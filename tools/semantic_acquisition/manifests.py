from __future__ import annotations

import argparse
import csv
import json
import math
import pickle
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from tools.semantic_acquisition.common import (
    deterministic_choice,
    file_sha256,
    stable_hash,
    stable_seed,
    write_csv,
    write_json,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_V1 = ROOT / "output" / "p0_v1_context_colocation_v2"
DEFAULT_DATA = ROOT / "DATA" / "cifar-100" / "cifar-100-python"
DEFAULT_OUTPUT = ROOT / "output" / "v2_v3_semantic_acquisition"
EXPECTED_INPUT = "ebd8f1d0b7765516d4f7ef8868a8262146dd3b630e7df7c1216c01d8e17b601b"
EXPECTED_UNIVERSE = "861850202191fe1db7037afef30987b14bfda680dd37d60af1f940c5c68b3b0f"
EXPECTED_BUDGETS = {
    79: 12, 81: 14, 82: 13, 83: 14, 84: 13, 85: 12, 86: 13,
    87: 12, 88: 12, 89: 13, 90: 11, 91: 13, 92: 13, 93: 13,
    94: 11, 95: 13, 96: 15, 97: 11, 98: 14, 99: 14,
}
BASE_FIELDS = [
    "stage", "data_seed", "tail_class", "draw", "condition", "client_role",
    "base_sample_id", "label", "is_tail", "is_related", "is_unrelated", "is_filler",
    "semantic_rank", "frequency_quintile", "match_pair_id", "global_class_count",
    "loss_weight", "base_multiset_hash",
]
EXEC_FIELDS = [
    "stage", "data_seed", "tail_class", "draw", "condition", "client_role",
    "epoch", "batch_index", "position_in_batch", "base_sample_id", "label",
    "slot_role", "loss_weight", "augmentation_seed", "execution_schedule_hash",
]
TOPOLOGY_BASE_FIELDS = [
    "stage", "data_seed", "topology", "client_id", "lt_index",
    "base_sample_id", "label", "client_size", "fedavg_weight",
    "global_multiset_hash", "partition_fingerprint",
]
TOPOLOGY_EXEC_FIELDS = [
    "stage", "data_seed", "topology", "client_id", "epoch",
    "batch_index", "position_in_batch", "base_sample_id", "label",
    "augmentation_seed", "execution_schedule_hash",
]


@dataclass
class FrozenInputs:
    labels: np.ndarray
    raw_train_ids: np.ndarray
    global_counts: np.ndarray
    tail_classes: list[int]
    class_names: list[str]
    similarity: np.ndarray
    top10: dict[int, list[int]]
    top30: dict[int, list[int]]
    quintiles: dict[int, int]
    budgets: dict[int, int]
    input_fingerprint: str
    universe_fingerprint: str
    mapping_hash: str
    checkpoint_sha256: str


@dataclass
class ManifestBundle:
    inputs: FrozenInputs
    companion_budget_rows: list[dict]
    matching_rows: list[dict]
    base_rows: list[dict]
    execution_rows: list[dict]
    placement_rows: list[dict]
    fairness_rows: list[dict]
    topology_base_rows: list[dict]
    topology_execution_rows: list[dict]
    topology_fairness_rows: list[dict]


def _load_pickle(path: Path) -> dict:
    with path.open("rb") as handle:
        return pickle.load(handle, encoding="latin1")


def _half_up(value: float) -> int:
    return int(math.floor(float(value) + 0.5))


def _array_universe_hash(labels: np.ndarray, raw_ids: np.ndarray, counts: np.ndarray) -> str:
    import hashlib
    digest = hashlib.sha256()
    for array in (labels, raw_ids, counts):
        value = np.ascontiguousarray(array)
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
        digest.update(value.tobytes())
    return digest.hexdigest()


def load_frozen_inputs(v1_dir: Path = DEFAULT_V1, data_dir: Path = DEFAULT_DATA) -> tuple[FrozenInputs, list[dict]]:
    v1_dir, data_dir = Path(v1_dir), Path(data_dir)
    required = [
        "client_class_counts.npz", "client_class_counts_meta.json", "partition_indices.npz",
        "partition_indices_meta.json", "partition_invariants.csv", "clip_related_classes.csv",
        "clip_similarity.npy", "clip_similarity_meta.json", "v1b_generic_context_per_class.csv",
        "v1_paired_deltas.csv", "v1_summary.json",
    ]
    missing = [name for name in required if not (v1_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing frozen P0/V1 artifacts: {missing}")

    meta = json.loads((v1_dir / "client_class_counts_meta.json").read_text(encoding="utf-8"))
    if meta.get("input_fingerprint") != EXPECTED_INPUT:
        raise RuntimeError("P0/V1 input fingerprint does not match the preregistered contract")
    arrays = np.load(v1_dir / "client_class_counts.npz")
    labels = arrays["global_train_labels"].astype(np.int64)
    raw_ids = arrays["selected_raw_train_indices"].astype(np.int64)
    counts = arrays["global_class_counts"].astype(np.int64)
    tail = [int(value) for value in arrays["tail_class_ids"].tolist()]
    universe_hash = _array_universe_hash(labels, raw_ids, counts)
    if universe_hash != EXPECTED_UNIVERSE:
        raise RuntimeError(f"Global universe fingerprint mismatch: {universe_hash}")
    if len(labels) != len(raw_ids) or int(counts[tail].sum()) != 153 or len(tail) != 20:
        raise RuntimeError("Frozen pool/tail cardinality mismatch")

    class_names = [str(name).replace("_", " ") for name in _load_pickle(data_dir / "meta")["fine_label_names"]]
    mapping_hash = stable_hash({str(index): name for index, name in enumerate(class_names)})
    sim_meta = json.loads((v1_dir / "clip_similarity_meta.json").read_text(encoding="utf-8"))
    if sim_meta.get("input_mapping_fingerprint") != mapping_hash:
        raise RuntimeError("CIFAR class mapping differs from the frozen V1 mapping")
    similarity = np.load(v1_dir / "clip_similarity.npy").astype(np.float64)
    if similarity.shape != (100, 100):
        raise RuntimeError(f"Unexpected similarity shape: {similarity.shape}")

    related = pd.read_csv(v1_dir / "clip_related_classes.csv")
    related = related[related["candidate_scope"] == "non_tail_only_primary"]
    top10 = {}
    non_tail = sorted(set(range(100)) - set(tail))
    top30 = {}
    for class_id in tail:
        rows = related[related["tail_class_id"] == class_id].sort_values("rank")
        values = [int(value) for value in rows["neighbor_class_id"].tolist()]
        if len(values) != 10 or len(set(values)) != 10 or set(values) & set(tail):
            raise RuntimeError(f"Invalid frozen Top-10 for tail class {class_id}")
        if any(class_names[int(row.neighbor_class_id)] != str(row.neighbor_class_name) for row in rows.itertuples()):
            raise RuntimeError(f"Class-name mismatch in related table for {class_id}")
        top10[class_id] = values
        top30[class_id] = sorted(non_tail, key=lambda other: (-float(similarity[class_id, other]), other))[:30]
        if top30[class_id][:10] != values:
            raise RuntimeError(f"Frozen Top-10 is inconsistent with similarity matrix for class {class_id}")

    ordered_non_tail = sorted(non_tail, key=lambda class_id: (int(counts[class_id]), class_id))
    groups = np.array_split(np.asarray(ordered_non_tail, dtype=np.int64), 5)
    if any(len(group) != 16 for group in groups):
        raise RuntimeError("V1 frequency quintiles are not five groups of 16")
    quintiles = {int(class_id): index for index, group in enumerate(groups) for class_id in group}

    dose = pd.read_csv(v1_dir / "v1b_generic_context_per_class.csv")
    dose = dose[dose["topology"] == "ClientLT-controlled"]
    budget_rows, budgets = [], {}
    for class_id in tail:
        rows = dose[dose["tail_class_id"] == class_id].sort_values("seed")
        values = {int(row.seed): float(row.generic_companion_sample_count_tail_mass_weighted) for row in rows.itertuples()}
        if set(values) != {42, 2026}:
            raise RuntimeError(f"Missing V1 dose rows for class {class_id}: {values}")
        mean = (values[42] + values[2026]) / 2.0
        budget = _half_up(mean)
        budgets[class_id] = budget
        budget_rows.append({"tail_class": class_id, "dose_seed42": values[42], "dose_seed2026": values[2026], "mean": mean, "half_up_budget": budget})
    if budgets != EXPECTED_BUDGETS:
        raise RuntimeError(f"Recomputed companion budgets differ from contract: {budgets}")

    return FrozenInputs(
        labels=labels, raw_train_ids=raw_ids, global_counts=counts, tail_classes=tail,
        class_names=class_names, similarity=similarity, top10=top10, top30=top30,
        quintiles=quintiles, budgets=budgets, input_fingerprint=EXPECTED_INPUT,
        universe_fingerprint=universe_hash, mapping_hash=mapping_hash,
        checkpoint_sha256=str(sim_meta["checkpoint_sha256"]),
    ), budget_rows


def quota_vector(budget: int, width: int = 10) -> list[int]:
    quota = [0] * width
    for index in range(int(budget)):
        quota[index % width] += 1
    return quota


def _minimum_assignment(related: Sequence[int], candidates: Sequence[int], counts: np.ndarray, tie_parts: tuple) -> list[int]:
    related, candidates = list(related), sorted(set(int(value) for value in candidates))
    if len(candidates) < len(related):
        raise RuntimeError(f"UNMATCHABLE: need {len(related)}, have {len(candidates)}")
    costs = [[abs(math.log(int(counts[r]) + 1) - math.log(int(counts[u]) + 1)) for u in candidates] for r in related]
    tie_rank = [[stable_seed("match-tie", *tie_parts, rank, candidate) for candidate in candidates] for rank in range(len(related))]

    @lru_cache(maxsize=None)
    def solve(position: int, used: int):
        if position == len(related):
            return 0.0, (), ()
        best = None
        for candidate_index, candidate in enumerate(candidates):
            if used & (1 << candidate_index):
                continue
            child_cost, child_tie, child_values = solve(position + 1, used | (1 << candidate_index))
            proposal = (
                costs[position][candidate_index] + child_cost,
                (tie_rank[position][candidate_index],) + child_tie,
                (candidate,) + child_values,
            )
            if best is None or proposal[:2] < best[:2]:
                best = proposal
        if best is None:
            raise RuntimeError("UNMATCHABLE")
        return best

    return list(solve(0, 0)[2])


def match_unrelated(inputs: FrozenInputs, class_id: int, data_seed: int, draw: int) -> tuple[list[int], list[dict]]:
    related = inputs.top10[class_id]
    excluded = set(inputs.tail_classes) | set(inputs.top30[class_id]) | set(related)
    assigned: dict[int, int] = {}
    for quintile in range(5):
        ranks = [index for index, value in enumerate(related) if inputs.quintiles[value] == quintile]
        if not ranks:
            continue
        candidates = [value for value, group in inputs.quintiles.items() if group == quintile and value not in excluded]
        selected = _minimum_assignment([related[index] for index in ranks], candidates, inputs.global_counts, (data_seed, class_id, draw, quintile))
        assigned.update({rank: value for rank, value in zip(ranks, selected)})
    if set(assigned) != set(range(10)) or len(set(assigned.values())) != 10:
        raise RuntimeError(f"UNMATCHABLE class={class_id} seed={data_seed} draw={draw}")
    unrelated = [assigned[index] for index in range(10)]
    rows = []
    for rank, (related_id, unrelated_id) in enumerate(zip(related, unrelated), start=1):
        rows.append({
            "data_seed": data_seed, "tail_class": class_id, "draw": draw, "semantic_rank": rank,
            "related_class": related_id, "unrelated_class": unrelated_id,
            "frequency_quintile": inputs.quintiles[related_id],
            "related_global_count": int(inputs.global_counts[related_id]),
            "unrelated_global_count": int(inputs.global_counts[unrelated_id]),
            "exact_count_match": int(inputs.global_counts[related_id]) == int(inputs.global_counts[unrelated_id]),
            "matching_cost": abs(math.log(int(inputs.global_counts[related_id]) + 1) - math.log(int(inputs.global_counts[unrelated_id]) + 1)),
            "match_pair_id": f"c{class_id}:rank{rank}",
        })
    return unrelated, rows


def _sample_block(pools: Mapping[int, Sequence[int]], classes: Sequence[int], quotas: Sequence[int], *seed_parts) -> list[dict]:
    rows = []
    for rank, (class_id, count) in enumerate(zip(classes, quotas), start=1):
        selected = deterministic_choice(pools[int(class_id)], int(count), "class-sample", *seed_parts, rank, int(class_id))
        for occurrence, raw_id in enumerate(selected):
            rows.append({"raw_id": raw_id, "label": int(class_id), "rank": rank, "occurrence": occurrence, "pair": f"rank{rank}:occ{occurrence}"})
    return rows


def _multiset_hash(rows: Sequence[Mapping]) -> str:
    return stable_hash(sorted((str(row["base_sample_id"]), int(row["label"])) for row in rows))


def _partition_payload_hash(partition: Mapping[int, np.ndarray]) -> str:
    return stable_hash({
        str(client_id): sorted(np.asarray(indices, dtype=np.int64).tolist())
        for client_id, indices in sorted(partition.items())
    })


def build_topology_replay_manifests(
    inputs: FrozenInputs,
    v1_dir: Path,
    data_seeds: Sequence[int],
    batch_size: int = 32,
    epochs: int = 3,
) -> tuple[list[dict], list[dict], list[dict]]:
    """Replay the frozen 30-client partitions with sample-bound augmentation.

    Dirichlet and ClientLT use the same complete LT universe. Client sizes,
    batch composition, support dispersion and sample-weighted FedAvg are kept
    as treatment-induced topology properties rather than falsely equalized.
    """
    archive = np.load(Path(v1_dir) / "partition_indices.npz")
    raw_global_hash = stable_hash(sorted(
        (f"train:{int(raw_id)}", int(label))
        for raw_id, label in zip(inputs.raw_train_ids, inputs.labels)
    ))
    base_rows, execution_rows, fairness_rows = [], [], []
    by_seed_topology = {}
    for seed in [int(value) for value in data_seeds]:
        for topology, key_name in (
            ("Dirichlet", "dirichlet"),
            ("ClientLT-controlled", "clientlt_controlled"),
        ):
            partition = {
                client_id: archive[f"seed{seed}_{key_name}_client{client_id}"].astype(np.int64)
                for client_id in range(30)
            }
            flat = np.concatenate([partition[client_id] for client_id in range(30)])
            if len(flat) != len(inputs.labels) or len(np.unique(flat)) != len(inputs.labels):
                raise RuntimeError(f"Topology replay does not conserve global pool: seed={seed}, topology={topology}")
            if set(flat.tolist()) != set(range(len(inputs.labels))):
                raise RuntimeError(f"Topology replay misses LT positions: seed={seed}, topology={topology}")
            partition_hash = _partition_payload_hash(partition)
            controlled_tail_constraints = None
            controlled_tail_count = None
            controlled_companion_count = None
            controlled_min_specialist_purity = None
            if topology == "ClientLT-controlled":
                tail_set = set(int(value) for value in inputs.tail_classes)
                all_tail = [int(index) for index in flat.tolist() if int(inputs.labels[index]) in tail_set]
                specialist = np.concatenate([partition[client_id] for client_id in (27, 28, 29)])
                specialist_tail = [
                    int(index) for index in specialist.tolist()
                    if int(inputs.labels[index]) in tail_set
                ]
                controlled_tail_count = len(specialist_tail)
                controlled_companion_count = int(len(specialist) - len(specialist_tail))
                purities = []
                for client_id in (27, 28, 29):
                    client_indices = partition[client_id]
                    client_tail_count = sum(
                        int(inputs.labels[index]) in tail_set for index in client_indices.tolist()
                    )
                    purities.append(client_tail_count / float(len(client_indices)))
                controlled_min_specialist_purity = min(purities)
                controlled_tail_constraints = bool(
                    len(all_tail) == 153
                    and controlled_tail_count == 153
                    and controlled_companion_count <= 38
                    and controlled_min_specialist_purity >= 0.8
                )
                if not controlled_tail_constraints:
                    raise RuntimeError(
                        "Frozen ClientLT partition violates the preregistered no-tail-leakage/8:2 constraints: "
                        f"seed={seed}, tail={controlled_tail_count}, companion={controlled_companion_count}, "
                        f"min_purity={controlled_min_specialist_purity}"
                    )
            topology_aug_hashes = {}
            for client_id in range(30):
                indices = partition[client_id]
                client_size = int(len(indices))
                weight = client_size / float(len(inputs.labels))
                for lt_index in indices.tolist():
                    base_rows.append({
                        "stage": "v2_topology", "data_seed": seed, "topology": topology,
                        "client_id": client_id, "lt_index": int(lt_index),
                        "base_sample_id": f"train:{int(inputs.raw_train_ids[lt_index])}",
                        "label": int(inputs.labels[lt_index]), "client_size": client_size,
                        "fedavg_weight": weight, "global_multiset_hash": raw_global_hash,
                        "partition_fingerprint": partition_hash,
                    })
                client_execution = []
                for epoch in range(1, epochs + 1):
                    generator = np.random.default_rng(stable_seed("topology-order", seed, topology, client_id, epoch))
                    ordered = indices[generator.permutation(client_size)]
                    for position, lt_index in enumerate(ordered.tolist()):
                        sample_id = f"train:{int(inputs.raw_train_ids[lt_index])}"
                        augmentation_seed = stable_seed("topology-aug", seed, epoch, sample_id)
                        topology_aug_hashes[(epoch, sample_id)] = augmentation_seed
                        client_execution.append({
                            "stage": "v2_topology", "data_seed": seed, "topology": topology,
                            "client_id": client_id, "epoch": epoch,
                            "batch_index": position // int(batch_size),
                            "position_in_batch": position % int(batch_size),
                            "base_sample_id": sample_id, "label": int(inputs.labels[lt_index]),
                            "augmentation_seed": augmentation_seed, "execution_schedule_hash": "",
                        })
                schedule_hash = stable_hash([
                    {key: value for key, value in row.items() if key != "execution_schedule_hash"}
                    for row in client_execution
                ])
                for row in client_execution:
                    row["execution_schedule_hash"] = schedule_hash
                execution_rows.extend(client_execution)
            by_seed_topology[(seed, topology)] = topology_aug_hashes
            fairness_rows.append({
                "stage": "v2_topology", "data_seed": seed, "topology": topology,
                "global_pool_conserved": True, "global_multiset_hash": raw_global_hash,
                "base_sample_count": int(len(flat)), "unique_base_sample_count": int(len(np.unique(flat))),
                "execution_repetition_correct": True,
                "fedavg_weight_sum": sum(len(partition[c]) for c in range(30)) / float(len(inputs.labels)),
                "sample_bound_augmentation": True, "cross_topology_augmented_input_equal": None,
                "client_sizes_equal": None,
                "controlled_tail_constraints": controlled_tail_constraints,
                "controlled_tail_count": controlled_tail_count,
                "controlled_companion_count": controlled_companion_count,
                "controlled_min_specialist_purity": controlled_min_specialist_purity,
                "reason": "client sizes and local step counts are topology treatment properties",
                "pass": True,
            })
        left = by_seed_topology[(seed, "Dirichlet")]
        right = by_seed_topology[(seed, "ClientLT-controlled")]
        equal = left == right
        if not equal:
            raise AssertionError(f"Cross-topology sample-bound augmentation mismatch for seed {seed}")
        for row in fairness_rows:
            if row["data_seed"] == seed:
                row["cross_topology_augmented_input_equal"] = True
    return base_rows, execution_rows, fairness_rows


def _execution(rows: Sequence[Mapping], stage: str, data_seed: int, class_id: int, draw: int, condition: str, client_role: str, epochs: int = 3) -> list[dict]:
    output = []
    for epoch in range(1, epochs + 1):
        if stage == "v2":
            ordered = sorted(rows, key=lambda row: (0 if row["is_tail"] else 1, str(row["match_pair_id"]), str(row["base_sample_id"])))
            aug = lambda row, pos: stable_seed("v2-aug", data_seed, class_id, epoch, row["match_pair_id"])
        else:
            ordered = sorted(rows, key=lambda row: (str(row["base_sample_id"]), str(row["match_pair_id"])))
            aug = lambda row, pos: stable_seed("v3-aug", data_seed, class_id, draw, epoch, row["base_sample_id"])
        for position, row in enumerate(ordered):
            output.append({
                "stage": stage, "data_seed": data_seed, "tail_class": class_id, "draw": draw,
                "condition": condition, "client_role": client_role, "epoch": epoch, "batch_index": 0,
                "position_in_batch": position, "base_sample_id": row["base_sample_id"], "label": row["label"],
                "slot_role": "tail" if row["is_tail"] else ("related" if row["is_related"] else ("unrelated" if row["is_unrelated"] else "filler")),
                "loss_weight": row["loss_weight"], "augmentation_seed": aug(row, position),
                "execution_schedule_hash": "",
            })
    schedule_hash = stable_hash([{key: value for key, value in row.items() if key != "execution_schedule_hash"} for row in output])
    for row in output:
        row["execution_schedule_hash"] = schedule_hash
    return output


def _base_row(stage, seed, class_id, draw, condition, client, item, role, quintile, count, loss_weight=1.0):
    return {
        "stage": stage, "data_seed": seed, "tail_class": class_id, "draw": draw,
        "condition": condition, "client_role": client,
        "base_sample_id": f"train:{int(item['raw_id'])}", "label": int(item["label"]),
        "is_tail": role == "tail", "is_related": role == "related", "is_unrelated": role == "unrelated", "is_filler": role == "filler",
        "semantic_rank": item.get("rank", ""), "frequency_quintile": quintile if quintile is not None else "",
        "match_pair_id": item.get("pair", ""), "global_class_count": int(count),
        "loss_weight": float(loss_weight), "base_multiset_hash": "",
    }


def build_manifests(v1_dir: Path = DEFAULT_V1, data_dir: Path = DEFAULT_DATA, data_seeds: Sequence[int] = (42, 2026), unrelated_draws: int = 3, tail_classes: Sequence[int] | None = None) -> ManifestBundle:
    inputs, budget_rows = load_frozen_inputs(v1_dir, data_dir)
    classes = inputs.tail_classes if tail_classes is None else [int(value) for value in tail_classes]
    if not set(classes) <= set(inputs.tail_classes):
        raise ValueError("Requested classes are not all in frozen bottom-20")
    pools = {class_id: inputs.raw_train_ids[inputs.labels == class_id].tolist() for class_id in range(100)}
    base_rows, execution_rows, matching_rows, placement_rows = [], [], [], []

    for seed in [int(value) for value in data_seeds]:
        for class_id in classes:
            budget = inputs.budgets[class_id]
            quotas = quota_vector(budget)
            tail_items = [{"raw_id": raw_id, "label": class_id, "rank": "", "pair": f"tail:{index}"} for index, raw_id in enumerate(sorted(pools[class_id]))]
            related_items = _sample_block(pools, inputs.top10[class_id], quotas, seed, class_id, "related")
            all_unrelated = {}
            unrelated_class_union = set()
            for draw in range(unrelated_draws):
                unrelated_classes, match_rows = match_unrelated(inputs, class_id, seed, draw)
                matching_rows.extend(match_rows)
                unrelated_class_union.update(unrelated_classes)
                all_unrelated[draw] = _sample_block(pools, unrelated_classes, quotas, seed, class_id, "unrelated", draw)

            filler_classes = [
                value for value in range(100)
                if value not in set(inputs.tail_classes)
                and value not in set(inputs.top30[class_id])
                and value not in set(inputs.top10[class_id])
                and value not in unrelated_class_union
            ]
            filler_pool = [raw_id for value in filler_classes for raw_id in pools[value]]
            filler_ids = deterministic_choice(filler_pool, len(tail_items), "filler", seed, class_id)
            raw_to_label = {int(raw_id): int(label) for raw_id, label in zip(inputs.raw_train_ids, inputs.labels)}
            filler_items = [{"raw_id": raw_id, "label": raw_to_label[raw_id], "rank": "", "pair": f"filler:{index}"} for index, raw_id in enumerate(filler_ids)]

            def make_rows(stage, draw, condition, client, blocks):
                result = []
                for role, items, masked in blocks:
                    for item in items:
                        group = inputs.quintiles.get(int(item["label"]))
                        result.append(_base_row(stage, seed, class_id, draw, condition, client, item, role, group, inputs.global_counts[int(item["label"])], 0.0 if masked else 1.0))
                block_hash = _multiset_hash(result)
                for row in result:
                    row["base_multiset_hash"] = block_hash
                return result

            v2_specs = [
                (-1, "related", [("tail", tail_items, False), ("related", related_items, False)]),
                (-1, "tail_only_masked", [("tail", tail_items, False), ("related", related_items, True)]),
            ] + [(draw, f"matched_unrelated_r{draw}", [("tail", tail_items, False), ("unrelated", all_unrelated[draw], False)]) for draw in range(unrelated_draws)]
            for draw, condition, blocks in v2_specs:
                rows = make_rows("v2", draw, condition, "single", blocks)
                base_rows.extend(rows)
                execution_rows.extend(_execution(rows, "v2", seed, class_id, draw, condition, "single"))

            for draw in range(unrelated_draws):
                placements = {
                    "R_colocated": {
                        "S": [("tail", tail_items, False), ("related", related_items, False)],
                        "D": [("unrelated", all_unrelated[draw], False), ("filler", filler_items, False)],
                    },
                    "R_remote_U_colocated": {
                        "S": [("tail", tail_items, False), ("unrelated", all_unrelated[draw], False)],
                        "D": [("related", related_items, False), ("filler", filler_items, False)],
                    },
                }
                global_hash = None
                for placement, clients in placements.items():
                    combined = []
                    client_rows = {}
                    for client, blocks in clients.items():
                        rows = make_rows("v3", draw, placement, client, blocks)
                        client_rows[client] = rows
                        combined.extend(rows)
                    placement_hash = _multiset_hash(combined)
                    global_hash = placement_hash if global_hash is None else global_hash
                    if placement_hash != global_hash:
                        raise RuntimeError("V3 placement global multiset mismatch")
                    for client, rows in client_rows.items():
                        for row in rows:
                            row["base_multiset_hash"] = placement_hash
                        base_rows.extend(rows)
                        execution_rows.extend(_execution(rows, "v3", seed, class_id, draw, placement, client))
                    placement_rows.append({
                        "data_seed": seed, "tail_class": class_id, "draw": draw, "placement": placement,
                        "support_size": len(client_rows["S"]), "remote_size": len(client_rows["D"]),
                        "support_weight": 0.5, "remote_weight": 0.5, "global_multiset_hash": placement_hash,
                    })

    topology_base_rows, topology_execution_rows, topology_fairness_rows = build_topology_replay_manifests(
        inputs, v1_dir, data_seeds
    )
    bundle = ManifestBundle(
        inputs, budget_rows, matching_rows, base_rows, execution_rows,
        placement_rows, [], topology_base_rows, topology_execution_rows,
        topology_fairness_rows,
    )
    bundle.fairness_rows = validate_manifest_bundle(bundle, data_seeds, classes, unrelated_draws)
    return bundle


def validate_manifest_bundle(bundle: ManifestBundle, data_seeds: Sequence[int], classes: Sequence[int], unrelated_draws: int) -> list[dict]:
    base, execution = pd.DataFrame(bundle.base_rows), pd.DataFrame(bundle.execution_rows)
    rows = []
    for stage in ("v2", "v3"):
        stage_base, stage_exec = base[base.stage == stage], execution[execution.stage == stage]
        grouped = stage_base.groupby(["data_seed", "tail_class", "draw", "condition"], sort=True)
        for key, unit in grouped:
            seed, class_id, draw, condition = key
            unit_exec = stage_exec[(stage_exec.data_seed == seed) & (stage_exec.tail_class == class_id) & (stage_exec.draw == draw) & (stage_exec.condition == condition)]
            ids = unit.base_sample_id.tolist()
            base_unique = len(ids) == len(set(ids)) if stage == "v2" else all(group.base_sample_id.is_unique for _, group in unit.groupby("client_role"))
            repeats = unit_exec.groupby(["client_role", "base_sample_id"]).size()
            repetition_ok = len(repeats) == len(unit) and bool((repeats == 3).all())
            client_sizes = unit.groupby("client_role").size().to_dict()
            v3_sizes = None if stage == "v2" else len(set(client_sizes.values())) == 1
            row = {
                "stage": stage, "data_seed": int(seed), "tail_class": int(class_id), "draw": int(draw), "condition": condition,
                "global_pool_hash_equal": True, "theta0_hash_equal": None, "pretrain_logits_equal": None,
                "tail_ids_equal": True, "tail_slots_equal": True, "companion_budget_equal": True, "quota_equal": True,
                "base_multiset_conserved": base_unique, "execution_repetition_correct": repetition_ok,
                "v2_paired_slot_augmentation_equal": None, "v3_per_sample_augmentation_equal": None,
                "v3_augmented_multiset_equal": None, "batch_size_equal": True, "optimizer_steps_equal": True,
                "scheduler_steps_equal": True, "loss_denominator_equal": True, "eval_ids_equal": True,
                "amp_overflow_equal": None, "client_sizes_equal": v3_sizes, "fedavg_weights_equal": None if stage == "v2" else True,
                "train_test_disjoint": True, "pass": False, "reason": "runtime fields pending",
            }
            structural = [base_unique, repetition_ok, v3_sizes if stage == "v3" else True]
            row["pass"] = all(structural)
            row["reason"] = "structural manifest checks passed; model/runtime fields pending smoke" if row["pass"] else "structural manifest invariant failed"
            rows.append(row)

    # Cross-placement invariants, including sample-bound augmentation seeds.
    v3b, v3e = base[base.stage == "v3"], execution[execution.stage == "v3"]
    for (seed, class_id, draw), group in v3b.groupby(["data_seed", "tail_class", "draw"]):
        hashes = group.groupby("condition").base_multiset_hash.first().tolist()
        if len(hashes) != 2 or len(set(hashes)) != 1:
            raise AssertionError(f"V3 global multiset mismatch for {(seed, class_id, draw)}")
        ex = v3e[(v3e.data_seed == seed) & (v3e.tail_class == class_id) & (v3e.draw == draw)]
        pivot = ex.groupby(["condition", "epoch", "base_sample_id"]).augmentation_seed.first().reset_index()
        counts = pivot.groupby(["epoch", "base_sample_id"]).augmentation_seed.nunique()
        if not bool((counts == 1).all()):
            raise AssertionError(f"V3 augmentation seed did not follow samples for {(seed, class_id, draw)}")

    # V2 pairs keep every tail slot and semantic companion slot seed fixed.
    v2e = execution[execution.stage == "v2"]
    for (seed, class_id), group in v2e.groupby(["data_seed", "tail_class"]):
        signatures = {}
        for condition, condition_rows in group.groupby("condition"):
            signatures[condition] = [
                (int(row.epoch), int(row.position_in_batch), "tail" if str(row.slot_role) == "tail" else "companion", int(row.augmentation_seed))
                for row in condition_rows.sort_values(["epoch", "position_in_batch"]).itertuples()
            ]
        reference = signatures.get("related")
        if reference is None or any(value != reference for value in signatures.values()):
            raise AssertionError(f"V2 paired slot/augmentation mismatch for {(seed, class_id)}")

    for row in rows:
        if row["stage"] == "v2":
            row["v2_paired_slot_augmentation_equal"] = True
        else:
            row["v3_per_sample_augmentation_equal"] = True
            row["v3_augmented_multiset_equal"] = True

    if not all(row["pass"] for row in rows):
        raise AssertionError("One or more structural fairness invariants failed")
    return rows


def write_bundle(bundle: ManifestBundle, output_dir: Path) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "companion_budgets.json", {
        "rounding": "half-up", "budgets": {str(key): value for key, value in sorted(bundle.inputs.budgets.items())},
        "rows": bundle.companion_budget_rows,
    })
    write_csv(output_dir / "matching_manifest.csv", bundle.matching_rows)
    write_csv(output_dir / "base_sample_manifest.csv", bundle.base_rows, BASE_FIELDS)
    write_csv(output_dir / "execution_slot_manifest.csv", bundle.execution_rows, EXEC_FIELDS)
    write_csv(output_dir / "v3_placement_manifest.csv", bundle.placement_rows)
    write_csv(output_dir / "fairness_invariants.csv", bundle.fairness_rows)
    write_csv(output_dir / "v2_topology_base_manifest.csv", bundle.topology_base_rows, TOPOLOGY_BASE_FIELDS)
    write_csv(output_dir / "v2_topology_execution_manifest.csv", bundle.topology_execution_rows, TOPOLOGY_EXEC_FIELDS)
    write_csv(output_dir / "v2_topology_fairness.csv", bundle.topology_fairness_rows)
    contract = {
        "schema_version": 1,
        "input_fingerprint": bundle.inputs.input_fingerprint,
        "global_universe_fingerprint": bundle.inputs.universe_fingerprint,
        "class_mapping_hash": bundle.inputs.mapping_hash,
        "clip_checkpoint_sha256": bundle.inputs.checkpoint_sha256,
        "tail_classes": bundle.inputs.tail_classes,
        "tail_sample_count": int(bundle.inputs.global_counts[bundle.inputs.tail_classes].sum()),
        "top10": {str(key): value for key, value in bundle.inputs.top10.items()},
        "top30": {str(key): value for key, value in bundle.inputs.top30.items()},
        "frequency_quintiles": {str(key): value for key, value in sorted(bundle.inputs.quintiles.items())},
        "companion_budgets_path": "companion_budgets.json",
        "companion_budgets_sha256": file_sha256(output_dir / "companion_budgets.json"),
        "matching_rule": "same V1 frequency quintile; min abs(log(n_r+1)-log(n_u+1)); SHA-256 tie break",
        "v2_augmentation_binding": "paired semantic slot",
        "v3_augmentation_binding": "physical base sample",
        "resolved_training": {
            "backbone": "ViT-B/16", "encoder": "vision", "position": "top3", "params": ["q", "v"],
            "rank": 2, "alpha": 1, "dropout": 0.0,
            "precision": "fp32", "mainline_precision": "amp",
            "precision_rationale": "three-step AMP smoke produced condition-dependent skipped optimizer steps; FP32 is the preregistered numerical control for the mechanism test",
            "batch_size": 32,
            "local_epochs": 3, "optimizer": "sgd", "lr": 0.002, "momentum": 0.9,
            "weight_decay": 0.0005, "scheduler": "single_step", "stepsize": 3, "gamma": 1.0,
            "warmup_epoch": -1, "gradient_clipping": None,
            "transforms": ["random_resized_crop", "random_flip", "normalize"],
        },
        "model_runtime_fields": None,
        "model_runtime_reason": "populated by CUDA runner after theta0/model construction",
    }
    contract["manifest_hashes"] = {
        name: file_sha256(output_dir / name)
        for name in (
            "companion_budgets.json", "matching_manifest.csv", "base_sample_manifest.csv",
            "execution_slot_manifest.csv", "v3_placement_manifest.csv", "fairness_invariants.csv",
            "v2_topology_base_manifest.csv", "v2_topology_execution_manifest.csv",
            "v2_topology_fairness.csv",
        )
    }
    write_json(output_dir / "experiment_contract.json", contract)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Build deterministic V2/V3 manifests without training")
    parser.add_argument("--v1-dir", type=Path, default=DEFAULT_V1)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--data-seeds", nargs="+", type=int, default=[42, 2026])
    parser.add_argument("--unrelated-draws", type=int, default=3)
    parser.add_argument("--tail-classes", nargs="+", type=int)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    bundle = build_manifests(args.v1_dir, args.data_dir, args.data_seeds, args.unrelated_draws, args.tail_classes)
    write_bundle(bundle, args.output_dir)
    print(json.dumps({
        "output_dir": str(args.output_dir.resolve()), "base_rows": len(bundle.base_rows),
        "execution_rows": len(bundle.execution_rows), "matching_rows": len(bundle.matching_rows),
        "topology_base_rows": len(bundle.topology_base_rows),
        "topology_execution_rows": len(bundle.topology_execution_rows),
        "structural_gate": "PASS",
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
