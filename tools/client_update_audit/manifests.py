from __future__ import annotations

import argparse
import contextlib
import importlib.util
import io
import json
import math
import pickle
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np

from tools.breadth_audit.inputs import load_preregistered_neighbors
from tools.client_update_audit.protocol import TAIL_CLASSES, TAIL_CLIENT_IDS, frozen_protocol
from tools.semantic_acquisition.common import file_sha256, stable_hash, stable_seed, write_csv, write_json
from utils.datasplit import partition_client_longtail_controlled, partition_fine_class_dirichlet


ROOT = Path(__file__).resolve().parents[2]
NUM_CLIENTS = 30
NUM_CLASSES = 100


def _load_pickle(path: Path) -> dict:
    with Path(path).open("rb") as handle:
        return pickle.load(handle, encoding="latin1")


def load_exact_lt_pool(data_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str], np.ndarray]:
    """Rebuild the exact repository CIFAR-100-LT pool and raw-index mapping."""
    data_dir = Path(data_dir)
    train = _load_pickle(data_dir / "train")
    test = _load_pickle(data_dir / "test")
    meta = _load_pickle(data_dir / "meta")
    raw_train_labels = np.asarray(train["fine_labels"], dtype=np.int64)
    raw_coarse_labels = np.asarray(train["coarse_labels"], dtype=np.int64)
    test_labels = np.asarray(test["fine_labels"], dtype=np.int64)
    class_names = [str(name).replace("_", " ") for name in meta["fine_label_names"]]

    spec = importlib.util.spec_from_file_location("e2_repository_long_tail", ROOT / "datasets" / "long_tail.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    by_class = [np.flatnonzero(raw_train_labels == class_id).tolist() for class_id in range(NUM_CLASSES)]
    with contextlib.redirect_stdout(io.StringIO()):
        _, selected_by_class = module.train_long_tail(by_class, NUM_CLASSES, 0.01, "exp")
    raw_ids = np.asarray(module.flatten_list(selected_by_class), dtype=np.int64)
    labels = raw_train_labels[raw_ids]
    # The repository's LT generator uses len(flat_train_indices)/num_classes
    # (=500) as the exponential head budget and realizes 10,847 samples.
    if len(labels) != 10847 or int(np.isin(labels, TAIL_CLASSES).sum()) != 153:
        raise RuntimeError(
            f"Unexpected CIFAR-100-LT pool: samples={len(labels)}, tail={int(np.isin(labels, TAIL_CLASSES).sum())}"
        )
    fine_to_coarse = np.full(NUM_CLASSES, -1, dtype=np.int64)
    for class_id in range(NUM_CLASSES):
        values = np.unique(raw_coarse_labels[raw_train_labels == class_id])
        if len(values) != 1:
            raise RuntimeError(f"Fine class {class_id} does not map to exactly one coarse class")
        fine_to_coarse[class_id] = int(values[0])
    return labels, raw_ids, test_labels, class_names, fine_to_coarse


def _partition_hash(partition: Mapping[int, np.ndarray]) -> str:
    return stable_hash({str(k): sorted(np.asarray(v, dtype=np.int64).tolist()) for k, v in sorted(partition.items())})


def _validate_partition(labels: np.ndarray, partition: Mapping[int, np.ndarray], *, condition: str) -> None:
    if set(partition) != set(range(NUM_CLIENTS)):
        raise RuntimeError(f"{condition}: client ids are incomplete")
    flat = np.concatenate([np.asarray(partition[k], dtype=np.int64) for k in range(NUM_CLIENTS)])
    if len(flat) != len(labels) or len(np.unique(flat)) != len(labels):
        raise RuntimeError(f"{condition}: global LT pool is not conserved")
    if set(flat.tolist()) != set(range(len(labels))):
        raise RuntimeError(f"{condition}: partition index universe differs")


def _partitions(labels: np.ndarray, test_labels: np.ndarray, seed: int) -> dict[str, dict[int, np.ndarray]]:
    dirichlet, _ = partition_fine_class_dirichlet(
        labels, test_labels, NUM_CLIENTS, NUM_CLASSES, 0.5, int(seed)
    )
    clientlt = partition_client_longtail_controlled(
        labels,
        NUM_CLIENTS,
        NUM_CLASSES,
        head_client_ratio=0.9,
        tail_client_ratio=0.1,
        tail_class_ratio=0.2,
        intra_group_alpha=0.5,
        tail_client_min_purity=0.8,
        tail_class_ids=TAIL_CLASSES,
        rng=np.random.RandomState(int(seed)),
    )
    output = {
        "dirichlet": {k: np.asarray(v, dtype=np.int64) for k, v in dirichlet.items()},
        "clientlt": {k: np.asarray(v, dtype=np.int64) for k, v in clientlt.items()},
    }
    for name, partition in output.items():
        _validate_partition(labels, partition, condition=name)
    tail_set = set(TAIL_CLASSES)
    tail_positions = set(np.flatnonzero(np.isin(labels, TAIL_CLASSES)).tolist())
    specialist_positions = set(np.concatenate([output["clientlt"][k] for k in TAIL_CLIENT_IDS]).tolist())
    if not tail_positions <= specialist_positions:
        raise RuntimeError("ClientLT has tail leakage outside specialist clients")
    companion_count = sum(
        int(np.sum(~np.isin(labels[output["clientlt"][k]], list(tail_set)))) for k in TAIL_CLIENT_IDS
    )
    if companion_count > 38:
        raise RuntimeError(f"ClientLT companion budget is {companion_count}, expected <=38")
    return output


def _counts(labels: np.ndarray, indices: Sequence[int]) -> np.ndarray:
    return np.bincount(labels[np.asarray(indices, dtype=np.int64)], minlength=NUM_CLASSES)


def _tail_relevance_by_class(tail_counts: np.ndarray, neighbors: Mapping[int, Sequence[int]]) -> np.ndarray:
    """Tail-mass weighted reciprocal neighbor-rank relevance for non-tail classes."""
    relevance = np.zeros(NUM_CLASSES, dtype=np.float64)
    total = float(tail_counts[TAIL_CLASSES].sum())
    if total <= 0:
        return relevance
    for tail_class in TAIL_CLASSES:
        mass = float(tail_counts[tail_class]) / total
        for rank, class_id in enumerate(neighbors[int(tail_class)], start=1):
            relevance[int(class_id)] += mass / float(rank)
    return relevance


def _taxonomy_relevance_by_class(tail_counts: np.ndarray, fine_to_coarse: np.ndarray) -> np.ndarray:
    """Independent semantic relevance from the official CIFAR-100 superclass taxonomy."""
    relevance = np.zeros(NUM_CLASSES, dtype=np.float64)
    total = float(tail_counts[TAIL_CLASSES].sum())
    if total <= 0:
        return relevance
    coarse_mass = np.zeros(20, dtype=np.float64)
    for tail_class in TAIL_CLASSES:
        coarse_mass[int(fine_to_coarse[tail_class])] += float(tail_counts[tail_class]) / total
    for class_id in range(80):
        relevance[class_id] = coarse_mass[int(fine_to_coarse[class_id])]
    return relevance


def _balanced_quotas(total: int, width: int) -> list[int]:
    if total < width:
        width = total
    return [total // width + int(i < total % width) for i in range(width)]


def _candidate_sets(
    relevance: np.ndarray,
    capacities: Mapping[int, int],
    width: int,
    total: int,
    *,
    seed_parts: tuple,
    draws: int,
    group_ids: np.ndarray | None = None,
    minimum_group_width: int = 1,
) -> list[tuple[float, tuple[int, ...], tuple[int, ...]]]:
    width = min(int(width), int(total))
    quotas = _balanced_quotas(int(total), width)
    eligible = [
        class_id for class_id in range(80)
        if int(capacities.get(class_id, 0)) >= min(quotas)
    ]
    if len(eligible) < width:
        raise RuntimeError(f"Only {len(eligible)} companion classes can support width={width}")
    generator = np.random.default_rng(stable_seed("e2b-class-sets", *seed_parts))
    sets = set()
    ranked = sorted(eligible, key=lambda c: (-float(relevance[c]), c))
    sets.add(tuple(sorted(ranked[:width])))
    sets.add(tuple(sorted(ranked[-width:])))
    if width == 2:
        for left in range(len(eligible)):
            for right in range(left + 1, len(eligible)):
                sets.add((eligible[left], eligible[right]))
    else:
        for _ in range(int(draws)):
            sets.add(tuple(sorted(int(x) for x in generator.choice(eligible, size=width, replace=False))))
    output = []
    for classes in sets:
        if group_ids is not None and len({int(group_ids[c]) for c in classes}) < int(minimum_group_width):
            continue
        ordered = tuple(sorted(classes, key=lambda c: (-float(relevance[c]), c)))
        if any(int(capacities.get(c, 0)) < quota for c, quota in zip(ordered, quotas)):
            continue
        mean = sum(float(relevance[c]) * quota for c, quota in zip(ordered, quotas)) / float(total)
        output.append((mean, ordered, tuple(quotas)))
    if not output:
        raise RuntimeError(f"No feasible companion class allocation for width={width}, total={total}")
    return sorted(output, key=lambda x: (x[0], x[1]))


def _matched_allocations(
    relevance: np.ndarray,
    capacities: Mapping[int, int],
    total: int,
    seed: int,
    client_id: int,
    tolerance: float,
    fine_to_coarse: np.ndarray | None = None,
) -> dict[str, tuple[float, tuple[int, ...], tuple[int, ...]]]:
    narrow = _candidate_sets(
        relevance, capacities, 2, total, seed_parts=(seed, client_id, "narrow"), draws=0,
        group_ids=fine_to_coarse, minimum_group_width=1,
    )
    broad = _candidate_sets(
        relevance, capacities, 8, total, seed_parts=(seed, client_id, "broad"), draws=50000,
        group_ids=fine_to_coarse, minimum_group_width=6,
    )
    max_score = max(max(x[0] for x in narrow), max(x[0] for x in broad))
    floor = 0.25 * max_score
    narrow = [x for x in narrow if x[0] >= floor]
    broad = [x for x in broad if x[0] >= floor]
    if not narrow or not broad:
        raise RuntimeError("Related companion allocation lost all positive-score candidates")

    narrow_scores = np.asarray([x[0] for x in narrow], dtype=np.float64)
    best = None
    for broad_item in broad:
        position = int(np.searchsorted(narrow_scores, broad_item[0]))
        for index in {max(0, position - 1), min(len(narrow) - 1, position)}:
            narrow_item = narrow[index]
            proposal = (
                abs(narrow_item[0] - broad_item[0]),
                -min(narrow_item[0], broad_item[0]),
                narrow_item[1],
                broad_item[1],
                narrow_item,
                broad_item,
            )
            if best is None or proposal[:4] < best[:4]:
                best = proposal
    assert best is not None
    if float(best[0]) > float(tolerance):
        raise RuntimeError(
            f"Cannot match narrow/broad relatedness for client {client_id}: diff={best[0]:.6f}"
        )
    unrelated = min(
        _candidate_sets(
            relevance, capacities, 8, total,
            seed_parts=(seed, client_id, "unrelated"), draws=50000,
            group_ids=fine_to_coarse, minimum_group_width=6,
        ),
        key=lambda x: (x[0], x[1]),
    )
    return {
        "narrow_related": best[4],
        "broad_related": best[5],
        "broad_unrelated": unrelated,
    }


def _owner_map(partition: Mapping[int, np.ndarray]) -> dict[int, int]:
    return {int(index): int(client) for client, values in partition.items() for index in values.tolist()}


def _choose_positions(
    labels: np.ndarray,
    owner: Mapping[int, int],
    allocation: tuple[float, tuple[int, ...], tuple[int, ...]],
    used: set[int],
    *seed_parts,
) -> list[int]:
    _, classes, quotas = allocation
    selected = []
    for class_id, quota in zip(classes, quotas):
        pool = [
            int(index) for index in np.flatnonzero(labels == int(class_id)).tolist()
            if owner[int(index)] not in TAIL_CLIENT_IDS and int(index) not in used
        ]
        if len(pool) < int(quota):
            raise RuntimeError(f"Class {class_id} has only {len(pool)} unused head-client samples for quota {quota}")
        generator = np.random.default_rng(stable_seed("e2b-sample-choice", *seed_parts, int(class_id)))
        chosen = [pool[int(i)] for i in generator.choice(len(pool), size=int(quota), replace=False)]
        selected.extend(chosen)
        used.update(chosen)
    return selected


def _swap_companions(
    labels: np.ndarray,
    base: Mapping[int, np.ndarray],
    allocations: Mapping[int, tuple[float, tuple[int, ...], tuple[int, ...]]],
    seed: int,
    condition: str,
) -> dict[int, np.ndarray]:
    owner = _owner_map(base)
    mutable = {client: list(np.asarray(values, dtype=np.int64).tolist()) for client, values in base.items()}
    used: set[int] = set()
    for client_id in TAIL_CLIENT_IDS:
        old = sorted(index for index in mutable[client_id] if int(labels[index]) not in TAIL_CLASSES)
        new = _choose_positions(labels, owner, allocations[client_id], used, seed, condition, client_id)
        if len(old) != len(new):
            raise RuntimeError(f"Client {client_id} companion swap differs: {len(old)} != {len(new)}")
        for old_index, new_index in zip(old, sorted(new)):
            donor = owner[new_index]
            mutable[client_id].remove(old_index)
            mutable[client_id].append(new_index)
            mutable[donor].remove(new_index)
            mutable[donor].append(old_index)
    result = {client: np.asarray(sorted(values), dtype=np.int64) for client, values in mutable.items()}
    _validate_partition(labels, result, condition=condition)
    for client_id in range(NUM_CLIENTS):
        if len(result[client_id]) != len(base[client_id]):
            raise RuntimeError(f"{condition}: client {client_id} size changed")
    for client_id in TAIL_CLIENT_IDS:
        before_tail = sorted(index for index in base[client_id] if int(labels[index]) in TAIL_CLASSES)
        after_tail = sorted(index for index in result[client_id] if int(labels[index]) in TAIL_CLASSES)
        if before_tail != after_tail:
            raise RuntimeError(f"{condition}: tail samples moved for specialist {client_id}")
    return result


def _manifest_rows(
    stage: str,
    seed: int,
    topology: str,
    condition: str,
    partition: Mapping[int, np.ndarray],
    labels: np.ndarray,
    raw_ids: np.ndarray,
    neighbors: Mapping[int, Sequence[int]],
) -> tuple[list[dict], list[dict], list[dict]]:
    partition_hash = _partition_hash(partition)
    sample_rows, support_rows, execution_rows = [], [], []
    for client_id in range(NUM_CLIENTS):
        indices = np.asarray(partition[client_id], dtype=np.int64)
        counts = _counts(labels, indices)
        tail_count = int(counts[TAIL_CLASSES].sum())
        companion_count = int(len(indices) - tail_count)
        relevance = _tail_relevance_by_class(counts, neighbors)
        present = counts > 0
        for lt_index in indices.tolist():
            label = int(labels[lt_index])
            sample_rows.append({
                "stage": stage, "data_seed": int(seed), "topology": topology,
                "condition": condition, "client_id": client_id, "lt_index": int(lt_index),
                "raw_train_index": int(raw_ids[lt_index]), "base_sample_id": f"train:{int(raw_ids[lt_index])}",
                "label": label, "is_tail": label in TAIL_CLASSES,
                "client_size": int(len(indices)), "client_tail_count": tail_count,
                "client_companion_count": companion_count, "partition_hash": partition_hash,
            })
        if tail_count <= 0:
            continue
        companion_classes = int(np.sum(present[:80]))
        for tail_class in TAIL_CLASSES:
            neighbor_values = [int(value) for value in neighbors[int(tail_class)]]
            coverage = sum((1.0 / rank) * int(present[class_id]) for rank, class_id in enumerate(neighbor_values, 1))
            support_rows.append({
                "stage": stage, "data_seed": int(seed), "topology": topology,
                "condition": condition, "client_id": client_id, "tail_class": tail_class,
                "tail_sample_count": int(counts[tail_class]), "supports_tail_class": int(counts[tail_class] > 0),
                "client_size": int(len(indices)), "client_tail_count": tail_count,
                "client_companion_count": companion_count,
                "tail_purity": tail_count / float(len(indices)),
                "companion_class_count": companion_classes,
                "tail_neighbor_access_score": float(coverage),
                "client_tail_mass_relatedness": float(np.dot(counts[:80] > 0, relevance[:80])),
                "partition_hash": partition_hash,
            })

        if stage == "e2a" or client_id in TAIL_CLIENT_IDS:
            # E2B tail positions are deliberately fixed across conditions. Sorting by
            # role keeps every tail sample in the same batch slot in all three arms.
            tail_indices = sorted(index for index in indices.tolist() if int(labels[index]) in TAIL_CLASSES)
            companion_indices = sorted(index for index in indices.tolist() if int(labels[index]) not in TAIL_CLASSES)
            for epoch in (1, 2, 3):
                if stage == "e2b":
                    tail_rng = np.random.default_rng(stable_seed("e2b-tail-order", seed, client_id, epoch))
                    comp_rng = np.random.default_rng(stable_seed("e2b-comp-order", seed, condition, client_id, epoch))
                    ordered = [tail_indices[i] for i in tail_rng.permutation(len(tail_indices))]
                    ordered += [companion_indices[i] for i in comp_rng.permutation(len(companion_indices))]
                else:
                    rng = np.random.default_rng(stable_seed("e2a-order", seed, topology, client_id, epoch))
                    ordered = [int(indices[i]) for i in rng.permutation(len(indices))]
                for position, lt_index in enumerate(ordered):
                    raw_id = int(raw_ids[lt_index])
                    execution_rows.append({
                        "stage": stage, "data_seed": int(seed), "topology": topology,
                        "condition": condition, "client_id": client_id, "epoch": epoch,
                        "batch_index": position // 32, "position_in_batch": position % 32,
                        "lt_index": int(lt_index), "base_sample_id": f"train:{raw_id}",
                        "label": int(labels[lt_index]),
                        "augmentation_seed": stable_seed("e2-sample-augmentation", seed, epoch, raw_id),
                        "partition_hash": partition_hash,
                    })
    return sample_rows, support_rows, execution_rows


def build_manifests(data_dir: Path, output_dir: Path, seeds: Sequence[int]) -> dict:
    labels, raw_ids, test_labels, class_names, fine_to_coarse = load_exact_lt_pool(data_dir)
    neighbors, neighbor_meta = load_preregistered_neighbors(TAIL_CLASSES)
    output_dir = Path(output_dir)
    sample_rows, support_rows, execution_rows, intervention_rows = [], [], [], []

    for seed in [int(value) for value in seeds]:
        base = _partitions(labels, test_labels, seed)
        for topology in ("dirichlet", "clientlt"):
            rows = _manifest_rows("e2a", seed, topology, "natural", base[topology], labels, raw_ids, neighbors)
            sample_rows.extend(rows[0]); support_rows.extend(rows[1]); execution_rows.extend(rows[2])

        owner = _owner_map(base["clientlt"])
        head_capacities = {
            class_id: sum(int(labels[index]) == class_id and owner[index] not in TAIL_CLIENT_IDS for index in range(len(labels)))
            for class_id in range(80)
        }
        condition_allocations: dict[str, dict[int, tuple[float, tuple[int, ...], tuple[int, ...]]]] = {
            name: {} for name in ("narrow_related", "broad_related", "broad_unrelated")
        }
        for client_id in TAIL_CLIENT_IDS:
            client_counts = _counts(labels, base["clientlt"][client_id])
            companion_count = int(client_counts[:80].sum())
            relevance = _taxonomy_relevance_by_class(client_counts, fine_to_coarse)
            allocations = _matched_allocations(
                relevance, head_capacities, companion_count, seed, client_id,
                float(frozen_protocol()["e2b"]["relatedness_match_tolerance"]),
                fine_to_coarse,
            )
            for condition, allocation in allocations.items():
                condition_allocations[condition][client_id] = allocation
                score, classes, quotas = allocation
                intervention_rows.append({
                    "data_seed": seed, "condition": condition, "client_id": client_id,
                    "companion_count": companion_count, "class_width": len(classes),
                    "companion_classes": json.dumps(list(classes), separators=(",", ":")),
                    "class_quotas": json.dumps(list(quotas), separators=(",", ":")),
                    "coarse_superclass_width": len({int(fine_to_coarse[c]) for c in classes}),
                    "coarse_superclasses": json.dumps(
                        sorted({int(fine_to_coarse[c]) for c in classes}), separators=(",", ":")
                    ),
                    "tail_mass_weighted_mean_relatedness": float(score),
                })
        for client_id in TAIL_CLIENT_IDS:
            narrow_score = condition_allocations["narrow_related"][client_id][0]
            broad_score = condition_allocations["broad_related"][client_id][0]
            if abs(narrow_score - broad_score) > float(frozen_protocol()["e2b"]["relatedness_match_tolerance"]):
                raise RuntimeError("E2B relatedness matching tolerance was violated")

        for condition in ("narrow_related", "broad_related", "broad_unrelated"):
            partition = _swap_companions(labels, base["clientlt"], condition_allocations[condition], seed, condition)
            rows = _manifest_rows("e2b", seed, "clientlt", condition, partition, labels, raw_ids, neighbors)
            sample_rows.extend(rows[0]); support_rows.extend(rows[1]); execution_rows.extend(rows[2])

    write_csv(output_dir / "partition_sample_manifest.csv", sample_rows)
    write_csv(output_dir / "client_tail_support_manifest.csv", support_rows)
    write_csv(output_dir / "local_execution_manifest.csv", execution_rows)
    write_csv(output_dir / "e2b_intervention_manifest.csv", intervention_rows)
    e2b_execution = [row for row in execution_rows if row["stage"] == "e2b"]
    execution_fairness = []
    for seed in [int(value) for value in seeds]:
        for client_id in TAIL_CLIENT_IDS:
            signatures = {}
            step_counts = {}
            for condition in ("narrow_related", "broad_related", "broad_unrelated"):
                rows = [
                    row for row in e2b_execution
                    if row["data_seed"] == seed and row["client_id"] == client_id
                    and row["condition"] == condition
                ]
                signatures[condition] = stable_hash(sorted(
                    (row["epoch"], row["batch_index"], row["position_in_batch"],
                     row["base_sample_id"], row["augmentation_seed"])
                    for row in rows if int(row["label"]) in TAIL_CLASSES
                ))
                step_counts[condition] = len(set((row["epoch"], row["batch_index"]) for row in rows))
            tail_equal = len(set(signatures.values())) == 1
            steps_equal = len(set(step_counts.values())) == 1
            if not tail_equal or not steps_equal:
                raise RuntimeError(
                    f"E2B execution fairness failed for seed={seed}, client={client_id}: "
                    f"tail_equal={tail_equal}, steps_equal={steps_equal}"
                )
            execution_fairness.append({
                "data_seed": seed, "client_id": client_id,
                "tail_execution_equal_across_conditions": tail_equal,
                "optimizer_steps_equal_across_conditions": steps_equal,
                "optimizer_steps": next(iter(step_counts.values())),
                "tail_execution_hash": next(iter(signatures.values())),
                "pass": True,
            })
    write_csv(output_dir / "e2b_execution_fairness.csv", execution_fairness)
    metadata = {
        "protocol": frozen_protocol(),
        "data_dir": str(Path(data_dir).resolve()),
        "data_seeds": [int(value) for value in seeds],
        "lt_sample_count": int(len(labels)),
        "tail_sample_count": int(np.isin(labels, TAIL_CLASSES).sum()),
        "class_names_hash": stable_hash(class_names),
        "fine_to_coarse": [int(value) for value in fine_to_coarse.tolist()],
        "lt_pool_hash": stable_hash([(int(raw), int(label)) for raw, label in zip(raw_ids, labels)]),
        "neighbor_metadata": neighbor_meta,
        "e2b_tail_execution_equal": True,
        "e2b_optimizer_steps_equal": True,
    }
    required = (
        "partition_sample_manifest.csv", "client_tail_support_manifest.csv",
        "local_execution_manifest.csv", "e2b_intervention_manifest.csv",
        "e2b_execution_fairness.csv",
    )
    metadata["manifest_hashes"] = {name: file_sha256(output_dir / name) for name in required}
    write_json(output_dir / "manifest_contract.json", metadata)
    return {
        "output_dir": str(output_dir.resolve()),
        "sample_rows": len(sample_rows), "support_rows": len(support_rows),
        "execution_rows": len(execution_rows), "intervention_rows": len(intervention_rows),
        "structural_gate": "PASS",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("DATA/cifar-100/cifar-100-python"))
    parser.add_argument("--output-dir", type=Path, default=Path("output/e2_client_update_audit/manifests"))
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    args = parser.parse_args()
    print(json.dumps(build_manifests(args.data_dir, args.output_dir, args.seeds)))


if __name__ == "__main__":
    main()
