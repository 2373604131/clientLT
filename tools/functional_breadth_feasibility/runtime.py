"""Forward-only Functional Breadth feasibility runtime.

The module intentionally contains no optimizer, backward, or test-split access.
It fails if any prerequisite state is absent instead of recreating it.
"""

from __future__ import annotations

import csv
import importlib.util
import io
import json
import pickle
import traceback
from collections import defaultdict
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

from tools.breadth_audit.inputs import load_preregistered_neighbors
from tools.carrier_access_audit.protocol import NON_TAIL_CLASSES, TAIL_CLASSES, frozen_protocol as carrier_protocol
from tools.carrier_access_audit.rewrite_protocol import frozen_rewrite_protocol
from tools.carrier_access_audit.rewrite_runtime import _writer_state_path
from tools.carrier_access_audit.runtime import _candidate_state_path, _combine_states
from tools.client_update_audit.runtime import _prepare_model, _repository_cifar_transform
from tools.functional_breadth_feasibility.matching import (
    coverage_metrics,
    enumerate_pair_screen,
    select_actual_match,
    shortlist_contrasts,
)
from tools.functional_breadth_feasibility.protocol import frozen_protocol, write_protocol
from tools.functional_breadth_feasibility.sampling import select_head_safety_ids
from tools.semantic_acquisition.common import (
    file_sha256,
    stable_hash,
    tensor_mapping_hash,
    write_csv,
    write_json,
)
from tools.semantic_acquisition.runtime import _predict_ids, flatten_named, load_lora_state


ROOT = Path(__file__).resolve().parents[2]


class TrainOnlyCifarRawStore:
    """Minimal store that never opens or materializes CIFAR's test file."""

    def __init__(self, data_dir: Path):
        data_dir = Path(data_dir)
        with (data_dir / "train").open("rb") as handle:
            train = pickle.load(handle, encoding="latin1")
        with (data_dir / "meta").open("rb") as handle:
            meta = pickle.load(handle, encoding="latin1")
        value = np.asarray(train["data"], dtype=np.uint8).reshape(-1, 3, 32, 32)
        self.train_images = value.transpose(0, 2, 3, 1)
        self.train_labels = np.asarray(train["fine_labels"], dtype=np.int64)
        self.class_names = [str(value).replace("_", " ") for value in meta["fine_label_names"]]

    def image(self, sample_id: str) -> Image.Image:
        split, raw = str(sample_id).split(":", 1)
        if split != "train":
            raise RuntimeError("Functional Breadth feasibility forbids non-train sample IDs")
        return Image.fromarray(self.train_images[int(raw)])


def _read_csv(path: Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _exact_lt_train_pool(store: TrainOnlyCifarRawStore) -> tuple[np.ndarray, np.ndarray]:
    spec = importlib.util.spec_from_file_location(
        "functional_breadth_long_tail", ROOT / "datasets" / "long_tail.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    by_class = [np.flatnonzero(store.train_labels == class_id).tolist() for class_id in range(100)]
    with redirect_stdout(io.StringIO()):
        _, selected = module.train_long_tail(by_class, 100, 0.01, "exp")
    raw_ids = np.asarray(module.flatten_list(selected), dtype=np.int64)
    labels = store.train_labels[raw_ids]
    if len(labels) != 10847 or int(np.isin(labels, TAIL_CLASSES).sum()) != 153:
        raise RuntimeError("Rebuilt CIFAR-100-LT train pool differs from the frozen parent protocol")
    return labels, raw_ids


def _verify_inputs(args) -> tuple[dict, pd.DataFrame, dict]:
    required = (
        args.manifest_dir / "manifest_contract.json",
        args.b_dir / "runtime_contract.json",
        args.d1_dir / "runtime_contract.json",
        args.theta0_file,
    )
    missing = [str(path) for path in required if not Path(path).is_file()]
    missing += [
        str(_candidate_state_path(args.b_dir, candidate_class))
        for candidate_class in NON_TAIL_CLASSES
        if not _candidate_state_path(args.b_dir, candidate_class).is_file()
    ]
    missing += [
        str(_writer_state_path(args.d1_dir, tail_class))
        for tail_class in TAIL_CLASSES
        if not _writer_state_path(args.d1_dir, tail_class).is_file()
    ]
    if missing:
        raise FileNotFoundError(
            "P1 is no-training and will not recreate missing prerequisites: " + ", ".join(missing)
        )
    manifest = json.loads((args.manifest_dir / "manifest_contract.json").read_text(encoding="utf-8"))
    if manifest.get("protocol") != carrier_protocol():
        raise RuntimeError("Carrier-B manifest protocol mismatch")
    for name, expected in manifest.get("manifest_hashes", {}).items():
        if file_sha256(args.manifest_dir / name) != expected:
            raise RuntimeError(f"Carrier-B manifest hash mismatch: {name}")
    b_contract = json.loads((args.b_dir / "runtime_contract.json").read_text(encoding="utf-8"))
    if (
        b_contract.get("stage") != "B"
        or b_contract.get("protocol") != carrier_protocol()
        or int(b_contract.get("completed_transfer_pairs", -1)) != 1600
    ):
        raise RuntimeError("Carrier-B runtime is incomplete or incompatible")
    d1_contract = json.loads((args.d1_dir / "runtime_contract.json").read_text(encoding="utf-8"))
    if int(d1_contract.get("completed_writer_classes", -1)) != 20:
        raise RuntimeError("D1 runtime is incomplete")
    if d1_contract.get("protocol") != frozen_rewrite_protocol():
        raise RuntimeError("D1 protocol mismatch")
    execution = pd.read_csv(args.manifest_dir / "training_execution.csv")
    return manifest, execution, {"b": b_contract, "d1": d1_contract}


def _private_tail_manifest(execution: pd.DataFrame) -> pd.DataFrame:
    rows = execution[execution.role == "private_tail_evidence"].drop_duplicates(
        ["class_id", "base_sample_id"]
    ).sort_values(["class_id", "slot"])
    counts = rows.groupby("class_id").size().to_dict()
    if counts != {class_id: 5 for class_id in TAIL_CLASSES}:
        raise RuntimeError("Expected five frozen private-tail train samples for every tail class")
    return rows


def _head_safety_manifest(
    store: TrainOnlyCifarRawStore, execution: pd.DataFrame, samples_per_class: int
) -> list[dict]:
    _, lt_raw_ids = _exact_lt_train_pool(store)
    used = set(
        execution.drop_duplicates("base_sample_id").base_sample_id.astype(str).tolist()
    )
    selected = select_head_safety_ids(
        store.train_labels, lt_raw_ids, used, NON_TAIL_CLASSES, samples_per_class
    )
    lt_raw_set = {int(value) for value in lt_raw_ids.tolist()}
    output = []
    for class_id in NON_TAIL_CLASSES:
        for slot, raw_id in enumerate(selected[class_id]):
            output.append({
                "data_seed": 42, "role": "heldout_head_safety", "class_id": class_id,
                "slot": slot, "raw_train_index": raw_id,
                "base_sample_id": f"train:{raw_id}", "label": class_id,
                "excluded_from_all_carrier_b_training": True,
                "outside_federated_lt_pool": bool(raw_id not in lt_raw_set),
            })
    return output


def _true_margin(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    rows = torch.arange(labels.numel())
    true = logits[rows, labels]
    negatives = logits.clone()
    negatives[rows, labels] = -torch.inf
    return true - negatives.max(dim=1).values


def _cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    denominator = float(left.norm().item() * right.norm().item())
    return float(torch.dot(left, right).item() / denominator) if denominator > 0 else 0.0


def _load_states(args, theta0, names, execution: pd.DataFrame, d1_contract: dict):
    theta_vector = flatten_named(theta0, names)
    theta_hash = tensor_mapping_hash(theta0)
    candidates, deltas, inventory = {}, {}, []
    for candidate_class in NON_TAIL_CLASSES:
        path = _candidate_state_path(args.b_dir, candidate_class)
        if not path.is_file():
            raise FileNotFoundError(
                f"Missing saved Carrier-B state (retraining is forbidden): {path}"
            )
        saved = torch.load(path, map_location="cpu")
        expected_sha = d1_contract.get("candidate_state_hashes", {}).get(str(candidate_class))
        if expected_sha and file_sha256(path) != expected_sha:
            raise RuntimeError(f"Carrier-B state differs from the D1-frozen state: {path}")
        if int(saved.get("candidate_class", -1)) != candidate_class or saved.get("theta0_hash") != theta_hash:
            raise RuntimeError(f"Stale Carrier-B state: {path}")
        state = saved["lora_state"]
        vector = flatten_named(state, names) - theta_vector
        meta = saved.get("train_meta", {})
        sample_count = int(execution[
            (execution.role == "candidate") & (execution.class_id == candidate_class)
        ].base_sample_id.nunique())
        candidates[candidate_class] = state
        deltas[candidate_class] = vector
        inventory.append({
            "candidate_class": candidate_class, "state_path": str(path.resolve()),
            "state_sha256": file_sha256(path), "theta0_hash": theta_hash,
            "update_l2": float(vector.norm().item()), "candidate_sample_count": sample_count,
            "optimizer_steps_successful": int(meta.get("optimizer_steps_successful", -1)),
            "scheduler_steps": int(meta.get("scheduler_steps", -1)),
            "amp_overflow_count": int(meta.get("amp_overflow_count", -1)),
            "source_reused_without_training": True,
        })
    writers, writer_deltas = {}, {}
    for tail_class in TAIL_CLASSES:
        path = _writer_state_path(args.d1_dir, tail_class)
        if not path.is_file():
            raise FileNotFoundError(
                f"Missing saved D1 direct-tail state (retraining is forbidden): {path}"
            )
        saved = torch.load(path, map_location="cpu")
        expected_sha = d1_contract.get("writer_state_hashes", {}).get(str(tail_class))
        if expected_sha and file_sha256(path) != expected_sha:
            raise RuntimeError(f"D1 writer state hash mismatch: {path}")
        if int(saved.get("tail_class", -1)) != tail_class or saved.get("theta0_hash") != theta_hash:
            raise RuntimeError(f"Stale D1 direct-tail state: {path}")
        writers[tail_class] = saved["lora_state"]
        writer_deltas[tail_class] = flatten_named(saved["lora_state"], names) - theta_vector
    return candidates, deltas, inventory, writers, writer_deltas


def _boundary_rows(
    baseline_logits: torch.Tensor,
    updated_logits: torch.Tensor,
    private_rows: pd.DataFrame,
    neighbors: dict[int, list[int]],
    candidate_class: int,
) -> list[dict]:
    output = []
    offset = 0
    for tail_class in TAIL_CLASSES:
        count = int((private_rows.class_id == tail_class).sum())
        before, after = baseline_logits[offset:offset + count], updated_logits[offset:offset + count]
        for rank, neighbor in enumerate(neighbors[tail_class], start=1):
            before_values = before[:, tail_class] - before[:, neighbor]
            after_values = after[:, tail_class] - after[:, neighbor]
            output.append({
                "tail_class": tail_class, "candidate_class": candidate_class,
                "boundary_neighbor_class": neighbor, "semantic_neighbor_rank": rank,
                "private_sample_count": count,
                "baseline_boundary_margin": float(before_values.mean().item()),
                "updated_boundary_margin": float(after_values.mean().item()),
                "private_boundary_gain": float((after_values - before_values).mean().item()),
                "selection_split": "private_train",
            })
        offset += count
    return output


def _actual_pair_metrics(
    pair: tuple[int, int], state: dict[str, torch.Tensor], model,
    store, transform, private_rows: pd.DataFrame, head_rows: list[dict],
    baseline_private_logits: torch.Tensor, baseline_head_logits: torch.Tensor,
    neighbors: dict[int, list[int]], writer_deltas: dict[int, torch.Tensor],
    theta0, names, inventory_by_candidate,
) -> tuple[dict[int, dict], list[dict]]:
    load_lora_state(model, state)
    private_logits, _ = _predict_ids(
        model, store, transform, private_rows.base_sample_id.tolist(),
        private_rows.label.astype(int).tolist(), batch_size=64,
    )
    head_labels = torch.as_tensor([int(row["label"]) for row in head_rows], dtype=torch.long)
    head_logits, _ = _predict_ids(
        model, store, transform, [row["base_sample_id"] for row in head_rows],
        head_labels.tolist(), batch_size=64,
    )
    head_margin_gain = float(
        (_true_margin(head_logits, head_labels) - _true_margin(baseline_head_logits, head_labels)).mean().item()
    )
    head_acc_gain = float(
        (head_logits.argmax(1) == head_labels).float().mean().item()
        - (baseline_head_logits.argmax(1) == head_labels).float().mean().item()
    )
    theta_vector = flatten_named(theta0, names)
    pair_delta = flatten_named(state, names) - theta_vector
    update_l2 = float(pair_delta.norm().item())
    metrics, gain_rows = {}, []
    offset = 0
    for tail_class in TAIL_CLASSES:
        count = int((private_rows.class_id == tail_class).sum())
        before, after = baseline_private_logits[offset:offset + count], private_logits[offset:offset + count]
        gains = []
        for rank, neighbor in enumerate(neighbors[tail_class], start=1):
            before_values = before[:, tail_class] - before[:, neighbor]
            after_values = after[:, tail_class] - after[:, neighbor]
            gain = float((after_values - before_values).mean().item())
            gains.append(gain)
            gain_rows.append({
                "tail_class": tail_class, "candidate_a": pair[0], "candidate_b": pair[1],
                "boundary_neighbor_class": neighbor, "semantic_neighbor_rank": rank,
                "private_sample_count": count, "actual_merged_boundary_gain": gain,
                "selection_split": "private_train",
            })
        metrics[tail_class] = {
            "tail_class": tail_class, "candidate_a": pair[0], "candidate_b": pair[1],
            **{f"actual_{key}": value for key, value in coverage_metrics(gains).items()},
            "actual_head_margin_gain": head_margin_gain, "actual_head_accuracy_gain": head_acc_gain,
            "update_l2": update_l2, "direct_tail_cosine": _cosine(pair_delta, writer_deltas[tail_class]),
            "candidate_sample_count": sum(
                inventory_by_candidate[value]["candidate_sample_count"] for value in pair
            ),
            "optimizer_steps": sum(
                inventory_by_candidate[value]["optimizer_steps_successful"] for value in pair
            ),
        }
        offset += count
    return metrics, gain_rows


def run(args) -> dict:
    args.output_dir = Path(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    protocol_path = write_protocol(args.output_dir)
    manifest_contract, execution, contracts = _verify_inputs(args)
    store = TrainOnlyCifarRawStore(args.data_dir)
    if not Path(args.theta0_file).is_file():
        raise FileNotFoundError("P1 refuses to initialize a missing theta0")
    _, model, theta0, names = _prepare_model(args, store)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.eval()
    if contracts["b"].get("theta0_hash") != tensor_mapping_hash(theta0):
        raise RuntimeError("Carrier-B and P1 theta0 differ")
    transform = _repository_cifar_transform()
    neighbors, neighbor_metadata = load_preregistered_neighbors(TAIL_CLASSES)
    private_rows = _private_tail_manifest(execution)
    head_rows = _head_safety_manifest(
        store, execution, int(frozen_protocol()["evidence"]["head_safety_samples_per_class"])
    )
    write_csv(args.output_dir / "head_safety_manifest.csv", head_rows)
    write_csv(args.output_dir / "hard_boundary_manifest.csv", [
        {"tail_class": tail, "boundary_neighbor_class": neighbor, "semantic_neighbor_rank": rank}
        for tail in TAIL_CLASSES for rank, neighbor in enumerate(neighbors[tail], start=1)
    ])
    candidates, deltas, inventory, _, writer_deltas = _load_states(
        args, theta0, names, execution, contracts["d1"]
    )
    inventory_by_candidate = {int(row["candidate_class"]): row for row in inventory}
    write_csv(args.output_dir / "candidate_state_inventory.csv", inventory)
    tensor_artifact = args.output_dir / "candidate_update_tensors.pt"
    torch.save({
        "theta0_hash": tensor_mapping_hash(theta0), "parameter_names": names,
        "candidate_deltas": {
            candidate: {name: candidates[candidate][name] - theta0[name] for name in names}
            for candidate in NON_TAIL_CLASSES
        },
        "source": "saved Carrier-B states; no training performed",
    }, tensor_artifact)

    load_lora_state(model, theta0)
    baseline_private_logits, _ = _predict_ids(
        model, store, transform, private_rows.base_sample_id.tolist(),
        private_rows.label.astype(int).tolist(), batch_size=args.eval_batch_size,
    )
    head_labels = [int(row["label"]) for row in head_rows]
    baseline_head_logits, _ = _predict_ids(
        model, store, transform, [row["base_sample_id"] for row in head_rows],
        head_labels, batch_size=args.eval_batch_size,
    )
    head_label_tensor = torch.as_tensor(head_labels, dtype=torch.long)
    baseline_head_margin = _true_margin(baseline_head_logits, head_label_tensor)
    boundary_gain_rows, head_gain_rows, cosine_rows = [], [], []
    for candidate_class in NON_TAIL_CLASSES:
        load_lora_state(model, candidates[candidate_class])
        private_logits, _ = _predict_ids(
            model, store, transform, private_rows.base_sample_id.tolist(),
            private_rows.label.astype(int).tolist(), batch_size=args.eval_batch_size,
        )
        boundary_gain_rows.extend(_boundary_rows(
            baseline_private_logits, private_logits, private_rows, neighbors, candidate_class
        ))
        head_logits, _ = _predict_ids(
            model, store, transform, [row["base_sample_id"] for row in head_rows],
            head_labels, batch_size=args.eval_batch_size,
        )
        head_margin_gain = float(
            (_true_margin(head_logits, head_label_tensor) - baseline_head_margin).mean().item()
        )
        head_gain_rows.append({
            "candidate_class": candidate_class, "head_sample_count": len(head_rows),
            "private_head_margin_gain": head_margin_gain,
            "private_head_accuracy_gain": float(
                (head_logits.argmax(1) == head_label_tensor).float().mean().item()
                - (baseline_head_logits.argmax(1) == head_label_tensor).float().mean().item()
            ),
            "private_head_harm": max(-head_margin_gain, 0.0),
            "selection_split": "private_train",
        })
        for tail_class in TAIL_CLASSES:
            cosine_rows.append({
                "tail_class": tail_class, "candidate_class": candidate_class,
                "candidate_direct_tail_cosine": _cosine(
                    deltas[candidate_class], writer_deltas[tail_class]
                ),
            })
        print(json.dumps({"stage": "P1-candidate-forward", "candidate_class": candidate_class}))
    write_csv(args.output_dir / "private_boundary_gains.csv", boundary_gain_rows)
    write_csv(args.output_dir / "private_head_safety.csv", head_gain_rows)
    write_csv(args.output_dir / "candidate_direct_tail_cosine.csv", cosine_rows)

    gains_lookup = defaultdict(dict)
    for row in boundary_gain_rows:
        gains_lookup[(int(row["tail_class"]), int(row["candidate_class"]))][
            int(row["semantic_neighbor_rank"])
        ] = float(row["private_boundary_gain"])
    head_lookup = {int(row["candidate_class"]): float(row["private_head_margin_gain"]) for row in head_gain_rows}
    cosine_lookup = {
        (int(row["tail_class"]), int(row["candidate_class"])): float(row["candidate_direct_tail_cosine"])
        for row in cosine_rows
    }
    pair_norms = {}
    for left in NON_TAIL_CLASSES:
        for right in range(left + 1, 80):
            pair_norms[(left, right)] = float((0.5 * deltas[left] + 0.5 * deltas[right]).norm().item())
    pair_screen, proposals = [], []
    shortlist_count = int(frozen_protocol()["merge"]["shortlist_contrasts_per_tail"])
    for tail_class in TAIL_CLASSES:
        vectors = {
            candidate: np.asarray(
                [gains_lookup[(tail_class, candidate)][rank] for rank in range(1, 11)],
                dtype=np.float64,
            ) for candidate in NON_TAIL_CLASSES
        }
        rows = enumerate_pair_screen(
            tail_class, vectors, pair_norms, head_lookup,
            {candidate: cosine_lookup[(tail_class, candidate)] for candidate in NON_TAIL_CLASSES},
            {candidate: inventory_by_candidate[candidate]["candidate_sample_count"] for candidate in NON_TAIL_CLASSES},
            {candidate: inventory_by_candidate[candidate]["optimizer_steps_successful"] for candidate in NON_TAIL_CLASSES},
        )
        pair_screen.extend(rows)
        proposals.extend(shortlist_contrasts(rows, shortlist_count))
    write_csv(args.output_dir / "pair_screen.csv", pair_screen)
    write_csv(args.output_dir / "contrast_shortlist.csv", proposals)

    unique_pairs = sorted({
        tuple(sorted(pair)) for proposal in proposals for pair in (
            (proposal["broad_a"], proposal["broad_b"]),
            (proposal["narrow_a"], proposal["narrow_b"]),
        )
    })
    needed_tail_by_pair = defaultdict(set)
    for proposal in proposals:
        needed_tail_by_pair[tuple(sorted((proposal["broad_a"], proposal["broad_b"])))].add(int(proposal["tail_class"]))
        needed_tail_by_pair[tuple(sorted((proposal["narrow_a"], proposal["narrow_b"])))].add(int(proposal["tail_class"]))
    actual_lookup, actual_summary_rows, actual_boundary_rows = {}, [], []
    for index, pair in enumerate(unique_pairs, start=1):
        state = _combine_states(theta0, [(candidates[pair[0]], 0.5), (candidates[pair[1]], 0.5)])
        all_metrics, gain_rows = _actual_pair_metrics(
            pair, state, model, store, transform, private_rows, head_rows,
            baseline_private_logits, baseline_head_logits, neighbors, writer_deltas,
            theta0, names, inventory_by_candidate,
        )
        for tail_class in sorted(needed_tail_by_pair[pair]):
            row = all_metrics[tail_class]
            actual_lookup[(tail_class, pair[0], pair[1])] = row
            actual_summary_rows.append(row)
            actual_boundary_rows.extend([
                gain for gain in gain_rows if int(gain["tail_class"]) == tail_class
            ])
        print(json.dumps({"stage": "P1-actual-merge", "pair": pair, "index": index, "total": len(unique_pairs)}))
    write_csv(args.output_dir / "actual_merged_summary.csv", actual_summary_rows)
    write_csv(args.output_dir / "actual_merged_boundary_gains.csv", actual_boundary_rows)

    matched_rows = []
    for tail_class in TAIL_CLASSES:
        chosen = select_actual_match(
            [row for row in proposals if int(row["tail_class"]) == tail_class],
            actual_lookup, frozen_protocol(),
        )
        chosen["selection_used_test_metrics"] = False
        matched_rows.append(chosen)
    write_csv(args.output_dir / "matched_broad_narrow_pairs.csv", matched_rows)
    pass_count = sum(bool(row["matched_pair_pass"]) for row in matched_rows)
    minimum = int(frozen_protocol()["feasibility_gate"]["minimum_tail_classes_with_matched_broad_narrow_pair"])
    verdict = "FEASIBLE" if pass_count >= minimum else ("PARTIAL" if pass_count > 0 else "INFEASIBLE")
    result_names = (
        "head_safety_manifest.csv", "hard_boundary_manifest.csv", "candidate_state_inventory.csv",
        "candidate_update_tensors.pt", "private_boundary_gains.csv", "private_head_safety.csv",
        "candidate_direct_tail_cosine.csv", "pair_screen.csv", "contrast_shortlist.csv",
        "actual_merged_summary.csv", "actual_merged_boundary_gains.csv", "matched_broad_narrow_pairs.csv",
    )
    summary = {
        "schema_version": "functional_breadth_feasibility_v2",
        "verdict": verdict, "matched_tail_classes": pass_count,
        "required_tail_classes": minimum, "tail_classes_total": 20,
        "training_performed": False, "optimizer_steps": 0, "gradient_calls": 0,
        "test_split_accessed": False, "selection_used_test_metrics": False,
        "server_deployable_method": False, "privacy_claim": False,
        "candidate_states_reused": 80, "direct_tail_states_reused": 20,
        "candidate_pairs_screened_per_tail": 3160,
        "actual_unique_pair_states_evaluated": len(unique_pairs),
        "protocol_hash": frozen_protocol()["protocol_hash"],
        "protocol_file_sha256": file_sha256(protocol_path),
        "parent_manifest_contract_hash": file_sha256(args.manifest_dir / "manifest_contract.json"),
        "parent_b_contract_hash": file_sha256(args.b_dir / "runtime_contract.json"),
        "parent_d1_contract_hash": file_sha256(args.d1_dir / "runtime_contract.json"),
        "theta0_hash": tensor_mapping_hash(theta0), "neighbor_metadata": neighbor_metadata,
        "candidate_update_tensor_sha256": file_sha256(tensor_artifact),
        "result_hashes": {name: file_sha256(args.output_dir / name) for name in result_names},
        "interpretation": (
            "FEASIBLE means the saved shared-LoRA candidate library contains private-train matched "
            "Broad/Narrow pairs. It does not yet show that breadth improves adaptation or retention."
        ),
    }
    write_json(args.output_dir / "p1_summary.json", summary)
    lines = [
        "# Phase 1 — Functional Breadth feasibility", "",
        f"- Verdict: **{verdict}**",
        f"- Matched tail classes: **{pass_count}/20** (gate: at least {minimum})",
        f"- Actual merged pair states evaluated: **{len(unique_pairs)}**",
        "- Training / gradient / optimizer calls: **0 / 0 / 0**",
        "- Test split accessed: **no**", "",
        "A pass only authorizes the next Broad/Narrow adaptation gate; it is not evidence of test improvement.",
    ]
    (args.output_dir / "p1_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def guarded_run(args) -> dict:
    failure = Path(args.output_dir) / "failure.json"
    if failure.is_file():
        failure.unlink()
    try:
        return run(args)
    except Exception as exc:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        write_json(failure, {
            "stage": "P1", "error_type": type(exc).__name__, "error": str(exc),
            "training_was_not_attempted": True, "traceback": traceback.format_exc(),
        })
        raise
