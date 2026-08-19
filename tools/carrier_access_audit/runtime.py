"""CUDA runtime for the controlled candidate-transfer (B) and placement (C) audits."""

from __future__ import annotations

import argparse
import csv
import json
import traceback
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

from tools.breadth_audit.inputs import load_preregistered_neighbors
from tools.breadth_audit.metrics import neighbor_discrimination_metrics
from tools.carrier_access_audit.protocol import NON_TAIL_CLASSES, TAIL_CLASSES, frozen_protocol
from tools.client_update_audit.runtime import (
    _prepare_model,
    _repository_cifar_transform,
)
from tools.semantic_acquisition.common import (
    file_sha256,
    tensor_mapping_hash,
    write_csv,
    write_json,
)
from tools.semantic_acquisition.metrics import classification_metrics
from tools.semantic_acquisition.runtime import (
    CifarRawStore,
    _load_cliplora_api,
    _materialize,
    _predict_ids,
    _train_client,
    flatten_named,
    load_lora_state,
    lora_state,
    trainable_named,
)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _verify_manifests(manifest_dir: Path) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    manifest_dir = Path(manifest_dir)
    contract = json.loads((manifest_dir / "manifest_contract.json").read_text(encoding="utf-8"))
    if contract.get("protocol") != frozen_protocol():
        raise RuntimeError("Carrier-access manifest protocol differs from the checked-in protocol")
    for name, expected in contract["manifest_hashes"].items():
        if file_sha256(manifest_dir / name) != expected:
            raise RuntimeError(f"Manifest hash mismatch: {name}")
    execution = pd.read_csv(manifest_dir / "training_execution.csv")
    semantic = pd.read_csv(manifest_dir / "semantic_pairs.csv")
    return contract, execution, semantic


def _combine_states(theta0, states_and_coefficients):
    result = {name: theta0[name].clone() for name in sorted(theta0)}
    for state, coefficient in states_and_coefficients:
        for name in result:
            result[name] += float(coefficient) * (state[name] - theta0[name])
    return result


def _update_norm(theta0, state) -> float:
    names = sorted(theta0)
    return float((flatten_named(state, names) - flatten_named(theta0, names)).norm().item())


def _target_rows(store: CifarRawStore, split: str, class_id: int, manifested: pd.DataFrame | None = None):
    if split == "test":
        raw = np.flatnonzero(store.test_labels == int(class_id)).tolist()
        return [f"test:{index}" for index in raw], [int(class_id)] * len(raw)
    if manifested is None:
        raise ValueError("Train evaluation requires manifested private samples")
    rows = manifested[manifested.class_id == int(class_id)].sort_values("slot")
    return rows.base_sample_id.tolist(), rows.label.astype(int).tolist()


def _target_metrics(model, store, transform, ids, labels, neighbors, class_id, batch_size):
    logits, checked = _predict_ids(model, store, transform, ids, labels, batch_size=batch_size)
    basic = classification_metrics(logits, checked, class_id)
    neighbor = neighbor_discrimination_metrics(
        logits.numpy(), checked.numpy(), {int(class_id): neighbors[int(class_id)]}, [int(class_id)]
    )[0]
    return basic, neighbor, logits


def _baseline_bundle(model, store, transform, private_samples, neighbors, batch_size):
    private, test = {}, {}
    for class_id in TAIL_CLASSES:
        ids, labels = _target_rows(store, "train", class_id, private_samples)
        private[class_id] = _target_metrics(
            model, store, transform, ids, labels, neighbors, class_id, batch_size
        )[:2]
        ids, labels = _target_rows(store, "test", class_id)
        test[class_id] = _target_metrics(
            model, store, transform, ids, labels, neighbors, class_id, batch_size
        )[:2]
    return private, test


def _gain_row(before, after, prefix: str) -> dict:
    before_basic, before_neighbor = before
    after_basic, after_neighbor = after
    return {
        f"{prefix}_accuracy": float(after_basic["accuracy"]),
        f"{prefix}_accuracy_gain": float(after_basic["accuracy"] - before_basic["accuracy"]),
        f"{prefix}_margin": float(after_basic["margin"]),
        f"{prefix}_margin_gain": float(after_basic["margin"] - before_basic["margin"]),
        f"{prefix}_nll": float(after_basic["nll"]),
        f"{prefix}_nll_gain": float(before_basic["nll"] - after_basic["nll"]),
        f"{prefix}_worst_neighbor_margin": float(after_neighbor["worst_neighbor_margin"]),
        f"{prefix}_worst_neighbor_margin_gain": float(
            after_neighbor["worst_neighbor_margin"] - before_neighbor["worst_neighbor_margin"]
        ),
        f"{prefix}_positive_neighbor_coverage": float(after_neighbor["positive_margin_neighbor_coverage"]),
        f"{prefix}_positive_neighbor_coverage_gain": float(
            after_neighbor["positive_margin_neighbor_coverage"]
            - before_neighbor["positive_margin_neighbor_coverage"]
        ),
    }


def _candidate_state_path(output_dir: Path, candidate_class: int) -> Path:
    return Path(output_dir) / "candidate_states" / f"candidate_{int(candidate_class):02d}.pt"


def run_b(args) -> dict:
    contract, execution, semantic = _verify_manifests(args.manifest_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    store = CifarRawStore(args.data_dir)
    cfg, model, theta0, names = _prepare_model(args, store)
    transform = _repository_cifar_transform()
    neighbors, neighbor_meta = load_preregistered_neighbors(TAIL_CLASSES)
    private_samples = execution[execution.role == "private_tail_evidence"].drop_duplicates(
        ["class_id", "base_sample_id"]
    )
    load_lora_state(model, theta0)
    baseline_private, baseline_test = _baseline_bundle(
        model, store, transform, private_samples, neighbors, args.eval_batch_size
    )

    previous_rows = _read_csv(args.output_dir / "transfer_matrix.csv") if (args.output_dir / "transfer_matrix.csv").is_file() else []
    rows_by_candidate = {}
    for row in previous_rows:
        rows_by_candidate.setdefault(int(row["candidate_class"]), []).append(row)
    transfer_rows = []
    fairness_rows = []
    semantic_lookup = {
        (int(row.tail_class), int(row.candidate_class)): row
        for row in semantic.itertuples(index=False)
    }
    theta_vector = flatten_named(theta0, names)
    for candidate_class in NON_TAIL_CLASSES:
        state_path = _candidate_state_path(args.output_dir, candidate_class)
        complete = len(rows_by_candidate.get(candidate_class, [])) == len(TAIL_CLASSES) and state_path.is_file()
        if complete:
            saved = torch.load(state_path, map_location="cpu")
            if saved.get("theta0_hash") != tensor_mapping_hash(theta0):
                raise RuntimeError(f"Stale candidate state for class {candidate_class}")
            candidate_state = saved["lora_state"]
            train_meta = saved["train_meta"]
            transfer_rows.extend(rows_by_candidate[candidate_class])
            status = "resumed"
        else:
            unit_execution = execution[
                (execution.role == "candidate") & (execution.class_id == candidate_class)
            ].copy()
            states, train_meta = _train_client(model, cfg, theta0, unit_execution, store, transform)
            candidate_state = states[3]
            state_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                "candidate_class": candidate_class,
                "theta0_hash": tensor_mapping_hash(theta0),
                "lora_state": candidate_state,
                "train_meta": train_meta,
            }, state_path)
            load_lora_state(model, candidate_state)
            for tail_class in TAIL_CLASSES:
                private_ids, private_labels = _target_rows(store, "train", tail_class, private_samples)
                test_ids, test_labels = _target_rows(store, "test", tail_class)
                private_after = _target_metrics(
                    model, store, transform, private_ids, private_labels,
                    neighbors, tail_class, args.eval_batch_size,
                )[:2]
                test_after = _target_metrics(
                    model, store, transform, test_ids, test_labels,
                    neighbors, tail_class, args.eval_batch_size,
                )[:2]
                semantic_row = semantic_lookup[(tail_class, candidate_class)]
                transfer_rows.append({
                    "data_seed": 42,
                    "tail_class": tail_class,
                    "candidate_class": candidate_class,
                    "semantic_rank": int(semantic_row.semantic_rank),
                    "cosine_similarity": float(semantic_row.cosine_similarity),
                    "related_top10": bool(semantic_row.related_top10),
                    "unrelated_bottom10": bool(semantic_row.unrelated_bottom10),
                    **_gain_row(baseline_private[tail_class], private_after, "private"),
                    **_gain_row(baseline_test[tail_class], test_after, "test"),
                })
            status = "trained"
        final_vector = flatten_named(candidate_state, names)
        fairness_rows.append({
            "data_seed": 42,
            "candidate_class": candidate_class,
            "candidate_sample_count": int(
                execution[(execution.role == "candidate") & (execution.class_id == candidate_class)]
                .drop_duplicates("base_sample_id").shape[0]
            ),
            "theta0_hash": tensor_mapping_hash(theta0),
            "optimizer_steps_successful": int(train_meta["optimizer_steps_successful"]),
            "scheduler_steps": int(train_meta["scheduler_steps"]),
            "amp_overflow_count": int(train_meta["amp_overflow_count"]),
            "lora_update_l2": float((final_vector - theta_vector).norm().item()),
            "pass": bool(
                train_meta["optimizer_steps_successful"] == 3
                and train_meta["scheduler_steps"] == 3
                and train_meta["amp_overflow_count"] == 0
            ),
        })
        transfer_rows.sort(key=lambda row: (int(row["candidate_class"]), int(row["tail_class"])))
        write_csv(args.output_dir / "transfer_matrix.csv", transfer_rows)
        write_csv(args.output_dir / "runtime_fairness.csv", fairness_rows)
        print(json.dumps({"stage": "B", "candidate_class": candidate_class, "status": status}))

    if len(transfer_rows) != 80 * 20 or not all(row["pass"] for row in fairness_rows):
        raise RuntimeError("Experiment B did not complete all valid candidate-tail pairs")
    runtime_contract = {
        "stage": "B",
        "protocol": frozen_protocol(),
        "manifest_contract_hash": file_sha256(Path(args.manifest_dir) / "manifest_contract.json"),
        "theta0_hash": tensor_mapping_hash(theta0),
        "neighbor_metadata": neighbor_meta,
        "trainable_parameter_count": int(sum(parameter.numel() for _, parameter in trainable_named(model))),
        "completed_candidates": 80,
        "completed_transfer_pairs": 1600,
        "test_metrics_used_for_candidate_selection": False,
        "result_hashes": {
            name: file_sha256(args.output_dir / name)
            for name in ("transfer_matrix.csv", "runtime_fairness.csv")
        },
    }
    write_json(args.output_dir / "runtime_contract.json", runtime_contract)
    return {"stage": "B", "completed_candidates": 80, "transfer_pairs": 1600}


def _joint_train(model, cfg, theta0, tail_execution, candidate_execution, store, transform):
    _, build_optimizer, _ = _load_cliplora_api()
    load_lora_state(model, theta0)
    optimizer, scheduler = build_optimizer(model, cfg)
    gradient_calls = 0
    tensor_hashes = []
    for epoch in (1, 2, 3):
        optimizer.zero_grad(set_to_none=True)
        for role_rows in (
            tail_execution[tail_execution.epoch == epoch],
            candidate_execution[candidate_execution.epoch == epoch],
        ):
            images, labels, _, hashes = _materialize(
                role_rows, store, transform, next(model.parameters()).device
            )
            model.train()
            (0.5 * F.cross_entropy(model(images), labels)).backward()
            gradient_calls += 1
            tensor_hashes.extend(hashes)
        optimizer.step()
        scheduler.step()
    return lora_state(model), {
        "gradient_calls": gradient_calls,
        "optimizer_steps": 3,
        "scheduler_steps": 3,
        "augmented_tensor_hashes": tensor_hashes,
    }


def _private_margin(model, state, store, transform, rows, class_id, batch_size) -> float:
    load_lora_state(model, state)
    ids, labels = _target_rows(store, "train", class_id, rows)
    logits, checked = _predict_ids(model, store, transform, ids, labels, batch_size=batch_size)
    return float(classification_metrics(logits, checked, class_id)["margin"])


def run_c(args) -> dict:
    contract, execution, semantic = _verify_manifests(args.manifest_dir)
    b_contract_path = Path(args.b_dir) / "runtime_contract.json"
    if not b_contract_path.is_file():
        raise FileNotFoundError("Experiment C requires completed Experiment B runtime")
    b_contract = json.loads(b_contract_path.read_text(encoding="utf-8"))
    if b_contract.get("stage") != "B" or b_contract.get("completed_transfer_pairs") != 1600:
        raise RuntimeError("Experiment B runtime is incomplete")
    transfer = pd.read_csv(Path(args.b_dir) / "transfer_matrix.csv")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    store = CifarRawStore(args.data_dir)
    cfg, model, theta0, names = _prepare_model(args, store)
    if b_contract["theta0_hash"] != tensor_mapping_hash(theta0):
        raise RuntimeError("Experiments B and C do not share theta0")
    transform = _repository_cifar_transform()
    neighbors, neighbor_meta = load_preregistered_neighbors(TAIL_CLASSES)
    private_samples = execution[execution.role == "private_tail_evidence"].drop_duplicates(
        ["class_id", "base_sample_id"]
    )
    load_lora_state(model, theta0)
    baseline_private, baseline_test = _baseline_bundle(
        model, store, transform, private_samples, neighbors, args.eval_batch_size
    )
    metrics_rows, selection_rows, fairness_rows = [], [], []
    lambda_grid = frozen_protocol()["experiment_c"]["private_readapt"]["lambda_grid"]

    for tail_class in TAIL_CLASSES:
        pair_rows = transfer[transfer.tail_class == tail_class]
        related_pool = pair_rows[pair_rows.semantic_rank <= 10]
        unrelated_pool = pair_rows[pair_rows.semantic_rank >= 71]
        related = related_pool.sort_values(
            ["private_margin_gain", "semantic_rank", "candidate_class"], ascending=[False, True, True]
        ).iloc[0]
        unrelated = unrelated_pool.sort_values(
            ["private_margin_gain", "semantic_rank", "candidate_class"], ascending=[False, False, True]
        ).iloc[0]
        related_class, unrelated_class = int(related.candidate_class), int(unrelated.candidate_class)
        related_saved = torch.load(_candidate_state_path(args.b_dir, related_class), map_location="cpu")
        related_state = related_saved["lora_state"]
        unrelated_saved = torch.load(
            _candidate_state_path(args.b_dir, unrelated_class), map_location="cpu"
        )
        unrelated_state = unrelated_saved["lora_state"]
        tail_execution = execution[
            (execution.role == "private_tail_evidence") & (execution.class_id == tail_class)
        ].copy()
        related_execution = execution[
            (execution.role == "candidate") & (execution.class_id == related_class)
        ].copy()
        unrelated_execution = execution[
            (execution.role == "candidate") & (execution.class_id == unrelated_class)
        ].copy()
        tail_states, tail_meta = _train_client(model, cfg, theta0, tail_execution, store, transform)
        tail_state = tail_states[3]
        joint_related, joint_related_meta = _joint_train(
            model, cfg, theta0, tail_execution, related_execution, store, transform
        )
        joint_unrelated, joint_unrelated_meta = _joint_train(
            model, cfg, theta0, tail_execution, unrelated_execution, store, transform
        )
        tail_hashes = sorted(
            value
            for epoch_values in tail_meta["augmented_tensor_hashes"].values()
            for value in epoch_values
        )
        related_hashes = sorted(
            value
            for epoch_values in related_saved["train_meta"]["augmented_tensor_hashes"].values()
            for value in epoch_values
        )
        related_joint_hashes = sorted(joint_related_meta["augmented_tensor_hashes"])
        unrelated_hashes = sorted(
            value
            for epoch_values in unrelated_saved["train_meta"]["augmented_tensor_hashes"].values()
            for value in epoch_values
        )
        same_related_augmented_tensors = related_joint_hashes == sorted(tail_hashes + related_hashes)
        same_unrelated_augmented_tensors = sorted(
            joint_unrelated_meta["augmented_tensor_hashes"]
        ) == sorted(tail_hashes + unrelated_hashes)
        separate = _combine_states(theta0, [(tail_state, 0.5), (related_state, 0.5)])
        lambda_scores = []
        readapt_states = {}
        for value in lambda_grid:
            state = _combine_states(theta0, [(tail_state, 0.5), (related_state, 0.5 * float(value))])
            readapt_states[float(value)] = state
            lambda_scores.append((
                _private_margin(
                    model, state, store, transform, private_samples,
                    tail_class, args.eval_batch_size,
                ),
                -float(value),
                float(value),
            ))
        _, _, chosen_lambda = max(lambda_scores)
        readapt = readapt_states[chosen_lambda]
        conditions = {
            "tail_only": tail_state,
            "joint_related": joint_related,
            "separate_merge_related": separate,
            "separate_readapt_related": readapt,
            "joint_unrelated": joint_unrelated,
        }
        test_ids, test_labels = _target_rows(store, "test", tail_class)
        private_ids, private_labels = _target_rows(store, "train", tail_class, private_samples)
        for condition, state in conditions.items():
            load_lora_state(model, state)
            private_after = _target_metrics(
                model, store, transform, private_ids, private_labels,
                neighbors, tail_class, args.eval_batch_size,
            )[:2]
            test_after = _target_metrics(
                model, store, transform, test_ids, test_labels,
                neighbors, tail_class, args.eval_batch_size,
            )[:2]
            metrics_rows.append({
                "data_seed": 42,
                "tail_class": tail_class,
                "condition": condition,
                "related_candidate_class": related_class,
                "unrelated_candidate_class": unrelated_class,
                "chosen_lambda": chosen_lambda if condition == "separate_readapt_related" else 1.0,
                "lora_update_l2": _update_norm(theta0, state),
                **_gain_row(baseline_private[tail_class], private_after, "private"),
                **_gain_row(baseline_test[tail_class], test_after, "test"),
            })
        selection_rows.append({
            "data_seed": 42,
            "tail_class": tail_class,
            "related_candidate_class": related_class,
            "related_semantic_rank": int(related.semantic_rank),
            "related_private_margin_gain_in_b": float(related.private_margin_gain),
            "related_test_margin_gain_in_b_audit_only": float(related.test_margin_gain),
            "unrelated_candidate_class": unrelated_class,
            "unrelated_semantic_rank": int(unrelated.semantic_rank),
            "unrelated_private_margin_gain_in_b": float(unrelated.private_margin_gain),
            "unrelated_test_margin_gain_in_b_audit_only": float(unrelated.test_margin_gain),
            "chosen_lambda": chosen_lambda,
            "selection_used_test_metrics": False,
        })
        fairness_rows.append({
            "data_seed": 42,
            "tail_class": tail_class,
            "theta0_hash_equal": True,
            "same_tail_ids": bool(tail_execution.base_sample_id.nunique() == 5),
            "same_related_ids_joint_separate_readapt": bool(related_execution.base_sample_id.nunique() == 12),
            "same_augmentation_seeds": bool(
                same_related_augmented_tensors and same_unrelated_augmented_tensors
            ),
            "tail_gradient_calls_joint": 3,
            "candidate_gradient_calls_joint": 3,
            "tail_gradient_calls_separate": 3,
            "candidate_gradient_calls_separate": 3,
            "joint_related_optimizer_steps": joint_related_meta["optimizer_steps"],
            "separate_total_optimizer_steps": int(tail_meta["optimizer_steps_successful"] + related_saved["train_meta"]["optimizer_steps_successful"]),
            "optimizer_trajectory_is_treatment": True,
            "selection_used_test_metrics": False,
            "pass": bool(
                joint_related_meta["gradient_calls"] == 6
                and joint_unrelated_meta["gradient_calls"] == 6
                and tail_meta["optimizer_steps_successful"] == 3
                and related_saved["train_meta"]["optimizer_steps_successful"] == 3
                and same_related_augmented_tensors
                and same_unrelated_augmented_tensors
            ),
        })
        write_csv(args.output_dir / "placement_metrics.csv", metrics_rows)
        write_csv(args.output_dir / "candidate_selection.csv", selection_rows)
        write_csv(args.output_dir / "runtime_fairness.csv", fairness_rows)
        print(json.dumps({
            "stage": "C", "tail_class": tail_class, "related": related_class,
            "unrelated": unrelated_class, "lambda": chosen_lambda, "status": "complete",
        }))

    if len(metrics_rows) != 20 * 5 or not all(row["pass"] for row in fairness_rows):
        raise RuntimeError("Experiment C did not complete all valid placement conditions")
    runtime_contract = {
        "stage": "C",
        "protocol": frozen_protocol(),
        "manifest_contract_hash": file_sha256(Path(args.manifest_dir) / "manifest_contract.json"),
        "experiment_b_contract_hash": file_sha256(b_contract_path),
        "theta0_hash": tensor_mapping_hash(theta0),
        "neighbor_metadata": neighbor_meta,
        "completed_tail_classes": 20,
        "completed_condition_rows": 100,
        "selection_used_test_metrics": False,
        "result_hashes": {
            name: file_sha256(args.output_dir / name)
            for name in ("placement_metrics.csv", "candidate_selection.csv", "runtime_fairness.csv")
        },
    }
    write_json(args.output_dir / "runtime_contract.json", runtime_contract)
    return {"stage": "C", "completed_tail_classes": 20, "condition_rows": 100}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["b", "c"], required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("DATA/cifar-100/cifar-100-python"))
    parser.add_argument("--manifest-dir", type=Path, default=Path("output/carrier_access_audit/manifests"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--b-dir", type=Path, default=Path("output/carrier_access_audit/experiment_b"))
    parser.add_argument("--theta0-file", type=Path, default=Path("output/e1_strength_breadth/protocol_v2/theta0_seed42.pt"))
    parser.add_argument("--model-init-seed", type=int, default=42)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    stale_failure = args.output_dir / "failure.json"
    if stale_failure.is_file():
        stale_failure.unlink()
    try:
        result = run_b(args) if args.stage == "b" else run_c(args)
        print(json.dumps(result))
    except Exception as exc:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        write_json(args.output_dir / "failure.json", {
            "stage": args.stage.upper(),
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        })
        raise


if __name__ == "__main__":
    main()
