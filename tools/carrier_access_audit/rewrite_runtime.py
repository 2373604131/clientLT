"""CUDA runtime for D1 post-write effects and D2 cumulative rewrite replay."""

from __future__ import annotations

import argparse
import csv
import json
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from tools.breadth_audit.inputs import load_preregistered_neighbors
from tools.carrier_access_audit.protocol import NON_TAIL_CLASSES, TAIL_CLASSES, frozen_protocol
from tools.carrier_access_audit.rewrite_protocol import frozen_rewrite_protocol
from tools.carrier_access_audit.runtime import (
    _baseline_bundle,
    _candidate_state_path,
    _gain_row,
    _target_metrics,
    _target_rows,
)
from tools.client_update_audit.runtime import _prepare_model, _repository_cifar_transform
from tools.semantic_acquisition.common import (
    file_sha256,
    stable_hash,
    stable_seed,
    tensor_mapping_hash,
    write_csv,
    write_json,
)
from tools.semantic_acquisition.runtime import (
    CifarRawStore,
    _train_client,
    flatten_named,
    load_lora_state,
)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _verify_inputs(manifest_dir: Path, b_dir: Path) -> tuple[dict, dict, pd.DataFrame]:
    manifest_dir, b_dir = Path(manifest_dir), Path(b_dir)
    manifest_contract = json.loads((manifest_dir / "manifest_contract.json").read_text(encoding="utf-8"))
    if manifest_contract.get("protocol") != frozen_protocol():
        raise RuntimeError("Carrier-access manifests do not match the frozen parent protocol")
    for name, expected in manifest_contract["manifest_hashes"].items():
        if file_sha256(manifest_dir / name) != expected:
            raise RuntimeError(f"Carrier-access manifest hash mismatch: {name}")
    b_contract = json.loads((b_dir / "runtime_contract.json").read_text(encoding="utf-8"))
    if (
        b_contract.get("stage") != "B"
        or b_contract.get("protocol") != frozen_protocol()
        or int(b_contract.get("completed_transfer_pairs", -1)) != 1600
    ):
        raise RuntimeError("Experiment B runtime is incomplete or incompatible")
    for name, expected in b_contract.get("result_hashes", {}).items():
        if file_sha256(b_dir / name) != expected:
            raise RuntimeError(f"Experiment B result hash mismatch: {name}")
    execution = pd.read_csv(manifest_dir / "training_execution.csv")
    return manifest_contract, b_contract, execution


def _candidate_states(b_dir: Path, theta0, names):
    states, norms = {}, {}
    theta_vector = flatten_named(theta0, names)
    theta_hash = tensor_mapping_hash(theta0)
    for candidate_class in NON_TAIL_CLASSES:
        path = _candidate_state_path(b_dir, candidate_class)
        if not path.is_file():
            raise FileNotFoundError(f"Experiment B candidate state is missing: {path}")
        saved = torch.load(path, map_location="cpu")
        if int(saved.get("candidate_class", -1)) != candidate_class or saved.get("theta0_hash") != theta_hash:
            raise RuntimeError(f"Stale Experiment B candidate state: {path}")
        state = saved["lora_state"]
        states[candidate_class] = state
        norms[candidate_class] = float((flatten_named(state, names) - theta_vector).norm().item())
    if min(norms.values()) <= 0:
        raise RuntimeError("A candidate update has zero norm")
    target_norm = float(np.median(list(norms.values())))
    return states, norms, target_norm


def _apply_normalized_candidate(anchor, theta0, candidate_state, candidate_norm, target_norm, coefficient):
    scale = float(coefficient) * float(target_norm) / float(candidate_norm)
    return {
        name: anchor[name] + scale * (candidate_state[name] - theta0[name])
        for name in sorted(theta0)
    }


def _apply_sequence(anchor, theta0, candidate_states, candidate_norms, target_norm, classes, beta):
    result = {name: anchor[name].clone() for name in sorted(theta0)}
    for candidate_class in classes:
        scale = float(beta) * float(target_norm) / float(candidate_norms[int(candidate_class)])
        candidate = candidate_states[int(candidate_class)]
        for name in result:
            result[name] += scale * (candidate[name] - theta0[name])
    return result


def _split_private(execution: pd.DataFrame):
    private = execution[execution.role == "private_tail_evidence"].copy()
    unique = private.drop_duplicates(["class_id", "base_sample_id"])
    write = unique[unique.slot.isin([0, 1, 2])].copy()
    evidence = unique[unique.slot.isin([3, 4])].copy()
    for class_id in TAIL_CLASSES:
        if len(write[write.class_id == class_id]) != 3 or len(evidence[evidence.class_id == class_id]) != 2:
            raise RuntimeError(f"Tail class {class_id} does not have the frozen 3/2 private split")
        if set(write[write.class_id == class_id].base_sample_id) & set(evidence[evidence.class_id == class_id].base_sample_id):
            raise RuntimeError(f"Tail write/evidence samples overlap for class {class_id}")
    write_execution = private[private.slot.isin([0, 1, 2])].copy()
    return write, evidence, write_execution


def _writer_state_path(output_dir: Path, tail_class: int) -> Path:
    return Path(output_dir) / "tail_writer_states" / f"tail_{int(tail_class):02d}.pt"


def _load_existing(path: Path, keys: tuple[str, ...]) -> dict[tuple, dict]:
    if not path.is_file():
        return {}
    rows = _read_csv(path)
    return {tuple(str(row[key]) for key in keys): row for row in rows}


def _ordered_rows(mapping: dict[tuple, dict], numeric_positions: tuple[int, ...]) -> list[dict]:
    def key(value):
        parts = list(value)
        for position in numeric_positions:
            parts[position] = int(parts[position])
        return tuple(parts)
    return [mapping[item] for item in sorted(mapping, key=key)]


def _effect_row(before, after, prefix: str) -> dict:
    return _gain_row(before, after, prefix)


def _effect_sign(value: float) -> str:
    if float(value) > 0:
        return "donor"
    if float(value) < 0:
        return "rewriter"
    return "neutral"


def run_d1(args) -> dict:
    manifest_contract, b_contract, execution = _verify_inputs(args.manifest_dir, args.b_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    store = CifarRawStore(args.data_dir)
    cfg, model, theta0, names = _prepare_model(args, store)
    if b_contract["theta0_hash"] != tensor_mapping_hash(theta0):
        raise RuntimeError("D1 and Experiment B do not share theta0")
    transform = _repository_cifar_transform()
    neighbors, neighbor_meta = load_preregistered_neighbors(TAIL_CLASSES)
    write_samples, evidence_samples, write_execution = _split_private(execution)
    candidates, candidate_norms, target_norm = _candidate_states(args.b_dir, theta0, names)
    alpha = float(frozen_rewrite_protocol()["candidate_updates"]["d1_alpha"])

    load_lora_state(model, theta0)
    baseline_private, baseline_test = _baseline_bundle(
        model, store, transform, evidence_samples, neighbors, args.eval_batch_size
    )
    writer_states, writer_rows, writer_fairness = {}, [], []
    split_hash = stable_hash({
        str(class_id): {
            "write": write_samples[write_samples.class_id == class_id].base_sample_id.tolist(),
            "evidence": evidence_samples[evidence_samples.class_id == class_id].base_sample_id.tolist(),
        }
        for class_id in TAIL_CLASSES
    })
    for tail_class in TAIL_CLASSES:
        state_path = _writer_state_path(args.output_dir, tail_class)
        class_execution = write_execution[write_execution.class_id == tail_class].copy()
        if state_path.is_file():
            saved = torch.load(state_path, map_location="cpu")
            if (
                int(saved.get("tail_class", -1)) != tail_class
                or saved.get("theta0_hash") != tensor_mapping_hash(theta0)
                or saved.get("split_hash") != split_hash
            ):
                raise RuntimeError(f"Stale D1 writer state: {state_path}")
            state, train_meta, status = saved["lora_state"], saved["train_meta"], "resumed"
        else:
            states, train_meta = _train_client(model, cfg, theta0, class_execution, store, transform)
            state, status = states[3], "trained"
            state_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                "tail_class": tail_class, "theta0_hash": tensor_mapping_hash(theta0),
                "split_hash": split_hash, "lora_state": state, "train_meta": train_meta,
            }, state_path)
        writer_states[tail_class] = state
        load_lora_state(model, state)
        private_ids, private_labels = _target_rows(store, "train", tail_class, evidence_samples)
        test_ids, test_labels = _target_rows(store, "test", tail_class)
        private_after = _target_metrics(
            model, store, transform, private_ids, private_labels, neighbors, tail_class, args.eval_batch_size
        )[:2]
        test_after = _target_metrics(
            model, store, transform, test_ids, test_labels, neighbors, tail_class, args.eval_batch_size
        )[:2]
        writer_rows.append({
            "data_seed": 42, "tail_class": tail_class,
            "write_sample_count": 3, "evidence_sample_count": 2,
            "writer_update_l2": float(
                (flatten_named(state, names) - flatten_named(theta0, names)).norm().item()
            ),
            **_effect_row(baseline_private[tail_class], private_after, "direct_private"),
            **_effect_row(baseline_test[tail_class], test_after, "direct_test"),
        })
        writer_fairness.append({
            "data_seed": 42, "tail_class": tail_class,
            "write_evidence_disjoint": True,
            "write_sample_count": 3, "evidence_sample_count": 2,
            "optimizer_steps_successful": int(train_meta["optimizer_steps_successful"]),
            "scheduler_steps": int(train_meta["scheduler_steps"]),
            "amp_overflow_count": int(train_meta["amp_overflow_count"]),
            "pass": bool(
                train_meta["optimizer_steps_successful"] == 3
                and train_meta["scheduler_steps"] == 3
                and train_meta["amp_overflow_count"] == 0
            ),
        })
        print(json.dumps({"stage": "D1-writer", "tail_class": tail_class, "status": status}))
    write_csv(args.output_dir / "tail_writer_metrics.csv", writer_rows)
    write_csv(args.output_dir / "tail_writer_fairness.csv", writer_fairness)

    pre_path = args.output_dir / "matched_pre_effects.csv"
    pre_mapping = _load_existing(pre_path, ("tail_class", "candidate_class"))
    for candidate_class in NON_TAIL_CLASSES:
        if all((str(tail_class), str(candidate_class)) in pre_mapping for tail_class in TAIL_CLASSES):
            print(json.dumps({"stage": "D1-pre", "candidate_class": candidate_class, "status": "resumed"}))
            continue
        state = _apply_normalized_candidate(
            theta0, theta0, candidates[candidate_class], candidate_norms[candidate_class], target_norm, alpha
        )
        load_lora_state(model, state)
        private_after, test_after = _baseline_bundle(
            model, store, transform, evidence_samples, neighbors, args.eval_batch_size
        )
        for tail_class in TAIL_CLASSES:
            pre_mapping[(str(tail_class), str(candidate_class))] = {
                "data_seed": 42, "tail_class": tail_class, "candidate_class": candidate_class,
                "candidate_raw_l2": candidate_norms[candidate_class],
                "candidate_normalized_l2": target_norm, "alpha": alpha,
                **_effect_row(baseline_private[tail_class], private_after[tail_class], "private_pre"),
                **_effect_row(baseline_test[tail_class], test_after[tail_class], "test_pre"),
            }
        write_csv(pre_path, _ordered_rows(pre_mapping, (0, 1)))
        print(json.dumps({"stage": "D1-pre", "candidate_class": candidate_class, "status": "complete"}))

    pre_numeric = {
        (int(key[0]), int(key[1])): row for key, row in pre_mapping.items()
    }
    post_path = args.output_dir / "post_write_effects.csv"
    post_mapping = _load_existing(post_path, ("tail_class", "candidate_class"))
    for tail_class in TAIL_CLASSES:
        private_ids, private_labels = _target_rows(store, "train", tail_class, evidence_samples)
        test_ids, test_labels = _target_rows(store, "test", tail_class)
        load_lora_state(model, writer_states[tail_class])
        writer_private = _target_metrics(
            model, store, transform, private_ids, private_labels, neighbors, tail_class, args.eval_batch_size
        )[:2]
        writer_test = _target_metrics(
            model, store, transform, test_ids, test_labels, neighbors, tail_class, args.eval_batch_size
        )[:2]
        for candidate_class in NON_TAIL_CLASSES:
            key = (str(tail_class), str(candidate_class))
            if key in post_mapping:
                continue
            state = _apply_normalized_candidate(
                writer_states[tail_class], theta0, candidates[candidate_class],
                candidate_norms[candidate_class], target_norm, alpha,
            )
            load_lora_state(model, state)
            private_after = _target_metrics(
                model, store, transform, private_ids, private_labels,
                neighbors, tail_class, args.eval_batch_size,
            )[:2]
            test_after = _target_metrics(
                model, store, transform, test_ids, test_labels,
                neighbors, tail_class, args.eval_batch_size,
            )[:2]
            pre = pre_numeric[(tail_class, candidate_class)]
            row = {
                "data_seed": 42, "tail_class": tail_class, "candidate_class": candidate_class,
                "candidate_raw_l2": candidate_norms[candidate_class],
                "candidate_normalized_l2": target_norm, "alpha": alpha,
                **_effect_row(writer_private, private_after, "private_post"),
                **_effect_row(writer_test, test_after, "test_post"),
            }
            row["private_margin_turnover"] = float(row["private_post_margin_gain"] - float(pre["private_pre_margin_gain"]))
            row["test_margin_turnover"] = float(row["test_post_margin_gain"] - float(pre["test_pre_margin_gain"]))
            row["pre_test_sign"] = _effect_sign(float(pre["test_pre_margin_gain"]))
            row["post_test_sign"] = _effect_sign(row["test_post_margin_gain"])
            row["sign_transition"] = f"{row['pre_test_sign']}_to_{row['post_test_sign']}"
            post_mapping[key] = row
            write_csv(post_path, _ordered_rows(post_mapping, (0, 1)))
        print(json.dumps({"stage": "D1-post", "tail_class": tail_class, "status": "complete"}))

    candidate_fairness = [{
        "data_seed": 42, "candidate_class": candidate_class,
        "theta0_hash_equal": True,
        "raw_l2": candidate_norms[candidate_class],
        "normalized_l2": target_norm,
        "normalized_l2_relative_error": 0.0,
        "alpha": alpha,
        "pass": True,
    } for candidate_class in NON_TAIL_CLASSES]
    write_csv(args.output_dir / "candidate_norm_fairness.csv", candidate_fairness)
    complete_pre = len(pre_mapping) == 1600
    complete_post = len(post_mapping) == 1600
    if not complete_pre or not complete_post or not all(row["pass"] for row in writer_fairness):
        raise RuntimeError("D1 did not complete all valid writer and pre/post units")
    runtime_contract = {
        "stage": "D1", "protocol": frozen_rewrite_protocol(),
        "parent_manifest_contract_hash": file_sha256(Path(args.manifest_dir) / "manifest_contract.json"),
        "experiment_b_contract_hash": file_sha256(Path(args.b_dir) / "runtime_contract.json"),
        "theta0_hash": tensor_mapping_hash(theta0), "private_split_hash": split_hash,
        "candidate_target_norm": target_norm, "alpha": alpha,
        "completed_writer_classes": 20, "completed_pre_pairs": 1600, "completed_post_pairs": 1600,
        "neighbor_metadata": neighbor_meta,
        "test_metrics_used_for_candidate_selection": False,
        "test_write_gain_used_for_tail_eligibility": True,
        "candidate_state_hashes": {
            str(candidate_class): file_sha256(_candidate_state_path(args.b_dir, candidate_class))
            for candidate_class in NON_TAIL_CLASSES
        },
        "writer_state_hashes": {
            str(tail_class): file_sha256(_writer_state_path(args.output_dir, tail_class))
            for tail_class in TAIL_CLASSES
        },
        "result_hashes": {
            name: file_sha256(args.output_dir / name) for name in (
                "tail_writer_metrics.csv", "tail_writer_fairness.csv", "candidate_norm_fairness.csv",
                "matched_pre_effects.csv", "post_write_effects.csv",
            )
        },
    }
    write_json(args.output_dir / "runtime_contract.json", runtime_contract)
    return {"stage": "D1", "writers": 20, "pre_pairs": 1600, "post_pairs": 1600}


def run_d2(args) -> dict:
    _, b_contract, execution = _verify_inputs(args.manifest_dir, args.b_dir)
    d1_dir = Path(args.d1_dir)
    d1_contract = json.loads((d1_dir / "runtime_contract.json").read_text(encoding="utf-8"))
    d1_summary = json.loads((Path(args.d1_summary)).read_text(encoding="utf-8"))
    if d1_contract.get("stage") != "D1" or d1_contract.get("protocol") != frozen_rewrite_protocol():
        raise RuntimeError("D2 requires a complete compatible D1 runtime")
    for name, expected in d1_contract.get("result_hashes", {}).items():
        if file_sha256(d1_dir / name) != expected:
            raise RuntimeError(f"D1 result hash mismatch: {name}")
    for tail_class, expected in d1_contract.get("writer_state_hashes", {}).items():
        if file_sha256(_writer_state_path(d1_dir, int(tail_class))) != expected:
            raise RuntimeError(f"D1 writer state hash mismatch: tail {tail_class}")
    for candidate_class, expected in d1_contract.get("candidate_state_hashes", {}).items():
        if file_sha256(_candidate_state_path(args.b_dir, int(candidate_class))) != expected:
            raise RuntimeError(f"Experiment B candidate state changed after D1: class {candidate_class}")
    if not bool(d1_summary.get("valid_comparison", False)):
        raise RuntimeError("D2 requires D1 to pass the tail-write validity Gate")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    store = CifarRawStore(args.data_dir)
    _, model, theta0, names = _prepare_model(args, store)
    if b_contract["theta0_hash"] != tensor_mapping_hash(theta0) or d1_contract["theta0_hash"] != tensor_mapping_hash(theta0):
        raise RuntimeError("D2, D1 and Experiment B do not share theta0")
    transform = _repository_cifar_transform()
    neighbors, neighbor_meta = load_preregistered_neighbors(TAIL_CLASSES)
    _, evidence_samples, _ = _split_private(execution)
    candidates, candidate_norms, target_norm = _candidate_states(args.b_dir, theta0, names)
    post = pd.read_csv(d1_dir / "post_write_effects.csv")
    writer_metrics = pd.read_csv(d1_dir / "tail_writer_metrics.csv")
    valid_tails = sorted(
        int(value) for value in writer_metrics[writer_metrics.direct_test_margin_gain > 0].tail_class.tolist()
    )
    protocol = frozen_rewrite_protocol()["d2"]
    beta = float(protocol["per_update_beta"])
    alpha = float(frozen_rewrite_protocol()["candidate_updates"]["d1_alpha"])
    lengths = [int(value) for value in protocol["sequence_lengths"]]
    blind_draws = int(protocol["blind_draws"])
    output_path = args.output_dir / "replay_metrics.csv"
    mapping = _load_existing(output_path, ("tail_class", "condition", "sequence_length", "draw"))
    fairness_rows = []
    for tail_class in valid_tails:
        writer_saved = torch.load(_writer_state_path(d1_dir, tail_class), map_location="cpu")
        writer_state = writer_saved["lora_state"]
        load_lora_state(model, writer_state)
        private_ids, private_labels = _target_rows(store, "train", tail_class, evidence_samples)
        test_ids, test_labels = _target_rows(store, "test", tail_class)
        writer_private = _target_metrics(
            model, store, transform, private_ids, private_labels, neighbors, tail_class, args.eval_batch_size
        )[:2]
        writer_test = _target_metrics(
            model, store, transform, test_ids, test_labels, neighbors, tail_class, args.eval_batch_size
        )[:2]
        write_gain = float(
            writer_metrics[writer_metrics.tail_class == tail_class].direct_test_margin_gain.iloc[0]
        )
        effects = post[post.tail_class == tail_class].sort_values(
            ["private_post_margin_gain", "candidate_class"], ascending=[False, True]
        )
        ranked = effects.candidate_class.astype(int).tolist()
        private_effect = {
            int(row.candidate_class): float(row.private_post_margin_gain)
            for row in effects.itertuples(index=False)
        }
        specifications = []
        for length in lengths:
            specifications.append(("low_risk", length, -1, ranked[:length]))
            specifications.append(("high_risk", length, -1, ranked[-length:]))
            for draw in range(blind_draws):
                generator = np.random.default_rng(stable_seed("d2-blind", 42, tail_class, length, draw))
                chosen = sorted(int(value) for value in generator.choice(NON_TAIL_CLASSES, size=length, replace=False))
                specifications.append(("blind", length, draw, chosen))
        for condition, length, draw, classes in specifications:
            key = (str(tail_class), condition, str(length), str(draw))
            if key in mapping:
                continue
            state = _apply_sequence(
                writer_state, theta0, candidates, candidate_norms, target_norm, classes, beta
            )
            load_lora_state(model, state)
            private_after = _target_metrics(
                model, store, transform, private_ids, private_labels, neighbors, tail_class, args.eval_batch_size
            )[:2]
            test_after = _target_metrics(
                model, store, transform, test_ids, test_labels, neighbors, tail_class, args.eval_batch_size
            )[:2]
            row = {
                "data_seed": 42, "tail_class": tail_class, "condition": condition,
                "sequence_length": length, "draw": draw,
                "candidate_classes": json.dumps(classes, separators=(",", ":")),
                "per_update_beta": beta,
                "predicted_private_rewrite_risk": float(
                    (beta / alpha) * sum(max(-private_effect[class_id], 0.0) for class_id in classes)
                ),
                "predicted_private_benefit": float(
                    (beta / alpha) * sum(max(private_effect[class_id], 0.0) for class_id in classes)
                ),
                "private_positive_candidate_fraction": float(np.mean([
                    private_effect[class_id] > 0 for class_id in classes
                ])),
                "sequence_update_l2": float(
                    (flatten_named(state, names) - flatten_named(writer_state, names)).norm().item()
                ),
                **_effect_row(writer_private, private_after, "private_replay"),
                **_effect_row(writer_test, test_after, "test_replay"),
            }
            row["test_forgetting"] = float(-row["test_replay_margin_gain"])
            row["test_retention"] = float((write_gain + row["test_replay_margin_gain"]) / write_gain)
            mapping[key] = row
            write_csv(output_path, _ordered_rows(mapping, (0, 2, 3)))
        fairness_rows.append({
            "data_seed": 42, "tail_class": tail_class,
            "writer_test_margin_gain_positive": write_gain > 0,
            "selection_uses_private_post_effect_only": True,
            "test_metrics_used_for_sequence_selection": False,
            "candidate_norm_equalized": True,
            "expected_sequence_rows": 3 * (2 + blind_draws),
            "observed_sequence_rows": sum(int(key[0]) == tail_class for key in mapping),
            "pass": bool(write_gain > 0),
        })
        write_csv(args.output_dir / "runtime_fairness.csv", fairness_rows)
        print(json.dumps({"stage": "D2", "tail_class": tail_class, "status": "complete"}))
    expected_rows = len(valid_tails) * len(lengths) * (2 + blind_draws)
    if len(mapping) != expected_rows or not all(row["pass"] for row in fairness_rows):
        raise RuntimeError(f"D2 expected {expected_rows} valid replay rows, observed {len(mapping)}")
    runtime_contract = {
        "stage": "D2", "protocol": frozen_rewrite_protocol(),
        "d1_contract_hash": file_sha256(d1_dir / "runtime_contract.json"),
        "d1_summary_hash": file_sha256(Path(args.d1_summary)),
        "theta0_hash": tensor_mapping_hash(theta0), "valid_tail_classes": valid_tails,
        "candidate_target_norm": target_norm, "per_update_beta": beta,
        "completed_replay_rows": len(mapping), "neighbor_metadata": neighbor_meta,
        "test_metrics_used_for_sequence_selection": False,
        "test_write_gain_used_for_tail_eligibility": True,
        "result_hashes": {
            name: file_sha256(args.output_dir / name)
            for name in ("replay_metrics.csv", "runtime_fairness.csv")
        },
    }
    write_json(args.output_dir / "runtime_contract.json", runtime_contract)
    return {"stage": "D2", "valid_tail_classes": len(valid_tails), "replay_rows": len(mapping)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["d1", "d2"], required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("DATA/cifar-100/cifar-100-python"))
    parser.add_argument("--manifest-dir", type=Path, default=Path("output/carrier_access_audit/manifests"))
    parser.add_argument("--b-dir", type=Path, default=Path("output/carrier_access_audit/experiment_b"))
    parser.add_argument("--d1-dir", type=Path, default=Path("output/post_write_rewrite_audit/d1"))
    parser.add_argument("--d1-summary", type=Path, default=Path("output/post_write_rewrite_audit/analysis_d1/d1_summary.json"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--theta0-file", type=Path, default=Path("output/e1_strength_breadth/protocol_v2/theta0_seed42.pt"))
    parser.add_argument("--model-init-seed", type=int, default=42)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    stale_failure = args.output_dir / "failure.json"
    if stale_failure.is_file():
        stale_failure.unlink()
    try:
        result = run_d1(args) if args.stage == "d1" else run_d2(args)
        print(json.dumps(result))
    except Exception as exc:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        write_json(args.output_dir / "failure.json", {
            "stage": args.stage.upper(), "error_type": type(exc).__name__,
            "error": str(exc), "traceback": traceback.format_exc(),
        })
        raise


if __name__ == "__main__":
    main()
