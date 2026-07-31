#!/usr/bin/env python
"""Minimal single-round Oracle CUSP pilot.

The real path consumes one round-10 dump produced by federated_main.py. It
freezes all candidates before any official test access, then evaluates the
frozen candidates on the same test cache. The synthetic path exercises the same
output schema without CIFAR, CLIP, CVXPY, or GPU.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.oracle_cusp import (
    ROUND1_METHODS,
    FlatSpec,
    classwise_weighting_delta,
    ensure_fedavg_in_subspace,
    finite_difference_utility,
    flatten_state,
    load_round_dump,
    random_reweight,
    scale_to_budget,
    sha256_file,
    sha256_json,
    solve_cusp,
    subspace_from_updates,
    summarize_values,
    unflatten_state,
    validate_train_feature_cache,
    write_json,
)


NUM_RANDOM = 10
RANDOM_SEED = 42
NORM_ATOL = 1e-6
NORM_RTOL = 1e-6


def now_stamp():
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def write_csv(path: Path, rows, fields=None):
    if fields is None:
        fields = []
        for row in rows:
            for key in row:
                if key not in fields:
                    fields.append(key)
        fields = fields or ["method"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def jsonable(value):
    if isinstance(value, torch.Tensor):
        if value.ndim == 0:
            return value.item()
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    return value


def fail_before_test(output_dir: Path, solver_report: dict, metadata: dict, message: str, exit_code: int = 2):
    output_dir.mkdir(parents=True, exist_ok=True)
    report = {**solver_report, "status": solver_report.get("status", "failed"), "failure_reason": message}
    write_json(output_dir / "oracle_solver.json", jsonable(report))
    write_json(output_dir / "oracle_metadata.json", jsonable({
        **metadata,
        "minimal_pilot_status": "INCOMPLETE",
        "failure_reason": message,
        "test_accessed": False,
        "finished_at": now_stamp(),
    }))
    raise SystemExit(exit_code)


def load_train_cache(path: Path, metadata: dict):
    if not metadata.get("train_feature_cache_sha256"):
        raise RuntimeError("metadata missing train_feature_cache_sha256")
    observed = sha256_file(path)
    if observed != metadata["train_feature_cache_sha256"]:
        raise RuntimeError("train_feature_cache.pt SHA-256 mismatch")
    cache = torch.load(path, map_location="cpu", weights_only=False)
    validate_train_feature_cache(cache, metadata["global_class_counts"])
    return {
        "features": torch.as_tensor(cache["features"], dtype=torch.float32),
        "labels": torch.as_tensor(cache["labels"], dtype=torch.long),
        "class_counts": torch.as_tensor(cache["class_counts"], dtype=torch.long),
        "sha256": sha256_file(path),
    }


def load_verified_dump(dump_dir: Path):
    payload, metadata = load_round_dump(dump_dir)
    if not metadata.get("round_state_sha256"):
        raise RuntimeError("metadata missing round_state_sha256")
    observed = sha256_file(dump_dir / "round_state.pt")
    if observed != metadata["round_state_sha256"]:
        raise RuntimeError("round_state.pt SHA-256 mismatch")
    cache = load_train_cache(dump_dir / "train_feature_cache.pt", metadata)
    if metadata.get("utility_data_source") != "train" or bool(metadata.get("test_used_for_utility", True)):
        raise RuntimeError("dump metadata violates train-only utility invariant")
    return payload, metadata, cache


def class_groups(metadata: dict, num_classes: int):
    tail = set(int(x) for x in metadata["tail_class_ids"])
    head = set(int(x) for x in metadata["head_class_ids"])
    if tail & head or tail | head != set(range(num_classes)):
        raise RuntimeError("head/tail metadata must cover all classes exactly once")
    return head, tail


def decision_margins_by_class(logits: torch.Tensor, labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    logits = logits.detach().cpu().to(torch.float64)
    labels = labels.detach().cpu().to(torch.long)
    masked = logits.clone()
    masked[torch.arange(labels.numel()), labels] = -torch.inf
    margins = logits[torch.arange(labels.numel()), labels] - masked.max(dim=1).values
    out = torch.full((num_classes,), torch.nan, dtype=torch.float64)
    for class_id in range(num_classes):
        selected = labels == class_id
        if bool(selected.any()):
            out[class_id] = margins[selected].mean()
    return out


def accuracy_metrics(logits: torch.Tensor, labels: torch.Tensor, head_ids, tail_ids, support_counts=None):
    logits = logits.detach().cpu()
    labels = labels.detach().cpu().to(torch.long)
    num_classes = logits.shape[1]
    preds = logits.argmax(dim=1)
    per_class = []
    for class_id in range(num_classes):
        selected = labels == class_id
        total = int(selected.sum().item())
        correct = int((preds[selected] == labels[selected]).sum().item()) if total else 0
        acc = 100.0 * correct / total if total else math.nan
        group = "tail" if class_id in tail_ids else "head"
        per_class.append({
            "class_id": class_id,
            "group": group,
            "support_count": int(support_counts[class_id]) if support_counts is not None else total,
            "test_count": total,
            "correct_count": correct,
            "class_acc": acc,
        })
    finite_acc = [row["class_acc"] for row in per_class if math.isfinite(row["class_acc"])]
    head_acc = [row["class_acc"] for row in per_class if row["class_id"] in head_ids and math.isfinite(row["class_acc"])]
    tail_acc = [row["class_acc"] for row in per_class if row["class_id"] in tail_ids and math.isfinite(row["class_acc"])]
    return {
        "overall_acc": 100.0 * float((preds == labels).to(torch.float64).mean().item()) if labels.numel() else math.nan,
        "macro_acc": float(sum(finite_acc) / len(finite_acc)) if finite_acc else math.nan,
        "head_acc": float(sum(head_acc) / len(head_acc)) if head_acc else math.nan,
        "tail_acc": float(sum(tail_acc) / len(tail_acc)) if tail_acc else math.nan,
        "per_class": per_class,
    }


def build_promptfl_model(metadata: dict):
    from Dassl.dassl.engine import build_trainer
    from federated_main import setup_cfg

    args = SimpleNamespace(**metadata["resolved_args"])
    cfg = setup_cfg(args)
    trainer = build_trainer(cfg)
    trainer.fed_before_train(is_global=True)
    return cfg, trainer


def logits_from_vector(model, spec: FlatSpec, theta: torch.Tensor, features: torch.Tensor):
    state = unflatten_state(theta, spec)
    with torch.no_grad():
        return model.logits_from_cached_features(features, state)


def build_test_cache(trainer, output_dir: Path):
    model = trainer.model
    was_training = model.training
    model.eval()
    features, labels = [], []
    try:
        with torch.no_grad():
            for batch in trainer.test_loader:
                inputs, batch_labels = trainer.parse_batch_train(batch)
                image_features = model.image_encoder(inputs.type(model.dtype))
                image_features = image_features / image_features.norm(dim=-1, keepdim=True).clamp_min(1e-12)
                features.append(image_features.detach().float().cpu())
                labels.append(batch_labels.detach().long().cpu())
    finally:
        model.train(was_training)
    if not features:
        raise RuntimeError("official test cache is empty")
    cache = {
        "schema_version": "cusp_round1_v1",
        "source": "official_test",
        "purpose": "test_evaluation_only",
        "features": torch.cat(features, dim=0),
        "labels": torch.cat(labels, dim=0),
    }
    path = output_dir / "official_test_feature_cache.pt"
    torch.save(cache, path)
    return {"path": str(path), "sha256": sha256_file(path), "features": cache["features"], "labels": cache["labels"]}


def candidate_hash(delta: torch.Tensor) -> str:
    return sha256_json({"delta": [round(float(x), 12) for x in delta.detach().cpu().reshape(-1).tolist()]})


def norm_ok(final_norm: float, budget: float) -> bool:
    return abs(final_norm - budget) <= NORM_ATOL + NORM_RTOL * abs(budget)


def build_candidates(payload: dict, metadata: dict, train_margin_fn, *, synthetic_cusp: bool = False):
    spec = FlatSpec.from_dict(payload["flatten_spec"])
    theta0 = flatten_state(payload["global_before_trainable"], spec)
    after = flatten_state(payload["global_after_fedavg_trainable"], spec)
    local_vectors = torch.stack([flatten_state(state, spec) for state in payload["local_trainable_states"]])
    client_deltas = (local_vectors - theta0.unsqueeze(0)).T
    weights = torch.as_tensor(payload["fedavg_weights"], dtype=torch.float64)
    delta_fedavg = after - theta0
    budget = float(delta_fedavg.norm().item())
    if budget <= 1e-12:
        raise RuntimeError("FedAvg update norm is zero; minimal pilot round is invalid")

    concrete = []
    concrete.append({
        "candidate_id": "fedavg",
        "method": "fedavg",
        "delta": delta_fedavg,
        "raw_norm": budget,
        "final_norm": budget,
        "scale_factor": 1.0,
        "status": "ok",
    })

    for random_item in random_reweight(client_deltas, budget, count=NUM_RANDOM, seed=RANDOM_SEED):
        if random_item["delta"] is None:
            concrete.append({
                "candidate_id": f"random_reweight_{random_item['index']:03d}",
                "method": "random_reweight",
                "delta": None,
                "status": "invalid",
                **{key: random_item[key] for key in ("raw_norm", "final_norm", "scale_factor", "coefficient_hash")},
            })
            continue
        concrete.append({
            "candidate_id": f"random_reweight_{random_item['index']:03d}",
            "method": "random_reweight",
            "delta": random_item["delta"],
            "raw_norm": random_item["raw_norm"],
            "final_norm": random_item["final_norm"],
            "scale_factor": random_item["scale_factor"],
            "coefficient_hash": random_item["coefficient_hash"],
            "status": "ok",
        })

    classwise_delta, classwise_report = classwise_weighting_delta(
        payload["global_before_trainable"],
        payload["local_trainable_states"],
        weights,
        torch.as_tensor(payload["client_class_counts"]),
        spec,
        int(payload["num_classes"]),
        budget,
    )
    concrete.append({
        "candidate_id": "classwise_weighting",
        "method": "classwise_weighting",
        "delta": classwise_delta,
        "status": "ok" if classwise_delta is not None else "invalid",
        **classwise_report,
    })

    subspace = ensure_fedavg_in_subspace(subspace_from_updates(client_deltas), delta_fedavg)
    q = subspace["Q"]
    a, fd = finite_difference_utility(train_margin_fn, theta0, q, epsilon=1e-3)
    solver_report = {"status": "not_run", "failure_reason": ""}
    if not fd["stable"] and not synthetic_cusp:
        return None, {"status": "finite_difference_unstable", "failure_reason": "finite difference failed stability thresholds"}, {
            "spec": spec, "theta0": theta0, "budget": budget, "concrete": concrete,
            "subspace": {k: v for k, v in subspace.items() if k != "Q"},
            "finite_difference": fd, "classwise": classwise_report,
        }

    if synthetic_cusp:
        raw = q @ a.nan_to_num(0.0).mean(dim=0)
        cusp_delta, cusp_norm = scale_to_budget(raw, budget)
        solver_report = {"status": "synthetic_success", "failure_reason": ""}
    else:
        norms = a.norm(dim=1)
        valid = norms > 1e-12
        valid_classes = torch.arange(int(payload["num_classes"]))[valid]
        head_ids = set(int(x) for x in metadata["head_class_ids"])
        head_mask = torch.tensor([int(x) in head_ids for x in valid_classes.tolist()], dtype=torch.bool)
        valid_a = a[valid] / norms[valid].unsqueeze(1)
        u_fedavg = (q.T @ delta_fedavg) / budget
        cusp_u, solver_report = solve_cusp(valid_a, u_fedavg, head_mask)
        if cusp_u is None:
            return None, solver_report, {
                "spec": spec, "theta0": theta0, "budget": budget, "concrete": concrete,
                "subspace": {k: v for k, v in subspace.items() if k != "Q"},
                "finite_difference": fd, "classwise": classwise_report,
            }
        raw = q @ (budget * cusp_u)
        cusp_delta, cusp_norm = scale_to_budget(raw, budget)

    concrete.append({
        "candidate_id": "oracle_cusp",
        "method": "oracle_cusp",
        "delta": cusp_delta,
        "status": "ok" if cusp_delta is not None else "invalid",
        **cusp_norm,
    })
    context = {
        "spec": spec,
        "theta0": theta0,
        "budget": budget,
        "concrete": concrete,
        "subspace": {k: v for k, v in subspace.items() if k != "Q"},
        "finite_difference": fd,
        "classwise": classwise_report,
    }
    return concrete, solver_report, context


def freeze_candidates(output_dir: Path, context: dict):
    output_dir.mkdir(parents=True, exist_ok=True)
    states, manifest_rows = {}, []
    budget = context["budget"]
    for item in context["concrete"]:
        delta = item["delta"]
        if delta is None or item["status"] != "ok":
            raise RuntimeError(f"candidate is invalid before freeze: {item['candidate_id']}")
        final_norm = float(delta.norm().item())
        if not norm_ok(final_norm, budget):
            raise RuntimeError(f"candidate norm mismatch: {item['candidate_id']} norm={final_norm} budget={budget}")
        states[item["candidate_id"]] = unflatten_state(context["theta0"] + delta, context["spec"])
        manifest_rows.append({
            "candidate_id": item["candidate_id"],
            "method": item["method"],
            "raw_norm": float(item["raw_norm"]),
            "final_norm": final_norm,
            "scale_factor": float(item["scale_factor"]),
            "candidate_hash": candidate_hash(delta),
        })
    states_path = output_dir / "candidate_states.pt"
    torch.save(states, states_path)
    manifest = {
        "schema_version": "cusp_round1_v1",
        "candidate_frozen_at": now_stamp(),
        "test_accessed": False,
        "num_concrete_candidates": len(states),
        "methods": list(ROUND1_METHODS),
        "norm_budget": budget,
        "norm_atol": NORM_ATOL,
        "norm_rtol": NORM_RTOL,
        "candidate_states_sha256": sha256_file(states_path),
        "candidates": manifest_rows,
    }
    write_json(output_dir / "candidate_manifest.json", manifest)
    return states, manifest


def summarize_results(context: dict, manifest: dict, metrics_by_id: dict, metadata: dict, output_dir: Path):
    rows = []
    fedavg = metrics_by_id["fedavg"]
    random_ids = [cid for cid in metrics_by_id if cid.startswith("random_reweight_")]
    random_summary = {
        key: summarize_values([metrics_by_id[cid][key] for cid in random_ids])
        for key in ("overall_acc", "macro_acc", "head_acc", "tail_acc")
    }
    norm_summary = summarize_values([
        item["final_norm"] for item in manifest["candidates"] if item["method"] == "random_reweight"
    ])
    for method in ROUND1_METHODS:
        if method == "random_reweight":
            row = {"method": method, "candidate_count": NUM_RANDOM, "update_norm": norm_summary["mean"]}
            for key, summary in random_summary.items():
                row[f"{key}_mean"] = summary["mean"]
                row[f"{key}_std"] = summary["std"]
                row[f"{key}_min"] = summary["min"]
                row[f"{key}_p25"] = summary["p25"]
                row[f"{key}_median"] = summary["median"]
                row[f"{key}_p75"] = summary["p75"]
                row[f"{key}_max"] = summary["max"]
                row[f"{key}_delta_vs_fedavg_median"] = summary["median"] - fedavg[key]
            rows.append(row)
            continue
        candidate_id = method
        metric = metrics_by_id[candidate_id]
        candidate = next(item for item in manifest["candidates"] if item["candidate_id"] == candidate_id)
        row = {
            "method": method,
            "candidate_count": 1,
            "overall_acc": metric["overall_acc"],
            "macro_acc": metric["macro_acc"],
            "head_acc": metric["head_acc"],
            "tail_acc": metric["tail_acc"],
            "update_norm": candidate["final_norm"],
        }
        for key in ("overall_acc", "macro_acc", "head_acc", "tail_acc"):
            row[f"{key}_delta_vs_fedavg"] = metric[key] - fedavg[key]
        rows.append(row)
    write_csv(output_dir / "oracle_method_summary.csv", rows)

    per_class_rows = []
    for candidate_id, metric in metrics_by_id.items():
        method = candidate_id if not candidate_id.startswith("random_reweight_") else "random_reweight"
        for row in metric["per_class"]:
            per_class_rows.append({"candidate_id": candidate_id, "method": method, **row})
    write_csv(output_dir / "oracle_per_class.csv", per_class_rows)

    random_rows = []
    for candidate in manifest["candidates"]:
        if candidate["method"] != "random_reweight":
            continue
        metric = metrics_by_id[candidate["candidate_id"]]
        row = {
            "candidate_id": candidate["candidate_id"],
            "coefficient_hash": next(
                item.get("coefficient_hash", "") for item in context["concrete"]
                if item["candidate_id"] == candidate["candidate_id"]
            ),
            "raw_norm": candidate["raw_norm"],
            "final_norm": candidate["final_norm"],
            "scale_factor": candidate["scale_factor"],
        }
        for key in ("overall_acc", "macro_acc", "head_acc", "tail_acc"):
            row[key] = metric[key]
            row[f"{key}_delta_vs_fedavg"] = metric[key] - fedavg[key]
        random_rows.append(row)
    write_csv(output_dir / "random_reweight_distribution.csv", random_rows)

    cusp = metrics_by_id["oracle_cusp"]
    classwise = metrics_by_id["classwise_weighting"]
    random_tail_median = random_summary["tail_acc"]["median"]
    pass_minimal = (
        cusp["tail_acc"] > fedavg["tail_acc"]
        and cusp["tail_acc"] > classwise["tail_acc"]
        and cusp["tail_acc"] > random_tail_median
        and cusp["head_acc"] >= fedavg["head_acc"] - 0.5
        and cusp["overall_acc"] >= fedavg["overall_acc"] - 0.5
    )
    verdict = "PASS_MINIMAL" if pass_minimal else "FAIL_MINIMAL"
    report_lines = [
        "# CUSP Minimal Pilot",
        "",
        f"Verdict: **{verdict}**",
        "",
        "| method | overall | macro | head/non-tail | tail | update norm |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        if row["method"] == "random_reweight":
            report_lines.append(
                f"| random_reweight median | {row['overall_acc_median']:.4f} | {row['macro_acc_median']:.4f} | "
                f"{row['head_acc_median']:.4f} | {row['tail_acc_median']:.4f} | {row['update_norm']:.6f} |"
            )
        else:
            report_lines.append(
                f"| {row['method']} | {row['overall_acc']:.4f} | {row['macro_acc']:.4f} | "
                f"{row['head_acc']:.4f} | {row['tail_acc']:.4f} | {row['update_norm']:.6f} |"
            )
    report_lines.extend([
        "",
        "This is a single-topology, single-seed, single-round centralized Oracle result and is not a paper conclusion.",
    ])
    (output_dir / "oracle_report.md").write_text("\n".join(report_lines), encoding="utf-8")
    return verdict


class SyntheticReplayModel:
    def __init__(self, weight: torch.Tensor):
        self.weight = weight.to(torch.float64)

    def logits_from_cached_features(self, features, state):
        vector = torch.cat([value.detach().cpu().to(torch.float64).reshape(-1) for value in state.values()])
        return features.to(torch.float64) @ (self.weight + vector.unsqueeze(1) * 0.05)


def synthetic_run(output_dir: Path):
    before = {
        "prompt_learner.general_ctx": torch.zeros(1, 2),
        "prompt_learner.class_aware_ctx": torch.zeros(3, 2),
    }
    spec = FlatSpec(
        keys=("prompt_learner.general_ctx", "prompt_learner.class_aware_ctx"),
        shapes=((1, 2), (3, 2)),
        dtypes=("torch.float32", "torch.float32"),
        offsets=((0, 2), (2, 8)),
    )
    theta0 = flatten_state(before, spec)
    client_deltas = torch.tensor([
        [1.0, 0.0, 0.5, 0.1, 0.0, 0.0, 0.2, 0.0],
        [0.2, 0.3, 0.1, 0.0, 0.0, 0.1, 0.0, -0.2],
        [-0.1, 0.2, 0.0, 0.0, 1.0, 0.8, 0.3, 0.2],
        [0.1, -0.2, 0.0, 0.0, -0.5, -0.4, -0.1, 0.0],
    ], dtype=torch.float64)
    local_states = [unflatten_state(theta0 + row, spec) for row in client_deltas]
    weights = torch.tensor([0.4, 0.3, 0.2, 0.1], dtype=torch.float64)
    delta_fedavg = client_deltas.T @ weights
    payload = {
        "flatten_spec": spec.as_dict(),
        "global_before_trainable": before,
        "local_trainable_states": local_states,
        "global_after_fedavg_trainable": unflatten_state(theta0 + delta_fedavg, spec),
        "selected_client_ids": [0, 1, 2, 3],
        "fedavg_weights": weights.tolist(),
        "client_sample_counts": [4, 3, 2, 1],
        "client_class_counts": torch.tensor([[3, 0, 0], [1, 2, 0], [0, 0, 2], [0, 0, 1]]),
        "num_classes": 3,
    }
    metadata = {
        "head_class_ids": [0, 1],
        "tail_class_ids": [2],
        "global_class_counts": [4, 2, 3],
        "accuracy_scale": "percent",
    }
    train_features = torch.eye(8, dtype=torch.float64).repeat(2, 1)
    train_labels = torch.tensor([0, 0, 1, 1, 2, 2, 2, 0] * 2)
    utility = torch.tensor([
        [1., 0., .1, 0., 0., 0., 0., 0.],
        [0., 1., 0., .1, 0., 0., 0., 0.],
        [-1., -.5, 0., 0., 1., 1., .1, 0.],
    ], dtype=torch.float64)
    train_margin = lambda theta: utility @ theta
    concrete, solver_report, context = build_candidates(payload, metadata, train_margin, synthetic_cusp=True)
    if concrete is None:
        fail_before_test(output_dir, solver_report, metadata, solver_report.get("failure_reason", "synthetic build failed"))
    states, manifest = freeze_candidates(output_dir, context)
    test_first_accessed_at = now_stamp()
    model = SyntheticReplayModel(torch.randn(8, 3, generator=torch.Generator().manual_seed(1), dtype=torch.float64))
    test_cache = {
        "features": train_features,
        "labels": train_labels,
        "sha256": "synthetic",
    }
    head, tail = class_groups(metadata, int(payload["num_classes"]))
    metrics = {}
    for item in context["concrete"]:
        logits = model.logits_from_cached_features(test_cache["features"], states[item["candidate_id"]])
        metrics[item["candidate_id"]] = accuracy_metrics(
            logits, test_cache["labels"], head, tail, support_counts=metadata["global_class_counts"]
        )
    verdict = summarize_results(context, manifest, metrics, metadata, output_dir)
    write_json(output_dir / "oracle_solver.json", jsonable(solver_report))
    write_json(output_dir / "oracle_metadata.json", jsonable({
        "schema_version": "cusp_round1_v1",
        "synthetic": True,
        "minimal_pilot_status": verdict,
        "candidate_frozen_at": manifest["candidate_frozen_at"],
        "test_first_accessed_at": test_first_accessed_at,
        "candidate_frozen_before_test": manifest["candidate_frozen_at"] <= test_first_accessed_at,
        "candidate_methods": list(ROUND1_METHODS),
        "num_concrete_candidates": len(context["concrete"]),
        "num_random": NUM_RANDOM,
        "accuracy_scale": "percent",
        "head_class_ids": metadata["head_class_ids"],
        "tail_class_ids": metadata["tail_class_ids"],
        "finite_difference": context["finite_difference"],
        "subspace": context["subspace"],
        "norm_budget": context["budget"],
        "test_leakage_check": {"utility_data_source": "train", "test_used_for_utility": False},
        "runtime_seconds": 0.0,
    }))


def real_run(args):
    started = time.time()
    dump_dir = args.run_dir / "oracle_cusp" / f"round_{args.communication_round:03d}"
    payload, metadata, train_cache = load_verified_dump(dump_dir)
    metadata_out = {
        "schema_version": "cusp_round1_v1",
        "synthetic": False,
        "dump_dir": str(dump_dir),
        "round_state_sha256_verified": bool(metadata.get("round_state_sha256")),
        "train_feature_cache_sha256_verified": bool(metadata.get("train_feature_cache_sha256")),
        "candidate_methods": list(ROUND1_METHODS),
        "num_random": NUM_RANDOM,
        "accuracy_scale": "percent",
        "head_class_ids": metadata["head_class_ids"],
        "tail_class_ids": metadata["tail_class_ids"],
        "trainable_keys": payload["trainable_keys"],
        "resolved_args": metadata.get("resolved_args", {}),
        "resolved_config": metadata.get("resolved_config", ""),
        "test_leakage_check": {"utility_data_source": "train", "test_used_for_utility": False},
    }
    cfg, trainer = build_promptfl_model(metadata)
    trainer.model.load_state_dict(payload["global_before_trainable"], strict=False)
    spec = FlatSpec.from_dict(payload["flatten_spec"])

    def train_margin(theta):
        logits = logits_from_vector(trainer.model, spec, theta, train_cache["features"])
        return decision_margins_by_class(logits, train_cache["labels"], int(payload["num_classes"]))

    concrete, solver_report, context = build_candidates(payload, metadata, train_margin, synthetic_cusp=False)
    metadata_out["finite_difference"] = context["finite_difference"]
    metadata_out["subspace"] = context["subspace"]
    metadata_out["norm_budget"] = context["budget"]
    if concrete is None:
        fail_before_test(
            args.output_dir,
            solver_report,
            metadata_out,
            solver_report.get("failure_reason", solver_report.get("status", "candidate build failed")),
        )
    states, manifest = freeze_candidates(args.output_dir, context)
    test_first_accessed_at = now_stamp()
    test_cache = build_test_cache(trainer, args.output_dir)
    head, tail = class_groups(metadata, int(payload["num_classes"]))
    metrics = {}
    for item in context["concrete"]:
        logits = trainer.model.logits_from_cached_features(test_cache["features"], states[item["candidate_id"]])
        metrics[item["candidate_id"]] = accuracy_metrics(
            logits, test_cache["labels"], head, tail, support_counts=metadata["global_class_counts"]
        )
    verdict = summarize_results(context, manifest, metrics, metadata, args.output_dir)
    write_json(args.output_dir / "oracle_solver.json", jsonable(solver_report))
    write_json(args.output_dir / "oracle_metadata.json", jsonable({
        **metadata_out,
        "minimal_pilot_status": verdict,
        "candidate_frozen_at": manifest["candidate_frozen_at"],
        "test_first_accessed_at": test_first_accessed_at,
        "candidate_frozen_before_test": manifest["candidate_frozen_at"] <= test_first_accessed_at,
        "official_test_cache_sha256": test_cache["sha256"],
        "runtime_seconds": time.time() - started,
    }))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--communication-round", type=int, default=10)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--synthetic-smoke", action="store_true")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.synthetic_smoke:
        synthetic_run(args.output_dir)
        return
    if args.run_dir is None:
        parser.error("--run-dir is required unless --synthetic-smoke is used")
    real_run(args)


if __name__ == "__main__":
    main()
