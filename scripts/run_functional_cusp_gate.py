#!/usr/bin/env python
"""Offline Functional CUSP Gate for one existing round-10 dump.

This script does not train. It consumes a CUSP minimal dump, rebuilds the same
train partition from the stored args/seed, creates a one-off train feature
cache, freezes four equal-norm candidates, and only then reads official test.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
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

from utils.cusp_minimal import (
    FlatSpec,
    SCHEMA_VERSION,
    _classwise_delta,
    _oracle_cusp_delta,
    flatten_state,
    load_cusp_minimal_dump,
    sha256_json,
    unflatten_state,
    write_csv,
    write_json,
)
from utils.functional_cusp import (
    build_functional_cusp_delta,
    candidate_hash_from_delta,
    fedavg_delta_from_payload,
    spearmanr_simple,
)


GATE_METHODS = ("fedavg", "classwise_aggregation", "fixed_weight_cusp", "functional_cusp")


def now_stamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_promptfl_trainer(metadata: dict, output_dir: Path):
    from Dassl.dassl.engine import build_trainer
    from federated_main import setup_cfg

    args = SimpleNamespace(**metadata["resolved_args"])
    args.output_dir = str(output_dir)
    cfg = setup_cfg(args)
    trainer = build_trainer(cfg)
    trainer.fed_before_train(is_global=True)
    return cfg, trainer


def validate_cache_counts(cache: dict, payload: dict) -> None:
    labels = torch.as_tensor(cache["labels"], dtype=torch.long)
    client_ids = torch.as_tensor(cache["client_ids"], dtype=torch.long)
    selected = [int(x) for x in payload["selected_client_ids"]]
    expected = torch.as_tensor(payload["client_class_counts"], dtype=torch.long)
    observed = torch.zeros_like(expected)
    for row_id, client_id in enumerate(selected):
        mask = client_ids == client_id
        observed[row_id] = torch.bincount(labels[mask], minlength=int(payload["num_classes"]))[: int(payload["num_classes"])]
    if not torch.equal(observed, expected):
        mismatch = (observed != expected).nonzero()
        first = mismatch[0].tolist() if mismatch.numel() else []
        raise RuntimeError(
            "Rebuilt train cache counts do not match dump client_class_counts; "
            f"first_mismatch={first}"
        )
    global_observed = observed.sum(dim=0)
    global_expected = torch.as_tensor(payload["global_class_counts"], dtype=torch.long)
    if not torch.equal(global_observed, global_expected):
        raise RuntimeError("Rebuilt train cache counts do not match dump global_class_counts")


def build_train_feature_cache(cfg, trainer, payload: dict, output_dir: Path) -> tuple[dict, str]:
    from Dassl.dassl.data.data_manager import build_data_loader
    from Dassl.dassl.data.transforms import build_transform

    selected = [int(x) for x in payload["selected_client_ids"]]
    transform = build_transform(cfg, is_train=False)
    model = trainer.model
    was_training = model.training
    model.eval()
    features, labels, client_ids = [], [], []
    try:
        with torch.no_grad():
            for client_id in selected:
                data_source = trainer.dm.dataset.federated_train_x[client_id]
                loader = build_data_loader(
                    cfg,
                    sampler_type="SequentialSampler",
                    data_source=data_source,
                    batch_size=cfg.DATALOADER.TEST.BATCH_SIZE,
                    tfm=transform,
                    is_train=False,
                    dataset_wrapper=None,
                    class_names=trainer.dm.dataset.classnames,
                )
                for batch in loader:
                    images = batch["img"].to(model.logit_scale.device)
                    batch_labels = batch["label"].detach().long().cpu()
                    image_features = model.image_encoder(images.type(model.dtype))
                    image_features = image_features / image_features.norm(dim=-1, keepdim=True).clamp_min(1e-12)
                    features.append(image_features.detach().float().cpu())
                    labels.append(batch_labels)
                    client_ids.append(torch.full_like(batch_labels, int(client_id)))
    finally:
        model.train(was_training)
    cache = {
        "schema_version": SCHEMA_VERSION,
        "source": "train",
        "test_used": False,
        "features": torch.cat(features, dim=0),
        "labels": torch.cat(labels, dim=0),
        "client_ids": torch.cat(client_ids, dim=0),
        "classnames": list(trainer.dm.dataset.classnames),
        "num_classes": int(payload["num_classes"]),
        "num_clients": int(max(selected) + 1),
    }
    validate_cache_counts(cache, payload)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "train_feature_cache.pt"
    torch.save(cache, path)
    return cache, sha256_file(path)


def build_gate_candidates(payload: dict, metadata: dict, train_cache: dict, trainer, args) -> tuple[dict, list[dict], list[dict], dict]:
    spec = FlatSpec.from_dict(payload["flatten_spec"])
    theta_t, _, client_deltas, delta_avg = fedavg_delta_from_payload(payload, spec)
    budget = float(delta_avg.norm().item())
    weights = torch.as_tensor(payload["fedavg_weights"], dtype=torch.float64)
    states, rows = {}, []

    def add_candidate(candidate_id: str, method: str, delta: torch.Tensor, extra: dict | None = None):
        final_norm = float(delta.norm().item())
        rows.append({
            "candidate_id": candidate_id,
            "method": method,
            "update_norm": final_norm,
            "norm_budget": budget,
            "candidate_hash": candidate_hash_from_delta(delta),
            **(extra or {}),
        })
        states[candidate_id] = unflatten_state(theta_t + delta, spec)

    add_candidate("fedavg", "fedavg", delta_avg, {"fallback": False})

    classwise_delta, classwise_report = _classwise_delta(payload, spec, theta_t, budget)
    add_candidate("classwise_aggregation", "classwise_aggregation", classwise_delta, {
        "fallback": False,
        "raw_norm": classwise_report.get("raw_norm"),
        "scale_factor": classwise_report.get("scale_factor"),
    })

    fixed_delta, fixed_report = _oracle_cusp_delta(payload, metadata, client_deltas.T, delta_avg, budget)
    add_candidate("fixed_weight_cusp", "fixed_weight_cusp", fixed_delta, {
        "fallback": False,
        "raw_norm": fixed_report.get("raw_norm"),
        "scale_factor": fixed_report.get("scale_factor"),
    })

    functional_delta, functional_report, diagnostics = build_functional_cusp_delta(
        payload,
        train_cache,
        trainer.model,
        rank_max=args.rank_max,
        probe_rel_step=args.probe_rel_step,
        steer_ratio=args.steer_ratio,
        class_count_power=args.class_count_power,
        batch_size=args.probe_batch_size,
    )
    add_candidate("functional_cusp", "functional_cusp", functional_delta, {
        "fallback": bool(functional_report.get("fallback", False)),
        "fallback_reason": functional_report.get("fallback_reason", ""),
    })
    context = {
        "norm_budget": budget,
        "weights_hash": sha256_json({"fedavg_weights": [round(float(x), 12) for x in weights.tolist()]}),
        "functional_report": functional_report,
    }
    return states, rows, diagnostics, context


def freeze_candidates(output_dir: Path, states: dict, rows: list[dict], manifest: dict) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(states, output_dir / "candidate_states.pt")
    write_csv(output_dir / "candidate_manifest.csv", rows)
    frozen = {
        **manifest,
        "candidate_names": [row["candidate_id"] for row in rows],
        "candidate_hashes": {row["candidate_id"]: row["candidate_hash"] for row in rows},
        "candidate_count": len(rows),
        "selection_source": "train_probe_only",
        "candidates_frozen": True,
        "candidate_frozen_at": now_stamp(),
    }
    write_json(output_dir / "candidate_manifest.json", frozen)
    return frozen


def build_test_cache(trainer, output_dir: Path) -> dict:
    model = trainer.model
    was_training = model.training
    model.eval()
    features, labels = [], []
    try:
        with torch.no_grad():
            for batch in trainer.test_loader:
                images, batch_labels = trainer.parse_batch_train(batch)
                image_features = model.image_encoder(images.type(model.dtype))
                image_features = image_features / image_features.norm(dim=-1, keepdim=True).clamp_min(1e-12)
                features.append(image_features.detach().float().cpu())
                labels.append(batch_labels.detach().long().cpu())
    finally:
        model.train(was_training)
    cache = {
        "schema_version": SCHEMA_VERSION,
        "source": "official_test",
        "features": torch.cat(features, dim=0),
        "labels": torch.cat(labels, dim=0),
    }
    torch.save(cache, output_dir / "official_test_cache.pt")
    return cache


def compute_metrics(logits: torch.Tensor, labels: torch.Tensor, head_ids: set[int], tail_ids: set[int]) -> tuple[dict, list[dict]]:
    logits = logits.detach().cpu()
    labels = labels.detach().cpu().long()
    preds = logits.argmax(dim=1)
    num_classes = logits.shape[1]
    per_class = []
    for class_id in range(num_classes):
        mask = labels == class_id
        total = int(mask.sum().item())
        correct = int((preds[mask] == labels[mask]).sum().item()) if total else 0
        acc = 100.0 * correct / total if total else math.nan
        per_class.append({
            "class_id": class_id,
            "test_count": total,
            "correct_count": correct,
            "class_acc": acc,
        })
    finite = [row["class_acc"] for row in per_class if math.isfinite(row["class_acc"])]

    def group_mean(ids):
        vals = [row["class_acc"] for row in per_class if row["class_id"] in ids and math.isfinite(row["class_acc"])]
        return float(sum(vals) / len(vals)) if vals else math.nan

    return {
        "overall_acc": 100.0 * float((preds == labels).double().mean().item()),
        "macro_acc": float(sum(finite) / len(finite)) if finite else math.nan,
        "head_acc": group_mean(head_ids),
        "tail_acc": group_mean(tail_ids),
    }, per_class


def evaluate_frozen_candidates(output_dir: Path, metadata: dict, payload: dict, trainer) -> tuple[list[dict], list[dict]]:
    states = torch.load(output_dir / "candidate_states.pt", map_location="cpu", weights_only=False)
    manifest = json.loads((output_dir / "candidate_manifest.json").read_text(encoding="utf-8"))
    test_first_accessed_at = now_stamp()
    test_cache = build_test_cache(trainer, output_dir)
    head_ids = set(int(x) for x in metadata["head_class_ids"])
    tail_ids = set(int(x) for x in metadata["tail_class_ids"])
    metric_rows, per_class_rows = [], []
    for row in csv.DictReader((output_dir / "candidate_manifest.csv").open(encoding="utf-8")):
        candidate_id = row["candidate_id"]
        logits = trainer.model.logits_from_cached_features(test_cache["features"], states[candidate_id])
        metrics, per_class = compute_metrics(logits, test_cache["labels"], head_ids, tail_ids)
        metric_rows.append({
            "partition": metadata["resolved_args"].get("partition", ""),
            "seed": metadata["resolved_args"].get("seed", ""),
            "round": metadata["communication_round"],
            "candidate_id": candidate_id,
            "method": row["method"],
            "update_norm": row["update_norm"],
            **metrics,
        })
        for item in per_class:
            per_class_rows.append({"candidate_id": candidate_id, "method": row["method"], **item})
    manifest["test_first_accessed_at"] = test_first_accessed_at
    manifest["candidate_frozen_before_test"] = manifest["candidate_frozen_at"] <= test_first_accessed_at
    write_json(output_dir / "candidate_manifest.json", manifest)
    return metric_rows, per_class_rows


def finalize_diagnostics(output_dir: Path, diagnostics: list[dict], metric_rows: list[dict], per_class_rows: list[dict], context: dict, metadata: dict):
    fed_per_class = {int(row["class_id"]): float(row["class_acc"]) for row in per_class_rows if row["method"] == "fedavg"}
    func_per_class = {int(row["class_id"]): float(row["class_acc"]) for row in per_class_rows if row["method"] == "functional_cusp"}
    for row in diagnostics:
        class_id = int(row["class_id"])
        row["partition"] = metadata["resolved_args"].get("partition", "")
        row["seed"] = metadata["resolved_args"].get("seed", "")
        if class_id in fed_per_class and class_id in func_per_class:
            row["realized_test_class_delta"] = func_per_class[class_id] - fed_per_class[class_id]
        else:
            row["realized_test_class_delta"] = math.nan
    write_csv(output_dir / "class_functional_diagnostics.csv", diagnostics)
    write_csv(output_dir / "candidate_metrics.csv", metric_rows)
    write_csv(output_dir / "candidate_per_class_metrics.csv", per_class_rows)

    metrics = {row["method"]: row for row in metric_rows}
    fedavg = metrics["fedavg"]
    classwise = metrics["classwise_aggregation"]
    functional = metrics["functional_cusp"]
    predicted = [row.get("predicted_steering_utility", math.nan) for row in diagnostics]
    realized = [row.get("realized_test_class_delta", math.nan) for row in diagnostics]
    corr = spearmanr_simple(predicted, realized)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "partition": metadata["resolved_args"].get("partition", ""),
        "seed": metadata["resolved_args"].get("seed", ""),
        "round": metadata["communication_round"],
        "functional_minus_fedavg": {
            "overall_acc": float(functional["overall_acc"]) - float(fedavg["overall_acc"]),
            "macro_acc": float(functional["macro_acc"]) - float(fedavg["macro_acc"]),
            "head_acc": float(functional["head_acc"]) - float(fedavg["head_acc"]),
            "tail_acc": float(functional["tail_acc"]) - float(fedavg["tail_acc"]),
        },
        "functional_minus_classwise": {
            "overall_acc": float(functional["overall_acc"]) - float(classwise["overall_acc"]),
            "macro_acc": float(functional["macro_acc"]) - float(classwise["macro_acc"]),
            "head_acc": float(functional["head_acc"]) - float(classwise["head_acc"]),
            "tail_acc": float(functional["tail_acc"]) - float(classwise["tail_acc"]),
        },
        "predicted_realized_spearman": corr,
        "fallback": bool(context["functional_report"].get("fallback", False)),
        "functional_report": context["functional_report"],
    }
    write_json(output_dir / "gate_summary.json", summary)
    return summary


def run_gate(args) -> None:
    output_dir = args.output_dir
    payload, metadata = load_cusp_minimal_dump(args.dump_dir)
    cfg, trainer = build_promptfl_trainer(metadata, output_dir / "model_build")
    cache, cache_hash = build_train_feature_cache(cfg, trainer, payload, output_dir)
    states, candidate_rows, diagnostics, context = build_gate_candidates(payload, metadata, cache, trainer, args)
    dump_hash = sha256_file(Path(args.dump_dir) / "round_state.pt")
    manifest = freeze_candidates(output_dir, states, candidate_rows, {
        "schema_version": SCHEMA_VERSION,
        "partition": metadata["resolved_args"].get("partition", ""),
        "seed": metadata["resolved_args"].get("seed", ""),
        "round": metadata["communication_round"],
        "dump_hash": dump_hash,
        "cache_hash": cache_hash,
        "hyperparameters": context["functional_report"].get("hyperparameters", {}),
    })
    metric_rows, per_class_rows = evaluate_frozen_candidates(output_dir, metadata, payload, trainer)
    summary = finalize_diagnostics(output_dir, diagnostics, metric_rows, per_class_rows, context, metadata)
    write_json(output_dir / "functional_gate_metadata.json", {
        **manifest,
        "gate_summary": summary,
        "test_isolation": "train cache and candidates are written before official_test_cache.pt is created",
    })
    print(f"Functional CUSP Gate finished: {output_dir}")


def synthetic_smoke(output_dir: Path) -> None:
    from utils.functional_cusp import client_disagreement_subspace, solve_safe_direction

    delta_avg = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64)
    client_deltas = torch.tensor([[1.0, 0.2, 0.0], [1.0, -0.2, 0.0], [1.0, 0.0, 0.3]], dtype=torch.float64)
    weights = torch.tensor([1 / 3, 1 / 3, 1 / 3], dtype=torch.float64)
    q, report = client_disagreement_subspace(client_deltas, delta_avg, weights, rank_max=2)
    if q is None:
        raise RuntimeError("synthetic subspace unexpectedly fell back")
    if float((q.T @ delta_avg).abs().max().item()) > 1e-10:
        raise RuntimeError("synthetic Q is not orthogonal to FedAvg")
    v, safe = solve_safe_direction(torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float64), torch.tensor([10, 2]))
    if v is None or safe.get("safe_dot", 0.0) < -1e-10:
        raise RuntimeError("synthetic safe projection failed")
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "synthetic_smoke.json", {"status": "PASS", "subspace": report, "safe": safe})
    print(f"Synthetic smoke passed: {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dump-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--stage", choices=["gate", "synthetic"], default="gate")
    parser.add_argument("--rank-max", type=int, default=8)
    parser.add_argument("--probe-rel-step", type=float, default=0.1)
    parser.add_argument("--steer-ratio", type=float, default=0.25)
    parser.add_argument("--class-count-power", type=float, default=0.5)
    parser.add_argument("--probe-batch-size", type=int, default=2048)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.stage == "synthetic":
        synthetic_smoke(args.output_dir or Path("output/functional_cusp_synthetic"))
        return
    if args.dump_dir is None or args.output_dir is None:
        raise SystemExit("--dump-dir and --output-dir are required for --stage gate")
    run_gate(args)


if __name__ == "__main__":
    main()
