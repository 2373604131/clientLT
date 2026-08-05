#!/usr/bin/env python
"""Run the first-version Visual-Semantic Boundary Repair offline Gate.

All audit diagnostics, repair solving, safety backtracking, candidate hashing,
and final-update norm matching happen before this script optionally reads the
official test loader.  The selected parameters are command-line values and are
saved in the frozen candidate manifest.
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
from typing import Mapping

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.boundary_audit import build_audit_cache, load_boundary_round_dump, sha256_file, validate_audit_cache
from utils.boundary_gate import BoundaryGateConfig, build_boundary_candidates
from utils.cusp_minimal import jsonable, write_csv, write_json


# ``build_boundary_candidates`` returns a mixed context: some entries are
# compact diagnostics intended for the report, while others are runtime-only
# tensors/objects used to construct and evaluate candidates.  Keep this list
# explicit so a future runtime object cannot silently produce a huge JSON file
# (or fail serialization, as ``FlatSpec`` does).
DIAGNOSTIC_CONTEXT_FIELDS = (
    "fragile_edge_ids",
    "edge_catalog",
    "head_class_ids",
    "tail_class_ids",
    "norm_budget",
    "config",
    "gradient_rows",
    "solver_report",
    "cap_report",
    "repair_choice",
    "accepted_candidate_edge_nonregression_rate",
    "substantive_repair_edge_rate",
    "substantive_repair_all_fragile_edges",
    "boundary_reversal_rate",
    "support_counterfactuals_norm_matched",
)


def diagnostic_context_for_json(context: Mapping) -> dict:
    """Return the bounded, report-only portion of the Gate context."""
    return {field: context[field] for field in DIAGNOSTIC_CONTEXT_FIELDS if field in context}


def now_stamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def build_promptfl_trainer(metadata: dict, output_dir: Path):
    from Dassl.dassl.engine import build_trainer
    from federated_main import setup_cfg

    args = SimpleNamespace(**metadata["resolved_args"])
    args.output_dir = str(output_dir)
    cfg = setup_cfg(args)
    trainer = build_trainer(cfg)
    trainer.fed_before_train(is_global=True)
    return cfg, trainer


def load_or_create_audit_cache(cfg, trainer, payload: dict, dump_dir: Path, output_dir: Path, batch_size: int) -> tuple[dict, str]:
    cache_path = dump_dir / "audit_cache.pt"
    if cache_path.exists():
        cache = torch.load(cache_path, map_location="cpu", weights_only=False)
        validate_audit_cache(cache, payload)
        return cache, sha256_file(cache_path)
    return build_audit_cache(cfg, trainer, payload, output_dir, batch_size=batch_size)


def freeze_candidates(output_dir: Path, states: dict, rows: list[dict], manifest: dict) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(states, output_dir / "candidate_states.pt")
    write_csv(output_dir / "candidate_manifest.csv", rows)
    frozen = {
        **manifest,
        "candidate_names": [row["candidate_id"] for row in rows],
        "candidate_hashes": {row["candidate_id"]: row["candidate_hash"] for row in rows},
        "candidate_count": len(rows),
        "candidate_frozen_at": now_stamp(),
        "test_accessed": False,
    }
    write_json(output_dir / "candidate_manifest.json", frozen)
    return frozen


def build_test_cache(trainer, output_dir: Path) -> dict:
    """Read official test exactly once, after the candidate manifest is frozen."""
    model = trainer.model
    was_training = model.training
    features, labels = [], []
    try:
        model.eval()
        with torch.no_grad():
            for batch in trainer.test_loader:
                images, batch_labels = trainer.parse_batch_train(batch)
                features.append(model.encode_audit_images(images).detach().float().cpu())
                labels.append(batch_labels.detach().long().cpu())
    finally:
        model.train(was_training)
    cache = {"source": "official_test", "features": torch.cat(features), "labels": torch.cat(labels)}
    torch.save(cache, output_dir / "official_test_cache.pt")
    return cache


def compute_metrics(logits: torch.Tensor, labels: torch.Tensor, head_ids: set[int], tail_ids: set[int]) -> tuple[dict, list[dict]]:
    logits = logits.detach().cpu()
    labels = labels.detach().cpu().long()
    prediction = logits.argmax(dim=1)
    per_class = []
    for class_id in range(int(logits.shape[1])):
        mask = labels == class_id
        count = int(mask.sum().item())
        accuracy = 100.0 * float((prediction[mask] == labels[mask]).double().mean().item()) if count else math.nan
        per_class.append({"class_id": class_id, "test_count": count, "class_acc": accuracy})

    def group_mean(ids: set[int]) -> float:
        values = [row["class_acc"] for row in per_class if row["class_id"] in ids and math.isfinite(row["class_acc"])]
        return float(sum(values) / len(values)) if values else math.nan

    finite = [row["class_acc"] for row in per_class if math.isfinite(row["class_acc"])]
    return {
        "overall_acc": 100.0 * float((prediction == labels).double().mean().item()),
        "macro_acc": float(sum(finite) / len(finite)) if finite else math.nan,
        "non_tail_acc": group_mean(set(range(int(logits.shape[1]))) - tail_ids),
        "head_acc": group_mean(head_ids),
        "tail_acc": group_mean(tail_ids),
    }, per_class


def evaluate_frozen_candidates(output_dir: Path, metadata: dict, context: dict, trainer) -> tuple[list[dict], list[dict]]:
    states = torch.load(output_dir / "candidate_states.pt", map_location="cpu", weights_only=False)
    candidate_rows = list(csv.DictReader((output_dir / "candidate_manifest.csv").open(encoding="utf-8")))
    test_first_accessed_at = now_stamp()
    test_cache = build_test_cache(trainer, output_dir)
    head_ids = {int(value) for value in context["head_class_ids"]}
    tail_ids = {int(value) for value in context["tail_class_ids"]}
    metrics, per_class_rows = [], []
    for row in candidate_rows:
        candidate_id = row["candidate_id"]
        logits = trainer.model.logits_from_cached_features(test_cache["features"], states[candidate_id])
        values, per_class = compute_metrics(logits, test_cache["labels"], head_ids, tail_ids)
        metrics.append({
            "candidate_id": candidate_id,
            "method": row["method"],
            "partition": metadata["resolved_args"].get("partition", ""),
            "seed": metadata["resolved_args"].get("seed", ""),
            "round": metadata.get("communication_round", ""),
            **values,
        })
        per_class_rows.extend({"candidate_id": candidate_id, "method": row["method"], **item} for item in per_class)
    manifest_path = output_dir / "candidate_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.update({"test_accessed": True, "test_first_accessed_at": test_first_accessed_at, "candidate_frozen_before_test": True})
    write_json(manifest_path, manifest)
    return metrics, per_class_rows


def summarize_diagnostics(rows: list[dict], context: dict) -> dict:
    fragile = [row for row in rows if row.get("fragile_selected")]

    def values(key: str) -> list[float]:
        return [float(row[key]) for row in fragile if isinstance(row.get(key), (int, float)) and math.isfinite(float(row[key]))]

    def summary(key: str) -> dict:
        items = values(key)
        return {
            "count": len(items),
            "mean": float(sum(items) / len(items)) if items else math.nan,
            "median": float(sorted(items)[len(items) // 2]) if items else math.nan,
            "positive_rate": float(sum(item > 0 for item in items) / len(items)) if items else math.nan,
        }

    return {
        "diagnostic_edge_count": len(rows),
        "fragile_edge_count": len(fragile),
        "local_audit_gain": summary("local_audit_gain"),
        "gain_support_normalized": summary("gain_support_normalized"),
        "gain_support_actual": summary("gain_support_actual"),
        "gain_all_fedavg": summary("gain_all_fedavg"),
        "dilution": summary("dilution"),
        "interference": summary("interference"),
        "solver": context["solver_report"],
        "repair": context["repair_choice"],
        "accepted_candidate_edge_nonregression_rate": context["accepted_candidate_edge_nonregression_rate"],
        "substantive_repair_edge_rate": context["substantive_repair_edge_rate"],
        "substantive_repair_all_fragile_edges": context["substantive_repair_all_fragile_edges"],
        "boundary_reversal_rate": context["boundary_reversal_rate"],
    }


def gate_config(args: argparse.Namespace) -> BoundaryGateConfig:
    return BoundaryGateConfig(
        gamma=args.gamma,
        tau=args.tau,
        min_support_clients=args.min_support_clients,
        max_edges_per_class=args.max_edges_per_class,
        max_total_edges=args.max_total_edges,
        repair_ratio=args.repair_ratio,
        min_deficit_closure=args.min_deficit_closure,
        substantive_deficit_closure=args.substantive_deficit_closure,
        max_non_target_margin_drop=args.max_non_target_margin_drop,
        max_semantic_repair_drift=args.max_semantic_repair_drift,
        gradient_batch_size=args.batch_size,
        solver_max_iterations=args.solver_max_iterations,
        solver_tolerance=args.solver_tolerance,
        solver_ridge=args.solver_ridge,
        random_seed=args.random_seed,
        tail_class_ratio=args.tail_class_ratio,
    )


def run_gate(args: argparse.Namespace) -> None:
    payload, metadata = load_boundary_round_dump(args.dump_dir)
    if bool(metadata.get("test_used_before_dump", False)):
        raise RuntimeError("dump is invalid: test access occurred before audit/candidate construction")
    cfg, trainer = build_promptfl_trainer(metadata, args.output_dir / "model_build")
    audit_cache, cache_hash = load_or_create_audit_cache(cfg, trainer, payload, args.dump_dir, args.output_dir, args.batch_size)
    config = gate_config(args)
    states, candidate_rows, diagnostics, context = build_boundary_candidates(trainer.model, payload, metadata, audit_cache, config)
    for row in diagnostics:
        row.update({
            "partition": metadata["resolved_args"].get("partition", ""),
            "seed": metadata["resolved_args"].get("seed", ""),
            "round": metadata.get("communication_round", ""),
        })
    write_csv(args.output_dir / "edge_diagnostics.csv", diagnostics)
    write_json(
        args.output_dir / "edge_diagnostics.json",
        {"rows": diagnostics, "context": diagnostic_context_for_json(context)},
    )
    manifest = freeze_candidates(args.output_dir, states, candidate_rows, {
        "schema_version": "visual_semantic_boundary_v1",
        "dump_hash": sha256_file(args.dump_dir / "round_state.pt"),
        "audit_cache_hash": cache_hash,
        "partition": metadata["resolved_args"].get("partition", ""),
        "seed": metadata["resolved_args"].get("seed", ""),
        "round": metadata.get("communication_round", ""),
        "hyperparameters": jsonable(context["config"]),
        "diagnostic_counterfactuals_norm_matched": False,
    })
    summary = summarize_diagnostics(diagnostics, context)
    if args.evaluate_official_test:
        metrics, per_class = evaluate_frozen_candidates(args.output_dir, metadata, context, trainer)
        write_csv(args.output_dir / "candidate_metrics.csv", metrics)
        write_csv(args.output_dir / "candidate_per_class_metrics.csv", per_class)
        manifest = json.loads((args.output_dir / "candidate_manifest.json").read_text(encoding="utf-8"))
    write_json(args.output_dir / "gate_summary.json", {**summary, "candidate_manifest": manifest})
    print(f"Boundary Gate finished: {args.output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dump-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--evaluate-official-test", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--gamma", type=float, default=0.5)
    parser.add_argument("--tau", type=float, default=0.0)
    parser.add_argument("--min-support-clients", type=int, default=2)
    parser.add_argument("--max-edges-per-class", type=int, default=3)
    parser.add_argument("--max-total-edges", type=int, default=60)
    parser.add_argument("--repair-ratio", type=float, default=0.25)
    parser.add_argument("--min-deficit-closure", type=float, default=0.0)
    parser.add_argument("--substantive-deficit-closure", type=float, default=0.1)
    parser.add_argument("--max-non-target-margin-drop", type=float, default=0.05)
    parser.add_argument("--max-semantic-repair-drift", type=float, default=0.01)
    parser.add_argument("--solver-max-iterations", type=int, default=500)
    parser.add_argument("--solver-tolerance", type=float, default=1e-8)
    parser.add_argument("--solver-ridge", type=float, default=1e-10)
    parser.add_argument("--random-seed", type=int, default=2026)
    parser.add_argument("--tail-class-ratio", type=float, default=0.2)
    return parser.parse_args()


def main() -> None:
    run_gate(parse_args())


if __name__ == "__main__":
    main()
