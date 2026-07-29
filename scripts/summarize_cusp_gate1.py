#!/usr/bin/env python
"""Summarize two Round-1 Oracle replay directories without recomputing candidates."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


REQUIRED_METHODS = {"fedavg", "random_reweight", "classwise_weighting", "oracle_cusp"}


def read_rows(path: Path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def as_float(row, key):
    try:
        value = float(row.get(key, ""))
    except Exception:
        return math.nan
    return value if math.isfinite(value) else math.nan


def load_topology(path: Path):
    summary_path = path / "oracle_method_summary.csv"
    solver_path = path / "oracle_solver.json"
    metadata_path = path / "oracle_metadata.json"
    failures = []
    if not summary_path.exists():
        return None, [f"missing {summary_path}"]
    rows = {row["method"]: row for row in read_rows(summary_path)}
    if set(rows) != REQUIRED_METHODS:
        failures.append(f"{summary_path} methods must be exactly {sorted(REQUIRED_METHODS)}")
    solver = json.loads(solver_path.read_text(encoding="utf-8")) if solver_path.exists() else {"status": "missing"}
    metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
    if solver.get("status") not in {"optimal", "optimal_inaccurate"}:
        failures.append(f"oracle_cusp solver not successful in {path}: {solver.get('status')}")
    if metadata.get("train_test_leakage_check", {}).get("test_used_for_utility", True):
        failures.append(f"test leakage flag is true in {path}")
    fedavg = rows.get("fedavg", {})
    random_row = rows.get("random_reweight", {})
    classwise = rows.get("classwise_weighting", {})
    cusp = rows.get("oracle_cusp", {})
    budget_ok = as_float(cusp, "norm_ratio") <= 1.0 + 1e-6
    tail_margin_ok = as_float(cusp, "test_tail_margin") >= as_float(fedavg, "test_tail_margin")
    random_ok = as_float(cusp, "predicted_tail_margin") >= as_float(random_row, "predicted_tail_margin_p50")
    classwise_ok = as_float(cusp, "predicted_tail_margin") >= as_float(classwise, "predicted_tail_margin")
    head_drop_ok = as_float(cusp, "test_head_acc") >= as_float(fedavg, "test_head_acc") - 0.005
    overall_drop_ok = as_float(cusp, "test_overall_acc") >= as_float(fedavg, "test_overall_acc") - 0.005
    checks = {
        "budget_ok": budget_ok,
        "tail_margin_or_accuracy_not_worse_than_fedavg": tail_margin_ok,
        "beats_random_median": random_ok,
        "beats_classwise": classwise_ok,
        "head_drop_within_half_point": head_drop_ok,
        "overall_drop_within_half_point": overall_drop_ok,
    }
    for key, passed in checks.items():
        if not passed:
            failures.append(f"{path}: {key} failed or missing")
    return {"path": str(path), "checks": checks, "solver_status": solver.get("status")}, failures


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oracle-dirs", type=Path, nargs=2, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    topologies, failures = [], []
    for path in args.oracle_dirs:
        topology, local_failures = load_topology(path)
        if topology is not None:
            topologies.append(topology)
        failures.extend(local_failures)
    gate = "PASS" if not failures and len(topologies) == 2 else "INCOMPLETE"
    summary = {
        "schema_version": "cusp_round1_v1",
        "gate": gate,
        "failure_reasons": failures,
        "topologies": topologies,
    }
    (args.output_dir / "cusp_gate1_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (args.output_dir / "cusp_gate1_report.md").write_text(
        "# CUSP Gate 1\n\n"
        f"Result: **{gate}**\n\n"
        "This script reads frozen Oracle replay outputs only; it does not rebuild candidates or re-read raw data.\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
