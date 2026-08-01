#!/usr/bin/env python
"""Summarize the two Functional CUSP Gate outputs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def read_metrics(directory: Path) -> dict:
    rows = list(csv.DictReader((directory / "candidate_metrics.csv").open(encoding="utf-8")))
    return {row["method"]: row for row in rows}


def read_case(directory: Path) -> dict:
    summary = json.loads((directory / "gate_summary.json").read_text(encoding="utf-8"))
    metrics = read_metrics(directory)
    fedavg = metrics["fedavg"]
    classwise = metrics["classwise_aggregation"]
    functional = metrics["functional_cusp"]
    tail_gt_fedavg = float(functional["tail_acc"]) > float(fedavg["tail_acc"])
    overall_safe = float(functional["overall_acc"]) >= float(fedavg["overall_acc"]) - 0.5
    common_safe = float(functional["head_acc"]) >= float(fedavg["head_acc"]) - 0.5
    beats_classwise = float(functional["tail_acc"]) > float(classwise["tail_acc"])
    positive_corr = float(summary["predicted_realized_spearman"]) > 0
    return {
        "directory": str(directory),
        "partition": summary["partition"],
        "tail_gt_fedavg": tail_gt_fedavg,
        "overall_safe": overall_safe,
        "common_safe": common_safe,
        "beats_classwise": beats_classwise,
        "positive_corr": positive_corr,
        "fallback": bool(summary["fallback"]),
        "summary": summary,
        "metrics": metrics,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clientlt-dir", type=Path, required=True)
    parser.add_argument("--dirichlet-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    cases = [read_case(args.clientlt_dir), read_case(args.dirichlet_dir)]
    both_tail = all(case["tail_gt_fedavg"] for case in cases)
    both_safe = all(case["overall_safe"] and case["common_safe"] for case in cases)
    one_beats_classwise = any(case["beats_classwise"] for case in cases)
    both_corr = all(case["positive_corr"] for case in cases)
    clientlt_pass = cases[0]["tail_gt_fedavg"] and cases[0]["overall_safe"] and cases[0]["common_safe"]
    dirichlet_pass = cases[1]["tail_gt_fedavg"] and cases[1]["overall_safe"] and cases[1]["common_safe"]
    if both_tail and both_safe and one_beats_classwise and both_corr:
        verdict = "STRONG_PASS"
    elif clientlt_pass and not dirichlet_pass:
        verdict = "PARTIAL_PASS"
    else:
        verdict = "FAIL"

    args.output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "verdict": verdict,
        "criteria": {
            "both_tail_gt_fedavg": both_tail,
            "both_overall_and_common_safe": both_safe,
            "at_least_one_beats_classwise": one_beats_classwise,
            "both_predicted_realized_correlations_positive": both_corr,
        },
        "cases": cases,
    }
    (args.output_dir / "two_topology_gate_summary.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Two-topology Functional CUSP Gate: {verdict}")
    print(args.output_dir / "two_topology_gate_summary.json")


if __name__ == "__main__":
    main()
