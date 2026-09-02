#!/usr/bin/env python3
"""Summarize the one-experiment Client-LT functional-coverage validation."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"Refusing to write empty CSV: {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def _protocol(run_dir: Path) -> dict:
    path = run_dir / "functional_coverage" / "protocol.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _count_matrix(run_dir: Path) -> np.ndarray:
    rows = _read_csv(run_dir / "client_class_counts.csv")
    rows.sort(key=lambda row: int(row["client_id"]))
    return np.asarray(
        [[int(row[f"class_{class_id}"]) for class_id in range(100)] for row in rows],
        dtype=np.int64,
    )


def _schedule(run_dir: Path) -> list[list[int]]:
    rows = _read_csv(run_dir / "lora_aggregation_weights.csv")
    grouped: dict[int, list[int]] = {}
    for row in rows:
        grouped.setdefault(int(row["communication_round"]), []).append(int(row["client_id"]))
    return [grouped[round_id] for round_id in sorted(grouped)]


def _round_metrics(run_dir: Path) -> list[dict]:
    rows = []
    for raw in _read_csv(run_dir / "round_metrics.csv"):
        epoch = int(raw["epoch"])
        if epoch < 0:
            continue
        head = float(raw["non_tail_acc"])
        tail = float(raw["bottom20_tail_acc"])
        rows.append(
            {
                "epoch": epoch,
                "communication_round": epoch + 1,
                "overall": float(raw["overall_acc"]),
                "head": head,
                "tail": tail,
                "hmean": 2.0 * head * tail / (head + tail) if head + tail else 0.0,
                "macro_f1": float(raw["macro_f1"]),
            }
        )
    if not rows:
        raise RuntimeError(f"No post-training round metrics under {run_dir}")
    return sorted(rows, key=lambda row: row["communication_round"])


def _per_class_accuracy(run_dir: Path, epoch: int) -> dict[int, float]:
    return {
        int(row["class_id"]): float(row["per_class_acc"])
        for row in _read_csv(run_dir / f"per_class_accuracy_epoch_{int(epoch)}.csv")
    }


def _coverage(run_dir: Path) -> list[dict]:
    return [
        {
            "round": int(row["communication_round"]),
            "class_id": int(row["class_id"]),
            "available": float(row["available_functional_coverage"]),
            "realized": float(row["realized_functional_coverage"]),
        }
        for row in _read_csv(
            run_dir / "functional_coverage" / "coverage_per_class_round.csv"
        )
    ]


def _safe_spearman(left, right) -> dict:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    if len(left) < 3 or np.allclose(left, left[0]) or np.allclose(right, right[0]):
        return {"rho": 0.0, "pvalue": 1.0, "defined": False}
    value = spearmanr(left, right)
    rho = float(value.statistic)
    pvalue = float(value.pvalue)
    if not math.isfinite(rho) or not math.isfinite(pvalue):
        return {"rho": 0.0, "pvalue": 1.0, "defined": False}
    return {"rho": rho, "pvalue": pvalue, "defined": True}


def analyze(output_root: Path) -> dict:
    output_root = Path(output_root)
    runs = {
        "clientlt": output_root / "clientlt",
        "matched_dirichlet": output_root / "matched_dirichlet",
    }
    protocols = {name: _protocol(path) for name, path in runs.items()}
    common_keys = (
        "schema_version",
        "selected_rounds",
        "tail_classes",
        "samples_per_tail_class",
        "probe_manifest_hash",
        "common_lora_anchor_sha256",
        "gain_epsilon",
        "lora_keys",
    )
    mismatches = {
        key: {name: protocol.get(key) for name, protocol in protocols.items()}
        for key in common_keys
        if protocols["clientlt"].get(key) != protocols["matched_dirichlet"].get(key)
    }
    if mismatches:
        raise RuntimeError(f"Dual-topology protocol mismatch: {mismatches}")
    boundary_rows = {
        name: _read_csv(path / "functional_coverage" / "frozen_boundary_weights.csv")
        for name, path in runs.items()
    }
    boundary_index = {
        name: {
            (int(row["class_id"]), int(row["competitor_class"])): float(
                row["confusion_weight"]
            )
            for row in rows
        }
        for name, rows in boundary_rows.items()
    }
    if set(boundary_index["clientlt"]) != set(boundary_index["matched_dirichlet"]):
        raise RuntimeError("The two topology runs use different functional boundary units")
    max_boundary_weight_diff = max(
        abs(
            boundary_index["clientlt"][unit]
            - boundary_index["matched_dirichlet"][unit]
        )
        for unit in boundary_index["clientlt"]
    )
    if max_boundary_weight_diff > 1e-6:
        raise RuntimeError(
            "Frozen confusion weights differ across topology runs: "
            f"max_abs_diff={max_boundary_weight_diff}"
        )

    matrices = {name: _count_matrix(path) for name, path in runs.items()}
    class_margins_equal = bool(
        np.array_equal(matrices["clientlt"].sum(axis=0), matrices["matched_dirichlet"].sum(axis=0))
    )
    client_margins_equal = bool(
        np.array_equal(matrices["clientlt"].sum(axis=1), matrices["matched_dirichlet"].sum(axis=1))
    )
    schedules_equal = _schedule(runs["clientlt"]) == _schedule(runs["matched_dirichlet"])
    if not class_margins_equal or not client_margins_equal or not schedules_equal:
        raise RuntimeError(
            "Causal protocol audit failed: "
            f"class_margins={class_margins_equal}, client_margins={client_margins_equal}, "
            f"schedule={schedules_equal}"
        )

    metrics = {name: _round_metrics(path) for name, path in runs.items()}
    metric_rounds = {name: [row["communication_round"] for row in rows] for name, rows in metrics.items()}
    if metric_rounds["clientlt"] != metric_rounds["matched_dirichlet"]:
        raise RuntimeError("The two topologies have different evaluation rounds")
    final_round = metric_rounds["clientlt"][-1]
    final_epoch = final_round - 1
    final = {name: rows[-1] for name, rows in metrics.items()}
    best_tail = {name: max(row["tail"] for row in rows) for name, rows in metrics.items()}
    tail_drop = {name: best_tail[name] - final[name]["tail"] for name in runs}

    coverage_rows = {name: _coverage(path) for name, path in runs.items()}
    coverage_index = {
        name: {(row["round"], row["class_id"]): row for row in rows}
        for name, rows in coverage_rows.items()
    }
    if set(coverage_index["clientlt"]) != set(coverage_index["matched_dirichlet"]):
        raise RuntimeError("The two topologies have different class-by-round coverage units")
    units = sorted(coverage_index["clientlt"])
    available_gap = float(
        np.mean(
            [
                coverage_index["matched_dirichlet"][unit]["available"]
                - coverage_index["clientlt"][unit]["available"]
                for unit in units
            ]
        )
    )
    realized_gap = float(
        np.mean(
            [
                coverage_index["matched_dirichlet"][unit]["realized"]
                - coverage_index["clientlt"][unit]["realized"]
                for unit in units
            ]
        )
    )

    tail_classes = [int(value) for value in protocols["clientlt"]["tail_classes"]]
    final_acc = {name: _per_class_accuracy(path, final_epoch) for name, path in runs.items()}
    all_per_class = {
        name: {
            epoch: _per_class_accuracy(path, epoch)
            for epoch in [row["epoch"] for row in metrics[name]]
        }
        for name, path in runs.items()
    }
    per_class_rows = []
    for class_id in tail_classes:
        class_units = [unit for unit in units if unit[1] == class_id]
        matched_available = float(
            np.mean([coverage_index["matched_dirichlet"][unit]["available"] for unit in class_units])
        )
        clientlt_available = float(
            np.mean([coverage_index["clientlt"][unit]["available"] for unit in class_units])
        )
        matched_realized = float(
            np.mean([coverage_index["matched_dirichlet"][unit]["realized"] for unit in class_units])
        )
        clientlt_realized = float(
            np.mean([coverage_index["clientlt"][unit]["realized"] for unit in class_units])
        )
        class_drop = {}
        for name in runs:
            trajectory = [all_per_class[name][epoch].get(class_id, 0.0) for epoch in all_per_class[name]]
            class_drop[name] = max(trajectory) - trajectory[-1]
        per_class_rows.append(
            {
                "class_id": class_id,
                "matched_minus_clientlt_available_coverage": matched_available - clientlt_available,
                "matched_minus_clientlt_realized_coverage": matched_realized - clientlt_realized,
                "matched_minus_clientlt_final_accuracy_pp": (
                    final_acc["matched_dirichlet"].get(class_id, 0.0)
                    - final_acc["clientlt"].get(class_id, 0.0)
                ),
                "clientlt_minus_matched_best_to_final_drop_pp": (
                    class_drop["clientlt"] - class_drop["matched_dirichlet"]
                ),
            }
        )
    _write_csv(output_root / "analysis" / "per_class_contrasts.csv", per_class_rows)

    accuracy_relation = _safe_spearman(
        [row["matched_minus_clientlt_realized_coverage"] for row in per_class_rows],
        [row["matched_minus_clientlt_final_accuracy_pp"] for row in per_class_rows],
    )
    retention_relation = _safe_spearman(
        [row["matched_minus_clientlt_realized_coverage"] for row in per_class_rows],
        [row["clientlt_minus_matched_best_to_final_drop_pp"] for row in per_class_rows],
    )

    final_tail_gap = final["matched_dirichlet"]["tail"] - final["clientlt"]["tail"]
    final_hmean_gap = final["matched_dirichlet"]["hmean"] - final["clientlt"]["hmean"]
    extra_clientlt_drop = tail_drop["clientlt"] - tail_drop["matched_dirichlet"]
    coverage_supported = available_gap > 0.0 and realized_gap > 0.0
    performance_supported = final_tail_gap > 0.0 and final_hmean_gap > 0.0
    retention_supported = extra_clientlt_drop > 0.0
    if coverage_supported and performance_supported and retention_supported:
        verdict = "FULL_CHAIN_SUPPORTED"
    elif coverage_supported and performance_supported:
        verdict = "ADAPTATION_CHAIN_ONLY"
    elif coverage_supported:
        verdict = "COVERAGE_WITHOUT_TASK_EFFECT"
    elif performance_supported:
        verdict = "TASK_GAP_WITHOUT_COVERAGE_MEDIATOR"
    else:
        verdict = "FUNCTIONAL_COVERAGE_STORY_NOT_SUPPORTED"

    scorecard = [
        {
            "claim": "available_coverage",
            "metric": "matched_dirichlet_minus_clientlt",
            "value": available_gap,
            "expected": ">0",
            "passed": available_gap > 0.0,
        },
        {
            "claim": "fedavg_realized_coverage",
            "metric": "matched_dirichlet_minus_clientlt",
            "value": realized_gap,
            "expected": ">0",
            "passed": realized_gap > 0.0,
        },
        {
            "claim": "tail_performance",
            "metric": "matched_dirichlet_minus_clientlt_final_tail_accuracy_pp",
            "value": final_tail_gap,
            "expected": ">0",
            "passed": final_tail_gap > 0.0,
        },
        {
            "claim": "late_retention",
            "metric": "clientlt_minus_matched_best_to_final_tail_drop_pp",
            "value": extra_clientlt_drop,
            "expected": ">0",
            "passed": extra_clientlt_drop > 0.0,
        },
    ]
    _write_csv(output_root / "analysis" / "main_scorecard.csv", scorecard)

    summary = {
        "schema_version": "functional_coverage_validation_analysis_v1",
        "verdict": verdict,
        "single_seed_descriptive": True,
        "primary_results": {
            "matched_minus_clientlt_available_coverage": available_gap,
            "matched_minus_clientlt_realized_coverage": realized_gap,
            "matched_minus_clientlt_final_tail_accuracy_pp": final_tail_gap,
            "matched_minus_clientlt_final_hmean_pp": final_hmean_gap,
            "clientlt_minus_matched_best_to_final_tail_drop_pp": extra_clientlt_drop,
        },
        "component_support": {
            "topology_to_coverage": coverage_supported,
            "topology_to_task_performance": performance_supported,
            "stronger_clientlt_late_forgetting": retention_supported,
        },
        "secondary_per_class_relations": {
            "realized_coverage_gap_vs_final_accuracy_gap": accuracy_relation,
            "realized_coverage_gap_vs_extra_clientlt_drop": retention_relation,
            "expected_direction_classes": {
                "available_coverage": sum(
                    row["matched_minus_clientlt_available_coverage"] > 0 for row in per_class_rows
                ),
                "realized_coverage": sum(
                    row["matched_minus_clientlt_realized_coverage"] > 0 for row in per_class_rows
                ),
                "final_accuracy": sum(
                    row["matched_minus_clientlt_final_accuracy_pp"] > 0 for row in per_class_rows
                ),
                "tail_class_count": len(per_class_rows),
            },
        },
        "raw_outcomes": {
            name: {
                "final": final[name],
                "best_tail_accuracy": best_tail[name],
                "best_to_final_tail_drop": tail_drop[name],
            }
            for name in runs
        },
        "protocol_audit": {
            "passed": True,
            "fixed_class_margins": class_margins_equal,
            "fixed_client_margins": client_margins_equal,
            "same_client_schedule": schedules_equal,
            "same_lora_anchor": True,
            "same_train_only_probe_bank": True,
            "max_frozen_boundary_weight_abs_diff": max_boundary_weight_diff,
            "test_controls_coverage_or_training": False,
            "coverage_rounds": protocols["clientlt"]["selected_rounds"],
            "final_round": final_round,
        },
        "interpretation": (
            "This is the preregistered direct seed-42 validation. Per-class correlations are "
            "secondary diagnostics and do not override the four primary directional results."
        ),
    }
    _write_json(output_root / "analysis" / "validation_summary.json", summary)
    primary = summary["primary_results"]
    report = "\n".join(
        [
            "# Client-LT functional-coverage validation",
            "",
            f"- Verdict: **{verdict}**",
            f"- Available coverage (Dir - Client-LT): **{primary['matched_minus_clientlt_available_coverage']:+.6f}**",
            f"- Realized coverage (Dir - Client-LT): **{primary['matched_minus_clientlt_realized_coverage']:+.6f}**",
            f"- Final tail accuracy (Dir - Client-LT): **{primary['matched_minus_clientlt_final_tail_accuracy_pp']:+.3f} pp**",
            f"- Final H-mean (Dir - Client-LT): **{primary['matched_minus_clientlt_final_hmean_pp']:+.3f} pp**",
            f"- Extra Client-LT best-to-final tail drop: **{primary['clientlt_minus_matched_best_to_final_tail_drop_pp']:+.3f} pp**",
            "",
            "The coverage diagnostic used only held-out training probes and all selected clients,",
            "including class-absent donors. Test metrics were never used to control training.",
            "This seed-42 result is descriptive; repeat seeds only after the directional gate passes.",
            "",
        ]
    )
    report_path = output_root / "analysis" / "validation_report.md"
    report_path.write_text(report, encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("output/functional_coverage_validation_seed42"),
    )
    args = parser.parse_args()
    result = analyze(args.output_root)
    print(json.dumps({"verdict": result["verdict"], **result["primary_results"]}, indent=2))


if __name__ == "__main__":
    main()
