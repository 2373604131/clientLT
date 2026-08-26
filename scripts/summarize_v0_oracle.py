#!/usr/bin/env python
"""Aggregate V0 oracle units across seeds, rounds, and topologies."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.cusp_minimal import write_csv, write_json


def read_units(directories: list[Path]) -> list[dict]:
    units = []
    for directory in directories:
        manifest = json.loads((directory / "v0_manifest.json").read_text(encoding="utf-8"))
        rows = list(csv.DictReader((directory / "test_metrics.csv").open(encoding="utf-8")))
        for row in rows:
            for key in (
                "gamma", "overall_acc", "balanced_acc", "head_acc", "mid_acc", "tail_acc",
                "non_tail_acc", "h3", "tail_gain", "head_damage", "mid_damage", "gap_closure",
                "support_only_tail_ceiling",
            ):
                row[key] = float(row[key])
        units.append({"directory": str(directory), "manifest": manifest, "rows": rows})
    return units


def bootstrap_ci(values: list[float], seed: int = 2026, draws: int = 4000) -> tuple[float, float]:
    values = np.asarray([value for value in values if math.isfinite(value)], dtype=float)
    if values.size == 0:
        return math.nan, math.nan
    if values.size == 1:
        return float(values[0]), float(values[0])
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, values.size, size=(int(draws), values.size))
    means = values[indices].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def summarize(units: list[dict]) -> tuple[list[dict], dict]:
    grouped: dict[tuple[str, str, float], list[dict]] = defaultdict(list)
    random_lookup: dict[tuple[str, str, str, float], list[float]] = defaultdict(list)
    for unit in units:
        manifest = unit["manifest"]
        partition = str(manifest.get("partition", ""))
        seed = str(manifest.get("seed", ""))
        round_id = str(manifest.get("round", ""))
        for row in unit["rows"]:
            grouped[(partition, row["method"], float(row["gamma"]))].append({
                **row, "seed": seed, "round": round_id, "directory": unit["directory"]
            })
            if row["method"] == "random_span":
                random_lookup[(partition, seed, round_id, float(row["gamma"]))].append(float(row["tail_gain"]))

    summary_rows = []
    formal_candidates = []
    for (partition, method, gamma), rows in sorted(grouped.items()):
        tail_gain = [float(row["tail_gain"]) for row in rows]
        head_damage = [float(row["head_damage"]) for row in rows]
        gap = [float(row["gap_closure"]) for row in rows if math.isfinite(float(row["gap_closure"]))]
        lower, upper = bootstrap_ci(tail_gain)
        random_superiority = []
        if method == "oracle_span":
            for row in rows:
                random_values = random_lookup.get((partition, row["seed"], row["round"], gamma), [])
                if random_values:
                    random_superiority.append(float(row["tail_gain"]) > float(np.percentile(random_values, 95)))
        record = {
            "partition": partition,
            "method": method,
            "gamma": gamma,
            "unit_count": len(rows),
            "seed_count": len({row["seed"] for row in rows}),
            "round_count": len({row["round"] for row in rows}),
            "tail_gain_mean": float(np.mean(tail_gain)),
            "tail_gain_std": float(np.std(tail_gain)),
            "tail_gain_ci95_low": lower,
            "tail_gain_ci95_high": upper,
            "head_damage_mean": float(np.mean(head_damage)),
            "gap_closure_mean": float(np.mean(gap)) if gap else math.nan,
            "positive_unit_rate": float(np.mean(np.asarray(tail_gain) > 0.0)),
            "random_p95_superiority_rate": float(np.mean(random_superiority)) if random_superiority else math.nan,
        }
        summary_rows.append(record)
        if method == "oracle_span":
            formal_candidates.append(record)

    for row in formal_candidates:
        row["strong_pass"] = (
            row["tail_gain_ci95_low"] > 0.0
            and row["head_damage_mean"] <= 0.5
            and row["positive_unit_rate"] >= 2.0 / 3.0
            and (math.isnan(row["gap_closure_mean"]) or row["gap_closure_mean"] >= 0.1)
            and (
                math.isnan(row["random_p95_superiority_rate"])
                or row["random_p95_superiority_rate"] >= 2.0 / 3.0
            )
            and row["seed_count"] >= 3
            and row["round_count"] >= 3
        )
    verdict = {
        "verdict": "PASS" if any(row.get("strong_pass", False) for row in formal_candidates) else "NOT_YET_PASS",
        "warning": (
            "Gamma-wise results are all reported. A paper method may not select gamma from official-test results; "
            "use the validation protocol frozen in each unit."
        ),
        "formal_oracle_span_candidates": formal_candidates,
    }
    return summary_rows, verdict


def plot_pareto(summary_rows: list[dict], output_dir: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    methods = (
        "oracle_span", "oracle_convex_search", "equal_client",
        "support_weighting", "random_span",
    )
    for partition in sorted({row["partition"] for row in summary_rows}):
        figure, axis = plt.subplots(figsize=(6.0, 4.5))
        for method in methods:
            rows = [row for row in summary_rows if row["partition"] == partition and row["method"] == method]
            if not rows:
                continue
            rows.sort(key=lambda row: row["gamma"])
            axis.plot(
                [row["head_damage_mean"] for row in rows],
                [row["tail_gain_mean"] for row in rows],
                marker="o",
                label=method,
            )
            for row in rows:
                axis.annotate(f"γ={row['gamma']:g}", (row["head_damage_mean"], row["tail_gain_mean"]), fontsize=7)
        axis.axhline(0.0, color="black", linewidth=0.8)
        axis.axvline(0.5, color="gray", linewidth=0.8, linestyle="--")
        axis.set_xlabel("Head damage (pp; lower is better)")
        axis.set_ylabel("Tail gain over FedAvg (pp)")
        axis.set_title(f"V0 Pareto — {partition}")
        axis.legend()
        figure.tight_layout()
        safe_name = partition.replace("/", "_").replace("\\", "_") or "unknown"
        figure.savefig(output_dir / f"v0_pareto_{safe_name}.pdf")
        figure.savefig(output_dir / f"v0_pareto_{safe_name}.png", dpi=180)
        plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dirs", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    units = read_units(args.input_dirs)
    summary_rows, verdict = summarize(units)
    write_csv(args.output_dir / "v0_aggregate.csv", summary_rows)
    write_json(args.output_dir / "v0_verdict.json", verdict)
    plot_pareto(summary_rows, args.output_dir)
    print(f"V0 summary finished: {args.output_dir}")


if __name__ == "__main__":
    main()
