#!/usr/bin/env python
import argparse
import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path


DEFAULT_ROOT = Path("output/cifar100_LT/PromptFL_fedavg_vit_b16_batchSize32/ExperimentD_LocalEpochPilot")
RUN_RE = re.compile(
    r"client-longtail_lambda=(?P<lambda>[0-9.]+)_alpha=(?P<alpha>[0-9.]+)_rho=(?P<rho>[0-9.]+)_"
    r"localE=(?P<local_epochs>\d+)_seed=(?P<seed>\d+)"
)

ALL_FIELDS = [
    "partition",
    "specialization_lambda",
    "intra_group_alpha",
    "head_leakage_scale",
    "local_epochs",
    "seed",
    "run_dir",
    "final_epoch",
    "final_overall_acc",
    "final_non_tail_acc",
    "final_bottom20_tail_acc",
    "final_macro_per_class_acc",
    "best_overall_acc",
    "best_bottom20_tail_acc",
    "last5_overall_acc_mean",
    "last5_overall_acc_std",
    "last5_bottom20_tail_acc_mean",
    "last5_bottom20_tail_acc_std",
    "num_diagnostic_rounds",
    "mean_gain_support_actual",
    "mean_gain_support_normalized",
    "mean_gain_all",
    "mean_offset_gap",
    "support_actual_positive_rate",
    "support_normalized_positive_rate",
    "offset_observed_rate",
    "full_reversal_rate",
    "mean_support_fedavg_weight",
    "mean_num_support_clients",
    "mean_positive_sample_specialist_ratio",
    "mean_update_norm_all",
    "mean_update_norm_head_clients",
    "mean_update_norm_tail_specialists",
    "mean_local_training_seconds",
    "mean_experimentD_diagnostic_seconds",
    "mean_normal_global_eval_seconds",
    "mean_round_total_seconds",
    "cumulative_seconds",
]

SUMMARY_METRICS = [
    "final_overall_acc",
    "final_bottom20_tail_acc",
    "final_macro_per_class_acc",
    "best_overall_acc",
    "last5_overall_acc_mean",
    "last5_bottom20_tail_acc_mean",
    "mean_gain_support_actual",
    "mean_gain_support_normalized",
    "mean_gain_all",
    "mean_offset_gap",
    "support_actual_positive_rate",
    "offset_observed_rate",
    "full_reversal_rate",
    "mean_update_norm_all",
    "mean_update_norm_head_clients",
    "mean_update_norm_tail_specialists",
    "mean_local_training_seconds",
    "mean_experimentD_diagnostic_seconds",
    "mean_round_total_seconds",
    "cumulative_seconds",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Summarize Experiment D local-epochs pilot results.")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--strict", action="store_true")
    return parser.parse_args()


def as_float(value, default=math.nan):
    if value in (None, ""):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def parse_run_dir(run_dir):
    match = RUN_RE.search(run_dir.name)
    if not match:
        return None
    return {
        "partition": "client-longtail",
        "specialization_lambda": float(match.group("lambda")),
        "intra_group_alpha": float(match.group("alpha")),
        "head_leakage_scale": float(match.group("rho")),
        "local_epochs": int(match.group("local_epochs")),
        "seed": int(match.group("seed")),
    }


def read_csv(path):
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def mean(values):
    valid = [float(x) for x in values if not math.isnan(float(x))]
    return sum(valid) / len(valid) if valid else math.nan


def sample_std(values):
    valid = [float(x) for x in values if not math.isnan(float(x))]
    if not valid:
        return math.nan
    if len(valid) == 1:
        return 0.0
    avg = mean(valid)
    return math.sqrt(sum((x - avg) ** 2 for x in valid) / (len(valid) - 1))


def write_csv(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                key: "" if isinstance(row.get(key), float) and math.isnan(row.get(key)) else row.get(key, "")
                for key in fields
            })


def pick_last_round(metrics_rows):
    sorted_rows = sorted(metrics_rows, key=lambda row: int(float(row.get("epoch", -1))))
    return sorted_rows[-1]


def collect_round_metric_summary(metrics_rows):
    last = pick_last_round(metrics_rows)
    tail_values = [as_float(row.get("bottom20_tail_acc")) for row in metrics_rows]
    overall_values = [as_float(row.get("overall_acc")) for row in metrics_rows]
    last5 = sorted(metrics_rows, key=lambda row: int(float(row.get("epoch", -1))))[-5:]
    return {
        "final_epoch": int(float(last.get("epoch", -1))),
        "final_overall_acc": as_float(last.get("overall_acc")),
        "final_non_tail_acc": as_float(last.get("non_tail_acc")),
        "final_bottom20_tail_acc": as_float(last.get("bottom20_tail_acc")),
        "final_macro_per_class_acc": as_float(last.get("macro_per_class_acc")),
        "best_overall_acc": max(overall_values) if overall_values else math.nan,
        "best_bottom20_tail_acc": max(tail_values) if tail_values else math.nan,
        "last5_overall_acc_mean": mean([as_float(row.get("overall_acc")) for row in last5]),
        "last5_overall_acc_std": sample_std([as_float(row.get("overall_acc")) for row in last5]),
        "last5_bottom20_tail_acc_mean": mean([as_float(row.get("bottom20_tail_acc")) for row in last5]),
        "last5_bottom20_tail_acc_std": sample_std([as_float(row.get("bottom20_tail_acc")) for row in last5]),
    }


def collect_diagnostic_summary(run_dir):
    path = run_dir / "experiment_d" / "experiment_d_round_summary.csv"
    if not path.exists():
        return {
            "num_diagnostic_rounds": 0,
            "mean_gain_support_actual": math.nan,
            "mean_gain_support_normalized": math.nan,
            "mean_gain_all": math.nan,
            "mean_offset_gap": math.nan,
            "support_actual_positive_rate": math.nan,
            "support_normalized_positive_rate": math.nan,
            "offset_observed_rate": math.nan,
            "full_reversal_rate": math.nan,
            "mean_support_fedavg_weight": math.nan,
            "mean_num_support_clients": math.nan,
            "mean_positive_sample_specialist_ratio": math.nan,
        }
    rows = read_csv(path)
    fields = [
        "mean_gain_support_actual",
        "mean_gain_support_normalized",
        "mean_gain_all",
        "mean_offset_gap",
        "support_actual_positive_rate",
        "support_normalized_positive_rate",
        "offset_observed_rate",
        "full_reversal_rate",
        "mean_support_fedavg_weight",
        "mean_num_support_clients",
        "mean_positive_sample_specialist_ratio",
    ]
    out = {"num_diagnostic_rounds": len(rows)}
    for field in fields:
        out[field] = mean([as_float(row.get(field)) for row in rows])
    return out


def collect_update_norm_summary(run_dir):
    path = run_dir / "experiment_d" / "client_update_norm_summary.csv"
    if not path.exists():
        return {
            "mean_update_norm_all": math.nan,
            "mean_update_norm_head_clients": math.nan,
            "mean_update_norm_tail_specialists": math.nan,
        }
    rows = read_csv(path)
    return {
        "mean_update_norm_all": mean([as_float(row.get("mean_update_norm_all")) for row in rows]),
        "mean_update_norm_head_clients": mean([as_float(row.get("mean_update_norm_head_clients")) for row in rows]),
        "mean_update_norm_tail_specialists": mean([as_float(row.get("mean_update_norm_tail_specialists")) for row in rows]),
    }


def collect_runtime_summary(run_dir):
    path = run_dir / "experiment_d" / "runtime_metrics.csv"
    if not path.exists():
        return {
            "mean_local_training_seconds": math.nan,
            "mean_experimentD_diagnostic_seconds": math.nan,
            "mean_normal_global_eval_seconds": math.nan,
            "mean_round_total_seconds": math.nan,
            "cumulative_seconds": math.nan,
        }
    rows = read_csv(path)
    return {
        "mean_local_training_seconds": mean([as_float(row.get("local_training_seconds")) for row in rows]),
        "mean_experimentD_diagnostic_seconds": mean([as_float(row.get("experimentD_diagnostic_seconds")) for row in rows]),
        "mean_normal_global_eval_seconds": mean([as_float(row.get("normal_global_eval_seconds")) for row in rows]),
        "mean_round_total_seconds": mean([as_float(row.get("round_total_seconds")) for row in rows]),
        "cumulative_seconds": as_float(rows[-1].get("cumulative_seconds")) if rows else math.nan,
    }


def collect(root, strict):
    if not root.exists():
        raise FileNotFoundError(
            f"Experiment D root does not exist: {root}. Run experiments before summarizing."
        )

    rows = []
    warnings = []
    for metrics_path in sorted(root.rglob("round_metrics.csv")):
        run_dir = metrics_path.parent
        info = parse_run_dir(run_dir)
        if info is None:
            warnings.append(f"skip unrecognized run directory: {run_dir}")
            continue

        metrics_rows = read_csv(metrics_path)
        if not metrics_rows:
            message = f"empty round_metrics.csv: {metrics_path}"
            if strict:
                raise RuntimeError(message)
            warnings.append(message)
            continue

        rows.append(
            {
                **info,
                "run_dir": str(run_dir),
                **collect_round_metric_summary(metrics_rows),
                **collect_diagnostic_summary(run_dir),
                **collect_update_norm_summary(run_dir),
                **collect_runtime_summary(run_dir),
            }
        )
    return rows, warnings


def build_summary(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[(row["partition"], row["intra_group_alpha"], row["local_epochs"])].append(row)

    summary = []
    for (partition, alpha, local_epochs), group in sorted(groups.items(), key=lambda item: item[0]):
        out = {
            "partition": partition,
            "intra_group_alpha": alpha,
            "local_epochs": local_epochs,
            "num_runs": len(group),
        }
        for metric in SUMMARY_METRICS:
            values = [as_float(row.get(metric)) for row in group]
            out[f"{metric}_mean"] = mean(values)
            out[f"{metric}_std"] = sample_std(values)
        summary.append(out)
    return summary


def build_comparison(summary_rows, rows):
    by_local_e = {int(row["local_epochs"]): row for row in summary_rows}
    comparison = {
        "description": (
            "Experiment D local_epochs pilot. This is a single-seed pilot by default; "
            "standard deviations in summary CSV describe available repeated runs, not cross-seed significance "
            "unless SEEDS was explicitly expanded."
        ),
        "single_seed_default": True,
        "groups": summary_rows,
    }
    if 1 in by_local_e and 3 in by_local_e:
        comparison["localE3_minus_localE1"] = {
            metric: (
                as_float(by_local_e[3].get(f"{metric}_mean"))
                - as_float(by_local_e[1].get(f"{metric}_mean"))
            )
            for metric in [
                "final_overall_acc",
                "final_bottom20_tail_acc",
                "mean_gain_support_actual",
                "mean_gain_all",
                "mean_round_total_seconds",
            ]
        }
    comparison["num_runs"] = len(rows)
    return comparison


def main():
    args = parse_args()
    output_dir = args.output_dir or (args.root / "summary")
    try:
        rows, warnings = collect(args.root, args.strict)
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}")
        return 1

    if not rows:
        print(f"ERROR: no completed Experiment D local-epochs pilot runs found under {args.root}")
        return 1

    all_path = output_dir / "experimentD_local_epochs_all_runs.csv"
    summary_path = output_dir / "experimentD_local_epochs_summary.csv"
    json_path = output_dir / "experimentD_local_epochs_comparison.json"

    write_csv(all_path, rows, ALL_FIELDS)
    summary_rows = build_summary(rows)
    summary_fields = ["partition", "intra_group_alpha", "local_epochs", "num_runs"]
    for metric in SUMMARY_METRICS:
        summary_fields.extend([f"{metric}_mean", f"{metric}_std"])
    write_csv(summary_path, summary_rows, summary_fields)
    output_dir.mkdir(parents=True, exist_ok=True)
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(build_comparison(summary_rows, rows), f, indent=2)

    for warning in warnings:
        print(f"WARNING: {warning}")
    print(f"Wrote {len(rows)} rows to {all_path}")
    print(f"Wrote {len(summary_rows)} rows to {summary_path}")
    print(f"Wrote {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
