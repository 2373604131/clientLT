#!/usr/bin/env python
import argparse
import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path


DEFAULT_ROOT = Path("output/cifar100_LT/PromptFL_fedavg_vit_b16_batchSize32/PanelC")
RUN_RE = re.compile(
    r"partition=(?P<partition>noniid-labeldir-fine|client-longtail).*?"
    r"alpha=(?P<alpha>[0-9.]+).*?"
    r"seed=(?P<seed>\d+)"
)

ALL_RUN_FIELDS = [
    "partition",
    "alpha",
    "seed",
    "run_dir",
    "overall_acc",
    "non_tail_acc",
    "bottom20_tail_acc",
    "macro_per_class_acc",
    "tail_effective_client_number",
    "tail_top1_client_mass",
    "tail_top2_client_mass",
    "client_sample_cv",
    "tail_to_tail_budget",
    "non_tail_to_tail_budget",
    "actual_tail_client_purity",
]

SUMMARY_METRICS = [
    "overall_acc",
    "non_tail_acc",
    "bottom20_tail_acc",
    "macro_per_class_acc",
    "tail_effective_client_number",
    "tail_top1_client_mass",
    "tail_top2_client_mass",
    "client_sample_cv",
    "tail_to_tail_budget",
    "non_tail_to_tail_budget",
    "actual_tail_client_purity",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Summarize Panel C runs at the fixed final epoch."
    )
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--epoch", type=int, default=99)
    parser.add_argument("--strict", action="store_true")
    return parser.parse_args()


def parse_run_name(run_dir):
    match = RUN_RE.search(run_dir.name)
    if not match:
        return None
    return {
        "partition": match.group("partition"),
        "alpha": float(match.group("alpha")),
        "seed": int(match.group("seed")),
    }


def as_float(value, default=math.nan):
    if value in (None, ""):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def read_final_metrics(path, epoch):
    with path.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        if int(float(row.get("epoch", -1))) == epoch:
            return row
    return None


def mean(values):
    valid = [x for x in values if not math.isnan(x)]
    if not valid:
        return math.nan
    return sum(valid) / len(valid)


def sample_std(values):
    valid = [x for x in values if not math.isnan(x)]
    if len(valid) <= 1:
        return 0.0 if valid else math.nan
    avg = mean(valid)
    return math.sqrt(sum((x - avg) ** 2 for x in valid) / (len(valid) - 1))


def load_tail_topology_from_csv(path):
    fallback = {}
    if not path.exists():
        return fallback
    with path.open("r", newline="", encoding="utf-8") as f:
        rows = [row for row in csv.DictReader(f) if row.get("class_group") == "tail"]
    if not rows:
        return fallback
    fallback["tail_effective_client_number"] = mean(
        [as_float(row.get("effective_client_number")) for row in rows]
    )
    fallback["tail_top1_client_mass"] = mean(
        [as_float(row.get("top1_client_mass", row.get("concentration"))) for row in rows]
    )
    fallback["tail_top2_client_mass"] = mean(
        [as_float(row.get("top2_client_mass")) for row in rows]
    )
    return fallback


def load_partition_summary(run_dir):
    path = run_dir / "partition_summary.json"
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def collect_runs(root, epoch, strict):
    rows = []
    warnings = []
    for metrics_path in sorted(root.rglob("round_metrics.csv")):
        run_dir = metrics_path.parent
        info = parse_run_name(run_dir)
        if info is None:
            warnings.append(f"skip unrecognized run directory: {run_dir}")
            continue
        metrics = read_final_metrics(metrics_path, epoch)
        if metrics is None:
            message = f"missing epoch={epoch} in {metrics_path}"
            if strict:
                raise RuntimeError(message)
            warnings.append(message)
            continue

        summary = load_partition_summary(run_dir)
        fallback = load_tail_topology_from_csv(run_dir / "class_topology.csv")
        is_clientlt = info["partition"] == "client-longtail"
        row = {
            "partition": info["partition"],
            "alpha": info["alpha"],
            "seed": info["seed"],
            "run_dir": str(run_dir),
            "overall_acc": as_float(metrics.get("overall_acc")),
            "non_tail_acc": as_float(metrics.get("non_tail_acc")),
            "bottom20_tail_acc": as_float(metrics.get("bottom20_tail_acc")),
            "macro_per_class_acc": as_float(metrics.get("macro_per_class_acc")),
            "tail_effective_client_number": as_float(
                summary.get("tail_effective_client_number_mean"),
                fallback.get("tail_effective_client_number", math.nan),
            ),
            "tail_top1_client_mass": as_float(
                summary.get("tail_top1_client_mass_mean"),
                fallback.get("tail_top1_client_mass", math.nan),
            ),
            "tail_top2_client_mass": as_float(
                summary.get("tail_top2_client_mass_mean"),
                fallback.get("tail_top2_client_mass", math.nan),
            ),
            "client_sample_cv": as_float(summary.get("client_sample_cv")),
            "tail_to_tail_budget": (
                as_float(summary.get("tail_to_tail_budget")) if is_clientlt else math.nan
            ),
            "non_tail_to_tail_budget": (
                as_float(summary.get("non_tail_to_tail_budget")) if is_clientlt else math.nan
            ),
            "actual_tail_client_purity": (
                as_float(summary.get("actual_tail_client_purity")) if is_clientlt else math.nan
            ),
        }
        rows.append(row)
    return rows, warnings


def write_csv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                key: "" if isinstance(row.get(key), float) and math.isnan(row.get(key)) else row.get(key)
                for key in fieldnames
            })


def build_summary(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[(row["partition"], row["alpha"])].append(row)

    summary_rows = []
    for (partition, alpha), group in sorted(groups.items(), key=lambda x: (x[0][0], x[0][1])):
        out = {
            "partition": partition,
            "alpha": alpha,
            "num_runs": len(group),
        }
        for metric in SUMMARY_METRICS:
            values = [as_float(row.get(metric)) for row in group]
            out[f"{metric}_mean"] = mean(values)
            out[f"{metric}_std"] = sample_std(values)
        summary_rows.append(out)
    return summary_rows


def build_paired_delta(rows):
    by_key = {(row["partition"], row["alpha"], row["seed"]): row for row in rows}
    pairs = []
    alphas = sorted({row["alpha"] for row in rows})
    seeds = sorted({row["seed"] for row in rows})
    for alpha in alphas:
        for seed in seeds:
            dir_row = by_key.get(("noniid-labeldir-fine", alpha, seed))
            clt_row = by_key.get(("client-longtail", alpha, seed))
            if dir_row is None or clt_row is None:
                continue
            dir_tail = as_float(dir_row.get("bottom20_tail_acc"))
            clt_tail = as_float(clt_row.get("bottom20_tail_acc"))
            pairs.append(
                {
                    "alpha": alpha,
                    "seed": seed,
                    "dirichlet_tail_acc": dir_tail,
                    "clientlt_tail_acc": clt_tail,
                    "clientlt_minus_dirichlet": clt_tail - dir_tail,
                }
            )
    return pairs


def main():
    args = parse_args()
    output_dir = args.output_dir or (args.root / "summary")
    rows, warnings = collect_runs(args.root, args.epoch, args.strict)
    if not rows:
        raise RuntimeError(f"No Panel C runs found under {args.root}")

    all_runs_path = output_dir / "panel_c_all_runs.csv"
    summary_path = output_dir / "panel_c_summary.csv"
    paired_path = output_dir / "panel_c_paired_delta.csv"

    write_csv(all_runs_path, rows, ALL_RUN_FIELDS)

    summary_rows = build_summary(rows)
    summary_fields = ["partition", "alpha", "num_runs"]
    for metric in SUMMARY_METRICS:
        summary_fields.extend([f"{metric}_mean", f"{metric}_std"])
    write_csv(summary_path, summary_rows, summary_fields)

    paired_rows = build_paired_delta(rows)
    write_csv(
        paired_path,
        paired_rows,
        [
            "alpha",
            "seed",
            "dirichlet_tail_acc",
            "clientlt_tail_acc",
            "clientlt_minus_dirichlet",
        ],
    )

    for warning in warnings:
        print(f"WARNING: {warning}")
    print(f"Wrote {len(rows)} rows to {all_runs_path}")
    print(f"Wrote {len(summary_rows)} rows to {summary_path}")
    print(f"Wrote {len(paired_rows)} rows to {paired_path}")


if __name__ == "__main__":
    main()
