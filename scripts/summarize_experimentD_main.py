#!/usr/bin/env python
import argparse
import csv
import math
import re
from collections import defaultdict
from pathlib import Path


DEFAULT_ROOT = Path("output/cifar100_LT/PromptFL_fedavg_vit_b16_batchSize32/ExperimentD_Main")

DIRICHLET_RE = re.compile(
    r"partition=(?P<partition>noniid-labeldir-fine)_alpha=(?P<alpha>[0-9.]+)_"
    r"IF=(?P<imb_factor>[0-9.]+)_localE=(?P<local_epochs>\d+)_seed=(?P<seed>\d+)"
)
CLIENTLT_RE = re.compile(
    r"partition=(?P<partition>client-longtail)_lambda=(?P<lambda>[0-9.]+)_alpha=(?P<alpha>[0-9.]+)_"
    r"rho=(?P<rho>[0-9.]+)_IF=(?P<imb_factor>[0-9.]+)_localE=(?P<local_epochs>\d+)_seed=(?P<seed>\d+)"
)

ROUND_METRICS = [
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

PAIR_METRICS = [
    "mean_gain_support_actual",
    "mean_gain_support_normalized",
    "mean_gain_all",
    "mean_offset_gap",
    "support_actual_positive_rate",
    "offset_observed_rate",
    "full_reversal_rate",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Summarize formal Experiment D results.")
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


def read_csv(path):
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows, fieldnames=None):
    rows = list(rows)
    if fieldnames is None:
        fieldnames = []
        seen = set()
        for row in rows:
            for key in row:
                if key not in seen:
                    seen.add(key)
                    fieldnames.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({
                key: "" if isinstance(row.get(key), float) and math.isnan(row.get(key)) else row.get(key, "")
                for key in fieldnames
            })


def parse_run_dir(run_dir):
    match = CLIENTLT_RE.search(run_dir.name)
    if match:
        return {
            "partition": match.group("partition"),
            "alpha": float(match.group("alpha")),
            "specialization_lambda": float(match.group("lambda")),
            "head_leakage_scale": float(match.group("rho")),
            "imb_factor": float(match.group("imb_factor")),
            "local_epochs": int(match.group("local_epochs")),
            "seed": int(match.group("seed")),
        }
    match = DIRICHLET_RE.search(run_dir.name)
    if match:
        return {
            "partition": match.group("partition"),
            "alpha": float(match.group("alpha")),
            "specialization_lambda": "",
            "head_leakage_scale": "",
            "imb_factor": float(match.group("imb_factor")),
            "local_epochs": int(match.group("local_epochs")),
            "seed": int(match.group("seed")),
        }
    return None


def common_run_fields(info, run_dir):
    return {
        **info,
        "run_dir": str(run_dir),
    }


def collect(root, strict=False):
    if not root.exists():
        raise FileNotFoundError(f"Experiment D main root does not exist: {root}")

    per_class_rows = []
    round_rows = []
    warnings = []

    for run_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        info = parse_run_dir(run_dir)
        if info is None:
            warnings.append(f"skip unrecognized run directory: {run_dir}")
            continue
        common = common_run_fields(info, run_dir)
        per_class_path = run_dir / "experiment_d" / "experiment_d_per_class.csv"
        round_path = run_dir / "experiment_d" / "experiment_d_round_summary.csv"
        missing = [str(path) for path in (per_class_path, round_path) if not path.exists()]
        if missing:
            message = f"missing Experiment D files for {run_dir}: {missing}"
            if strict:
                raise RuntimeError(message)
            warnings.append(message)
            continue

        for row in read_csv(per_class_path):
            per_class_rows.append({**common, **row})
        for row in read_csv(round_path):
            round_rows.append({**common, **row})

    return per_class_rows, round_rows, warnings


def aggregate_rounds(round_rows):
    groups = defaultdict(list)
    for row in round_rows:
        key = (
            row["partition"],
            as_float(row["alpha"]),
            int(float(row["communication_round"])),
        )
        groups[key].append(row)

    out = []
    for (partition, alpha, communication_round), rows in sorted(groups.items(), key=lambda item: item[0]):
        summary = {
            "partition": partition,
            "alpha": alpha,
            "communication_round": communication_round,
            "num_seeds": len({int(float(row["seed"])) for row in rows}),
            "num_runs": len(rows),
        }
        for metric in ROUND_METRICS:
            values = [as_float(row.get(metric)) for row in rows]
            summary[f"{metric}_mean"] = mean(values)
            summary[f"{metric}_std"] = sample_std(values)
        out.append(summary)
    return out


def paired_by_round(round_rows):
    by_key = defaultdict(dict)
    for row in round_rows:
        key = (
            int(float(row["seed"])),
            as_float(row["alpha"]),
            int(float(row["communication_round"])),
        )
        by_key[key][row["partition"]] = row

    paired = []
    for (seed, alpha, communication_round), rows in sorted(by_key.items()):
        clientlt = rows.get("client-longtail")
        dirichlet = rows.get("noniid-labeldir-fine")
        if clientlt is None or dirichlet is None:
            continue
        out = {
            "seed": seed,
            "alpha": alpha,
            "communication_round": communication_round,
        }
        for metric in PAIR_METRICS:
            out[f"clientlt_{metric}"] = as_float(clientlt.get(metric))
            out[f"dirichlet_{metric}"] = as_float(dirichlet.get(metric))
            out[f"clientlt_minus_dirichlet_{metric}"] = (
                as_float(clientlt.get(metric)) - as_float(dirichlet.get(metric))
            )
        paired.append(out)
    return paired


def aggregate_paired(paired_rows):
    groups = defaultdict(list)
    for row in paired_rows:
        groups[(as_float(row["alpha"]), int(float(row["communication_round"])))].append(row)

    out = []
    delta_fields = [f"clientlt_minus_dirichlet_{metric}" for metric in PAIR_METRICS]
    for (alpha, communication_round), rows in sorted(groups.items()):
        summary = {
            "alpha": alpha,
            "communication_round": communication_round,
            "num_pairs": len(rows),
        }
        for field in delta_fields:
            values = [as_float(row.get(field)) for row in rows]
            summary[f"{field}_mean"] = mean(values)
            summary[f"{field}_std"] = sample_std(values)
        out.append(summary)
    return out


def main():
    args = parse_args()
    output_dir = args.output_dir or (args.root / "summary")
    try:
        per_class_rows, round_rows, warnings = collect(args.root, args.strict)
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}")
        return 1

    if not per_class_rows or not round_rows:
        print(f"ERROR: no complete formal Experiment D runs found under {args.root}")
        return 1

    round_summary = aggregate_rounds(round_rows)
    paired_rows = paired_by_round(round_rows)
    paired_summary = aggregate_paired(paired_rows)

    write_csv(output_dir / "experimentD_main_per_class_all.csv", per_class_rows)
    write_csv(output_dir / "experimentD_main_round_all.csv", round_rows)
    write_csv(output_dir / "experimentD_main_round_summary.csv", round_summary)
    write_csv(output_dir / "experimentD_main_paired_by_round.csv", paired_rows)
    write_csv(output_dir / "experimentD_main_clientlt_minus_dirichlet_summary.csv", paired_summary)

    for warning in warnings:
        print(f"WARNING: {warning}")
    print(f"Wrote {len(per_class_rows)} per-class rows")
    print(f"Wrote {len(round_rows)} run-round rows")
    print(f"Wrote {len(round_summary)} partition-round summary rows")
    print(f"Wrote {len(paired_rows)} paired seed-round rows")
    print(f"Wrote outputs to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
