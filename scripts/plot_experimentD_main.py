#!/usr/bin/env python
import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path


DEFAULT_SUMMARY_DIR = Path("output/cifar100_LT/PromptFL_fedavg_vit_b16_batchSize32/ExperimentD_Main/summary")


def parse_args():
    parser = argparse.ArgumentParser(description="Plot formal Experiment D results.")
    parser.add_argument("--summary-dir", type=Path, default=DEFAULT_SUMMARY_DIR)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def as_float(value):
    if value in (None, ""):
        return math.nan
    return float(value)


def read_csv(path):
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def require_csv(path):
    if not path.exists():
        raise FileNotFoundError(f"missing required CSV: {path}")
    rows = read_csv(path)
    if not rows:
        raise RuntimeError(f"empty required CSV: {path}")
    return rows


def plot_gain_by_round(round_summary, output_path):
    import matplotlib.pyplot as plt

    metrics = [
        ("mean_gain_support_actual_mean", "G_support_actual"),
        ("mean_gain_all_mean", "G_all"),
        ("mean_offset_gap_mean", "offset gap"),
    ]
    partitions = ["noniid-labeldir-fine", "client-longtail"]
    colors = {"noniid-labeldir-fine": "#3A6EA5", "client-longtail": "#C44E52"}

    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.6), sharex=True)
    for ax, (field, title) in zip(axes, metrics):
        for partition in partitions:
            rows = sorted(
                [row for row in round_summary if row["partition"] == partition],
                key=lambda row: int(float(row["communication_round"])),
            )
            if not rows:
                continue
            xs = [int(float(row["communication_round"])) for row in rows]
            ys = [as_float(row[field]) for row in rows]
            yerr_field = field.replace("_mean", "_std")
            yerr = [as_float(row.get(yerr_field)) for row in rows]
            ax.errorbar(
                xs,
                ys,
                yerr=yerr,
                marker="o",
                linewidth=2.0,
                capsize=4,
                color=colors.get(partition),
                label=partition,
            )
        ax.set_title(title)
        ax.set_xlabel("Communication round")
        ax.grid(True, axis="y", alpha=0.25)
    axes[0].set_ylabel("Accuracy gain (pp)")
    axes[-1].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_paired_delta(paired_summary, output_path):
    import matplotlib.pyplot as plt

    fields = [
        ("clientlt_minus_dirichlet_mean_gain_support_actual_mean", "Delta G_support_actual"),
        ("clientlt_minus_dirichlet_mean_gain_all_mean", "Delta G_all"),
        ("clientlt_minus_dirichlet_mean_offset_gap_mean", "Delta offset gap"),
    ]
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    for field, label in fields:
        rows = sorted(paired_summary, key=lambda row: int(float(row["communication_round"])))
        xs = [int(float(row["communication_round"])) for row in rows]
        ys = [as_float(row[field]) for row in rows]
        yerr = [as_float(row.get(field.replace("_mean", "_std"))) for row in rows]
        ax.errorbar(xs, ys, yerr=yerr, marker="o", linewidth=2.0, capsize=4, label=label)
    ax.axhline(0.0, color="#555555", linewidth=1.0)
    ax.set_xlabel("Communication round")
    ax.set_ylabel("Client-LT minus Dirichlet (pp)")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_rates(round_summary, output_path):
    import matplotlib.pyplot as plt

    fields = [
        ("support_actual_positive_rate_mean", "positive G_support"),
        ("offset_observed_rate_mean", "offset observed"),
        ("full_reversal_rate_mean", "full reversal"),
    ]
    partitions = ["noniid-labeldir-fine", "client-longtail"]
    linestyles = {"noniid-labeldir-fine": "-", "client-longtail": "--"}

    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    for partition in partitions:
        rows = sorted(
            [row for row in round_summary if row["partition"] == partition],
            key=lambda row: int(float(row["communication_round"])),
        )
        for field, label in fields:
            if not rows:
                continue
            xs = [int(float(row["communication_round"])) for row in rows]
            ys = [as_float(row[field]) for row in rows]
            ax.plot(xs, ys, marker="o", linewidth=1.8, linestyle=linestyles[partition], label=f"{partition}: {label}")
    ax.set_xlabel("Communication round")
    ax.set_ylabel("Rate")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_class_scatter(per_class_rows, output_path):
    import matplotlib.pyplot as plt

    colors = {"noniid-labeldir-fine": "#3A6EA5", "client-longtail": "#C44E52"}
    fig, ax = plt.subplots(figsize=(5.4, 4.6))
    for partition in ["noniid-labeldir-fine", "client-longtail"]:
        rows = [row for row in per_class_rows if row["partition"] == partition]
        if not rows:
            continue
        xs = [as_float(row["gain_support_actual"]) for row in rows]
        ys = [as_float(row["gain_all"]) for row in rows]
        ax.scatter(xs, ys, s=18, alpha=0.65, color=colors.get(partition), label=partition)
    ax.axhline(0.0, color="#555555", linewidth=1.0)
    ax.axvline(0.0, color="#555555", linewidth=1.0)
    ax.set_xlabel("G_support_actual (pp)")
    ax.set_ylabel("G_all (pp)")
    ax.grid(True, alpha=0.2)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def main():
    args = parse_args()
    output_dir = args.output_dir or (args.summary_dir / "figures")
    try:
        round_summary = require_csv(args.summary_dir / "experimentD_main_round_summary.csv")
        paired_summary = require_csv(args.summary_dir / "experimentD_main_clientlt_minus_dirichlet_summary.csv")
        per_class_rows = require_csv(args.summary_dir / "experimentD_main_per_class_all.csv")
    except (FileNotFoundError, RuntimeError) as exc:
        print(f"ERROR: {exc}")
        return 1

    output_dir.mkdir(parents=True, exist_ok=True)
    plot_gain_by_round(round_summary, output_dir / "experimentD_main_gain_by_round.pdf")
    plot_paired_delta(paired_summary, output_dir / "experimentD_main_clientlt_minus_dirichlet_by_round.pdf")
    plot_rates(round_summary, output_dir / "experimentD_main_rates_by_round.pdf")
    plot_class_scatter(per_class_rows, output_dir / "experimentD_main_class_scatter_support_vs_all.pdf")
    print(f"Wrote formal Experiment D figures to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
