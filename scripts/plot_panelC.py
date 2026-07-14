#!/usr/bin/env python
import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path


DEFAULT_SUMMARY = Path(
    "output/cifar100_LT/PromptFL_fedavg_vit_b16_batchSize32/PanelC/summary/panel_c_summary.csv"
)
DEFAULT_OUTPUT = Path(
    "output/cifar100_LT/PromptFL_fedavg_vit_b16_batchSize32/PanelC/summary/panel_c.pdf"
)


def parse_args():
    parser = argparse.ArgumentParser(description="Plot Panel C tail accuracy curves.")
    parser.add_argument("--summary-csv", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def as_float(value):
    if value in (None, ""):
        return math.nan
    return float(value)


def read_summary(path):
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main():
    args = parse_args()
    rows = read_summary(args.summary_csv)
    if not rows:
        raise RuntimeError(f"No rows found in {args.summary_csv}")

    by_partition = defaultdict(list)
    for row in rows:
        by_partition[row["partition"]].append(row)

    import matplotlib.pyplot as plt

    label_map = {
        "noniid-labeldir-fine": "Standard Dirichlet",
        "client-longtail": "Client-LT",
    }
    color_map = {
        "noniid-labeldir-fine": "#3A6EA5",
        "client-longtail": "#C44E52",
    }

    fig, ax = plt.subplots(figsize=(5.6, 3.8))
    for partition in ("noniid-labeldir-fine", "client-longtail"):
        group = sorted(by_partition.get(partition, []), key=lambda r: as_float(r["alpha"]))
        if not group:
            continue
        xs = [as_float(row["alpha"]) for row in group]
        ys = [as_float(row["bottom20_tail_acc_mean"]) for row in group]
        yerr = [as_float(row["bottom20_tail_acc_std"]) for row in group]
        ax.errorbar(
            xs,
            ys,
            yerr=yerr,
            marker="o",
            linewidth=2.0,
            capsize=4,
            label=label_map.get(partition, partition),
            color=color_map.get(partition),
        )

    ax.set_xlabel("Concentration parameter")
    ax.set_ylabel("Bottom-20 tail accuracy (%)")
    ax.set_xticks(sorted({as_float(row["alpha"]) for row in rows}))
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
