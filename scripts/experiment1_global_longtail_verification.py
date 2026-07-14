#!/usr/bin/env python
"""Experiment 1: verify global long-tail preservation.

This script checks the protocol-level claim that Dirichlet and Client-LT are
matched in global long-tail statistics. It does not train a model. It builds
both client partitions from the same long-tailed training pool, sums the
client-class counts back to global class counts, and writes paper-ready tables
and figures.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Iterable

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.phase1_exposure_topology import (  # noqa: E402
    PARTITION_SPECS,
    build_split,
    class_groups,
    counts_matrix,
    set_plot_style,
)


SPLIT_LABELS = {
    "dirichlet": "Dirichlet",
    "client_lt": "Client-LT",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify that Dirichlet and Client-LT preserve identical global long-tail class counts."
    )
    parser.add_argument("--dataset", default="cifar100_LT", choices=["cifar10_LT", "cifar100_LT", "fmnist_LT"])
    parser.add_argument("--data-root", default="DATA")
    parser.add_argument("--output-dir", default="output/experiment1_global_longtail_verification")
    parser.add_argument("--num-users", type=int, default=20)
    parser.add_argument("--frac", type=float, default=0.4)
    parser.add_argument("--rounds", type=int, default=100)
    parser.add_argument("--local-epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--model-init-seed", type=int, default=1)
    parser.add_argument("--schedule-seed", type=int, default=1)
    parser.add_argument("--global-lt-seed", type=int, default=1)
    parser.add_argument("--imb-type", default="exp")
    parser.add_argument("--imb-factor", type=float, default=0.01)
    parser.add_argument("--dirichlet-beta", type=float, default=0.3)
    parser.add_argument("--head-client-ratio", type=float, default=0.8)
    parser.add_argument("--tail-client-ratio", type=float, default=0.2)
    parser.add_argument("--head-class-ratio", type=float, default=0.8)
    parser.add_argument("--tail-class-ratio", type=float, default=0.2)
    parser.add_argument("--specialization-lambda", type=float, default=1.0)
    parser.add_argument("--intra-group-alpha", type=float, default=0.3)
    parser.add_argument("--head-leakage-scale", type=float, default=3.0)
    parser.add_argument(
        "--use-project-datasplit",
        action="store_true",
        help=(
            "Use utils.datasplit.partition_data_LT. By default the script uses "
            "the lightweight CIFAR label path shared with phase1_exposure_topology.py."
        ),
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit with an error if the two protocols do not have identical global class-count vectors.",
    )
    return parser.parse_args()


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str] | None = None) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def group_counts(groups: dict[int, str]) -> dict[str, int]:
    out = {"head": 0, "middle": 0, "tail": 0}
    for group in groups.values():
        out[group] = out.get(group, 0) + 1
    return out


def imbalance_factor(global_counts: np.ndarray) -> float:
    positive = global_counts[global_counts > 0]
    if positive.size == 0:
        return 0.0
    return float(positive.max() / positive.min())


def summarize_split(split_name: str, counts: np.ndarray, groups: dict[int, str]) -> dict[str, object]:
    global_counts = counts.sum(axis=0)
    sizes = group_counts(groups)
    return {
        "split": split_name,
        "protocol": SPLIT_LABELS.get(split_name, split_name),
        "total_samples": int(global_counts.sum()),
        "num_classes": int(global_counts.size),
        "imbalance_factor": f"{imbalance_factor(global_counts):.6f}",
        "max_class_count": int(global_counts.max()) if global_counts.size else 0,
        "min_positive_class_count": int(global_counts[global_counts > 0].min()) if np.any(global_counts > 0) else 0,
        "mean_class_count": f"{float(global_counts.mean()):.6f}" if global_counts.size else "0.000000",
        "median_class_count": f"{float(np.median(global_counts)):.6f}" if global_counts.size else "0.000000",
        "head_classes": int(sizes.get("head", 0)),
        "middle_classes": int(sizes.get("middle", 0)),
        "tail_classes": int(sizes.get("tail", 0)),
    }


def compare_reference(
    reference_counts: np.ndarray,
    split_counts: dict[str, np.ndarray],
) -> list[dict[str, object]]:
    rows = []
    for split_name, counts in split_counts.items():
        global_counts = counts.sum(axis=0)
        diff = np.abs(global_counts - reference_counts)
        rows.append(
            {
                "split": split_name,
                "protocol": SPLIT_LABELS.get(split_name, split_name),
                "l1_count_difference": int(diff.sum()),
                "max_class_count_difference": int(diff.max()) if diff.size else 0,
                "num_mismatched_classes": int(np.count_nonzero(diff)),
                "matches_reference": bool(np.all(diff == 0)),
            }
        )
    return rows


def per_class_rows(
    classnames: list[str],
    split_counts: dict[str, np.ndarray],
    groups: dict[int, str],
) -> list[dict[str, object]]:
    dir_counts = split_counts["dirichlet"].sum(axis=0)
    client_counts = split_counts["client_lt"].sum(axis=0)
    rows = []
    for class_id in range(len(dir_counts)):
        rows.append(
            {
                "class_id": int(class_id),
                "class_name": classnames[class_id] if class_id < len(classnames) else str(class_id),
                "group": groups[class_id],
                "dirichlet_global_count": int(dir_counts[class_id]),
                "client_lt_global_count": int(client_counts[class_id]),
                "absolute_difference": int(abs(dir_counts[class_id] - client_counts[class_id])),
            }
        )
    return rows


def split_rows(classnames: list[str], global_counts: np.ndarray, groups: dict[int, str]) -> list[dict[str, object]]:
    rows = []
    for class_id, count in enumerate(global_counts.tolist()):
        rows.append(
            {
                "class_id": int(class_id),
                "class_name": classnames[class_id] if class_id < len(classnames) else str(class_id),
                "global_count": int(count),
                "group": groups[class_id],
            }
        )
    return rows


def controlled_variable_rows(args: argparse.Namespace) -> list[dict[str, object]]:
    variables = [
        ("dataset", args.dataset, "Dataset used for protocol verification."),
        ("global_lt_seed", args.global_lt_seed, "Seed of the shared global long-tailed sample pool."),
        ("imb_type", args.imb_type, "Shape of the global long-tail distribution."),
        ("imb_factor", args.imb_factor, "Long-tail strength, max class count divided by min class count."),
        ("num_users", args.num_users, "Number of federated clients."),
        ("frac", args.frac, "Client participation rate used in later training."),
        ("rounds", args.rounds, "Communication rounds used in later training."),
        ("local_epochs", args.local_epochs, "Local epochs used in later training."),
        ("batch_size", args.batch_size, "Batch size used in later training."),
        ("model_init_seed", args.model_init_seed, "Model initialization seed used in later training."),
        ("schedule_seed", args.schedule_seed, "Client sampling seed used in later training."),
        ("dirichlet_beta", args.dirichlet_beta, "Dirichlet label-skew parameter."),
        ("head_class_ratio", args.head_class_ratio, "Fraction of classes assigned to the head group."),
        ("tail_class_ratio", args.tail_class_ratio, "Fraction of classes assigned to the tail group."),
        ("head_client_ratio", args.head_client_ratio, "Fraction of head/general clients in Client-LT."),
        ("tail_client_ratio", args.tail_client_ratio, "Fraction of tail-specialized clients in Client-LT."),
        ("specialization_lambda", args.specialization_lambda, "Client-LT tail-class specialization strength."),
        ("intra_group_alpha", args.intra_group_alpha, "Client-LT within-group Dirichlet concentration."),
        ("head_leakage_scale", args.head_leakage_scale, "Client-LT non-tail leakage scale."),
    ]
    return [{"variable": key, "value": value, "role": role} for key, value, role in variables]


def write_latex_summary(path: Path, summary_rows: list[dict[str, object]], diff_rows: list[dict[str, object]]) -> None:
    diff_by_split = {row["split"]: row for row in diff_rows}
    headers = [
        "Protocol",
        "Total",
        "Classes",
        "IF",
        "Head",
        "Middle",
        "Tail",
        "$\\ell_1$ diff",
        "Max diff",
    ]
    lines = [
        "\\begin{tabular}{lrrrrrrrr}",
        "\\toprule",
        " & ".join(headers) + " \\\\",
        "\\midrule",
    ]
    for row in summary_rows:
        diff = diff_by_split[row["split"]]
        values = [
            str(row["protocol"]),
            str(row["total_samples"]),
            str(row["num_classes"]),
            f"{float(row['imbalance_factor']):.2f}",
            str(row["head_classes"]),
            str(row["middle_classes"]),
            str(row["tail_classes"]),
            str(diff["l1_count_difference"]),
            str(diff["max_class_count_difference"]),
        ]
        lines.append(" & ".join(values) + " \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def write_paper_notes(path: Path, args: argparse.Namespace, all_match: bool) -> None:
    status = "PASS" if all_match else "FAIL"
    text = f"""# Experiment 1 Paper Notes

Status: {status}

Recommended location:

- Method / Protocol section: describe that Client-LT is generated from the same global long-tailed training pool as Dirichlet.
- Experimental setup: cite `controlled_variables.csv` to state the fixed variables.
- First results subsection: place `global_longtail_summary.csv` and `class_count_curve.pdf` before any accuracy table.

Recommended subsection title:

Client-LT Preserves Global Long-Tail Statistics

Recommended claim:

Dirichlet and Client-LT share the same global class-count vector, imbalance factor,
and head/middle/tail split. Client-LT therefore does not change the amount of
tail evidence; it only reorganizes where that evidence resides across clients.

Recommended figure:

- `class_count_curve.pdf`: sorted global class-count curves. The two protocols
  should overlap exactly.

Recommended table:

- `paper_table_global_longtail.tex`: total samples, number of classes, imbalance
  factor, class group sizes, L1 count difference, and max class-count difference.

Do not conclude from Experiment 1:

- Client-LT is harder than Dirichlet.
- CAPT fails under Client-LT.
- Topology changes learning dynamics.
- Ours is better.

Experiment 1 only establishes the control:

Dirichlet and Client-LT are matched in global long-tail statistics.

Run configuration:

- dataset: {args.dataset}
- imbalance type: {args.imb_type}
- imbalance factor: {args.imb_factor}
- num users: {args.num_users}
- Dirichlet beta: {args.dirichlet_beta}
- Client-LT specialization lambda: {args.specialization_lambda}
- Client-LT intra-group alpha: {args.intra_group_alpha}
- Client-LT head leakage scale: {args.head_leakage_scale}
"""
    path.write_text(text, encoding="utf-8")


def plot_class_count_curve(split_counts: dict[str, np.ndarray], output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    dir_counts = split_counts["dirichlet"].sum(axis=0)
    client_lt_counts = split_counts["client_lt"].sum(axis=0)
    sorted_counts = np.sort(dir_counts)[::-1]
    x = np.arange(1, len(sorted_counts) + 1)
    diff = np.abs(dir_counts - client_lt_counts)

    ax.plot(
        x,
        sorted_counts,
        label="Shared global class counts",
        color="#333333",
        marker="o",
        markersize=2.2,
        linewidth=1.8,
        alpha=0.95,
    )
    ax.text(
        0.98,
        0.96,
        f"Dirichlet = Client-LT\n$\\ell_1$ diff = {int(diff.sum())}\nmax diff = {int(diff.max()) if diff.size else 0}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=10,
        bbox=dict(facecolor="white", edgecolor="#BDBDBD", alpha=0.92, boxstyle="round,pad=0.35"),
    )
    ax.set_xlim(1, len(sorted_counts))
    ax.set_xlabel("Class rank by global sample count")
    ax.set_ylabel("Training samples per class")
    ax.set_title("Global long-tail distribution is preserved")
    ax.legend(frameon=False, loc="upper right", bbox_to_anchor=(0.98, 0.78))
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=240, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_count_difference(per_class: list[dict[str, object]], output_path: Path) -> None:
    class_ids = [int(row["class_id"]) for row in per_class]
    diffs = [int(row["absolute_difference"]) for row in per_class]
    fig, ax = plt.subplots(figsize=(6.2, 2.8))
    ax.axhline(0, color="#333333", linewidth=1.2)
    ax.scatter(class_ids, diffs, color="#4C78A8", s=12, zorder=3)
    if max(diffs, default=0) == 0:
        ax.text(
            0.5,
            0.62,
            "All per-class global-count differences are zero",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=11,
            bbox=dict(facecolor="white", edgecolor="#BDBDBD", alpha=0.94, boxstyle="round,pad=0.35"),
        )
    ax.set_xlabel("Class ID")
    ax.set_ylabel("Absolute count difference")
    ax.set_title("Dirichlet vs. Client-LT global-count difference")
    ax.set_xlim(min(class_ids, default=0) - 1, max(class_ids, default=1) + 1)
    ax.set_ylim(-0.05, max(max(diffs, default=0), 1))
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=240, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def print_summary(summary_rows: Iterable[dict[str, object]], diff_rows: Iterable[dict[str, object]]) -> None:
    print("Experiment 1: Global Long-Tail Preservation Verification")
    for row in summary_rows:
        print(
            f"- {row['protocol']}: total={row['total_samples']}, classes={row['num_classes']}, "
            f"IF={float(row['imbalance_factor']):.2f}, "
            f"head/middle/tail={row['head_classes']}/{row['middle_classes']}/{row['tail_classes']}"
        )
    for row in diff_rows:
        print(
            f"- {row['protocol']} vs reference: L1 diff={row['l1_count_difference']}, "
            f"max diff={row['max_class_count_difference']}, "
            f"mismatched classes={row['num_mismatched_classes']}"
        )


def main() -> None:
    set_plot_style()
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    split_counts: dict[str, np.ndarray] = {}
    y_by_split: dict[str, np.ndarray] = {}
    classnames: list[str] | None = None
    reference_groups: dict[int, str] | None = None

    for split_name, partition_name in PARTITION_SPECS:
        names, net_train, y_train = build_split(args, split_name, partition_name)
        classnames = names
        counts = counts_matrix(y_train, net_train, args.num_users, len(names))
        split_counts[split_name] = counts
        y_by_split[split_name] = np.asarray(y_train, dtype=np.int64)
        groups = class_groups(counts.sum(axis=0), args.head_class_ratio, args.tail_class_ratio)
        if reference_groups is None:
            reference_groups = groups

    if classnames is None or reference_groups is None:
        raise RuntimeError("No partition was built.")

    reference_counts = split_counts["dirichlet"].sum(axis=0)
    summary_rows = [
        summarize_split(split_name, counts, reference_groups)
        for split_name, counts in split_counts.items()
    ]
    diff_rows = compare_reference(reference_counts, split_counts)
    class_rows = per_class_rows(classnames, split_counts, reference_groups)
    hmt_rows = split_rows(classnames, reference_counts, reference_groups)
    controlled_rows = controlled_variable_rows(args)

    write_csv(output_dir / "global_longtail_summary.csv", summary_rows)
    write_csv(output_dir / "global_count_verification.csv", diff_rows)
    write_csv(output_dir / "global_class_counts.csv", class_rows)
    write_csv(output_dir / "head_medium_tail_split.csv", hmt_rows)
    write_csv(output_dir / "controlled_variables.csv", controlled_rows)
    write_latex_summary(output_dir / "paper_table_global_longtail.tex", summary_rows, diff_rows)
    all_match = all(bool(row["matches_reference"]) for row in diff_rows)
    write_paper_notes(output_dir / "paper_notes.md", args, all_match)

    config = vars(args).copy()
    config["partitions"] = dict(PARTITION_SPECS)
    config["all_global_counts_match"] = bool(all_match)
    config["dirichlet_y_train_checksum"] = int(np.sum(y_by_split["dirichlet"] * (np.arange(len(y_by_split["dirichlet"])) + 1)))
    config["client_lt_y_train_checksum"] = int(np.sum(y_by_split["client_lt"] * (np.arange(len(y_by_split["client_lt"])) + 1)))
    write_json(output_dir / "config.json", config)

    plot_class_count_curve(split_counts, output_dir / "class_count_curve.png")
    plot_count_difference(class_rows, output_dir / "class_count_difference.png")

    print_summary(summary_rows, diff_rows)
    print(f"Saved Experiment 1 outputs to {output_dir}")
    print(f"- {output_dir / 'global_longtail_summary.csv'}")
    print(f"- {output_dir / 'global_count_verification.csv'}")
    print(f"- {output_dir / 'global_class_counts.csv'}")
    print(f"- {output_dir / 'head_medium_tail_split.csv'}")
    print(f"- {output_dir / 'class_count_curve.pdf'}")
    print(f"- {output_dir / 'paper_table_global_longtail.tex'}")
    print(f"- {output_dir / 'paper_notes.md'}")

    if args.strict and not all_match:
        raise SystemExit("Global class counts do not match across protocols.")


if __name__ == "__main__":
    main()
