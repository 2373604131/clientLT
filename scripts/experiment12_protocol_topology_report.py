#!/usr/bin/env python
"""Combined Experiment 1+2 protocol/topology report.

This is the paper-facing protocol verification script. It combines:

1. Global long-tail preservation:
   Dirichlet and Client-LT use identical global class-count vectors.
2. Tail evidence topology difference:
   The same tail samples are organized differently across clients.

The intended paper claim is:

    Same global long-tail, different client-level tail evidence topology.

This script does not train a model.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.experiment1_global_longtail_verification import (  # noqa: E402
    SPLIT_LABELS,
    compare_reference,
    controlled_variable_rows,
    per_class_rows,
    split_rows,
    summarize_split,
    write_latex_summary,
)
from scripts.phase1_exposure_topology import (  # noqa: E402
    PARTITION_SPECS,
    build_split,
    class_groups,
    counts_matrix,
    plot_tail_mass_heatmap,
    plot_tail_metric_boxplots,
    plot_topology_overview,
    set_plot_style,
    summarize,
    topology_rows,
    write_counts_csv,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate the combined same-global-LT/different-topology report."
    )
    parser.add_argument("--dataset", default="cifar100_LT", choices=["cifar10_LT", "cifar100_LT", "fmnist_LT"])
    parser.add_argument("--data-root", default="DATA")
    parser.add_argument("--output-dir", default="output/experiment12_protocol_topology_report")
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
            "Use utils.datasplit.partition_data_LT. The default path uses the "
            "lightweight CIFAR-label loader shared with phase1_exposure_topology.py."
        ),
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit with an error if global class-count vectors do not match.",
    )
    return parser.parse_args()


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str] | None = None) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def write_combined_paper_notes(
    path: Path,
    global_match: bool,
    summary_rows: list[dict[str, object]],
    topology_summary_rows: list[dict[str, object]],
) -> None:
    dir_row = next(row for row in summary_rows if row["split"] == "dirichlet")
    client_row = next(row for row in summary_rows if row["split"] == "client_lt")
    tail_topology = [
        row for row in topology_summary_rows
        if row["group"] == "tail" and row["split"] in {"dirichlet", "client_lt"}
    ]

    def metric(split: str, name: str) -> float:
        row = next(r for r in tail_topology if r["split"] == split)
        return float(row[name])

    status = "PASS" if global_match else "FAIL"
    text = f"""# Combined Experiment 1+2 Paper Notes

Status: {status}

Recommended paper subsection title:

Same Global Long-Tail, Different Tail Evidence Topology

What this section proves:

1. Dirichlet and Client-LT preserve the same global long-tail statistics.
2. Under the same global class counts, Client-LT reorganizes tail samples into
   a more client-specialized, concentrated topology.

Use in the paper:

- Put `figure_protocol_topology_overview.pdf` in the main text.
- Put `paper_table_protocol_control.tex` either in the main text or immediately
  below the figure.
- Put the complete per-class count table in the appendix if space is limited.

Global control:

- Dirichlet total samples: {dir_row['total_samples']}
- Client-LT total samples: {client_row['total_samples']}
- Number of classes: {dir_row['num_classes']}
- Imbalance factor: {float(dir_row['imbalance_factor']):.2f}
- Head/middle/tail classes: {dir_row['head_classes']}/{dir_row['middle_classes']}/{dir_row['tail_classes']}

Tail topology contrast:

- Top-1 client mass: Dirichlet {metric('dirichlet', 'top1_client_mass_mean'):.3f}, Client-LT {metric('client_lt', 'top1_client_mass_mean'):.3f}
- Top-2 client mass: Dirichlet {metric('dirichlet', 'top2_client_mass_mean'):.3f}, Client-LT {metric('client_lt', 'top2_client_mass_mean'):.3f}
- Concentration: Dirichlet {metric('dirichlet', 'concentration_C_mean'):.3f}, Client-LT {metric('client_lt', 'concentration_C_mean'):.3f}
- Expected temporal exposure: Dirichlet {metric('dirichlet', 'expected_temporal_exposure_mean'):.3f}, Client-LT {metric('client_lt', 'expected_temporal_exposure_mean'):.3f}

Recommended claim:

Dirichlet and Client-LT are matched in global long-tail statistics but differ
substantially in client-level tail evidence topology. Therefore, subsequent
differences in learning behavior should not be attributed to a change in the
amount of tail data, but to how tail evidence is organized across clients.

Do not write:

- Client-LT is harder than Dirichlet.
- Client-LT changes the global long-tail distribution.
- CAPT fails under Client-LT.

Preferred wording:

Client-LT captures a distinct client-specialization axis while preserving the
same global long-tailed class-count vector as standard Dirichlet partitions.
"""
    path.write_text(text, encoding="utf-8")


def main() -> None:
    set_plot_style()
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_topology_rows: list[dict[str, object]] = []
    split_counts: dict[str, np.ndarray] = {}
    classnames: list[str] | None = None
    reference_groups: dict[int, str] | None = None

    for split_name, partition_name in PARTITION_SPECS:
        names, net_train, y_train = build_split(args, split_name, partition_name)
        classnames = names
        counts = counts_matrix(y_train, net_train, args.num_users, len(names))
        groups = class_groups(counts.sum(axis=0), args.head_class_ratio, args.tail_class_ratio)
        if reference_groups is None:
            reference_groups = groups
        split_counts[split_name] = counts

        write_counts_csv(output_dir / f"client_class_counts_{split_name}.csv", counts)
        all_topology_rows.extend(
            topology_rows(
                split_name,
                partition_name,
                counts,
                names,
                reference_groups,
                args.frac,
                args.rounds,
                args.schedule_seed,
            )
        )

    if classnames is None or reference_groups is None:
        raise RuntimeError("No partition was built.")

    reference_counts = split_counts["dirichlet"].sum(axis=0)
    global_summary_rows = [
        summarize_split(split_name, counts, reference_groups)
        for split_name, counts in split_counts.items()
    ]
    global_diff_rows = compare_reference(reference_counts, split_counts)
    global_match = all(bool(row["matches_reference"]) for row in global_diff_rows)
    topology_summary_rows = summarize(all_topology_rows)

    write_csv(output_dir / "global_longtail_summary.csv", global_summary_rows)
    write_csv(output_dir / "global_count_verification.csv", global_diff_rows)
    write_csv(output_dir / "global_class_counts.csv", per_class_rows(classnames, split_counts, reference_groups))
    write_csv(output_dir / "head_medium_tail_split.csv", split_rows(classnames, reference_counts, reference_groups))
    write_csv(output_dir / "controlled_variables.csv", controlled_variable_rows(args))
    write_csv(output_dir / "class_topology.csv", all_topology_rows)
    write_csv(output_dir / "summary_by_group.csv", topology_summary_rows)

    write_latex_summary(output_dir / "paper_table_protocol_control.tex", global_summary_rows, global_diff_rows)
    write_combined_paper_notes(
        output_dir / "paper_notes.md",
        global_match,
        global_summary_rows,
        topology_summary_rows,
    )

    plot_topology_overview(
        all_topology_rows,
        split_counts,
        reference_groups,
        output_dir / "figure_protocol_topology_overview.png",
    )
    plot_tail_metric_boxplots(
        all_topology_rows,
        output_dir / "figure_tail_topology_boxplots.png",
    )
    plot_tail_mass_heatmap(
        split_counts,
        reference_groups,
        output_dir / "figure_tail_mass_heatmap.png",
    )

    config = vars(args).copy()
    config["partitions"] = dict(PARTITION_SPECS)
    config["all_global_counts_match"] = bool(global_match)
    config["split_labels"] = SPLIT_LABELS
    write_json(output_dir / "config.json", config)

    print("Combined Experiment 1+2 report script finished.")
    print(f"Output dir: {output_dir}")
    print(f"Global counts match: {global_match}")
    print(f"Main figure: {output_dir / 'figure_protocol_topology_overview.pdf'}")
    print(f"Protocol table: {output_dir / 'paper_table_protocol_control.tex'}")
    print(f"Paper notes: {output_dir / 'paper_notes.md'}")

    if args.strict and not global_match:
        raise SystemExit("Global class counts do not match across protocols.")


if __name__ == "__main__":
    main()
