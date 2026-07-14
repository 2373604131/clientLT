#!/usr/bin/env python
"""Phase 2 mechanism analysis for tail exposure topology.

This script analyzes baseline runs (PromptFL/CAPT by default) under Dirichlet
and client-longtail partitions. It focuses on the part that can be measured
from the project's existing outputs:

  * global tail acquisition/survival trajectories from round_metrics.csv;
  * per-tail-class peak-to-final retention from per_class_accuracy_epoch_*.csv;
  * topology-to-survival coupling from client_class_counts.csv.

Pairwise confuser analysis requires confusion matrices or prediction dumps.
If those files are absent, the script writes a short instrumentation note that
specifies the expected files for the next training pass.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


METHODS = ("PromptFL", "CAPT")
PARTITIONS = ("noniid-labeldir", "client-longtail")
PARTITION_LABEL = {
    "noniid-labeldir": "Dirichlet",
    "client-longtail": "Client-LT",
}
METHOD_COLOR = {
    "PromptFL": "#4C78A8",
    "CAPT": "#F58518",
}
PARTITION_COLOR = {
    "noniid-labeldir": "#4C78A8",
    "client-longtail": "#F58518",
}


def set_plot_style() -> None:
    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 9,
            "figure.titlesize": 13,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Analyze Phase 2 tail failure mechanisms.")
    parser.add_argument(
        "--run-roots",
        nargs="+",
        default=[
            "output/cifar100_LT/motivation1_topology",
            "output/cifar100_LT/capt_main_matched",
        ],
        help="Roots recursively searched for run directories.",
    )
    parser.add_argument("--output-dir", default="output/phase2_failure_mechanisms")
    parser.add_argument("--methods", nargs="+", default=list(METHODS))
    parser.add_argument("--partitions", nargs="+", default=list(PARTITIONS))
    parser.add_argument("--tail-class-ratio", type=float, default=0.2)
    parser.add_argument("--frac", type=float, default=0.4)
    parser.add_argument("--fracs", nargs="+", default=["0.4"], help="Only analyze run dirs with these participation rates.")
    parser.add_argument("--num-users", type=int, default=20)
    parser.add_argument("--min-final-epoch", type=int, default=0)
    parser.add_argument("--write-commands", default="", help="Optional PowerShell command file for missing runs.")
    parser.add_argument("--dataset", default="cifar100_LT")
    parser.add_argument("--rounds", type=int, default=100)
    parser.add_argument("--seeds", nargs="+", default=["1", "42", "3407"])
    parser.add_argument("--gpu", default="0")
    return parser.parse_args()


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


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


def infer_method_partition(run_dir: Path, rows: list[dict[str, str]]) -> tuple[str | None, str | None]:
    method = rows[0].get("method") if rows else None
    partition = rows[0].get("partition") if rows else None
    name = run_dir.name
    for candidate in METHODS:
        if candidate.lower() in name.lower():
            method = candidate
    for candidate in PARTITIONS:
        if candidate in name:
            partition = candidate
    return method, partition


def infer_frac(run_dir: Path, rows: list[dict[str, str]], default: float) -> float:
    if rows and rows[0].get("frac") not in (None, ""):
        return float(rows[0]["frac"])
    for part in run_dir.parts:
        match = re.fullmatch(r"frac([0-9.]+)", part)
        if match:
            return float(match.group(1))
    return float(default)


def discover_runs(args: argparse.Namespace) -> list[Path]:
    run_dirs = []
    seen = set()
    for root in args.run_roots:
        root_path = Path(root)
        if not root_path.exists():
            continue
        for metrics_path in root_path.rglob("round_metrics.csv"):
            run_dir = metrics_path.parent
            if run_dir in seen:
                continue
            seen.add(run_dir)
            rows = read_csv_rows(metrics_path)
            method, partition = infer_method_partition(run_dir, rows)
            frac = infer_frac(run_dir, rows, args.frac)
            allowed_fracs = {round(float(x), 8) for x in args.fracs}
            if method in args.methods and partition in args.partitions:
                if round(frac, 8) not in allowed_fracs:
                    continue
                if rows:
                    last_epoch = max(int(float(r.get("epoch", -1))) for r in rows)
                    if last_epoch < args.min_final_epoch:
                        continue
                run_dirs.append(run_dir)
    return sorted(run_dirs)


def load_counts(path: Path) -> np.ndarray:
    rows = read_csv_rows(path)
    if not rows:
        raise ValueError(f"Empty client counts file: {path}")
    class_cols = [c for c in rows[0].keys() if c.startswith("class_")]
    class_cols.sort(key=lambda x: int(x.split("_", 1)[1]))
    counts = []
    for row in rows:
        counts.append([int(float(row[c])) for c in class_cols])
    return np.asarray(counts, dtype=np.float64)


def get_tail_classes(run_dir: Path, counts: np.ndarray, tail_class_ratio: float) -> list[int]:
    topology_path = run_dir / "class_topology.csv"
    if topology_path.exists():
        rows = read_csv_rows(topology_path)
        tail = [
            int(float(r["class_id"]))
            for r in rows
            if r.get("class_group", "").strip().lower() == "tail"
        ]
        if tail:
            return sorted(tail)
    global_counts = counts.sum(axis=0)
    order = np.argsort(-global_counts)
    n_tail = max(1, int(len(global_counts) * tail_class_ratio))
    return sorted(order[-n_tail:].tolist())


def topology_for_counts(counts: np.ndarray, tail_classes: list[int], frac: float) -> dict[int, dict[str, float]]:
    num_users = counts.shape[0]
    clients_per_round = max(int(frac * num_users), 1)
    out = {}
    for cls in tail_classes:
        per_client = counts[:, cls].astype(np.float64)
        total = float(per_client.sum())
        support_clients = int(np.count_nonzero(per_client))
        if total > 0:
            mass = np.sort(per_client / total)[::-1]
            concentration = float(np.sum(per_client ** 2) / (total ** 2))
            local_depth = float(np.sum(per_client ** 2) / total)
            top1_mass = float(mass[0])
            top2_mass = float(mass[:2].sum())
            neff = float(1.0 / concentration) if concentration > 0 else 0.0
        else:
            concentration = local_depth = top1_mass = top2_mass = neff = 0.0
        if support_clients <= 0:
            exposure = 0.0
        elif support_clients >= num_users or clients_per_round >= num_users:
            exposure = 1.0
        elif num_users - support_clients < clients_per_round:
            exposure = 1.0
        else:
            miss = math.comb(num_users - support_clients, clients_per_round) / math.comb(num_users, clients_per_round)
            exposure = 1.0 - miss
        out[int(cls)] = {
            "support_clients": support_clients,
            "top1_mass": top1_mass,
            "top2_mass": top2_mass,
            "concentration": concentration,
            "local_depth": local_depth,
            "effective_clients": neff,
            "temporal_exposure": exposure,
            "global_count": total,
        }
    return out


def read_per_class_history(run_dir: Path) -> dict[int, dict[int, float]]:
    history = defaultdict(dict)
    pattern = re.compile(r"per_class_accuracy_epoch_(\d+)\.csv$")
    for path in run_dir.glob("per_class_accuracy_epoch_*.csv"):
        match = pattern.search(path.name)
        if not match:
            continue
        epoch = int(match.group(1))
        for row in read_csv_rows(path):
            cls = int(float(row["class_id"]))
            acc = float(row["per_class_acc"])
            history[cls][epoch] = acc
    return dict(history)


def analyze_run(run_dir: Path, args: argparse.Namespace) -> tuple[dict[str, object], list[dict[str, object]], list[dict[str, object]]]:
    round_rows = read_csv_rows(run_dir / "round_metrics.csv")
    method, partition = infer_method_partition(run_dir, round_rows)
    seed = round_rows[0].get("seed", "") if round_rows else ""
    frac = infer_frac(run_dir, round_rows, args.frac)
    counts = load_counts(run_dir / "client_class_counts.csv")
    tail_classes = get_tail_classes(run_dir, counts, args.tail_class_ratio)
    topology = topology_for_counts(counts, tail_classes, frac)
    per_class_history = read_per_class_history(run_dir)

    trajectory = []
    for row in round_rows:
        trajectory.append(
            {
                "run_dir": str(run_dir),
                "method": method,
                "partition": partition,
                "partition_label": PARTITION_LABEL.get(str(partition), str(partition)),
                "seed": seed,
                "frac": frac,
                "epoch": int(float(row["epoch"])),
                "overall_acc": float(row.get("overall_acc", 0.0)),
                "tail_acc": float(row.get("tail_acc", row.get("bottom20_tail_acc", 0.0))),
                "head_acc": float(row.get("head_acc", row.get("non_tail_acc", 0.0))),
                "macro_acc": float(row.get("macro_per_class_acc", 0.0)),
            }
        )

    class_rows = []
    for cls in tail_classes:
        hist = per_class_history.get(int(cls), {})
        if not hist:
            continue
        epochs = sorted(hist)
        values = np.asarray([hist[e] for e in epochs], dtype=np.float64)
        initial = float(values[0])
        final = float(values[-1])
        best_idx = int(values.argmax())
        best = float(values[best_idx])
        best_epoch = int(epochs[best_idx])
        best_gain = best - initial
        final_gain = final - initial
        survival_drop = best - final
        retention_ratio = (final_gain / best_gain) if best_gain > 1e-8 else float("nan")
        topo = topology[int(cls)]
        class_rows.append(
            {
                "run_dir": str(run_dir),
                "method": method,
                "partition": partition,
                "partition_label": PARTITION_LABEL.get(str(partition), str(partition)),
                "seed": seed,
                "frac": frac,
                "class_id": int(cls),
                "initial_acc": initial,
                "best_acc": best,
                "best_epoch": best_epoch,
                "final_acc": final,
                "best_global_gain": best_gain,
                "final_global_gain": final_gain,
                "survival_drop": survival_drop,
                "retention_ratio": retention_ratio,
                "acc_std": float(values.std()),
                **topo,
            }
        )

    tail_traj = [r for r in trajectory]
    if tail_traj:
        tail_values = np.asarray([float(r["tail_acc"]) for r in tail_traj], dtype=np.float64)
        epochs = [int(r["epoch"]) for r in tail_traj]
        best_idx = int(tail_values.argmax())
        run_summary = {
            "run_dir": str(run_dir),
            "method": method,
            "partition": partition,
            "partition_label": PARTITION_LABEL.get(str(partition), str(partition)),
            "seed": seed,
            "frac": frac,
            "initial_tail_acc": float(tail_values[0]),
            "best_tail_acc": float(tail_values[best_idx]),
            "best_epoch": int(epochs[best_idx]),
            "final_tail_acc": float(tail_values[-1]),
            "tail_best_gain": float(tail_values[best_idx] - tail_values[0]),
            "tail_final_gain": float(tail_values[-1] - tail_values[0]),
            "tail_survival_drop": float(tail_values[best_idx] - tail_values[-1]),
            "tail_retention_ratio": float((tail_values[-1] - tail_values[0]) / (tail_values[best_idx] - tail_values[0]))
            if (tail_values[best_idx] - tail_values[0]) > 1e-8
            else float("nan"),
            "mean_tail_top1_mass": float(np.mean([topology[c]["top1_mass"] for c in tail_classes])),
            "mean_tail_concentration": float(np.mean([topology[c]["concentration"] for c in tail_classes])),
            "mean_tail_temporal_exposure": float(np.mean([topology[c]["temporal_exposure"] for c in tail_classes])),
            "mean_tail_local_depth": float(np.mean([topology[c]["local_depth"] for c in tail_classes])),
        }
    else:
        run_summary = {"run_dir": str(run_dir), "method": method, "partition": partition, "seed": seed, "frac": frac}
    return run_summary, class_rows, trajectory


def group_mean_trajectory(rows: list[dict[str, object]]) -> dict[tuple[str, str], tuple[list[int], list[float], list[float]]]:
    grouped = defaultdict(lambda: defaultdict(list))
    for row in rows:
        key = (str(row["method"]), str(row["partition"]))
        grouped[key][int(row["epoch"])].append(float(row["tail_acc"]))
    out = {}
    for key, epoch_map in grouped.items():
        epochs = sorted(epoch_map)
        means = [float(np.mean(epoch_map[e])) for e in epochs]
        sems = [
            float(np.std(epoch_map[e], ddof=1) / math.sqrt(len(epoch_map[e]))) if len(epoch_map[e]) > 1 else 0.0
            for e in epochs
        ]
        out[key] = (epochs, means, sems)
    return out


def plot_phase2_overview(
    summary_rows: list[dict[str, object]],
    class_rows: list[dict[str, object]],
    trajectory_rows: list[dict[str, object]],
    output_path: Path,
) -> None:
    fig = plt.figure(figsize=(12.5, 7.0))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.2, 1.0], height_ratios=[1.0, 1.0])
    ax_traj = fig.add_subplot(gs[0, 0])
    ax_bar = fig.add_subplot(gs[0, 1])
    ax_scatter = fig.add_subplot(gs[1, :])

    markers = {"noniid-labeldir": "o", "client-longtail": "s"}
    linestyles = {"noniid-labeldir": "-", "client-longtail": "--"}
    for (method, partition), (epochs, means, sems) in group_mean_trajectory(trajectory_rows).items():
        color = METHOD_COLOR.get(method, "gray")
        label = f"{method} / {PARTITION_LABEL.get(partition, partition)}"
        ax_traj.plot(epochs, means, color=color, linestyle=linestyles.get(partition, "-"), marker=markers.get(partition, "o"), markersize=3, label=label)
        if any(sems):
            lo = np.asarray(means) - np.asarray(sems)
            hi = np.asarray(means) + np.asarray(sems)
            ax_traj.fill_between(epochs, lo, hi, color=color, alpha=0.12)
    ax_traj.set_title("Global tail accuracy trajectory")
    ax_traj.set_xlabel("Round")
    ax_traj.set_ylabel("Tail accuracy (%)")
    ax_traj.grid(axis="y", alpha=0.25)
    ax_traj.legend(frameon=False, ncol=1)

    combo = sorted({(str(r["method"]), str(r["partition"])) for r in summary_rows})
    x = np.arange(len(combo))
    drops = []
    labels = []
    colors = []
    for method, partition in combo:
        vals = [
            float(r["tail_survival_drop"])
            for r in summary_rows
            if r["method"] == method and r["partition"] == partition
        ]
        drops.append(float(np.mean(vals)))
        labels.append(f"{method}\n{PARTITION_LABEL.get(partition, partition)}")
        colors.append(PARTITION_COLOR.get(partition, "gray"))
    ax_bar.bar(x, drops, color=colors, alpha=0.9)
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(labels)
    ax_bar.set_title("Peak-to-final tail drop")
    ax_bar.set_ylabel("Accuracy drop (%)")
    ax_bar.grid(axis="y", alpha=0.25)

    for partition in PARTITIONS:
        subset = [r for r in class_rows if r["partition"] == partition]
        if not subset:
            continue
        ax_scatter.scatter(
            [float(r["top1_mass"]) for r in subset],
            [float(r["survival_drop"]) for r in subset],
            s=38,
            alpha=0.62,
            label=PARTITION_LABEL.get(partition, partition),
            color=PARTITION_COLOR.get(partition, "gray"),
            edgecolors="none",
        )
    ax_scatter.axhline(0.0, color="black", linewidth=0.8, alpha=0.5)
    ax_scatter.set_title("Tail concentration vs. global survival loss")
    ax_scatter.set_xlabel("Top-1 client mass of a tail class")
    ax_scatter.set_ylabel("Per-class peak-to-final drop (%)")
    ax_scatter.grid(axis="both", alpha=0.22)
    ax_scatter.legend(frameon=False)

    fig.suptitle("Fragmented weak evidence vs. concentrated intermittent evidence", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=240, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def find_pairwise_inputs(run_dirs: list[Path]) -> list[Path]:
    files = []
    patterns = [
        "confusion_epoch_*.csv",
        "confusion_matrix_epoch_*.csv",
        "predictions_epoch_*.csv",
        "per_client_predictions_epoch_*.csv",
    ]
    for run_dir in run_dirs:
        for pattern in patterns:
            files.extend(run_dir.glob(pattern))
    return sorted(files)


def write_pairwise_note(path: Path) -> None:
    text = """# Phase 2 pairwise confuser logging needed

The current analysis found no confusion/prediction dumps, so it can analyze
global survival but cannot yet measure tail-confuser stability.

For Experiment 3, save one of the following after each global evaluation round:

1. `confusion_epoch_{epoch}.csv`
   - square matrix, rows are true labels, columns are predicted labels; or
   - long format with columns: `true_class,pred_class,count`.

2. `per_client_predictions_epoch_{epoch}.csv`
   - columns: `client_id,sample_id,true_class,pred_class`.
   - This enables expert-client confuser stability.

The script will use these files to compute:

- tail-to-head error;
- top-confuser class for each tail class;
- top-confuser stability across rounds;
- whether expert clients repeatedly identify the same tail-vs-confuser pairs.
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_run_commands(args: argparse.Namespace, path: Path) -> None:
    lines = [
        "# Phase 2 baseline commands generated by phase2_failure_mechanisms.py",
        "# Run from repository root. Adjust CUDA device / batch size if needed.",
        "",
    ]
    for seed in args.seeds:
        for partition in PARTITIONS:
            for method in METHODS:
                model = "cluster" if method == "CAPT" else "fedavg"
                config = f"configs/trainers/{method}/vit_b16.yaml"
                out_dir = f"output/{args.dataset}/phase2_mechanisms/seed{seed}/frac{args.frac}/{method}_{partition}"
                cmd = [
                    f"$env:CUDA_VISIBLE_DEVICES='{args.gpu}'",
                    "python federated_main.py",
                    "--root DATA",
                    f"--model {model}",
                    f"--dataset {args.dataset}",
                    f"--seed {seed}",
                    "--num_users 20",
                    f"--frac {args.frac}",
                    "--lr 0.001",
                    "--csc True",
                    "--gamma 1",
                    f"--trainer {method}",
                    f"--round {args.rounds}",
                    f"--partition {partition}",
                    "--beta 1.0",
                    "--n_ctx 4",
                    f"--dataset-config-file configs/datasets/{args.dataset}.yaml",
                    f"--config-file {config}",
                    f"--output-dir {out_dir}",
                    "--imb_factor 0.01",
                    "--imb_type exp",
                    "--ctx_init False",
                    "--train_batch_size 32",
                    "--test_batch_size 64",
                    "--global_eval_interval 5",
                    "--num_classes 100",
                    "--n_general 1",
                    "--head_client_ratio 0.8",
                    "--tail_client_ratio 0.2",
                    "--head_class_ratio 0.8",
                    "--tail_class_ratio 0.2",
                    "--specialization_lambda 1.0",
                    "--intra_group_alpha 0.3",
                    "--head_leakage_scale 3.0",
                    "DATALOADER.NUM_WORKERS 4",
                ]
                lines.append(" `\n  ".join(cmd))
                lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    set_plot_style()
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    run_dirs = discover_runs(args)
    if not run_dirs:
        raise SystemExit(
            "No matching run directories found. Use --write-commands to generate baseline commands."
        )

    summary_rows = []
    class_rows = []
    trajectory_rows = []
    for run_dir in run_dirs:
        try:
            summary, classes, trajectory = analyze_run(run_dir, args)
        except FileNotFoundError as exc:
            print(f"Skip {run_dir}: missing {exc.filename}")
            continue
        summary_rows.append(summary)
        class_rows.extend(classes)
        trajectory_rows.extend(trajectory)

    write_csv(output_dir / "phase2_run_summary.csv", summary_rows)
    write_csv(output_dir / "phase2_tail_class_mechanisms.csv", class_rows)
    write_csv(output_dir / "phase2_tail_trajectories.csv", trajectory_rows)
    plot_phase2_overview(summary_rows, class_rows, trajectory_rows, output_dir / "figure2_failure_mechanisms.png")

    pairwise_files = find_pairwise_inputs(run_dirs)
    if not pairwise_files:
        write_pairwise_note(output_dir / "phase2_required_pairwise_logging.md")
    else:
        with (output_dir / "pairwise_input_files.json").open("w", encoding="utf-8") as f:
            json.dump([str(p) for p in pairwise_files], f, indent=2)

    if args.write_commands:
        write_run_commands(args, Path(args.write_commands))

    print(f"Analyzed {len(summary_rows)} runs.")
    print(f"Saved Phase 2 analysis to {output_dir}")
    print(f"- {output_dir / 'phase2_run_summary.csv'}")
    print(f"- {output_dir / 'phase2_tail_class_mechanisms.csv'}")
    print(f"- {output_dir / 'figure2_failure_mechanisms.png'}")
    if not pairwise_files:
        print(f"- {output_dir / 'phase2_required_pairwise_logging.md'}")


if __name__ == "__main__":
    main()
