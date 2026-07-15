#!/usr/bin/env python
import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path


DEFAULT_ALL_RUNS = Path(
    "output/cifar100_LT/PromptFL_fedavg_vit_b16_batchSize32/"
    "ExperimentD_LocalEpochPilot/summary/experimentD_local_epochs_all_runs.csv"
)
DEFAULT_OUTPUT_DIR = Path(
    "output/cifar100_LT/PromptFL_fedavg_vit_b16_batchSize32/"
    "ExperimentD_LocalEpochPilot/summary/figures"
)


def parse_args():
    parser = argparse.ArgumentParser(description="Plot Experiment D local-epochs pilot results.")
    parser.add_argument("--all-runs", type=Path, default=DEFAULT_ALL_RUNS)
    parser.add_argument("--summary", type=Path, default=None, help="compatibility alias; prefer --all-runs")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def as_float(value):
    if value in (None, ""):
        return math.nan
    return float(value)


def read_rows(path):
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def mean(values):
    valid = [float(x) for x in values if not math.isnan(float(x))]
    return sum(valid) / len(valid) if valid else math.nan


def grouped_mean_series(series_by_group):
    out = {}
    for group, rows in series_by_group.items():
        by_round = defaultdict(list)
        for communication_round, value in rows:
            by_round[int(communication_round)].append(float(value))
        out[group] = sorted((round_id, mean(values)) for round_id, values in by_round.items())
    return out


def collect_metric_from_csv(all_runs, relative_path, round_field, value_fields):
    series = {field: defaultdict(list) for field in value_fields}
    for run in all_runs:
        run_dir = Path(run["run_dir"])
        path = run_dir / relative_path
        if not path.exists():
            continue
        local_epochs = int(run["local_epochs"])
        rows = read_rows(path)
        for row in rows:
            if round_field == "epoch":
                communication_round = int(float(row[round_field])) + 1
            else:
                communication_round = int(float(row[round_field]))
            for field in value_fields:
                value = as_float(row.get(field))
                if not math.isnan(value):
                    series[field][local_epochs].append((communication_round, value))
    return {field: grouped_mean_series(groups) for field, groups in series.items()}


def plot_lines(series_by_group, ylabel, output_path, title=None):
    import matplotlib.pyplot as plt

    if not any(series_by_group.values()):
        print(f"WARNING: no data for {output_path.name}; figure not created")
        return False

    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    for local_epochs, series in sorted(series_by_group.items()):
        if not series:
            continue
        xs = [x for x, _ in series]
        ys = [y for _, y in series]
        ax.plot(xs, ys, marker="o", linewidth=2.0, label=f"local_epochs={local_epochs}")
    ax.set_xlabel("Communication round")
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    return True


def plot_update_norm(norm_series, output_path):
    import matplotlib.pyplot as plt

    head = norm_series.get("mean_update_norm_head_clients", {})
    tail = norm_series.get("mean_update_norm_tail_specialists", {})
    if not any(head.values()) and not any(tail.values()):
        print(f"WARNING: no data for {output_path.name}; figure not created")
        return False

    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    for local_epochs, series in sorted(head.items()):
        if series:
            ax.plot(
                [x for x, _ in series],
                [y for _, y in series],
                marker="o",
                linewidth=2.0,
                label=f"head clients, localE={local_epochs}",
            )
    for local_epochs, series in sorted(tail.items()):
        if series:
            ax.plot(
                [x for x, _ in series],
                [y for _, y in series],
                marker="s",
                linestyle="--",
                linewidth=2.0,
                label=f"tail specialists, localE={local_epochs}",
            )
    ax.set_xlabel("Communication round")
    ax.set_ylabel("Update norm")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    return True


def plot_runtime(runtime_series, output_path):
    import matplotlib.pyplot as plt

    local = runtime_series.get("local_training_seconds", {})
    diag = runtime_series.get("experimentD_diagnostic_seconds", {})
    if not any(local.values()):
        print(f"WARNING: no data for {output_path.name}; figure not created")
        return False

    labels = []
    local_means = []
    diag_means = []
    for local_epochs in sorted(local):
        labels.append(str(local_epochs))
        local_means.append(mean([value for _, value in local[local_epochs]]))
        diag_means.append(mean([value for _, value in diag.get(local_epochs, [])]))

    x = list(range(len(labels)))
    fig, ax = plt.subplots(figsize=(5.6, 3.8))
    ax.bar([v - 0.18 for v in x], local_means, width=0.36, label="local training")
    ax.bar([v + 0.18 for v in x], diag_means, width=0.36, label="Experiment D diagnostics")
    ax.set_xlabel("Local epochs")
    ax.set_ylabel("Seconds per round")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    return True


def main():
    args = parse_args()
    all_runs_path = args.summary or args.all_runs
    if not all_runs_path.exists():
        print(f"ERROR: all-runs CSV does not exist: {all_runs_path}")
        return 1
    all_runs = read_rows(all_runs_path)
    if not all_runs:
        print(f"ERROR: all-runs CSV is empty: {all_runs_path}")
        return 1
    if "run_dir" not in all_runs[0]:
        print(
            "ERROR: plotting needs experimentD_local_epochs_all_runs.csv with a run_dir column; "
            "run the summarizer first and pass --all-runs."
        )
        return 1

    args.output_dir.mkdir(parents=True, exist_ok=True)

    round_series = collect_metric_from_csv(
        all_runs,
        Path("round_metrics.csv"),
        "epoch",
        ["overall_acc", "bottom20_tail_acc"],
    )
    diag_series = collect_metric_from_csv(
        all_runs,
        Path("experiment_d") / "experiment_d_round_summary.csv",
        "communication_round",
        [
            "mean_gain_support_actual",
            "mean_gain_support_normalized",
            "mean_gain_all",
            "mean_offset_gap",
        ],
    )
    norm_series = collect_metric_from_csv(
        all_runs,
        Path("experiment_d") / "client_update_norm_summary.csv",
        "communication_round",
        ["mean_update_norm_head_clients", "mean_update_norm_tail_specialists"],
    )
    runtime_series = collect_metric_from_csv(
        all_runs,
        Path("experiment_d") / "runtime_metrics.csv",
        "communication_round",
        [
            "local_training_seconds",
            "experimentD_diagnostic_seconds",
            "normal_global_eval_seconds",
            "round_total_seconds",
        ],
    )

    created = 0
    created += plot_lines(
        round_series.get("overall_acc", {}),
        "Overall accuracy (%)",
        args.output_dir / "experimentD_overall_accuracy_vs_round.pdf",
    )
    created += plot_lines(
        round_series.get("bottom20_tail_acc", {}),
        "Bottom-20 tail accuracy (%)",
        args.output_dir / "experimentD_tail_accuracy_vs_round.pdf",
    )
    created += plot_lines(
        diag_series.get("mean_gain_support_actual", {}),
        "Mean G_support_actual (pp)",
        args.output_dir / "experimentD_gain_support_actual_vs_round.pdf",
    )
    created += plot_lines(
        diag_series.get("mean_gain_support_normalized", {}),
        "Mean G_support_normalized (pp)",
        args.output_dir / "experimentD_gain_support_normalized_vs_round.pdf",
    )
    created += plot_lines(
        diag_series.get("mean_gain_all", {}),
        "Mean G_all (pp)",
        args.output_dir / "experimentD_gain_all_vs_round.pdf",
    )
    created += plot_lines(
        diag_series.get("mean_offset_gap", {}),
        "Mean offset gap (pp)",
        args.output_dir / "experimentD_offset_gap_vs_round.pdf",
    )
    created += plot_update_norm(
        norm_series,
        args.output_dir / "experimentD_head_tail_update_norm_vs_round.pdf",
    )
    created += plot_runtime(
        runtime_series,
        args.output_dir / "experimentD_runtime_comparison.pdf",
    )

    if created == 0:
        print("ERROR: no figures were created because required result CSVs were missing.")
        return 1
    print(f"Wrote {created} figures to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
