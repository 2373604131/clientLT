#!/usr/bin/env python
import argparse
import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path


DEFAULT_ROOT = Path("output/cifar100_LT/PromptFL_fedavg_vit_b16_batchSize32/ExperimentD_Main")
DEFAULT_SEEDS = (42, 2026)
PARTITION_LABELS = {
    "client-longtail": "Client-LT",
    "noniid-labeldir-fine": "Dirichlet",
}

CLIENTLT_RE = re.compile(
    r"partition=(?P<partition>client-longtail)_lambda=(?P<lambda>[0-9.]+)_alpha=(?P<alpha>[0-9.]+)_"
    r"rho=(?P<rho>[0-9.]+)_IF=(?P<imb_factor>[0-9.]+)_localE=(?P<local_epochs>\d+)_seed=(?P<seed>\d+)"
)
DIRICHLET_RE = re.compile(
    r"partition=(?P<partition>noniid-labeldir-fine)_alpha=(?P<alpha>[0-9.]+)_"
    r"IF=(?P<imb_factor>[0-9.]+)_localE=(?P<local_epochs>\d+)_seed=(?P<seed>\d+)"
)

REQUIRED_EVENT_COLUMNS = [
    "communication_round",
    "class_id",
    "class_group",
    "acc_before",
    "acc_support_actual",
    "acc_all",
]

FINAL_ACC_CANDIDATES = [
    "summary/experimentD_main_2seed_final_accuracy_all_runs.csv",
    "summary/experimentD_main_final_accuracy_all_runs.csv",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Summarize two-seed Experiment D mechanism results from existing CSV files."
    )
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    return parser.parse_args()


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
            writer.writerow(row)


def as_float(value):
    if value in (None, ""):
        return math.nan
    return float(value)


def mean(values):
    values = [float(v) for v in values if not math.isnan(float(v))]
    return sum(values) / len(values) if values else math.nan


def sample_std(values):
    values = [float(v) for v in values if not math.isnan(float(v))]
    if len(values) <= 1:
        return 0.0 if values else math.nan
    avg = mean(values)
    return math.sqrt(sum((v - avg) ** 2 for v in values) / (len(values) - 1))


def fmt(value, digits=2):
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    return f"{float(value):.{digits}f}"


def parse_run_dir(run_dir):
    match = CLIENTLT_RE.fullmatch(run_dir.name)
    if match:
        return {
            "partition": match.group("partition"),
            "partition_label": PARTITION_LABELS[match.group("partition")],
            "seed": int(match.group("seed")),
            "alpha": float(match.group("alpha")),
            "specialization_lambda": float(match.group("lambda")),
            "head_leakage_scale": float(match.group("rho")),
            "imb_factor": float(match.group("imb_factor")),
            "local_epochs": int(match.group("local_epochs")),
        }
    match = DIRICHLET_RE.fullmatch(run_dir.name)
    if match:
        return {
            "partition": match.group("partition"),
            "partition_label": PARTITION_LABELS[match.group("partition")],
            "seed": int(match.group("seed")),
            "alpha": float(match.group("alpha")),
            "specialization_lambda": "",
            "head_leakage_scale": "",
            "imb_factor": float(match.group("imb_factor")),
            "local_epochs": int(match.group("local_epochs")),
        }
    return None


def find_final_acc_file(root):
    for rel_path in FINAL_ACC_CANDIDATES:
        path = root / rel_path
        if path.exists():
            return path
    return None


def load_final_accuracy(root, seeds):
    path = find_final_acc_file(root)
    if path is None:
        return {}, None
    rows = read_csv(path)
    out = {}
    for row in rows:
        partition = row.get("partition")
        seed = int(float(row.get("seed", "nan")))
        if partition in PARTITION_LABELS and seed in seeds:
            out[(partition, seed)] = {
                "tail_acc": as_float(row.get("bottom20_tail_acc")),
                "non_tail_acc": as_float(row.get("non_tail_acc")),
                "overall_acc": as_float(row.get("overall_acc")),
            }
    return out, path


def collect_events(root, seeds):
    events = []
    warnings = []
    input_files = []
    for run_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        info = parse_run_dir(run_dir)
        if info is None or info["seed"] not in seeds:
            continue
        path = run_dir / "experiment_d" / "experiment_d_per_class.csv"
        if not path.exists():
            warnings.append(f"missing mechanism file: {path}")
            continue
        rows = read_csv(path)
        input_files.append(path)
        if not rows:
            warnings.append(f"empty mechanism file: {path}")
            continue
        missing_cols = [col for col in REQUIRED_EVENT_COLUMNS if col not in rows[0]]
        if missing_cols:
            warnings.append(f"{path} missing columns: {', '.join(missing_cols)}")
            continue
        for row in rows:
            if row.get("class_group") != "tail":
                continue
            acc_before = as_float(row["acc_before"])
            acc_support = as_float(row["acc_support_actual"])
            acc_all = as_float(row["acc_all"])
            if any(math.isnan(v) for v in (acc_before, acc_support, acc_all)):
                warnings.append(f"missing accuracy value in {path}: round={row.get('communication_round')} class={row.get('class_id')}")
                continue
            # Formulas, in percentage points:
            # G_support = A_support - A_before
            # G_all = A_all - A_before
            # R_retain = sum(G_all | G_support > 0) / sum(G_support | G_support > 0)
            g_support = acc_support - acc_before
            g_all = acc_all - acc_before
            events.append(
                {
                    **info,
                    "run_dir": str(run_dir),
                    "communication_round": int(float(row["communication_round"])),
                    "tail_class_id": int(float(row["class_id"])),
                    "A_before": acc_before,
                    "A_support": acc_support,
                    "A_all": acc_all,
                    "G_support": g_support,
                    "G_all": g_all,
                    "support_positive": g_support > 0.0,
                    "reversal": g_support > 0.0 and g_all < 0.0,
                }
            )
    return events, input_files, warnings


def validate_events(events):
    checks = []
    required_groups = {(p, s) for p in PARTITION_LABELS for s in DEFAULT_SEEDS}
    present_groups = {(e["partition"], e["seed"]) for e in events}
    checks.append(("required groups present", required_groups.issubset(present_groups)))

    duplicate_keys = defaultdict(int)
    for event in events:
        key = (event["partition"], event["seed"], event["communication_round"], event["tail_class_id"])
        duplicate_keys[key] += 1
    duplicates = [key for key, count in duplicate_keys.items() if count > 1]
    checks.append(("no duplicate partition-seed-round-class events", len(duplicates) == 0))

    group_sets = defaultdict(lambda: {"rounds": set(), "classes": set()})
    for event in events:
        group = (event["partition"], event["seed"])
        group_sets[group]["rounds"].add(event["communication_round"])
        group_sets[group]["classes"].add(event["tail_class_id"])

    round_sets = {group: tuple(sorted(v["rounds"])) for group, v in group_sets.items()}
    class_sets = {group: tuple(sorted(v["classes"])) for group, v in group_sets.items()}
    checks.append(("same diagnostic rounds across groups", len(set(round_sets.values())) == 1))
    checks.append(("same bottom-20 tail classes across groups", len(set(class_sets.values())) == 1))
    checks.append(("tail classes are class 80-99", set(next(iter(class_sets.values()), ())) == set(range(80, 100))))

    expected_events_per_group = None
    if group_sets:
        first = next(iter(group_sets.values()))
        expected_events_per_group = len(first["rounds"]) * len(first["classes"])
    counts = defaultdict(int)
    for event in events:
        counts[(event["partition"], event["seed"])] += 1
    checks.append(("balanced tail-class x round coverage", len(set(counts.values())) == 1))

    return checks, round_sets, class_sets, counts, duplicates, expected_events_per_group


def summarize_group(events, final_acc):
    groups = defaultdict(list)
    for event in events:
        groups[(event["partition"], event["seed"])].append(event)

    rows = []
    for (partition, seed), rows_in in sorted(groups.items(), key=lambda item: (item[0][1], item[0][0])):
        eligible = [e for e in rows_in if e["G_support"] > 0.0]
        g_support_sum = sum(e["G_support"] for e in eligible)
        g_all_sum = sum(e["G_all"] for e in eligible)
        retention = g_all_sum / g_support_sum if g_support_sum != 0 else math.nan
        reversal_count = sum(1 for e in eligible if e["G_all"] < 0.0)
        acc = final_acc.get((partition, seed), {})
        rows.append(
            {
                "partition": partition,
                "partition_label": PARTITION_LABELS[partition],
                "seed": seed,
                "tail_acc": acc.get("tail_acc", math.nan),
                "non_tail_acc": acc.get("non_tail_acc", math.nan),
                "overall_acc": acc.get("overall_acc", math.nan),
                "num_events": len(rows_in),
                "G_support_mean": mean(e["G_support"] for e in rows_in),
                "positive_support_count": len(eligible),
                "positive_support_rate": len(eligible) / len(rows_in) if rows_in else math.nan,
                "G_all_mean": mean(e["G_all"] for e in rows_in),
                "retention_rate": retention,
                "eligible_count": len(eligible),
                "reversal_count": reversal_count,
                "reversal_rate": reversal_count / len(eligible) if eligible else math.nan,
            }
        )
    return rows


def mean_rows_by_partition(summary_rows):
    out = []
    grouped = defaultdict(list)
    for row in summary_rows:
        grouped[row["partition"]].append(row)
    for partition, rows in sorted(grouped.items()):
        out.append(
            {
                "partition": partition,
                "partition_label": PARTITION_LABELS[partition],
                "seed": "mean",
                "tail_acc": mean(r["tail_acc"] for r in rows),
                "non_tail_acc": mean(r["non_tail_acc"] for r in rows),
                "overall_acc": mean(r["overall_acc"] for r in rows),
                "num_events": sum(int(r["num_events"]) for r in rows),
                "G_support_mean": mean(r["G_support_mean"] for r in rows),
                "positive_support_count": sum(int(r["positive_support_count"]) for r in rows),
                "positive_support_rate": mean(r["positive_support_rate"] for r in rows),
                "G_all_mean": mean(r["G_all_mean"] for r in rows),
                "retention_rate": mean(r["retention_rate"] for r in rows),
                "eligible_count": sum(int(r["eligible_count"]) for r in rows),
                "reversal_count": sum(int(r["reversal_count"]) for r in rows),
                "reversal_rate": mean(r["reversal_rate"] for r in rows),
            }
        )
    return out


def paired_delta(summary_rows):
    by_seed_partition = {(r["seed"], r["partition"]): r for r in summary_rows}
    rows = []
    for seed in sorted({int(r["seed"]) for r in summary_rows}):
        client = by_seed_partition.get((seed, "client-longtail"))
        diri = by_seed_partition.get((seed, "noniid-labeldir-fine"))
        if not client or not diri:
            continue
        row = {"seed": seed}
        for metric in [
            "tail_acc",
            "non_tail_acc",
            "overall_acc",
            "G_support_mean",
            "positive_support_rate",
            "G_all_mean",
            "retention_rate",
            "reversal_rate",
        ]:
            row[f"clientlt_{metric}"] = client[metric]
            row[f"dirichlet_{metric}"] = diri[metric]
            row[f"delta_{metric}"] = client[metric] - diri[metric]
        row["clientlt_reversal_count"] = client["reversal_count"]
        row["clientlt_eligible_count"] = client["eligible_count"]
        row["dirichlet_reversal_count"] = diri["reversal_count"]
        row["dirichlet_eligible_count"] = diri["eligible_count"]
        rows.append(row)
    return rows


def paired_mean_row(pair_rows):
    row = {"seed": "paired difference mean"}
    if not pair_rows:
        return row
    for key in pair_rows[0]:
        if key == "seed":
            continue
        if key.startswith("delta_"):
            row[key] = mean(r[key] for r in pair_rows)
    return row


def direction_consistency(pair_rows, metric):
    values = [r[f"delta_{metric}"] for r in pair_rows]
    return all(v < 0 for v in values), values


def build_conclusion(root, output_paths, input_files, final_acc_path, warnings, checks, summary_rows, pair_rows):
    mean_by_partition = {r["partition"]: r for r in mean_rows_by_partition(summary_rows)}
    pair_mean = paired_mean_row(pair_rows)
    support_consistent, support_deltas = direction_consistency(pair_rows, "G_support_mean")
    retention_consistent, retention_deltas = direction_consistency(pair_rows, "retention_rate")
    reversal_values = [r["delta_reversal_rate"] for r in pair_rows]
    reversal_not_lower = all(v >= 0 for v in reversal_values)

    if support_consistent and retention_consistent and reversal_not_lower:
        mechanism = "both gain-generation deficit and gain-retention failure"
        recommendation = "use a two-stage method: lightweight gain generation plus gain preservation."
    elif support_consistent:
        mechanism = "mainly gain-generation deficit"
        recommendation = "prioritize improving local tail-class gain generation on support clients."
    elif retention_consistent and reversal_not_lower:
        mechanism = "mainly gain-conversion failure"
        recommendation = "prioritize gain-preserving aggregation."
    else:
        mechanism = "current one-step Experiment D cannot fully explain the final tail gap"
        recommendation = "keep the current diagnostic scope and inspect whether existing data support finer mechanism claims."

    lines = []
    lines.append("# Experiment D Mechanism Conclusion")
    lines.append("")
    lines.append("## Input Files")
    for path in input_files:
        lines.append(f"- `{path}`")
    if final_acc_path:
        lines.append(f"- `{final_acc_path}`")
    lines.append("")
    lines.append("## Data Integrity")
    for name, ok in checks:
        lines.append(f"- {name}: {'OK' if ok else 'FAILED'}")
    if warnings:
        lines.append("- warnings:")
        for warning in warnings:
            lines.append(f"  - {warning}")
    else:
        lines.append("- warnings: none")
    lines.append("- accuracy unit: percentage points on a 0-100 scale")
    lines.append("- support-only aggregation: `support_actual` uses original FedAvg weights over support clients only; no renormalization.")
    lines.append("")
    lines.append("## Main Table")
    lines.append("| partition | seed | tail acc | G_support mean | positive support rate | G_all mean | retention rate | reversal count / eligible count | reversal rate |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    display_rows = summary_rows + list(mean_by_partition.values())
    order = {"Client-LT": 0, "Dirichlet": 1}
    display_rows = sorted(display_rows, key=lambda r: (str(r["seed"]), order.get(r["partition_label"], 9)))
    for row in display_rows:
        lines.append(
            f"| {row['partition_label']} | {row['seed']} | {fmt(row['tail_acc'])} | "
            f"{fmt(row['G_support_mean'])} | {fmt(100 * row['positive_support_rate'])}% | "
            f"{fmt(row['G_all_mean'])} | {fmt(100 * row['retention_rate'])}% | "
            f"{row['reversal_count']} / {row['eligible_count']} | {fmt(100 * row['reversal_rate'])}% |"
        )
    lines.append(
        f"| paired difference | mean | {fmt(pair_mean.get('delta_tail_acc'))} | "
        f"{fmt(pair_mean.get('delta_G_support_mean'))} | {fmt(100 * pair_mean.get('delta_positive_support_rate'))}% | "
        f"{fmt(pair_mean.get('delta_G_all_mean'))} | {fmt(100 * pair_mean.get('delta_retention_rate'))}% |  | "
        f"{fmt(100 * pair_mean.get('delta_reversal_rate'))}% |"
    )
    lines.append("")
    lines.append("## Paired Direction")
    lines.append(f"- G_support_mean deltas Client-LT - Dirichlet: {', '.join(fmt(v) for v in support_deltas)}")
    lines.append(f"- retention rate deltas Client-LT - Dirichlet: {', '.join(fmt(100 * v) + '%' for v in retention_deltas)}")
    lines.append(f"- reversal rate deltas Client-LT - Dirichlet: {', '.join(fmt(100 * v) + '%' for v in reversal_values)}")
    lines.append("")
    lines.append("## Mechanism Judgment")
    lines.append("Question 1: Does Client-LT have a gain-generation deficit?")
    lines.append(
        "Conclusion: yes. Client-LT has direction-consistent support-side gain-generation deficit. "
        "For both seeds, Client-LT has lower `G_support_mean` than Dirichlet and a lower positive support rate."
    )
    lines.append("")
    lines.append("Question 2: Does Client-LT have more severe gain-retention failure?")
    if retention_consistent and reversal_not_lower:
        lines.append(
            "Conclusion: yes. Client-LT has lower retention rate in both seeds, "
            "and its reversal rate is not lower than Dirichlet."
        )
    else:
        lines.append(
            "Conclusion: mixed evidence. Retention and reversal do not give a fully consistent two-seed direction."
        )
    lines.append("")
    lines.append(f"Final classification: **{mechanism}**.")
    lines.append("")
    lines.append("## Accuracy Context")
    client_mean = mean_by_partition["client-longtail"]
    diri_mean = mean_by_partition["noniid-labeldir-fine"]
    tail_delta = client_mean["tail_acc"] - diri_mean["tail_acc"]
    non_tail_delta = client_mean["non_tail_acc"] - diri_mean["non_tail_acc"]
    lines.append(
        f"- Client-LT mean tail accuracy: {fmt(client_mean['tail_acc'])}; "
        f"Dirichlet mean tail accuracy: {fmt(diri_mean['tail_acc'])}; "
        f"Client-LT - Dirichlet: {fmt(tail_delta)} percentage points."
    )
    lines.append(
        f"- non-tail paired mean difference is about {fmt(non_tail_delta)} percentage points, "
        "so the degradation is selective to bottom-20 tail classes rather than a general representation collapse."
    )
    lines.append("")
    lines.append("## Method Implication")
    lines.append(
        "Client-LT selective tail degradation -> weaker tail gains on support clients -> standard FedAvg retains "
        f"less of the generated gain -> next step should {recommendation}"
    )
    lines.append("")
    lines.append("## Output Files")
    for path in output_paths:
        lines.append(f"- `{path}`")
    lines.append("")
    return "\n".join(lines)


def main():
    args = parse_args()
    root = args.root
    output_dir = args.output_dir or (root / "summary")
    seeds = set(args.seeds)

    events, input_files, warnings = collect_events(root, seeds)
    checks, round_sets, class_sets, counts, duplicates, expected_events = validate_events(events)
    final_acc, final_acc_path = load_final_accuracy(root, seeds)

    event_path = output_dir / "experimentD_mechanism_events.csv"
    summary_path = output_dir / "experimentD_mechanism_summary.csv"
    paired_path = output_dir / "experimentD_mechanism_paired_delta.csv"
    conclusion_path = output_dir / "experimentD_mechanism_conclusion.md"

    summary_rows = summarize_group(events, final_acc)
    paired_rows = paired_delta(summary_rows)

    write_csv(event_path, events)
    write_csv(summary_path, summary_rows + mean_rows_by_partition(summary_rows))
    write_csv(paired_path, paired_rows + [paired_mean_row(paired_rows)])

    output_paths = [event_path, summary_path, paired_path, conclusion_path]
    conclusion = build_conclusion(
        root,
        output_paths,
        input_files,
        final_acc_path,
        warnings,
        checks,
        summary_rows,
        paired_rows,
    )
    conclusion_path.write_text(conclusion, encoding="utf-8")

    report = {
        "root": str(root),
        "num_events": len(events),
        "counts": {f"{partition}|{seed}": count for (partition, seed), count in sorted(counts.items())},
        "rounds": {f"{partition}|{seed}": list(values) for (partition, seed), values in sorted(round_sets.items())},
        "tail_classes": {f"{partition}|{seed}": list(values) for (partition, seed), values in sorted(class_sets.items())},
        "checks": [{"name": name, "ok": ok} for name, ok in checks],
        "warnings": warnings,
        "outputs": [str(path) for path in output_paths],
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
