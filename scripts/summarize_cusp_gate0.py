#!/usr/bin/env python
"""Gate 0 summary for Round-1 CUSP using existing Experiment-D output."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path


REQUIRED_PARTITIONS = ("client-longtail", "noniid-labeldir-fine")
REQUIRED_ROUNDS = (5, 10, 20)
REQUIRED_SEED = 42
REQUIRED_CONFIG = {"num_users": 30, "frac": 1.0, "local_epochs": 3}

EVENT_FIELDS = [
    "partition", "seed", "communication_round", "class_id", "class_group", "global_count",
    "num_support_clients", "support_fedavg_weight", "acc_before", "acc_support_actual",
    "acc_support_normalized", "acc_all", "gain_support_actual", "gain_support_normalized",
    "gain_all", "offset_gap", "renorm_gain_gap", "eligible_positive_support", "reversal",
]

ROUND_FIELDS = [
    "partition", "seed", "communication_round", "tail_event_count", "eligible_count",
    "eligible_rate", "mean_gain_support_actual", "median_gain_support_actual",
    "mean_gain_all", "median_gain_all", "eligible_mean_offset_gap",
    "eligible_median_offset_gap", "eligible_mean_renorm_gain_gap",
    "eligible_median_renorm_gain_gap", "pooled_retained_gain", "reversal_count",
    "reversal_rate",
]


def read_csv(path: Path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def finite_float(value, label, errors):
    try:
        out = float(value)
    except Exception:
        errors.append(f"non-numeric {label}: {value!r}")
        return math.nan
    if not math.isfinite(out):
        errors.append(f"non-finite {label}: {value!r}")
    return out


def mean(values):
    values = [float(x) for x in values if math.isfinite(float(x))]
    return sum(values) / len(values) if values else math.nan


def median(values):
    values = sorted(float(x) for x in values if math.isfinite(float(x)))
    if not values:
        return math.nan
    middle = len(values) // 2
    return values[middle] if len(values) % 2 else (values[middle - 1] + values[middle]) / 2


def dynamic_groups(summary):
    counts = [int(float(x)) for x in summary["global_class_counts"]]
    tail_ids = {int(x) for x in summary["tail_classes"]}
    head_ids = {int(x) for x in summary["head_classes"]}
    all_ids = set(range(len(counts)))
    return counts, head_ids, tail_ids, all_ids


def load_run(run_dir: Path, errors: list[str]):
    summary_path = run_dir / "partition_summary.json"
    events_path = run_dir / "experiment_d" / "experiment_d_per_class.csv"
    if not summary_path.exists():
        errors.append(f"missing partition_summary.json: {run_dir}")
        return None, []
    if not events_path.exists():
        errors.append(f"missing experiment_d_per_class.csv: {run_dir}")
        return None, []
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    partition = summary.get("partition")
    if partition not in REQUIRED_PARTITIONS:
        errors.append(f"unexpected partition in {summary_path}: {partition}")
    if int(summary.get("seed", -1)) != REQUIRED_SEED:
        errors.append(f"seed must be 42 in {summary_path}")
    if int(summary.get("num_clients", -1)) != REQUIRED_CONFIG["num_users"]:
        errors.append(f"num_clients must be 30 in {summary_path}")
    try:
        counts, head_ids, tail_ids, all_ids = dynamic_groups(summary)
        if not tail_ids:
            errors.append(f"tail_classes empty in {summary_path}")
        if head_ids & tail_ids or head_ids | tail_ids != all_ids:
            errors.append(f"head/tail classes must cover all classes exactly once in {summary_path}")
    except Exception as exc:
        errors.append(f"invalid head/tail metadata in {summary_path}: {exc}")
        return summary, []

    output = []
    seen = Counter()
    required_columns = {
        "partition", "seed", "communication_round", "num_users", "frac", "local_epochs",
        "class_id", "class_group", "global_class_count", "num_support_clients",
        "support_fedavg_weight", "acc_before", "acc_support_actual",
        "acc_support_normalized", "acc_all", "gain_support_actual",
        "gain_support_normalized", "gain_all", "offset_gap",
    }
    rows = read_csv(events_path)
    if rows and required_columns - set(rows[0].keys()):
        errors.append(f"{events_path} missing columns: {sorted(required_columns - set(rows[0].keys()))}")
        return summary, []
    for raw in rows:
        row_errors = []
        row_partition = raw.get("partition")
        seed = int(finite_float(raw.get("seed"), "seed", row_errors))
        comm_round = int(finite_float(raw.get("communication_round"), "communication_round", row_errors))
        class_id = int(finite_float(raw.get("class_id"), "class_id", row_errors))
        if row_partition != partition:
            row_errors.append(f"row partition mismatch: {row_partition} != {partition}")
        if seed != REQUIRED_SEED:
            row_errors.append(f"row seed must be 42: {seed}")
        if int(finite_float(raw.get("num_users"), "num_users", row_errors)) != REQUIRED_CONFIG["num_users"]:
            row_errors.append("row num_users must be 30")
        if abs(finite_float(raw.get("frac"), "frac", row_errors) - REQUIRED_CONFIG["frac"]) > 1e-12:
            row_errors.append("row frac must be 1.0")
        if int(finite_float(raw.get("local_epochs"), "local_epochs", row_errors)) != REQUIRED_CONFIG["local_epochs"]:
            row_errors.append("row local_epochs must be 3")
        if comm_round not in REQUIRED_ROUNDS:
            row_errors.append(f"unexpected diagnostic round: {comm_round}")
        expected_group = "tail" if class_id in tail_ids else "head" if class_id in head_ids else None
        if expected_group is None:
            row_errors.append(f"unknown class id: {class_id}")
        if raw.get("class_group") != expected_group:
            row_errors.append(f"class_group mismatch for class {class_id}")
        numeric = {name: finite_float(raw.get(name), name, row_errors) for name in [
            "global_class_count", "num_support_clients", "support_fedavg_weight",
            "acc_before", "acc_support_actual", "acc_support_normalized", "acc_all",
            "gain_support_actual", "gain_support_normalized", "gain_all", "offset_gap",
        ]}
        if row_errors:
            errors.extend(f"{events_path}: {message}" for message in row_errors)
            continue
        if not math.isclose(numeric["gain_support_actual"], numeric["acc_support_actual"] - numeric["acc_before"], abs_tol=1e-8):
            errors.append(f"gain_support_actual semantic mismatch: {events_path} class={class_id}")
        if not math.isclose(numeric["gain_all"], numeric["acc_all"] - numeric["acc_before"], abs_tol=1e-8):
            errors.append(f"gain_all semantic mismatch: {events_path} class={class_id}")
        if not math.isclose(numeric["offset_gap"], numeric["gain_support_actual"] - numeric["gain_all"], abs_tol=1e-8):
            errors.append(f"offset_gap semantic mismatch: {events_path} class={class_id}")
        key = (partition, seed, comm_round, class_id)
        seen[key] += 1
        if expected_group == "tail":
            output.append({
                "partition": partition,
                "seed": seed,
                "communication_round": comm_round,
                "class_id": class_id,
                "class_group": expected_group,
                "global_count": numeric["global_class_count"],
                "num_support_clients": int(numeric["num_support_clients"]),
                "support_fedavg_weight": numeric["support_fedavg_weight"],
                "acc_before": numeric["acc_before"],
                "acc_support_actual": numeric["acc_support_actual"],
                "acc_support_normalized": numeric["acc_support_normalized"],
                "acc_all": numeric["acc_all"],
                "gain_support_actual": numeric["gain_support_actual"],
                "gain_support_normalized": numeric["gain_support_normalized"],
                "gain_all": numeric["gain_all"],
                "offset_gap": numeric["offset_gap"],
                "renorm_gain_gap": numeric["gain_support_normalized"] - numeric["gain_support_actual"],
                "eligible_positive_support": numeric["gain_support_actual"] > 0,
                "reversal": numeric["gain_support_actual"] > 0 and numeric["gain_all"] < 0,
            })
    for key, count in seen.items():
        if count > 1:
            errors.append(f"duplicate class event: {key}")
    for required_round in REQUIRED_ROUNDS:
        observed = {event["class_id"] for event in output if event["communication_round"] == required_round}
        if observed != tail_ids:
            errors.append(
                f"{partition} round {required_round} tail classes incomplete: "
                f"missing={sorted(tail_ids - observed)} extra={sorted(observed - tail_ids)}"
            )
    return summary, output


def summarize_rounds(events):
    grouped = defaultdict(list)
    for event in events:
        grouped[(event["partition"], event["seed"], event["communication_round"])].append(event)
    rows = []
    for key, group in sorted(grouped.items()):
        eligible = [row for row in group if row["eligible_positive_support"]]
        support_sum = sum(row["gain_support_actual"] for row in eligible)
        rows.append({
            "partition": key[0],
            "seed": key[1],
            "communication_round": key[2],
            "tail_event_count": len(group),
            "eligible_count": len(eligible),
            "eligible_rate": len(eligible) / len(group) if group else math.nan,
            "mean_gain_support_actual": mean(row["gain_support_actual"] for row in group),
            "median_gain_support_actual": median(row["gain_support_actual"] for row in group),
            "mean_gain_all": mean(row["gain_all"] for row in group),
            "median_gain_all": median(row["gain_all"] for row in group),
            "eligible_mean_offset_gap": mean(row["offset_gap"] for row in eligible),
            "eligible_median_offset_gap": median(row["offset_gap"] for row in eligible),
            "eligible_mean_renorm_gain_gap": mean(row["renorm_gain_gap"] for row in eligible),
            "eligible_median_renorm_gain_gap": median(row["renorm_gain_gap"] for row in eligible),
            "pooled_retained_gain": sum(row["gain_all"] for row in eligible) / support_sum if support_sum > 0 else math.nan,
            "reversal_count": sum(bool(row["reversal"]) for row in eligible),
            "reversal_rate": mean(float(row["reversal"]) for row in eligible),
        })
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dirs", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    errors, metadata, events = [], [], []
    if len(args.run_dirs) != 2:
        errors.append("Gate 0 requires exactly two run dirs")
    for run_dir in args.run_dirs:
        summary, run_events = load_run(run_dir, errors)
        if summary is not None:
            metadata.append(summary)
        events.extend(run_events)

    partitions = Counter(item.get("partition") for item in metadata)
    for partition in REQUIRED_PARTITIONS:
        if partitions[partition] != 1:
            errors.append(f"Gate 0 requires exactly one {partition} run")
    if len(metadata) == 2:
        for key in ("global_class_counts", "tail_classes", "head_classes"):
            values = {json.dumps(item.get(key), sort_keys=True) for item in metadata}
            if len(values) != 1:
                errors.append(f"paired runs disagree on {key}")
        fingerprints = [item.get("global_lt_fingerprint") for item in metadata if item.get("global_lt_fingerprint")]
        if fingerprints and len(set(fingerprints)) != 1:
            errors.append("paired runs disagree on Global-LT fingerprint")

    round_rows = summarize_rounds(events)
    write_csv(args.output_dir / "cusp_gate0_events.csv", events, EVENT_FIELDS)
    write_csv(args.output_dir / "cusp_gate0_by_partition_round.csv", round_rows, ROUND_FIELDS)

    complete = not errors
    clientlt = [row for row in round_rows if row["partition"] == "client-longtail"]
    clientlt_by_round = {int(row["communication_round"]): row for row in clientlt}
    all_three_eligible = all(clientlt_by_round.get(round_id, {}).get("eligible_count", 0) > 0 for round_id in REQUIRED_ROUNDS)
    positive_offset_rounds = sum(
        clientlt_by_round.get(round_id, {}).get("eligible_mean_offset_gap", math.nan) > 0
        for round_id in REQUIRED_ROUNDS
    )
    eligible_events = [
        event for event in events
        if event["partition"] == "client-longtail" and event["eligible_positive_support"]
    ]
    support_sum = sum(event["gain_support_actual"] for event in eligible_events)
    pooled_retained = sum(event["gain_all"] for event in eligible_events) / support_sum if support_sum > 0 else math.nan
    gate_pass = bool(complete and all_three_eligible and positive_offset_rounds >= 2 and pooled_retained < 1)
    summary = {
        "schema_version": "cusp_round1_v1",
        "scientific_data_complete": complete,
        "gate": "PASS" if gate_pass else ("INCOMPLETE" if not complete else "FAIL_GATE0"),
        "failure_reasons": errors,
        "clientlt_all_three_rounds_have_eligible_tail": bool(all_three_eligible),
        "clientlt_positive_eligible_offset_rounds": int(positive_offset_rounds),
        "clientlt_pooled_retained_gain": None if not math.isfinite(pooled_retained) else pooled_retained,
        "round_rows": round_rows,
    }
    (args.output_dir / "cusp_gate0_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (args.output_dir / "cusp_gate0_report.md").write_text(
        "# CUSP Gate 0\n\n"
        f"Result: **{summary['gate']}**\n\n"
        f"Scientific data complete: `{str(complete).lower()}`\n\n"
        "Gate uses eligible tail events (`gain_support_actual > 0`) for offset/interference statistics.\n",
        encoding="utf-8",
    )
    if not complete:
        for reason in errors:
            print(f"INCOMPLETE: {reason}", file=sys.stderr)
        raise SystemExit(2)


if __name__ == "__main__":
    main()
