#!/usr/bin/env python
"""Aggregate frozen Boundary Gate diagnostics and candidate performance.

Each topology × seed is reduced to one value before any cross-seed average.
This deliberately prevents runs with more fragile edges from receiving more
weight in a formal multi-seed comparison.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Mapping

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.cusp_minimal import write_csv, write_json


EDGE_METRICS = (
    "local_audit_gain",
    "gain_support_normalized",
    "gain_support_actual",
    "gain_all_fedavg",
    "dilution",
    "interference",
)
PERFORMANCE_METRICS = ("overall_acc", "non_tail_acc", "tail_acc")


def as_float(value) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return math.nan
    return parsed if math.isfinite(parsed) else math.nan


def mean(values: Iterable[float]) -> float:
    valid = [float(value) for value in values if math.isfinite(float(value))]
    return float(sum(valid) / len(valid)) if valid else math.nan


def is_true(value) -> bool:
    return value is True or str(value).strip().lower() == "true"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def topology_key(manifest: Mapping) -> str:
    """Prefer an explicit future topology label, with partition as fallback."""
    return str(manifest.get("topology", manifest.get("partition", "")))


def summarize_gate(gate_dir: Path) -> tuple[dict, dict, list[dict]]:
    manifest = load_json(gate_dir / "candidate_manifest.json")
    summary = load_json(gate_dir / "gate_summary.json")
    diagnostics = list(csv.DictReader((gate_dir / "edge_diagnostics.csv").open(encoding="utf-8")))
    fragile = [row for row in diagnostics if is_true(row.get("fragile_selected"))]
    gate_row = {
        "gate_dir": str(gate_dir),
        "topology": topology_key(manifest),
        "partition": manifest.get("partition", ""),
        "seed": manifest.get("seed", ""),
        "round": manifest.get("round", ""),
        "fragile_edge_count": len(fragile),
        "repair_accepted": summary.get("repair", {}).get("accepted", False),
        "substantive_repair_edge_rate": summary.get("substantive_repair_edge_rate", math.nan),
        "substantive_repair_all_fragile_edges": summary.get("substantive_repair_all_fragile_edges", False),
    }
    for metric in EDGE_METRICS:
        gate_row[f"{metric}_mean"] = mean(as_float(row.get(metric)) for row in fragile)

    edge_per_gate = []
    for class_group in sorted({str(row.get("class_group", "unknown")) for row in fragile}):
        group = [row for row in fragile if str(row.get("class_group", "unknown")) == class_group]
        item = {
            "gate_dir": str(gate_dir),
            "topology": topology_key(manifest),
            "partition": manifest.get("partition", ""),
            "seed": manifest.get("seed", ""),
            "round": manifest.get("round", ""),
            "class_group": class_group,
            "fragile_edge_count": len(group),
        }
        for metric in EDGE_METRICS:
            item[metric] = mean(as_float(row.get(metric)) for row in group)
        edge_per_gate.append(item)
    return gate_row, manifest, edge_per_gate


def collapse_equal_units(rows: Iterable[Mapping], key_fields: tuple[str, ...], metric_fields: tuple[str, ...]) -> list[dict]:
    """Average duplicate runs within one topology × seed, never raw edges."""
    groups: dict[tuple, list[Mapping]] = defaultdict(list)
    for row in rows:
        groups[tuple(str(row.get(field, "")) for field in key_fields)].append(row)
    collapsed = []
    for key, items in sorted(groups.items()):
        result = {field: value for field, value in zip(key_fields, key)}
        result["contributing_gate_count"] = len(items)
        for metric in metric_fields:
            result[metric] = mean(as_float(item.get(metric)) for item in items)
        collapsed.append(result)
    return collapsed


def summarize_edge_groups(edge_per_gate: list[dict]) -> tuple[list[dict], list[dict]]:
    per_seed = collapse_equal_units(
        edge_per_gate,
        ("topology", "partition", "seed", "round", "class_group"),
        ("fragile_edge_count", *EDGE_METRICS),
    )
    cross_seed = collapse_equal_units(
        per_seed,
        ("topology", "partition", "class_group"),
        ("fragile_edge_count", *EDGE_METRICS),
    )
    for row in cross_seed:
        row["equal_weight_seed_count"] = row.pop("contributing_gate_count")
    return per_seed, cross_seed


def candidate_rows_for_gate(gate_dir: Path, manifest: Mapping) -> list[dict]:
    metrics_path = gate_dir / "candidate_metrics.csv"
    if not metrics_path.exists():
        return []
    source = list(csv.DictReader(metrics_path.open(encoding="utf-8")))
    fedavg = next((row for row in source if row.get("method") == "fedavg"), None)
    if fedavg is None:
        raise RuntimeError(f"candidate_metrics.csv has no FedAvg reference: {metrics_path}")
    output = []
    for row in source:
        item = {
            "gate_dir": str(gate_dir),
            "topology": topology_key(manifest),
            "partition": manifest.get("partition", ""),
            "seed": manifest.get("seed", ""),
            "round": manifest.get("round", ""),
            "candidate_id": row.get("candidate_id", ""),
            "method": row.get("method", ""),
        }
        for metric in PERFORMANCE_METRICS:
            value = as_float(row.get(metric))
            item[metric] = value
            item[f"{metric}_minus_fedavg"] = value - as_float(fedavg.get(metric)) if math.isfinite(value) else math.nan
        output.append(item)
    return output


def summarize_candidate_performance(candidate_per_gate: list[dict]) -> tuple[list[dict], list[dict]]:
    metric_fields = tuple(PERFORMANCE_METRICS) + tuple(f"{metric}_minus_fedavg" for metric in PERFORMANCE_METRICS)
    per_seed = collapse_equal_units(
        candidate_per_gate,
        ("topology", "partition", "seed", "round", "candidate_id", "method"),
        metric_fields,
    )
    cross_seed = collapse_equal_units(
        per_seed,
        ("topology", "partition", "candidate_id", "method"),
        metric_fields,
    )
    for row in cross_seed:
        row["equal_weight_seed_count"] = row.pop("contributing_gate_count")
    return per_seed, cross_seed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gate-dir", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    gate_rows, edge_per_gate, candidate_per_gate = [], [], []
    for gate_dir in args.gate_dir:
        gate_row, manifest, edge_rows = summarize_gate(gate_dir)
        gate_rows.append(gate_row)
        edge_per_gate.extend(edge_rows)
        candidate_per_gate.extend(candidate_rows_for_gate(gate_dir, manifest))
    edge_per_seed, edge_cross_seed = summarize_edge_groups(edge_per_gate)
    candidate_per_seed, candidate_cross_seed = summarize_candidate_performance(candidate_per_gate)

    write_csv(args.output_dir / "boundary_gate_summary.csv", gate_rows)
    write_csv(args.output_dir / "boundary_edge_per_seed_summary.csv", edge_per_seed)
    write_csv(args.output_dir / "boundary_edge_group_summary.csv", edge_cross_seed)
    write_csv(args.output_dir / "boundary_candidate_per_seed_metrics.csv", candidate_per_seed)
    write_csv(args.output_dir / "boundary_candidate_summary.csv", candidate_cross_seed)
    write_json(args.output_dir / "boundary_gate_summary.json", {
        "gate_count": len(gate_rows),
        "gate_rows": gate_rows,
        "edge_per_seed": edge_per_seed,
        "edge_cross_seed": edge_cross_seed,
        "candidate_per_seed": candidate_per_seed,
        "candidate_cross_seed": candidate_cross_seed,
    })
    print(args.output_dir / "boundary_gate_summary.json")


if __name__ == "__main__":
    main()
