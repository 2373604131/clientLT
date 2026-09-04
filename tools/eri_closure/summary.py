"""Aggregate ERI closure outputs into topology, outcome, and intervention tests."""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean

import numpy as np

from utils.cusp_minimal import write_csv, write_json


def _read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _float(value, default=math.nan) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _rank(values: list[float]) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    cursor = 0
    while cursor < len(values):
        end = cursor + 1
        while end < len(values) and values[order[end]] == values[order[cursor]]:
            end += 1
        ranks[order[cursor:end]] = (cursor + end - 1) / 2.0 + 1.0
        cursor = end
    return ranks


def spearman(x: list[float], y: list[float]) -> float:
    pairs = [(a, b) for a, b in zip(x, y) if math.isfinite(a) and math.isfinite(b)]
    if len(pairs) < 3:
        return math.nan
    rx, ry = _rank([item[0] for item in pairs]), _rank([item[1] for item in pairs])
    if np.std(rx) == 0 or np.std(ry) == 0:
        return math.nan
    return float(np.corrcoef(rx, ry)[0, 1])


def _run_identity(analysis_file: Path) -> dict:
    run_dir = analysis_file.parents[2]
    dump_metadata = sorted((run_dir / "eri_closure" / "dumps").glob("round_*/metadata.json"))
    if not dump_metadata:
        raise FileNotFoundError(f"Cannot identify run without ERI dump metadata: {run_dir}")
    metadata = json.loads(dump_metadata[0].read_text(encoding="utf-8"))
    args = metadata.get("resolved_args", {})
    return {
        "run_dir": str(run_dir),
        "seed": int(args.get("seed", -1)),
        "partition": str(args.get("partition", metadata.get("partition", ""))),
        "aggregation": str(args.get("cliplora_aggregation", metadata.get("aggregation", "fedavg"))),
    }


def _interval_weights(rounds: list[int]) -> dict[int, float]:
    ordered = sorted(set(rounds))
    if not ordered:
        return {}
    return {
        round_id: float((ordered[index + 1] - round_id) if index + 1 < len(ordered) else 1)
        for index, round_id in enumerate(ordered)
    }


def summarize(output_root: str | Path, *, output_dir: str | Path | None = None) -> Path:
    root = Path(output_root)
    output = Path(output_dir) if output_dir else root / "eri_closure_summary"
    analysis_files = sorted(root.glob("**/eri_closure/analysis/round_signed_budgets.csv"))
    if not analysis_files:
        raise FileNotFoundError(f"No ERI analysis files found below {root}")
    class_rows: list[dict] = []
    run_rows: list[dict] = []
    for file in analysis_files:
        identity = _run_identity(file)
        budgets = _read_csv(file)
        by_class: dict[int, list[dict]] = defaultdict(list)
        for row in budgets:
            if row.get("method") == "trained_server":
                by_class[int(row["class_id"])].append(row)
        if not by_class:
            continue
        per_run_eris = []
        for class_id, rows in by_class.items():
            weights = _interval_weights([int(row["communication_round"]) for row in rows])
            total_w = sum(_float(row["W"]) * weights[int(row["communication_round"])] for row in rows)
            total_h = sum(_float(row["H"]) * weights[int(row["communication_round"])] for row in rows)
            total_d = sum(_float(row["D"]) * weights[int(row["communication_round"])] for row in rows)
            total_r = sum(_float(row["R"]) * weights[int(row["communication_round"])] for row in rows)
            ceri = total_r / (total_w + total_d + 1e-12)
            per_run_eris.append(ceri)
            class_rows.append({
                **identity, "class_id": class_id, "weighted_W": total_w,
                "weighted_H": total_h, "weighted_D": total_d, "weighted_R": total_r,
                "positive_functional_refresh": total_w + total_d,
                "CERI": ceri,
            })
        run_rows.append({**identity, "tail_CERI_mean": mean(per_run_eris), "tail_class_count": len(per_run_eris)})

    # Add best-to-final drops from the normal post-round test logging.
    drops: dict[tuple[str, int], float] = {}
    for item in class_rows:
        run_dir, class_id = Path(item["run_dir"]), int(item["class_id"])
        key = (str(run_dir), class_id)
        if key in drops:
            continue
        path = run_dir / "eri_closure" / "test_per_class_metrics.csv"
        if not path.exists():
            drops[key] = math.nan
            continue
        points = [
            _float(row["accuracy_percent"])
            for row in _read_csv(path)
            if int(row["class_id"]) == class_id and int(row["communication_round"]) >= 1
        ]
        drops[key] = max(points) - points[-1] if points else math.nan
    for item in class_rows:
        item["best_to_final_drop_pp"] = drops[(item["run_dir"], int(item["class_id"]))]
    # Outcome retention is computed from the untouched normal global-test
    # curve, after all train-only attribution work is complete.
    run_outcomes = {}
    grouped_for_outcome: dict[str, list[dict]] = defaultdict(list)
    for row in class_rows:
        grouped_for_outcome[row["run_dir"]].append(row)
    for run_dir, rows in grouped_for_outcome.items():
        path = Path(run_dir) / "eri_closure" / "test_per_class_metrics.csv"
        by_class: dict[int, list[float]] = defaultdict(list)
        if path.exists():
            for row in _read_csv(path):
                if int(row["communication_round"]) >= 1:
                    by_class[int(row["class_id"])].append(_float(row["accuracy_percent"]))
        best_values = [max(by_class[int(row["class_id"])]) for row in rows if by_class.get(int(row["class_id"]))]
        final_values = [by_class[int(row["class_id"])][-1] for row in rows if by_class.get(int(row["class_id"]))]
        best_mean = mean(best_values) if best_values else math.nan
        final_mean = mean(final_values) if final_values else math.nan
        run_outcomes[run_dir] = {
            "tail_best_accuracy_percent": best_mean,
            "tail_final_accuracy_percent": final_mean,
            "tail_retention_percent": 100.0 * final_mean / best_mean if best_mean > 0 else math.nan,
            "tail_BFD_pp": mean([row["best_to_final_drop_pp"] for row in rows if math.isfinite(row["best_to_final_drop_pp"])]) if any(math.isfinite(row["best_to_final_drop_pp"]) for row in rows) else math.nan,
        }
    for row in run_rows:
        row.update(run_outcomes.get(row["run_dir"], {}))
    write_csv(output / "per_class_ceri_bfd.csv", class_rows)
    write_csv(output / "per_run_ceri.csv", run_rows)

    correlation_rows = []
    by_run: dict[str, list[dict]] = defaultdict(list)
    for row in class_rows:
        by_run[row["run_dir"]].append(row)
    for run_dir, rows in by_run.items():
        correlation_rows.append({
            **{key: rows[0][key] for key in ("run_dir", "seed", "partition", "aggregation")},
            "spearman_CERI_vs_BFD": spearman(
                [row["CERI"] for row in rows], [row["best_to_final_drop_pp"] for row in rows]
            ),
            "tail_classes_with_outcome": sum(math.isfinite(row["best_to_final_drop_pp"]) for row in rows),
        })
    write_csv(output / "per_run_ceri_bfd_correlation.csv", correlation_rows)

    # Paired conditions: same seed and aggregation, Client-LT minus matched Dirichlet.
    index = {(row["seed"], row["aggregation"], row["partition"]): row for row in run_rows}
    paired = []
    for (seed, aggregation, partition), client_lt in index.items():
        if partition != "client-longtail":
            continue
        matched = index.get((seed, aggregation, "matched-dirichlet"))
        if matched:
            paired.append({
                "seed": seed, "aggregation": aggregation,
                "clientlt_tail_CERI": client_lt["tail_CERI_mean"],
                "dirichlet_tail_CERI": matched["tail_CERI_mean"],
                "clientlt_minus_dirichlet": client_lt["tail_CERI_mean"] - matched["tail_CERI_mean"],
            })
    write_csv(output / "paired_topology_CERI.csv", paired)
    # H3: identical topology/seed, support-normalized intervention minus
    # ordinary FedAvg. Retention is the result-side consistency check.
    intervention = []
    for (seed, aggregation, partition), fedavg in index.items():
        if aggregation != "fedavg":
            continue
        controlled = index.get((seed, "support_normalized", partition))
        if controlled:
            intervention.append({
                "seed": seed, "partition": partition,
                "fedavg_tail_CERI": fedavg["tail_CERI_mean"],
                "support_normalized_tail_CERI": controlled["tail_CERI_mean"],
                "CERI_delta_control_minus_fedavg": controlled["tail_CERI_mean"] - fedavg["tail_CERI_mean"],
                "fedavg_tail_retention_percent": fedavg.get("tail_retention_percent", math.nan),
                "support_normalized_tail_retention_percent": controlled.get("tail_retention_percent", math.nan),
                "retention_delta_pp": controlled.get("tail_retention_percent", math.nan) - fedavg.get("tail_retention_percent", math.nan),
            })
    write_csv(output / "paired_intervention_effects.csv", intervention)
    summary = {
        "schema_version": "eri_closure_summary_v1",
        "num_runs": len(run_rows), "num_per_class_records": len(class_rows),
        "num_paired_topology_records": len(paired),
        "mean_clientlt_minus_dirichlet_CERI": mean([row["clientlt_minus_dirichlet"] for row in paired]) if paired else math.nan,
        "mean_per_run_spearman_CERI_vs_BFD": mean([
            row["spearman_CERI_vs_BFD"] for row in correlation_rows
            if math.isfinite(row["spearman_CERI_vs_BFD"])
        ]) if any(math.isfinite(row["spearman_CERI_vs_BFD"]) for row in correlation_rows) else math.nan,
        "mean_clientlt_intervention_CERI_delta": mean([
            row["CERI_delta_control_minus_fedavg"] for row in intervention
            if row["partition"] == "client-longtail"
        ]) if any(row["partition"] == "client-longtail" for row in intervention) else math.nan,
        "mean_clientlt_intervention_retention_delta_pp": mean([
            row["retention_delta_pp"] for row in intervention
            if row["partition"] == "client-longtail" and math.isfinite(row["retention_delta_pp"])
        ]) if any(row["partition"] == "client-longtail" and math.isfinite(row["retention_delta_pp"]) for row in intervention) else math.nan,
        "interpretation": "ERI is destructive class-absent rewrite divided by all positive functional refresh; it is not an absolute rewrite magnitude.",
    }
    write_json(output / "eri_closure_summary.json", summary)
    return output
