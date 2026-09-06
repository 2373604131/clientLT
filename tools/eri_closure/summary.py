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
    metadata = {}
    args = {}
    if dump_metadata:
        metadata = json.loads(dump_metadata[0].read_text(encoding="utf-8"))
        args = metadata.get("resolved_args", {})
    manifest_path = analysis_file.parent / "analysis_manifest.json"
    if not args and manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        args = manifest.get("run_identity", {})

    # Analysis-only share archives intentionally omit large round dumps. Keep
    # them summarizable by falling back to the stable run-directory contract.
    case = run_dir.parent.name
    inferred = {
        "clientlt_fedavg": ("client-longtail", "fedavg"),
        "matched_dirichlet_fedavg": ("matched-dirichlet", "fedavg"),
        "clientlt_support_normalized": ("client-longtail", "support_normalized"),
        "matched_dirichlet_support_normalized": ("matched-dirichlet", "support_normalized"),
    }.get(case)
    if inferred is None and not args:
        raise FileNotFoundError(f"Cannot identify analysis-only ERI run: {run_dir}")
    seed_text = run_dir.name.removeprefix("seed")
    frac = 1.0
    for parent in run_dir.parents:
        if parent.name.startswith("frac") and "p" in parent.name:
            frac = float(parent.name.removeprefix("frac").replace("p", "."))
            break
    return {
        "run_dir": str(run_dir),
        "seed": int(args.get("seed", seed_text)),
        "frac": float(args.get("frac", frac)),
        "partition": str(args.get("partition", metadata.get("partition", inferred[0] if inferred else ""))),
        "aggregation": str(args.get("cliplora_aggregation", metadata.get("aggregation", inferred[1] if inferred else "fedavg"))),
    }


def _trapezoid_total(rows: list[dict], field: str) -> float:
    """Integrate an audited trajectory without pretending audits are uniform."""
    points = sorted(
        (int(row["communication_round"]), _float(row[field]))
        for row in rows
    )
    if not points:
        return math.nan
    if len(points) == 1:
        return points[0][1]
    return sum(
        0.5 * (left_value + right_value) * (right_round - left_round)
        for (left_round, left_value), (right_round, right_value)
        in zip(points, points[1:])
    )


def _relative_percent(delta: float, reference: float) -> float:
    if not math.isfinite(delta) or not math.isfinite(reference) or abs(reference) < 1e-12:
        return math.nan
    return 100.0 * delta / reference


RUN_METRICS = (
    "tail_W_mean",
    "tail_H_mean",
    "tail_D_mean",
    "tail_R_mean",
    "tail_P_mean",
    "tail_balance_ratio",
    "tail_CERI_mean",
    "no_support_class_round_rate",
    "mean_selected_supporters_per_class_round",
    "tail_best_accuracy_percent",
    "tail_final_accuracy_percent",
    "tail_retention_percent",
    "tail_BFD_pp",
)


def _paired_metrics(reference: dict, target: dict, *, reference_prefix: str, target_prefix: str) -> dict:
    result = {}
    for metric in RUN_METRICS:
        reference_value = _float(reference.get(metric))
        target_value = _float(target.get(metric))
        delta = target_value - reference_value
        result[f"{reference_prefix}_{metric}"] = reference_value
        result[f"{target_prefix}_{metric}"] = target_value
        result[f"delta_{metric}"] = delta
        result[f"relative_delta_{metric}_percent"] = _relative_percent(delta, reference_value)
    return result


def _mechanism_claim_row(client_lt: dict, matched: dict) -> dict:
    paired = _paired_metrics(
        matched, client_lt,
        reference_prefix="dirichlet", target_prefix="clientlt",
    )
    w_delta = paired["delta_tail_W_mean"]
    p_delta = paired["delta_tail_P_mean"]
    w_relative = paired["relative_delta_tail_W_mean_percent"]
    p_relative = paired["relative_delta_tail_P_mean_percent"]
    r_relative = paired["relative_delta_tail_R_mean_percent"]
    donor_present = _float(client_lt.get("tail_D_mean")) > 0.0
    donor_partly_closes_write_gap = w_delta < 0.0 and p_delta > w_delta
    checks = {
        "W_reduced": w_delta < 0.0,
        "W_largest_relative_reduction": (
            math.isfinite(w_relative)
            and math.isfinite(p_relative)
            and math.isfinite(r_relative)
            and w_relative < min(p_relative, r_relative)
        ),
        "donor_present": donor_present,
        "donor_increased_vs_dirichlet": paired["delta_tail_D_mean"] > 0.0,
        "donor_partly_closes_write_gap": donor_partly_closes_write_gap,
        "P_reduced_more_than_R_relative": (
            math.isfinite(p_relative)
            and math.isfinite(r_relative)
            and p_relative < r_relative
        ),
        "balance_shifted_toward_rewrite": paired["delta_tail_balance_ratio"] > 0.0,
    }
    required = (
        "W_reduced",
        "W_largest_relative_reduction",
        "donor_present",
        "donor_partly_closes_write_gap",
        "P_reduced_more_than_R_relative",
        "balance_shifted_toward_rewrite",
    )
    return {
        "seed": client_lt["seed"],
        "frac": client_lt["frac"],
        "aggregation": client_lt["aggregation"],
        **paired,
        **{key: int(value) for key, value in checks.items()},
        "mechanism_statement_supported": int(all(checks[key] for key in required)),
    }


def _format_number(value, digits: int = 3) -> str:
    number = _float(value)
    return "NA" if not math.isfinite(number) else f"{number:.{digits}f}"


def _write_fraction_report(
    path: Path,
    mechanism_checks: list[dict],
    fraction_effects: list[dict],
    *,
    maintenance_start_round: int,
) -> None:
    lines = [
        "# ERI participation-fraction comparison",
        "",
        f"Maintenance window starts at communication round {int(maintenance_start_round)}. ",
        "W/H/D/R/P are trapezoidal time integrals followed by a Bottom-20 class macro mean.",
        "",
        "## Topology mechanism check at each fraction",
        "",
        "| Seed | frac | aggregation | W: Client-LT / Dir. | D: Client-LT / Dir. | P relative delta | R relative delta | R/P: Client-LT / Dir. | no-support rate: Client-LT / Dir. | Statement |",
        "|---:|---:|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in sorted(mechanism_checks, key=lambda item: (item["seed"], item["frac"], item["aggregation"])):
        lines.append(
            "| {seed} | {frac} | {aggregation} | {cw} / {dw} | {cd} / {dd} | {pd}% | {rd}% | "
            "{cb} / {db} | {cn} / {dn} | {verdict} |".format(
                seed=row["seed"], frac=_format_number(row["frac"], 1),
                aggregation=row["aggregation"],
                cw=_format_number(row["clientlt_tail_W_mean"]),
                dw=_format_number(row["dirichlet_tail_W_mean"]),
                cd=_format_number(row["clientlt_tail_D_mean"]),
                dd=_format_number(row["dirichlet_tail_D_mean"]),
                pd=_format_number(row["relative_delta_tail_P_mean_percent"], 1),
                rd=_format_number(row["relative_delta_tail_R_mean_percent"], 1),
                cb=_format_number(row["clientlt_tail_balance_ratio"]),
                db=_format_number(row["dirichlet_tail_balance_ratio"]),
                cn=_format_number(row["clientlt_no_support_class_round_rate"], 3),
                dn=_format_number(row["dirichlet_no_support_class_round_rate"], 3),
                verdict="SUPPORTED" if row["mechanism_statement_supported"] else "NOT SUPPORTED",
            )
        )
    lines.extend([
        "",
        "The statement column is a directional consistency check, not a significance test. "
        "Independent-seed inference is still required before using the word statistically significant.",
        "",
        "## Partial participation minus full participation",
        "",
        "| Seed | Topology | target frac | W delta | D delta | P delta | R delta | R/P delta | no-support-rate delta | retention delta (pp) |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for row in sorted(fraction_effects, key=lambda item: (item["seed"], item["partition"], item["target_frac"])):
        lines.append(
            "| {seed} | {topology} | {frac} | {w} | {d} | {p} | {r} | {b} | {n} | {ret} |".format(
                seed=row["seed"], topology=row["partition"],
                frac=_format_number(row["target_frac"], 1),
                w=_format_number(row["delta_tail_W_mean"]),
                d=_format_number(row["delta_tail_D_mean"]),
                p=_format_number(row["delta_tail_P_mean"]),
                r=_format_number(row["delta_tail_R_mean"]),
                b=_format_number(row["delta_tail_balance_ratio"]),
                n=_format_number(row["delta_no_support_class_round_rate"]),
                ret=_format_number(row["delta_tail_retention_percent"]),
            )
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def summarize(
    output_root: str | Path,
    *,
    output_dir: str | Path | None = None,
    maintenance_start_round: int = 10,
) -> Path:
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
            if (
                row.get("method") == "trained_server"
                and int(row["communication_round"]) >= int(maintenance_start_round)
            ):
                by_class[int(row["class_id"])].append(row)
        if not by_class:
            continue
        for class_id, rows in by_class.items():
            total_w = _trapezoid_total(rows, "W")
            total_h = _trapezoid_total(rows, "H")
            total_d = _trapezoid_total(rows, "D")
            total_r = _trapezoid_total(rows, "R")
            total_p = total_w + total_d
            ceri = total_r / (total_w + total_d + 1e-12)
            class_rows.append({
                **identity, "class_id": class_id, "weighted_W": total_w,
                "weighted_H": total_h, "weighted_D": total_d, "weighted_R": total_r,
                "positive_functional_refresh": total_p,
                "CERI": ceri,
                "maintenance_start_round": int(maintenance_start_round),
                "maintenance_end_round": max(int(row["communication_round"]) for row in rows),
                "maintenance_audit_count": len(rows),
                "no_support_audit_count": sum(int(row["supporter_count"]) == 0 for row in rows),
                "no_support_audit_rate": mean(int(row["supporter_count"]) == 0 for row in rows),
                "mean_selected_supporters": mean(int(row["supporter_count"]) for row in rows),
            })
        run_class_rows = [row for row in class_rows if row["run_dir"] == identity["run_dir"]]
        if not run_class_rows:
            continue
        w_mean = mean(row["weighted_W"] for row in run_class_rows)
        d_mean = mean(row["weighted_D"] for row in run_class_rows)
        r_mean = mean(row["weighted_R"] for row in run_class_rows)
        total_audits = sum(row["maintenance_audit_count"] for row in run_class_rows)
        no_support_audits = sum(row["no_support_audit_count"] for row in run_class_rows)
        run_rows.append({
            **identity,
            "maintenance_start_round": int(maintenance_start_round),
            "tail_class_count": len(run_class_rows),
            "tail_W_mean": w_mean,
            "tail_H_mean": mean(row["weighted_H"] for row in run_class_rows),
            "tail_D_mean": d_mean,
            "tail_R_mean": r_mean,
            "tail_P_mean": w_mean + d_mean,
            "tail_balance_ratio": r_mean / (w_mean + d_mean + 1e-12),
            "tail_CERI_mean": mean(row["CERI"] for row in run_class_rows),
            "maintenance_class_round_count": total_audits,
            "no_support_class_round_count": no_support_audits,
            "no_support_class_round_rate": no_support_audits / total_audits,
            "mean_selected_supporters_per_class_round": mean(
                row["mean_selected_supporters"] for row in run_class_rows
            ),
        })

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
            **{key: rows[0][key] for key in ("run_dir", "seed", "frac", "partition", "aggregation")},
            "spearman_CERI_vs_BFD": spearman(
                [row["CERI"] for row in rows], [row["best_to_final_drop_pp"] for row in rows]
            ),
            "tail_classes_with_outcome": sum(math.isfinite(row["best_to_final_drop_pp"]) for row in rows),
        })
    write_csv(output / "per_run_ceri_bfd_correlation.csv", correlation_rows)

    # Paired conditions: same seed and aggregation, Client-LT minus matched Dirichlet.
    index = {
        (row["seed"], row["frac"], row["aggregation"], row["partition"]): row
        for row in run_rows
    }
    paired = []
    mechanism_checks = []
    for (seed, frac, aggregation, partition), client_lt in index.items():
        if partition != "client-longtail":
            continue
        matched = index.get((seed, frac, aggregation, "matched-dirichlet"))
        if matched:
            paired.append({
                "seed": seed, "frac": frac, "aggregation": aggregation,
                "clientlt_tail_CERI": client_lt["tail_CERI_mean"],
                "dirichlet_tail_CERI": matched["tail_CERI_mean"],
                "clientlt_minus_dirichlet": client_lt["tail_CERI_mean"] - matched["tail_CERI_mean"],
            })
            mechanism_checks.append(_mechanism_claim_row(client_lt, matched))
    write_csv(output / "paired_topology_CERI.csv", paired)
    write_csv(output / "mechanism_claim_check.csv", mechanism_checks)
    # H3: identical topology/seed, support-normalized intervention minus
    # ordinary FedAvg. Retention is the result-side consistency check.
    intervention = []
    for (seed, frac, aggregation, partition), fedavg in index.items():
        if aggregation != "fedavg":
            continue
        controlled = index.get((seed, frac, "support_normalized", partition))
        if controlled:
            intervention.append({
                "seed": seed, "frac": frac, "partition": partition,
                "fedavg_tail_CERI": fedavg["tail_CERI_mean"],
                "support_normalized_tail_CERI": controlled["tail_CERI_mean"],
                "CERI_delta_control_minus_fedavg": controlled["tail_CERI_mean"] - fedavg["tail_CERI_mean"],
                "fedavg_tail_retention_percent": fedavg.get("tail_retention_percent", math.nan),
                "support_normalized_tail_retention_percent": controlled.get("tail_retention_percent", math.nan),
                "retention_delta_pp": controlled.get("tail_retention_percent", math.nan) - fedavg.get("tail_retention_percent", math.nan),
            })
    write_csv(output / "paired_intervention_effects.csv", intervention)

    # Directly answer how partial participation changes the same trained
    # topology relative to its full-participation counterpart.
    fraction_effects = []
    fraction_index: dict[tuple[int, str, str], list[dict]] = defaultdict(list)
    for row in run_rows:
        fraction_index[(row["seed"], row["partition"], row["aggregation"])].append(row)
    for (seed, partition, aggregation), rows in fraction_index.items():
        reference = next((row for row in rows if math.isclose(row["frac"], 1.0)), None)
        if reference is None:
            continue
        for target in rows:
            if math.isclose(target["frac"], 1.0):
                continue
            fraction_effects.append({
                "seed": seed,
                "partition": partition,
                "aggregation": aggregation,
                "reference_frac": 1.0,
                "target_frac": target["frac"],
                **_paired_metrics(
                    reference, target,
                    reference_prefix="frac1", target_prefix="partial",
                ),
            })
    write_csv(output / "paired_fraction_effects.csv", fraction_effects)
    _write_fraction_report(
        output / "fraction_comparison_report.md",
        mechanism_checks,
        fraction_effects,
        maintenance_start_round=maintenance_start_round,
    )
    summary = {
        "schema_version": "eri_closure_summary_v2",
        "maintenance_start_round": int(maintenance_start_round),
        "fractions": sorted({row["frac"] for row in run_rows}),
        "num_runs": len(run_rows), "num_per_class_records": len(class_rows),
        "num_paired_topology_records": len(paired),
        "num_paired_fraction_records": len(fraction_effects),
        "num_mechanism_claim_checks": len(mechanism_checks),
        "num_mechanism_claim_checks_passed": sum(
            row["mechanism_statement_supported"] for row in mechanism_checks
        ),
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
