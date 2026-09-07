"""Aggregate ERI closure outputs into topology, outcome, and intervention tests."""

from __future__ import annotations

import csv
import itertools
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


def _finite_mean(values) -> float:
    finite = [float(value) for value in values if math.isfinite(_float(value))]
    return mean(finite) if finite else math.nan


def _product_shapley(reference: dict[str, float], target: dict[str, float]) -> dict[str, float]:
    """Exact order-averaged attribution for target product minus reference product."""
    keys = ("A", "rho", "mu")
    contributions = {key: 0.0 for key in keys}
    for order in itertools.permutations(keys):
        state = dict(reference)
        previous = math.prod(state[key] for key in keys)
        for key in order:
            state[key] = target[key]
            current = math.prod(state[item] for item in keys)
            contributions[key] += current - previous
            previous = current
    scale = float(math.factorial(len(keys)))
    return {key: value / scale for key, value in contributions.items()}


def _factor_ratio(target: float, reference: float) -> float:
    return target / reference if reference > 0.0 and math.isfinite(target) else math.nan


def _method_location(a_ratio: float, rho_ratio: float, mu_ratio: float, *, bound: float = 0.10) -> str:
    """Preregistered practical-equivalence routing, not a significance test."""
    if not all(math.isfinite(value) for value in (a_ratio, rho_ratio, mu_ratio)):
        return "unavailable_missing_factor_data"
    lower = 1.0 - float(bound)
    access_low = math.isfinite(a_ratio) and a_ratio < lower
    rho_low = math.isfinite(rho_ratio) and rho_ratio < lower
    mu_low = math.isfinite(mu_ratio) and mu_ratio < lower
    if access_low and not rho_low and not mu_low:
        return "server_aggregation"
    if not access_low and rho_low and not mu_low:
        return "local_training_or_update_selection"
    if not access_low and not rho_low and mu_low:
        return "local_functional_efficiency"
    if access_low and (rho_low or mu_low):
        return "joint_function_conditioned_aggregation"
    if rho_low or mu_low:
        return "local_training"
    return "no_practically_large_factor_drop"


def _paired_class_bootstrap(
    reference_rows: list[dict],
    target_rows: list[dict],
    *,
    seed: int,
    draws: int = 5000,
) -> dict[str, float]:
    """Class-resampling intervals; these do not replace multi-seed uncertainty."""
    reference_by_class = {int(row["class_id"]): row for row in reference_rows}
    target_by_class = {int(row["class_id"]): row for row in target_rows}
    classes = sorted(set(reference_by_class) & set(target_by_class))
    if len(classes) < 2:
        return {}
    arrays = {}
    for name, field in (("A", "integrated_support_access_A"),
                        ("B", "integrated_positive_support_access_B"),
                        ("W", "weighted_W")):
        arrays[f"reference_{name}"] = np.asarray(
            [_float(reference_by_class[c][field]) for c in classes], dtype=float
        )
        arrays[f"target_{name}"] = np.asarray(
            [_float(target_by_class[c][field]) for c in classes], dtype=float
        )
    if not all(np.isfinite(values).all() for values in arrays.values()):
        return {}
    rng = np.random.default_rng(int(seed))
    ratios = {key: [] for key in ("A", "rho", "mu")}
    for _ in range(int(draws)):
        chosen = rng.integers(0, len(classes), size=len(classes))
        values = {
            key: float(array[chosen].mean()) for key, array in arrays.items()
        }
        reference = {
            "A": values["reference_A"],
            "rho": values["reference_B"] / values["reference_A"],
            "mu": values["reference_W"] / values["reference_B"],
        }
        target = {
            "A": values["target_A"],
            "rho": values["target_B"] / values["target_A"],
            "mu": values["target_W"] / values["target_B"],
        }
        for key in ratios:
            ratios[key].append(_factor_ratio(target[key], reference[key]))
    result = {"paired_tail_class_count": len(classes), "class_bootstrap_draws": int(draws)}
    for key, values in ratios.items():
        low, high = np.quantile(np.asarray(values), [0.025, 0.975])
        result[f"{key}_ratio_class_bootstrap_ci_low"] = float(low)
        result[f"{key}_ratio_class_bootstrap_ci_high"] = float(high)
    return result


def _write_factorization_report(path: Path, rows: list[dict], *, maintenance_start_round: int) -> None:
    lines = [
        "# Supporter write factorization",
        "",
        f"Maintenance window starts at round {int(maintenance_start_round)}. All A, B, and W totals use the same trapezoidal audit-time weights.",
        "",
        "For each class, B is the aggregation mass of positive supporter accesses, rho=B/A, mu=W/B, and W=A*rho*mu. The tail row uses macro totals before taking ratios, so it also closes exactly.",
        "",
        "The automatic method-location label uses a preregistered 10% practical-equivalence bound. It is descriptive; class-bootstrap intervals do not substitute for uncertainty across training seeds.",
        "",
    ]
    for row in rows:
        lines.extend([
            f"## seed={row['seed']}, frac={row['frac']}, aggregation={row['aggregation']}",
            "",
            "| quantity | matched Dirichlet | Client-LT | CLT/Dir |",
            "|---|---:|---:|---:|",
            f"| A: supporter access | {row['dirichlet_A']:.6f} | {row['clientlt_A']:.6f} | {row['A_ratio']:.4f} |",
            f"| rho: write success | {row['dirichlet_rho']:.6f} | {row['clientlt_rho']:.6f} | {row['rho_ratio']:.4f} |",
            f"| mu: positive strength | {row['dirichlet_mu']:.6f} | {row['clientlt_mu']:.6f} | {row['mu_ratio']:.4f} |",
            f"| W | {row['dirichlet_W']:.6f} | {row['clientlt_W']:.6f} | {row['W_ratio']:.4f} |",
            "",
            f"Exact Shapley contributions to CLT-Dirichlet delta W={row['W_delta']:.6f}: access={row['shapley_A_contribution']:.6f}, success={row['shapley_rho_contribution']:.6f}, strength={row['shapley_mu_contribution']:.6f}; closure error={row['shapley_closure_absolute_error']:.3g}.",
            "",
            f"Method-location result: `{row['recommended_method_location']}`. Dominant negative factor: `{row['dominant_negative_factor']}`.",
            "",
        ])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _budgets_with_access(budgets: list[dict], run_dir: Path) -> list[dict]:
    """Backfill the round-level W=A*rho*mu factors for old analyses.

    ``client_effects.functional_effect`` is the server-realized contribution
    q_k e_{k,c}, not the unweighted e_{k,c}.  For positive q_k their signs are
    identical.  Thus B=sum q_k 1[e_{k,c}>0] can be recovered exactly from the
    light client-effect and aggregation-weight CSVs; no model replay is needed.
    """
    required = {
        "support_access", "positive_support_access",
        "support_write_success_rate", "positive_write_strength",
    }
    if budgets and all(required.issubset(row) for row in budgets):
        return budgets
    count_path = run_dir / "client_class_counts.csv"
    weight_path = run_dir / "lora_aggregation_weights.csv"
    client_effect_path = run_dir / "eri_closure" / "analysis" / "client_effects.csv"
    if not weight_path.exists():
        return budgets
    counts = (
        {int(row["client_id"]): row for row in _read_csv(count_path)}
        if count_path.exists() else {}
    )
    weights_by_round: dict[int, dict[int, float]] = defaultdict(dict)
    for row in _read_csv(weight_path):
        weights_by_round[int(row["communication_round"])][int(row["client_id"])] = _float(
            row["aggregation_weight"]
        )
    effects_by_key: dict[tuple[int, int, str], list[dict]] = defaultdict(list)
    if client_effect_path.exists():
        for client_row in _read_csv(client_effect_path):
            effects_by_key[(
                int(client_row["communication_round"]),
                int(client_row["class_id"]),
                str(client_row.get("method", "trained_server")),
            )].append(client_row)
    enriched = []
    for original in budgets:
        row = dict(original)
        communication_round = int(row["communication_round"])
        class_id = int(row["class_id"])
        method = str(row.get("method", "trained_server"))
        round_weights = weights_by_round.get(communication_round, {})
        class_key = f"class_{int(row['class_id'])}"
        effect_rows = effects_by_key.get((communication_round, class_id, method), [])
        supporter_weights = []
        positive_supporter_weights = []
        if effect_rows:
            for effect_row in effect_rows:
                client_id = int(effect_row["client_id"])
                q = _float(effect_row.get("aggregation_weight"))
                if not math.isfinite(q):
                    q = round_weights.get(client_id, math.nan)
                if not math.isfinite(q) or not bool(int(effect_row["supports_class"])):
                    continue
                supporter_weights.append(q)
                # q_k is non-negative, so sign(q_k e)=sign(e) whenever q_k>0.
                if _float(effect_row["functional_effect"]) > 0.0:
                    positive_supporter_weights.append(q)
        elif counts:
            supporter_weights = [
                weight
                for client_id, weight in round_weights.items()
                if client_id in counts and int(counts[client_id][class_key]) > 0
            ]
        if round_weights:
            access = _float(row.get("support_access"), sum(supporter_weights))
            if not math.isfinite(access):
                access = sum(supporter_weights)
            positive_access = _float(
                row.get("positive_support_access"),
                sum(positive_supporter_weights) if effect_rows else math.nan,
            )
            squared_mass = sum(weight * weight for weight in supporter_weights)
            row["support_access"] = access
            row["positive_support_access"] = positive_access
            row["support_write_success_rate"] = (
                positive_access / access
                if access > 0.0 and math.isfinite(positive_access) else math.nan
            )
            row["positive_write_strength"] = (
                _float(row["W"]) / positive_access
                if math.isfinite(positive_access) and positive_access > 0.0 else math.nan
            )
            row["positive_write_efficiency"] = (
                _float(row["W"]) / access if access > 0.0 else math.nan
            )
            reconstructed = (
                access
                * row["support_write_success_rate"]
                * row["positive_write_strength"]
            )
            row["write_factorization_absolute_error"] = (
                abs(_float(row["W"]) - reconstructed)
                if math.isfinite(reconstructed) else math.nan
            )
            row["support_effective_clients"] = (
                access * access / squared_mass if squared_mass > 0.0 else math.nan
            )
        enriched.append(row)
    return enriched


RUN_METRICS = (
    "tail_W_mean",
    "tail_A_integral_mean",
    "tail_support_access_time_mean",
    "tail_positive_write_efficiency",
    "tail_support_write_success_rate",
    "tail_positive_write_strength",
    "tail_support_effective_clients_time_mean",
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
    clientlt_a = _float(client_lt.get("tail_A_integral_mean"))
    dirichlet_a = _float(matched.get("tail_A_integral_mean"))
    clientlt_e = _float(client_lt.get("tail_positive_write_efficiency"))
    dirichlet_e = _float(matched.get("tail_positive_write_efficiency"))
    access_component = (clientlt_a - dirichlet_a) * (clientlt_e + dirichlet_e) / 2.0
    efficiency_component = (clientlt_e - dirichlet_e) * (clientlt_a + dirichlet_a) / 2.0
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
        "shapley_access_component_of_W_delta": access_component,
        "shapley_efficiency_component_of_W_delta": efficiency_component,
        "shapley_component_sum": access_component + efficiency_component,
        "access_share_of_W_delta_percent": _relative_percent(access_component, w_delta),
        "efficiency_share_of_W_delta_percent": _relative_percent(efficiency_component, w_delta),
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
    correlation_rows: list[dict],
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
        "## W = A * efficiency decomposition",
        "",
        "| Seed | frac | aggregation | mean A: Client-LT / Dir. | efficiency: Client-LT / Dir. | W delta | access component | efficiency component | access share of W delta |",
        "|---:|---:|---|---:|---:|---:|---:|---:|---:|",
    ])
    for row in sorted(mechanism_checks, key=lambda item: (item["seed"], item["frac"], item["aggregation"])):
        lines.append(
            "| {seed} | {frac} | {aggregation} | {ca} / {da} | {ce} / {de} | {w} | {ac} | {ec} | {share}% |".format(
                seed=row["seed"], frac=_format_number(row["frac"], 1),
                aggregation=row["aggregation"],
                ca=_format_number(row["clientlt_tail_support_access_time_mean"], 4),
                da=_format_number(row["dirichlet_tail_support_access_time_mean"], 4),
                ce=_format_number(row["clientlt_tail_positive_write_efficiency"], 4),
                de=_format_number(row["dirichlet_tail_positive_write_efficiency"], 4),
                w=_format_number(row["delta_tail_W_mean"]),
                ac=_format_number(row["shapley_access_component_of_W_delta"]),
                ec=_format_number(row["shapley_efficiency_component_of_W_delta"]),
                share=_format_number(row["access_share_of_W_delta_percent"], 1),
            )
        )
    lines.extend([
        "",
        "## Per-class diagnostic associations",
        "",
        "| Seed | frac | topology | rho(A,W) | rho(efficiency,W) | rho(A,BFD) | rho(efficiency,BFD) | rho(A,final acc.) | rho(efficiency,final acc.) |",
        "|---:|---:|---|---:|---:|---:|---:|---:|---:|",
    ])
    for row in sorted(correlation_rows, key=lambda item: (item["seed"], item["frac"], item["partition"], item["aggregation"])):
        lines.append(
            "| {seed} | {frac} | {topology}/{aggregation} | {aw} | {ew} | {ab} | {eb} | {af} | {ef} |".format(
                seed=row["seed"], frac=_format_number(row["frac"], 1),
                topology=row["partition"], aggregation=row["aggregation"],
                aw=_format_number(row["spearman_A_vs_W"]),
                ew=_format_number(row["spearman_efficiency_vs_W"]),
                ab=_format_number(row["spearman_A_vs_BFD"]),
                eb=_format_number(row["spearman_efficiency_vs_BFD"]),
                af=_format_number(row["spearman_A_vs_final_accuracy"]),
                ef=_format_number(row["spearman_efficiency_vs_final_accuracy"]),
            )
        )
    lines.extend([
        "",
        "These class-wise correlations are diagnostic associations. The end-to-end "
        "support-normalized intervention is required for an accuracy-level causal claim.",
        "",
        "## Partial participation minus full participation",
        "",
        "| Seed | Topology | target frac | W delta | mean A delta | efficiency delta | D delta | P delta | R delta | R/P delta | no-support-rate delta | retention delta (pp) |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for row in sorted(fraction_effects, key=lambda item: (item["seed"], item["partition"], item["target_frac"])):
        lines.append(
            "| {seed} | {topology} | {frac} | {w} | {a} | {e} | {d} | {p} | {r} | {b} | {n} | {ret} |".format(
                seed=row["seed"], topology=row["partition"],
                frac=_format_number(row["target_frac"], 1),
                w=_format_number(row["delta_tail_W_mean"]),
                a=_format_number(row["delta_tail_support_access_time_mean"], 4),
                e=_format_number(row["delta_tail_positive_write_efficiency"], 4),
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
        budgets = _budgets_with_access(_read_csv(file), Path(identity["run_dir"]))
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
            total_a = (
                _trapezoid_total(rows, "support_access")
                if all(math.isfinite(_float(row.get("support_access"))) for row in rows)
                else math.nan
            )
            total_b = (
                _trapezoid_total(rows, "positive_support_access")
                if all(math.isfinite(_float(row.get("positive_support_access"))) for row in rows)
                else math.nan
            )
            total_h = _trapezoid_total(rows, "H")
            total_d = _trapezoid_total(rows, "D")
            total_r = _trapezoid_total(rows, "R")
            total_p = total_w + total_d
            ceri = total_r / (total_w + total_d + 1e-12)
            start_round = min(int(row["communication_round"]) for row in rows)
            end_round = max(int(row["communication_round"]) for row in rows)
            span = max(end_round - start_round, 1)
            efficiency = total_w / total_a if math.isfinite(total_a) and total_a > 0.0 else math.nan
            success_rate = (
                total_b / total_a
                if math.isfinite(total_b) and math.isfinite(total_a) and total_a > 0.0
                else math.nan
            )
            positive_strength = (
                total_w / total_b
                if math.isfinite(total_b) and total_b > 0.0 else math.nan
            )
            factorized_w = total_a * success_rate * positive_strength
            effective_integral = (
                _trapezoid_total(rows, "support_effective_clients")
                if all(math.isfinite(_float(row.get("support_effective_clients"))) for row in rows)
                else math.nan
            )
            class_rows.append({
                **identity, "class_id": class_id, "weighted_W": total_w,
                "integrated_support_access_A": total_a,
                "integrated_positive_support_access_B": total_b,
                "support_access_time_mean": total_a / span if math.isfinite(total_a) else math.nan,
                "support_write_success_rate_rho": success_rate,
                "positive_write_strength_mu": positive_strength,
                "factorized_W": factorized_w,
                "write_factorization_absolute_error": (
                    abs(total_w - factorized_w) if math.isfinite(factorized_w) else math.nan
                ),
                "positive_write_efficiency_W_over_A": efficiency,
                "support_effective_clients_time_mean": (
                    effective_integral / span if math.isfinite(effective_integral) else math.nan
                ),
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
        a_mean = _finite_mean(row["integrated_support_access_A"] for row in run_class_rows)
        b_mean = _finite_mean(row["integrated_positive_support_access_B"] for row in run_class_rows)
        d_mean = mean(row["weighted_D"] for row in run_class_rows)
        r_mean = mean(row["weighted_R"] for row in run_class_rows)
        total_audits = sum(row["maintenance_audit_count"] for row in run_class_rows)
        no_support_audits = sum(row["no_support_audit_count"] for row in run_class_rows)
        run_rows.append({
            **identity,
            "maintenance_start_round": int(maintenance_start_round),
            "tail_class_count": len(run_class_rows),
            "tail_W_mean": w_mean,
            "tail_A_integral_mean": a_mean,
            "tail_B_integral_mean": b_mean,
            "tail_support_access_time_mean": _finite_mean(
                row["support_access_time_mean"] for row in run_class_rows
            ),
            # Macro factors are ratios of macro totals rather than unweighted
            # means of per-class ratios. This preserves W=A*rho*mu exactly.
            "tail_support_write_success_rate": (
                b_mean / a_mean
                if math.isfinite(a_mean) and a_mean > 0.0 and math.isfinite(b_mean)
                else math.nan
            ),
            "tail_positive_write_strength": (
                w_mean / b_mean
                if math.isfinite(b_mean) and b_mean > 0.0 else math.nan
            ),
            # Ratio of macro integrated budgets preserves the exact identity
            # tail_W_mean = tail_A_integral_mean * efficiency.
            "tail_positive_write_efficiency": (
                w_mean / a_mean if math.isfinite(a_mean) and a_mean > 0.0 else math.nan
            ),
            "tail_class_macro_positive_write_efficiency": _finite_mean(
                row["positive_write_efficiency_W_over_A"] for row in run_class_rows
            ),
            "tail_class_macro_support_write_success_rate": _finite_mean(
                row["support_write_success_rate_rho"] for row in run_class_rows
            ),
            "tail_class_macro_positive_write_strength": _finite_mean(
                row["positive_write_strength_mu"] for row in run_class_rows
            ),
            "tail_support_effective_clients_time_mean": _finite_mean(
                row["support_effective_clients_time_mean"] for row in run_class_rows
            ),
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
    finals: dict[tuple[str, int], float] = {}
    for item in class_rows:
        run_dir, class_id = Path(item["run_dir"]), int(item["class_id"])
        key = (str(run_dir), class_id)
        if key in drops:
            continue
        path = run_dir / "eri_closure" / "test_per_class_metrics.csv"
        if not path.exists():
            drops[key] = math.nan
            finals[key] = math.nan
            continue
        points = [
            _float(row["accuracy_percent"])
            for row in _read_csv(path)
            if int(row["class_id"]) == class_id and int(row["communication_round"]) >= 1
        ]
        drops[key] = max(points) - points[-1] if points else math.nan
        finals[key] = points[-1] if points else math.nan
    for item in class_rows:
        item["best_to_final_drop_pp"] = drops[(item["run_dir"], int(item["class_id"]))]
        item["final_accuracy_percent"] = finals[(item["run_dir"], int(item["class_id"]))]
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
    write_csv(output / "per_class_supporter_write_factorization.csv", [{
        key: row.get(key) for key in (
            "run_dir", "seed", "frac", "partition", "aggregation", "class_id",
            "weighted_W", "integrated_support_access_A",
            "integrated_positive_support_access_B", "support_write_success_rate_rho",
            "positive_write_strength_mu", "factorized_W",
            "write_factorization_absolute_error", "maintenance_start_round",
            "maintenance_end_round", "maintenance_audit_count",
        )
    } for row in class_rows])
    write_csv(output / "per_run_supporter_write_factorization.csv", [{
        key: row.get(key) for key in (
            "run_dir", "seed", "frac", "partition", "aggregation", "tail_class_count",
            "tail_W_mean", "tail_A_integral_mean", "tail_B_integral_mean",
            "tail_support_write_success_rate", "tail_positive_write_strength",
            "tail_positive_write_efficiency", "maintenance_start_round",
        )
    } for row in run_rows])

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
            "spearman_A_vs_W": spearman(
                [row["integrated_support_access_A"] for row in rows],
                [row["weighted_W"] for row in rows],
            ),
            "spearman_efficiency_vs_W": spearman(
                [row["positive_write_efficiency_W_over_A"] for row in rows],
                [row["weighted_W"] for row in rows],
            ),
            "spearman_A_vs_BFD": spearman(
                [row["integrated_support_access_A"] for row in rows],
                [row["best_to_final_drop_pp"] for row in rows],
            ),
            "spearman_efficiency_vs_BFD": spearman(
                [row["positive_write_efficiency_W_over_A"] for row in rows],
                [row["best_to_final_drop_pp"] for row in rows],
            ),
            "spearman_A_vs_final_accuracy": spearman(
                [row["integrated_support_access_A"] for row in rows],
                [row["final_accuracy_percent"] for row in rows],
            ),
            "spearman_efficiency_vs_final_accuracy": spearman(
                [row["positive_write_efficiency_W_over_A"] for row in rows],
                [row["final_accuracy_percent"] for row in rows],
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
    factorization_pairs = []
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
            reference_factors = {
                "A": _float(matched["tail_A_integral_mean"]),
                "rho": _float(matched["tail_support_write_success_rate"]),
                "mu": _float(matched["tail_positive_write_strength"]),
            }
            target_factors = {
                "A": _float(client_lt["tail_A_integral_mean"]),
                "rho": _float(client_lt["tail_support_write_success_rate"]),
                "mu": _float(client_lt["tail_positive_write_strength"]),
            }
            contributions = _product_shapley(reference_factors, target_factors)
            a_ratio = _factor_ratio(target_factors["A"], reference_factors["A"])
            rho_ratio = _factor_ratio(target_factors["rho"], reference_factors["rho"])
            mu_ratio = _factor_ratio(target_factors["mu"], reference_factors["mu"])
            w_delta = _float(client_lt["tail_W_mean"]) - _float(matched["tail_W_mean"])
            negative = {key: value for key, value in contributions.items() if value < 0.0}
            bootstrap = _paired_class_bootstrap(
                [row for row in class_rows if row["run_dir"] == matched["run_dir"]],
                [row for row in class_rows if row["run_dir"] == client_lt["run_dir"]],
                seed=(int(seed) * 10007 + int(round(float(frac) * 10000))
                      + sum(ord(char) for char in str(aggregation))),
            )
            factorization_pairs.append({
                "seed": seed,
                "frac": frac,
                "aggregation": aggregation,
                "practical_equivalence_bound_percent": 10.0,
                "factorization_complete": int(all(
                    math.isfinite(value) and value > 0.0
                    for value in (*reference_factors.values(), *target_factors.values())
                )),
                "dirichlet_A": reference_factors["A"],
                "clientlt_A": target_factors["A"],
                "A_ratio": a_ratio,
                "dirichlet_rho": reference_factors["rho"],
                "clientlt_rho": target_factors["rho"],
                "rho_ratio": rho_ratio,
                "dirichlet_mu": reference_factors["mu"],
                "clientlt_mu": target_factors["mu"],
                "mu_ratio": mu_ratio,
                "dirichlet_W": _float(matched["tail_W_mean"]),
                "clientlt_W": _float(client_lt["tail_W_mean"]),
                "W_ratio": _factor_ratio(
                    _float(client_lt["tail_W_mean"]), _float(matched["tail_W_mean"])
                ),
                "W_delta": w_delta,
                "shapley_A_contribution": contributions["A"],
                "shapley_rho_contribution": contributions["rho"],
                "shapley_mu_contribution": contributions["mu"],
                "shapley_closure_absolute_error": abs(
                    w_delta - sum(contributions.values())
                ),
                "clientlt_factorization_absolute_error": abs(
                    _float(client_lt["tail_W_mean"])
                    - math.prod(target_factors.values())
                ),
                "dirichlet_factorization_absolute_error": abs(
                    _float(matched["tail_W_mean"])
                    - math.prod(reference_factors.values())
                ),
                "dominant_negative_factor": (
                    min(negative, key=negative.get) if negative else "none"
                ),
                "recommended_method_location": _method_location(a_ratio, rho_ratio, mu_ratio),
                **bootstrap,
            })
    write_csv(output / "paired_topology_CERI.csv", paired)
    write_csv(output / "mechanism_claim_check.csv", mechanism_checks)
    write_csv(output / "paired_supporter_write_factorization.csv", factorization_pairs)
    _write_factorization_report(
        output / "supporter_write_factorization_report.md",
        factorization_pairs,
        maintenance_start_round=maintenance_start_round,
    )
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
        correlation_rows,
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
        "num_supporter_write_factorization_pairs": len(factorization_pairs),
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
