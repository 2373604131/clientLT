from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from tools.client_update_audit.protocol import TAIL_CLASSES, frozen_protocol
from tools.semantic_acquisition.common import write_csv, write_json
from tools.semantic_acquisition.metrics import cluster_bootstrap


def _load_valid_runtime(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    path = Path(path)
    contract = json.loads((path / "runtime_contract.json").read_text(encoding="utf-8"))
    if contract.get("protocol") != frozen_protocol() or contract.get("server_aggregation_called") is not False:
        raise RuntimeError(f"Invalid local-only E2 runtime contract: {path}")
    fairness = pd.read_csv(path / "runtime_fairness.csv")
    if fairness.empty or not fairness["pass"].astype(bool).all() or fairness["server_aggregation_called"].astype(bool).any():
        raise RuntimeError(f"Runtime fairness failed: {path}")
    return pd.read_csv(path / "local_tail_metrics.csv"), pd.read_csv(path / "local_client_summaries.csv")


def _weighted_tail(metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    keys = ["stage", "data_seed", "topology", "condition", "local_epoch", "tail_class"]
    value_columns = [
        "accuracy", "margin", "accuracy_gain", "margin_gain",
        "target_vs_neighbor_pairwise_margin", "worst_neighbor_margin",
        "positive_margin_neighbor_coverage", "target_vs_neighbor_pairwise_margin_gain",
        "worst_neighbor_margin_gain", "positive_margin_neighbor_coverage_gain",
    ]
    for key, group in metrics.groupby(keys, sort=True):
        supported = group[group.tail_sample_count > 0]
        if supported.empty:
            raise RuntimeError(f"No local writer supports tail class for group={key}")
        weights = supported.tail_sample_count.to_numpy(dtype=np.float64)
        weights /= weights.sum()
        row = dict(zip(keys, key))
        row["support_client_count"] = int(len(supported))
        row["tail_sample_count"] = int(supported.tail_sample_count.sum())
        row["tail_mass_weight_sum"] = float(weights.sum())
        for column in value_columns:
            row[column] = float(np.dot(weights, supported[column].to_numpy(dtype=np.float64)))
        row["tail_neighbor_access_score"] = float(
            np.dot(weights, supported.tail_neighbor_access_score.to_numpy(dtype=np.float64))
        )
        row["companion_class_count"] = float(
            np.dot(weights, supported.companion_class_count.to_numpy(dtype=np.float64))
        )
        rows.append(row)
    return pd.DataFrame(rows)


def _accuracy_matched_pairs(
    weighted: pd.DataFrame,
    left_condition: str,
    right_condition: str,
    *,
    topology: str,
    tolerance: float,
    effect_prefix: str,
) -> list[dict]:
    rows = []
    data = weighted[(weighted.topology == topology) & (weighted.local_epoch > 0)]
    for (seed, tail_class), group in data.groupby(["data_seed", "tail_class"], sort=True):
        left = group[group.condition == left_condition]
        right = group[group.condition == right_condition]
        if left.empty or right.empty:
            raise RuntimeError(f"Missing paired E2 condition for seed={seed}, class={tail_class}")
        candidates = []
        for lrow in left.itertuples():
            for rrow in right.itertuples():
                gap = abs(float(lrow.accuracy) - float(rrow.accuracy))
                candidates.append((gap, int(lrow.local_epoch) + int(rrow.local_epoch), int(lrow.local_epoch), int(rrow.local_epoch), lrow, rrow))
        gap, _, left_epoch, right_epoch, lrow, rrow = min(candidates, key=lambda x: x[:4])
        rows.append({
            "data_seed": int(seed), "tail_class": int(tail_class),
            "left_condition": left_condition, "right_condition": right_condition,
            "left_epoch": left_epoch, "right_epoch": right_epoch,
            "left_accuracy": float(lrow.accuracy), "right_accuracy": float(rrow.accuracy),
            "accuracy_gap_abs": float(gap), "accuracy_matched": bool(gap <= tolerance),
            f"{effect_prefix}_worst_neighbor_margin": float(lrow.worst_neighbor_margin - rrow.worst_neighbor_margin),
            f"{effect_prefix}_worst_neighbor_margin_gain": float(lrow.worst_neighbor_margin_gain - rrow.worst_neighbor_margin_gain),
            f"{effect_prefix}_pairwise_neighbor_margin": float(
                lrow.target_vs_neighbor_pairwise_margin - rrow.target_vs_neighbor_pairwise_margin
            ),
            f"{effect_prefix}_positive_neighbor_coverage": float(
                lrow.positive_margin_neighbor_coverage - rrow.positive_margin_neighbor_coverage
            ),
            f"{effect_prefix}_margin": float(lrow.margin - rrow.margin),
        })
    return rows


def _bootstrap_effect(rows: Sequence[Mapping], column: str, draws: int) -> dict:
    by_class: dict[int, list[float]] = {}
    for row in rows:
        if bool(row["accuracy_matched"]):
            by_class.setdefault(int(row["tail_class"]), []).append(float(row[column]))
    if not by_class:
        return {"mean": 0.0, "median": 0.0, "ci_low": 0.0, "ci_high": 0.0, "class_count": 0, "bootstrap_draws": draws}
    return cluster_bootstrap(by_class, draws=draws, seed=20260817)


def summarize(args) -> dict:
    e2a_metrics, e2a_clients = _load_valid_runtime(args.e2a_dir)
    e2b_metrics, e2b_clients = _load_valid_runtime(args.e2b_dir)
    e2b_difficulty = pd.read_csv(Path(args.e2b_dir) / "companion_initial_difficulty.csv")
    e2a_weighted = _weighted_tail(e2a_metrics)
    e2b_weighted = _weighted_tail(e2b_metrics)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    e2a_weighted.to_csv(args.output_dir / "e2a_tail_mass_weighted_metrics.csv", index=False)
    e2b_weighted.to_csv(args.output_dir / "e2b_tail_mass_weighted_metrics.csv", index=False)

    tolerance = float(frozen_protocol()["e2b"]["confirmation_rule"]["accuracy_match_tolerance"])
    access_rows = _accuracy_matched_pairs(
        e2b_weighted, "broad_related", "narrow_related", topology="clientlt",
        tolerance=tolerance, effect_prefix="broad_related_minus_narrow",
    )
    specificity_rows = _accuracy_matched_pairs(
        e2b_weighted, "broad_related", "broad_unrelated", topology="clientlt",
        tolerance=tolerance, effect_prefix="broad_related_minus_unrelated",
    )
    write_csv(args.output_dir / "e2b_accuracy_matched_access_effects.csv", access_rows)
    write_csv(args.output_dir / "e2b_accuracy_matched_specificity_effects.csv", specificity_rows)

    primary_column = "broad_related_minus_narrow_worst_neighbor_margin_gain"
    specificity_column = "broad_related_minus_unrelated_worst_neighbor_margin_gain"
    access_bootstrap = _bootstrap_effect(access_rows, primary_column, args.bootstrap_draws)
    specificity_bootstrap = _bootstrap_effect(specificity_rows, specificity_column, args.bootstrap_draws)
    matched_access = [row for row in access_rows if row["accuracy_matched"]]
    matched_specificity = [row for row in specificity_rows if row["accuracy_matched"]]
    class_access_means = {
        class_id: float(np.mean([row[primary_column] for row in matched_access if int(row["tail_class"]) == class_id]))
        for class_id in TAIL_CLASSES
        if any(int(row["tail_class"]) == class_id for row in matched_access)
    }
    positive_classes = sum(value > 0 for value in class_access_means.values())

    intervention = pd.read_csv(Path(args.manifest_dir) / "e2b_intervention_manifest.csv")
    related_pivot = intervention.pivot_table(
        index=["data_seed", "client_id"], columns="condition",
        values="tail_mass_weighted_mean_relatedness", aggfunc="first",
    ).reset_index()
    max_relatedness_gap = float(np.max(np.abs(
        related_pivot.broad_related.to_numpy() - related_pivot.narrow_related.to_numpy()
    )))
    relatedness_matched = max_relatedness_gap <= float(
        frozen_protocol()["e2b"]["relatedness_match_tolerance"]
    )
    unrelated_is_lower = bool((related_pivot.broad_related > related_pivot.broad_unrelated).all())

    difficulty_smd_rows = []
    for (seed, client_id), group in e2b_difficulty.groupby(["data_seed", "client_id"], sort=True):
        narrow_values = group[group.condition == "narrow_related"].theta0_nll.to_numpy(dtype=np.float64)
        broad_values = group[group.condition == "broad_related"].theta0_nll.to_numpy(dtype=np.float64)
        if not len(narrow_values) or not len(broad_values):
            raise RuntimeError(f"Missing E2B companion difficulty rows for seed={seed}, client={client_id}")
        pooled = math.sqrt((float(np.var(narrow_values)) + float(np.var(broad_values))) / 2.0)
        smd = (float(np.mean(broad_values)) - float(np.mean(narrow_values))) / max(pooled, 1e-12)
        difficulty_smd_rows.append({
            "data_seed": int(seed), "client_id": int(client_id),
            "narrow_theta0_nll_mean": float(np.mean(narrow_values)),
            "broad_related_theta0_nll_mean": float(np.mean(broad_values)),
            "broad_minus_narrow_nll_smd": float(smd), "absolute_smd": abs(float(smd)),
        })
    write_csv(args.output_dir / "e2b_companion_difficulty_balance.csv", difficulty_smd_rows)
    difficulty_threshold = float(
        frozen_protocol()["e2b"]["initial_companion_difficulty_control"]
        ["maximum_absolute_standardized_mean_difference"]
    )
    max_difficulty_smd = max(row["absolute_smd"] for row in difficulty_smd_rows)
    difficulty_balanced = max_difficulty_smd <= difficulty_threshold

    access_by_condition = e2b_weighted[e2b_weighted.local_epoch == 0].groupby("condition")[
        "tail_neighbor_access_score"
    ].mean().to_dict()
    access_manipulation = bool(
        access_by_condition.get("broad_related", -np.inf)
        > access_by_condition.get("narrow_related", np.inf)
    )

    required_pairs = 15 * max(1, len(set(e2b_weighted.data_seed.tolist())))
    sufficient_matching = len(matched_access) >= required_pairs and len(matched_specificity) >= required_pairs
    primary_positive = access_bootstrap["mean"] > 0
    primary_ci_positive = access_bootstrap["ci_low"] > 0
    majority_positive = positive_classes >= 12
    specificity_positive = specificity_bootstrap["mean"] > 0
    causal_gate = all([
        relatedness_matched, unrelated_is_lower, access_manipulation,
        difficulty_balanced,
        sufficient_matching, primary_positive, primary_ci_positive,
        majority_positive, specificity_positive,
    ])
    if causal_gate:
        verdict = "CAUSAL_SEMANTIC_ACCESS_HARM_SUPPORTED"
    elif not difficulty_balanced:
        verdict = "COMPANION_DIFFICULTY_CONFOUNDED"
    elif not sufficient_matching:
        verdict = "INVALID_ACCURACY_MATCHED_COMPARISON"
    elif primary_positive and majority_positive:
        verdict = "EXPLORATORY_SEMANTIC_ACCESS_SIGNAL"
    else:
        verdict = "NO_CAUSAL_SEMANTIC_ACCESS_SUPPORT"

    # E2A remains explicitly descriptive. At epoch 3 compare natural local
    # footprints after tail-mass weighting; no causal verdict is attached.
    e2a_final = e2a_weighted[e2a_weighted.local_epoch == 3]
    e2a_pairs = []
    for (seed, tail_class), group in e2a_final.groupby(["data_seed", "tail_class"], sort=True):
        d = group[group.topology == "dirichlet"].iloc[0]
        c = group[group.topology == "clientlt"].iloc[0]
        e2a_pairs.append({
            "data_seed": int(seed), "tail_class": int(tail_class),
            "clientlt_minus_dirichlet_margin_gain": float(c.margin_gain - d.margin_gain),
            "dirichlet_minus_clientlt_worst_neighbor_margin_gain": float(
                d.worst_neighbor_margin_gain - c.worst_neighbor_margin_gain
            ),
            "dirichlet_minus_clientlt_neighbor_access": float(
                d.tail_neighbor_access_score - c.tail_neighbor_access_score
            ),
            "clientlt_support_clients": int(c.support_client_count),
            "dirichlet_support_clients": int(d.support_client_count),
        })
    write_csv(args.output_dir / "e2a_natural_partition_paired_effects.csv", e2a_pairs)
    e2a_local_stronger = float(np.mean([x["clientlt_minus_dirichlet_margin_gain"] for x in e2a_pairs])) > 0
    e2a_local_narrower = float(np.mean([x["dirichlet_minus_clientlt_worst_neighbor_margin_gain"] for x in e2a_pairs])) > 0

    summary = {
        "scope": "preaggregation_client_local_functional_audit",
        "server_aggregation_used": False,
        "verdict": verdict,
        "causal_gate_pass": causal_gate,
        "e2a_descriptive": {
            "clientlt_local_direct_margin_gain_stronger": e2a_local_stronger,
            "dirichlet_local_worst_neighbor_gain_broader": e2a_local_narrower,
            "paired_seed_class_count": len(e2a_pairs),
            "mean_clientlt_minus_dirichlet_margin_gain": float(np.mean([
                x["clientlt_minus_dirichlet_margin_gain"] for x in e2a_pairs
            ])),
            "mean_dirichlet_minus_clientlt_worst_neighbor_margin_gain": float(np.mean([
                x["dirichlet_minus_clientlt_worst_neighbor_margin_gain"] for x in e2a_pairs
            ])),
            "interpretation": "association_only_because_natural_client_data_and_step_counts_differ",
        },
        "e2b_causal": {
            "primary_endpoint": "accuracy_matched_worst_neighbor_margin_gain",
            "access_effect": access_bootstrap,
            "specificity_effect": specificity_bootstrap,
            "positive_tail_classes": positive_classes,
            "required_positive_tail_classes": 12,
            "accuracy_matched_access_units": len(matched_access),
            "accuracy_matched_specificity_units": len(matched_specificity),
            "required_matched_units": required_pairs,
            "max_narrow_broad_relatedness_gap": max_relatedness_gap,
            "relatedness_matched": relatedness_matched,
            "broad_unrelated_has_lower_relatedness": unrelated_is_lower,
            "maximum_companion_difficulty_absolute_smd": max_difficulty_smd,
            "companion_difficulty_smd_threshold": difficulty_threshold,
            "companion_initial_difficulty_balanced": difficulty_balanced,
            "broad_related_has_higher_neighbor_access_than_narrow": access_manipulation,
            "access_score_by_condition": {str(k): float(v) for k, v in access_by_condition.items()},
            "gate_checks": {
                "sufficient_accuracy_matching": sufficient_matching,
                "initial_companion_difficulty_balanced": difficulty_balanced,
                "primary_mean_positive": primary_positive,
                "primary_ci_low_positive": primary_ci_positive,
                "tail_class_majority_positive": majority_positive,
                "semantic_specificity_mean_positive": specificity_positive,
            },
        },
        "evidence_boundary": (
            "E2 tests local LoRA formation before aggregation. A positive E2B gate supports a "
            "controlled semantic-access effect under this CIFAR-100-LT vision-LoRA protocol, "
            "not a universal federated-learning claim."
        ),
    }
    write_json(args.output_dir / "e2_client_update_summary.json", summary)
    lines = [
        "# E2 client-local update audit", "",
        f"- Verdict: `{verdict}`",
        f"- Server aggregation used: `False`",
        f"- E2A Client-LT direct local gain stronger: `{e2a_local_stronger}`",
        f"- E2A Dirichlet local neighbor footprint broader: `{e2a_local_narrower}`",
        f"- E2B access effect mean [95% CI]: `{access_bootstrap['mean']:.6f}` "
        f"[`{access_bootstrap['ci_low']:.6f}`, `{access_bootstrap['ci_high']:.6f}`]",
        f"- E2B positive tail classes: `{positive_classes}/20`",
        f"- Accuracy-matched E2B units: `{len(matched_access)}/{required_pairs}` required",
        "",
        "E2A is descriptive; only E2B is allowed to support the causal semantic-access statement.",
    ]
    (args.output_dir / "e2_client_update_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--e2a-dir", type=Path, required=True)
    parser.add_argument("--e2b-dir", type=Path, required=True)
    parser.add_argument("--manifest-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-draws", type=int, default=10000)
    args = parser.parse_args()
    summary = summarize(args)
    print(json.dumps({
        "verdict": summary["verdict"], "causal_gate_pass": summary["causal_gate_pass"],
        "output_dir": str(args.output_dir.resolve()),
    }))


if __name__ == "__main__":
    main()
