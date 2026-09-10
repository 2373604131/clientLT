#!/usr/bin/env python
"""Turn a SelectiveSync-V1 run into numerical answers to the audit questions."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np


def read_csv(path: Path):
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def number(row, key):
    value = row.get(key, "")
    return float(value) if value not in ("", None) else math.nan


def mean(values):
    finite = []
    for value in values:
        if value is None:
            continue
        value = float(value)
        if math.isfinite(value):
            finite.append(value)
    return float(np.mean(finite)) if finite else math.nan


def ratio(numerator, denominator):
    return float(numerator / denominator) if denominator > 0 else math.nan


def correlation(left, right, method="pearson"):
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    valid = np.isfinite(left) & np.isfinite(right)
    left, right = left[valid], right[valid]
    if len(left) < 2 or np.std(left) == 0 or np.std(right) == 0:
        return math.nan
    if method == "spearman":
        left = np.argsort(np.argsort(left, kind="stable"), kind="stable")
        right = np.argsort(np.argsort(right, kind="stable"), kind="stable")
    return float(np.corrcoef(left, right)[0, 1])


def linear_slope(rows, field):
    x = np.asarray([number(row, "round") for row in rows], dtype=np.float64)
    y = np.asarray([number(row, field) for row in rows], dtype=np.float64)
    valid = np.isfinite(x) & np.isfinite(y)
    return float(np.polyfit(x[valid], y[valid], 1)[0]) if valid.sum() >= 2 else math.nan


def sign_reversal_rate(values):
    changes = np.diff(np.asarray(values, dtype=np.float64))
    signs = np.sign(changes[changes != 0])
    if len(signs) < 2:
        return math.nan
    return float(np.mean(signs[1:] != signs[:-1]))


def clean_json(value):
    if isinstance(value, dict):
        return {key: clean_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [clean_json(item) for item in value]
    if isinstance(value, (float, np.floating)) and not math.isfinite(float(value)):
        return None
    if isinstance(value, (np.integer,)):
        return int(value)
    return value


def summarize_run(run_dir: Path):
    sync_dir = run_dir / "selective_sync"
    config = json.loads((sync_dir / "config.json").read_text(encoding="utf-8"))
    rounds = read_csv(run_dir / "round_metrics.csv")
    classes = read_csv(sync_dir / "class_functional_metrics.csv")
    clients = read_csv(sync_dir / "client_sync_metrics.csv")
    states = read_csv(sync_dir / "round_state_metrics.csv")
    access = read_csv(sync_dir / "server_access_metrics.csv")
    capacity_path = sync_dir / "fixed_a_capacity.csv"
    capacity = read_csv(capacity_path) if capacity_path.exists() else []

    final_round = rounds[-1]
    tail_curve = [number(row, "bottom20_tail_acc") for row in rounds]
    performance = {
        "final_round": int(number(final_round, "epoch")) + 1,
        "final_overall_acc": number(final_round, "overall_acc"),
        "final_tail_acc": number(final_round, "bottom20_tail_acc"),
        "best_tail_acc": float(np.max(tail_curve)),
        "best_tail_round": int(number(rounds[int(np.argmax(tail_curve))], "epoch")) + 1,
    }

    capacity_by_round = {}
    for round_id in sorted({int(number(row, "round")) for row in capacity}):
        selected = [row for row in capacity if int(number(row, "round")) == round_id]
        full = sum(number(row, "full_gradient_energy") for row in selected)
        projected = sum(number(row, "projected_gradient_energy") for row in selected)
        ratios = [number(row, "capture_ratio") for row in selected]
        capacity_by_round[str(round_id)] = {
            "energy_weighted_capture_ratio": ratio(projected, full),
            "mean_module_capture_ratio": mean(ratios),
            "minimum_module_capture_ratio": min(ratios) if ratios else math.nan,
            "mean_A_rank": mean(number(row, "a_rank") for row in selected),
            "measurements": len(selected),
        }
    all_full = sum(number(row, "full_gradient_energy") for row in capacity)
    all_projected = sum(number(row, "projected_gradient_energy") for row in capacity)

    probability_harm = [number(row, "raw_harm") for row in classes]
    margin_harm = [number(row, "raw_margin_harm") for row in classes]
    epsilon = float(config["harm_epsilon"])
    harm_sign_agreement = mean(
        (probability > epsilon) == (margin_value > 0)
        for probability, margin_value in zip(probability_harm, margin_harm)
    )
    grouped_classes = defaultdict(list)
    for row in classes:
        grouped_classes[(int(number(row, "round")), int(number(row, "client_id")))].append(row)
    top_l_overlaps = []
    for group in grouped_classes.values():
        probability_set = {
            int(number(row, "class_id")) for row in group if int(number(row, "selected")) == 1
        }
        margin_candidates = sorted(
            (row for row in group if number(row, "raw_margin_harm") > 0),
            key=lambda row: (-number(row, "raw_margin_harm"), int(number(row, "class_id"))),
        )[: int(config["top_l"])]
        margin_set = {int(number(row, "class_id")) for row in margin_candidates}
        union = probability_set | margin_set
        top_l_overlaps.append(ratio(len(probability_set & margin_set), len(union)))

    selected_rows = [row for row in classes if int(number(row, "selected")) == 1]
    raw_harm_total = sum(number(row, "raw_harm") for row in classes)
    selected_raw_harm = sum(number(row, "raw_harm") for row in selected_rows)
    accepted_harm_total = sum(number(row, "accepted_harm") for row in classes)
    selected_accepted_harm = sum(number(row, "accepted_harm") for row in selected_rows)
    selected_counts = [number(row, "selected_class_count") for row in clients]
    harmed_counts = [number(row, "raw_harmed_class_count") for row in clients]
    protection = {
        "configured_L": int(config["top_l"]),
        "mean_selected_classes": mean(selected_counts),
        "mean_raw_harmed_classes": mean(harmed_counts),
        "budget_saturation_rate": mean(
            harmed >= int(config["top_l"]) for harmed in harmed_counts
        ),
        "raw_harm_mass_covered_by_TopL": ratio(selected_raw_harm, raw_harm_total),
        "selected_constraint_success_rate": mean(
            number(row, "selected_success_rate") for row in clients
        ),
    }

    backtracks = [number(row, "backtracks") for row in clients]
    kappa_path = {
        "configured_kappa": float(config["success_kappa"]),
        "selected_accepted_to_raw_harm_ratio": ratio(
            selected_accepted_harm, selected_raw_harm
        ),
        "all_class_damage_reduction": 1.0 - ratio(accepted_harm_total, raw_harm_total),
        "clients_backtracked_rate": mean(value > 0 for value in backtracks),
        "mean_backtracks": mean(backtracks),
        "zero_receive_scale_rate": mean(
            number(row, "accepted_scale") == 0 for row in clients
        ),
        "mean_accepted_scale": mean(number(row, "accepted_scale") for row in clients),
    }

    tail_by_class = {
        int(number(row, "class_id")): bool(int(number(row, "tail20"))) for row in access
    }

    def donor_metrics(rows_subset):
        donor = [
            row for row in rows_subset
            if number(row, "raw_score") > number(row, "private_score")
        ]
        raw_gain = sum(
            number(row, "raw_score") - number(row, "private_score") for row in donor
        )
        accepted_gain = sum(
            max(number(row, "accepted_score") - number(row, "private_score"), 0.0)
            for row in donor
        )
        postlocal_gain = sum(
            max(number(row, "postlocal_score") - number(row, "private_score"), 0.0)
            for row in donor
        )
        return {
            "raw_donor_gain_mass": raw_gain,
            "accepted_donor_gain_mass": accepted_gain,
            "accepted_donor_gain_retention": ratio(accepted_gain, raw_gain),
            "postlocal_donor_gain_retention": ratio(postlocal_gain, raw_gain),
            "donor_class_observations": len(donor),
        }

    donor = donor_metrics(classes)
    donor["tail20"] = donor_metrics(
        [row for row in classes if tail_by_class.get(int(number(row, "class_id")), False)]
    )

    final_state = states[-1]
    divergence = {
        "final_mean_pairwise_B_distance": number(final_state, "mean_pairwise_private_distance"),
        "peak_mean_pairwise_B_distance": max(
            number(row, "mean_pairwise_private_distance") for row in states
        ),
        "per_round_distance_slope": linear_slope(states, "mean_pairwise_private_distance"),
        "final_mean_pairwise_B_cosine": number(final_state, "mean_pairwise_private_cosine"),
        "final_rms_private_global_disagreement": number(
            final_state, "rms_private_global_disagreement"
        ),
        "final_disagreement_over_global_B_norm": ratio(
            number(final_state, "rms_private_global_disagreement"),
            number(final_state, "global_b_norm"),
        ),
        "final_mean_private_B_rank": number(final_state, "mean_private_b_rank"),
    }

    step_cosines = [
        number(row, "global_step_cosine_previous") for row in states
        if math.isfinite(number(row, "global_step_cosine_previous"))
    ]
    oscillation = {
        "mean_global_B_step_norm": mean(number(row, "global_step_norm") for row in states),
        "max_global_B_step_norm": max(number(row, "global_step_norm") for row in states),
        "final_global_B_step_norm": number(final_state, "global_step_norm"),
        "mean_consecutive_step_cosine": mean(step_cosines),
        "opposite_step_rate": mean(value < 0 for value in step_cosines),
        "tail_accuracy_sign_reversal_rate": sign_reversal_rate(tail_curve),
        "tail_accuracy_last20_std": float(np.std(tail_curve[-20:])),
    }

    access_by_round = defaultdict(dict)
    for row in access:
        access_by_round[int(number(row, "round"))][int(number(row, "class_id"))] = number(
            row, "supporter_access"
        )
    first_access = access_by_round[min(access_by_round)]
    final_access = access_by_round[max(access_by_round)]
    access_deltas = {
        class_id: final_access[class_id] - value for class_id, value in first_access.items()
    }
    low_count = max(int(round(0.2 * len(first_access))), 1)
    low_access_classes = sorted(first_access, key=first_access.get)[:low_count]
    access_summary = {
        "server_weighting": config["server_aggregation"],
        "dynamic_alpha_enabled": False,
        "mean_absolute_access_change": mean(abs(value) for value in access_deltas.values()),
        "maximum_absolute_access_change": max(abs(value) for value in access_deltas.values()),
        "low_access_mean_initial_Ac": mean(first_access[c] for c in low_access_classes),
        "low_access_mean_final_Ac": mean(final_access[c] for c in low_access_classes),
        "low_access_mean_Ac_change": mean(access_deltas[c] for c in low_access_classes),
    }

    effective_counts = [number(row, "effective_client_count") for row in states]
    neff = {
        "configured_Neff_lower_bound": None,
        "minimum_effective_client_count": min(effective_counts),
        "mean_effective_client_count": mean(effective_counts),
        "final_effective_client_count": effective_counts[-1],
        "physical_client_count": len({int(number(row, "client_id")) for row in clients}),
    }

    immediate_gain = {
        "raw_direct_fusion_harm_mass": raw_harm_total,
        "scheduled_fusion_harm_mass": accepted_harm_total,
        "functional_damage_reduction": 1.0 - ratio(accepted_harm_total, raw_harm_total),
        "donor_gain_retention": donor["accepted_donor_gain_retention"],
    }

    questions = {
        "1_fixed_A0_expressiveness": {
            "status": "proxy_answer_only",
            "metrics": {
                "all_round_energy_weighted_gradient_capture": ratio(all_projected, all_full),
                "by_round": capacity_by_round,
            },
            "interpretation": "This measures the CE-gradient energy representable by delta-B times fixed A0; sufficiency still needs a trainable-A matched control.",
        },
        "2_functional_score_definition": {
            "status": "definition_answered_metric_comparison_diagnostic",
            "answer": "correct_class_probability",
            "metrics": {
                "probability_margin_harm_pearson": correlation(probability_harm, margin_harm),
                "probability_margin_harm_spearman": correlation(
                    probability_harm, margin_harm, method="spearman"
                ),
                "harm_sign_agreement": harm_sign_agreement,
                "TopL_probability_margin_mean_jaccard": mean(top_l_overlaps),
            },
            "interpretation": "A causal choice between probability and normalized margin requires a matched margin-score rerun.",
        },
        "3_protected_class_budget_L": {
            "status": "answered_for_this_run",
            "metrics": protection,
        },
        "4_kappa_sensitivity": {
            "status": "path_answered_sensitivity_requires_multiple_kappa_runs",
            "metrics": kappa_path,
        },
        "5_donor_gain_retention": {
            "status": "answered_on_functional_memory",
            "metrics": donor,
        },
        "6_private_B_divergence": {
            "status": "trajectory_answered_no_failure_threshold_defined",
            "metrics": divergence,
        },
        "7_global_B_oscillation": {
            "status": "trajectory_answered_cause_requires_delta_aggregation_control",
            "metrics": oscillation,
        },
        "8_dynamic_alpha_and_low_access": {
            "status": "not_applicable_v1_uses_static_fedavg",
            "metrics": access_summary,
        },
        "9_effective_client_count": {
            "status": "measured_but_no_lower_bound_in_v1",
            "metrics": neff,
        },
        "10_independent_scheduler_gain": {
            "status": "immediate_effect_answered_end_to_end_gain_requires_controls",
            "metrics": immediate_gain,
            "interpretation": "Raw Bg-Bk is the within-run direct-fusion counterfactual before local training. Final accuracy attribution requires no-projection/direct-fusion and static-access matched runs.",
        },
    }
    return clean_json(
        {
            "run_dir": str(run_dir),
            "configuration": config,
            "performance": performance,
            "questions": questions,
        }
    )


def print_report(report):
    print(f"\nRun: {report['run_dir']}")
    print("Final:", json.dumps(report["performance"], ensure_ascii=False))
    for index, (name, answer) in enumerate(report["questions"].items(), start=1):
        print(f"Q{index} {name} [{answer['status']}]")
        print(json.dumps(answer.get("metrics", {}), ensure_ascii=False, sort_keys=True))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", action="append", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    reports = [summarize_run(path) for path in args.run_dir]
    for report, run_dir in zip(reports, args.run_dir):
        print_report(report)
        destination = run_dir / "selective_sync" / "audit_answers.json"
        destination.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(f"Saved: {destination}")

    kappas = defaultdict(list)
    for report in reports:
        kappas[report["configuration"]["success_kappa"]].append(report)
    comparison = {
        "run_count": len(reports),
        "distinct_kappas": sorted(kappas),
        "kappa_sensitivity_answerable": len(kappas) >= 2,
        "runs": [
            {
                "run_dir": report["run_dir"],
                "kappa": report["configuration"]["success_kappa"],
                **report["performance"],
            }
            for report in reports
        ],
    }
    if len(kappas) >= 2:
        kappa_rows = []
        for kappa, group in sorted(kappas.items()):
            kappa_rows.append(
                {
                    "kappa": kappa,
                    "mean_final_overall_acc": mean(
                        item["performance"]["final_overall_acc"] for item in group
                    ),
                    "mean_final_tail_acc": mean(
                        item["performance"]["final_tail_acc"] for item in group
                    ),
                    "mean_donor_gain_retention": mean(
                        item["questions"]["5_donor_gain_retention"]["metrics"][
                            "accepted_donor_gain_retention"
                        ]
                        for item in group
                    ),
                    "mean_zero_receive_scale_rate": mean(
                        item["questions"]["4_kappa_sensitivity"]["metrics"][
                            "zero_receive_scale_rate"
                        ]
                        for item in group
                    ),
                }
            )
        comparison["kappa_summary"] = kappa_rows
        comparison["kappa_observed_ranges"] = {
            "final_overall_acc": max(row["mean_final_overall_acc"] for row in kappa_rows)
            - min(row["mean_final_overall_acc"] for row in kappa_rows),
            "final_tail_acc": max(row["mean_final_tail_acc"] for row in kappa_rows)
            - min(row["mean_final_tail_acc"] for row in kappa_rows),
            "donor_gain_retention": max(row["mean_donor_gain_retention"] for row in kappa_rows)
            - min(row["mean_donor_gain_retention"] for row in kappa_rows),
        }
    if len(reports) > 1 or args.output is not None:
        output = args.output or Path("selective_sync_audit_comparison.json")
        output.write_text(
            json.dumps(clean_json(comparison), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"Saved comparison: {output}")


if __name__ == "__main__":
    main()
