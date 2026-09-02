from __future__ import annotations

import itertools
import math

import numpy as np


EPS = 1e-12


def coverage_metrics(gains) -> dict[str, float]:
    values = np.asarray(gains, dtype=np.float64)
    positive = np.maximum(values, 0.0)
    strength = float(positive.sum())
    if strength <= EPS:
        breadth = 0.0
        entropy = 0.0
        coverage = 0
    else:
        probabilities = positive / strength
        nonzero = probabilities[probabilities > 0]
        entropy = float(-(nonzero * np.log(nonzero)).sum())
        breadth = float(math.exp(entropy))
        coverage = int((positive > 0).sum())
    return {
        "positive_strength": strength,
        "effective_breadth": breadth,
        "positive_boundary_count": coverage,
        "negative_boundary_harm": float(np.maximum(-values, 0.0).sum()),
    }


def symmetric_relative_gap(left: float, right: float) -> float:
    return float(2.0 * abs(left - right) / (abs(left) + abs(right) + EPS))


def enumerate_pair_screen(
    tail_class: int,
    boundary_vectors: dict[int, np.ndarray],
    pair_norms: dict[tuple[int, int], float],
    head_margin_gains: dict[int, float],
    direct_cosines: dict[int, float],
    candidate_samples: dict[int, int],
    candidate_steps: dict[int, int],
) -> list[dict]:
    rows = []
    for left, right in itertools.combinations(sorted(boundary_vectors), 2):
        vector = 0.5 * (boundary_vectors[left] + boundary_vectors[right])
        metrics = coverage_metrics(vector)
        rows.append({
            "tail_class": int(tail_class), "candidate_a": left, "candidate_b": right,
            "donor_count": 2, "predicted_update_l2": pair_norms[(left, right)],
            "predicted_head_margin_gain": 0.5 * (head_margin_gains[left] + head_margin_gains[right]),
            "predicted_direct_tail_cosine": 0.5 * (direct_cosines[left] + direct_cosines[right]),
            "candidate_sample_count": candidate_samples[left] + candidate_samples[right],
            "optimizer_steps": candidate_steps[left] + candidate_steps[right],
            **{f"predicted_{key}": value for key, value in metrics.items()},
        })
    return rows


def constraint_aware_shortlist(
    rows: list[dict], count: int, protocol: dict
) -> tuple[list[dict], dict]:
    """Search the complete pair-state contrast space before actual evaluation.

    V2 compared only the lowest and highest breadth quartiles.  That heuristic
    systematically coupled breadth with positive strength.  V3 instead applies
    the preregistered scientific constraints as hard filters to every possible
    Broad/Narrow contrast.  The filters use predicted private-train quantities
    only; every shortlisted contrast is checked again with actual merged forward
    passes by :func:`select_actual_match`.
    """

    if not rows:
        return [], {
            "tail_class": "", "pair_states_screened": 0,
            "pair_state_contrasts_possible": 0,
            "predicted_feasible_contrasts": 0, "shortlisted_contrasts": 0,
            "shortlisted_unique_pair_states": 0,
        }
    count = int(count)
    if count < 1:
        raise ValueError("constraint-aware shortlist count must be positive")

    thresholds = protocol["matching"]
    strength_max = float(thresholds["actual_symmetric_relative_strength_gap_max"])
    norm_max = float(thresholds["relative_update_norm_gap_max"])
    head_max = float(thresholds["absolute_head_margin_gain_gap_max"])
    cosine_max = float(thresholds["absolute_direct_tail_cosine_gap_max"])
    breadth_min = float(thresholds["minimum_actual_effective_breadth_gap"])

    breadth = np.asarray(
        [float(row["predicted_effective_breadth"]) for row in rows], dtype=np.float64
    )
    strength = np.asarray(
        [float(row["predicted_positive_strength"]) for row in rows], dtype=np.float64
    )
    update_norm = np.asarray(
        [float(row["predicted_update_l2"]) for row in rows], dtype=np.float64
    )
    head_gain = np.asarray(
        [float(row["predicted_head_margin_gain"]) for row in rows], dtype=np.float64
    )
    cosine = np.asarray(
        [float(row["predicted_direct_tail_cosine"]) for row in rows], dtype=np.float64
    )
    donor_count = np.asarray([int(row["donor_count"]) for row in rows], dtype=np.int64)
    sample_count = np.asarray(
        [int(row["candidate_sample_count"]) for row in rows], dtype=np.int64
    )
    optimizer_steps = np.asarray(
        [int(row["optimizer_steps"]) for row in rows], dtype=np.int64
    )

    feasible = []
    for broad_index, broad_row in enumerate(rows):
        breadth_gap = breadth[broad_index] - breadth
        strength_gap = (
            2.0 * np.abs(strength[broad_index] - strength)
            / (np.abs(strength[broad_index]) + np.abs(strength) + EPS)
        )
        norm_gap = (
            2.0 * np.abs(update_norm[broad_index] - update_norm)
            / (np.abs(update_norm[broad_index]) + np.abs(update_norm) + EPS)
        )
        head_gap = np.abs(head_gain[broad_index] - head_gain)
        cosine_gap = np.abs(cosine[broad_index] - cosine)
        budget_match = (
            (donor_count[broad_index] == donor_count)
            & (sample_count[broad_index] == sample_count)
            & (optimizer_steps[broad_index] == optimizer_steps)
        )
        mask = (
            (breadth_gap >= breadth_min)
            & (strength_gap <= strength_max)
            & (norm_gap <= norm_max)
            & (head_gap <= head_max)
            & (cosine_gap <= cosine_max)
            & budget_match
        )
        for narrow_index in np.flatnonzero(mask).tolist():
            narrow_row = rows[narrow_index]
            normalized_utilization = (
                float(strength_gap[narrow_index] / strength_max),
                float(norm_gap[narrow_index] / norm_max),
                float(head_gap[narrow_index] / head_max),
                float(cosine_gap[narrow_index] / cosine_max),
            )
            max_utilization = max(normalized_utilization)
            feasible.append({
                "tail_class": int(broad_row["tail_class"]),
                "broad_a": int(broad_row["candidate_a"]),
                "broad_b": int(broad_row["candidate_b"]),
                "narrow_a": int(narrow_row["candidate_a"]),
                "narrow_b": int(narrow_row["candidate_b"]),
                "predicted_breadth_gap": float(breadth_gap[narrow_index]),
                "predicted_strength_symmetric_relative_gap": float(
                    strength_gap[narrow_index]
                ),
                "predicted_update_norm_symmetric_relative_gap": float(
                    norm_gap[narrow_index]
                ),
                "predicted_head_margin_gain_gap": float(head_gap[narrow_index]),
                "predicted_direct_tail_cosine_gap": float(cosine_gap[narrow_index]),
                "predicted_constraint_max_utilization": max_utilization,
                "predicted_minimum_constraint_slack": 1.0 - max_utilization,
                "predicted_match_distance": float(
                    math.sqrt(sum(value * value for value in normalized_utilization))
                ),
                "predicted_budget_match": True,
                "predicted_constraints_pass": True,
            })

    deterministic_key = lambda row: (
        int(row["broad_a"]), int(row["broad_b"]),
        int(row["narrow_a"]), int(row["narrow_b"]),
    )
    breadth_order = sorted(feasible, key=lambda row: (
        -float(row["predicted_breadth_gap"]),
        -float(row["predicted_minimum_constraint_slack"]),
        float(row["predicted_match_distance"]),
        deterministic_key(row),
    ))
    robustness_order = sorted(feasible, key=lambda row: (
        -float(row["predicted_minimum_constraint_slack"]),
        -float(row["predicted_breadth_gap"]),
        float(row["predicted_match_distance"]),
        deterministic_key(row),
    ))

    # Interleave two deterministic lanes: large separations make the feasibility
    # contrast meaningful, while large constraint slack supplies backups that are
    # less likely to cross a threshold after the actual merged forward pass.
    output, seen = [], set()
    cursors = {"breadth": 0, "robustness": 0}
    orders = {"breadth": breadth_order, "robustness": robustness_order}
    while len(output) < min(count, len(feasible)):
        made_progress = False
        for lane in ("breadth", "robustness"):
            ordered = orders[lane]
            while cursors[lane] < len(ordered):
                row = ordered[cursors[lane]]
                cursors[lane] += 1
                key = deterministic_key(row)
                if key in seen:
                    continue
                selected = dict(row)
                selected["selection_lane"] = lane
                selected["selection_rank"] = len(output) + 1
                selected["shortlist_score"] = float(row["predicted_breadth_gap"])
                output.append(selected)
                seen.add(key)
                made_progress = True
                break
            if len(output) >= min(count, len(feasible)):
                break
        if not made_progress:
            break

    selected_pairs = {
        tuple(sorted(pair))
        for proposal in output
        for pair in (
            (proposal["broad_a"], proposal["broad_b"]),
            (proposal["narrow_a"], proposal["narrow_b"]),
        )
    }
    selected_gaps = [float(row["predicted_breadth_gap"]) for row in output]
    audit = {
        "tail_class": int(rows[0]["tail_class"]),
        "pair_states_screened": len(rows),
        "pair_state_contrasts_possible": len(rows) * (len(rows) - 1) // 2,
        "predicted_feasible_contrasts": len(feasible),
        "shortlisted_contrasts": len(output),
        "shortlisted_unique_pair_states": len(selected_pairs),
        "shortlisted_breadth_gap_min": min(selected_gaps) if selected_gaps else "",
        "shortlisted_breadth_gap_median": (
            float(np.median(selected_gaps)) if selected_gaps else ""
        ),
        "shortlisted_breadth_gap_max": max(selected_gaps) if selected_gaps else "",
    }
    return output, audit


def evaluate_actual_matches(
    proposals: list[dict], actual: dict[tuple[int, int, int], dict], protocol: dict
) -> list[dict]:
    thresholds = protocol["matching"]
    evaluated = []
    for proposal in proposals:
        tail = int(proposal["tail_class"])
        broad_key = (tail, min(proposal["broad_a"], proposal["broad_b"]), max(proposal["broad_a"], proposal["broad_b"]))
        narrow_key = (tail, min(proposal["narrow_a"], proposal["narrow_b"]), max(proposal["narrow_a"], proposal["narrow_b"]))
        broad, narrow = actual[broad_key], actual[narrow_key]
        checks = {
            "strength_match": symmetric_relative_gap(
                broad["actual_positive_strength"], narrow["actual_positive_strength"]
            ) <= thresholds["actual_symmetric_relative_strength_gap_max"],
            "norm_match": symmetric_relative_gap(
                broad["update_l2"], narrow["update_l2"]
            ) <= thresholds["relative_update_norm_gap_max"],
            "head_match": abs(
                broad["actual_head_margin_gain"] - narrow["actual_head_margin_gain"]
            ) <= thresholds["absolute_head_margin_gain_gap_max"],
            "cosine_match": abs(
                broad["direct_tail_cosine"] - narrow["direct_tail_cosine"]
            ) <= thresholds["absolute_direct_tail_cosine_gap_max"],
            "breadth_separated": (
                broad["actual_effective_breadth"] - narrow["actual_effective_breadth"]
            ) >= thresholds["minimum_actual_effective_breadth_gap"],
            "budget_match": (
                broad["candidate_sample_count"] == narrow["candidate_sample_count"]
                and broad["optimizer_steps"] == narrow["optimizer_steps"]
            ),
        }
        breadth_gap = broad["actual_effective_breadth"] - narrow["actual_effective_breadth"]
        evaluated.append({
            **proposal,
            "broad_actual_positive_strength": broad["actual_positive_strength"],
            "narrow_actual_positive_strength": narrow["actual_positive_strength"],
            "broad_actual_effective_breadth": broad["actual_effective_breadth"],
            "narrow_actual_effective_breadth": narrow["actual_effective_breadth"],
            "actual_effective_breadth_gap": breadth_gap,
            "actual_strength_symmetric_relative_gap": symmetric_relative_gap(
                broad["actual_positive_strength"], narrow["actual_positive_strength"]
            ),
            "actual_update_norm_symmetric_relative_gap": symmetric_relative_gap(
                broad["update_l2"], narrow["update_l2"]
            ),
            "actual_head_margin_gain_gap": abs(
                broad["actual_head_margin_gain"] - narrow["actual_head_margin_gain"]
            ),
            "actual_direct_tail_cosine_gap": abs(
                broad["direct_tail_cosine"] - narrow["direct_tail_cosine"]
            ),
            **checks, "matched_pair_pass": all(checks.values()),
            "selection_used_test_metrics": False,
        })
    return evaluated


def select_evaluated_actual_match(evaluated: list[dict]) -> dict:
    if not evaluated:
        raise ValueError("No constraint-aware proposals were available for actual selection")
    passing = [row for row in evaluated if row["matched_pair_pass"]]
    chosen = max(passing, key=lambda row: row["actual_effective_breadth_gap"]) if passing else max(
        evaluated, key=lambda row: (
            sum(bool(row[name]) for name in (
                "strength_match", "norm_match", "head_match", "cosine_match", "breadth_separated", "budget_match"
            )), row["actual_effective_breadth_gap"],
        )
    )
    return chosen


def select_actual_match(
    proposals: list[dict], actual: dict[tuple[int, int, int], dict], protocol: dict
) -> dict:
    return select_evaluated_actual_match(
        evaluate_actual_matches(proposals, actual, protocol)
    )
