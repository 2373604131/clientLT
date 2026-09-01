from __future__ import annotations

import itertools
import math
from collections import defaultdict

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


def shortlist_contrasts(rows: list[dict], count: int) -> list[dict]:
    if not rows:
        return []
    ordered = sorted(rows, key=lambda row: row["predicted_effective_breadth"])
    quartile = max(1, len(ordered) // 4)
    narrow, broad = ordered[:quartile], ordered[-quartile:]
    feature_names = (
        "predicted_positive_strength", "predicted_update_l2",
        "predicted_head_margin_gain", "predicted_direct_tail_cosine",
    )
    scales = {}
    for name in feature_names:
        values = np.asarray([float(row[name]) for row in rows], dtype=np.float64)
        scales[name] = float(values.std()) or 1.0
    proposals = []
    for broad_row in broad:
        compatible = [row for row in narrow if (
            row["candidate_sample_count"] == broad_row["candidate_sample_count"]
            and row["optimizer_steps"] == broad_row["optimizer_steps"]
        )]
        if not compatible:
            continue
        narrow_row = min(compatible, key=lambda row: sum(
            ((float(row[name]) - float(broad_row[name])) / scales[name]) ** 2
            for name in feature_names
        ))
        distance = math.sqrt(sum(
            ((float(narrow_row[name]) - float(broad_row[name])) / scales[name]) ** 2
            for name in feature_names
        ))
        gap = float(broad_row["predicted_effective_breadth"] - narrow_row["predicted_effective_breadth"])
        proposals.append({
            "tail_class": int(broad_row["tail_class"]),
            "broad_a": int(broad_row["candidate_a"]), "broad_b": int(broad_row["candidate_b"]),
            "narrow_a": int(narrow_row["candidate_a"]), "narrow_b": int(narrow_row["candidate_b"]),
            "predicted_breadth_gap": gap, "predicted_match_distance": distance,
            "shortlist_score": gap - 0.10 * distance,
        })
    proposals.sort(key=lambda row: (-row["shortlist_score"], row["predicted_match_distance"]))
    output, seen = [], set()
    for row in proposals:
        key = (row["broad_a"], row["broad_b"], row["narrow_a"], row["narrow_b"])
        if key in seen:
            continue
        seen.add(key)
        output.append(row)
        if len(output) >= int(count):
            break
    return output


def select_actual_match(proposals: list[dict], actual: dict[tuple[int, int, int], dict], protocol: dict) -> dict:
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
        })
    passing = [row for row in evaluated if row["matched_pair_pass"]]
    chosen = max(passing, key=lambda row: row["actual_effective_breadth_gap"]) if passing else max(
        evaluated, key=lambda row: (
            sum(bool(row[name]) for name in (
                "strength_match", "norm_match", "head_match", "cosine_match", "breadth_separated", "budget_match"
            )), row["actual_effective_breadth_gap"],
        )
    )
    return chosen
