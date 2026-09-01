from __future__ import annotations

import math

import numpy as np


EPS = 1e-12


def breadth_metrics(boundary_gains) -> dict[str, float | int]:
    values = np.asarray(boundary_gains, dtype=np.float64)
    if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all():
        raise ValueError("boundary_gains must be a non-empty finite vector")
    positive = np.maximum(values, 0.0)
    strength = float(positive.sum())
    if strength <= EPS:
        entropy, effective = 0.0, 0.0
    else:
        probabilities = positive[positive > 0] / strength
        entropy = float(-(probabilities * np.log(probabilities)).sum())
        effective = float(math.exp(entropy))
    return {
        "positive_strength": strength,
        "coverage_entropy": entropy,
        "effective_breadth": effective,
        "positive_boundary_count": int((values > 0).sum()),
        "worst_boundary_gain": float(values.min()),
        "negative_boundary_harm": float(np.maximum(-values, 0.0).sum()),
    }


def potential_pool_metrics(gain_matrix, weights) -> dict[str, float | int]:
    gains = np.asarray(gain_matrix, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if gains.ndim != 2 or weights.ndim != 1 or gains.shape[0] != len(weights):
        raise ValueError("gain_matrix/weights shape mismatch")
    if gains.shape[1] == 0 or np.any(weights < 0) or not np.isfinite(gains).all():
        raise ValueError("invalid pool inputs")
    if float(weights.sum()) <= 0:
        return {
            "positive_donor_count": 0, "mean_positive_donors_per_boundary": 0.0,
            **{f"potential_{key}": value for key, value in breadth_metrics(np.zeros(gains.shape[1])).items()},
        }
    normalized = weights / weights.sum()
    positive_only_vector = (normalized[:, None] * np.maximum(gains, 0.0)).sum(axis=0)
    return {
        "positive_donor_count": int(np.any(gains > 0, axis=1).sum()),
        "mean_positive_donors_per_boundary": float((gains > 0).sum(axis=0).mean()),
        **{f"potential_{key}": value for key, value in breadth_metrics(positive_only_vector).items()},
    }

