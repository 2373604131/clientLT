from __future__ import annotations

import math
from typing import Iterable, Sequence

import numpy as np


def weighted_mean(values: Sequence[float], weights: Sequence[float]) -> float:
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if values.size == 0 or values.shape != weights.shape or float(weights.sum()) <= 0:
        raise ValueError("weighted_mean received invalid values or weights")
    return float(np.dot(values, weights / weights.sum()))


def normalized_positive_entropy(values: Sequence[float]) -> float:
    positive = np.maximum(np.asarray(values, dtype=np.float64), 0.0)
    if positive.size <= 1 or float(positive.sum()) <= 0:
        return 0.0
    probabilities = positive / positive.sum()
    nonzero = probabilities > 0
    return float(-(probabilities[nonzero] * np.log(probabilities[nonzero])).sum() / math.log(positive.size))


def weighted_pairwise_cosine_diversity(vectors: Sequence[Sequence[float]], weights: Sequence[float]) -> float:
    matrix = np.asarray(vectors, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != weights.size:
        raise ValueError("vectors and weights do not align")
    if matrix.shape[0] < 2:
        return 0.0
    weights = weights / weights.sum()
    numerator = 0.0
    denominator = 0.0
    for left in range(matrix.shape[0]):
        for right in range(left + 1, matrix.shape[0]):
            pair_weight = float(weights[left] * weights[right])
            left_norm = float(np.linalg.norm(matrix[left]))
            right_norm = float(np.linalg.norm(matrix[right]))
            cosine = 0.0 if left_norm == 0 or right_norm == 0 else float(
                np.dot(matrix[left], matrix[right]) / (left_norm * right_norm)
            )
            numerator += pair_weight * (1.0 - cosine)
            denominator += pair_weight
    return float(numerator / denominator) if denominator > 0 else 0.0


def effective_count(weights: Sequence[float]) -> float:
    weights = np.asarray(weights, dtype=np.float64)
    if weights.size == 0 or float(weights.sum()) <= 0:
        return 0.0
    probabilities = weights / weights.sum()
    return float(1.0 / np.square(probabilities).sum())


def rankdata(values: Sequence[float]) -> np.ndarray:
    """Average ranks for ties, matching scipy.stats.rankdata(method='average')."""
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=np.float64)
    start = 0
    while start < values.size:
        end = start + 1
        while end < values.size and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1) + 1.0
        start = end
    return ranks


def spearman(left: Sequence[float], right: Sequence[float]) -> float:
    left_rank, right_rank = rankdata(left), rankdata(right)
    if left_rank.size < 2 or float(left_rank.std()) == 0 or float(right_rank.std()) == 0:
        return 0.0
    return float(np.corrcoef(left_rank, right_rank)[0, 1])


def summarize(values: Iterable[float]) -> dict:
    array = np.asarray(list(values), dtype=np.float64)
    if array.size == 0:
        return {"count": 0, "mean": 0.0, "median": 0.0, "positive_count": 0, "positive_fraction": 0.0}
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "positive_count": int((array > 0).sum()),
        "positive_fraction": float(np.mean(array > 0)),
    }
