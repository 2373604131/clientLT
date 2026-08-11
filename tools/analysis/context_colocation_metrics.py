"""Pure metrics for the P0/V1 context co-location audit.

These functions consume client-by-class counts only.  They do not import a
trainer, construct an optimizer, or mutate model parameters.
"""

from __future__ import annotations

import hashlib
import json
from typing import Iterable, Mapping, Sequence

import numpy as np


WEIGHTINGS = ("client_unweighted", "tail_mass_weighted", "fedavg_weighted")


def stable_sha256(value) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def support_weights(counts: np.ndarray, tail_class: int) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    counts = np.asarray(counts, dtype=np.float64)
    support = np.flatnonzero(counts[:, tail_class] > 0)
    if support.size == 0:
        raise ValueError(f"Tail class {tail_class} has empty support")
    class_mass = counts[support, tail_class]
    client_mass = counts[support].sum(axis=1)
    weights = {
        "client_unweighted": np.repeat(1.0 / support.size, support.size),
        "tail_mass_weighted": class_mass / class_mass.sum(),
        "fedavg_weighted": client_mass / client_mass.sum(),
    }
    return support, weights


def topology_metrics(counts: np.ndarray, tail_class: int) -> dict[str, float]:
    counts = np.asarray(counts, dtype=np.float64)
    support, _ = support_weights(counts, tail_class)
    mass = counts[support, tail_class]
    probabilities = mass / mass.sum()
    return {
        "support_client_count": float(support.size),
        "top2_tail_client_mass": float(np.sort(probabilities)[-2:].sum()),
        "effective_support_clients": float(1.0 / np.square(probabilities).sum()),
        "tail_mass_accounted_for": float(mass.sum() / counts[:, tail_class].sum()),
    }


def _weighted_pairwise(values: Sequence[float], pair_weights: Sequence[float]) -> float:
    if not values:
        return 0.0
    weights = np.asarray(pair_weights, dtype=np.float64)
    if weights.sum() <= 0:
        return 0.0
    return float(np.dot(np.asarray(values, dtype=np.float64), weights / weights.sum()))


def generic_context_metrics(
    counts: np.ndarray,
    tail_class: int,
    context_classes: Iterable[int],
    embeddings: np.ndarray | None = None,
) -> dict[str, float]:
    """Describe non-tail positive context on clients supporting ``tail_class``."""
    counts = np.asarray(counts, dtype=np.float64)
    context = np.asarray(sorted(set(int(x) for x in context_classes)), dtype=np.int64)
    if context.size == 0:
        raise ValueError("context_classes must be non-empty")
    support, weights = support_weights(counts, tail_class)
    local = counts[support][:, context]
    presence = local > 0
    class_count = presence.sum(axis=1).astype(np.float64)
    sample_count = local.sum(axis=1)
    client_total = counts[support].sum(axis=1)
    sample_fraction = sample_count / np.maximum(client_total, 1.0)
    result: dict[str, float] = {}

    for weighting, q in weights.items():
        result[f"generic_companion_class_count_{weighting}"] = float(np.dot(q, class_count))
        result[f"generic_companion_class_fraction_{weighting}"] = float(
            np.dot(q, class_count) / context.size
        )
        result[f"generic_companion_sample_count_{weighting}"] = float(np.dot(q, sample_count))
        result[f"generic_companion_sample_fraction_{weighting}"] = float(
            np.dot(q, sample_fraction)
        )

        marginal = np.dot(q, presence.astype(np.float64))
        if marginal.sum() <= 0 or context.size <= 1:
            entropy = 0.0
        else:
            distribution = marginal / marginal.sum()
            nonzero = distribution[distribution > 0]
            entropy = float(-np.sum(nonzero * np.log(nonzero)) / np.log(context.size))
        result[f"companion_presence_entropy_{weighting}"] = entropy

        jaccards: list[float] = []
        centroid_distances: list[float] = []
        pair_weights: list[float] = []
        for left in range(support.size):
            for right in range(left + 1, support.size):
                union = np.logical_or(presence[left], presence[right]).sum()
                intersection = np.logical_and(presence[left], presence[right]).sum()
                # Diversity is one minus similarity; two empty contexts are identical.
                jaccards.append(0.0 if union == 0 else 1.0 - intersection / union)
                pair_weights.append(float(q[left] * q[right]))
                if embeddings is not None:
                    def centroid(row: np.ndarray) -> np.ndarray | None:
                        if row.sum() <= 0:
                            return None
                        vector = np.dot(row / row.sum(), embeddings[context])
                        norm = np.linalg.norm(vector)
                        return None if norm <= 0 else vector / norm

                    centroid_left = centroid(local[left])
                    centroid_right = centroid(local[right])
                    if centroid_left is None or centroid_right is None:
                        centroid_distances.append(0.0)
                    else:
                        centroid_distances.append(float(1.0 - np.dot(centroid_left, centroid_right)))
        result[f"context_jaccard_diversity_{weighting}"] = _weighted_pairwise(
            jaccards, pair_weights
        )
        result[f"context_clip_centroid_cosine_diversity_{weighting}"] = (
            _weighted_pairwise(centroid_distances, pair_weights)
            if embeddings is not None
            else float("nan")
        )
    return result


def class_set_coverage(
    counts: np.ndarray,
    tail_class: int,
    class_set: Sequence[int],
    weighting: str,
    class_weights: Sequence[float] | None = None,
) -> float:
    classes = np.asarray(class_set, dtype=np.int64)
    if classes.size == 0 or np.unique(classes).size != classes.size:
        raise ValueError("class_set must be non-empty and contain distinct class ids")
    support, weights = support_weights(counts, tail_class)
    if weighting not in weights:
        raise ValueError(f"Unknown weighting {weighting}")
    if class_weights is None:
        w = np.repeat(1.0 / classes.size, classes.size)
    else:
        w = np.asarray(class_weights, dtype=np.float64)
        if w.shape != classes.shape or np.any(w < 0) or w.sum() <= 0:
            raise ValueError("Invalid class_weights")
        w = w / w.sum()
    per_client = np.dot((counts[support][:, classes] > 0).astype(np.float64), w)
    return float(np.dot(weights[weighting], per_client))


def related_sample_metrics(
    counts: np.ndarray,
    tail_class: int,
    related_classes: Sequence[int],
    context_classes: Iterable[int],
) -> dict[str, float]:
    counts = np.asarray(counts, dtype=np.float64)
    related = np.asarray(related_classes, dtype=np.int64)
    context = np.asarray(sorted(set(int(x) for x in context_classes)), dtype=np.int64)
    support, weights = support_weights(counts, tail_class)
    related_count = counts[support][:, related].sum(axis=1)
    companion_count = counts[support][:, context].sum(axis=1)
    fraction = related_count / np.maximum(companion_count, 1.0)
    result = {}
    for weighting, q in weights.items():
        result[f"related_companion_absolute_sample_count_{weighting}"] = float(
            np.dot(q, related_count)
        )
        result[f"related_companion_fraction_among_companions_{weighting}"] = float(
            np.dot(q, fraction)
        )
    return result


def frequency_quintiles(class_counts: Sequence[int]) -> dict[int, int]:
    counts = np.asarray(class_counts, dtype=np.int64)
    ordered = sorted(range(len(counts)), key=lambda class_id: (int(counts[class_id]), class_id))
    groups = np.array_split(np.asarray(ordered, dtype=np.int64), 5)
    return {int(class_id): quintile for quintile, group in enumerate(groups) for class_id in group}


def generate_frequency_matched_null_sets(
    tail_class: int,
    related_classes: Sequence[int],
    quintile_by_class: Mapping[int, int],
    draws: int = 1000,
    master_seed: int = 20260811,
) -> list[list[int]]:
    related = [int(x) for x in related_classes]
    if len(set(related)) != len(related):
        raise ValueError("related_classes contains duplicates")
    composition = {q: 0 for q in range(5)}
    for class_id in related:
        composition[int(quintile_by_class[class_id])] += 1
    candidates = {
        q: [
            int(class_id)
            for class_id, class_quintile in sorted(quintile_by_class.items())
            if int(class_quintile) == q and int(class_id) != int(tail_class)
        ]
        for q in range(5)
    }
    for q, required in composition.items():
        if len(candidates[q]) < required:
            raise ValueError(
                f"Quintile {q} has {len(candidates[q])} candidates but {required} are required"
            )
    output: list[list[int]] = []
    for draw_id in range(int(draws)):
        # SeedSequence is stable and avoids Python's randomized hash().
        rng = np.random.default_rng(np.random.SeedSequence([master_seed, tail_class, draw_id]))
        sampled: list[int] = []
        for q in range(5):
            required = composition[q]
            if required:
                sampled.extend(rng.choice(candidates[q], size=required, replace=False).tolist())
        # Canonical ordering makes the artifact and its hash deterministic.
        output.append(sorted(int(x) for x in sampled))
    return output


def cluster_bootstrap(
    values_by_class: Mapping[int, Sequence[float]],
    draws: int = 10000,
    seed: int = 20260811,
) -> dict[str, object]:
    class_ids = np.asarray(sorted(values_by_class), dtype=np.int64)
    if class_ids.size == 0:
        raise ValueError("No class clusters for bootstrap")
    cluster_means = np.asarray(
        [np.mean(np.asarray(values_by_class[int(c)], dtype=np.float64)) for c in class_ids]
    )
    rng = np.random.default_rng(seed)
    sampled = rng.choice(cluster_means, size=(int(draws), class_ids.size), replace=True).mean(axis=1)
    return {
        "mean": float(cluster_means.mean()),
        "median": float(np.median(cluster_means)),
        "ci_low": float(np.quantile(sampled, 0.025)),
        "ci_high": float(np.quantile(sampled, 0.975)),
        "bootstrap_means": sampled,
    }
