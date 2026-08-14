from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np


def _array(value, *, ndim: int, name: str, dtype=None) -> np.ndarray:
    result = np.asarray(value, dtype=dtype)
    if result.ndim != ndim:
        raise ValueError(f"{name} must have {ndim} dimensions, got {result.shape}")
    if not np.isfinite(result).all():
        raise ValueError(f"{name} contains non-finite values")
    return result


def _validate_logits_labels(logits, labels) -> tuple[np.ndarray, np.ndarray]:
    logits = _array(logits, ndim=2, name="logits", dtype=np.float64)
    labels = _array(labels, ndim=1, name="labels", dtype=np.int64)
    if logits.shape[0] != labels.shape[0]:
        raise ValueError("logits and labels have different sample counts")
    if logits.shape[1] < 2:
        raise ValueError("at least two classes are required")
    if labels.size and (labels.min() < 0 or labels.max() >= logits.shape[1]):
        raise ValueError("labels are outside the logit class range")
    return logits, labels


def _true_class_margins(logits: np.ndarray, labels: np.ndarray) -> np.ndarray:
    rows = np.arange(labels.size)
    true_logits = logits[rows, labels]
    negatives = logits.copy()
    negatives[rows, labels] = -np.inf
    return true_logits - negatives.max(axis=1)


def _tail_ids(labels: np.ndarray, tail_classes: Sequence[int]) -> list[int]:
    requested = sorted(set(int(value) for value in tail_classes))
    missing = [class_id for class_id in requested if not np.any(labels == class_id)]
    if missing:
        raise ValueError(f"tail classes have no evaluation samples: {missing}")
    return requested


def visual_subgroup_metrics(
    logits,
    labels,
    cluster_ids,
    tail_classes: Sequence[int],
    *,
    recognized_accuracy_threshold: float = 0.5,
) -> list[dict]:
    """Compute fixed-DINO-subgroup coverage for every tail class.

    ``cluster_ids`` must have been generated once from a frozen independent
    encoder and then reused for every topology, checkpoint, seed, and method.
    """
    logits, labels = _validate_logits_labels(logits, labels)
    cluster_ids = _array(cluster_ids, ndim=1, name="cluster_ids", dtype=np.int64)
    if cluster_ids.shape[0] != labels.shape[0]:
        raise ValueError("cluster_ids and labels have different sample counts")
    if not 0.0 <= float(recognized_accuracy_threshold) <= 1.0:
        raise ValueError("recognized_accuracy_threshold must be in [0,1]")
    predictions = logits.argmax(axis=1)
    output = []
    for class_id in _tail_ids(labels, tail_classes):
        class_mask = labels == class_id
        class_clusters = sorted(set(int(value) for value in cluster_ids[class_mask]))
        if not class_clusters or class_clusters[0] < 0:
            raise ValueError(f"invalid cluster IDs for tail class {class_id}: {class_clusters}")
        cluster_accuracies, cluster_sizes = [], []
        for cluster_id in class_clusters:
            mask = class_mask & (cluster_ids == cluster_id)
            size = int(mask.sum())
            if size <= 0:
                raise AssertionError("empty cluster survived cluster enumeration")
            cluster_sizes.append(size)
            cluster_accuracies.append(float(np.mean(predictions[mask] == labels[mask])))
        values = np.asarray(cluster_accuracies, dtype=np.float64)
        recognized = values >= float(recognized_accuracy_threshold)
        output.append({
            "tail_class": class_id,
            "sample_count": int(class_mask.sum()),
            "cluster_count": len(class_clusters),
            "cluster_sizes": cluster_sizes,
            "cluster_accuracies": cluster_accuracies,
            "worst_cluster_accuracy": float(values.min()),
            "cluster_accuracy_std": float(values.std(ddof=0)),
            "recognized_cluster_count_at_50": int(recognized.sum()),
            "recognized_cluster_fraction_at_50": float(recognized.mean()),
            "cluster_balanced_accuracy": float(values.mean()),
            "recognized_accuracy_threshold": float(recognized_accuracy_threshold),
        })
    return output


def multiview_robustness_metrics(
    logits_by_view: Mapping[str, np.ndarray],
    labels,
    tail_classes: Sequence[int],
    *,
    clean_view: str = "clean",
    expected_views: Sequence[str] = (
        "clean", "crop", "color_jitter", "blur", "occlusion", "resize"
    ),
) -> list[dict]:
    """Compute deterministic multi-view robustness metrics per tail class."""
    if set(logits_by_view) != set(expected_views):
        raise ValueError(
            f"view set differs from frozen protocol: {sorted(logits_by_view)} != "
            f"{sorted(expected_views)}"
        )
    labels = _array(labels, ndim=1, name="labels", dtype=np.int64)
    normalized = {}
    for view in expected_views:
        logits, checked_labels = _validate_logits_labels(logits_by_view[view], labels)
        if not np.array_equal(checked_labels, labels):
            raise AssertionError("label validation changed labels")
        normalized[view] = logits
    clean_predictions = normalized[clean_view].argmax(axis=1)
    corrupted = [view for view in expected_views if view != clean_view]
    output = []
    for class_id in _tail_ids(labels, tail_classes):
        mask = labels == class_id
        view_accuracies, view_margins = {}, {}
        consistency_parts = []
        prediction_stack = []
        for view in expected_views:
            predictions = normalized[view].argmax(axis=1)
            margins = _true_class_margins(normalized[view], labels)
            view_accuracies[view] = float(np.mean(predictions[mask] == labels[mask]))
            view_margins[view] = float(np.mean(margins[mask]))
            prediction_stack.append(predictions[mask])
            if view != clean_view:
                consistency_parts.append(predictions[mask] == clean_predictions[mask])
        corruption_accuracy = float(np.mean([view_accuracies[view] for view in corrupted]))
        consistency = float(np.mean(np.concatenate(consistency_parts)))
        stack = np.stack(prediction_stack, axis=0)
        all_view_consistency = float(np.mean(np.all(stack == stack[0:1], axis=0)))
        output.append({
            "tail_class": class_id,
            "sample_count": int(mask.sum()),
            "view_accuracies": view_accuracies,
            "view_margins": view_margins,
            "worst_view_accuracy": float(min(view_accuracies.values())),
            "prediction_consistency": consistency,
            "all_view_prediction_consistency": all_view_consistency,
            "worst_view_margin": float(min(view_margins.values())),
            "clean_accuracy": float(view_accuracies[clean_view]),
            "mean_corruption_accuracy": corruption_accuracy,
            "clean_to_corruption_accuracy_drop": float(
                view_accuracies[clean_view] - corruption_accuracy
            ),
            "clean_to_worst_view_accuracy_drop": float(
                view_accuracies[clean_view] - min(view_accuracies.values())
            ),
        })
    return output


def neighbor_discrimination_metrics(
    logits,
    labels,
    neighbors_by_tail: Mapping[int, Sequence[int]],
    tail_classes: Sequence[int],
) -> list[dict]:
    """Measure each target tail class against its frozen semantic neighbors."""
    logits, labels = _validate_logits_labels(logits, labels)
    output = []
    for class_id in _tail_ids(labels, tail_classes):
        if class_id not in neighbors_by_tail:
            raise ValueError(f"missing frozen neighbors for tail class {class_id}")
        neighbors = [int(value) for value in neighbors_by_tail[class_id]]
        if not neighbors or len(set(neighbors)) != len(neighbors):
            raise ValueError(f"neighbors must be non-empty and unique for {class_id}")
        if class_id in neighbors:
            raise ValueError(f"target class {class_id} appears in its own neighbors")
        if min(neighbors) < 0 or max(neighbors) >= logits.shape[1]:
            raise ValueError(f"neighbor IDs outside logit class range for {class_id}")
        mask = labels == class_id
        target = logits[mask, class_id][:, None]
        pairwise = target - logits[mask][:, neighbors]
        neighbor_means = pairwise.mean(axis=0)
        output.append({
            "tail_class": class_id,
            "sample_count": int(mask.sum()),
            "neighbor_count": len(neighbors),
            "neighbor_class_ids": neighbors,
            "neighbor_mean_margins": [float(value) for value in neighbor_means],
            "target_vs_neighbor_pairwise_margin": float(pairwise.mean()),
            "worst_neighbor_margin": float(neighbor_means.min()),
            "positive_margin_neighbor_coverage": float(np.mean(neighbor_means > 0.0)),
            "neighbor_margin_variance": float(neighbor_means.var(ddof=0)),
        })
    return output
