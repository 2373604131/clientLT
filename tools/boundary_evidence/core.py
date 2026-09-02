"""Pure calculations for the boundary-evidence experiment.

This module deliberately contains no model, dataset, or filesystem code.  The
same definitions are therefore used by the runtime, summarizer, and tests.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import torch


def hard_negative_ranking(mean_logits: torch.Tensor, target_class: int) -> list[int]:
    """Rank all non-target classes by the target-side pair margin.

    For fixed target-class probes, sorting ``z_c-z_h`` ascending is equivalent
    to sorting the competing logit ``z_h`` descending.  The explicit margin
    form keeps the implementation aligned with the preregistered definition.
    """
    values = torch.as_tensor(mean_logits, dtype=torch.float64).reshape(-1)
    target = int(target_class)
    if values.numel() < 2 or target < 0 or target >= values.numel():
        raise ValueError("Invalid mean-logit vector or target class")
    if not bool(torch.isfinite(values).all()):
        raise ValueError("Hard-negative logits contain NaN or Inf")
    margins = values[target] - values
    return sorted(
        (class_id for class_id in range(values.numel()) if class_id != target),
        key=lambda class_id: (float(margins[class_id].item()), class_id),
    )


def choose_matched_control(
    global_counts: Sequence[int],
    target_class: int,
    hard_class: int,
    excluded_top20: Sequence[int],
) -> int:
    """Choose the frequency-nearest class outside the target's frozen top-20."""
    counts = np.asarray(global_counts, dtype=np.int64).reshape(-1)
    target, hard = int(target_class), int(hard_class)
    excluded = {int(value) for value in excluded_top20}
    excluded.add(target)
    candidates = [class_id for class_id in range(len(counts)) if class_id not in excluded]
    if not candidates:
        raise ValueError("No matched-control class remains outside frozen top-20")
    return min(candidates, key=lambda class_id: (abs(int(counts[class_id]) - int(counts[hard])), class_id))


def coexposure_rate(counts: np.ndarray, target_class: int, hard_class: int) -> dict:
    """Compute q(c,h) over clients carrying the target class."""
    matrix = np.asarray(counts, dtype=np.int64)
    if matrix.ndim != 2 or np.any(matrix < 0):
        raise ValueError("counts must be a non-negative [clients, classes] matrix")
    target, hard = int(target_class), int(hard_class)
    carriers = matrix[:, target] > 0
    carrier_count = int(carriers.sum())
    if carrier_count == 0:
        raise ValueError(f"Target class {target} has no carrier")
    joint_count = int(np.logical_and(carriers, matrix[:, hard] > 0).sum())
    return {
        "carrier_count": carrier_count,
        "joint_carrier_count": joint_count,
        "q": float(joint_count / carrier_count),
    }


def pairwise_boundary_metrics(
    logits_c: torch.Tensor,
    logits_h: torch.Tensor,
    target_class: int,
    hard_class: int,
) -> dict[str, float]:
    """Return M_c, M_h and balanced pairwise accuracy for one (c,h) edge."""
    left = torch.as_tensor(logits_c, dtype=torch.float64)
    right = torch.as_tensor(logits_h, dtype=torch.float64)
    target, hard = int(target_class), int(hard_class)
    if left.ndim != 2 or right.ndim != 2 or left.shape[1] != right.shape[1]:
        raise ValueError("Pairwise logits must be [samples, classes] with equal class width")
    if left.shape[0] == 0 or right.shape[0] == 0:
        raise ValueError("Pairwise evaluation requires probes from both classes")
    if min(target, hard) < 0 or max(target, hard) >= left.shape[1] or target == hard:
        raise ValueError("Invalid pairwise class ids")
    if not bool(torch.isfinite(left).all()) or not bool(torch.isfinite(right).all()):
        raise ValueError("Pairwise logits contain NaN or Inf")
    margin_c = left[:, target] - left[:, hard]
    margin_h = right[:, hard] - right[:, target]
    acc_c = (margin_c > 0).to(torch.float64).mean()
    acc_h = (margin_h > 0).to(torch.float64).mean()
    return {
        "m_c": float(margin_c.mean().item()),
        "m_h": float(margin_h.mean().item()),
        "pair_accuracy": float((0.5 * (acc_c + acc_h)).item()),
    }


def metric_deltas(before: Mapping[str, float], after: Mapping[str, float]) -> dict[str, float]:
    required = ("m_c", "m_h", "pair_accuracy")
    missing = [name for name in required if name not in before or name not in after]
    if missing:
        raise KeyError(f"Missing pairwise metrics: {missing}")
    return {
        "delta_m_c": float(after["m_c"] - before["m_c"]),
        "delta_m_h": float(after["m_h"] - before["m_h"]),
        "delta_pair_accuracy": float(after["pair_accuracy"] - before["pair_accuracy"]),
    }


def class_cluster_summary(
    values: Mapping[int, Sequence[float]],
    *,
    draws: int = 10_000,
    seed: int = 20260903,
) -> dict[str, float | int]:
    """Average within tail class, then bootstrap the tail-class means."""
    class_ids = sorted(int(value) for value in values)
    if not class_ids:
        raise ValueError("No tail-class clusters were provided")
    class_means = np.asarray(
        [np.mean(np.asarray(values[class_id], dtype=np.float64)) for class_id in class_ids],
        dtype=np.float64,
    )
    if not np.isfinite(class_means).all():
        raise ValueError("Non-finite class-cluster value")
    generator = np.random.default_rng(int(seed))
    sampled = generator.choice(
        class_means, size=(int(draws), len(class_means)), replace=True
    ).mean(axis=1)
    return {
        "mean": float(class_means.mean()),
        "ci_low": float(np.quantile(sampled, 0.025)),
        "ci_high": float(np.quantile(sampled, 0.975)),
        "tail_class_count": int(len(class_means)),
        "bootstrap_draws": int(draws),
    }
