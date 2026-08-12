from __future__ import annotations

import math
from typing import Mapping, Sequence

import numpy as np
import torch
from torch.nn import functional as F


def classification_metrics(logits: torch.Tensor, labels: torch.Tensor, target_class: int | None = None) -> dict:
    if logits.ndim != 2 or logits.shape[1] != 100:
        raise ValueError(f"Expected [N,100] logits, got {tuple(logits.shape)}")
    if labels.ndim != 1 or labels.numel() != logits.shape[0]:
        raise ValueError("Labels do not align with logits")
    if not torch.isfinite(logits).all():
        raise ValueError("Non-finite logits")
    labels = labels.long()
    rows = torch.arange(labels.numel(), device=logits.device)
    correct_logits = logits[rows, labels]
    masked = logits.clone()
    masked[rows, labels] = -torch.inf
    hardest_logits, hardest_classes = masked.max(dim=1)
    margins = correct_logits - hardest_logits
    nll = F.cross_entropy(logits, labels, reduction="none")
    predictions = logits.argmax(dim=1)
    result = {
        "sample_count": int(labels.numel()),
        "margin": float(margins.mean().item()),
        "nll": float(nll.mean().item()),
        "accuracy": float((predictions == labels).float().mean().item()),
        "correct_logit": float(correct_logits.mean().item()),
        "hardest_negative_logit": float(hardest_logits.mean().item()),
    }
    if target_class is not None:
        if not bool((labels == int(target_class)).all()):
            raise ValueError("Target metric received labels from another class")
        counts = torch.bincount(hardest_classes.cpu(), minlength=100)
        result["hardest_negative_class"] = int(torch.argmax(counts).item())
    return result


def metric_gain(before: Mapping[str, float], after: Mapping[str, float]) -> dict:
    return {
        "g_margin": float(after["margin"] - before["margin"]),
        "g_nll": float(before["nll"] - after["nll"]),
        "g_acc": float(after["accuracy"] - before["accuracy"]),
        "correct_logit_change": float(after["correct_logit"] - before["correct_logit"]),
        "hardest_negative_logit_change": float(after["hardest_negative_logit"] - before["hardest_negative_logit"]),
    }


def vector_comparison(left: torch.Tensor, right: torch.Tensor, eps: float = 1e-12) -> dict:
    left, right = left.detach().float().reshape(-1), right.detach().float().reshape(-1)
    if left.shape != right.shape:
        raise ValueError(f"Vector shapes differ: {left.shape} vs {right.shape}")
    difference = left - right
    denominator = max(float(left.norm().item()), float(right.norm().item()), eps)
    cosine = float(F.cosine_similarity(left, right, dim=0, eps=eps).item()) if left.numel() else 1.0
    return {
        "relative_l2": float(difference.norm().item()) / denominator,
        "max_abs": float(difference.abs().max().item()) if difference.numel() else 0.0,
        "cosine": cosine,
    }


def cluster_bootstrap(values: Mapping[int, Sequence[float]], draws: int = 10000, seed: int = 20260811) -> dict:
    """Resample class IDs, retaining all seed-level observations per class."""
    classes = sorted(int(key) for key in values)
    if not classes:
        raise ValueError("No class clusters")
    cluster_means = np.asarray([np.mean(np.asarray(values[class_id], dtype=np.float64)) for class_id in classes])
    if not np.isfinite(cluster_means).all():
        raise ValueError("Non-finite cluster values")
    generator = np.random.default_rng(int(seed))
    samples = np.empty(int(draws), dtype=np.float64)
    for index in range(int(draws)):
        chosen = generator.integers(0, len(classes), size=len(classes))
        samples[index] = float(cluster_means[chosen].mean())
    return {
        "mean": float(cluster_means.mean()),
        "median": float(np.median(cluster_means)),
        "ci_low": float(np.quantile(samples, 0.025)),
        "ci_high": float(np.quantile(samples, 0.975)),
        "class_count": len(classes),
        "bootstrap_draws": int(draws),
    }


def summarize_v2_rows(rows: Sequence[Mapping], bootstrap_draws: int = 10000) -> dict:
    """Average unrelated draws within seed/class before class-cluster inference."""
    grouped = {}
    for row in rows:
        key = (int(row["data_seed"]), int(row["tail_class"]))
        grouped.setdefault(key, {"related": [], "unrelated": [], "tail_only": []})
        condition = str(row["condition"])
        if condition == "related":
            grouped[key]["related"].append(float(row["g_margin"]))
        elif condition.startswith("matched_unrelated"):
            grouped[key]["unrelated"].append(float(row["g_margin"]))
        elif condition == "tail_only_masked":
            grouped[key]["tail_only"].append(float(row["g_margin"]))
    paired = []
    for (seed, class_id), values in sorted(grouped.items()):
        if len(values["related"]) != 1 or not values["unrelated"] or len(values["tail_only"]) != 1:
            raise ValueError(f"Incomplete paired V2 unit {(seed, class_id)}")
        sem = values["related"][0] - float(np.mean(values["unrelated"]))
        pos = values["related"][0] - values["tail_only"][0]
        paired.append({"data_seed": seed, "tail_class": class_id, "delta_sem": sem, "delta_pos": pos})
    by_class_sem, by_class_pos = {}, {}
    for row in paired:
        by_class_sem.setdefault(row["tail_class"], []).append(row["delta_sem"])
        by_class_pos.setdefault(row["tail_class"], []).append(row["delta_pos"])
    return {
        "paired_rows": paired,
        "delta_sem_bootstrap": cluster_bootstrap(by_class_sem, bootstrap_draws),
        "delta_pos_bootstrap": cluster_bootstrap(by_class_pos, bootstrap_draws),
    }
