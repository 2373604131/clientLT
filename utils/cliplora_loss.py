"""Loss primitive shared by baseline ClipLora and mechanism experiments."""

from __future__ import annotations

import torch
from torch.nn import functional as F


def fixed_denominator_cross_entropy(logits, labels, loss_weight=None):
    """100-way CE with an optional sample mask and actual-batch denominator."""
    if loss_weight is None:
        return F.cross_entropy(logits, labels)
    weights = loss_weight.to(device=logits.device, dtype=logits.dtype).reshape(-1)
    if weights.numel() != labels.numel():
        raise ValueError(
            f"loss_weight has {weights.numel()} entries for {labels.numel()} labels"
        )
    if not torch.isfinite(weights).all() or bool((weights < 0).any()):
        raise ValueError("loss_weight must be finite and non-negative")
    per_sample = F.cross_entropy(logits, labels, reduction="none")
    return (per_sample * weights).sum() / labels.numel()
