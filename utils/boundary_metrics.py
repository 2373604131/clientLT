"""Pure tensor metrics shared by the boundary-audit Gate.

The functions in this module do not construct a model, read a dataset, or
access official test data.  Keeping the causal diagnostics and candidate-norm
rules here makes them straightforward to unit test.
"""

from __future__ import annotations

import math
from typing import Iterable

import torch


EPS = 1e-12


def tensor64(value) -> torch.Tensor:
    """Return a detached CPU float64 tensor."""
    return torch.as_tensor(value, dtype=torch.float64, device="cpu").detach().clone()


def support_mask(edge_counts: torch.Tensor) -> torch.Tensor:
    """Return one boolean support mask per edge from [clients, edges] counts."""
    counts = tensor64(edge_counts)
    if counts.ndim != 2:
        raise ValueError("edge_counts must have shape [num_clients, num_edges]")
    return counts > 0


def edge_sample_weights(edge_counts: torch.Tensor, edge_id: int) -> torch.Tensor:
    """Audit-sample weights over supporting clients for one edge."""
    counts = tensor64(edge_counts)[:, int(edge_id)]
    total = float(counts.sum().item())
    if total <= EPS:
        return torch.zeros_like(counts)
    return counts / total


def support_mass(fedavg_weights: torch.Tensor, support: torch.Tensor) -> float:
    weights = tensor64(fedavg_weights).reshape(-1)
    support = torch.as_tensor(support, dtype=torch.bool).reshape(-1)
    if weights.numel() != support.numel():
        raise ValueError("fedavg_weights and support must have the same length")
    return float(weights[support].sum().item())


def support_counterfactual_delta(
    client_deltas: torch.Tensor,
    fedavg_weights: torch.Tensor,
    support: torch.Tensor,
    *,
    normalized: bool,
) -> tuple[torch.Tensor, dict]:
    """Build a support-only delta with either raw or renormalized FedAvg mass.

    `normalized=False` preserves the original FedAvg denominator and is the
    causal ``support_actual`` state.  It must never be norm matched when used
    for dilution/interference diagnostics.
    """
    deltas = tensor64(client_deltas)
    weights = tensor64(fedavg_weights).reshape(-1)
    support = torch.as_tensor(support, dtype=torch.bool).reshape(-1)
    if deltas.ndim != 2:
        raise ValueError("client_deltas must have shape [num_clients, num_parameters]")
    if deltas.shape[0] != weights.numel() or support.numel() != weights.numel():
        raise ValueError("client dimensions do not match")
    mass = float(weights[support].sum().item())
    if not bool(support.any()) or mass <= EPS:
        return torch.zeros(deltas.shape[1], dtype=torch.float64), {
            "available": False,
            "support_mass": mass,
            "normalized": bool(normalized),
            "reason": "support_mass_zero",
        }
    selected_weights = weights.clone()
    selected_weights[~support] = 0.0
    if normalized:
        selected_weights = selected_weights / mass
    delta = (deltas * selected_weights[:, None]).sum(dim=0)
    return delta, {
        "available": True,
        "support_mass": mass,
        "normalized": bool(normalized),
        "raw_norm": float(delta.norm().item()),
    }


def match_final_update_norm(delta: torch.Tensor, budget: float, *, eps: float = EPS) -> tuple[torch.Tensor, dict]:
    """Scale a complete model update to the FedAvg final-update budget."""
    delta = tensor64(delta)
    raw_norm = float(delta.norm().item())
    if not math.isfinite(raw_norm) or raw_norm <= eps or budget <= eps:
        return torch.zeros_like(delta), {
            "matched": False,
            "raw_final_norm": raw_norm,
            "target_final_norm": float(budget),
            "matched_final_norm": 0.0,
            "scale_factor": 0.0,
            "reason": "zero_or_nonfinite_update",
        }
    scale = float(budget) / raw_norm
    matched = delta * scale
    final_norm = float(matched.norm().item())
    return matched, {
        "matched": True,
        "raw_final_norm": raw_norm,
        "target_final_norm": float(budget),
        "matched_final_norm": final_norm,
        "scale_factor": scale,
        "relative_error": abs(final_norm - float(budget)) / max(float(budget), eps),
    }


def cap_repair_norm(repair_delta: torch.Tensor, budget: float, repair_ratio: float, *, eps: float = EPS) -> tuple[torch.Tensor, dict]:
    """Apply the pre-matching repair trust region ``||delta|| <= rho * B``."""
    repair = tensor64(repair_delta)
    raw_norm = float(repair.norm().item())
    limit = max(float(repair_ratio), 0.0) * max(float(budget), 0.0)
    if raw_norm <= eps or limit <= eps:
        return torch.zeros_like(repair), {
            "raw_repair_norm": raw_norm,
            "capped_repair_norm": 0.0,
            "repair_norm_limit": limit,
            "scale_factor": 0.0,
            "capped": raw_norm > eps,
        }
    scale = min(1.0, limit / raw_norm)
    capped = repair * scale
    return capped, {
        "raw_repair_norm": raw_norm,
        "capped_repair_norm": float(capped.norm().item()),
        "repair_norm_limit": limit,
        "scale_factor": scale,
        "capped": scale < 1.0,
    }


def matched_repair_update(
    fedavg_delta: torch.Tensor,
    repair_delta: torch.Tensor,
    alpha: float,
    *,
    eps: float = EPS,
) -> tuple[torch.Tensor, dict]:
    """Blend a repair into FedAvg and match the *complete* update norm."""
    fedavg = tensor64(fedavg_delta)
    repair = tensor64(repair_delta)
    if fedavg.shape != repair.shape:
        raise ValueError("fedavg_delta and repair_delta must have matching shapes")
    budget = float(fedavg.norm().item())
    raw = fedavg + float(alpha) * repair
    matched, report = match_final_update_norm(raw, budget, eps=eps)
    report.update({
        "alpha": float(alpha),
        "fedavg_norm_budget": budget,
        "added_repair_norm": float((float(alpha) * repair).norm().item()),
    })
    return matched, report


def pair_margin(logits: torch.Tensor, labels: torch.Tensor, negatives: torch.Tensor) -> torch.Tensor:
    """Return per-example margin for fixed true/negative class pairs."""
    logits = torch.as_tensor(logits, dtype=torch.float64)
    labels = torch.as_tensor(labels, dtype=torch.long, device=logits.device).reshape(-1)
    negatives = torch.as_tensor(negatives, dtype=torch.long, device=logits.device).reshape(-1)
    if logits.ndim != 2 or logits.shape[0] != labels.numel() or labels.numel() != negatives.numel():
        raise ValueError("logits, labels, and negatives have incompatible shapes")
    rows = torch.arange(labels.numel(), device=logits.device)
    return logits[rows, labels] - logits[rows, negatives]


def finite_mean(values: Iterable[float]) -> float:
    items = [float(value) for value in values if math.isfinite(float(value))]
    return float(sum(items) / len(items)) if items else math.nan


def finite_median(values: Iterable[float]) -> float:
    items = sorted(float(value) for value in values if math.isfinite(float(value)))
    if not items:
        return math.nan
    middle = len(items) // 2
    return items[middle] if len(items) % 2 else 0.5 * (items[middle - 1] + items[middle])


def safe_ratio(numerator: float, denominator: float, *, eps: float = EPS) -> float:
    return float(numerator / denominator) if abs(float(denominator)) > eps else math.nan
