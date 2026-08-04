"""Minimum-norm boundary repair and final-update matching.

This module deliberately contains no model or dataset code.  The Gate supplies
one row gradient per fragile edge and receives a repair vector plus an explicit
solver report.
"""

from __future__ import annotations

import math
from typing import Callable, Mapping, Sequence

import torch

from utils.boundary_metrics import EPS, cap_repair_norm, matched_repair_update, tensor64


def solve_minimum_norm_repair(
    gradients: torch.Tensor,
    deficits: torch.Tensor,
    *,
    max_iterations: int = 500,
    tolerance: float = 1e-8,
    ridge: float = 1e-10,
) -> tuple[torch.Tensor, dict]:
    """Solve the nonnegative dual with projected gradient ascent.

    The primal is ``min .5 ||delta||^2`` subject to ``G delta >= d``.  A
    positive residual after convergence is retained in the report rather than
    being hidden as a successful repair; trust-region clipping can also make a
    previously feasible solution only partially closed.
    """
    matrix = tensor64(gradients)
    target = tensor64(deficits).reshape(-1)
    if matrix.ndim != 2 or matrix.shape[0] != target.numel():
        raise ValueError("gradients must be [num_edges, num_parameters] and match deficits")
    if matrix.shape[0] == 0:
        return torch.zeros(matrix.shape[1], dtype=torch.float64), {
            "status": "empty",
            "iterations": 0,
            "max_linear_violation": 0.0,
            "active_edge_count": 0,
        }
    if bool((target < -tolerance).any()):
        raise ValueError("deficits must be nonnegative")

    gram = matrix @ matrix.T
    if ridge > 0:
        gram = gram + float(ridge) * torch.eye(gram.shape[0], dtype=torch.float64)
    eigenvalues = torch.linalg.eigvalsh(gram)
    lipschitz = max(float(eigenvalues.max().item()), EPS)
    step = 1.0 / lipschitz
    dual = torch.zeros_like(target)
    converged = False
    for iteration in range(1, int(max_iterations) + 1):
        next_dual = torch.clamp(dual + step * (target - gram @ dual), min=0.0)
        delta_dual = float((next_dual - dual).abs().max().item())
        dual = next_dual
        if delta_dual <= tolerance:
            converged = True
            break
    repair = matrix.T @ dual
    residual = target - matrix @ repair
    max_violation = max(float(residual.max().item()), 0.0)
    positive_eigenvalues = eigenvalues[eigenvalues > EPS]
    condition = (
        float(positive_eigenvalues.max().item() / positive_eigenvalues.min().item())
        if positive_eigenvalues.numel() else math.inf
    )
    return repair, {
        "status": "converged" if converged else "max_iterations",
        "iterations": iteration,
        "dual_step_size": step,
        "ridge": float(ridge),
        "gram_condition_number": condition,
        "max_linear_violation": max_violation,
        "mean_linear_violation": float(torch.clamp(residual, min=0.0).mean().item()),
        "active_edge_count": int((dual > tolerance).sum().item()),
        "repair_norm": float(repair.norm().item()),
        "dual_objective": float(target.dot(dual).item() - 0.5 * dual.dot(gram @ dual).item()),
    }


def build_repair_candidates(
    fedavg_delta: torch.Tensor,
    raw_repair: torch.Tensor,
    *,
    repair_ratio: float,
    alphas: Sequence[float],
) -> tuple[list[tuple[float, torch.Tensor, dict]], dict]:
    """Cap a repair then construct final-norm-matched backtracking candidates."""
    fedavg = tensor64(fedavg_delta)
    capped, cap_report = cap_repair_norm(raw_repair, float(fedavg.norm().item()), repair_ratio)
    candidates = []
    for alpha in alphas:
        delta, report = matched_repair_update(fedavg, capped, float(alpha))
        candidates.append((float(alpha), delta, report))
    return candidates, cap_report


def choose_safe_backtracking_candidate(
    candidates: Sequence[tuple[float, torch.Tensor, Mapping]],
    safety_check: Callable[[torch.Tensor], tuple[bool, dict]],
) -> tuple[torch.Tensor | None, dict]:
    """Return the first safe candidate and preserve every rejection reason."""
    attempts = []
    for alpha, delta, norm_report in candidates:
        safe, safety_report = safety_check(delta)
        attempts.append({"alpha": alpha, "norm": dict(norm_report), "safety": safety_report, "accepted": bool(safe)})
        if safe:
            return delta, {"accepted": True, "selected_alpha": alpha, "attempts": attempts}
    return None, {"accepted": False, "selected_alpha": None, "attempts": attempts}
