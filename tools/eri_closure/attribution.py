"""Mathematical ERI attribution primitives.

These functions contain no data loading, model construction, or test-set
access.  They are intentionally unit-testable against analytic objectives.
"""

from __future__ import annotations

import math
from typing import Callable, Iterable, Sequence
import torch


GradientFn = Callable[[torch.Tensor, int], torch.Tensor]


def gauss_legendre_rule(points: int, *, device=None, dtype=torch.float64) -> tuple[torch.Tensor, torch.Tensor]:
    if int(points) < 1:
        raise ValueError("Quadrature needs at least one point")
    # Keep this tiny rule table dependency-free. In particular, NumPy's
    # ``leggauss`` may initialize an MKL eigensolver on import, which is an
    # unnecessary failure mode in minimal Slurm analysis environments.
    rules = {
        1: ([0.0], [2.0]),
        2: ([-0.5773502691896257, 0.5773502691896257], [1.0, 1.0]),
        4: (
            [-0.8611363115940526, -0.3399810435848563, 0.3399810435848563, 0.8611363115940526],
            [0.3478548451374538, 0.6521451548625461, 0.6521451548625461, 0.3478548451374538],
        ),
        8: (
            [-0.9602898564975363, -0.7966664774136267, -0.5255324099163290, -0.1834346424956498,
             0.1834346424956498, 0.5255324099163290, 0.7966664774136267, 0.9602898564975363],
            [0.1012285362903763, 0.2223810344533745, 0.3137066458778873, 0.3626837833783620,
             0.3626837833783620, 0.3137066458778873, 0.2223810344533745, 0.1012285362903763],
        ),
    }
    if int(points) not in rules:
        raise ValueError("Supported Gauss-Legendre ERI quadrature points are 1, 2, 4, or 8")
    nodes, weights = rules[int(points)]
    # Map [-1, 1] -> [0, 1].
    return (
        (torch.as_tensor(nodes, device=device, dtype=dtype) + 1.0) / 2.0,
        torch.as_tensor(weights, device=device, dtype=dtype) / 2.0,
    )


def aggregate_delta(client_deltas: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    client_deltas = torch.as_tensor(client_deltas, dtype=torch.float64)
    weights = torch.as_tensor(weights, dtype=torch.float64).reshape(-1)
    if client_deltas.ndim != 2 or client_deltas.shape[0] != weights.numel():
        raise ValueError("client_deltas must be [clients, parameters] aligned with weights")
    if not torch.isfinite(weights).all() or (weights < 0).any() or abs(float(weights.sum()) - 1.0) > 1e-6:
        raise ValueError("server weights must be finite, non-negative, and sum to one")
    return (client_deltas * weights[:, None]).sum(dim=0)


def signed_budgets(effects: torch.Tensor, supports: torch.Tensor, *, epsilon: float = 1e-12) -> dict[str, float]:
    """Split one class's client effects into evidence, donor, and rewrite mass.

    ``effects`` already includes the aggregation coefficient q_k, so all
    reported values are *server-realized functional contributions*.
    """
    effects = torch.as_tensor(effects, dtype=torch.float64).reshape(-1)
    supports = torch.as_tensor(supports, dtype=torch.bool).reshape(-1)
    if effects.numel() != supports.numel():
        raise ValueError("effects and supports must be client-aligned")
    positive = effects.clamp_min(0)
    negative = (-effects).clamp_min(0)
    write = positive[supports].sum()
    supporter_harm = negative[supports].sum()
    donor = positive[~supports].sum()
    rewrite = negative[~supports].sum()
    denominator = write + donor + float(epsilon)
    return {
        "W": float(write.item()),
        "H": float(supporter_harm.item()),
        "D": float(donor.item()),
        "R": float(rewrite.item()),
        "positive_refresh": float((write + donor).item()),
        "ERI": float((rewrite / denominator).item()),
        "non_support_net": float((effects[~supports].sum()).item()),
        "support_net": float((effects[supports].sum()).item()),
    }


def integrated_client_effects(
    theta_before: torch.Tensor,
    client_deltas: torch.Tensor,
    weights: torch.Tensor,
    class_ids: Sequence[int],
    gradient_fn: GradientFn,
    *,
    quadrature_points: int = 8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Path-integrated per-client functional effects.

    Returns ``[classes, clients]`` effects e_{k,c} and their aggregate server
    delta.  With the exact line integral, summing client effects equals the
    functional change of the aggregate update.  Numerical completeness is
    checked by the caller against direct endpoint evaluation.
    """
    theta = torch.as_tensor(theta_before, dtype=torch.float64).reshape(-1)
    deltas = torch.as_tensor(client_deltas, dtype=torch.float64)
    q = torch.as_tensor(weights, dtype=torch.float64).reshape(-1)
    aggregate = aggregate_delta(deltas, q)
    alphas, quad_weights = gauss_legendre_rule(quadrature_points, dtype=torch.float64)
    effects = torch.zeros((len(class_ids), deltas.shape[0]), dtype=torch.float64)
    for class_index, class_id in enumerate(class_ids):
        integral_gradient = torch.zeros_like(theta)
        for alpha, quad_weight in zip(alphas, quad_weights):
            gradient = torch.as_tensor(
                gradient_fn(theta + alpha * aggregate, int(class_id)), dtype=torch.float64
            ).reshape(-1)
            if gradient.shape != theta.shape:
                raise ValueError("gradient_fn returned a vector with wrong shape")
            integral_gradient.add_(gradient, alpha=float(quad_weight))
        effects[class_index] = q * (deltas @ integral_gradient)
    return effects, aggregate


def first_order_client_effects(
    theta_before: torch.Tensor,
    client_deltas: torch.Tensor,
    weights: torch.Tensor,
    class_ids: Sequence[int],
    gradient_fn: GradientFn,
) -> tuple[torch.Tensor, torch.Tensor]:
    theta = torch.as_tensor(theta_before, dtype=torch.float64).reshape(-1)
    deltas = torch.as_tensor(client_deltas, dtype=torch.float64)
    q = torch.as_tensor(weights, dtype=torch.float64).reshape(-1)
    aggregate = aggregate_delta(deltas, q)
    effects = torch.stack(
        [q * (deltas @ torch.as_tensor(gradient_fn(theta, int(c)), dtype=torch.float64).reshape(-1)) for c in class_ids]
    )
    return effects, aggregate


def rows_from_effects(
    effects: torch.Tensor,
    class_ids: Sequence[int],
    selected_client_ids: Sequence[int],
    client_class_counts: torch.Tensor,
    *,
    communication_round: int,
    method: str,
    aggregation_weights: torch.Tensor | None = None,
    epsilon: float = 1e-12,
) -> tuple[list[dict], list[dict]]:
    """Return client effect and per-class signed-budget records."""
    effects = torch.as_tensor(effects, dtype=torch.float64)
    counts = torch.as_tensor(client_class_counts)
    if effects.shape != (len(class_ids), len(selected_client_ids)):
        raise ValueError("effect shape does not match class/client ids")
    q = None
    if aggregation_weights is not None:
        q = torch.as_tensor(aggregation_weights, dtype=torch.float64).reshape(-1)
        if q.numel() != len(selected_client_ids):
            raise ValueError("aggregation_weights must be client-aligned")
        if not torch.isfinite(q).all() or (q < 0).any() or abs(float(q.sum()) - 1.0) > 1e-6:
            raise ValueError("aggregation_weights must be finite, non-negative, and sum to one")
    client_rows: list[dict] = []
    budget_rows: list[dict] = []
    for ci, class_id in enumerate(class_ids):
        support = counts[:, int(class_id)] > 0
        budget = signed_budgets(effects[ci], support, epsilon=epsilon)
        access = float(q[support].sum().item()) if q is not None else float("nan")
        # ``effects`` is already q_k times the unweighted functional effect.
        # Since every selected q_k is strictly positive, testing effects > 0
        # is exactly the same as testing e_{k,c}=effects/q_k > 0, without a
        # numerically unnecessary division.  B is the aggregation mass of
        # supporter accesses that actually write in the positive direction.
        positive_support = support & (effects[ci] > 0)
        positive_support_access = (
            float(q[positive_support].sum().item()) if q is not None else float("nan")
        )
        write_success_rate = (
            positive_support_access / access
            if math.isfinite(positive_support_access) and access > 0.0
            else float("nan")
        )
        positive_write_strength = (
            budget["W"] / positive_support_access
            if math.isfinite(positive_support_access) and positive_support_access > 0.0
            else float("nan")
        )
        positive_efficiency = (
            budget["W"] / access if math.isfinite(access) and access > 0.0 else float("nan")
        )
        effective_supporters = float("nan")
        if q is not None and access > 0.0:
            squared_mass = float(q[support].square().sum().item())
            effective_supporters = access * access / squared_mass if squared_mass > 0.0 else float("nan")
        budget_rows.append(
            {
                "communication_round": int(communication_round),
                "class_id": int(class_id),
                "method": str(method),
                "supporter_count": int(support.sum().item()),
                "non_supporter_count": int((~support).sum().item()),
                "support_access": access,
                "positive_support_access": positive_support_access,
                "support_write_success_rate": write_success_rate,
                "positive_write_strength": positive_write_strength,
                "positive_write_efficiency": positive_efficiency,
                "write_factorization_absolute_error": (
                    abs(budget["W"] - access * write_success_rate * positive_write_strength)
                    if all(math.isfinite(value) for value in (
                        access, write_success_rate, positive_write_strength
                    ))
                    else float("nan")
                ),
                "support_effective_clients": effective_supporters,
                **budget,
            }
        )
        for ki, client_id in enumerate(selected_client_ids):
            value = float(effects[ci, ki].item())
            if value > 0:
                signed_role = "support_write" if support[ki] else "donor"
            elif value < 0:
                signed_role = "support_harm" if support[ki] else "rewriter"
            else:
                signed_role = "neutral"
            client_rows.append(
                {
                    "communication_round": int(communication_round),
                    "class_id": int(class_id),
                    "method": str(method),
                    "client_id": int(client_id),
                    "supports_class": int(bool(support[ki].item())),
                    "aggregation_weight": float(q[ki].item()) if q is not None else float("nan"),
                    "functional_effect": value,
                    "functional_effect_per_unit_weight": (
                        value / float(q[ki].item())
                        if q is not None and float(q[ki].item()) > 0.0 else float("nan")
                    ),
                    "signed_role": signed_role,
                }
            )
    return client_rows, budget_rows


def completeness_record(
    effects: torch.Tensor,
    direct_before: Iterable[float],
    direct_after: Iterable[float],
    class_ids: Sequence[int],
    *,
    communication_round: int,
    method: str,
) -> list[dict]:
    rows = []
    for index, (before, after, class_id) in enumerate(zip(direct_before, direct_after, class_ids)):
        attributed = float(torch.as_tensor(effects[index]).sum().item())
        actual = float(after) - float(before)
        rows.append(
            {
                "communication_round": int(communication_round),
                "class_id": int(class_id),
                "method": str(method),
                "attributed_change": attributed,
                "direct_change": actual,
                "absolute_completeness_error": abs(attributed - actual),
            }
        )
    return rows
