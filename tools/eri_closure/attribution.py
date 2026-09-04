"""Mathematical ERI attribution primitives.

These functions contain no data loading, model construction, or test-set
access.  They are intentionally unit-testable against analytic objectives.
"""

from __future__ import annotations

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
    epsilon: float = 1e-12,
) -> tuple[list[dict], list[dict]]:
    """Return client effect and per-class signed-budget records."""
    effects = torch.as_tensor(effects, dtype=torch.float64)
    counts = torch.as_tensor(client_class_counts)
    if effects.shape != (len(class_ids), len(selected_client_ids)):
        raise ValueError("effect shape does not match class/client ids")
    client_rows: list[dict] = []
    budget_rows: list[dict] = []
    for ci, class_id in enumerate(class_ids):
        support = counts[:, int(class_id)] > 0
        budget = signed_budgets(effects[ci], support, epsilon=epsilon)
        budget_rows.append(
            {
                "communication_round": int(communication_round),
                "class_id": int(class_id),
                "method": str(method),
                "supporter_count": int(support.sum().item()),
                "non_supporter_count": int((~support).sum().item()),
                **budget,
            }
        )
        for ki, client_id in enumerate(selected_client_ids):
            value = float(effects[ci, ki].item())
            client_rows.append(
                {
                    "communication_round": int(communication_round),
                    "class_id": int(class_id),
                    "method": str(method),
                    "client_id": int(client_id),
                    "supports_class": int(bool(support[ki].item())),
                    "functional_effect": value,
                    "signed_role": "support_write" if support[ki] and value > 0 else (
                        "donor" if value > 0 else ("rewriter" if value < 0 else "neutral")
                    ),
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
