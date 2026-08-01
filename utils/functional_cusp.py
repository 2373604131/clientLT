"""Functional CUSP utilities for the offline two-topology Gate.

The functions here are train-only candidate construction code. They do not
build datasets, create official-test loaders, or choose parameters from test.
"""

from __future__ import annotations

import math
from typing import Mapping

import numpy as np
import torch

from utils.cusp_minimal import FlatSpec, flatten_state, unflatten_state


def spearmanr_simple(x, y) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if x.size < 2:
        return math.nan
    rx = _rankdata(x)
    ry = _rankdata(y)
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    denom = float(np.linalg.norm(rx) * np.linalg.norm(ry))
    return float(rx.dot(ry) / denom) if denom > 0 else math.nan


def _rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=float)
    sorted_values = values[order]
    start = 0
    while start < values.size:
        end = start + 1
        while end < values.size and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1) + 1.0
        start = end
    return ranks


def candidate_hash_from_delta(delta: torch.Tensor) -> str:
    from utils.cusp_minimal import sha256_json

    return sha256_json({"delta": [round(float(x), 12) for x in delta.detach().cpu().reshape(-1).tolist()]})


def fedavg_delta_from_payload(payload: Mapping, spec: FlatSpec | None = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    spec = spec or FlatSpec.from_dict(payload["flatten_spec"])
    theta_t = flatten_state(payload["global_before_trainable"], spec)
    local_vectors = torch.stack([flatten_state(state, spec) for state in payload["local_trainable_states"]], dim=0)
    weights = torch.as_tensor(payload["fedavg_weights"], dtype=torch.float64).reshape(-1)
    client_deltas = local_vectors - theta_t.unsqueeze(0)
    delta_avg = (client_deltas * weights[:, None]).sum(dim=0)
    return theta_t, local_vectors, client_deltas, delta_avg


def client_disagreement_subspace(
    client_deltas: torch.Tensor,
    delta_avg: torch.Tensor,
    weights: torch.Tensor,
    *,
    rank_max: int = 8,
    eps: float = 1e-12,
) -> tuple[torch.Tensor | None, dict]:
    budget = float(delta_avg.norm().item())
    if budget <= eps:
        return None, {"fallback": True, "fallback_reason": "fedavg_update_norm_too_small"}
    u_avg = delta_avg / budget
    columns = []
    for delta_k, weight in zip(client_deltas, weights):
        residual = delta_k - delta_avg
        residual = residual - torch.dot(residual, u_avg) * u_avg
        columns.append(math.sqrt(float(weight)) * residual)
    matrix = torch.stack(columns, dim=1)
    if float(matrix.norm().item()) <= eps:
        return None, {"fallback": True, "fallback_reason": "no_client_disagreement_after_fedavg_projection"}
    u, singular, _ = torch.linalg.svd(matrix, full_matrices=False)
    effective_rank = int((singular > eps).sum().item())
    rank = min(int(rank_max), effective_rank)
    if rank < 1:
        return None, {"fallback": True, "fallback_reason": "svd_rank_zero"}
    q = u[:, :rank]
    orth_error = float((q.T @ q - torch.eye(rank, dtype=torch.float64)).abs().max().item())
    fedavg_orth_error = float((q.T @ delta_avg).abs().max().item() / max(budget, eps))
    energy = singular.square()
    retained = float(energy[:rank].sum().item() / max(float(energy.sum().item()), eps))
    return q, {
        "fallback": False,
        "rank": rank,
        "effective_rank": effective_rank,
        "singular_values": [float(x) for x in singular.tolist()],
        "energy_retained": retained,
        "orthogonality_error": orth_error,
        "fedavg_orthogonality_error": fedavg_orth_error,
    }


def _logits_from_theta(model, spec: FlatSpec, theta: torch.Tensor, features: torch.Tensor, batch_size: int = 2048) -> torch.Tensor:
    state = unflatten_state(theta, spec)
    chunks = []
    with torch.no_grad():
        for start in range(0, features.shape[0], batch_size):
            chunks.append(model.logits_from_cached_features(features[start:start + batch_size], state).detach().cpu())
    return torch.cat(chunks, dim=0)


def _true_class_margin(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    logits = logits.detach().cpu().to(torch.float64)
    labels = labels.detach().cpu().long()
    masked = logits.clone()
    masked[torch.arange(labels.numel()), labels] = -torch.inf
    return logits[torch.arange(labels.numel()), labels] - masked.max(dim=1).values


def probe_class_client_utilities(
    model,
    spec: FlatSpec,
    theta_bar: torch.Tensor,
    q: torch.Tensor,
    train_cache: Mapping,
    *,
    epsilon: float,
    batch_size: int = 2048,
) -> tuple[torch.Tensor, dict]:
    features = torch.as_tensor(train_cache["features"], dtype=torch.float32)
    labels = torch.as_tensor(train_cache["labels"], dtype=torch.long)
    client_ids = torch.as_tensor(train_cache["client_ids"], dtype=torch.long)
    num_clients = int(train_cache.get("num_clients", int(client_ids.max().item()) + 1))
    num_classes = int(train_cache.get("num_classes", int(labels.max().item()) + 1))
    utilities = torch.full((num_clients, num_classes, q.shape[1]), torch.nan, dtype=torch.float64)
    for direction_id in range(q.shape[1]):
        direction = q[:, direction_id]
        plus = _logits_from_theta(model, spec, theta_bar + epsilon * direction, features, batch_size=batch_size)
        minus = _logits_from_theta(model, spec, theta_bar - epsilon * direction, features, batch_size=batch_size)
        margin_delta = (_true_class_margin(plus, labels) - _true_class_margin(minus, labels)) / (2.0 * epsilon)
        for client_id in range(num_clients):
            client_mask = client_ids == client_id
            if not bool(client_mask.any()):
                continue
            for class_id in torch.unique(labels[client_mask]).tolist():
                mask = client_mask & (labels == int(class_id))
                utilities[client_id, int(class_id), direction_id] = margin_delta[mask].mean()
    report = {"epsilon": float(epsilon), "probe_rank": int(q.shape[1])}
    return utilities, report


def aggregate_class_utilities(utilities: torch.Tensor, client_class_counts: torch.Tensor) -> tuple[torch.Tensor, list[dict]]:
    counts = torch.as_tensor(client_class_counts, dtype=torch.float64)
    num_clients, num_classes, rank = utilities.shape
    class_utilities = torch.zeros(num_classes, rank, dtype=torch.float64)
    rows = []
    for class_id in range(num_classes):
        support = counts[:, class_id] > 0
        support_count = int(support.sum().item())
        if support_count == 0:
            rows.append({
                "class_id": class_id,
                "support_client_count": 0,
                "effective_client_number": 0.0,
                "utility_mean": math.nan,
                "utility_std": math.nan,
                "sign_agreement": math.nan,
            })
            continue
        weights = torch.sqrt(counts[support, class_id])
        weights = weights / weights.sum()
        selected = utilities[support, class_id, :]
        selected = torch.nan_to_num(selected, nan=0.0)
        class_utilities[class_id] = (selected * weights[:, None]).sum(dim=0)
        mass = counts[support, class_id] / counts[support, class_id].sum()
        effective = float(1.0 / torch.square(mass).sum().item())
        scalar = selected.mean(dim=1)
        target_sign = torch.sign(class_utilities[class_id].mean())
        if float(target_sign.item()) == 0.0:
            agreement = math.nan
        else:
            agreement = float((torch.sign(scalar) == target_sign).double().mean().item())
        rows.append({
            "class_id": class_id,
            "support_client_count": support_count,
            "effective_client_number": effective,
            "utility_mean": float(scalar.mean().item()),
            "utility_std": float(scalar.std(unbiased=False).item()) if scalar.numel() else math.nan,
            "sign_agreement": agreement,
        })
    return class_utilities, rows


def solve_safe_direction(
    class_utilities: torch.Tensor,
    global_class_counts: torch.Tensor,
    *,
    class_count_power: float = 0.5,
    eps: float = 1e-12,
) -> tuple[torch.Tensor | None, dict]:
    counts = torch.as_tensor(global_class_counts, dtype=torch.float64)
    weights = torch.pow(counts + 1.0, -float(class_count_power))
    weights = weights / weights.sum().clamp_min(eps)
    g = (class_utilities * weights[:, None]).sum(dim=0)
    positive_counts = counts[counts > 0]
    median = float(torch.median(positive_counts).item()) if positive_counts.numel() else 0.0
    common = counts >= median
    h = class_utilities[common].mean(dim=0) if bool(common.any()) else torch.zeros_like(g)
    direction = g.clone()
    h_norm_sq = float(torch.dot(h, h).item())
    projected = False
    if h_norm_sq > eps and float(torch.dot(h, direction).item()) < 0:
        direction = direction - (torch.dot(h, direction) / h_norm_sq) * h
        projected = True
    norm = float(direction.norm().item())
    if norm <= eps:
        return None, {
            "fallback": True,
            "fallback_reason": "functional_direction_norm_too_small",
            "common_count_median": median,
            "safe_projection_applied": projected,
        }
    v = direction / norm
    return v, {
        "fallback": False,
        "common_count_median": median,
        "common_class_count": int(common.sum().item()),
        "safe_projection_applied": projected,
        "safe_dot": float(torch.dot(h, v).item()),
        "class_count_power": float(class_count_power),
    }


def build_functional_cusp_delta(
    payload: Mapping,
    train_cache: Mapping,
    model,
    *,
    rank_max: int = 8,
    probe_rel_step: float = 0.1,
    steer_ratio: float = 0.25,
    class_count_power: float = 0.5,
    batch_size: int = 2048,
) -> tuple[torch.Tensor, dict, list[dict]]:
    spec = FlatSpec.from_dict(payload["flatten_spec"])
    theta_t, _, client_deltas, delta_avg = fedavg_delta_from_payload(payload, spec)
    weights = torch.as_tensor(payload["fedavg_weights"], dtype=torch.float64).reshape(-1)
    budget = float(delta_avg.norm().item())
    q, subspace_report = client_disagreement_subspace(client_deltas, delta_avg, weights, rank_max=rank_max)
    if q is None:
        return delta_avg.clone(), {
            "fallback": True,
            "fallback_reason": subspace_report.get("fallback_reason", "subspace_unavailable"),
            "norm_budget": budget,
            "subspace": subspace_report,
            "hyperparameters": {
                "rank_max": rank_max,
                "probe_rel_step": probe_rel_step,
                "steer_ratio": steer_ratio,
                "class_count_power": class_count_power,
            },
        }, []

    theta_bar = theta_t + delta_avg
    epsilon = float(probe_rel_step) * budget
    utilities, probe_report = probe_class_client_utilities(
        model, spec, theta_bar, q, train_cache, epsilon=epsilon, batch_size=batch_size
    )
    selected_ids = [int(x) for x in payload["selected_client_ids"]]
    compact_counts = torch.as_tensor(payload["client_class_counts"], dtype=torch.float64)
    full_counts = torch.zeros(utilities.shape[0], compact_counts.shape[1], dtype=torch.float64)
    for row_id, client_id in enumerate(selected_ids):
        if client_id < full_counts.shape[0]:
            full_counts[client_id] = compact_counts[row_id]
    class_utilities, diagnostic_rows = aggregate_class_utilities(
        utilities, full_counts
    )
    v, direction_report = solve_safe_direction(
        class_utilities,
        torch.as_tensor(payload["global_class_counts"], dtype=torch.float64),
        class_count_power=class_count_power,
    )
    if v is None:
        return delta_avg.clone(), {
            "fallback": True,
            "fallback_reason": direction_report.get("fallback_reason", "direction_unavailable"),
            "norm_budget": budget,
            "subspace": subspace_report,
            "probe": probe_report,
            "direction": direction_report,
            "hyperparameters": {
                "rank_max": rank_max,
                "probe_rel_step": probe_rel_step,
                "steer_ratio": steer_ratio,
                "class_count_power": class_count_power,
            },
        }, diagnostic_rows

    functional = math.sqrt(1.0 - float(steer_ratio) ** 2) * delta_avg + float(steer_ratio) * budget * (q @ v)
    norm_error = abs(float(functional.norm().item()) - budget) / max(budget, 1e-12)
    if norm_error > 1e-6:
        raise RuntimeError(f"Functional CUSP equal-norm check failed: relative_error={norm_error:.6g}")
    predicted = class_utilities @ v
    for row in diagnostic_rows:
        class_id = int(row["class_id"])
        row["predicted_steering_utility"] = float(predicted[class_id].item())
        row["global_train_count"] = int(torch.as_tensor(payload["global_class_counts"])[class_id].item())
    return functional, {
        "fallback": False,
        "fallback_reason": "",
        "norm_budget": budget,
        "norm_relative_error": norm_error,
        "subspace": subspace_report,
        "probe": probe_report,
        "direction": direction_report,
        "hyperparameters": {
            "rank_max": rank_max,
            "probe_rel_step": probe_rel_step,
            "steer_ratio": steer_ratio,
            "class_count_power": class_count_power,
        },
    }, diagnostic_rows
