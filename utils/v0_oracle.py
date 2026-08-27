"""Label-oracle aggregation headroom utilities.

V0 is an offline kill-test.  Labels may be used to select a direction, but
the candidate update must remain in the disagreement subspace spanned by the
client uploads.  This module deliberately contains no CMSA mode discovery,
clustering, or official-test access.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from utils.cusp_minimal import (
    FlatSpec,
    flatten_state,
    make_flat_spec,
    write_json,
)
from utils.functional_cusp import client_disagreement_subspace, fedavg_delta_from_payload


V0_SCHEMA_VERSION = "v0_oracle_v1"


def class_groups_from_counts(
    global_class_counts: Sequence[float] | torch.Tensor,
    *,
    head_ratio: float = 0.4,
    tail_ratio: float = 0.2,
) -> dict[str, list[int]]:
    """Return deterministic frequency-ranked head/mid/tail class groups."""
    counts = torch.as_tensor(global_class_counts, dtype=torch.float64).reshape(-1)
    num_classes = int(counts.numel())
    if num_classes < 3:
        raise ValueError("V0 requires at least three classes")
    if not (0.0 < float(head_ratio) < 1.0 and 0.0 < float(tail_ratio) < 1.0):
        raise ValueError("head_ratio and tail_ratio must be in (0, 1)")
    if float(head_ratio) + float(tail_ratio) >= 1.0:
        raise ValueError("head_ratio + tail_ratio must be smaller than one")

    # Python's stable sort gives a deterministic class-id tie break.
    order = sorted(range(num_classes), key=lambda class_id: (-float(counts[class_id]), class_id))
    num_head = max(1, int(round(num_classes * float(head_ratio))))
    num_tail = max(1, int(round(num_classes * float(tail_ratio))))
    if num_head + num_tail >= num_classes:
        num_head = max(1, num_classes - num_tail - 1)
    return {
        "head": sorted(order[:num_head]),
        "mid": sorted(order[num_head : num_classes - num_tail]),
        "tail": sorted(order[num_classes - num_tail :]),
        "non_tail": sorted(order[: num_classes - num_tail]),
    }


def weighted_disagreement_scale(
    client_deltas: torch.Tensor,
    fedavg_delta: torch.Tensor,
    weights: torch.Tensor,
) -> float:
    client_deltas = torch.as_tensor(client_deltas, dtype=torch.float64)
    fedavg_delta = torch.as_tensor(fedavg_delta, dtype=torch.float64).reshape(-1)
    weights = torch.as_tensor(weights, dtype=torch.float64).reshape(-1)
    if client_deltas.ndim != 2 or client_deltas.shape[0] != weights.numel():
        raise ValueError("client_deltas must be [num_clients, num_parameters]")
    if client_deltas.shape[1] != fedavg_delta.numel():
        raise ValueError("client and FedAvg updates have incompatible dimensions")
    if not torch.isclose(weights.sum(), torch.tensor(1.0, dtype=torch.float64), atol=1e-8):
        raise ValueError("weights must sum to one")
    squared = torch.square(client_deltas - fedavg_delta.unsqueeze(0)).sum(dim=1)
    return math.sqrt(max(0.0, float(torch.dot(weights, squared).item())))


def maximum_trust_angle(
    norm_budget: float,
    trust_radius: float,
    *,
    max_angle: float = math.pi / 2.0,
) -> float:
    """Maximum equal-norm spherical angle allowed by a chord-distance trust region."""
    budget = float(norm_budget)
    radius = max(0.0, float(trust_radius))
    if budget <= 1e-12 or radius <= 0.0:
        return 0.0
    ratio = min(1.0, radius / (2.0 * budget))
    return min(float(max_angle), 2.0 * math.asin(ratio))


def sphere_candidate_from_coordinates(
    fedavg_delta: torch.Tensor,
    basis: torch.Tensor,
    coordinates: torch.Tensor,
    *,
    trust_radius: float,
    max_angle: float = math.pi / 2.0,
) -> tuple[torch.Tensor, dict]:
    """Map tangent coordinates to an equal-norm, trust-region-safe update."""
    fedavg_delta = torch.as_tensor(fedavg_delta, dtype=torch.float64).reshape(-1)
    basis = torch.as_tensor(basis, dtype=torch.float64)
    coordinates = torch.as_tensor(coordinates, dtype=torch.float64).reshape(-1)
    budget = float(fedavg_delta.norm().item())
    if budget <= 1e-12:
        return fedavg_delta.clone(), {
            "fallback": True,
            "fallback_reason": "fedavg_update_norm_too_small",
            "angle": 0.0,
            "trust_distance": 0.0,
        }
    if basis.ndim != 2 or basis.shape[0] != fedavg_delta.numel() or basis.shape[1] != coordinates.numel():
        raise ValueError("basis/coordinate dimensions do not match FedAvg")

    angle_cap = maximum_trust_angle(budget, trust_radius, max_angle=max_angle)
    coordinate_norm = float(coordinates.norm().item())
    if coordinate_norm <= 1e-15 or angle_cap <= 0.0:
        candidate = fedavg_delta.clone()
        angle = 0.0
    else:
        angle = min(coordinate_norm, angle_cap)
        tangent = basis @ (coordinates / coordinate_norm)
        tangent = tangent - torch.dot(tangent, fedavg_delta / budget) * (fedavg_delta / budget)
        tangent_norm = float(tangent.norm().item())
        if tangent_norm <= 1e-12:
            candidate = fedavg_delta.clone()
            angle = 0.0
        else:
            tangent = tangent / tangent_norm
            candidate = budget * (math.cos(angle) * fedavg_delta / budget + math.sin(angle) * tangent)

    final_norm = float(candidate.norm().item())
    trust_distance = float((candidate - fedavg_delta).norm().item())
    norm_relative_error = abs(final_norm - budget) / max(budget, 1e-12)
    if norm_relative_error > 1e-8:
        raise RuntimeError(f"V0 equal-norm construction failed: relative_error={norm_relative_error:.6g}")
    if trust_distance > float(trust_radius) + 1e-8:
        raise RuntimeError(
            "V0 trust-region construction failed: "
            f"distance={trust_distance:.6g} radius={float(trust_radius):.6g}"
        )
    return candidate, {
        "fallback": False,
        "fallback_reason": "",
        "angle": float(angle),
        "angle_cap": float(angle_cap),
        "trust_distance": trust_distance,
        "norm_budget": budget,
        "final_norm": final_norm,
        "norm_relative_error": norm_relative_error,
        "fedavg_alignment": float(torch.dot(candidate, fedavg_delta).item()),
    }


def harmonic_mean(values: Sequence[float]) -> float:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    if len(finite) != len(values) or any(value <= 0.0 for value in finite):
        return math.nan
    return float(len(finite) / sum(1.0 / value for value in finite))


def metrics_from_logits(
    logits: torch.Tensor,
    labels: torch.Tensor,
    groups: Mapping[str, Sequence[int]],
) -> tuple[dict[str, float], list[dict]]:
    """Compute class-balanced losses and head/mid/tail accuracies."""
    logits = torch.as_tensor(logits).detach().cpu().float()
    labels = torch.as_tensor(labels).detach().cpu().long().reshape(-1)
    if logits.ndim != 2 or logits.shape[0] != labels.numel():
        raise ValueError("logits must be [num_examples, num_classes]")
    losses = F.cross_entropy(logits, labels, reduction="none")
    predictions = logits.argmax(dim=1)
    per_class = []
    for class_id in range(int(logits.shape[1])):
        mask = labels == class_id
        count = int(mask.sum().item())
        per_class.append({
            "class_id": class_id,
            "count": count,
            "class_acc": (
                100.0 * float((predictions[mask] == labels[mask]).double().mean().item())
                if count else math.nan
            ),
            "class_loss": float(losses[mask].mean().item()) if count else math.nan,
        })

    def group_mean(group_name: str, key: str) -> float:
        ids = {int(value) for value in groups[group_name]}
        values = [float(row[key]) for row in per_class if row["class_id"] in ids and math.isfinite(float(row[key]))]
        return float(sum(values) / len(values)) if values else math.nan

    finite_acc = [float(row["class_acc"]) for row in per_class if math.isfinite(float(row["class_acc"]))]
    metrics = {
        "overall_acc": 100.0 * float((predictions == labels).double().mean().item()),
        "balanced_acc": float(sum(finite_acc) / len(finite_acc)) if finite_acc else math.nan,
        "head_acc": group_mean("head", "class_acc"),
        "mid_acc": group_mean("mid", "class_acc"),
        "tail_acc": group_mean("tail", "class_acc"),
        "non_tail_acc": group_mean("non_tail", "class_acc"),
        "head_loss": group_mean("head", "class_loss"),
        "mid_loss": group_mean("mid", "class_loss"),
        "tail_loss": group_mean("tail", "class_loss"),
    }
    metrics["h3"] = harmonic_mean([metrics["head_acc"], metrics["mid_acc"], metrics["tail_acc"]])
    return metrics, per_class


def gap_closure(candidate: float, fedavg: float, support_only: float, *, eps: float = 1e-12) -> float:
    denominator = float(support_only) - float(fedavg)
    if not all(math.isfinite(float(value)) for value in (candidate, fedavg, support_only)) or denominator <= eps:
        return math.nan
    return (float(candidate) - float(fedavg)) / denominator


def oracle_objective(metrics: Mapping[str, float], lambda_head: float, lambda_mid: float) -> float:
    return (
        float(metrics["tail_loss"])
        + float(lambda_head) * float(metrics["head_loss"])
        + float(lambda_mid) * float(metrics["mid_loss"])
    )


@dataclass(frozen=True)
class SpanOracleResult:
    delta: torch.Tensor
    coordinates: torch.Tensor
    metrics: dict
    report: dict


def optimize_span_oracle(
    evaluate_delta: Callable[[torch.Tensor], Mapping[str, float]],
    fedavg_delta: torch.Tensor,
    basis: torch.Tensor,
    *,
    gamma: float,
    disagreement_scale: float,
    lambda_head: float,
    lambda_mid: float,
    iterations: int = 4,
    probe_angle: float = 0.02,
    max_angle: float = math.pi / 2.0,
    initial_coordinates: torch.Tensor | None = None,
    initialization: str = "fedavg",
) -> SpanOracleResult:
    """Derivative-free projected optimization in the client disagreement span."""
    fedavg_delta = torch.as_tensor(fedavg_delta, dtype=torch.float64).reshape(-1)
    basis = torch.as_tensor(basis, dtype=torch.float64)
    trust_radius = max(0.0, float(gamma) * float(disagreement_scale))
    angle_cap = maximum_trust_angle(float(fedavg_delta.norm().item()), trust_radius, max_angle=max_angle)
    if initial_coordinates is None:
        coordinates = torch.zeros(basis.shape[1], dtype=torch.float64)
    else:
        coordinates = torch.as_tensor(initial_coordinates, dtype=torch.float64).reshape(-1).clone()
        if coordinates.numel() != basis.shape[1]:
            raise ValueError(
                "initial_coordinates dimension does not match disagreement basis: "
                f"{coordinates.numel()} != {basis.shape[1]}"
            )
        if not bool(torch.isfinite(coordinates).all()):
            raise ValueError("initial_coordinates contains NaN or Inf")
        coordinate_norm = float(coordinates.norm().item())
        if coordinate_norm > angle_cap and coordinate_norm > 0.0:
            coordinates.mul_(angle_cap / coordinate_norm)

    def candidate_and_metrics(current: torch.Tensor) -> tuple[torch.Tensor, dict, dict]:
        delta, geometry = sphere_candidate_from_coordinates(
            fedavg_delta, basis, current, trust_radius=trust_radius, max_angle=max_angle
        )
        metrics = dict(evaluate_delta(delta))
        return delta, metrics, geometry

    best_delta, best_metrics, best_geometry = candidate_and_metrics(coordinates)
    best_objective = oracle_objective(best_metrics, lambda_head, lambda_mid)
    evaluations = 1
    accepted_steps = 0

    if angle_cap > 0.0 and basis.shape[1] > 0:
        for iteration in range(max(0, int(iterations))):
            step_probe = min(float(probe_angle), max(angle_cap / 4.0, 1e-4))
            gradient = torch.zeros_like(coordinates)
            for direction_id in range(basis.shape[1]):
                unit = torch.zeros_like(coordinates)
                unit[direction_id] = step_probe
                plus = coordinates + unit
                minus = coordinates - unit
                for point in (plus, minus):
                    point_norm = float(point.norm().item())
                    if point_norm > angle_cap:
                        point.mul_(angle_cap / point_norm)
                _, plus_metrics, _ = candidate_and_metrics(plus)
                _, minus_metrics, _ = candidate_and_metrics(minus)
                evaluations += 2
                gradient[direction_id] = (
                    oracle_objective(plus_metrics, lambda_head, lambda_mid)
                    - oracle_objective(minus_metrics, lambda_head, lambda_mid)
                ) / (2.0 * step_probe)

            gradient_norm = float(gradient.norm().item())
            if not math.isfinite(gradient_norm) or gradient_norm <= 1e-12:
                break
            base_step = angle_cap / max(1.0, float(iterations))
            improved = False
            for multiplier in (2.0, 1.0, 0.5, 0.25):
                proposal = coordinates - multiplier * base_step * gradient / gradient_norm
                proposal_norm = float(proposal.norm().item())
                if proposal_norm > angle_cap:
                    proposal = proposal * (angle_cap / proposal_norm)
                proposal_delta, proposal_metrics, proposal_geometry = candidate_and_metrics(proposal)
                evaluations += 1
                proposal_objective = oracle_objective(proposal_metrics, lambda_head, lambda_mid)
                if proposal_objective < best_objective - 1e-10:
                    coordinates = proposal
                    best_delta = proposal_delta
                    best_metrics = proposal_metrics
                    best_geometry = proposal_geometry
                    best_objective = proposal_objective
                    accepted_steps += 1
                    improved = True
                    break
            if not improved:
                break

    report = {
        **best_geometry,
        "gamma": float(gamma),
        "disagreement_scale": float(disagreement_scale),
        "trust_radius": trust_radius,
        "lambda_head": float(lambda_head),
        "lambda_mid": float(lambda_mid),
        "iterations_requested": int(iterations),
        "accepted_steps": int(accepted_steps),
        "evaluation_count": int(evaluations),
        "objective": float(best_objective),
        "coordinate_norm": float(coordinates.norm().item()),
        "initialization": str(initialization),
    }
    return SpanOracleResult(best_delta, coordinates, best_metrics, report)


def random_span_candidates(
    fedavg_delta: torch.Tensor,
    basis: torch.Tensor,
    *,
    gamma: float,
    disagreement_scale: float,
    count: int,
    seed: int,
) -> list[tuple[torch.Tensor, dict]]:
    basis = torch.as_tensor(basis, dtype=torch.float64)
    budget = float(torch.as_tensor(fedavg_delta, dtype=torch.float64).norm().item())
    trust_radius = max(0.0, float(gamma) * float(disagreement_scale))
    angle_cap = maximum_trust_angle(budget, trust_radius)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    rows = []
    for index in range(max(0, int(count))):
        coordinates = torch.randn(basis.shape[1], generator=generator, dtype=torch.float64)
        coordinates = coordinates / coordinates.norm().clamp_min(1e-12) * angle_cap
        delta, report = sphere_candidate_from_coordinates(
            fedavg_delta, basis, coordinates, trust_radius=trust_radius
        )
        rows.append((delta, {**report, "random_index": index, "gamma": float(gamma)}))
    return rows


def support_normalized_deltas(payload: Mapping) -> dict[int, torch.Tensor]:
    """Build the per-class, non-deployable support-only diagnostic states."""
    spec = FlatSpec.from_dict(payload["flatten_spec"])
    _, _, client_deltas, _ = fedavg_delta_from_payload(payload, spec)
    weights = torch.as_tensor(payload["fedavg_weights"], dtype=torch.float64).reshape(-1)
    counts = torch.as_tensor(payload["client_class_counts"], dtype=torch.float64)
    if counts.shape[0] != client_deltas.shape[0]:
        raise ValueError("client_class_counts must use the selected-client row order")
    result = {}
    for class_id in range(int(counts.shape[1])):
        support = counts[:, class_id] > 0
        support_mass = float(weights[support].sum().item())
        if support_mass <= 1e-12:
            continue
        normalized = weights[support] / support_mass
        result[class_id] = (client_deltas[support] * normalized[:, None]).sum(dim=0)
    return result


def save_v0_round_dump(
    *,
    output_dir: str | Path,
    args,
    cfg,
    epoch: int,
    global_before: Mapping[str, torch.Tensor],
    global_after: Mapping[str, torch.Tensor],
    local_weights: Sequence[Mapping[str, torch.Tensor]],
    selected_clients: Sequence[int],
    client_sample_counts: Sequence[int],
    client_class_counts: Mapping[int, torch.Tensor],
    global_class_counts: torch.Tensor,
    trainable_keys: Sequence[str],
) -> Path:
    """Save a compact ClipLoRA round without accessing validation or test."""
    selected = [int(client_id) for client_id in selected_clients]
    keys = tuple(sorted(str(key) for key in trainable_keys))
    if not keys or any("lora_" not in key for key in keys):
        raise ValueError("V0 ClipLoRA dump requires only LoRA trainable keys")
    before = {key: global_before[key].detach().cpu().clone() for key in keys}
    after = {key: global_after[key].detach().cpu().clone() for key in keys}
    compact_local = [
        {key: local_weights[client_id][key].detach().cpu().clone() for key in keys}
        for client_id in selected
    ]
    spec = make_flat_spec(before, keys)
    sample_counts = torch.tensor(
        [float(client_sample_counts[client_id]) for client_id in selected], dtype=torch.float64
    )
    weights = sample_counts / sample_counts.sum()
    compact_class_counts = torch.stack(
        [client_class_counts[client_id].detach().cpu().long() for client_id in selected], dim=0
    )
    groups = class_groups_from_counts(global_class_counts)
    run_dir = Path(output_dir) / "v0_oracle" / f"round_{int(epoch) + 1:03d}"
    run_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": V0_SCHEMA_VERSION,
        "flatten_spec": spec.as_dict(),
        "trainable_keys": list(keys),
        "global_before_trainable": before,
        "global_after_fedavg_trainable": after,
        "local_trainable_states": compact_local,
        "selected_client_ids": selected,
        "fedavg_weights": weights,
        "client_sample_counts": [int(client_sample_counts[client_id]) for client_id in selected],
        "client_class_counts": compact_class_counts,
        "global_class_counts": global_class_counts.detach().cpu().long(),
        "num_classes": int(len(global_class_counts)),
    }
    metadata = {
        "schema_version": V0_SCHEMA_VERSION,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "communication_round": int(epoch) + 1,
        "parameter_space": "uploaded_lora_parameter_state",
        "head_class_ids": groups["head"],
        "mid_class_ids": groups["mid"],
        "tail_class_ids": groups["tail"],
        "non_tail_class_ids": groups["non_tail"],
        "resolved_args": vars(args),
        "resolved_config": str(cfg),
        "validation_used_before_dump": False,
        "test_used_before_dump": False,
    }
    torch.save(payload, run_dir / "round_state.pt")
    write_json(run_dir / "metadata.json", metadata)
    return run_dir
