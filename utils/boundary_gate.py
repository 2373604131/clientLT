"""Train-only candidate construction for the Boundary Repair Gate.

This module coordinates the four causal diagnostic states, the minimum-norm
repair, norm-matched baselines, and the exact post-matching safety check.  It
does not build datasets or touch official test data; the command script owns
those two boundary operations explicitly.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Mapping, Sequence

import torch

from utils.boundary_audit import (
    build_edge_catalog,
    diagnose_edges,
    edge_gradient,
    evaluate_state_edge_margins,
)
from utils.boundary_metrics import (
    EPS,
    edge_sample_weights,
    match_final_update_norm,
    safe_ratio,
    tensor64,
)
from utils.boundary_repair import (
    build_repair_candidates,
    choose_safe_backtracking_candidate,
    solve_minimum_norm_repair,
)
from utils.cusp_minimal import (
    FlatSpec,
    _classwise_delta,
    _oracle_cusp_delta,
    flatten_state,
    unflatten_state,
)
from utils.functional_cusp import candidate_hash_from_delta, fedavg_delta_from_payload


@dataclass(frozen=True)
class BoundaryGateConfig:
    """Frozen train-only hyperparameters for a first-version Boundary Gate."""

    gamma: float = 0.5
    tau: float = 0.0
    min_support_clients: int = 2
    max_edges_per_class: int = 3
    max_total_edges: int = 60
    repair_ratio: float = 0.25
    backtracking_alphas: tuple[float, ...] = (1.0, 0.5, 0.25, 0.125)
    min_deficit_closure: float = 0.0
    substantive_deficit_closure: float = 0.1
    max_non_target_margin_drop: float = 0.05
    max_semantic_repair_drift: float = 0.01
    gradient_batch_size: int = 2048
    solver_max_iterations: int = 500
    solver_tolerance: float = 1e-8
    solver_ridge: float = 1e-10
    random_seed: int = 2026
    tail_class_ratio: float = 0.2


def _head_tail_ids(payload: Mapping, metadata: Mapping, tail_ratio: float) -> tuple[list[int], list[int]]:
    """Use frozen dump groups when available, otherwise derive them from counts."""
    if "head_class_ids" in metadata and "tail_class_ids" in metadata:
        return [int(value) for value in metadata["head_class_ids"]], [int(value) for value in metadata["tail_class_ids"]]
    counts = torch.as_tensor(payload["global_class_counts"], dtype=torch.float64)
    num_tail = max(1, int(round(int(counts.numel()) * float(tail_ratio))))
    order = torch.argsort(counts, descending=True).tolist()
    tail_ids = sorted(int(value) for value in order[-num_tail:])
    head_ids = sorted(int(value) for value in order[:-num_tail])
    return head_ids, tail_ids


def _candidate_report(
    delta: torch.Tensor,
    budget: float,
    raw_delta: torch.Tensor | None = None,
    *,
    raw_norm_override: float | None = None,
) -> dict:
    """Create one uniform manifest record for a performance candidate."""
    final = tensor64(delta)
    raw = final if raw_delta is None else tensor64(raw_delta)
    final_norm = float(final.norm().item())
    raw_norm = float(raw.norm().item()) if raw_norm_override is None else float(raw_norm_override)
    return {
        "raw_final_norm": raw_norm,
        "norm_budget": float(budget),
        "final_norm": final_norm,
        "norm_scale_factor": float(budget / raw_norm) if raw_norm > EPS else 0.0,
        "norm_relative_error": abs(final_norm - float(budget)) / max(float(budget), EPS),
        "candidate_hash": candidate_hash_from_delta(final),
    }


def _normalise_direction(direction: torch.Tensor, norm: float) -> torch.Tensor:
    direction = tensor64(direction)
    source_norm = float(direction.norm().item())
    if source_norm <= EPS or norm <= EPS:
        return torch.zeros_like(direction)
    return direction * (float(norm) / source_norm)


def _inverse_support_mass_delta(
    client_deltas: torch.Tensor,
    fedavg_weights: torch.Tensor,
    selected_edge_ids: Sequence[int],
    edge_counts: torch.Tensor,
    selected_clients: Sequence[int],
    budget: float,
) -> tuple[torch.Tensor, dict]:
    """Reweight clients by inverse fragile-edge support mass, then norm match.

    A client receives ``alpha_k * sum_e 1[k in S_e] / mu_e`` before the
    coefficients are normalized.  This is a transparent comparison baseline,
    not a causal support counterfactual.
    """
    weights = tensor64(fedavg_weights).reshape(-1)
    selected = torch.as_tensor(selected_clients, dtype=torch.long)
    counts = tensor64(edge_counts)[selected]
    score = torch.zeros_like(weights)
    for edge_id in selected_edge_ids:
        support = counts[:, int(edge_id)] > 0
        mass = float(weights[support].sum().item())
        if mass > EPS:
            score[support] += 1.0 / mass
    coefficients = weights * score
    if float(coefficients.sum().item()) <= EPS:
        raw = (client_deltas * weights[:, None]).sum(dim=0)
        reason = "no_fragile_support"
    else:
        coefficients = coefficients / coefficients.sum()
        raw = (client_deltas * coefficients[:, None]).sum(dim=0)
        reason = "inverse_support_mass"
    delta, report = match_final_update_norm(raw, budget)
    report.update({
        "coefficient_rule": reason,
        "nonzero_client_count": int((coefficients > 0).sum().item()),
        "coefficient_sum": float(coefficients.sum().item()),
    })
    return delta, report


def _pooled_gain(values: torch.Tensor, before: torch.Tensor, edge_counts: torch.Tensor, edge_id: int) -> float:
    weights = edge_sample_weights(edge_counts, int(edge_id))
    valid = torch.isfinite(values[:, int(edge_id)]) & torch.isfinite(before[:, int(edge_id)]) & (weights > 0)
    if not bool(valid.any()):
        return math.nan
    return float((weights[valid] * (values[valid, int(edge_id)] - before[valid, int(edge_id)])).sum().item())


def _pooled_margin(values: torch.Tensor, edge_counts: torch.Tensor, edge_id: int) -> float:
    weights = edge_sample_weights(edge_counts, int(edge_id))
    valid = torch.isfinite(values[:, int(edge_id)]) & (weights > 0)
    if not bool(valid.any()):
        return math.nan
    return float((weights[valid] * values[valid, int(edge_id)]).sum().item())


def _semantic_distance(text_features: torch.Tensor, reference: torch.Tensor) -> float:
    gram = text_features.to(torch.float64) @ text_features.to(torch.float64).T
    reference_gram = reference.to(torch.float64) @ reference.to(torch.float64).T
    classes = max(int(gram.shape[0]), 1)
    return float((gram - reference_gram).norm().item() / float(classes * classes))


def _safety_checker(model, context: Mapping, cache: Mapping, edges: Sequence[Mapping], fragile_edges: Sequence[Mapping], config: BoundaryGateConfig):
    """Build an exact post-norm-matching safety predicate for backtracking."""
    theta_before = context["theta_before"]
    fedavg_delta = context["fedavg_delta"]
    spec: FlatSpec = context["spec"]
    before = context["before_margins"]
    fedavg = context["fedavg_margins"]
    edge_counts = context["edge_counts"]
    fragile_ids = {int(row["edge_id"]) for row in fragile_edges}
    before_text = model.text_features_from_trainable_state(unflatten_state(theta_before, spec))
    fedavg_text = model.text_features_from_trainable_state(unflatten_state(theta_before + fedavg_delta, spec))
    fedavg_semantic = _semantic_distance(fedavg_text, before_text)

    def check(delta: torch.Tensor) -> tuple[bool, dict]:
        values = evaluate_state_edge_margins(
            model, spec, theta_before + delta, cache, edges, batch_size=config.gradient_batch_size
        )
        finite = bool(torch.isfinite(delta).all()) and not bool(torch.isinf(values).any())
        closure_rows = []
        for row in fragile_edges:
            edge_id = int(row["edge_id"])
            gain = _pooled_gain(values, before, edge_counts, edge_id)
            margin = _pooled_margin(values, edge_counts, edge_id)
            denominator = float(config.gamma) * float(row["local_audit_gain"]) - float(row["gain_all_fedavg"])
            raw_closure = safe_ratio(gain - float(row["gain_all_fedavg"]), denominator)
            closure_rows.append({
                "edge_id": edge_id,
                "repair_gain": gain,
                "raw_deficit_closure": raw_closure,
                "clipped_deficit_closure": min(1.0, max(0.0, raw_closure)) if math.isfinite(raw_closure) else math.nan,
                "repair_margin": margin,
                "boundary_reversed": margin < 0.0,
            })
        closure_values = [row["raw_deficit_closure"] for row in closure_rows if math.isfinite(row["raw_deficit_closure"])]
        min_closure = min(closure_values) if closure_values else math.nan
        non_target_drops = []
        for edge_id in range(len(edges)):
            if edge_id in fragile_ids:
                continue
            candidate_gain = _pooled_gain(values, before, edge_counts, edge_id)
            fedavg_gain = _pooled_gain(fedavg, before, edge_counts, edge_id)
            if math.isfinite(candidate_gain) and math.isfinite(fedavg_gain):
                non_target_drops.append(candidate_gain - fedavg_gain)
        mean_non_target_drop = float(sum(non_target_drops) / len(non_target_drops)) if non_target_drops else 0.0
        worst_non_target_drop = min(non_target_drops) if non_target_drops else 0.0
        candidate_text = model.text_features_from_trainable_state(unflatten_state(theta_before + delta, spec))
        semantic_total = _semantic_distance(candidate_text, before_text)
        semantic_repair = semantic_total - fedavg_semantic
        checks = {
            "finite": finite,
            "minimum_deficit_closure": min_closure,
            "closure_safe": bool(closure_values) and min_closure >= float(config.min_deficit_closure),
            "mean_non_target_margin_drop": mean_non_target_drop,
            "worst_non_target_margin_drop": worst_non_target_drop,
            "non_target_safe": (
                mean_non_target_drop >= -float(config.max_non_target_margin_drop)
                and worst_non_target_drop >= -float(config.max_non_target_margin_drop)
            ),
            "semantic_total_drift": semantic_total,
            "semantic_fedavg_drift": fedavg_semantic,
            "semantic_repair_drift": semantic_repair,
            "semantic_safe": abs(semantic_repair) <= float(config.max_semantic_repair_drift),
            "edge_closure": closure_rows,
            "boundary_reversal_rate": (
                float(sum(item["boundary_reversed"] for item in closure_rows) / len(closure_rows)) if closure_rows else math.nan
            ),
        }
        return bool(checks["finite"] and checks["closure_safe"] and checks["non_target_safe"] and checks["semantic_safe"]), checks

    return check


def build_boundary_candidates(
    model,
    payload: Mapping,
    metadata: Mapping,
    audit_cache: Mapping,
    config: BoundaryGateConfig,
) -> tuple[dict[str, dict[str, torch.Tensor]], list[dict], list[dict], dict]:
    """Construct and freeze all train-only Gate candidates before test access."""
    head_ids, tail_ids = _head_tail_ids(payload, metadata, config.tail_class_ratio)
    edges, edge_counts = build_edge_catalog(audit_cache, max_edges_per_class=config.max_edges_per_class)
    diagnostics, fragile_edges, context = diagnose_edges(
        model,
        payload,
        audit_cache,
        edges,
        edge_counts,
        gamma=config.gamma,
        tau=config.tau,
        min_support_clients=config.min_support_clients,
        max_fragile_edges_per_class=config.max_edges_per_class,
        max_total_edges=config.max_total_edges,
        batch_size=config.gradient_batch_size,
    )
    theta_before = context["theta_before"]
    fedavg_delta = context["fedavg_delta"]
    spec: FlatSpec = context["spec"]
    _, _, client_deltas, _ = fedavg_delta_from_payload(payload, spec)
    weights = torch.as_tensor(payload["fedavg_weights"], dtype=torch.float64)
    budget = float(fedavg_delta.norm().item())
    if budget <= EPS:
        raise RuntimeError("Boundary Gate cannot match candidates because the FedAvg update norm is zero")

    gradients, gradient_rows = [], []
    for edge in fragile_edges:
        gradient, report = edge_gradient(model, spec, theta_before + fedavg_delta, audit_cache, edge)
        gradients.append(gradient)
        gradient_rows.append({**edge, **report})
    if gradients:
        gradient_matrix = torch.stack(gradients, dim=0)
        deficits = torch.tensor([float(row["visibility_deficit"]) for row in fragile_edges], dtype=torch.float64)
        raw_repair, solver_report = solve_minimum_norm_repair(
            gradient_matrix,
            deficits,
            max_iterations=config.solver_max_iterations,
            tolerance=config.solver_tolerance,
            ridge=config.solver_ridge,
        )
        candidates, cap_report = build_repair_candidates(
            fedavg_delta,
            raw_repair,
            repair_ratio=config.repair_ratio,
            alphas=config.backtracking_alphas,
        )
        safety_check = _safety_checker(model, context, audit_cache, edges, fragile_edges, config)
        repair_delta, repair_choice = choose_safe_backtracking_candidate(candidates, safety_check)
        selected_alpha = float(repair_choice["selected_alpha"]) if repair_delta is not None else 0.0
        if repair_delta is None:
            repair_delta = fedavg_delta.clone()
        capped_repair_norm = float(cap_report["capped_repair_norm"])
        matched_added_repair_norm = selected_alpha * capped_repair_norm
    else:
        gradient_matrix = torch.empty((0, theta_before.numel()), dtype=torch.float64)
        raw_repair = torch.zeros_like(theta_before)
        solver_report = {"status": "no_fragile_edges", "iterations": 0, "max_linear_violation": 0.0}
        cap_report = {"raw_repair_norm": 0.0, "capped_repair_norm": 0.0, "repair_norm_limit": config.repair_ratio * budget}
        repair_choice = {"accepted": False, "selected_alpha": None, "attempts": [], "reason": "no_fragile_edges"}
        selected_alpha = 0.0
        repair_delta = fedavg_delta.clone()
        matched_added_repair_norm = 0.0

    selected_safety = None
    for attempt in repair_choice.get("attempts", []):
        if attempt.get("accepted"):
            selected_safety = attempt.get("safety", {})
            break
    closure_by_edge = {
        int(item["edge_id"]): item
        for item in (selected_safety or {}).get("edge_closure", [])
    }
    for row in diagnostics:
        row["class_group"] = "tail" if int(row["class_id"]) in tail_ids else "non_tail"
        closure = closure_by_edge.get(int(row["edge_id"]))
        if closure is None:
            row.update({
                "repair_candidate_selected": False,
                "repair_gain": math.nan,
                "repair_margin": math.nan,
                "raw_deficit_closure": math.nan,
                "clipped_deficit_closure": math.nan,
                "boundary_reversed_after_repair": math.nan,
                "substantive_repair": False,
            })
        else:
            row.update({
                "repair_candidate_selected": True,
                "repair_gain": closure["repair_gain"],
                "repair_margin": closure["repair_margin"],
                "raw_deficit_closure": closure["raw_deficit_closure"],
                "clipped_deficit_closure": closure["clipped_deficit_closure"],
                "boundary_reversed_after_repair": closure["boundary_reversed"],
                "substantive_repair": (
                    bool(math.isfinite(closure["raw_deficit_closure"])
                         and closure["raw_deficit_closure"] >= float(config.substantive_deficit_closure))
                ),
            })
    closure_values = [
        float(item["raw_deficit_closure"])
        for item in closure_by_edge.values()
        if math.isfinite(float(item["raw_deficit_closure"]))
    ]

    cusp_metadata = {"head_class_ids": head_ids, "tail_class_ids": tail_ids}
    classwise_delta, classwise_report = _classwise_delta(payload, spec, theta_before, budget)
    cusp_delta, cusp_report = _oracle_cusp_delta(payload, cusp_metadata, client_deltas.T, fedavg_delta, budget)
    inverse_delta, inverse_report = _inverse_support_mass_delta(
        client_deltas,
        weights,
        [int(row["edge_id"]) for row in fragile_edges],
        edge_counts,
        payload["selected_client_ids"],
        budget,
    )
    ordinary_direction = gradient_matrix.mean(dim=0) if gradients else torch.zeros_like(theta_before)
    ordinary_repair = _normalise_direction(ordinary_direction, matched_added_repair_norm)
    ordinary_delta, ordinary_report = match_final_update_norm(fedavg_delta + ordinary_repair, budget)
    generator = torch.Generator(device="cpu").manual_seed(int(config.random_seed))
    random_repair = _normalise_direction(torch.randn(theta_before.numel(), generator=generator, dtype=torch.float64), matched_added_repair_norm)
    random_delta, random_report = match_final_update_norm(fedavg_delta + random_repair, budget)

    capped_repair = raw_repair * float(cap_report.get("scale_factor", 0.0))
    raw_for_method = {
        "fedavg": fedavg_delta,
        "inverse_support_mass": None,
        "classwise_aggregation": None,
        "cusp_minimal": None,
        "random_repair": fedavg_delta + random_repair,
        "ordinary_audit_gradient": fedavg_delta + ordinary_repair,
        "edge_level_boundary_repair": fedavg_delta + selected_alpha * capped_repair,
    }
    deltas = {
        "fedavg": fedavg_delta,
        "inverse_support_mass": inverse_delta,
        "classwise_aggregation": classwise_delta,
        "cusp_minimal": cusp_delta,
        "random_repair": random_delta,
        "ordinary_audit_gradient": ordinary_delta,
        "edge_level_boundary_repair": repair_delta,
    }
    reports = {
        "fedavg": {"fallback": False},
        "inverse_support_mass": inverse_report,
        "classwise_aggregation": classwise_report,
        "cusp_minimal": cusp_report,
        "random_repair": random_report,
        "ordinary_audit_gradient": ordinary_report,
        "edge_level_boundary_repair": {
            "repair_accepted": bool(repair_choice["accepted"]),
            "selected_backtracking_alpha": selected_alpha if repair_choice["accepted"] else None,
            "raw_repair_norm": float(raw_repair.norm().item()),
            "capped_repair_norm": capped_repair_norm,
            "matched_added_repair_norm": matched_added_repair_norm,
            "solver": solver_report,
            "backtracking": repair_choice,
        },
    }
    raw_norm_overrides = {
        "inverse_support_mass": inverse_report.get("raw_final_norm"),
        "classwise_aggregation": classwise_report.get("raw_norm"),
        "cusp_minimal": cusp_report.get("raw_norm"),
    }
    states, candidate_rows = {}, []
    for method, delta in deltas.items():
        raw_delta = raw_for_method[method]
        state = unflatten_state(theta_before + delta, spec)
        states[method] = state
        report = _candidate_report(delta, budget, raw_delta, raw_norm_override=raw_norm_overrides.get(method))
        candidate_rows.append({"candidate_id": method, "method": method, **report, **reports[method]})

    context.update({
        "edge_catalog": edges,
        "client_edge_counts": edge_counts,
        "head_class_ids": head_ids,
        "tail_class_ids": tail_ids,
        "norm_budget": budget,
        "config": asdict(config),
        "gradient_rows": gradient_rows,
        "solver_report": solver_report,
        "cap_report": cap_report,
        "repair_choice": repair_choice,
        "accepted_candidate_edge_nonregression_rate": (
            float(sum(value >= float(config.min_deficit_closure) for value in closure_values) / len(closure_values))
            if closure_values else math.nan
        ),
        "substantive_repair_edge_rate": (
            float(sum(value >= float(config.substantive_deficit_closure) for value in closure_values) / len(closure_values))
            if closure_values else math.nan
        ),
        "substantive_repair_all_fragile_edges": (
            bool(closure_values) and all(value >= float(config.substantive_deficit_closure) for value in closure_values)
        ),
        "boundary_reversal_rate": (
            float(sum(bool(item["boundary_reversed"]) for item in closure_by_edge.values()) / len(closure_by_edge))
            if closure_by_edge else math.nan
        ),
        "support_counterfactuals_norm_matched": False,
    })
    return states, candidate_rows, diagnostics, context
