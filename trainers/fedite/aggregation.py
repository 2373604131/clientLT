import copy
from typing import Dict, Iterable, List, Sequence

import torch

from .utils import group_update_norm, weighted_average_tensors


def _fedavg_group(previous_global_state, local_states, weights, keys):
    new_values = {}
    for key in keys:
        pairs = [
            (state[key], weight)
            for state, weight in zip(local_states, weights)
            if key in state
        ]
        if not pairs:
            continue
        values = [value for value, _weight in pairs]
        key_weights = [weight for _value, weight in pairs]
        new_values[key] = weighted_average_tensors(values, key_weights).to(previous_global_state[key].dtype)
    return new_values


def _is_classwise_tail_key(previous_global_state, key, num_classes):
    value = previous_global_state.get(key)
    return value is not None and value.ndim > 0 and int(value.shape[0]) == int(num_classes)


def _tailagg_group(previous_global_state, local_states, local_stats, tail_keys, reliability, tail_momentum=1.0, eps=1e-12):
    """Aggregate tail keys while preserving class-wise evidence rows.

    Class-wise tail parameters, such as class-specific routing logits, are
    updated only from clients that observed the corresponding protected class.
    Other tail parameters are aggregated from clients with any protected
    positive evidence.
    """

    num_classes = int(reliability.numel())
    tail_momentum = float(max(0.0, min(float(tail_momentum), 1.0)))
    updates = {}
    eligible_clients = []
    eligible_weights = []
    protected_positive_total = 0

    for idx, stats in enumerate(local_stats):
        positive = int(stats.get("protected_positive_count", 0))
        protected_positive_total += positive
        if positive <= 0:
            continue
        support = torch.as_tensor(stats.get("protected_class_support", torch.zeros_like(reliability)), dtype=torch.float32)
        supported = support > 0
        client_reliability = float(reliability[supported].mean().item()) if supported.any() else 0.0
        weight = float(positive) * max(client_reliability, 0.0)
        if weight <= eps:
            continue
        eligible_clients.append(idx)
        eligible_weights.append(weight)

    for key in tail_keys:
        if key not in previous_global_state:
            continue
        old_value = previous_global_state[key].detach().cpu()
        if _is_classwise_tail_key(previous_global_state, key, num_classes):
            new_value = old_value.clone()
            row_updated = False
            for class_id in range(num_classes):
                row_values = []
                row_weights = []
                class_reliability = max(float(reliability[class_id].item()), 0.0)
                if class_reliability <= eps:
                    continue
                for idx, stats in enumerate(local_stats):
                    if int(stats.get("protected_positive_count", 0)) <= 0:
                        continue
                    support = torch.as_tensor(
                        stats.get("protected_class_support", torch.zeros_like(reliability)),
                        dtype=torch.float32,
                    )
                    support_count = float(support[class_id].item()) if class_id < int(support.numel()) else 0.0
                    if support_count <= 0 or key not in local_states[idx]:
                        continue
                    row_values.append(local_states[idx][key].detach().cpu()[class_id])
                    row_weights.append(support_count * class_reliability)
                if not row_values:
                    continue
                avg_row = weighted_average_tensors(row_values, row_weights).to(old_value.dtype)
                new_value[class_id] = old_value[class_id] + tail_momentum * (avg_row - old_value[class_id])
                row_updated = True
            if row_updated:
                updates[key] = new_value.to(previous_global_state[key].dtype)
        elif eligible_clients:
            values = [local_states[idx][key] for idx in eligible_clients if key in local_states[idx]]
            weights = [eligible_weights[pos] for pos, idx in enumerate(eligible_clients) if key in local_states[idx]]
            if values:
                avg_value = weighted_average_tensors(values, weights).to(old_value.dtype)
                updates[key] = (old_value + tail_momentum * (avg_value - old_value)).to(previous_global_state[key].dtype)

    diagnostics = {
        "eligible": eligible_clients,
        "tail_weights": eligible_weights,
        "protected_positive_total": protected_positive_total,
    }
    return updates, diagnostics


def aggregate_round_stats(local_stats, num_classes):
    M = torch.zeros(num_classes, dtype=torch.float32)
    Q = torch.zeros(num_classes, dtype=torch.float32)
    H = torch.zeros(num_classes, dtype=torch.float32)
    write_sum = torch.zeros(num_classes, dtype=torch.float32)
    write_count = torch.zeros(num_classes, dtype=torch.float32)
    protected_positive = 0
    for stats in local_stats:
        M += torch.as_tensor(stats.get("M", torch.zeros(num_classes)), dtype=torch.float32)
        Q += torch.as_tensor(stats.get("Q", torch.zeros(num_classes)), dtype=torch.float32)
        H += torch.as_tensor(stats.get("H", torch.zeros(num_classes)), dtype=torch.float32)
        write_sum += torch.as_tensor(stats.get("write_sum", torch.zeros(num_classes)), dtype=torch.float32)
        write_count += torch.as_tensor(stats.get("write_count", torch.zeros(num_classes)), dtype=torch.float32)
        protected_positive += int(stats.get("protected_positive_count", 0))
    U = write_sum / write_count.clamp_min(1.0)
    return {
        "M": M,
        "Q": Q,
        "H": H,
        "U": U,
        "write_sum": write_sum,
        "write_count": write_count,
        "protected_positive_count": protected_positive,
    }


def aggregate_fedite(
    local_states,
    local_stats,
    previous_global_state,
    shared_keys,
    gate_keys,
    tail_keys,
    observer_state,
    tail_active=True,
    tail_momentum=1.0,
    eps=1e-12,
):
    """Reliability-weighted survival-aware aggregation for FedITE."""

    new_state = copy.deepcopy(previous_global_state)
    sample_weights = [max(float(stats.get("num_samples", 0)), 1.0) for stats in local_stats]
    for group_keys in (shared_keys, gate_keys):
        updates = _fedavg_group(previous_global_state, local_states, sample_weights, group_keys)
        new_state.update(updates)

    R = observer_state["class_state"][:, 1].detach().cpu().float()

    if not tail_active:
        for key in tail_keys:
            if key in previous_global_state:
                new_state[key] = previous_global_state[key].detach().cpu().clone()
        diagnostics = {
            "eligible_tail_clients": 0,
            "selected_clients": len(local_states),
            "protected_positive_samples": 0,
            "tail_weight_min": 0.0,
            "tail_weight_max": 0.0,
            "tail_weight_mean": 0.0,
            "mean_reliability_eligible": 0.0,
            "kept_previous_tail": True,
            "tail_active": False,
        }
    else:
        updates, tail_diag = _tailagg_group(
            previous_global_state,
            local_states,
            local_stats,
            tail_keys,
            R,
            tail_momentum=tail_momentum,
            eps=eps,
        )
        eligible = tail_diag["eligible"]
        tail_weights = tail_diag["tail_weights"]
        protected_positive_total = tail_diag["protected_positive_total"]
        if updates:
            new_state.update(updates)
            weight_tensor = torch.as_tensor(tail_weights, dtype=torch.float32)
            diagnostics = {
                "eligible_tail_clients": len(eligible),
                "selected_clients": len(local_states),
                "protected_positive_samples": protected_positive_total,
                "tail_weight_min": float(weight_tensor.min().item()),
                "tail_weight_max": float(weight_tensor.max().item()),
                "tail_weight_mean": float(weight_tensor.mean().item()),
                "mean_reliability_eligible": float((weight_tensor / torch.as_tensor([max(float(local_stats[i].get("protected_positive_count", 1)), 1.0) for i in eligible])).mean().item()),
                "kept_previous_tail": False,
                "tail_active": True,
            }
        else:
            for key in tail_keys:
                if key in previous_global_state:
                    new_state[key] = previous_global_state[key].detach().cpu().clone()
            diagnostics = {
                "eligible_tail_clients": 0,
                "selected_clients": len(local_states),
                "protected_positive_samples": protected_positive_total,
                "tail_weight_min": 0.0,
                "tail_weight_max": 0.0,
                "tail_weight_mean": 0.0,
                "mean_reliability_eligible": 0.0,
                "kept_previous_tail": True,
                "tail_active": True,
            }

    diagnostics["shared_global_update_norm"] = group_update_norm(previous_global_state, new_state, shared_keys)
    diagnostics["gate_global_update_norm"] = group_update_norm(previous_global_state, new_state, gate_keys)
    diagnostics["tail_global_update_norm"] = group_update_norm(previous_global_state, new_state, tail_keys)
    local_tail_norms = [float(stats.get("tail_update_norm", 0.0)) for stats in local_stats if int(stats.get("protected_positive_count", 0)) > 0]
    mean_local_tail = sum(local_tail_norms) / max(len(local_tail_norms), 1)
    diagnostics["tail_retention_ratio"] = diagnostics["tail_global_update_norm"] / (mean_local_tail + eps)
    return new_state, diagnostics
