import copy

import torch


def is_tail_stream_key(key):
    return key.startswith("tail_stream.") or key == "routed_prompt_delta"


def is_shared_stream_key(key, train_img_adap=True, train_lora=False, aggregate_logit_scale=False):
    if is_tail_stream_key(key):
        return False
    if key.startswith("prompt_learner."):
        return True
    if train_img_adap and (key.startswith("img_adap.") or key.startswith("image_adapter.")):
        return True
    if aggregate_logit_scale and key == "logit_scale":
        return True
    if train_lora and "lora_" in key:
        return True
    return False


def _selected_total(selected_clients, client_weights):
    if client_weights is None:
        return float(len(selected_clients))
    return float(sum(client_weights[idx] for idx in selected_clients))


def fedavg_keys(global_state, local_states, selected_clients, client_weights=None, keys=None):
    new_state = copy.deepcopy(global_state)
    if keys is None:
        keys = list(global_state.keys())
    total = max(_selected_total(selected_clients, client_weights), 1e-12)
    for key in keys:
        avg = torch.zeros_like(global_state[key].detach().cpu())
        for idx in selected_clients:
            weight = 1.0 if client_weights is None else float(client_weights[idx])
            avg += (weight / total) * local_states[idx][key].detach().cpu()
        new_state[key] = avg.to(dtype=global_state[key].dtype)
    return new_state


def fedavg_shared(global_state, local_states, selected_clients, client_weights, shared_keys):
    return fedavg_keys(global_state, local_states, selected_clients, client_weights, shared_keys)


def _classwise_tail_keys(global_state, num_classes, tail_keys=None):
    if tail_keys is None:
        tail_keys = [key for key in global_state.keys() if is_tail_stream_key(key)]
    keys = []
    for key in tail_keys:
        value = global_state[key]
        if value.ndim > 0 and value.shape[0] == num_classes:
            keys.append(key)
    return keys


def _client_concat_delta(global_state, local_state, classwise_keys, num_classes):
    chunks = []
    for key in classwise_keys:
        old = global_state[key].detach().cpu()
        new = local_state[key].detach().cpu()
        chunks.append((new - old).reshape(num_classes, -1).float())
    if not chunks:
        return torch.zeros(num_classes, 1)
    return torch.cat(chunks, dim=1)


def compute_survival_ratio(row_deltas, eps=1e-6):
    if not row_deltas:
        raise ValueError("row_deltas must contain at least one tensor")
    stacked = torch.stack([delta.float() for delta in row_deltas], dim=0)
    sum_delta = stacked.sum(dim=0)
    sum_norm = stacked.norm(dim=2).sum(dim=0)
    return sum_delta.norm(dim=1) / (sum_norm + float(eps))


def compute_tail_update_stats(
    global_state,
    local_states,
    selected_clients,
    num_classes,
    tail_keys=None,
    eps=1e-6,
):
    classwise_keys = _classwise_tail_keys(global_state, num_classes, tail_keys)
    exposure_proxy = torch.zeros(num_classes)
    observed_count = torch.zeros(num_classes)
    client_deltas = []

    for idx in selected_clients:
        concat_delta = _client_concat_delta(
            global_state,
            local_states[idx],
            classwise_keys,
            num_classes,
        )
        row_norm = concat_delta.norm(dim=1)
        exposure_proxy += row_norm
        observed_count += (row_norm > float(eps)).float()
        client_deltas.append(concat_delta)

    if client_deltas:
        survival_ratio = compute_survival_ratio(client_deltas, eps=eps)
    else:
        survival_ratio = torch.ones(num_classes)
    return exposure_proxy, observed_count, survival_ratio


def compute_tail_stream_positive_update_stats(*args, **kwargs):
    return compute_tail_update_stats(*args, **kwargs)


def compute_tail_stream_gradient_prior_proxy(
    global_state,
    local_states,
    selected_clients,
    num_classes,
    eps=1e-6,
):
    tail_keys = [
        key for key in global_state.keys()
        if key.startswith("tail_stream.")
        and global_state[key].ndim > 0
        and global_state[key].shape[0] == num_classes
    ]
    proxy, observed, _ = compute_tail_update_stats(
        global_state,
        local_states,
        selected_clients,
        num_classes,
        tail_keys=tail_keys,
        eps=eps,
    )
    return proxy, observed


def evidence_preserving_tailagg(
    global_state,
    local_states,
    selected_clients,
    tail_keys,
    gate,
    survival_ratio,
    base_momentum=0.6,
    low_survival_momentum=0.25,
    evidence_threshold=1e-6,
    update_clip=10.0,
    eps=1e-6,
):
    new_state = copy.deepcopy(global_state)
    gate = torch.as_tensor(gate).cpu().float()
    survival_ratio = torch.as_tensor(survival_ratio).cpu().float().clamp(0.0, 1.0)
    num_classes = int(gate.numel())
    classwise_keys = _classwise_tail_keys(global_state, num_classes, tail_keys)

    old_global = copy.deepcopy(global_state)
    exposure_proxy = torch.zeros(num_classes)
    observed_count = torch.zeros(num_classes)
    local_energy_sum = torch.zeros(num_classes)
    fedavg_row_energy = torch.zeros(num_classes)
    tailagg_row_energy = torch.zeros(num_classes)
    memory_row_norm = torch.zeros(num_classes)

    for key in tail_keys:
        old = global_state[key].detach().cpu()
        if key not in classwise_keys:
            new_state = fedavg_keys(new_state, local_states, selected_clients, None, [key])
            continue

        new_param = old.clone()
        for cls in range(num_classes):
            if gate[cls] <= 0:
                continue

            values = []
            norms = []
            raw_norms = []
            for idx in selected_clients:
                local_value = local_states[idx][key].detach().cpu()
                delta = local_value[cls] - old[cls]
                norm = delta.float().norm()
                if norm > float(evidence_threshold):
                    values.append(local_value[cls])
                    norms.append(norm.clamp(max=float(update_clip)))
                    raw_norms.append(norm)

            if not values:
                new_param[cls] = old[cls]
                continue

            weights = torch.stack(norms)
            weights = weights / weights.sum().clamp_min(float(eps))
            agg_value = torch.zeros_like(old[cls])
            for weight, value in zip(weights, values):
                agg_value += weight.to(value.dtype) * value

            s = survival_ratio[cls].clamp(0.0, 1.0)
            momentum = float(low_survival_momentum) + (
                float(base_momentum) - float(low_survival_momentum)
            ) * float(s.item())
            new_param[cls] = (1.0 - momentum) * old[cls] + momentum * agg_value

            raw_norm = torch.stack(raw_norms).sum()
            exposure_proxy[cls] += float(torch.stack(norms).sum().item())
            observed_count[cls] += float(len(values))
            local_energy_sum[cls] += float(raw_norm.item())

        new_state[key] = new_param.to(dtype=global_state[key].dtype)
        delta_tailagg = (new_state[key].detach().cpu() - old).reshape(num_classes, -1).float()
        tailagg_row_energy += delta_tailagg.norm(dim=1)

        fedavg_reference = fedavg_keys(copy.deepcopy(old_global), local_states, selected_clients, None, [key])
        delta_fedavg = (fedavg_reference[key].detach().cpu() - old).reshape(num_classes, -1).float()
        fedavg_row_energy += delta_fedavg.norm(dim=1)
        memory_row_norm += new_state[key].detach().cpu().reshape(num_classes, -1).float().norm(dim=1)

    _, update_observed_count, survival = compute_tail_update_stats(
        global_state,
        local_states,
        selected_clients,
        num_classes,
        tail_keys=classwise_keys,
        eps=eps,
    )
    observed_count = torch.maximum(observed_count, update_observed_count)
    local_energy_mean = local_energy_sum / observed_count.clamp_min(1.0)
    diagnostics = {
        "mode": "evidence_preserving",
        "observed_client_count": observed_count,
        "local_energy_sum": local_energy_sum,
        "local_energy_mean_observed": local_energy_mean,
        "fedavg_row_energy": fedavg_row_energy,
        "tailagg_row_energy": tailagg_row_energy,
        "memory_row_norm": memory_row_norm,
        "updated_rows": int((tailagg_row_energy > float(eps)).sum().item()),
        "updated_row_tensors": int((tailagg_row_energy > float(eps)).sum().item()),
        "updated_classes": int((observed_count > 0).sum().item()),
        "exposure_proxy": exposure_proxy,
        "survival_ratio": survival,
    }
    return new_state, {
        "exposure_proxy": exposure_proxy,
        "observed_count": observed_count,
        "survival_ratio": survival,
    }, diagnostics


def fedtef_v10_evidence_preserving_tailagg(
    global_state,
    local_states,
    selected_clients,
    gate,
    num_classes,
    survival_ratio=None,
    evidence_threshold=1e-6,
    update_clip=10.0,
    base_momentum=0.6,
    low_survival_momentum=0.25,
    eps=1e-6,
    return_diagnostics=False,
):
    if survival_ratio is None:
        survival_ratio = torch.ones(num_classes)
    tail_keys = [key for key in global_state.keys() if is_tail_stream_key(key)]
    new_state, stats, diagnostics = evidence_preserving_tailagg(
        global_state=global_state,
        local_states=local_states,
        selected_clients=selected_clients,
        tail_keys=tail_keys,
        gate=gate,
        survival_ratio=survival_ratio,
        base_momentum=base_momentum,
        low_survival_momentum=low_survival_momentum,
        evidence_threshold=evidence_threshold,
        update_clip=update_clip,
        eps=eps,
    )
    if return_diagnostics:
        return new_state, stats["exposure_proxy"], diagnostics
    return new_state, stats["exposure_proxy"]
