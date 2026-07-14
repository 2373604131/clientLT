from dataclasses import dataclass
from typing import Dict, List

import torch

from .state import sanitize_residual


def compute_pre_reliability(M, D, age, m0, d0, stale_horizon):
    r_m = M.float() / (M.float() + float(m0))
    r_d = D.float() / (D.float() + float(d0))
    r_delta = torch.exp(-age.float() / max(float(stale_horizon), 1e-6))
    r_pre = (r_m.clamp_min(0.0) * r_d.clamp_min(0.0) * r_delta.clamp_min(0.0)).clamp_min(0.0).pow(1.0 / 3.0)
    return r_pre.clamp(0.0, 1.0)


def direction_consistency(unit_direction_sum, valid_count):
    valid = torch.as_tensor(valid_count, dtype=torch.float32).clamp_min(0.0)
    denom = valid.view(-1, 1).clamp_min(1.0)
    c_dir = unit_direction_sum.float().norm(dim=1) / denom.squeeze(1)
    c_dir = torch.where(valid > 0, c_dir, torch.zeros_like(c_dir))
    return c_dir.clamp(0.0, 1.0)


def corroboration(valid_count, nu0=2.0):
    valid = torch.as_tensor(valid_count, dtype=torch.float32).clamp_min(0.0)
    return (1.0 - torch.exp(-valid / max(float(nu0), 1e-6))).clamp(0.0, 1.0)


def aggregate_prompt_states(prompt_states, weights):
    total = float(sum(max(float(w), 0.0) for w in weights))
    if not prompt_states or total <= 0:
        return None, 0.0
    out = {}
    for key in prompt_states[0].keys():
        acc = None
        for state, weight in zip(prompt_states, weights):
            value = state[key].detach().cpu().float()
            acc = value * float(weight) if acc is None else acc + value * float(weight)
        out[key] = acc / total
    return out, total


def empty_sufficient_stats(num_tail, dim, device="cpu"):
    return {
        "update_sum": torch.zeros(num_tail, dim, dtype=torch.float32, device=device),
        "update_weight": torch.zeros(num_tail, dtype=torch.float32, device=device),
        "unit_direction_sum": torch.zeros(num_tail, dim, dtype=torch.float32, device=device),
        "gain_sum": torch.zeros(num_tail, dtype=torch.float32, device=device),
        "valid_count": torch.zeros(num_tail, dtype=torch.float32, device=device),
        "adapt_count": torch.zeros(num_tail, dtype=torch.float32, device=device),
        "candidate_skip_count": torch.zeros(num_tail, dtype=torch.float32, device=device),
        "hbs_loss_sum": torch.tensor(0.0, dtype=torch.float32, device=device),
        "hbs_loss_count": torch.tensor(0.0, dtype=torch.float32, device=device),
    }


def merge_sufficient_stats(stats_list, num_tail, dim):
    merged = empty_sufficient_stats(num_tail, dim)
    for stats in stats_list:
        for key in ["update_sum", "update_weight", "unit_direction_sum", "gain_sum", "valid_count", "adapt_count", "candidate_skip_count"]:
            merged[key] += stats.get(key, torch.zeros_like(merged[key])).detach().cpu().float()
        merged["hbs_loss_sum"] += torch.as_tensor(stats.get("hbs_loss_sum", 0.0)).detach().cpu().float()
        merged["hbs_loss_count"] += torch.as_tensor(stats.get("hbs_loss_count", 0.0)).detach().cpu().float()
    return merged


def update_core_state(
    state,
    sufficient_stats,
    variant="tcrm_core",
    disable_width=False,
    disable_write=False,
    disable_survival=False,
    server_rho_lr=1.0,
    gamma_decay=0.15,
    corroboration_scale_nu0=2.0,
):
    num_tail, dim = state.rho.shape
    state.r_pre = compute_pre_reliability(state.M, state.D, state.age, state.m0, state.d0, state.stale_horizon)

    update_weight = sufficient_stats["update_weight"].float()
    mean_update = sufficient_stats["update_sum"].float() / update_weight.view(-1, 1).clamp_min(1e-12)
    has_update = update_weight > 0

    if variant == "prompt_only":
        W = torch.zeros(num_tail, dtype=torch.float32)
        decay = torch.zeros_like(W)
        state.age = state.age + 1.0
    elif variant == "decoupled_residual_fedavg":
        W = has_update.float()
        decay = torch.zeros_like(W)
        tail_z = state.zero_shot_text[state.tail_class_ids].float()
        rho_new = state.rho.float().clone()
        rho_new[has_update] = rho_new[has_update] + mean_update[has_update]
        state.rho = sanitize_residual(rho_new, tail_z, state.rho_norm_bound)
        state.age = torch.where(has_update, torch.zeros_like(state.age), state.age + 1.0)
    else:
        valid = sufficient_stats["valid_count"].float()
        c_dir = direction_consistency(sufficient_stats["unit_direction_sum"].float(), valid)
        B = corroboration(valid, corroboration_scale_nu0)
        G = torch.where(valid > 0, sufficient_stats["gain_sum"].float() / valid.clamp_min(1.0), torch.zeros_like(valid)).clamp(0.0, 1.0)
        if bool(disable_write):
            W = has_update.float()
        else:
            # R_pre asks whether residual capacity is worth attempting; B asks
            # whether multiple clients corroborate the evidence; C_dir asks
            # whether their update directions agree; G asks whether the update
            # helped local held-out data. W is the write permission for global
            # long-term residual memory.
            W = (state.r_pre * B * c_dir * G).clamp(0.0, 1.0)
        decay = torch.zeros_like(W) if bool(disable_survival) else (
            float(gamma_decay) * (1.0 - W) * (1.0 - state.r_pre)
        ).clamp(0.0, 1.0)
        tail_z = state.zero_shot_text[state.tail_class_ids].float()
        rho_new = (1.0 - decay).view(-1, 1) * state.rho.float() + float(server_rho_lr) * W.view(-1, 1) * mean_update
        state.rho = sanitize_residual(rho_new, tail_z, state.rho_norm_bound)
        state.age = (1.0 - W) * (state.age + 1.0)
        state.last_direction_consistency = c_dir.detach().cpu()
        state.last_local_gain = G.detach().cpu()
        state.last_corroboration = B.detach().cpu()

    state.last_write = W.detach().cpu()
    state.last_decay = decay.detach().cpu()
    state.last_num_valid_contributors = sufficient_stats["valid_count"].detach().cpu().float()
    state.last_candidate_skip_count = sufficient_stats.get("candidate_skip_count", torch.zeros_like(state.last_num_valid_contributors)).detach().cpu().float()
    state.r_pre = compute_pre_reliability(state.M, state.D, state.age, state.m0, state.d0, state.stale_horizon)
    state.width_gate = torch.ones_like(state.r_pre) if bool(disable_width) else state.r_pre.clone()
    state.sanitize_()
    return state
