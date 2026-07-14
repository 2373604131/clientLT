import copy
from typing import Dict, List

import torch
from torch.nn.utils import clip_grad_norm_

from .classifier import assemble_text_bank, build_tail_direction, compute_logits
from .losses import cross_entropy_with_optional_adjustment, hbs_loss, true_class_margin
from .server_core import empty_sufficient_stats
from .state import sanitize_residual


def _clone_prompt(prompt_learner, prompt_state):
    local = copy.deepcopy(prompt_learner)
    local.load_trainable_state(prompt_state)
    return local


def _prompt_grad_norm(module):
    total = 0.0
    for param in module.parameters():
        if param.requires_grad and param.grad is not None:
            total += float(param.grad.detach().float().norm().item()) ** 2
    return total ** 0.5


def _make_text_bank(prompt_learner, state, logit_scale, device):
    non_tail_ids = torch.as_tensor(state.non_tail_class_ids, dtype=torch.long, device=device)
    non_tail_text = prompt_learner(non_tail_ids).float()
    tail_z = state.zero_shot_text[state.tail_class_ids].to(device, dtype=torch.float32)
    tail_text = build_tail_direction(tail_z, state.rho.to(device), state.width_gate.to(device), state.rho_norm_bound)
    return assemble_text_bank(
        int(state.zero_shot_text.shape[0]),
        state.zero_shot_text.to(device),
        state.non_tail_class_ids,
        non_tail_text,
        state.tail_class_ids,
        tail_text,
    )


def _resolved_variant(args):
    return str(getattr(args, "tcrm_variant", "") or getattr(args, "method", "tcrm_core"))


def _tail_split(indices, holdout_ratio, holdout_min):
    if isinstance(indices, torch.Tensor):
        indices = indices.to(dtype=torch.long)
    else:
        indices = torch.as_tensor(indices, dtype=torch.long)
    if indices.numel() == 0:
        return indices, indices.new_empty((0,))
    min_holdout = max(int(holdout_min), 0)
    if indices.numel() <= min_holdout:
        return indices, indices.new_empty((0,))
    holdout = max(int(holdout_min), int(round(float(holdout_ratio) * int(indices.numel()))))
    holdout = min(max(holdout, 0), int(indices.numel()) - 1)
    if holdout <= 0:
        return indices, indices.new_empty((0,))
    return indices[:-holdout], indices[-holdout:]


def train_tcrm_client(
    prompt_learner,
    state,
    features,
    labels,
    args,
    logit_scale,
    device="cpu",
):
    features = features.to(device, dtype=torch.float32)
    labels = labels.to(device).long()
    local_prompt = _clone_prompt(prompt_learner, state.prompt_state).to(device)
    local_prompt.train()
    before_state = {k: v.clone() for k, v in state.prompt_state.items()}
    prompt_opt = torch.optim.AdamW(
        [local_prompt.ctx_delta],
        lr=float(args.prompt_lr),
        weight_decay=float(getattr(args, "prompt_weight_decay", 0.0)),
        eps=float(getattr(args, "prompt_adam_eps", 1e-8)),
    )
    non_tail_set = set(state.non_tail_class_ids)
    tail_set = set(state.tail_class_ids)
    non_tail_mask = torch.as_tensor([int(y) in non_tail_set for y in labels.detach().cpu().tolist()], dtype=torch.bool, device=device)
    variant = _resolved_variant(args)
    tail_splits = {}
    hbs_adapt_indices = []
    for class_id in state.tail_class_ids:
        cls_idx = torch.where(labels == int(class_id))[0]
        if cls_idx.numel() == 0:
            continue
        adapt_idx, holdout_idx = _tail_split(cls_idx, args.tail_holdout_ratio, args.tail_holdout_min)
        tail_splits[int(class_id)] = (adapt_idx, holdout_idx)
        if adapt_idx.numel() > 0:
            hbs_adapt_indices.append(adapt_idx)
    hbs_idx = torch.cat(hbs_adapt_indices) if hbs_adapt_indices else labels.new_empty((0,))
    prompt_grad = 0.0
    hbs_values = []
    for _ in range(int(args.local_prompt_epochs)):
        if bool(non_tail_mask.any()) or (float(args.lambda_hbs) > 0 and hbs_idx.numel() > 0):
            prompt_opt.zero_grad(set_to_none=True)
            text_bank = _make_text_bank(local_prompt, state, logit_scale, device)
            loss = features.sum() * 0.0
            if bool(non_tail_mask.any()):
                logits = compute_logits(features[non_tail_mask], text_bank, logit_scale)
                loss = loss + cross_entropy_with_optional_adjustment(
                    logits,
                    labels[non_tail_mask],
                    class_prior=state.class_prior.to(device),
                    tau=float(args.logit_adjust_tau),
                )
            if float(args.lambda_hbs) > 0 and not bool(args.disable_hbs) and hbs_idx.numel() > 0:
                logits_all = compute_logits(features[hbs_idx], text_bank, logit_scale)
                zero_logits = compute_logits(features[hbs_idx], state.zero_shot_text.to(device), logit_scale)
                hbs = hbs_loss(logits_all, zero_logits, labels[hbs_idx], state.tail_class_ids, state.non_tail_class_ids, epsilon=float(args.epsilon_hbs))
                hbs_values.append(float(hbs.detach().item()))
                loss = loss + float(args.lambda_hbs) * hbs
            if torch.isfinite(loss):
                loss.backward()
                if local_prompt.ctx_delta.grad is not None and not torch.isfinite(local_prompt.ctx_delta.grad).all():
                    local_prompt.load_trainable_state(before_state)
                    break
                if float(getattr(args, "prompt_grad_clip", 0.0)) > 0:
                    clip_grad_norm_([local_prompt.ctx_delta], float(args.prompt_grad_clip))
                prompt_grad = max(prompt_grad, _prompt_grad_norm(local_prompt))
                prompt_opt.step()
                if not torch.isfinite(local_prompt.ctx_delta).all():
                    local_prompt.load_trainable_state(before_state)
                    break
            else:
                raise FloatingPointError("TCRM prompt loss became NaN/Inf")
    prompt_state = local_prompt.trainable_state()
    prompt_delta = 0.0
    for key, value in prompt_state.items():
        prompt_delta += float((value - before_state[key]).float().norm().item()) ** 2
    prompt_delta = prompt_delta ** 0.5

    dim = int(features.shape[1])
    stats = empty_sufficient_stats(len(state.tail_class_ids), dim, device=device)
    if hbs_values:
        stats["hbs_loss_sum"] = torch.tensor(sum(hbs_values), dtype=torch.float32, device=device)
        stats["hbs_loss_count"] = torch.tensor(len(hbs_values), dtype=torch.float32, device=device)

    if variant == "prompt_only":
        return {
            "prompt_state": prompt_state,
            "prompt_weight": int(non_tail_mask.sum().item()),
            "prompt_grad_norm": prompt_grad,
            "prompt_delta_norm": prompt_delta,
            "sufficient_stats": {k: v.detach().cpu() for k, v in stats.items()},
        }

    with torch.no_grad():
        non_tail_ids = torch.as_tensor(state.non_tail_class_ids, dtype=torch.long, device=device)
        cached_non_tail_text = local_prompt(non_tail_ids).detach().float()
    tail_z_all = state.zero_shot_text[state.tail_class_ids].to(device, dtype=torch.float32)
    base_tail_text = build_tail_direction(
        tail_z_all,
        state.rho.to(device),
        state.width_gate.to(device),
        state.rho_norm_bound,
    ).detach()
    base_text_bank = assemble_text_bank(
        int(state.zero_shot_text.shape[0]),
        state.zero_shot_text.to(device),
        state.non_tail_class_ids,
        cached_non_tail_text,
        state.tail_class_ids,
        base_tail_text,
    )

    rho_grad_norms = []
    for class_id in state.tail_class_ids:
        tail_index = state.tail_index_of_class[int(class_id)]
        if int(class_id) not in tail_splits:
            continue
        adapt_idx, holdout_idx = tail_splits[int(class_id)]
        if adapt_idx.numel() == 0:
            continue
        rho_param = torch.nn.Parameter(state.rho[tail_index].to(device).float().clone())
        opt = torch.optim.SGD([rho_param], lr=float(args.local_rho_lr))
        gate = 1.0 if variant == "decoupled_residual_fedavg" else float(state.width_gate[tail_index].item())
        candidate_valid = True
        for _ in range(int(args.local_rho_steps)):
            opt.zero_grad(set_to_none=True)
            rho_s = sanitize_residual(rho_param.view(1, -1), tail_z_all[tail_index].view(1, -1), float(args.rho_norm_bound)).squeeze(0)
            tail_text = base_tail_text.clone()
            tail_text[tail_index] = build_tail_direction(tail_z_all[tail_index], rho_s, gate, float(args.rho_norm_bound))
            text_bank = assemble_text_bank(
                int(state.zero_shot_text.shape[0]),
                state.zero_shot_text.to(device),
                state.non_tail_class_ids,
                cached_non_tail_text,
                state.tail_class_ids,
                tail_text,
            )
            logits = compute_logits(features[adapt_idx], text_bank, logit_scale)
            if not torch.isfinite(logits).all():
                stats["candidate_skip_count"][tail_index] += 1.0
                candidate_valid = False
                break
            loss = cross_entropy_with_optional_adjustment(
                logits,
                labels[adapt_idx],
                class_prior=state.class_prior.to(device),
                tau=float(args.logit_adjust_tau),
            )
            if torch.isfinite(loss):
                loss.backward()
                if rho_param.grad is not None and torch.isfinite(rho_param.grad).all():
                    if float(getattr(args, "rho_grad_clip", 0.0)) > 0:
                        clip_grad_norm_([rho_param], float(args.rho_grad_clip))
                    rho_grad_norms.append(float(rho_param.grad.detach().float().norm().item()))
                elif rho_param.grad is not None:
                    stats["candidate_skip_count"][tail_index] += 1.0
                    candidate_valid = False
                    break
                opt.step()
                with torch.no_grad():
                    rho_param.copy_(sanitize_residual(rho_param.view(1, -1), tail_z_all[tail_index].view(1, -1), float(args.rho_norm_bound)).squeeze(0))
            else:
                stats["candidate_skip_count"][tail_index] += 1.0
                candidate_valid = False
                break
        if not candidate_valid:
            continue
        with torch.no_grad():
            rho_new = sanitize_residual(rho_param.detach().view(1, -1), tail_z_all[tail_index].view(1, -1), float(args.rho_norm_bound)).squeeze(0)
            if not torch.isfinite(rho_new).all():
                stats["candidate_skip_count"][tail_index] += 1.0
                continue
            update = rho_new - state.rho[tail_index].to(device)
            update_norm = float(update.norm().item())
            if update_norm < float(args.update_norm_min):
                continue
            unit = update / update.norm().clamp_min(1e-12)
            weight = float(adapt_idx.numel())
            if variant == "decoupled_residual_fedavg":
                stats["update_sum"][tail_index] += weight * update
                stats["update_weight"][tail_index] += weight
                stats["unit_direction_sum"][tail_index] += unit
                stats["valid_count"][tail_index] += 1.0
                stats["adapt_count"][tail_index] += weight
                continue
            if holdout_idx.numel() == 0:
                continue
            base_logits = compute_logits(features[holdout_idx], base_text_bank, logit_scale)
            tail_text = base_tail_text.clone()
            tail_text[tail_index] = build_tail_direction(tail_z_all[tail_index], rho_new, gate, float(args.rho_norm_bound))
            candidate_bank = assemble_text_bank(
                int(state.zero_shot_text.shape[0]),
                state.zero_shot_text.to(device),
                state.non_tail_class_ids,
                cached_non_tail_text,
                state.tail_class_ids,
                tail_text,
            )
            candidate_logits = compute_logits(features[holdout_idx], candidate_bank, logit_scale)
            base_margin = true_class_margin(base_logits, labels[holdout_idx]).mean()
            candidate_margin = true_class_margin(candidate_logits, labels[holdout_idx]).mean()
            local_gain = torch.clamp((candidate_margin - base_margin) / max(float(args.gain_margin_scale), 1e-12), 0.0, 1.0)
            stats["update_sum"][tail_index] += weight * update
            stats["update_weight"][tail_index] += weight
            stats["unit_direction_sum"][tail_index] += unit
            stats["gain_sum"][tail_index] += local_gain
            stats["valid_count"][tail_index] += 1.0
            stats["adapt_count"][tail_index] += weight

    return {
        "prompt_state": prompt_state,
        "prompt_weight": int(non_tail_mask.sum().item()),
        "prompt_grad_norm": prompt_grad,
        "prompt_delta_norm": prompt_delta,
        "rho_grad_norm_mean": float(sum(rho_grad_norms) / max(len(rho_grad_norms), 1)),
        "rho_grad_norm_max": float(max(rho_grad_norms) if rho_grad_norms else 0.0),
        "sufficient_stats": {k: v.detach().cpu() for k, v in stats.items()},
    }
