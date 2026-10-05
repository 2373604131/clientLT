"""Shared-model LoRA baselines adapted to the frozen CLIP protocol.

Notation throughout: delta_W = scaling * B @ A. No client-private factors.
Paper equations and adaptation choices are recorded in third_party/paper_baselines.
"""
from contextlib import contextmanager
import math

import torch
from torch.nn import functional as F

from trainers.baselines.common import trainable_state

METHODS = ('ffa-lora', 'rolora', 'fedsvd', 'lora-a2')


def factor_pairs(state):
    pairs, consumed = [], set()
    for a in sorted(state):
        if not a.endswith('_lora_A'):
            continue
        b = a[:-1] + 'B'
        if b not in state:
            raise ValueError('Missing LoRA partner: ' + b)
        av, bv = state[a], state[b]
        if av.ndim != 2 or bv.ndim != 2 or av.shape[0] != bv.shape[1] or av.shape[0] > min(av.shape[1], bv.shape[0]):
            raise ValueError('Invalid LoRA factor dimensions: ' + a)
        if not bool(torch.isfinite(av).all() and torch.isfinite(bv).all()):
            raise FloatingPointError('Nonfinite LoRA state: ' + a)
        pairs.append((a, b)); consumed.update((a, b))
    if not pairs or consumed != set(state):
        raise ValueError('Expected complete LoRA A/B state, including frozen factors')
    return pairs


def active_factor(method, round_id):
    if method not in METHODS or round_id < 1:
        raise ValueError('Unknown factor baseline or invalid communication round')
    # First learn zero-initialized B; freezing B=0 would kill A's gradient.
    return 'B' if method in ('ffa-lora', 'fedsvd') or round_id % 2 == 1 else 'A'


def validate_options(options):
    settings = options['factor_baselines']
    a2 = settings['lora-a2']
    if (settings['privacy'] != 'non-DP' or not settings['fedsvd']['reparameterize_every_round']
        or settings['rolora']['first_active'] != 'B' or not settings['rolora']['switch_every_round']
        or not a2['probe_reset'] or a2['global_rank'] != 4):
        raise ValueError('Unsupported factor-baseline configuration')
    if (not isinstance(a2['selection_epochs'], int) or a2['selection_epochs'] < 1
        or not isinstance(a2['local_rank_budget'], int) or not 1 <= a2['local_rank_budget'] <= a2['global_rank']
        or not math.isfinite(a2['b_lr_multiplier']) or a2['b_lr_multiplier'] <= 0):
        raise ValueError('Invalid LoRA-A2 selection budget or learning-rate multiplier')


@contextmanager
def freeze_other_factor(model, active):
    parameters = dict(model.named_parameters())
    flags = {k: p.requires_grad for k, p in parameters.items()}
    pairs = factor_pairs(trainable_state(model))
    keys = [a if active == 'A' else b for a, b in pairs]
    try:
        for key, p in parameters.items():
            p.requires_grad_(key in keys)
            p.grad = None
        yield keys
    finally:
        for key, p in parameters.items():
            p.requires_grad_(flags[key])
            p.grad = None


def rank_masks(before, probe, active, local_rank_budget):
    """LoRA-A2 Eq.(4): ||outer(delta_B_i,A_i)|| or ||outer(B_i,delta_A_i)||.

    A single top-k is taken across ALL modules. Unselected deltas are zero,
    not unselected absolute parameters; previously learned factors survive.
    """
    pairs = factor_pairs(before)
    if set(probe) != set(before):
        raise ValueError('Probe state keys differ')
    if not isinstance(local_rank_budget, int) or isinstance(local_rank_budget, bool) or local_rank_budget < 1:
        raise ValueError('local_rank_budget must be a positive integer')
    candidates, masks = [], {}
    for a, b in pairs:
        key = a if active == 'A' else b
        if probe[key].shape != before[key].shape:
            raise ValueError('Probe shape differs')
        delta = (probe[key] - before[key]).double()
        if active == 'A':
            scores = before[b].double().norm(dim=0) * delta.norm(dim=1)
        else:
            scores = delta.norm(dim=0) * before[a].double().norm(dim=1)
        if not bool(torch.isfinite(scores).all()):
            raise FloatingPointError('Nonfinite importance scores')
        masks[key] = torch.zeros_like(before[key], dtype=torch.bool)
        candidates.extend((float(score), key, i) for i, score in enumerate(scores))
    keep = local_rank_budget * len(pairs)
    if keep > len(candidates):
        raise ValueError('Local rank budget exceeds global capacity')
    # Deterministic tie break by module name and rank index, including zero scores.
    chosen = sorted(candidates, key=lambda x: (-x[0], x[1], x[2]))[:keep]
    indices = {k: [] for k in masks}
    for _, key, i in chosen:
        if active == 'A':
            masks[key][i, :] = True
        else:
            masks[key][:, i] = True
        indices[key].append(i)
    return masks, indices


def svd_reparameterize(state):
    """FedSVD Eq.(7): A'=Vh[:r], B'=U[:,:r] S[:r], preserving raw BA.

    Unlike symmetric sqrt(S) factorization, this makes A's rows orthonormal.
    The model's scaling stays unchanged. CPU float64 gives reproducible
    restart behavior and handles rank-deficient/zero adapters without division.
    """
    result = {k: v.clone() for k, v in state.items()}
    errors = []
    for a, b in factor_pairs(state):
        rank = state[a].shape[0]
        product = state[b].double().cpu() @ state[a].double().cpu()
        u, singular, vh = torch.linalg.svd(product, full_matrices=False)
        result[a] = vh[:rank].to(state[a])
        result[b] = (u[:, :rank] * singular[:rank]).to(state[b])
        reconstructed = result[b].double().cpu() @ result[a].double().cpu()
        error = float((product - reconstructed).norm() / product.norm().clamp_min(1e-12))
        if not math.isfinite(error) or error > 1e-5:
            raise FloatingPointError('FedSVD did not preserve the effective adapter')
        errors.append(error)
    return result, max(errors)


def aggregate(method, before, local, clients, sizes, round_id):
    """Sample-weighted active-factor deltas; missing sparse ranks mean zero delta."""
    pairs = factor_pairs(before)
    if not clients or len(clients) != len(sizes) or any(n <= 0 for n in sizes):
        raise ValueError('Invalid client weights')
    active = active_factor(method, round_id)
    active_keys = {a if active == 'A' else b for a, b in pairs}
    result = {k: v.clone() for k, v in before.items()}
    total = float(sum(sizes))
    for client, size in zip(clients, sizes):
        if set(local[client]) != set(before):
            raise ValueError('Client state is incomplete')
        for key, value in local[client].items():
            if value.shape != before[key].shape or not bool(torch.isfinite(value).all()):
                raise ValueError('Invalid client factor: ' + key)
            if key not in active_keys and not torch.equal(value, before[key]):
                raise ValueError('Frozen factor changed: ' + key)
            if key in active_keys:
                result[key].add_(value - before[key], alpha=size / total)
    error = 0.
    if method == 'fedsvd':
        result, error = svd_reparameterize(result)
    return result, {'svd_relative_error': error}


def local_train(model, loader, method, options, device, round_id, smoke=False):
    """CE local training, one active factor, fresh client optimizer each round.

    LoRA-A2 first probes one training epoch, discards that probe state AND its
    momentum, then trains masked deltas from the round-start global state.
    Probe compute is counted, not quietly hidden inside the 3 training epochs.
    """
    settings = options['factor_baselines']
    validate_options(options)
    active = active_factor(method, round_id)
    before = trainable_state(model)
    pairs = factor_pairs(before)
    if method == 'lora-a2' and any(before[a].shape[0] != settings['lora-a2']['global_rank'] for a, _ in pairs):
        raise ValueError('LoRA-A2 global capacity differs from its declared configuration')
    params = dict(model.named_parameters())
    costs = dict(optimizer_steps=0, student_forward_images=0, teacher_forward_images=0,
        backward_images=0, projection_conflicts=0, loss_sum=0., selection_optimizer_steps=0,
        selection_forward_images=0, selected_rank_slots=0, update_parameters=0,
        payload_upload_bytes=0, payload_download_bytes=0)
    lr = options['lr']
    if method == 'lora-a2' and active == 'B':
        lr *= settings['lora-a2']['b_lr_multiplier']
    if not math.isfinite(lr) or lr <= 0:
        raise ValueError('Invalid active-factor learning rate')
    audit = dict(method=method, round=round_id, active_factor=active, learning_rate=lr,
                 initialization='shared protocol Kaiming A, zero B', rank_indices={})
    model.train()
    with freeze_other_factor(model, active) as keys:
        def optimize(epochs, masks=None, selection=False):
            optimizer = torch.optim.SGD([params[k] for k in keys], lr=lr,
                momentum=options['momentum'], weight_decay=options['weight_decay'])
            for _ in range(1 if smoke else epochs):
                for batch in loader:
                    images, labels = batch['img'].to(device), batch['label'].to(device)
                    optimizer.zero_grad(set_to_none=True)
                    loss = F.cross_entropy(model(images), labels)
                    if not bool(torch.isfinite(loss)):
                        raise FloatingPointError('Nonfinite factor-baseline loss')
                    loss.backward()
                    for key in keys:
                        grad = params[key].grad
                        if grad is not None:
                            if not bool(torch.isfinite(grad).all()):
                                raise FloatingPointError('Nonfinite factor gradient')
                            if masks is not None:
                                grad.mul_(masks[key])
                    optimizer.step()
                    # SGD momentum and weight decay must not move masked slots.
                    if masks is not None:
                        with torch.no_grad():
                            for key in keys:
                                params[key].copy_(torch.where(masks[key], params[key], before[key].to(params[key])))
                                momentum = optimizer.state[params[key]].get('momentum_buffer')
                                if momentum is not None:
                                    momentum.mul_(masks[key])
                    costs['optimizer_steps'] += 1
                    costs['student_forward_images'] += len(labels)
                    costs['backward_images'] += len(labels)
                    costs['loss_sum'] += float(loss.detach())
                    if selection:
                        costs['selection_optimizer_steps'] += 1
                        costs['selection_forward_images'] += len(labels)
                    if smoke:
                        break
        masks = None
        if method == 'lora-a2':
            optimize(settings['lora-a2']['selection_epochs'], selection=True)
            probe = {k: params[k].detach().cpu().clone() for k in before}
            masks, indices = rank_masks(before, probe, active, settings['lora-a2']['local_rank_budget'])
            audit['rank_indices'] = indices
            masks = {k: m.to(device) for k, m in masks.items()}
            with torch.no_grad():
                for key in before:
                    params[key].copy_(before[key])
        optimize(options['local_epochs'], masks=masks)
        costs['update_parameters'] = sum(int(masks[k].sum()) if masks is not None else params[k].numel() for k in keys)
        costs['selected_rank_slots'] = sum(len(v) for v in audit['rank_indices'].values()) if masks is not None else sum(before[a].shape[0] for a, _ in pairs)
        costs['payload_upload_bytes'] = sum((int(masks[k].sum()) if masks is not None else params[k].numel()) * params[k].element_size() for k in keys)
        if masks is not None:
            # Sparse wire-format estimate: int64 slot indices and per-module counts.
            costs['payload_upload_bytes'] += 8 * (costs['selected_rank_slots'] + len(keys))
        down_keys = list(before) if method == 'fedsvd' else keys
        costs['payload_download_bytes'] = sum(before[k].numel() * before[k].element_size() for k in down_keys)
    # Full A/B state must survive checkpoints and subsequent alternating rounds.
    state = trainable_state(model)
    factor_pairs(state)
    return state, costs, audit
