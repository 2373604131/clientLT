"""Global Fed-LT mechanisms, retaining the pinned upstream calculations.

Architecture adaptation: shared CLIP visual LoRA, frozen text head. Raw visual
features (before CLIP's L2 normalization) are used for prototype statistics and
FedLF feature losses. The original CLIP forward still produces all logits.
"""
from contextlib import contextmanager, redirect_stdout
import io
import math
from types import SimpleNamespace

import numpy as np
import torch
from torch.nn import functional as F

from trainers.baselines.common import trainable_state
from trainers.baselines.upstream import definitions


def validate_options(options):
    lf, yy, rl = (options['longtail_baselines'][k] for k in ('fedlf', 'fedyoyo', 'fedrela'))
    values = [v for cfg in (lf, yy, rl) for v in cfg.values() if isinstance(v, (int, float))]
    if any(not math.isfinite(v) or v < 0 for v in values):
        raise ValueError('Long-tail hyperparameters must be finite and nonnegative')
    if not 0 < lf['rs_alpha'] <= 1 or lf['distance_epsilon'] <= 0:
        raise ValueError('Invalid FedLF logit scale/distance epsilon')
    if not 0 <= yy['local_prior_gamma'] <= 1 or not 0 <= yy['global_prior_momentum'] < 1:
        raise ValueError('Invalid FedYoYo prior mixing')
    if yy['temperature'] <= 0 or yy['warmup_rounds'] <= 0 or yy['effective_epsilon'] <= 0:
        raise ValueError('Invalid FedYoYo temperature/warm-up/epsilon')
    if not isinstance(rl['relabel_after_round'], int) or not 0 <= rl['relabel_after_round'] < options['rounds']:
        raise ValueError('FedReLa must start before the last training round')
    if not isinstance(rl['posterior_passes'], int) or rl['posterior_passes'] < 1 or not 0 <= rl['threshold_percent'] <= 100:
        raise ValueError('Invalid FedReLa posterior passes/percentile')
    if rl['threshold_mode'] != 'adaptive' or rl['relabel_period'] != 0 or rl['host'] != 'FedAvg-LoRA CE':
        raise ValueError('This suite uses one-shot adaptive FedReLa on the CE host')
    if rl['post_relabel_lr_multiplier'] <= 0:
        raise ValueError('FedReLa learning rate multiplier must be positive')


@contextmanager
def visual_features(model):
    """Observe raw features in the SAME forward, without editing frozen A/AB."""
    captured = []
    handle = model.image_encoder.register_forward_hook(lambda module, args, output: captured.append(output))
    def forward(images):
        captured.clear()
        logits = model(images)
        if len(captured) != 1 or captured[0].ndim != 2:
            raise ValueError('Expected exactly one matrix of raw visual features')
        return captured.pop(), logits
    try:
        yield forward
    finally:
        handle.remove()
        captured.clear()


def fedlf_losses(features, logits, labels, centers, local_counts, options):
    scale = local_counts.to(logits) / local_counts.max()
    scale = scale * (1. - options['rs_alpha']) + options['rs_alpha']
    ce = F.cross_entropy(logits * scale[None, :], labels)
    # Upstream sqrt of a roundoff-negative squared distance can produce NaNs.
    # clamp only at numerical zero; leave all nondegenerate distances unchanged.
    squared = features.square().sum(1, keepdim=True) - 2 * features @ centers.T + centers.square().sum(1)[None, :]
    distances = squared.clamp_min(options['distance_epsilon']).sqrt()
    gap = torch.pdist(centers).max().clamp_max(100.)
    distances = distances + F.one_hot(labels, logits.shape[1]).to(logits) * gap
    center = F.cross_entropy(-distances, labels)
    upstream = definitions('fedlf', 'algorithm/fedlf.py', ('DecorrLoss',))
    decorr = upstream.DecorrLoss()(features)
    if not torch.is_tensor(decorr):  # official singleton-batch branch returns 0.0
        decorr = features.sum() * 0.
    loss = ce + options['center_weight'] * center + options['decorr_weight'] * decorr
    return loss, dict(ce=ce, center=center, decorr=decorr)


def effective_weight(features, labels, centers, num_classes, epsilon=1e-6):
    """Upstream Local.calculate_eff_weight, for ONE shuffled batch."""
    centered = features.detach() - centers[labels]
    effective = features.new_zeros(num_classes)
    for label in labels.unique():
        matrix = centered[labels == label]
        if matrix.shape[0] == 1:
            effective[label] = 1.
        else:
            dot = matrix @ matrix.T
            norm = matrix.square().sum(1).sqrt()[:, None]
            denominator = norm @ norm.T
            denominator = torch.where(denominator == 0, epsilon, denominator)
            correlation = dot / denominator
            average = matrix.new_ones(1, len(matrix)) / len(matrix)
            c = (average @ correlation @ average.T).squeeze().clamp_min(epsilon)
            effective[label] = 1. / c
    return effective


def global_effective_prior(client_values, previous, options):
    current = torch.ones_like(client_values[0]) + torch.stack(client_values).sum(0)
    if previous is None:
        abnormal = current > options['first_round_outlier_threshold']
        # Upstream empty mean is undefined if every class is an outlier.
        if abnormal.all():
            current = torch.ones_like(current)
        elif abnormal.any():
            current[abnormal] = current[~abnormal].mean()
    else:
        momentum = options['global_prior_momentum']
        current = previous.to(current) * momentum + current * (1. - momentum)
        helper = definitions('fedyoyo', 'data_loader/tools.py', ('apply_change_threshold',))
        current = helper.apply_change_threshold(current, previous.to(current), options['change_threshold'])
    if not torch.isfinite(current).all() or (current <= 0).any():
        raise FloatingPointError('Invalid FedYoYo effective class distribution')
    return current.detach(), (current / current.sum()).detach()


def fedyoyo_loss(logits, labels, prior, local_counts, round_id, options):
    local = local_counts.to(logits) / local_counts.sum()
    gamma = options['local_prior_gamma']
    updated_prior = (1. - gamma) * prior + gamma * local
    adjusted = logits + (updated_prior.pow(options['tau']) + 1e-9).log()
    weak, strong = adjusted.chunk(2, dim=0)
    if len(weak) != len(labels) or len(strong) != len(labels):
        raise ValueError('FedYoYo requires paired weak and strong views')
    selected = weak.detach().argmax(1) == labels
    if selected.any():
        kd = F.kl_div(F.log_softmax(strong[selected] / options['temperature'], dim=1),
                      F.softmax(weak[selected] / options['temperature'], dim=1).detach(), reduction='batchmean')
    else:
        kd = strong.sum() * 0.  # same zero update as upstream's empty-mask NaN guard
    ce = F.cross_entropy(adjusted, torch.cat([labels, labels]))
    # The official implementation does NOT multiply KD by temperature squared.
    loss = ce + options['distillation_weight'] * kd * min(round_id / options['warmup_rounds'], 1.)
    return loss, updated_prior.detach(), dict(ce=ce, kd=kd, selected=int(selected.sum()))


def fedrela_mapping(probabilities, labels, raw_ids, threshold_percent):
    """Execute official prior/z-score/threshold/stochastic relabel functions."""
    names = ('compute_normalized_priors', 'compute_classwise_statistics', 'zscore',
             'compute_adaptive_threshold', 'get_rho_with_tanh_norm')
    upstream = definitions('fedrela', 'util/FedReLa.py', names)
    probabilities = np.asarray(probabilities)
    labels = np.asarray(labels, dtype=np.int64)
    keys = [int(x) for x in raw_ids]
    if probabilities.ndim != 2 or probabilities.shape[0] != len(labels) or len(keys) != len(set(keys)) or len(keys) != len(labels):
        raise ValueError('Invalid FedReLa sample alignment')
    if not len(keys) or not np.isfinite(probabilities).all() or (probabilities < 0).any():
        raise ValueError('Invalid FedReLa posterior probabilities')
    if not np.allclose(probabilities.sum(1), 1., atol=1e-5):
        raise ValueError('FedReLa expects probabilities, not logits')
    classes = probabilities.shape[1]
    if (labels < 0).any() or (labels >= classes).any():
        raise ValueError('FedReLa labels outside the class range')
    pred = {k: (p.copy(), int(y)) for k, p, y in zip(keys, probabilities, labels)}
    args = SimpleNamespace(id='cliplora_adaptive')
    with redirect_stdout(io.StringIO()):
        priors, counts = upstream.compute_normalized_priors(labels, classes)
        means, stds, scores = upstream.compute_classwise_statistics(probabilities, labels, keys, classes)
        _, matrix = upstream.zscore(pred, means, stds, scores, 0)
        thresholds = upstream.compute_adaptive_threshold(matrix, threshold_percent, args, 0)
        mapping, new_counts, confusion, changed = upstream.get_rho_with_tanh_norm(
            pred, scores, priors, thresholds, classes, args)
    mapping = {int(k): int(v) for k, v in mapping.items()}
    for raw_id, original in zip(keys, labels):
        if raw_id in mapping and counts[mapping[raw_id]] >= counts[original]:
            raise ValueError('Upstream relabel violated the local majority-to-minority rule')
    return mapping, dict(changed=int(changed), samples=len(keys),
        original_counts=counts.tolist(), relabeled_counts=[int(new_counts[i]) for i in range(classes)],
        thresholds=np.asarray(thresholds).tolist())


def new_costs():
    return dict(optimizer_steps=0, student_forward_images=0, teacher_forward_images=0,
                backward_images=0, auxiliary_forward_images=0, loss_sum=0.,
                ce_loss_sum=0., center_loss_sum=0., decorr_loss_sum=0., kd_loss_sum=0.,
                selected_distillation_images=0, relabeled_training_images=0,
                statistics_upload_bytes=0, statistics_download_bytes=0)


@torch.no_grad()
def class_centers(model, loader, num_classes, device, smoke=False):
    sums = None
    counts = torch.zeros(num_classes, device=device)
    seen = 0
    with visual_features(model) as forward:
        for batch in loader:
            labels = batch['label'].to(device)
            features, _ = forward(batch['img'].to(device))
            if sums is None:
                sums = features.new_zeros(num_classes, features.shape[1])
            sums.index_add_(0, labels, features)
            counts.index_add_(0, labels, torch.ones_like(labels, dtype=counts.dtype))
            seen += len(labels)
            if smoke:
                break
    if sums is None:
        raise ValueError('Empty client statistics loader')
    centers = sums / counts.clamp_min(1)[:, None]
    return centers, counts, seen


@torch.no_grad()
def estimate_effective(model, loader, num_classes, device, options, smoke=False):
    model.eval()
    centers, _, seen = class_centers(model, loader, num_classes, device, smoke)
    total = centers.new_zeros(num_classes)
    with visual_features(model) as forward:
        for batch in loader:
            labels = batch['label'].to(device)
            features, _ = forward(batch['img'].to(device))
            total += effective_weight(features, labels, centers, num_classes, options['effective_epsilon'])
            seen += len(labels)
            if smoke:
                break
    return total, seen


@torch.no_grad()
def collect_posteriors(model, loader, device, passes):
    model.eval()
    records = {}
    for _ in range(passes):
        for batch in loader:
            probabilities = model(batch['img'].to(device)).softmax(1).cpu().numpy()
            for raw_id, label, p in zip(batch['index'].tolist(), batch['label'].tolist(), probabilities):
                if raw_id not in records:
                    records[raw_id] = [int(label), []]
                if records[raw_id][0] != label:
                    raise ValueError('Training label changed during posterior collection')
                records[raw_id][1].append(p)
    if not records or any(len(x[1]) != passes for x in records.values()):
        raise ValueError('Posterior collection must visit each training sample exactly once per pass')
    ids = sorted(records)
    probabilities = np.stack([np.mean(np.stack(records[k][1]), axis=0) for k in ids])
    return probabilities, np.array([records[k][0] for k in ids]), ids, len(ids) * passes


def local_train(model, loader, method, options, local_counts, device, round_id,
                prior=None, relabels=None, relabel_active=False, smoke=False):
    costs = new_costs()
    spec = options['longtail_baselines'][method]
    params = [p for p in model.parameters() if p.requires_grad]
    lr = options['lr'] * (spec['post_relabel_lr_multiplier'] if method == 'fedrela' and relabel_active else 1.)
    optimizer = torch.optim.SGD(params, lr=lr, momentum=options['momentum'], weight_decay=options['weight_decay'])
    model.train()
    centers = None
    if method == 'fedlf':
        centers, seen_counts, seen = class_centers(model, loader, len(local_counts), device, smoke)
        centers[seen_counts == 0] = 1e-8
        costs['auxiliary_forward_images'] += seen
    if method == 'fedyoyo':
        if prior is None:
            raise ValueError('FedYoYo requires its estimated round-start global prior')
        prior = prior.detach().to(device).clone()
    with visual_features(model) as forward:
        for _ in range(1 if smoke else options['local_epochs']):
            for batch in loader:
                labels = batch['label'].to(device)
                if method == 'fedrela' and relabel_active:
                    if relabels is None:
                        raise ValueError('Missing committed FedReLa relabel state')
                    labels = torch.tensor([relabels.get(int(i), int(y)) for i, y in zip(batch['index'], batch['label'])], device=device)
                    costs['relabeled_training_images'] += int((labels != batch['label'].to(device)).sum())
                optimizer.zero_grad(set_to_none=True)
                images = batch['img'].to(device)
                if method == 'fedyoyo':
                    images = torch.cat([images, batch['img_strong'].to(device)])
                features, logits = forward(images)
                if method == 'fedlf':
                    objective, parts = fedlf_losses(features, logits, labels, centers, local_counts.to(device), spec)
                elif method == 'fedyoyo':
                    objective, prior, parts = fedyoyo_loss(logits, labels, prior, local_counts.to(device), round_id, spec)
                    costs['selected_distillation_images'] += parts.pop('selected')
                elif method == 'fedrela':
                    objective = F.cross_entropy(logits, labels)
                    parts = dict(ce=objective)
                else:
                    raise ValueError('Unknown long-tail method: ' + method)
                if not torch.isfinite(objective):
                    raise FloatingPointError('Nonfinite ' + method + ' objective')
                objective.backward()
                if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in params):
                    raise FloatingPointError('Nonfinite ' + method + ' gradient')
                optimizer.step()
                costs['optimizer_steps'] += 1
                costs['student_forward_images'] += len(images)
                costs['backward_images'] += len(images)
                costs['loss_sum'] += float(objective.detach())
                for key, value in parts.items():
                    costs[key + '_loss_sum'] += float(value.detach())
                if smoke:
                    break
    return trainable_state(model), costs
