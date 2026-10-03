"""Training-only, client-local penalties for a single shared donor-C update."""
import math

import torch


VARIANTS = ('off', 'client', 'class')


def guard_config(base, variant):
    expected = dict(mode='shared', calibration_profile='coverage_tradeoff',
                    non_tail_sampling='class-cyclic', tail_weight=.35,
                    learning_rate=.3, probe_step=.1, regularization=.001, steps=2)
    if variant not in VARIANTS or any(base.get(k) != v for k, v in expected.items()):
        raise ValueError('Class guard requires frozen shared B, class-cyclic w0.35, lr0.3, two steps')
    result = dict(base)
    result.update(schema_version='donor_b_class_guard_v1', calibration_profile='class_guard',
        guard_variant=variant, guard_weight=0. if variant == 'off' else 1.,
        guard_reference='same event, same client/batch/images, ordinary post-FedAvg model; detached',
        guard_normalization='original client/group/sample weights; no extra class-macro weighting',
        guard_scope='client tail/non-tail group mean before ReLU' if variant == 'client'
                    else 'ReLU of each client-class mean before sample-weighted group mean',
        guard_commit='fixed second step; no rejection, replay, test gating or extra correction',
        calibration_objective='original mixed LA + one C penalty + fixed-weight local guard',
        guard_gradient_diagnostics='extra autograd traversal only when selected guard is active; separately counted')
    return result


def guard_losses(losses, reference, labels, count_tail, tail_weight=.35):
    """Return both penalties, preserving class sample counts within each group.

    Reference and grouping are constant. Each call concerns ONE client's batch;
    averaging happens after ReLU, never across clients. Zero has zero derivative.
    """
    if losses.ndim != 1 or losses.shape != labels.shape or losses.numel() == 0:
        raise ValueError('Expected a nonempty vector of per-image losses and labels')
    if labels.dtype not in (torch.int32, torch.int64) or not 0 < count_tail <= losses.numel():
        raise ValueError('Invalid labels or tail count')
    if not math.isfinite(tail_weight) or not 0 < tail_weight < 1:
        raise ValueError('Invalid tail weight')
    reference = reference.detach().to(losses)
    if reference.shape != losses.shape or not torch.isfinite(losses).all() or not torch.isfinite(reference).all():
        raise ValueError('Nonfinite or mismatched current/reference losses')
    tail_labels = set(labels[:count_tail].detach().cpu().tolist())
    if tail_labels & set(labels[count_tail:].detach().cpu().tolist()):
        raise ValueError('One class cannot occur in both tail and non-tail groups')
    groups = [('tail', slice(0, count_tail), tail_weight if count_tail < len(losses) else 1.)]
    if count_tail < len(losses):
        groups.append(('non_tail', slice(count_tail, None), 1-tail_weight))
    client_harm = losses.sum()*0.
    class_harm = losses.sum()*0.
    cells, group_rows = [], []
    for name, indices, weight in groups:
        current, baseline, target = losses[indices], reference[indices], labels[indices]
        # Mean of differences preserves exact zero for an identical reference.
        delta = (current-baseline).mean()
        client_harm = client_harm + weight*delta.relu()
        group_rows.append(dict(group=name, samples=len(current), group_weight=weight,
                               delta_la=float(delta.detach()), active=bool(delta.detach() > 0)))
        for c in target.unique(sorted=True).tolist():
            mask = target == c
            n = int(mask.sum())
            change = (current[mask]-baseline[mask]).mean()
            coefficient = weight*n/len(current)
            class_harm = class_harm + coefficient*change.relu()
            cells.append(dict(class_id=c, group=name, samples=n, local_weight=coefficient,
                reference_la=float(baseline[mask].mean()), current_la=float(current[mask].detach().mean()),
                delta_la=float(change.detach()), active=bool(change.detach() > 0),
                weighted_harm=coefficient*float(change.detach().relu())))
    return dict(client=client_harm, **{'class': class_harm}, cells=cells, groups=group_rows)
