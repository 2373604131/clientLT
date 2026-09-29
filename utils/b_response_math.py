"""Class-conditioned positive responses for a shared, donor-indexed C student."""
import math

import torch

from utils.b_directed_math import directed_config


def response_config(base, args):
    if not getattr(args, 'b_directed_enable', False):
        raise ValueError('Response B requires explicit directed target/source rules')
    config = directed_config(base, args)
    variant = getattr(args, 'b_response_variant', 'positive')
    weight = float(getattr(args, 'b_response_weight', 1.))
    if variant not in ('positive', 'zero') or not math.isfinite(weight) or weight <= 0:
        raise ValueError('Response variant must be positive/zero and weight finite and positive')
    config.update(schema_version='donor_b_tail_response_c_v1', calibration_profile='tail_response',
        response_variant=variant, response_weight=weight, probe_views=2,
        response_views=['deterministic_eval', 'horizontal_flip'],
        donor_rule='outside target group; absent target label; min-view class-mean raw margin gain; top-K',
        donor_pool='union is donor-indexed C basis; class edges additionally determine response targets',
        c_scope='one independent rank-by-rank C per selected donor and module',
        source_weights='normalized positive min-view class margin gains; fixed teacher weights, not C coefficients',
        response_target='baseline pairwise raw-logit margins + positive min-view weighted donor increment',
        response_competitors='frozen baseline softmax over all non-true classes',
        response_loss='one-sided squared margin deficit; sample mean within class; equal class mean; two-view mean',
        calibration_objective='class-macro two-view tail LA + response_weight * response_loss + one C penalty',
        calibration_transform='deterministic evaluation tensor and width flip; autograd only through C',
        residual_normalization='mean over selected donor union, as in original independent-donor C',
        no_response_rule='both variants skip if no positive increment BEFORE zero ablation',
        zero_control='same screen, teacher construction, C basis, views and steps; zero only the teacher increment',
        commit_rule='fixed second C step; no evaluation selection or rollback',
        communication='modeled dense donor/C gradients and class score statistics; teacher tensors stay at receivers')
    return config


def pairwise_margins(logits, labels):
    if logits.ndim != 2 or labels.ndim != 1 or len(logits) != len(labels) or logits.shape[1] < 2:
        raise ValueError('Expected N x K logits and N labels, with K >= 2')
    if not torch.isfinite(logits).all():
        raise ValueError('Nonfinite response logits')
    if labels.numel() and (labels.min() < 0 or labels.max() >= logits.shape[1]):
        raise ValueError('Label outside complete classifier')
    return logits.gather(1, labels[:, None]) - logits


def decision_margins(logits, labels):
    margins = pairwise_margins(logits, labels)
    return margins.scatter(1, labels[:, None], torch.inf).min(1).values


@torch.no_grad()
def build_response_targets(baseline, donor_logits, labels, mixtures, variant='positive'):
    """All tensors are local to one receiver; mix sources BEFORE clipping.

    baseline/donor_logits[j]: V x N x K, with V=2. Return frozen targets
    and both the measured increment and the increment actually supervised.
    """
    if variant not in ('positive', 'zero') or baseline.ndim != 3 or baseline.shape[0] != 2:
        raise ValueError('Response targets require two views and a supported variant')
    base_margin = torch.stack([pairwise_margins(view, labels) for view in baseline])
    increment = torch.zeros_like(base_margin[0])
    for c in labels.unique().tolist():
        sources = mixtures.get(c, {})
        if not sources:
            continue
        if any(not math.isfinite(w) or w <= 0 for w in sources.values()) or not math.isclose(sum(sources.values()), 1., abs_tol=1e-7):
            raise ValueError('Invalid class-conditioned teacher weights')
        mask = labels == c
        response = torch.zeros_like(base_margin[:, mask])
        for donor, weight in sources.items():
            value = donor_logits[donor]
            if value.shape != baseline.shape:
                raise ValueError('Teacher/baseline sample or view mismatch')
            margin = torch.stack([pairwise_margins(view[mask], labels[mask]) for view in value])
            response.add_(margin - base_margin[:, mask], alpha=weight)
        increment[mask] = response.min(0).values.clamp_min(0)
    increment.scatter_(1, labels[:, None], 0.)
    applied = increment.clone() if variant == 'positive' else torch.zeros_like(increment)
    competitor_logits = baseline.clone()
    competitor_logits.scatter_(2, labels[None, :, None].expand(2, -1, 1), -torch.inf)
    weights = competitor_logits.softmax(-1)
    target = base_margin + applied.unsqueeze(0)
    if not all(torch.isfinite(v).all() for v in (increment, weights, target)):
        raise ValueError('Nonfinite response target')
    return dict(baseline_margins=base_margin.detach(), positive_increment=increment.detach(),
                applied_increment=applied.detach(), target_margins=target.detach(),
                competitor_weights=weights.detach())


def response_losses(logits, labels, target_margins, competitor_weights):
    if target_margins.shape != logits.shape or competitor_weights.shape != logits.shape:
        raise ValueError('Response supervision shape differs from student')
    if not torch.isfinite(target_margins).all() or not torch.isfinite(competitor_weights).all():
        raise ValueError('Nonfinite response supervision')
    deficit = (target_margins.detach() - pairwise_margins(logits, labels)).clamp_min(0)
    return (competitor_weights.detach() * deficit.square()).sum(-1)


def donor_residual(donor_tensors, matrices):
    if not donor_tensors or donor_tensors.keys() != matrices.keys():
        raise ValueError('Missing or inconsistent donor C basis')
    return {key: torch.bmm(value, matrices[key]).mean(0) for key, value in donor_tensors.items()}


def image_view(images, view):
    if view not in (0, 1) or images.ndim != 4:
        raise ValueError('Response views require NCHW images and view 0/1')
    return images if view == 0 else images.flip(-1)
