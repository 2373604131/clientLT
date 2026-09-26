"""Class weights and loss for the four shared-C problem-2 controls."""
from collections import Counter, defaultdict
import math

import torch


VARIANTS = ('E00', 'E10', 'E01', 'E11')


def problem2_config(base, variant, beta=1.):
    if variant not in VARIANTS or base.get('mode') != 'shared':
        raise ValueError('Problem 2 requires shared C and E00/E10/E01/E11')
    if not math.isfinite(beta) or beta < 0:
        raise ValueError('Harm beta must be finite and nonnegative')
    config = dict(base)
    config.update(schema_version='donor_b_problem2_v1', calibration_profile='problem2',
        problem2_variant=variant, class_averaging=variant[1] == '1',
        harm_beta=beta if variant[2] == '1' else 0., harm_tolerance=0.,
        non_tail_sampling=base.get('non_tail_sampling', 'sample'),
        tail_weight=base.get('tail_weight', .5),
        donor_rule='all_selected_clients; no positive-gain screen',
        donor_pool='all selected raw updates, including receiver updates',
        calibration_objective='class means with fixed group mass; optional per-unit positive loss increase; one C penalty',
        group_mass_rule='preserve original actual tail/non-tail mass separately for each step',
        harm_anchor='stop-gradient C=0 losses on the SAME step images',
        reference_forward='C=0 reference on both batches for every variant, counted as algorithm forwards')
    return config


def calibration_weights(labels_by_client, tail_ids, tail_weight, class_averaging):
    """Global weights for (receiver, class), summing to one across receivers."""
    if not labels_by_client or not math.isfinite(tail_weight) or not 0 < tail_weight < 1:
        raise ValueError('Empty feedback or invalid tail weight')
    tail_ids = set(tail_ids)
    original, receivers = {}, defaultdict(list)
    counts = {k: Counter(map(int, labels)) for k, labels in labels_by_client.items()}
    for k, local in counts.items():
        nt = sum(n for c, n in local.items() if c in tail_ids)
        nn = sum(local.values()) - nt
        if nt == 0:
            raise ValueError(f'Receiver {k} has no tail samples')
        for c, n in local.items():
            is_tail = c in tail_ids
            group_weight = (tail_weight if nn else 1.) if is_tail else 1-tail_weight
            original[k, c] = group_weight*n/(nt if is_tail else nn)/len(counts)
            receivers[c].append(k)
    mass = {g: sum(w for (_, c), w in original.items() if (c in tail_ids) == g)
            for g in (False, True)}
    classes = {g: [c for c in receivers if (c in tail_ids) == g] for g in (False, True)}
    if class_averaging:
        weights = {(k, c): mass[c in tail_ids]/len(classes[c in tail_ids])/len(receivers[c])
                   for k, c in original}
    else:
        weights = original
    if not math.isclose(sum(weights.values()), 1., abs_tol=1e-12):
        raise ValueError('Calibration weights do not sum to one')
    for g in (False, True):
        if not math.isclose(sum(w for (_, c), w in weights.items() if (c in tail_ids) == g),
                            mass[g], abs_tol=1e-12):
            raise ValueError('Class averaging changed tail/non-tail mass')
    return weights, mass


def class_means(losses, labels):
    if losses.ndim != 1 or labels.ndim != 1 or losses.shape != labels.shape:
        raise ValueError('Expected one loss and one label per image')
    return {int(c): losses[labels == c].mean() for c in labels.unique(sorted=True)}


def feedback_loss(losses, labels, client, weights, baseline, beta):
    """Loss already has global receiver weights; do not divide by K again."""
    means = class_means(losses, labels)
    classification = sum(weights[client, c]*value for c, value in means.items())
    harm = sum(weights[client, c]*torch.relu(value-value.new_tensor(baseline[c]).detach())
               for c, value in means.items())
    return classification+beta*harm, classification, harm


def summarize_changes(rows):
    """Common class-macro metrics, independent of a variant's fitted weights."""
    result = []
    for group in ('tail', 'non_tail'):
        selected = [r for r in rows if r['group'] == group]
        if not selected:
            continue
        classes = defaultdict(list)
        for row in selected:
            classes[row['class_id']].append(row['gain'])
        macro = lambda transform: sum(sum(transform(v) for v in values)/len(values)
                                     for values in classes.values())/len(classes)
        result.append(dict(group=group, units=len(selected), classes=len(classes),
            improved_units=sum(r['gain'] > 0 for r in selected),
            harmed_units=sum(r['gain'] < 0 for r in selected),
            unchanged_units=sum(r['gain'] == 0 for r in selected),
            harmed_classes=sum(sum(v)/len(v) < 0 for v in classes.values()),
            macro_gain=macro(lambda v: v), positive_gain=macro(lambda v: max(v, 0.)),
            harm=macro(lambda v: max(-v, 0.)), max_harm=max(max(-r['gain'], 0.) for r in selected)))
    return result
