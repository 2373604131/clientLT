"""Tail-class mass rescaling; preserve the original within-class weights."""
from collections import defaultdict
import math


ARM = 'F-tail-equal'
OBJECTIVE = 'tail_class_rescale_original_coefficients_v1'


def rescale_tail_coefficients(labels_by_client, tail_ids, tail_weight=.35):
    """Return per-image old/new GLOBAL coefficients, in the supplied order.

    Empty clients are excluded, as in the original F optimizer. Single-group
    clients contribute their group mean with weight one. All declared tail
    classes must appear: silently renormalizing to observed classes is forbidden.
    """
    tail = set(tail_ids)
    if not tail or not math.isfinite(tail_weight) or not 0 < tail_weight < 1:
        raise ValueError('Invalid tail classes/weight')
    active = {k: list(v) for k, v in labels_by_client.items() if len(v)}
    if not active:
        raise ValueError('No active feedback clients')
    old, masses = {}, defaultdict(float)
    for k, labels in active.items():
        nt = sum(c in tail for c in labels)
        nn = len(labels)-nt
        wt = tail_weight if nt and nn else 1.
        wn = 1-tail_weight if nt and nn else 1.
        old[k] = [(wt/nt if c in tail else wn/nn)/len(active) for c in labels]
        for c, value in zip(labels, old[k]):
            masses[c] += value
    missing = sorted(c for c in tail if masses[c] == 0)
    if missing:
        raise ValueError('Missing declared tail classes in feedback batch: '+str(missing))
    beta = math.fsum(masses[c] for c in tail)
    new = {k: [w*beta/(len(tail)*masses[c]) if c in tail else w
               for c, w in zip(active[k], weights)] for k, weights in old.items()}
    new_masses = defaultdict(float)
    for k, labels in active.items():
        for c, w in zip(labels, new[k]):
            new_masses[c] += w
    if (not math.isclose(math.fsum(masses.values()), 1., abs_tol=1e-12, rel_tol=0)
            or not math.isclose(math.fsum(new_masses.values()), 1., abs_tol=1e-12, rel_tol=0)
            or any(not math.isclose(new_masses[c], beta/len(tail), abs_tol=1e-12, rel_tol=0) for c in tail)):
        raise ValueError('Feedback mass conservation failed')
    rows = [dict(class_id=c, is_tail=c in tail, old_mass=masses[c], new_mass=new_masses[c],
                 beta_tail=beta, multiplier=beta/(len(tail)*masses[c]) if c in tail else 1.)
            for c in sorted(masses)]
    return old, new, rows


def manifest_coefficients(manifests, groups, tail_ids, tail_weight=.35):
    """Bind coefficients to client/local-position/class identities, for two batches."""
    lookup = {k: {int(p): int(c) for c, ids in groups[k].items() for p in ids} for k in manifests}
    prepared, samples, classes = [], [], []
    for index in range(2):
        labels = {}
        positions = {}
        for k, item in manifests.items():
            if len(item['batches']) != 2:
                raise ValueError('Expected exactly two feedback batches')
            batch = item['batches'][index]
            positions[k] = batch['tail_positions']+batch['non_tail_positions']
            if len(positions[k]) != len(set(positions[k])):
                raise ValueError('Duplicate sample identity in feedback batch')
            labels[k] = [lookup[k][int(p)] for p in positions[k]]
            nt = len(batch['tail_positions'])
            if any((c in tail_ids) != (i < nt) for i, c in enumerate(labels[k])):
                raise ValueError('Feedback tail/non-tail labels disagree with manifest')
        old, new, class_rows = rescale_tail_coefficients(labels, tail_ids, tail_weight)
        prepared.append({k: dict(labels=labels[k], weights=new[k]) for k in new})
        classes.extend(dict(batch=index+1, **r) for r in class_rows)
        samples.extend(dict(batch=index+1, client_id=k, local_position=int(p), class_id=c,
                            old_coefficient=a, new_coefficient=b)
                       for k in new for p, c, a, b in zip(positions[k], labels[k], old[k], new[k]))
    return prepared, samples, classes
