"""Image-identity folds and bounded diagnostic sampling; no model/test access."""
from collections import defaultdict

import numpy as np


def image_identity(rows):
    identities, labels = {}, {}
    for row in rows:
        key = int(row['client_id']), int(row['local_position'])
        raw, label = int(row['raw_sample_id']), int(row['class_id'])
        if key in identities or (raw in labels and labels[raw] != label):
            raise ValueError('Duplicate position or conflicting raw-image labels')
        identities[key], labels[raw] = raw, label
    return identities


def cyclic_positions(groups, cap, rng):
    if not groups:
        return []
    order = list(map(int, rng.permutation(sorted(groups))))
    bags = {c: list(map(int, rng.permutation(groups[c]))) for c in groups}
    chosen, cursor = [], 0
    cap = min(cap, sum(map(len, bags.values())))
    while len(chosen) < cap:
        c = order[cursor % len(order)]
        cursor += 1
        if bags[c]:
            chosen.append(bags[c].pop())
    return chosen


def make_folds(groups, identities, recipients, tail, seed, rnd, fit_fold):
    if fit_fold not in (0, 1):
        raise ValueError('fit_fold must be 0 or 1')
    by_class = defaultdict(set)
    raw_labels = {}
    for k in recipients:
        for c, positions in groups[k].items():
            for p in positions:
                raw = identities[k, p]
                if raw in raw_labels and raw_labels[raw] != c:
                    raise ValueError('Inconsistent image label across clients')
                raw_labels[raw] = c
                by_class[c].add(raw)
    membership, coverage = {}, []
    for c, raw in sorted(by_class.items()):
        rng = np.random.default_rng(np.random.SeedSequence([seed, c, 202610081]))
        order = list(map(int, rng.permutation(sorted(raw))))
        split = len(order)//2
        membership.update({p: int(i >= split) for i, p in enumerate(order)})
        coverage.append(dict(class_id=c, distinct_images=len(order), fold0=split, fold1=len(order)-split))
    fit = [dict() for _ in groups]
    diagnostic, probes = {}, {'fit': {}, 'diagnostic': {}}
    for k in recipients:
        held = {}
        for c, positions in groups[k].items():
            a = [p for p in positions if membership[identities[k, p]] == fit_fold]
            b = [p for p in positions if membership[identities[k, p]] != fit_fold]
            if a:
                fit[k][c] = a
            if b:
                held[c] = b
        rng = np.random.default_rng(np.random.SeedSequence([seed, rnd, k, 202610082]))
        diagnostic[k] = sorted(p for c, ps in held.items() if c in tail for p in ps)
        diagnostic[k] += cyclic_positions({c: ps for c, ps in held.items() if c not in tail}, 16, rng)
        for name, pool in [('fit', fit[k]), ('diagnostic', held)]:
            probes[name][k] = {c: sorted(ps)[:8] for c, ps in pool.items() if c in tail}
    rows = [dict(raw_sample_id=raw, class_id=raw_labels[raw], fold=fold)
            for raw, fold in sorted(membership.items())]
    return fit, diagnostic, probes, dict(fit_fold=fit_fold, images=rows, coverage=coverage,
        scope='held out from this B fit only; ordinary local training can have seen these images')
