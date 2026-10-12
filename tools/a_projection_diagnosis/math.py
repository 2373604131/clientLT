"""Small CPU calculations shared by the real experiment and its tests."""
import math

import numpy as np
import torch


def row_basis(a, rtol=1e-6):
    a = torch.as_tensor(a).detach().cpu().double()
    if a.ndim != 2 or not torch.isfinite(a).all():
        raise ValueError('Expected a finite matrix')
    _, singular, vh = torch.linalg.svd(a, full_matrices=False)
    rank = int((singular > singular[0] * rtol).sum()) if singular.numel() and singular[0] > 0 else 0
    # Padding zeros keeps the feature dimension fixed without inventing directions.
    q = torch.zeros_like(a)
    q[:rank] = vh[:rank]
    return q, rank, singular


def project_summary(summary, a):
    q, rank, _ = row_basis(a)
    x = torch.as_tensor(summary).double()
    if x.ndim != 3 or x.shape[1] != 2 or x.shape[2] != q.shape[1]:
        raise ValueError('Expected [images, CLS+patch_mean, input_dim]')
    return (x @ q.T).flatten(1).numpy(), rank


def balanced_weights(labels, classes):
    labels = np.asarray(labels, dtype=np.int64)
    if labels.ndim != 1 or (labels < 0).any() or (labels >= classes).any():
        raise ValueError('Invalid fit labels')
    counts = np.bincount(labels, minlength=classes)
    if (counts == 0).any():
        raise ValueError('Every class must occur in the FIT sample')
    return 1. / (classes * counts[labels])


def ridge_probe(train, labels, test, classes=100, regularization=.01):
    """Class-balanced squared loss, unpenalized bias, one scalar train RMS.

    Predictions are regression scores, not probabilities. All preprocessing uses
    fit images only. No per-channel normalization and no hyperparameter search.
    """
    x, z = np.asarray(train, dtype=np.float64), np.asarray(test, dtype=np.float64)
    if x.ndim != 2 or z.ndim != 2 or x.shape[1] != z.shape[1] or regularization <= 0:
        raise ValueError('Invalid probe inputs')
    if not np.isfinite(x).all() or not np.isfinite(z).all():
        raise ValueError('Nonfinite probe features')
    w = balanced_weights(labels, classes)
    y = np.eye(classes)[np.asarray(labels, dtype=int)]
    center, bias = w @ x, w @ y
    x = x - center
    rms = math.sqrt(float(np.sum(w[:, None] * x ** 2) / x.shape[1]))
    if rms <= 1e-15:
        return np.broadcast_to(bias, (len(z), classes)).copy(), dict(rms=rms, constant=True)
    x, z = x / rms, (z - center) / rms
    matrix = x.T @ (w[:, None] * x) + regularization * np.eye(x.shape[1])
    beta = np.linalg.solve(matrix, x.T @ (w[:, None] * (y - bias)))
    return z @ beta + bias, dict(rms=rms, constant=False)


def aggregate(anchor, proposals, weights):
    """Preserve uploaded client order and the historical FP32 delta arithmetic."""
    result = {k: v.clone() for k, v in anchor.items()}
    keys = [k for k in anchor if k.endswith('_lora_A')]
    if set(proposals) != set(weights) or abs(sum(weights.values()) - 1.) > 1e-8:
        raise ValueError('Proposals and normalized weights must cover the same clients')
    if any(not math.isfinite(v) or v < 0 for v in weights.values()):
        raise ValueError('Invalid weights')
    for k in keys:
        delta = torch.zeros_like(anchor[k])
        for client, upload in proposals.items():
            delta.add_(upload[k].to(delta), alpha=float(weights[client]))
        result[k] = anchor[k] + delta
    return result


def with_delta(anchor, delta):
    return {k: v + delta[k].to(v) if k in delta else v.clone() for k, v in anchor.items()}


def effective_norm(anchor, delta, scales):
    total = 0.
    for k, d in delta.items():
        b, d = anchor[k[:-1] + 'B'].double(), d.double()
        total += scales[k] ** 2 * float(((b.T @ b) * (d @ d.T).T).sum().clamp_min(0))
    return math.sqrt(total)


def split_delta(anchor, delta):
    parallel, outside = {}, {}
    for k, d in delta.items():
        q, _, _ = row_basis(anchor[k])
        parallel[k] = ((d.double() @ q.T) @ q).to(d)
        outside[k] = d - parallel[k]
    return parallel, outside


def shrink_panel(anchor, deltas, scales):
    """One coefficient per six-module candidate; zeros do not set the target."""
    norms = {name: effective_norm(anchor, d, scales) for name, d in deltas.items()}
    positive = [n for n in norms.values() if n > 0]
    target = min(positive) if positive else 0.
    maximum = max(positive) if positive else 0.
    matched, records = {}, []
    for name, delta in deltas.items():
        ratio = target / norms[name] if norms[name] > 0 else 1.
        matched[name] = {k: d * ratio for k, d in delta.items()}
        records.append(dict(candidate=name, raw_norm=norms[name], target_norm=target,
                            coefficient=ratio, zero_reference=norms[name] == 0,
                            low_power=target < 1e-10 or (maximum > 0 and target / maximum < 1e-3)))
    return matched, records


def metric_rows(values, groups, anchor=None):
    y = np.asarray(values['labels'])
    correct = values['prediction'] == y
    if anchor is not None and (not np.array_equal(y, anchor['labels']) or
                              not np.array_equal(values['sample_id'], anchor['sample_id'])):
        raise ValueError('Paired evaluation sample identities differ')
    rows = []
    for group, classes in groups.items():
        mask = np.isin(y, classes)
        if not mask.any():
            continue
        row = dict(group=group, samples=int(mask.sum()),
                   accuracy=float(np.mean([correct[y == c].mean() for c in classes if (y == c).any()]) * 100))
        for name in ('margin', 'ce', 'squared_error'):
            if name in values:
                row[name] = float(np.mean([values[name][y == c].mean() for c in classes if (y == c).any()]))
        if anchor is not None:
            old = anchor['prediction'] == y
            row.update(lost=int((mask & old & ~correct).sum()), gained=int((mask & ~old & correct).sum()))
            for name in ('margin', 'ce'):
                if name in values:
                    row['delta_' + name] = float(np.mean([
                        (values[name][y == c] - anchor[name][y == c]).mean()
                        for c in classes if (y == c).any()]))
            row['delta_accuracy'] = float(np.mean([
                correct[y == c].mean() - old[y == c].mean() for c in classes if (y == c).any()]) * 100)
        rows.append(row)
    return rows


def response_decomposition(local, aggregated, anchor, weights, counts):
    """Descriptive responses, NOT an additive causal decomposition of accuracy."""
    rows = []
    for c in range(counts.shape[1]):
        mask = anchor['labels'] == c
        effects = {j: float((p['margin'][mask] - anchor['margin'][mask]).mean()) for j, p in local.items()}
        for rule, w in weights.items():
            row = dict(class_id=c, rule=rule,
                       aggregate_delta_margin=float((aggregated[rule]['margin'][mask] - anchor['margin'][mask]).mean()),
                       weighted_local_delta_margin=sum(w[j] * effects[j] for j in effects))
            for held in (True, False):
                prefix = 'holder' if held else 'nonholder'
                ids = [j for j in effects if bool(counts[j, c] > 0) == held]
                row[prefix + '_weight'] = sum(w[j] for j in ids)
                for sign in ('positive', 'negative'):
                    row[prefix + '_' + sign] = sum(w[j] * effects[j] for j in ids
                                                      if (effects[j] > 0 if sign == 'positive' else effects[j] < 0))
            row['nonlinear_gap'] = row['aggregate_delta_margin'] - row['weighted_local_delta_margin']
            rows.append(row)
    return rows
