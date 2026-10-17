"""Paired tail-only allocation sweep for the existing IF=100 J baseline.

This writes partition manifests; it does not change the production partitioner.
Non-tail assignments match the ordinary lambda=.75 Client-LT reference exactly.
Both group Dirichlet draws are consumed even at empty endpoint groups so their
realizations, as well as within-class sample permutations, stay paired.
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from utils.datasplit import (
    _allocate_class_budgets, _append_dirichlet, _validate_partition_map,
    partition_client_longtail,
)

LAMBDAS = (0.0, 0.25, 0.5, 0.75, 1.0)
ALPHA = 0.5
REFERENCE_LAMBDA = 0.75
TAIL_CLIENTS = (27, 28, 29)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, payload):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(payload, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')


def write_csv(path, rows):
    rows = list(rows)
    with Path(path).open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_pool(data_root):
    from utils.functional_coverage_validation import (
        _TrainOnlyCifar100, _exact_lt_raw_ids, _locate_cifar100,
    )
    location = _locate_cifar100(Path(data_root))
    store = _TrainOnlyCifar100(location)
    raw_ids = _exact_lt_raw_ids(store.labels, 0.01, 'exp')
    labels = store.labels[raw_ids]
    if len(raw_ids) != 10847 or int((labels >= 80).sum()) != 153:
        raise ValueError('The J protocol requires the original CIFAR-100-LT IF=100 pool (10847 / 153 tail).')
    return raw_ids, labels, location


def reference_partition(labels, split_seed):
    return partition_client_longtail(
        labels, 30, 100, specialization_lambda=REFERENCE_LAMBDA,
        intra_group_alpha=ALPHA, head_leakage_scale=3.0,
        rng=np.random.RandomState(split_seed),
    )


def controlled_partition(labels, concentration, split_seed):
    if not np.isfinite(concentration) or not 0 <= concentration <= 1:
        raise ValueError('concentration must be in [0, 1]')
    labels = np.asarray(labels, dtype=np.int64)
    counts = np.bincount(labels, minlength=100)
    n_tail = int(counts[80:].sum())
    tail_budget = round(n_tail * (0.1 + 0.9 * concentration))
    # Freeze leakage at the original lambda=.75 reference: 11 images at IF=100.
    non_tail_budget = min(round(3.0 * n_tail * 0.1 * (1 - REFERENCE_LAMBDA)), int(counts[:80].sum()))
    budgets = _allocate_class_budgets(dict(enumerate(counts[:80])), non_tail_budget)
    budgets.update(_allocate_class_budgets({c: counts[c] for c in range(80, 100)}, tail_budget))
    rng = np.random.RandomState(split_seed)
    mapping = {k: np.array([], dtype=np.int64) for k in range(30)}
    for c in range(100):
        indices = np.flatnonzero(labels == c)
        rng.shuffle(indices)
        cut = budgets[c]
        # For non-tail classes, retain the production RNG consumption exactly.
        # For tail classes, draw both groups even if one is empty at lambda=0/1.
        if cut > 0 or c >= 80:
            _append_dirichlet(indices[:cut], list(TAIL_CLIENTS), mapping, ALPHA, rng)
        if cut < len(indices) or c >= 80:
            _append_dirichlet(indices[cut:], list(range(27)), mapping, ALPHA, rng)
    _validate_partition_map(labels, mapping, 30, 100, 'fixed-non-tail-lambda-sweep')
    return mapping


def audit_partition(labels, mapping, reference, concentration):
    counts = np.stack([np.bincount(labels[mapping[k]], minlength=100) for k in range(30)])
    reference_non_tail = [reference[k][labels[reference[k]] < 80] for k in range(30)]
    for k in range(30):
        actual = mapping[k][labels[mapping[k]] < 80]
        if not np.array_equal(actual, reference_non_tail[k]):
            raise AssertionError(f'Non-tail sample identities/order changed at client {k}')
    if concentration == REFERENCE_LAMBDA:
        if not all(np.array_equal(mapping[k], reference[k]) for k in range(30)):
            raise AssertionError('lambda=.75 must reproduce the production reference partition exactly')
    totals = counts.sum(1)
    if np.any(totals == 0):
        raise ValueError('An empty client cannot be used by this training runtime; no automatic resampling is performed.')
    all_indices = np.concatenate(list(mapping.values()))
    if not np.array_equal(np.sort(all_indices), np.arange(len(labels))):
        raise AssertionError('Missing or duplicate training samples')
    if not np.array_equal(counts.sum(0), np.bincount(labels, minlength=100)):
        raise AssertionError('Global class counts changed')
    tail_samples = int(counts[:, 80:].sum())
    target_tail = int(counts[27:, 80:].sum())
    target_non_tail = int(counts[27:, :80].sum())
    if target_tail != round(tail_samples * (0.1 + 0.9 * concentration)):
        raise AssertionError('Tail concentration budget mismatch')
    steps_per_epoch = int(((totals + 31) // 32).sum())
    stats = {
        'lambda': concentration, 'nominal_tail_share': 0.1 + 0.9 * concentration,
        'actual_tail_share': target_tail / tail_samples,
        'total_samples': int(totals.sum()), 'tail_samples': tail_samples,
        'tail_samples_on_designated_clients': target_tail,
        'non_tail_samples_on_designated_clients': target_non_tail,
        'tail_client_totals': totals[27:].tolist(), 'client_totals': totals.tolist(),
        'tail_client_fedavg_weight': float(totals[27:].sum() / totals.sum()),
        'mean_tail_client_coverage': float((counts[:, 80:] > 0).sum(0).mean()),
        'non_tail_assignments_identical_to_reference': True,
        'lambda_075_matches_production_reference': concentration == REFERENCE_LAMBDA,
        'expected_optimizer_steps': steps_per_epoch * 309,
    }
    return stats, counts


def write_partition(root, raw_ids, labels, mapping, counts):
    write_csv(root / 'partition_source.csv', (
        {'raw_sample_id': int(raw_ids[i]), 'class_id': int(labels[i]),
         'client_id': k, 'local_position': pos}
        for k in range(30) for pos, i in enumerate(mapping[k])
    ))
    write_csv(root / 'expected_client_class_counts.csv', (
        {'client_id': k, **{f'class_{c}': int(counts[k, c]) for c in range(100)}}
        for k in range(30)
    ))
