"""The preregistered longitudinal, proposal, weighting and row-space panels."""
import json
import time

import numpy as np
import torch

from tools.a_learning_schedule.runtime import save_arrays
from tools.a_projection_diagnosis.math import (
    aggregate, with_delta, split_delta, shrink_panel, row_basis, ridge_probe,
    metric_rows, response_decomposition, effective_norm,
)
from tools.a_projection_diagnosis.protocol import SNAPSHOTS, aggregation_weights, write_csv, write_json
from utils.cliplora_a_refresh import state_hash


def probe_bundle(backend, fit, evaluation, state, context, root, fit_labels=None):
    y = backend.labels['fit'] if fit_labels is None else fit_labels
    split = context['split']
    labels, ids = backend.labels[split], backend.ids[split]
    rows, values, scores = [], {}, {}
    groups = dict(backend.groups, **{'class_%03d' % c: [c] for c in range(100)})
    start = time.perf_counter()
    for module in backend.a_keys + ['combined48']:
        keys = backend.a_keys if module == 'combined48' else [module]
        x = np.concatenate([fit[k] for k in keys], axis=1)
        z = np.concatenate([evaluation[k] for k in keys], axis=1)
        score, info = ridge_probe(x, y, z)
        ranks = [row_basis(state[k])[1] for k in keys]
        value = dict(labels=labels, sample_id=ids, prediction=score.argmax(1),
                     squared_error=((score - np.eye(100)[labels]) ** 2).sum(1))
        values[module], scores[module] = value, score
        for m in metric_rows(value, groups):
            rows.append(dict(context, module=module, rank=json.dumps(ranks), ridge_lambda=.01,
                             fit_sha256=backend.fit_hash, eval_sha256=backend.eval_hash,
                             A_sha256=state_hash(state, backend.a_keys), **info, **m))
    export = dict(labels=labels, sample_id=ids)
    for i, module in enumerate(values):
        for field in ('prediction', 'squared_error'):
            export['m%d_%s' % (i, field)] = values[module][field]
    save_arrays(root / 'probe_predictions' / (context['candidate'] + '_' + split + '.npz'), **export)
    backend.costs.append(dict(kind='probe_fit', fits=len(values), seconds=time.perf_counter() - start, forward_images=0))
    return rows, values, scores


def full_rows(backend, values, context, anchor=None):
    groups = dict(backend.groups, **{'class_%03d' % c: [c] for c in range(100)})
    return [dict(context, **m) for m in metric_rows(values, groups, anchor)]


def check_scores(left, right, keys, tolerance=2e-4):
    error = max(float(np.max(np.abs(left[k] - right[k]))) for k in keys)
    if error > tolerance:
        raise ValueError('Same-space probe prediction invariance failed: %g' % error)
    return error


def geometry(backend, state, reference, context):
    rows = []
    for k in backend.a_keys:
        a, b = state[k].double(), state[k[:-1] + 'B'].double()
        q, rank, singular = row_basis(a)
        oldq, oldrank, _ = row_basis(reference[k])
        distance = max(0., rank + oldrank - 2 * float((q @ oldq.T).square().sum())) ** .5
        rows.append(dict(context, module=k, rank=rank, singular_values=json.dumps(singular.tolist()),
            row_projector_distance=distance, A_norm=float(a.norm()),
            effective_BA_norm=backend.scales[k] * float(((b.T @ b) * (a @ a.T).T).sum().clamp_min(0).sqrt())))
    return rows


def longitudinal(backend, store, root):
    states = {r: store.committed(r) for r in SNAPSHOTS}
    names = {'r%03d' % r: s for r, s in states.items()}
    if any(not torch.equal(states[90][k], states[100][k]) for k in backend.a_keys):
        raise ValueError('A changed between rounds90 and100 in a B-only interval')
    reference_fit = backend.measure(states[0], names, 'fit')
    reference_test = backend.measure(states[0], names, 'test')
    curves, model_rows, checks, geometries = [], [], [], []
    fixed90 = None
    for rnd, state in states.items():
        name = 'r%03d' % rnd
        geometries.extend(geometry(backend, state, states[0], dict(round=rnd, trajectory=backend.source['rule'])))
        current_fit = backend.measure(state, dict(current=state, initial=states[0]), 'fit')
        current_test = backend.measure(state, dict(current=state, initial=states[0]), 'test')
        for split, reference, current in [('fit', reference_fit, current_fit), ('test', reference_test, current_test)]:
            same = reference['record']['block9_input_hash'] == current['record']['block9_input_hash']
            checks.append(dict(check='block9_frozen_upstream_input', round=rnd, split=split, passed=same))
            if not same:
                raise ValueError('Block9 actual q/v inputs changed despite frozen upstream and fixed preprocessing')
        pairs = [('fixed_input', reference_fit['features'][name], reference_test['features'][name], state, 0),
                 ('fixed_A', current_fit['features']['initial'], current_test['features']['initial'], states[0], rnd),
                 ('live_input', current_fit['features']['current'], current_test['features']['current'], state, rnd)]
        for combo, fit, test, a_state, input_round in pairs:
            context = dict(trajectory=backend.source['rule'], seed=42, topology='client-longtail',
                           round=rnd, phase='committed', candidate=name + '_' + combo, combination=combo,
                           split='test', input_round=input_round,
                           input_state_sha256=state_hash(states[input_round], backend.keys))
            rows, _, scores = probe_bundle(backend, fit, test, a_state, context, root)
            curves.extend(rows)
            if combo == 'fixed_input' and rnd == 90:
                fixed90 = scores
            if combo == 'fixed_input' and rnd == 100:
                error = check_scores(fixed90, scores, list(scores), tolerance=1e-10)
                checks.append(dict(check='round90_100_fixed_input_probe', passed=True, max_score_error=error))
        model_rows.extend(full_rows(backend, current_test['predictions'], dict(trajectory=backend.source['rule'],
            round=rnd, phase='committed', candidate=name, split='test'), reference_test['predictions']))
        save_arrays(root / 'model_predictions' / (name + '.npz'), **current_test['predictions'])
        write_csv(root / 'representation_curves.csv', curves)
        write_csv(root / 'full_model_curves.csv', model_rows)
        print('E1 %s round%d finished' % (backend.source['rule'], rnd), flush=True)
    # Fixed label shuffle: diagnostic only; do not require exactly chance accuracy.
    labels = np.random.default_rng(20261012).permutation(backend.labels['fit'])
    rows, _, _ = probe_bundle(backend, reference_fit['features']['r000'], reference_test['features']['r000'],
        states[0], dict(trajectory=backend.source['rule'], round=0, phase='negative_control',
                       candidate='shuffled_fit_labels', split='test'), root, fit_labels=labels)
    write_csv(root / 'shuffled_label_control.csv', rows)
    write_json(root / 'numerical_checks.json', checks)
    write_csv(root / 'projection_geometry.csv', geometries)
    write_csv(backend.root / 'costs.csv', backend.costs)
    write_json(root / 'completion.json', dict(experiment='E1', trajectory=backend.source['rule'], snapshots=list(SNAPSHOTS)))


def event_panels(backend, store, rnd, root):
    anchor, provenance = store.anchor(rnd)
    proposals, original = store.historical_proposals(rnd, anchor)
    if proposals is None:
        proposals, original = backend.regenerate(anchor, rnd)
    provenance.update(original)
    provenance['anchor_sha256'] = state_hash(anchor, backend.keys)
    write_json(root / 'provenance.json', provenance)
    weights = aggregation_weights(backend.source)
    write_json(root / 'weights.json', {k: {str(j): w for j, w in v.items()} for k, v in weights.items()})
    candidates = dict(no_A=anchor)
    for client, delta in proposals.items():
        candidates['local_%02d' % client] = with_delta(anchor, delta)
    aggregates = {name: aggregate(anchor, proposals, w) for name, w in weights.items()}
    raw = {name: {k: s[k] - anchor[k] for k in backend.a_keys} for name, s in aggregates.items()}
    candidates.update({'weight_raw_' + name: s for name, s in aggregates.items()})
    matched, norm_rows = shrink_panel(anchor, raw, backend.scales)
    norm_rows = [dict(panel='weight', **r) for r in norm_rows]
    candidates.update({'weight_matched_' + n: with_delta(anchor, d) for n, d in matched.items()})
    parallel, outside = split_delta(anchor, raw['fedavg'])
    intervention = dict(full=raw['fedavg'], parallel=parallel, outside=outside)
    matched, norms = shrink_panel(anchor, intervention, backend.scales)
    norm_rows.extend(dict(panel='rowspace', **r) for r in norms)
    candidates.update({'space_raw_' + n: with_delta(anchor, d) for n, d in intervention.items()})
    candidates.update({'space_matched_' + n: with_delta(anchor, d) for n, d in matched.items()})
    write_csv(root / 'norm_matching.csv', norm_rows)
    checks = []
    for k in backend.a_keys:
        q, rank, _ = row_basis(anchor[k])
        error = float((raw['fedavg'][k] - parallel[k] - outside[k]).abs().max())
        orth = float((anchor[k].double() @ outside[k].double().T).abs().max())
        checks.append(dict(check='delta_decomposition', module=k, max_reconstruction_error=error,
                           max_A_outside_inner_product=orth, rank=rank, passed=error < 1e-6))
        if error >= 1e-6:
            raise ValueError('Rowspace decomposition failed')
    fit = backend.measure(anchor, candidates, 'fit')
    panel = backend.measure(anchor, candidates, 'panel')
    base_predictions = panel['predictions']
    repeated = backend.measure(anchor, {}, 'panel', force=True)['predictions']
    noise = float(np.max(np.abs(repeated['margin'] - base_predictions['margin'])))
    checks.append(dict(check='repeat_unchanged_model', max_cosine_margin_error=noise,
                       prediction_flips=int(np.sum(repeated['prediction'] != base_predictions['prediction'])),
                       passed=noise <= 1e-6))
    if noise > 1e-6:
        raise ValueError('Unchanged-model numerical noise is too large for local response diagnosis')
    nonlocal_candidates = {n: s for n, s in candidates.items() if not n.startswith('local_')}
    remaining = backend.measure(anchor, nonlocal_candidates, 'remaining')
    probes, full, local_predictions, aggregate_predictions, geometries = [], [], {}, {}, []
    anchor_scores = {}
    for name, state in candidates.items():
        geometries.extend(geometry(backend, state, anchor, dict(round=rnd, candidate=name,
            effective_delta_norm=effective_norm(anchor, {k: state[k] - anchor[k] for k in backend.a_keys}, backend.scales))))
        for split, reference in [('panel', panel)] + ([] if name.startswith('local_') else [('remaining', remaining)]):
            context = dict(trajectory='fedavg', seed=42, topology='client-longtail', round=rnd,
                           phase='post_B_pre_A', candidate=name, split=split,
                           input_state_sha256=provenance['anchor_sha256'])
            rows, _, scores = probe_bundle(backend, fit['features'][name], reference['features'][name],
                                           state, context, root)
            probes.extend(rows)
            if name == 'no_A':
                anchor_scores[split] = scores
            if name in ('space_raw_parallel', 'space_matched_parallel'):
                same_rank = [k for k in backend.a_keys if row_basis(state[k])[1] == row_basis(anchor[k])[1]]
                test_keys = same_rank + (['combined48'] if len(same_rank) == len(backend.a_keys) else [])
                error = check_scores(anchor_scores[split], scores, test_keys) if test_keys else None
                checks.append(dict(check='parallel_fixed_input_probe', candidate=name, split=split,
                                   same_rank_modules=same_rank, max_score_error=error,
                                   rank_drop_modules=sorted(set(backend.a_keys) - set(same_rank)), passed=True))
            values = reference['predictions'] if name == 'no_A' else backend.measure(state, {}, split)['predictions']
            full.extend(full_rows(backend, values, context, reference['predictions']))
            save_arrays(root / 'model_predictions' / (name + '_' + split + '.npz'), **values)
            if split == 'panel' and name.startswith('local_'):
                local_predictions[int(name[6:])] = values
            if split == 'panel' and name.startswith('weight_raw_'):
                aggregate_predictions[name[len('weight_raw_'):]] = values
        print('E2/E3/space round%d %s finished' % (rnd, name), flush=True)
        write_csv(root / 'projection_probes.csv', probes)
        write_csv(root / 'full_model_candidates.csv', full)
    decomposition = response_decomposition(local_predictions, aggregate_predictions, base_predictions,
                                            weights, backend.source['counts'])
    write_csv(root / 'response_decomposition.csv', [dict(round=rnd, **r) for r in decomposition])
    local_rows = []
    for r in full:
        if not r['candidate'].startswith('local_') or not r['group'].startswith('class_'):
            continue
        client, c = int(r['candidate'][6:]), int(r['group'][6:])
        local_rows.append(dict(r, client_id=client, class_id=c, class_count=int(backend.source['counts'][client, c]),
                              holds_class=bool(backend.source['counts'][client, c]),
                              **{'weight_' + n: w[client] for n, w in weights.items()}))
    write_csv(root / 'local_class_response.csv', local_rows)
    write_csv(root / 'a_weight_counterfactuals.csv', [r for r in full if r['candidate'] == 'no_A' or r['candidate'].startswith('weight_')])
    write_csv(root / 'rowspace_interventions.csv', [r for r in full if r['candidate'] == 'no_A' or r['candidate'].startswith('space_')])
    write_json(root / 'numerical_checks.json', checks)
    write_csv(root / 'projection_geometry.csv', geometries)
    write_csv(backend.root / 'costs.csv', backend.costs)
    write_json(root / 'completion.json', dict(experiment='E2_E3_rowspace', round=rnd,
        proposals=provenance['proposals'], candidates=list(candidates),
        remaining_scope='all predefined nonlocal candidates; no winner selection'))
