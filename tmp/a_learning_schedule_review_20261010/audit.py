"""Audit the received archive and derive descriptive, within-seed comparisons."""
import csv
import json
import statistics as st
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
ROOT = REPO / 'output/a_learning_schedule_v1_analysis/a_learning_schedule_v1'
OUT = Path(__file__).resolve().parent
ARMS = ['BB', 'AB', 'AA_short', 'AA_long']
CONTRASTS = [('AB', 'BB'), ('AA_long', 'AB'), ('AA_long', 'AA_short')]
GROUPS = ['overall', 'head20', 'middle60', 'tail20']


def read_csv(path):
    with path.open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def write_csv(name, rows):
    with (OUT / name).open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_npz(path):
    with np.load(path, allow_pickle=False) as f:
        return dict(f)


def recalc(arr, anchor, groups):
    labels = arr['labels']
    assert np.array_equal(labels, anchor['labels'])
    for x in arr.values():
        assert len(x) == len(labels) and np.isfinite(x).all()
    correct = arr['prediction'] == labels
    old = anchor['prediction'] == labels
    result = {}
    for group, ids in groups.items():
        masks = [labels == c for c in ids if (labels == c).any()]
        mask = np.isin(labels, ids)
        result[group] = np.mean([100 * correct[m].mean() for m in masks])
        for metric in ('ce', 'la', 'margin'):
            result[group + '_' + metric] = np.mean([arr[metric][m].mean() for m in masks])
        result[group + '_new_correct'] = np.sum(mask & ~old & correct)
        result[group + '_forgotten'] = np.sum(mask & old & ~correct)
        result[group + '_retained_correct'] = np.sum(mask & old & correct)
    return result


def feedback_gain(rows, ids, metric, stage):
    class_means = []
    for c in ids:
        selected = [r for r in rows if int(r['class_id']) == c]
        if not selected:
            continue
        sign = 1 if metric == 'margin' else -1
        gains = [sign * (float(r[metric + '_' + stage]) - float(r[metric + '_before'])) for r in selected]
        class_means.append(np.average(gains, weights=[int(r['samples']) for r in selected]))
    return float(np.mean(class_means))


per_run = read_csv(ROOT / 'analysis/per_run.csv')
paired = read_csv(ROOT / 'analysis/paired.csv')
assert len(per_run) == 24 and len(paired) == 18
request_values = []
checks, norm_rows, feedback_rows, curve_pairs = [], [], [], []
max_raw_error = max_summary_error = 0.
npz_count = 0
for origin in ['e2', 'e3']:
    for rnd in [20, 50, 80]:
        job = ROOT / f'runs/seed42/{origin}/round{rnd:03d}'
        info, complete, request = [read_json(job / name) for name in ['anchor_info.json', 'complete.json', 'request.json']]
        request_values.append(request)
        curves, events, budget, local, probes = [read_csv(job / name) for name in ['curves.csv', 'events.csv', 'budget.csv', 'local_feedback.csv', 'norm_probes.csv']]
        assert info['state_key'] == 'anchor_lora_state' and complete['frozen_state_unchanged']
        assert len(curves) == 44 and {(r['arm'], int(r['h'])) for r in curves} == {(a, h) for a in ARMS for h in range(11)}
        unique_events = {r['id']: r for r in events}
        assert len(events) == 48 and len(unique_events) == 40
        assert len(budget) == 1200
        assert sum(int(r['optimizer_steps']) for r in budget) == complete['actual_optimizer_steps'] == 38016
        for arm in ARMS:
            selected_events = [r for r in events if r['arm'] == arm]
            ids = {r['id'] for r in selected_events}
            assert sum(int(r['optimizer_steps']) for r in budget if r['node_id'] in ids) == complete['logical_optimizer_steps_per_arm'] == 11264
            extras = [(int(r['h']), r['factor']) for r in selected_events if r['kind'] == 'extra']
            assert extras == [(0, 'B' if arm == 'BB' else 'A'), (1 if arm == 'AA_short' else 5, 'B' if arm in ('BB', 'AB') else 'A')]
        anchor = {d: load_npz(job / f'evaluations/anchor/{d}.npz') for d in ('test', 'feedback')}
        assert len(anchor['test']['labels']) == 10000
        assert np.array_equal(np.bincount(anchor['test']['labels']), np.full(100, 100))
        computed = {}
        eval_dirs = sorted((job / 'evaluations').iterdir())
        assert len(eval_dirs) == 45
        for ev in eval_dirs:
            meta = read_json(ev / 'metrics.json')
            values = {}
            for domain in ('test', 'feedback'):
                arrays = load_npz(ev / (domain + '.npz'))
                metrics = recalc(arrays, anchor[domain], info['groups'])
                npz_count += 1
                for key, value in meta['metrics'][domain].items():
                    error = abs(float(value) - metrics[key])
                    max_raw_error = max(max_raw_error, error)
                    assert error < 1e-5, (origin, rnd, ev.name, domain, key, error)
                values.update({domain + '_' + k: float(v) for k, v in metrics.items()})
            computed[ev.name] = values
            if ev.name in unique_events:
                event = unique_events[ev.name]
                assert event['after_sha256'] == meta['state_sha256']
                parent = read_json(job / 'evaluations' / event['parent'] / 'metrics.json')
                assert event['before_sha256'] == parent['state_sha256']
        for row in curves:
            for key, value in computed[row['node_id']].items():
                assert abs(float(row[key]) - value) < 1e-5
        for row in per_run:
            if row['origin'] != origin or int(row['anchor_round']) != rnd:
                continue
            for key in computed['anchor']:
                value = st.mean(float(r[key]) for r in curves if r['arm'] == row['arm'] and int(r['h']) in (8, 9, 10))
                max_summary_error = max(max_summary_error, abs(float(row[key]) - value))
                assert abs(float(row[key]) - value) < 1e-8
        for a, b in CONTRASTS:
            for h in range(11):
                ra = next(r for r in curves if r['arm'] == a and int(r['h']) == h)
                rb = next(r for r in curves if r['arm'] == b and int(r['h']) == h)
                curve_pairs.append(dict(origin=origin, anchor_round=rnd, comparison=a+'-'+b, h=h,
                                        **{g: float(ra['test_'+g])-float(rb['test_'+g]) for g in GROUPS}))
        for h in (0, 5):
            pa, pb = [next(r for r in probes if int(r['h']) == h and r['factor'] == f) for f in ('A', 'B')]
            assert pa['comparable'] == pb['comparable'] == 'True'
            assert pa['parent_sha256'] == pb['parent_sha256']
            ea, eb = [next(r for r in events if r['arm'] == arm and int(r['h']) == h and r['kind'] == 'extra') for arm in (('AB', 'BB') if h == 0 else ('AA_long', 'AB'))]
            assert ea['parent'] == eb['parent'] and ea['before_sha256'] == eb['before_sha256'] == pa['parent_sha256']
            norm_rows.append(dict(origin=origin, anchor_round=rnd, h=h,
                                  raw_norm_A_B_ratio=float(pa['raw_norm']) / float(pb['raw_norm']),
                                  max_matching_error=max(float(pa['relative_error']), float(pb['relative_error'])),
                                  **{'matched_delta_'+g: float(pa['test_'+g])-float(pb['test_'+g]) for g in GROUPS},
                                  **{'raw_delta_'+g: computed[ea['id']]['test_'+g]-computed[eb['id']]['test_'+g] for g in GROUPS},
                                  matched_delta_tail_ce=float(pa['test_tail20_ce'])-float(pb['test_tail20_ce'])))
            for event in (ea, eb):
                rows = [r for r in local if r['node_id'] == event['id']]
                assert len({r['client_id'] for r in rows}) == 30
                for group in GROUPS:
                    feedback_rows.append(dict(origin=origin, anchor_round=rnd, h=h, factor=event['factor'], group=group,
                                              **{metric+'_'+stage+'_gain': feedback_gain(rows, info['groups'][group], metric, stage)
                                                 for metric in ('ce', 'la', 'margin') for stage in ('local', 'aggregated')}))
        checks.append(dict(origin=origin, anchor_round=rnd, curve_rows=len(curves), unique_training_nodes=len(unique_events),
                           evaluations=len(eval_dirs), test_samples=10000, feedback_samples=len(anchor['feedback']['labels']),
                           logical_optimizer_steps_per_arm=11264, anchor_overall=computed['anchor']['test_overall'],
                           anchor_tail20=computed['anchor']['test_tail20']))
for key in ('protocol', 'partition_sha256', 'code_fingerprint', 'num_workers', 'eval_batch_size'):
    assert len({json.dumps(r[key], sort_keys=True) for r in request_values}) == 1, key
for pair in paired:
    a, b = pair['comparison'].split('-')
    rows = [next(r for r in per_run if r['origin'] == pair['origin'] and r['anchor_round'] == pair['anchor_round'] and r['arm'] == arm) for arm in (a, b)]
    for key in pair:
        if key.startswith(('test_', 'feedback_')):
            assert abs(float(pair[key]) - (float(rows[0][key])-float(rows[1][key]))) < 1e-8
aggregates = []
for origin in ('all', 'e2', 'e3'):
    for a, b in CONTRASTS:
        rows = [r for r in paired if r['comparison'] == a+'-'+b and (origin == 'all' or r['origin'] == origin)]
        aggregates.append(dict(origin=origin, comparison=a+'-'+b, anchors=len(rows),
                               **{g+'_mean_pp': st.mean(float(r['test_'+g]) for r in rows) for g in GROUPS},
                               **{g+'_positive': sum(float(r['test_'+g]) > 1e-8 for r in rows) for g in GROUPS},
                               **{g+'_'+s+'_delta': st.mean(float(r['test_'+g+'_'+s]) for r in rows) for g in ('overall', 'tail20') for s in ('new_correct', 'forgotten')}))
write_csv('descriptive_contrasts.csv', aggregates)
write_csv('norm_matched_contrasts.csv', norm_rows)
write_csv('paired_local_feedback.csv', feedback_rows)
write_csv('paired_trajectories.csv', curve_pairs)
audit = dict(anchors=checks, npz_files_verified=npz_count, max_raw_metric_error=float(max_raw_error),
             max_summary_error=float(max_summary_error), common_code_fingerprint=request_values[0]['code_fingerprint'],
             caveat='Single seed. Anchor averages are descriptive, not independent-seed inference. Training checkpoint tensors absent from archive.')
(OUT / 'audit.json').write_text(json.dumps(audit, indent=2), encoding='utf-8')
print(json.dumps(audit, indent=2))
print('DESCRIPTIVE CONTRASTS')
for r in aggregates:
    print(json.dumps(r))
print('NORM PROBES')
for r in norm_rows:
    print(json.dumps(r))
print('FEEDBACK MEAN BY FACTOR/GROUP: class-macro, paired parents h=0,5; positive = improvement')
for group in ('overall', 'tail20'):
    for factor in ('A', 'B'):
        selected = [r for r in feedback_rows if r['factor'] == factor and r['group'] == group]
        print(json.dumps(dict(group=group, factor=factor, **{k:st.mean(r[k] for r in selected) for k in selected[0] if k.endswith('_gain')})))
