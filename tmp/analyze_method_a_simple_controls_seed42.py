"""Reproduce the local seed42 review without modifying imported analysis/plan files."""
import json
import statistics
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from tools.sfra.simple_controls import load_json, read_csv, static_tables, write_table
from tools.sfra.simple_controls_summary import read_result, sample_metrics, cost_metrics

ROOT = REPO / 'output/method_a_simple_controls_v1_probe_fix_results/method_a_simple_controls_v1_probe_fix'
BASE = REPO / 'output/ab_decision_42_0_3407_analysis/ab_decision_42_0_3407'
OUT = ROOT / 'review'
OUT.mkdir(exist_ok=True)
plan = load_json(ROOT / 'control_protocol.json')
jobs = {job['method']: job for job in plan['jobs']}
runs = {
    's': BASE / 'runs/s/seed42/client-longtail/s/baseline_protocol42_fast_v2_f128_c4',
    'full-cp': BASE / 'runs/a/seed42/client-longtail/full-cp/lambda10_mu1_protocol42_fast_v2_f128_c4',
}
runs.update({m: ROOT / job['run'] for m, job in jobs.items()})
results, per_class, audits, samples, retention, costs = {}, {}, [], [], [], []
for method, run in runs.items():
    result = read_result(run, method, jobs.get(method))
    transitions, note, signature = sample_metrics(result)
    result['sample_signature'] = signature
    result['initial_per_class'] = {
        int(r['class_id']): float(r['per_class_acc'])
        for r in read_csv(run / 'per_class_accuracy_epoch_-1.csv')
    }
    assert set(result['initial_per_class']) == set(range(100))
    result['environment'] = load_json(run / 'bridge_metadata.json')['environment']
    results[method] = result
    samples.extend(transitions)
    costs.append(cost_metrics(result))
    for group in ('Tail20', 'Non-tail'):
        rows = [r for r in transitions if r['group'] == group and 81 <= r['round'] <= 100]
        if rows:
            retention.append(dict(method=method, group=group, initial_correct=rows[0]['initial_correct'],
                initial_wrong=rows[0]['initial_wrong'], **{
                    'last20_mean_' + k: statistics.mean(r[k] for r in rows)
                    for k in ('retained', 'learned', 'retention_percent', 'acquisition_percent')}))
    round_values = [{int(r['class_id']): float(r['per_class_acc'])
        for r in read_csv(run / f'per_class_accuracy_epoch_{rnd-1}.csv')} for rnd in range(81, 101)]
    per_class[method] = {c: statistics.mean(r[c] for r in round_values) for c in range(100)}
    audits.append(dict(method=method, run=str(run.relative_to(REPO)), status='passed',
        completion_round=result['completion']['completed_round'], sample_record_status=note,
        initial_sample_signature=signature, environment=result['environment']))

comparisons = [('full-cp', 's'), ('tailrw-g1', 's'), ('tailrw-g4', 's'), ('tailrw-g16', 's'),
    ('cover-cp', 's'), ('full-cp', 'cover-cp'), ('tailrw-g16', 'full-cp'),
    ('tailrw-g16', 'cover-cp'), ('tailrw-g16', 'tailrw-g1')]
pairs, deltas, class_deltas, group_counts = [], [], [], []
for left, right in comparisons:
    a, b = results[left], results[right]
    mismatches = [k for k in a['signature'] if a['signature'][k] != b['signature'][k]]
    assert not mismatches, (left, right, mismatches)
    assert a['initial_per_class'] == b['initial_per_class'], (left, right, 'initial per-class')
    if a['witness'] is not None and b['witness'] is not None:
        assert a['witness'] == b['witness']
    if a['sample_signature'] is not None and b['sample_signature'] is not None:
        assert a['sample_signature'] == b['sample_signature']
    name = left + ' minus ' + right
    same_env = a['environment'] == b['environment']
    pairs.append(dict(comparison=name, protocol_signature_match=True,
        initial_per_class_match=True, initial_sample_match=(a['sample_signature'] == b['sample_signature']
            if a['sample_signature'] is not None and b['sample_signature'] is not None else 'unavailable'),
        recorded_environment_match=same_env,
        interpretation='single-seed descriptive; ' + ('same recorded environment' if same_env else
            'cross-environment reference: do not attribute small differences solely to method or compare wall time')))
    deltas.append(dict(comparison=name, **{'delta_' + k: a['performance'][k] - b['performance'][k]
        for k in a['performance'] if k.startswith('last20_') or k in (
            'final_tail', 'tail_change_90_to_100', 'tail_peak_to_final')}))
    for c in range(100):
        class_deltas.append(dict(comparison=name, class_id=c,
            delta_last20_accuracy=per_class[left][c] - per_class[right][c]))
    for group, ids in a['groups'].items():
        differences = [per_class[left][c] - per_class[right][c] for c in ids]
        group_counts.append(dict(comparison=name, group=group, improved=sum(d > 1e-8 for d in differences),
            tied=sum(abs(d) <= 1e-8 for d in differences), declined=sum(d < -1e-8 for d in differences),
            mean_delta=statistics.mean(differences)))

clients, classes = static_tables(results['s']['counts'])
tables = dict(performance=[r['performance'] for r in results.values()], comparisons=deltas,
    pair_audit=pairs, costs=costs, sample_retention=retention, sample_transitions=samples,
    class_comparisons=class_deltas, group_class_counts=group_counts,
    per_class=[dict(method=m, class_id=c, last20_mean_accuracy=v)
        for m, values in per_class.items() for c, v in values.items()],
    client_weights=clients, class_counts=classes,
    curves=[dict(row, method=m) for m, r in results.items() for row in r['curves']])
for name, rows in tables.items():
    write_table(OUT / (name + '.csv'), rows)

receipt = dict(endpoint='mean accuracy over rounds 81..100', seed=42, independent_seeds=1,
    sources=audits, pair_audits=pairs,
    baseline_registration='Recovered independently from local AB-decision archive; original control plan unchanged.',
    interpretation_limits=[
        'S and Full-CP use RTX 3090 / torch 2.12.0+cu126; four new controls use RTX 4090 / torch 2.6.0+cu118.',
        'Protocol, partition, initialization, core training code, optimizer budget and initial class accuracies match.',
        'S is staged LoRA A/B training with LA, not jointly trained FedAvg or always-frozen A.',
        'TailRW weights both A and B aggregation; this does not isolate B-only improvement.',
        'Cover-CP changes priorities only and still uses original response measurement, targets and correction.',
        'S and Full-CP lack per-sample predictions in the reused archive.',
        'Gamma=4 was the prespecified representative; gamma=16 is strongest on this seed, not a validated optimum.',
        'Round/class/sample variation is not independent-seed uncertainty; no significance or equivalence claim.',
        'Communication bytes are protocol accounting in sequential simulation, not measured network transfer.',
    ])
(OUT / 'audit.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
print(json.dumps(dict(output=str(OUT), performance=tables['performance'],
    sample_retention=retention, pair_audit=pairs), ensure_ascii=False, indent=2))
