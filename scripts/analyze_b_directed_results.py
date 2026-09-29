"""Audit packed directed-B results without loading checkpoints or running training.

Run from the repository: python scripts/analyze_b_directed_results.py
Only derived files under presentation/b_directed_review_20260928 are written.
"""
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'presentation/b_directed_review_20260928'
EPS = 1e-6


def js(path):
    return json.loads(path.read_text(encoding='utf-8'))


def rows(path):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def save(name, records):
    with (OUT / (name + '.csv')).open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(dict.fromkeys(k for r in records for k in r)))
        writer.writeheader()
        writer.writerows(records)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    configs = list((ROOT / 'output/sfra_b_directed_analysis').rglob('sfra_config.json'))
    if len(configs) != 1:
        raise ValueError(f'Expected exactly one directed run, found {len(configs)}')
    directed = configs[0].parent
    runs = {'A_only': ROOT / 'output/sfra_cp_analysis/sfra_cp/seed42/client-longtail/full-cp/lambda10_mu1_protocol42',
            'old_shared': next((ROOT / 'output/sfra_b_shared_transfer_analysis').rglob('sfra_config.json')).parent}
    for cfg in (ROOT / 'output/sfra_b_shared_tradeoff_analysis').rglob('sfra_config.json'):
        runs['w' + str(js(cfg)['b_transfer']['tail_weight'])] = cfg.parent
    runs['directed'] = directed
    checks = []

    def check(name, value):
        checks.append(dict(check=name, passed=bool(value)))
        if not value:
            raise ValueError('Audit failed: ' + name)

    performance, perclass, provenance, curves = [], [], [], {}
    metrics = ['overall_acc', 'head20_acc', 'middle60_acc', 'bottom20_tail_acc', 'non_tail_acc']
    for label, run in runs.items():
        curve = rows(run / 'round_metrics.csv')
        curves[label] = curve
        check(label + ': rounds 0..100 complete', [int(r['round']) for r in curve] == list(range(101)))
        check(label + ': committed round 100', js(run / 'completion.json')['completed_round'] == 100)
        perf = dict(run=label)
        for metric in metrics:
            perf['last20_' + metric] = mean(float(r[metric]) for r in curve[81:101])
            perf['final_' + metric] = float(curve[100][metric])
        prior = js(run / 'class_prior.json')['counts']
        cls = defaultdict(list)
        for epoch in range(80, 100):
            cr = rows(run / f'per_class_accuracy_epoch_{epoch}.csv')
            check(f'{label}: epoch {epoch} classes complete', {int(r['class_id']) for r in cr} == set(range(100)) and len(cr) == 100)
            for r in cr:
                cls[int(r['class_id'])].append(float(r['per_class_acc']))
        for c, values in sorted(cls.items()):
            perclass.append(dict(run=label, class_id=c, train_count=prior[c], last20_acc=mean(values)))
        for name, ids in [('many', [c for c, n in enumerate(prior) if n > 100]),
                          ('medium', [c for c, n in enumerate(prior) if 20 <= n <= 100]),
                          ('few', [c for c, n in enumerate(prior) if n < 20])]:
            perf['last20_' + name] = mean(mean(cls[c]) for c in ids)
        check(label + ': Tail20 agrees with per-class files', abs(perf['last20_bottom20_tail_acc'] - mean(mean(cls[c]) for c in range(80, 100))) < 1e-9)
        performance.append(perf)
        meta = js(run / 'bridge_metadata.json')
        provenance.append(dict(run=label, path=str(run.relative_to(ROOT)),
            hashes={k: v for k, v in meta.items() if k.endswith('_sha256')},
            partition_sha256=digest(run / 'partition_manifest.csv'),
            witness_sha256=digest(run / 'private_witness_manifest.json'),
            execution=js(run / 'execution_config.json') if (run / 'execution_config.json').exists() else None))

    for p in performance:
        p['tail_delta_vs_A_pp'] = p['last20_bottom20_tail_acc'] - performance[0]['last20_bottom20_tail_acc']
    for p in provenance[1:]:
        for key in ['initial_lora_sha256', 'frozen_model_sha256', 'schedule_sha256', 'pool_sha256', 'test_sha256']:
            check(p['run'] + ': ' + key, p['hashes'][key] == provenance[0]['hashes'][key])
        for key in ['partition_sha256', 'witness_sha256']:
            check(p['run'] + ': ' + key, p[key] == provenance[0][key])
        check(p['run'] + ': pre-B recorded accuracy identical', all(
            float(r[k]) == float(b[k]) for r, b in zip(curves[p['run']][:30], curves['A_only'][:30]) for k in metrics))
        cfg, basecfg = js(runs[p['run']] / 'sfra_config.json'), js(runs['A_only'] / 'sfra_config.json')
        check(p['run'] + ': same non-transfer SFRA config', {k: v for k, v in cfg.items() if k != 'b_transfer'} == {k: v for k, v in basecfg.items() if k != 'b_transfer'})

    cfg = js(directed / 'sfra_config.json')['b_transfer']
    bm = js(directed / 'b_transfer_manifest.json')
    target = bm['target_clients']
    tail = bm['tail_ids']
    totals = {int(c): n for c, n in bm['target_class_totals'].items()}
    partition = rows(directed / 'partition_manifest.csv')
    counts = Counter((int(r['client_id']), int(r['class_id'])) for r in partition)
    labels = {(int(r['client_id']), int(r['local_position'])): int(r['class_id']) for r in partition}
    clients = {k for k, _ in labels}
    expected_edges = {(c, j) for c in totals for j in clients if j not in target and counts[j, c] == 0}
    check('target clients exactly 27,28,29', target == [27, 28, 29])
    check('target class pool counts match partition', totals == {c: sum(counts[k, c] for k in target) for c in tail})
    check('all 20 target classes present', len(totals) == 20)
    check('100 rounds, 8 events, 16 C steps', all(js(directed / 'completion.json')[k] == v for k, v in
          [('completed_round', 100), ('b_transfer_events', 8), ('b_transfer_optimizer_steps', 16)]))
    events, class_changes, receiver_changes, source_rows, probe_rows, trace = [], [], [], [], [], []
    for rnd in cfg['rounds']:
        folder = directed / f'b_transfer_rounds/r{rnd:03d}'
        cal, summary = js(folder / 'calibration_manifest.json'), js(folder / 'summary.json')
        scores, sources = rows(folder / 'donor_scores.csv'), rows(folder / 'source_weights.csv')
        source_rows += sources
        probe_rows += scores
        scored = {(int(r['class_id']), int(r['donor'])): r for r in scores}
        selected = {(int(r['class_id']), int(r['donor'])): float(r['weight']) for r in sources}
        check(f'r{rnd}: complete eligible class-donor candidates', set(scored) == expected_edges and len(scores) == len(expected_edges))
        check(f'r{rnd}: unique selected edges', len(selected) == len(sources))
        check(f'r{rnd}: all selected donors external and label-absent', all(j not in target and counts[j, c] == 0 and int(scored[c, j]['donor_class_count']) == 0 for c, j in selected))
        expected = {}
        for c in totals:
            ranked = sorted([(j, float(row['gain'])) for (cc, j), row in scored.items() if cc == c and float(row['gain']) > cfg['min_gain']], key=lambda x: (-x[1], x[0]))[:cfg['donors_per_class']]
            denominator = sum(g for _, g in ranked)
            expected.update({(c, j): g / denominator for j, g in ranked})
        check(f'r{rnd}: exact positive top-K and normalized weights', selected.keys() == expected.keys() and all(abs(selected[k] - expected[k]) < 1e-12 for k in expected))
        check(f'r{rnd}: scored flags and weights match selections', all(
            (row['selected'].lower() == 'true') == ((c, j) in selected) and
            abs(float(row['source_weight']) - selected.get((c, j), 0.)) < 1e-12 and
            abs(float(row['baseline_la']) - float(row['injected_la']) - float(row['gain'])) < 1e-12
            for (c, j), row in scored.items()))
        mix = {(int(c), int(j)): w for c, donors in cal['source_mixtures'].items() for j, w in donors.items()}
        check(f'r{rnd}: source manifest matches actual weights', mix == selected)
        check(f'r{rnd}: feedback clients match targets', cal['feedback_clients'] == target and sorted(map(int, cal['recipients'])) == target)
        for client, rec in cal['recipients'].items():
            k = int(client)
            full = {p for (i, p), c in labels.items() if i == k and c in tail}
            check(f'r{rnd}: client {k} full tail-only pool', set(rec['tail_positions']) == full and len(rec['tail_positions']) == len(full) and not rec['non_tail_positions'])
        feedback = rows(folder / 'client_feedback_steps.csv')
        check(f'r{rnd}: both optimizer steps use full tail-only pool', len(feedback) == 6 and
              {(int(r['step']), int(r['client_id'])) for r in feedback} == {(s, k) for s in [1, 2] for k in target} and
              all(int(r['non_tail_samples']) == 0 and int(r['tail_samples']) == sum(counts[int(r['client_id']), c] for c in tail) for r in feedback))
        with np.load(folder / 'matrices.npz') as archive:
            basis_classes = archive['basis_class_ids'].tolist()
            donors = archive['donor_ids'].tolist()
            check(f'r{rnd}: NPZ source weights match CSV', np.allclose(archive['source_weights'], [[selected.get((c, j), 0.) for j in donors] for c in basis_classes], atol=1e-12, rtol=0))
            check(f'r{rnd}: class and donor axes match manifest', basis_classes == cal['basis_class_ids'] == sorted(totals) and donors == cal['shared_donor_ids'] == sorted({j for c, j in selected}))
            matrices = [archive[f'module_{i:02d}'].astype(np.float64) for i in range(len(cal['module_order']))]
            check(f'r{rnd}: finite class-indexed 20x4x4 matrices', all(a.shape == (20, 4, 4) and np.isfinite(a).all() for a in matrices))
            penalty = cfg['regularization'] * sum(float(np.square(a).sum()) for a in matrices) / (len(basis_classes) * len(matrices))
        check(f'r{rnd}: summary source counts match files', summary['unique_donors'] == len(donors) and summary['donor_links'] == len(selected) and summary['supported_classes'] == len(basis_classes) and summary['candidate_pairs'] == len(scores))
        check(f'r{rnd}: nonzero transfer, two steps, no non-tail samples', summary['effective_global_transfer_norm'] > 0 and summary['optimizer_steps'] == 2 and summary['non_tail_samples'] == 0)
        probes = rows(folder / 'probe_metrics.csv')
        by_unit = defaultdict(dict)
        for r in probes:
            by_unit[int(r['client_id']), int(r['class_id'])][r['phase']] = r
        expected_units = {(k, c) for k in target for c in totals if counts[k, c]}
        check(f'r{rnd}: complete three-phase target measurements', set(by_unit) == expected_units and len(probes) == 3 * len(expected_units))
        pooled = defaultdict(list)
        for (k, c), phases in by_unit.items():
            b, t, a = [phases[p] for p in ['ordinary_global_B', 'transferred_global_B', 'committed_global']]
            check(f'r{rnd}: client {k}, class {c} same full pool', all(int(v['samples']) == counts[k, c] for v in phases.values()))
            change = dict(round=rnd, client_id=k, class_id=c, samples=int(b['samples']),
                la_before=float(b['la']), la_after_B=float(t['la']), la_after_A=float(a['la']),
                B_la_gain=float(b['la']) - float(t['la']), A_la_gain=float(t['la']) - float(a['la']),
                accuracy_before=float(b['accuracy']), accuracy_after_B=float(t['accuracy']),
                B_accuracy_delta_pp=float(t['accuracy']) - float(b['accuracy']))
            receiver_changes.append(change)
            pooled[c].append(change)
        current = []
        for c, recs in sorted(pooled.items()):
            rec = dict(round=rnd, class_id=c, samples=sum(v['samples'] for v in recs))
            for k in ['la_before', 'la_after_B', 'la_after_A', 'B_la_gain', 'A_la_gain', 'accuracy_before', 'accuracy_after_B', 'B_accuracy_delta_pp']:
                rec[k] = sum(v[k] * v['samples'] for v in recs) / rec['samples']
            current.append(rec)
        class_changes += current
        for phase, name in [('ordinary_global_B', 'la_before'), ('transferred_global_B', 'la_after_B'), ('committed_global', 'la_after_A')]:
            check(f'r{rnd}: {phase} is sample-mean then class-macro', abs(mean(v[name] for v in current) - summary[phase + '_tail_la']) < 1e-9)
        steps = rows(folder / 'optimization_steps.csv')
        objectives = [float(s['objective_before']) for s in steps] + [summary['transferred_global_B_tail_la'] + penalty]
        check(f'r{rnd}: regularized objective falls at both steps', len(steps) == 2 and objectives[0] > objectives[1] > objectives[2])
        for step, objective in enumerate(objectives):
            trace.append(dict(round=rnd, completed_C_steps=step, objective=objective,
                la=float(steps[step]['la_before']) if step < 2 else summary['transferred_global_B_tail_la'],
                regularization=float(steps[step]['regularization_penalty_before']) if step < 2 else penalty))
        events.append(dict(round=rnd, candidate_pairs=len(scores), positive_pairs=sum(float(s['gain']) > cfg['min_gain'] for s in scores),
            selected_pairs=len(selected), unique_donors=len(donors), supported_classes=len(basis_classes),
            mean_B_la_gain=mean(v['B_la_gain'] for v in current), mean_A_la_gain=mean(v['A_la_gain'] for v in current),
            improved_classes=sum(v['B_la_gain'] > EPS for v in current), harmed_classes=sum(v['B_la_gain'] < -EPS for v in current),
            accuracy_improved_classes=sum(v['B_accuracy_delta_pp'] > EPS for v in current),
            accuracy_harmed_classes=sum(v['B_accuracy_delta_pp'] < -EPS for v in current),
            mean_B_accuracy_delta_pp=mean(v['B_accuracy_delta_pp'] for v in current),
            transfer_norm=summary['effective_global_transfer_norm'], ordinary_update_norm=summary['ordinary_B_effective_update_norm'],
            norm_ratio=summary['effective_global_transfer_norm'] / summary['ordinary_B_effective_update_norm'],
            objective_before=objectives[0], objective_after_first=objectives[1], objective_after_second=objectives[2],
            final_regularization=penalty, algorithm_forward_images=summary['algorithm_forward_images'],
            algorithm_backward_images=summary['algorithm_backward_images'], seconds=summary['seconds']))

    lookup = {(p['run'], p['class_id']): p['last20_acc'] for p in perclass}
    tail_summary = []
    for c in totals:
        cc = [r for r in class_changes if r['class_id'] == c]
        tail_summary.append(dict(class_id=c, target_train_samples=totals[c], mean_B_la_gain=mean(r['B_la_gain'] for r in cc),
            harmed_events=sum(r['B_la_gain'] < -EPS for r in cc),
            A_only_last20_acc=lookup['A_only', c], directed_last20_acc=lookup['directed', c],
            directed_minus_A_pp=lookup['directed', c] - lookup['A_only', c]))
    selected_gains = [float(r['gain']) for r in probe_rows if r['selected'].lower() == 'true']
    audit = dict(runs=provenance, target_clients=target, target_samples=sum(totals.values()),
        target_class_totals=totals, target_receiver_class_units=len(expected_units),
        performance=performance, events=events,
        class_event_count=len(class_changes), improved_class_events=sum(r['B_la_gain'] > EPS for r in class_changes),
        harmed_class_events=sum(r['B_la_gain'] < -EPS for r in class_changes),
        mean_B_la_gain=mean(r['B_la_gain'] for r in class_changes),
        accuracy_improved_class_events=sum(r['B_accuracy_delta_pp'] > EPS for r in class_changes),
        accuracy_harmed_class_events=sum(r['B_accuracy_delta_pp'] < -EPS for r in class_changes),
        selected_probe_gain=dict(min=min(selected_gains), median=median(selected_gains), max=max(selected_gains)),
        checkpoint_files=len(list(directed.rglob('*.pt'))),
        limitations=['Single seed; rounds are repeated measures, not independent runs.',
                     'Same training pool used for screening, C optimization, and transfer diagnostics.',
                     'Source/weight/matrix axes checked; absent raw deltas and commit.pt prevent independent tensor reconstruction.',
                     'Historical B differs in feedback pool and parameterization; performance comparison does not isolate one changed component.'],
        checks=checks)
    OUT.mkdir(parents=True, exist_ok=True)
    for name, data in [('performance', performance), ('per_class', perclass), ('tail_class_summary', tail_summary),
                       ('events', events), ('class_changes', class_changes), ('receiver_changes', receiver_changes),
                       ('objective_trace', trace), ('source_weights', source_rows), ('donor_scores', probe_rows), ('checks', checks)]:
        save(name, data)
    (OUT / 'audit.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({k: audit[k] for k in ['target_samples', 'class_event_count', 'improved_class_events', 'harmed_class_events',
        'mean_B_la_gain', 'accuracy_improved_class_events', 'accuracy_harmed_class_events', 'selected_probe_gain', 'checkpoint_files']}, indent=2))
    print(f'Checks passed: {len(checks)}; output: {OUT.relative_to(ROOT)}')
    for p in performance:
        print(p['run'], 'last20 Tail20=', round(p['last20_bottom20_tail_acc'], 4), 'final Tail20=', p['final_bottom20_tail_acc'])


if __name__ == '__main__':
    main()
