"""Read-only experiment audit; writes derived tables to a separate review folder."""
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'presentation/b_shared_tradeoff_review_20260927'
OUT.mkdir(parents=True, exist_ok=True)


def js(path):
    return json.loads(path.read_text(encoding='utf-8'))


def csvrows(path):
    with path.open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def save(name, rows):
    if not rows:
        return
    with (OUT / (name + '.csv')).open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        writer.writeheader()
        writer.writerows(rows)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


runs = {}
for cfg in (ROOT / 'output/sfra_b_shared_tradeoff_analysis').rglob('sfra_config.json'):
    runs['w' + str(js(cfg)['b_transfer']['tail_weight'])] = cfg.parent
runs['old_shared'] = next((ROOT / 'output/sfra_b_shared_transfer_analysis').rglob('sfra_config.json')).parent
runs['A_only'] = ROOT / 'output/sfra_cp_analysis/sfra_cp/seed42/client-longtail/full-cp/lambda10_mu1_protocol42'
metrics = ['overall_acc', 'head20_acc', 'middle60_acc', 'bottom20_tail_acc', 'non_tail_acc']
performance, coverage, changes, events, trace, provenance, perclass = [], [], [], [], [], [], []
round_data, manifests = {}, {}

for label, run in runs.items():
    cfg = js(run / 'sfra_config.json')
    done = js(run / 'completion.json')
    rows = csvrows(run / 'round_metrics.csv')
    assert [int(r['round']) for r in rows] == list(range(101)), label
    assert done['completed_round'] == 100, label
    round_data[label] = rows
    perf = {'run': label, 'completed_round': 100}
    for metric in metrics:
        perf['last20_' + metric] = mean(float(r[metric]) for r in rows[81:101])
        perf['final_' + metric] = float(rows[100][metric])
    peak = max(rows[1:], key=lambda r: float(r['bottom20_tail_acc']))
    perf.update(tail_peak=float(peak['bottom20_tail_acc']), tail_peak_round=int(peak['round']),
                tail_peak_to_final=float(peak['bottom20_tail_acc']) - float(rows[100]['bottom20_tail_acc']))
    prior = js(run / 'class_prior.json')['counts']
    cls = defaultdict(list)
    for epoch in range(80, 100):
        crows = csvrows(run / f'per_class_accuracy_epoch_{epoch}.csv')
        assert len(crows) == 100 and len({r['class_id'] for r in crows}) == 100
        for r in crows:
            cls[int(r['class_id'])].append(float(r['per_class_acc']))
    for c, vals in cls.items():
        perclass.append(dict(run=label, class_id=c, train_count=prior[c], last20_acc=mean(vals)))
    for group, ids in [('many', [c for c, n in enumerate(prior) if n > 100]),
                       ('medium', [c for c, n in enumerate(prior) if 20 <= n <= 100]),
                       ('few', [c for c, n in enumerate(prior) if n < 20])]:
        perf['last20_' + group] = mean(mean(cls[c]) for c in ids)
    performance.append(perf)
    meta = js(run / 'bridge_metadata.json')
    provenance.append(dict(run=label, **{k: meta[k] for k in ['seed', 'initial_lora_sha256', 'frozen_model_sha256',
        'schedule_sha256', 'pool_sha256', 'test_sha256']},
        partition_sha256=digest(run/'partition_manifest.csv'),
        witness_sha256=digest(run/'private_witness_manifest.json'),
        execution=js(run/'execution_config.json') if (run/'execution_config.json').exists() else None,
        b_transfer=cfg.get('b_transfer')))
    if label == 'A_only':
        continue
    assert done['b_transfer_events'] == 8 and done['b_transfer_optimizer_steps'] == 16
    labels = {(int(r['client_id']), int(r['local_position'])): int(r['class_id']) for r in csvrows(run/'partition_manifest.csv')}
    bm = js(run/'b_transfer_manifest.json')
    manifests[label] = bm
    tail = set(bm['tail_ids'])
    for ev in range(30, 101, 10):
        folder = run / f'b_transfer_rounds/r{ev:03d}'
        cal = js(folder / 'calibration_manifest.json')
        event_non_tail_classes = set()
        for client, rec in cal['recipients'].items():
            k = int(client)
            eligible = {c for (i, _), c in labels.items() if i == k and c not in tail}
            nt = [p for b in rec['batches'] for p in b['non_tail_positions']]
            ts = [p for b in rec['batches'] for p in b['tail_positions']]
            covered = {labels[k, p] for p in nt}
            event_non_tail_classes |= covered
            monitored = bm['monitors'][client]
            mon_nt, mon_t = set(monitored['non_tail_positions']), set(monitored['tail_positions'])
            coverage.append(dict(run=label, round=ev, client=k, non_tail_images=len(nt), tail_images=len(ts),
                non_tail_classes=len(covered), available_non_tail_classes=len(eligible),
                coverage=len(covered)/len(eligible) if eligible else '',
                non_tail_monitor_overlap=len(mon_nt & set(nt)), non_tail_monitor_images=len(mon_nt),
                tail_monitor_overlap=len(mon_t & set(ts)), tail_monitor_images=len(mon_t)))
        probe = csvrows(folder/'probe_metrics.csv')
        byphase = defaultdict(dict)
        for r in probe:
            key = (int(r['client_id']), int(r['class_id']))
            assert key not in byphase[r['phase']]
            byphase[r['phase']][key] = r
        assert set(byphase['ordinary_global_B']) == set(byphase['transferred_global_B'])
        for key, before in byphase['ordinary_global_B'].items():
            after = byphase['transferred_global_B'][key]
            assert before['samples'] == after['samples']
            changes.append(dict(run=label, round=ev, client=key[0], class_id=key[1],
                group='tail' if before['is_tail'] == 'True' else 'non_tail', samples=int(before['samples']),
                gain=float(before['la'])-float(after['la']),
                accuracy_delta=float(after['accuracy'])-float(before['accuracy'])))
        if (folder/'c_loss_trace.csv').exists():
            rr = {int(r['c_step']): r for r in csvrows(folder/'c_loss_trace.csv')}
            if set(rr) == {0, 1, 2}:
                trace.append(dict(run=label, round=ev,
                    objective_improvement=float(rr[0]['fixed_pool_objective'])-float(rr[2]['fixed_pool_objective']),
                    objective_improvement_step1=float(rr[0]['fixed_pool_objective'])-float(rr[1]['fixed_pool_objective']),
                    tail_gain=float(rr[0]['tail_la'])-float(rr[2]['tail_la']),
                    non_tail_gain=float(rr[0]['non_tail_la'])-float(rr[2]['non_tail_la']),
                    norm=float(rr[2]['effective_transfer_norm']), ratio=float(rr[2]['transfer_to_ordinary_ratio'])))
        events.append(dict(run=label, round=ev, unique_donors=len(cal['shared_donor_ids']),
                           feedback_clients=len(cal['feedback_clients']), non_tail_classes=len(event_non_tail_classes)))

harm = []
for label in runs:
    for group in ['tail', 'non_tail']:
        vals = [r['gain'] for r in changes if r['run'] == label and r['group'] == group]
        if not vals:
            continue
        harm.append(dict(run=label, group=group, units=len(vals), benefited=sum(v > 1e-6 for v in vals),
            harmed=sum(v < -1e-6 for v in vals), harmed_raw=sum(v < 0 for v in vals),
            harmed_1e4=sum(v < -1e-4 for v in vals), harmed_1e3=sum(v < -1e-3 for v in vals),
            harmed_percent=100*sum(v < -1e-6 for v in vals)/len(vals), mean_gain=mean(vals),
            mean_positive=mean(max(v, 0) for v in vals), mean_harm=mean(max(-v, 0) for v in vals),
            max_harm=max(-v for v in vals),
            accuracy_up=sum(r['accuracy_delta'] > 1e-6 for r in changes if r['run']==label and r['group']==group),
            accuracy_down=sum(r['accuracy_delta'] < -1e-6 for r in changes if r['run']==label and r['group']==group)))

cov_summary = []
for label in runs:
    rr = [r for r in coverage if r['run']==label]
    if not rr:
        continue
    cov_summary.append(dict(run=label, avg_nt_classes_per_client_event=mean(r['non_tail_classes'] for r in rr),
        pooled_coverage=sum(r['non_tail_classes'] for r in rr)/sum(r['available_non_tail_classes'] for r in rr),
        avg_nt_images_per_event=sum(r['non_tail_images'] for r in rr)/8,
        avg_tail_images_per_event=sum(r['tail_images'] for r in rr)/8,
        nt_overlap=sum(r['non_tail_monitor_overlap'] for r in rr)/sum(r['non_tail_monitor_images'] for r in rr),
        tail_overlap=sum(r['tail_monitor_overlap'] for r in rr)/sum(r['tail_monitor_images'] for r in rr)))

checks = []
ref = round_data['w0.5']
for label, rows in round_data.items():
    checks.append(dict(run=label, max_accuracy_diff_before_first_transfer=max(abs(float(a[m])-float(b[m]))
        for a, b in zip(ref[:30], rows[:30]) for m in metrics),
        transfer_manifest_equal=manifests.get(label) == manifests['w0.5'] if label in manifests else None))
baseline = {r['class_id']:r['last20_acc'] for r in perclass if r['run']=='old_shared'}
for row in perclass:
    row['delta_vs_old_shared'] = row['last20_acc'] - baseline[row['class_id']]

for name, values in [('performance',performance),('coverage',coverage),('coverage_summary',cov_summary),
                     ('class_changes',changes),('harm_summary',harm),('events',events),('c_trace',trace),
                     ('per_class',perclass),('checks',checks)]:
    save(name, values)
result = dict(performance=performance, coverage=cov_summary, harm=harm, checks=checks,
              provenance=provenance, events=events, c_trace=trace)
(OUT/'audit.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
for name, values in [('PERFORMANCE', performance),('COVERAGE',cov_summary),('HARM',harm),('CHECKS',checks)]:
    print(name, json.dumps(values))
print('PROVENANCE_SAME', {key:len({str(r[key]) for r in provenance}) == 1 for key in
      ['initial_lora_sha256','frozen_model_sha256','schedule_sha256','pool_sha256','test_sha256','partition_sha256','witness_sha256']})
print('DONORS', {label:sorted({r['unique_donors'] for r in events if r['run']==label}) for label in manifests})
print('TRACE',json.dumps(trace))
print('OUT', OUT)
