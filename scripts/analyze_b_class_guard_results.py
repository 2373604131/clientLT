"""Audit imported guard results without modifying training code or input packages."""
import argparse
import csv
import itertools
import json
import math
from pathlib import Path
import statistics as stats
import sys

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from tools.sfra.b_class_guard import read_run, source_hashes, baseline_kind
from tools.sfra.summary import METRICS


def read(path):
    with path.open(encoding='utf-8-sig', newline='') as handle:
        return list(csv.DictReader(handle))


def js(path):
    return json.loads(path.read_text(encoding='utf-8'))


def save(path, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fields)
        writer.writeheader()
        writer.writerows(rows)


def close(a, b, tol=1e-8):
    if not math.isclose(float(a), float(b), rel_tol=0, abs_tol=tol):
        raise ValueError(f'Values differ: {a} vs {b}')


def mean(rows, key):
    return stats.mean(float(r[key]) for r in rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=REPO/'output/sfra_b_class_guard_analysis/sfra_b_class_guard')
    parser.add_argument('--baseline-root', type=Path, default=REPO/'output/ab_decision_42_0_3407_analysis/ab_decision_42_0_3407')
    parser.add_argument('--output', type=Path, default=REPO/'output/b_class_guard_review_20261004')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    candidates = []
    for path in sorted(args.root.glob('seed*/*/*/*/guard_spec.json')):
        spec = js(path)
        if spec['code_sha256'] != source_hashes(REPO):
            raise ValueError('Imported guard source hashes differ from local source')
        candidates.append((path.parent, spec['settings']['guard'], spec))
    for path in sorted(args.baseline_root.rglob('sfra_config.json')):
        kind = baseline_kind(js(path))
        if kind:
            candidates.append((path.parent, kind, None))
    records, audits, performances = {}, [], []
    event_rows, monitor_rows, client_rows, guard_rows = [], [], [], []
    for run, kind, spec in candidates:
        record = read_run(run, kind, spec)
        perf, signature, per_class, curve, tables, tail = record
        seed = perf['seed']
        if (seed, kind) in records:
            raise ValueError('Duplicate run')
        records[seed, kind] = record
        performances.append(perf)
        prior = js(run/'class_prior.json')
        head = set(sorted(range(100), key=lambda c: (-prior['counts'][c], c))[:20])
        groups = dict(overall_acc=set(range(100)), head20_acc=head,
                      middle60_acc=set(range(100))-head-tail,
                      bottom20_tail_acc=tail, non_tail_acc=set(range(100))-tail)
        # Independently recompute every recorded evaluation, including initial state.
        for rnd in range(101):
            values = read(run/f'per_class_accuracy_epoch_{rnd-1}.csv')
            acc = {int(x['class_id']): float(x['per_class_acc']) for x in values}
            assert len(values) == 100 and set(acc) == set(range(100))
            assert all(math.isfinite(v) and 0 <= v <= 100 for v in acc.values())
            for metric, ids in groups.items():
                close(stats.mean(acc[c] for c in ids), curve[rnd][metric])
        audits.append(dict(seed=seed, guard=kind, check='completion_receipt_protocol_and_505_recomputed_metrics', passed=True))
        if not tables:
            continue
        for event in tables['b_transfer_rounds']:
            rnd = int(event['round'])
            folder = run/f'b_transfer_rounds/r{rnd:03d}'
            manifest = js(folder/'calibration_manifest.json')
            trace = read(folder/'c_loss_trace.csv')
            steps = read(folder/'optimization_steps.csv')
            event_rows.append(dict(seed=seed, guard=kind, round=rnd,
                donors=len(manifest['shared_donor_ids']),
                norm=float(event['effective_global_transfer_norm']),
                ordinary_norm=float(event['ordinary_B_effective_update_norm']),
                tail_la_gain=float(event['tail_la_gain_after_B_aggregation']),
                fixed_pool_la_gain=float(trace[0]['fixed_pool_la'])-float(trace[-1]['fixed_pool_la']),
                objective_decreased_steps=sum(float(x['objective_change_same_batch']) < 0 for x in steps),
                step1_norm=float(steps[0]['effective_transfer_norm_after']),
                step2_gradient_ratio=float(steps[1].get('guard_to_base_gradient_ratio') or 0),
                step2_guard_before=float(steps[1].get('guard_before') or 0)))
            if spec is not None and kind != 'off':
                cells = read(folder/'guard_class_trace.csv')
                clients = read(folder/'guard_client_trace.csv')
                with np.load(folder/'guard_references.npz', allow_pickle=False) as refs:
                    for client, rec in manifest['recipients'].items():
                        for batch, sampled in enumerate(rec['batches'], 1):
                            key = f'client{int(client):02d}_batch{batch}'
                            positions = sampled['tail_positions']+sampled['non_tail_positions']
                            assert np.array_equal(refs[key+'_positions'], positions)
                            labels, losses = refs[key+'_labels'], refs[key+'_losses']
                            assert np.isfinite(losses).all() and len(losses) == len(positions)
                            zero = [r for r in cells if r['client_id']==client and int(r['batch'])==batch and int(r['c_step'])==0]
                            assert {int(x['class_id']) for x in zero} == set(labels.tolist())
                            for row in zero:
                                selected = losses[labels == int(row['class_id'])]
                                assert len(selected) == int(row['samples'])
                                close(selected.mean(), row['reference_la'], 2e-6)
                                close(row['delta_la'], 0)
                for step in (0, 1, 2):
                    state = [r for r in cells if int(r['c_step'])==step]
                    cell_harm = sum(float(r['weighted_harm'])*float(r['client_weight'])/2 for r in state)
                    group_changes = {}
                    for row in state:
                        key = (row['client_id'], row['batch'], row['group'])
                        group_changes[key] = group_changes.get(key, 0.) + float(row['local_weight'])*float(row['delta_la'])
                    group_harm = sum(max(v, 0.) for v in group_changes.values())/len(manifest['feedback_clients'])/2
                    logged = [r for r in clients if int(r['c_step'])==step]
                    close(cell_harm, mean(logged, 'class_guard'), 2e-8)
                    close(group_harm, mean(logged, 'client_guard'), 2e-8)
                    close(trace[step]['fixed_pool_guard'], cell_harm if kind=='class' else group_harm, 2e-8)
                    guard_rows.append(dict(seed=seed, guard=kind, round=rnd, c_step=step,
                        cell_harm=cell_harm, group_harm=group_harm,
                        observed_cells=len(state), active_cells=sum(float(r['delta_la'])>0 for r in state)))
                close(steps[0]['guard_gradient_norm'], 0)
                assert float(steps[1]['guard_gradient_norm']) > 0
        probes = tables['probe_metrics']
        after = {(r['round'], r['client_id'], r['class_id']): r for r in probes if r['phase']=='transferred_global_B'}
        by_client = {}
        for before in probes:
            if before['phase'] != 'ordinary_global_B':
                continue
            key = (before['round'], before['client_id'], before['class_id'])
            after_row = after[key]
            assert before['samples']==after_row['samples'] and before['is_tail']==after_row['is_tail']
            row = dict(seed=seed, guard=kind, round=int(key[0]), client_id=int(key[1]),
                       class_id=int(key[2]), is_tail=before['is_tail']=='True', samples=int(before['samples']),
                       delta_la=float(after_row['la'])-float(before['la']),
                       delta_acc=float(after_row['accuracy'])-float(before['accuracy']))
            monitor_rows.append(row)
            by_client.setdefault(key[:2], []).append(row)
        for key, rows in by_client.items():
            grouped_delta = {}
            for is_tail in (True, False):
                group = [r for r in rows if r['is_tail']==is_tail]
                grouped_delta[is_tail] = sum(r['samples']*r['delta_la'] for r in group)/sum(r['samples'] for r in group)
            delta = .35*grouped_delta[True]+.65*grouped_delta[False]
            gain = any(r['delta_la'] < -.001 for r in rows)
            harm = any(r['delta_la'] > .001 for r in rows)
            client_rows.append(dict(seed=seed, guard=kind, round=int(key[0]), client_id=int(key[1]),
                               weighted_delta_la=delta, mixed=gain and harm, mean_improves_some_harm=delta<-.001 and harm))
        audits.append(dict(seed=seed, guard=kind, check='transfer_budgets_and_guard_references_and_penalty_reconstruction', passed=True))
    pairs, pair_classes = [], []
    for seed in sorted({k[0] for k in records}):
        kinds = [k for k in ('A','B-old','client','class') if (seed,k) in records]
        for right, left in itertools.combinations(kinds, 2):
            a,b = records[seed,left],records[seed,right]
            assert a[1]==b[1], 'Source/protocol fingerprints differ'
            for x,y in zip(a[3][:30],b[3][:30]):
                for metric in METRICS:
                    close(x[metric], y[metric])
            if left!='A' and right!='A':
                for rnd in range(30,101,10):
                    ma,mb=(js(Path(r[0]['run'])/f'b_transfer_rounds/r{rnd:03d}/calibration_manifest.json') for r in (a,b))
                    assert ma['feedback_clients']==mb['feedback_clients']
                    assert {k:v['batches'] for k,v in ma['recipients'].items()}=={k:v['batches'] for k,v in mb['recipients'].items()}
                keys = lambda r: {(x['round'],x['client_id'],x['class_id'],x['samples'],x['is_tail']) for x in r[4]['probe_metrics'] if x['phase']=='ordinary_global_B'}
                assert keys(a)==keys(b)
            differences = {c:a[2][c]-b[2][c] for c in a[5]}
            pairs.append(dict(seed=seed,left=left,right=right,
                **{'delta_'+m:a[0]['last20_'+m]-b[0]['last20_'+m] for m in METRICS},
                tail_improved=sum(v>1e-8 for v in differences.values()),
                tail_tied=sum(abs(v)<=1e-8 for v in differences.values()),
                tail_harmed=sum(v< -1e-8 for v in differences.values()),
                tail_negative_sum=sum(min(v,0) for v in differences.values()),worst_tail_delta=min(differences.values())))
            pair_classes.extend(dict(seed=seed,left=left,right=right,class_id=c,is_tail=c in a[5],delta_acc=a[2][c]-b[2][c]) for c in range(100))
            audits.append(dict(seed=seed,guard=left+' vs '+right,check='source_protocol_pre_B_curve_and_feedback_identity',passed=True))
    monitor_summary=[]
    for seed,kind in records:
        selected=[r for r in monitor_rows if r['seed']==seed and r['guard']==kind]
        if not selected:
            continue
        for tail in (True,False):
            rows=[r for r in selected if r['is_tail']==tail]
            monitor_summary.append(dict(seed=seed,guard=kind,group='tail' if tail else 'non_tail',cells=len(rows),
                la_harmed_gt_001=sum(r['delta_la']>.001 for r in rows),
                la_improved_lt_neg001=sum(r['delta_la']<-.001 for r in rows),
                positive_la_sum_unweighted=sum(max(r['delta_la'],0) for r in rows),
                accuracy_harmed=sum(r['delta_acc']< -1e-8 for r in rows),
                accuracy_improved=sum(r['delta_acc']>1e-8 for r in rows)))
    for name,rows in dict(audit=audits,performance=performances,paired=pairs,paired_per_class=pair_classes,
        events=event_rows,guard_trace=guard_rows,monitor_cells=monitor_rows,monitor_clients=client_rows,
        monitor_summary=monitor_summary).items():
        save(args.output/(name+'.csv'),rows)
    summary=dict(runs=len(records),new_seeds=sorted({s for s,k in records if k in ('client','class')}),
        evaluations_recomputed=101*len(records),group_metrics_recomputed=505*len(records),
        audit_rows=len(audits),all_passed=True,
        limitations=['No checkpoint or GPU replay; artifact and receipt audit only.',
            'Probe comparisons use each run own pre-B reference; later model states differ.',
            'Client/class/round observations are repeated training diagnostics, not independent seeds.',
            'Primary endpoint remains rounds81..100 Tail20 test accuracy, not best round or training loss.'])
    (args.output/'audit_summary.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(summary,indent=2))
    print('Saved:',args.output)


if __name__=='__main__':
    main()
