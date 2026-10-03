"""Read-only audit and decision evidence for completed frozen AB experiments."""
import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO))
from tools.sfra.ab_validation import (PLAN_NAME,CONTRACT,CONTRASTS,SCHEMA,code_hashes,
    load_json,read_result,safe_run,control_pair_check)
from tools.sfra.summary import METRICS,read_csv,write_csv


def mean(values):
    return statistics.mean(values)


def close(a,b):
    return math.isclose(float(a),float(b),rel_tol=0,abs_tol=1e-8)


def analyze(root,out):
    root=Path(root).resolve();out=Path(out).resolve()
    if out==root or out.is_relative_to(root) or root.is_relative_to(out):
        raise ValueError('Use a separate output directory; preserve imported results')
    plan=load_json(root/PLAN_NAME)
    assert plan['schema_version']==SCHEMA and plan['contract']==CONTRACT
    jobs=plan['jobs'];results={};blocks=defaultdict(dict);audits=[];runs=[];classes=[]
    events=[];costs=[];pairs=[];pair_classes=[];coverage=[];checks=0;source_differences=[]
    local_hashes=code_hashes(REPO)
    snapshot={p:hashlib.sha256(p.read_bytes()).hexdigest() for p in [root/PLAN_NAME,*sorted((root/'analysis').glob('*'))] if p.is_file()}
    for job in jobs:
        run=safe_run(root,job['run'])
        if not (run/'completion.json').exists():
            audits.append(dict(seed=job['seed'],arm=job['arm'],status='incomplete_or_missing'))
            continue
        result=read_result(run,job)
        results[job['seed'],job['arm']]=result;blocks[job['seed']][job['arm']]=job
        r,signature,witness,per_class,curve=result
        prior=load_json(run/'class_prior.json');tail=set(prior['tail_ids'])
        head=set(sorted(range(100),key=lambda c:(-prior['counts'][c],c))[:20])
        groups=dict(overall_acc=set(range(100)),head20_acc=head,bottom20_tail_acc=tail,
                    middle60_acc=set(range(100))-head-tail,non_tail_acc=set(range(100))-tail)
        # Recompute every official curve point from all 101 per-class tables.
        for row in curve:
            values=read_csv(run/f'per_class_accuracy_epoch_{int(row["round"])-1}.csv')
            assert len(values)==100 and {int(x['class_id']) for x in values}==set(range(100))
            pc={int(x['class_id']):float(x['per_class_acc']) for x in values}
            assert all(math.isfinite(v) and 0<=v<=100 for v in pc.values())
            for metric,ids in groups.items():
                assert close(mean(pc[c] for c in ids),row[metric]),(job['run'],row['round'],metric)
                checks+=1
        source_differences.extend(dict(seed=job['seed'],arm=job['arm'],file=k)
            for k,v in job['code_sha256'].items() if local_hashes.get(k)!=v)
        audits.append(dict(seed=job['seed'],arm=job['arm'],status='verified',official_rounds=len(curve),
            local_steps=load_json(run/'completion.json')['total_local_optimizer_steps'],
            functional_steps=load_json(run/'completion.json')['functional_correction_steps']))
        runs.append(r)
        classes.extend(dict(seed=job['seed'],arm=job['arm'],class_id=c,is_tail=c in tail,last20_acc=v) for c,v in per_class.items())
        if job['arm'] not in ('ab','a-calibration'):
            continue
        b_rows=read_csv(run/'b_transfer_rounds.csv')
        tail_ids_seen=set();nontail_ids_seen=set()
        for row in b_rows:
            rnd=int(row['round']);folder=run/f'b_transfer_rounds/r{rnd:03d}'
            trace=sorted(read_csv(folder/'c_loss_trace.csv'),key=lambda x:int(x['c_step']))
            steps=read_csv(folder/'optimization_steps.csv');receiver=read_csv(folder/'receiver_summary.csv')
            manifest=load_json(folder/'calibration_manifest.json')
            for client,info in manifest['recipients'].items():
                for batch in info['batches']:
                    tail_ids_seen.update((int(client),i) for i in batch['tail_positions'])
                    nontail_ids_seen.update((int(client),i) for i in batch['non_tail_positions'])
            delta=[float(b['fixed_pool_objective'])-float(a['fixed_pool_objective']) for a,b in zip(trace,trace[1:])]
            gain=[float(x['before_tail_la'])-float(x['after_tail_la']) for x in receiver]
            norm_error=[abs(float(x['effective_transfer_norm_after'])-float(x['reference_effective_norm']))/float(x['reference_effective_norm'])
                        for x in steps if x.get('reference_effective_norm') and float(x['reference_effective_norm'])>0]
            events.append(dict(seed=job['seed'],arm=job['arm'],round=rnd,unique_donors=int(row['unique_donors']),
                optimizer_steps=len(steps),fixed_objective_descending_steps=sum(x<0 for x in delta),
                fixed_objective_gain=float(trace[0]['fixed_pool_objective'])-float(trace[-1]['fixed_pool_objective']),
                fixed_la_gain=float(trace[0]['fixed_pool_la'])-float(trace[-1]['fixed_pool_la']),
                same_batch_objective_increase_steps=sum(float(x['objective_change_same_batch'])>0 for x in steps if x.get('objective_change_same_batch')),
                mean_tail_la_gain=float(row['tail_la_gain_after_B_aggregation']),
                during_A_tail_la_change=float(row['tail_la_change_during_A']),
                receiver_events=len(receiver),receiver_tail_la_improved=sum(x>1e-10 for x in gain),
                receiver_tail_la_harmed=sum(x< -1e-10 for x in gain),max_norm_relative_error=max(norm_error,default=0.),
                ordinary_tail_la=float(row['ordinary_global_B_tail_la']),
                effective_norm=float(row['effective_global_transfer_norm'])))
        upload=sum(float(x['extra_upload_bytes']) for x in b_rows)
        downlink=sum(float(x['extra_downlink_bytes']) for x in b_rows)
        costs.append(dict(seed=job['seed'],arm=job['arm'],
            calibration_parameters=max(max(int(x.get('learned_c_parameters') or 0),int(x.get('learned_direct_parameters') or 0)) for x in b_rows),
            upload_MiB=upload/2**20,downlink_MiB=downlink/2**20,total_MiB=(upload+downlink)/2**20,
            algorithm_forward_images=sum(float(x['algorithm_forward_images']) for x in b_rows),
            algorithm_backward_images=sum(float(x['algorithm_backward_images']) for x in b_rows),
            diagnostic_forward_images=sum(float(x['diagnostic_forward_images']) for x in b_rows),
            event_seconds_including_diagnostics=sum(float(x['seconds']) for x in b_rows),
            unique_tail_feedback_images=len(tail_ids_seen),unique_nontail_feedback_images=len(nontail_ids_seen)))
    for seed,by_arm in sorted(blocks.items()):
        if 'ab' in by_arm and 'a-calibration' in by_arm:
            control_pair_check(root,by_arm['ab'],by_arm['a-calibration'])
        for left,right in CONTRASTS:
            if left not in by_arm or right not in by_arm:
                continue
            a,b=results[seed,left],results[seed,right]
            assert a[1]==b[1],(seed,left,right,'paired fingerprints')
            assert a[2] is None or b[2] is None or a[2]==b[2],(seed,left,right,'witness')
            if {left,right}<=set(('a','ab','a-calibration')):
                assert all(close(x[m],y[m]) for x,y in zip(a[4][:30],b[4][:30]) for m in METRICS)
            d=dict(seed=seed,comparison=left+' minus '+right,cohort=a[0]['cohort'])
            d.update({'delta_'+m:a[0]['last20_'+m]-b[0]['last20_'+m] for m in METRICS})
            d['delta_final_tail']=a[0]['final_bottom20_tail_acc']-b[0]['final_bottom20_tail_acc']
            d['delta_tail_peak_to_final']=a[0]['tail_peak_to_final']-b[0]['tail_peak_to_final']
            pairs.append(d)
            tail=set(load_json(root/by_arm[left]['run']/'class_prior.json')['tail_ids'])
            ds={c:a[3][c]-b[3][c] for c in range(100)}
            pair_classes.extend(dict(seed=seed,comparison=d['comparison'],class_id=c,is_tail=c in tail,delta_last20_acc=v) for c,v in ds.items())
            coverage.append(dict(seed=seed,comparison=d['comparison'],tail_improved=sum(ds[c]>1e-9 for c in tail),
                tail_unchanged=sum(abs(ds[c])<=1e-9 for c in tail),tail_harmed=sum(ds[c]< -1e-9 for c in tail),
                worst_tail_delta=min(ds[c] for c in tail),best_tail_delta=max(ds[c] for c in tail)))
    summary=dict(input_root=str(root),seeds_present=sorted(blocks),planned_runs=len(jobs),verified_runs=len(runs),
        verified_pairs=len(pairs),per_class_metric_recomputations=checks,training_files_compared=len(local_hashes),
        local_source_differences=source_differences,requested_seed0_present=0 in blocks,
        note='seed42 is development evidence; other seeds are confirmation; no rounds/classes treated as seed replicates')
    assert all(hashlib.sha256(p.read_bytes()).hexdigest()==v for p,v in snapshot.items())
    out.mkdir(parents=True,exist_ok=True)
    for name,rows in [('audit',audits),('per_run',runs),('paired',pairs),('per_class',classes),
        ('paired_per_class',pair_classes),('tail_coverage',coverage),('transfer_events',events),('transfer_costs',costs)]:
        write_csv(out/(name+'.csv'),rows)
    (out/'audit_summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    print(json.dumps(summary,indent=2,ensure_ascii=False))
    return summary


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-root',type=Path,default=Path('output/ab_decision_42_0_3407_analysis/ab_decision_42_0_3407'))
    parser.add_argument('--output-root',type=Path,default=Path('output/ab_decision_review_20261003'))
    args=parser.parse_args()
    analyze(args.input_root,args.output_root)
