"""Offline AB summaries compatible with the frozen v1 training source hashes.

The v1 training launcher fingerprints tools/sfra/ab_validation.py itself.
Keep that file unchanged while existing studies are running or resumable.
This separate report builder reuses its validation rules and statistics; the
curve export merges CSV metadata without passing duplicate dict keywords.
No training, checkpoint loading, or provenance rewriting is performed here.
"""
import argparse
from collections import defaultdict
from pathlib import Path
import statistics
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from tools.sfra.ab_validation import (
    SCHEMA, CONTRACT, PLAN_NAME, CONTRASTS, load_json, safe_run,
    read_result, control_pair_check,
)
from tools.sfra.summary import METRICS, write_csv, pack


def summarize(root):
    root=Path(root).resolve();plan=load_json(root/PLAN_NAME)
    if plan.get('schema_version')!=SCHEMA or plan.get('contract')!=CONTRACT:
        raise ValueError('Different AB validation contract')
    out=root/'analysis';out.mkdir(parents=True,exist_ok=True)
    status=[];results={};blocks=defaultdict(dict);class_rows=[];curve_rows=[]
    for job in plan['jobs']:
        path=safe_run(root,job['run']);arm=job['arm']
        key=(job['partition'],job['protocol_seed'],job['dirichlet_beta'],job['seed'])
        if arm in blocks[key]:
            raise ValueError('Duplicate arm in paired block')
        blocks[key][arm]=job
        row=dict(run=job['run'],arm=arm,seed=job['seed'],partition=job['partition'],status='missing',reason='')
        if path.is_dir():
            row['status']='incomplete'
            if (path/'completion.json').is_file():
                try:
                    results[job['run']]=read_result(path,job)
                    row['status']='complete'
                except (OSError,ValueError,KeyError,TypeError) as error:
                    row.update(status='invalid',reason=str(error))
        status.append(row)
    # An invalid control must be excluded from EVERY comparison, including control-A.
    for row in status:
        if row['arm']!='a-calibration' or row['run'] not in results:
            continue
        job=next(j for j in plan['jobs'] if j['run']==row['run'])
        try:
            if job['dependency'] not in results:
                row.update(status='unverified',reason='Paired AB is not a verified complete result')
            else:
                ab=next(j for j in plan['jobs'] if j['run']==job['dependency'])
                control_pair_check(root,ab,job)
        except (OSError,ValueError,KeyError,TypeError) as error:
            row.update(status='invalid',reason=str(error))
        if row['status']!='complete':
            results.pop(row['run'])
    paired=[];audits=[];paired_classes=[]
    for block,jobs in sorted(blocks.items()):
        for left,right in CONTRASTS:
            if left not in jobs or right not in jobs:
                continue
            jl,jr=jobs[left],jobs[right];kl,kr=jl['run'],jr['run']
            audit=dict(partition=block[0],protocol_seed=block[1],dirichlet_beta=block[2],seed=block[3],comparison=left+' minus '+right,eligible=False,reason='pending')
            if kl in results and kr in results:
                a,sa,wa,_,ca=results[kl];b,sb,wb,_,cb=results[kr]
                mismatches=[k for k in sa if sa[k]!=sb[k]]
                if wa is not None and wb is not None and wa!=wb:
                    mismatches.append('witness')
                if 's' not in (left,right) and left not in ('a-flat','a-current','a-no-cp') and right not in ('a-flat','a-current','a-no-cp'):
                    if any(abs(float(x[m])-float(y[m]))>1e-8 for x,y in zip(ca[:30],cb[:30]) for m in METRICS):
                        mismatches.append('pre-transfer test trajectory')
                if (left,right)==('ab','a-calibration'):
                    try: control_pair_check(root,jl,jr)
                    except (OSError,ValueError,KeyError) as error: mismatches.append(str(error))
                audit.update(eligible=not mismatches,reason='; '.join(mismatches))
                if not mismatches:
                    row={k:v for k,v in audit.items() if k not in ('eligible','reason')}
                    row['cohort']=a['cohort']
                    for m in METRICS:
                        row['delta_last20_'+m]=a['last20_'+m]-b['last20_'+m]
                    row['delta_tail_peak_to_final']=a['tail_peak_to_final']-b['tail_peak_to_final']
                    paired.append(row)
                    tail_ids=set(load_json(root/kl/'class_prior.json')['tail_ids'])
                    for c in sorted(results[kl][3]):
                        paired_classes.append(dict(partition=block[0],protocol_seed=block[1],dirichlet_beta=block[2],
                            seed=block[3],cohort=a['cohort'],comparison=left+' minus '+right,class_id=c,
                            is_tail=c in tail_ids,delta_last20_acc=results[kl][3][c]-results[kr][3][c]))
            audits.append(audit)
    records=[r[0] for r in results.values()]
    for key,(r,s,w,classes,curve) in results.items():
        class_rows.extend(dict(run=key,arm=r['arm'],seed=r['seed'],partition=r['partition'],class_id=c,last20_acc=v) for c,v in classes.items())
        curve_rows.extend({**x, 'run':key, 'arm':r['arm'], 'seed':r['seed'], 'partition':r['partition']} for x in curve)
    aggregates=[]
    groups=defaultdict(list)
    for r in paired:
        groups[(r['partition'],r['protocol_seed'],r['dirichlet_beta'],r['comparison'],'all')].append(r)
        groups[(r['partition'],r['protocol_seed'],r['dirichlet_beta'],r['comparison'],r['cohort'])].append(r)
    for (partition,protocol,beta,comparison,cohort),values in sorted(groups.items()):
        row=dict(partition=partition,protocol_seed=protocol,dirichlet_beta=beta,comparison=comparison,cohort=cohort,
                 n=len(values),seeds=','.join(str(x['seed']) for x in values))
        for metric in [k for k in values[0] if k.startswith('delta_')]:
            v=[x[metric] for x in values];row[metric+'_mean']=statistics.mean(v)
            row[metric+'_sd']=statistics.stdev(v) if len(v)>1 else ''
        aggregates.append(row)
    methods=[];groups=defaultdict(list)
    for r in records:
        for cohort in ('all',r['cohort']):
            groups[(r['partition'],r['protocol_seed'],r['dirichlet_beta'],r['arm'],cohort)].append(r)
    for (partition,protocol,beta,arm,cohort),values in sorted(groups.items()):
        row=dict(partition=partition,protocol_seed=protocol,dirichlet_beta=beta,arm=arm,cohort=cohort,
                 n=len(values),seeds=','.join(str(x['seed']) for x in values))
        for metric in [k for k,v in values[0].items() if isinstance(v,(int,float)) and k not in ('seed','protocol_seed','dirichlet_beta')]:
            numbers=[x[metric] for x in values];row[metric+'_mean']=statistics.mean(numbers)
            row[metric+'_sd']=statistics.stdev(numbers) if len(numbers)>1 else ''
        methods.append(row)
    for name,data in [('status',status),('per_run',records),('paired_per_seed',paired),('paired_summary',aggregates),
                      ('method_summary',methods),('pair_audit',audits),('per_class',class_rows),
                      ('paired_per_class',paired_classes),('curves',curve_rows)]:
        write_csv(out/(name+'.csv'),data)
    lines=['# Frozen AB validation','', 'Primary endpoint: fixed Tail20, committed rounds 81�C100 mean.',
        'Seed42 is development evidence; confirmation-seed differences are also reported separately.',
        'Each comparison uses its own complete, fingerprint-matched pairs; missing unrelated arms never block it.',
        'method_summary.csv is descriptive; use paired_summary.csv for comparisons with matching seed sets.',
        'Incomplete/mismatched runs are not silently treated as zero or included in paired averages.', '',
        '| Partition | Seed | Arm | Tail20 | Overall |','|---|---:|---|---:|---:|']
    for r in sorted(records,key=lambda x:(x['partition'],x['seed'],x['arm'])):
        lines.append(f'| {r["partition"]} | {r["seed"]} | {r["arm"]} | {r["last20_bottom20_tail_acc"]:.4f} | {r["last20_overall_acc"]:.4f} |')
    lines+=['','| Comparison | Partition | Cohort | n | Tail20 delta (pp) | SD |','|---|---|---|---:|---:|---:|']
    for r in aggregates:
        sd=r['delta_last20_bottom20_tail_acc_sd']
        lines.append(f'| {r["comparison"]} | {r["partition"]} | {r["cohort"]} | {r["n"]} | {r["delta_last20_bottom20_tail_acc_mean"]:+.4f} | '+(f'{sd:.4f}' if sd!='' else '��')+' |')
    lines+=['','A-calibration uses AB training-side step norms, no checkpoints/test selection; it is a reference-dependent diagnostic.',
        'Same calibration images, LA weights, steps and effective norm do not imply equal parameter count, optimizer geometry, total compute or communication.',
        'AB minus calibration probes added value over this control, not optimal screening or all possible extra-training baselines.',
        'AB minus A is conditional benefit; no special A/B interaction is established without the corresponding factorial evidence.',
        'Standard Dirichlet is a separate partition protocol, not a fixed-marginal concentration intervention.',
        'Ordinary/A/B costs are listed separately in per_run.csv; network bytes are modeled, elapsed time is serial simulation.',
        'No winner selection, significance declaration, or automatic method redesign is performed.']
    (out/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return status,audits


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=['summary', 'pack'], default='pack')
    parser.add_argument('--output-root', type=Path, default=Path('output/cifar100_LT/ab_validation'))
    args = parser.parse_args()
    root = args.output_root.resolve()
    try:
        status, audits = summarize(root)
        if args.stage == 'pack':
            pack(root)
        print('Report:', root/'analysis/report.md', flush=True)
        if any(x['status'] == 'invalid' for x in status) or any(x['reason'] not in ('', 'pending') for x in audits):
            raise ValueError('Invalid/mismatched results excluded; see analysis/status.csv and pair_audit.csv')
    except (OSError, ValueError, KeyError) as error:
        parser.error(str(error))


if __name__ == '__main__':
    main()

