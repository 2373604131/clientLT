"""Frozen AB contracts, independent paired comparisons, and auditable summaries."""
import hashlib
import math
import statistics
from collections import defaultdict
from pathlib import Path

from tools.sfra.maintext import FROZEN, PAIR_HASHES, TRAINING_FILES, digest, load_json
from tools.sfra.calibration_reference import ROUNDS
from tools.sfra.summary import METRICS, read_csv, write_csv

SCHEMA = 'frozen_ab_validation_v1'
PLAN_NAME = 'ab_validation_plan.json'
ARMS = {'s':'s','a':'full-cp','ab':'full-cp','a-calibration':'full-cp',
        'a-flat':'flat-cp','a-current':'current-cp','a-no-cp':'full'}
CONTRASTS = [('ab','a'),('a','s'),('ab','s'),('ab','a-calibration'),
             ('a-calibration','a'),('a','a-flat'),('a','a-current'),('a','a-no-cp')]
CONTRACT = dict(A=FROZEN,mu=1.,B=dict(non_tail_sampling='class-cyclic',tail_weight=.35,
    learning_rate=.3,regularization=.001,probe_step=.1,steps=2,rounds=list(ROUNDS)),
    execution=dict(version=2,feedback_batch_size=128,cache_gib=4.,precision='fp32'),
    primary_endpoint='committed rounds81..100 fixed Tail20 mean',
    calibration='direct residual; two Adam steps; paired AB training norm after each step; no C penalty',
    development_seed=42,dataset='cifar100_LT',protocol='30 clients, full participation, rank4, 100 rounds')
EXTRA_FILES = ('utils/cliplora_b_transfer.py','utils/cliplora_b_shared_transfer.py','utils/cliplora_b_calibration.py',
    'utils/b_aggregation.py','utils/sfra_execution.py','utils/sfra_fast_feedback.py','utils/sfra_resident_feedback.py',
    'utils/cliplora_bridge_audit.py','utils/cliplora_functional_feedback.py','utils/pfrf.py',
    'datasets/cifar100_LT.py','Dassl/dassl/data/data_manager.py','Dassl/dassl/engine/trainer.py',
    'tools/sfra/maintext.py','tools/sfra/summary.py','tools/sfra/calibration_reference.py',
    'tools/sfra/ab_validation.py','scripts/run_ab_validation.py')


def code_hashes(repo):
    return {name:hashlib.sha256((Path(repo)/name).read_text(encoding='utf-8').encode()).hexdigest()
            for name in dict.fromkeys(TRAINING_FILES+EXTRA_FILES)}


def safe_run(root, relative):
    root=Path(root).resolve();path=(root/relative).resolve()
    if path==root or not path.is_relative_to(root):
        raise ValueError('Run must stay inside the validation root')
    return path


def validate_config(run, job):
    cfg=load_json(run/'sfra_config.json')
    expected={**FROZEN,'variant':ARMS[job['arm']],'seed':job['seed'],
              'protocol_seed':job['protocol_seed'],'partition':job['partition'],'witness_batch_size':8}
    if ARMS[job['arm']].endswith('-cp'):
        expected['classification_weight']=1.
    if any(cfg.get(k)!=v for k,v in expected.items()):
        raise ValueError('Frozen A configuration differs: '+', '.join(k for k,v in expected.items() if cfg.get(k)!=v))
    base=cfg['base_training_config']
    if cfg.get('b_aggregation'):
        raise ValueError('Frozen suite requires ordinary sample-weighted B aggregation')
    if base['dirichlet_beta']!=job['dirichlet_beta'] or base['seed']!=job['seed'] or base['protocol_seed']!=job['protocol_seed']:
        raise ValueError('Base training protocol differs')
    ex=load_json(run/'execution_config.json')
    if ex.get('version')!=2 or ex.get('feedback_forward_batch_size')!=128 or ex.get('device_cache_gib')!=4.:
        raise ValueError('All formal runs require identical fast-v2 f128 c4 execution')
    transfer=cfg.get('b_transfer')
    if job['arm'] in ('ab','a-calibration'):
        if not transfer or transfer.get('mode')!='shared':
            raise ValueError('Missing shared transfer/calibration configuration')
        expected_b=dict(CONTRACT['B'])
        expected_b['calibration_profile']='coverage_tradeoff' if job['arm']=='ab' else 'direct_norm_matched'
        if job['arm']=='a-calibration':
            expected_b['regularization']=0.
        if any(transfer.get(k)!=v for k,v in expected_b.items()):
            raise ValueError('Frozen B/control settings differ')
        if job['arm']=='a-calibration':
            if digest(transfer['norm_reference'])!=transfer['norm_reference_sha256']:
                raise ValueError('Calibration reference digest mismatch')
    elif transfer:
        raise ValueError('Unexpected B in A-only/baseline arm')
    return cfg


def check_receipt(run,job):
    receipt=load_json(run/'ab_validation_run.json')
    if receipt.get('schema_version')!=SCHEMA or receipt.get('job')!=job or not receipt.get('code_unchanged'):
        raise ValueError('Run source/registration differs from frozen job')


def finite(value):
    x=float(value)
    if not math.isfinite(x):
        raise ValueError('Nonfinite result')
    return x


def read_result(run,job):
    cfg=validate_config(run,job);check_receipt(run,job)
    done=load_json(run/'completion.json')
    if done.get('completed_round')!=100 or done.get('official_test_passes')!=101 or done.get('variant')!=ARMS[job['arm']]:
        raise ValueError('Expected 100 completed rounds and 101 official evaluations')
    for actual,expected in [('normal_optimizer_steps','normal_steps_expected'),('extra_optimizer_steps','extra_steps_expected')]:
        if done[actual]!=cfg['base_training_config'][expected]:
            raise ValueError('Local training budget differs')
    curve=sorted(read_csv(run/'round_metrics.csv'),key=lambda x:int(x['round']))
    if [int(x['round']) for x in curve]!=list(range(101)):
        raise ValueError('Missing/duplicate committed rounds')
    for row in curve:
        if any(not 0<=finite(row[m])<=100 for m in METRICS):
            raise ValueError('Accuracy outside [0,100]')
    result={k:job[k] for k in ['run','arm','seed','partition','protocol_seed','dirichlet_beta']}
    result['cohort']='development' if job['seed']==42 else 'confirmation'
    for metric in METRICS:
        result['last20_'+metric]=statistics.mean(float(x[metric]) for x in curve[81:101])
        result['final_'+metric]=float(curve[100][metric])
    tail=[float(x['bottom20_tail_acc']) for x in curve]
    result.update(tail_change_90_to_100=tail[100]-tail[90],tail_peak_to_final=max(tail[1:])-tail[100],
                  elapsed_seconds=finite(done['elapsed_seconds']))
    cls=defaultdict(list)
    for epoch in range(80,100):
        cr=read_csv(run/f'per_class_accuracy_epoch_{epoch}.csv')
        if len(cr)!=100 or {int(x['class_id']) for x in cr}!=set(range(100)):
            raise ValueError('Missing/duplicate per-class results')
        for x in cr:
            value=finite(x['per_class_acc'])
            if not 0<=value<=100:
                raise ValueError('Per-class accuracy outside [0,100]')
            cls[int(x['class_id'])].append(value)
    prior=load_json(run/'class_prior.json')
    tail_ids=prior['tail_ids']
    if len(tail_ids)!=20 or len(set(tail_ids))!=20:
        raise ValueError('Invalid fixed Tail20 definition')
    if not math.isclose(statistics.mean(statistics.mean(cls[c]) for c in tail_ids),result['last20_bottom20_tail_acc'],abs_tol=1e-8):
        raise ValueError('Tail20 table differs from per-class results')
    transfer_rows=[]
    if job['arm'] in ('ab','a-calibration'):
        transfer_rows=read_csv(run/'b_transfer_rounds.csv')
        if sorted(int(x['round']) for x in transfer_rows)!=list(ROUNDS) or done.get('b_transfer_events')!=8:
            raise ValueError('Transfer event budget differs')
        total_steps=0
        if job['arm']=='a-calibration' and load_json(run/'calibration_reference.json')!=cfg['b_transfer']['norm_reference']:
            raise ValueError('Control reference file differs from recorded configuration')
        for rnd in ROUNDS:
            manifest=load_json(run/f'b_transfer_rounds/r{rnd:03d}/calibration_manifest.json')
            steps=read_csv(run/f'b_transfer_rounds/r{rnd:03d}/optimization_steps.csv')
            active=bool(manifest['shared_donor_ids']) if job['arm']=='ab' else bool(cfg['b_transfer']['norm_reference']['events'][str(rnd)]['norms'])
            if [int(x['step']) for x in steps]!=([1,2] if active else []):
                raise ValueError('Transfer step budget differs')
            total_steps+=len(steps)
            if job['arm']=='a-calibration':
                norms=cfg['b_transfer']['norm_reference']['events'][str(rnd)]['norms']
                for x,target in zip(steps,norms):
                    if not math.isclose(finite(x['effective_transfer_norm_after']),target,rel_tol=1e-5,abs_tol=1e-10):
                        raise ValueError('Control effective update norm differs from AB reference')
        if done.get('b_transfer_optimizer_steps')!=total_steps:
            raise ValueError('Transfer completion budget differs from recorded steps')
    result['B_optimizer_steps']=sum(len(read_csv(run/f'b_transfer_rounds/r{rnd:03d}/optimization_steps.csv')) for rnd in ROUNDS) if transfer_rows else 0
    for key in ['algorithm_forward_images','algorithm_backward_images','diagnostic_forward_images','extra_upload_bytes','extra_downlink_bytes','seconds']:
        result['B_'+key]=sum(finite(x.get(key) or 0) for x in transfer_rows)
    budget=read_csv(run/'budget.csv');costs=read_csv(run/'sfra_costs.csv')
    for label,items,key in [('local_upload_bytes',budget,'upload_bytes'),('local_downlink_bytes',budget,'modeled_downlink_bytes'),
        ('A_upload_bytes',costs,'upload_bytes'),('A_downlink_bytes',costs,'downlink_bytes'),
        ('A_forward_images',costs,'forward_images'),('A_backward_images',costs,'backward_images')]:
        result[label]=sum(finite(x.get(key) or 0) for x in items)
    meta=load_json(run/'bridge_metadata.json')
    signature={k:meta[k] for k in PAIR_HASHES}
    signature.update(partition=digest(read_csv(run/'partition_manifest.csv')),execution=digest(load_json(run/'execution_config.json')),
        base_training=digest(cfg['base_training_config']),code=digest(job['code_sha256']),tail_ids=digest(tail_ids))
    witness=digest(load_json(run/'private_witness_manifest.json')) if job['arm']!='s' else None
    return result,signature,witness,{c:statistics.mean(v) for c,v in cls.items()},curve


def control_pair_check(root,ab,control):
    """Actual batches and reference fingerprint, not just CLI promises."""
    from tools.sfra.calibration_reference import build_reference
    expected=build_reference(root/ab['run'])
    saved=load_json(root/control['run']/'calibration_reference.json')
    if saved!=expected:
        raise ValueError('Control borrowed norms from a different AB run')
    for rnd in ROUNDS:
        event=load_json(root/control['run']/f'b_transfer_rounds/r{rnd:03d}/calibration_manifest.json')
        if event['feedback_clients']!=expected['events'][str(rnd)]['feedback_clients'] or {k:v['batches'] for k,v in event['recipients'].items()}!=expected['events'][str(rnd)]['batches']:
            raise ValueError('Control calibration batches differ from AB')


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
        curve_rows.extend(dict(run=key,arm=r['arm'],seed=r['seed'],partition=r['partition'],**x) for x in curve)
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
    lines=['# Frozen AB validation','', 'Primary endpoint: fixed Tail20, committed rounds 81–100 mean.',
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
        lines.append(f'| {r["comparison"]} | {r["partition"]} | {r["cohort"]} | {r["n"]} | {r["delta_last20_bottom20_tail_acc_mean"]:+.4f} | '+(f'{sd:.4f}' if sd!='' else '—')+' |')
    lines+=['','A-calibration uses AB training-side step norms, no checkpoints/test selection; it is a reference-dependent diagnostic.',
        'Same calibration images, LA weights, steps and effective norm do not imply equal parameter count, optimizer geometry, total compute or communication.',
        'AB minus calibration probes added value over this control, not optimal screening or all possible extra-training baselines.',
        'AB minus A is conditional benefit; no special A/B interaction is established without the corresponding factorial evidence.',
        'Standard Dirichlet is a separate partition protocol, not a fixed-marginal concentration intervention.',
        'Ordinary/A/B costs are listed separately in per_run.csv; network bytes are modeled, elapsed time is serial simulation.',
        'No winner selection, significance declaration, or automatic method redesign is performed.']
    (out/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return status,audits
