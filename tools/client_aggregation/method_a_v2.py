"""Av2 row checks and optional, post-training comparisons; no accuracy gate."""
from pathlib import Path
from statistics import mean

import numpy as np

from scripts.run_ab_validation import write_json
from tools.client_aggregation.protocol import load_json, read_csv, write_table
from utils.method_a_v2_math import leq, VARIANT

METHODS = ('tailrw16','joint','joint_floor16')
ARM = 'method_a_v2'


def audit_rounds(rows, progress, completed):
    if [int(r['round']) for r in rows] != list(range(1,completed+1)):
        raise ValueError('Missing Av2 submitted round records')
    steps = 0
    for r in rows:
        rnd,n = int(r['round']),int(r['correction_steps'])
        if r['variant'] != VARIANT or n not in ((0,3) if rnd<=90 else (0,)):
            raise ValueError('Wrong Av2 method or correction budget')
        if n == 0 and r['skip_reason'] not in (
                ('no_valid_history','zero_proposal_radius','no_history_deficit') if rnd<=90 else ('B_only_schedule',)):
            raise ValueError('Unexpected Av2 correction skip')
        steps += n
        if int(r['history_last_committed']) != min(rnd,90):
            raise ValueError('Av2 history and submitted round differ')
        if rnd>90:
            continue
        if int(r['history_last_used']) != rnd-1 or int(r['window_count']) != rnd%5:
            raise ValueError('Av2 history window or next-round activation differs')
        beta = float(r['selected_beta'])
        if beta not in (0.,.5,1.) or (n==0 and beta!=0):
            raise ValueError('Unregistered Av2 candidate fraction')
        for g in ('H','M','T'):
            if not bool(leq(float(r['final_CE_'+g]),float(r['limit_CE_'+g]))):
                raise ValueError('Submitted Av2 group loss violates its bound')
        if not bool(leq(float(r['J_final']),float(r['J0']))):
            raise ValueError('Submitted Av2 objective exceeds ordinary A')
        if beta==1. and r['full_reject'] or beta==.5 and (not r['full_reject'] or r['half_reject']):
            raise ValueError('Av2 candidate order and rejection reasons differ')
        if n==3 and beta==0 and (not r['full_reject'] or not r['half_reject']):
            raise ValueError('Av2 zero fallback requires both candidates to fail')
    if steps != progress['functional_correction_steps']:
        raise ValueError('Av2 attempted correction budget differs')


def status_rows(root):
    rows=[]
    for rule in METHODS:
        run=Path(root)/rule/'runs/seed42'/ARM
        complete=(run/'completion.json').is_file()
        path=run/('completion.json' if complete else 'progress.json')
        p=load_json(path) if path.is_file() else {}
        rows.append(dict(method=rule,status='completed_unverified' if complete else 'incomplete' if p else 'not_started',
                         completed_round=p.get('completed_round',0)))
    return rows


def compare(candidate, baseline):
    from tools.client_aggregation.frozen_tailrw import paired_audit
    from tools.client_aggregation.method_a import INTEGRATION_FILES
    pair=paired_audit(candidate,baseline)
    allowed=INTEGRATION_FILES | {
        'utils/cliplora_sfra.py','federated_main.py','utils/method_a_v2_math.py',
        'utils/method_a_v2_feedback.py','utils/cliplora_method_a_v2.py',
        'tools/client_aggregation/method_a_v2.py','scripts/run_cliplora_method_a_v2_aggregation.py',
        'tools/client_aggregation/weights.py','tools/client_aggregation/floor16.py',
        'scripts/run_cliplora_joint_floor.py','references/joint_floor16_seed42/aggregation_spec.json'}
    pair['mismatches']=[m for m in pair['mismatches'] if m not in
        {'variant','budget.functional_correction_steps',*('code.'+p for p in allowed)}]
    for p in pair['source_differences']:
        if p not in allowed and 'code.'+p not in pair['mismatches']:
            pair['mismatches'].append('code.'+p)
    if not np.allclose(candidate['job']['aggregation']['weights'],baseline['job']['aggregation']['weights'],rtol=0,atol=1e-8):
        pair['mismatches'].append('aggregation.weights')
    pair['status']='mismatch' if pair['mismatches'] else 'execution_differs' if not pair['same_recorded_environment'] else 'eligible'
    pair['contrast']='source-memory-v2 minus same-aggregation ordinary A'
    pair['local_SGD_budget_matched']=not any(x.startswith('budget.') for x in pair['mismatches'])
    pair['total_compute_budget_matched']=False
    return pair


def summarize(root, plain_root, floor_plain_root):
    from tools.client_aggregation.analysis import (METRICS, audit_run, sample_dynamics, stage_comparisons)
    from tools.client_aggregation.plain_a import load_plain
    from scripts.run_cliplora_joint_aggregation import make_job
    root=Path(root); out=root/'analysis'; out.mkdir(parents=True,exist_ok=True)
    statuses=status_rows(root); performance=[]; costs=[]; comparisons=[]; results={}
    for row in statuses:
        rule=row['method']
        if row['status']!='completed_unverified':
            continue
        try:
            plan=load_json(root/rule/'experiment_plan.json'); job=make_job(plan,ARM)
            r=audit_run(root/rule/job['run'],job);results[rule]=r
            window=[x for x in r['curves'] if 81<=int(x['round'])<=100]
            ep={m:mean(float(x[m]) for x in window) for m in METRICS}
            performance.append(dict(method=rule,arm=ARM,**ep))
            costs.append(dict(method=rule,**r['progress']))
            write_table(out/(rule+'_sample_dynamics.csv'),sample_dynamics(r))
            write_table(out/(rule+'_stages.csv'),stage_comparisons(r))
            row['status']='complete'
            source=Path(floor_plain_root) if rule=='joint_floor16' else Path(plain_root)/rule
            pair=dict(method=rule,status='unavailable',baseline_root=str(source))
            if (source/'runs/seed42/plain_a/completion.json').is_file():
                try:
                    baseline=load_plain(source,rule)
                    pair.update(compare(r,baseline))
                    if pair['status']=='eligible':
                        base={m:mean(float(x[m]) for x in baseline['curves'] if 81<=int(x['round'])<=100) for m in METRICS}
                        performance.append(dict(method=rule,arm='plain_a',**base))
                        pair['delta_pp']={m:ep[m]-base[m] for m in METRICS}
                except (ValueError,OSError,KeyError) as e:
                    pair.update(status='unavailable',reason=str(e))
            comparisons.append(pair)
        except (ValueError,OSError,KeyError) as e:
            row.update(status='invalid',reason=str(e))
    # Cross-combination tradeoffs are descriptive, never single-component attribution.
    anchor=Path(plain_root)/'tailrw16'; tradeoffs=[]
    if (anchor/'runs/seed42/plain_a/completion.json').is_file():
        try:
            base=load_plain(anchor,'tailrw16')
            ep={m:mean(float(x[m]) for x in base['curves'] if 81<=int(x['round'])<=100) for m in METRICS}
            for rule,r in results.items():
                pair=compare(r,base)
                pair['mismatches']=[m for m in pair['mismatches'] if m!='aggregation.weights']
                comparable=not pair['mismatches'] and pair['same_recorded_environment']
                tradeoffs.append(dict(method=rule,anchor='tailrw16+ordinary_A',
                    status='descriptive_cross_combination' if comparable else 'mismatch',
                    **({'delta_'+m:mean(float(x[m]) for x in r['curves'] if 81<=int(x['round'])<=100)-ep[m] for m in METRICS} if comparable else {})))
        except (ValueError,OSError,KeyError):
            pass
    write_table(out/'status.csv',statuses);write_table(out/'performance.csv',performance)
    write_table(out/'costs.csv',costs);write_table(out/'vs_tailrw16_ordinary.csv',tradeoffs)
    write_json(out/'comparisons.json',dict(comparisons=comparisons,independent_seeds=1,
        endpoint='mean accuracy of submitted rounds81..100, not model averaging',
        note='No Av2 accuracy gate; missing controls are never rerun; aggregation adoption remains separate'))
    (out/'report.md').write_text('source-memory-v2, seed42\n\n'+
        '\n'.join(str(r) for r in statuses)+'\n\n'+
        '\n'.join(str(r) for r in performance)+'\n\nComparisons: '+str(comparisons)+
        '\nOne seed; twenty consecutive rounds are not independent repetitions.\n',encoding='utf-8')
    return statuses
