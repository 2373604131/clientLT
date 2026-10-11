"""One frozen floor16 candidate, an audited baseline snapshot and an integer gate."""
import hashlib
from pathlib import Path
import shutil
from statistics import mean

import numpy as np

from scripts.run_ab_validation import write_json
from tools.client_aggregation.analysis import (METRICS, audit_run, prediction,
    metrics_from_prediction, sample_dynamics, stage_comparisons)
from tools.client_aggregation.frozen_tailrw import paired_audit
from tools.client_aggregation.method_a import INTEGRATION_FILES as A_INTEGRATION_FILES
from tools.client_aggregation.plain_a import load_plain
from tools.client_aggregation.protocol import load_json, digest, write_table, check_aggregation

REFERENCE = 'references/joint_floor16_seed42/aggregation_spec.json'
RULE = 'joint_floor16'
GATE = dict(baseline_overall_correct=141730, baseline_tail_correct=28683,
    overall_predictions=200000, tail_predictions=40000,
    min_overall_gain_correct=200, min_tail_gain_correct=120, rounds=list(range(81,101)))
EXPERIMENT = dict(name='joint_floor16_plain_a_seed42_v1', rule=RULE, arm='plain_a',
    seed=42, full_runs=1, client_concurrency=8, aggregation_lambda=.1,
    floor_source='N/(N+16*N_tail), all training clients', weights='frozen offline FP64 solution',
    gate=GATE, stop_rule='pass: adopt joint_floor16; otherwise: retain joint; stop aggregation changes')
INTEGRATION_FILES = A_INTEGRATION_FILES | {
    'tools/client_aggregation/weights.py', 'tools/client_aggregation/floor16.py',
    'scripts/run_cliplora_joint_floor.py', REFERENCE,
}


def endpoint(result):
    counts = {key:0 for key in METRICS}
    totals = {key:0 for key in METRICS}
    per_round = []
    for rnd in GATE['rounds']:
        sample = result['predictions'][rnd]
        row = dict(round=rnd)
        for key, ids in result['groups'].items():
            mask = np.isin(sample['class_id'],ids)
            row[key] = int(sample['correct'][mask].sum())
            counts[key] += row[key]
            totals[key] += int(mask.sum())
        per_round.append(row)
    if totals['overall_acc'] != 200000 or totals['bottom20_tail_acc'] != 40000:
        raise ValueError('Unexpected test prediction denominators')
    return dict(correct=counts, predictions=totals, per_round_correct=per_round,
        accuracy={k:100*counts[k]/totals[k] for k in counts})


def integer_decision(overall_correct, tail_correct):
    if (type(overall_correct) is not int or type(tail_correct) is not int
            or not 0 <= tail_correct <= 40000 or not tail_correct <= overall_correct <= 200000):
        raise ValueError('Gate inputs must be raw integer correct counts')
    checks = dict(overall_non_decreasing=overall_correct>=141730,
        tail_non_decreasing=tail_correct>=28683,
        overall_gain_at_least_0_1pp=overall_correct>=141930,
        tail_gain_at_least_0_3pp=tail_correct>=28803)
    passed = (checks['overall_non_decreasing'] and checks['tail_non_decreasing'] and
              (checks['overall_gain_at_least_0_1pp'] or checks['tail_gain_at_least_0_3pp']))
    return dict(checks=checks, passed=passed, decision='adopt_joint_floor16' if passed else 'retain_original_joint',
        aggregation_revision_finished=True, statistical_significance=False)


def verify_baseline(result):
    if result['job']['arm'] != 'plain_a' or result['job']['aggregation'].get('rule','joint') != 'joint':
        raise ValueError('Baseline must be the completed original joint + ordinary A')
    result_endpoint = endpoint(result)
    if (result_endpoint['correct']['overall_acc'] != GATE['baseline_overall_correct']
            or result_endpoint['correct']['bottom20_tail_acc'] != GATE['baseline_tail_correct']):
        raise ValueError('Raw baseline predictions do not match the preregistered integer gate')
    return result_endpoint


def snapshot_baseline(root, source):
    """Audit once, then preserve raw predictions so a results pack is self-contained."""
    result = load_plain(source,'joint')
    baseline_endpoint = verify_baseline(result)
    folder = Path(root)/'baseline'
    files = {}
    for rnd in range(101):
        src = result['run']/'predictions'/f'r{rnd:03d}.npz'
        dest = folder/'predictions'/src.name
        dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(src,dest)
        files[src.name] = hashlib.sha256(dest.read_bytes()).hexdigest()
    record = {k:result[k] for k in ('job','config','progress','metadata','groups','curves')}
    record.update(endpoint=baseline_endpoint, prediction_sha256=files, source=str(Path(source).resolve()))
    write_json(folder/'record.json',record)
    return record


def read_baseline(root):
    folder = Path(root)/'baseline'
    record = load_json(folder/'record.json')
    samples = {}
    for rnd in range(101):
        path = folder/'predictions'/f'r{rnd:03d}.npz'
        if hashlib.sha256(path.read_bytes()).hexdigest() != record['prediction_sha256'][path.name]:
            raise ValueError('Baseline raw prediction snapshot changed')
        samples[rnd] = prediction(path)
        if not np.array_equal(samples[rnd]['class_id'],samples[0]['class_id']):
            raise ValueError('Baseline sample ordering changed')
    result = dict(record,predictions=samples)
    if verify_baseline(result) != record['endpoint']:
        raise ValueError('Baseline endpoint snapshot changed')
    return result


def comparison_baseline(root, plan, source=None):
    """Attach results for comparison without changing the training registration."""
    root = Path(root)
    record_path = root/'baseline/record.json'
    identity_path = root/'baseline/identity.json'
    expected = plan.get('baseline_record_sha256')  # Support previously registered snapshots.
    if identity_path.is_file():
        attached = load_json(identity_path)['record_sha256']
        if expected is not None and expected != attached:
            raise ValueError('Registered comparison baseline changed')
        expected = attached
    if expected is not None:
        if digest(load_json(record_path)) != expected:
            raise ValueError('Registered comparison baseline changed')
        result = read_baseline(root)
    else:
        if source is None or not (Path(source)/'experiment_plan.json').is_file():
            return None
        source = Path(source).resolve(); target = root.resolve()
        if source == target or source in target.parents or target in source.parents:
            raise ValueError('Use an independent baseline directory outside the new experiment')
        # Re-audit any incomplete/unbound snapshot using the original artifacts.
        record = snapshot_baseline(root,source)
        result = read_baseline(root)
        if result['job']['aggregation']['weights'] != plan['aggregation']['solver']['joint_reference_weights']:
            raise ValueError('Original joint weight identity differs from the registered reference')
        write_json(identity_path,dict(record_sha256=digest(record)))
    if result['job']['aggregation']['weights'] != plan['aggregation']['solver']['joint_reference_weights']:
        raise ValueError('Original joint weight identity differs from the registered reference')
    return result


def comparison_audit(result, baseline, smoke=False):
    pair = paired_audit(result,baseline)
    allowed = {'code.'+k for k in INTEGRATION_FILES}
    if smoke:
        allowed |= {'budget.normal_optimizer_steps','budget.extra_optimizer_steps'}
    pair['mismatches'] = [x for x in pair['mismatches'] if x not in allowed]
    for name in pair['source_differences']:
        if name not in INTEGRATION_FILES and 'code.'+name not in pair['mismatches']:
            pair['mismatches'].append('code.'+name)
    if not pair['same_recorded_environment']:
        pair['mismatches'].append('recorded_environment')
    pair['status'] = 'mismatch' if pair['mismatches'] else 'eligible'
    pair['contrast'] = 'joint_floor16 minus original joint; ordinary A; one seed'
    pair['smoke_only'] = smoke
    pair['source_change_note'] = 'Only listed aggregation/integration changes permitted; legacy paths covered by regression checks'
    return pair


def status(root):
    run = Path(root)/'runs/seed42/plain_a'
    path = run/('completion.json' if (run/'completion.json').is_file() else 'progress.json')
    progress = load_json(path) if path.is_file() else {}
    return dict(rule=RULE, status='completed_unverified' if path.name=='completion.json' else
        'incomplete' if progress else 'not_started', completed_round=progress.get('completed_round',0),
        B_steps=progress.get('normal_optimizer_steps',0), A_steps=progress.get('extra_optimizer_steps',0))


def audit_smoke(root, plan):
    from scripts.run_cliplora_joint_aggregation import make_job
    job = make_job(plan,'plain_a','smoke')
    result = audit_run(Path(root)/job['run'],job,1)
    if result['config']['client_execution']['max_concurrent_clients'] != 8:
        raise ValueError('Smoke must use the registered eight-client execution')
    write_json(Path(root)/'smoke_audit.json',dict(comparison=dict(status='deferred_to_summary'),
        weights_sha256=job['aggregation']['weights_sha256'],
        completed_round=1, B_steps=result['progress']['normal_optimizer_steps'],
        A_steps=result['progress']['extra_optimizer_steps'],
        checks=['exact offline weights by client ID in A and B','all 30 clients, 8 slots',
                'ordinary A; no correction or C']))
    return result


def write_result_tables(out, result, current_endpoint, baseline=None):
    entries = [(RULE,result,current_endpoint)]
    if baseline is not None:
        entries.append(('joint',baseline,verify_baseline(baseline)))
    performance,curves,dynamics,per_class = [],[],[],[]
    for label,r,ep in entries:
        tails = [float(x['bottom20_tail_acc']) for x in r['curves']]
        performance.append(dict(method=label,**{'last20_'+m:ep['accuracy'][m] for m in METRICS},
            final_tail=tails[-1],tail_peak_to_final=max(tails)-tails[-1],tail_change_90_to_100=tails[-1]-tails[90]))
        curves.extend(dict(row,method=label) for row in r['curves'])
        dynamics.extend(dict(row,method=label) for row in sample_dynamics(r))
        per_class.extend(dict(method=label,class_id=c,last20_accuracy=mean(
            int(r['predictions'][t]['correct'][r['predictions'][t]['class_id']==c].sum()) for t in GATE['rounds'])) for c in range(100))
    costs = [dict(method=label,**{k:r['progress'].get(k,0) for k in
        ('normal_optimizer_steps','extra_optimizer_steps','functional_correction_steps',
         'b_transfer_optimizer_steps','elapsed_seconds','stage_diagnostic_seconds')})
        for label,r,_ in entries]
    for name,rows in dict(performance=performance,curves=curves,sample_dynamics=dynamics,
            per_class=per_class,costs=costs,stage_effects=stage_comparisons(result),
            endpoint_correct=current_endpoint['per_round_correct']).items():
        write_table(out/(name+'.csv'),rows)


def summarize(root, baseline_root=None):
    from scripts.run_cliplora_joint_aggregation import make_job
    root = Path(root); out = root/'analysis'; out.mkdir(parents=True,exist_ok=True)
    state = status(root)
    decision = dict(status='pending', decision='not_evaluated', aggregation_revision_finished=False, gate=GATE)
    write_json(out/'decision.json',decision)  # Clear any stale decision before validation.
    write_json(out/'selected_aggregation.json',dict(decision='not_evaluated',aggregation=None))
    write_json(out/'pair_audit.json',dict(status='pending'))
    if state['status'] != 'completed_unverified':
        write_table(out/'status.csv',[state]); print(RULE,state['status'],'round='+str(state['completed_round']))
        return decision
    try:
        plan = load_json(root/'experiment_plan.json')
        if plan['floor16_experiment'] != EXPERIMENT:
            raise ValueError('Registered experiment or adoption gate changed')
        job = make_job(plan,'plain_a')
        result = audit_run(root/job['run'],job)
        current_endpoint = endpoint(result)
        decision['endpoint'] = current_endpoint
        write_result_tables(out,result,current_endpoint)
        state['status']='complete'
        baseline = comparison_baseline(root,plan,baseline_root)
        if baseline is None:
            decision.update(status='pending_baseline',reason='Training completed and audited. Original joint results are absent; '
                'results can be packed now. Supply --baseline-root to summary/pack later to evaluate the fixed gate.')
            write_json(out/'pair_audit.json',dict(status='pending_baseline'))
        else:
            pair = comparison_audit(result,baseline)
            write_json(out/'pair_audit.json',pair)
            if pair['status'] != 'eligible':
                raise ValueError('Full-run pairing failed: '+str(pair['mismatches']))
            base_endpoint = verify_baseline(baseline)
            decision.update(integer_decision(current_endpoint['correct']['overall_acc'],current_endpoint['correct']['bottom20_tail_acc']),
                status='evaluated', baseline_endpoint=base_endpoint,
                delta_pp={m:current_endpoint['accuracy'][m]-base_endpoint['accuracy'][m] for m in METRICS})
            write_result_tables(out,result,current_endpoint,baseline)
            write_json(out/'selected_aggregation.json',dict(decision=decision['decision'],
                aggregation=result['job']['aggregation'] if decision['passed'] else baseline['job']['aggregation'],
                note='Selection for subsequent work; legacy joint definition and existing experiments remain unchanged'))
    except (ValueError,OSError,KeyError,TypeError,AssertionError) as error:
        state.update(status='invalid',reason=str(error))
        decision.update(status='invalid',reason=str(error))
    write_table(out/'status.csv',[state]); write_json(out/'decision.json',decision)
    lines = ['joint_floor16 + ordinary A, seed42','', 'Status: '+decision['status'],
        'Decision: '+decision['decision'],
        'Gate uses raw integer correct occurrences in committed rounds81..100.',
        'No gamma search, no additional full run; this is an engineering gate, not statistical significance.']
    if decision['status']=='evaluated':
        lines += ['','Metric, joint_floor16, original_joint, delta_pp']
        lines += [f'{m}, {current_endpoint["accuracy"][m]:.4f}, {base_endpoint["accuracy"][m]:.4f}, {decision["delta_pp"][m]:+.4f}' for m in METRICS]
        lines += ['', 'Correct occurrences: '+str(current_endpoint['correct']), 'Checks: '+str(decision['checks']),
                  'Head/Middle and forgetting are reported; they do not change the fixed gate.']
    elif decision['status']=='pending_baseline':
        lines += ['',decision['reason'],'','Metric, joint_floor16']
        lines += [f'{m}, {current_endpoint["accuracy"][m]:.4f}' for m in METRICS]
        lines += ['','Correct occurrences: '+str(current_endpoint['correct'])]
    elif 'reason' in decision:
        lines.append(decision['reason'])
    (out/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('Decision:',decision['decision'],'status='+decision['status'],flush=True)
    if decision['status']=='invalid':
        raise ValueError('Invalid result/pair; no adoption decision. See analysis/decision.json')
    return decision
