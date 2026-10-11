"""Audited two-arm summaries; stage diagnostics never control training."""
import json
import math
from pathlib import Path
import statistics
import tarfile

import numpy as np

from scripts.run_ab_validation import write_json
from tools.client_aggregation.protocol import (ARMS, B_ROUNDS, DIAGNOSTIC_ROUNDS, SCHEMA,
    PAIR_HASHES, audit_runtime_config, check_aggregation, digest, load_json, read_csv, write_table, partition_signature)

METRICS = ('overall_acc', 'head20_acc', 'middle60_acc', 'bottom20_tail_acc', 'non_tail_acc')


def prediction(path):
    with np.load(path, allow_pickle=False) as z:
        values = {k: z[k].copy() for k in ('sample_id', 'class_id', 'prediction', 'correct', 'logit_margin')}
    ids, labels, pred, hit, margin = (values[k] for k in values)
    if (not np.array_equal(ids, np.arange(10000)) or any(x.shape != ids.shape for x in (labels, pred, hit, margin))
            or not np.isin(labels, np.arange(100)).all() or not np.isin(pred, np.arange(100)).all()
            or not np.all(np.bincount(labels, minlength=100) == 100)
            or not np.array_equal(hit, labels == pred) or not np.isfinite(margin).all()):
        raise ValueError('Invalid per-sample prediction file: ' + str(path))
    values['correct'] = hit.astype(bool)
    return values


def groups_for(job):
    counts = job['aggregation']['counts']
    head = sorted(range(100), key=lambda c: (-counts['class_counts'][c], c))[:20]
    tail = counts['tail_ids']
    middle = [c for c in range(100) if c not in head and c not in tail]
    return dict(overall_acc=list(range(100)), head20_acc=head, middle60_acc=middle,
        bottom20_tail_acc=tail, non_tail_acc=[c for c in range(100) if c not in tail])


def metrics_from_prediction(sample, groups):
    classes = np.bincount(sample['class_id'], weights=sample['correct'], minlength=100)
    return {name: float(classes[ids].mean()) for name, ids in groups.items()}


def expected_stages(job, completed):
    rounds = [r for r in DIAGNOSTIC_ROUNDS if r <= completed]
    transfers = [1] if job['mode'] == 'smoke' else B_ROUNDS
    for rnd in rounds:
        for stage in ('ordinary_B', 'sample_weighted_B_counterfactual'):
            yield rnd, stage
        if job['arm'] in ('ab', 'plain_a', 'method_a'):
            if job['arm'] == 'ab' and rnd in transfers:
                yield rnd, 'after_B_transfer'
            if rnd <= 90:
                yield rnd, 'ordinary_A'
                yield rnd, 'sample_weighted_A_counterfactual'


def audit_run(run, job, completed=100):
    run = Path(run)
    if not 0 <= completed <= 100 or job['arm'] not in (*ARMS, 'plain_a', 'method_a'):
        raise ValueError('Invalid run request')
    check_aggregation(job['aggregation'])
    if load_json(run / 'joint_job.json') != job:
        raise ValueError('Job differs from the frozen plan')
    receipt = load_json(run / 'joint_receipt.json')
    if (receipt.get('schema') != SCHEMA or receipt.get('job_digest') != digest(job)
            or not receipt.get('code_unchanged') or receipt.get('exit_code') != 0):
        raise ValueError('Missing/failed stable-code worker receipt')
    cfg = audit_runtime_config(run, job)
    progress = load_json(run / ('completion.json' if completed == 100 else 'progress.json'))
    if (progress['completed_round'] != completed or progress['official_test_passes'] != completed + 1
            or progress.get('weights_sha256') != job['aggregation']['weights_sha256']
            or progress.get('arm') != job['arm'] or progress.get('mode') != job['mode']):
        raise ValueError('Completion/weight identity mismatch')
    frozen = job['arm'] == 'frozen'
    a_rounds = 0 if frozen else min(completed, 90)
    steps = sum((n + 31) // 32 for n in job['aggregation']['counts']['sizes'])
    if (progress['normal_optimizer_steps'] != completed * 3 * steps
            or progress['extra_optimizer_steps'] != a_rounds * steps):
        raise ValueError('Wrong local optimization budget')
    budgets = read_csv(run / 'budget.csv')
    if len(budgets) != 30 * (completed + a_rounds):
        raise ValueError('Missing/duplicate local training budget records')
    phases = [(r, 'B', 'normal_B', 3) for r in range(1, completed + 1)]
    phases += [(r, 'A', 'refresh_A', 1) for r in range(1, a_rounds + 1)]
    for rnd, factor, phase, epochs in phases:
        selected = [r for r in budgets if int(r['round']) == rnd and r['phase'] == phase]
        if len(selected) != 30 or {int(r['client_id']) for r in selected} != set(range(30)):
            raise ValueError('Local clients missing from phase')
        for row in selected:
            n = job['aggregation']['counts']['sizes'][int(row['client_id'])]
            if int(row['optimizer_steps']) != epochs * ((n + 31) // 32) or int(row['sample_presentations']) != epochs * n:
                raise ValueError('Per-client budget mismatch')
        weights = read_csv(run / 'aggregation_weights' / f'r{rnd:03d}_{factor}.csv')
        if len(weights) != 30 or {int(row['client_id']) for row in weights} != set(range(30)):
            raise ValueError('Missing/duplicate aggregation weights')
        for row in weights:
            j = int(row['client_id'])
            if (int(row['round']) != rnd or row['factor'] != factor
                    or not math.isclose(float(row['weight']), job['aggregation']['weights'][j], abs_tol=1e-12)
                    or not math.isclose(float(row['original_weight']), job['aggregation']['counts']['sizes'][j] / 10847, abs_tol=1e-12)):
                raise ValueError('Actual phase aggregation differs from the frozen weights')
        event = load_json(run / 'events' / f'r{rnd:03d}_c000_main_{phase}' / 'event.json')
        if not event['reconstruction_passed'] or float(event['frozen_factor_max_abs_error']) != 0:
            raise ValueError('Actual aggregation reconstruction/frozen-factor audit failed')
        if event['selected_client_ids'] != [int(j) for j in event['selected_client_ids']] or sorted(event['selected_client_ids']) != list(range(30)):
            raise ValueError('Bad event client identities')
        if not np.allclose(event['server_weights'], [job['aggregation']['weights'][j] for j in event['selected_client_ids']], rtol=0, atol=1e-12):
            raise ValueError('Bridge recorded different aggregation weights')
        slots = job.get('settings', {}).get('client_concurrency', 1)
        if slots > 1:
            execution = load_json(run / 'parallel_execution' / f'r{rnd:03d}_{factor}.json')
            clients = execution['selected_client_ids']
            if (execution['round'] != rnd or execution['factor'] != factor or execution['slots'] != slots
                    or execution['clients'] != 30 or clients != event['selected_client_ids']
                    or [r['client_id'] for r in execution['client_audits']] != clients
                    or any(not 0 <= r['slot'] < slots for r in execution['client_audits'])):
                raise ValueError('Invalid parallel client execution record')
    # Numerical comparison pilots are no longer part of smoke or completion acceptance.
    from tools.sfra.simple_controls import partition_counts
    actual_counts = partition_counts(read_csv(run / 'partition_manifest.csv'), job['aggregation']['counts']['tail_ids'])
    if partition_signature(read_csv(run / 'partition_manifest.csv')) != job['aggregation']['partition_sha256']:
        raise ValueError('Sample-level client partition differs from the registered source')
    if actual_counts != job['aggregation']['counts']:
        raise ValueError('Training partition counts differ')
    matrix = np.zeros((30, 100), dtype=np.int64)
    for row in read_csv(run / 'partition_manifest.csv'):
        matrix[int(row['client_id']), int(row['class_id'])] += 1
    if digest(matrix.tolist()) != job['aggregation']['matrix_sha256']:
        raise ValueError('Client/class count matrix differs')
    prior = load_json(run / 'class_prior.json')
    if prior['counts'] != matrix.sum(0).tolist() or not np.allclose(prior['prior'], matrix.sum(0) / matrix.sum(), rtol=0, atol=1e-12):
        raise ValueError('LA prior was changed with the aggregation weights')
    corrections = read_csv(run / 'sfra_rounds.csv')
    if job['arm'] in ('frozen', 'plain_a') and (corrections or progress['functional_correction_steps']):
        raise ValueError('Frozen/plain-A arm executed functional correction')
    if job['arm'] in ('plain_a', 'method_a') and (progress.get('b_transfer_events', 0)
            or progress.get('b_transfer_optimizer_steps', 0)
            or ((run / 'b_transfer_rounds.csv').is_file() and read_csv(run / 'b_transfer_rounds.csv'))):
        raise ValueError('A-only experiment executed source B transfer')
    if frozen and (progress.get('b_transfer_events', 0) or progress.get('b_transfer_optimizer_steps', 0)
            or any(row.get('trainable_factor', 'B') != 'B' or int(row.get('a_optimizer_steps') or 0)
                   for row in budgets)):
        raise ValueError('Frozen arm executed A training or source B transfer')
    if job['arm'] in ('ab', 'method_a'):
        if [int(r['round']) for r in corrections] != list(range(1, completed + 1)):
            raise ValueError('Missing A functional records')
        total = 0
        for row in corrections:
            if int(row['correction_steps']) not in ((0, 3) if int(row['round']) <= 90 else (0,)):
                raise ValueError('Unexpected A correction budget')
            if job['arm'] == 'method_a' and int(row['correction_steps']) == 0:
                allowed = ('no_active_tokens', 'zero_proposal_radius') if int(row['round']) <= 90 else ('B_only_schedule',)
                if row.get('skip_reason') not in allowed:
                    raise ValueError('Method A skipped correction without a permitted reason')
            total += int(row['correction_steps'])
        if total != progress['functional_correction_steps']:
            raise ValueError('A correction completion budget differs')
    if job['arm'] == 'ab':
        b = read_csv(run / 'b_transfer_rounds.csv')
        expected = [r for r in ([1] if job['mode'] == 'smoke' else B_ROUNDS) if r <= completed]
        if [int(r['round']) for r in b] != expected or progress['b_transfer_events'] != len(expected):
            raise ValueError('Wrong B transfer schedule')
        for row in b:
            rnd = int(row['round'])
            manifest = load_json(run / 'b_transfer_rounds' / f'r{rnd:03d}' / 'calibration_manifest.json')
            wanted = 2 if manifest['shared_donor_ids'] else 0
            if int(row['optimizer_steps']) != wanted:
                raise ValueError('Wrong shared C optimization steps')
    groups = groups_for(job)
    curves = sorted(read_csv(run / 'round_metrics.csv'), key=lambda r: int(r['round']))
    if [int(r['round']) for r in curves] != list(range(completed + 1)):
        raise ValueError('Missing/duplicate committed rounds')
    predictions, stages = {}, {}
    initial_labels = initial_a = last_a = None
    for rnd in range(completed + 1):
        sample = prediction(run / 'predictions' / f'r{rnd:03d}.npz')
        record = load_json(run / 'predictions' / f'r{rnd:03d}.json')
        if rnd == 0:
            initial_labels, initial_a = sample['class_id'], record['a_sha256']
        if not np.array_equal(sample['class_id'], initial_labels) or frozen and record['a_sha256'] != initial_a:
            raise ValueError('Test identities or frozen A changed')
        if job['arm'] in ('plain_a', 'method_a') and rnd > 90 and record['a_sha256'] != last_a:
            raise ValueError('A changed after round 90')
        last_a = record['a_sha256']
        actual = metrics_from_prediction(sample, groups)
        for metric in METRICS:
            if not math.isclose(actual[metric], float(curves[rnd][metric]), abs_tol=1e-8):
                raise ValueError('Predictions and committed accuracy differ')
        classes = read_csv(run / f'per_class_accuracy_epoch_{rnd-1}.csv')
        if len(classes) != 100 or {int(row['class_id']) for row in classes} != set(range(100)):
            raise ValueError('Incomplete per-class metrics')
        acc = np.bincount(sample['class_id'], weights=sample['correct'], minlength=100)
        if any(not math.isclose(float(row['per_class_acc']), acc[int(row['class_id'])], abs_tol=1e-8) for row in classes):
            raise ValueError('Class metrics differ from saved predictions')
        predictions[rnd] = sample
    for rnd, stage in expected_stages(job, completed):
        sample = prediction(run / 'stage_predictions' / stage / f'r{rnd:03d}.npz')
        if not np.array_equal(sample['class_id'], initial_labels):
            raise ValueError('Diagnostic sample order differs')
        record = load_json(run / 'stage_predictions' / stage / f'r{rnd:03d}.json')
        actual = metrics_from_prediction(sample, groups)
        if any(not math.isclose(actual[m], float(record[m]), abs_tol=1e-8) for m in METRICS):
            raise ValueError('Stage metadata differs from predictions')
        stages[rnd, stage] = sample
        if job['arm'] == 'plain_a' and stage == ('ordinary_A' if rnd <= 90 else 'ordinary_B'):
            committed = load_json(run / 'predictions' / f'r{rnd:03d}.json')
            if any(not record.get(k) or record[k] != committed.get(k) for k in ('a_sha256', 'b_sha256')):
                raise ValueError('Plain-A committed state differs from its ordinary phase')
        if job['arm'] == 'method_a' and stage == 'ordinary_B':
            committed = load_json(run / 'predictions' / f'r{rnd:03d}.json')
            keys = ('b_sha256',) if rnd <= 90 else ('a_sha256', 'b_sha256')
            if any(not record.get(k) or record[k] != committed.get(k) for k in keys):
                raise ValueError('Method A altered B after B aggregation or A after round 90')
    return dict(job=job, run=run, config=cfg, progress=progress, curves=curves,
        predictions=predictions, stages=stages, groups=groups, budgets=budgets,
        metadata=load_json(run / 'bridge_metadata.json'))


def sample_dynamics(result):
    rows = []
    seen = np.zeros(len(result['predictions'][0]['correct']), dtype=bool)
    previous = None
    samples = result['predictions']
    for rnd, sample in samples.items():
        current = sample['correct']
        for group, ids in result['groups'].items():
            mask = np.isin(sample['class_id'], ids)
            initial = samples[0]['correct']
            rows.append(dict(arm=result['job']['arm'], round=rnd, group=group,
                accuracy=float(current[mask].mean()*100), initial_correct=int((mask & initial).sum()),
                retained_initial=int((mask & initial & current).sum()), learned_initial_wrong=int((mask & ~initial & current).sum()),
                never_correct_so_far=int((mask & ~(seen | current)).sum()),
                previously_correct_now_wrong=int((mask & seen & ~current).sum()),
                correct_to_wrong=int((mask & previous & ~current).sum()) if previous is not None else 0,
                wrong_to_correct=int((mask & ~previous & current).sum()) if previous is not None else 0,
                mean_logit_margin=float(sample['logit_margin'][mask].mean())))
        seen |= current
        previous = current
    return rows


def stage_comparisons(result):
    rows = []
    for rnd in sorted({r for r, _ in result['stages']}):
        stages = {s: p for (r, s), p in result['stages'].items() if r == rnd}
        stages['previous_committed'] = result['predictions'][rnd-1]
        stages['committed'] = result['predictions'][rnd]
        pairs = [('ordinary_B', 'previous_committed', 'ordinary_B_learning'),
            ('ordinary_B', 'sample_weighted_B_counterfactual', 'B_aggregation_same_local_updates')]
        if 'after_B_transfer' in stages:
            pairs.append(('after_B_transfer', 'ordinary_B', 'B_shared_transfer'))
        if 'ordinary_A' in stages:
            pairs += [('ordinary_A', 'after_B_transfer' if 'after_B_transfer' in stages else 'ordinary_B', 'ordinary_A_learning'),
                ('ordinary_A', 'sample_weighted_A_counterfactual', 'A_aggregation_same_local_updates')]
            if result['job']['arm'] != 'plain_a':
                pairs.append(('committed', 'ordinary_A', 'A_functional_correction'))
        for left, right, operation in pairs:
            a, b = stages[left], stages[right]
            for group, ids in result['groups'].items():
                mask = np.isin(a['class_id'], ids)
                rows.append(dict(arm=result['job']['arm'], round=rnd, operation=operation, group=group,
                    before_accuracy=float(b['correct'][mask].mean()*100), after_accuracy=float(a['correct'][mask].mean()*100),
                    delta_accuracy=float((a['correct'][mask].astype(float)-b['correct'][mask]).mean()*100),
                    corrected_examples=int((mask & ~b['correct'] & a['correct']).sum()),
                    damaged_examples=int((mask & b['correct'] & ~a['correct']).sum()),
                    delta_mean_logit_margin=float((a['logit_margin'][mask]-b['logit_margin'][mask]).mean()),
                    scope='same-state immediate observation; not an independent long-term component ablation'))
    return rows


def summarize(root, compare_root=None):
    from scripts.run_cliplora_joint_aggregation import make_job
    root = Path(root)
    plan = load_json(root / 'experiment_plan.json')
    if plan['aggregation'].get('rule') in ('tailrw16', 'fedavg'):
        from tools.client_aggregation.frozen_tailrw import summarize as summarize_tailrw
        return summarize_tailrw(root, compare_root)
    out = root / 'analysis'; out.mkdir(exist_ok=True)
    status, results, performance, costs, dynamics, stages, classes, cohorts = [], {}, [], [], [], [], [], []
    for arm in ARMS:
        job = make_job(plan, arm)
        run = root / job['run']
        row = dict(arm=arm, status='missing', completed_round=0, reason='')
        if (run / 'progress.json').is_file():
            row.update(status='incomplete', completed_round=load_json(run / 'progress.json')['completed_round'])
        if (run / 'completion.json').is_file():
            try:
                result = audit_run(run, job)
                results[arm] = result
                row.update(status='complete', completed_round=100)
            except (ValueError, OSError, KeyError, TypeError, AssertionError) as error:
                row.update(status='invalid', reason=str(error))
        status.append(row)
        print(arm, row['status'], 'round=' + str(row['completed_round']), row['reason'])
        if arm not in results:
            continue
        result = results[arm]
        curve = result['curves']
        perf = dict(arm=arm, seed=42, **{'last20_'+m: statistics.mean(float(r[m]) for r in curve[81:101]) for m in METRICS})
        perf.update(final_tail=float(curve[100]['bottom20_tail_acc']),
            tail_peak_to_final=max(float(r['bottom20_tail_acc']) for r in curve[1:])-float(curve[100]['bottom20_tail_acc']),
            tail_change_90_to_100=float(curve[100]['bottom20_tail_acc'])-float(curve[90]['bottom20_tail_acc']))
        performance.append(perf)
        dynamics.extend(sample_dynamics(result)); stages.extend(stage_comparisons(result))
        for c in range(100):
            values = [100*float(p['correct'][p['class_id'] == c].mean()) for r, p in result['predictions'].items() if r >= 81]
            classes.append(dict(arm=arm, class_id=c, last20_accuracy=statistics.mean(values),
                train_count=job['aggregation']['counts']['class_counts'][c], holders=job['aggregation']['counts']['coverage'][c]))
        for anchor in (0, 50, 80):
            reference = result['predictions'][anchor]
            for group, ids in result['groups'].items():
                mask = np.isin(reference['class_id'], ids)
                old = mask & reference['correct']; wrong = mask & ~reference['correct']
                cohorts.append(dict(arm=arm, anchor_round=anchor, group=group,
                    reference_correct=int(old.sum()), reference_wrong=int(wrong.sum()),
                    last20_retained=statistics.mean(int((p['correct'] & old).sum()) for r,p in result['predictions'].items() if r>=81),
                    last20_acquired=statistics.mean(int((p['correct'] & wrong).sum()) for r,p in result['predictions'].items() if r>=81)))
        done = result['progress']
        evaluations = read_csv(run / 'evaluation_budget.csv')
        seconds = sum(float(r['seconds']) for r in evaluations)
        feedback = read_csv(run / 'sfra_costs.csv')
        transfer = read_csv(run / 'b_transfer_rounds.csv') if (run / 'b_transfer_rounds.csv').is_file() else []
        def total(records, key):
            return sum(float(row.get(key) or 0) for row in records)
        costs.append(dict(arm=arm, normal_steps=done['normal_optimizer_steps'], extra_A_steps=done['extra_optimizer_steps'],
            client_concurrency=job.get('settings', {}).get('client_concurrency', 1),
            A_correction_steps=done['functional_correction_steps'], B_transfer_steps=done.get('b_transfer_optimizer_steps',0),
            elapsed_seconds=done['elapsed_seconds'], official_test_seconds=seconds,
            stage_test_seconds=done['stage_diagnostic_seconds'],
            elapsed_excluding_official_and_stage_tests=float(done['elapsed_seconds'])-seconds-float(done['stage_diagnostic_seconds']),
            local_training_images=sum(int(r['sample_presentations']) for r in result['budgets']),
            local_upload_bytes=sum(int(r['upload_bytes']) for r in result['budgets']),
            local_modeled_downlink_bytes=total(result['budgets'],'modeled_downlink_bytes'),
            A_feedback_forward_images=total(feedback,'forward_images'), A_feedback_backward_images=total(feedback,'backward_images'),
            A_feedback_upload_bytes=total(feedback,'upload_bytes'), A_feedback_downlink_bytes=total(feedback,'downlink_bytes'),
            B_algorithm_forward_images=total(transfer,'algorithm_forward_images'),
            B_algorithm_backward_images=total(transfer,'algorithm_backward_images'),
            B_internal_diagnostic_forward_images=total(transfer,'diagnostic_forward_images'),
            B_extra_upload_bytes=total(transfer,'extra_upload_bytes'), B_extra_downlink_bytes=total(transfer,'extra_downlink_bytes'),
            modeled_count_upload_bytes=30*100*4,
            environment=json.dumps(result['metadata'].get('environment',{}),ensure_ascii=False)))
    pair, paired_samples = {}, []
    if len(results) == 2:
        a,b=results['ab'],results['frozen']
        mismatches=[key for key in PAIR_HASHES if a['metadata'][key]!=b['metadata'][key]]
        base_a,base_b=a['config']['base_training_config'],b['config']['base_training_config']
        intentional={'candidate_rounds','extra_trainable_factor','extra_steps_expected'}
        mismatches += ['base.'+k for k in set(base_a)|set(base_b) if k not in intentional and base_a.get(k)!=base_b.get(k)]
        for key in ('class_id','prediction'):
            if not np.array_equal(a['predictions'][0][key],b['predictions'][0][key]):
                mismatches.append('initial_'+key)
        environment_keys=('python','torch','cuda','cudnn','gpu')
        same_environment=all(a['metadata'].get('environment',{}).get(k)==b['metadata'].get('environment',{}).get(k)
                             for k in environment_keys)
        pair=dict(status='eligible' if not mismatches else 'mismatch', mismatches=mismatches,
            same_recorded_environment=same_environment, independent_seeds=1,
            budget_matched=False, separate_A_B_effect_identifiable=False,
            interpretation='Full AB package versus permanently frozen A under identical new aggregation; no component attribution or significance claim')
        if not mismatches:
            perf={r['arm']:r for r in performance}
            pair['ab_minus_frozen']={m:perf['ab']['last20_'+m]-perf['frozen']['last20_'+m] for m in METRICS}
            for group,ids in a['groups'].items():
                mask=np.isin(a['predictions'][0]['class_id'],ids)
                for rnd in range(81,101):
                    ac,bc=a['predictions'][rnd]['correct'],b['predictions'][rnd]['correct']
                    paired_samples.append(dict(round=rnd,group=group,
                        both_correct=int((mask&ac&bc).sum()),only_ab_correct=int((mask&ac&~bc).sum()),
                        only_frozen_correct=int((mask&~ac&bc).sum()),both_wrong=int((mask&~ac&~bc).sum())))
    pilots = []
    for arm in ARMS:
        for path in sorted((root/'smoke/seed42'/arm/'parallel_benchmark').glob('factor_*.json')):
            bench = load_json(path)
            pilots.append(dict(arm=arm, factor=bench['factor'], passed=bench['passed'],
                serial_seconds=bench['serial']['seconds'], parallel_seconds=bench['parallel']['seconds'],
                speedup=bench['speedup'], max_parameter_error=max(r['max_abs'] for r in bench['comparisons']),
                short_group_clients=bench['short_group']['clients'],
                process_peak_allocated_gib=bench['parallel']['process_peak_allocated_bytes']/2**30))
    stage_summary=[]
    for arm,operation,group in sorted({(r['arm'],r['operation'],r['group']) for r in stages}):
        rows=[r for r in stages if (r['arm'],r['operation'],r['group'])==(arm,operation,group)]
        stage_summary.append(dict(arm=arm,operation=operation,group=group,events=len(rows),
            mean_delta_accuracy=statistics.mean(r['delta_accuracy'] for r in rows),
            mean_corrected_examples=statistics.mean(r['corrected_examples'] for r in rows),
            mean_damaged_examples=statistics.mean(r['damaged_examples'] for r in rows),
            scope='descriptive mean of prespecified events, not independent replicates'))
    tables=dict(status=status,performance=performance,costs=costs,sample_dynamics=dynamics,parallel_pilot=pilots,
        stage_summary=stage_summary,
        stage_effects=stages,per_class=classes,retention_cohorts=cohorts,paired_samples=paired_samples,
        curves=[dict(row,arm=arm) for arm,r in results.items() for row in r['curves']])
    for name, rows in tables.items():
        write_table(out / (name+'.csv'), rows)
    write_json(out/'pair_audit.json',pair)
    from tools.client_aggregation.plots import render_plots
    figures, figure_note = render_plots(out, tables)
    write_json(out/'figure_manifest.json', dict(files=figures, note=figure_note))
    lines=['# 新聚合：冻结 LoRA A 与当前 A+B 的两组实验','',
        '固定终点：第81—100轮提交模型均值。仅seed42、lambda= '+str(plan['settings']['lambda_value'])+'。',
        '冻结组没有额外A训练；AB组保留A训练、保持修正和共享C。两组总预算不同，不能分离三个操作的长期贡献。','',
        '| 实验 | 状态 | Overall | Head20 | Middle60 | Tail20 |','|---|---|---:|---:|---:|---:|']
    for row in status:
        p=next((v for v in performance if v['arm']==row['arm']),None)
        values=[f'{p["last20_"+m]:.4f}' if p else '—' for m in METRICS[:4]]
        lines.append('| '+row['arm']+' | '+row['status']+' | '+' | '.join(values)+' |')
        if row['reason']:
            lines.append('\n审计失败：'+row['reason'])
    lines += ['', '## 如何解释结果','',
        '- performance.csv：联合查看头、中、尾及Overall，不能只凭一个组的提升宣布全面改善。',
        '- sample_dynamics.csv / retention_cohorts.csv：区分保住原本会的样本、学会原本不会的样本，以及后期再次答错；这是轨迹描述，不是直接测量知识。',
        '- stage_effects.csv：同轮同起点的即时变化；原样本量聚合只做不提交的反事实，不代表其完整训练结果。',
        '- paired_samples.csv：AB新增答对和新增答错的样本分别有多少。',
        '- costs.csv：额外A训练、保持优化和B优化都单列；扣除测试后的时间仍包含保存和原有训练侧诊断。','',
        '## 后续优化依据','',
        '1. 普通A更新改善头中部、保持修正又削弱这些收益：优先检查保持目标/范围；即时现象不能单独证明长期原因。',
        '2. 普通A更新改善学习且修正保留收益：保留该机制，再看尾类旧能力能否长期维持。',
        '3. B补充在训练反馈改善、测试阶段没有改善：检查反馈覆盖、目标和来源利用；不要仅扩大C或步数。',
        '4. 新聚合与原样本量聚合即时取舍相似：检查类别计数代理与实际更新是否一致，不能据此宣布完整聚合无效。',
        '5. 两组都存在大量始终答错样本：提示仍有学习空间，不能仅凭此断言是LoRA容量不足。',
        '6. 若AB整体没有额外收益，先按阶段证据定位，再决定增加单组件对照；本轮不自动追加实验。']
    if pair.get('status')=='eligible':
        d=pair['ab_minus_frozen']
        lines += ['', 'AB减冻结组：'+', '.join(f'{m}={v:+.4f} pp' for m,v in d.items())+'。',
            '本次仅说明整套AB在新聚合基础上的结果变化；没有证明新聚合优于旧聚合，也没有排除额外训练预算的影响。']
        if not pair['same_recorded_environment']:
            lines.append('两个运行的环境记录不同；小差异与耗时比较需谨慎。')
    elif pair:
        lines += ['', '两组配对审计未通过：'+', '.join(pair['mismatches'])+'；不输出配对结论。']
    if stage_summary:
        lines += ['', '## 阶段即时变化', '', '以下为预先指定诊断轮次的描述性平均，不是独立重复或长期组件贡献。', '',
            '| 实验 | 操作 | Head20变化 | Middle60变化 | Tail20变化 |', '|---|---|---:|---:|---:|']
        for arm,operation in sorted({(r['arm'],r['operation']) for r in stage_summary}):
            rows={r['group']:r for r in stage_summary if r['arm']==arm and r['operation']==operation}
            values=[f'{rows[m]["mean_delta_accuracy"]:+.4f}' for m in ('head20_acc','middle60_acc','bottom20_tail_acc')]
            lines.append('| '+arm+' | '+operation+' | '+' | '.join(values)+' |')
    if dynamics:
        lines += ['', '## 冻结与AB各自还存在哪些错误', '',
            '下表为最后20轮平均样本数。保留/学会相对第0轮；曾会后错包含任何此前提交轮次曾答对的样本。它们不是互斥类别，不能直接相加。', '',
            '| 实验 | 类别组 | 保留起初答对 | 学会起初答错 | 始终未答对 | 曾答对后又错 |', '|---|---|---:|---:|---:|---:|']
        for arm in results:
            for group in ('head20_acc','middle60_acc','bottom20_tail_acc'):
                rows=[r for r in dynamics if r['arm']==arm and r['group']==group and r['round']>=81]
                values=[f'{statistics.mean(r[k] for r in rows):.2f}' for k in
                    ('retained_initial','learned_initial_wrong','never_correct_so_far','previously_correct_now_wrong')]
                lines.append('| '+arm+' | '+group+' | '+' | '.join(values)+' |')
    if pilots:
        lines += ['', '## 四客户端并行尝试', '',
            '以下来自独立smoke，同一起点与批次的单次局部训练计时。大于1表示此次四路更快，不代表完整训练也按此倍数加速。显存是进程峰值，包含模型等常驻占用。', '',
            '| 实验 | 因子 | 数值通过 | 串行秒 | 四路秒 | 速度比 | 最大参数差 | 短组客户端数 |',
            '|---|---|---|---:|---:|---:|---:|---:|']
        for p in pilots:
            lines.append(f'| {p["arm"]} | {p["factor"]} | {p["passed"]} | {p["serial_seconds"]:.2f} | '
                         f'{p["parallel_seconds"]:.2f} | {p["speedup"]:.3f} | {p["max_parameter_error"]:.3g} | {p["short_group_clients"]} |')
    for figure in figures:
        if figure.endswith('.png'):
            lines += ['', '!['+figure+']('+figure+')']
    if figure_note:
        lines += ['', figure_note]
    (out/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('Report:',out/'report.md')
    return status


def pack_results(root):
    root=Path(root).resolve()
    target=root.with_name(root.name+'_results.tar.gz')
    temp=target.with_suffix('.tmp')
    with tarfile.open(temp,'w:gz') as archive:
        for path in sorted(root.rglob('*')):
            parts = path.relative_to(root).parts
            smoke_pilot = 'smoke' in parts and 'parallel_benchmark' in parts
            if (path.is_file() and not path.is_symlink() and path.suffix in ('.json','.csv','.npz','.md','.log','.yaml','.png','.svg')
                    and not {'checkpoints','locks'}.intersection(parts) and ('smoke' not in parts or smoke_pilot)):
                archive.add(path,arcname=str(Path(root.name)/path.relative_to(root)))
    temp.replace(target)
    return target
