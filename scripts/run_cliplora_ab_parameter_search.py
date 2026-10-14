"""Six fixed full-method trajectories; no pruning, retries, or test-driven updates."""
import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
from statistics import mean
import subprocess
import sys
import time
import traceback

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from scripts import run_cliplora_joint_aggregation as single
from scripts.run_ab_validation import file_lock
from tools.client_aggregation.ab_factorial import contract_for, feedback_preflight
from tools.client_aggregation.analysis import METRICS, audit_run
from tools.client_aggregation.protocol import code_hashes, load_json, read_csv, write_table

GRID = {'S1': (1/3, 1.), 'S2': (1., 1.), 'S3': (3., 1.),
        'S4': (1., 3.), 'S5': (3., 3.), 'S6': (9., 3.)}
PRIORITY = ('S2', 'S5', 'S4', 'S1', 'S3', 'S6')
RUN = 'runs/seed42/method_a_margin'
IDENTITIES = ('initial_lora_sha256', 'frozen_model_sha256', 'schedule_sha256',
              'pool_sha256', 'test_sha256', 'probe_images_sha256', 'probe_manifest_sha256',
              'client_sample_counts', 'classnames')
EVALUATION = ('CIFAR-100 official test, repeatedly inspected and used for hyperparameter '
              'exploration; development results, NOT independent test conclusions. No independent dev set.')


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')
    os.replace(tmp, path)


def candidate_plan(baseline, group, hashes):
    plan = copy.deepcopy(baseline)
    keep, learn = GRID[group]
    plan['settings']['margin_parameters'].update(keep_weight=keep, learn_weight=learn, correction_step_size=1.)
    plan['contract'] = contract_for('new_a_new_b_joint', plan['settings']['margin_parameters'])
    plan['code_sha256'] = hashes
    # Every other method/training/aggregation field must remain identical.
    old, new = copy.deepcopy(baseline['contract']), copy.deepcopy(plan['contract'])
    for field in ('keep_weight', 'learn_weight', 'retention_weight', 'correction_step_size'):
        old['method_a_margin'].pop(field)
        new['method_a_margin'].pop(field)
    if old != new:
        raise ValueError('Unexpected change outside the authorized A coefficients and z step size')
    single.make_job(plan, 'method_a_margin')
    return plan


def endpoint(rows):
    selected = [r for r in rows if 81 <= int(r['round']) <= 100]
    if sorted(int(r['round']) for r in selected) != list(range(81, 101)):
        raise ValueError('Endpoint requires every committed round 81..100 exactly once')
    values = {m: mean(float(r[m]) for r in selected) for m in METRICS[:4]}
    if not all(math.isfinite(v) for v in values.values()):
        raise ValueError('Nonfinite endpoint')
    return values


def dominates_baseline(scores, baseline):
    keys = ('overall_acc', 'bottom20_tail_acc')
    return all(scores[k] >= baseline[k] for k in keys) and any(scores[k] > baseline[k] for k in keys)


def prepare(root, baseline_root, gpus):
    if (root/'search_plan.json').exists() or any((root/g/RUN).exists() for g in GRID):
        raise ValueError('Search root already registered or occupied; preserve it and use a fresh root')
    baseline = load_json(baseline_root/'experiment_plan.json')
    if baseline['settings']['factorial_cell'] != 'new_a_new_b_joint':
        raise ValueError('Reference must be the current complete method')
    baseline_job = single.make_job(baseline, 'method_a_margin')
    audited = audit_run(baseline_root/RUN, baseline_job)
    hashes = code_hashes(REPO)
    changed = sorted(k for k in hashes if hashes[k] != baseline['code_sha256'].get(k))
    allowed = {'federated_main.py', 'utils/method_a_margin_math.py',
               'utils/cliplora_method_a_margin.py', 'tools/client_aggregation/protocol.py'}
    if set(changed) != allowed:
        raise ValueError('Unexpected training source differences: ' + str(changed))
    suite = dict(schema='full_ab_a_parameter_search_v1', source_root=str(REPO),
        baseline_root=str(baseline_root), priority=list(PRIORITY), gpu_assignment=dict(zip(PRIORITY, gpus)),
        evaluation=EVALUATION, endpoint='arithmetic mean of committed rounds 81..100, accuracy in percent',
        selection='Tail20 and Overall >= current full baseline, at least one >; prioritize Tail20; inspect Head20/Middle60/time; retain tradeoffs; no composite score',
        fixed=dict(seed=42, rounds=100, z_step_size=1., proximity_mu=1., correction_steps=3,
                   ordinary_A_lr=.001, A_epochs=1, B_epochs=3, A_last_round=90, joint_lambda=.1),
        execution='Six fresh formal runs only; independent histories; no smoke trajectories, pruning, retry, rollback or method changes',
        baseline_scores=endpoint(audited['curves']), baseline_identity={k: audited['metadata'][k] for k in IDENTITIES},
        source_differences=changed, launcher_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), groups={})
    for group, (keep, learn) in GRID.items():
        plan = candidate_plan(baseline, group, hashes)
        single.preflight(plan)
        save(root/group/'experiment_plan.json', plan)
        suite['groups'][group] = dict(root=str(root/group), keep=keep, learn=learn, c=learn, r=keep/learn)
    save(root/'feedback_preflight.json', feedback_preflight(baseline['settings']['reference_run']))
    save(root/'search_plan.json', suite)
    print(json.dumps(dict(registered=str(root), priority=PRIORITY, baseline_scores=suite['baseline_scores']), indent=2), flush=True)


def actual_corrections(run, destination):
    """Read existing snapshots after training; no additional model evaluations."""
    import torch
    torch.set_num_threads(1)
    cfg = load_json(run/'sfra_config.json')
    keys = cfg['a_parameter_order']
    args = load_json(run/'bridge_metadata.json')['resolved_args']
    scaling = float(args['cliplora_alpha']) / math.sqrt(int(args['cliplora_rank']))
    records = []
    for row in read_csv(run/'sfra_rounds.csv')[:90]:
        rnd = int(row['round'])
        def event(r, phase):
            return torch.load(run/'events'/f'r{r:03d}_c000_main_{phase}'/'state.pt', map_location='cpu', weights_only=False)
        ordinary = event(rnd, 'refresh_A')
        submitted = event(rnd+1, 'normal_B')['anchor_lora_state']
        before, after = ordinary['anchor_lora_state'], ordinary['actual_after_lora_state']
        da2 = ord2 = dw2 = 0.
        for key in keys:
            delta = submitted[key].double() - after[key].double()
            da2 += float(delta.square().sum())
            ord2 += float((after[key].double()-before[key].double()).square().sum())
            b_key = key.replace('lora_A', 'lora_B')
            if not torch.equal(submitted[b_key], after[b_key]):
                raise ValueError('B changed during A correction')
            b = after[b_key].double()
            dw2 += scaling**2 * float(((b.T @ b) * (delta @ delta.T)).sum())
        norm, ordinary_norm = math.sqrt(da2), math.sqrt(ord2)
        records.append(dict(round=rnd, actual_A_correction_norm=norm, ordinary_A_update_norm=ordinary_norm,
            correction_to_ordinary_A=norm/ordinary_norm if ordinary_norm else 0.,
            effective_LoRA_correction_norm=math.sqrt(max(dw2, 0.)),
            ideal_radius_z_norm=float(row['radius'])*float(row['correction_z_norm'])))
    if [r['round'] for r in records] != list(range(1, 91)):
        raise ValueError('Expected 90 A correction records')
    write_table(destination, records)
    return {k+'_mean_rounds1_90': mean(r[k] for r in records) for k in records[0] if k != 'round'}


def summarize(root):
    suite = load_json(root/'search_plan.json')
    out = root/'analysis'
    out.mkdir(parents=True, exist_ok=True)
    rows, errors = [], {}
    # Baseline uses its original eta_z=.1, keep=learn=1, and completed full A+B trajectory.
    sources = [('baseline', Path(suite['baseline_root']))] + [(g, root/g) for g in GRID]
    for group, folder in sources:
        try:
            if not (folder/RUN/'completion.json').is_file():
                raise ValueError('Not completed; excluded from endpoint selection')
            plan = load_json(folder/'experiment_plan.json')
            result = audit_run(folder/RUN, single.make_job(plan, 'method_a_margin'))
            if any(result['metadata'][k] != suite['baseline_identity'][k] for k in IDENTITIES):
                raise ValueError('Initialization/data/schedule identity differs from baseline')
            scores = endpoint(result['curves'])
            a_rows = [r for r in read_csv(folder/RUN/'sfra_rounds.csv') if int(r['round']) <= 90]
            b_rows = read_csv(folder/RUN/'b_transfer_rounds.csv')
            params = result['config']
            row = dict(group=group, keep_weight=params['keep_weight'], learn_weight=params['learn_weight'],
                z_step_size=params['correction_step_size'], **scores,
                A_correction_seconds_total=sum(float(r['repair_seconds']) for r in a_rows),
                A_correction_seconds_mean=mean(float(r['repair_seconds']) for r in a_rows),
                B_transfer_seconds_total=sum(float(r['seconds']) for r in b_rows),
                elapsed_seconds=result['progress']['elapsed_seconds'],
                **actual_corrections(folder/RUN, out/(group+'_actual_corrections.csv')))
            row.update({k+'_delta_pp': scores[k]-suite['baseline_scores'][k] for k in scores})
            row['qualifies_vs_baseline'] = group != 'baseline' and dominates_baseline(scores, suite['baseline_scores'])
            write_table(out/(group+'_A_losses_and_costs.csv'), a_rows)
            rows.append(row)
        except Exception as exc:
            errors[group] = str(exc)
            (out/(group+'_error.txt')).write_text(traceback.format_exc(), encoding='utf-8')
    write_table(out/'summary.csv', rows)
    eligible = sorted((r for r in rows if r['qualifies_vs_baseline']), key=lambda r: -r['bottom20_tail_acc'])
    report = dict(evaluation=EVALUATION, endpoint=suite['endpoint'], selection=suite['selection'],
        all_six_completed_and_audited=not errors and len(rows)==7, rows=rows, errors=errors,
        qualifying_groups_by_tail=[r['group'] for r in eligible],
        automatic_followup_training=False, selected_formal_parameters_changed=False)
    save(out/'report.json', report)
    lines = ['完整方法 A 参数搜索：开发性结果', EVALUATION,
             '终点：第 81—100 轮均值；单位：%；差值：百分点。',
             '全部六组完成并审计通过：' + str(report['all_six_completed_and_audited']),
             'group | Overall | Head20 | Middle60 | Tail20 | A修正总秒数 | 实际A修正/普通A更新均值']
    for r in rows:
        lines.append(f"{r['group']} | {r['overall_acc']:.4f} | {r['head20_acc']:.4f} | {r['middle60_acc']:.4f} | {r['bottom20_tail_acc']:.4f} | {r['A_correction_seconds_total']:.2f} | {r['correction_to_ordinary_A_mean_rounds1_90']:.8f}")
    lines += ['满足预先规则的配置（按尾类排序）：' + ', '.join(report['qualifying_groups_by_tail']),
              '完整修正量、损失和耗时见同目录 CSV；Head/Middle/耗时须共同审阅，不自动切换配置或追加训练。',
              'errors: ' + json.dumps(errors, ensure_ascii=False)]
    (out/'report.txt').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    return report


def worker(root, group):
    folder = root/group
    run = folder/RUN
    if run.exists():
        raise ValueError('Worker requires a fresh run directory; automatic resume is disabled for this search')
    plan = load_json(folder/'experiment_plan.json')
    single.preflight(plan, check_cuda=True)
    single.execute(single.make_job(plan, 'method_a_margin'), folder)


def dispatch(root):
    suite = load_json(root/'search_plan.json')
    with file_lock(root/'locks/search_dispatch.lock', timeout=.1):
        if any((root/g/RUN).exists() for g in GRID):
            raise ValueError('One or more runs already exist; dispatcher will not restart/resume them')
        processes, streams, statuses = {}, [], {}
        (root/'launcher_logs').mkdir(parents=True, exist_ok=True)
        for group in PRIORITY:
            gpu = suite['gpu_assignment'][group]
            stream = (root/'launcher_logs'/(group+'_worker.log')).open('x', encoding='utf-8')
            streams.append(stream)
            command = [sys.executable, '-u', str(Path(__file__).resolve()), '--stage', 'worker',
                       '--output-root', str(root), '--group', group]
            process = subprocess.Popen(command, cwd=REPO, env=single.server_environment(gpu),
                                       stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT)
            processes[group] = process
            statuses[group] = dict(pid=process.pid, gpu=gpu, state='running', launched_at=time.time())
            print('START', group, 'gpu', gpu, 'pid', process.pid, flush=True)
        while True:
            for group, process in processes.items():
                code = process.poll()
                statuses[group].update(state='running' if code is None else 'completed' if code==0 else 'failed', exit_code=code)
                progress = root/group/RUN/'progress.json'
                if progress.is_file():
                    try: statuses[group]['completed_round'] = load_json(progress)['completed_round']
                    except (OSError, ValueError): pass  # A concurrently written status is retried, never a training action.
            save(root/'status.json', dict(updated_at=time.time(), evaluation=EVALUATION, groups=statuses))
            if all(p.poll() is not None for p in processes.values()):
                break
            time.sleep(30)
        for stream in streams:
            stream.close()
        report = summarize(root)
        if not report['all_six_completed_and_audited']:
            raise ValueError('Search has failed/incomplete runs; preserved without retries. See analysis/report.json')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('prepare', 'server', 'worker', 'summary', 'status'), required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--baseline-root', type=Path)
    parser.add_argument('--gpus', type=int, nargs=6, default=list(range(6)))
    parser.add_argument('--group', choices=GRID)
    args = parser.parse_args()
    root = args.output_root.resolve()
    os.chdir(REPO)
    if args.stage == 'prepare':
        if args.baseline_root is None or len(set(args.gpus)) != 6 or min(args.gpus) < 0:
            parser.error('prepare requires the full-method baseline and six distinct nonnegative GPU IDs')
        prepare(root, args.baseline_root.resolve(), args.gpus)
    elif args.stage == 'server': dispatch(root)
    elif args.stage == 'worker': worker(root, args.group)
    elif args.stage == 'summary': summarize(root)
    else: print(json.dumps(load_json(root/'status.json'), indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
