"""One fresh frozen-A/no-extra-B run, paired to the existing S4-A/no-extra-B run."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
FROZEN_RUN = 'runs/seed42/frozen'
A_RUN = 'runs/seed42/method_a_margin'
BASE_A_FIELDS = {'candidate_rounds', 'extra_trainable_factor', 'extra_steps_expected'}


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def save(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    os.replace(temporary, path)


def frozen_plan(reference, hashes):
    from tools.client_aggregation.protocol import CONTRACT
    s = reference['settings']
    if (s.get('factorial_cell') != 'new_a_joint' or s['client_concurrency'] != 8
            or s['margin_parameters']['keep_weight'] != 1.
            or s['margin_parameters']['learn_weight'] != 3.
            or s['margin_parameters']['correction_step_size'] != 1.):
        raise ValueError('Expected existing S4-A/no-extra-B, parallel8 reference')
    if hashes != reference['code_sha256']:
        raise ValueError('Reuse the exact reference training source')
    allowed = {'seed', 'lambda_value', 'reference_run', 'data_root', 'num_workers',
               'client_concurrency', 'parallel_validation', 'cuda_policy', 'aggregation_rule'}
    removed = {'factorial_experiment', 'factorial_cell', 'margin_parameters'}
    if set(s)-allowed-removed:
        raise ValueError('Unexpected settings: '+str(set(s)-allowed-removed))
    plan = copy.deepcopy(reference)
    plan['settings'] = {k:v for k,v in s.items() if k in allowed}
    plan['contract'] = copy.deepcopy(CONTRACT)
    return plan


def compare_startup(left, right):
    import numpy as np
    from scripts.run_cliplora_ab_parameter_search import IDENTITIES
    from scripts import run_cliplora_joint_aggregation as single
    from tools.client_aggregation.protocol import audit_runtime_config
    folders = (Path(left), Path(right))
    configs, metas = [], []
    plans = [load(f/'experiment_plan.json') for f in folders]
    for folder, plan, arm, suffix in zip(folders, plans, ('frozen', 'method_a_margin'), (FROZEN_RUN, A_RUN)):
        run = folder/suffix
        cfg = audit_runtime_config(run, single.make_job(plan, arm))
        if cfg.get('b_transfer'):
            raise ValueError('Extra B unexpectedly active')
        command = load(run/'command.json')
        if command[command.index('--sfra_resume')+1] or '--b_transfer_enable' in command:
            raise ValueError('Expected fresh initialization, no B transfer')
        configs.append(cfg)
        metas.append(load(run/'bridge_metadata.json'))
    if plans[0]['aggregation'] != plans[1]['aggregation'] or plans[0]['code_sha256'] != plans[1]['code_sha256']:
        raise ValueError('Aggregation/source mismatch')
    if configs[0]['refresh_rounds'] != [] or configs[1]['refresh_rounds'] != list(range(1, 91)):
        raise ValueError('A schedule mismatch')
    if configs[0]['client_execution'] != configs[1]['client_execution']:
        raise ValueError('Execution mismatch')
    base = [c['base_training_config'] for c in configs]
    if any(base[0].get(k) != base[1].get(k) for k in set(base[0])|set(base[1]) if k not in BASE_A_FIELDS):
        raise ValueError('Ordinary training configuration mismatch')
    if any(metas[0][k] != metas[1][k] for k in IDENTITIES):
        raise ValueError('Initialization/data/client schedule mismatch')
    with np.load(folders[0]/FROZEN_RUN/'predictions/r000.npz') as a, np.load(folders[1]/A_RUN/'predictions/r000.npz') as b:
        if any(not np.array_equal(a[k], b[k]) for k in ('sample_id', 'class_id', 'prediction', 'correct')):
            raise ValueError('Initial predictions mismatch')
    return dict(passed=True, fresh_initialization=True, identical_source=True,
        same_data_initialization_schedule=True, ordinary_B_matched=True,
        A_rounds=[[], list(range(1, 91))], extra_B=[False, False],
        total_compute_budget_matched=False, reference_root=str(right))


def prepare(args):
    from scripts import run_cliplora_joint_aggregation as single
    from tools.client_aggregation.protocol import code_hashes, write_weight_tables
    if (args.output_root/'experiment_plan.json').exists() or (args.output_root/FROZEN_RUN).exists():
        raise ValueError('Preserve existing registration and outputs')
    reference = load(args.reference_root/'experiment_plan.json')
    plan = frozen_plan(reference, code_hashes(REPO))
    single.preflight(plan)
    job = single.make_job(plan, 'frozen')
    command = single.command_for(job, args.output_root)
    if command[command.index('--sfra_variant')+1] != 's' or '--b_transfer_enable' in command:
        raise ValueError('Wrong frozen/no-B command')
    save(args.output_root/'experiment_plan.json', plan)
    write_weight_tables(args.output_root, plan['aggregation'])
    from scripts.run_cliplora_ab_parameter_search import EVALUATION
    save(args.output_root/'ablation_plan.json', dict(reference_root=str(args.reference_root),
        source_root=str(REPO), gpu=args.gpu, reused_run=str(args.reference_root/A_RUN),
        new_run=str(args.output_root/FROZEN_RUN), formal_rounds=100,
        A='initial A frozen throughout; no A optimization, history or correction',
        B='ordinary local B, 3 epochs every round; no supplemental B of either kind',
        endpoint='mean committed rounds81..100 Overall/Head20/Middle60/Tail20',
        evaluation=EVALUATION, auto_retry=False, rollback=False,
        interpretation='A training plus new-A correction versus freezing A; not correction-only attribution',
        launcher_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
    print('REGISTERED', args.output_root, flush=True)


def summarize(root):
    from scripts import run_cliplora_joint_aggregation as single
    from scripts.run_cliplora_ab_parameter_search import endpoint, actual_corrections
    from tools.client_aggregation.analysis import audit_run
    from tools.client_aggregation.protocol import read_csv, write_table
    suite = load(root/'ablation_plan.json')
    reference = Path(suite['reference_root'])
    pair = compare_startup(root, reference)
    rows = []
    for name, folder, arm, suffix in (('frozen_A_no_extra_B', root, 'frozen', FROZEN_RUN),
            ('new_A_no_extra_B', reference, 'method_a_margin', A_RUN)):
        run = folder/suffix
        result = audit_run(run, single.make_job(load(folder/'experiment_plan.json'), arm))
        b_records = read_csv(run/'b_transfer_rounds.csv') if (run/'b_transfer_rounds.csv').exists() else []
        if b_records or result['progress'].get('b_transfer_optimizer_steps', 0):
            raise ValueError('No-B arm contains transfer steps')
        a_records = read_csv(run/'sfra_rounds.csv')
        row = dict(method=name, **endpoint(result['curves']),
            normal_B_steps=result['progress']['normal_optimizer_steps'],
            A_local_steps=result['progress']['extra_optimizer_steps'],
            A_correction_steps=result['progress']['functional_correction_steps'],
            A_repair_seconds_total=sum(float(r.get('repair_seconds') or 0) for r in a_records),
            elapsed_seconds=result['progress']['elapsed_seconds'])
        if arm == 'method_a_margin':
            row.update(actual_corrections(run, root/'analysis/new_A_actual_corrections.csv'))
        rows.append(row)
    keys = ('overall_acc', 'head20_acc', 'middle60_acc', 'bottom20_tail_acc')
    pair['new_A_minus_frozen_A_pp'] = {k:rows[1][k]-rows[0][k] for k in keys}
    write_table(root/'analysis/performance.csv', rows)
    save(root/'analysis/report.json', dict(completed_and_audited=True, evaluation=suite['evaluation'],
        endpoint=suite['endpoint'], interpretation=suite['interpretation'], performance=rows, pair=pair))
    save(root/'finalization_receipt.json', dict(completed_and_audited=True, finished_at=time.time()))


def dispatch(args):
    from scripts import run_cliplora_joint_aggregation as single
    from scripts.run_ab_validation import file_lock
    root = args.output_root
    suite = load(root/'ablation_plan.json')
    with file_lock(root/'locks/dispatch.lock', timeout=.1):
        if (root/'worker_launch.json').exists() or (root/FROZEN_RUN).exists():
            raise ValueError('Refusing duplicate launch or automatic resume')
        if hashlib.sha256(Path(__file__).read_bytes()).hexdigest() != suite['launcher_sha256']:
            raise ValueError('Registered launcher changed')
        raw = subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True)
        free = {int(row[0]):(int(row[1]),int(row[2])) for row in (line.split(',') for line in raw.strip().splitlines())}
        gpu = suite['gpu']
        if free[gpu][0] < 17000 or free[gpu][1] > 10:
            raise ValueError('Registered GPU no longer idle; do not alter running experiments')
        command = [sys.executable,'-u',str(Path(__file__).resolve()),'--stage','worker','--output-root',str(root)]
        log = root/'launcher_logs/worker.log'
        log.parent.mkdir(parents=True,exist_ok=True)
        with log.open('x') as stream:
            proc = subprocess.Popen(command,cwd=REPO,env=single.server_environment(gpu),
                stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        save(root/'worker_launch.json', dict(pid=proc.pid,gpu=gpu,command=command,time=time.time()))
        reference = Path(suite['reference_root'])
        while True:
            progress = root/FROZEN_RUN/'progress.json'
            save(root/'status.json',dict(pid=proc.pid,running=proc.poll() is None,
                progress=load(progress) if progress.exists() else {}))
            if not (root/'startup_audit.json').exists() and (root/FROZEN_RUN/'predictions/r000.npz').exists():
                save(root/'startup_audit.json', compare_startup(root, reference))
            if proc.poll() is not None:
                if proc.returncode:
                    raise ValueError('Frozen worker failed; no automatic retry')
                break
            time.sleep(30)
        while not (reference/A_RUN/'joint_receipt.json').exists():
            time.sleep(30)
        summarize(root)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage', choices=('prepare','dispatch','worker','summary','startup-audit'), required=True)
    p.add_argument('--output-root', type=Path, required=True)
    p.add_argument('--reference-root', type=Path)
    p.add_argument('--gpu', type=int, default=3)
    args = p.parse_args()
    if args.stage == 'prepare':
        prepare(args)
    elif args.stage == 'worker':
        from scripts import run_cliplora_joint_aggregation as single
        plan = load(args.output_root/'experiment_plan.json')
        if (args.output_root/FROZEN_RUN).exists():
            raise ValueError('Fresh run required')
        single.preflight(plan,check_cuda=True)
        single.execute(single.make_job(plan,'frozen'),args.output_root)
    elif args.stage == 'dispatch':
        try:
            dispatch(args)
        except Exception:
            save(args.output_root/'failure_receipt.json',dict(time=time.time(),traceback=traceback.format_exc(),auto_retry=False))
            raise
    elif args.stage == 'summary':
        summarize(args.output_root)
    else:
        reference = Path(load(args.output_root/'ablation_plan.json')['reference_root'])
        save(args.output_root/'startup_audit.json',compare_startup(args.output_root,reference))


if __name__ == '__main__':
    main()
