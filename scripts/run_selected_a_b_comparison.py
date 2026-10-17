"""Freeze completed S4 selection and compare absent, archived, and current B."""
import argparse
import copy
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
from types import SimpleNamespace

CELLS = ('new_a_old_b_joint', 'new_a_joint')
RUN = 'runs/seed42/method_a_margin'


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    p.add_argument('--stage', choices=('prepare', 'dispatch', 'worker', 'monitor', 'startup-audit', 'summary', 'status'), required=True)
    p.add_argument('--source-root', type=Path, required=True)
    p.add_argument('--output-root', type=Path, required=True)
    p.add_argument('--search-root', type=Path, required=True)
    p.add_argument('--gpus', type=int, nargs=2, default=[5, 4])
    p.add_argument('--cell', choices=CELLS)
    return p.parse_args()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def no_b_plan(reference, hashes):
    from tools.client_aggregation.ab_factorial import contract_for
    plan = copy.deepcopy(reference)
    plan['settings']['factorial_cell'] = 'new_a_joint'
    plan['contract'] = contract_for('new_a_joint', plan['settings']['margin_parameters'])
    plan['code_sha256'] = hashes
    if plan['contract']['method_a_margin'] != reference['contract']['method_a_margin']:
        raise ValueError('Removing extra B changed A')
    return plan


def prepare(args):
    from scripts import run_cliplora_joint_aggregation as single
    from scripts import run_cliplora_new_a_old_b as old
    from scripts.run_cliplora_ab_parameter_search import dominates_baseline
    from tools.client_aggregation.protocol import code_hashes, write_weight_tables
    from tools.client_aggregation.new_a_old_b import EVALUATION
    root, search = args.output_root, args.search_root
    if (root/'comparison_plan.json').exists() or any((root/c/RUN).exists() for c in CELLS):
        raise ValueError('Comparison already registered or occupied; preserve its outputs')
    receipt = load_json(search/'finalization_receipt.json')
    report = load_json(search/'analysis/report.json')
    if not receipt.get('all_six_completed_and_audited') or not report.get('all_six_completed_and_audited'):
        raise ValueError('Wait for all six completed, audited trajectories')
    rows = {r['group']: r for r in report['rows']}
    eligible = [g for g in ('S1', 'S2', 'S3', 'S4', 'S5', 'S6') if dominates_baseline(rows[g], rows['baseline'])]
    if eligible != ['S4'] or report['qualifying_groups_by_tail'] != ['S4']:
        raise ValueError('The expected unique S4 selection does not match complete results')
    reference = search/'S4'
    source, audited = old.load_reference(reference)
    if old.endpoint(audited) != {k: rows['S4'][k] for k in old.endpoint(audited)}:
        raise ValueError('Selected endpoint differs from audited S4 predictions')
    params = source['settings']['margin_parameters']
    if (params['keep_weight'], params['learn_weight'], params['correction_step_size']) != (1., 3., 1.):
        raise ValueError('Wrong selected A parameters')
    # Reuse the guarded old-B registration; it verifies all permitted source differences.
    old_args = SimpleNamespace(output_root=root/CELLS[0], reference_a=reference, compare_root=[])
    old_plan = old.register(old_args)
    plain_plan = no_b_plan(source, code_hashes(args.source_root))
    plans = {CELLS[0]: old_plan, CELLS[1]: plain_plan}
    for cell, plan in plans.items():
        single.preflight(plan)
        job = single.make_job(plan, 'method_a_margin')
        command = single.command_for(job, root/cell)
        for flag, value in (('--a_margin_keep_weight', 1.), ('--a_margin_learn_weight', 3.),
                            ('--a_margin_correction_step_size', 1.)):
            if float(command[command.index(flag)+1]) != value:
                raise ValueError('Selected parameter is not wired into command: '+flag)
        if command[command.index('--sfra_resume')+1] or '--b_transfer_enable' in command:
            raise ValueError('Unexpected resume or legacy B CLI activation')
        save(root/cell/'experiment_plan.json', plan)
        write_weight_tables(root/cell, plan['aggregation'])
    no_audit = copy.deepcopy(load_json(root/CELLS[0]/'reference_audit.json'))
    no_audit.update(B=None, execution='one fresh formal100; no supplemental B; independently initialized history')
    save(root/CELLS[1]/'reference_audit.json', no_audit)
    smoke = Path('/data/yzh/clientLT/output/cifar100_LT/new_a_old_b_joint_20261014_01/smoke/seed42/method_a_margin')
    if load_json(smoke/'smoke_complete.json')['completed_round'] != 1:
        raise ValueError('Previous CUDA integration check is missing')
    if load_json(smoke/'joint_job.json')['code_sha256'] != code_hashes(args.source_root):
        raise ValueError('Integration code changed after its CUDA check')
    suite = dict(source_root=str(args.source_root), search_root=str(search),
        selected_group='S4', selection=report['selection'], evaluation=EVALUATION,
        selected_result=rows['S4'], baseline_result=rows['baseline'],
        evidence={str(search/p): sha(search/p) for p in ('finalization_receipt.json', 'analysis/report.json', 'analysis/summary.csv')},
        endpoint='mean of committed rounds81..100; Overall, Head20, Middle60, Tail20',
        reused_current_B_root=str(reference), A=source['contract']['method_a_margin'],
        groups={c: dict(root=str(root/c), gpu=g) for c,g in zip(CELLS,args.gpus)},
        limitations='Single seed; tiny development score differences do not establish significance or global optimality.',
        execution='Two fresh formal100 runs; independent histories; no pruning, auto retry, rollback, or extra parameter sweep.',
        validation='Same source already passed 34 CPU tests and one CUDA old-B integration round; S4 A parameters completed100. No repeated smoke trajectory.',
        previous_integration_smoke=str(smoke), launcher_sha256=sha(__file__),
        registered_at=time.time())
    save(root/'comparison_plan.json', suite)
    print(json.dumps(dict(selected='S4', groups=suite['groups'], parameters=params), indent=2), flush=True)


def startup_audit(folder):
    """Check the same identities with optional B absent from sfra_config."""
    import numpy as np
    from scripts import run_cliplora_joint_aggregation as single
    from scripts.run_cliplora_new_a_old_b import IDENTITIES
    from tools.client_aggregation.protocol import audit_runtime_config
    folder = Path(folder)
    plan = load_json(folder/'experiment_plan.json')
    run = folder/RUN
    cfg = audit_runtime_config(run, single.make_job(plan, 'method_a_margin'))
    meta = load_json(run/'bridge_metadata.json')
    cmd = load_json(run/'command.json')
    if cmd[cmd.index('--sfra_resume')+1]:
        raise ValueError('Fresh initialization required')
    checks = []
    for ref in load_json(folder/'reference_audit.json')['references']:
        ref_run = Path(ref['root'])/RUN
        if any(meta[k] != ref['identity'][k] for k in IDENTITIES):
            raise ValueError('Startup initialization/data/schedule mismatch')
        for name in ('private_witness_manifest.json', 'margin_witness_manifest.json'):
            if load_json(run/name) != load_json(ref_run/name):
                raise ValueError('Witness mismatch: '+name)
        with np.load(run/'predictions/r000.npz') as a, np.load(ref_run/'predictions/r000.npz') as b:
            for key in ('sample_id', 'class_id', 'prediction', 'correct'):
                if not np.array_equal(a[key], b[key]):
                    raise ValueError('Initial predictions mismatch: '+key)
        checks.append(dict(reference=ref['root'], passed=True))
    result = dict(passed=True, fresh_initialization=True, independent_history=True,
                  references=checks, A=plan['contract']['method_a_margin'], B=cfg.get('b_transfer'))
    save(folder/'startup_audit.json', result)
    print(json.dumps(dict(root=str(folder), passed=True, extra_B=bool(result['B']))), flush=True)
    return result


def summarize(args):
    from scripts import run_cliplora_new_a_old_b as old
    from scripts import run_cliplora_joint_aggregation as single
    from scripts.run_cliplora_ab_parameter_search import actual_corrections
    from tools.client_aggregation.analysis import audit_run
    from tools.client_aggregation.protocol import read_csv, write_table
    from tools.client_aggregation.new_a_old_b import pair_check
    root = args.output_root
    (root/'analysis').mkdir(parents=True, exist_ok=True)
    suite = load_json(root/'comparison_plan.json')
    results, rows = {}, []
    sources = {**{c: root/c for c in CELLS}, 'new_a_new_b_joint': Path(suite['reused_current_B_root'])}
    for cell, folder in sources.items():
        plan = load_json(folder/'experiment_plan.json')
        result = audit_run(folder/RUN, single.make_job(plan, 'method_a_margin'))
        results[cell] = result
        if cell in CELLS:
            startup_audit(folder)
        a_rows = read_csv(folder/RUN/'sfra_rounds.csv')
        b_path = folder/RUN/'b_transfer_rounds.csv'
        if cell == 'new_a_joint':
            b_rows = read_csv(b_path) if b_path.exists() else []
            if b_rows:
                raise ValueError('No-B comparison unexpectedly contains B transfer events')
        else:
            b_rows = read_csv(b_path)
        rows.append(dict(method=cell, **old.endpoint(result),
            A_repair_seconds_total=sum(float(r.get('repair_seconds') or 0) for r in a_rows),
            B_transfer_seconds_total=sum(float(r['seconds']) for r in b_rows),
            B_effective_LoRA_transfer_norm_mean=(sum(float(r['effective_global_transfer_norm']) for r in b_rows)/len(b_rows) if b_rows else 0.),
            elapsed_seconds=result['progress']['elapsed_seconds'],
            **actual_corrections(folder/RUN, root/'analysis'/(cell+'_actual_A_corrections.csv'))))
    pairs = []
    for left, right in ((CELLS[0], CELLS[1]), (CELLS[0], 'new_a_new_b_joint'), ('new_a_new_b_joint', CELLS[1])):
        pair = pair_check(results[left], results[right])
        pair.update(left=left, right=right, contrast=left+' minus '+right)
        if pair['status'] == 'mismatch':
            save(root/'analysis/pair_audit.json', pairs+[pair])
            raise ValueError('Final comparison audit failed: '+str(pair['mismatches']))
        l, r = old.endpoint(results[left]), old.endpoint(results[right])
        pair['delta_pp'] = {k:l[k]-r[k] for k in l}
        pairs.append(pair)
    write_table(root/'analysis/performance.csv', rows)
    save(root/'analysis/report.json', dict(all_three_completed_and_audited=True,
        evaluation=suite['evaluation'], selection=suite['selection'], selected_group='S4',
        endpoint=suite['endpoint'], performance=rows, comparisons=pairs,
        local_SGD_matched=True, total_compute_budget_matched=False, independent_seeds=1))
    save(root/'finalization_receipt.json', dict(all_three_completed_and_audited=True, reporting_only=True, finished_at=time.time()))


def monitor(args):
    """Observe already-started workers; this path never starts or restarts training."""
    root = args.output_root
    status = load_json(root/'status.json')
    try:
        while True:
            for cell in CELLS:
                entry = status[cell]
                proc = Path('/proc')/str(entry['pid'])
                running = False
                if proc.exists():
                    stat = (proc/'stat').read_text().rsplit(')', 1)[1].split()
                    command = (proc/'cmdline').read_bytes().replace(b'\0', b' ').decode()
                    running = stat[0] != 'Z' and '--stage worker' in command and str(root) in command and cell in command
                folder = root/cell/RUN
                progress = folder/'progress.json'
                if progress.exists():
                    entry['progress'] = load_json(progress)
                receipt = folder/'joint_receipt.json'
                exit_code = load_json(receipt)['exit_code'] if receipt.exists() else None
                complete = (folder/'completion.json').exists() and exit_code == 0
                entry.update(state='running' if running else 'completed' if complete else 'failed', exit_code=None if running else exit_code)
                if not (root/cell/'startup_audit.json').exists() and (folder/'predictions/r000.npz').exists():
                    startup_audit(root/cell)
            save(root/'status.json', status)
            if all(s['state'] != 'running' for s in status.values()):
                break
            time.sleep(30)
        if any(s['state'] != 'completed' for s in status.values()):
            raise ValueError('Worker failed; no automatic restart')
        summarize(args)
    except Exception:
        save(root/'reporting_failure.json', dict(time=time.time(), traceback=traceback.format_exc(), training_restarted=False))
        raise


def dispatch(args):
    from scripts.run_ab_validation import file_lock
    root = args.output_root
    suite = load_json(root/'comparison_plan.json')
    if sha(__file__) != suite['launcher_sha256']:
        raise ValueError('Registered orchestration source changed')
    with file_lock(root/'locks/dispatch.lock', timeout=.1):
        if (root/'dispatch_receipt.json').exists() or any((root/c/RUN).exists() for c in CELLS):
            raise ValueError('Refusing duplicate launch/resume')
        raw = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.free,utilization.gpu', '--format=csv,noheader,nounits'], text=True)
        gpus = {int(r[0]):dict(free_mib=int(r[1]), utilization=int(r[2])) for r in csv.reader(io.StringIO(raw))}
        assigned = [suite['groups'][c]['gpu'] for c in CELLS]
        if len(set(assigned)) != 2 or any(gpus[g]['free_mib'] < 17000 or gpus[g]['utilization'] > 10 for g in assigned):
            raise ValueError('Assigned GPUs are no longer free; do not alter other jobs')
        save(root/'dispatch_receipt.json', dict(started_at=time.time(), gpus=gpus, pid=os.getpid(), auto_retry=False))
        processes, status, streams = {}, {}, []
        try:
            for cell in CELLS:
                gpu = suite['groups'][cell]['gpu']
                command = [sys.executable, '-u', str(Path(__file__).resolve()), '--stage', 'worker',
                    '--source-root', str(args.source_root), '--output-root', str(root),
                    '--search-root', str(args.search_root), '--cell', cell]
                log = root/'launcher_logs'/(cell+'.log')
                log.parent.mkdir(parents=True, exist_ok=True)
                stream = log.open('x', encoding='utf-8')
                streams.append(stream)
                proc = subprocess.Popen(command, cwd=args.source_root,
                    env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), CUDA_DEVICE_ORDER='PCI_BUS_ID'),
                    stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
                processes[cell] = proc
                status[cell] = dict(pid=proc.pid, gpu=gpu, state='running', launched_at=time.time(), command=command, log=str(log))
                save(root/'status.json', status)
            while True:
                for cell, proc in processes.items():
                    code = proc.poll()
                    status[cell].update(state='running' if code is None else 'completed' if code==0 else 'failed', exit_code=code)
                    progress = root/cell/RUN/'progress.json'
                    if progress.exists():
                        status[cell]['progress'] = load_json(progress)
                save(root/'status.json', status)
                if all(p.poll() is not None for p in processes.values()):
                    break
                time.sleep(30)
            if any(p.returncode for p in processes.values()):
                raise ValueError('At least one worker failed; outputs preserved, no automatic retry')
            summarize(args)
        except Exception:
            save(root/'failure_receipt.json', dict(time=time.time(), traceback=traceback.format_exc(), training_restarted=False))
            raise
        finally:
            for stream in streams:
                stream.close()


def main():
    args = parse_args()
    for key in ('source_root', 'output_root', 'search_root'):
        setattr(args, key, getattr(args, key).resolve())
    sys.path.insert(0, str(args.source_root))
    os.chdir(args.source_root)
    global load_json, save
    from tools.client_aggregation.protocol import load_json
    from scripts.run_cliplora_ab_parameter_search import save
    if args.stage == 'prepare':
        prepare(args)
    elif args.stage == 'dispatch':
        dispatch(args)
    elif args.stage == 'monitor':
        monitor(args)
    elif args.stage == 'worker':
        from scripts import run_cliplora_joint_aggregation as single
        if not args.cell:
            raise ValueError('Worker requires a cell')
        folder = args.output_root/args.cell
        if (folder/RUN).exists():
            raise ValueError('Fresh initialization required; no automatic resume')
        suite = load_json(args.output_root/'comparison_plan.json')
        if sha(__file__) != suite['launcher_sha256']:
            raise ValueError('Orchestration source changed')
        plan = load_json(folder/'experiment_plan.json')
        single.preflight(plan, check_cuda=True)
        single.execute(single.make_job(plan, 'method_a_margin'), folder)
    elif args.stage == 'startup-audit':
        for cell in CELLS:
            startup_audit(args.output_root/cell)
    elif args.stage == 'summary':
        summarize(args)
    else:
        print(json.dumps(load_json(args.output_root/'status.json'), indent=2))


if __name__ == '__main__':
    main()
