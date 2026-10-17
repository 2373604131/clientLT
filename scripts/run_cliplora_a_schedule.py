"""Three fresh100 A-schedule experiments, retaining the original donor-C B."""
import argparse
import csv
import io
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
from scripts.run_cliplora_ab_parameter_search import endpoint, save, IDENTITIES, EVALUATION
from scripts.run_cliplora_new_a_old_b import startup_audit
from tools.client_aggregation.protocol import code_hashes, load_json, read_csv, write_table, write_weight_tables
from tools.client_aggregation.a_schedule import SCHEDULES, candidate_plan, rounds_for, FILES

RUN = 'runs/seed42/method_a_margin'
CHANGES = FILES | {'tools/client_aggregation/protocol.py', 'tools/client_aggregation/runtime.py',
    'tools/client_aggregation/analysis.py', 'tools/client_aggregation/new_a_old_b.py'}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    p.add_argument('--stage', choices=('prepare','dispatch','worker','status','summary','startup-audit'), required=True)
    p.add_argument('--output-root', type=Path, required=True)
    p.add_argument('--reference-root', type=Path, help='Existing full90 A + same old B run, allowed still running')
    p.add_argument('--gpus', type=int, nargs=3, default=[0,1,2])
    p.add_argument('--cell', choices=SCHEDULES)
    return p.parse_args(argv)


def prepare(args):
    root, reference = args.output_root.resolve(), args.reference_root.resolve()
    if (root/'schedule_plan.json').exists() or any((root/c/RUN).exists() for c in SCHEDULES):
        raise ValueError('Occupied experiment root; preserve it')
    ref = load_json(reference/'experiment_plan.json')
    if ref['settings'].get('b_comparison_experiment') != 'new_a_old_b_joint_v1':
        raise ValueError('Reference must be the same selected-A original donor-C B trajectory')
    if not load_json(reference/'startup_audit.json')['passed']:
        raise ValueError('Reference startup was not audited')
    from tools.client_aggregation.protocol import audit_runtime_config
    audit_runtime_config(reference/RUN, single.make_job(ref, 'method_a_margin'))
    hashes = code_hashes(REPO)
    changes = sorted(k for k in set(hashes)|set(ref['code_sha256']) if hashes.get(k) != ref['code_sha256'].get(k))
    if set(changes)-CHANGES:
        raise ValueError('Unexpected source differences: '+str(set(changes)-CHANGES))
    meta = load_json(reference/RUN/'bridge_metadata.json')
    identity = {k:meta[k] for k in IDENTITIES}
    suite = dict(experiment='selected_a_schedule_old_b_v1', source=str(REPO),
        reference_root=str(reference), reference_completed_at_registration=load_json(reference/RUN/'progress.json')['completed_round'],
        evaluation=EVALUATION, endpoint='committed rounds81..100 mean Overall/Head20/Middle60/Tail20',
        exploratory_schedule_choice='A60 chosen after inspecting the prior test trajectory; development experiment, not independently validated early stopping',
        source_differences=changes, reference_identity=identity, groups={}, registered_at=time.time(),
        execution='Three fresh100 runs; independent histories; no automatic resume/retry/pruning; reference read only',
        history_policy='Five consecutive calendar-round windows through the last scheduled A update; no history or A gradients for fully frozen A',
        budgets='Ordinary B identical; A local training, A gradients, and history-observation costs intentionally differ')
    if len(set(args.gpus)) != 3 or min(args.gpus)<0:
        raise ValueError('Need three distinct GPUs')
    for (cell, rounds), gpu in zip(SCHEDULES.items(), args.gpus):
        folder = root/cell
        plan = candidate_plan(ref, cell, hashes)
        if plan['contract']['old_B'] != ref['contract']['old_B'] or plan['aggregation'] != ref['aggregation']:
            raise ValueError('B or aggregation changed')
        single.preflight(plan)
        command = single.command_for(single.make_job(plan, 'method_a_margin'), folder)
        for flag, expected in (('--a_margin_keep_weight',1.),('--a_margin_learn_weight',3.),('--a_margin_correction_step_size',1.)):
            if float(command[command.index(flag)+1]) != expected:
                raise ValueError('S4 argument mismatch: '+flag)
        if command[command.index('--sfra_resume')+1] or '--b_transfer_enable' in command:
            raise ValueError('Unexpected resume/legacy activation')
        save(folder/'experiment_plan.json', plan)
        save(folder/'reference_audit.json', dict(references=[dict(root=str(reference), identity=identity)],
            evaluation=EVALUATION, role='same initialization/data and old-B configuration; A schedule intentionally changes'))
        write_weight_tables(folder, plan['aggregation'])
        suite['groups'][cell] = dict(root=str(folder), gpu=gpu, A_rounds=rounds,
            local_A_steps=plan['contract']['extra_A_steps'], max_correction_steps=3*len(rounds))
    save(root/'schedule_plan.json', suite)
    print(json.dumps(suite['groups'], indent=2), flush=True)


def actual_corrections(folder, job, destination):
    """Existing snapshots only; off-schedule A contributes exactly zero."""
    import torch
    torch.set_num_threads(1)
    run = folder/RUN
    cfg = load_json(run/'sfra_config.json')
    args = load_json(run/'bridge_metadata.json')['resolved_args']
    scale = float(args['cliplora_alpha'])/math.sqrt(int(args['cliplora_rank']))
    active = set(rounds_for(job))
    rows = []
    for rnd in range(1,101):
        da2=dw2=ordinary2=0.
        if rnd in active:
            ordinary = torch.load(run/'events'/f'r{rnd:03d}_c000_main_refresh_A/state.pt', map_location='cpu', weights_only=False)
            after = ordinary['actual_after_lora_state']
            before = ordinary['anchor_lora_state']
            submitted = torch.load(run/'events'/f'r{rnd+1:03d}_c000_main_normal_B/state.pt', map_location='cpu', weights_only=False)['anchor_lora_state']
            for key in cfg['a_parameter_order']:
                delta = submitted[key].double()-after[key].double()
                da2 += float(delta.square().sum())
                ordinary2 += float((after[key].double()-before[key].double()).square().sum())
                bkey = key.replace('lora_A','lora_B')
                if not torch.equal(submitted[bkey], after[bkey]):
                    raise ValueError('A correction changed B')
                b = after[bkey].double()
                dw2 += scale**2*float(((b.T@b)*(delta@delta.T)).sum())
        rows.append(dict(round=rnd, A_open=rnd in active, actual_A_correction_norm=math.sqrt(da2),
            ordinary_A_update_norm=math.sqrt(ordinary2), effective_LoRA_correction_norm=math.sqrt(max(0.,dw2)),
            correction_to_ordinary_A=math.sqrt(da2/ordinary2) if ordinary2 else 0.))
    write_table(destination, rows)
    chosen = [r for r in rows if r['A_open']]
    return dict(A_update_rounds=len(active),
        actual_A_correction_norm_mean_active_rounds=mean(r['actual_A_correction_norm'] for r in chosen) if chosen else 0.,
        correction_to_ordinary_A_mean_active_rounds=mean(r['correction_to_ordinary_A'] for r in chosen) if chosen else 0.,
        effective_LoRA_correction_norm_mean_active_rounds=mean(r['effective_LoRA_correction_norm'] for r in chosen) if chosen else 0.)


def summarize(root, require_reference=False):
    from tools.client_aggregation.analysis import audit_run
    from tools.client_aggregation.frozen_tailrw import paired_audit
    root = Path(root)
    suite = load_json(root/'schedule_plan.json')
    out = root/'analysis'
    out.mkdir(parents=True, exist_ok=True)
    reference = Path(suite['reference_root'])
    sources = {c:root/c for c in SCHEDULES}
    if (reference/RUN/'completion.json').exists() and (reference/RUN/'joint_receipt.json').exists():
        sources['a_until90_reference'] = reference
    elif require_reference:
        raise ValueError('Full90 same-B reference is not yet complete')
    results, rows = {}, []
    for cell, folder in sources.items():
        plan = load_json(folder/'experiment_plan.json')
        job = single.make_job(plan, 'method_a_margin')
        result = audit_run(folder/RUN, job)
        results[cell] = result
        if any(result['metadata'][k] != suite['reference_identity'][k] for k in IDENTITIES):
            raise ValueError('Reference initialization/data identity differs')
        if cell in SCHEDULES:
            startup_audit(folder)
        a_rows = read_csv(folder/RUN/'sfra_rounds.csv')
        b_rows = read_csv(folder/RUN/'b_transfer_rounds.csv')
        scores = endpoint(result['curves'])
        rows.append(dict(cell=cell, **scores,
            **actual_corrections(folder,job,out/(cell+'_actual_A_corrections.csv')),
            A_repair_seconds_total=sum(float(r.get('repair_seconds') or 0) for r in a_rows),
            history_only_seconds_total=sum(float(r.get('history_observation_seconds') or 0) for r in a_rows),
            B_transfer_seconds_total=sum(float(r['seconds']) for r in b_rows),
            elapsed_seconds=result['progress']['elapsed_seconds'],
            local_A_steps=result['progress']['extra_optimizer_steps'],
            local_B_steps=result['progress']['normal_optimizer_steps']))
        write_table(out/(cell+'_A_losses.csv'),a_rows)
        write_table(out/(cell+'_B_transfers.csv'),b_rows)
    comparisons = []
    if 'a_until90_reference' in results:
        baseline = results['a_until90_reference']
        for cell in SCHEDULES:
            candidate = results[cell]
            pair = paired_audit(candidate, baseline)
            allowed = {'base.candidate_rounds','base.extra_steps_expected','base.extra_trainable_factor',
                       'refresh_rounds','budget.extra_optimizer_steps','budget.functional_correction_steps',
                       *('code.'+p for p in CHANGES)}
            pair['mismatches'] = [m for m in pair['mismatches'] if m not in allowed]
            if candidate['config']['b_transfer'] != baseline['config']['b_transfer'] or candidate['job']['aggregation'] != baseline['job']['aggregation']:
                pair['mismatches'].append('B_or_aggregation')
            pair.update(cell=cell, status='mismatch' if pair['mismatches'] else 'eligible',
                contrast='changed A schedule minus full90 A, with identical archived B',
                budget_matched=False, local_A_budget_matched=False, local_B_budget_matched=True,
                total_compute_budget_matched=False)
            if pair['mismatches']:
                save(out/'pair_audit.json', comparisons+[pair])
                raise ValueError('Pair mismatch: '+str(pair['mismatches']))
            l,r=endpoint(candidate['curves']),endpoint(baseline['curves'])
            pair['delta_pp']={k:l[k]-r[k] for k in l}
            comparisons.append(pair)
    write_table(out/'performance.csv', rows)
    report = dict(three_new_runs_completed_and_audited=True, reference_complete='a_until90_reference' in results,
        evaluation=EVALUATION, endpoint=suite['endpoint'], performance=rows, comparisons=comparisons,
        exploratory_schedule_choice=suite['exploratory_schedule_choice'], budget_note=suite['budgets'])
    save(out/'report.json', report)
    save(root/'finalization_receipt.json', dict(three_new_runs_completed_and_audited=True,
        full_reference_comparison_complete=report['reference_complete'], finished_at=time.time(), reporting_only=True))
    return report


def dispatch(root):
    suite=load_json(root/'schedule_plan.json')
    with file_lock(root/'locks/dispatch.lock', timeout=.1):
        if (root/'dispatch_receipt.json').exists() or any((root/c/RUN).exists() for c in SCHEDULES):
            raise ValueError('Refusing duplicate launch/resume')
        raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True)
        gpus={int(r[0]):dict(free_mib=int(r[1]),utilization=int(r[2])) for r in csv.reader(io.StringIO(raw))}
        if any(gpus[g['gpu']]['free_mib']<17000 or gpus[g['gpu']]['utilization']>10 for g in suite['groups'].values()):
            raise ValueError('Assigned GPUs are not sufficiently free')
        save(root/'dispatch_receipt.json',dict(pid=os.getpid(),launched_at=time.time(),gpu_before=gpus))
        processes,streams,status={},{},{}
        try:
            for cell,item in suite['groups'].items():
                log=root/'launcher_logs'/(cell+'.log')
                log.parent.mkdir(parents=True,exist_ok=True)
                streams[cell]=log.open('x')
                command=[sys.executable,'-u',str(Path(__file__).resolve()),'--stage','worker','--output-root',str(root),'--cell',cell]
                p=subprocess.Popen(command,cwd=REPO,env=single.server_environment(item['gpu']),stdin=subprocess.DEVNULL,
                    stdout=streams[cell],stderr=subprocess.STDOUT,start_new_session=True)
                processes[cell]=p
                status[cell]=dict(pid=p.pid,gpu=item['gpu'],command=command,log=str(log),state='running')
                save(root/'status.json',status)
            while True:
                for cell,p in processes.items():
                    code=p.poll()
                    status[cell].update(state='running' if code is None else 'completed' if code==0 else 'failed',exit_code=code)
                    run=root/cell/RUN
                    if (run/'progress.json').exists():
                        status[cell]['progress']=load_json(run/'progress.json')
                    if not (root/cell/'startup_audit.json').exists() and (run/'predictions/r000.npz').exists():
                        startup_audit(root/cell)
                save(root/'status.json',status)
                if all(p.poll() is not None for p in processes.values()):
                    break
                time.sleep(30)
            if any(p.returncode for p in processes.values()):
                raise ValueError('A worker failed; no automatic retry')
            report=summarize(root)
            if not report['reference_complete']:
                reference=Path(suite['reference_root'])/RUN
                save(root/'waiting_for_reference.json',dict(reference=str(reference),training_finished=True,reporting_only=True))
                while not (reference/'joint_receipt.json').exists():
                    time.sleep(30)
                if load_json(reference/'joint_receipt.json')['exit_code'] != 0:
                    raise ValueError('Existing full90 reference failed; new results preserved')
                summarize(root,require_reference=True)
        except Exception:
            save(root/'failure_receipt.json',dict(time=time.time(),traceback=traceback.format_exc(),training_restarted=False))
            raise
        finally:
            for stream in streams.values():
                stream.close()


def main(argv=None):
    args=parse_args(argv)
    root=args.output_root.resolve()
    os.chdir(REPO)
    if args.stage=='prepare':
        if args.reference_root is None:
            raise ValueError('Preparation requires a same-B reference')
        prepare(args)
    elif args.stage=='dispatch':
        dispatch(root)
    elif args.stage=='worker':
        if args.cell is None:
            raise ValueError('Missing cell')
        folder=root/args.cell
        if (folder/RUN).exists():
            raise ValueError('Worker requires a fresh initialization')
        plan=load_json(folder/'experiment_plan.json')
        single.preflight(plan,check_cuda=True)
        single.execute(single.make_job(plan,'method_a_margin'),folder)
    elif args.stage=='startup-audit':
        for cell in SCHEDULES:
            startup_audit(root/cell)
    elif args.stage=='summary':
        print(json.dumps(summarize(root),indent=2))
    else:
        print(json.dumps(load_json(root/'status.json'),indent=2))


if __name__=='__main__':
    main()
