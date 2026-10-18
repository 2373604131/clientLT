"""Six FedAvg ablations: method A includes LA; plain A uses CE; two jobs per GPU."""
import argparse
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from scripts import run_cliplora_joint_aggregation as single
from scripts.run_ab_validation import file_lock, write_json
from tools.client_aggregation.protocol import (SCHEMA, PAIR_HASHES, aggregation_spec,
    code_hashes, load_json, write_table, write_weight_tables)
from tools.client_aggregation.six_ablation import CELLS, EXPERIMENT, contract_for, losses_for


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    p.add_argument('--stage', choices=('plan', 'preflight', 'server', 'smoke', 'run', 'status', 'summary'), default='plan')
    p.add_argument('--cells', nargs='+', choices=tuple(CELLS), default=list(CELLS))
    p.add_argument('--gpus', nargs=3, type=int, default=[0, 1, 2],
        help='Physical GPU IDs for frozen-A pair, plain-A pair, method-A pair')
    p.add_argument('--client-concurrency', type=int, default=10,
        help='Concurrent clients PER EXPERIMENT (2..30); two experiments share each GPU')
    p.add_argument('--num-workers', type=int, default=2)
    p.add_argument('--reference-run', type=Path, default=Path('references/full10_clientlt'))
    p.add_argument('--data-root', type=Path, default=Path('DATA'))
    p.add_argument('--output-root', type=Path)
    p.add_argument('--stop-after-round', type=int, default=0)
    args = p.parse_args(argv)
    if len(set(args.gpus)) != 3 or min(args.gpus) < 0:
        p.error('Use three distinct nonnegative physical GPU IDs')
    if not 2 <= args.client_concurrency <= 30:
        p.error('--client-concurrency must be 2..30; every cell uses parallel clients')
    if len(set(args.cells)) != len(args.cells) or args.num_workers < 0 or not 0 <= args.stop_after_round <= 100:
        p.error('Duplicate cells, negative workers or invalid stopping round')
    if args.stage == 'run' and len(args.cells) != 1:
        p.error('Use --stage server for all six; --stage run requires exactly one --cells value')
    if args.output_root is None:
        args.output_root = Path(f'output/cifar100_LT/six_ablation_methodA_LA_fedavg_parallel{args.client_concurrency}')
    return args


def assignment(args):
    return {cell: args.gpus[list(CELLS).index(cell) % 3] for cell in args.cells}


def register(args):
    root = args.output_root.resolve()
    settings = dict(seed=42, reference_run=str(args.reference_run.resolve()),
        data_root=str(args.data_root.resolve()), num_workers=args.num_workers,
        client_concurrency=args.client_concurrency, parallel_validation='off',
        aggregation_rule='fedavg', lambda_value=None, six_ablation_experiment=EXPERIMENT)
    spec = aggregation_spec(args.reference_run, .1, 'fedavg')
    hashes = code_hashes(REPO)
    suite = dict(experiment=EXPERIMENT, settings=settings, code_sha256=hashes,
        cells={c: dict(arm=arm, method_B=b, root=c) for c, (arm, b) in CELLS.items()})
    with file_lock(root/'locks/registration.lock'):
        path = root/'six_ablation_plan.json'
        if path.exists() and load_json(path) != suite:
            raise ValueError('Registered settings/source changed. Preserve existing results and use a new --output-root.')
        plans = {}
        # Validate every existing registration before creating missing cells.
        for cell, (arm, _) in CELLS.items():
            plan = dict(schema=SCHEMA, contract=contract_for(cell), code_sha256=hashes,
                aggregation=spec, settings=dict(settings, six_ablation_cell=cell))
            single.make_job(plan, arm)
            path = root/cell/'experiment_plan.json'
            if path.exists() and load_json(path) != plan:
                raise ValueError('Cell registration changed: '+cell)
            plans[cell] = plan
        for cell, plan in plans.items():
            path = root/cell/'experiment_plan.json'
            if not path.exists():
                write_json(path, plan)
                write_weight_tables(root/cell, spec)
        write_json(root/'six_ablation_plan.json', suite)
    return suite


def child_command(args, cell, smoke=False):
    return [sys.executable, '-u', str(REPO/'scripts/run_six_ablation.py'),
        '--stage', 'run', '--cells', cell, '--gpus', *map(str, args.gpus),
        '--client-concurrency', str(args.client_concurrency), '--num-workers', str(args.num_workers),
        '--reference-run', str(args.reference_run.resolve()), '--data-root', str(args.data_root.resolve()),
        '--output-root', str(args.output_root.resolve()),
        '--stop-after-round', str(args.stop_after_round)] + (['--smoke-worker'] if smoke else [])


def stop_owned(process):
    if process.poll() is not None:
        return
    if os.name == 'nt':
        subprocess.run(['taskkill', '/PID', str(process.pid), '/T', '/F'],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
    else:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            return
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
    process.wait()


def dispatch(args):
    root = args.output_root.resolve()
    jobs = assignment(args)
    commands = {cell: child_command(args, cell, args.stage == 'smoke') for cell in jobs}
    with file_lock(root/'locks/server_dispatch.lock', timeout=.1):
        write_json(root/'server_launch.json', dict(gpu_assignment=jobs,
            client_concurrency_per_experiment=args.client_concurrency,
            maximum_client_tasks_per_gpu=2*args.client_concurrency, commands=commands,
            note='Six concurrent processes, one visible GPU each; independent smoke before formal training'))
        processes, streams, results = {}, [], {}
        try:
            for cell, gpu in jobs.items():
                log = root/'launcher_logs'/f'{cell}_server.log'
                log.parent.mkdir(parents=True, exist_ok=True)
                stream = log.open('a', encoding='utf-8')
                streams.append(stream)
                stream.write('\n'+shlex.join(commands[cell])+'\n'); stream.flush()
                group = (dict(creationflags=subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW)
                    if os.name == 'nt' else dict(start_new_session=True))
                processes[cell] = subprocess.Popen(commands[cell], cwd=REPO,
                    env=single.server_environment(gpu), stdout=stream, stderr=subprocess.STDOUT, **group)
                print(f'START {cell}: GPU={gpu}, clients={args.client_concurrency}, PID={processes[cell].pid}, log={log}', flush=True)
            while len(results) < len(processes):
                for cell, process in processes.items():
                    if cell not in results and process.poll() is not None:
                        results[cell] = process.returncode
                        print(f'EXIT {cell}: {process.returncode}', flush=True)
                if len(results) < len(processes):
                    time.sleep(.5)
        finally:
            for process in processes.values():
                stop_owned(process)
            for stream in streams:
                stream.close()
        failed = [cell for cell, code in results.items() if code]
        if failed:
            raise ValueError('Failed cells: '+', '.join(failed)+'. Inspect launcher_logs; rerun the same command to resume committed rounds.')


def status(root):
    suite = load_json(root/'six_ablation_plan.json')
    for cell, item in suite['cells'].items():
        paths = [root/cell/f'{mode}/seed42'/item['arm'] for mode in ('smoke', 'runs')]
        rounds = [load_json(p/'progress.json')['completed_round'] if (p/'progress.json').exists() else 0 for p in paths]
        print(f'{cell}: smoke={rounds[0]}/1, formal={rounds[1]}/100', flush=True)


def pair_check(left, right):
    import numpy as np
    mismatches = [key for key in PAIR_HASHES if left['metadata'].get(key) is None
        or left['metadata'].get(key) != right['metadata'].get(key)]
    for key in ('aggregation', 'code_sha256'):
        if left['job'][key] != right['job'][key]:
            mismatches.append(key)
    for key in ('client_execution',):
        if left['config'].get(key) != right['config'].get(key):
            mismatches.append(key)
    for name, result in (('left', left), ('right', right)):
        if result['config'].get('ordinary_phase_losses') != losses_for(result['job']):
            mismatches.append(name+'.ordinary_phase_losses')
    for key in ('sample_id', 'class_id', 'prediction', 'correct'):
        if not np.array_equal(left['predictions'][0][key], right['predictions'][0][key]):
            mismatches.append('initial.'+key)
    return dict(status='mismatch' if mismatches else 'eligible', mismatches=mismatches,
        A_local_loss_left=losses_for(left['job'])['A'], A_local_loss_right=losses_for(right['job'])['A'],
        method_A_scope='A-side LA plus Full-CP; contrast does not isolate their individual effects',
        total_compute_matched=False, independent_seeds=1)


def summarize(root):
    from statistics import mean
    from tools.client_aggregation.analysis import METRICS, audit_run
    suite = load_json(root/'six_ablation_plan.json')
    if suite.get('experiment') != EXPERIMENT:
        raise ValueError('This output belongs to a different ablation version; do not mix v1 and v2 results')
    rows, results = [], {}
    for number, (cell, item) in enumerate(suite['cells'].items(), 1):
        plan = load_json(root/cell/'experiment_plan.json')
        job = single.make_job(plan, item['arm'])
        run = root/cell/job['run']
        progress = load_json(run/'progress.json') if (run/'progress.json').exists() else {}
        row = dict(row=number, cell=cell, A=item['arm'], method_B=item['method_B'],
            status='incomplete' if progress else 'not_started', completed_round=progress.get('completed_round', 0),
            **{m: '' for m in METRICS}, normal_B_steps='', ordinary_A_steps='', correction_steps='', B_steps='', reason='')
        if (run/'completion.json').exists():
            try:
                result = audit_run(run, job)
                results[cell] = result
                p = result['progress']
                row.update(status='complete', completed_round=100,
                    **{m: mean(float(r[m]) for r in result['curves'][81:101]) for m in METRICS},
                    normal_B_steps=p['normal_optimizer_steps'], ordinary_A_steps=p['extra_optimizer_steps'],
                    correction_steps=p['functional_correction_steps'], B_steps=p.get('b_transfer_optimizer_steps', 0))
            except (ValueError, OSError, KeyError, TypeError, AssertionError) as error:
                row.update(status='invalid', reason=str(error))
        rows.append(row)
        print(cell, row['status'], row['reason'], flush=True)
    by_cell = {r['cell']: r for r in rows}
    contrasts = []
    for label, left, right in (
        ('ordinary_A_without_B', 'plain_a', 'frozen_a'),
        ('ordinary_A_with_B', 'plain_a_b', 'frozen_a_b'),
        ('method_A_without_B', 'method_a', 'plain_a'),
        ('method_A_with_B', 'method_a_b', 'plain_a_b'),
        ('B_with_frozen_A', 'frozen_a_b', 'frozen_a'),
        ('B_with_plain_A', 'plain_a_b', 'plain_a'),
        ('B_with_method_A', 'method_a_b', 'method_a')):
        pair = dict(contrast=label, left=left, right=right, status='unavailable')
        if left in results and right in results:
            pair.update(pair_check(results[left], results[right]))
            if pair['status'] == 'eligible':
                pair['delta_pp'] = {m: by_cell[left][m]-by_cell[right][m] for m in METRICS}
        contrasts.append(pair)
    interaction = None
    required = ('method_A_without_B', 'method_A_with_B', 'B_with_plain_A', 'B_with_method_A')
    if all(next(p for p in contrasts if p['contrast'] == label)['status'] == 'eligible' for label in required):
        interaction = {m: by_cell['method_a_b'][m]-by_cell['method_a'][m]
            -by_cell['plain_a_b'][m]+by_cell['plain_a'][m] for m in METRICS}
    out = root/'analysis'
    write_table(out/'ablation_table.csv', rows)
    write_json(out/'contrasts.json', dict(pairs=contrasts, interaction_pp=interaction,
        endpoint='mean committed rounds81..100; single seed42, no inferential significance claim'))
    lines = ['Six-cell ablation: CIFAR-100-LT / Client-LT; accuracy (%)',
        'Plain A: CE. Method A: A-side LA tau=1 plus Full-CP. All ordinary B: LA tau=1. Both factors: FedAvg.',
        'Method-A contrasts measure its combined components. B: original shared donor C.',
        'Endpoint: mean rounds81..100, seed42. Unfinished/invalid results are left blank.', '',
        'Cell\tOverall\tHead20\tMiddle60\tTail20\tStatus']
    for row in rows:
        values = [f'{row[m]:.2f}' if isinstance(row[m], (int, float)) else '—' for m in METRICS[:4]]
        lines.append('\t'.join([row['cell'], *values, row['status']]))
    (out/'ablation_table.txt').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    if any(r['status'] == 'invalid' for r in rows) or any(p['status'] == 'mismatch' for p in contrasts):
        raise ValueError('Result audit failed; inspect analysis/ablation_table.csv and contrasts.json')
    return rows


def main(argv=None):
    raw = list(sys.argv[1:] if argv is None else argv)
    smoke_worker = '--smoke-worker' in raw
    if smoke_worker:
        raw.remove('--smoke-worker')
    args = parse_args(raw)
    if smoke_worker and args.stage != 'run':
        raise ValueError('--smoke-worker is internal to a single-cell run')
    root = args.output_root.resolve()
    if args.stage == 'status':
        return status(root)
    if args.stage == 'summary':
        return summarize(root)
    register(args)
    print('GPU assignment:', assignment(args), flush=True)
    if args.stage == 'plan':
        print('Registered six cells:', root, flush=True)
        return
    plan = load_json(root/args.cells[0]/'experiment_plan.json')
    single.preflight(plan, check_cuda=args.stage == 'run')
    if args.stage == 'preflight':
        print('CPU file/config preflight passed; CUDA smoke still required.', flush=True)
    elif args.stage in ('server', 'smoke'):
        dispatch(args)
        if args.stage == 'server' and args.stop_after_round == 0:
            summarize(root)
    elif args.stage == 'run':
        arm = CELLS[args.cells[0]][0]
        cell_root = root/args.cells[0]
        single.execute(single.make_job(plan, arm, 'smoke'), cell_root)
        if not smoke_worker:
            single.execute(single.make_job(plan, arm), cell_root, args.stop_after_round)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        raise SystemExit(130)
    except (OSError, ValueError) as error:
        raise SystemExit(str(error))
