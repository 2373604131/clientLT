"""Four independent GPUs, eight concurrent local clients per experiment."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import shlex
import subprocess
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from scripts import run_cliplora_joint_aggregation as single
from scripts.run_ab_validation import file_lock, write_json
from tools.client_aggregation.protocol import (SCHEMA, aggregation_spec, code_hashes,
    load_json, write_weight_tables)
from tools.client_aggregation.ab_factorial import CELLS, EXPERIMENT, contract_for, summarize
from utils.method_a_margin_math import DEFAULTS, algorithm_config


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    p.add_argument('--stage', choices=('plan', 'preflight', 'server', 'smoke', 'run', 'status', 'summary', 'pack'), default='plan')
    p.add_argument('--cells', nargs='+', choices=tuple(CELLS), default=list(CELLS))
    p.add_argument('--gpus', nargs=4, type=int, default=[0, 1, 2, 3],
        help='Physical GPUs in order: plain_joint, new_a_joint, new_a_new_b_joint, plain_new_b_joint')
    p.add_argument('--client-concurrency', type=int, choices=(8,), default=8)
    p.add_argument('--num-workers', type=int, default=8)
    p.add_argument('--reference-run', type=Path, default=Path('references/full10_clientlt'))
    p.add_argument('--data-root', type=Path, default=Path('DATA'))
    p.add_argument('--output-root', type=Path, default=Path('output/cifar100_LT/ab_factorial_joint_v1_parallel8'))
    p.add_argument('--reuse-baseline', type=Path, help='Completed plain-A joint root containing experiment_plan.json; read-only')
    p.add_argument('--stop-after-round', type=int, default=0)
    for key, value in DEFAULTS.items():
        p.add_argument('--'+key.replace('_', '-'), type=float, default=value)
    args = p.parse_args(argv)
    if len(set(args.gpus)) != 4 or min(args.gpus) < 0:
        p.error('Use four distinct nonnegative GPU IDs')
    if len(set(args.cells)) != len(args.cells) or args.num_workers < 0 or not 0 <= args.stop_after_round <= 100:
        p.error('Duplicate cells, negative workers or invalid stopping round')
    try:
        algorithm_config(**parameters(args))
    except ValueError as error:
        p.error(str(error))
    return args


def parameters(args):
    return {k: getattr(args, k) for k in DEFAULTS}


def register(args):
    root = args.output_root.resolve()
    settings = dict(seed=42, reference_run=str(args.reference_run.resolve()), data_root=str(args.data_root.resolve()),
        num_workers=args.num_workers, client_concurrency=8, parallel_validation='off',
        aggregation_rule='joint', lambda_value=.1, margin_parameters=parameters(args), factorial_experiment=EXPERIMENT)
    spec, hashes = aggregation_spec(args.reference_run, .1, 'joint'), code_hashes(REPO)
    suite = dict(experiment=EXPERIMENT, settings=settings, code_sha256=hashes, cells={})
    with file_lock(root/'locks/registration.lock'):
        for cell, (arm, _) in CELLS.items():
            reused = cell == 'plain_joint' and args.reuse_baseline is not None
            cell_root = args.reuse_baseline.resolve() if reused else root/cell
            suite['cells'][cell] = dict(root=str(cell_root), arm=arm, reused=reused)
        suite_path = root/'factorial_plan.json'
        if suite_path.is_file() and load_json(suite_path) != suite:
            raise ValueError('Registered source/parameters/baseline changed; preserve this suite and use a new output root')
        # Solver wall time is diagnostic, not part of the immutable weight identity.
        # Reuse the original registered solution on restart (or the reused baseline's solution).
        for item in suite['cells'].values():
            path = Path(item['root'])/'experiment_plan.json'
            if path.is_file():
                frozen_spec = load_json(path)['aggregation']
                identity = ('counts', 'matrix_sha256', 'partition_sha256', 'input_fingerprint',
                            'weights_sha256', 'lambda_value', 'rule')
                if any(frozen_spec.get(k) != spec.get(k) for k in identity):
                    raise ValueError('Registered aggregation inputs/weights changed')
                spec = frozen_spec
                break
        for cell, item in suite['cells'].items():
            cell_root = Path(item['root'])
            if item['reused']:
                from tools.client_aggregation.plain_a import load_plain
                baseline = load_plain(cell_root, 'joint')
                old = baseline['job']
                if (old['aggregation'] != spec or old['settings'].get('client_concurrency') != 8
                        or baseline['config'].get('b_transfer') or baseline['progress']['functional_correction_steps']):
                    raise ValueError('Reused baseline must be completed ordinary A + same joint, eight clients and no B/correction')
                continue
            plan = dict(schema=SCHEMA, settings=dict(settings, factorial_cell=cell),
                contract=contract_for(cell, parameters(args)), code_sha256=hashes, aggregation=spec)
            path = cell_root/'experiment_plan.json'
            if path.is_file():
                if load_json(path) != plan:
                    raise ValueError('Cell registration changed: '+cell)
            else:
                write_json(path, plan)
                write_weight_tables(cell_root, spec)
        write_json(suite_path, suite)
    return suite


def selected_jobs(args, suite):
    return [(cell, args.gpus[list(CELLS).index(cell)]) for cell in args.cells if not suite['cells'][cell]['reused']]


def child_commands(args, cell):
    common = [sys.executable, '-u', str(REPO/'scripts/run_cliplora_ab_factorial.py'), '--cells', cell,
        '--gpus', *map(str, args.gpus), '--client-concurrency', '8', '--num-workers', str(args.num_workers),
        '--reference-run', str(args.reference_run.resolve()), '--data-root', str(args.data_root.resolve()),
        '--output-root', str(args.output_root.resolve())]
    if args.reuse_baseline:
        common += ['--reuse-baseline', str(args.reuse_baseline.resolve())]
    for key, value in parameters(args).items():
        common += ['--'+key.replace('_', '-'), str(value)]
    stages = ('smoke',) if args.stage == 'smoke' else ('smoke', 'run')
    return [common+['--stage', stage, '--stop-after-round', str(args.stop_after_round)] for stage in stages]


def dispatch(args, suite):
    jobs = selected_jobs(args, suite)
    if not jobs:
        print('Selected baseline is reused; no GPU process needed.')
        return
    root = args.output_root.resolve()
    with file_lock(root/'locks/server_dispatch.lock', timeout=.1):
        write_json(root/'server_launch.json', dict(gpu_assignment=dict(jobs), client_concurrency=8,
            cells=suite['cells'], commands={c: child_commands(args, c) for c, _ in jobs},
            note='Each child sees one GPU; all30 clients participate in each ordinary phase with max8 in flight'))
        def launch(cell, gpu):
            log = root/'launcher_logs'/f'{cell}_server.log'
            log.parent.mkdir(parents=True, exist_ok=True)
            print(f'START {cell}: physical GPU={gpu}, concurrent clients=8, log={log}', flush=True)
            with log.open('a', encoding='utf-8') as stream:
                for command in child_commands(args, cell):
                    stream.write('\n'+shlex.join(command)+'\n'); stream.flush()
                    result = subprocess.run(command, cwd=REPO, env=single.server_environment(gpu),
                        stdout=stream, stderr=subprocess.STDOUT)
                    if result.returncode:
                        print(f'FAILED {cell}: {log}', flush=True)
                        return result.returncode
            print('FINISHED '+cell, flush=True)
            return 0
        with ThreadPoolExecutor(max_workers=len(jobs)) as executor:
            futures = [executor.submit(launch, cell, gpu) for cell, gpu in jobs]
            codes = [future.result() for future in futures]
        if any(codes):
            raise ValueError('One or more GPU jobs failed; see launcher_logs. Rerun the same command to resume committed rounds.')


def status(root):
    suite = load_json(root/'factorial_plan.json')
    for cell, item in suite['cells'].items():
        run = Path(item['root'])/'runs/seed42'/item['arm']
        path = run/'progress.json'
        progress = load_json(path) if path.is_file() else {}
        smoke = Path(item['root'])/'smoke/seed42'/item['arm']/'progress.json'
        print(cell, 'reused' if item['reused'] else 'new',
            f'formal={progress.get("completed_round", 0)}/100',
            f'smoke={load_json(smoke).get("completed_round", 0) if smoke.is_file() else 0}/6',
            'completed_unverified' if (run/'completion.json').is_file() else '')


def main(argv=None):
    args = parse_args(argv)
    os.chdir(REPO)
    root = args.output_root.resolve()
    if args.stage == 'status':
        status(root)
        return
    if args.stage in ('summary', 'pack'):
        statuses, pairs = summarize(root)
        if any(r['status'] == 'invalid' for r in statuses) or any(p['status'] == 'mismatch' for p in pairs):
            raise ValueError('Invalid or unpaired results; see analysis/status.csv and contrasts.json')
        if args.stage == 'pack':
            from tools.client_aggregation.analysis import pack_results
            print('Archive:', pack_results(root))
        return
    suite = register(args)
    for cell, gpu in selected_jobs(args, suite):
        print(f'{cell} -> GPU {gpu}; max8 concurrent clients; {suite["cells"][cell]["root"]}', flush=True)
    if args.stage == 'plan':
        return
    for cell, _ in selected_jobs(args, suite):
        single.preflight(load_json(Path(suite['cells'][cell]['root'])/'experiment_plan.json'))
    from tools.client_aggregation.ab_factorial import feedback_preflight
    write_json(root/'feedback_preflight.json', feedback_preflight(args.reference_run))
    if args.stage == 'preflight':
        print('Inputs/configuration passed; GPU execution has NOT been tested.')
        return
    if args.stage == 'server' or args.stage == 'smoke' and len(args.cells) > 1:
        dispatch(args, suite)
        return
    jobs = selected_jobs(args, suite)
    if len(jobs) != 1:
        raise ValueError('Use --stage server for parallel execution or select one cell for --stage run')
    cell, gpu = jobs[0]
    # A direct single-cell invocation has the same physical assignment as the server launcher.
    os.environ.update(CUDA_VISIBLE_DEVICES=str(gpu), CUDA_DEVICE_ORDER='PCI_BUS_ID')
    cell_root = Path(suite['cells'][cell]['root'])
    plan = load_json(cell_root/'experiment_plan.json')
    single.preflight(plan, check_cuda=True)
    single.execute(single.make_job(plan, CELLS[cell][0], 'smoke'), cell_root)
    if args.stage == 'run':
        single.execute(single.make_job(plan, CELLS[cell][0]), cell_root, args.stop_after_round)


if __name__ == '__main__':
    try:
        main()
    except (ValueError, OSError, KeyError) as error:
        raise SystemExit(str(error))
