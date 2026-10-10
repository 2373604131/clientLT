"""Two seed42 controls: frozen A, ordinary B, TailRW16 on GPU2 and FedAvg on GPU3."""
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
from tools.client_aggregation.protocol import load_json

METHODS = ('tailrw16', 'fedavg')


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('plan', 'preflight', 'smoke', 'server', 'run', 'status', 'summary', 'pack'), default='plan')
    parser.add_argument('--methods', nargs='+', choices=METHODS, default=list(METHODS))
    parser.add_argument('--seed', type=int, choices=(42,), default=42)
    parser.add_argument('--gpus', nargs=2, type=int, default=[2, 3], metavar=('TAILRW_GPU', 'FEDAVG_GPU'))
    parser.add_argument('--client-concurrency', type=int, choices=(1, 4, 6), default=6)
    parser.add_argument('--cuda-policy', choices=('legacy', 'deterministic'), default='legacy')
    parser.add_argument('--reference-run', type=Path, default=Path('references/full10_clientlt'))
    parser.add_argument('--data-root', type=Path, default=Path('DATA'))
    parser.add_argument('--output-root', type=Path, default=Path('output/cifar100_LT/frozen_aggregation_controls_v3_parallel6'))
    parser.add_argument('--compare-root', type=Path, default=Path('output/cifar100_LT/client_aggregation_v2_parallel4'),
                        help='Read-only historical joint frozen reference; never launched')
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--stop-after-round', type=int, default=0)
    args = parser.parse_args(argv)
    if len(set(args.methods)) != len(args.methods) or len(set(args.gpus)) != 2 or min(args.gpus) < 0:
        parser.error('Methods must be distinct; specify two distinct nonnegative GPU IDs')
    if args.num_workers < 0 or not 0 <= args.stop_after_round <= 100:
        parser.error('Invalid workers or stopping round')
    return args


def child_options(args, rule):
    return ['--aggregation-rule', rule, '--arms', 'frozen', '--seed', str(args.seed),
        '--reference-run', str(args.reference_run.resolve()), '--data-root', str(args.data_root.resolve()),
        '--output-root', str(args.output_root.resolve()/rule), '--num-workers', str(args.num_workers),
        '--client-concurrency', str(args.client_concurrency), '--cuda-policy', args.cuda_policy]


def register(args):
    root = args.output_root.resolve()
    plans = {}
    # Register both plans before dispatch. The old joint root is only used by reporting.
    for rule in METHODS:
        plans[rule] = single.register(single.parse_args(child_options(args, rule)))
    manifest = dict(schema='frozen_aggregation_controls_v1', seed=42, methods=list(METHODS),
        settings=plans['fedavg']['settings'], code_sha256=plans['fedavg']['code_sha256'],
        primary='rounds81..100 mean Overall; Head20/Middle60/Tail20 jointly; no test-selected checkpoints',
        controls='initial A fixed; B3 epochs/round, 100 rounds, 30 clients; no extra A/B, CP, C or direct calibration')
    with file_lock(root/'locks/suite_plan.lock'):
        path = root/'suite_plan.json'
        if path.is_file() and load_json(path) != manifest:
            raise ValueError('Registered suite changed; preserve it and choose a new output root')
        if not path.is_file():
            write_json(path, manifest)
    return plans


def commands(args, rule):
    common = [sys.executable, '-u', str(REPO/'scripts/run_cliplora_joint_aggregation.py'), *child_options(args, rule)]
    result = [common+['--stage', 'smoke']]
    if args.stage != 'smoke':
        result.append(common+['--stage', 'run', '--stop-after-round', str(args.stop_after_round)])
    return result


def launch(args):
    root = args.output_root.resolve()
    assignments = {rule: args.gpus[METHODS.index(rule)] for rule in args.methods}
    with file_lock(root/'locks/suite_dispatch.lock', timeout=.1):
        write_json(root/'server_launch.json', dict(stage=args.stage, gpu_assignment=assignments,
            client_concurrency=args.client_concurrency, stop_after_round=args.stop_after_round))

        def run_one(rule, gpu):
            log = root/'launcher_logs'/(rule+'_server.log')
            log.parent.mkdir(parents=True, exist_ok=True)
            print(f'START {rule}: physical GPU={gpu}, max local clients={args.client_concurrency}, log={log}', flush=True)
            with log.open('a', encoding='utf-8') as stream:
                for command in commands(args, rule):
                    stage = command[command.index('--stage')+1]
                    print(f'{rule}: {stage}', flush=True)
                    stream.write('\n'+shlex.join(command)+'\n'); stream.flush()
                    result = subprocess.run(command, cwd=REPO, env=single.server_environment(gpu),
                                            stdout=stream, stderr=subprocess.STDOUT)
                    if result.returncode:
                        print(f'FAILED {rule}: {stage}, exit={result.returncode}. See {log}', flush=True)
                        return result.returncode
            target = 1 if args.stage == 'smoke' else args.stop_after_round or 100
            print(f'FINISHED {rule} on GPU {gpu}; target round={target}', flush=True)
            return 0

        with ThreadPoolExecutor(max_workers=len(assignments)) as pool:
            futures = [pool.submit(run_one, rule, gpu) for rule, gpu in assignments.items()]
            codes = [future.result() for future in futures]
        if any(codes):
            raise ValueError('A control failed. Successful results are retained; rerun the same command to resume.')


def main(argv=None):
    args = parse_args(argv)
    os.chdir(REPO)
    if args.stage in ('status', 'summary', 'pack'):
        from tools.client_aggregation.frozen_controls import summarize
        from tools.client_aggregation.analysis import pack_results
        statuses, pairs = summarize(args.output_root, args.compare_root)
        if any(row['status'] == 'invalid' for row in statuses) or any(row['status'] == 'mismatch' for row in pairs):
            raise ValueError('Invalid/mismatched results; see analysis/report.md and pair_audit.json')
        if args.stage == 'pack':
            print('Archive:', pack_results(args.output_root))
        return
    plans = register(args)
    if args.stage == 'plan':
        print('Registered two frozen-A controls:', args.output_root.resolve())
        return
    for plan in plans.values():
        single.preflight(plan, check_cuda=False)
    print('File/configuration preflight passed for two controls. GPU execution has NOT been tested.', flush=True)
    if args.stage != 'preflight':
        launch(args)


if __name__ == '__main__':
    try:
        main()
    except (ValueError, OSError, KeyError) as error:
        raise SystemExit(str(error))
