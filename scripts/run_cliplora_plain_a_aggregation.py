"""Three seed42 ordinary-A runs, with static FedAvg/TailRW16/joint client weights.

Server: python scripts/run_cliplora_plain_a_aggregation.py --stage server --gpus 2 3
Each GPU drains a queue, one experiment at a time. Each run first passes a separate
one-round B+A smoke. Repeating the same command resumes committed checkpoints.
Use --stage status for progress and --stage pack for audited results collection.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
from queue import Empty, Queue
import shlex
import subprocess
import sys
import threading

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from scripts import run_cliplora_joint_aggregation as single
from scripts.run_ab_validation import file_lock, write_json
from tools.client_aggregation.plain_a import METHODS, status_rows, summarize
from tools.client_aggregation.protocol import (SCHEMA, PLAIN_A_CONTRACT, aggregation_spec,
    check_aggregation, code_hashes, count_matrix, digest, load_json, write_weight_tables)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('plan', 'preflight', 'smoke', 'run', 'server', 'status', 'summary', 'pack'), default='plan')
    parser.add_argument('--methods', nargs='+', choices=METHODS, default=list(METHODS))
    parser.add_argument('--seed', type=int, choices=(42,), default=42)
    parser.add_argument('--gpus', nargs='+', type=int, default=[2, 3], help='One queued experiment per GPU; order is physical GPU order')
    parser.add_argument('--client-concurrency', type=int, choices=(1, 4, 6), default=6)
    parser.add_argument('--cuda-policy', choices=('legacy', 'deterministic'), default='legacy')
    parser.add_argument('--reference-run', type=Path, default=Path('references/full10_clientlt'))
    parser.add_argument('--data-root', type=Path, default=Path('DATA'))
    parser.add_argument('--output-root', type=Path, default=Path('output/cifar100_LT/plain_a_aggregation_v1_parallel6'))
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--stop-after-round', type=int, default=0)
    args = parser.parse_args(argv)
    if len(set(args.methods)) != len(args.methods) or len(set(args.gpus)) != len(args.gpus) or min(args.gpus) < 0:
        parser.error('Use distinct methods and distinct nonnegative GPU IDs')
    if args.num_workers < 0 or not 0 <= args.stop_after_round <= 100:
        parser.error('Invalid workers or stopping round')
    return args


def register(args):
    root = args.output_root.resolve()
    fingerprint, counts, matrix = count_matrix(args.reference_run)
    hashes = code_hashes(REPO)
    settings = dict(seed=42, reference_run=str(args.reference_run.resolve()), data_root=str(args.data_root.resolve()),
                    num_workers=args.num_workers, client_concurrency=args.client_concurrency, parallel_validation='off')
    if args.cuda_policy != 'legacy':
        settings['cuda_policy'] = args.cuda_policy
    plans = {}
    with file_lock(root / 'locks/suite_plan.lock'):
        suite = dict(schema='plain_a_aggregation_v1', methods=list(METHODS), settings=settings,
                     code_sha256=hashes, contract=PLAIN_A_CONTRACT, aggregation_lambda=.1, tailrw_gamma=16.)
        manifest = root / 'suite_plan.json'
        if manifest.is_file() and load_json(manifest) != suite:
            raise ValueError('Registered suite changed. Preserve existing results and use a new output root.')
        for rule in METHODS:
            folder = root / rule
            rule_settings = dict(settings, aggregation_rule=rule, lambda_value=.1 if rule=='joint' else None)
            path = folder / 'experiment_plan.json'
            if path.is_file():
                plan = load_json(path)
                if (plan['contract'] != PLAIN_A_CONTRACT or plan['settings'] != rule_settings
                        or plan['code_sha256'] != hashes or plan['aggregation']['input_fingerprint'] != fingerprint
                        or plan['aggregation']['matrix_sha256'] != digest(matrix.tolist())
                        or plan['aggregation'].get('rule', 'joint') != rule):
                    raise ValueError('Registered inputs changed: ' + str(folder))
                check_aggregation(plan['aggregation'])
            else:
                plan = dict(schema=SCHEMA, contract=PLAIN_A_CONTRACT, settings=rule_settings,
                            code_sha256=hashes, aggregation=aggregation_spec(args.reference_run, .1, rule))
                write_json(path, plan)
                write_weight_tables(folder, plan['aggregation'])
            plans[rule] = plan
        if not manifest.is_file():
            write_json(manifest, suite)
    return plans


def child_command(args, rule):
    return [sys.executable, '-u', str(Path(__file__).resolve()), '--stage', 'run', '--methods', rule,
        '--output-root', str(args.output_root.resolve()), '--reference-run', str(args.reference_run.resolve()),
        '--data-root', str(args.data_root.resolve()), '--seed', '42', '--num-workers', str(args.num_workers),
        '--client-concurrency', str(args.client_concurrency), '--cuda-policy', args.cuda_policy,
        '--stop-after-round', str(args.stop_after_round)]


def launch(args):
    root = args.output_root.resolve()
    queue, active, failures, assignments = Queue(), set(), [], []
    lock, stop = threading.Lock(), threading.Event()
    for rule in args.methods:
        queue.put(rule)

    def worker(gpu):
        while not stop.is_set():
            try:
                rule = queue.get_nowait()
            except Empty:
                return
            command = child_command(args, rule)
            log = root / 'launcher_logs' / (rule+'_server.log')
            log.parent.mkdir(parents=True, exist_ok=True)
            with lock:
                assignments.append(dict(method=rule, gpu=gpu))
                write_json(root / 'server_launch.json', dict(assignments=assignments,
                    client_concurrency=args.client_concurrency, dispatch='one experiment per GPU; queued methods'))
            print(f'START {rule}: GPU={gpu}, clients={args.client_concurrency}, log={log}', flush=True)
            with log.open('a', encoding='utf-8') as stream:
                stream.write('\n'+shlex.join(command)+'\n'); stream.flush()
                # A new POSIX process group lets Ctrl+C stop the worker AND training child.
                with lock:
                    if stop.is_set():
                        return
                    process = subprocess.Popen(command, cwd=REPO, env=single.server_environment(gpu),
                        stdout=stream, stderr=subprocess.STDOUT, start_new_session=os.name=='posix')
                    active.add(process)
                try:
                    code = process.wait()
                finally:
                    with lock:
                        active.discard(process)
            print(('FAILED' if code else 'FINISHED'), rule, 'GPU='+str(gpu), flush=True)
            if code:
                with lock:
                    failures.append(rule)

    with file_lock(root / 'locks/suite_dispatch.lock', timeout=.1):
        pool = ThreadPoolExecutor(max_workers=min(len(args.gpus), len(args.methods)))
        try:
            futures = [pool.submit(worker, gpu) for gpu in args.gpus[:len(args.methods)]]
            for future in futures:
                future.result()
        except BaseException:
            stop.set()
            with lock:
                for process in active:
                    if process.poll() is None:
                        if os.name == 'posix':
                            import signal
                            try:
                                os.killpg(process.pid, signal.SIGTERM)
                            except ProcessLookupError:
                                pass
                        else:
                            process.terminate()
            raise
        finally:
            pool.shutdown(wait=True)
    if failures:
        raise ValueError('Failed methods: '+', '.join(failures)+'. Rerun the same command to resume; completed methods are retained.')


def main(argv=None):
    args = parse_args(argv)
    os.chdir(REPO)
    if args.stage == 'status':
        for row in status_rows(args.output_root):
            print(f'{row["method"]}: {row["status"]}, round={row["completed_round"]}/100, '
                  f'B_steps={row["normal_B_steps"]}, A_steps={row["extra_A_steps"]}')
        return
    if args.stage in ('summary', 'pack'):
        statuses, pairs = summarize(args.output_root)
        if any(s['status']=='invalid' for s in statuses) or any(p['status']=='mismatch' for p in pairs):
            raise ValueError('Invalid/mismatched results. See analysis/status.csv and pair_audit.json.')
        if args.stage == 'pack':
            from tools.client_aggregation.analysis import pack_results
            print('Archive:', pack_results(args.output_root))
        return
    plans = register(args)
    print('Registered ordinary A: B3 -> A1 in rounds1..90; B3 only in rounds91..100.', flush=True)
    if args.stage == 'plan':
        return
    for rule in args.methods:
        single.preflight(plans[rule], check_cuda=args.stage in ('smoke', 'run'))
    print('Input/configuration preflight passed; this is not a training result.', flush=True)
    if args.stage == 'preflight':
        return
    if args.stage == 'server':
        launch(args)
        return
    for rule in args.methods:
        root = args.output_root / rule
        single.execute(single.make_job(plans[rule], 'plain_a', 'smoke'), root)
        if args.stage == 'run':
            single.execute(single.make_job(plans[rule], 'plain_a'), root, args.stop_after_round)


if __name__ == '__main__':
    try:
        main()
    except (ValueError, OSError, KeyError) as error:
        raise SystemExit(str(error))
