"""Seed42 comparison suite: plan/preflight/smoke/train/status/summary/pack/import-ab."""
import argparse
import os
from pathlib import Path
import shlex
import subprocess
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from tools.benchmarks.common import (ALL_METHODS, METHODS, FACTOR_METHODS, LONGTAIL_METHODS, job_id, job_path, make_job,
    preflight, read_json, register, run_path)


def command_for(job, resume=False, stop_after=None):
    method = job['method']
    if method in ('a', 'ab'):
        if stop_after is not None:
            raise ValueError('--stop-after is only supported by the new baseline worker')
        command = [sys.executable, '-u', str(REPO / 'scripts/run_ab_validation.py'), '--stage', 'train',
            '--arms', method, '--seeds', '42', '--protocol-seed', '42',
            '--reference-run', job['reference_run'], '--data-root', job['data_root'],
            '--output-root', str(Path(job['output_root']) / 'ab_runs'), '--num-workers', str(job['workers'])]
    else:
        command = [sys.executable, '-u', str(REPO / 'scripts/train_paper_baseline.py'),
                   '--job', str(job_path(job['output_root'], method))]
        if stop_after is not None:
            command += ['--stop-after', str(stop_after)]
    if resume:
        command += ['--resume']
    return command


def execute(job, resume=False, stop_after=None):
    from scripts.run_ab_validation import file_lock
    from tools.benchmarks.collect import completed_result, imported_run
    root = Path(job['output_root'])
    with file_lock(root / 'locks' / (job['method'] + '.lock'), timeout=.1):
        preflight(job, require_cuda=True)
        imported = imported_run(root, job['method'])
        if imported is not None or (run_path(job) / 'completion.json').exists():
            completed_result(root, job)
            print('SKIP verified complete:', job['method'], flush=True)
            return
        command = command_for(job, resume, stop_after)
        print(shlex.join(command), flush=True)
        logs = root / 'logs'
        logs.mkdir(parents=True, exist_ok=True)
        with (logs / (job['method'] + '.log')).open('a', encoding='utf-8') as stream:
            process = subprocess.Popen(command, cwd=REPO, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                       text=True, encoding='utf-8', errors='replace', bufsize=1,
                                       env={**os.environ, 'PYTHONIOENCODING': 'utf-8'})
            try:
                for line in process.stdout:
                    print(line, end='', flush=True)
                    stream.write(line); stream.flush()
                code = process.wait()
            except BaseException:
                process.terminate()
                process.wait()
                raise
        if code:
            raise RuntimeError(f"{job['method']} exited with {code}; see {logs / (job['method'] + '.log')}")
        if stop_after is None:
            completed_result(root, job)


def parser_for_cli(suite='main'):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage', choices=['plan', 'preflight', 'smoke', 'train', 'status', 'summary', 'pack', 'import-ab'], default='plan')
    choices = ALL_METHODS if suite == 'main' else (LONGTAIL_METHODS if suite == 'longtail' else FACTOR_METHODS) + ('a', 'ab')
    defaults = METHODS if suite == 'main' else (LONGTAIL_METHODS if suite == 'longtail' else FACTOR_METHODS)
    p.add_argument('--methods', nargs='+', choices=choices, default=list(defaults))
    p.add_argument('--seed', type=int, choices=[42], default=42, help='This first suite is fixed to seed42')
    p.add_argument('--reference-run', type=Path, default=Path('references/full10_clientlt'))
    p.add_argument('--data-root', type=Path, default=Path('DATA'))
    output = {'main': 'paper_benchmarks_seed42', 'factors': 'factor_benchmarks_seed42',
              'longtail': 'longtail_benchmarks_seed42_v1'}[suite]
    p.add_argument('--output-root', type=Path, default=Path('output/cifar100_LT') / output)
    p.add_argument('--num-workers', type=int, default=8)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--stop-after', type=int, help='External baselines only: stop at a committed round, then resume')
    p.add_argument('--import-ab-root', type=Path, help='Existing completed ab_validation suite to audit and reuse')
    return p


def main(suite='main'):
    parser = parser_for_cli(suite)
    args = parser.parse_args()
    try:
        if len(set(args.methods)) != len(args.methods) or args.num_workers < 0:
            raise ValueError('Methods must be distinct and workers nonnegative')
        if suite == 'main' and any(m in FACTOR_METHODS for m in args.methods):
            raise ValueError('Use scripts/run_factor_benchmarks.py for second-batch factor baselines')
        if args.stop_after is not None and not 1 <= args.stop_after <= 100:
            raise ValueError('--stop-after must be in 1..100')
        root = args.output_root.resolve()
        from tools.benchmarks.collect import collect, pack, status, import_ab
        if args.stage in ('status', 'summary', 'pack'):
            if args.stage == 'status':
                for row in status(root):
                    print(row)
            else:
                collect(root)
                if args.stage == 'pack':
                    print(pack(root))
                print('Report:', root / 'analysis/report.md')
            return
        methods = args.methods
        smoke = args.stage == 'smoke'
        if smoke:
            root = root / 'smoke'  # never mix short pilot data into the formal table
            methods = [x for x in methods if x not in ('a', 'ab')]
            if not methods:
                raise ValueError('Smoke is for external baseline adapters, not frozen A/AB')
        if args.stage == 'import-ab':
            if args.import_ab_root is None:
                raise ValueError('--import-ab-root is required')
            methods = ['a', 'ab']
        jobs = [make_job(m, args.reference_run, args.data_root, root, args.num_workers, smoke, suite) for m in methods]
        for job in jobs:
            register(job)
        if args.stage == 'import-ab':
            import_ab(root, args.import_ab_root)
            collect(root)
            return
        for job in jobs:
            if args.stage == 'plan':
                print(job['label'], '\n ', shlex.join(command_for(job, args.resume, args.stop_after)))
            elif args.stage == 'preflight':
                print(job['method'], preflight(job, require_cuda=True))
            else:
                execute(job, args.resume, args.stop_after)
    except (OSError, ValueError, RuntimeError, KeyError) as error:
        parser.error(str(error))


if __name__ == '__main__':
    main()
