"""Seed42 paired A/B interventions: one host or three independent single-GPU shards."""
import sys
if sys.version_info < (3, 10):
    raise SystemExit('Python >=3.10 required. Activate the clientlt conda environment first.')

import argparse
import os
from pathlib import Path
import shlex
import signal
import subprocess
import time
import zipfile

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from tools.a_learning_schedule.config import parse_archived_config, validate_input_config
from tools.a_learning_schedule.protocol import (
    ARMS, ROUNDS, PROTOCOL, check_source, digest, discover_source, file_digest,
    job_root, plan, read_json, write_csv, write_json,
)
from tools.a_learning_schedule.summary import status_rows, summarize


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--stage', choices=('preflight', 'smoke', 'run', 'status', 'summary', 'collect', '_worker'), default='preflight')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--summary-seeds', nargs='+', type=int, help='Summary/status/collect only; no new training is launched')
    parser.add_argument('--source-root', type=Path, default=Path('output/cifar100_LT/la_control'))
    parser.add_argument('--e2-run', type=Path)
    parser.add_argument('--e3-run', type=Path)
    parser.add_argument('--output-root', type=Path, default=Path('output/cifar100_LT/a_learning_schedule_v1'))
    parser.add_argument('--data-root', type=Path, default=Path('DATA'))
    parser.add_argument('--origins', nargs='+', choices=('e2', 'e3'))
    parser.add_argument('--rounds', nargs='+', type=int, choices=ROUNDS)
    parser.add_argument('--shard', type=int, choices=(1, 2, 3),
                        help='Independent single-GPU node: 1=E2/E3 round20, 2=round50, 3=round80')
    parser.add_argument('--gpus', nargs='+', help='CUDA_VISIBLE_DEVICES tokens, e.g. 0 1 2 3 4 5; overrides inherited mask')
    parser.add_argument('--num-workers', type=int, default=0, help='Per-client DataLoader workers; 0 avoids repeatedly spawning workers')
    parser.add_argument('--eval-batch-size', type=int, default=64)
    parser.add_argument('--no-plots', action='store_true')
    parser.add_argument('--archive', type=Path, help='Optional output zip for collect')
    parser.add_argument('--origin', choices=('e2', 'e3'), help=argparse.SUPPRESS)
    parser.add_argument('--anchor-round', type=int, help=argparse.SUPPRESS)
    parser.add_argument('--source-run', type=Path, help=argparse.SUPPRESS)
    parser.add_argument('--worker-mode', choices=('smoke', 'run'), default='run', help=argparse.SUPPRESS)
    parser.add_argument('--device', default='cuda:0', help=argparse.SUPPRESS)
    parser.add_argument('--code-fingerprint', help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.shard is not None:
        if args.origins is not None or args.rounds is not None:
            parser.error('--shard already selects E2/E3 and its anchor round; omit --origins/--rounds')
        if args.stage in ('collect', '_worker'):
            parser.error('--shard is not used with this stage; collect all three shards without --shard')
        if args.gpus is not None and len(args.gpus) != 1:
            parser.error('--shard uses exactly one GPU on this node')
        args.origins, args.rounds = ['e2', 'e3'], [ROUNDS[args.shard - 1]]
    else:
        args.origins = args.origins or ['e2', 'e3']
        args.rounds = args.rounds or list(ROUNDS)
    if args.summary_seeds and args.stage not in ('summary', 'status', 'collect'):
        parser.error('--summary-seeds is only for reading existing results')
    if args.num_workers < 0 or args.eval_batch_size <= 0 or args.seed < 0:
        parser.error('Invalid worker count, batch size, or seed')
    if len(set(args.origins)) != len(args.origins) or len(set(args.rounds)) != len(args.rounds):
        parser.error('Repeated origins/rounds are not allowed')
    if args.gpus and (len(set(args.gpus)) != len(args.gpus) or any(not g or ',' in g for g in args.gpus)):
        parser.error('Supply distinct GPU tokens separated by spaces')
    return args


def selection_label(args, seeds=None):
    seeds = [args.seed] if seeds is None else seeds
    return 'seed%s_%s_r%s' % ('-'.join(map(str, sorted(seeds))),
                            '-'.join(sorted(args.origins)),
                            '-'.join('%03d' % r for r in sorted(args.rounds)))


def full_selection(args):
    return set(args.origins) == {'e2', 'e3'} and set(args.rounds) == set(ROUNDS)


def analysis_directory(args, seeds=None):
    root = args.output_root / 'analysis'
    return root if full_selection(args) else root / 'selections' / selection_label(args, seeds)


def preflight_path(args):
    label = 'seed%d' % args.seed if full_selection(args) else selection_label(args)
    return args.output_root / ('preflight_%s.json' % label)


def selected_gpus(args):
    gpus = args.gpus
    if gpus is None:
        visible = os.environ.get('CUDA_VISIBLE_DEVICES')
        gpus = ['0'] if visible is None else visible.split(',')
    gpus = [g.strip() for g in gpus]
    if not gpus or len(set(gpus)) != len(gpus) or any(not g or g == '-1' for g in gpus):
        raise ValueError('No usable GPUs in CUDA_VISIBLE_DEVICES/--gpus; run inside your GPU allocation.')
    if args.shard is not None and len(gpus) != 1:
        raise ValueError('--shard needs one allocated GPU. Select it with --gpus TOKEN; '
                         'CUDA_VISIBLE_DEVICES currently exposes: ' + ','.join(gpus))
    return gpus


def code_fingerprint():
    paths = [Path('scripts/run_cliplora_a_learning_schedule.py')]
    paths += [p.relative_to(REPO) for p in sorted((REPO / 'tools/a_learning_schedule').glob('*.py'))]
    paths += [Path(p) for p in ('trainers/cliplora.py', 'utils/cliplora_a_refresh.py',
              'utils/sfra_execution.py', 'utils/loralib/layers.py', 'Dassl/dassl/data/data_manager.py',
              'Dassl/dassl/data/samplers.py', 'utils/cliplora_loss.py')]
    return digest({str(p): file_digest(REPO / p) for p in paths})


def preflight(args):
    sources = {}
    input_configs = {}
    errors = []
    for origin in args.origins:
        try:
            source = discover_source(args.source_root, args.seed, origin, getattr(args, origin + '_run'))
            checked = check_source(source, origin, args.seed, args.rounds)
            values = parse_archived_config(checked['meta']['resolved_config'])
            input_configs[origin] = validate_input_config(values)
            sources[origin] = checked
        except (OSError, ValueError, KeyError) as error:
            errors.append('%s: %s' % (origin, error))
    if len(sources) == 2:
        a, b = sources['e2'], sources['e3']
        if a['rows'] != b['rows'] or a['meta']['schedule'] != b['meta']['schedule']:
            errors.append('E2/E3 training partitions or participation schedules differ')
        for key in ('initial_lora_sha256', 'frozen_model_sha256', 'pool_sha256', 'test_sha256'):
            if a['meta'].get(key) != b['meta'].get(key):
                errors.append('E2/E3 source mismatch: ' + key)
    # Do not import torch, torchvision or the model in file-only preflight.
    candidates = [args.data_root, args.data_root / 'cifar-100-python', args.data_root / 'cifar-100/cifar-100-python']
    if not any(all((p / name).is_file() for name in ('train', 'test', 'meta')) for p in candidates):
        errors.append('CIFAR-100 train/test/meta files not found under --data-root ' + str(args.data_root))
    if errors:
        raise ValueError('Preflight failed:\n' + '\n'.join(errors))
    fingerprint = code_fingerprint()
    jobs = [dict(origin=origin, anchor_round=rnd, source_run=sources[origin]['path'])
            for origin in args.origins for rnd in args.rounds]
    unique_nodes = {n['id']: n for arm in ARMS for n in plan(arm)}
    report = dict(protocol=PROTOCOL, seed=args.seed, code_fingerprint=fingerprint, jobs=jobs,
                  arms=list(ARMS), branches=len(jobs) * 4, nodes_per_anchor=len(unique_nodes),
                  normal_nodes_per_anchor=sum(n['kind'] == 'normal' for n in unique_nodes.values()),
                  input_configs=input_configs, gpu_tested=False)
    write_json(preflight_path(args), report)
    print('File/configuration preflight passed: %d anchors, %d arms. GPU execution has NOT been tested.' % (len(jobs), 4 * len(jobs)), flush=True)
    for job in jobs:
        print('  %s round%03d <- %s' % (job['origin'], job['anchor_round'], job['source_run']), flush=True)
    return jobs, fingerprint


def child_command(args, job, fingerprint):
    return [sys.executable, '-u', str(Path(__file__).resolve()), '--stage', '_worker',
            '--worker-mode', args.stage, '--seed', str(args.seed), '--origin', job['origin'],
            '--anchor-round', str(job['anchor_round']), '--source-run', job['source_run'],
            '--output-root', str(args.output_root.resolve()), '--data-root', str(args.data_root.resolve()),
            '--num-workers', str(args.num_workers), '--eval-batch-size', str(args.eval_batch_size),
            '--code-fingerprint', fingerprint]


def stop_owned(process):
    if process.poll() is not None:
        return
    if os.name == 'nt':
        subprocess.run(['taskkill', '/PID', str(process.pid), '/T', '/F'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    else:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            return
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
    process.wait()


def print_failure_tail(log, start, max_bytes=32768, max_lines=60):
    """Show only this attempt's output, not errors from older appended attempts."""
    try:
        with Path(log).open('rb') as stream:
            stream.seek(0, os.SEEK_END)
            end = stream.tell()
            stream.seek(max(start, end - max_bytes))
            lines = stream.read().decode('utf-8', errors='replace').splitlines()[-max_lines:]
    except OSError as error:
        print('Unable to read failure log %s: %s' % (log, error), flush=True)
        return
    print('Current attempt failure details: ' + str(log), flush=True)
    print('\n'.join(lines) if lines else '(No child output was written.)', flush=True)


def launch(args, jobs, fingerprint):
    from scripts.run_ab_validation import file_lock
    gpus = selected_gpus(args)
    log_root = args.output_root / 'launcher_logs' / ('seed%d' % args.seed)
    log_root.mkdir(parents=True, exist_ok=True)
    queue, active, failures = list(jobs), {}, []
    selection = digest([(j['origin'], j['anchor_round']) for j in jobs])[:12]
    # Disjoint selections on different hosts can share one result root.
    with file_lock(args.output_root / ('launcher_seed%d_%s.lock' % (args.seed, selection)), timeout=.1):
        try:
            while queue or active:
                for gpu in gpus:
                    if gpu in active or not queue:
                        continue
                    job = queue.pop(0)
                    command = child_command(args, job, fingerprint)
                    log = log_root / ('%s_r%03d_%s.log' % (job['origin'], job['anchor_round'], args.stage))
                    stream = log.open('a', encoding='utf-8')
                    stream.write('\n[%s] CUDA_VISIBLE_DEVICES=%s %s\n' % (time.strftime('%Y-%m-%d %H:%M:%S'), gpu, shlex.join(command)))
                    stream.flush()
                    log_start = log.stat().st_size
                    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), PYTHONUNBUFFERED='1')
                    env.setdefault('OMP_NUM_THREADS', '1')
                    group = dict(creationflags=subprocess.CREATE_NEW_PROCESS_GROUP) if os.name == 'nt' else dict(start_new_session=True)
                    try:
                        process = subprocess.Popen(command, cwd=REPO, env=env, stdout=stream, stderr=subprocess.STDOUT, **group)
                    except Exception:
                        stream.close()
                        raise
                    active[gpu] = (process, stream, job, log, log_start)
                    print('START GPU=%s %s round=%d PID=%d log=%s' % (gpu, job['origin'], job['anchor_round'], process.pid, log), flush=True)
                for gpu, (process, stream, job, log, log_start) in list(active.items()):
                    code = process.poll()
                    if code is None:
                        continue
                    stream.close()
                    print('EXIT GPU=%s %s round=%d code=%d' % (gpu, job['origin'], job['anchor_round'], code), flush=True)
                    if code:
                        failures.append(dict(**job, returncode=code, log=str(log)))
                        print_failure_tail(log, log_start)
                    del active[gpu]
                if active:
                    time.sleep(1)
        finally:
            for process, stream, _, _, _ in active.values():
                stop_owned(process)
                stream.close()
    write_json(log_root / (args.stage + '_results_' + selection + '.json'), dict(failures=failures, jobs=jobs))
    if failures:
        raise ValueError('%d anchor jobs failed. Successful jobs are retained; rerun the same command to resume. See %s'
                         % (len(failures), log_root))


def collect(args):
    root = args.output_root.resolve()
    target = args.archive.resolve() if args.archive else root.with_name(root.name + '_analysis.zip')
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + '.tmp')
    with zipfile.ZipFile(temporary, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(root.rglob('*')):
            if path.is_file() and path.suffix.lower() in ('.csv', '.json', '.npz', '.png', '.pdf', '.md'):
                archive.write(path, Path(root.name) / path.relative_to(root))
    os.replace(temporary, target)
    print('Analysis archive (no training checkpoint tensors): ' + str(target))


def main(argv=None):
    args = parse_args(argv)
    os.chdir(REPO)
    args.output_root, args.data_root = args.output_root.resolve(), args.data_root.resolve()
    if args.stage == '_worker':
        from tools.a_learning_schedule.runtime import run_worker
        run_worker(args)
        return
    seeds = args.summary_seeds or [args.seed]
    destination = analysis_directory(args, seeds)
    if args.stage == 'status':
        rows = status_rows(args.output_root, seeds, args.origins, args.rounds)
        write_csv(destination / 'status.csv', rows)
        for row in rows:
            print('seed%s %s round%03d: %-10s saved_nodes=%d %s' % (row['seed'], row['origin'], row['anchor_round'], row['status'], row['saved_training_nodes'], row['reason']))
        return
    if args.stage in ('summary', 'collect'):
        summarize(args.output_root, seeds, args.origins, args.rounds,
                  plots=not args.no_plots, destination=destination)
        print('Summary: ' + str(destination / 'report.md'))
        if args.stage == 'collect':
            collect(args)
        return
    jobs, fingerprint = preflight(args)
    if args.stage in ('smoke', 'run'):
        launch(args, jobs, fingerprint)
        if args.stage == 'run':
            summarize(args.output_root, [args.seed], args.origins, args.rounds,
                      plots=not args.no_plots, destination=destination)
            print('Summary: ' + str(destination / 'report.md'))


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        raise SystemExit(130)
    except (OSError, ValueError, KeyError, RuntimeError) as error:
        raise SystemExit(str(error))
