"""Independent diagnostic jobs, bounded GPU queue, and owned child processes."""
from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
from queue import Empty, Queue
import signal
import subprocess
import sys
import threading
import time

from tools.a_projection_diagnosis.protocol import write_json


def selected_jobs(args):
    # Single GPU follows the protocol: E1 first, then the three anchor panels.
    # The optional parallel queue starts its more expensive panels first.
    jobs = []
    if args.part in ('all', 'events') and 'fedavg' in args.trajectories:
        jobs.extend(dict(id='fedavg_round%03d' % r, trajectory='fedavg', part='events', round=r)
                    for r in sorted(args.rounds))
    if args.part in ('all', 'e1'):
        jobs.extend(dict(id=rule + '_e1', trajectory=rule, part='e1', round=None) for rule in args.trajectories)
    if not getattr(args, 'gpus', None):
        jobs.sort(key=lambda job: job['part'] != 'e1')
    return jobs


def task_root(output_root, job):
    return Path(output_root) / job['trajectory'] / ('e1' if job['part'] == 'e1' else 'round%03d' % job['round'])


def gpu_tokens(indices, environ=None):
    """Indices refer to the current allocation, matching torch device numbering."""
    env = os.environ if environ is None else environ
    visible = env.get('CUDA_VISIBLE_DEVICES')
    if not indices or len(indices) > 4 or len(set(indices)) != len(indices) or min(indices) < 0:
        raise ValueError('Select one to four distinct nonnegative visible GPU indices')
    if visible is None:
        return [str(i) for i in indices]
    allocated = [s.strip() for s in visible.split(',')]
    if any(not s or s == '-1' for s in allocated) or max(indices) >= len(allocated):
        raise ValueError('--gpus indexes CUDA_VISIBLE_DEVICES; requested indices exceed the visible allocation: ' + visible)
    if len(set(allocated)) != len(allocated):
        raise ValueError('CUDA_VISIBLE_DEVICES contains duplicate devices')
    return [allocated[i] for i in indices]


def child_command(args, job, repo):
    command = [sys.executable, '-u', str(Path(repo) / 'scripts/run_a_projection_diagnosis.py'),
        '--stage', args.stage, '--trajectories', job['trajectory'], '--part', job['part'],
        '--source-root', str(args.source_root.resolve()), '--output-root', str(args.output_root.resolve()),
        '--data-root', str(args.data_root.resolve()), '--device', 'cuda:0', '--seed', str(args.seed),
        '--num-workers', str(args.num_workers), '--eval-batch-size', str(args.eval_batch_size),
        '--cpu-threads', str(args.cpu_threads)]
    if job['round'] is not None:
        command += ['--rounds', str(job['round'])]
    for rule in ('fedavg', 'tailrw16'):
        path = getattr(args, rule + '_run')
        if path is not None:
            command += ['--' + rule + '-run', str(path.resolve())]
    if args.no_plots:
        command.append('--no-plots')
    # Deliberately omit --gpus: the child owns one visible GPU and one task.
    return command


def drain_queue(jobs, slots, run_job):
    """One worker per slot. A failure is reported without dropping other jobs."""
    queue = Queue()
    for job in jobs:
        queue.put(job)
    stop = threading.Event()

    def worker(slot):
        outcomes = []
        while not stop.is_set():
            try:
                job = queue.get_nowait()
            except Empty:
                return outcomes
            try:
                outcomes.append(run_job(slot, job))
            except Exception as error:
                outcomes.append(dict(job=job['id'], status='failed', reason=str(error)))
        return outcomes

    pool = ThreadPoolExecutor(max_workers=min(len(jobs), len(slots)))
    try:
        futures = [pool.submit(worker, slot) for slot in slots[:len(jobs)]]
        return [outcome for f in futures for outcome in f.result()]
    except BaseException:
        stop.set()
        # The launcher stops its owned subprocesses before joining these threads.
        pool.shutdown(wait=False, cancel_futures=True)
        raise
    finally:
        if not stop.is_set():
            pool.shutdown(wait=True)


def stop_owned(process):
    if process.poll() is not None:
        return
    if os.name == 'nt':
        subprocess.run(['taskkill', '/PID', str(process.pid), '/T', '/F'],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
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


def launch(args, repo):
    from scripts.run_ab_validation import file_lock
    jobs = selected_jobs(args)
    tokens = gpu_tokens(args.gpus)
    root = args.output_root
    lock, active, cancelled = threading.Lock(), set(), threading.Event()
    records = {}

    def publish():
        write_json(root / 'parallel_status.json', dict(stage=args.stage, visible_indices=args.gpus,
            device_tokens=tokens, jobs=[records[j['id']] for j in jobs if j['id'] in records]))

    def run_job(token, job):
        command = child_command(args, job, repo)
        log = root / 'launcher_logs' / (job['id'] + '.log')
        log.parent.mkdir(parents=True, exist_ok=True)
        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = token
        for variable in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
            env[variable] = str(args.cpu_threads)
        start = time.perf_counter()
        with log.open('a', encoding='utf-8') as stream:
            stream.write('\nCOMMAND ' + repr(command) + '\n')
            stream.flush()
            with lock:
                if cancelled.is_set():
                    return dict(job=job['id'], status='cancelled')
                process = subprocess.Popen(command, cwd=repo, env=env, stdout=stream, stderr=subprocess.STDOUT,
                                           start_new_session=os.name != 'nt')
                active.add(process)
                records[job['id']] = dict(job=job['id'], device_token=token, status='running', log=str(log))
                publish()
            print('START %s GPU=%s log=%s' % (job['id'], token, log), flush=True)
            try:
                code = process.wait()
            finally:
                with lock:
                    active.discard(process)
            result = dict(job=job['id'], device_token=token, status='complete' if code == 0 else 'failed',
                          return_code=code, seconds=time.perf_counter() - start, log=str(log))
            with lock:
                records[job['id']] = result
                publish()
            print('%s %s GPU=%s' % (result['status'].upper(), job['id'], token), flush=True)
            if code:
                with log.open('rb') as reader:
                    reader.seek(max(0, log.stat().st_size - 8000))
                    print(reader.read().decode('utf-8', errors='replace'), flush=True)
            return result

    with file_lock(root / 'parallel_launcher.lock', timeout=.1):
        write_json(root / 'parallel_plan.json', dict(jobs=jobs, visible_indices=args.gpus, device_tokens=tokens,
                   dispatch='one independent diagnostic task per GPU; longest panels queued first',
                   cpu_threads_per_worker=args.cpu_threads))
        try:
            results = drain_queue(jobs, tokens, run_job)
        except BaseException:
            with lock:
                cancelled.set()
                owned = list(active)
            for process in owned:
                stop_owned(process)
            raise
        failures = [r for r in results if r['status'] != 'complete']
        # Also include launch exceptions that happened before a child could start.
        with lock:
            records.update({r['job']: r for r in results})
            publish()
        if failures:
            raise RuntimeError('Failed diagnostic jobs: ' + ', '.join(r['job'] for r in failures)
                               + '. See launcher_logs and parallel_status.json; rerun the same command after fixing the cause.')
    return results
