"""Registration, source locks, one worker per GPU and resumable batch execution."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import hashlib
import importlib.util
import os
import subprocess
import sys
import threading
import time

from tools.benchmarks.common import REPO, read_json, write_json, digest
from tools.multidataset.catalog import BATCHES, GROUPS, METHODS
from tools.multidataset.protocol import verify


def source_hashes():
    files = []
    for folder in ('tools/multidataset', 'tools/benchmarks', 'tools/client_aggregation', 'tools/sfra',
                   'trainers', 'utils', 'clip', 'Dassl/dassl'):
        files.extend((REPO / folder).rglob('*.py'))
    files.extend(p for p in (REPO / 'configs/benchmarks').glob('*.json') if p.name != 'multidataset_data.json')
    files.extend(REPO / 'scripts' / name for name in ('run_multidataset_benchmarks.py',
        'train_multidataset_benchmark.py', 'prepare_multidataset_data.py', 'run_ab_validation.py'))
    files.extend((REPO / 'third_party/paper_baselines').rglob('*.py'))
    files.extend((REPO / 'third_party/paper_baselines').rglob('UPSTREAM.json'))
    return {p.relative_to(REPO).as_posix(): hashlib.sha256(p.read_text(encoding='utf-8-sig').encode()).hexdigest()
            for p in sorted(set(files))}


def verify_source(job):
    if source_hashes() != job['source_hashes']:
        raise ValueError('Training source changed. Preserve this run and use its original source snapshot to resume.')


def selected_methods(group, methods):
    values = methods or (METHODS if group == 'all' else GROUPS[group])
    if len(values) != len(set(values)) or any(m not in METHODS for m in values):
        raise ValueError('Select distinct registered methods')
    return list(values)


def options():
    config = read_json(REPO / 'configs/benchmarks/clientlt_seed42.json')
    config['factor_baselines'] = read_json(REPO / 'configs/benchmarks/factor_baselines_seed42.json')
    config['longtail_baselines'] = read_json(REPO / 'configs/benchmarks/longtail_baselines_seed42.json')
    # All group IDs are supplied by each dataset protocol, never this CIFAR template.
    for key in ('head_ids', 'middle_ids', 'tail_ids'):
        config.pop(key, None)
    return config


def make_job(root, batch, method, seed, workers, reference, smoke=False):
    root = Path(root).resolve()
    protocol_root = root / batch / 'protocol'
    protocol = verify(protocol_root)
    config = options()
    config['seed'] = seed
    return dict(schema='clientlt_multidataset_job_v1', batch=batch, method=method, seed=seed,
        workers=workers, smoke=smoke, config=config, protocol=protocol,
        protocol_root=str(protocol_root), reference_run=str(Path(reference).resolve()),
        run_root=str(root / batch / ('smoke' if smoke else 'runs') / f'seed{seed}' / method),
        source_hashes=source_hashes(), numerical_policy='cudnn deterministic, benchmark disabled; fp32',
        execution='one GPU per method; serial clients; all 30 clients per formal round')


def register(root, job):
    from scripts.run_ab_validation import file_lock
    target = Path(root) / job['batch'] / 'jobs' / f"{job['method']}_seed{job['seed']}{'_smoke' if job['smoke'] else ''}.json"
    with file_lock(target.with_suffix('.lock')):
        if target.exists() and read_json(target) != job:
            raise ValueError('Registered experiment changed; use a new output root: ' + str(target))
        write_json(target, job)
    return target


def preflight(jobs):
    missing = [name for name in ('torch', 'torchvision', 'yacs', 'yaml', 'ftfy', 'regex', 'scipy',
        'sklearn', 'timm', 'tensorboard', 'tabulate', 'gdown', 'matplotlib', 'PIL', 'tqdm')
        if importlib.util.find_spec(name) is None]
    if missing:
        raise RuntimeError('Missing server environment dependencies: ' + ', '.join(missing))
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('No CUDA. Plan/prepare/status/summary/pack are CPU commands; training requires the server.')
    checked = set()
    for job in jobs:
        verify_source(job)
        if job['protocol_root'] not in checked:
            verify(job['protocol_root'], images=True)
            checked.add(job['protocol_root'])


def run_jobs(job_paths, gpus, stop_after=0, *, smoke_paths=None, duration_estimates=None, schedule_path=None):
    """Each GPU takes the longest pending job, with its own smoke gate first."""
    from tools.multidataset.scheduling import estimate_seconds, prioritize, project_loads
    if not gpus or len(gpus) != len(set(gpus)) or any(g < 0 for g in gpus):
        raise ValueError('GPU IDs must be distinct and nonnegative')
    if len(set(map(str, job_paths))) != len(job_paths):
        raise ValueError('Duplicate jobs in the shared queue')
    identity = lambda job: (job['batch'], job['method'], job['seed'])
    gates = {identity(read_json(p)): p for p in (smoke_paths or [])}
    if smoke_paths is not None and len(gates) != len(job_paths):
        raise ValueError('Every formal job needs exactly one smoke gate')
    entries = []
    for path in job_paths:
        job = read_json(path)
        gate = gates.get(identity(job))
        if smoke_paths is not None and gate is None:
            raise ValueError('Missing corresponding smoke gate')
        entries.append(dict(path=str(path), key='/'.join(map(str, identity(job))),
            batch=job['batch'], method=job['method'], seed=job['seed'],
            smoke_path=str(gate) if gate else None,
            estimated_seconds=estimate_seconds(job, duration_estimates, stop_after)))
    pending = prioritize(entries)
    projection = project_loads(pending, gpus)
    for gpu, value in projection.items():
        print(f"PLAN GPU={gpu}: {len(value['jobs'])} jobs, estimated {value['estimated_seconds']/3600:.2f} hours", flush=True)
    state = dict(schema='multidataset_shared_queue_v1', policy='longest estimated remaining job first; work conserving',
        gpus=gpus, estimates=duration_estimates or {'basis': 'relative method costs'},
        projection=projection, pending=pending, active={}, completed=[], failed=[], started=time.time())
    lock = threading.Lock()
    failures = []

    def persist():
        if schedule_path is not None:
            write_json(schedule_path, dict(state, updated=time.time()))

    def launch(path, gpu, limit):
        path = Path(path)
        job = read_json(path)
        log = path.parent.parent / 'launcher_logs' / (path.stem + '.log')
        log.parent.mkdir(parents=True, exist_ok=True)
        command = [sys.executable, '-u', str(REPO / 'scripts/train_multidataset_benchmark.py'),
                   '--job', str(path), '--resume', '--stop-after', str(limit)]
        environment = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
        phase = 'smoke' if job['smoke'] else 'formal'
        print(f"START {phase} {job['batch']}/{job['method']}/seed{job['seed']} GPU={gpu} log={log}", flush=True)
        with log.open('a', encoding='utf-8') as stream:
            result = subprocess.run(command, cwd=REPO, env=environment, stdout=stream, stderr=subprocess.STDOUT)
        return result.returncode, str(log)

    def queue(gpu):
        while True:
            with lock:
                if not pending:
                    return
                entry = pending.pop(0)
                active = dict(entry, gpu=gpu, started=time.time(), phase='smoke' if entry['smoke_path'] else 'formal')
                state['active'][str(gpu)] = active
                persist()
            error, log = '', ''
            try:
                if entry['smoke_path']:
                    code, log = launch(entry['smoke_path'], gpu, 0)
                    if code:
                        raise RuntimeError('Smoke failed; formal job was not started')
                    from tools.multidataset.collect import audit_job
                    audit_job(read_json(entry['smoke_path']))
                with lock:
                    active['phase'] = 'formal'
                    persist()
                code, log = launch(entry['path'], gpu, stop_after)
                if code:
                    raise RuntimeError('Worker exited with code ' + str(code))
            except Exception as exc:
                error = str(exc)
            with lock:
                finished = dict(active, finished=time.time(), wall_seconds=time.time()-active['started'], log=log)
                del state['active'][str(gpu)]
                if error:
                    finished['error'] = error
                    failures.append(log or entry['path'])
                    state['failed'].append(finished)
                    print('FAILED:', entry['key'], error, log, flush=True)
                else:
                    state['completed'].append(finished)
                    print('DONE:', entry['key'], f"GPU={gpu}", flush=True)
                persist()

    persist()
    with ThreadPoolExecutor(max_workers=len(gpus)) as executor:
        futures = [executor.submit(queue, gpu) for gpu in gpus]
        for future in futures:
            future.result()
    if failures:
        raise RuntimeError('Some independent jobs failed; rerun resumes successful checkpoints. Logs: ' + ', '.join(failures))


def worker(job, resume=False, stop_after=0):
    from scripts.run_ab_validation import file_lock
    from tools.multidataset.collect import audit_job
    root = Path(job['run_root'])
    with file_lock(root.parent / (root.name + '.lock'), timeout=.1):
        verify_source(job)
        if verify(job['protocol_root'], images=True) != job['protocol']:
            raise ValueError('Registered protocol changed')
        if (root / 'completion.json').exists() and (root / 'receipt.json').exists():
            audit_job(job)
            print('SKIP verified complete:', root, flush=True)
            return
        if root.exists() and any(root.iterdir()):
            if not (root / 'job.json').exists() or read_json(root / 'job.json') != job:
                raise ValueError('Run directory is occupied by another job')
            if not resume:
                raise ValueError('Use --resume for an existing run')
        root.mkdir(parents=True, exist_ok=True)
        write_json(root / 'job.json', job)
        import torch
        from tools.multidataset.model import build_config, build_model, seed_all
        from tools.multidataset.data import build_data
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError('Worker requires exactly one visible CUDA GPU')
        torch.set_num_threads(1)
        seed_all(job['seed'])
        cfg = build_config(job)
        loaders, statistics, test_loader, clients = build_data(job, cfg)
        model, teacher = build_model(job, cfg, torch.device('cuda:0'))
        (root / 'resolved_config.yaml').write_text(str(cfg), encoding='utf-8')
        write_json(root / 'environment.json', dict(torch=str(torch.__version__), cuda=torch.version.cuda,
            gpu=torch.cuda.get_device_name(0), visible_devices=os.getenv('CUDA_VISIBLE_DEVICES')))
        try:
            if job['method'] == 'joint-old-ab':
                from tools.multidataset.ours import train
                train(job, model, cfg, loaders, test_loader, clients, resume, stop_after)
            else:
                from tools.multidataset.external import train
                train(job, model, teacher, loaders, statistics, test_loader, resume, stop_after)
            verify_source(job)
            write_json(root / 'receipt.json', dict(job_id=digest(job), source_unchanged=True,
                peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(), exit_code=0))
            if (root / 'completion.json').exists():
                audit_job(job)
        except Exception as error:
            write_json(root / 'failure.json', dict(error=repr(error), job_id=digest(job)))
            raise
