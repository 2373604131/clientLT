"""CPU-light protocol, source provenance, and transactional run registration."""
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import uuid

REPO = Path(__file__).resolve().parents[2]
CONFIG = REPO / 'configs/benchmarks/clientlt_seed42.json'
FACTOR_CONFIG = REPO / 'configs/benchmarks/factor_baselines_seed42.json'
LONGTAIL_CONFIG = REPO / 'configs/benchmarks/longtail_baselines_seed42.json'
METHODS = ('fedavg-lora', 'capt', 'fedpurel', 'fedntd', 'a', 'ab')
FACTOR_METHODS = ('ffa-lora', 'rolora', 'fedsvd', 'lora-a2')
LONGTAIL_METHODS = ('fedavg-lora-la', 'fedlf', 'fedyoyo', 'fedrela')
ALL_METHODS = METHODS + FACTOR_METHODS + LONGTAIL_METHODS
LABELS = {
    'fedavg-lora': 'FedAvg + LoRA (joint A/B, CE)',
    'fedavg-lora-la': 'FedAvg + LoRA + global LA (joint A/B)',
    'capt': 'CAPT (fixed aggregation; protocol-adapted)',
    'fedpurel': 'FedPuReL-Global (matched LoRA; protocol-adapted)',
    'fedntd': 'FedNTD (CLIP-LoRA adaptation)',
    'a': 'A (Full-CP)', 'ab': 'A+B (class-cyclic w=0.35)',
    'ffa-lora': 'FFA-LoRA (shared CLIP-LoRA; protocol-adapted)',
    'rolora': 'RoLoRA (alternating shared factors; protocol-adapted)',
    'fedsvd': 'FedSVD (non-DP shared CLIP-LoRA; protocol-adapted)',
    'lora-a2': 'LoRA-A2 (global rank4, local budget2; protocol-adapted)',
    'fedlf': 'FedLF (CLIP-LoRA adaptation)',
    'fedyoyo': 'FedYoYo (CLIP-LoRA adaptation; estimated prior)',
    'fedrela': 'FedAvg-LoRA + FedReLa (CE host; one-shot relabel)',
}
REFERENCE_FILES = ('partition_manifest.csv', 'bridge_metadata.json', 'protocol/full_schedule.json',
                   'protocol/eri_protocol.json', 'protocol/probe_manifest.csv')
METRICS = ('overall_acc', 'head20_acc', 'middle60_acc', 'bottom20_tail_acc', 'non_tail_acc')


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')
    tmp.replace(path)


def read_csv(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    with tmp.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def manifest_rows(path):
    rows = [{k: int(r[k]) for k in ('raw_sample_id', 'class_id', 'client_id', 'local_position')}
            for r in read_csv(path)]
    rows.sort(key=lambda r: (r['client_id'], r['local_position']))
    if not rows or len({r['raw_sample_id'] for r in rows}) != len(rows):
        raise ValueError('Empty partition or duplicate raw training samples')
    if any(not 0 <= r['class_id'] < 100 or not 0 <= r['client_id'] < 30 or r['raw_sample_id'] < 0 for r in rows):
        raise ValueError('Invalid partition indices')
    for client in range(30):
        positions = [r['local_position'] for r in rows if r['client_id'] == client]
        if not positions or positions != list(range(len(positions))):
            raise ValueError('Missing client or non-contiguous local positions')
    if {r['class_id'] for r in rows} != set(range(100)):
        raise ValueError('Expected all 100 classes')
    return rows


def reference_contract(root):
    root = Path(root)
    for name in REFERENCE_FILES:
        if not (root / name).is_file():
            raise FileNotFoundError(f'Missing reference file: {root / name}; use the complete Full-10/A/S reference run')
    meta = read_json(root / 'bridge_metadata.json')
    if meta['topology'] != 'client-longtail' or len(meta['classnames']) != 100:
        raise ValueError('This first comparison suite is CIFAR100 Client-LT only')
    if not meta.get('resolved_config'):
        raise ValueError('Reference must include resolved_config for exact model/input reconstruction')
    rows = manifest_rows(root / 'partition_manifest.csv')
    schedule = read_json(root / 'protocol/full_schedule.json')
    schedule = schedule['schedule'] if isinstance(schedule, dict) else schedule
    if len(schedule) != 100 or any(sorted(x) != list(range(30)) for x in schedule):
        raise ValueError('Expected 100 rounds with exactly 30 distinct clients each')
    if digest(schedule) != meta['schedule_sha256'] or digest(sorted(r['raw_sample_id'] for r in rows)) != meta['pool_sha256']:
        raise ValueError('Reference manifest/schedule disagrees with source metadata')
    return dict(partition=digest(rows), schedule=meta['schedule_sha256'], pool=meta['pool_sha256'],
                test=meta['test_sha256'], classnames=digest(meta['classnames']),
                reference_files={p: file_hash(root / p) for p in REFERENCE_FILES})


def source_hashes():
    from tools.sfra.ab_validation import code_hashes
    files = code_hashes(REPO)  # normalize newlines, as in the frozen A/AB receipts
    extra = [CONFIG, FACTOR_CONFIG, LONGTAIL_CONFIG, REPO / 'scripts/run_longtail_benchmarks.py',
             REPO / 'scripts/vendor_paper_baselines.py', REPO / 'scripts/run_factor_benchmarks.py',
             REPO / 'scripts/run_paper_benchmarks.py', REPO / 'scripts/train_paper_baseline.py',
             REPO / 'scripts/collect_paper_benchmarks.py', REPO / 'trainers/capt.py', REPO / 'loss/prompt_loss.py']
    for folder in ('tools/benchmarks', 'trainers/baselines', 'clip', 'utils/loralib', 'Dassl/dassl/data/transforms'):
        extra.extend((REPO / folder).rglob('*.py'))
    for path in extra:
        files[path.relative_to(REPO).as_posix()] = hashlib.sha256(path.read_text(encoding='utf-8').encode()).hexdigest()
    for method in ('capt', 'fedntd', 'fedpurel', *FACTOR_METHODS, 'fedlf', 'fedyoyo', 'fedrela'):
        base = REPO / 'third_party/paper_baselines' / method
        upstream = read_json(base / 'UPSTREAM.json')
        for name, expected in upstream['files'].items():
            if file_hash(base / name) != expected:
                raise ValueError('Official reference file changed: ' + str(base / name))
        files[(base / 'UPSTREAM.json').relative_to(REPO).as_posix()] = file_hash(base / 'UPSTREAM.json')
    return files


def make_job(method, reference, data_root, output, workers=8, smoke=False, suite='main'):
    if method not in ALL_METHODS or workers < 0:
        raise ValueError('Invalid method or worker count')
    if smoke and method in ('a', 'ab'):
        raise ValueError('Smoke mode is for the new external adapters; frozen A/AB remain 100-round jobs')
    if suite not in ('main', 'factors', 'longtail'):
        raise ValueError('Unknown benchmark suite')
    config = read_json(CONFIG)
    if suite == 'factors' or method in FACTOR_METHODS:
        config['factor_baselines'] = read_json(FACTOR_CONFIG)
    if suite == 'longtail' or method in ('fedlf', 'fedyoyo', 'fedrela'):
        config['longtail_baselines'] = read_json(LONGTAIL_CONFIG)
    return dict(schema='paper_benchmarks_seed42_v1', method=method, label=LABELS[method], seed=42,
                config=config, reference_run=str(Path(reference).resolve()),
                data_root=str(Path(data_root).resolve()), output_root=str(Path(output).resolve()),
                workers=workers, smoke=bool(smoke), source_hashes=source_hashes(),
                protocol=reference_contract(reference))


def job_path(root, method):
    return Path(root) / 'jobs' / (method + '.json')


def job_id(job):
    return digest(job)


def register(job):
    from scripts.run_ab_validation import file_lock
    root = Path(job['output_root'])
    with file_lock(root / 'locks/registration.lock'):
        mode = root / 'suite.json'
        contract = dict(config=job['config'], protocol=job['protocol'], smoke=job['smoke'])
        if mode.exists() and read_json(mode) != contract:
            raise ValueError('Existing suite has another protocol/config/smoke mode; use a new output root')
        path = job_path(root, job['method'])
        if path.exists() and read_json(path) != job:
            raise ValueError('Registered job changed (source/settings/path); use original settings or a new output root')
        write_json(mode, contract)
        write_json(path, job)


def run_path(job):
    if job['method'] not in ('a', 'ab'):
        return Path(job['output_root']) / 'runs' / job['method']
    from scripts.run_ab_validation import launcher_args
    from scripts.run_cliplora_sfra import run_directory
    from types import SimpleNamespace
    args = SimpleNamespace(output_root=Path(job['output_root']) / 'ab_runs',
                           partition='client-longtail', protocol_seed=42, dirichlet_beta=.5)
    return run_directory(launcher_args(args, job['method'], 42))


def preflight(job, require_cuda=False):
    if source_hashes() != job['source_hashes']:
        raise ValueError('Source changed after job registration')
    if reference_contract(job['reference_run']) != job['protocol']:
        raise ValueError('Reference changed after job registration')
    folder = Path(job['data_root']) / 'cifar-100/cifar-100-python'
    for name in ('train', 'test', 'meta'):
        if not (folder / name).is_file() or (folder / name).stat().st_size == 0:
            raise FileNotFoundError('Missing CIFAR100 data: ' + str(folder / name))
    missing = [name for name in ('torch', 'torchvision', 'yacs', 'yaml', 'ftfy', 'regex', 'scipy', 'sklearn',
                                'timm', 'tensorboard', 'tabulate', 'gdown', 'matplotlib', 'PIL', 'tqdm')
               if importlib.util.find_spec(name) is None]
    if missing:
        raise RuntimeError('Missing dependencies in this Python environment: ' + ', '.join(missing))
    if require_cuda:
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError('Training requires CUDA; this preflight does not launch a CPU training fallback')
    return 'ready'
