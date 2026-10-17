"""Only complete, identity-checked runs enter the main table. Never best-round selection."""
from collections import defaultdict
from pathlib import Path
import csv
import math
import tarfile
import numpy as np

from tools.benchmarks.common import read_json, write_json, write_csv as _write_csv, digest
from tools.multidataset.protocol import metrics

METRICS = ('overall_acc', 'macro_acc', 'head_acc', 'middle_acc', 'tail_acc')


def write_csv(path, rows):
    if rows:
        return _write_csv(path, rows)
    # Empty analyses remain parseable tables, never headerless files.
    fields = ['batch', 'dataset', 'specialization_lambda', 'seed', 'method']
    fields += ['status', 'completed_round', 'error'] if Path(path).name == 'status.csv' else list(METRICS)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open('w', encoding='utf-8', newline='') as stream:
        csv.DictWriter(stream, fieldnames=fields).writeheader()


def audit_job(job):
    root = Path(job['run_root'])
    identity = digest(job)
    complete = read_json(root / 'completion.json')
    final = 2 if job['smoke'] else 100
    if complete != dict(job_id=identity, completed_round=final, smoke=job['smoke']):
        raise ValueError('Invalid completion record')
    receipt = read_json(root / 'receipt.json')
    if receipt.get('job_id') != identity or receipt.get('source_unchanged') is not True or receipt.get('exit_code') != 0:
        raise ValueError('Missing successful source-verified worker receipt')
    if read_json(root / 'job.json') != job:
        raise ValueError('Run metadata differs from registered job')
    rows, expected_total = [], None
    test = read_json(Path(job['protocol_root']) / 'test.json')
    expected_total = np.bincount([r['class_id'] for r in test], minlength=job['protocol']['classes']).tolist()
    for rnd in range(final + 1):
        row = read_json(root / 'rounds' / f'r{rnd:03d}.json')
        if row['job_id'] != identity or row['round'] != rnd or row['total'] != expected_total:
            raise ValueError('Round identity/test population changed')
        values = metrics(row['correct'], row['total'], job['protocol']['groups'])
        if set(values) != set(row['metrics']) or any(not math.isfinite(row['metrics'][key]) or abs(values[key] - row['metrics'][key]) > 1e-9 for key in values):
            raise ValueError('Saved metrics disagree with per-class counts')
        rows.append(dict(round=rnd, **values))
    if job['method'] == 'joint-old-ab':
        cfg = read_json(root / 'sfra_config.json')
        transfer = cfg['b_transfer']
        if (cfg['variant'] != 'full-cp' or cfg['retention_weight'] != 10. or cfg['classification_weight'] != 1.
            or cfg['refresh_rounds'] != list(range(1, 91)) or transfer['mode'] != 'shared'
            or transfer['tail_weight'] != .35 or transfer['non_tail_sampling'] != 'class-cyclic'
            or transfer['learning_rate'] != .3 or transfer['probe_step'] != .1 or transfer['regularization'] != .001
            or transfer['rounds'] != ([1] if job['smoke'] else list(range(30, 101, 10)))
            or cfg['joint_aggregation']['weights'] != job['protocol']['joint_solution']['weights']):
            raise ValueError('Main method is not the frozen joint + old A + old B recipe')
    return rows


def collect(root):
    from scripts.run_ab_validation import file_lock
    root = Path(root).resolve()
    with file_lock(root / 'locks/analysis.lock'):
        return _collect(root)


def _collect(root):
    root = Path(root).resolve()
    status, performance, curves = [], [], []
    for path in sorted(root.glob('*/jobs/*.json')):
        job = read_json(path)
        run = Path(job['run_root'])
        base = dict(batch=job['batch'], dataset=job['protocol']['dataset'],
            specialization_lambda=job['protocol']['specialization_lambda'], seed=job['seed'],
            method=job['method'], smoke=job['smoke'])
        try:
            progress = read_json(run / 'progress.json') if (run / 'progress.json').exists() else {}
        except (ValueError, OSError):
            progress = {'phase': 'progress_file_updating'}
        state, error = 'pending', ''
        if progress:
            state = 'incomplete'
        if (run / 'failure.json').exists():
            state = 'failed_or_interrupted'
        if (run / 'completion.json').exists():
            try:
                records = audit_job(job)
                state = 'smoke_verified' if job['smoke'] else 'complete_verified'
                if not job['smoke']:
                    last = [r for r in records if 81 <= r['round'] <= 100]
                    performance.append({**base, 'aggregation': 'mean_committed_rounds_81_100',
                        **{key: float(np.mean([r[key] for r in last])) for key in records[0] if key != 'round'}})
                    curves.extend({**base, **r} for r in records)
            except (OSError, ValueError, KeyError, TypeError) as exc:
                state, error = 'audit_failed', str(exc)
        status.append(dict(**base, status=state, completed_round=progress.get('completed_round', 0),
            active_round=progress.get('active_round', ''), phase=progress.get('phase', ''),
            client_position=progress.get('client_position', ''), client_id=progress.get('client_id', ''), error=error))
    out = root / 'analysis'
    write_csv(out / 'status.csv', status)
    write_csv(out / 'performance.csv', performance)
    write_csv(out / 'imagenet_standard_groups.csv', [
        {key: r[key] for key in ('seed', 'method', 'overall_acc', 'macro_acc', 'many_shot_acc', 'medium_shot_acc', 'few_shot_acc')}
        for r in performance if r['dataset'] == 'imagenet-lt'])
    write_csv(out / 'curves.csv', curves)
    grouped = defaultdict(list)
    for row in performance:
        grouped[(row['batch'], row['method'])].append(row)
    seed_summary = []
    for (batch, method), rows in sorted(grouped.items()):
        result = dict(batch=batch, method=method, seeds=','.join(str(r['seed']) for r in rows), n_seeds=len(rows))
        for key in METRICS:
            values = [r[key] for r in rows]
            result[key + '_mean'] = float(np.mean(values))
            result[key + '_std'] = float(np.std(values, ddof=1)) if len(values) > 1 else ''
        seed_summary.append(result)
    write_csv(out / 'seed_summary.csv', seed_summary)
    for filename, batches in (('main_table_cifar_lambda.csv', ['cifar-lambda050', 'cifar-lambda075', 'cifar-lambda100']),
                              ('main_table_multidataset.csv', ['imagenet', 'food', 'cub', 'dtd'])):
        keys = sorted({(r['seed'], r['method']) for r in performance if r['batch'] in batches})
        lookup = {(r['batch'], r['seed'], r['method']): r for r in performance}
        table = []
        for seed, method in keys:
            row = dict(seed=seed, method=method)
            for batch in batches:
                values = lookup.get((batch, seed, method), {})
                row.update({batch+'_'+key: values.get(key, '') for key in METRICS})
            table.append(row)
        write_csv(out / filename, table)
    report = ['# Multi-dataset Client-LT results', '',
        'Main method: joint + old Full-CP A + old shared-C B. Only verified formal runs enter this table.',
        'Each row is the fixed mean of committed rounds 81–100. It is not a multi-seed standard deviation.', '',
        '| Batch | Seed | Method | Overall | Macro | Head | Middle | Tail |',
        '|---|---:|---|---:|---:|---:|---:|---:|']
    for row in performance:
        report.append('| ' + ' | '.join([row['batch'], str(row['seed']), row['method']] + [f'{row[k]:.4f}' for k in METRICS]) + ' |')
    report.extend(['', f'Formal verified runs: {len(performance)}. Registered jobs including smoke: {len(status)}.',
        'See status.csv for pending, failed and incomplete runs; none are replaced with zeros.',
        'Cross-seed summaries are in seed_summary.csv. Compare methods on the SAME seed set.'])
    (out / 'report.md').write_text('\n'.join(report) + '\n', encoding='utf-8')
    return status


def pack(root):
    root = Path(root).resolve()
    collect(root)
    target = root.parent / (root.name + '_analysis.tar.gz')
    allowed = {'.json', '.csv', '.yaml', '.md', '.txt'}
    with tarfile.open(target, 'w:gz') as archive:
        for path in sorted(root.rglob('*')):
            if path.is_file() and path.suffix in allowed and 'checkpoints' not in path.parts:
                archive.add(path, arcname=str(Path(root.name) / path.relative_to(root)))
    return target
