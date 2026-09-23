"""Plan/run the five frozen Method A settings and summarize paired results."""
import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import uuid

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from scripts.run_cliplora_sfra import run_directory
from tools.sfra.maintext import (METHODS, PLAN_NAME, SCHEMA, FROZEN, code_hashes, digest,
                                load_json, validate_config, check_receipt, summarize_maintext)


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp')
    temp.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False)+'\n', encoding='utf-8')
    temp.replace(path)


def plan_jobs(args):
    hashes = code_hashes(REPO)
    jobs = []
    for seed in args.seeds:
        for method in METHODS:
            single = argparse.Namespace(**{**vars(args), 'seed': seed, 'method': method},
                                        retention_weight=10., classification_weight=1., b_transfer=False)
            run = run_directory(single).relative_to(args.output_root.resolve()).as_posix()
            jobs.append(dict(run=run, seed=seed, method=method, partition=args.partition,
                             protocol_seed=args.protocol_seed, dirichlet_beta=args.dirichlet_beta,
                             witness_batch_size=args.witness_batch_size, num_workers=args.num_workers,
                             data_root=str(args.data_root.resolve()),
                             reference_run=str(args.reference_run.resolve()) if args.reference_run else None,
                             schedule_file=str(args.schedule_file.resolve()) if args.schedule_file else None,
                             code_sha256=hashes))
    return jobs


def merge_plan(root, jobs):
    path = Path(root)/PLAN_NAME
    existing = load_json(path) if path.exists() else dict(schema_version=SCHEMA, frozen=FROZEN, jobs=[])
    if existing.get('schema_version') != SCHEMA or existing.get('frozen') != FROZEN:
        raise ValueError('The output root contains a different experiment contract; use a new --output-root')
    saved = {j['run']: j for j in existing['jobs']}
    for job in jobs:
        if job['run'] in saved and saved[job['run']] != job:
            raise ValueError(f'Configuration or training source changed for {job["run"]}; use a new --output-root')
        saved[job['run']] = job
    existing['jobs'] = list(saved.values())
    write_json(path, existing)


def command_for(job, root):
    command = [sys.executable, '-u', str(REPO/'scripts/run_cliplora_sfra.py'),
               '--method', job['method'], '--retention-weight', '10', '--classification-weight', '1',
               '--seed', str(job['seed']), '--protocol-seed', str(job['protocol_seed']),
               '--partition', job['partition'], '--dirichlet-beta', str(job['dirichlet_beta']),
               '--data-root', job['data_root'], '--output-root', str(Path(root).resolve()),
               '--num-workers', str(job['num_workers']), '--witness-batch-size', str(job['witness_batch_size'])]
    if job['reference_run']:
        command += ['--reference-run', job['reference_run']]
    else:
        command += ['--fresh-protocol']
    if job['schedule_file']:
        command += ['--schedule-file', job['schedule_file']]
    return command


def start_record_path(root, job):
    return Path(root)/'maintext_launches'/(digest(job['run'])+'.json')


def verify_or_recover_receipt(run, job, root, resume):
    try:
        check_receipt(run, job)
    except FileNotFoundError:
        # A hard-killed launcher cannot execute its finally block. Recover only
        # from its pre-launch registration, with the same source verified now.
        if not resume:
            raise ValueError(f'Missing run receipt: {run}; use --resume after an interrupted launcher')
        started = load_json(start_record_path(root, job))
        if started != dict(schema_version=SCHEMA, job=job):
            raise ValueError(f'Launch registration differs: {run}')
        write_json(run/'maintext_run.json', dict(schema_version=SCHEMA, job=job, code_unchanged=True))


def execute_job(job, root, resume=False):
    if code_hashes(REPO) != job['code_sha256']:
        raise ValueError('Training source differs from the plan; use a new --output-root')
    run = Path(root)/job['run']
    command = command_for(job, root)
    if (run/'completion.json').is_file():
        validate_config(run, job)
        verify_or_recover_receipt(run, job, root, resume)
        if load_json(run/'completion.json').get('completed_round') != 100:
            raise ValueError(f'Invalid completion marker: {run}')
        print(f'SKIP completed: {run}', flush=True)
        return
    if run.exists() and any(run.iterdir()):
        if not resume:
            raise ValueError(f'Partial run: {run}. Use --resume to continue its saved configuration.')
        if not (run/'checkpoints/sfra_last.pt').is_file():
            raise ValueError(f'No resumable checkpoint in {run}; keep it for diagnosis and use a new output root.')
        validate_config(run, job)
        verify_or_recover_receipt(run, job, root, resume)
        command += ['--resume']
    render = subprocess.list2cmdline(command) if os.name == 'nt' else shlex.join(command)
    print(render, flush=True)
    write_json(start_record_path(root, job), dict(schema_version=SCHEMA, job=job))
    try:
        subprocess.run(command, cwd=REPO, check=True)
    finally:
        # Preserve provenance even when a training process exits with an error.
        unchanged = code_hashes(REPO) == job['code_sha256']
        if run.is_dir():
            write_json(run/'maintext_run.json', dict(schema_version=SCHEMA, job=job, code_unchanged=unchanged))
    if not unchanged:
        raise ValueError('Training source changed during the run; results cannot be pooled into this suite')
    if not (run/'completion.json').is_file():
        raise ValueError(f'Training returned without a completion marker: {run}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=['plan', 'train', 'summary', 'pack'], default='plan')
    parser.add_argument('--methods', nargs='+', choices=METHODS, default=list(METHODS))
    parser.add_argument('--seeds', nargs='+', type=int, default=[42])
    parser.add_argument('--protocol-seed', type=int, default=42)
    parser.add_argument('--partition', choices=['client-longtail', 'noniid-labeldir-fine'], default='client-longtail')
    parser.add_argument('--dirichlet-beta', type=float, default=.5)
    parser.add_argument('--data-root', type=Path, default=Path('DATA'))
    parser.add_argument('--output-root', type=Path, default=Path('output/cifar100_LT/method_a_maintext'))
    parser.add_argument('--reference-run', type=Path)
    parser.add_argument('--schedule-file', type=Path)
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--witness-batch-size', type=int, default=8)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    # Relative paths follow repository conventions, independently of the invoking shell's cwd.
    os.chdir(REPO)
    args.output_root = args.output_root.resolve()
    if args.stage in ('summary', 'pack'):
        statuses = summarize_maintext(args.output_root)
        if args.stage == 'pack':
            from tools.sfra.summary import pack
            pack(args.output_root)
        if any(r['status'].startswith('invalid') for r in statuses.values()):
            raise SystemExit('Invalid/unpaired results were excluded; see maintext_analysis/status.csv')
        return
    import math
    if len(set(args.seeds)) != len(args.seeds) or any(s < 0 for s in args.seeds) or args.protocol_seed < 0:
        parser.error('Seeds must be distinct nonnegative integers')
    if not math.isfinite(args.dirichlet_beta) or args.dirichlet_beta <= 0:
        parser.error('--dirichlet-beta must be finite and positive')
    if args.num_workers < 0 or args.witness_batch_size < 1:
        parser.error('Invalid worker or witness batch size')
    if args.reference_run and args.schedule_file:
        parser.error('--reference-run already specifies its schedule; do not also pass --schedule-file')
    jobs = plan_jobs(args)
    merge_plan(args.output_root, jobs)
    selected = [job for job in jobs if job['method'] in args.methods]
    print(f'Frozen Method A: lambda=10, mu=1, 100 rounds, B transfer off. Selected runs: {len(selected)}.', flush=True)
    print(f'Plan: {args.output_root/PLAN_NAME}', flush=True)
    if args.stage == 'plan':
        for job in selected:
            command = command_for(job, args.output_root)
            print(subprocess.list2cmdline(command) if os.name == 'nt' else shlex.join(command))
        return
    for job in selected:
        execute_job(job, args.output_root, args.resume)
    statuses = summarize_maintext(args.output_root)
    if any(r['status'].startswith('invalid') for r in statuses.values()):
        raise SystemExit('Protocol checks failed; inspect maintext_analysis/status.csv')


if __name__ == '__main__':
    main()
