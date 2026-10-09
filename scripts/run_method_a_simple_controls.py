"""Plan, run and collect the frozen seed42 TailRW and coverage-priority controls."""
import argparse
import os
from pathlib import Path
import shlex
import subprocess
import sys
from types import SimpleNamespace

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from scripts.run_ab_validation import file_lock, write_json
from scripts.run_cliplora_a_refresh import build_command
from scripts.run_cliplora_sfra import prepare_protocol
from tools.sfra.simple_controls import (CONTRACT, METHODS, NEW_METHODS, PLAN_NAME, SCHEMA,
    code_hashes, digest, implementation_variant, load_json, protocol_info, static_tables,
    probe_manifest_replay, read_csv, validate_config, write_table)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('plan', 'preflight', 'train', 'summary', 'pack'), default='plan')
    parser.add_argument('--methods', nargs='+', choices=METHODS, default=list(NEW_METHODS))
    parser.add_argument('--seed', type=int, choices=(42,), default=42)
    parser.add_argument('--reference-run', type=Path, default=Path('references/full10_clientlt'))
    parser.add_argument('--data-root', type=Path, default=Path('DATA'))
    parser.add_argument('--output-root', type=Path, default=Path('output/cifar100_LT/method_a_simple_controls_v1'))
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--baseline', action='append', default=[], metavar='METHOD=PATH',
                        help='Reuse completed seed42 s/full-cp; otherwise detect the existing AB result tree')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args(argv)
    if len(set(args.methods)) != len(args.methods) or args.num_workers < 0:
        parser.error('Methods must be distinct; num-workers must be nonnegative')
    return args


def baseline_paths(args, saved=None):
    explicit = {}
    for item in args.baseline:
        method, sep, value = item.partition('=')
        if not sep or method not in ('s', 'full-cp') or method in explicit or not value:
            raise ValueError('Use --baseline s=PATH and/or --baseline full-cp=PATH once each')
        explicit[method] = str(Path(value).resolve())
    result = dict(saved or {})
    for method, value in explicit.items():
        if method in result and result[method] != value:
            raise ValueError('Reference path is already frozen; use a separate output root')
        result[method] = value
    for method in ('s', 'full-cp'):
        if method in args.methods:
            if method in explicit or method in result:
                raise ValueError('Do not both train and reuse the same reference: ' + method)
            continue
        if method not in result:
            arm = 's' if method == 's' else 'a'
            setting = 'baseline_protocol42' if method == 's' else 'lambda10_mu1_protocol42'
            suffix = Path('runs')/arm/'seed42/client-longtail'/method/(setting+'_fast_v2_f128_c4')
            for root in (REPO/'output/cifar100_LT/ab_decision_42_0_3407',
                         REPO/'output/ab_decision_42_0_3407_analysis/ab_decision_42_0_3407'):
                path = root/suffix
                if (path/'completion.json').is_file():
                    result[method] = str(path.resolve())
                    break
    return result


def plan_jobs(args):
    fingerprint, counts = protocol_info(args.reference_run)
    return [dict(schema_version=SCHEMA, contract=CONTRACT, method=method, seed=42,
        run=f'runs/seed42/{method}', reference_run=str(args.reference_run.resolve()),
        data_root=str(args.data_root.resolve()), num_workers=args.num_workers,
        input_fingerprint=fingerprint, counts=counts, code_sha256=code_hashes(REPO))
        for method in args.methods]


def merge_plan(args, jobs):
    root = args.output_root
    with file_lock(root/'locks/plan.lock'):
        path = root/PLAN_NAME
        plan = load_json(path) if path.is_file() else dict(contract=CONTRACT, jobs=[], baselines={})
        if plan['contract'] != CONTRACT:
            raise ValueError('Different experiment contract; use a separate output root')
        saved = {j['method']: j for j in plan['jobs']}
        for job in jobs:
            if job['method'] in saved and saved[job['method']] != job:
                raise ValueError('Registered configuration/source changed: ' + job['method'])
            if saved and any(j['input_fingerprint'] != job['input_fingerprint'] or j['counts'] != job['counts']
                             for j in saved.values()):
                raise ValueError('Cannot mix different training partitions in a suite')
            saved[job['method']] = job
        plan['baselines'] = baseline_paths(args, plan['baselines'])
        if set(plan['baselines']) & set(saved):
            raise ValueError('A reference is both reused and registered for training')
        plan['jobs'] = list(saved.values())
        write_json(path, plan)
        clients, classes = static_tables(jobs[0]['counts'])
        write_table(root/'client_weights.csv', clients)
        write_table(root/'class_coverage.csv', classes)
    return plan


def command_for(job, root, resume=False):
    run = Path(root).resolve()/job['run']
    args = SimpleNamespace(output_root=Path(root), data_root=Path(job['data_root']), seed=42,
        partition='client-longtail', rank=4, schedule_file=run/'protocol/full_schedule.json',
        matched_beta=.5, refresh_interval=10, refresh_epochs=1, refresh_lr=.001,
        resume=None, num_workers=job['num_workers'])
    command, _ = build_command(args, 'off')
    command[command.index('federated_main.py')] = str(REPO/'federated_main.py')
    command[command.index('--output-dir')+1] = str(run)
    at = command.index('DATALOADER.NUM_WORKERS')
    command[at:at] = ['--lac_method', 's', '--lac_partition_manifest', str(run/'protocol/partition_source.csv'),
        '--lac_la_tau', '1', '--lac_a_lr_mult', '1', '--sfra_variant', implementation_variant(job['method']),
        '--sfra_retention_weight', '10', '--sfra_classification_weight', '1', '--sfra_witness_batch_size', '8',
        '--sfra_fast_execution_v2', '--sfra_feedback_batch_size', '128', '--sfra_feedback_cache_gib', '4',
        '--sfra_resume', str(run/'checkpoints/sfra_last.pt') if resume else '',
        '--method_a_simple_control_manifest', str(run/'simple_control_job.json')]
    return command


def preflight_job(job):
    if job['seed'] != 42 or job['contract'] != CONTRACT or code_hashes(REPO) != job['code_sha256']:
        raise ValueError('Seed, contract or implementation differs from the registered job')
    fingerprint, counts = protocol_info(job['reference_run'])
    if fingerprint != job['input_fingerprint'] or counts != job['counts']:
        raise ValueError('Reference training partition/protocol changed')
    reference = Path(job['reference_run'])
    _, replay = probe_manifest_replay(reference/'protocol/probe_manifest.csv',
        load_json(reference/'bridge_metadata.json')['probe_manifest_sha256'])
    if replay['newline_conversion'] != 'unchanged':
        print('Probe manifest: will restore '+replay['newline_conversion']+' line endings in the run copy '
              'to reproduce recorded SHA256 '+replay['expected_sha256'], flush=True)
    dataset = Path(job['data_root'])/'cifar-100/cifar-100-python'
    for name in ('train', 'test', 'meta'):
        if not (dataset/name).is_file() or not (dataset/name).stat().st_size:
            raise ValueError('Missing/empty CIFAR-100 file: ' + str(dataset/name))


def prepare_control_protocol(job, run):
    """Preserve the historical byte hash, without editing the shared reference or metadata."""
    run = Path(run)
    prepare_protocol(SimpleNamespace(reference_run=Path(job['reference_run']), partition='client-longtail'), run)
    manifest = run/'protocol/probe_manifest.csv'
    expected = load_json(run/'protocol/source_metadata.json')['probe_manifest_sha256']
    payload, replay = probe_manifest_replay(manifest, expected)
    if replay['newline_conversion'] != 'unchanged':
        temporary = manifest.with_suffix('.csv.tmp')
        temporary.write_bytes(payload)
        temporary.replace(manifest)
        print(f'Probe manifest restored to recorded bytes ({replay["newline_conversion"]}): {expected}', flush=True)
    write_json(run/'probe_manifest_replay.json', replay)


def execute_job(job, root, resume=False):
    root = Path(root)
    run = root/job['run']
    with file_lock(root/'locks'/f'{job["method"]}.lock', timeout=.1):
        preflight_job(job)
        if (run/'completion.json').is_file():
            from tools.sfra.simple_controls_summary import read_result
            read_result(run, job['method'], job)
            print('SKIP completed:', run, flush=True)
            return
        occupied = run.exists() and any(run.iterdir())
        if occupied:
            if not resume:
                raise ValueError('Partial output exists; use --resume: ' + str(run))
            if load_json(run/'simple_control_job.json') != job or not (run/'checkpoints/sfra_last.pt').is_file():
                raise ValueError('Resume needs the original registered job and a round-boundary checkpoint')
            validate_config(run, job['method'])
        else:
            prepare_control_protocol(job, run)
            write_json(run/'simple_control_job.json', job)
        command = command_for(job, root, resume=occupied)
        if not occupied:
            write_json(run/'command.json', command)
        else:
            original = load_json(run/'command.json')
            expected = command_for(job, root)
            original[0] = expected[0]
            if original != expected:
                raise ValueError('Saved original command differs; refusing to alter resumed training')
        print(shlex.join(command), flush=True)
        try:
            subprocess.run(command, cwd=REPO, check=True)
        finally:
            unchanged = code_hashes(REPO) == job['code_sha256']
            write_json(run/'simple_control_receipt.json', dict(schema_version=SCHEMA, code_unchanged=unchanged))
        if not unchanged:
            raise ValueError('Implementation changed while this run was training')
        from tools.sfra.simple_controls_summary import read_result
        read_result(run, job['method'], job)


def main(argv=None):
    args = parse_args(argv)
    os.chdir(REPO)
    args.output_root = args.output_root.resolve()
    if args.stage in ('summary', 'pack'):
        from tools.sfra.simple_controls_summary import summarize, pack
        plan = load_json(args.output_root/PLAN_NAME)
        if args.baseline:
            with file_lock(args.output_root/'locks/plan.lock'):
                plan = load_json(args.output_root/PLAN_NAME)
                plan['baselines'] = baseline_paths(args, plan['baselines'])
                if set(plan['baselines']) & {j['method'] for j in plan['jobs']}:
                    raise ValueError('A reference is both reused and registered for training')
                write_json(args.output_root/PLAN_NAME, plan)
        status = summarize(args.output_root)
        if args.stage == 'pack':
            print('Archive:', pack(args.output_root, REPO))
        print('Report:', args.output_root/'analysis/report.md')
        if any(r['status'] in ('invalid', 'mismatch') for r in status):
            raise ValueError('Invalid/mismatched results excluded; see analysis/status.csv')
        return
    jobs = plan_jobs(args)
    plan = merge_plan(args, jobs)
    print(f'Seed42 only; selected {len(jobs)} runs. Plan: {args.output_root/PLAN_NAME}', flush=True)
    for method in ('s', 'full-cp'):
        print('Reference', method, plan['baselines'].get(method, 'not reused; use --baseline or explicitly train it'))
    if args.stage in ('preflight', 'train'):
        from tools.sfra.simple_controls_summary import read_result
        for method, path in plan['baselines'].items():
            result = read_result(path, method)
            # A stale source record must not silently qualify as a paired reference.
            from tools.sfra.simple_controls import PAIR_CODE_FILES, reference_code
            code = reference_code(path)
            if any(code[n] != jobs[0]['code_sha256'][n] for n in PAIR_CODE_FILES):
                raise ValueError('Legacy reference training implementation differs: ' + method)
            if result['counts'] != jobs[0]['counts']:
                raise ValueError('Legacy reference training partition counts differ: ' + method)
            reference = Path(jobs[0]['reference_run'])
            if result['signature']['partition'] != digest(read_csv(reference/'partition_manifest.csv')):
                raise ValueError('Legacy reference sample assignment differs: ' + method)
            if load_json(Path(path)/'protocol/full_schedule.json') != load_json(reference/'protocol/full_schedule.json'):
                raise ValueError('Legacy reference client schedule differs: ' + method)
    for job in jobs:
        if args.stage == 'plan':
            print(shlex.join(command_for(job, args.output_root)))
        elif args.stage == 'preflight':
            preflight_job(job)
            print('File/protocol preflight ready:', job['method'], '(not a CUDA validation)')
        else:
            execute_job(job, args.output_root, args.resume)


if __name__ == '__main__':
    try:
        main()
    except (OSError, ValueError, KeyError) as error:
        raise SystemExit(str(error))
