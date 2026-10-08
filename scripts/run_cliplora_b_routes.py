"""Plan, audit and run B route experiments. Default stage prints a plan only."""
import argparse
import math
import os
from pathlib import Path
import shlex
import subprocess
import sys
from types import SimpleNamespace

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from scripts.run_ab_validation import file_lock, input_fingerprint, write_json
from scripts.run_cliplora_a_refresh import build_command
from scripts.run_cliplora_sfra import prepare_protocol
from tools.sfra.b_routes import (ARMS, SCHEMA, check_sources, file_hash, replay_files, replay_preflight,
                                source_hashes, summarize, summarize_replay, validate_profile)
from tools.sfra.maintext import digest, load_json


def parser():
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    p.add_argument('--mode', choices=['full', 'replay'], default='full')
    p.add_argument('--stage', choices=['plan', 'preflight', 'run', 'status', 'summary', 'pack'], default='plan')
    p.add_argument('--arms', nargs='+', choices=ARMS, default=None)
    p.add_argument('--seeds', nargs='+', type=int, default=[42], help='Full runs only; replay uses source seed')
    p.add_argument('--reference-run', type=Path, default=Path('references/full10_clientlt'))
    p.add_argument('--source-run', type=Path, help='Replay: complete source snapshots, not an analysis archive')
    p.add_argument('--data-root', type=Path, default=Path('DATA'))
    p.add_argument('--output-root', type=Path, default=Path('output/cifar100_LT/b_routes'))
    p.add_argument('--baseline-run', type=Path, action='append', default=[], help='Optional completed A-only run for summary')
    p.add_argument('--num-workers', type=int, default=8)
    p.add_argument('--rounds', nargs='+', type=int, default=[30, 60, 90])
    p.add_argument('--rhos', nargs='+', type=float, default=[.5, 1., 2.], help='Replay radii only; full runs always use rho=1')
    p.add_argument('--no-rotations', action='store_true', help='Replay only: omit the three rotation controls')
    p.add_argument('--include-legacy', action='store_true', help='Replay original shared-C optimizer and verify old commit')
    p.add_argument('--skip-source-probe', action='store_true')
    p.add_argument('--stop-after-round', type=int)
    p.add_argument('--resume', action='store_true')
    return p


def validate_args(args):
    args.arms = args.arms or (['P', 'C', 'F'] if args.mode == 'full' else ['N', 'P', 'M', 'S', 'C', 'F'])
    for name in ('arms', 'seeds', 'rounds', 'rhos'):
        values = getattr(args, name)
        if len(values) != len(set(values)):
            raise ValueError('Duplicate '+name)
    if args.num_workers < 0 or any(s < 0 for s in args.seeds):
        raise ValueError('Workers/seeds must be nonnegative')
    if args.stop_after_round is not None and not 0 <= args.stop_after_round <= 100:
        raise ValueError('stop-after-round must be in 0..100 (0 means all rounds)')
    if any(not math.isfinite(r) or r <= 0 for r in args.rhos):
        raise ValueError('Rhos must be positive and finite')
    if args.mode == 'replay':
        if args.source_run is None:
            raise ValueError('Replay requires --source-run')
        if args.stop_after_round is not None:
            raise ValueError('Replay resumes at branch boundaries; stop-after-round is for full runs')
        if args.seeds != [42]:
            raise ValueError('Replay derives its seed from --source-run; omit --seeds')
        if any(a.endswith('-rot') for a in args.arms):
            raise ValueError('Replay rotations are generated separately at rho=1; use --no-rotations to omit them')
        separate_paths(args.output_root, args.source_run)
    else:
        if (args.source_run is not None or args.rhos != [.5, 1., 2.] or args.rounds != [30, 60, 90]
                or args.no_rotations or args.include_legacy or args.skip_source_probe):
            raise ValueError('Source/radii/event/legacy/probe options require --mode replay')
        separate_paths(args.output_root, args.reference_run)


def separate_paths(output, source):
    a, b = Path(output).resolve(), Path(source).resolve()
    if a == b or a in b.parents or b in a.parents:
        raise ValueError('Output and source/reference directories must be separate, non-nested paths')


def set_flag(cmd, flag, value):
    result = list(cmd)
    if flag in result:
        result[result.index(flag)+1] = str(value)
    else:
        i = result.index('DATALOADER.NUM_WORKERS')
        result[i:i] = [flag, str(value)]
    return result


def full_command(args, run, seed, arm):
    base = SimpleNamespace(output_root=args.output_root, data_root=args.data_root.resolve(), seed=seed,
        schedule_file=run/'protocol/full_schedule.json', partition='client-longtail', matched_beta=.5,
        rank=4, refresh_interval=10, refresh_epochs=1, refresh_lr=.001, resume=None, num_workers=args.num_workers)
    cmd, _ = build_command(base, 'off')
    cmd[2] = str(REPO/'scripts/train_cliplora_b_routes.py')
    cmd = set_flag(set_flag(cmd, '--output-dir', run), '--split_seed', 42)
    flags = ['--route-spec', str(run/'route_spec.json'), '--lac_method', 's',
        '--lac_partition_manifest', str(run/'protocol/partition_source.csv'), '--lac_la_tau', '1', '--lac_a_lr_mult', '1',
        '--sfra_variant', 'full-cp', '--sfra_retention_weight', '10.0', '--sfra_classification_weight', '1.0',
        '--sfra_witness_batch_size', '8', '--sfra_resume', '', '--sfra_fast_execution_v2',
        '--sfra_feedback_batch_size', '128', '--sfra_feedback_cache_gib', '4.0']
    if arm != 'N':
        flags += ['--b_transfer_enable', '--b_transfer_mode', 'shared', '--b_transfer_lr', '0.3',
            '--b_transfer_probe_step', '0.1', '--b_transfer_reg', '0.001',
            '--b_transfer_non_tail_sampling', 'class-cyclic', '--b_transfer_tail_weight', '0.35']
    i = cmd.index('DATALOADER.NUM_WORKERS')
    cmd[i:i] = flags
    return cmd


def replay_command(args, run):
    source = args.source_run.resolve()
    cmd = list(load_json(source/'command.json'))
    indices = [i for i, value in enumerate(cmd) if Path(value).name in
               ('federated_main.py', 'train_cliplora_b_routes.py')]
    if len(indices) != 1:
        raise ValueError('Unsupported source worker; use a standard A/AB or route run')
    cmd[0], cmd[indices[0]] = sys.executable, str(REPO/'scripts/train_cliplora_b_routes.py')
    # No training checkpoint or another diagnostic runtime may override replay.
    for flag in ('--route-spec', '--sfra_resume', '--sfra_stop_after_round', '--b_problem2_replay_manifest',
                 '--method_a_diagnostic_manifest'):
        if flag in cmd:
            i = cmd.index(flag)
            del cmd[i:i+2]
    for flag, value in {'--route-spec': run/'route_spec.json', '--sfra_resume': '',
        '--root': args.data_root.resolve(), '--output-dir': run,
        '--client_schedule_file': run/'protocol/full_schedule.json',
        '--lac_partition_manifest': run/'protocol/partition_source.csv'}.items():
        cmd = set_flag(cmd, flag, value)
    cmd[cmd.index('DATALOADER.NUM_WORKERS')+1] = str(args.num_workers)
    return cmd


def jobs(args, hash_inputs=True):
    hashes = source_hashes(REPO)
    common = dict(schema_version=SCHEMA, mode=args.mode, code_sha256=hashes)
    if args.mode == 'full':
        fingerprint = input_fingerprint(SimpleNamespace(reference_run=args.reference_run, partition='client-longtail'))
        for seed in args.seeds:
            for arm in args.arms:
                run = args.output_root.resolve()/'runs'/arm/f'seed{seed}'
                spec = dict(common, input_fingerprint=fingerprint, settings=dict(
                    arm=arm, seed=seed, protocol_seed=42, partition='client-longtail', rho=1., null_index=0,
                    reference_run=str(args.reference_run.resolve()), data_root=str(args.data_root.resolve()),
                    num_workers=args.num_workers, feedback_batch_size=128, feedback_cache_gib=4.))
                yield run, spec, full_command(args, run, seed, arm)
    else:
        report = replay_preflight(args.source_run, args.rounds, args.include_legacy, hash_inputs)
        if not report['ready']:
            raise ValueError('Replay inputs unavailable: '+str(report))
        cfg = load_json(args.source_run/'sfra_config.json')
        for rnd in args.rounds:
            run = args.output_root.resolve()/'replay'/f'seed{cfg["seed"]}'/f'r{rnd:03d}'
            inputs = {n: report['source_sha256'][n] for n in replay_files([rnd], args.include_legacy)} if hash_inputs else {}
            spec = dict(common, source_run=str(args.source_run.resolve()), source_sha256=inputs,
                round=rnd, seed=cfg['seed'], arms=args.arms, rhos=args.rhos, rotations=not args.no_rotations,
                include_legacy=args.include_legacy, source_probe=not args.skip_source_probe,
                data_root=str(args.data_root.resolve()), num_workers=args.num_workers)
            yield run, spec, replay_command(args, run)


def dataset_preflight(args):
    folder = args.data_root/'cifar-100/cifar-100-python'
    missing = [str(folder/n) for n in ('train', 'test', 'meta') if not (folder/n).is_file() or (folder/n).stat().st_size == 0]
    if missing:
        raise ValueError('Missing CIFAR-100 files: '+str(missing))


def sources_unchanged(spec):
    try:
        if source_hashes(REPO) != spec['code_sha256']:
            return False
        if spec['mode'] == 'full':
            return input_fingerprint(SimpleNamespace(reference_run=Path(spec['settings']['reference_run']),
                partition='client-longtail')) == spec['input_fingerprint']
        return all(file_hash(Path(spec['source_run'])/n) == h for n, h in spec['source_sha256'].items())
    except (OSError, ValueError):
        return False


def merge_plan(root, planned):
    root = Path(root)
    # Per-run locks protect execution; a separate short lock protects the
    # shared read/merge/write when different GPUs register different jobs.
    with file_lock(root/'locks/route_plan.lock'):
        path = root/'route_plan.json'
        previous = load_json(path)['jobs'] if path.is_file() else []
        registered = {j['run']: j for j in previous}
        for run, spec, _ in planned:
            row = dict(run=str(run), spec=spec)
            if str(run) in registered and registered[str(run)] != row:
                raise ValueError('Registered route job changed: '+str(run))
            registered[str(run)] = row
        write_json(path, dict(schema_version=SCHEMA, primary_endpoint='committed rounds81..100 fixed Tail20 mean',
            practical_difference_pp=.10, jobs=[registered[k] for k in sorted(registered)]))


def write_preflight(args, result):
    # Retain every invocation's preflight; the root file is just the latest.
    request = dict(mode=args.mode, arms=args.arms, seeds=args.seeds, rounds=args.rounds, rhos=args.rhos,
        source_run=str(args.source_run.resolve()) if args.source_run else None,
        reference_run=str(args.reference_run.resolve()), data_root=str(args.data_root.resolve()),
        rotations=not args.no_rotations, include_legacy=args.include_legacy,
        source_probe=not args.skip_source_probe, num_workers=args.num_workers)
    record = dict(result, request=request)
    root = args.output_root.resolve()
    with file_lock(root/'locks/route_preflight.lock'):
        write_json(root/'preflights'/(digest(record)+'.json'), record)
        write_json(root/'preflight.json', record)


def execute(args, run, spec, command):
    check_sources(spec, REPO)
    with file_lock(args.output_root.resolve()/'locks'/(digest(str(run))+'.lock'), timeout=.1):
        saved, receipt_path = run/'route_spec.json', run/'route_receipt.json'
        if run.exists() and any(run.iterdir()):
            if not saved.is_file() or load_json(saved) != spec:
                raise ValueError('Nonempty output has a different route contract: '+str(run))
            if not args.resume:
                raise ValueError('Existing run: add --resume or use a new output root')
            if receipt_path.is_file():
                receipt = load_json(receipt_path)
                if receipt.get('spec_digest') != digest(spec) or not receipt.get('code_unchanged'):
                    raise ValueError('Invalid source receipt; cannot resume')
            old_command = list(load_json(run/'command.json'))
            old_command[0] = sys.executable
            if '--sfra_stop_after_round' in old_command:
                i = old_command.index('--sfra_stop_after_round')
                del old_command[i:i+2]
            if old_command != command:
                raise ValueError('Saved worker command changed')
            if spec['mode'] == 'full':
                if (run/'completion.json').is_file():
                    from tools.sfra.b_routes import read_run
                    read_run(run, spec)
                    print('SKIP completed', run, flush=True)
                    return
                checkpoint = run/'checkpoints/sfra_last.pt'
                if not checkpoint.is_file():
                    raise ValueError('No round-boundary checkpoint; preserve partial output and use a new root')
                validate_profile(load_json(run/'sfra_config.json'), spec)
                command = set_flag(command, '--sfra_resume', checkpoint)
            # Replay always revalidates branch artifacts inside the worker.
        else:
            protocol_source = args.reference_run if spec['mode'] == 'full' else args.source_run
            prepare_protocol(SimpleNamespace(reference_run=protocol_source, partition='client-longtail'), run)
            write_json(saved, spec)
            write_json(run/'command.json', command)
        if args.stop_after_round is not None:
            command = set_flag(command, '--sfra_stop_after_round', args.stop_after_round)
        print(shlex.join(command), flush=True)
        try:
            subprocess.run(command, cwd=REPO, check=True)
        finally:
            unchanged = sources_unchanged(spec)
            write_json(receipt_path, dict(spec_digest=digest(spec), code_unchanged=unchanged))
        if not unchanged:
            raise ValueError('Code or replay inputs changed during execution')
        if spec['mode'] == 'replay':
            done = load_json(run/'replay_completion.json')
            if not done.get('complete') or done.get('spec_digest') != digest(spec):
                raise ValueError('Replay worker did not finish all branches')
        elif not (args.stop_after_round or 0):
            if not (run/'completion.json').is_file():
                raise ValueError('Training returned without completing 100 rounds; resume explicitly')
        else:
            done = load_json(run/'progress.json')
            if done.get('completed_round', -1) < args.stop_after_round:
                raise ValueError('Training stopped before the requested boundary')


def main(argv=None):
    p = parser()
    args = p.parse_args(argv)
    os.chdir(REPO)
    try:
        if args.stage in ('summary', 'pack'):
            status, audits = summarize(args.output_root, args.baseline_run)
            replay_status = summarize_replay(args.output_root)
            if args.stage == 'pack':
                from tools.sfra.summary import pack
                pack(args.output_root)
            print('Reports:', args.output_root/'analysis')
            if any(x['status'] == 'invalid' for x in status+replay_status) or any(not x['valid'] for x in audits):
                raise ValueError('Invalid results were excluded; inspect the audit tables')
            return
        if args.stage == 'status':
            for path in sorted(args.output_root.rglob('route_spec.json')):
                run = path.parent
                marker = 'completion.json' if load_json(path)['mode'] == 'full' else 'replay_completion.json'
                print(run, 'complete' if (run/marker).is_file() else 'incomplete')
            return
        validate_args(args)
        # A plan can identify missing replay snapshots without hashing gigabytes.
        if args.stage == 'plan' and args.mode == 'replay':
            report = replay_preflight(args.source_run, args.rounds, args.include_legacy)
            print('Replay input availability:', report)
            print('Events:', args.rounds, 'Arms:', args.arms, 'Rhos:', args.rhos,
                  'Folds: 2; steps: 2 and 8; rotation controls:', not args.no_rotations)
            return
        try:
            planned = list(jobs(args))
            if args.stage != 'plan':
                dataset_preflight(args)
                for run, spec, _ in planned:
                    check_sources(spec, REPO)
                    saved = run/'route_spec.json'
                    if saved.is_file() and load_json(saved) != spec:
                        raise ValueError('Saved contract differs: '+str(run))
        except (OSError, ValueError, KeyError) as error:
            if args.stage == 'preflight':
                write_preflight(args, dict(ready=False, mode=args.mode, error=str(error)))
            raise
        if args.stage == 'plan':
            for run, _, command in planned:
                print('Run:', run)
                print(shlex.join(command))
            return
        write_preflight(args, dict(ready=True, mode=args.mode,
            jobs=[dict(run=str(r), spec_digest=digest(s)) for r, s, _ in planned], gpu_execution_tested=False))
        if args.stage == 'preflight':
            print(f'File/configuration preflight passed for {len(planned)} jobs. GPU execution has not been tested.')
            return
        import torch
        if not torch.cuda.is_available():
            raise ValueError('Full CLIP experiments require CUDA; use CPU unit tests to verify the implementation')
        merge_plan(args.output_root, planned)
        for run, spec, command in planned:
            execute(args, run, spec, command)
    except (OSError, ValueError, KeyError, subprocess.CalledProcessError) as error:
        p.error(str(error))


if __name__ == '__main__':
    main()
