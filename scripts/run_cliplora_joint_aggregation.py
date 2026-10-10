"""Seed42 joint aggregation experiments and opt-in frozen-A aggregation controls."""
import argparse
from concurrent.futures import ThreadPoolExecutor
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
from scripts.run_method_a_simple_controls import prepare_control_protocol
from tools.client_aggregation.protocol import (ARMS, CONTRACT, SCHEMA, aggregation_spec,
    check_aggregation, code_hashes, count_matrix, digest, load_json, write_weight_tables)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('plan', 'preflight', 'smoke', 'run', 'server', 'status', 'summary', 'pack'), default='plan')
    parser.add_argument('--arms', nargs='+', choices=ARMS)
    parser.add_argument('--aggregation-rule', choices=('joint', 'tailrw16', 'fedavg'), default='joint',
                        help='tailrw16/fedavg keep A frozen and train ordinary B only')
    parser.add_argument('--seed', type=int, choices=(42,), default=42)
    parser.add_argument('--aggregation-lambda', type=float, default=.1)
    parser.add_argument('--reference-run', type=Path, default=Path('references/full10_clientlt'))
    parser.add_argument('--data-root', type=Path, default=Path('DATA'))
    parser.add_argument('--output-root', type=Path,
                        help='Defaults to a separate frozen_tailrw16_v1 directory for the tailrw16 rule')
    parser.add_argument('--compare-root', type=Path,
                        help='For TailRW16 status/summary/pack: completed joint-aggregation root; read-only')
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--client-concurrency', type=int, choices=(1, 4, 6), default=4,
                        help='Maximum independent clients on each experiment GPU; 1 retains serial training')
    parser.add_argument('--cuda-policy', choices=('legacy', 'deterministic'), default='legacy',
                        help='Original CUDA execution by default; deterministic is an explicit opt-in')
    parser.add_argument('--gpus', type=int, nargs=2, default=[2, 3], metavar=('FROZEN_GPU', 'AB_GPU'),
                        help='Physical GPU IDs for --stage server, mapped as frozen then ab')
    parser.add_argument('--stop-after-round', type=int, default=0, help='Pause a formal run at a committed round; rerun with 0 to continue')
    args = parser.parse_args(argv)
    if args.arms is None:
        args.arms = list(ARMS) if args.aggregation_rule == 'joint' else ['frozen']
    if args.output_root is None:
        args.output_root = Path('output/cifar100_LT/' + dict(joint='client_aggregation_v2_parallel4',
            tailrw16='frozen_tailrw16_v1', fedavg='frozen_fedavg_v1')[args.aggregation_rule])
    if args.aggregation_rule != 'joint' and (args.arms != ['frozen'] or args.aggregation_lambda != .1):
        parser.error('Aggregation controls support only --arms frozen; --aggregation-lambda does not apply')
    if len(set(args.arms)) != len(args.arms) or args.num_workers < 0 or not 0 <= args.stop_after_round <= 100:
        parser.error('Duplicate arms, negative workers or invalid stopping round')
    if len(set(args.gpus)) != 2 or min(args.gpus) < 0:
        parser.error('Use two distinct nonnegative GPU IDs')
    return args


def register(args):
    root = args.output_root.resolve()
    settings = dict(seed=42, lambda_value=args.aggregation_lambda,
        reference_run=str(args.reference_run.resolve()), data_root=str(args.data_root.resolve()),
        num_workers=args.num_workers, client_concurrency=args.client_concurrency, parallel_validation='off')
    if args.cuda_policy != 'legacy':
        settings['cuda_policy'] = args.cuda_policy
    contract = CONTRACT
    if args.aggregation_rule != 'joint':
        from tools.client_aggregation.frozen_tailrw import control_contract
        contract = control_contract(args.aggregation_rule)
        settings.update(aggregation_rule=args.aggregation_rule, lambda_value=None)
    fingerprint, counts, matrix = count_matrix(args.reference_run)
    hashes = code_hashes(REPO)
    with file_lock(root / 'locks/plan.lock'):
        path = root / 'experiment_plan.json'
        if path.is_file():
            plan = load_json(path)
            if (plan['contract'] != contract or plan['settings'] != settings or plan['code_sha256'] != hashes
                    or plan['aggregation']['input_fingerprint'] != fingerprint
                    or plan['aggregation']['matrix_sha256'] != digest(matrix.tolist())):
                raise ValueError('Registered inputs/settings/source changed. Use a new output root; existing runs are preserved.')
            check_aggregation(plan['aggregation'])
        else:
            plan = dict(schema=SCHEMA, contract=contract, settings=settings, code_sha256=hashes,
                aggregation=aggregation_spec(args.reference_run, args.aggregation_lambda, args.aggregation_rule))
            write_json(path, plan)
            write_weight_tables(root, plan['aggregation'])
    return plan


def make_job(plan, arm, mode='formal'):
    from tools.client_aggregation.protocol import validate_arm
    job = dict(schema=SCHEMA, arm=arm, mode=mode, settings=plan['settings'], contract=plan['contract'],
        aggregation=plan['aggregation'], code_sha256=plan['code_sha256'],
        run=('smoke' if mode == 'smoke' else 'runs') + '/seed42/' + arm)
    validate_arm(job)
    return job


def preflight(plan, check_cuda=False):
    settings = plan['settings']
    fingerprint, counts, matrix = count_matrix(settings['reference_run'])
    if fingerprint != plan['aggregation']['input_fingerprint'] or counts != plan['aggregation']['counts']:
        raise ValueError('Reference changed after registration')
    check_aggregation(plan['aggregation'])
    dataset = Path(settings['data_root']) / 'cifar-100/cifar-100-python'
    for filename in ('train', 'test', 'meta'):
        if not (dataset / filename).is_file() or not (dataset / filename).stat().st_size:
            raise ValueError('Missing CIFAR-100 file: ' + str(dataset / filename))
    if code_hashes(REPO) != plan['code_sha256']:
        raise ValueError('Training source changed')
    if check_cuda:
        import torch
        if not torch.cuda.is_available():
            raise ValueError('CUDA is unavailable in this Python environment. Run smoke/run on your GPU server; preflight is not a GPU test.')


def command_for(job, root, resume=False, stop_after=0):
    run = Path(root).resolve() / job['run']
    setting = job['settings']
    base = SimpleNamespace(output_root=Path(root), data_root=Path(setting['data_root']), seed=42,
        partition='client-longtail', rank=4, schedule_file=run / 'protocol/full_schedule.json',
        matched_beta=.5, refresh_interval=10, refresh_epochs=1, refresh_lr=.001,
        resume=None, num_workers=setting['num_workers'])
    command, _ = build_command(base, 'off')
    command[command.index('federated_main.py')] = str(REPO / 'scripts/train_cliplora_joint_aggregation.py')
    command[command.index('--output-dir') + 1] = str(run)
    extra = ['--joint-spec', str(run / 'joint_job.json'), '--lac_method', 's',
        '--lac_partition_manifest', str(run / 'protocol/partition_source.csv'),
        '--lac_la_tau', '1', '--lac_a_lr_mult', '1',
        '--sfra_variant', 'full-cp' if job['arm'] == 'ab' else 's',
        '--sfra_retention_weight', '0' if job['arm'] == 'plain_a' else '10', '--sfra_classification_weight', '1',
        '--sfra_witness_batch_size', '8', '--sfra_fast_execution_v2',
        '--sfra_feedback_batch_size', '128', '--sfra_feedback_cache_gib', '4',
        '--sfra_resume', str(run / 'checkpoints/sfra_last.pt') if resume else '',
        '--sfra_stop_after_round', str(1 if job['mode'] == 'smoke' else stop_after)]
    if job['arm'] == 'ab':
        extra += ['--b_transfer_enable', '--b_transfer_mode', 'shared', '--b_transfer_lr', '.3',
            '--b_transfer_probe_step', '.1', '--b_transfer_reg', '.001',
            '--b_transfer_non_tail_sampling', 'class-cyclic', '--b_transfer_tail_weight', '.35']
    command[command.index('DATALOADER.NUM_WORKERS'):command.index('DATALOADER.NUM_WORKERS')] = extra
    return command


def execute(job, root, stop_after=0):
    from tools.client_aggregation.analysis import audit_run
    root = Path(root).resolve()
    run = root / job['run']
    with file_lock(root / 'locks' / (job['mode'] + '_' + job['arm'] + '.lock'), timeout=.1):
        if (run / 'completion.json').is_file() or (run / 'smoke_complete.json').is_file():
            audit_run(run, job, 1 if job['mode'] == 'smoke' else 100)
            print('SKIP verified completed:', run)
            return
        if (run / 'joint_job.json').is_file():
            if load_json(run / 'joint_job.json') != job:
                raise ValueError('Existing run belongs to a different job')
        else:
            if run.exists() and any(run.iterdir()):
                raise ValueError('Unregistered output directory is occupied: ' + str(run))
            prepare_control_protocol(dict(reference_run=job['settings']['reference_run']), run)
            write_json(run / 'joint_job.json', job)
        resume = (run / 'checkpoints/sfra_last.pt').is_file()
        if not resume and (any((run / 'events').glob('*/state.pt')) or any((run / 'predictions').glob('*.npz'))):
            raise ValueError('Training artifacts exist without a committed checkpoint. Preserve them and use a new output root.')
        command = command_for(job, root, resume, stop_after)
        write_json(run / ('resume_command.json' if resume else 'command.json'), command)
        log = root / 'launcher_logs' / (job['arm'] + '_' + job['mode'] + '.log')
        log.parent.mkdir(parents=True, exist_ok=True)
        print('START', job['arm'], job['mode'], 'resume=' + str(resume), 'log=' + str(log), flush=True)
        print(shlex.join(command), flush=True)
        with log.open('a', encoding='utf-8') as stream:
            stream.write('\nCOMMAND ' + shlex.join(command) + '\n'); stream.flush()
            result = subprocess.run(command, cwd=REPO, stdout=stream, stderr=subprocess.STDOUT)
        unchanged = code_hashes(REPO) == job['code_sha256']
        write_json(run / 'joint_receipt.json', dict(schema=SCHEMA, job_digest=digest(job),
            code_unchanged=unchanged, exit_code=result.returncode))
        if result.returncode or not unchanged:
            print('\n'.join(log.read_text(encoding='utf-8', errors='replace').splitlines()[-70:]), flush=True)
            raise ValueError('Worker failed; same command resumes its last committed round. Log: ' + str(log))
        completed = int(load_json(run / 'progress.json')['completed_round'])
        audit_run(run, job, completed)
        if job['mode'] == 'smoke':
            if completed != 1:
                raise ValueError('Smoke did not commit exactly one round')
            write_json(run / 'smoke_complete.json', dict(schema=SCHEMA, completed_round=1,
                note='Separate one-round smoke; no serial/parallel comparison; excluded from formal analysis'))
        print('DONE', job['arm'], 'completed_round=' + str(completed), flush=True)


def server_commands(args, arm):
    common = [sys.executable, '-u', str(REPO/'scripts/run_cliplora_joint_aggregation.py'),
        '--arms', arm, '--seed', '42', '--aggregation-lambda', str(args.aggregation_lambda),
        '--aggregation-rule', args.aggregation_rule,
        '--reference-run', str(args.reference_run.resolve()), '--data-root', str(args.data_root.resolve()),
        '--output-root', str(args.output_root.resolve()), '--num-workers', str(args.num_workers),
        '--client-concurrency', str(args.client_concurrency), '--cuda-policy', args.cuda_policy]
    return [common + ['--stage', 'smoke'],
            common + ['--stage', 'run', '--stop-after-round', str(args.stop_after_round)]]


def server_environment(gpu):
    return dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), CUDA_DEVICE_ORDER='PCI_BUS_ID')


def run_server(args):
    """Two isolated processes. Each GPU must pass its smoke before its formal run."""
    root = args.output_root.resolve()
    jobs = [(arm, args.gpus[ARMS.index(arm)]) for arm in args.arms]
    with file_lock(root/'locks/server_dispatch.lock', timeout=.1):
        write_json(root/'server_launch.json', dict(gpu_assignment=dict(jobs),
            client_concurrency=args.client_concurrency,
            note='One visible GPU per child; no DataParallel across GPU 2 and 3'))

        def launch(arm, gpu):
            print(f'SERVER {arm} -> physical GPU {gpu}, max local clients={args.client_concurrency}', flush=True)
            log = root/'launcher_logs'/f'{arm}_server.log'
            log.parent.mkdir(parents=True, exist_ok=True)
            with log.open('a', encoding='utf-8') as stream:
                for command in server_commands(args, arm):
                    stream.write('\n' + shlex.join(command) + '\n'); stream.flush()
                    result = subprocess.run(command, cwd=REPO, env=server_environment(gpu),
                                            stdout=stream, stderr=subprocess.STDOUT)
                    if result.returncode:
                        print(f'FAILED {arm} on GPU {gpu}. See {log}', flush=True)
                        return result.returncode
            print(f'FINISHED {arm} on GPU {gpu}', flush=True)
            return 0

        with ThreadPoolExecutor(max_workers=len(jobs)) as executor:
            futures = [executor.submit(launch, arm, gpu) for arm, gpu in jobs]
            codes = [future.result() for future in futures]
        if any(codes):
            raise ValueError('One or more server jobs failed; completed jobs are retained. See launcher_logs/*_server.log.')


def main(argv=None):
    args = parse_args(argv)
    os.chdir(REPO)
    root = args.output_root.resolve()
    if args.stage in ('status', 'summary', 'pack'):
        from tools.client_aggregation.analysis import summarize, pack_results
        statuses = summarize(root, compare_root=args.compare_root)
        if args.stage == 'pack':
            print('Archive:', pack_results(root))
        pair = load_json(root / 'analysis/pair_audit.json')
        if any(r['status'] == 'invalid' for r in statuses) or pair.get('status') == 'mismatch':
            raise ValueError('Invalid/mismatched results were excluded. See analysis/status.csv and pair_audit.json.')
        return
    plan = register(args)
    print('Registered seed42, rule=' + args.aggregation_rule, root, flush=True)
    if args.stage == 'plan':
        return
    preflight(plan, check_cuda=args.stage in ('smoke', 'run'))
    print('File/configuration/aggregation-weight preflight passed. GPU execution has NOT been tested.', flush=True)
    if args.stage == 'preflight':
        return
    if args.stage == 'server':
        # Children initialize CUDA only after their per-GPU environment is set.
        run_server(args)
        return
    for arm in args.arms:
        if args.stage == 'run' and args.client_concurrency > 1:
            execute(make_job(plan, arm, 'smoke'), root)
        execute(make_job(plan, arm, 'smoke' if args.stage == 'smoke' else 'formal'), root, args.stop_after_round)


if __name__ == '__main__':
    try:
        main()
    except (ValueError, OSError, KeyError) as error:
        raise SystemExit(str(error))
