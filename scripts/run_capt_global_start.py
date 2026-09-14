"""Run CAPT cluster with a common server-model start on both partitions."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.create_client_schedule import (
    create_schedule, load_schedule, validate_schedule, write_schedule_atomic,
)

PARTITIONS = ('client-longtail', 'noniid-labeldir-fine', 'matched-dirichlet', 'noniid-labeldir')


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', default='DATA/')
    parser.add_argument('--output-root', type=Path, default=Path('output/cifar100_LT/capt_cluster_global_start'))
    parser.add_argument('--python-bin', default=sys.executable)
    parser.add_argument('--gpu', default='0', help='GPU index within inherited CUDA_VISIBLE_DEVICES, or a device ID')
    parser.add_argument('--seeds', nargs='+', type=int, default=[1, 42, 2026])
    parser.add_argument('--partitions', nargs='+', choices=PARTITIONS,
                        default=['client-longtail', 'noniid-labeldir-fine'])
    parser.add_argument('--num-users', type=int, default=30)
    parser.add_argument('--frac', type=float, default=1.0)
    parser.add_argument('--rounds', type=int, default=100)
    parser.add_argument('--local-epochs', type=int, default=3)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--alpha', type=float, default=0.5)
    parser.add_argument('--specialization-lambda', type=float, default=0.75)
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--fixed-global-agg-freq', type=int, default=0,
                        help='0 retains the original MAB; 1 aggregates every round without test-controlled MAB')
    parser.add_argument('--reset-optimizer', action='store_true',
                        help='Also enable existing capt_matched_v2 optimizer reset; off for the parameter-only comparison')
    parser.add_argument('--dry-run', action='store_true', help='Print commands without writing files or training')
    args = parser.parse_args(argv)
    if args.num_users < 4 or not 0 < args.frac <= 1:
        parser.error('num-users must be >= 4 and frac must be in (0, 1]')
    if max(int(args.frac * args.num_users), 1) < 4:
        parser.error('CAPT uses four clusters: at least four clients must participate per round')
    if args.rounds < 1 or args.local_epochs < 1 or args.lr <= 0 or args.alpha <= 0:
        parser.error('rounds, local-epochs, lr and alpha must be positive')
    if args.fixed_global_agg_freq < 0 or args.num_workers < 0:
        parser.error('fixed-global-agg-freq and num-workers must be nonnegative')
    if not 0 <= args.specialization_lambda <= 1:
        parser.error('specialization-lambda must be in [0, 1]')
    if len(set(args.seeds)) != len(args.seeds) or len(set(args.partitions)) != len(args.partitions):
        parser.error('duplicate seeds or partitions would overwrite a run')
    args.output_root = (ROOT / args.output_root).resolve()
    return args


def schedule_path(args, seed):
    return args.output_root / 'schedules' / (
        f'users{args.num_users}_frac{args.frac}_rounds{args.rounds}_seed{seed}.json'
    )


def run_directory(args, seed, partition):
    return args.output_root / f'seed{seed}' / partition


def build_command(args, seed, partition):
    if partition not in PARTITIONS:
        raise ValueError(f'Unsupported partition: {partition}')
    options = [
        ('--root', args.data_root), ('--model', 'cluster'), ('--trainer', 'CAPT'),
        ('--dataset', 'cifar100_LT'), ('--seed', seed), ('--split_seed', seed),
        ('--num_users', args.num_users), ('--frac', args.frac), ('--round', args.rounds),
        ('--local_epochs', args.local_epochs), ('--lr', args.lr), ('--gamma', 1),
        ('--n_ctx', 4), ('--n_general', 1), ('--ctx_init', 'False'), ('--csc', 'True'),
        ('--dataset-config-file', 'configs/datasets/cifar100_LT.yaml'),
        ('--config-file', 'configs/trainers/CAPT/vit_b16.yaml'),
        ('--output-dir', run_directory(args, seed, partition)),
        ('--imb_factor', 0.01), ('--imb_type', 'exp'),
        ('--train_batch_size', 32), ('--test_batch_size', 64),
        ('--global_eval_interval', 1), ('--num_classes', 100),
        ('--head_client_ratio', 0.9), ('--tail_client_ratio', 0.1),
        ('--head_class_ratio', 0.8), ('--tail_class_ratio', 0.2),
        ('--specialization_lambda', args.specialization_lambda),
        ('--intra_group_alpha', args.alpha), ('--head_leakage_scale', 3.0),
        ('--partition', partition), ('--beta', args.alpha),
        ('--n_simclusters', 4), ('--n_disclusters', 4),
        ('--client_schedule_file', schedule_path(args, seed)), ('--client_schedule_seed', seed),
        ('--capt_reset_global_before_client', 'True'),
        ('--capt_fixed_global_agg_freq', args.fixed_global_agg_freq),
        ('--capt_matched_v2', str(args.reset_optimizer)),
    ]
    cmd = [args.python_bin, '-u', 'federated_main.py']
    for name, value in options:
        cmd.extend([name, str(value)])
    cmd.extend(['DATALOADER.NUM_WORKERS', str(args.num_workers)])
    return cmd


def ensure_schedule(args, seed):
    path = schedule_path(args, seed)
    expected = create_schedule(args.rounds, args.num_users, args.frac, seed)
    if path.exists():
        actual = load_schedule(path)
        validate_schedule(actual, args.rounds, args.num_users, args.frac)
        if actual != expected:
            raise ValueError(f'Existing schedule differs from the requested seed: {path}')
    else:
        write_schedule_atomic(path, dict(
            num_rounds=args.rounds, num_users=args.num_users, frac=args.frac,
            seed=seed, clients_per_round=max(int(args.frac * args.num_users), 1),
            schedule=expected,
        ))
    return hashlib.sha256(json.dumps(expected, separators=(',', ':')).encode()).hexdigest()


def gpu_environment(gpu):
    env = os.environ.copy()
    env['PYTHONIOENCODING'] = 'utf-8'
    visible = [x.strip() for x in env.get('CUDA_VISIBLE_DEVICES', '').split(',') if x.strip()]
    env['CUDA_VISIBLE_DEVICES'] = visible[int(gpu)] if visible and gpu.isdigit() and int(gpu) < len(visible) else gpu
    return env


def main(argv=None):
    args = parse_args(argv)
    jobs = [(seed, part) for seed in args.seeds for part in args.partitions]
    # Fail before starting any expensive run if any target contains old results.
    for seed, part in jobs:
        path = run_directory(args, seed, part)
        if not args.dry_run and path.exists() and any(path.iterdir()):
            raise FileExistsError(f'Output already contains files; choose a new --output-root: {path}')
    env = gpu_environment(args.gpu)
    print('CAPT cluster: restore global parameters before every client.', flush=True)
    print(f'MAB enabled: {args.fixed_global_agg_freq == 0}; optimizer reset: {args.reset_optimizer}', flush=True)
    for seed, part in jobs:
        cmd = build_command(args, seed, part)
        print(f'\nseed={seed}, partition={part}, CUDA_VISIBLE_DEVICES={env["CUDA_VISIBLE_DEVICES"]}', flush=True)
        print(subprocess.list2cmdline(cmd) if os.name == 'nt' else shlex.join(cmd), flush=True)
        if args.dry_run:
            continue
        schedule_hash = ensure_schedule(args, seed)
        out = run_directory(args, seed, part)
        out.mkdir(parents=True, exist_ok=True)
        manifest = dict(command=cmd, cwd=str(ROOT), cuda_visible_devices=env['CUDA_VISIBLE_DEVICES'],
                        client_schedule_sha256=schedule_hash,
                        parameter_start='latest_server_state_before_every_client',
                        optimizer_reset=args.reset_optimizer,
                        official_test_controls_mab=args.fixed_global_agg_freq == 0,
                        fixed_global_aggregation_frequency=args.fixed_global_agg_freq,
                        source_sha256={name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
                                       for name in ['federated_main.py', 'trainers/capt.py', 'loss/prompt_loss.py',
                                                    'clip/model.py', 'utils/datasplit.py',
                                                    'Dassl/dassl/engine/trainer.py', 'utils/fed_utils.py']})
        (out/'command.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
        with (out/'run.log').open('w', encoding='utf-8') as log:
            with subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=subprocess.PIPE,
                                  stderr=subprocess.STDOUT, text=True, encoding='utf-8', errors='replace') as process:
                for line in process.stdout:
                    print(line, end='', flush=True)
                    log.write(line)
                    log.flush()
                code = process.wait()
        if code:
            raise SystemExit(f'CAPT failed (exit {code}); inspect {out / "run.log"}')
        (out/'finished.json').write_text(json.dumps({'exit_code': 0, 'requested_rounds': args.rounds}), encoding='utf-8')


if __name__ == '__main__':
    main()
