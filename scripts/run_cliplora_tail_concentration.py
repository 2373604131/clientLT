"""Prepare or explicitly launch five J-baseline tail-concentration experiments.

Default --stage plan only prints commands. No GPU/model is used in plan/prepare.
Training is foreground, sequential, and requires --stage train explicitly.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
from types import SimpleNamespace

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from scripts.run_cliplora_a_refresh import build_command

VALUES = (0.0, 0.25, 0.5, 0.75, 1.0)
DEFAULT_OUTPUT = 'output/cifar100_LT/j_tail_concentration_if100'


@contextmanager
def preparation_lock(path):
    # open('x') either gives us ownership or fails without touching another lock.
    handle = path.open('x', encoding='utf-8')
    try:
        with handle:
            yield
    finally:
        path.unlink()


def run_path(args, value):
    return args.output_root.resolve() / f'seed{args.seed}_split{args.protocol_seed}' / f'lambda_{value:g}'


def command_for(args, value, run):
    baseline = SimpleNamespace(
        output_root=args.output_root.resolve(), data_root=args.data_root.resolve(),
        seed=args.seed, schedule_file=run / 'protocol/full_schedule.json',
        partition='client-longtail', matched_beta=.5, rank=4,
        refresh_interval=10, refresh_epochs=1, refresh_lr=.001,
        resume=None, num_workers=args.num_workers,
    )
    command, _ = build_command(baseline, 'off')
    for flag, val in {
        '--output-dir': str(run), '--split_seed': str(args.protocol_seed),
        '--client_schedule_seed': str(args.protocol_seed), '--specialization_lambda': str(value),
    }.items():
        command[command.index(flag) + 1] = val
    index = command.index('DATALOADER.NUM_WORKERS')
    command[index:index] = [
        '--lac_method', 'j', '--lac_partition_manifest', str(run / 'protocol/partition_source.csv'),
        '--lac_la_tau', '1.0', '--lac_a_lr_mult', '1.0',
        '--lac_lookahead_rounds', '2', '--lac_min_gain', '0.0001',
        '--lac_tail_tolerance', '0.002', '--lac_patience', '2',
    ]
    return command


def specification(args, value):
    from scripts.cliplora_tail_concentration_protocol import sha256
    sources = ['scripts/cliplora_tail_concentration_protocol.py',
               'scripts/run_cliplora_tail_concentration.py', 'scripts/run_cliplora_a_refresh.py',
               'utils/datasplit.py', 'utils/cliplora_la_control.py',
               'utils/cliplora_functional_feedback.py', 'datasets/cifar100_LT.py']
    return {
        'schema': 'j_tail_concentration_v1', 'lambda': value, 'method': 'j',
        'seed': args.seed, 'split_seed': args.protocol_seed,
        'imbalance_ratio': 100, 'imb_factor_argument': .01, 'intra_group_alpha': .5,
        'clients': 30, 'classes': 100, 'tail_clients': [27, 28, 29],
        'tail_classes': list(range(80, 100)), 'rounds': 100, 'local_epochs': 3,
        'la_tau': 1., 'rank': 4, 'reference_lambda': .75,
        'non_tail_allocation': 'fixed identities, clients and order from lambda=.75 / rho=3',
        'tail_randomness': 'same class permutations and group Dirichlet draws at every lambda',
        'client_totals': 'free to change; actual sample-weighted FedAvg',
        'schedule': 'default_rng(split_seed), full participation, 100 rounds',
        'code_sha256': {name: sha256(REPO / name) for name in sources},
    }


def verify_prepared(args, value, run):
    from scripts.cliplora_tail_concentration_protocol import sha256
    info = json.loads((run / 'experiment.json').read_text(encoding='utf-8'))
    if info['specification'] != specification(args, value):
        raise RuntimeError(f'Prepared experiment differs from requested configuration/code: {run}. Use a new --output-root.')
    for name, expected in info['protocol_sha256'].items():
        if sha256(run / 'protocol' / name) != expected:
            raise RuntimeError(f'Prepared protocol was changed: {run / "protocol" / name}')
    return info


def prepare_run(args, value, run):
    from scripts.cliplora_tail_concentration_protocol import (
        load_pool, reference_partition, controlled_partition, audit_partition,
        write_partition, write_json, sha256,
    )
    from tools.eri_closure.protocol import build_protocol
    import numpy as np
    run.mkdir(parents=True, exist_ok=True)
    with preparation_lock(run / '.prepare.lock'):
        if (run / 'experiment.json').exists():
            return verify_prepared(args, value, run)
        if any(p.name != '.prepare.lock' for p in run.iterdir()):
            raise RuntimeError(f'Unrecognized or incomplete output directory: {run}. Use a new --output-root.')
        raw_ids, labels, location = load_pool(args.data_root)
        reference = reference_partition(labels, args.protocol_seed)
        mapping = controlled_partition(labels, value, args.protocol_seed)
        stats, counts = audit_partition(labels, mapping, reference, value)
        root = run / 'protocol'
        build_protocol(root, data_root=args.data_root, imbalance_factor=.01,
                       audit_rounds=list(range(10, 101, 10)))
        rng = np.random.default_rng(args.protocol_seed)
        schedule = [rng.choice(30, 30, replace=False).tolist() for _ in range(100)]
        write_json(root / 'full_schedule.json', {'schedule': schedule})
        write_partition(root, raw_ids, labels, mapping, counts)
        write_json(root / 'partition_protocol.json', {
            **stats, 'description': 'Tail-concentration intervention; fixed non-tail allocation',
            'partition_source': 'partition_source.csv overrides nominal Client-LT training allocation',
            'evaluation': 'unchanged full balanced official test set; no per-client test metric',
            'lambda_zero_is_ordinary_dirichlet': False,
            'tail_class_share_formula': 'q_T + (1-q_T)*lambda, q_T=0.1; integer budget rounding',
        })
        info = {
            'specification': specification(args, value), 'statistics': stats,
            'source_train_sha256': sha256(location / 'train'),
            'source_meta_sha256': sha256(location / 'meta'),
            'pool_raw_ids_sha256': hashlib.sha256(raw_ids.astype('<i8').tobytes()).hexdigest(),
            'protocol_sha256': {p.name: sha256(p) for p in root.iterdir() if p.is_file()},
        }
        write_json(run / 'command.json', command_for(args, value, run))
        write_json(run / 'experiment.json', info)
        return info


def train_run(args, value, run):
    from scripts.cliplora_tail_concentration_protocol import write_json
    complete = run / 'completion.json'
    if complete.exists():
        if json.loads(complete.read_text(encoding='utf-8')).get('completed_round') != 100:
            raise RuntimeError(f'Invalid completion marker: {complete}')
        print(f'Already complete; skipping {run}', flush=True)
        return
    if any((run / name).exists() for name in ('launch.json', 'round_metrics.csv', 'progress.json', 'log.txt')):
        raise RuntimeError(f'Existing/unfinished training found: {run}. This baseline has no automatic resume; use a new --output-root.')
    command = command_for(args, value, run)
    # Exclusive marker prevents accidentally launching the same run twice.
    with (run / 'launch.json').open('x', encoding='utf-8') as stream:
        json.dump({'command': command, 'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES')}, stream, indent=2)
    write_json(run / 'command.json', command)
    print(shlex.join(command), flush=True)
    subprocess.run(command, cwd=REPO, check=True)
    if not complete.exists() or json.loads(complete.read_text(encoding='utf-8')).get('completed_round') != 100:
        raise RuntimeError(f'Training exited without a complete 100-round result: {run}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=['plan', 'prepare', 'train', 'plot'], default='plan')
    parser.add_argument('--lambdas', type=float, nargs='+', default=list(VALUES), choices=VALUES)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--protocol-seed', type=int, default=42)
    parser.add_argument('--data-root', type=Path, default=Path('DATA'))
    parser.add_argument('--output-root', type=Path, default=Path(DEFAULT_OUTPUT))
    parser.add_argument('--num-workers', type=int, default=8)
    args = parser.parse_args()
    if len(set(args.lambdas)) != len(args.lambdas):
        parser.error('--lambdas must be unique')
    if min(args.seed, args.protocol_seed, args.num_workers) < 0:
        parser.error('seeds and num-workers must be non-negative')
    os.chdir(REPO)
    if args.stage == 'plot':
        from scripts.plot_cliplora_tail_concentration import plot_results
        plot_results(args)
        return
    print('J baseline: IF=100, alpha=0.5, 30 clients, 100 classes, 100 rounds; non-tail allocation fixed.', flush=True)
    for value in args.lambdas:
        run = run_path(args, value)
        if args.stage == 'plan':
            print(f'\nlambda={value:g}; nominal tail share={100*(.1+.9*value):g}%')
            print(shlex.join(command_for(args, value, run)))
            continue
        info = prepare_run(args, value, run)
        print(f'Prepared lambda={value:g}: {info["statistics"]}', flush=True)
        if args.stage == 'train':
            train_run(args, value, run)


if __name__ == '__main__':
    main()
