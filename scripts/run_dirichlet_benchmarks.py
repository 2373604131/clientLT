"""Shared seed42 Dirichlet main table: 8 long-tail controls + 4 factors + A/AB.

Default is plan only. Train runs in the foreground on the visible GPU; shards
allow independent GPUs/nodes to share the same immutable partition and output.
"""
import argparse
import math
from pathlib import Path
import shlex
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from tools.benchmarks.common import make_job, preflight, read_json, register
from tools.benchmarks.dirichlet import ALL, GROUPS, prepare_reference, select_methods
from scripts.run_paper_benchmarks import command_for, execute


def parser_for_cli():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=['plan', 'prepare', 'preflight', 'smoke', 'train',
                                          'status', 'summary', 'pack'], default='plan')
    parser.add_argument('--group', choices=['all', *GROUPS], default='all')
    parser.add_argument('--methods', nargs='+', choices=ALL,
                        help='Explicit subset; overrides --group')
    parser.add_argument('--seed', type=int, choices=[42], default=42)
    parser.add_argument('--dirichlet-beta', type=float, default=.5,
                        help='Symmetric per-fine-class Dirichlet concentration (alpha)')
    parser.add_argument('--reference-run', type=Path, default=Path('references/full10_clientlt'),
                        help='Existing Client-LT reference; reuse its global pool, not its allocation')
    parser.add_argument('--data-root', type=Path, default=Path('DATA'))
    parser.add_argument('--output-root', type=Path,
                        help='Default: output/cifar100_LT/dirichlet_benchmarks_seed42_beta<VALUE>_v1')
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--num-shards', type=int, default=1)
    parser.add_argument('--shard-index', type=int, default=0)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--stop-after', type=int, help='External methods only: stop after a committed round')
    return parser


def main(argv=None):
    parser = parser_for_cli()
    args = parser.parse_args(argv)
    try:
        if not math.isfinite(args.dirichlet_beta) or args.dirichlet_beta <= 0 or args.num_workers < 0:
            raise ValueError('Require a positive finite beta and nonnegative workers')
        if args.stop_after is not None and not 1 <= args.stop_after <= 100:
            raise ValueError('--stop-after must be in 1..100')
        beta_id = format(args.dirichlet_beta, '.12g')
        root = (args.output_root or Path('output/cifar100_LT') /
                f'dirichlet_benchmarks_seed42_beta{beta_id}_v1').resolve()
        if args.stage in ('status', 'summary', 'pack'):
            from tools.benchmarks.collect import collect, pack, status
            if args.stage == 'status':
                for row in status(root):
                    print(row)
            else:
                collect(root)
                if args.stage == 'pack':
                    print(pack(root))
                print('Report:', root / 'analysis/report.md')
            return
        selected = select_methods(args.group, args.methods, args.shard_index, args.num_shards)
        if args.stage == 'smoke':
            skipped = [m for m in selected if m in GROUPS['ours']]
            selected = [m for m in selected if m not in GROUPS['ours']]
            if skipped:
                print('A/AB retain the frozen full-training entry; no short-run score:', ', '.join(skipped))
            if not selected:
                raise ValueError('Smoke requires at least one external method; use --group longtail or factors')
        if args.stop_after is not None and any(m in GROUPS['ours'] for m in selected):
            raise ValueError('--stop-after applies only to external methods; select a baseline group')
        reference = prepare_reference(args.reference_run, root / 'reference', args.dirichlet_beta)
        recipe = read_json(reference / 'protocol/partition_protocol.json')
        print('Ordinary Dirichlet beta=', args.dirichlet_beta, '; seed42; clients=30; rounds=100', sep='')
        print('Selected', len(selected), 'methods:', ', '.join(selected))
        print('Client sizes:', min(recipe['client_sample_counts']), '..', max(recipe['client_sample_counts']),
              '; aggregate steps/epoch:', recipe['steps_per_epoch'])
        if args.stage == 'prepare':
            print('Reference:', reference)
            return
        smoke = args.stage == 'smoke'
        run_root = root / 'smoke' if smoke else root
        jobs = [make_job(m, reference, args.data_root, run_root, args.num_workers, smoke,
                         suite='dirichlet') for m in selected]
        for job in jobs:
            register(job)
        if args.stage == 'plan':
            for job in jobs:
                print(job['label'], '\n ', shlex.join(command_for(job, args.resume, args.stop_after)))
            print('Plan only: no model was trained. Jobs:', run_root / 'jobs')
            return
        # Validate every selected method before spending GPU time on the first.
        for job in jobs:
            print(job['method'], preflight(job, require_cuda=True))
        if args.stage == 'preflight':
            return
        for index, job in enumerate(jobs, 1):
            print(f"[{index}/{len(jobs)}] START {job['method']}", flush=True)
            execute(job, args.resume, args.stop_after)
        from tools.benchmarks.collect import collect
        collect(run_root)
        print('Report:', run_root / 'analysis/report.md')
    except (OSError, ValueError, RuntimeError, KeyError) as error:
        parser.error(str(error))


if __name__ == '__main__':
    main()
