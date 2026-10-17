"""Seven independent Client-LT batches; plan by default, never implicit training."""
import argparse
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from tools.benchmarks.common import read_json
from tools.multidataset.catalog import BATCHES, GROUPS, METHODS


def parser_for_cli():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('plan', 'prepare', 'preflight', 'smoke', 'run', 'status', 'summary', 'pack'), default='plan')
    parser.add_argument('--batches', nargs='+', choices=BATCHES, default=list(BATCHES))
    parser.add_argument('--group', choices=('all', *GROUPS), default='all')
    parser.add_argument('--methods', nargs='+', choices=METHODS)
    parser.add_argument('--seeds', nargs='+', type=int, default=[42])
    parser.add_argument('--data-config', type=Path, default=Path('configs/benchmarks/multidataset_data.json'))
    parser.add_argument('--reference-run', type=Path, default=Path('references/full10_clientlt'))
    parser.add_argument('--output-root', type=Path, default=Path('output/multidataset_clientlt_old_ab_v1'))
    parser.add_argument('--gpus', nargs='+', type=int, default=[0])
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--stop-after', type=int, default=0, help='Pause formal jobs at this committed round; 0 means finish')
    parser.add_argument('--duration-estimates', type=Path, help='Historical per-method seconds, for scheduling only')
    return parser


def main(argv=None):
    parser = parser_for_cli()
    args = parser.parse_args(argv)
    try:
        from tools.multidataset.runner import selected_methods, make_job, register, preflight, run_jobs
        from tools.multidataset.protocol import prepare
        from tools.multidataset.collect import collect, pack
        if (len(args.seeds) != len(set(args.seeds)) or min(args.seeds) < 0 or max(args.seeds) >= 2**32
            or len(args.batches) != len(set(args.batches)) or args.num_workers < 0 or not 0 <= args.stop_after <= 100):
            raise ValueError('Invalid seeds, duplicate batches, worker count or stopping round')
        root = args.output_root.resolve()
        if args.stage in ('status', 'summary', 'pack'):
            for row in collect(root):
                print(row)
            if args.stage == 'pack':
                print('Archive:', pack(root))
            print('Report:', root / 'analysis/report.md')
            return
        methods = selected_methods(args.group, args.methods)
        total = len(args.batches) * len(args.seeds) * len(methods)
        print(f'{len(methods)} methods; {len(args.batches)} batches; {len(args.seeds)} seeds; {total} formal runs')
        for batch in args.batches:
            name, value = BATCHES[batch]
            print(f'  {batch}: {name}, Client-LT specialization_lambda={value}, joint weight strength=0.1')
        if args.stage == 'plan':
            print('Methods:', ', '.join(methods))
            print('Plan only: no data required, no models trained. Edit data-config paths, then prepare/smoke/run.')
            return
        data = read_json(args.data_config)
        for batch in args.batches:
            name, value = BATCHES[batch]
            spec = data[name]
            protocol = prepare(name, spec['root'], root / batch / 'protocol', value,
                args.reference_run, spec.get('split_id', 1))
            print('PREPARED', batch, 'classes=', protocol['classes'],
                'train=', sum(sum(row) for row in protocol['matrix']), flush=True)
        if args.stage == 'prepare':
            return
        jobs = [make_job(root, batch, method, seed, args.num_workers, args.reference_run, args.stage == 'smoke')
                for batch in args.batches for seed in args.seeds for method in methods]
        paths = [register(root, job) for job in jobs]
        preflight(jobs)
        if args.stage == 'preflight':
            print('Data, source and CUDA preflight passed. Run smoke to verify actual method execution.')
            return
        if args.stage == 'run':
            # Each formal job must first pass its corresponding isolated smoke.
            smoke_jobs = [make_job(root, batch, method, seed, args.num_workers, args.reference_run, True)
                for batch in args.batches for seed in args.seeds for method in methods]
            smoke_paths = [register(root, job) for job in smoke_jobs]
            estimates = read_json(args.duration_estimates) if args.duration_estimates else None
            from scripts.run_ab_validation import file_lock
            with file_lock(root / 'locks/launcher.lock', timeout=.1):
                run_jobs(paths, args.gpus, args.stop_after, smoke_paths=smoke_paths,
                    duration_estimates=estimates, schedule_path=root / 'scheduling/run_queue.json')
        else:
            run_jobs(paths, args.gpus, schedule_path=root / 'scheduling/smoke_queue.json')
        collect(root)
        print('Report:', root / 'analysis/report.md')
    except (OSError, ValueError, RuntimeError, KeyError) as exc:
        parser.error(str(exc))


if __name__ == '__main__':
    main()
