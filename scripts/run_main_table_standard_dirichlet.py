"""Launch only the five missing standard-Dirichlet controls in the main slide."""

import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
from types import SimpleNamespace

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from scripts.run_cliplora_v2 import build_command as build_capt_command


def build_jobs(args):
    jobs = []
    for method in args.experiments:
        if method == 'capt':
            config = SimpleNamespace(seed=args.seed, data_root=args.data_root,
                                     output_root=args.capt_output_root, num_workers=args.num_workers)
            command = build_capt_command(config, 'capt')
            command[command.index('--partition') + 1] = 'noniid-labeldir-fine'
            # Retain the schedule used by the slide's original CAPT launcher.
            schedule = args.capt_schedule_file or Path(f'output/cifar100_LT/v2_matched/full_schedule_seed{args.seed}.json')
            command[command.index('--client_schedule_file') + 1] = str(schedule)
            output = args.capt_output_root / f'seed{args.seed}' / 'capt'
        else:
            command = [sys.executable, '-u', 'scripts/run_cliplora_la_control.py',
                       '--stage', 'train', '--method', method,
                       '--partition', 'noniid-labeldir-fine', '--dirichlet-beta', '0.5',
                       '--seed', str(args.seed), '--protocol-seed', str(args.seed),
                       '--data-root', str(args.data_root), '--output-root', str(args.la_output_root),
                       '--num-workers', str(args.num_workers)]
            tau = 0 if method in ('e0', 'e1') else 1
            config_id = f'tau{tau}_a1'
            if method == 'e5':
                config_id += '_h2_t0.002_hist0.002_gain0.0001_p2'
            config_id += f'_protocol{args.seed}_beta0.5'
            output = args.la_output_root / f'seed{args.seed}' / 'noniid-labeldir-fine' / method / config_id
        jobs.append((method, command, output))
    return jobs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiments', nargs='+', choices=['e0', 'e1', 'e2', 'e5', 'capt'],
                        default=['e0', 'e1', 'e2', 'e5', 'capt'])
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--data-root', type=Path, default=Path('DATA'))
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--la-output-root', type=Path, default=Path('output/cifar100_LT/la_control'))
    parser.add_argument('--capt-output-root', type=Path, default=Path('output/cifar100_LT/capt_matched_standard_dirichlet'))
    parser.add_argument('--capt-schedule-file', type=Path)
    parser.add_argument('--dry-run', action='store_true', help='Print commands only; create no files and run no training')
    args = parser.parse_args()
    if len(set(args.experiments)) != len(args.experiments):
        parser.error('Do not repeat an experiment in one invocation.')
    jobs = build_jobs(args)
    # Check every destination before starting any training. Never append to old results.
    if not args.dry_run:
        if args.capt_schedule_file and not (REPO / args.capt_schedule_file).is_file():
            parser.error(f'Schedule not found: {args.capt_schedule_file}')
        for _, _, output in jobs:
            destination = REPO / output
            if destination.exists() and (not destination.is_dir() or any(destination.iterdir())):
                parser.error(f'Output is not empty: {destination}. Select only missing experiments or use a new output root.')
    for i, (method, command, output) in enumerate(jobs, 1):
        print(f'[{i}/{len(jobs)}] {method}: {output}', flush=True)
        print(subprocess.list2cmdline(command) if os.name == 'nt' else shlex.join(command), flush=True)
        if args.dry_run:
            continue
        if method == 'capt':
            destination = REPO / output
            destination.mkdir(parents=True, exist_ok=True)
            (destination / 'command.json').write_text(json.dumps(command, indent=2), encoding='utf-8')
        subprocess.run(command, cwd=REPO, check=True)
        print(f'DONE {method}', flush=True)


if __name__ == '__main__':
    main()
