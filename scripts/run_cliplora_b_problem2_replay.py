"""Preflight or replay E00/E10/E01/E11 from saved B events; never run local training."""
import argparse
import math
from pathlib import Path
import shutil
import subprocess
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from tools.sfra.b_problem2 import (SCHEMA, ROUNDS, VARIANTS, preflight, receipt, file_hash,
                                  read_json, write_json, replace_argument, summarize)
from tools.sfra.maintext import TRAINING_FILES

CODE_FILES = tuple(dict.fromkeys((*TRAINING_FILES, 'utils/b_problem2_math.py',
    'utils/cliplora_b_problem2.py', 'utils/b_problem2_replay.py',
    'utils/cliplora_b_shared_transfer.py', 'utils/cliplora_b_transfer.py',
    'utils/cliplora_bridge_audit.py', 'utils/cliplora_functional_feedback.py',
    'utils/sfra_execution.py', 'utils/sfra_fast_feedback.py', 'utils/sfra_resident_feedback.py',
    'tools/sfra/b_problem2.py', 'scripts/run_cliplora_b_problem2_replay.py')))


def build_command(source, run, data_root, workers):
    command = read_json(source/'command.json')
    command[0] = sys.executable
    entries = [i for i, value in enumerate(command) if Path(value).name == 'federated_main.py']
    if len(entries) != 1:
        raise ValueError('Expected a federated_main.py reference command')
    command[entries[0]] = str(REPO/'federated_main.py')
    for flag, value in {'--root':data_root, '--output-dir':run,
            '--client_schedule_file':run/'protocol/full_schedule.json',
            '--lac_partition_manifest':run/'protocol/partition_source.csv',
            '--sfra_resume':'', '--sfra_stop_after_round':0, '--method_a_diagnostic_manifest':'',
            '--b_problem2_variant':'off', '--b_problem2_replay_manifest':run/'replay_job.json'}.items():
        replace_argument(command, flag, value)
    command[command.index('DATALOADER.NUM_WORKERS')+1] = str(workers)
    return command


def run(args):
    source, root = args.source_run.resolve(), args.output_root.resolve()
    if root == source or source in root.parents or root in source.parents:
        raise ValueError('Replay output must be separate from the source directory')
    check = preflight(source, args.rounds)
    write_json(root/'preflight.json', check)
    if not check['ready']:
        raise ValueError('Source not ready; see preflight.json for missing files/configuration errors')
    if args.data_root is None or not args.data_root.is_dir():
        raise ValueError('--data-root must point to existing training data')
    if 'E00' not in args.variants:
        raise ValueError('Include E00 for equivalence and norm-matched comparisons')
    import torch
    if not torch.cuda.is_available():
        raise ValueError('Replay needs the training GPU environment; preflight/summary do not')
    hashes = {name:file_hash(REPO/name) for name in CODE_FILES}
    plan = dict(schema_version=SCHEMA, source=str(source), rounds=args.rounds, variants=args.variants,
                harm_beta=args.harm_beta, donor_diagnostics=not args.skip_donor_diagnostics,
                source_sha256=receipt(source, args.rounds), code_sha256=hashes,
                data_root=str(args.data_root.resolve()), num_workers=args.num_workers)
    path = root/'replay_plan.json'
    if path.is_file() and read_json(path) != plan:
        raise ValueError('Replay plan, source or code changed; use a new output directory')
    if not path.is_file() and any(p.name != 'preflight.json' for p in root.iterdir()):
        raise ValueError('Unregistered nonempty replay output')
    write_json(path, plan)
    for rnd in args.rounds:
        folder = root/f'r{rnd:03d}'
        if (folder/'replay_completion.json').is_file():
            print(f'Skipping complete replay r{rnd:03d}', flush=True)
            continue
        if folder.exists() and any(folder.iterdir()):
            raise ValueError(f'Incomplete event: {folder}; use a new output root (no partial-C resume)')
        (folder/'protocol').mkdir(parents=True, exist_ok=True)
        for p in (source/'protocol').iterdir():
            if p.is_file():
                shutil.copy2(p, folder/'protocol'/p.name)
        shutil.copy2(source/'partition_manifest.csv', folder/'protocol/partition_source.csv')
        shutil.copy2(source/'bridge_metadata.json', folder/'protocol/source_metadata.json')
        write_json(folder/'replay_job.json', dict(plan, round=rnd))
        command = build_command(source, folder, args.data_root.resolve(), args.num_workers)
        write_json(folder/'command.json', command)
        subprocess.run(command, cwd=REPO, check=True)
        if not (folder/'replay_completion.json').is_file():
            raise ValueError(f'Child exited without completion: {folder}')
        if {name:file_hash(REPO/name) for name in CODE_FILES} != hashes:
            raise ValueError('Code changed during replay; results require audit')
    if receipt(source, args.rounds) != plan['source_sha256']:
        raise ValueError('Source changed during replay; results require audit')
    return summarize(root)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=['preflight','run','summary','pack'], default='preflight')
    parser.add_argument('--source-run', type=Path)
    parser.add_argument('--output-root', type=Path, default=Path('output/cifar100_LT/b_problem2_replay'))
    parser.add_argument('--data-root', type=Path)
    parser.add_argument('--rounds', type=int, nargs='+', default=list(ROUNDS))
    parser.add_argument('--variants', choices=VARIANTS, nargs='+', default=list(VARIANTS))
    parser.add_argument('--harm-beta', type=float, default=1.)
    parser.add_argument('--skip-donor-diagnostics', action='store_true')
    parser.add_argument('--num-workers', type=int, default=8)
    args = parser.parse_args()
    if not math.isfinite(args.harm_beta) or args.harm_beta < 0 or len(set(args.variants)) != len(args.variants):
        parser.error('Require finite nonnegative beta and unique variants')
    if args.num_workers < 0:
        parser.error('--num-workers must be nonnegative')
    if args.stage in ('preflight','run') and args.source_run is None:
        parser.error('--source-run is required')
    if args.source_run is not None:
        source, root = args.source_run.resolve(), args.output_root.resolve()
        if source == root or source in root.parents or root in source.parents:
            parser.error('Use a separate output directory, outside the source tree')
    if args.stage == 'preflight':
        check = preflight(args.source_run, args.rounds)
        write_json(args.output_root/'preflight.json', check)
        print(f'Ready={check["ready"]}; missing={len(check["missing"])}; errors={check["errors"]}')
        for name in check['missing']:
            print('Missing: '+name)
    elif args.stage == 'run':
        print(run(args))
    else:
        print(summarize(args.output_root))
        if args.stage == 'pack':
            from tools.sfra.summary import pack
            pack(args.output_root)


if __name__ == '__main__':
    main()
