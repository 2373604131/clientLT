"""Foreground SFRA / classification-preserving SFRA runs and result collection."""
import argparse
import json
import math
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
from types import SimpleNamespace

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from scripts.run_cliplora_a_refresh import build_command


def run_directory(args):
    setting = f'lambda{args.retention_weight:g}_protocol{args.protocol_seed}'
    if args.method == 'full-cp':
        setting = f'lambda{args.retention_weight:g}_mu{args.classification_weight:g}_protocol{args.protocol_seed}'
    if args.method == 's':
        setting = f'baseline_protocol{args.protocol_seed}'
    if args.partition == 'noniid-labeldir-fine':
        setting += f'_beta{args.dirichlet_beta:g}'
    return args.output_root.resolve()/f'seed{args.seed}'/args.partition/args.method/setting


def prepare_protocol(args, run):
    if args.reference_run is not None:
        source = args.reference_run.resolve()
    elif args.partition == 'client-longtail' and not args.fresh_protocol:
        source = args.bridge_root.resolve()/f'seed{args.protocol_seed}'/'client-longtail'/'c1'
    else:
        from scripts.cliplora_fresh_protocol import prepare_fresh_protocol
        return prepare_fresh_protocol(args, run), ''
    required = ('partition_manifest.csv', 'bridge_metadata.json', 'protocol/full_schedule.json',
                'protocol/eri_protocol.json', 'protocol/probe_manifest.csv')
    for name in required:
        if not (source/name).is_file():
            raise FileNotFoundError(f'{source/name}; provide --reference-run with the complete Full-10 or S run, '
                                    'or use --fresh-protocol for all new groups.')
    metadata = json.loads((source/'bridge_metadata.json').read_text(encoding='utf-8'))
    if metadata['topology'] != args.partition:
        raise ValueError('Reference partition differs; never copy Client-LT capacities into Dirichlet.')
    (run/'protocol').mkdir(parents=True)
    for path in (source/'protocol').iterdir():
        if path.is_file():
            shutil.copy2(path, run/'protocol'/path.name)
    manifest = run/'protocol/partition_source.csv'
    shutil.copy2(source/'partition_manifest.csv', manifest)
    shutil.copy2(source/'bridge_metadata.json', run/'protocol/source_metadata.json')
    return run/'protocol/full_schedule.json', str(manifest)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=['train', 'summary', 'pack'], default='train')
    parser.add_argument('--method', choices=['s', 'current', 'full', 'flat', 'full-cp'], default='full')
    parser.add_argument('--retention-weight', type=float, default=10., help='Retention lambda; keep 10 for the full-cp pilot')
    parser.add_argument('--classification-weight', type=float, default=1., help='Classification preservation mu; full-cp only')
    parser.add_argument('--partition', choices=['client-longtail', 'noniid-labeldir-fine'], default='client-longtail')
    parser.add_argument('--dirichlet-beta', type=float, default=.5)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--protocol-seed', type=int, default=42)
    parser.add_argument('--data-root', type=Path, default=Path('DATA'))
    parser.add_argument('--output-root', type=Path,
                        help='Default: output/cifar100_LT/sfra_cp for full-cp, sfra_v1 otherwise')
    parser.add_argument('--bridge-root', type=Path, default=Path('output/cifar100_LT/a_refresh_topology_bridge'))
    parser.add_argument('--reference-run', type=Path,
                        help='Replay the partition/order/schedule of an existing complete Full-10 or S run; not its trained weights')
    parser.add_argument('--fresh-protocol', action='store_true')
    parser.add_argument('--schedule-file', type=Path)
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--witness-batch-size', type=int, default=8, help='Memory setting, not witness count or algorithm budget')
    parser.add_argument('--resume', action='store_true', help='Resume this method/lambda/mu directory at its last completed round')
    args = parser.parse_args()
    os.chdir(REPO)
    if args.output_root is None:
        args.output_root = Path('output/cifar100_LT') / ('sfra_cp' if args.method == 'full-cp' else 'sfra_v1')
    if args.stage != 'train':
        from tools.sfra.summary import summarize, pack
        summarize(args.output_root, args.reference_run)
        if args.stage == 'pack':
            pack(args.output_root)
        return
    if not math.isfinite(args.retention_weight) or args.retention_weight < 0:
        parser.error('--retention-weight must be finite and nonnegative')
    if not math.isfinite(args.classification_weight) or args.classification_weight < 0:
        parser.error('--classification-weight must be finite and nonnegative')
    if args.witness_batch_size < 1:
        parser.error('--witness-batch-size must be positive')
    run = run_directory(args)
    if args.resume:
        checkpoint = run/'checkpoints/sfra_last.pt'
        if not checkpoint.is_file():
            parser.error(f'Missing round-boundary checkpoint: {checkpoint}')
        command = json.loads((run/'command.json').read_text(encoding='utf-8'))
        command[0] = sys.executable
        command[command.index('--sfra_resume')+1] = str(checkpoint)
        print('Resuming the ORIGINAL saved configuration; other training flags are not changed.', flush=True)
    else:
        if run.exists() and any(run.iterdir()):
            parser.error(f'Output is not empty: {run}. Use --resume or a different --output-root.')
        schedule, manifest = prepare_protocol(args, run)
        baseline = SimpleNamespace(**{**vars(args), 'schedule_file':schedule, 'resume':None}, rank=4,
                                   matched_beta=args.dirichlet_beta, refresh_interval=10, refresh_epochs=1, refresh_lr=.001)
        command, _ = build_command(baseline, 'off')
        command[command.index('--output-dir')+1] = str(run)
        command[command.index('--split_seed')+1] = str(args.protocol_seed)
        index = command.index('DATALOADER.NUM_WORKERS')
        command[index:index] = ['--lac_method','s','--lac_partition_manifest',manifest,
            '--lac_la_tau','1','--lac_a_lr_mult','1', '--sfra_variant',args.method,
            '--sfra_retention_weight',str(args.retention_weight),
            '--sfra_classification_weight',str(args.classification_weight),
            '--sfra_witness_batch_size',str(args.witness_batch_size), '--sfra_resume','']
        (run/'command.json').write_text(json.dumps(command, indent=2), encoding='utf-8')
    print(shlex.join(command), flush=True)
    subprocess.run(command, check=True)
    print(f'Training complete: {run}', flush=True)


if __name__ == '__main__':
    main()
