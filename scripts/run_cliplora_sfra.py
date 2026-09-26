"""Foreground SFRA runs, optional donor B transfer, and result collection."""
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


def shared_tradeoff_enabled(args):
    return (args.b_transfer and getattr(args, 'transfer_mode', 'local') == 'shared' and
            (getattr(args, 'transfer_non_tail_sampling', 'sample') != 'sample' or
             getattr(args, 'transfer_tail_weight', .5) != .5))


def run_directory(args):
    setting = f'lambda{args.retention_weight:g}_protocol{args.protocol_seed}'
    if args.method.endswith('-cp'):
        setting = f'lambda{args.retention_weight:g}_mu{args.classification_weight:g}_protocol{args.protocol_seed}'
    if args.method == 's':
        setting = f'baseline_protocol{args.protocol_seed}'
    if args.partition == 'noniid-labeldir-fine':
        setting += f'_beta{args.dirichlet_beta:g}'
    if args.b_transfer:
        if getattr(args, 'transfer_mode', 'local') == 'shared':
            setting += '_bshared'
        setting += f'_b_lr{args.transfer_lr:g}_probe{args.probe_step:g}_reg{args.transfer_reg:g}'
        if shared_tradeoff_enabled(args):
            setting += f'_nt{args.transfer_non_tail_sampling}_tw{args.transfer_tail_weight:g}'
    if getattr(args, 'b_aggregation', 'sample') == 'uniform-transfer-rounds':
        setting += '_bagg_uniform8'
    if getattr(args, 'fast_execution_v2', False):
        setting += f'_fast_v2_f{args.feedback_batch_size}_c{args.feedback_cache_gib:g}'
    elif getattr(args, 'fast_execution', False):
        setting += '_fast'
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


def main(default_b_transfer=False, default_transfer_mode='local',
         default_non_tail_sampling='sample', default_tail_weight=.5):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=['train', 'summary', 'pack'], default='train')
    parser.add_argument('--method', choices=['s', 'current', 'full', 'flat', 'full-cp', 'flat-cp', 'current-cp'],
                        default='full-cp' if default_b_transfer else 'full')
    parser.add_argument('--b-transfer', action='store_true', default=default_b_transfer)
    parser.add_argument('--transfer-mode', choices=['local', 'shared'], default=default_transfer_mode,
                        help='local: original recipient C; shared: one donor C jointly calibrated on shared B')
    parser.add_argument('--b-aggregation', choices=['sample', 'uniform-transfer-rounds'], default='sample',
                        help='Uniform whole-client B aggregation only at rounds 30,40,...,100; A is unchanged')
    parser.add_argument('--transfer-lr', type=float, default=.3 if default_transfer_mode == 'shared' else .1,
                        help='Adam learning rate for donor matrices C')
    parser.add_argument('--probe-step', type=float, default=.1, help='Temporary injection epsilon, not final transfer strength')
    parser.add_argument('--transfer-reg', type=float, default=.001, help='Mean squared-matrix penalty coefficient')
    parser.add_argument('--transfer-non-tail-sampling', choices=['sample', 'class-cyclic'],
                        default=default_non_tail_sampling,
                        help='Shared C only: image-uniform original batches or class-cyclic non-tail coverage')
    parser.add_argument('--transfer-tail-weight', type=float, default=default_tail_weight,
                        help='Shared C tail LA loss fraction; non-tail uses 1-weight, tail-only clients use 1')
    parser.add_argument('--retention-weight', type=float, default=10., help='Retention lambda; keep 10 for the full-cp pilot')
    parser.add_argument('--classification-weight', type=float, default=1., help='Classification preservation mu; all -cp variants')
    parser.add_argument('--partition', choices=['client-longtail', 'noniid-labeldir-fine'], default='client-longtail')
    parser.add_argument('--dirichlet-beta', type=float, default=.5)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--protocol-seed', type=int, default=42)
    parser.add_argument('--data-root', type=Path, default=Path('DATA'))
    parser.add_argument('--output-root', type=Path,
                        help='Default under output/cifar100_LT: sfra_b_shared_tradeoff for new C calibration; sfra_b_shared_transfer for original shared C; otherwise the corresponding baseline root')
    parser.add_argument('--bridge-root', type=Path, default=Path('output/cifar100_LT/a_refresh_topology_bridge'))
    parser.add_argument('--reference-run', type=Path,
                        help='Replay the partition/order/schedule of an existing complete Full-10 or S run; not its trained weights')
    parser.add_argument('--fresh-protocol', action='store_true')
    parser.add_argument('--schedule-file', type=Path)
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--witness-batch-size', type=int, default=8, help='Memory setting, not witness count or algorithm budget')
    parser.add_argument('--fast-execution', action='store_true',
                        help='Opt-in FP32 execution caches and equivalent feedback batching; separate _fast directory')
    parser.add_argument('--fast-execution-v2', action='store_true',
                        help='Bounded GPU prefix cache and vectorized feedback; separate _fast_v2 directory')
    parser.add_argument('--feedback-batch-size', type=int, default=64,
                        help='V2 correction/measurement forward batch, not the training batch or witness count')
    parser.add_argument('--feedback-cache-gib', type=float, default=4.,
                        help='V2 GPU prefix cache limit, 0-4 GiB; 0 uses CPU storage; reserve 4 GiB for computations')
    parser.add_argument('--stop-after-round', type=int, default=None,
                        help='Pause at this completed round for timing; 0 continues to round 100, also on resume')
    parser.add_argument('--resume', action='store_true', help='Resume this configuration directory at its last completed round')
    args = parser.parse_args()
    os.chdir(REPO)
    if args.output_root is None:
        name = ('sfra_b_shared_tradeoff' if shared_tradeoff_enabled(args) else
                'sfra_b_shared_transfer' if args.b_transfer and args.transfer_mode == 'shared' else
                'sfra_b_aggregation' if args.b_aggregation == 'uniform-transfer-rounds' else
                ('sfra_b_transfer' if args.b_transfer else ('sfra_cp' if args.method.endswith('-cp') else 'sfra_v1')))
        args.output_root = Path('output/cifar100_LT') / name
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
    if args.stop_after_round is not None and not 0 <= args.stop_after_round <= 100:
        parser.error('--stop-after-round must be between 0 and 100')
    if args.fast_execution_v2:
        if args.feedback_batch_size < 8:
            parser.error('--feedback-batch-size must be at least 8')
        if not math.isfinite(args.feedback_cache_gib) or not 0 <= args.feedback_cache_gib <= 4:
            parser.error('--feedback-cache-gib must be finite and between 0 and 4')
    elif args.feedback_batch_size != 64 or args.feedback_cache_gib != 4.:
        parser.error('--feedback-batch-size / --feedback-cache-gib require --fast-execution-v2')
    if args.b_transfer:
        if args.method == 's':
            parser.error('B transfer currently attaches to a functional A method: use full-cp or full.')
        if not all(math.isfinite(v) and v > 0 for v in (args.transfer_lr, args.probe_step)):
            parser.error('--transfer-lr and --probe-step must be finite and positive')
        if not math.isfinite(args.transfer_reg) or args.transfer_reg < 0:
            parser.error('--transfer-reg must be finite and nonnegative')
    if args.transfer_mode == 'shared' and (not args.b_transfer or args.b_aggregation != 'sample'):
        parser.error('Shared C requires B transfer and the unchanged sample-weighted B aggregation.')
    if not math.isfinite(args.transfer_tail_weight) or not 0 < args.transfer_tail_weight < 1:
        parser.error('--transfer-tail-weight must be strictly between 0 and 1')
    if (args.transfer_non_tail_sampling != 'sample' or args.transfer_tail_weight != .5) and not (
            args.b_transfer and args.transfer_mode == 'shared'):
        parser.error('The new calibration options require shared C transfer.')
    run = run_directory(args)
    if args.resume:
        checkpoint = run/'checkpoints/sfra_last.pt'
        if not checkpoint.is_file():
            parser.error(f'Missing round-boundary checkpoint: {checkpoint}')
        command = json.loads((run/'command.json').read_text(encoding='utf-8'))
        command[0] = sys.executable
        command[command.index('--sfra_resume')+1] = str(checkpoint)
        print('Resuming the ORIGINAL saved configuration; only an explicit stop-after-round may change.', flush=True)
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
        if args.b_aggregation != 'sample':
            index = command.index('DATALOADER.NUM_WORKERS')
            command[index:index] = ['--sfra_b_aggregation', args.b_aggregation]
        if args.fast_execution_v2:
            index = command.index('DATALOADER.NUM_WORKERS')
            command[index:index] = ['--sfra_fast_execution_v2',
                                   '--sfra_feedback_batch_size', str(args.feedback_batch_size),
                                   '--sfra_feedback_cache_gib', str(args.feedback_cache_gib)]
        elif args.fast_execution:
            index = command.index('DATALOADER.NUM_WORKERS')
            command[index:index] = ['--sfra_fast_execution']
        if args.b_transfer:
            index = command.index('DATALOADER.NUM_WORKERS')
            command[index:index] = ['--b_transfer_enable', '--b_transfer_lr', str(args.transfer_lr),
                                   '--b_transfer_probe_step', str(args.probe_step),
                                   '--b_transfer_reg', str(args.transfer_reg)]
            if args.transfer_mode == 'shared':
                index = command.index('DATALOADER.NUM_WORKERS')
                command[index:index] = ['--b_transfer_mode', 'shared']
                if shared_tradeoff_enabled(args):
                    index = command.index('DATALOADER.NUM_WORKERS')
                    command[index:index] = ['--b_transfer_non_tail_sampling', args.transfer_non_tail_sampling,
                                           '--b_transfer_tail_weight', str(args.transfer_tail_weight)]
    if args.stop_after_round is not None:
        flag = '--sfra_stop_after_round'
        if flag in command:
            command[command.index(flag)+1] = str(args.stop_after_round)
        else:
            index = command.index('DATALOADER.NUM_WORKERS')
            command[index:index] = [flag, str(args.stop_after_round)]
    if not args.resume:
        (run/'command.json').write_text(json.dumps(command, indent=2), encoding='utf-8')
    print(shlex.join(command), flush=True)
    subprocess.run(command, check=True)
    print(f'Run finished; see progress.json for completed rounds: {run}', flush=True)


if __name__ == '__main__':
    main()
