"""One foreground LA/control experiment per allocated GPU; no background launch."""
import argparse
import json
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


def configuration_id(args):
    tau = args.la_tau if args.method in ('e2', 'e3', 'e5', 'j', 's') else 0
    value = f'tau{tau:g}_a{args.a_lr_mult:g}'
    if args.method in ('e4', 'e5'):
        history = args.tail_tolerance if args.history_tolerance is None else args.history_tolerance
        value += f'_h{args.lookahead_rounds}_t{args.tail_tolerance:g}_hist{history:g}_gain{args.min_gain:g}_p{args.patience}'
    value += f'_protocol{args.protocol_seed}'
    if args.partition == 'noniid-labeldir-fine':
        value += f'_beta{args.dirichlet_beta:g}'
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=['train', 'analyze', 'summary'], default='train')
    parser.add_argument('--method', choices=[f'e{i}' for i in range(6)] + ['j', 's'])
    parser.add_argument('--partition', choices=['client-longtail', 'noniid-labeldir-fine', 'matched-dirichlet'])
    parser.add_argument('--dirichlet-beta', type=float, default=.5)
    parser.add_argument('--fresh-protocol', action='store_true',
                        help='Build an independent protocol without requiring a prior bridge run')
    parser.add_argument('--schedule-file', type=Path)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--protocol-seed', type=int, default=42)
    parser.add_argument('--bridge-root', type=Path, default=Path('output/cifar100_LT/a_refresh_topology_bridge'))
    parser.add_argument('--data-root', type=Path, default=Path('DATA'))
    parser.add_argument('--output-root', type=Path, default=Path('output/cifar100_LT/la_control'))
    parser.add_argument('--la-tau', type=float, default=1.)
    parser.add_argument('--a-lr-mult', type=float, default=1.)
    parser.add_argument('--lookahead-rounds', type=int, default=2)
    parser.add_argument('--min-gain', type=float, default=1e-4)
    parser.add_argument('--tail-tolerance', type=float, default=.002)
    parser.add_argument('--history-tolerance', type=float)
    parser.add_argument('--patience', type=int, default=2)
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--normal-rounds', default='10,11,12,20,21,22,30,31,32,40,41,42,50,51,52,60,61,62,70,71,72,80,81,82,90,91,92,100', help='Offline normal B/AB attribution; all for every round')
    parser.add_argument('--quadrature-segments', type=int, default=1)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    os.chdir(REPO)
    if args.stage == 'summary':
        from tools.la_control.summary import summarize
        summarize(args.output_root)
        return
    if args.method is None or args.partition is None:
        parser.error('--method and --partition are required')
    assert 0 <= args.lookahead_rounds < 10
    assert args.patience >= 1 and args.a_lr_mult > 0 and args.la_tau >= 0
    assert args.min_gain >= 0 and args.tail_tolerance >= 0
    assert args.history_tolerance is None or args.history_tolerance >= 0
    run = args.output_root.resolve() / f'seed{args.seed}' / args.partition / args.method / configuration_id(args)
    if args.stage == 'analyze':
        from tools.la_control.analysis import analyze
        analyze(run, args.data_root.resolve(), args.device, args.normal_rounds, args.quadrature_segments)
        return
    if args.partition == 'matched-dirichlet':
        parser.error('matched-dirichlet is historical-analysis only. New training uses noniid-labeldir-fine.')
    if run.exists() and any(run.iterdir()):
        parser.error(f'Output is not empty: {run}. Use --stage analyze or another --output-root.')
    if args.partition == 'noniid-labeldir-fine' or args.fresh_protocol:
        from scripts.cliplora_fresh_protocol import prepare_fresh_protocol
        schedule = prepare_fresh_protocol(args, run)
        manifest = ''
    else:
        source = args.bridge_root.resolve() / f'seed{args.protocol_seed}' / args.partition / 'c1'
        # Historical CLT replay keeps its exact within-client data order.
        assert (source / 'partition_manifest.csv').is_file(), source
        assert (source / 'protocol/full_schedule.json').is_file(), source
        assert (source / 'protocol/eri_protocol.json').is_file(), source
        assert (source / 'protocol/probe_manifest.csv').is_file(), source
        (run / 'protocol').mkdir(parents=True)
        for path in (source / 'protocol').iterdir():
            if path.is_file():
                shutil.copy2(path, run / 'protocol' / path.name)
        shutil.copy2(source / 'partition_manifest.csv', run / 'protocol/partition_source.csv')
        shutil.copy2(source / 'bridge_metadata.json', run / 'protocol/source_metadata.json')
        schedule = run / 'protocol/full_schedule.json'
        manifest = str(run / 'protocol/partition_source.csv')
    baseline = SimpleNamespace(**{**vars(args), 'schedule_file': schedule}, rank=4,
                               matched_beta=args.dirichlet_beta, refresh_interval=10,
                               refresh_epochs=1, refresh_lr=.001, resume=None)
    command, _ = build_command(baseline, 'off')
    command[command.index('--output-dir')+1] = str(run)
    command[command.index('--split_seed')+1] = str(args.protocol_seed)
    index = command.index('DATALOADER.NUM_WORKERS')
    flags = ['--lac_method', args.method, '--lac_partition_manifest', manifest,
             '--lac_la_tau', str(args.la_tau), '--lac_a_lr_mult', str(args.a_lr_mult),
             '--lac_lookahead_rounds', str(args.lookahead_rounds), '--lac_min_gain', str(args.min_gain),
             '--lac_tail_tolerance', str(args.tail_tolerance), '--lac_patience', str(args.patience)]
    if args.history_tolerance is not None:
        flags += ['--lac_history_tolerance', str(args.history_tolerance)]
    command[index:index] = flags
    (run / 'command.json').write_text(json.dumps(command, indent=2), encoding='utf-8')
    print(shlex.join(command), flush=True)
    subprocess.run(command, check=True)
    print(f'Training complete: {run}', flush=True)


if __name__ == '__main__':
    main()
