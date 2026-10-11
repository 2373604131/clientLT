"""Independent Tail30 partition experiment: original joint x three A paths."""
import argparse
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_cliplora_joint_aggregation as single
from tools.client_aggregation.tail30 import ARMS, register, status, summarize


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('plan','preflight','smoke','run','status','summary','pack'), default='plan')
    parser.add_argument('--arms', nargs='+', choices=ARMS, default=list(ARMS))
    parser.add_argument('--output-root', type=Path, default=Path('output/cifar100_LT/joint_tail30_v1_parallel8'))
    parser.add_argument('--data-root', type=Path, default=Path('DATA'))
    parser.add_argument('--client-concurrency', type=int, choices=(1,4,6,8), default=8)
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--cuda-policy', choices=('legacy','deterministic'), default='legacy')
    parser.add_argument('--stop-after-round', type=int, default=0)
    args = parser.parse_args(argv)
    if len(set(args.arms)) != len(args.arms) or args.num_workers<0 or not 0<=args.stop_after_round<=100:
        parser.error('Duplicate arms or invalid worker/round count')
    return args


def main(argv=None):
    args = parse_args(argv)
    os.chdir(ROOT)
    if args.stage == 'status':
        for row in status(args.output_root):
            print(row)
        return
    if args.stage in ('summary','pack'):
        for row in summarize(args.output_root):
            print(row)
        if args.stage == 'pack':
            from tools.client_aggregation.analysis import pack_results
            print('Archive:',pack_results(args.output_root))
        return
    plans = register(args)
    print('Tail30 partition registered; original joint lambda=0.1; seed42; no B/C; '
          'online diagnostics retain Tail20, final summary adds Tail30.',flush=True)
    if args.stage == 'plan':
        return
    for arm in args.arms:
        single.preflight(plans[arm], check_cuda=args.stage in ('smoke','run'))
    if args.stage == 'preflight':
        print('CPU/data/weights/configuration preflight passed; GPU execution not tested.')
        return
    for arm in args.arms:
        folder = args.output_root/arm
        single.execute(single.make_job(plans[arm],arm,'smoke'),folder)
        if args.stage == 'run':
            single.execute(single.make_job(plans[arm],arm),folder,args.stop_after_round)


if __name__ == '__main__':
    try:
        main()
    except (ValueError,OSError,KeyError) as error:
        raise SystemExit(str(error))
