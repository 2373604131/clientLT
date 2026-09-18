"""Paired, one-epoch A/B role interventions from existing S checkpoints."""
import argparse
import os
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=['run', 'summary'], default='run')
    parser.add_argument('--anchor-origin', choices=['both', 'client-longtail', 'noniid-labeldir-fine'], default='both')
    parser.add_argument('--rounds', nargs='+', type=int, choices=[20, 50, 80], default=[20, 50, 80])
    parser.add_argument('--clt-run', type=Path, default=Path(
        'output/cifar100_LT/la_control/seed42/client-longtail/s/tau1_a1_protocol42'))
    parser.add_argument('--dirichlet-run', type=Path, default=Path(
        'output/cifar100_LT/la_control/seed42/noniid-labeldir-fine/s/tau1_a1_protocol42_beta0.5'))
    parser.add_argument('--data-root', type=Path, default=Path('DATA'))
    parser.add_argument('--output-root', type=Path, default=Path('output/cifar100_LT/factor_roles'))
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--num-workers', type=int, default=0,
                        help='Fresh workers per client; 0 is simplest for reproducible paired streams')
    args = parser.parse_args()
    os.chdir(REPO)
    if args.stage == 'summary':
        from tools.factor_roles.summary import summarize
        summarize(args.output_root)
    else:
        from tools.factor_roles.runtime import run
        run(args)


if __name__ == '__main__':
    main()
