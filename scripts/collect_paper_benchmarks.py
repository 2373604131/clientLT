"""Read-only training-result collection and model-free archive creation."""
import argparse
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def main():
    from tools.benchmarks.collect import collect, pack, status
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root', type=Path, default=Path('output/cifar100_LT/paper_benchmarks_seed42'))
    parser.add_argument('--status', action='store_true')
    parser.add_argument('--pack', action='store_true')
    args = parser.parse_args()
    if args.status:
        for row in status(args.output_root):
            print(row)
    else:
        collect(args.output_root)
        print('Report:', args.output_root / 'analysis/report.md')
        if args.pack:
            print('Archive:', pack(args.output_root))


if __name__ == '__main__':
    main()
