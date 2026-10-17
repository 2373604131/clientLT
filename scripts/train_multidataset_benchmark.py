"""Internal worker; launched with exactly one visible GPU by the batch runner."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    from tools.benchmarks.common import read_json
    from tools.multidataset.runner import worker
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--job', required=True, type=Path)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--stop-after', type=int, default=0)
    args = parser.parse_args()
    if not 0 <= args.stop_after <= 100:
        parser.error('--stop-after must be in 0..100')
    worker(read_json(args.job), args.resume, args.stop_after)


if __name__ == '__main__':
    main()
