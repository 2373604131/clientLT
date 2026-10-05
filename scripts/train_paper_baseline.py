"""Internal worker for one registered external-baseline job."""
import argparse
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def main():
    from tools.benchmarks.common import preflight, read_json
    from tools.benchmarks.runtime import train
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--job', type=Path, required=True)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--stop-after', type=int)
    args = parser.parse_args()
    job = read_json(args.job)
    preflight(job, require_cuda=True)
    train(job, resume=args.resume, stop_after=args.stop_after)


if __name__ == '__main__':
    main()
