"""FedAvg / TailRW16 / joint + frozen Method A (Full-CP), without Method B.

Run each policy with --stage run --methods <rule> --client-concurrency 8.
CUDA_VISIBLE_DEVICES selects its GPU; --gpus applies to --stage server only.
The default output root is separate from completed ordinary-A experiments.
"""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.run_cliplora_plain_a_aggregation import main as run_suite, parse_args as parse_suite


def parse_args(argv=None):
    return parse_suite(argv, arm='method_a')


def main(argv=None):
    return run_suite(argv, arm='method_a')


if __name__ == '__main__':
    try:
        main()
    except (ValueError, OSError, KeyError) as error:
        raise SystemExit(str(error))
