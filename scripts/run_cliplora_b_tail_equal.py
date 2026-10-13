"""Independent F-tail-equal entry; reuses the existing route training worker."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_cliplora_b_routes import main as run_routes


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    reserved = {'--mode', '--arms', '--source-run', '--rounds', '--rhos',
                '--no-rotations', '--include-legacy', '--skip-source-probe'}
    if any(a.split('=')[0] in reserved for a in argv):
        raise SystemExit('This entry fixes full mode and F-tail-equal; replay/other arms use run_cliplora_b_routes.py')
    return run_routes(['--mode', 'full', '--arms', 'F-tail-equal',
        '--output-root', 'output/cifar100_LT/b_tail_equal_v1', *argv])


if __name__ == '__main__':
    main()
