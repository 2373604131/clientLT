"""Compatibility name: the controls now use separate GPUs through the parallel launcher."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.run_method_a_simple_controls_parallel import cli


if __name__ == '__main__':
    print('Use run_method_a_simple_controls_parallel.py: four separate GPUs (default 0, 1, 2, 3).', flush=True)
    cli()
