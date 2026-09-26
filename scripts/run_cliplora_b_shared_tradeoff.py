"""Donor + full C transfer with class-cyclic non-tail coverage and one loss weight."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.run_cliplora_sfra import main


if __name__ == '__main__':
    main(default_b_transfer=True, default_transfer_mode='shared',
         default_non_tail_sampling='class-cyclic', default_tail_weight=.35)
