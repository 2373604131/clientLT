"""Tail-directed transfer from clients missing the target class label."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.run_cliplora_sfra import main

if __name__ == '__main__':
    main(default_b_transfer=True, default_transfer_mode='shared', default_directed=True)
