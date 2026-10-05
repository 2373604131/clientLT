"""Second-batch shared-factor baselines; first-batch defaults stay unchanged."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.run_paper_benchmarks import main

if __name__ == '__main__':
    main(suite='factors')
