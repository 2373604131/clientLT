"""Run the seven standard-Dirichlet replacements sequentially in the foreground."""

import argparse
from pathlib import Path
import shlex
import subprocess
import sys


REPO = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiments', nargs='+', choices=['e2','e3','e5','j','s','c1','c2'],
                        default=['e3','j','s','e2','e5','c1','c2'])
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--data-root', type=Path, default=Path('DATA'))
    args = parser.parse_args()
    for index, method in enumerate(args.experiments,1):
        bridge = method in ('c1','c2')
        script = 'scripts/run_cliplora_topology_bridge.py' if bridge else 'scripts/run_cliplora_la_control.py'
        command = [sys.executable,'-u',script,'--stage','train','--method',method,
                   '--partition','noniid-labeldir-fine','--dirichlet-beta','0.5',
                   '--seed',str(args.seed),'--data-root',str(args.data_root)]
        if not bridge:
            command += ['--protocol-seed',str(args.seed)]
        print(f'[{index}/{len(args.experiments)}] START {method.upper()} / standard Dirichlet beta=0.5',flush=True)
        print(shlex.join(command),flush=True)
        subprocess.run(command,cwd=REPO,check=True)
        print(f'[{index}/{len(args.experiments)}] DONE {method.upper()}',flush=True)


if __name__=='__main__':
    main()
