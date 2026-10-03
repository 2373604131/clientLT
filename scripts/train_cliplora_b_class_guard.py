"""Dedicated worker: install the guard runtime in this process only."""
import argparse
from pathlib import Path
import runpy
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO))


def main(argv=None):
    parser=argparse.ArgumentParser(add_help=False,allow_abbrev=False)
    parser.add_argument('--guard-spec',type=Path,required=True)
    args,remaining=parser.parse_known_args(argv)
    from tools.sfra.b_class_guard import check_sources
    from tools.sfra.maintext import load_json
    spec=load_json(args.guard_spec)
    check_sources(spec,REPO)
    if '--output-dir' not in remaining or Path(remaining[remaining.index('--output-dir')+1]).resolve()!=args.guard_spec.resolve().parent:
        raise ValueError('Guard spec and worker output directory differ')
    from utils.cliplora_b_guard_runtime import runtime_class
    from utils import cliplora_sfra
    original=cliplora_sfra.SFRARuntime
    original_argv=sys.argv
    try:
        cliplora_sfra.SFRARuntime=runtime_class(spec)
        sys.argv=[str(REPO/'federated_main.py'),*remaining]
        runpy.run_path(str(REPO/'federated_main.py'),run_name='__main__')
    finally:
        cliplora_sfra.SFRARuntime=original
        sys.argv=original_argv


if __name__=='__main__':
    main()
