"""Install the route runtime only inside this dedicated training/replay worker."""
import argparse
from pathlib import Path
import runpy
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def main(argv=None):
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument('--route-spec', type=Path, required=True)
    args, remaining = parser.parse_known_args(argv)
    from tools.sfra.b_routes import check_sources, file_hash, verify_prepared_probe
    from tools.sfra.maintext import load_json
    spec = load_json(args.route_spec)
    check_sources(spec, REPO)
    if '--output-dir' not in remaining or Path(remaining[remaining.index('--output-dir')+1]).resolve() != args.route_spec.resolve().parent:
        raise ValueError('Route spec and worker output directory differ')
    source = Path(spec['source_run'] if spec['mode'] == 'replay' else spec['settings']['reference_run']).resolve()
    output = args.route_spec.resolve().parent
    if output == source or source in output.parents or output in source.parents:
        raise ValueError('Worker output must be separate from its source/reference')
    verify_prepared_probe(output)
    if spec['mode'] == 'replay':
        for name, expected in spec['source_sha256'].items():
            if file_hash(Path(spec['source_run'])/name) != expected:
                raise ValueError('Replay input changed: '+name)
    from utils.cliplora_b_route_runtime import runtime_class
    from utils import cliplora_sfra
    original, original_argv = cliplora_sfra.SFRARuntime, sys.argv
    try:
        cliplora_sfra.SFRARuntime = runtime_class(spec)
        sys.argv = [str(REPO/'federated_main.py'), *remaining]
        runpy.run_path(str(REPO/'federated_main.py'), run_name='__main__')
    finally:
        cliplora_sfra.SFRARuntime, sys.argv = original, original_argv


if __name__ == '__main__':
    main()
