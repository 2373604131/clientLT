"""Dedicated worker: patch only this process, never the legacy training source."""
import argparse
from pathlib import Path
import runpy
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def main(argv=None):
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument('--joint-spec', type=Path, required=True)
    args, remaining = parser.parse_known_args(argv)
    from tools.client_aggregation.protocol import SCHEMA, code_hashes, load_json
    from tools.sfra.simple_controls import probe_manifest_replay
    job = load_json(args.joint_spec)
    if job['schema'] != SCHEMA or job['code_sha256'] != code_hashes(REPO):
        raise ValueError('Worker specification/source mismatch')
    run = args.joint_spec.resolve().parent
    if '--output-dir' not in remaining or Path(remaining[remaining.index('--output-dir')+1]).resolve() != run:
        raise ValueError('Worker output differs from its specification directory')
    expected = load_json(run / 'protocol/source_metadata.json')['probe_manifest_sha256']
    payload, replay = probe_manifest_replay(run / 'protocol/probe_manifest.csv', expected)
    if replay['newline_conversion'] != 'unchanged':
        raise ValueError('Prepared probe bytes differ; use the launcher to prepare the protocol')
    if job.get('settings', {}).get('client_concurrency', 1) == 4:
        import torch
        # Four CPU feeders should not each launch a full intra-op thread pool.
        torch.set_num_threads(1)
    from tools.client_aggregation.runtime import runtime_class
    from utils import cliplora_sfra
    original, original_argv = cliplora_sfra.SFRARuntime, sys.argv
    try:
        cliplora_sfra.SFRARuntime = runtime_class(job)
        sys.argv = [str(REPO / 'federated_main.py'), *remaining]
        runpy.run_path(str(REPO / 'federated_main.py'), run_name='__main__')
    finally:
        cliplora_sfra.SFRARuntime, sys.argv = original, original_argv


if __name__ == '__main__':
    main()
