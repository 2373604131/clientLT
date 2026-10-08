"""Opt-in real-data GPU smoke with explicit local-initialization provenance.

Never writes the real reference. A Windows/PyTorch initialization may differ
from the server's exact tensor hash. The separate test fixture records BOTH
hashes and only changes its initialization receipt; the production identity
check remains mandatory. This output must never enter a paper table.
"""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--reference-run', type=Path, default=Path('references/full10_clientlt'))
    parser.add_argument('--num-workers', type=int, default=2)
    args = parser.parse_args()
    from tools.benchmarks import common, runtime
    from utils.cliplora_a_refresh import state_hash
    import torch
    import Dassl.dassl.engine
    from trainers.cliplora import build_cliplora_model
    if not torch.cuda.is_available():
        raise RuntimeError('This validation requires an actual CUDA device')
    root = args.output_root.resolve()
    reference = root / 'local_initialization_fixture'
    if reference.exists():
        raise ValueError('Use a new validation directory; do not overwrite provenance')
    job = common.make_job('fedavg-lora-la', args.reference_run, REPO / 'DATA', root, args.num_workers, True, 'longtail')
    meta = common.read_json(args.reference_run / 'bridge_metadata.json')
    cfg = runtime.build_config(job, meta)
    model = build_cliplora_model(cfg, meta['classnames'])
    state = model.state_dict()
    keys = sorted(k for k in state if k.endswith(('_lora_A', '_lora_B')))
    frozen = state_hash(state, sorted(set(state) - set(keys)))
    if frozen != meta['frozen_model_sha256']:
        raise ValueError('Frozen pretrained model differs; this fixture cannot substitute its pretrained backbone')
    local_sha = state_hash(state, keys)
    receipt = dict(purpose='GPU execution validation only; never a paper result',
        reference_initial_lora_sha256=meta['initial_lora_sha256'], local_initial_lora_sha256=local_sha,
        initialization_matches_server=local_sha == meta['initial_lora_sha256'], frozen_model_sha256=frozen,
        torch=str(torch.__version__), cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(0),
        original_reference=str(args.reference_run.resolve()), original_protocol=job['protocol'])
    for name in common.REFERENCE_FILES:
        target = reference / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(args.reference_run / name, target)
    meta['initial_lora_sha256'] = local_sha
    meta['local_gpu_validation_only'] = True
    common.write_json(reference / 'bridge_metadata.json', meta)
    common.write_json(root / 'validation_provenance.json', receipt)
    del model, state
    command = [sys.executable, str(REPO / 'scripts/run_longtail_benchmarks.py'), '--stage', 'smoke',
        '--reference-run', str(reference), '--output-root', str(root / 'suite'),
        '--data-root', str(REPO / 'DATA'), '--num-workers', str(args.num_workers)]
    subprocess.run(command, cwd=REPO, check=True)
    from tools.benchmarks.collect import collect
    collect(root / 'suite/smoke')
    print('Real-data CUDA smoke complete; provenance:', root / 'validation_provenance.json')


if __name__ == '__main__':
    main()
