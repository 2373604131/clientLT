#!/usr/bin/env python
"""Run the formal E2 client-local audit on one CUDA compute node."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from importlib.util import find_spec
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

CLIP_SHA256 = "5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _preflight(args) -> None:
    missing = [name for name in ("yacs", "torch", "torchvision", "pandas") if find_spec(name) is None]
    if missing:
        raise RuntimeError(f"E2 dependencies are missing from the active environment: {missing}")
    for name in ("train", "test", "meta"):
        path = args.data_dir / name
        if not path.is_file():
            raise FileNotFoundError(f"CIFAR-100 file is missing: {path}")
    clip_path = Path.home() / ".cache" / "clip" / "ViT-B-16.pt"
    if not clip_path.is_file():
        raise FileNotFoundError(f"CLIP checkpoint is missing: {clip_path}")
    observed = _sha256(clip_path)
    if observed != CLIP_SHA256:
        raise RuntimeError(f"CLIP checkpoint hash mismatch: {observed} != {CLIP_SHA256}")


def _call(parts: list[str]) -> None:
    print(json.dumps({"command": parts}))
    subprocess.run([sys.executable, *parts], cwd=REPO_ROOT, check=True)


def _runtime_complete(path: Path, stage: str) -> bool:
    contract = path / "runtime_contract.json"
    if not contract.is_file():
        return False
    value = json.loads(contract.read_text(encoding="utf-8"))
    return value.get("stage") == stage and value.get("server_aggregation_called") is False


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["all", "manifests", "e2a", "e2b", "summarize"], default="all")
    parser.add_argument("--data-dir", type=Path, default=Path("DATA/cifar-100/cifar-100-python"))
    parser.add_argument("--output-root", type=Path, default=Path("output/e2_client_update_audit"))
    parser.add_argument("--theta0-file", type=Path, default=Path("output/e1_strength_breadth/protocol_v2/theta0_seed42.pt"))
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--bootstrap-draws", type=int, default=10000)
    args = parser.parse_args()
    _preflight(args)

    protocol_dir = args.output_root / "protocol"
    manifest_dir = args.output_root / "manifests"
    e2a_dir = args.output_root / "e2a_local_footprint"
    e2b_dir = args.output_root / "e2b_access_intervention"
    analysis_dir = args.output_root / "analysis"

    if args.stage in ("all", "manifests"):
        _call(["-m", "tools.client_update_audit.protocol", "--output-dir", str(protocol_dir)])
        if (manifest_dir / "manifest_contract.json").exists():
            existing = json.loads((manifest_dir / "manifest_contract.json").read_text(encoding="utf-8"))
            if existing.get("data_seeds") != [int(seed) for seed in args.seeds]:
                raise RuntimeError(
                    "Existing E2 manifests use different seeds: "
                    f"{existing.get('data_seeds')} != {args.seeds}. Choose a new --output-root."
                )
            print(json.dumps({"stage": "manifests", "status": "already_exists", "path": str(manifest_dir)}))
        else:
            _call([
                "-m", "tools.client_update_audit.manifests",
                "--data-dir", str(args.data_dir), "--output-dir", str(manifest_dir),
                "--seeds", *[str(seed) for seed in args.seeds],
            ])
    if args.stage in ("e2a", "e2b", "summarize") and not (manifest_dir / "manifest_contract.json").is_file():
        raise FileNotFoundError("E2 manifests are missing; run --stage manifests first")

    if args.stage in ("all", "e2a"):
        if _runtime_complete(e2a_dir, "e2a"):
            print(json.dumps({"stage": "e2a", "status": "already_complete", "path": str(e2a_dir)}))
        else:
            _call([
                "-m", "tools.client_update_audit.runtime", "--stage", "e2a",
                "--data-dir", str(args.data_dir), "--manifest-dir", str(manifest_dir),
                "--output-dir", str(e2a_dir), "--theta0-file", str(args.theta0_file),
                "--eval-batch-size", str(args.eval_batch_size), "--device", "cuda",
            ])
    if args.stage in ("all", "e2b"):
        if _runtime_complete(e2b_dir, "e2b"):
            print(json.dumps({"stage": "e2b", "status": "already_complete", "path": str(e2b_dir)}))
        else:
            _call([
                "-m", "tools.client_update_audit.runtime", "--stage", "e2b",
                "--data-dir", str(args.data_dir), "--manifest-dir", str(manifest_dir),
                "--output-dir", str(e2b_dir), "--theta0-file", str(args.theta0_file),
                "--eval-batch-size", str(args.eval_batch_size), "--device", "cuda",
            ])
    if args.stage in ("all", "summarize"):
        if not _runtime_complete(e2a_dir, "e2a") or not _runtime_complete(e2b_dir, "e2b"):
            raise RuntimeError("Both E2A and E2B must complete before summarization")
        _call([
            "-m", "tools.client_update_audit.summarize",
            "--e2a-dir", str(e2a_dir), "--e2b-dir", str(e2b_dir),
            "--manifest-dir", str(manifest_dir), "--output-dir", str(analysis_dir),
            "--bootstrap-draws", str(args.bootstrap_draws),
        ])


if __name__ == "__main__":
    main()
