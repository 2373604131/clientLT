#!/usr/bin/env python
"""Run Experiments A/B/C for the frozen seed-42 carrier-access audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.carrier_access_audit.protocol import write_protocol  # noqa: E402


CLIP_SHA256 = "5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _call(parts: list[str]) -> None:
    print(json.dumps({"command": parts}))
    subprocess.run([sys.executable, *parts], cwd=REPO_ROOT, check=True)


def _runtime_complete(path: Path, stage: str, count_key: str, expected: int) -> bool:
    contract = path / "runtime_contract.json"
    if not contract.is_file():
        return False
    value = json.loads(contract.read_text(encoding="utf-8"))
    return value.get("stage") == stage and int(value.get(count_key, -1)) == expected


def _preflight(args, needs_cuda: bool) -> None:
    needs_data = args.stage in {"all", "manifests", "b", "c"}
    needs_similarity = args.stage in {"all", "manifests"}
    if needs_data:
        for name in ("train", "test", "meta"):
            if not (args.data_dir / name).is_file():
                raise FileNotFoundError(f"CIFAR-100 file is missing: {args.data_dir / name}")
    if needs_similarity and not args.similarity_file.is_file():
        raise FileNotFoundError(f"Frozen CLIP similarity matrix is missing: {args.similarity_file}")
    if not needs_cuda:
        return
    try:
        import pandas  # noqa: F401
        import torch
        import torchvision  # noqa: F401
        import yacs  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(f"Active environment lacks a runtime dependency: {exc.name}") from exc
    if not torch.cuda.is_available():
        raise RuntimeError("Experiments B/C require a CUDA compute node")
    clip_path = Path.home() / ".cache" / "clip" / "ViT-B-16.pt"
    if not clip_path.is_file() or _sha256(clip_path) != CLIP_SHA256:
        raise RuntimeError(f"Verified CLIP checkpoint is missing or invalid: {clip_path}")
    if not args.theta0_file.is_file():
        raise FileNotFoundError(f"Shared theta0 is missing: {args.theta0_file}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=["all", "protocol", "manifests", "a", "b", "summarize-b", "c", "summarize-c"],
        default="all",
    )
    parser.add_argument("--data-dir", type=Path, default=Path("DATA/cifar-100/cifar-100-python"))
    parser.add_argument("--similarity-file", type=Path, default=Path("output/p0_v1_context_colocation_v2/clip_similarity.npy"))
    parser.add_argument("--e2a-dir", type=Path, default=Path("output/e2_client_update_audit/e2a_local_footprint"))
    parser.add_argument("--theta0-file", type=Path, default=Path("output/e1_strength_breadth/protocol_v2/theta0_seed42.pt"))
    parser.add_argument("--output-root", type=Path, default=Path("output/carrier_access_audit"))
    parser.add_argument("--eval-batch-size", type=int, default=64)
    args = parser.parse_args()
    needs_cuda = args.stage in {"all", "b", "c"}
    _preflight(args, needs_cuda)

    protocol_dir = args.output_root / "protocol"
    manifest_dir = args.output_root / "manifests"
    a_dir = args.output_root / "experiment_a"
    b_dir = args.output_root / "experiment_b"
    b_analysis = args.output_root / "analysis_b"
    c_dir = args.output_root / "experiment_c"
    c_analysis = args.output_root / "analysis_c"

    if args.stage in {"all", "protocol"}:
        path = write_protocol(protocol_dir)
        print(json.dumps({"stage": "protocol", "path": str(path.resolve())}))
    if args.stage in {"all", "manifests"}:
        _call([
            "-m", "tools.carrier_access_audit.manifests",
            "--data-dir", str(args.data_dir),
            "--similarity-file", str(args.similarity_file),
            "--output-dir", str(manifest_dir),
        ])
    if args.stage in {"all", "a"}:
        _call([
            "-m", "tools.carrier_access_audit.footprint",
            "--e2a-dir", str(args.e2a_dir), "--output-dir", str(a_dir),
        ])
    if args.stage in {"all", "b"}:
        if _runtime_complete(b_dir, "B", "completed_transfer_pairs", 1600):
            print(json.dumps({"stage": "B", "status": "already_complete"}))
        else:
            _call([
                "-m", "tools.carrier_access_audit.runtime", "--stage", "b",
                "--data-dir", str(args.data_dir), "--manifest-dir", str(manifest_dir),
                "--output-dir", str(b_dir), "--theta0-file", str(args.theta0_file),
                "--eval-batch-size", str(args.eval_batch_size), "--device", "cuda",
            ])
    if args.stage in {"all", "summarize-b"}:
        _call([
            "-m", "tools.carrier_access_audit.summarize", "--stage", "b",
            "--input-dir", str(b_dir), "--output-dir", str(b_analysis),
        ])
    if args.stage in {"all", "c"}:
        if _runtime_complete(c_dir, "C", "completed_condition_rows", 100):
            print(json.dumps({"stage": "C", "status": "already_complete"}))
        else:
            _call([
                "-m", "tools.carrier_access_audit.runtime", "--stage", "c",
                "--data-dir", str(args.data_dir), "--manifest-dir", str(manifest_dir),
                "--output-dir", str(c_dir), "--b-dir", str(b_dir),
                "--theta0-file", str(args.theta0_file),
                "--eval-batch-size", str(args.eval_batch_size), "--device", "cuda",
            ])
    if args.stage in {"all", "summarize-c"}:
        _call([
            "-m", "tools.carrier_access_audit.summarize", "--stage", "c",
            "--input-dir", str(c_dir), "--output-dir", str(c_analysis),
        ])


if __name__ == "__main__":
    main()
