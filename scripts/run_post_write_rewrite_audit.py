#!/usr/bin/env python
"""Run the frozen D1 post-write and D2 cumulative rewrite audits."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.carrier_access_audit.rewrite_protocol import write_rewrite_protocol  # noqa: E402


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


def _runtime_complete(path: Path, stage: str, fields: dict[str, int]) -> bool:
    contract_path = path / "runtime_contract.json"
    if not contract_path.is_file():
        return False
    value = json.loads(contract_path.read_text(encoding="utf-8"))
    return value.get("stage") == stage and all(int(value.get(key, -1)) == expected for key, expected in fields.items())


def _preflight(args) -> None:
    runtime_stage = args.stage in {"all", "d1", "d2"}
    if not runtime_stage:
        return
    missing = []
    for module in ("pandas", "torch", "torchvision", "yacs"):
        try:
            importlib.import_module(module)
        except ImportError:
            missing.append(module)
    if missing:
        raise RuntimeError(f"Active environment lacks D1/D2 dependencies: {missing}")
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("D1/D2 require a CUDA compute node")
    for name in ("train", "test", "meta"):
        if not (args.data_dir / name).is_file():
            raise FileNotFoundError(f"CIFAR-100 file is missing: {args.data_dir / name}")
    required = (
        args.manifest_dir / "manifest_contract.json",
        args.manifest_dir / "training_execution.csv",
        args.b_dir / "runtime_contract.json",
        args.b_dir / "transfer_matrix.csv",
        args.theta0_file,
    )
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(f"Required completed carrier-access artifact is missing: {path}")
    missing_states = [
        args.b_dir / "candidate_states" / f"candidate_{candidate:02d}.pt"
        for candidate in range(80)
        if not (args.b_dir / "candidate_states" / f"candidate_{candidate:02d}.pt").is_file()
    ]
    if missing_states:
        raise FileNotFoundError(
            f"Experiment B candidate states are incomplete: {len(missing_states)} missing; first={missing_states[0]}"
        )
    clip_path = Path.home() / ".cache" / "clip" / "ViT-B-16.pt"
    if not clip_path.is_file() or _sha256(clip_path) != CLIP_SHA256:
        raise RuntimeError(f"Verified CLIP checkpoint is missing or invalid: {clip_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=["all", "protocol", "d1", "summarize-d1", "d2", "summarize-d2"],
        default="all",
    )
    parser.add_argument("--data-dir", type=Path, default=Path("DATA/cifar-100/cifar-100-python"))
    parser.add_argument("--manifest-dir", type=Path, default=Path("output/carrier_access_audit/manifests"))
    parser.add_argument("--b-dir", type=Path, default=Path("output/carrier_access_audit/experiment_b"))
    parser.add_argument("--theta0-file", type=Path, default=Path("output/e1_strength_breadth/protocol_v2/theta0_seed42.pt"))
    parser.add_argument("--output-root", type=Path, default=Path("output/post_write_rewrite_audit"))
    parser.add_argument("--eval-batch-size", type=int, default=64)
    args = parser.parse_args()
    _preflight(args)

    protocol_dir = args.output_root / "protocol"
    d1_dir = args.output_root / "d1"
    d1_analysis = args.output_root / "analysis_d1"
    d1_summary = d1_analysis / "d1_summary.json"
    d2_dir = args.output_root / "d2"
    d2_analysis = args.output_root / "analysis_d2"

    if args.stage in {"all", "protocol"}:
        path = write_rewrite_protocol(protocol_dir)
        print(json.dumps({"stage": "protocol", "path": str(path.resolve())}))
    if args.stage in {"all", "d1"}:
        if _runtime_complete(d1_dir, "D1", {
            "completed_writer_classes": 20,
            "completed_pre_pairs": 1600,
            "completed_post_pairs": 1600,
        }):
            print(json.dumps({"stage": "D1", "status": "already_complete"}))
        else:
            _call([
                "-m", "tools.carrier_access_audit.rewrite_runtime", "--stage", "d1",
                "--data-dir", str(args.data_dir), "--manifest-dir", str(args.manifest_dir),
                "--b-dir", str(args.b_dir), "--output-dir", str(d1_dir),
                "--theta0-file", str(args.theta0_file),
                "--eval-batch-size", str(args.eval_batch_size), "--device", "cuda",
            ])
    if args.stage in {"all", "summarize-d1"}:
        _call([
            "-m", "tools.carrier_access_audit.rewrite_summarize", "--stage", "d1",
            "--input-dir", str(d1_dir), "--output-dir", str(d1_analysis),
        ])
    if args.stage == "all":
        summary = json.loads(d1_summary.read_text(encoding="utf-8"))
        if not summary.get("valid_comparison", False):
            print(json.dumps({
                "stage": "D2", "status": "skipped",
                "reason": "D1 has fewer than 12 positive tail writers",
            }))
            return
    if args.stage in {"all", "d2"}:
        expected_valid_tails = len(json.loads(d1_summary.read_text(encoding="utf-8"))["valid_tail_classes"])
        expected_rows = expected_valid_tails * 3 * 7
        if _runtime_complete(d2_dir, "D2", {"completed_replay_rows": expected_rows}):
            print(json.dumps({"stage": "D2", "status": "already_complete"}))
        else:
            _call([
                "-m", "tools.carrier_access_audit.rewrite_runtime", "--stage", "d2",
                "--data-dir", str(args.data_dir), "--manifest-dir", str(args.manifest_dir),
                "--b-dir", str(args.b_dir), "--d1-dir", str(d1_dir),
                "--d1-summary", str(d1_summary), "--output-dir", str(d2_dir),
                "--theta0-file", str(args.theta0_file),
                "--eval-batch-size", str(args.eval_batch_size), "--device", "cuda",
            ])
    if args.stage in {"all", "summarize-d2"}:
        _call([
            "-m", "tools.carrier_access_audit.rewrite_summarize", "--stage", "d2",
            "--input-dir", str(d2_dir), "--d1-summary", str(d1_summary),
            "--output-dir", str(d2_analysis),
        ])


if __name__ == "__main__":
    main()
