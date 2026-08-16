#!/usr/bin/env python
"""Run the paired, formal E1 seed-42 mechanism experiment on one compute node."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
CLIP_SHA256 = "5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f"


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _runtime_preflight() -> None:
    """Fail before creating a run directory when runtime inputs are unavailable."""
    for module in ("yacs.config", "torchvision", "tools.breadth_audit.runtime_e1"):
        try:
            importlib.import_module(module)
        except ModuleNotFoundError as error:
            raise RuntimeError(f"E1 runtime dependency is missing: {error.name}") from error
    clip_path = Path.home() / ".cache" / "clip" / "ViT-B-16.pt"
    if not clip_path.is_file():
        raise FileNotFoundError(
            f"CLIP ViT-B/16 checkpoint is missing: {clip_path}. "
            "Download it on the login node before allocating the compute run."
        )
    observed = _file_sha256(clip_path)
    if observed != CLIP_SHA256:
        raise RuntimeError(f"CLIP checkpoint hash mismatch: {observed} != {CLIP_SHA256}")


def _completed(run_dir: Path) -> bool:
    path = run_dir / "e1_round_manifest.csv"
    if not path.is_file():
        return False
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return len(rows) == 101 and [int(row["round"]) for row in rows] == list(range(101))


def _common(args, run_dir: Path) -> list[str]:
    return [
        sys.executable,
        "federated_main.py",
        "--root", str(args.data_root),
        "--model", "fedavg",
        "--trainer", "ClipLora",
        "--dataset", "cifar100_LT",
        "--seed", "42",
        "--split_seed", "42",
        "--num_users", "30",
        "--frac", "1.0",
        "--round", "100",
        "--local_epochs", "3",
        "--client_schedule_seed", "42",
        "--client_schedule_file", str(args.output_root / "shared_full_schedule_seed42.json"),
        "--isolate_local_optimizer_state", "True",
        "--federated_single_scheduler_step", "True",
        "--lr", "0.001",
        "--gamma", "1",
        "--n_ctx", "4",
        "--n_general", "1",
        "--ctx_init", "False",
        "--csc", "True",
        "--dataset-config-file", "configs/datasets/cifar100_LT.yaml",
        "--config-file", "configs/trainers/PromptFL/vit_b16.yaml",
        "--output-dir", str(run_dir),
        "--imb_factor", "0.01",
        "--imb_type", "exp",
        "--train_batch_size", "32",
        "--test_batch_size", "64",
        "--global_eval_interval", "1",
        "--num_classes", "100",
        "--tail_class_ratio", "0.2",
        "--head_client_ratio", "0.9",
        "--tail_client_ratio", "0.1",
        "--head_class_ratio", "0.8",
        "--encoder", "vision",
        "--cliplora_position", "top3",
        "--cliplora_rank", "2",
        "--cliplora_alpha", "1",
        "--cliplora_dropout_rate", "0.0",
        "--cliplora_params", "q", "v",
        "--cliplora_lr_policy", "constant",
        "--cliplora_precision", "fp32",
        "--cliplora_aggregation", "fedavg",
        "--experimentD_enable", "False",
        "--e1_enable", "True",
        "--e1_protocol_file", str(args.protocol_file),
        "--e1_dino_artifact", str(args.dino_artifact),
        "--e1_data_dir", str(args.raw_data_dir),
        "--e1_theta0_file", str(args.theta0_file),
        "--e1_model_seed", "42",
        "--e1_eval_batch_size", str(args.eval_batch_size),
    ]


def _command(args, case: str) -> tuple[list[str], Path]:
    run_dir = args.output_root / "seed42" / case
    command = _common(args, run_dir)
    if case == "dirichlet":
        command += [
            "--partition", "noniid-labeldir-fine",
            "--beta", "0.5",
            "--intra_group_alpha", "0.5",
            "--controlled_tail_min_purity", "0.8",
        ]
    elif case == "clientlt_controlled":
        command += [
            "--partition", "client-longtail-controlled",
            "--beta", "0.5",
            "--intra_group_alpha", "0.5",
            "--controlled_tail_min_purity", "0.8",
            "--specialization_lambda", "1.0",
            "--head_leakage_scale", "0.0",
        ]
    else:
        raise ValueError(case)
    command += ["DATALOADER.NUM_WORKERS", str(args.num_workers)]
    return command, run_dir


def _run_case(args, case: str) -> None:
    command, run_dir = _command(args, case)
    if _completed(run_dir):
        print(json.dumps({"case": case, "status": "already_complete", "run_dir": str(run_dir)}))
        return
    if run_dir.exists() and any(run_dir.iterdir()):
        raise RuntimeError(
            f"Incomplete non-empty E1 directory exists: {run_dir}. Preserve it for "
            "diagnosis, then choose a new --output-root or move it aside before rerunning."
        )
    run_dir.mkdir(parents=True, exist_ok=True)
    print(json.dumps({"case": case, "status": "starting", "run_dir": str(run_dir)}))
    subprocess.run(command, cwd=REPO_ROOT, check=True)
    if not _completed(run_dir):
        raise RuntimeError(f"E1 case returned without all rounds 0--100: {run_dir}")


def _summarize(args) -> None:
    dirichlet = args.output_root / "seed42" / "dirichlet"
    clientlt = args.output_root / "seed42" / "clientlt_controlled"
    if not _completed(dirichlet) or not _completed(clientlt):
        raise RuntimeError("Both paired E1 cases must complete before summarization")
    subprocess.run([
        sys.executable,
        "-m", "tools.breadth_audit.summarize_e1",
        "--dirichlet-dir", str(dirichlet),
        "--clientlt-dir", str(clientlt),
        "--output-dir", str(args.output_root / "seed42" / "analysis"),
    ], cwd=REPO_ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--case", choices=["both", "dirichlet", "clientlt_controlled", "summarize"],
        default="both",
    )
    parser.add_argument("--data-root", type=Path, default=Path("DATA"))
    parser.add_argument(
        "--raw-data-dir", type=Path,
        default=Path("DATA/cifar-100/cifar-100-python"),
    )
    parser.add_argument(
        "--output-root", type=Path,
        default=Path("output/e1_strength_breadth/formal"),
    )
    parser.add_argument(
        "--protocol-file", type=Path,
        default=Path("output/e1_strength_breadth/protocol_v2/mechanism_validation_protocol.json"),
    )
    parser.add_argument(
        "--dino-artifact", type=Path,
        default=Path("output/e1_strength_breadth/frozen_eval/dino_tail_clusters.npz"),
    )
    parser.add_argument(
        "--theta0-file", type=Path,
        default=Path("output/e1_strength_breadth/protocol_v2/theta0_seed42.pt"),
    )
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()
    for path, label in (
        (args.protocol_file, "V2 protocol"),
        (args.dino_artifact, "DINO artifact"),
        (args.raw_data_dir / "train", "CIFAR train"),
        (args.raw_data_dir / "test", "CIFAR test"),
        (args.raw_data_dir / "meta", "CIFAR meta"),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"{label} is missing: {path}")
    _runtime_preflight()
    if args.case in ("both", "dirichlet"):
        _run_case(args, "dirichlet")
    if args.case in ("both", "clientlt_controlled"):
        _run_case(args, "clientlt_controlled")
    if args.case in ("both", "summarize"):
        _summarize(args)


if __name__ == "__main__":
    main()
