#!/usr/bin/env python
"""Run the minimal Client-LT/Dirichlet ClipLoRA LA-aggregation pair."""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
PARTITIONS = {
    "clientlt": "client-longtail",
    "dirichlet": "noniid-labeldir-fine",
}


def run_dir(output_root: Path, condition: str, seed: int) -> Path:
    return output_root / f"seed{seed}" / condition / "la_aggregation"


def build_command(args, condition: str) -> list[str]:
    partition = PARTITIONS[condition]
    output_dir = run_dir(args.output_root, condition, args.seed)
    schedule_file = args.client_schedule_file
    if schedule_file is None:
        schedule_file = args.output_root / f"shared_full_schedule_seed{args.seed}.json"

    command = [
        args.python_bin,
        "-u",
        "federated_main.py",
        "--root", str(args.data_root),
        "--model", "fedavg",
        "--trainer", "ClipLora",
        "--dataset", "cifar100_LT",
        "--dataset-config-file", "configs/datasets/cifar100_LT.yaml",
        "--config-file", "configs/trainers/PromptFL/vit_b16.yaml",
        "--output-dir", str(output_dir),
        "--seed", str(args.seed),
        "--split_seed", str(args.seed),
        "--num_users", "30",
        "--frac", "1.0",
        "--round", "100",
        "--local_epochs", "3",
        "--client_schedule_seed", str(args.seed),
        "--client_schedule_file", str(schedule_file),
        "--partition", partition,
        "--beta", "0.5",
        "--imb_factor", "0.01",
        "--imb_type", "exp",
        "--specialization_lambda", "0.75",
        "--intra_group_alpha", "0.5",
        "--head_leakage_scale", "3.0",
        "--head_client_ratio", "0.9",
        "--tail_client_ratio", "0.1",
        "--head_class_ratio", "0.8",
        "--tail_class_ratio", "0.2",
        "--train_batch_size", "32",
        "--test_batch_size", "64",
        "--global_eval_interval", "1",
        "--lr", "0.001",
        "--gamma", "1",
        "--n_ctx", "4",
        "--n_general", "1",
        "--ctx_init", "False",
        "--csc", "True",
        "--encoder", "vision",
        "--cliplora_position", "top3",
        "--cliplora_rank", "2",
        "--cliplora_alpha", "1",
        "--cliplora_dropout_rate", "0.0",
        "--cliplora_params", "q", "v",
        "--cliplora_lr_policy", "constant",
        "--cliplora_precision", "amp",
        "--cliplora_aggregation", "la_aggregation",
        "--cliplora_la_temperature", str(args.la_temperature),
        "--cliplora_la_regularization", str(args.la_regularization),
        "--cliplora_la_max_iter", str(args.la_max_iter),
        "--isolate_local_optimizer_state", "True",
        "--federated_single_scheduler_step", "True",
        "--experimentD_enable", "False",
        "DATALOADER.NUM_WORKERS", str(args.num_workers),
    ]
    return command


def command_text(command: list[str]) -> str:
    return subprocess.list2cmdline(command) if os.name == "nt" else shlex.join(command)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--condition",
        choices=["clientlt", "dirichlet", "all"],
        default="all",
        help="run one topology or both sequentially",
    )
    parser.add_argument("--data-root", type=Path, default=Path("DATA"))
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("output/cifar100_LT/ClipLora_LA_Aggregation_seed42"),
    )
    parser.add_argument("--client-schedule-file", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--la-temperature", type=float, default=1.0)
    parser.add_argument("--la-regularization", type=float, default=1.0)
    parser.add_argument("--la-max-iter", type=int, default=100)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--python-bin", default=sys.executable)
    args = parser.parse_args()

    if not 0.0 <= args.la_temperature <= 1.0:
        raise ValueError("--la-temperature must be in [0, 1]")
    if args.la_regularization < 0.0:
        raise ValueError("--la-regularization must be non-negative")
    if args.la_max_iter < 1:
        raise ValueError("--la-max-iter must be positive")
    if args.client_schedule_file is not None and not args.client_schedule_file.is_file():
        raise FileNotFoundError(
            f"Client schedule does not exist: {args.client_schedule_file}"
        )
    conditions = list(PARTITIONS) if args.condition == "all" else [args.condition]
    for condition in conditions:
        output_dir = run_dir(args.output_root, condition, args.seed)
        if output_dir.exists() and any(output_dir.iterdir()):
            raise FileExistsError(
                f"Refusing to append to non-empty run directory: {output_dir}"
            )
        command = build_command(args, condition)
        print(f"\n[{condition}]\n{command_text(command)}", flush=True)
        subprocess.run(command, cwd=REPO_ROOT, check=True)


if __name__ == "__main__":
    main()
