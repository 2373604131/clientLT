#!/usr/bin/env python
"""Run the first fixed-A, private-B functional selective-sync experiment."""

from __future__ import annotations

import argparse
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
PARTITIONS = {
    "clientlt": "client-longtail",
    "dirichlet": "noniid-labeldir-fine",
}


def _number(value) -> str:
    return (f"{float(value):g}").replace("-", "m").replace(".", "p")


def experiment_name(args) -> str:
    if args.run_name:
        return re.sub(r"[^A-Za-z0-9_.-]+", "_", args.run_name)
    loss = "ordinary" if args.local_loss == "ordinary_ce" else "reweighted"
    return (
        f"{loss}_g{_number(args.receive_ratio)}_L{args.top_l}_"
        f"mu{_number(args.slack_mu)}_k{_number(args.success_kappa)}"
    )


def run_dir(output_root: Path, condition: str, seed: int, name: str) -> Path:
    return output_root / f"seed{seed}" / condition / name


def build_command(args, condition: str) -> list[str]:
    output_dir = run_dir(args.output_root, condition, args.seed, experiment_name(args))
    schedule_file = args.client_schedule_file or (
        args.output_root / f"shared_full_schedule_seed{args.seed}.json"
    )
    return [
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
        "--round", str(args.rounds),
        "--local_epochs", "3",
        "--client_schedule_seed", str(args.seed),
        "--client_schedule_file", str(schedule_file),
        "--partition", PARTITIONS[condition],
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
        "--cliplora_dropout_rate", "0",
        "--cliplora_params", "q", "v",
        "--cliplora_lr_policy", "constant",
        "--cliplora_precision", "amp",
        "--cliplora_common_init_seed", str(args.seed),
        "--cliplora_freeze_a", "True",
        "--cliplora_aggregation", "fedavg",
        "--selective_sync_enable", "True",
        "--selective_sync_receive_ratio", str(args.receive_ratio),
        "--selective_sync_top_l", str(args.top_l),
        "--selective_sync_harm_epsilon", str(args.harm_epsilon),
        "--selective_sync_temperature", str(args.temperature),
        "--selective_sync_slack_mu", str(args.slack_mu),
        "--selective_sync_success_kappa", str(args.success_kappa),
        "--selective_sync_backtrack_factor", str(args.backtrack_factor),
        "--selective_sync_max_backtracks", str(args.max_backtracks),
        "--selective_sync_memory_size", str(args.memory_size),
        "--selective_sync_local_loss", args.local_loss,
        "--selective_sync_memory_lambda", str(args.memory_lambda),
        "--selective_sync_memory_start_epoch", str(args.memory_start_epoch),
        "--selective_sync_capacity_rounds", args.capacity_rounds,
        "--isolate_local_optimizer_state", "True",
        "--federated_single_scheduler_step", "True",
        "--pfrf_enable", "False",
        "--cliplora_sca_enable", "False",
        "--experimentD_enable", "False",
        "--e1_enable", "False",
        "--stage3_enable", "False",
        "DATALOADER.NUM_WORKERS", str(args.num_workers),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--condition", choices=["clientlt", "dirichlet", "all"], default="clientlt")
    parser.add_argument("--data-root", type=Path, default=Path("DATA"))
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("output/cifar100_LT/ClipLora_SelectiveSyncV1_seed42"),
    )
    parser.add_argument("--client-schedule-file", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rounds", type=int, default=100)
    parser.add_argument("--run-name", default="")
    parser.add_argument("--receive-ratio", type=float, default=1.0)
    parser.add_argument("--top-l", type=int, default=5)
    parser.add_argument("--harm-epsilon", type=float, default=1e-4)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--slack-mu", type=float, default=100.0)
    parser.add_argument("--success-kappa", type=float, default=0.5)
    parser.add_argument("--backtrack-factor", type=float, default=0.5)
    parser.add_argument("--max-backtracks", type=int, default=8)
    parser.add_argument("--memory-size", type=int, default=32)
    parser.add_argument(
        "--local-loss",
        choices=["ordinary_ce", "class_reweighted_memory_ce"],
        default="ordinary_ce",
    )
    parser.add_argument("--memory-lambda", type=float, default=1.0)
    parser.add_argument("--memory-start-epoch", type=int, default=2)
    parser.add_argument(
        "--capacity-rounds",
        default="1,20,50,100",
        help="one-based rounds used to measure fixed-A gradient-subspace capacity",
    )
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--python-bin", default=sys.executable)
    args = parser.parse_args()

    conditions = list(PARTITIONS) if args.condition == "all" else [args.condition]
    for condition in conditions:
        output_dir = run_dir(
            args.output_root, condition, args.seed, experiment_name(args)
        )
        if output_dir.exists() and any(output_dir.iterdir()):
            raise FileExistsError(f"Run directory is not empty: {output_dir}")
        command = build_command(args, condition)
        command_text = subprocess.list2cmdline(command) if os.name == "nt" else shlex.join(command)
        print(f"[{condition}] {command_text}", flush=True)
        subprocess.run(command, cwd=REPO_ROOT, check=True)


if __name__ == "__main__":
    main()
