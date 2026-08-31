#!/usr/bin/env python
"""Foreground launcher for the first deployable online SCA experiment.

The baseline and SCA runs share the same seed-42 partial-participation client
schedule.  Partial participation is intentional: it matches CAPT's practical
setting and makes D4-A supporter-absence trajectories observable.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def read_frozen_lora(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing frozen G0 LoRA configuration: {path}. "
            "Pass --freeze-file or finish G0 first."
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    config = payload.get("selected_config")
    if payload.get("verdict") != "PASS" or not config:
        raise RuntimeError(f"Frozen LoRA configuration did not pass G0: {path}")
    expected = {"position", "rank", "alpha", "params"}
    if not expected.issubset(config):
        raise ValueError(f"Malformed frozen LoRA configuration: {config}")
    return config


def common_command(args, output_dir: Path, config: dict) -> list[str]:
    schedule = args.output_root / "schedules" / (
        f"frac{str(args.frac).replace('.', 'p')}_users30_rounds{args.rounds}_seed42.json"
    )
    return [
        args.python_bin, "-u", "federated_main.py",
        "--root", str(args.data_root),
        "--model", "fedavg",
        "--trainer", "ClipLora",
        "--dataset", "cifar100_LT",
        "--seed", "42",
        "--split_seed", "42",
        "--num_users", "30",
        "--frac", str(args.frac),
        "--round", str(args.rounds),
        "--local_epochs", "3",
        "--client_schedule_seed", "42",
        "--client_schedule_file", str(schedule),
        "--isolate_local_optimizer_state", "True",
        "--federated_single_scheduler_step", "True",
        "--lr", str(args.lr),
        "--gamma", "1",
        "--n_ctx", "4",
        "--n_general", "1",
        "--ctx_init", "False",
        "--csc", "True",
        "--dataset-config-file", "configs/datasets/cifar100_LT.yaml",
        "--config-file", "configs/trainers/PromptFL/vit_b16.yaml",
        "--output-dir", str(output_dir),
        "--imb_factor", "0.01",
        "--imb_type", "exp",
        "--train_batch_size", "32",
        "--test_batch_size", str(args.test_batch_size),
        "--global_eval_interval", str(args.eval_interval),
        "--num_classes", "100",
        "--tail_class_ratio", "0.2",
        "--head_client_ratio", "0.9",
        "--tail_client_ratio", "0.1",
        "--head_class_ratio", "0.8",
        "--partition", "client-longtail",
        "--beta", "0.5",
        "--specialization_lambda", "0.75",
        "--intra_group_alpha", "0.5",
        "--head_leakage_scale", "3.0",
        "--encoder", "vision",
        "--cliplora_position", str(config["position"]),
        "--cliplora_rank", str(config["rank"]),
        "--cliplora_alpha", str(config["alpha"]),
        "--cliplora_params", *[str(value) for value in config["params"]],
        "--cliplora_dropout_rate", "0.0",
        "--cliplora_lr_policy", "constant",
        "--cliplora_precision", "fp32",
        "--cliplora_aggregation", "fedavg",
        "--experimentD_enable", "False",
        "DATALOADER.NUM_WORKERS", str(args.num_workers),
    ]


def build_command(args, condition: str, config: dict) -> tuple[Path, list[str]]:
    output_dir = args.output_root / condition
    command = common_command(args, output_dir, config)
    enabled = condition == "online_sca"
    command[command.index("DATALOADER.NUM_WORKERS"):command.index("DATALOADER.NUM_WORKERS")] = [
        "--cliplora_sca_enable", str(enabled),
        "--cliplora_sca_d4_enable", str(enabled),
        "--cliplora_sca_scale", str(args.sca_scale),
        "--cliplora_sca_clamp", str(args.sca_clamp),
        "--cliplora_sca_lr_mult", str(args.sca_lr_mult),
        "--cliplora_sca_use_bias", "False",
        "--cliplora_sca_support_min_fraction", str(args.support_min_fraction),
        "--cliplora_sca_weighting", args.support_weighting,
    ]
    return output_dir, command


def run(command: list[str], gpu: str, dry_run: bool) -> None:
    print("\n" + "=" * 78, flush=True)
    print(shlex.join(command), flush=True)
    print("=" * 78, flush=True)
    if dry_run:
        return
    env = os.environ.copy()
    # Treat --gpu as a logical slot inside the Slurm allocation.  If Slurm
    # exposes physical ids/UUIDs such as "2,3" or "GPU-...,GPU-...", preserve
    # that mapping instead of accidentally escaping the allocated devices.
    visible = [value.strip() for value in env.get("CUDA_VISIBLE_DEVICES", "").split(",") if value.strip()]
    requested = str(gpu)
    if visible and requested.isdigit() and int(requested) < len(visible):
        env["CUDA_VISIBLE_DEVICES"] = visible[int(requested)]
    else:
        env["CUDA_VISIBLE_DEVICES"] = requested
    subprocess.run(command, cwd=REPO_ROOT, env=env, check=True)


def last_metrics(path: Path) -> dict:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise RuntimeError(f"No metrics in {path}")
    row = dict(rows[-1])
    head = float(row["non_tail_acc"])
    tail = float(row["bottom20_tail_acc"])
    row["balanced_acc"] = float(row["macro_per_class_acc"])
    row["head_tail_h_mean"] = 2.0 * head * tail / (head + tail) if head + tail > 0 else 0.0
    return row


def summarize(args) -> None:
    rows = {}
    for condition in ("fedavg", "online_sca"):
        path = args.output_root / condition / "round_metrics.csv"
        if path.exists():
            rows[condition] = last_metrics(path)
    payload = {
        "schema_version": "online_sca_seed42_v1",
        "seed": 42,
        "frac": args.frac,
        "rounds": args.rounds,
        "conditions_found": sorted(rows),
        "final_metrics": rows,
        "d4a_path": str(args.output_root / "online_sca" / "d4a" / "d4a_per_class_round.csv"),
        "method_ready": False,
        "decision_note": (
            "Seed 42 is a discovery run. Compare the matched FedAvg and online SCA "
            "trajectory before freezing a fresh-seed confirmation."
        ),
    }
    if {"fedavg", "online_sca"}.issubset(rows):
        for metric in (
            "overall_acc",
            "non_tail_acc",
            "bottom20_tail_acc",
            "macro_per_class_acc",
            "head_tail_h_mean",
        ):
            payload[f"final_delta_{metric}"] = (
                float(rows["online_sca"][metric]) - float(rows["fedavg"][metric])
            )
    path = args.output_root / "online_sca_summary.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["baseline", "sca", "both", "summary"], default="sca")
    parser.add_argument("--output-root", type=Path, default=Path("output/online_sca_seed42"))
    parser.add_argument("--data-root", type=Path, default=Path("DATA"))
    parser.add_argument("--freeze-file", type=Path, default=Path("output/g0_d1_seed42/lora_freeze.json"))
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--rounds", type=int, default=80)
    parser.add_argument("--frac", type=float, default=0.4)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--test-batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--eval-interval", type=int, default=1)
    parser.add_argument("--sca-scale", type=float, default=10.0)
    parser.add_argument("--sca-clamp", type=float, default=3.0)
    parser.add_argument("--sca-lr-mult", type=float, default=5.0)
    parser.add_argument("--support-min-fraction", type=float, default=0.0)
    parser.add_argument("--support-weighting", choices=["class_count", "uniform"], default="class_count")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    args.output_root = args.output_root.resolve()
    args.data_root = args.data_root.resolve()
    args.freeze_file = args.freeze_file.resolve()
    if args.stage == "summary":
        summarize(args)
        return
    config = read_frozen_lora(args.freeze_file)
    conditions = {
        "baseline": ["fedavg"],
        "sca": ["online_sca"],
        "both": ["fedavg", "online_sca"],
    }[args.stage]
    for condition in conditions:
        output_dir, command = build_command(args, condition, config)
        if output_dir.exists() and any(output_dir.iterdir()) and not args.dry_run:
            raise FileExistsError(
                f"Refusing to mix a new run into non-empty directory: {output_dir}"
            )
        run(command, args.gpu, args.dry_run)
    if not args.dry_run:
        summarize(args)


if __name__ == "__main__":
    main()
