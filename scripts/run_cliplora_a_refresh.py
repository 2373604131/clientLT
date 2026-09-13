"""Launch one C0--C3 controlled-A-refresh experiment on the allocated GPU."""

import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def build_command(args, method):
    output = args.output_root / f"seed{args.seed}" / method
    command = [
        sys.executable, "-u", "federated_main.py",
        "--root", str(args.data_root), "--output-dir", str(output),
        "--model", "fedavg", "--trainer", "ClipLora", "--dataset", "cifar100_LT",
        "--dataset-config-file", "configs/datasets/cifar100_LT.yaml",
        "--config-file", "configs/trainers/PromptFL/vit_b16.yaml",
        "--seed", str(args.seed), "--split_seed", str(args.seed),
        "--num_users", "30", "--frac", "1.0", "--round", "100", "--local_epochs", "3",
        "--client_schedule_seed", str(args.seed),
        "--client_schedule_file", str(args.schedule_file or Path(f"output/cifar100_LT/v2_matched/full_schedule_seed{args.seed}.json")),
        "--partition", "client-longtail", "--beta", "0.5",
        "--imb_factor", "0.01", "--imb_type", "exp",
        "--specialization_lambda", "0.75", "--intra_group_alpha", "0.5",
        "--head_leakage_scale", "3.0", "--head_client_ratio", "0.9",
        "--tail_client_ratio", "0.1", "--head_class_ratio", "0.8", "--tail_class_ratio", "0.2",
        "--train_batch_size", "32", "--test_batch_size", "64", "--global_eval_interval", "1",
        "--lr", "0.001", "--gamma", "1", "--n_ctx", "4", "--n_general", "1",
        "--ctx_init", "False", "--csc", "True", "--isolate_local_optimizer_state", "True",
        "--experimentD_enable", "False", "--encoder", "vision", "--cliplora_position", "top3",
        "--cliplora_rank", str(args.rank), "--cliplora_alpha", "1",
        "--cliplora_dropout_rate", "0", "--cliplora_params", "q", "v",
        "--cliplora_lr_policy", "constant", "--cliplora_precision", "fp32",
        "--cliplora_common_init_seed", str(args.seed), "--cliplora_freeze_a", "True",
        "--cliplora_aggregation", "fedavg", "--cliplora_v2", "off",
        "--cliplora_la_temperature", "1", "--cliplora_la_regularization", "0",
        "--cliplora_la_max_iter", "100", "--federated_single_scheduler_step", "True",
        "--selective_sync_enable", "False", "--pfrf_enable", "False",
        "--cliplora_sca_enable", "False", "--e1_enable", "False", "--stage3_enable", "False",
        "--a_refresh_variant", method, "--a_refresh_interval", str(args.refresh_interval),
        "--a_refresh_epochs", str(args.refresh_epochs), "--a_refresh_lr", str(args.refresh_lr),
    ]
    if args.resume:
        command += ["--a_refresh_resume", str(args.resume)]
    command += [
        "DATALOADER.NUM_WORKERS", str(args.num_workers),
        "OPTIM.LR", "0.001", "OPTIM.LR_SCHEDULER", "single_step",
        "OPTIM.STEPSIZE", "(3,)", "OPTIM.GAMMA", "1.0", "OPTIM.WARMUP_EPOCH", "-1",
    ]
    return command, output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=["c0", "c1", "c2", "c3", "all"], required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--data-root", type=Path, default=Path("DATA"))
    parser.add_argument("--output-root", type=Path, default=Path("output/cifar100_LT/a_refresh_pilot"))
    parser.add_argument("--schedule-file", type=Path)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--refresh-interval", type=int, default=10)
    parser.add_argument("--refresh-epochs", type=int, default=1)
    parser.add_argument("--refresh-lr", type=float, default=0.001)
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    methods = ["c0", "c1", "c2", "c3"] if args.method == "all" else [args.method]
    for method in methods:
        command, output = build_command(args, method)
        destination = REPO_ROOT / output
        destination.mkdir(parents=True, exist_ok=True)
        (destination / "command.json").write_text(json.dumps(command, indent=2), encoding="utf-8")
        print(subprocess.list2cmdline(command) if os.name == "nt" else shlex.join(command), flush=True)
        subprocess.run(command, cwd=REPO_ROOT, check=True)


if __name__ == "__main__":
    main()
