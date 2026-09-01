#!/usr/bin/env python
"""Run the single seed-42 Client-LT/SCA Stage 2-C diagnostic trajectory."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

try:
    from scripts.run_online_sca import common_command, read_frozen_lora, run
except ImportError:  # Direct execution puts scripts/ rather than the repo on sys.path.
    from run_online_sca import common_command, read_frozen_lora, run


def build_stage2c_command(args, config):
    output_dir = args.output_root / "stage2c_temporal_clientlt"
    command = common_command(
        args,
        output_dir,
        config,
        partition="client-longtail",
    )
    insertion = command.index("DATALOADER.NUM_WORKERS")
    command[insertion:insertion] = [
        "--cliplora_sca_enable", "True",
        "--cliplora_residual_aggregation", "class_separable",
        "--cliplora_sca_d4_enable", "True",
        "--cliplora_sca_scale", str(args.sca_scale),
        "--cliplora_sca_clamp", str(args.sca_clamp),
        "--cliplora_sca_lr_mult", str(args.sca_lr_mult),
        "--cliplora_sca_use_bias", "False",
        "--cliplora_sca_support_min_fraction", str(args.support_min_fraction),
        "--cliplora_sca_weighting", args.support_weighting,
        "--cliplora_stage2c_enable", "True",
        "--cliplora_stage2c_rounds", args.checkpoint_rounds,
        "--cliplora_stage2c_substantive_drop", str(args.substantive_drop),
    ]
    return output_dir, command


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("output/online_sca_seed42_v2"),
    )
    parser.add_argument("--data-root", type=Path, default=Path("DATA"))
    parser.add_argument(
        "--freeze-file",
        type=Path,
        default=Path("output/g0_d1_seed42/lora_freeze.json"),
    )
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
    parser.add_argument(
        "--support-weighting", choices=["class_count", "uniform"], default="class_count"
    )
    parser.add_argument("--matched-beta", type=float, default=0.5)
    parser.add_argument("--checkpoint-rounds", default="3,20,50,80")
    parser.add_argument(
        "--substantive-drop",
        type=float,
        default=2.0,
        help="descriptive post-training H-mean threshold; never controls training",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    args.output_root = args.output_root.resolve()
    args.data_root = args.data_root.resolve()
    args.freeze_file = args.freeze_file.resolve()
    if args.rounds != 80 or abs(args.frac - 0.4) > 1e-12 or args.eval_interval != 1:
        raise ValueError(
            "The formal Stage 2-C rerun is frozen to 80 rounds, frac=0.4, eval_interval=1"
        )
    if args.checkpoint_rounds.replace(" ", "") != "3,20,50,80":
        raise ValueError("The formal Stage 2-C checkpoint rounds are frozen to 3,20,50,80")
    schedule = args.output_root / "schedules" / "frac0p4_users30_rounds80_seed42.json"
    if not schedule.exists() and not args.dry_run:
        raise FileNotFoundError(
            "The original frozen client schedule is missing: {}".format(schedule)
        )
    config = read_frozen_lora(args.freeze_file)
    output_dir, command = build_stage2c_command(args, config)
    summary_path = output_dir / "stage2c" / "stage2c_summary.json"
    if summary_path.exists() and not args.dry_run:
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        raise FileExistsError(
            "Stage 2-C is already complete at {} (route={}). Refusing to overwrite it.".format(
                summary_path,
                payload.get("route", {}).get("recommended_route", "unknown"),
            )
        )
    if output_dir.exists() and any(output_dir.iterdir()) and not args.dry_run:
        raise FileExistsError(
            "The Stage 2-C output directory is not empty: {}. Use a new --output-root "
            "or archive the partial run before retrying.".format(output_dir)
        )
    run(command, args.gpu, args.dry_run)
    if not args.dry_run and not summary_path.exists():
        raise RuntimeError("Training ended without the Stage 2-C summary: {}".format(summary_path))
    if not args.dry_run:
        print(summary_path.read_text(encoding="utf-8"), flush=True)


if __name__ == "__main__":
    main()
