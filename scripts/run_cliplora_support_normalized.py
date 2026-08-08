#!/usr/bin/env python
"""Run the paired ClipLoRA support-normalization experiment.

The canonical design is a 2x2 comparison:

  topology:    client-longtail vs matched fine-class Dirichlet
  aggregation: sample-weighted FedAvg vs end-to-end support-normalized

All four runs share the global CIFAR-100-LT class marginal, seed, model,
optimizer lifecycle, client schedule, and LoRA configuration.  After training,
the launcher verifies the saved global-long-tail fingerprints before writing a
paired manifest.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]

CASES = (
    "clientlt_fedavg",
    "clientlt_support_normalized",
    "dirichlet_fedavg",
    "dirichlet_support_normalized",
)


def case_parts(case: str) -> tuple[str, str]:
    if case.startswith("clientlt_"):
        topology = "clientlt"
        aggregation = case.removeprefix("clientlt_")
    elif case.startswith("dirichlet_"):
        topology = "dirichlet"
        aggregation = case.removeprefix("dirichlet_")
    else:
        raise ValueError(f"Unknown experiment case: {case}")
    return topology, aggregation


def run_dir(output_root: Path, case: str, seed: int) -> Path:
    topology, aggregation = case_parts(case)
    return output_root / f"seed{seed}" / topology / aggregation


def build_command(args, case: str, seed: int) -> list[str]:
    topology, aggregation = case_parts(case)
    output_dir = run_dir(args.output_root, case, seed)
    schedule_file = args.output_root / f"shared_full_schedule_seed{seed}.json"

    if topology == "clientlt":
        partition_args = [
            "--partition", "client-longtail",
            "--beta", str(args.dirichlet_beta),
            "--specialization_lambda", str(args.specialization_lambda),
            "--intra_group_alpha", str(args.intra_group_alpha),
            "--head_leakage_scale", str(args.head_leakage_scale),
        ]
    else:
        partition_args = [
            "--partition", "noniid-labeldir-fine",
            "--beta", str(args.dirichlet_beta),
            # Retained in metadata only; these do not affect Dirichlet splits.
            "--specialization_lambda", str(args.specialization_lambda),
            "--intra_group_alpha", str(args.intra_group_alpha),
            "--head_leakage_scale", str(args.head_leakage_scale),
        ]

    experiment_d = aggregation == "fedavg"
    command = [
        args.python_bin,
        "federated_main.py",
        "--root", str(args.data_root),
        "--model", "fedavg",
        "--trainer", "ClipLora",
        "--dataset", "cifar100_LT",
        "--seed", str(seed),
        "--split_seed", str(seed),
        "--num_users", str(args.num_users),
        "--frac", "1.0",
        "--round", str(args.rounds),
        "--local_epochs", str(args.local_epochs),
        "--client_schedule_seed", str(seed),
        "--client_schedule_file", str(schedule_file),
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
        "--imb_factor", str(args.imb_factor),
        "--imb_type", "exp",
        "--train_batch_size", str(args.train_batch_size),
        "--test_batch_size", str(args.test_batch_size),
        "--global_eval_interval", str(args.global_eval_interval),
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
        "--cliplora_precision", "amp",
        "--cliplora_aggregation", aggregation,
        "--experimentD_enable", str(experiment_d),
    ]
    if experiment_d:
        command.extend([
            "--experimentD_rounds", "20,50,80",
            "--experimentD_include_normalized", "True",
            "--experimentD_log_update_norm", "True",
            "--experimentD_require_full_participation", "True",
            "--experimentD_verify_fedavg", "True",
            "--experimentD_eval_mode", "class_filtered",
        ])
    command.extend(partition_args)
    command.extend(["DATALOADER.NUM_WORKERS", str(args.num_workers)])
    return command


def command_text(command: list[str]) -> str:
    if os.name == "nt":
        return subprocess.list2cmdline(command)
    return shlex.join(command)


def is_complete(path: Path, rounds: int) -> bool:
    metrics = path / "round_metrics.csv"
    if not metrics.exists():
        return False
    lines = metrics.read_text(encoding="utf-8").strip().splitlines()
    return any(line.startswith(f"{rounds - 1},") for line in lines[1:])


def load_partition_summary(path: Path) -> dict:
    summary_path = path / "partition_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing partition summary: {summary_path}")
    return json.loads(summary_path.read_text(encoding="utf-8"))


def verify_matched_class_marginals(output_root: Path, cases: list[str], seeds: list[int]) -> Path:
    rows = []
    for seed in seeds:
        summaries = {
            case: load_partition_summary(run_dir(output_root, case, seed))
            for case in cases
        }
        fingerprints = {
            summary["global_lt_fingerprint"] for summary in summaries.values()
        }
        if len(fingerprints) != 1:
            details = {
                case: summary["global_lt_fingerprint"]
                for case, summary in summaries.items()
            }
            raise RuntimeError(
                f"Seed {seed} does not have a matched global class marginal: {details}"
            )
        for case, summary in summaries.items():
            topology, aggregation = case_parts(case)
            rows.append(
                {
                    "seed": seed,
                    "case": case,
                    "topology": topology,
                    "aggregation": aggregation,
                    "run_dir": str(run_dir(output_root, case, seed)),
                    "global_lt_fingerprint": summary["global_lt_fingerprint"],
                    "global_class_counts": summary["global_class_counts"],
                }
            )

    topologies = {case_parts(case)[0] for case in cases}
    manifest = output_root / "paired_experiment_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "design": "2x2: topology x ClipLora aggregation",
                "class_marginal_match_verified": True,
                "cross_topology_match_verified": len(topologies) > 1,
                "runs": rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path("DATA"))
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--cases", nargs="+", choices=CASES, default=list(CASES))
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--rounds", type=int, default=100)
    parser.add_argument("--local-epochs", type=int, default=3)
    parser.add_argument("--num-users", type=int, default=30)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--imb-factor", type=float, default=0.01)
    parser.add_argument("--dirichlet-beta", type=float, default=0.5)
    parser.add_argument("--specialization-lambda", type=float, default=0.75)
    parser.add_argument("--intra-group-alpha", type=float, default=0.5)
    parser.add_argument("--head-leakage-scale", type=float, default=3.0)
    parser.add_argument("--train-batch-size", type=int, default=32)
    parser.add_argument("--test-batch-size", type=int, default=64)
    parser.add_argument("--global-eval-interval", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--skip-completed",
        action="store_true",
        help="Skip a run only when its metrics contain the requested final round.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_root = args.output_root.resolve()
    if not args.dry_run:
        args.output_root.mkdir(parents=True, exist_ok=True)

    commands = []
    for seed in args.seeds:
        for case in args.cases:
            output_dir = run_dir(args.output_root, case, seed)
            command = build_command(args, case, seed)
            commands.append((case, seed, output_dir, command))

    for case, seed, output_dir, command in commands:
        print(f"\n[{case} / seed {seed}]\n{command_text(command)}", flush=True)
        if args.dry_run:
            continue
        if output_dir.exists() and any(output_dir.iterdir()):
            if args.skip_completed and is_complete(output_dir, args.rounds):
                print(f"Skipping completed run: {output_dir}", flush=True)
                continue
            raise FileExistsError(
                f"Refusing to append to non-empty run directory: {output_dir}. "
                "Choose a fresh --output-root or use --skip-completed."
            )
        subprocess.run(command, cwd=REPO_ROOT, check=True)

    if args.dry_run:
        return
    manifest = verify_matched_class_marginals(
        args.output_root,
        list(args.cases),
        list(args.seeds),
    )
    print(f"\nMatched class marginals verified. Manifest: {manifest}")


if __name__ == "__main__":
    main()
