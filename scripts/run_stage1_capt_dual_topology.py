#!/usr/bin/env python3
"""Run the exact seed-42 CAPT dual-topology diagnostic baseline."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
CONDITIONS = {
    "clientlt": "client-longtail",
    "matched": "matched-dirichlet",
}
STRICT_STAGE1_PROTOCOL = {
    "seed": 42,
    "split_seed": 42,
    "schedule_seed": 42,
    "num_users": 30,
    "frac": 0.4,
    "rounds": 80,
    "local_epochs": 3,
}


def _schedule_path(args) -> Path:
    frac = str(args.frac).replace(".", "p")
    return args.sca_output_root / "schedules" / (
        f"frac{frac}_users{args.num_users}_rounds{args.rounds}_seed{args.seed}.json"
    )


def _ensure_schedule(path: Path, args) -> dict:
    clients_per_round = max(int(args.frac * args.num_users), 1)
    if path.exists():
        payload = json.loads(path.read_text(encoding="utf-8"))
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        rng = np.random.default_rng(args.schedule_seed)
        payload = {
            "num_rounds": args.rounds,
            "num_users": args.num_users,
            "frac": args.frac,
            "clients_per_round": clients_per_round,
            "seed": args.schedule_seed,
            "schedule": [
                [
                    int(value)
                    for value in rng.choice(
                        args.num_users, clients_per_round, replace=False
                    ).tolist()
                ]
                for _ in range(args.rounds)
            ],
        }
        temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}")
        temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        os.replace(temporary, path)
    schedule = payload.get("schedule", payload) if isinstance(payload, dict) else payload
    if len(schedule) < args.rounds:
        raise ValueError(f"Schedule {path} has fewer than {args.rounds} rounds")
    for epoch, clients in enumerate(schedule[: args.rounds]):
        if len(clients) != clients_per_round or len(set(clients)) != len(clients):
            raise ValueError(f"Invalid schedule epoch {epoch}: {clients}")
        if any(int(value) < 0 or int(value) >= args.num_users for value in clients):
            raise ValueError(f"Out-of-range client in schedule epoch {epoch}: {clients}")
    return payload


def _read_sca_actual_schedule(run_dir: Path, rounds: int) -> list[list[int]]:
    path = run_dir / "lora_aggregation_weights.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Strict CAPT preflight requires the executed SCA client audit: {path}"
        )
    grouped: dict[int, list[int]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for raw in csv.DictReader(handle):
            grouped.setdefault(int(raw["epoch_index"]), []).append(
                int(raw["client_id"])
            )
    if sorted(grouped) != list(range(rounds)):
        raise ValueError(f"SCA audit does not cover exactly {rounds} rounds: {path}")
    return [sorted(grouped[epoch]) for epoch in range(rounds)]


def _audit_schedule_against_sca(args, payload: dict) -> None:
    requested = payload.get("schedule", payload) if isinstance(payload, dict) else payload
    requested = [sorted(int(value) for value in row) for row in requested[: args.rounds]]
    actual_clientlt = _read_sca_actual_schedule(
        args.sca_output_root / "online_sca", args.rounds
    )
    actual_matched = _read_sca_actual_schedule(
        args.sca_output_root / "online_sca_matched_dirichlet", args.rounds
    )
    if actual_clientlt != actual_matched:
        raise ValueError("Existing Client-LT and matched SCA runs used different schedules")
    if requested != actual_clientlt:
        raise ValueError(
            "Requested CAPT schedule differs from the clients actually used by SCA"
        )


def _schedule_hash(payload: dict, rounds: int) -> str:
    schedule = payload.get("schedule", payload) if isinstance(payload, dict) else payload
    normalized = [sorted(int(value) for value in row) for row in schedule[:rounds]]
    encoded = json.dumps(normalized, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _output_dir(args, condition: str) -> Path:
    return args.output_root / f"capt_{condition}"


def _is_finished(output_dir: Path, rounds: int) -> bool:
    path = output_dir / "round_metrics.csv"
    if not path.exists():
        return False
    with path.open(newline="", encoding="utf-8") as handle:
        epochs = [int(row["epoch"]) for row in csv.DictReader(handle)]
    return bool(epochs) and max(epochs) >= rounds - 1


def build_command(args, condition: str, schedule: Path) -> list[str]:
    if condition not in CONDITIONS:
        raise ValueError(f"Unknown CAPT Stage-1 condition: {condition}")
    partition = CONDITIONS[condition]
    output_dir = _output_dir(args, condition)
    return [
        args.python_bin,
        "-u",
        "federated_main.py",
        "--root",
        str(args.data_root),
        "--model",
        "cluster",
        "--trainer",
        "CAPT",
        "--dataset",
        "cifar100_LT",
        "--seed",
        str(args.seed),
        "--split_seed",
        str(args.split_seed),
        "--num_users",
        str(args.num_users),
        "--frac",
        str(args.frac),
        "--round",
        str(args.rounds),
        "--local_epochs",
        str(args.local_epochs),
        "--client_schedule_seed",
        str(args.schedule_seed),
        "--client_schedule_file",
        str(schedule),
        "--lr",
        str(args.lr),
        "--gamma",
        "1",
        "--n_ctx",
        "4",
        "--n_general",
        "1",
        "--ctx_init",
        "False",
        "--csc",
        "True",
        "--dataset-config-file",
        "configs/datasets/cifar100_LT.yaml",
        "--config-file",
        "configs/trainers/CAPT/vit_b16.yaml",
        "--output-dir",
        str(output_dir),
        "--imb_factor",
        "0.01",
        "--imb_type",
        "exp",
        "--train_batch_size",
        "32",
        "--test_batch_size",
        str(args.test_batch_size),
        "--global_eval_interval",
        "1",
        "--num_classes",
        "100",
        "--tail_class_ratio",
        "0.2",
        "--head_client_ratio",
        "0.9",
        "--tail_client_ratio",
        "0.1",
        "--head_class_ratio",
        "0.8",
        "--partition",
        partition,
        "--beta",
        str(args.matched_beta),
        "--specialization_lambda",
        "0.75",
        "--intra_group_alpha",
        "0.5",
        "--head_leakage_scale",
        "3.0",
        "--n_simclusters",
        "4",
        "--n_disclusters",
        "4",
        "--capt_fixed_global_agg_freq",
        "1",
        "DATALOADER.NUM_WORKERS",
        str(args.num_workers),
    ]


def _gpu_environment(gpu: str) -> dict:
    environment = os.environ.copy()
    visible = [
        value.strip()
        for value in environment.get("CUDA_VISIBLE_DEVICES", "").split(",")
        if value.strip()
    ]
    if visible and str(gpu).isdigit() and int(gpu) < len(visible):
        environment["CUDA_VISIBLE_DEVICES"] = visible[int(gpu)]
    else:
        environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
    return environment


def _write_protocol(
    args, condition: str, command: list[str], schedule: Path, schedule_payload: dict
) -> None:
    output_dir = _output_dir(args, condition)
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "stage1b_capt_dual_topology_v1",
        "condition": condition,
        "partition": CONDITIONS[condition],
        "seed": args.seed,
        "split_seed": args.split_seed,
        "schedule_seed": args.schedule_seed,
        "schedule_file": str(schedule),
        "client_schedule_sha256": _schedule_hash(schedule_payload, args.rounds),
        "num_users": args.num_users,
        "frac": args.frac,
        "rounds": args.rounds,
        "local_epochs": args.local_epochs,
        "matched_beta": args.matched_beta,
        "clientlt": {
            "specialization_lambda": 0.75,
            "intra_group_alpha": 0.5,
            "head_leakage_scale": 3.0,
            "head_client_ratio": 0.9,
            "tail_client_ratio": 0.1,
            "head_class_ratio": 0.8,
            "tail_class_ratio": 0.2,
        },
        "capt_protocol": {
            "model": "cluster",
            "trainer": "CAPT",
            "fixed_global_aggregation_frequency": 1,
            "official_test_controls_future_training": False,
            "reason": (
                "CAPT's default MAB consumes official-test metrics and may skip the "
                "final aggregation; Stage-1 fixes aggregation every round for an "
                "auditable two-topology diagnostic while preserving default behavior "
                "outside this explicit flag."
            ),
        },
        "command": command,
    }
    (output_dir / "stage1b_capt_protocol.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )


def run_condition(args, condition: str, schedule: Path, schedule_payload: dict) -> None:
    output_dir = _output_dir(args, condition)
    if _is_finished(output_dir, args.rounds):
        if args.skip_finished:
            print(f"Skip finished CAPT condition: {output_dir}", flush=True)
            return
        raise FileExistsError(f"Completed output already exists: {output_dir}")
    if output_dir.exists() and any(output_dir.iterdir()) and not args.dry_run:
        allowed = {"stage1b_capt_protocol.json"}
        unexpected = {path.name for path in output_dir.iterdir()} - allowed
        if unexpected:
            raise FileExistsError(
                f"Refusing to mix a new run into non-empty directory {output_dir}: "
                f"{sorted(unexpected)}"
            )
    command = build_command(args, condition, schedule)
    print("=" * 78, flush=True)
    print(shlex.join(command), flush=True)
    print("=" * 78, flush=True)
    if args.dry_run:
        return
    _write_protocol(args, condition, command, schedule, schedule_payload)
    subprocess.run(
        command,
        cwd=REPO_ROOT,
        env=_gpu_environment(args.gpu),
        check=True,
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage", choices=["clientlt", "matched", "both"], default="both"
    )
    parser.add_argument(
        "--sca-output-root", type=Path, default=Path("output/online_sca_seed42_v2")
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("output/online_sca_seed42_v2/stage1b_capt"),
    )
    parser.add_argument("--data-root", type=Path, default=Path("DATA"))
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--schedule-seed", type=int, default=42)
    parser.add_argument("--num-users", type=int, default=30)
    parser.add_argument("--frac", type=float, default=0.4)
    parser.add_argument("--rounds", type=int, default=80)
    parser.add_argument("--local-epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--matched-beta", type=float, default=0.5)
    parser.add_argument("--test-batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--skip-finished", action="store_true")
    parser.add_argument("--skip-analysis", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    mismatches = {
        key: {"required": expected, "received": getattr(args, key)}
        for key, expected in STRICT_STAGE1_PROTOCOL.items()
        if getattr(args, key) != expected
    }
    if mismatches:
        raise ValueError(
            f"Strict seed-42 Stage-1B protocol mismatch: {mismatches}"
        )
    if args.matched_beta <= 0:
        raise ValueError("--matched-beta must be positive")
    args.sca_output_root = args.sca_output_root.resolve()
    args.output_root = args.output_root.resolve()
    args.data_root = args.data_root.resolve()
    schedule = _schedule_path(args)
    schedule_payload = _ensure_schedule(schedule, args)
    _audit_schedule_against_sca(args, schedule_payload)
    stages = ["clientlt", "matched"] if args.stage == "both" else [args.stage]
    for condition in stages:
        run_condition(args, condition, schedule, schedule_payload)
    if args.dry_run or args.skip_analysis:
        return
    if all(
        _is_finished(_output_dir(args, condition), args.rounds)
        for condition in CONDITIONS
    ):
        subprocess.run(
            [
                args.python_bin,
                "-u",
                "scripts/analyze_stage1_capt_gap.py",
                "--sca-output-root",
                str(args.sca_output_root),
                "--capt-output-root",
                str(args.output_root),
            ],
            cwd=REPO_ROOT,
            check=True,
        )


if __name__ == "__main__":
    main()
