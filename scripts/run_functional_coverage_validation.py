#!/usr/bin/env python3
"""Launch the minimal two-topology functional-coverage validation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.create_client_schedule import (
    create_schedule,
    load_schedule,
    validate_schedule,
    write_schedule_atomic,
)
from scripts.run_online_sca import read_frozen_lora


TOPOLOGIES = {
    "clientlt": "client-longtail",
    "matched": "matched-dirichlet",
}


def _sha256_json(value) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _schedule_path(args) -> Path:
    return args.output_root / "schedule" / "frac0p4_users30_rounds80_seed42.json"


def prepare(args) -> dict:
    args.output_root.mkdir(parents=True, exist_ok=True)
    schedule_path = _schedule_path(args)
    if schedule_path.exists():
        schedule = load_schedule(schedule_path)
    else:
        schedule = create_schedule(args.rounds, 30, args.frac, 42)
        write_schedule_atomic(
            schedule_path,
            {
                "num_rounds": args.rounds,
                "num_users": 30,
                "frac": args.frac,
                "clients_per_round": int(args.frac * 30),
                "seed": 42,
                "schedule": schedule,
            },
        )
    validate_schedule(schedule, args.rounds, 30, args.frac)
    config = read_frozen_lora(args.freeze_file)
    if not args.theta0_file.is_file():
        raise FileNotFoundError(
            f"Missing common LoRA theta0: {args.theta0_file}. "
            "Use the frozen seed-42 anchor already produced by E1."
        )
    protocol = {
        "schema_version": "functional_coverage_validation_launcher_v1",
        "single_question": (
            "Does fixed-margin Client-LT reduce functional coverage and accompany worse "
            "tail accuracy and late retention than matched Dirichlet?"
        ),
        "topologies": TOPOLOGIES,
        "seed": 42,
        "split_seed": 42,
        "num_users": 30,
        "frac": args.frac,
        "rounds": args.rounds,
        "local_epochs": 3,
        "coverage_rounds": args.coverage_rounds,
        "samples_per_tail_class": args.samples_per_class,
        "model": "ClipLora vision-only ordinary FedAvg",
        "lora_config": config,
        "common_theta0": str(args.theta0_file.resolve()),
        "schedule_file": str(schedule_path.resolve()),
        "schedule_sha256": _sha256_json(schedule),
        "primary_metrics": [
            "matched_minus_clientlt_available_coverage",
            "matched_minus_clientlt_realized_coverage",
            "matched_minus_clientlt_final_tail_accuracy_pp",
            "clientlt_minus_matched_best_to_final_tail_drop_pp",
        ],
        "decision_rule": {
            "coverage": "both coverage gaps > 0",
            "performance": "final tail and H-mean gaps > 0",
            "retention": "Client-LT best-to-final tail drop minus matched drop > 0",
        },
        "test_split_role": "outcome evaluation only; never selects boundaries or controls training",
    }
    protocol_path = args.output_root / "frozen_experiment_protocol.json"
    if protocol_path.exists():
        existing = json.loads(protocol_path.read_text(encoding="utf-8"))
        if existing != protocol:
            raise RuntimeError(f"Refusing to change frozen protocol: {protocol_path}")
    else:
        protocol_path.write_text(
            json.dumps(protocol, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
    return protocol


def common_command(args, partition: str, output_dir: Path, config: dict) -> list[str]:
    return [
        args.python_bin,
        "-u",
        "federated_main.py",
        "--root",
        str(args.data_root),
        "--model",
        "fedavg",
        "--trainer",
        "ClipLora",
        "--dataset",
        "cifar100_LT",
        "--seed",
        "42",
        "--split_seed",
        "42",
        "--num_users",
        "30",
        "--frac",
        str(args.frac),
        "--round",
        str(args.rounds),
        "--local_epochs",
        "3",
        "--client_schedule_seed",
        "42",
        "--client_schedule_file",
        str(_schedule_path(args)),
        "--isolate_local_optimizer_state",
        "True",
        "--federated_single_scheduler_step",
        "True",
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
        "configs/trainers/PromptFL/vit_b16.yaml",
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
        "0.5",
        "--specialization_lambda",
        "0.75",
        "--intra_group_alpha",
        "0.5",
        "--head_leakage_scale",
        "3.0",
        "--encoder",
        "vision",
        "--cliplora_position",
        str(config["position"]),
        "--cliplora_rank",
        str(config["rank"]),
        "--cliplora_alpha",
        str(config["alpha"]),
        "--cliplora_params",
        *[str(value) for value in config["params"]],
        "--cliplora_dropout_rate",
        "0.0",
        "--cliplora_lr_policy",
        "constant",
        "--cliplora_precision",
        "fp32",
        "--cliplora_aggregation",
        "fedavg",
        "--cliplora_sca_enable",
        "False",
        "--experimentD_enable",
        "False",
        "--functional_coverage_validation_enable",
        "True",
        "--functional_coverage_validation_rounds",
        args.coverage_rounds,
        "--functional_coverage_theta0_file",
        str(args.theta0_file),
        "--functional_coverage_samples_per_class",
        str(args.samples_per_class),
        "--functional_coverage_gain_epsilon",
        "0.0",
        "--functional_coverage_eval_batch_size",
        str(args.coverage_batch_size),
        "DATALOADER.NUM_WORKERS",
        str(args.num_workers),
    ]


def _gpu_environment(slot: str) -> dict:
    env = os.environ.copy()
    visible = [
        value.strip()
        for value in env.get("CUDA_VISIBLE_DEVICES", "").split(",")
        if value.strip()
    ]
    if visible and str(slot).isdigit() and int(slot) < len(visible):
        env["CUDA_VISIBLE_DEVICES"] = visible[int(slot)]
    else:
        env["CUDA_VISIBLE_DEVICES"] = str(slot)
    return env


def run_topology(args, short_name: str) -> None:
    protocol = prepare(args)
    config = protocol["lora_config"]
    output_name = "clientlt" if short_name == "clientlt" else "matched_dirichlet"
    output_dir = args.output_root / output_name
    if (output_dir / "round_metrics.csv").exists():
        raise RuntimeError(
            f"Run directory already contains training metrics: {output_dir}. "
            "Use a new --output-root to avoid appending incompatible trajectories."
        )
    command = common_command(args, TOPOLOGIES[short_name], output_dir, config)
    print(shlex.join(command), flush=True)
    if args.dry_run:
        return
    subprocess.run(
        command,
        cwd=ROOT,
        env=_gpu_environment(args.gpu),
        check=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=["prepare", "clientlt", "matched", "analyze", "all"],
        default="all",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("output/functional_coverage_validation_seed42"),
    )
    parser.add_argument("--data-root", type=Path, default=Path("DATA"))
    parser.add_argument(
        "--freeze-file",
        type=Path,
        default=Path("output/g0_d1_seed42/lora_freeze.json"),
    )
    parser.add_argument(
        "--theta0-file",
        type=Path,
        default=Path("output/e1_strength_breadth/protocol_v2/theta0_seed42.pt"),
    )
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--rounds", type=int, default=80)
    parser.add_argument("--frac", type=float, default=0.4)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--coverage-rounds", default="1,10,20,40,60,80")
    parser.add_argument("--samples-per-class", type=int, default=10)
    parser.add_argument("--coverage-batch-size", type=int, default=100)
    parser.add_argument("--test-batch-size", type=int, default=100)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.rounds != 80 or abs(args.frac - 0.4) > 1e-12:
        raise ValueError("The frozen primary validation requires rounds=80 and frac=0.4")

    if args.stage in {"prepare", "all"}:
        result = prepare(args)
        print(json.dumps({"stage": "prepare", "schedule_sha256": result["schedule_sha256"]}))
    if args.stage in {"clientlt", "all"}:
        run_topology(args, "clientlt")
    if args.stage in {"matched", "all"}:
        run_topology(args, "matched")
    if args.stage in {"analyze", "all"} and not args.dry_run:
        from scripts.analyze_functional_coverage_validation import analyze

        result = analyze(args.output_root)
        print(json.dumps({"stage": "analyze", "verdict": result["verdict"]}))


if __name__ == "__main__":
    main()

