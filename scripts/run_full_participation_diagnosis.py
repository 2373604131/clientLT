#!/usr/bin/env python3
"""Run the two-topology CIFAR100-LT full-participation diagnosis."""

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

from scripts.create_client_schedule import validate_schedule, write_schedule_atomic
from scripts.run_online_sca import read_frozen_lora


TOPOLOGIES = {
    "clientlt": "client-longtail",
    "matched": "matched-dirichlet",
}
NUM_USERS = 30
ROUNDS = 80
FRAC = 1.0
SEED = 42


def _sha256_json(value) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _schedule_path(args) -> Path:
    return args.output_root / "schedule" / "frac1p0_users30_rounds80_seed42.json"


def _full_schedule() -> list[list[int]]:
    # A canonical order makes the full-participation counterfactual identical
    # across topologies and removes schedule generation as another moving part.
    return [list(range(NUM_USERS)) for _ in range(ROUNDS)]


def prepare(args) -> dict:
    args.output_root.mkdir(parents=True, exist_ok=True)
    schedule_path = _schedule_path(args)
    schedule = _full_schedule()
    if schedule_path.exists():
        payload = json.loads(schedule_path.read_text(encoding="utf-8"))
        existing = payload.get("schedule", payload) if isinstance(payload, dict) else payload
        if existing != schedule:
            raise RuntimeError(f"Existing full-participation schedule is incompatible: {schedule_path}")
    else:
        write_schedule_atomic(
            schedule_path,
            {
                "num_rounds": ROUNDS,
                "num_users": NUM_USERS,
                "frac": FRAC,
                "clients_per_round": NUM_USERS,
                "seed": SEED,
                "schedule": schedule,
            },
        )
    validate_schedule(schedule, ROUNDS, NUM_USERS, FRAC)

    lora_config = read_frozen_lora(args.freeze_file)
    protocol = {
        "schema_version": "full_participation_diagnosis_v1",
        "question": (
            "Does the Client-LT final-tail gap and/or best-to-final collapse remain "
            "when all 30 clients participate in every round?"
        ),
        "dataset": "cifar100_LT",
        "topologies": TOPOLOGIES,
        "seed": SEED,
        "split_seed": SEED,
        "num_users": NUM_USERS,
        "frac": FRAC,
        "rounds": ROUNDS,
        "local_epochs": 3,
        "tail_class_ratio": 0.2,
        "equivalence_threshold_pp": float(args.equivalence_threshold_pp),
        "model": "ClipLora vision-only ordinary sample-weighted FedAvg",
        "lora_config": lora_config,
        "common_init_seed": int(args.common_init_seed),
        "schedule_file": str(schedule_path.resolve()),
        "schedule_sha256": _sha256_json(schedule),
        "partial_baseline_root": (
            str(args.partial_root.resolve()) if args.partial_root is not None else None
        ),
        "primary_metrics_only": [
            "final_tail_accuracy_gap_pp",
            "best_to_final_drop_gap_pp",
        ],
        "required_audits": [
            "initial_model_hash_equal",
            "global_class_counts_equal",
            "client_total_samples_equal",
            "thirty_unique_clients_every_round",
            "fedavg_weight_sum_one_every_round",
        ],
        "excluded_analyses": [
            "margin",
            "retention_ratio",
            "coverage",
            "correlation",
        ],
    }
    protocol_path = args.output_root / "frozen_protocol.json"
    if protocol_path.exists():
        existing = json.loads(protocol_path.read_text(encoding="utf-8"))
        if existing != protocol:
            started = any(
                (args.output_root / name / "round_metrics.csv").is_file()
                for name in ("clientlt", "matched_dirichlet")
            )
            if started:
                raise RuntimeError(f"Refusing to change a started protocol: {protocol_path}")
            protocol_path.write_text(
                json.dumps(protocol, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
                encoding="utf-8",
            )
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
        str(SEED),
        "--split_seed",
        str(SEED),
        "--num_users",
        str(NUM_USERS),
        "--frac",
        str(FRAC),
        "--round",
        str(ROUNDS),
        "--local_epochs",
        "3",
        "--client_schedule_seed",
        str(SEED),
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
        "--cliplora_common_init_seed",
        str(args.common_init_seed),
        "--cliplora_aggregation",
        "fedavg",
        "--cliplora_sca_enable",
        "False",
        "--experimentD_enable",
        "False",
        "--functional_coverage_validation_enable",
        "False",
        "DATALOADER.NUM_WORKERS",
        str(args.num_workers),
    ]


def _gpu_environment(slot: str) -> dict:
    env = os.environ.copy()
    visible = [value.strip() for value in env.get("CUDA_VISIBLE_DEVICES", "").split(",") if value.strip()]
    if visible and str(slot).isdigit() and int(slot) < len(visible):
        env["CUDA_VISIBLE_DEVICES"] = visible[int(slot)]
    else:
        env["CUDA_VISIBLE_DEVICES"] = str(slot)
    return env


def run_topology(args, short_name: str) -> None:
    protocol = prepare(args)
    output_name = "clientlt" if short_name == "clientlt" else "matched_dirichlet"
    output_dir = args.output_root / output_name
    if (output_dir / "round_metrics.csv").exists():
        raise RuntimeError(
            f"Run directory already contains training metrics: {output_dir}. "
            "Use a new --output-root rather than appending another trajectory."
        )
    command = common_command(args, TOPOLOGIES[short_name], output_dir, protocol["lora_config"])
    print(shlex.join(command), flush=True)
    if not args.dry_run:
        subprocess.run(command, cwd=ROOT, env=_gpu_environment(args.gpu), check=True)


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
        default=Path("output/full_participation_diagnosis_seed42"),
    )
    parser.add_argument(
        "--partial-root",
        type=Path,
        default=None,
        help=(
            "Optional frozen frac=0.4 result root. Omit it to analyze only the two "
            "required frac=1.0 runs."
        ),
    )
    parser.add_argument("--data-root", type=Path, default=Path("DATA"))
    parser.add_argument(
        "--freeze-file",
        type=Path,
        default=Path("output/g0_d1_seed42/lora_freeze.json"),
    )
    parser.add_argument("--common-init-seed", type=int, default=424242)
    parser.add_argument("--equivalence-threshold-pp", type=float, default=2.0)
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--test-batch-size", type=int, default=100)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.equivalence_threshold_pp <= 0:
        raise ValueError("--equivalence-threshold-pp must be positive")

    if args.stage in {"prepare", "all"}:
        protocol = prepare(args)
        print(json.dumps({"stage": "prepare", "schedule_sha256": protocol["schedule_sha256"]}))
    if args.stage in {"clientlt", "all"}:
        run_topology(args, "clientlt")
    if args.stage in {"matched", "all"}:
        run_topology(args, "matched")
    if args.stage in {"analyze", "all"} and not args.dry_run:
        from scripts.analyze_full_participation_diagnosis import analyze

        result = analyze(args.output_root, args.partial_root)
        print(json.dumps({"stage": "analyze", "verdict": result["verdict"]}))


if __name__ == "__main__":
    main()
