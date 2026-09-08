#!/usr/bin/env python
"""Run and verify the deterministic six-condition PFRF smoke gate."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Mapping

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_eri_closure import ensure_client_schedule
from utils.pfrf import PFRF_CHECKPOINT_SCHEMA, PFRF_CONDITIONS


RESUME_CONDITIONS = ("pfrf_max", "pfrf_add")
UNIT_TESTS = (
    "tests/test_pfrf_smoke.py",
    "tests/test_lora_aggregation.py",
    "tests/test_pfrf_runner.py",
    "tests/test_pfrf_federated_integration.py",
)


def checkpoint_path(run_dir: Path) -> Path:
    return run_dir / "pfrf" / "checkpoint_latest.pt"


def load_checkpoint_file(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    try:
        value = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        value = torch.load(path, map_location="cpu")
    if value.get("schema_version") != PFRF_CHECKPOINT_SCHEMA:
        raise ValueError(f"Unexpected checkpoint schema in {path}")
    return value


def load_checkpoint(run_dir: Path) -> dict:
    return load_checkpoint_file(checkpoint_path(run_dir))


def run_directory(root: Path, condition: str) -> Path:
    return root / "runs" / condition / "seed42"


def resume_directory(root: Path, condition: str, part: str) -> Path:
    return root / "resume" / condition / part


def build_command(
    args,
    *,
    condition: str,
    output_dir: Path,
    rounds: int,
    schedule: Path,
    resume: Path | None = None,
) -> list[str]:
    command = [
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
        "--dataset-config-file",
        "configs/datasets/cifar100_LT.yaml",
        "--config-file",
        "configs/trainers/PromptFL/vit_b16.yaml",
        "--output-dir",
        str(output_dir),
        "--seed",
        "42",
        "--split_seed",
        "42",
        "--num_users",
        str(args.num_users),
        "--frac",
        "1.0",
        "--round",
        str(rounds),
        "--local_epochs",
        "3",
        "--client_schedule_seed",
        "42",
        "--client_schedule_file",
        str(schedule),
        "--partition",
        "client-longtail",
        "--imb_factor",
        "0.01",
        "--imb_type",
        "exp",
        "--specialization_lambda",
        "0.75",
        "--intra_group_alpha",
        "0.5",
        "--head_leakage_scale",
        "3.0",
        "--head_client_ratio",
        "0.9",
        "--tail_client_ratio",
        "0.1",
        "--head_class_ratio",
        "0.8",
        "--tail_class_ratio",
        "0.2",
        "--train_batch_size",
        "32",
        "--test_batch_size",
        "64",
        "--global_eval_interval",
        "1",
        "--lr",
        "0.001",
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
        "--encoder",
        "vision",
        "--cliplora_position",
        "top3",
        "--cliplora_rank",
        "2",
        "--cliplora_alpha",
        "1",
        "--cliplora_dropout_rate",
        "0",
        "--cliplora_params",
        "q",
        "v",
        "--cliplora_lr_policy",
        "constant",
        "--cliplora_precision",
        "fp32",
        "--cliplora_common_init_seed",
        "42",
        "--cliplora_aggregation",
        "effective_svd",
        "--isolate_local_optimizer_state",
        "True",
        "--federated_single_scheduler_step",
        "True",
        "--pfrf_enable",
        "True",
        "--pfrf_condition",
        condition,
        "--pfrf_temperature",
        "1",
        "--pfrf_lambda",
        "1",
        "--pfrf_proposal_epochs",
        "2",
        "--pfrf_add_cap",
        "1",
        "--pfrf_deterministic",
        "True",
        "--cliplora_sca_enable",
        "False",
        "--experimentD_enable",
        "False",
        "--e1_enable",
        "False",
        "--stage3_enable",
        "False",
    ]
    if resume is not None:
        command.extend(["--resume", str(resume)])
    # argparse exposes trailing YACS overrides through a positional REMAINDER.
    # Every ordinary --flag, especially --resume, must precede this boundary.
    command.extend(["DATALOADER.NUM_WORKERS", "0"])
    return command


def commands(args) -> list[dict]:
    schedule = args.output_root / "protocol" / "schedule_users30_frac1_rounds5_seed42.json"
    result = []
    for condition in args.conditions:
        result.append(
            {
                "label": f"full:{condition}",
                "command": build_command(
                    args,
                    condition=condition,
                    output_dir=run_directory(args.output_root, condition),
                    rounds=5,
                    schedule=schedule,
                ),
            }
        )
    for condition in RESUME_CONDITIONS:
        if condition not in args.conditions:
            continue
        part1 = resume_directory(args.output_root, condition, "round2")
        part2 = resume_directory(args.output_root, condition, "round5")
        result.extend(
            [
                {
                    "label": f"resume-prefix:{condition}",
                    "command": build_command(
                        args,
                        condition=condition,
                        output_dir=part1,
                        rounds=2,
                        schedule=schedule,
                    ),
                },
                {
                    "label": f"resume-suffix:{condition}",
                    "command": build_command(
                        args,
                        condition=condition,
                        output_dir=part2,
                        rounds=5,
                        schedule=schedule,
                        resume=part1,
                    ),
                },
            ]
        )
    return result


def _run(command: list[str]) -> None:
    environment = os.environ.copy()
    # torch.use_deterministic_algorithms(True) requires a deterministic
    # cuBLAS workspace on CUDA >= 10.2. Set it on the child process before
    # Python imports torch or initializes CUDA.
    environment["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    subprocess.run(command, cwd=REPO_ROOT, check=True, env=environment)


def run_unit_tests(args) -> None:
    _run([args.python_bin, "-m", "pytest", "-q", *UNIT_TESTS])
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "unit_test_report.json").write_text(
        json.dumps(
            {
                "schema_version": "pfrf_unit_test_report_v1",
                "status": "pass",
                "tests": list(UNIT_TESTS),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def run_training(args) -> None:
    schedule = args.output_root / "protocol" / "schedule_users30_frac1_rounds5_seed42.json"
    ensure_client_schedule(
        schedule, rounds=5, users=args.num_users, frac=1.0, seed=42
    )
    for item in commands(args):
        command = item["command"]
        output_index = command.index("--output-dir") + 1
        output_dir = Path(command[output_index])
        expected_round = int(command[command.index("--round") + 1])
        path = checkpoint_path(output_dir)
        if path.is_file() and int(load_checkpoint(output_dir)["completed_round"]) == expected_round:
            print(f"Skip complete PFRF smoke run: {item['label']}")
            continue
        if output_dir.exists() and any(output_dir.iterdir()):
            raise RuntimeError(
                f"Refusing to append to incomplete smoke output {output_dir}; "
                "move it aside and rerun"
            )
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "smoke_command.json").write_text(
            json.dumps(
                {
                    "schema_version": "pfrf_smoke_command_v1",
                    "label": item["label"],
                    "command": command,
                },
                indent=2,
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"Run PFRF smoke: {item['label']}")
        _run(command)


def _finite_tensor_mapping(value: Mapping[str, torch.Tensor]) -> bool:
    return all(bool(torch.isfinite(tensor).all()) for tensor in value.values())


def _state_close(left: Mapping[str, torch.Tensor], right: Mapping[str, torch.Tensor]) -> tuple[bool, float]:
    if set(left) != set(right):
        return False, math.inf
    maximum = 0.0
    for key in left:
        a = torch.as_tensor(left[key]).cpu()
        b = torch.as_tensor(right[key]).cpu()
        if a.shape != b.shape or a.dtype != b.dtype:
            return False, math.inf
        if a.numel():
            maximum = max(maximum, float(torch.max(torch.abs(a.float() - b.float())).item()))
        if not torch.allclose(a, b, atol=1e-6, rtol=1e-5):
            return False, maximum
    return True, maximum


def _csv_rounds(path: Path) -> set[int]:
    if not path.is_file():
        return set()
    with path.open(encoding="utf-8", newline="") as handle:
        return {int(row["communication_round"]) for row in csv.DictReader(handle)}


def _rng_state_equal(left: Mapping, right: Mapping) -> bool:
    if left["python"] != right["python"]:
        return False
    left_numpy, right_numpy = left["numpy"], right["numpy"]
    if (
        left_numpy[0] != right_numpy[0]
        or not np.array_equal(left_numpy[1], right_numpy[1])
        or left_numpy[2:] != right_numpy[2:]
    ):
        return False
    if not torch.equal(torch.as_tensor(left["torch"]), torch.as_tensor(right["torch"])):
        return False
    left_cuda, right_cuda = left.get("cuda"), right.get("cuda")
    if left_cuda is None or right_cuda is None:
        return left_cuda is None and right_cuda is None
    return len(left_cuda) == len(right_cuda) and all(
        torch.equal(torch.as_tensor(a), torch.as_tensor(b))
        for a, b in zip(left_cuda, right_cuda)
    )


def _final_metric_state_close(left_dir: Path, right_dir: Path) -> tuple[bool, float]:
    def metric_row(run_dir: Path) -> dict:
        path = run_dir / "round_metrics.csv"
        with path.open(encoding="utf-8", newline="") as handle:
            rows = [row for row in csv.DictReader(handle) if int(row["epoch"]) == 4]
        if len(rows) != 1:
            raise ValueError(f"Expected one epoch-4 metric row in {path}, got {len(rows)}")
        return rows[0]

    keys = (
        "overall_acc",
        "non_tail_acc",
        "bottom20_tail_acc",
        "macro_per_class_acc",
        "macro_f1",
    )
    maximum = 0.0
    left_row, right_row = metric_row(left_dir), metric_row(right_dir)
    for key in keys:
        error = abs(float(left_row[key]) - float(right_row[key]))
        maximum = max(maximum, error)
        if not math.isclose(float(left_row[key]), float(right_row[key]), rel_tol=1e-8, abs_tol=1e-8):
            return False, maximum

    def class_values(run_dir: Path) -> dict[int, float]:
        path = run_dir / "per_class_accuracy_epoch_4.csv"
        with path.open(encoding="utf-8", newline="") as handle:
            return {
                int(row["class_id"]): float(row["per_class_acc"])
                for row in csv.DictReader(handle)
            }

    left_classes, right_classes = class_values(left_dir), class_values(right_dir)
    if set(left_classes) != set(right_classes) or len(left_classes) != 100:
        return False, math.inf
    for class_id in left_classes:
        error = abs(left_classes[class_id] - right_classes[class_id])
        maximum = max(maximum, error)
        if not math.isclose(
            left_classes[class_id], right_classes[class_id],
            rel_tol=1e-8, abs_tol=1e-8,
        ):
            return False, maximum
    return True, maximum


def verify(args) -> dict:
    checks = []

    def record(name: str, passed: bool, **details):
        checks.append({"name": name, "passed": bool(passed), **details})

    unit_report = args.output_root / "unit_test_report.json"
    unit_passed = False
    if unit_report.is_file():
        unit_passed = json.loads(unit_report.read_text(encoding="utf-8")).get("status") == "pass"
    record("deterministic_unit_tests", unit_passed)

    split_fingerprints = {}
    for condition in args.conditions:
        run_dir = run_directory(args.output_root, condition)
        try:
            required_contracts = (
                run_dir / "smoke_command.json",
                run_dir / "resolved_config.yaml",
                run_dir / "client_split_fingerprint.json",
                run_dir / "effective_svd_contract.json",
                run_dir / "pfrf" / "runtime_contract.json",
            )
            record(
                f"{condition}:run_contracts",
                all(path.is_file() and path.stat().st_size > 0 for path in required_contracts),
                missing=[str(path) for path in required_contracts if not path.is_file()],
            )
            checkpoint = load_checkpoint(run_dir)
            record(
                f"{condition}:checkpoint_round",
                int(checkpoint["completed_round"]) == 5,
                observed=int(checkpoint["completed_round"]),
            )
            record(
                f"{condition}:finite_global_state",
                _finite_tensor_mapping(checkpoint["global_lora_state"]),
            )
            svd_rounds = _csv_rounds(run_dir / "effective_svd_diagnostics.csv")
            record(
                f"{condition}:svd_all_rounds",
                svd_rounds == {1, 2, 3, 4, 5},
                observed=sorted(svd_rounds),
            )
            with (run_dir / "effective_svd_diagnostics.csv").open(
                encoding="utf-8", newline=""
            ) as handle:
                svd_rows = list(csv.DictReader(handle))
            contract = json.loads(
                (run_dir / "effective_svd_contract.json").read_text(encoding="utf-8")
            )
            runtime_scaling = float(checkpoint["runtime"]["aggregation_scaling"])
            checkpoint_scaling = float(checkpoint["aggregation_scaling"])
            contract_scaling = float(contract["common_scaling"])
            svd_values_valid = bool(svd_rows) and all(
                int(row["numerical_rank"]) <= 2
                and math.isfinite(float(row["tail_singular_energy"]))
                and math.isfinite(float(row["projection_norm_error"]))
                and math.isclose(
                    float(row["projection_error_squared"]),
                    float(row["expected_projection_error_squared"]),
                    rel_tol=1e-8,
                    abs_tol=1e-10,
                )
                and float(row["projection_norm_error"]) <= 1e-10
                and math.isclose(
                    float(row["scaling"]), contract_scaling,
                    rel_tol=0.0, abs_tol=1e-12,
                )
                for row in svd_rows
            )
            per_round_svd = {
                round_id: sum(
                    int(row["communication_round"]) == round_id for row in svd_rows
                )
                for round_id in range(1, 6)
            }
            svd_values_valid = (
                svd_values_valid
                and len(set(per_round_svd.values())) == 1
                and next(iter(per_round_svd.values()), 0) == int(contract["matrix_count"])
                and math.isclose(
                    runtime_scaling, checkpoint_scaling,
                    rel_tol=0.0, abs_tol=1e-12,
                )
                and math.isclose(
                    runtime_scaling, contract_scaling,
                    rel_tol=0.0, abs_tol=1e-12,
                )
            )
            record(f"{condition}:svd_numeric_contract", svd_values_valid)
            functional_csv = run_dir / "pfrf" / "client_functional_state.csv"
            record(f"{condition}:functional_log", functional_csv.is_file())
            target_updates_valid = False
            if functional_csv.is_file():
                with functional_csv.open(encoding="utf-8", newline="") as handle:
                    functional_rows = list(csv.DictReader(handle))
                observed_pairs = {
                    (int(row["round"]), int(row["client_id"]))
                    for row in functional_rows
                }
                expected_pairs = {
                    (round_id, client_id)
                    for round_id in range(1, 6)
                    for client_id in range(args.num_users)
                }
                target_updates_valid = (
                    bool(functional_rows)
                    and observed_pairs == expected_pairs
                    and all(
                        all(
                            math.isfinite(float(row[key]))
                            for key in (
                                "f_incoming", "f_proposal", "target_before",
                                "f_upload", "f_next_global",
                            )
                        )
                        for row in functional_rows
                    )
                )
                for row in functional_rows:
                    before = float(row["target_before"])
                    after = float(row["target_after"])
                    incoming = float(row["f_incoming"])
                    proposal = float(row["f_proposal"])
                    if condition == "pfrf_max":
                        expected = max(before, proposal)
                    elif condition == "pfrf_add":
                        expected = min(
                            incoming
                            + max(before - incoming, 0.0)
                            + max(proposal - incoming, 0.0),
                            1.0,
                        )
                    elif condition == "instantaneous_target":
                        expected = proposal
                    else:
                        expected = math.nan
                    if math.isnan(expected):
                        target_updates_valid = target_updates_valid and math.isnan(after)
                    else:
                        target_updates_valid = target_updates_valid and math.isclose(
                            after, expected, rel_tol=1e-6, abs_tol=1e-7
                        )
                    if condition == "pfrf_add":
                        raw = (
                            incoming
                            + max(before - incoming, 0.0)
                            + max(proposal - incoming, 0.0)
                        )
                        target_updates_valid = target_updates_valid and math.isclose(
                            float(row["discarded_target"]),
                            max(raw - 1.0, 0.0),
                            rel_tol=1e-6,
                            abs_tol=1e-7,
                        )
            record(f"{condition}:target_update_contract", target_updates_valid)
            budget_csv = run_dir / "pfrf" / "budget.csv"
            budget_valid = False
            if budget_csv.is_file():
                with budget_csv.open(encoding="utf-8", newline="") as handle:
                    budget_rows = list(csv.DictReader(handle))
                expected_rows = 5 * args.num_users
                budget_valid = len(budget_rows) == expected_rows and all(
                    int(row["scheduler_steps"]) == 3
                    and int(row["optimizer_steps"]) == 3 * int(row["train_batches"])
                    and int(row["memory_backward_calls"])
                    == (0 if condition == "ordinary_ce" else int(row["train_batches"]))
                    for row in budget_rows
                )
            record(f"{condition}:step_budget", budget_valid)
            split_payload = json.loads(
                (run_dir / "client_split_fingerprint.json").read_text(encoding="utf-8")
            )
            split_fingerprints[condition] = (
                split_payload["global_ordered_sha256"],
                split_payload["global_membership_sha256"],
            )
        except Exception as error:
            record(f"{condition}:artifacts", False, error=str(error))

    record(
        "six_condition_split_identity",
        len(split_fingerprints) == len(args.conditions)
        and len(set(split_fingerprints.values())) == 1,
        observed=split_fingerprints,
    )

    for condition in RESUME_CONDITIONS:
        if condition not in args.conditions:
            continue
        try:
            continuous_dir = run_directory(args.output_root, condition)
            resumed_dir = resume_directory(args.output_root, condition, "round5")
            continuous = load_checkpoint(continuous_dir)
            resumed = load_checkpoint(resumed_dir)
            prefix = load_checkpoint(resume_directory(args.output_root, condition, "round2"))
            continuous_round2 = load_checkpoint_file(
                continuous_dir / "pfrf" / "checkpoint_round_002.pt"
            )
            prefix_state_equal, prefix_max_error = _state_close(
                continuous_round2["global_lora_state"], prefix["global_lora_state"]
            )
            prefix_equal = (
                int(continuous_round2["completed_round"]) == 2
                and int(prefix["completed_round"]) == 2
                and prefix_state_equal
                and continuous_round2["runtime"] == prefix["runtime"]
                and _rng_state_equal(
                    continuous_round2["rng_state"], prefix["rng_state"]
                )
            )
            record(
                f"{condition}:continuous_vs_prefix_round2",
                prefix_equal,
                global_max_abs_error=prefix_max_error,
            )
            state_equal, max_error = _state_close(
                continuous["global_lora_state"], resumed["global_lora_state"]
            )
            targets_equal = (
                continuous["runtime"]["targets"] == resumed["runtime"]["targets"]
            )
            evidence_equal = (
                continuous["runtime"]["evidence_store"]
                == resumed["runtime"]["evidence_store"]
            )
            rng_equal = _rng_state_equal(
                continuous["rng_state"], resumed["rng_state"]
            )
            metrics_equal, metric_max_error = _final_metric_state_close(
                continuous_dir, resumed_dir
            )
            record(
                f"{condition}:continuous_vs_resume",
                int(prefix["completed_round"]) == 2
                and int(resumed["completed_round"]) == 5
                and state_equal
                and targets_equal
                and evidence_equal
                and rng_equal
                and metrics_equal,
                # State equivalence without RNG equivalence is not a valid
                # round-boundary resume, even when five rounds happened to
                # reach numerically matching weights.
                global_max_abs_error=max_error,
                targets_equal=targets_equal,
                evidence_equal=evidence_equal,
                rng_equal=rng_equal,
                metrics_equal=metrics_equal,
                metric_max_abs_error=metric_max_error,
            )
        except Exception as error:
            record(f"{condition}:continuous_vs_resume", False, error=str(error))

    report = {
        "schema_version": "pfrf_smoke_report_v1",
        "status": "pass" if all(item["passed"] for item in checks) else "fail",
        "seed": 42,
        "rounds": 5,
        "conditions": list(args.conditions),
        "checks": checks,
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "smoke_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if report["status"] != "pass":
        failed = [item["name"] for item in checks if not item["passed"]]
        raise RuntimeError(f"PFRF smoke gate failed: {failed}")
    return report


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("unit", "commands", "train", "verify", "all"), default="all")
    parser.add_argument("--output-root", type=Path, default=Path("output/pfrf_smoke_v1"))
    parser.add_argument("--data-root", type=Path, default=Path("DATA"))
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--num-users", type=int, default=30)
    parser.add_argument("--conditions", nargs="+", choices=PFRF_CONDITIONS, default=list(PFRF_CONDITIONS))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.num_users != 30:
        raise ValueError("The formal PFRF smoke gate is frozen to 30 clients")
    if args.stage in {"unit", "all"}:
        run_unit_tests(args)
    if args.stage == "commands":
        for item in commands(args):
            print(json.dumps(item, ensure_ascii=False))
        return
    if args.stage in {"train", "all"}:
        run_training(args)
    if args.stage in {"verify", "all"}:
        report = verify(args)
        print(f"PFRF smoke gate: {report['status']}")


if __name__ == "__main__":
    main()
