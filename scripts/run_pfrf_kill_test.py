#!/usr/bin/env python
"""Run and validate the six-condition, seed-42 PFRF 100-round kill test."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import subprocess
import sys
import uuid
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_eri_closure import ensure_client_schedule
from scripts.run_pfrf_smoke import (
    _csv_rounds,
    _finite_tensor_mapping,
    _run,
    build_command,
    load_checkpoint,
)
from utils.pfrf import PFRF_CONDITIONS


KILL_ROUNDS = 100
KILL_SEED = 42
KILL_USERS = 30
METRIC_KEYS = (
    "overall_acc",
    "non_tail_acc",
    "bottom20_tail_acc",
    "macro_per_class_acc",
    "macro_f1",
)


def run_directory(root: Path, condition: str) -> Path:
    return root / "runs" / condition / "seed42"


def schedule_path(root: Path) -> Path:
    return root / "protocol" / "schedule_users30_frac1_rounds100_seed42.json"


def _is_within(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def validate_roots(output_root: Path, smoke_root: Path) -> None:
    if _is_within(output_root, smoke_root) or _is_within(smoke_root, output_root):
        raise ValueError("Kill-test and smoke output roots must be disjoint")


def load_smoke_gate(smoke_root: Path) -> dict:
    path = smoke_root / "smoke_report.json"
    if not path.is_file():
        raise FileNotFoundError(f"Smoke gate report does not exist: {path}")
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("schema_version") != "pfrf_smoke_report_v1":
        raise ValueError("Unexpected PFRF smoke report schema")
    if report.get("status") != "pass":
        raise ValueError("PFRF smoke gate did not pass")
    if tuple(report.get("conditions", ())) != tuple(PFRF_CONDITIONS):
        raise ValueError("PFRF smoke gate did not cover the frozen six conditions")
    if int(report.get("rounds", -1)) != 5 or int(report.get("seed", -1)) != KILL_SEED:
        raise ValueError("PFRF smoke gate has the wrong seed or round count")
    checks = report.get("checks", [])
    if not checks or any(not bool(item.get("passed")) for item in checks):
        raise ValueError("PFRF smoke report contains a failed correctness check")
    return report


def commands(args) -> list[dict]:
    schedule = schedule_path(args.output_root)
    result = []
    for condition in args.conditions:
        command = build_command(
            args,
            condition=condition,
            output_dir=run_directory(args.output_root, condition),
            rounds=KILL_ROUNDS,
            schedule=schedule,
        )
        if "--resume" in command:
            raise RuntimeError("Formal kill-test commands must start from scratch")
        result.append({"label": condition, "command": command})
    return result


def gpu_assignments(args) -> dict[str, int | None]:
    gpu_ids = list(getattr(args, "gpu_ids", []) or [])
    if not gpu_ids:
        return {condition: None for condition in args.conditions}
    gpu_ids = [int(gpu_id) for gpu_id in gpu_ids]
    if len(gpu_ids) != len(args.conditions):
        raise ValueError(
            "--gpu-ids must contain exactly one GPU id per selected condition"
        )
    if len(set(gpu_ids)) != len(gpu_ids) or any(gpu_id < 0 for gpu_id in gpu_ids):
        raise ValueError("--gpu-ids must be distinct non-negative integers")
    return dict(zip(args.conditions, gpu_ids))


def child_environment(gpu_id: int) -> dict[str, str]:
    environment = os.environ.copy()
    environment["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    environment["CUDA_VISIBLE_DEVICES"] = str(int(gpu_id))
    return environment


def prepare(args) -> Path:
    validate_roots(args.output_root, args.smoke_root)
    smoke_report = load_smoke_gate(args.smoke_root)
    ensure_client_schedule(
        schedule_path(args.output_root),
        rounds=KILL_ROUNDS,
        users=KILL_USERS,
        frac=1.0,
        seed=KILL_SEED,
    )
    smoke_bytes = (args.smoke_root / "smoke_report.json").read_bytes()
    manifest = {
        "schema_version": "pfrf_kill_protocol_v1",
        "status": "prepared",
        "seed": KILL_SEED,
        "rounds": KILL_ROUNDS,
        "num_users": KILL_USERS,
        "frac": 1.0,
        "conditions": list(PFRF_CONDITIONS),
        "fresh_initialization": True,
        "resume_from_smoke": False,
        "gpu_assignment": "recorded_per_condition_in_kill_command.json",
        "smoke_report": str(args.smoke_root / "smoke_report.json"),
        "smoke_report_sha256": hashlib.sha256(smoke_bytes).hexdigest(),
        "smoke_status": smoke_report["status"],
        "schedule": str(schedule_path(args.output_root)),
    }
    path = args.output_root / "protocol" / "kill_protocol.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        f"{path.name}.tmp.{os.getpid()}.{uuid.uuid4().hex}"
    )
    temporary.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)
    return path


def _prepare_launch(item: dict, assigned_gpu: int | None) -> dict | None:
    command = item["command"]
    output_index = command.index("--output-dir") + 1
    output_dir = Path(command[output_index])
    checkpoint = output_dir / "pfrf" / "checkpoint_latest.pt"
    if checkpoint.is_file():
        payload = load_checkpoint(output_dir)
        if int(payload["completed_round"]) == KILL_ROUNDS:
            print(f"Skip complete PFRF kill-test run: {item['label']}")
            return None
    if output_dir.exists() and any(output_dir.iterdir()):
        raise RuntimeError(
            f"Refusing to append to incomplete kill-test output {output_dir}; "
            "move it aside before rerunning"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    invocation = {
        "schema_version": "pfrf_kill_command_v1",
        "condition": item["label"],
        "fresh_initialization": True,
        "resume": None,
        "assigned_physical_gpu": assigned_gpu,
        "inherited_cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "command": command,
    }
    (output_dir / "kill_command.json").write_text(
        json.dumps(invocation, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return {
        "condition": item["label"],
        "command": command,
        "output_dir": output_dir,
        "gpu": assigned_gpu,
    }


def _run_parallel(launches: list[dict]) -> None:
    running = []
    try:
        for launch in launches:
            environment = child_environment(launch["gpu"])
            log_path = launch["output_dir"] / "launcher.log"
            log_handle = log_path.open("w", encoding="utf-8")
            print(
                f"Launch {launch['condition']} on physical GPU {launch['gpu']}; "
                f"log={log_path}"
            )
            try:
                process = subprocess.Popen(
                    launch["command"],
                    cwd=REPO_ROOT,
                    env=environment,
                    stdout=log_handle,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
            except Exception:
                log_handle.close()
                raise
            running.append({**launch, "process": process, "log_handle": log_handle})

        failures = []
        for launch in running:
            return_code = launch["process"].wait()
            launch["log_handle"].close()
            print(
                f"Finish {launch['condition']} on GPU {launch['gpu']}: "
                f"return_code={return_code}"
            )
            if return_code != 0:
                failures.append(
                    f"{launch['condition']} (GPU {launch['gpu']}, "
                    f"log {launch['output_dir'] / 'launcher.log'})"
                )
        if failures:
            raise RuntimeError("Parallel PFRF kill-test failures: " + "; ".join(failures))
    except BaseException:
        for launch in running:
            if launch["process"].poll() is None:
                launch["process"].terminate()
        for launch in running:
            if launch["process"].poll() is None:
                launch["process"].wait()
            if not launch["log_handle"].closed:
                launch["log_handle"].close()
        raise


def run_training(args) -> None:
    prepare(args)
    assignments = gpu_assignments(args)
    launches = [
        launch
        for item in commands(args)
        if (
            launch := _prepare_launch(
                item, assignments[item["label"]]
            )
        ) is not None
    ]
    if not launches:
        return
    if any(launch["gpu"] is not None for launch in launches):
        _run_parallel(launches)
        return
    for launch in launches:
        print(f"Run PFRF 100-round kill test: {launch['condition']}")
        _run(launch["command"])


def _read_csv(path: Path) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _verify_condition(args, condition: str) -> tuple[list[dict], dict | None]:
    checks = []

    def record(name: str, passed: bool, **details):
        checks.append({"name": name, "passed": bool(passed), **details})

    run_dir = run_directory(args.output_root, condition)
    checkpoint = load_checkpoint(run_dir)
    record(
        f"{condition}:checkpoint_round_100",
        int(checkpoint["completed_round"]) == KILL_ROUNDS,
        observed=int(checkpoint["completed_round"]),
    )
    record(
        f"{condition}:finite_global_lora",
        _finite_tensor_mapping(checkpoint["global_lora_state"]),
    )

    invocation = json.loads((run_dir / "kill_command.json").read_text(encoding="utf-8"))
    command = invocation.get("command", [])
    fresh = (
        invocation.get("fresh_initialization") is True
        and invocation.get("resume") is None
        and "--resume" not in command
        and int(command[command.index("--round") + 1]) == KILL_ROUNDS
    )
    record(f"{condition}:fresh_initialization", fresh)

    required = (
        run_dir / "resolved_config.yaml",
        run_dir / "client_split_fingerprint.json",
        run_dir / "cliplora_initialization_audit.json",
        run_dir / "effective_svd_contract.json",
        run_dir / "effective_svd_diagnostics.csv",
        run_dir / "round_metrics.csv",
        run_dir / "per_class_accuracy_epoch_99.csv",
        run_dir / "pfrf" / "runtime_contract.json",
        run_dir / "pfrf" / "client_functional_state.csv",
        run_dir / "pfrf" / "budget.csv",
    )
    record(
        f"{condition}:required_artifacts",
        all(path.is_file() and path.stat().st_size > 0 for path in required),
        missing=[str(path) for path in required if not path.is_file()],
    )

    svd_path = run_dir / "effective_svd_diagnostics.csv"
    svd_rows = _read_csv(svd_path)
    contract = json.loads(
        (run_dir / "effective_svd_contract.json").read_text(encoding="utf-8")
    )
    expected_svd_rows = KILL_ROUNDS * int(contract["matrix_count"])
    scaling = float(contract["common_scaling"])
    svd_valid = (
        _csv_rounds(svd_path) == set(range(1, KILL_ROUNDS + 1))
        and len(svd_rows) == expected_svd_rows
        and all(
            int(row["numerical_rank"]) <= 2
            and math.isfinite(float(row["projection_error_squared"]))
            and math.isfinite(float(row["expected_projection_error_squared"]))
            and math.isclose(
                float(row["projection_error_squared"]),
                float(row["expected_projection_error_squared"]),
                rel_tol=1e-8,
                abs_tol=1e-10,
            )
            and float(row["projection_norm_error"]) <= 1e-10
            and math.isclose(
                float(row["scaling"]), scaling, rel_tol=0.0, abs_tol=1e-12
            )
            for row in svd_rows
        )
    )
    record(
        f"{condition}:svd_100_round_contract",
        svd_valid,
        observed_rows=len(svd_rows),
        expected_rows=expected_svd_rows,
    )

    budget_rows = _read_csv(run_dir / "pfrf" / "budget.csv")
    budget_pairs = {
        (int(row["round"]), int(row["client_id"])) for row in budget_rows
    }
    expected_pairs = {
        (round_id, client_id)
        for round_id in range(1, KILL_ROUNDS + 1)
        for client_id in range(KILL_USERS)
    }
    budget_valid = (
        len(budget_rows) == KILL_ROUNDS * KILL_USERS
        and budget_pairs == expected_pairs
        and all(
            int(row["scheduler_steps"]) == 3
            and int(row["optimizer_steps"]) == 3 * int(row["train_batches"])
            and int(row["memory_backward_calls"])
            == (0 if condition == "ordinary_ce" else int(row["train_batches"]))
            for row in budget_rows
        )
    )
    record(f"{condition}:step_budget_100_rounds", budget_valid)

    functional_rows = _read_csv(run_dir / "pfrf" / "client_functional_state.csv")
    functional_pairs = {
        (int(row["round"]), int(row["client_id"])) for row in functional_rows
    }
    functional_valid = (
        functional_pairs == expected_pairs
        and all(
            all(
                math.isfinite(float(row[key]))
                for key in (
                    "f_incoming",
                    "f_proposal",
                    "target_before",
                    "f_upload",
                    "f_next_global",
                )
            )
            for row in functional_rows
        )
    )
    record(
        f"{condition}:functional_state_100_rounds",
        functional_valid,
        observed_rows=len(functional_rows),
    )

    metric_rows = _read_csv(run_dir / "round_metrics.csv")
    training_metrics = {
        int(row["epoch"]): row for row in metric_rows if int(row["epoch"]) >= 0
    }
    metrics_valid = (
        set(training_metrics) == set(range(KILL_ROUNDS))
        and all(
            all(math.isfinite(float(row[key])) for key in METRIC_KEYS)
            for row in training_metrics.values()
        )
    )
    record(f"{condition}:metrics_100_rounds", metrics_valid)

    per_class_rows = _read_csv(run_dir / "per_class_accuracy_epoch_99.csv")
    final_classes = {int(row["class_id"]) for row in per_class_rows}
    per_class_valid = (
        len(per_class_rows) == 100
        and final_classes == set(range(100))
        and all(math.isfinite(float(row["per_class_acc"])) for row in per_class_rows)
    )
    record(f"{condition}:final_per_class_metrics", per_class_valid)

    identity = {
        "split": json.loads(
            (run_dir / "client_split_fingerprint.json").read_text(encoding="utf-8")
        )["global_ordered_sha256"],
        "initial_lora": json.loads(
            (run_dir / "cliplora_initialization_audit.json").read_text(encoding="utf-8")
        )["initial_lora_sha256"],
        "assigned_physical_gpu": invocation.get("assigned_physical_gpu"),
    }
    return checks, identity


def _metric_summary(args) -> list[dict]:
    summaries = []
    for condition in PFRF_CONDITIONS:
        rows = _read_csv(run_directory(args.output_root, condition) / "round_metrics.csv")
        rows = sorted(
            (row for row in rows if int(row["epoch"]) >= 0),
            key=lambda row: int(row["epoch"]),
        )
        if len(rows) != KILL_ROUNDS:
            raise ValueError(f"{condition} does not contain 100 training metric rows")
        summary = {"condition": condition}
        for key in METRIC_KEYS:
            values = [float(row[key]) for row in rows]
            summary[f"final_{key}"] = values[-1]
            summary[f"late81_100_mean_{key}"] = sum(values[80:]) / 20.0
            summary[f"round1_100_mean_{key}"] = sum(values) / KILL_ROUNDS
        summaries.append(summary)
    return summaries


def write_metric_summary(args) -> Path:
    rows = _metric_summary(args)
    by_condition = {row["condition"]: row for row in rows}
    controls = (
        "matched_memory_ce",
        "class_reweighted_ce",
        "instantaneous_target",
    )
    max_row = by_condition["pfrf_max"]
    ordinary = by_condition["ordinary_ce"]
    best_control_final = max(
        by_condition[name]["final_bottom20_tail_acc"] for name in controls
    )
    best_control_late = max(
        by_condition[name]["late81_100_mean_bottom20_tail_acc"] for name in controls
    )
    screening = {
        "pfrf_max_final_tail_gain_vs_best_control_pp": (
            max_row["final_bottom20_tail_acc"] - best_control_final
        ),
        "pfrf_max_late_tail_gain_vs_best_control_pp": (
            max_row["late81_100_mean_bottom20_tail_acc"] - best_control_late
        ),
        "pfrf_max_final_non_tail_change_vs_ordinary_pp": (
            max_row["final_non_tail_acc"] - ordinary["final_non_tail_acc"]
        ),
        "pfrf_max_final_overall_change_vs_ordinary_pp": (
            max_row["final_overall_acc"] - ordinary["final_overall_acc"]
        ),
        "primary_gain_at_least_1pp": (
            max_row["final_bottom20_tail_acc"] - best_control_final >= 1.0
        ),
        "late_gain_positive": (
            max_row["late81_100_mean_bottom20_tail_acc"] - best_control_late > 0.0
        ),
        "non_tail_drop_within_1pp": (
            max_row["final_non_tail_acc"] - ordinary["final_non_tail_acc"] >= -1.0
        ),
        "overall_drop_within_1pp": (
            max_row["final_overall_acc"] - ordinary["final_overall_acc"] >= -1.0
        ),
        "scientific_verdict": "pending_external_probe_and_additional_seeds",
    }
    analysis_dir = args.output_root / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    csv_path = analysis_dir / "six_condition_metrics.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (analysis_dir / "screening_summary.json").write_text(
        json.dumps(screening, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return csv_path


def verify(args) -> dict:
    validate_roots(args.output_root, args.smoke_root)
    load_smoke_gate(args.smoke_root)
    checks = []
    identities = {}
    for condition in PFRF_CONDITIONS:
        try:
            condition_checks, identity = _verify_condition(args, condition)
            checks.extend(condition_checks)
            identities[condition] = identity
        except Exception as error:
            checks.append(
                {
                    "name": f"{condition}:artifacts",
                    "passed": False,
                    "error": str(error),
                }
            )
    checks.append(
        {
            "name": "six_condition_split_and_initialization_identity",
            "passed": (
                len(identities) == len(PFRF_CONDITIONS)
                and len({item["split"] for item in identities.values()}) == 1
                and len({item["initial_lora"] for item in identities.values()}) == 1
            ),
            "observed": identities,
        }
    )
    assigned_gpus = [item["assigned_physical_gpu"] for item in identities.values()]
    checks.append(
        {
            "name": "parallel_gpu_assignment",
            "passed": (
                all(item is None for item in assigned_gpus)
                or (
                    all(item is not None for item in assigned_gpus)
                    and len(set(int(item) for item in assigned_gpus))
                    == len(PFRF_CONDITIONS)
                )
            ),
            "observed": assigned_gpus,
        }
    )
    status = "pass" if all(item["passed"] for item in checks) else "fail"
    report = {
        "schema_version": "pfrf_kill_test_report_v1",
        "status": status,
        "seed": KILL_SEED,
        "rounds": KILL_ROUNDS,
        "conditions": list(PFRF_CONDITIONS),
        "fresh_initialization": True,
        "checks": checks,
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "kill_test_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if status != "pass":
        failed = [item["name"] for item in checks if not item["passed"]]
        raise RuntimeError(f"PFRF kill-test validation failed: {failed}")
    write_metric_summary(args)
    return report


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage",
        choices=("prepare", "commands", "train", "verify", "all"),
        default="all",
    )
    parser.add_argument("--output-root", type=Path, default=Path("output/pfrf_kill_v1"))
    parser.add_argument("--smoke-root", type=Path, default=Path("output/pfrf_smoke_v1"))
    parser.add_argument("--data-root", type=Path, default=Path("DATA"))
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--num-users", type=int, default=KILL_USERS)
    parser.add_argument(
        "--condition",
        choices=PFRF_CONDITIONS,
        default=None,
        help="run exactly one condition in a scheduler-provided single-GPU allocation",
    )
    parser.add_argument(
        "--gpu-ids",
        nargs="+",
        type=int,
        default=None,
        help="one physical GPU id per selected condition; enables parallel launch",
    )
    parser.add_argument(
        "--conditions",
        nargs="+",
        choices=PFRF_CONDITIONS,
        default=None,
    )
    args = parser.parse_args()
    if args.condition is not None and args.conditions is not None:
        parser.error("--condition and --conditions cannot be used together")
    if args.condition is not None:
        args.conditions = [args.condition]
    elif args.conditions is None:
        args.conditions = list(PFRF_CONDITIONS)
    return args


def main() -> None:
    args = parse_args()
    if args.num_users != KILL_USERS:
        raise ValueError("The formal PFRF kill test is frozen to 30 clients")
    if (
        args.stage == "train"
        and args.condition is not None
        and not args.gpu_ids
        and torch.cuda.device_count() != 1
    ):
        raise RuntimeError(
            "--condition expects the scheduler allocation to expose exactly one GPU; "
            f"torch sees {torch.cuda.device_count()}"
        )
    if args.stage in {"verify", "all"} and tuple(args.conditions) != tuple(PFRF_CONDITIONS):
        raise ValueError("Formal kill-test verification requires all six conditions")
    if args.stage == "prepare":
        print(f"Prepared PFRF kill-test protocol: {prepare(args)}")
        return
    if args.stage == "commands":
        prepare(args)
        for item in commands(args):
            print(json.dumps(item, ensure_ascii=False))
        return
    if args.stage in {"train", "all"}:
        run_training(args)
    if args.stage in {"verify", "all"}:
        report = verify(args)
        print(f"PFRF 100-round kill-test validation: {report['status']}")


if __name__ == "__main__":
    main()
