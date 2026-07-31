#!/usr/bin/env python
"""Cross-platform launcher for the minimal CUSP Oracle Round-1 pilot.

Use the same entry point on Windows and Linux:

  python scripts/cusp_oracle_round1.py --dry-run
  python scripts/cusp_oracle_round1.py --stage synthetic --run
  python scripts/cusp_oracle_round1.py --stage all --run --cuda-visible-devices 0

The full experiment still requires the Linux training environment and CIFAR-100
data.  The synthetic stage is a local smoke test for the offline CUSP path.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def bool_from_env(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def print_command(command: list[str]) -> None:
    print(subprocess.list2cmdline(command))


def require_empty_dir(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise SystemExit(f"Refusing to overwrite non-empty directory: {path}")


def require_file(path: Path) -> None:
    if not path.is_file():
        raise SystemExit(f"Missing required file: {path}")


def preflight(python_bin: str, data_dir: Path) -> None:
    required = ["torch", "torchvision", "tensorboard", "yacs", "cvxpy"]
    missing = [name for name in required if importlib.util.find_spec(name) is None]
    if missing:
        raise SystemExit("Missing required Python dependencies before training: " + ", ".join(missing))

    import cvxpy as cp

    if not cp.installed_solvers():
        raise SystemExit("CVXPY is installed but no solver is available")
    require_file(REPO_ROOT / "configs/datasets/cifar100_LT.yaml")
    require_file(REPO_ROOT / "configs/trainers/PromptFL/vit_b16.yaml")
    if not data_dir.is_dir():
        raise SystemExit(f"Missing DATA directory: {data_dir}")
    print(f"Preflight OK: using {python_bin}")


def build_paths(output_root: Path) -> dict[str, Path]:
    train_dir = output_root / "client-longtail_seed42_round10"
    oracle_dir = output_root / "oracle_client-longtail_seed42_round10"
    return {
        "output_root": output_root,
        "train_dir": train_dir,
        "oracle_dir": oracle_dir,
        "schedule_file": output_root / "shared_client_schedule_seed42_round10.json",
        "dump_dir": train_dir / "oracle_cusp" / "round_010",
    }


def build_train_command(python_bin: str, data_dir: Path, paths: dict[str, Path]) -> list[str]:
    return [
        python_bin, "federated_main.py",
        "--root", str(data_dir),
        "--model", "fedavg",
        "--trainer", "PromptFL",
        "--dataset", "cifar100_LT",
        "--seed", "42",
        "--split_seed", "42",
        "--client_schedule_seed", "42",
        "--client_schedule_file", str(paths["schedule_file"]),
        "--num_users", "30",
        "--frac", "1.0",
        "--round", "10",
        "--local_epochs", "3",
        "--lr", "0.001",
        "--gamma", "1",
        "--n_ctx", "4",
        "--n_general", "1",
        "--ctx_init", "False",
        "--csc", "True",
        "--imb_type", "exp",
        "--imb_factor", "0.01",
        "--train_batch_size", "32",
        "--test_batch_size", "64",
        "--global_eval_interval", "999999",
        "--num_classes", "100",
        "--tail_class_ratio", "0.2",
        "--head_class_ratio", "0.8",
        "--head_client_ratio", "0.9",
        "--tail_client_ratio", "0.1",
        "--specialization_lambda", "0.75",
        "--intra_group_alpha", "0.5",
        "--head_leakage_scale", "3.0",
        "--isolate_local_optimizer_state", "True",
        "--federated_single_scheduler_step", "True",
        "--dataset-config-file", "configs/datasets/cifar100_LT.yaml",
        "--config-file", "configs/trainers/PromptFL/vit_b16.yaml",
        "--partition", "client-longtail",
        "--experimentD_enable", "False",
        "--oracle_cusp_enable", "True",
        "--oracle_cusp_round", "10",
        "--oracle_cusp_cache_train_features", "True",
        "--oracle_cusp_max_train_samples_per_class", "0",
        "--output-dir", str(paths["train_dir"]),
    ]


def build_oracle_command(python_bin: str, paths: dict[str, Path]) -> list[str]:
    return [
        python_bin, "scripts/oracle_cusp_single_round.py",
        "--run-dir", str(paths["train_dir"]),
        "--communication-round", "10",
        "--output-dir", str(paths["oracle_dir"]),
    ]


def build_synthetic_command(python_bin: str, output_dir: Path) -> list[str]:
    return [
        python_bin, "scripts/oracle_cusp_single_round.py",
        "--synthetic-smoke",
        "--output-dir", str(output_dir),
    ]


def run_command(command: list[str], env: dict[str, str]) -> None:
    print_command(command)
    subprocess.run(command, cwd=REPO_ROOT, env=env, check=True)


def verify_dump(dump_dir: Path) -> None:
    from utils.oracle_cusp import load_round_dump, sha256_file

    payload, metadata = load_round_dump(dump_dir)
    del payload
    require_file(dump_dir / "round_state.pt")
    require_file(dump_dir / "train_feature_cache.pt")
    require_file(dump_dir / "metadata.json")
    if metadata.get("round_state_sha256") != sha256_file(dump_dir / "round_state.pt"):
        raise SystemExit("round_state.pt hash mismatch")
    if metadata.get("train_feature_cache_sha256") != sha256_file(dump_dir / "train_feature_cache.pt"):
        raise SystemExit("train_feature_cache.pt hash mismatch")
    print(f"Oracle dump verified: {dump_dir}")


def verify_oracle_outputs(oracle_dir: Path) -> None:
    required = [
        "candidate_states.pt",
        "candidate_manifest.json",
        "oracle_method_summary.csv",
        "oracle_per_class.csv",
        "random_reweight_distribution.csv",
        "oracle_solver.json",
        "oracle_metadata.json",
        "oracle_report.md",
    ]
    missing = [name for name in required if not (oracle_dir / name).is_file()]
    if missing:
        raise SystemExit("Missing oracle outputs: " + ", ".join(missing))

    with (oracle_dir / "oracle_method_summary.csv").open(newline="", encoding="utf-8") as handle:
        methods = {row["method"] for row in csv.DictReader(handle)}
    expected_methods = {"fedavg", "random_reweight", "classwise_weighting", "oracle_cusp"}
    if methods != expected_methods:
        raise SystemExit(f"Unexpected oracle methods: {sorted(methods)}")

    with (oracle_dir / "random_reweight_distribution.csv").open(newline="", encoding="utf-8") as handle:
        count = sum(1 for _ in csv.DictReader(handle))
    if count != 10:
        raise SystemExit(f"Expected 10 random candidates, got {count}")
    print(f"Oracle outputs verified: {oracle_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python-bin", default=os.environ.get("PYTHON_BIN", sys.executable))
    parser.add_argument("--data", type=Path, default=Path(os.environ.get("DATA", "DATA/")))
    parser.add_argument("--output-root", type=Path, default=Path(os.environ.get("OUTPUT_ROOT", "output/cusp_minimal_seed42")))
    parser.add_argument("--stage", choices=["all", "train", "oracle", "synthetic"], default=os.environ.get("STAGE", "all"))
    parser.add_argument("--run", action="store_true", help="actually run commands; default is dry-run")
    parser.add_argument("--dry-run", action="store_true", help="print commands only")
    parser.add_argument("--cuda-visible-devices", default=os.environ.get("CUDA_VISIBLE_DEVICES"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dry_run = args.dry_run or (not args.run and bool_from_env("DRY_RUN", True))
    paths = build_paths(args.output_root)
    env = os.environ.copy()
    if args.cuda_visible_devices is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(args.cuda_visible_devices)

    train_cmd = build_train_command(args.python_bin, args.data, paths)
    oracle_cmd = build_oracle_command(args.python_bin, paths)
    synthetic_dir = args.output_root / "synthetic_oracle_smoke"
    synthetic_cmd = build_synthetic_command(args.python_bin, synthetic_dir)

    print("CUSP minimal pilot")
    print(f"Repo: {REPO_ROOT}")
    print(f"Stage: {args.stage}")
    print(f"Dry run: {dry_run}")
    print(f"Output root: {args.output_root}")
    if args.cuda_visible_devices is not None:
        print(f"CUDA_VISIBLE_DEVICES={args.cuda_visible_devices}")

    if dry_run:
        if args.stage in {"all", "train"}:
            print("Training command:")
            print_command(train_cmd)
        if args.stage in {"all", "oracle"}:
            print("Oracle command:")
            print_command(oracle_cmd)
        if args.stage == "synthetic":
            print("Synthetic smoke command:")
            print_command(synthetic_cmd)
        return

    os.chdir(REPO_ROOT)
    if args.stage == "synthetic":
        require_empty_dir(synthetic_dir)
        synthetic_dir.mkdir(parents=True, exist_ok=True)
        run_command(synthetic_cmd, env)
        verify_oracle_outputs(synthetic_dir)
        return

    preflight(args.python_bin, args.data)

    if args.stage in {"all", "train"}:
        require_empty_dir(paths["train_dir"])
        args.output_root.mkdir(parents=True, exist_ok=True)
        run_command(train_cmd, env)
        verify_dump(paths["dump_dir"])

    if args.stage in {"all", "oracle"}:
        verify_dump(paths["dump_dir"])
        require_empty_dir(paths["oracle_dir"])
        paths["oracle_dir"].mkdir(parents=True, exist_ok=True)
        run_command(oracle_cmd, env)
        verify_oracle_outputs(paths["oracle_dir"])


if __name__ == "__main__":
    main()
