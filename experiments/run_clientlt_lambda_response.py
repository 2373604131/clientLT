"""Experiment C: response of existing methods to ClientLT(lambda_T, alpha_T).

This orchestrator does not implement a new method. It only launches the
existing Zero-shot CLIP, PromptFL, and CAPT entry points with matched
Client-LT split parameters, then normalizes their outputs into the CSV/JSON
files and figures needed by Experiment C.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[1]

METHOD_SPECS = {
    "zeroshot": {
        "display": "Zero-shot CLIP",
        "trainer": "CLIP",
        "model": "fedavg",
        "config_attr": "clip_config",
        "extra_args": ["--no-train"],
    },
    "promptfl": {
        "display": "PromptFL",
        "trainer": "PromptFL",
        "model": "fedavg",
        "config_attr": "promptfl_config",
        "extra_args": [],
    },
    "capt": {
        "display": "CAPT",
        "trainer": "CAPT",
        "model": "cluster",
        "config_attr": "capt_config",
        "extra_args": [],
    },
}

METHOD_ORDER = ["zeroshot", "promptfl", "capt"]
METHOD_COLORS = {
    "zeroshot": "#4C78A8",
    "promptfl": "#F58518",
    "capt": "#54A24B",
}
METHOD_MARKERS = {
    "zeroshot": "o",
    "promptfl": "s",
    "capt": "^",
}

RUN_FIELDNAMES = [
    "method",
    "method_display",
    "lambda_T",
    "alpha_T",
    "seed",
    "final_overall_acc",
    "final_head_acc",
    "final_tail_acc",
    "final_macro_acc",
    "best_overall_acc",
    "best_tail_acc",
    "tail_peak_to_final_drop",
    "tail_client_purity",
    "head_leakage_to_tail_clients",
    "tail_leakage_to_head_clients",
    "tail_active_rounds_mean",
    "tail_active_rounds_std",
    "exposure_interval_mean",
    "exposure_interval_std",
    "max_exposure_gap_mean",
    "max_exposure_gap_std",
    "tail_top1_mass_mean",
    "tail_top2_mass_mean",
    "effective_client_number_mean",
    "normalized_entropy_mean",
    "output_dir",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run or collect Experiment C: ClientLT(lambda_T, alpha_T) method response."
    )
    parser.add_argument("--datadir", default="./DATA", help="Dataset root passed to federated_main.py --root.")
    parser.add_argument("--output_dir", default="output/expC_lambda_response")
    parser.add_argument("--dataset", default="cifar100_LT")
    parser.add_argument("--dataset_config_file", default="")
    parser.add_argument("--methods", nargs="+", default=["zeroshot", "promptfl", "capt"], choices=METHOD_ORDER)
    parser.add_argument("--lambda_values", type=float, nargs="+", default=[0.0, 0.25, 0.5, 0.75, 1.0])
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--num_clients", type=int, default=50)
    parser.add_argument("--tail_client_ratio", type=float, default=0.1)
    parser.add_argument("--tail_class_ratio", type=float, default=0.2)
    parser.add_argument("--intra_group_alpha", type=float, default=0.1)
    parser.add_argument("--head_leakage_scale", type=float, default=3.0)
    parser.add_argument("--imb_factor", type=float, default=0.01)
    parser.add_argument("--imb_type", default="exp")
    parser.add_argument("--num_classes", type=int, default=100)
    parser.add_argument("--rounds", type=int, default=100)
    parser.add_argument("--frac", type=float, default=0.4)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--n_ctx", type=int, default=4)
    parser.add_argument("--n_general", type=int, default=1)
    parser.add_argument("--train_batch_size", type=int, default=32)
    parser.add_argument("--test_batch_size", type=int, default=64)
    parser.add_argument(
        "--global_eval_interval",
        type=int,
        default=5,
        help="Global evaluation interval. Epoch 0 and the final round are always evaluated by federated_main.py.",
    )
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--ctx_init", default="False")
    parser.add_argument("--csc", default="True")
    parser.add_argument("--promptfl_config", default="configs/trainers/PromptFL/vit_b16.yaml")
    parser.add_argument("--capt_config", default="configs/trainers/CAPT/vit_b16.yaml")
    parser.add_argument("--clip_config", default="configs/trainers/PromptFL/vit_b16.yaml")
    parser.add_argument("--federated_entry", default="federated_main.py")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--gpu", default="", help="Optional CUDA_VISIBLE_DEVICES value for launched runs.")
    parser.add_argument("--extra_opts", nargs=argparse.REMAINDER, default=None)
    parser.add_argument("--dry_run", action="store_true", help="Print commands only.")
    parser.add_argument("--run", action="store_true", help="Execute commands and then collect results.")
    parser.add_argument("--collect_only", action="store_true", help="Only collect existing run directories.")
    parser.add_argument("--overwrite", action="store_true", help="Rerun commands even if metrics.json already exists.")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    modes = int(args.dry_run) + int(args.run) + int(args.collect_only)
    if modes != 1:
        raise SystemExit("Choose exactly one mode: --dry_run, --run, or --collect_only.")
    if not 0.0 <= args.tail_client_ratio <= 1.0:
        raise ValueError("--tail_client_ratio must be in [0, 1]")
    if not 0.0 <= args.tail_class_ratio <= 1.0:
        raise ValueError("--tail_class_ratio must be in [0, 1]")
    if args.intra_group_alpha <= 0:
        raise ValueError("--intra_group_alpha must be > 0")
    if args.head_leakage_scale < 0:
        raise ValueError("--head_leakage_scale must be >= 0")
    for value in args.lambda_values:
        if not 0.0 <= float(value) <= 1.0:
            raise ValueError(f"lambda_T must be in [0, 1], got {value}")
    if args.extra_opts is None:
        args.extra_opts = []


def lambda_dir_name(lambda_t: float) -> str:
    return f"lambda={float(lambda_t):.2f}"


def run_dir_for(args: argparse.Namespace, method: str, lambda_t: float, seed: int) -> Path:
    return (
        Path(args.output_dir)
        / f"method={method}"
        / lambda_dir_name(lambda_t)
        / f"seed={int(seed)}"
    )


def dataset_config(args: argparse.Namespace) -> str:
    if args.dataset_config_file:
        return args.dataset_config_file
    return f"configs/datasets/{args.dataset}.yaml"


def quote_cmd(cmd: list[str], gpu: str = "") -> str:
    import shlex

    text = " ".join(shlex.quote(str(x)) for x in cmd)
    if gpu:
        return f"CUDA_VISIBLE_DEVICES={shlex.quote(str(gpu))} {text}"
    return text


def build_command(args: argparse.Namespace, method: str, lambda_t: float, seed: int) -> list[str]:
    spec = METHOD_SPECS[method]
    run_dir = run_dir_for(args, method, lambda_t, seed)
    schedule_file = run_dir / "selected_clients_per_round.json"
    head_client_ratio = 1.0 - float(args.tail_client_ratio)
    head_class_ratio = 1.0 - float(args.tail_class_ratio)
    config_file = getattr(args, spec["config_attr"])

    cmd = [
        args.python,
        args.federated_entry,
        "--root",
        str(args.datadir),
        "--model",
        str(spec["model"]),
        "--dataset",
        str(args.dataset),
        "--seed",
        str(seed),
        "--num_users",
        str(args.num_clients),
        "--frac",
        str(args.frac),
        "--lr",
        str(args.lr),
        "--csc",
        str(args.csc),
        "--gamma",
        str(args.gamma),
        "--trainer",
        str(spec["trainer"]),
        "--round",
        str(args.rounds),
        "--partition",
        "client-longtail",
        "--beta",
        str(args.beta),
        "--n_ctx",
        str(args.n_ctx),
        "--dataset-config-file",
        dataset_config(args),
        "--config-file",
        str(config_file),
        "--output-dir",
        str(run_dir),
        "--imb_factor",
        str(args.imb_factor),
        "--imb_type",
        str(args.imb_type),
        "--ctx_init",
        str(args.ctx_init),
        "--train_batch_size",
        str(args.train_batch_size),
        "--test_batch_size",
        str(args.test_batch_size),
        "--global_eval_interval",
        str(args.global_eval_interval),
        "--num_classes",
        str(args.num_classes),
        "--n_general",
        str(args.n_general),
        "--head_client_ratio",
        str(head_client_ratio),
        "--tail_client_ratio",
        str(args.tail_client_ratio),
        "--head_class_ratio",
        str(head_class_ratio),
        "--tail_class_ratio",
        str(args.tail_class_ratio),
        "--specialization_lambda",
        str(lambda_t),
        "--intra_group_alpha",
        str(args.intra_group_alpha),
        "--head_leakage_scale",
        str(args.head_leakage_scale),
        "--client_schedule_file",
        str(schedule_file),
        "--client_schedule_seed",
        str(seed),
    ]
    cmd.extend(spec["extra_args"])
    cmd.extend(["DATALOADER.NUM_WORKERS", str(args.num_workers)])
    cmd.extend(args.extra_opts)
    return cmd


def get_git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def as_float(row: dict[str, Any], *names: str, default: float = float("nan")) -> float:
    for name in names:
        value = row.get(name)
        if value not in (None, ""):
            try:
                return float(value)
            except (TypeError, ValueError):
                continue
    return default


def as_int(row: dict[str, Any], *names: str, default: int = 0) -> int:
    value = as_float(row, *names, default=float(default))
    if math.isnan(value):
        return default
    return int(round(value))


def load_counts(path: Path) -> np.ndarray | None:
    rows = read_csv_rows(path)
    if not rows:
        return None
    class_cols = [col for col in rows[0] if col.startswith("class_")]
    class_cols.sort(key=lambda name: int(name.split("_", 1)[1]))
    matrix = []
    for row in rows:
        matrix.append([int(float(row[col])) for col in class_cols])
    return np.asarray(matrix, dtype=np.float64)


def head_tail_clients(num_clients: int, tail_client_ratio: float) -> tuple[list[int], list[int]]:
    head_count = int(num_clients * (1.0 - float(tail_client_ratio)))
    return list(range(head_count)), list(range(head_count, num_clients))


def head_tail_classes(num_classes: int, tail_class_ratio: float) -> tuple[list[int], list[int]]:
    head_count = int(num_classes * (1.0 - float(tail_class_ratio)))
    return list(range(head_count)), list(range(head_count, num_classes))


def safe_div(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return float("nan")
    return float(numerator) / float(denominator)


def max_zero_run(active: list[bool]) -> int:
    best = 0
    current = 0
    for item in active:
        if item:
            current = 0
        else:
            current += 1
            best = max(best, current)
    return best


def exposure_interval(active: list[bool]) -> float:
    indices = [idx for idx, item in enumerate(active) if item]
    if len(indices) < 2:
        return float("nan")
    gaps = [indices[idx + 1] - indices[idx] for idx in range(len(indices) - 1)]
    return float(np.mean(gaps))


def load_schedule(path: Path) -> list[list[int]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    schedule = payload.get("schedule", payload) if isinstance(payload, dict) else payload
    return [[int(x) for x in row] for row in schedule]


def normalized_entropy(counts: np.ndarray, denominator_clients: int) -> float:
    total = float(counts.sum())
    if total <= 0:
        return float("nan")
    probs = counts[counts > 0] / total
    entropy = float(-(probs * np.log(probs)).sum())
    if denominator_clients <= 1:
        return 0.0
    return entropy / float(np.log(denominator_clients))


def nanmean(values: list[float]) -> float:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0 or np.all(np.isnan(arr)):
        return float("nan")
    return float(np.nanmean(arr))


def nanstd(values: list[float]) -> float:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0 or np.all(np.isnan(arr)):
        return float("nan")
    return float(np.nanstd(arr))


def compute_topology_metrics(
    counts: np.ndarray | None,
    schedule: list[list[int]],
    args: argparse.Namespace,
) -> dict[str, float]:
    metrics = {
        "tail_client_purity": float("nan"),
        "head_leakage_to_tail_clients": float("nan"),
        "tail_leakage_to_head_clients": float("nan"),
        "tail_active_rounds_mean": float("nan"),
        "tail_active_rounds_std": float("nan"),
        "exposure_interval_mean": float("nan"),
        "exposure_interval_std": float("nan"),
        "max_exposure_gap_mean": float("nan"),
        "max_exposure_gap_std": float("nan"),
        "tail_top1_mass_mean": float("nan"),
        "tail_top2_mass_mean": float("nan"),
        "effective_client_number_mean": float("nan"),
        "normalized_entropy_mean": float("nan"),
    }
    if counts is None:
        return metrics

    num_clients, num_classes = counts.shape
    head_clients, tail_clients = head_tail_clients(num_clients, args.tail_client_ratio)
    head_classes, tail_classes = head_tail_classes(num_classes, args.tail_class_ratio)

    tail_client_total = float(counts[np.ix_(tail_clients, range(num_classes))].sum())
    tail_on_tail_clients = float(counts[np.ix_(tail_clients, tail_classes)].sum())
    head_on_tail_clients = float(counts[np.ix_(tail_clients, head_classes)].sum())
    tail_on_head_clients = float(counts[np.ix_(head_clients, tail_classes)].sum())
    head_global_total = float(counts[:, head_classes].sum())
    tail_global_total = float(counts[:, tail_classes].sum())
    metrics.update(
        {
            "tail_client_purity": safe_div(tail_on_tail_clients, tail_client_total),
            "head_leakage_to_tail_clients": safe_div(head_on_tail_clients, head_global_total),
            "tail_leakage_to_head_clients": safe_div(tail_on_head_clients, tail_global_total),
        }
    )

    top1_values = []
    top2_values = []
    effective_values = []
    entropy_values = []
    active_rounds = []
    max_gaps = []
    intervals = []
    support = counts > 0
    for class_id in tail_classes:
        class_counts = counts[:, class_id].astype(np.float64)
        total = float(class_counts.sum())
        if total <= 0:
            continue
        sorted_counts = np.sort(class_counts)[::-1]
        top1_values.append(float(sorted_counts[0] / total))
        top2_values.append(float(sorted_counts[:2].sum() / total))
        denom = float((class_counts ** 2).sum())
        effective_values.append(float((total ** 2) / denom) if denom > 0 else float("nan"))
        entropy_values.append(normalized_entropy(class_counts, num_clients))
        if schedule:
            active = [bool(support[np.asarray(selected, dtype=int), class_id].any()) for selected in schedule]
            active_rounds.append(float(sum(active)))
            max_gaps.append(float(max_zero_run(active)))
            intervals.append(exposure_interval(active))

    metrics.update(
        {
            "tail_top1_mass_mean": nanmean(top1_values),
            "tail_top2_mass_mean": nanmean(top2_values),
            "effective_client_number_mean": nanmean(effective_values),
            "normalized_entropy_mean": nanmean(entropy_values),
        }
    )
    if schedule:
        metrics.update(
            {
                "tail_active_rounds_mean": nanmean(active_rounds),
                "tail_active_rounds_std": nanstd(active_rounds),
                "exposure_interval_mean": nanmean(intervals),
                "exposure_interval_std": nanstd(intervals),
                "max_exposure_gap_mean": nanmean(max_gaps),
                "max_exposure_gap_std": nanstd(max_gaps),
            }
        )
    return metrics


def find_final_per_class_file(run_dir: Path, final_round: int) -> Path | None:
    exact = run_dir / f"per_class_accuracy_epoch_{final_round}.csv"
    if exact.exists():
        return exact
    candidates = sorted(run_dir.glob("per_class_accuracy_epoch_*.csv"))
    if not candidates:
        return None
    return candidates[-1]


def normalize_round_metrics(run_dir: Path) -> tuple[list[dict[str, Any]], int]:
    rows = read_csv_rows(run_dir / "round_metrics.csv")
    normalized = []
    for row in rows:
        round_id = as_int(row, "round", "epoch")
        normalized.append(
            {
                "round": round_id,
                "overall_acc": as_float(row, "overall_acc"),
                "head_acc": as_float(row, "head_acc", "non_tail_acc"),
                "tail_acc": as_float(row, "tail_acc", "bottom20_tail_acc"),
                "macro_acc": as_float(row, "macro_acc", "macro_per_class_acc"),
            }
        )
    normalized.sort(key=lambda item: int(item["round"]))
    write_csv(run_dir / "per_round_metrics.csv", normalized, ["round", "overall_acc", "head_acc", "tail_acc", "macro_acc"])
    final_round = int(normalized[-1]["round"]) if normalized else -1
    return normalized, final_round


def write_per_class_metrics(
    run_dir: Path,
    final_round: int,
    counts: np.ndarray | None,
    args: argparse.Namespace,
) -> None:
    source = find_final_per_class_file(run_dir, final_round)
    if source is None:
        return
    rows = read_csv_rows(source)
    num_classes = args.num_classes
    head_classes, tail_classes = head_tail_classes(num_classes, args.tail_class_ratio)
    head_set = set(head_classes)
    tail_set = set(tail_classes)
    global_counts = counts.sum(axis=0) if counts is not None else np.full(num_classes, np.nan)
    support_counts = (counts > 0).sum(axis=0) if counts is not None else np.full(num_classes, np.nan)

    out = []
    for row in rows:
        class_id = as_int(row, "class_id")
        if class_id in head_set:
            group = "head"
        elif class_id in tail_set:
            group = "tail"
        else:
            group = "middle"
        out.append(
            {
                "class_id": class_id,
                "class_group": group,
                "global_count": float(global_counts[class_id]) if class_id < len(global_counts) else "",
                "num_support_clients": int(support_counts[class_id]) if class_id < len(support_counts) and not np.isnan(support_counts[class_id]) else "",
                "per_class_acc": as_float(row, "per_class_acc"),
            }
        )
    write_csv(
        run_dir / "per_class_metrics.csv",
        out,
        ["class_id", "class_group", "global_count", "num_support_clients", "per_class_acc"],
    )


def metrics_from_rounds(round_rows: list[dict[str, Any]]) -> dict[str, float]:
    if not round_rows:
        return {}
    final = round_rows[-1]
    best_overall_row = max(round_rows, key=lambda row: float(row["overall_acc"]))
    best_tail_row = max(round_rows, key=lambda row: float(row["tail_acc"]))
    best_tail = float(best_tail_row["tail_acc"])
    final_tail = float(final["tail_acc"])
    return {
        "final_overall_acc": float(final["overall_acc"]),
        "final_head_acc": float(final["head_acc"]),
        "final_tail_acc": final_tail,
        "final_macro_acc": float(final["macro_acc"]),
        "best_overall_acc": float(best_overall_row["overall_acc"]),
        "best_tail_acc": best_tail,
        "tail_peak_to_final_drop": best_tail - final_tail,
        "final_round": int(final["round"]),
        "best_tail_round": int(best_tail_row["round"]),
    }


def write_run_config(
    run_dir: Path,
    args: argparse.Namespace,
    method: str,
    lambda_t: float,
    seed: int,
    command: list[str],
    git_commit: str | None,
) -> None:
    spec = METHOD_SPECS[method]
    payload = {
        "method": method,
        "method_display": spec["display"],
        "trainer": spec["trainer"],
        "model": spec["model"],
        "dataset": args.dataset,
        "partition": "client-longtail",
        "num_clients": int(args.num_clients),
        "head_client_ratio": 1.0 - float(args.tail_client_ratio),
        "tail_client_ratio": float(args.tail_client_ratio),
        "head_class_ratio": 1.0 - float(args.tail_class_ratio),
        "tail_class_ratio": float(args.tail_class_ratio),
        "specialization_lambda": float(lambda_t),
        "intra_group_alpha": float(args.intra_group_alpha),
        "head_leakage_scale": float(args.head_leakage_scale),
        "imb_factor": float(args.imb_factor),
        "imb_type": args.imb_type,
        "seed": int(seed),
        "rounds": int(args.rounds),
        "frac": float(args.frac),
        "local_epoch": "from_config",
        "batch_size": int(args.train_batch_size),
        "test_batch_size": int(args.test_batch_size),
        "learning_rate": float(args.lr),
        "backbone": "from_config",
        "config_file": getattr(args, spec["config_attr"]),
        "dataset_config_file": dataset_config(args),
        "client_schedule_file": str(run_dir / "selected_clients_per_round.json"),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "git_commit": git_commit,
        "command": command,
    }
    write_json(run_dir / "config.json", payload)


def postprocess_run(
    args: argparse.Namespace,
    method: str,
    lambda_t: float,
    seed: int,
    command: list[str],
    git_commit: str | None,
) -> dict[str, Any] | None:
    run_dir = run_dir_for(args, method, lambda_t, seed)
    if not (run_dir / "round_metrics.csv").exists():
        print(f"[warn] Missing round_metrics.csv, skip postprocess: {run_dir}")
        return None

    round_rows, final_round = normalize_round_metrics(run_dir)
    if not round_rows:
        print(f"[warn] Empty round_metrics.csv, skip postprocess: {run_dir}")
        return None

    counts = load_counts(run_dir / "client_class_counts.csv")
    schedule = load_schedule(run_dir / "selected_clients_per_round.json")
    write_per_class_metrics(run_dir, final_round, counts, args)

    method_metrics = metrics_from_rounds(round_rows)
    topology_metrics = compute_topology_metrics(counts, schedule, args)
    metrics = {
        "method": method,
        "method_display": METHOD_SPECS[method]["display"],
        "lambda_T": float(lambda_t),
        "alpha_T": float(args.intra_group_alpha),
        "seed": int(seed),
        **method_metrics,
        **topology_metrics,
        "output_dir": str(run_dir),
    }
    write_json(run_dir / "metrics.json", metrics)
    write_run_config(run_dir, args, method, lambda_t, seed, command, git_commit)
    return metrics


def launch_run(args: argparse.Namespace, method: str, lambda_t: float, seed: int, command: list[str]) -> None:
    run_dir = run_dir_for(args, method, lambda_t, seed)
    metrics_path = run_dir / "metrics.json"
    if metrics_path.exists() and not args.overwrite:
        print(f"[skip] Existing metrics.json: {run_dir}")
        return

    run_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = run_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    if args.gpu:
        env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    log_path = logs_dir / "train.log"
    print(f"[run] {METHOD_SPECS[method]['display']} lambda={lambda_t:g} seed={seed}")
    with log_path.open("w", encoding="utf-8") as log_file:
        log_file.write(quote_cmd(command, args.gpu) + "\n\n")
        log_file.flush()
        result = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
        )
    if result.returncode != 0:
        raise RuntimeError(f"Run failed with exit code {result.returncode}: {run_dir}. See {log_path}")


def collect_all(args: argparse.Namespace, commands: dict[tuple[str, float, int], list[str]]) -> list[dict[str, Any]]:
    git_commit = get_git_commit()
    rows = []
    for method in args.methods:
        for lambda_t in args.lambda_values:
            for seed in args.seeds:
                key = (method, float(lambda_t), int(seed))
                row = postprocess_run(args, method, float(lambda_t), int(seed), commands[key], git_commit)
                if row is not None:
                    rows.append(row)
    return rows


def write_summaries(args: argparse.Namespace, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output_dir = Path(args.output_dir)
    write_csv(output_dir / "summary_all_runs.csv", rows, RUN_FIELDNAMES)

    metrics = [
        "final_overall_acc",
        "final_head_acc",
        "final_tail_acc",
        "final_macro_acc",
        "tail_peak_to_final_drop",
        "tail_active_rounds_mean",
        "max_exposure_gap_mean",
    ]
    grouped: dict[tuple[str, float], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["method"]), float(row["lambda_T"]))].append(row)

    summary_rows = []
    for method in METHOD_ORDER:
        if method not in args.methods:
            continue
        for lambda_t in sorted({float(row["lambda_T"]) for row in rows if row["method"] == method}):
            group_rows = grouped[(method, lambda_t)]
            out: dict[str, Any] = {
                "method": method,
                "method_display": METHOD_SPECS[method]["display"],
                "lambda_T": lambda_t,
                "alpha_T": float(args.intra_group_alpha),
                "num_runs": len(group_rows),
            }
            for metric in metrics:
                values = [float(row.get(metric, float("nan"))) for row in group_rows]
                out[f"{metric}_mean"] = nanmean(values)
                out[f"{metric}_std"] = nanstd(values)
            summary_rows.append(out)

    fieldnames = ["method", "method_display", "lambda_T", "alpha_T", "num_runs"]
    for metric in metrics:
        fieldnames.extend([f"{metric}_mean", f"{metric}_std"])
    write_csv(output_dir / "summary_by_method_lambda.csv", summary_rows, fieldnames)
    return summary_rows


def set_plot_style() -> None:
    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 9,
            "figure.titlesize": 13,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def grouped_metric(summary_rows: list[dict[str, Any]], method: str, metric: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows = sorted([row for row in summary_rows if row["method"] == method], key=lambda row: float(row["lambda_T"]))
    x = np.asarray([float(row["lambda_T"]) for row in rows], dtype=np.float64)
    y = np.asarray([float(row[f"{metric}_mean"]) for row in rows], dtype=np.float64)
    yerr = np.asarray([float(row[f"{metric}_std"]) for row in rows], dtype=np.float64)
    return x, y, yerr


def save_figure(fig: plt.Figure, output_base: Path) -> None:
    output_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_base.with_suffix(".png"), dpi=240, bbox_inches="tight")
    fig.savefig(output_base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_metric_vs_lambda(summary_rows: list[dict[str, Any]], methods: list[str], metric: str, ylabel: str, output_base: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for method in methods:
        x, y, yerr = grouped_metric(summary_rows, method, metric)
        if x.size == 0:
            continue
        ax.errorbar(
            x,
            y,
            yerr=yerr,
            marker=METHOD_MARKERS.get(method, "o"),
            color=METHOD_COLORS.get(method, "gray"),
            linewidth=1.8,
            capsize=3,
            label=METHOD_SPECS[method]["display"],
        )
    ax.set_xlabel(r"Tail specialization strength $\lambda_T$")
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    save_figure(fig, output_base)


def plot_all_metrics(summary_rows: list[dict[str, Any]], methods: list[str], output_base: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10.2, 7.0), sharex=True)
    specs = [
        ("final_overall_acc", "Overall accuracy"),
        ("final_head_acc", "Head accuracy"),
        ("final_tail_acc", "Tail accuracy"),
        ("final_macro_acc", "Macro accuracy"),
    ]
    for ax, (metric, title) in zip(axes.ravel(), specs):
        for method in methods:
            x, y, yerr = grouped_metric(summary_rows, method, metric)
            if x.size == 0:
                continue
            ax.errorbar(
                x,
                y,
                yerr=yerr,
                marker=METHOD_MARKERS.get(method, "o"),
                color=METHOD_COLORS.get(method, "gray"),
                linewidth=1.6,
                capsize=3,
                label=METHOD_SPECS[method]["display"],
            )
        ax.set_title(title)
        ax.set_xlabel(r"$\lambda_T$")
        ax.set_ylabel("Accuracy")
        ax.grid(axis="y", alpha=0.25)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, frameon=False, loc="upper center", ncol=min(len(handles), 3))
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    save_figure(fig, output_base)


def read_trajectory(run_dir: Path) -> list[dict[str, float]]:
    rows = read_csv_rows(run_dir / "per_round_metrics.csv")
    out = []
    for row in rows:
        out.append({"round": as_float(row, "round"), "tail_acc": as_float(row, "tail_acc")})
    return out


def plot_tail_trajectory(args: argparse.Namespace, output_base: Path) -> None:
    target_lambdas = [0.0, 0.5, 1.0]
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 3.9), sharey=True)
    for ax, lambda_t in zip(axes, target_lambdas):
        for method in ["promptfl", "capt"]:
            if method not in args.methods:
                continue
            round_map: dict[int, list[float]] = defaultdict(list)
            for seed in args.seeds:
                run_dir = run_dir_for(args, method, lambda_t, seed)
                for row in read_trajectory(run_dir):
                    if math.isnan(row["round"]) or math.isnan(row["tail_acc"]):
                        continue
                    round_map[int(row["round"])].append(float(row["tail_acc"]))
            if not round_map:
                continue
            rounds = np.asarray(sorted(round_map), dtype=np.float64)
            means = np.asarray([nanmean(round_map[int(r)]) for r in rounds], dtype=np.float64)
            stds = np.asarray([nanstd(round_map[int(r)]) for r in rounds], dtype=np.float64)
            color = METHOD_COLORS.get(method, "gray")
            ax.plot(rounds, means, color=color, linewidth=1.8, label=METHOD_SPECS[method]["display"])
            if np.any(np.isfinite(stds)):
                ax.fill_between(rounds, means - stds, means + stds, color=color, alpha=0.14)
        ax.set_title(rf"$\lambda_T={lambda_t:g}$")
        ax.set_xlabel("Round")
        ax.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("Tail accuracy")
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, frameon=False, loc="upper center", ncol=len(handles))
    fig.tight_layout(rect=[0, 0, 1, 0.9])
    save_figure(fig, output_base)


def plot_scatter(rows: list[dict[str, Any]], y_metric: str, ylabel: str, output_base: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.3, 4.2))
    for method in METHOD_ORDER:
        subset = [row for row in rows if row["method"] == method]
        if not subset:
            continue
        x = [float(row.get("max_exposure_gap_mean", float("nan"))) for row in subset]
        y = [float(row.get(y_metric, float("nan"))) for row in subset]
        ax.scatter(
            x,
            y,
            marker=METHOD_MARKERS.get(method, "o"),
            color=METHOD_COLORS.get(method, "gray"),
            alpha=0.78,
            s=48,
            label=METHOD_SPECS[method]["display"],
            edgecolors="none",
        )
    ax.set_xlabel("Mean max exposure gap")
    ax.set_ylabel(ylabel)
    ax.grid(axis="both", alpha=0.22)
    ax.legend(frameon=False)
    fig.tight_layout()
    save_figure(fig, output_base)


def plot_figures(args: argparse.Namespace, rows: list[dict[str, Any]], summary_rows: list[dict[str, Any]]) -> None:
    if not rows or not summary_rows:
        print("[warn] No rows available for figures.")
        return
    set_plot_style()
    figures_dir = Path(args.output_dir) / "figures"
    methods = [method for method in METHOD_ORDER if method in args.methods]
    plot_metric_vs_lambda(
        summary_rows,
        methods,
        "final_tail_acc",
        "Final tail accuracy",
        figures_dir / "figure_C1_tail_acc_vs_lambda",
    )
    plot_all_metrics(summary_rows, methods, figures_dir / "figure_C2_all_metrics_vs_lambda")
    drop_methods = [method for method in ["promptfl", "capt"] if method in args.methods]
    plot_metric_vs_lambda(
        summary_rows,
        drop_methods,
        "tail_peak_to_final_drop",
        "Tail peak-to-final drop",
        figures_dir / "figure_C3_tail_drop_vs_lambda",
    )
    plot_tail_trajectory(args, figures_dir / "figure_C4_tail_trajectory_lambda_0_05_1")
    plot_scatter(rows, "final_tail_acc", "Final tail accuracy", figures_dir / "figure_C5_tail_acc_vs_exposure_gap")
    plot_scatter(rows, "tail_peak_to_final_drop", "Tail peak-to-final drop", figures_dir / "figure_C6_drop_vs_exposure_gap")


def monotonic(values: list[float], direction: str, tol: float = 1e-8) -> bool | None:
    finite = [value for value in values if not math.isnan(value)]
    if len(finite) < 2:
        return None
    if direction == "increasing":
        return all(finite[idx + 1] >= finite[idx] - tol for idx in range(len(finite) - 1))
    if direction == "decreasing":
        return all(finite[idx + 1] <= finite[idx] + tol for idx in range(len(finite) - 1))
    raise ValueError(direction)


def write_response_checks(args: argparse.Namespace, rows: list[dict[str, Any]]) -> None:
    output_dir = Path(args.output_dir)
    by_lambda_seed: dict[tuple[float, int], dict[str, Any]] = {}
    for row in rows:
        by_lambda_seed.setdefault((float(row["lambda_T"]), int(row["seed"])), row)

    topo_metrics = {
        "tail_client_purity": "increasing",
        "head_leakage_to_tail_clients": "decreasing",
        "tail_leakage_to_head_clients": "decreasing",
        "tail_active_rounds_mean": "decreasing",
        "max_exposure_gap_mean": "increasing",
    }
    topology_checks = {}
    for metric, direction in topo_metrics.items():
        values = []
        for lambda_t in sorted({key[0] for key in by_lambda_seed}):
            lambda_rows = [row for (lam, _seed), row in by_lambda_seed.items() if lam == lambda_t]
            values.append(nanmean([float(row.get(metric, float("nan"))) for row in lambda_rows]))
        topology_checks[metric] = {
            "expected": direction,
            "values": values,
            "passed": monotonic(values, direction),
        }

    method_response = {}
    for method in ["promptfl", "capt"]:
        subset = [row for row in rows if row["method"] == method]
        lambda_values = sorted({float(row["lambda_T"]) for row in subset})
        tail_means = []
        drop_means = []
        for lambda_t in lambda_values:
            group = [row for row in subset if float(row["lambda_T"]) == lambda_t]
            tail_means.append(nanmean([float(row["final_tail_acc"]) for row in group]))
            drop_means.append(nanmean([float(row["tail_peak_to_final_drop"]) for row in group]))
        method_response[method] = {
            "final_tail_acc_by_lambda_mean": tail_means,
            "tail_drop_by_lambda_mean": drop_means,
            "final_tail_acc_range": float(np.nanmax(tail_means) - np.nanmin(tail_means)) if tail_means else float("nan"),
            "tail_drop_range": float(np.nanmax(drop_means) - np.nanmin(drop_means)) if drop_means else float("nan"),
            "final_tail_acc_std_across_lambda": nanstd(tail_means),
            "tail_drop_mean_across_lambda": nanmean(drop_means),
        }

    prompt = method_response.get("promptfl", {})
    capt = method_response.get("capt", {})
    capt_more_stable = {
        "by_final_tail_acc_std": None,
        "by_tail_drop_mean": None,
        "combined": None,
    }
    if prompt and capt:
        by_std = float(capt["final_tail_acc_std_across_lambda"]) <= float(prompt["final_tail_acc_std_across_lambda"])
        by_drop = float(capt["tail_drop_mean_across_lambda"]) <= float(prompt["tail_drop_mean_across_lambda"])
        capt_more_stable = {
            "by_final_tail_acc_std": bool(by_std),
            "by_tail_drop_mean": bool(by_drop),
            "combined": bool(by_std and by_drop),
        }

    payload = {
        "topology_checks": topology_checks,
        "method_response": method_response,
        "capt_more_stable_than_promptfl": capt_more_stable,
        "note": "Model-response checks are descriptive only and should not abort the experiment.",
    }
    write_json(output_dir / "response_checks.json", payload)


def main() -> None:
    args = parse_args()
    validate_args(args)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    commands: dict[tuple[str, float, int], list[str]] = {}
    for method in args.methods:
        for lambda_t in args.lambda_values:
            for seed in args.seeds:
                key = (method, float(lambda_t), int(seed))
                commands[key] = build_command(args, method, float(lambda_t), int(seed))

    if args.dry_run:
        for method in args.methods:
            for lambda_t in args.lambda_values:
                for seed in args.seeds:
                    print(quote_cmd(commands[(method, float(lambda_t), int(seed))], args.gpu))
        return

    if args.run:
        for method in args.methods:
            for lambda_t in args.lambda_values:
                for seed in args.seeds:
                    launch_run(args, method, float(lambda_t), int(seed), commands[(method, float(lambda_t), int(seed))])

    rows = collect_all(args, commands)
    summary_rows = write_summaries(args, rows)
    plot_figures(args, rows, summary_rows)
    write_response_checks(args, rows)
    print(f"[done] Wrote Experiment C outputs to {Path(args.output_dir).resolve()}")


if __name__ == "__main__":
    main()
