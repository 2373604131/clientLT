"""ClientLT(lambda_T, alpha_T) topology controllability sweep.

Example:
    python experiments/analyze_clientlt_topology_sweep.py \
      --datadir DATA/cifar-100 \
      --output_dir output/topology_sweep/clientlt_control \
      --num_clients 50 \
      --tail_client_ratio 0.1 \
      --tail_class_ratio 0.2 \
      --imb_factor 0.01 \
      --imb_type exp \
      --seeds 1 2 3 \
      --num_rounds 100 \
      --participation_rate 0.2

The script only generates Client-LT partitions, statistics, CSV files, and
figures. It does not train any model.
"""

from __future__ import annotations

import argparse
import csv
import inspect
import json
import math
import os
import pickle
import random
import subprocess
import sys
import types
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


OLD_CLIENTLT_PARAMS = {
    "head_to_head_ratio",
    "head_to_tail_ratio",
    "tail_to_head_ratio",
    "tail_to_tail_ratio",
    "client_longtail_alpha",
}
NEW_CLIENTLT_PARAMS = {
    "head_client_ratio",
    "tail_client_ratio",
    "head_class_ratio",
    "tail_class_ratio",
    "specialization_lambda",
    "intra_group_alpha",
    "head_leakage_scale",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze ClientLT(lambda_T, alpha_T) topology controllability without training."
    )
    parser.add_argument("--dataset", default="cifar100_LT", choices=["cifar100_LT"])
    parser.add_argument("--datadir", default="DATA/cifar-100")
    parser.add_argument("--output_dir", default="output/topology_sweep/clientlt_control")
    parser.add_argument("--num_clients", type=int, default=50)
    parser.add_argument("--tail_client_ratio", type=float, default=0.1)
    parser.add_argument("--tail_class_ratio", type=float, default=0.2)
    parser.add_argument("--imb_factor", type=float, default=0.01)
    parser.add_argument("--imb_type", default="exp")
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--lambda_values", type=float, nargs="+", default=[0.0, 0.25, 0.5, 0.75, 1.0])
    parser.add_argument("--alpha_values", type=float, nargs="+", default=[0.05, 0.1, 0.5, 1.0, 5.0])
    parser.add_argument("--head_leakage_scale", type=float, default=3.0)
    parser.add_argument("--fixed_alpha", type=float, default=0.1)
    parser.add_argument("--fixed_lambda", type=float, default=1.0)
    parser.add_argument("--num_rounds", type=int, default=100)
    parser.add_argument("--participation_rate", type=float, default=0.2)
    parser.add_argument("--monotonic_tolerance", type=float, default=0.02)
    return parser.parse_args()


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if hasattr(torch, "cuda") and torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def _make_dummy_loader_module() -> types.ModuleType:
    module = types.ModuleType("utils.dataloader")

    def _missing_loader(*_args, **_kwargs):
        raise RuntimeError("utils.dataloader is unavailable in this environment")

    for name in [
        "load_mnist_data",
        "load_fmnist_data",
        "load_fmnist_LT_data",
        "load_cifar10_data",
        "load_cifar100_data",
        "load_cifar10_LT_data",
        "load_cifar100_LT_data",
        "load_svhn_data",
        "load_celeba_data",
        "load_femnist_data",
    ]:
        setattr(module, name, _missing_loader)
    return module


def import_partition_client_longtail():
    try:
        from utils.datasplit import partition_client_longtail

        return partition_client_longtail
    except Exception as exc:
        # datasplit imports dataloader at module import time. In lightweight
        # analysis environments without torchvision, stub only those imports;
        # partition_client_longtail itself does not need image loading code.
        print(f"[info] Direct utils.datasplit import failed: {exc}")
        print("[info] Retrying datasplit import with lightweight dataloader stubs.")
        sys.modules.pop("utils.datasplit", None)
        sys.modules.setdefault("utils.dataloader", _make_dummy_loader_module())
        dataset_module = types.ModuleType("utils.dataset")
        dataset_module.mkdirs = lambda *_args, **_kwargs: None
        sys.modules.setdefault("utils.dataset", dataset_module)
        from utils.datasplit import partition_client_longtail

        return partition_client_longtail


def assert_new_clientlt_api(partition_fn) -> None:
    signature = inspect.signature(partition_fn)
    params = set(signature.parameters)
    old_present = sorted(OLD_CLIENTLT_PARAMS.intersection(params))
    missing_new = sorted(NEW_CLIENTLT_PARAMS.difference(params))
    if old_present or missing_new:
        raise RuntimeError(
            "partition_client_longtail must use the new ClientLT(lambda_T, alpha_T) API. "
            f"old params still present={old_present}; missing new params={missing_new}"
        )


def _find_cifar100_python_root(datadir: str | Path) -> Path:
    root = Path(datadir)
    candidates = [
        root / "cifar-100-python",
        root / "cifar-100" / "cifar-100-python",
        root,
    ]
    for candidate in candidates:
        if (candidate / "train").exists() and (candidate / "meta").exists():
            return candidate
    raise FileNotFoundError(
        "Could not locate CIFAR-100 python files. Expected one of: "
        + ", ".join(str(c) for c in candidates)
    )


def _load_pickle(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        return pickle.load(handle, encoding="latin1")


def _build_lt_labels_from_raw_cifar100(datadir: str, imb_factor: float, imb_type: str) -> tuple[np.ndarray, list[str]]:
    if imb_type != "exp":
        raise ValueError(f"Raw CIFAR100 fallback only supports imb_type='exp', got {imb_type}")
    cifar_root = _find_cifar100_python_root(datadir)
    train_payload = _load_pickle(cifar_root / "train")
    meta_payload = _load_pickle(cifar_root / "meta")
    raw_labels = np.asarray(train_payload["fine_labels"], dtype=np.int64)
    classnames = list(meta_payload.get("fine_label_names", [str(i) for i in range(100)]))

    set_all_seeds(1)
    num_classes = 100
    img_max = len(raw_labels) / num_classes
    selected_indices: list[int] = []
    for class_id in range(num_classes):
        indices = np.where(raw_labels == class_id)[0].astype(np.int64)
        np.random.shuffle(indices)
        keep = int(img_max * (float(imb_factor) ** (class_id / (num_classes - 1.0))))
        selected_indices.extend(indices[:keep].tolist())
    return raw_labels[np.asarray(selected_indices, dtype=np.int64)], classnames


def load_labels(args: argparse.Namespace) -> tuple[np.ndarray, list[str]]:
    try:
        from utils.dataloader import load_cifar100_LT_data

        _x_train, y_train, _x_test, _y_test, _data_train, _data_test, _lab2cname, classnames = (
            load_cifar100_LT_data(args.datadir, args.imb_factor, args.imb_type)
        )
        print("[info] Loaded CIFAR100-LT labels via utils.dataloader.load_cifar100_LT_data")
        return np.asarray(y_train, dtype=np.int64), list(classnames)
    except Exception as exc:
        print(f"[info] Project dataloader failed: {exc}")
        print("[info] Falling back to raw CIFAR-100 pickle label loading.")
        return _build_lt_labels_from_raw_cifar100(args.datadir, args.imb_factor, args.imb_type)


def get_head_tail_clients(num_clients: int, tail_client_ratio: float) -> tuple[list[int], list[int]]:
    head_client_ratio = 1.0 - float(tail_client_ratio)
    head_client_count = int(num_clients * head_client_ratio)
    head_clients = list(range(head_client_count))
    tail_clients = list(range(head_client_count, num_clients))
    return head_clients, tail_clients


def get_head_tail_classes(num_classes: int, tail_class_ratio: float) -> tuple[list[int], list[int]]:
    head_class_ratio = 1.0 - float(tail_class_ratio)
    head_class_count = int(num_classes * head_class_ratio)
    head_classes = list(range(head_class_count))
    tail_classes = list(range(head_class_count, num_classes))
    return head_classes, tail_classes


def compute_theoretical_ratios(lambda_T: float, tail_client_ratio: float) -> dict[str, float]:
    q_t = float(tail_client_ratio)
    lambda_t = float(lambda_T)
    tail_to_tail_ratio = q_t + (1.0 - q_t) * lambda_t
    tail_to_head_ratio = 1.0 - tail_to_tail_ratio
    head_to_tail_ratio = q_t * (1.0 - lambda_t)
    head_to_head_ratio = 1.0 - head_to_tail_ratio
    return {
        "head_to_head_ratio_theory": head_to_head_ratio,
        "head_to_tail_ratio_theory": head_to_tail_ratio,
        "tail_to_head_ratio_theory": tail_to_head_ratio,
        "tail_to_tail_ratio_theory": tail_to_tail_ratio,
    }


def validate_partition(labels: np.ndarray, net_dataidx_map: dict[int, np.ndarray]) -> None:
    all_indices = []
    for client_id, indices in net_dataidx_map.items():
        array = np.asarray(indices, dtype=np.int64)
        if np.any(array < 0) or np.any(array >= len(labels)):
            raise AssertionError(f"Client {client_id} has out-of-range indices")
        all_indices.extend(array.tolist())
    if len(all_indices) != len(labels):
        raise AssertionError(f"Partition has {len(all_indices)} indices, expected {len(labels)}")
    unique = set(all_indices)
    if len(unique) != len(all_indices):
        raise AssertionError("Partition contains duplicate indices")
    expected = set(range(len(labels)))
    if unique != expected:
        missing = len(expected.difference(unique))
        extra = len(unique.difference(expected))
        raise AssertionError(f"Partition index set mismatch: missing={missing}, extra={extra}")


def build_client_class_count_matrix(
    labels: np.ndarray,
    net_dataidx_map: dict[int, np.ndarray],
    num_clients: int,
    num_classes: int,
) -> np.ndarray:
    validate_partition(labels, net_dataidx_map)
    counts = np.zeros((num_clients, num_classes), dtype=np.int64)
    for client_id in range(num_clients):
        indices = np.asarray(net_dataidx_map.get(client_id, []), dtype=np.int64)
        if indices.size == 0:
            continue
        client_labels = labels[indices].astype(np.int64)
        counts[client_id] = np.bincount(client_labels, minlength=num_classes)
    if int(counts.sum()) != len(labels):
        raise AssertionError(f"Count matrix sums to {counts.sum()}, expected {len(labels)}")
    return counts


def safe_divide(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return float("nan")
    return float(numerator) / float(denominator)


def compute_purity_and_leakage_metrics(
    client_class_counts: np.ndarray,
    head_clients: list[int],
    tail_clients: list[int],
    head_classes: list[int],
    tail_classes: list[int],
) -> dict[str, float]:
    tail_client_total = float(client_class_counts[np.ix_(tail_clients, range(client_class_counts.shape[1]))].sum())
    tail_on_tail_clients = float(client_class_counts[np.ix_(tail_clients, tail_classes)].sum())
    head_on_tail_clients = float(client_class_counts[np.ix_(tail_clients, head_classes)].sum())
    tail_on_head_clients = float(client_class_counts[np.ix_(head_clients, tail_classes)].sum())
    head_global_total = float(client_class_counts[:, head_classes].sum())
    tail_global_total = float(client_class_counts[:, tail_classes].sum())
    return {
        "tail_client_purity": safe_divide(tail_on_tail_clients, tail_client_total),
        "head_leakage_to_tail_clients": safe_divide(head_on_tail_clients, head_global_total),
        "tail_leakage_to_head_clients": safe_divide(tail_on_head_clients, tail_global_total),
    }


def max_zero_run(active: np.ndarray) -> int:
    best = 0
    current = 0
    for value in active.tolist():
        if int(value) == 0:
            current += 1
            best = max(best, current)
        else:
            current = 0
    return int(best)


def simulate_temporal_exposure(
    client_class_counts: np.ndarray,
    tail_classes: list[int],
    num_rounds: int,
    participation_rate: float,
    seed: int,
) -> dict[str, float]:
    num_clients = client_class_counts.shape[0]
    clients_per_round = max(1, int(round(num_clients * float(participation_rate))))
    clients_per_round = min(num_clients, clients_per_round)
    rng = np.random.default_rng(seed + 100_003)
    schedule = [
        rng.choice(num_clients, size=clients_per_round, replace=False)
        for _ in range(int(num_rounds))
    ]

    active_round_counts = []
    max_gaps = []
    support = client_class_counts > 0
    for class_id in tail_classes:
        active = np.asarray([bool(support[selected, class_id].any()) for selected in schedule], dtype=np.int64)
        active_round_counts.append(float(active.sum()))
        max_gaps.append(float(max_zero_run(active)))
    return {
        "tail_active_rounds_mean": float(np.mean(active_round_counts)),
        "tail_active_rounds_std": float(np.std(active_round_counts)),
        "max_exposure_gap_mean": float(np.mean(max_gaps)),
        "max_exposure_gap_std": float(np.std(max_gaps)),
    }


def entropy_from_counts(counts: np.ndarray, denominator_clients: int) -> float:
    total = float(counts.sum())
    if total <= 0:
        return float("nan")
    positive = counts[counts > 0].astype(np.float64) / total
    entropy = float(-(positive * np.log(positive)).sum())
    if denominator_clients <= 1:
        return 0.0
    return entropy / float(np.log(denominator_clients))


def compute_tail_aggregation_metrics(
    client_class_counts: np.ndarray,
    tail_clients: list[int],
    tail_classes: list[int],
) -> dict[str, float]:
    top1_values = []
    top2_values = []
    effective_values = []
    entropy_values = []
    tail_entropy_values = []
    num_clients = client_class_counts.shape[0]
    for class_id in tail_classes:
        counts = client_class_counts[:, class_id].astype(np.float64)
        total = float(counts.sum())
        if total <= 0:
            continue
        sorted_counts = np.sort(counts)[::-1]
        top1_values.append(float(sorted_counts[0] / total))
        top2_values.append(float(sorted_counts[:2].sum() / total))
        denom = float((counts ** 2).sum())
        effective_values.append(float((total ** 2) / denom) if denom > 0 else float("nan"))
        entropy_values.append(entropy_from_counts(counts, num_clients))
        tail_entropy_values.append(entropy_from_counts(counts[np.asarray(tail_clients, dtype=np.int64)], len(tail_clients)))
    return {
        "tail_top1_mass_mean": float(np.nanmean(top1_values)),
        "tail_top1_mass_std": float(np.nanstd(top1_values)),
        "tail_top2_mass_mean": float(np.nanmean(top2_values)),
        "tail_top2_mass_std": float(np.nanstd(top2_values)),
        "effective_client_number_mean": float(np.nanmean(effective_values)),
        "effective_client_number_std": float(np.nanstd(effective_values)),
        "normalized_entropy_mean": float(np.nanmean(entropy_values)),
        "normalized_entropy_std": float(np.nanstd(entropy_values)),
        "tail_client_normalized_entropy_mean": float(np.nanmean(tail_entropy_values)),
        "tail_client_normalized_entropy_std": float(np.nanstd(tail_entropy_values)),
    }


def normalized_tail_mass_matrix(client_class_counts: np.ndarray, tail_classes: list[int]) -> np.ndarray:
    tail_counts = client_class_counts[:, tail_classes].astype(np.float64)
    totals = tail_counts.sum(axis=0, keepdims=True)
    totals[totals == 0] = 1.0
    return tail_counts / totals


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def summarize_rows(rows: list[dict[str, Any]], key: str, metrics: list[str]) -> list[dict[str, Any]]:
    grouped: dict[Any, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row[key]].append(row)
    summary = []
    for value in sorted(grouped):
        out = {key: value}
        for metric in metrics:
            vals = np.asarray([float(row[metric]) for row in grouped[value]], dtype=np.float64)
            out[f"{metric}_mean"] = float(np.nanmean(vals))
            out[f"{metric}_std"] = float(np.nanstd(vals))
        summary.append(out)
    return summary


def plot_experiment_A_curves(summary_rows: list[dict[str, Any]], output_dir: Path) -> None:
    x = np.asarray([float(row["lambda_T"]) for row in summary_rows], dtype=np.float64)
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    specs = [
        ("tail_client_purity", "Tail-client purity", "#1f77b4"),
        ("head_leakage_to_tail_clients", "Head leakage to tail clients", "#d62728"),
        ("tail_leakage_to_head_clients", "Tail leakage to head clients", "#2ca02c"),
    ]
    for metric, label, color in specs:
        y = np.asarray([float(row[f"{metric}_mean"]) for row in summary_rows])
        yerr = np.asarray([float(row[f"{metric}_std"]) for row in summary_rows])
        ax.errorbar(x, y, yerr=yerr, marker="o", linewidth=1.8, capsize=3, label=label, color=color)
    ax.set_xlabel(r"Tail specialization strength $\lambda_T$")
    ax.set_ylabel("Ratio")
    ax.set_ylim(-0.03, 1.03)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(output_dir / f"figure_A_lambda_purity_leakage.{suffix}", dpi=240, bbox_inches="tight")
    plt.close(fig)


def plot_experiment_A_temporal(summary_rows: list[dict[str, Any]], output_dir: Path) -> None:
    x = np.asarray([float(row["lambda_T"]) for row in summary_rows], dtype=np.float64)
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.8))
    temporal_specs = [
        (axes[0], "tail_active_rounds", "Tail active rounds", "#1f77b4"),
        (axes[1], "max_exposure_gap", "Max exposure gap", "#d62728"),
    ]
    for ax, metric, label, color in temporal_specs:
        y = np.asarray([float(row[f"{metric}_mean_mean"]) for row in summary_rows])
        yerr = np.asarray([float(row[f"{metric}_mean_std"]) for row in summary_rows])
        ax.errorbar(x, y, yerr=yerr, marker="o", linewidth=1.8, capsize=3, color=color)
        ax.set_xlabel(r"Tail specialization strength $\lambda_T$")
        ax.set_ylabel(label)
        ax.grid(alpha=0.25)
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(output_dir / f"figure_A_lambda_temporal_exposure.{suffix}", dpi=240, bbox_inches="tight")
    plt.close(fig)


def plot_heatmap_panels(
    matrices: dict[float, np.ndarray],
    values: list[float],
    title_prefix: str,
    output_path: Path,
    x_label: str = "Tail class id",
) -> None:
    fig, axes = plt.subplots(
        1,
        len(values),
        figsize=(3.2 * len(values), 4.2),
        sharey=True,
        constrained_layout=True,
    )
    if len(values) == 1:
        axes = [axes]
    image = None
    for ax, value in zip(axes, values):
        matrix = matrices[value]
        image = ax.imshow(matrix, aspect="auto", vmin=0.0, vmax=1.0, interpolation="nearest", cmap="viridis")
        ax.set_title(f"{title_prefix}={value:g}")
        ax.set_xlabel(x_label)
        ax.set_xticks([0, matrix.shape[1] - 1])
        ax.set_xticklabels(["first", "last"])
        ax.grid(False)
    axes[0].set_ylabel("Client id")
    if image is not None:
        fig.colorbar(image, ax=axes, fraction=0.025, pad=0.02, label="Normalized tail-class mass")
    for suffix in ("png", "pdf"):
        fig.savefig(output_path.with_suffix(f".{suffix}"), dpi=240, bbox_inches="tight")
    plt.close(fig)


def plot_experiment_A_heatmaps(
    heatmaps: dict[float, np.ndarray],
    lambda_values: list[float],
    output_dir: Path,
) -> None:
    plot_heatmap_panels(
        heatmaps,
        lambda_values,
        r"$\lambda_T$",
        output_dir / "figure_A_lambda_heatmaps",
    )


def plot_experiment_B_curves(summary_rows: list[dict[str, Any]], output_dir: Path) -> None:
    x = np.asarray([float(row["alpha_T"]) for row in summary_rows], dtype=np.float64)
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.9))

    for metric, label, color in [
        ("tail_top1_mass", "Tail top-1 client mass", "#1f77b4"),
        ("tail_top2_mass", "Tail top-2 client mass", "#ff7f0e"),
    ]:
        y = np.asarray([float(row[f"{metric}_mean_mean"]) for row in summary_rows])
        yerr = np.asarray([float(row[f"{metric}_mean_std"]) for row in summary_rows])
        axes[0].errorbar(x, y, yerr=yerr, marker="o", linewidth=1.8, capsize=3, label=label, color=color)
    axes[0].set_xscale("log")
    axes[0].set_xlabel(r"Tail aggregation alpha $\alpha_T$")
    axes[0].set_ylabel("Mass")
    axes[0].set_ylim(-0.03, 1.03)
    axes[0].grid(alpha=0.25)
    axes[0].legend(frameon=False)

    eff = np.asarray([float(row["effective_client_number_mean_mean"]) for row in summary_rows])
    eff_err = np.asarray([float(row["effective_client_number_mean_std"]) for row in summary_rows])
    entropy = np.asarray([float(row["normalized_entropy_mean_mean"]) for row in summary_rows])
    entropy_err = np.asarray([float(row["normalized_entropy_mean_std"]) for row in summary_rows])
    axes[1].errorbar(x, eff, yerr=eff_err, marker="o", linewidth=1.8, capsize=3, label="Effective client number", color="#2ca02c")
    axes[1].set_xscale("log")
    axes[1].set_xlabel(r"Tail aggregation alpha $\alpha_T$")
    axes[1].set_ylabel("Effective client number")
    axes[1].grid(alpha=0.25)

    ax2 = axes[1].twinx()
    ax2.errorbar(x, entropy, yerr=entropy_err, marker="s", linewidth=1.5, capsize=3, label="Normalized entropy", color="#9467bd")
    ax2.set_ylabel("Normalized entropy")
    ax2.set_ylim(-0.03, 1.03)
    lines1, labels1 = axes[1].get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    axes[1].legend(lines1 + lines2, labels1 + labels2, frameon=False, loc="best")

    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(output_dir / f"figure_B_alpha_aggregation_metrics.{suffix}", dpi=240, bbox_inches="tight")
    plt.close(fig)


def plot_experiment_B_heatmaps(
    heatmaps: dict[float, np.ndarray],
    alpha_values: list[float],
    output_dir: Path,
) -> None:
    plot_heatmap_panels(
        heatmaps,
        alpha_values,
        r"$\alpha_T$",
        output_dir / "figure_B_alpha_heatmaps",
    )


def monotonic_increasing(values: list[float], tolerance: float) -> bool:
    return all(values[i + 1] >= values[i] - tolerance for i in range(len(values) - 1))


def monotonic_decreasing(values: list[float], tolerance: float) -> bool:
    return all(values[i + 1] <= values[i] + tolerance for i in range(len(values) - 1))


def check_monotonicity(
    summary_A: list[dict[str, Any]],
    summary_B: list[dict[str, Any]],
    tolerance: float,
) -> dict[str, Any]:
    checks: dict[str, dict[str, Any]] = {}

    def add_check(name: str, values: list[float], expected: str) -> None:
        if expected == "increasing":
            passed = monotonic_increasing(values, tolerance)
        else:
            passed = monotonic_decreasing(values, tolerance)
        checks[name] = {
            "expected": expected,
            "tolerance": tolerance,
            "values": values,
            "passed": bool(passed),
        }

    add_check(
        "A_tail_client_purity_vs_lambda",
        [float(row["tail_client_purity_mean"]) for row in summary_A],
        "increasing",
    )
    add_check(
        "A_head_leakage_to_tail_clients_vs_lambda",
        [float(row["head_leakage_to_tail_clients_mean"]) for row in summary_A],
        "decreasing",
    )
    add_check(
        "A_tail_leakage_to_head_clients_vs_lambda",
        [float(row["tail_leakage_to_head_clients_mean"]) for row in summary_A],
        "decreasing",
    )
    add_check(
        "B_tail_top1_mass_vs_alpha",
        [float(row["tail_top1_mass_mean_mean"]) for row in summary_B],
        "decreasing",
    )
    add_check(
        "B_tail_top2_mass_vs_alpha",
        [float(row["tail_top2_mass_mean_mean"]) for row in summary_B],
        "decreasing",
    )
    add_check(
        "B_effective_client_number_vs_alpha",
        [float(row["effective_client_number_mean_mean"]) for row in summary_B],
        "increasing",
    )
    add_check(
        "B_normalized_entropy_vs_alpha",
        [float(row["normalized_entropy_mean_mean"]) for row in summary_B],
        "increasing",
    )
    return {
        "all_passed": all(item["passed"] for item in checks.values()),
        "checks": checks,
    }


def get_git_commit_hash() -> str | None:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True, stderr=subprocess.DEVNULL)
            .strip()
        )
    except Exception:
        return None


def run_partition(
    partition_fn,
    labels: np.ndarray,
    seed: int,
    num_clients: int,
    num_classes: int,
    head_client_ratio: float,
    tail_client_ratio: float,
    head_class_ratio: float,
    tail_class_ratio: float,
    lambda_T: float,
    alpha_T: float,
    head_leakage_scale: float,
) -> np.ndarray:
    set_all_seeds(seed)
    net_dataidx_map = partition_fn(
        labels=labels,
        n_parties=num_clients,
        num_classes=num_classes,
        head_client_ratio=head_client_ratio,
        tail_client_ratio=tail_client_ratio,
        head_class_ratio=head_class_ratio,
        tail_class_ratio=tail_class_ratio,
        specialization_lambda=lambda_T,
        intra_group_alpha=alpha_T,
        head_leakage_scale=head_leakage_scale,
    )
    return build_client_class_count_matrix(labels, net_dataidx_map, num_clients, num_classes)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    csv_dir = output_dir / "csv"
    figures_dir = output_dir / "figures"
    heatmaps_dir = output_dir / "heatmaps"
    for directory in (csv_dir, figures_dir, heatmaps_dir):
        directory.mkdir(parents=True, exist_ok=True)

    labels, classnames = load_labels(args)
    num_classes = len(classnames)
    if num_classes != 100:
        raise RuntimeError(f"This script expects CIFAR100-LT with 100 classes, got {num_classes}")

    head_client_ratio = 1.0 - float(args.tail_client_ratio)
    head_class_ratio = 1.0 - float(args.tail_class_ratio)
    head_clients, tail_clients = get_head_tail_clients(args.num_clients, args.tail_client_ratio)
    head_classes, tail_classes = get_head_tail_classes(num_classes, args.tail_class_ratio)

    partition_fn = import_partition_client_longtail()
    assert_new_clientlt_api(partition_fn)

    print("Client/head-tail definitions:")
    print(f"  head clients: {head_clients[0]}-{head_clients[-1]} (n={len(head_clients)})")
    print(f"  tail clients: {tail_clients[0]}-{tail_clients[-1]} (n={len(tail_clients)})")
    print(f"  head classes: {head_classes[0]}-{head_classes[-1]} (n={len(head_classes)})")
    print(f"  tail classes: {tail_classes[0]}-{tail_classes[-1]} (n={len(tail_classes)})")
    print(f"  y_train samples: {len(labels)}")

    rows_A: list[dict[str, Any]] = []
    rows_B: list[dict[str, Any]] = []
    heatmaps_A: dict[float, np.ndarray] = {}
    heatmaps_B: dict[float, np.ndarray] = {}
    heatmap_seed = int(args.seeds[0])

    for lambda_T in args.lambda_values:
        for seed in args.seeds:
            counts = run_partition(
                partition_fn,
                labels,
                seed,
                args.num_clients,
                num_classes,
                head_client_ratio,
                args.tail_client_ratio,
                head_class_ratio,
                args.tail_class_ratio,
                lambda_T,
                args.fixed_alpha,
                args.head_leakage_scale,
            )
            metrics = compute_purity_and_leakage_metrics(
                counts, head_clients, tail_clients, head_classes, tail_classes
            )
            metrics.update(
                simulate_temporal_exposure(
                    counts,
                    tail_classes,
                    args.num_rounds,
                    args.participation_rate,
                    seed,
                )
            )
            ratios = compute_theoretical_ratios(lambda_T, args.tail_client_ratio)
            row = {
                "experiment": "A_lambda_sweep",
                "seed": seed,
                "lambda_T": float(lambda_T),
                "alpha_T": float(args.fixed_alpha),
                "num_clients": args.num_clients,
                "tail_client_ratio": float(args.tail_client_ratio),
                "tail_class_ratio": float(args.tail_class_ratio),
                **metrics,
                **ratios,
            }
            rows_A.append(row)
            if int(seed) == heatmap_seed:
                heatmaps_A[float(lambda_T)] = normalized_tail_mass_matrix(counts, tail_classes)

    for alpha_T in args.alpha_values:
        for seed in args.seeds:
            counts = run_partition(
                partition_fn,
                labels,
                seed,
                args.num_clients,
                num_classes,
                head_client_ratio,
                args.tail_client_ratio,
                head_class_ratio,
                args.tail_class_ratio,
                args.fixed_lambda,
                alpha_T,
                args.head_leakage_scale,
            )
            metrics = compute_tail_aggregation_metrics(counts, tail_clients, tail_classes)
            row = {
                "experiment": "B_alpha_sweep",
                "seed": seed,
                "lambda_T": float(args.fixed_lambda),
                "alpha_T": float(alpha_T),
                "num_clients": args.num_clients,
                "tail_client_ratio": float(args.tail_client_ratio),
                "tail_class_ratio": float(args.tail_class_ratio),
                **metrics,
            }
            rows_B.append(row)
            if int(seed) == heatmap_seed:
                heatmaps_B[float(alpha_T)] = normalized_tail_mass_matrix(counts, tail_classes)

    fields_A = [
        "experiment",
        "seed",
        "lambda_T",
        "alpha_T",
        "num_clients",
        "tail_client_ratio",
        "tail_class_ratio",
        "tail_client_purity",
        "head_leakage_to_tail_clients",
        "tail_leakage_to_head_clients",
        "tail_active_rounds_mean",
        "tail_active_rounds_std",
        "max_exposure_gap_mean",
        "max_exposure_gap_std",
        "head_to_head_ratio_theory",
        "head_to_tail_ratio_theory",
        "tail_to_head_ratio_theory",
        "tail_to_tail_ratio_theory",
    ]
    fields_B = [
        "experiment",
        "seed",
        "lambda_T",
        "alpha_T",
        "num_clients",
        "tail_client_ratio",
        "tail_class_ratio",
        "tail_top1_mass_mean",
        "tail_top1_mass_std",
        "tail_top2_mass_mean",
        "tail_top2_mass_std",
        "effective_client_number_mean",
        "effective_client_number_std",
        "normalized_entropy_mean",
        "normalized_entropy_std",
        "tail_client_normalized_entropy_mean",
        "tail_client_normalized_entropy_std",
    ]
    write_csv(csv_dir / "experiment_A_per_seed.csv", rows_A, fields_A)
    write_csv(csv_dir / "experiment_B_per_seed.csv", rows_B, fields_B)

    metrics_A = [
        "tail_client_purity",
        "head_leakage_to_tail_clients",
        "tail_leakage_to_head_clients",
        "tail_active_rounds_mean",
        "tail_active_rounds_std",
        "max_exposure_gap_mean",
        "max_exposure_gap_std",
        "head_to_head_ratio_theory",
        "head_to_tail_ratio_theory",
        "tail_to_head_ratio_theory",
        "tail_to_tail_ratio_theory",
    ]
    metrics_B = [
        "tail_top1_mass_mean",
        "tail_top1_mass_std",
        "tail_top2_mass_mean",
        "tail_top2_mass_std",
        "effective_client_number_mean",
        "effective_client_number_std",
        "normalized_entropy_mean",
        "normalized_entropy_std",
        "tail_client_normalized_entropy_mean",
        "tail_client_normalized_entropy_std",
    ]
    summary_A = summarize_rows(rows_A, "lambda_T", metrics_A)
    summary_B = summarize_rows(rows_B, "alpha_T", metrics_B)
    write_csv(csv_dir / "experiment_A_summary.csv", summary_A)
    write_csv(csv_dir / "experiment_B_summary.csv", summary_B)

    plot_experiment_A_curves(summary_A, figures_dir)
    plot_experiment_A_temporal(summary_A, figures_dir)
    plot_experiment_A_heatmaps(heatmaps_A, [float(x) for x in args.lambda_values], heatmaps_dir)
    plot_experiment_B_curves(summary_B, figures_dir)
    plot_experiment_B_heatmaps(heatmaps_B, [float(x) for x in args.alpha_values], heatmaps_dir)

    monotonicity = check_monotonicity(summary_A, summary_B, args.monotonic_tolerance)
    with (csv_dir / "monotonicity_checks.json").open("w", encoding="utf-8") as handle:
        json.dump(monotonicity, handle, indent=2)

    metadata = {
        "dataset": args.dataset,
        "datadir": args.datadir,
        "num_clients": args.num_clients,
        "head_client_ratio": head_client_ratio,
        "tail_client_ratio": float(args.tail_client_ratio),
        "head_class_ratio": head_class_ratio,
        "tail_class_ratio": float(args.tail_class_ratio),
        "seeds": [int(x) for x in args.seeds],
        "lambda_values": [float(x) for x in args.lambda_values],
        "alpha_values": [float(x) for x in args.alpha_values],
        "fixed_alpha_for_experiment_A": float(args.fixed_alpha),
        "fixed_lambda_for_experiment_B": float(args.fixed_lambda),
        "num_rounds": int(args.num_rounds),
        "participation_rate": float(args.participation_rate),
        "formulas": {
            "tail_to_tail_ratio": "q_T + (1 - q_T) * lambda_T",
            "tail_to_head_ratio": "1 - tail_to_tail_ratio",
            "head_to_tail_ratio": "q_T * (1 - lambda_T)",
            "head_to_head_ratio": "1 - head_to_tail_ratio",
        },
        "head_clients": {"first": head_clients[0], "last": head_clients[-1], "count": len(head_clients)},
        "tail_clients": {"first": tail_clients[0], "last": tail_clients[-1], "count": len(tail_clients)},
        "head_classes": {"first": head_classes[0], "last": head_classes[-1], "count": len(head_classes)},
        "tail_classes": {"first": tail_classes[0], "last": tail_classes[-1], "count": len(tail_classes)},
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "git_commit_hash": get_git_commit_hash(),
    }
    with (output_dir / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    print("\nMonotonicity checks:")
    for name, result in monotonicity["checks"].items():
        status = "PASS" if result["passed"] else "WARNING"
        print(f"  [{status}] {name}: expected {result['expected']}, values={result['values']}")
    if monotonicity["all_passed"]:
        print("\nAll monotonicity checks passed.")
    else:
        failed = [name for name, result in monotonicity["checks"].items() if not result["passed"]]
        print("\nWARNING: Some monotonicity checks did not pass:")
        for name in failed:
            print(f"  - {name}")

    print(f"\nSaved outputs to: {output_dir}")
    print("Experiment A CSV: csv/experiment_A_per_seed.csv, csv/experiment_A_summary.csv")
    print("Experiment B CSV: csv/experiment_B_per_seed.csv, csv/experiment_B_summary.csv")
    print("Figures: figures/*.png, figures/*.pdf")
    print("Heatmaps: heatmaps/*.png, heatmaps/*.pdf")


if __name__ == "__main__":
    main()
