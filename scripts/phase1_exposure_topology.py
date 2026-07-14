#!/usr/bin/env python
"""Phase 1 topology report for tail exposure experiments.

This script compares the standard Dirichlet split with the cleaner
client-longtail split that concentrates each tail class on a small number of
tail clients. It does not train a model; the goal is to verify that the two
partitions induce different client-class exposure topologies under the same
global long-tailed class counts.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import pickle
import random
import sys
from pathlib import Path

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

partition_data_LT = None


PARTITION_SPECS = (
    ("dirichlet", "noniid-labeldir"),
    ("client_lt", "client-longtail"),
)


def set_plot_style() -> None:
    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "figure.titlesize": 13,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare Dirichlet and client-longtail exposure topology."
    )
    parser.add_argument("--dataset", default="cifar100_LT", choices=["cifar10_LT", "cifar100_LT", "fmnist_LT"])
    parser.add_argument("--data-root", default="DATA")
    parser.add_argument("--output-dir", default="output/phase1_exposure_topology")
    parser.add_argument("--num-users", type=int, default=20)
    parser.add_argument("--frac", type=float, default=0.4)
    parser.add_argument("--rounds", type=int, default=100)
    parser.add_argument("--schedule-seed", type=int, default=1)
    parser.add_argument("--imb-type", default="exp")
    parser.add_argument("--imb-factor", type=float, default=0.01)
    parser.add_argument("--dirichlet-beta", type=float, default=0.3)
    parser.add_argument("--head-client-ratio", type=float, default=0.8)
    parser.add_argument("--tail-client-ratio", type=float, default=0.2)
    parser.add_argument("--head-class-ratio", type=float, default=0.8)
    parser.add_argument("--tail-class-ratio", type=float, default=0.2)
    parser.add_argument("--specialization-lambda", type=float, default=1.0)
    parser.add_argument("--intra-group-alpha", type=float, default=0.3)
    parser.add_argument("--head-leakage-scale", type=float, default=3.0)
    parser.add_argument(
        "--use-project-datasplit",
        action="store_true",
        help=(
            "Use utils.datasplit.partition_data_LT. The default lightweight path "
            "uses only CIFAR label files to avoid importing torch/torchvision."
        ),
    )
    return parser.parse_args()


def dataset_dir(data_root: str, dataset: str) -> str:
    root = Path(data_root)
    if dataset == "cifar100_LT":
        return str(root / "cifar-100")
    if dataset == "cifar10_LT":
        return str(root / "cifar-10")
    if dataset == "fmnist_LT":
        return str(root / "fmnist")
    raise ValueError(f"Unsupported dataset: {dataset}")


def _counts_from_weights(total: int, weights: np.ndarray) -> np.ndarray:
    weights = np.asarray(weights, dtype=np.float64)
    raw_counts = weights * total
    counts = np.floor(raw_counts).astype(int)
    remainder = int(total - counts.sum())
    if remainder > 0:
        fractional = raw_counts - counts
        order = np.argsort(fractional)[::-1]
        counts[order[:remainder]] += 1
    return counts


def _validate_ratio_pair(name_a: str, value_a: float, name_b: str, value_b: float) -> None:
    if value_a < 0 or value_b < 0:
        raise ValueError(f"{name_a} and {name_b} must be non-negative, got {value_a} and {value_b}")
    total = value_a + value_b
    if not np.isclose(total, 1.0):
        raise ValueError(f"{name_a} + {name_b} must be 1.0, got {total:.6f}")


def _allocate_class_budgets(class_counts: dict[int, int], total_budget: int) -> dict[int, int]:
    class_counts = {int(class_id): int(count) for class_id, count in class_counts.items()}
    total_budget = int(total_budget)
    total_count = sum(class_counts.values())
    if total_budget < 0 or total_budget > total_count:
        raise ValueError(f"Invalid total_budget={total_budget} for total_count={total_count}")
    budgets = {class_id: 0 for class_id in class_counts}
    if total_budget == 0 or total_count == 0:
        return budgets
    raw_budgets = {
        class_id: (count / total_count) * total_budget
        for class_id, count in class_counts.items()
    }
    for class_id, raw_budget in raw_budgets.items():
        budgets[class_id] = min(int(np.floor(raw_budget)), class_counts[class_id])
    remainder = total_budget - sum(budgets.values())
    candidates = sorted(
        class_counts,
        key=lambda class_id: (
            raw_budgets[class_id] - np.floor(raw_budgets[class_id]),
            class_counts[class_id],
            -class_id,
        ),
        reverse=True,
    )
    while remainder > 0:
        progressed = False
        for class_id in candidates:
            if remainder <= 0:
                break
            if budgets[class_id] >= class_counts[class_id]:
                continue
            budgets[class_id] += 1
            remainder -= 1
            progressed = True
        if not progressed:
            raise RuntimeError("Unable to allocate class budgets within class capacities")
    return budgets


def _append_uniform(subset: np.ndarray, group_ids: list[int], net_map: dict[int, np.ndarray]) -> None:
    split = np.array_split(subset, len(group_ids))
    for client_id, chunk in zip(group_ids, split):
        if len(chunk) > 0:
            net_map[client_id] = np.append(net_map[client_id], chunk.astype(np.int64))


def _append_dirichlet(
    subset: np.ndarray,
    group_ids: list[int],
    net_map: dict[int, np.ndarray],
    alpha: float,
    rng: np.random.Generator,
) -> None:
    if len(group_ids) == 1 or alpha is None or alpha <= 0:
        _append_uniform(subset, group_ids, net_map)
        return
    counts = _counts_from_weights(len(subset), rng.dirichlet(np.repeat(alpha, len(group_ids))))
    offset = 0
    for client_id, count in zip(group_ids, counts):
        if count <= 0:
            continue
        chunk = subset[offset:offset + count]
        offset += count
        if len(chunk) > 0:
            net_map[client_id] = np.append(net_map[client_id], chunk.astype(np.int64))


def _partition_client_longtail_light(
    labels: np.ndarray,
    n_parties: int,
    num_classes: int,
    head_client_ratio: float,
    tail_client_ratio: float,
    head_class_ratio: float,
    tail_class_ratio: float,
    specialization_lambda: float,
    intra_group_alpha: float,
    head_leakage_scale: float,
) -> dict[int, np.ndarray]:
    _validate_ratio_pair("head_client_ratio", head_client_ratio, "tail_client_ratio", tail_client_ratio)
    _validate_ratio_pair("head_class_ratio", head_class_ratio, "tail_class_ratio", tail_class_ratio)
    if specialization_lambda < 0.0 or specialization_lambda > 1.0:
        raise ValueError(f"specialization_lambda must be in [0.0, 1.0], got {specialization_lambda}")
    if intra_group_alpha <= 0:
        raise ValueError(f"intra_group_alpha must be > 0, got {intra_group_alpha}")
    if head_leakage_scale < 0:
        raise ValueError(f"head_leakage_scale must be >= 0, got {head_leakage_scale}")

    rng = np.random.default_rng(1)
    head_client_count = int(n_parties * head_client_ratio)
    tail_client_count = n_parties - head_client_count
    if head_client_count <= 0 or tail_client_count <= 0:
        raise ValueError("client-longtail requires both head and tail clients")

    head_class_count = int(num_classes * head_class_ratio)
    tail_class_count = num_classes - head_class_count
    if head_class_count <= 0 or tail_class_count <= 0:
        raise ValueError("client-longtail requires both head and tail classes")

    head_clients = list(range(head_client_count))
    tail_clients = list(range(head_client_count, n_parties))
    net_map = {i: np.array([], dtype=np.int64) for i in range(n_parties)}
    head_classes = set(range(head_class_count))
    tail_classes = set(range(head_class_count, num_classes))

    class_counts = {class_id: int(np.sum(labels == class_id)) for class_id in range(num_classes)}
    tail_class_counts = {class_id: class_counts[class_id] for class_id in sorted(tail_classes)}
    non_tail_class_counts = {class_id: class_counts[class_id] for class_id in sorted(head_classes)}
    n_tail = sum(tail_class_counts.values())
    n_non_tail = sum(non_tail_class_counts.values())
    q_t = float(tail_client_ratio)
    lambda_t = float(specialization_lambda)
    rho = float(head_leakage_scale)

    tail_to_tail_budget = int(round(n_tail * (q_t + (1.0 - q_t) * lambda_t)))
    tail_to_tail_budget = min(max(tail_to_tail_budget, 0), n_tail)
    non_tail_to_tail_budget = int(round(rho * n_tail * q_t * (1.0 - lambda_t)))
    non_tail_to_tail_budget = min(max(non_tail_to_tail_budget, 0), n_non_tail)
    tail_budgets = _allocate_class_budgets(tail_class_counts, tail_to_tail_budget)
    non_tail_budgets = _allocate_class_budgets(non_tail_class_counts, non_tail_to_tail_budget)

    for class_id in range(num_classes):
        class_indices = np.where(labels == class_id)[0].astype(np.int64)
        rng.shuffle(class_indices)
        class_to_tail_count = (
            tail_budgets[class_id]
            if class_id in tail_classes
            else non_tail_budgets[class_id]
        )
        to_tail = class_indices[:class_to_tail_count]
        to_head = class_indices[class_to_tail_count:]
        if len(to_tail) > 0:
            _append_dirichlet(to_tail, tail_clients, net_map, intra_group_alpha, rng)
        if len(to_head) > 0:
            _append_dirichlet(to_head, head_clients, net_map, intra_group_alpha, rng)
    return net_map


def _partition_dirichlet_light(
    labels: np.ndarray,
    n_parties: int,
    num_classes: int,
    beta: float,
) -> dict[int, np.ndarray]:
    min_size = 0
    min_require_size = 10
    n_train = labels.shape[0]
    while min_size < min_require_size:
        idx_batch = [[] for _ in range(n_parties)]
        for class_id in range(num_classes):
            idx_k = np.where(labels == class_id)[0]
            np.random.shuffle(idx_k)
            proportions = np.random.dirichlet(np.repeat(beta, n_parties))
            proportions = np.array(
                [p * (len(idx_j) < n_train / n_parties) for p, idx_j in zip(proportions, idx_batch)]
            )
            proportions = proportions / proportions.sum()
            split_points = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
            idx_batch = [
                idx_j + idx.tolist()
                for idx_j, idx in zip(idx_batch, np.split(idx_k, split_points))
            ]
        min_size = min(len(idx_j) for idx_j in idx_batch)
    return {i: np.asarray(idx_batch[i], dtype=np.int64) for i in range(n_parties)}


def _read_pickle(path: Path):
    with path.open("rb") as f:
        return pickle.load(f, encoding="latin1")


def _make_longtail_labels(labels: np.ndarray, num_classes: int, imb_factor: float, imb_type: str) -> np.ndarray:
    if imb_type != "exp":
        raise ValueError(f"Lightweight loader only supports imb_type='exp', got {imb_type}")
    rng = np.random.default_rng(1)
    labels = np.asarray(labels, dtype=np.int64)
    img_max = len(labels) / num_classes
    selected = []
    for class_id in range(num_classes):
        cls_idx = np.where(labels == class_id)[0]
        rng.shuffle(cls_idx)
        keep = int(img_max * (imb_factor ** (class_id / (num_classes - 1.0))))
        selected.extend(cls_idx[:keep].tolist())
    selected = np.asarray(selected, dtype=np.int64)
    return labels[selected]


def _load_cifar_labels_light(datadir: str, dataset: str, imb_factor: float, imb_type: str) -> tuple[np.ndarray, list[str]]:
    root = Path(datadir)
    if dataset == "cifar100_LT":
        data_root = root / "cifar-100-python"
        train_payload = _read_pickle(data_root / "train")
        meta_payload = _read_pickle(data_root / "meta")
        labels = np.asarray(train_payload["fine_labels"], dtype=np.int64)
        classnames = list(meta_payload.get("fine_label_names", [str(i) for i in range(100)]))
        return _make_longtail_labels(labels, 100, imb_factor, imb_type), classnames
    if dataset == "cifar10_LT":
        data_root = root / "cifar-10-batches-py"
        labels_list = []
        for batch_id in range(1, 6):
            payload = _read_pickle(data_root / f"data_batch_{batch_id}")
            labels_list.extend(payload.get("labels", payload.get("fine_labels")))
        meta_payload = _read_pickle(data_root / "batches.meta")
        labels = np.asarray(labels_list, dtype=np.int64)
        classnames = list(meta_payload.get("label_names", [str(i) for i in range(10)]))
        return _make_longtail_labels(labels, 10, imb_factor, imb_type), classnames
    raise RuntimeError(
        "torchvision is not installed and the lightweight fallback only supports CIFAR-LT datasets"
    )


def _partition_data_lt_light(args: argparse.Namespace, split_name: str, partition_name: str):
    random.seed(1)
    np.random.seed(1)
    y_train, classnames = _load_cifar_labels_light(
        dataset_dir(args.data_root, args.dataset),
        args.dataset,
        args.imb_factor,
        args.imb_type,
    )
    num_classes = len(classnames)
    if split_name == "client_lt":
        net_train = _partition_client_longtail_light(
            y_train,
            args.num_users,
            num_classes,
            args.head_client_ratio,
            args.tail_client_ratio,
            args.head_class_ratio,
            args.tail_class_ratio,
            args.specialization_lambda,
            args.intra_group_alpha,
            args.head_leakage_scale,
        )
    elif partition_name == "noniid-labeldir":
        net_train = _partition_dirichlet_light(y_train, args.num_users, num_classes, args.dirichlet_beta)
    else:
        raise ValueError(f"Unsupported lightweight partition: {partition_name}")
    return classnames, net_train, y_train


def counts_matrix(y_train: np.ndarray, net_dataidx_map: dict[int, np.ndarray], num_users: int, num_classes: int) -> np.ndarray:
    counts = np.zeros((num_users, num_classes), dtype=np.int64)
    for client_id in range(num_users):
        indices = np.asarray(net_dataidx_map.get(client_id, []), dtype=np.int64)
        if indices.size == 0:
            continue
        labels = y_train[indices].astype(np.int64)
        counts[client_id] = np.bincount(labels, minlength=num_classes)
    return counts


def class_groups(global_counts: np.ndarray, head_ratio: float, tail_ratio: float) -> dict[int, str]:
    num_classes = len(global_counts)
    order = np.argsort(-global_counts)
    n_head = max(1, int(num_classes * head_ratio))
    n_tail = max(1, int(num_classes * tail_ratio))
    n_head = min(n_head, num_classes - n_tail)
    head = set(order[:n_head].tolist())
    tail = set(order[-n_tail:].tolist())
    groups = {}
    for class_id in range(num_classes):
        if class_id in head:
            groups[class_id] = "head"
        elif class_id in tail:
            groups[class_id] = "tail"
        else:
            groups[class_id] = "middle"
    return groups


def exposure_probability(num_users: int, clients_per_round: int, holders: int) -> float:
    if holders <= 0:
        return 0.0
    if holders >= num_users or clients_per_round >= num_users:
        return 1.0
    if num_users - holders < clients_per_round:
        return 1.0
    miss = math.comb(num_users - holders, clients_per_round) / math.comb(num_users, clients_per_round)
    return 1.0 - miss


def simulate_exposure(
    support: np.ndarray,
    num_users: int,
    frac: float,
    rounds: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    clients_per_round = max(int(frac * num_users), 1)
    rng = np.random.default_rng(seed)
    num_classes = support.shape[1]
    hit_counts = np.zeros(num_classes, dtype=np.int64)
    for _ in range(rounds):
        selected = rng.choice(num_users, size=clients_per_round, replace=False)
        hit_counts += support[selected].any(axis=0).astype(np.int64)
    return hit_counts / max(rounds, 1), hit_counts


def topology_rows(
    split_name: str,
    partition_name: str,
    counts: np.ndarray,
    classnames: list[str],
    groups: dict[int, str],
    frac: float,
    rounds: int,
    schedule_seed: int,
) -> list[dict[str, object]]:
    num_users, num_classes = counts.shape
    support = counts > 0
    global_counts = counts.sum(axis=0)
    sim_rate, sim_hits = simulate_exposure(support, num_users, frac, rounds, schedule_seed)
    clients_per_round = max(int(frac * num_users), 1)
    rows = []
    for class_id in range(num_classes):
        per_client = counts[:, class_id].astype(np.float64)
        total = float(global_counts[class_id])
        holder_count = int(np.count_nonzero(per_client))
        if total > 0:
            concentration = float(np.sum(per_client ** 2) / (total ** 2))
            effective_clients = float(1.0 / concentration) if concentration > 0 else 0.0
            local_depth = float(np.sum(per_client ** 2) / total)
            mass = np.sort(per_client / total)[::-1]
            top1_mass = float(mass[0]) if mass.size else 0.0
            top2_mass = float(mass[:2].sum()) if mass.size else 0.0
        else:
            concentration = 0.0
            effective_clients = 0.0
            local_depth = 0.0
            top1_mass = 0.0
            top2_mass = 0.0
        rows.append(
            {
                "split": split_name,
                "partition": partition_name,
                "class_id": class_id,
                "class_name": classnames[class_id] if class_id < len(classnames) else str(class_id),
                "group": groups[class_id],
                "global_count": int(total),
                "num_clients": holder_count,
                "concentration_C": concentration,
                "effective_clients_Neff": effective_clients,
                "local_depth_D": local_depth,
                "top1_client_mass": top1_mass,
                "top2_client_mass": top2_mass,
                "clients_per_round": clients_per_round,
                "expected_temporal_exposure": exposure_probability(num_users, clients_per_round, holder_count),
                "simulated_temporal_exposure": float(sim_rate[class_id]),
                "simulated_exposure_rounds": int(sim_hits[class_id]),
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str] | None = None) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_counts_csv(path: Path, counts: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["client_id"] + [f"class_{i}" for i in range(counts.shape[1])])
        for client_id, row in enumerate(counts):
            writer.writerow([client_id] + [int(x) for x in row.tolist()])


def summarize(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    metrics = [
        "concentration_C",
        "effective_clients_Neff",
        "local_depth_D",
        "top1_client_mass",
        "top2_client_mass",
        "expected_temporal_exposure",
        "simulated_temporal_exposure",
        "num_clients",
    ]
    splits = sorted({str(r["split"]) for r in rows})
    groups = ["head", "middle", "tail"]
    out = []
    for split in splits:
        for group in groups:
            subset = [r for r in rows if r["split"] == split and r["group"] == group]
            if not subset:
                continue
            row = {"split": split, "group": group, "num_classes": len(subset)}
            for metric in metrics:
                vals = np.asarray([float(r[metric]) for r in subset], dtype=np.float64)
                row[f"{metric}_mean"] = float(vals.mean())
                row[f"{metric}_median"] = float(np.median(vals))
            out.append(row)
    return out


def plot_tail_metric_boxplots(rows: list[dict[str, object]], output_path: Path) -> None:
    metrics = [
        ("concentration_C", "Concentration C"),
        ("local_depth_D", "Local depth D"),
        ("effective_clients_Neff", "Effective clients"),
        ("top2_client_mass", "Top-2 mass"),
        ("expected_temporal_exposure", "Temporal exposure"),
    ]
    splits = ["dirichlet", "client_lt"]
    fig, axes = plt.subplots(1, len(metrics), figsize=(18, 3.8))
    for ax, (metric, title) in zip(axes, metrics):
        data = []
        labels = []
        for split in splits:
            vals = [
                float(r[metric])
                for r in rows
                if r["split"] == split and r["group"] == "tail"
            ]
            data.append(vals)
            labels.append("Dirichlet" if split == "dirichlet" else "Client-LT")
        try:
            ax.boxplot(data, tick_labels=labels, showmeans=True)
        except TypeError:
            ax.boxplot(data, labels=labels, showmeans=True)
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=20)
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle("Tail-class exposure topology under the same global long-tail distribution", y=1.05)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_tail_mass_heatmap(
    split_counts: dict[str, np.ndarray],
    groups: dict[int, str],
    output_path: Path,
) -> None:
    tail_classes = [class_id for class_id, group in groups.items() if group == "tail"]
    client_lt_mass = split_counts["client_lt"][:, tail_classes].astype(np.float64)
    client_lt_totals = client_lt_mass.sum(axis=0, keepdims=True)
    client_lt_mass = np.divide(
        client_lt_mass,
        client_lt_totals,
        out=np.zeros_like(client_lt_mass),
        where=client_lt_totals > 0,
    )
    column_order = np.lexsort(
        (
            -client_lt_mass.max(axis=0),
            client_lt_mass.argmax(axis=0),
        )
    )
    tail_classes = [tail_classes[i] for i in column_order.tolist()]

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.6), sharey=True)
    for ax, split in zip(axes, ["dirichlet", "client_lt"]):
        counts = split_counts[split][:, tail_classes].astype(np.float64)
        totals = counts.sum(axis=0, keepdims=True)
        mass = np.divide(counts, totals, out=np.zeros_like(counts), where=totals > 0)
        im = ax.imshow(mass, aspect="auto", interpolation="nearest", cmap="magma", vmin=0.0, vmax=1.0)
        ax.set_title("Dirichlet" if split == "dirichlet" else "Client-LT")
        ax.set_xlabel("Tail class")
        ax.set_ylabel("Client ID")
        ax.set_xticks([0, max(len(tail_classes) // 2, 1), len(tail_classes) - 1])
        ax.set_xticklabels(["1", str(max(len(tail_classes) // 2 + 1, 1)), str(len(tail_classes))])
        y_ticks = sorted(set([0, 5, 10, 15, split_counts[split].shape[0] - 1]))
        ax.set_yticks(y_ticks)
        ax.set_yticklabels([str(y) for y in y_ticks])
    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.85)
    cbar.set_label("Class mass on client")
    fig.suptitle("Where are tail samples exposed?", y=0.98)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def _tail_metric_values(rows: list[dict[str, object]], split: str, metric: str) -> np.ndarray:
    return np.asarray(
        [float(r[metric]) for r in rows if r["split"] == split and r["group"] == "tail"],
        dtype=np.float64,
    )


def plot_topology_overview(
    rows: list[dict[str, object]],
    split_counts: dict[str, np.ndarray],
    groups: dict[int, str],
    output_path: Path,
) -> None:
    tail_classes = [class_id for class_id, group in groups.items() if group == "tail"]
    client_lt_mass = split_counts["client_lt"][:, tail_classes].astype(np.float64)
    client_lt_totals = client_lt_mass.sum(axis=0, keepdims=True)
    client_lt_mass = np.divide(
        client_lt_mass,
        client_lt_totals,
        out=np.zeros_like(client_lt_mass),
        where=client_lt_totals > 0,
    )
    column_order = np.lexsort((-client_lt_mass.max(axis=0), client_lt_mass.argmax(axis=0)))
    tail_classes = [tail_classes[i] for i in column_order.tolist()]

    fig = plt.figure(figsize=(11.4, 5.4))
    gs = fig.add_gridspec(2, 3, width_ratios=[1.0, 1.15, 1.15], height_ratios=[1.0, 1.0])
    heat_axes = [fig.add_subplot(gs[row, 0]) for row in range(2)]
    bar_ax = fig.add_subplot(gs[:, 1:])

    for ax, split in zip(heat_axes, ["dirichlet", "client_lt"]):
        counts = split_counts[split][:, tail_classes].astype(np.float64)
        totals = counts.sum(axis=0, keepdims=True)
        mass = np.divide(counts, totals, out=np.zeros_like(counts), where=totals > 0)
        ax.imshow(mass, aspect="auto", interpolation="nearest", cmap="magma", vmin=0.0, vmax=1.0)
        ax.set_title("Dirichlet: fragmented" if split == "dirichlet" else "Client-LT: concentrated")
        ax.set_ylabel("Client")
        ax.set_xticks([])
        y_ticks = sorted(set([0, split_counts[split].shape[0] // 2, split_counts[split].shape[0] - 1]))
        ax.set_yticks(y_ticks)
        ax.set_yticklabels([str(y) for y in y_ticks])
    heat_axes[-1].set_xlabel("Tail classes sorted by Client-LT owner")

    metrics = [
        ("top1_client_mass", "Top-1\nmass"),
        ("top2_client_mass", "Top-2\nmass"),
        ("concentration_C", "Concen-\ntration"),
        ("expected_temporal_exposure", "Temporal\nexposure"),
    ]
    x = np.arange(len(metrics))
    width = 0.36
    colors = {"dirichlet": "#4C78A8", "client_lt": "#F58518"}
    for offset, split, label in [(-width / 2, "dirichlet", "Dirichlet"), (width / 2, "client_lt", "Client-LT")]:
        means = []
        errors = []
        for metric, _ in metrics:
            vals = _tail_metric_values(rows, split, metric)
            means.append(float(vals.mean()))
            errors.append(float(vals.std(ddof=1) / math.sqrt(len(vals))) if len(vals) > 1 else 0.0)
        bar_ax.bar(x + offset, means, width, yerr=errors, capsize=3, label=label, color=colors[split], alpha=0.92)

    bar_ax.set_xticks(x)
    bar_ax.set_xticklabels([name for _, name in metrics])
    bar_ax.set_ylabel("Tail-class average")
    bar_ax.set_ylim(0.0, 1.08)
    bar_ax.set_title("Exposure topology statistics")
    bar_ax.legend(frameon=False, loc="upper right")
    bar_ax.grid(axis="y", alpha=0.22)
    bar_ax.text(
        0.02,
        0.98,
        "Client-LT concentrates tail evidence\non fewer, intermittently sampled clients.",
        transform=bar_ax.transAxes,
        va="top",
        ha="left",
        fontsize=10,
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.78, pad=3),
    )

    fig.suptitle("Same global long-tail, different tail exposure topology", y=0.995)
    fig.subplots_adjust(wspace=0.58, hspace=0.34, top=0.86)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=240, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def build_split(args: argparse.Namespace, split_name: str, partition_name: str):
    global partition_data_LT
    if not args.use_project_datasplit:
        classnames, net_train, y_train = _partition_data_lt_light(args, split_name, partition_name)
        return classnames, net_train, y_train

    if partition_data_LT is None:
        from utils.datasplit import partition_data_LT as project_partition_data_LT

        partition_data_LT = project_partition_data_LT

    common_kwargs = dict(
        dataset=args.dataset,
        datadir=dataset_dir(args.data_root, args.dataset),
        partition=partition_name,
        n_parties=args.num_users,
        imb_factor=args.imb_factor,
        imb_type=args.imb_type,
        beta=args.dirichlet_beta,
        logdir=None,
        head_client_ratio=args.head_client_ratio,
        tail_client_ratio=args.tail_client_ratio,
        head_class_ratio=args.head_class_ratio,
        tail_class_ratio=args.tail_class_ratio,
        specialization_lambda=args.specialization_lambda,
        intra_group_alpha=args.intra_group_alpha,
        head_leakage_scale=args.head_leakage_scale,
    )
    (
        _data_train,
        _data_test,
        _lab2cname,
        classnames,
        net_train,
        _net_test,
        _train_counts_dict,
        _test_counts_dict,
        y_train,
    ) = partition_data_LT(**common_kwargs)
    return classnames, net_train, y_train


def main() -> None:
    set_plot_style()
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_rows = []
    split_counts = {}
    reference_groups = None
    classnames = None

    for split_name, partition_name in PARTITION_SPECS:
        classnames, net_train, y_train = build_split(args, split_name, partition_name)
        num_classes = len(classnames)
        counts = counts_matrix(y_train, net_train, args.num_users, num_classes)
        groups = class_groups(counts.sum(axis=0), args.head_class_ratio, args.tail_class_ratio)
        if reference_groups is None:
            reference_groups = groups
        split_counts[split_name] = counts
        write_counts_csv(output_dir / f"client_class_counts_{split_name}.csv", counts)
        rows = topology_rows(
            split_name,
            partition_name,
            counts,
            classnames,
            groups,
            args.frac,
            args.rounds,
            args.schedule_seed,
        )
        all_rows.extend(rows)

    fieldnames = [
        "split",
        "partition",
        "class_id",
        "class_name",
        "group",
        "global_count",
        "num_clients",
        "concentration_C",
        "effective_clients_Neff",
        "local_depth_D",
        "top1_client_mass",
        "top2_client_mass",
        "clients_per_round",
        "expected_temporal_exposure",
        "simulated_temporal_exposure",
        "simulated_exposure_rounds",
    ]
    write_csv(output_dir / "class_topology.csv", all_rows, fieldnames)
    write_csv(output_dir / "summary_by_group.csv", summarize(all_rows))
    plot_topology_overview(all_rows, split_counts, reference_groups or {}, output_dir / "figure1_topology_overview.png")
    plot_tail_metric_boxplots(all_rows, output_dir / "figure1_tail_topology_boxplots.png")
    plot_tail_mass_heatmap(split_counts, reference_groups or {}, output_dir / "figure1_tail_mass_heatmap.png")

    config = vars(args).copy()
    config["partitions"] = dict(PARTITION_SPECS)
    with (output_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    print(f"Saved topology report to {output_dir}")
    print(f"- {output_dir / 'class_topology.csv'}")
    print(f"- {output_dir / 'summary_by_group.csv'}")
    print(f"- {output_dir / 'figure1_topology_overview.png'}")
    print(f"- {output_dir / 'figure1_tail_topology_boxplots.png'}")
    print(f"- {output_dir / 'figure1_tail_mass_heatmap.png'}")


if __name__ == "__main__":
    main()
