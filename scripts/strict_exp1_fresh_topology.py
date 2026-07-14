#!/usr/bin/env python
"""Strict fresh Experiment 1: same Global-LT, different client topology.

This script intentionally does not import the repository's existing partition
or topology-report scripts. It rebuilds the experimental data for Experiment 1
from raw CIFAR-100 training labels, then creates four matched global long-tail
client partitions:

  1. IID + Global-LT
  2. Dirichlet + Global-LT
  3. Client-LT + Global-LT
  4. Hybrid-LT + Global-LT

The invariant is the global class-count vector. Only the client-class evidence
organization is changed.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import pickle
from pathlib import Path
from typing import Iterable

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROTOCOLS = (
    "iid_global_lt",
    "dirichlet_global_lt",
    "client_lt_global_lt",
    "hybrid_lt_global_lt",
)

PROTOCOL_LABELS = {
    "iid_global_lt": "IID + Global-LT",
    "dirichlet_global_lt": "Dirichlet + Global-LT",
    "client_lt_global_lt": "Client-LT + Global-LT",
    "hybrid_lt_global_lt": "Hybrid-LT + Global-LT",
}

PROTOCOL_COLORS = {
    "iid_global_lt": "#4C78A8",
    "dirichlet_global_lt": "#72B7B2",
    "client_lt_global_lt": "#F58518",
    "hybrid_lt_global_lt": "#54A24B",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fresh protocol-controlled topology experiment for Client-LT rationality."
    )
    parser.add_argument("--data-root", default="DATA")
    parser.add_argument("--output-dir", default="output/strict_exp1_fresh_topology")
    parser.add_argument("--dataset", default="cifar100_LT", choices=["cifar100_LT"])
    parser.add_argument("--num-clients", type=int, default=20)
    parser.add_argument("--participation-rate", type=float, default=0.4)
    parser.add_argument("--rounds", type=int, default=100)
    parser.add_argument("--local-epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--global-lt-seed", type=int, default=20260710)
    parser.add_argument("--partition-seed", type=int, default=20260711)
    parser.add_argument("--schedule-seed", type=int, default=20260712)
    parser.add_argument("--imbalance-factor", type=float, default=0.01)
    parser.add_argument("--head-class-ratio", type=float, default=0.30)
    parser.add_argument("--tail-class-ratio", type=float, default=0.20)
    parser.add_argument("--dirichlet-alpha", type=float, default=1.0)
    parser.add_argument("--head-client-ratio", type=float, default=0.80)
    parser.add_argument("--tail-owner-count", type=int, default=2)
    parser.add_argument("--client-tail-alpha", type=float, default=0.30)
    parser.add_argument("--client-head-dispersion-alpha", type=float, default=8.0)
    parser.add_argument("--client-medium-dispersion-alpha", type=float, default=8.0)
    parser.add_argument("--tail-client-head-background-weight", type=float, default=0.05)
    parser.add_argument("--tail-client-medium-background-weight", type=float, default=0.10)
    parser.add_argument("--hybrid-lambda", type=float, default=0.50)
    return parser.parse_args()


def set_plot_style() -> None:
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "figure.titlesize": 13,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def read_pickle(path: Path):
    with path.open("rb") as handle:
        return pickle.load(handle, encoding="latin1")


def load_cifar100_train_labels(data_root: Path) -> tuple[np.ndarray, list[str]]:
    cifar_root = data_root / "cifar-100" / "cifar-100-python"
    train_path = cifar_root / "train"
    meta_path = cifar_root / "meta"
    if not train_path.exists() or not meta_path.exists():
        raise FileNotFoundError(
            f"Missing CIFAR-100 python files under {cifar_root}. Expected train and meta."
        )
    train_payload = read_pickle(train_path)
    meta_payload = read_pickle(meta_path)
    labels = np.asarray(train_payload["fine_labels"], dtype=np.int64)
    classnames = list(meta_payload.get("fine_label_names", [str(i) for i in range(100)]))
    return labels, classnames


def long_tail_counts(num_classes: int, max_count: int, imbalance_factor: float) -> np.ndarray:
    counts = []
    for class_id in range(num_classes):
        exponent = class_id / max(float(num_classes - 1), 1.0)
        count = int(max_count * (float(imbalance_factor) ** exponent))
        counts.append(max(count, 1))
    return np.asarray(counts, dtype=np.int64)


def build_global_lt_pool(
    raw_labels: np.ndarray,
    num_classes: int,
    imbalance_factor: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    raw_counts = np.bincount(raw_labels, minlength=num_classes)
    max_count = int(raw_counts.min())
    target_counts = long_tail_counts(num_classes, max_count, imbalance_factor)
    global_labels: list[int] = []
    source_indices: list[int] = []
    for class_id in range(num_classes):
        class_indices = np.where(raw_labels == class_id)[0].astype(np.int64)
        rng.shuffle(class_indices)
        keep = int(min(target_counts[class_id], len(class_indices)))
        chosen = class_indices[:keep]
        source_indices.extend(chosen.tolist())
        global_labels.extend([class_id] * keep)
    return (
        np.asarray(global_labels, dtype=np.int64),
        np.asarray(source_indices, dtype=np.int64),
        target_counts.astype(np.int64),
    )


def class_groups(
    global_counts: np.ndarray,
    head_ratio: float,
    tail_ratio: float,
) -> dict[int, str]:
    num_classes = int(global_counts.size)
    order = np.argsort(-global_counts)
    n_head = max(1, int(round(num_classes * float(head_ratio))))
    n_tail = max(1, int(round(num_classes * float(tail_ratio))))
    if n_head + n_tail >= num_classes:
        n_head = max(1, min(n_head, num_classes - 2))
        n_tail = max(1, min(n_tail, num_classes - n_head - 1))
    head = set(int(x) for x in order[:n_head])
    tail = set(int(x) for x in order[-n_tail:])
    groups = {}
    for class_id in range(num_classes):
        if class_id in head:
            groups[class_id] = "head"
        elif class_id in tail:
            groups[class_id] = "tail"
        else:
            groups[class_id] = "medium"
    return groups


def integer_counts_from_weights(total: int, weights: np.ndarray) -> np.ndarray:
    if total <= 0:
        return np.zeros_like(weights, dtype=np.int64)
    weights = np.asarray(weights, dtype=np.float64)
    if weights.sum() <= 0 or not np.isfinite(weights).all():
        weights = np.ones_like(weights, dtype=np.float64)
    weights = weights / weights.sum()
    raw = weights * int(total)
    counts = np.floor(raw).astype(np.int64)
    remainder = int(total - counts.sum())
    if remainder > 0:
        order = np.argsort(raw - counts)[::-1]
        counts[order[:remainder]] += 1
    return counts


def allocate_even(total: int, clients: Iterable[int], num_clients: int, rng: np.random.Generator) -> np.ndarray:
    clients = np.asarray(list(clients), dtype=np.int64)
    rng.shuffle(clients)
    out = np.zeros(num_clients, dtype=np.int64)
    if len(clients) == 0 or total <= 0:
        return out
    base = int(total) // len(clients)
    remainder = int(total) % len(clients)
    out[clients] = base
    if remainder > 0:
        out[clients[:remainder]] += 1
    return out


def allocate_dirichlet(
    total: int,
    clients: Iterable[int],
    num_clients: int,
    alpha: float,
    rng: np.random.Generator,
    base_weights: np.ndarray | None = None,
) -> np.ndarray:
    clients = np.asarray(list(clients), dtype=np.int64)
    out = np.zeros(num_clients, dtype=np.int64)
    if len(clients) == 0 or total <= 0:
        return out
    if len(clients) == 1 or alpha <= 0:
        out[int(clients[0])] = int(total)
        return out
    if base_weights is None:
        concentration = np.repeat(float(alpha), len(clients))
    else:
        base_weights = np.asarray(base_weights, dtype=np.float64)
        base_weights = base_weights / base_weights.sum()
        concentration = np.maximum(float(alpha) * len(clients) * base_weights, 1e-3)
    weights = rng.dirichlet(concentration)
    out[clients] = integer_counts_from_weights(int(total), weights)
    return out


def allocate_subset_with_min_support(
    total: int,
    clients: Iterable[int],
    num_clients: int,
    alpha: float,
    rng: np.random.Generator,
) -> np.ndarray:
    clients = np.asarray(list(clients), dtype=np.int64)
    out = np.zeros(num_clients, dtype=np.int64)
    if total <= 0 or len(clients) == 0:
        return out
    support = min(len(clients), int(total))
    selected = clients[:support].copy()
    rng.shuffle(selected)
    out[selected] += 1
    remaining = int(total) - support
    if remaining > 0:
        out += allocate_dirichlet(remaining, selected, num_clients, alpha, rng)
    return out


def build_iid_counts(global_counts: np.ndarray, num_clients: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    counts = np.zeros((num_clients, len(global_counts)), dtype=np.int64)
    all_clients = np.arange(num_clients)
    for class_id, total in enumerate(global_counts.tolist()):
        counts[:, class_id] = allocate_even(int(total), all_clients, num_clients, rng)
    return counts


def build_dirichlet_counts(
    global_counts: np.ndarray,
    num_clients: int,
    alpha: float,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    counts = np.zeros((num_clients, len(global_counts)), dtype=np.int64)
    all_clients = np.arange(num_clients)
    for class_id, total in enumerate(global_counts.tolist()):
        counts[:, class_id] = allocate_dirichlet(int(total), all_clients, num_clients, alpha, rng)
    return counts


def client_type_sets(num_clients: int, head_client_ratio: float) -> tuple[np.ndarray, np.ndarray]:
    num_head_clients = int(round(num_clients * float(head_client_ratio)))
    num_head_clients = min(max(num_head_clients, 1), num_clients - 1)
    head_clients = np.arange(num_head_clients, dtype=np.int64)
    tail_clients = np.arange(num_head_clients, num_clients, dtype=np.int64)
    return head_clients, tail_clients


def weighted_background_counts(
    total: int,
    num_clients: int,
    head_clients: np.ndarray,
    tail_clients: np.ndarray,
    tail_client_weight: float,
    alpha: float,
    rng: np.random.Generator,
) -> np.ndarray:
    clients = np.arange(num_clients, dtype=np.int64)
    weights = np.ones(num_clients, dtype=np.float64)
    weights[tail_clients] = float(tail_client_weight)
    weights[head_clients] = 1.0
    return allocate_dirichlet(total, clients, num_clients, alpha, rng, base_weights=weights)


def build_client_lt_counts(
    global_counts: np.ndarray,
    groups: dict[int, str],
    num_clients: int,
    head_client_ratio: float,
    tail_owner_count: int,
    tail_alpha: float,
    head_dispersion_alpha: float,
    medium_dispersion_alpha: float,
    tail_head_background_weight: float,
    tail_medium_background_weight: float,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    counts = np.zeros((num_clients, len(global_counts)), dtype=np.int64)
    head_clients, tail_clients = client_type_sets(num_clients, head_client_ratio)
    for class_id, total in enumerate(global_counts.tolist()):
        group = groups[class_id]
        if group == "tail":
            k = max(1, min(int(tail_owner_count), len(tail_clients), int(total)))
            chosen = np.asarray(rng.choice(tail_clients, size=k, replace=False), dtype=np.int64)
            counts[:, class_id] = allocate_subset_with_min_support(int(total), chosen, num_clients, tail_alpha, rng)
        elif group == "head":
            counts[:, class_id] = weighted_background_counts(
                int(total),
                num_clients,
                head_clients,
                tail_clients,
                tail_head_background_weight,
                head_dispersion_alpha,
                rng,
            )
        else:
            counts[:, class_id] = weighted_background_counts(
                int(total),
                num_clients,
                head_clients,
                tail_clients,
                tail_medium_background_weight,
                medium_dispersion_alpha,
                rng,
            )
    return counts


def build_hybrid_counts(
    global_counts: np.ndarray,
    groups: dict[int, str],
    num_clients: int,
    dirichlet_alpha: float,
    head_client_ratio: float,
    tail_owner_count: int,
    tail_alpha: float,
    hybrid_lambda: float,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    counts = np.zeros((num_clients, len(global_counts)), dtype=np.int64)
    all_clients = np.arange(num_clients, dtype=np.int64)
    _head_clients, tail_clients = client_type_sets(num_clients, head_client_ratio)
    lam = min(max(float(hybrid_lambda), 0.0), 1.0)
    for class_id, total in enumerate(global_counts.tolist()):
        total = int(total)
        if groups[class_id] != "tail" or lam <= 0:
            counts[:, class_id] = allocate_dirichlet(total, all_clients, num_clients, dirichlet_alpha, rng)
            continue
        concentrated_total = int(round(total * lam))
        background_total = total - concentrated_total
        if concentrated_total > 0:
            k = max(1, min(int(tail_owner_count), len(tail_clients), concentrated_total))
            chosen = np.asarray(rng.choice(tail_clients, size=k, replace=False), dtype=np.int64)
            counts[:, class_id] += allocate_subset_with_min_support(
                concentrated_total,
                chosen,
                num_clients,
                tail_alpha,
                rng,
            )
        if background_total > 0:
            counts[:, class_id] += allocate_dirichlet(
                background_total,
                all_clients,
                num_clients,
                dirichlet_alpha,
                rng,
            )
    return counts


def build_schedule(num_clients: int, participation_rate: float, rounds: int, seed: int) -> list[list[int]]:
    rng = np.random.default_rng(seed)
    clients_per_round = max(1, int(round(num_clients * float(participation_rate))))
    clients_per_round = min(num_clients, clients_per_round)
    return [
        [int(x) for x in rng.choice(num_clients, size=clients_per_round, replace=False).tolist()]
        for _ in range(int(rounds))
    ]


def mean_or_nan(values: list[float]) -> float:
    clean = [float(x) for x in values if np.isfinite(float(x))]
    if not clean:
        return float("nan")
    return float(np.mean(clean))


def exposure_stats(support: np.ndarray, schedule: list[list[int]]) -> tuple[int, float, int]:
    active_rounds = []
    for round_id, clients in enumerate(schedule):
        if np.any(support[np.asarray(clients, dtype=np.int64)]):
            active_rounds.append(round_id)
    if not active_rounds:
        return 0, float("nan"), len(schedule)
    if len(active_rounds) == 1:
        avg_interval = float("nan")
    else:
        avg_interval = float(np.mean(np.diff(np.asarray(active_rounds, dtype=np.float64))))
    missing_gaps = [active_rounds[0]]
    missing_gaps.extend(active_rounds[i] - active_rounds[i - 1] - 1 for i in range(1, len(active_rounds)))
    missing_gaps.append(len(schedule) - 1 - active_rounds[-1])
    return int(len(active_rounds)), avg_interval, int(max(missing_gaps))


def topology_rows(
    protocol: str,
    counts: np.ndarray,
    classnames: list[str],
    groups: dict[int, str],
    schedule: list[list[int]],
) -> list[dict[str, object]]:
    rows = []
    global_counts = counts.sum(axis=0).astype(np.float64)
    client_totals = counts.sum(axis=1).astype(np.float64)
    for class_id in range(counts.shape[1]):
        per_client = counts[:, class_id].astype(np.float64)
        total = float(global_counts[class_id])
        support_mask = per_client > 0
        support_count = int(np.count_nonzero(support_mask))
        if total > 0:
            masses = np.sort(per_client / total)[::-1]
            top1 = float(masses[0])
            top2 = float(masses[:2].sum())
            s2 = float(np.sum(per_client ** 2))
            effective_clients = float((total ** 2) / s2) if s2 > 0 else 0.0
        else:
            top1 = top2 = effective_clients = 0.0
        support_client_totals = client_totals[support_mask]
        support_class_counts = per_client[support_mask]
        local_shares = np.divide(
            support_class_counts,
            support_client_totals,
            out=np.zeros_like(support_class_counts, dtype=np.float64),
            where=support_client_totals > 0,
        )
        local_class_purity = float(local_shares.mean()) if local_shares.size else 0.0
        denom = float(support_client_totals.sum())
        weighted_local_class_purity = float(support_class_counts.sum() / denom) if denom > 0 else 0.0
        active_rounds, avg_interval, max_gap = exposure_stats(support_mask, schedule)
        rows.append(
            {
                "protocol": protocol,
                "protocol_label": PROTOCOL_LABELS[protocol],
                "class_id": int(class_id),
                "class_name": classnames[class_id],
                "class_group": groups[class_id],
                "global_count": int(total),
                "support_client_count": support_count,
                "top1_client_mass": top1,
                "top2_client_mass": top2,
                "effective_client_number": effective_clients,
                "local_class_purity": local_class_purity,
                "weighted_local_class_purity": weighted_local_class_purity,
                "tail_active_rounds": active_rounds,
                "tail_active_rate": float(active_rounds / max(len(schedule), 1)),
                "exposure_interval": avg_interval,
                "max_exposure_gap": max_gap,
            }
        )
    return rows


def summarize_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    metrics = [
        "global_count",
        "support_client_count",
        "top1_client_mass",
        "top2_client_mass",
        "effective_client_number",
        "local_class_purity",
        "weighted_local_class_purity",
        "tail_active_rounds",
        "tail_active_rate",
        "exposure_interval",
        "max_exposure_gap",
    ]
    out = []
    for protocol in PROTOCOLS:
        for group in ("head", "medium", "tail"):
            subset = [row for row in rows if row["protocol"] == protocol and row["class_group"] == group]
            if not subset:
                continue
            summary = {
                "protocol": protocol,
                "protocol_label": PROTOCOL_LABELS[protocol],
                "class_group": group,
                "num_classes": len(subset),
            }
            for metric in metrics:
                vals = np.asarray(
                    [float(row[metric]) for row in subset if np.isfinite(float(row[metric]))],
                    dtype=np.float64,
                )
                if vals.size == 0:
                    summary[f"{metric}_mean"] = ""
                    summary[f"{metric}_median"] = ""
                    summary[f"{metric}_std"] = ""
                else:
                    summary[f"{metric}_mean"] = float(vals.mean())
                    summary[f"{metric}_median"] = float(np.median(vals))
                    summary[f"{metric}_std"] = float(vals.std(ddof=0))
            out.append(summary)
    return out


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str] | None = None) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_counts_csv(path: Path, counts: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["client_id"] + [f"class_{i}" for i in range(counts.shape[1])])
        for client_id, row in enumerate(counts):
            writer.writerow([client_id] + [int(x) for x in row.tolist()])


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def controlled_variable_rows(args: argparse.Namespace, global_counts: np.ndarray, groups: dict[int, str]) -> list[dict[str, object]]:
    group_sizes = {name: sum(1 for group in groups.values() if group == name) for name in ("head", "medium", "tail")}
    variables = [
        ("dataset", args.dataset, "Dataset protocol."),
        ("source_split", "CIFAR-100 train", "Raw label source used to rebuild the LT pool."),
        ("num_classes", len(global_counts), "Number of classes."),
        ("total_train_samples", int(global_counts.sum()), "Total Global-LT training samples."),
        ("max_class_count", int(global_counts.max()), "Head-class sample count."),
        ("min_class_count", int(global_counts.min()), "Tail-class sample count."),
        ("imbalance_factor_observed", float(global_counts.max() / global_counts.min()), "Observed max/min class count."),
        ("head_classes", group_sizes["head"], "Number of head classes."),
        ("medium_classes", group_sizes["medium"], "Number of medium classes."),
        ("tail_classes", group_sizes["tail"], "Number of tail classes."),
        ("num_clients", args.num_clients, "Number of federated clients."),
        ("participation_rate", args.participation_rate, "Client participation rate."),
        ("clients_per_round", max(1, int(round(args.num_clients * args.participation_rate))), "Selected clients per round."),
        ("rounds", args.rounds, "Communication rounds used for temporal exposure metrics."),
        ("local_epochs", args.local_epochs, "Fixed local epoch for the later training protocol."),
        ("batch_size", args.batch_size, "Fixed batch size for the later training protocol."),
        ("global_lt_seed", args.global_lt_seed, "Seed for rebuilding the Global-LT sample pool."),
        ("partition_seed", args.partition_seed, "Seed for client allocation."),
        ("schedule_seed", args.schedule_seed, "Seed for client participation schedule."),
        ("dirichlet_alpha", args.dirichlet_alpha, "Dirichlet alpha for Experiment 1."),
        ("tail_owner_count", args.tail_owner_count, "Client-LT tail owners per tail class."),
        ("hybrid_lambda", args.hybrid_lambda, "Fraction of tail samples routed through Client-LT specialization in Hybrid-LT."),
    ]
    return [{"variable": key, "value": value, "role": role} for key, value, role in variables]


def global_count_verification_rows(split_counts: dict[str, np.ndarray]) -> list[dict[str, object]]:
    reference = next(iter(split_counts.values())).sum(axis=0)
    rows = []
    for protocol, counts in split_counts.items():
        global_counts = counts.sum(axis=0)
        diff = np.abs(global_counts - reference)
        rows.append(
            {
                "protocol": protocol,
                "protocol_label": PROTOCOL_LABELS[protocol],
                "l1_global_count_difference": int(diff.sum()),
                "max_global_count_difference": int(diff.max()) if diff.size else 0,
                "num_mismatched_classes": int(np.count_nonzero(diff)),
                "matches_reference": bool(np.all(diff == 0)),
            }
        )
    return rows


def global_count_rows(
    global_counts: np.ndarray,
    classnames: list[str],
    groups: dict[int, str],
    split_counts: dict[str, np.ndarray],
) -> list[dict[str, object]]:
    rows = []
    protocol_global = {name: counts.sum(axis=0) for name, counts in split_counts.items()}
    for class_id in range(len(global_counts)):
        row = {
            "class_id": class_id,
            "class_name": classnames[class_id],
            "class_group": groups[class_id],
            "reference_global_count": int(global_counts[class_id]),
        }
        for protocol in PROTOCOLS:
            row[f"{protocol}_global_count"] = int(protocol_global[protocol][class_id])
        rows.append(row)
    return rows


def plot_global_counts(global_counts: np.ndarray, groups: dict[int, str], output_path: Path) -> None:
    colors = {"head": "#4C78A8", "medium": "#72B7B2", "tail": "#F58518"}
    x = np.arange(len(global_counts))
    fig, ax = plt.subplots(figsize=(9.0, 3.8))
    ax.bar(x, global_counts, color=[colors[groups[int(i)]] for i in x], width=0.86)
    ax.set_title("Fixed Global-LT class-count vector shared by all protocols")
    ax.set_xlabel("Class ID")
    ax.set_ylabel("Training samples")
    ax.set_xlim(-1, len(global_counts))
    ax.grid(axis="y", alpha=0.22)
    handles = [
        plt.Line2D([0], [0], marker="s", color="none", markerfacecolor=colors[group], markersize=8, label=group)
        for group in ("head", "medium", "tail")
    ]
    ax.legend(handles=handles, frameon=False, ncol=3, loc="upper right")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=240, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def tail_values(rows: list[dict[str, object]], protocol: str, metric: str) -> list[float]:
    return [
        float(row[metric])
        for row in rows
        if row["protocol"] == protocol and row["class_group"] == "tail" and np.isfinite(float(row[metric]))
    ]


def plot_tail_mass(rows: list[dict[str, object]], output_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.8), sharey=True)
    for ax, metric, title in zip(
        axes,
        ("top1_client_mass", "top2_client_mass"),
        ("Tail top-1 client mass", "Tail top-2 client mass"),
    ):
        data = [tail_values(rows, protocol, metric) for protocol in PROTOCOLS]
        labels = [PROTOCOL_LABELS[p].replace(" + ", "\n+ ") for p in PROTOCOLS]
        box = ax.boxplot(data, tick_labels=labels, patch_artist=True, showmeans=True)
        for patch, protocol in zip(box["boxes"], PROTOCOLS):
            patch.set_facecolor(PROTOCOL_COLORS[protocol])
            patch.set_alpha(0.65)
        ax.set_title(title)
        ax.set_ylim(0, 1.05)
        ax.grid(axis="y", alpha=0.25)
        ax.tick_params(axis="x", rotation=15)
    fig.suptitle("Tail evidence concentration across clients", y=1.02)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=240, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_tail_effective_clients(rows: list[dict[str, object]], output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.6, 3.9))
    data = [tail_values(rows, protocol, "effective_client_number") for protocol in PROTOCOLS]
    labels = [PROTOCOL_LABELS[p].replace(" + ", "\n+ ") for p in PROTOCOLS]
    box = ax.boxplot(data, tick_labels=labels, patch_artist=True, showmeans=True)
    for patch, protocol in zip(box["boxes"], PROTOCOLS):
        patch.set_facecolor(PROTOCOL_COLORS[protocol])
        patch.set_alpha(0.65)
    ax.set_title("Tail effective client number")
    ax.set_ylabel("Effective clients")
    ax.grid(axis="y", alpha=0.25)
    ax.tick_params(axis="x", rotation=15)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=240, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_tail_local_purity(rows: list[dict[str, object]], output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.6, 3.9))
    data = [tail_values(rows, protocol, "local_class_purity") for protocol in PROTOCOLS]
    labels = [PROTOCOL_LABELS[p].replace(" + ", "\n+ ") for p in PROTOCOLS]
    box = ax.boxplot(data, tick_labels=labels, patch_artist=True, showmeans=True)
    for patch, protocol in zip(box["boxes"], PROTOCOLS):
        patch.set_facecolor(PROTOCOL_COLORS[protocol])
        patch.set_alpha(0.65)
    ax.set_title("Tail local purity in support clients")
    ax.set_ylabel("Mean local class share")
    ax.grid(axis="y", alpha=0.25)
    ax.tick_params(axis="x", rotation=15)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=240, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_tail_temporal_exposure(rows: list[dict[str, object]], output_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.8))
    specs = [
        ("tail_active_rounds", "Tail active rounds"),
        ("max_exposure_gap", "Max exposure gap"),
    ]
    for ax, (metric, title) in zip(axes, specs):
        data = [tail_values(rows, protocol, metric) for protocol in PROTOCOLS]
        labels = [PROTOCOL_LABELS[p].replace(" + ", "\n+ ") for p in PROTOCOLS]
        box = ax.boxplot(data, tick_labels=labels, patch_artist=True, showmeans=True)
        for patch, protocol in zip(box["boxes"], PROTOCOLS):
            patch.set_facecolor(PROTOCOL_COLORS[protocol])
            patch.set_alpha(0.65)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.25)
        ax.tick_params(axis="x", rotation=15)
    fig.suptitle("Temporal exposure under the same participation schedule", y=1.02)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=240, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_tail_client_mass_heatmap(
    split_counts: dict[str, np.ndarray],
    groups: dict[int, str],
    output_path: Path,
) -> None:
    tail_classes = [class_id for class_id, group in groups.items() if group == "tail"]
    fig, axes = plt.subplots(2, 2, figsize=(9.8, 6.5), sharex=True, sharey=True)
    axes = axes.ravel()
    for ax, protocol in zip(axes, PROTOCOLS):
        counts = split_counts[protocol][:, tail_classes].astype(np.float64)
        totals = counts.sum(axis=0, keepdims=True)
        mass = np.divide(counts, totals, out=np.zeros_like(counts), where=totals > 0)
        im = ax.imshow(mass, aspect="auto", interpolation="nearest", cmap="magma", vmin=0.0, vmax=1.0)
        ax.set_title(PROTOCOL_LABELS[protocol])
        ax.set_xlabel("Tail class")
        ax.set_ylabel("Client")
    cbar = fig.colorbar(im, ax=axes.tolist(), shrink=0.88)
    cbar.set_label("Class mass on client")
    fig.suptitle("Tail class mass distribution across clients", y=0.995)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=240, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def write_paper_notes(path: Path, summary_rows: list[dict[str, object]], verification_rows: list[dict[str, object]]) -> None:
    tail = {
        row["protocol"]: row
        for row in summary_rows
        if row["class_group"] == "tail"
    }
    all_match = all(bool(row["matches_reference"]) for row in verification_rows)

    def metric(protocol: str, name: str) -> float:
        value = tail[protocol].get(f"{name}_mean", "")
        return float(value) if value != "" else float("nan")

    lines = [
        "# Strict Fresh Experiment 1 Notes",
        "",
        f"Global class-count match: {'PASS' if all_match else 'FAIL'}",
        "",
        "This run rebuilds all partitions from the raw CIFAR-100 training labels.",
        "It does not reuse prior experiment outputs or prior topology scripts.",
        "",
        "## Tail Topology Means",
        "",
        "| Protocol | support clients | top1 mass | top2 mass | effective clients | local purity | active rounds | max gap |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for protocol in PROTOCOLS:
        lines.append(
            "| "
            + PROTOCOL_LABELS[protocol]
            + f" | {metric(protocol, 'support_client_count'):.3f}"
            + f" | {metric(protocol, 'top1_client_mass'):.3f}"
            + f" | {metric(protocol, 'top2_client_mass'):.3f}"
            + f" | {metric(protocol, 'effective_client_number'):.3f}"
            + f" | {metric(protocol, 'local_class_purity'):.5f}"
            + f" | {metric(protocol, 'tail_active_rounds'):.3f}"
            + f" | {metric(protocol, 'max_exposure_gap'):.3f} |"
        )
    lines.extend(
        [
            "",
            "## Intended Claim",
            "",
            "All protocols have the same global long-tailed class-count vector.",
            "The differences are client-level evidence topology differences:",
            "support-client count, concentration, effective support, local purity,",
            "and temporal exposure under partial participation.",
            "",
            "Do not phrase this as Client-LT being universally harder. The correct",
            "claim is that Client-LT explicitly controls client-class specialization",
            "while preserving the same global long-tail statistics.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    set_plot_style()
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_labels, classnames = load_cifar100_train_labels(Path(args.data_root))
    num_classes = len(classnames)
    global_labels, source_indices, global_counts = build_global_lt_pool(
        raw_labels,
        num_classes,
        args.imbalance_factor,
        args.global_lt_seed,
    )
    groups = class_groups(global_counts, args.head_class_ratio, args.tail_class_ratio)
    schedule = build_schedule(args.num_clients, args.participation_rate, args.rounds, args.schedule_seed)

    split_counts = {
        "iid_global_lt": build_iid_counts(global_counts, args.num_clients, args.partition_seed + 101),
        "dirichlet_global_lt": build_dirichlet_counts(
            global_counts,
            args.num_clients,
            args.dirichlet_alpha,
            args.partition_seed + 202,
        ),
        "client_lt_global_lt": build_client_lt_counts(
            global_counts,
            groups,
            args.num_clients,
            args.head_client_ratio,
            args.tail_owner_count,
            args.client_tail_alpha,
            args.client_head_dispersion_alpha,
            args.client_medium_dispersion_alpha,
            args.tail_client_head_background_weight,
            args.tail_client_medium_background_weight,
            args.partition_seed + 303,
        ),
        "hybrid_lt_global_lt": build_hybrid_counts(
            global_counts,
            groups,
            args.num_clients,
            args.dirichlet_alpha,
            args.head_client_ratio,
            args.tail_owner_count,
            args.client_tail_alpha,
            args.hybrid_lambda,
            args.partition_seed + 404,
        ),
    }

    all_rows: list[dict[str, object]] = []
    for protocol, counts in split_counts.items():
        write_counts_csv(output_dir / f"client_class_counts_{protocol}.csv", counts)
        all_rows.extend(topology_rows(protocol, counts, classnames, groups, schedule))

    verification_rows = global_count_verification_rows(split_counts)
    summary_rows = summarize_rows(all_rows)
    write_csv(output_dir / "class_topology.csv", all_rows)
    write_csv(output_dir / "summary_by_group.csv", summary_rows)
    write_csv(output_dir / "global_count_verification.csv", verification_rows)
    write_csv(output_dir / "global_class_counts.csv", global_count_rows(global_counts, classnames, groups, split_counts))
    write_csv(output_dir / "controlled_variables.csv", controlled_variable_rows(args, global_counts, groups))
    write_json(output_dir / "participation_schedule.json", {"schedule": schedule})
    write_json(
        output_dir / "config.json",
        {
            **vars(args),
            "protocols": {protocol: PROTOCOL_LABELS[protocol] for protocol in PROTOCOLS},
            "num_global_lt_samples": int(global_counts.sum()),
            "global_lt_source_indices_checksum": int(np.sum(source_indices * (np.arange(len(source_indices)) + 1))),
            "global_lt_labels_checksum": int(np.sum(global_labels * (np.arange(len(global_labels)) + 1))),
        },
    )

    plot_global_counts(global_counts, groups, output_dir / "figure1_global_class_counts.png")
    plot_tail_mass(all_rows, output_dir / "figure2_tail_top1_top2_mass.png")
    plot_tail_effective_clients(all_rows, output_dir / "figure3_tail_effective_clients.png")
    plot_tail_local_purity(all_rows, output_dir / "figure4_tail_local_purity.png")
    plot_tail_temporal_exposure(all_rows, output_dir / "figure5_tail_temporal_exposure.png")
    plot_tail_client_mass_heatmap(split_counts, groups, output_dir / "figure6_tail_client_mass_heatmap.png")
    write_paper_notes(output_dir / "paper_notes.md", summary_rows, verification_rows)

    print("Strict fresh Experiment 1 finished.")
    print(f"Output directory: {output_dir}")
    print(f"Global count match: {all(row['matches_reference'] for row in verification_rows)}")
    for row in summary_rows:
        if row["class_group"] != "tail":
            continue
        print(
            f"- {row['protocol_label']}: "
            f"support={float(row['support_client_count_mean']):.3f}, "
            f"top1={float(row['top1_client_mass_mean']):.3f}, "
            f"top2={float(row['top2_client_mass_mean']):.3f}, "
            f"Neff={float(row['effective_client_number_mean']):.3f}, "
            f"purity={float(row['local_class_purity_mean']):.5f}, "
            f"active_rounds={float(row['tail_active_rounds_mean']):.3f}, "
            f"max_gap={float(row['max_exposure_gap_mean']):.3f}"
        )


if __name__ == "__main__":
    main()
