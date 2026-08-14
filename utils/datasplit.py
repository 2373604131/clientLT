import numpy as np
import logging
import random
import time
import math
from collections import defaultdict
from collections import defaultdict, Counter   # 文件顶部
import torch
import os


def _data_loaders():
    """Import torchvision-backed loaders only when image data is requested.

    Keeping the pure partition helpers importable lets the P0/V1 audit run in
    lightweight environments without changing the training path.
    """
    from utils.dataloader import (
        load_mnist_data,
        load_fmnist_data,
        load_fmnist_LT_data,
        load_cifar10_data,
        load_cifar100_data,
        load_cifar10_LT_data,
        load_cifar100_LT_data,
        load_svhn_data,
        load_celeba_data,
        load_femnist_data,
    )

    return {
        "mnist": load_mnist_data,
        "fmnist": load_fmnist_data,
        "fmnist_LT": load_fmnist_LT_data,
        "cifar10": load_cifar10_data,
        "cifar100": load_cifar100_data,
        "cifar10_LT": load_cifar10_LT_data,
        "cifar100_LT": load_cifar100_LT_data,
        "svhn": load_svhn_data,
        "celeba": load_celeba_data,
        "femnist": load_femnist_data,
    }

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)

def set_random_seed(seed=None):
    rng_seed = (seed if (seed is not None and seed >= 0) else int(time.time()))
    rng = random.Random(rng_seed)
    np.random.seed(rng_seed)

    print(f"Using random seed: {rng_seed}")
    return rng, rng_seed

import numpy as np
import math
from collections import defaultdict

import numpy as np
from collections import defaultdict

def _get_num_classes(dataset, y_train, y_test=None):
    if dataset in ("cifar10", "cifar10_LT", "fmnist", "fmnist_LT"):
        return 10
    if dataset in ("cifar100", "cifar100_LT"):
        return 100
    labels = y_train if y_test is None else np.concatenate([y_train, y_test], axis=0)
    return int(np.max(labels)) + 1


def _validate_ratio_pair(name_a, value_a, name_b, value_b):
    if value_a < 0 or value_b < 0:
        raise ValueError(f"{name_a} and {name_b} must be non-negative, got {value_a} and {value_b}")
    total = value_a + value_b
    if not np.isclose(total, 1.0):
        raise ValueError(f"{name_a} + {name_b} must be 1.0, got {total:.6f}")


def _counts_from_weights(total, weights):
    weights = np.asarray(weights, dtype=np.float64)
    raw_counts = weights * total
    counts = np.floor(raw_counts).astype(int)
    remainder = total - counts.sum()
    if remainder > 0:
        fractional = raw_counts - counts
        order = np.argsort(fractional)[::-1]
        counts[order[:remainder]] += 1
    return counts


def _allocate_class_budgets(class_counts, total_budget):
    class_counts = {int(class_id): int(count) for class_id, count in class_counts.items()}
    total_budget = int(total_budget)
    total_count = sum(class_counts.values())

    if total_budget < 0:
        raise ValueError(f"total_budget must be non-negative, got {total_budget}")
    if total_budget > total_count:
        raise ValueError(
            f"total_budget must not exceed available samples, got {total_budget} > {total_count}"
        )

    budgets = {class_id: 0 for class_id in class_counts}
    if total_budget == 0 or total_count == 0:
        return budgets

    raw_budgets = {
        class_id: (count / total_count) * total_budget
        for class_id, count in class_counts.items()
    }
    for class_id, raw_budget in raw_budgets.items():
        budgets[class_id] = min(int(math.floor(raw_budget)), class_counts[class_id])

    remainder = total_budget - sum(budgets.values())
    while remainder > 0:
        candidates = [
            class_id
            for class_id, count in class_counts.items()
            if budgets[class_id] < count
        ]
        if not candidates:
            raise RuntimeError("Unable to allocate class budgets within class capacities")
        candidates.sort(
            key=lambda class_id: (
                raw_budgets[class_id] - math.floor(raw_budgets[class_id]),
                class_counts[class_id],
                -class_id,
            ),
            reverse=True,
        )
        for class_id in candidates:
            if remainder <= 0:
                break
            budgets[class_id] += 1
            remainder -= 1

    if sum(budgets.values()) != total_budget:
        raise RuntimeError(
            f"Class budget allocation mismatch: expected {total_budget}, got {sum(budgets.values())}"
        )
    for class_id, budget in budgets.items():
        if budget < 0 or budget > class_counts[class_id]:
            raise RuntimeError(
                f"Class budget out of bounds for class {class_id}: {budget} / {class_counts[class_id]}"
            )
    return budgets


def _append_uniform(subset, group_ids, net_map):
    split = np.array_split(subset, len(group_ids))
    for client_id, chunk in zip(group_ids, split):
        if len(chunk) == 0:
            continue
        net_map[client_id] = np.append(net_map[client_id], chunk.astype(np.int64))


def _append_dirichlet(subset, group_ids, net_map, alpha, rng):
    counts = _counts_from_weights(
        len(subset),
        rng.dirichlet(np.repeat(alpha, len(group_ids)))
    )
    offset = 0
    for client_id, count in zip(group_ids, counts):
        if count <= 0:
            continue
        chunk = subset[offset:offset + count]
        offset += count
        if len(chunk) == 0:
            continue
        net_map[client_id] = np.append(net_map[client_id], chunk.astype(np.int64))


def _bottom_classes_from_counts(labels, num_classes, tail_class_ratio):
    """Return the LT tail while preserving the generator's class order.

    Exponential LT construction can give adjacent boundary classes identical
    integer counts.  Larger ids are later in the LT schedule and therefore win
    an equal-count tie for tail membership.
    """
    labels = np.asarray(labels, dtype=np.int64)
    tail_count = max(1, int(round(int(num_classes) * float(tail_class_ratio))))
    tail_count = min(tail_count, int(num_classes))
    class_counts = {
        class_id: int(np.sum(labels == class_id))
        for class_id in range(int(num_classes))
    }
    return sorted(class_counts, key=lambda class_id: (class_counts[class_id], -class_id))[
        :tail_count
    ]


def _append_dirichlet_with_capacities(
    subset,
    group_ids,
    net_map,
    capacities,
    alpha,
    rng,
):
    """Append ``subset`` using Dirichlet preferences without exceeding capacities."""
    subset = np.asarray(subset, dtype=np.int64)
    group_ids = [int(client_id) for client_id in group_ids]
    if len(subset) == 0:
        return
    if len(subset) > sum(int(capacities[client_id]) for client_id in group_ids):
        raise ValueError("Subset exceeds the remaining client capacities")

    preferences = rng.dirichlet(np.repeat(float(alpha), len(group_ids)))
    assigned = {client_id: [] for client_id in group_ids}
    for sample_index in subset.tolist():
        available_positions = [
            position
            for position, client_id in enumerate(group_ids)
            if int(capacities[client_id]) > 0
        ]
        if not available_positions:
            raise RuntimeError("Ran out of controlled Client-LT companion capacity")
        probabilities = preferences[available_positions].astype(np.float64)
        probabilities /= probabilities.sum()
        selected_position = int(rng.choice(available_positions, p=probabilities))
        client_id = group_ids[selected_position]
        assigned[client_id].append(int(sample_index))
        capacities[client_id] = int(capacities[client_id]) - 1

    for client_id, indices in assigned.items():
        if indices:
            net_map[client_id] = np.append(
                net_map[client_id], np.asarray(indices, dtype=np.int64)
            )


def partition_client_longtail_controlled(
    labels,
    n_parties,
    num_classes,
    head_client_ratio=0.9,
    tail_client_ratio=0.1,
    tail_class_ratio=0.2,
    *,
    intra_group_alpha,
    tail_client_min_purity=0.8,
    tail_class_ids=None,
    rng=None,
):
    """Build the controlled Client-LT diagnostic split.

    Every sample from a train-derived tail class is assigned to the designated
    tail-client group. Non-tail samples may enter that group only up to each
    client's integer purity capacity ``floor(T_k * (1-purity) / purity)``.
    The legacy :func:`partition_client_longtail` behavior is intentionally
    unchanged.
    """
    if rng is None:
        rng = np.random.RandomState(1)
    _validate_ratio_pair(
        "head_client_ratio", head_client_ratio, "tail_client_ratio", tail_client_ratio
    )
    if not 0.0 < float(tail_client_min_purity) <= 1.0:
        raise ValueError(
            "tail_client_min_purity must be in (0, 1], got "
            f"{tail_client_min_purity}"
        )
    if intra_group_alpha is None or float(intra_group_alpha) <= 0:
        raise ValueError(f"intra_group_alpha must be > 0, got {intra_group_alpha}")

    head_client_count = int(int(n_parties) * float(head_client_ratio))
    tail_client_count = int(n_parties) - head_client_count
    if head_client_count <= 0 or tail_client_count <= 0:
        raise ValueError("controlled Client-LT requires head and tail clients")
    head_clients = list(range(head_client_count))
    tail_clients = list(range(head_client_count, int(n_parties)))

    labels = np.asarray(labels, dtype=np.int64)
    if tail_class_ids is None:
        tail_class_ids = _bottom_classes_from_counts(
            labels, num_classes, tail_class_ratio
        )
    tail_classes = sorted({int(class_id) for class_id in tail_class_ids})
    expected_tail_count = max(
        1, int(round(int(num_classes) * float(tail_class_ratio)))
    )
    if len(tail_classes) != expected_tail_count:
        raise ValueError(
            f"Expected {expected_tail_count} tail classes, got {len(tail_classes)}"
        )
    if any(class_id < 0 or class_id >= int(num_classes) for class_id in tail_classes):
        raise ValueError(f"Invalid tail class ids: {tail_classes}")
    tail_set = set(tail_classes)
    non_tail_classes = [
        class_id for class_id in range(int(num_classes)) if class_id not in tail_set
    ]

    net_dataidx_map = {
        client_id: np.asarray([], dtype=np.int64)
        for client_id in range(int(n_parties))
    }

    # First place every tail sample inside the specialist group.
    for class_id in tail_classes:
        class_indices = np.where(labels == class_id)[0].astype(np.int64)
        rng.shuffle(class_indices)
        _append_dirichlet(
            class_indices,
            tail_clients,
            net_dataidx_map,
            float(intra_group_alpha),
            rng,
        )

    tail_counts_by_client = {
        client_id: int(len(net_dataidx_map[client_id])) for client_id in tail_clients
    }
    if any(count <= 0 for count in tail_counts_by_client.values()):
        raise RuntimeError(
            "Controlled Client-LT produced an empty tail client before companion "
            f"allocation: {tail_counts_by_client}"
        )

    companion_ratio = (1.0 - float(tail_client_min_purity)) / float(
        tail_client_min_purity
    )
    capacities = {
        client_id: int(math.floor(count * companion_ratio + 1e-12))
        for client_id, count in tail_counts_by_client.items()
    }
    companion_budget = sum(capacities.values())
    non_tail_indices = np.where(~np.isin(labels, tail_classes))[0].astype(np.int64)
    companion_budget = min(companion_budget, len(non_tail_indices))
    # Select individual images uniformly from the real non-tail pool. This is
    # frequency-proportional in expectation and, unlike a class-budget largest
    # remainder rule, does not mechanically maximize companion-class breadth.
    rng.shuffle(non_tail_indices)
    selected_companions = set(non_tail_indices[:companion_budget].tolist())

    # Select non-tail companions without consulting semantic similarity, then
    # distribute them under the per-client purity capacities.
    for class_id in non_tail_classes:
        class_indices = np.where(labels == class_id)[0].astype(np.int64)
        rng.shuffle(class_indices)
        to_tail_indices = np.asarray(
            [index for index in class_indices if int(index) in selected_companions],
            dtype=np.int64,
        )
        to_head_indices = np.asarray(
            [index for index in class_indices if int(index) not in selected_companions],
            dtype=np.int64,
        )
        _append_dirichlet_with_capacities(
            to_tail_indices,
            tail_clients,
            net_dataidx_map,
            capacities,
            float(intra_group_alpha),
            rng,
        )
        _append_dirichlet(
            to_head_indices,
            head_clients,
            net_dataidx_map,
            float(intra_group_alpha),
            rng,
        )

    _validate_partition_map(
        labels,
        net_dataidx_map,
        int(n_parties),
        int(num_classes),
        "client-longtail-controlled",
    )

    actual_tail_counts = {}
    actual_companion_counts = {}
    for client_id in tail_clients:
        client_labels = labels[np.asarray(net_dataidx_map[client_id], dtype=np.int64)]
        actual_tail = int(np.isin(client_labels, tail_classes).sum())
        actual_companion = int(len(client_labels) - actual_tail)
        actual_tail_counts[client_id] = actual_tail
        actual_companion_counts[client_id] = actual_companion
        denominator = actual_tail + actual_companion
        purity = actual_tail / denominator if denominator else 0.0
        if purity + 1e-12 < float(tail_client_min_purity):
            raise RuntimeError(
                f"Tail client {client_id} purity {purity:.6f} is below "
                f"{tail_client_min_purity:.6f}"
            )

    head_indices = np.concatenate(
        [np.asarray(net_dataidx_map[client_id], dtype=np.int64) for client_id in head_clients]
    )
    if np.isin(labels[head_indices], tail_classes).any():
        raise RuntimeError("Controlled Client-LT leaked tail samples to head clients")

    logger.info(
        "Controlled ClientLT: tail_classes=%s tail_samples=%d "
        "companion_samples=%d min_purity=%.6f tail_counts=%s companion_counts=%s",
        tail_classes,
        sum(actual_tail_counts.values()),
        sum(actual_companion_counts.values()),
        float(tail_client_min_purity),
        actual_tail_counts,
        actual_companion_counts,
    )
    return net_dataidx_map


# ClientLT uses separate controls for tail specialization, within-group
# concentration, and non-tail leakage into tail clients.
def partition_client_longtail(
    labels, n_parties, num_classes,
    head_client_ratio=0.9, tail_client_ratio=0.1,
    head_class_ratio=0.8, tail_class_ratio=0.2,
    *,
    specialization_lambda,
    intra_group_alpha,
    head_leakage_scale,
    rng=None,
):
    if rng is None:
        rng = np.random.RandomState(1)

    _validate_ratio_pair("head_client_ratio", head_client_ratio, "tail_client_ratio", tail_client_ratio)
    _validate_ratio_pair("head_class_ratio", head_class_ratio, "tail_class_ratio", tail_class_ratio)
    if specialization_lambda < 0.0 or specialization_lambda > 1.0:
        raise ValueError(f"specialization_lambda must be in [0.0, 1.0], got {specialization_lambda}")
    if intra_group_alpha is None or intra_group_alpha <= 0:
        raise ValueError(f"intra_group_alpha must be > 0, got {intra_group_alpha}")
    if head_leakage_scale < 0:
        raise ValueError(f"head_leakage_scale must be >= 0, got {head_leakage_scale}")

    head_client_count = int(n_parties * head_client_ratio)
    tail_client_count = n_parties - head_client_count
    if head_client_count <= 0 or tail_client_count <= 0:
        raise ValueError(
            "client-longtail requires both head and tail clients. "
            f"Got head_client_count={head_client_count}, tail_client_count={tail_client_count}. "
            "Adjust head_client_ratio/tail_client_ratio or num_clients."
        )

    head_class_count = int(num_classes * head_class_ratio)
    tail_class_count = num_classes - head_class_count
    if head_class_count <= 0 or tail_class_count <= 0:
        raise ValueError(
            "client-longtail requires both head and tail classes. "
            f"Got head_class_count={head_class_count}, tail_class_count={tail_class_count}. "
            "Adjust head_class_ratio/tail_class_ratio or num_classes."
        )

    labels = np.asarray(labels)
    head_clients = list(range(head_client_count))
    tail_clients = list(range(head_client_count, n_parties))
    net_dataidx_map = {i: np.array([], dtype=np.int64) for i in range(n_parties)}

    head_classes = set(range(head_class_count))
    tail_classes = set(range(head_class_count, num_classes))
    class_counts = {
        class_id: int(np.sum(labels == class_id))
        for class_id in range(num_classes)
    }
    tail_class_counts = {
        class_id: class_counts[class_id]
        for class_id in sorted(tail_classes)
    }
    non_tail_class_counts = {
        class_id: class_counts[class_id]
        for class_id in sorted(head_classes)
    }

    N_tail = sum(tail_class_counts.values())
    N_non_tail = sum(non_tail_class_counts.values())
    q_t = float(tail_client_ratio)
    lambda_t = float(specialization_lambda)
    rho = float(head_leakage_scale)

    tail_to_tail_ratio = q_t + (1.0 - q_t) * lambda_t
    tail_to_tail_budget = int(round(N_tail * tail_to_tail_ratio))
    tail_to_tail_budget = min(max(tail_to_tail_budget, 0), N_tail)
    tail_to_head_budget = N_tail - tail_to_tail_budget

    non_tail_to_tail_budget = int(round(rho * N_tail * q_t * (1.0 - lambda_t)))
    non_tail_to_tail_budget = min(max(non_tail_to_tail_budget, 0), N_non_tail)
    non_tail_to_head_budget = N_non_tail - non_tail_to_tail_budget

    tail_class_to_tail_budgets = _allocate_class_budgets(tail_class_counts, tail_to_tail_budget)
    non_tail_class_to_tail_budgets = _allocate_class_budgets(
        non_tail_class_counts,
        non_tail_to_tail_budget,
    )

    for class_id in range(num_classes):
        class_indices = np.where(labels == class_id)[0].astype(np.int64)
        rng.shuffle(class_indices)
        if class_id in tail_classes:
            class_to_tail_count = tail_class_to_tail_budgets[class_id]
        else:
            class_to_tail_count = non_tail_class_to_tail_budgets[class_id]

        to_tail_indices = class_indices[:class_to_tail_count]
        to_head_indices = class_indices[class_to_tail_count:]

        if len(to_tail_indices) > 0:
            _append_dirichlet(
                to_tail_indices,
                tail_clients,
                net_dataidx_map,
                intra_group_alpha,
                rng,
            )
        if len(to_head_indices) > 0:
            _append_dirichlet(
                to_head_indices,
                head_clients,
                net_dataidx_map,
                intra_group_alpha,
                rng,
            )

    _validate_partition_map(labels, net_dataidx_map, n_parties, num_classes, "client-longtail")

    tail_client_indices = np.concatenate(
        [np.asarray(net_dataidx_map[client_id], dtype=np.int64) for client_id in tail_clients]
    )
    tail_client_labels = labels[tail_client_indices] if len(tail_client_indices) > 0 else np.asarray([], dtype=labels.dtype)
    actual_tail_client_tail_samples = int(np.isin(tail_client_labels, list(tail_classes)).sum())
    actual_tail_client_non_tail_samples = int(len(tail_client_labels) - actual_tail_client_tail_samples)
    if actual_tail_client_tail_samples != tail_to_tail_budget:
        raise RuntimeError(
            "Tail-client tail sample count mismatch: "
            f"expected {tail_to_tail_budget}, got {actual_tail_client_tail_samples}"
        )
    if actual_tail_client_non_tail_samples != non_tail_to_tail_budget:
        raise RuntimeError(
            "Tail-client non-tail sample count mismatch: "
            f"expected {non_tail_to_tail_budget}, got {actual_tail_client_non_tail_samples}"
        )

    denominator = actual_tail_client_tail_samples + actual_tail_client_non_tail_samples
    actual_tail_client_purity = (
        actual_tail_client_tail_samples / denominator
        if denominator > 0
        else 0.0
    )
    logger.info(
        "ClientLT budgets: "
        "specialization_lambda=%.6f intra_group_alpha=%.6f head_leakage_scale=%.6f "
        "N_tail=%d N_non_tail=%d tail_to_tail_budget=%d tail_to_head_budget=%d "
        "non_tail_to_tail_budget=%d non_tail_to_head_budget=%d "
        "actual_tail_client_tail_samples=%d actual_tail_client_non_tail_samples=%d "
        "actual_tail_client_purity=%.6f",
        specialization_lambda,
        intra_group_alpha,
        head_leakage_scale,
        N_tail,
        N_non_tail,
        tail_to_tail_budget,
        tail_to_head_budget,
        non_tail_to_tail_budget,
        non_tail_to_head_budget,
        actual_tail_client_tail_samples,
        actual_tail_client_non_tail_samples,
        actual_tail_client_purity,
    )

    return net_dataidx_map


def _validate_partition_map(labels, net_dataidx_map, n_parties, num_classes, split_name):
    labels = np.asarray(labels)
    expected_keys = set(range(n_parties))
    actual_keys = set(net_dataidx_map.keys())
    if actual_keys != expected_keys:
        raise RuntimeError(
            f"{split_name} partition keys mismatch: expected {sorted(expected_keys)}, got {sorted(actual_keys)}"
        )

    arrays = []
    for client_id in range(n_parties):
        values = np.asarray(net_dataidx_map[client_id], dtype=np.int64)
        if values.ndim != 1:
            raise RuntimeError(f"{split_name} client {client_id} indices must be one-dimensional")
        arrays.append(values)

    if arrays:
        merged = np.concatenate(arrays) if sum(len(a) for a in arrays) > 0 else np.asarray([], dtype=np.int64)
    else:
        merged = np.asarray([], dtype=np.int64)

    if len(merged) != len(labels):
        raise RuntimeError(
            f"{split_name} partition coverage length mismatch: expected {len(labels)}, got {len(merged)}"
        )
    if np.unique(merged).size != len(labels):
        raise RuntimeError(f"{split_name} partition contains duplicate or missing indices")
    if not np.array_equal(np.sort(merged), np.arange(len(labels), dtype=np.int64)):
        raise RuntimeError(f"{split_name} partition index set is not exactly 0..{len(labels) - 1}")

    original_counts = np.bincount(labels.astype(np.int64), minlength=num_classes)[:num_classes]
    assigned_counts = np.bincount(labels[merged].astype(np.int64), minlength=num_classes)[:num_classes]
    if not np.array_equal(original_counts, assigned_counts):
        raise RuntimeError(f"{split_name} partition class counts are not conserved")


def _append_class_indices(target_lists, indices, counts):
    offset = 0
    for client_id, count in enumerate(counts):
        if count > 0:
            target_lists[client_id].extend(indices[offset:offset + count].tolist())
        offset += int(count)


def partition_fine_class_dirichlet(
    y_train,
    y_test,
    n_parties,
    num_classes,
    beta,
    split_seed,
    min_client_train_samples=10,
    max_retries=1000,
):
    if beta <= 0:
        raise ValueError(f"beta must be > 0, got {beta}")
    if n_parties <= 1:
        raise ValueError(f"n_parties must be > 1, got {n_parties}")
    if max_retries <= 0:
        raise ValueError(f"max_retries must be > 0, got {max_retries}")

    y_train = np.asarray(y_train, dtype=np.int64)
    y_test = np.asarray(y_test, dtype=np.int64)
    base_seed = int(split_seed)
    last_min_client_size = None

    for retry in range(int(max_retries)):
        seed_offset = retry * 3
        proportion_rng = np.random.RandomState(base_seed + seed_offset)
        train_rng = np.random.RandomState(base_seed + 1_000_003 + seed_offset)
        test_rng = np.random.RandomState(base_seed + 2_000_003 + seed_offset)

        train_lists = {client_id: [] for client_id in range(n_parties)}
        test_lists = {client_id: [] for client_id in range(n_parties)}

        for class_id in range(num_classes):
            proportions = proportion_rng.dirichlet(np.repeat(float(beta), n_parties))

            train_indices = np.where(y_train == class_id)[0].astype(np.int64).copy()
            test_indices = np.where(y_test == class_id)[0].astype(np.int64).copy()

            train_rng.shuffle(train_indices)
            test_rng.shuffle(test_indices)

            train_counts = train_rng.multinomial(len(train_indices), proportions)
            test_counts = test_rng.multinomial(len(test_indices), proportions)

            _append_class_indices(train_lists, train_indices, train_counts)
            _append_class_indices(test_lists, test_indices, test_counts)

        net_dataidx_map_train = {
            client_id: np.asarray(train_lists[client_id], dtype=np.int64)
            for client_id in range(n_parties)
        }
        net_dataidx_map_test = {
            client_id: np.asarray(test_lists[client_id], dtype=np.int64)
            for client_id in range(n_parties)
        }

        last_min_client_size = min(len(net_dataidx_map_train[client_id]) for client_id in range(n_parties))
        if last_min_client_size < min_client_train_samples:
            continue

        _validate_partition_map(y_train, net_dataidx_map_train, n_parties, num_classes, "train")
        _validate_partition_map(y_test, net_dataidx_map_test, n_parties, num_classes, "test")
        return net_dataidx_map_train, net_dataidx_map_test

    raise RuntimeError(
        "Unable to satisfy fine-class Dirichlet min client size: "
        f"beta={beta}, n_parties={n_parties}, "
        f"min_client_train_samples={min_client_train_samples}, "
        f"max_retries={max_retries}, last_min_client_size={last_min_client_size}"
    )


def record_net_data_stats(y_train, net_dataidx_map, logdir=None):

    net_cls_counts = {}

    for net_i, dataidx in net_dataidx_map.items():
        unq, unq_cnt = np.unique(y_train[dataidx], return_counts=True)
        tmp = {unq[i]: unq_cnt[i] for i in range(len(unq))}
        net_cls_counts[net_i] = tmp
    if logdir != None:
        logger.info('Data statistics: %s' % str(net_cls_counts))

    return net_cls_counts

def renormalize(weights, index):
    """
    :param weights: vector of non negative weights summing to 1.
    :type weights: numpy.array
    :param index: index of the weight to remove
    :type index: int
    """
    renormalized_weights = np.delete(weights, index)
    renormalized_weights /= renormalized_weights.sum()

    return renormalized_weights

def partition_data(
    dataset, datadir, partition, n_parties, beta=0.4, logdir=None,
    head_client_ratio=0.9, tail_client_ratio=0.1,
    head_class_ratio=0.8, tail_class_ratio=0.2,
    *,
    specialization_lambda,
    intra_group_alpha,
    head_leakage_scale,
):
    # np.random.seed(2020)
    # torch.manual_seed(2020)

    loaders = _data_loaders()
    if dataset == 'mnist':
        X_train, y_train, X_test, y_test = loaders["mnist"](datadir)
    elif dataset == 'fmnist':
        X_train, y_train, X_test, y_test, data_train, data_test, lab2cname, classnames = loaders["fmnist"](datadir)
        y = np.concatenate([y_train, y_test], axis=0)
    elif dataset == 'cifar10':
        X_train, y_train, X_test, y_test, data_train, data_test, lab2cname, classnames = loaders["cifar10"](datadir)
        y = np.concatenate([y_train, y_test], axis=0)
    elif dataset == 'cifar100':
        # X_train, y_train, X_test, y_test = load_cifar100_data(datadir)
        X_train, y_train, X_test, y_test, data_train, data_test, lab2cname, classnames = loaders["cifar100"](datadir)
        y = np.concatenate([y_train, y_test], axis=0)

    elif dataset == 'svhn':
        X_train, y_train, X_test, y_test = loaders["svhn"](datadir)
    elif dataset == 'celeba':
        X_train, y_train, X_test, y_test = loaders["celeba"](datadir)
    elif dataset == 'femnist':
        X_train, y_train, u_train, X_test, y_test, u_test = loaders["femnist"](datadir)
    elif dataset == 'generated':
        X_train, y_train = [], []
        for loc in range(4):
            for i in range(1000):
                p1 = random.random()
                p2 = random.random()
                p3 = random.random()
                if loc > 1:
                    p2 = -p2
                if loc % 2 ==1:
                    p3 = -p3
                if i % 2 == 0:
                    X_train.append([p1, p2, p3])
                    y_train.append(0)
                else:
                    X_train.append([-p1, -p2, -p3])
                    y_train.append(1)
        X_test, y_test = [], []
        for i in range(1000):
            p1 = random.random() * 2 - 1
            p2 = random.random() * 2 - 1
            p3 = random.random() * 2 - 1
            X_test.append([p1, p2, p3])
            if p1 >0:
                y_test.append(0)
            else:
                y_test.append(1)
        X_train = np.array(X_train, dtype=np.float32)
        X_test = np.array(X_test, dtype=np.float32)
        y_train = np.array(y_train, dtype=np.int32)
        y_test = np.array(y_test, dtype=np.int64)
        idxs = np.linspace(0 ,3999 ,4000 ,dtype=np.int64)
        batch_idxs = np.array_split(idxs, n_parties)
        net_dataidx_map = {i: batch_idxs[i] for i in range(n_parties)}
        os.makedirs("data/generated/", exist_ok=True)
        np.save("data/generated/X_train.npy" ,X_train)
        np.save("data/generated/X_test.npy" ,X_test)
        np.save("data/generated/y_train.npy" ,y_train)
        np.save("data/generated/y_test.npy" ,y_test)

    n_train = y_train.shape[0]
    n_test = y_test.shape[0]

    if partition == "homo":
        idxs_train = np.random.permutation(n_train)
        idxs_test = np.random.permutation(n_test)

        batch_idxs_train = np.array_split(idxs_train, n_parties)
        batch_idxs_test = np.array_split(idxs_test, n_parties)

        net_dataidx_map_train = {i: batch_idxs_train[i] for i in range(n_parties)}
        net_dataidx_map_test = {i: batch_idxs_test[i] for i in range(n_parties)}

    elif partition == "iid-label100":
        seed = 12345
        n_fine_labels = 100
        n_coarse_labels = 20
        coarse_labels = \
            np.array([
                4, 1, 14, 8, 0, 6, 7, 7, 18, 3,
                3, 14, 9, 18, 7, 11, 3, 9, 7, 11,
                6, 11, 5, 10, 7, 6, 13, 15, 3, 15,
                0, 11, 1, 10, 12, 14, 16, 9, 11, 5,
                5, 19, 8, 8, 15, 13, 14, 17, 18, 10,
                16, 4, 17, 4, 2, 0, 17, 4, 18, 17,
                10, 3, 2, 12, 12, 16, 12, 1, 9, 19,
                2, 10, 0, 1, 16, 12, 9, 13, 15, 13,
                16, 19, 2, 4, 6, 19, 5, 5, 8, 19,
                18, 1, 2, 15, 6, 0, 17, 8, 14, 13
            ])
        rng_seed = (seed if (seed is not None and seed >= 0) else int(time.time()))
        rng = random.Random(rng_seed)
        np.random.seed(rng_seed)

        n_samples_train = y_train.shape[0]
        n_samples_test = y_test.shape[0]

        selected_indices_train = rng.sample(list(range(n_samples_train)), n_samples_train)
        selected_indices_test = rng.sample(list(range(n_samples_test)), n_samples_test)

        n_samples_by_client_train = int((n_samples_train / n_parties) // 5)
        n_samples_by_client_test = int((n_samples_test / n_parties) // 5)

        indices_by_fine_labels_train = {k: list() for k in range(n_fine_labels)}
        indices_by_coarse_labels_train = {k: list() for k in range(n_coarse_labels)}

        indices_by_fine_labels_test = {k: list() for k in range(n_fine_labels)}
        indices_by_coarse_labels_test = {k: list() for k in range(n_coarse_labels)}

        for idx in selected_indices_train:
            fine_label = y_train[idx]
            coarse_label = coarse_labels[fine_label]

            indices_by_fine_labels_train[fine_label].append(idx)
            indices_by_coarse_labels_train[coarse_label].append(idx)

        for idx in selected_indices_test:
            fine_label = y_test[idx]
            coarse_label = coarse_labels[fine_label]

            indices_by_fine_labels_test[fine_label].append(idx)
            indices_by_coarse_labels_test[coarse_label].append(idx)

        fine_labels_by_coarse_labels = {k: list() for k in range(n_coarse_labels)}

        for fine_label, coarse_label in enumerate(coarse_labels):
            fine_labels_by_coarse_labels[coarse_label].append(fine_label)

        net_dataidx_map_train = {i: np.ndarray(0, dtype=np.int64) for i in range(n_parties)}
        net_dataidx_map_test = {i: np.ndarray(0, dtype=np.int64) for i in range(n_parties)}

        for client_idx in range(n_parties):
            coarse_idx = client_idx // 5
            fine_idx = fine_labels_by_coarse_labels[coarse_idx]
            for k in range(5):
                fine_label = fine_idx[k]
                sample_idx = rng.sample(list(indices_by_fine_labels_train[fine_label]), n_samples_by_client_train)
                net_dataidx_map_train[client_idx] = np.append(net_dataidx_map_train[client_idx], sample_idx)
                for idx in sample_idx:
                    indices_by_fine_labels_train[fine_label].remove(idx)

        for client_idx in range(n_parties):
            coarse_idx = client_idx // 5
            fine_idx = fine_labels_by_coarse_labels[coarse_idx]
            for k in range(5):
                fine_label = fine_idx[k]
                sample_idx = rng.sample(list(indices_by_fine_labels_test[fine_label]), n_samples_by_client_test)
                net_dataidx_map_test[client_idx] = np.append(net_dataidx_map_test[client_idx], sample_idx)
                for idx in sample_idx:
                    indices_by_fine_labels_test[fine_label].remove(idx)

    elif partition == "noniid-labeluni":
        if dataset == "cifar10":
            num = 2
        elif dataset == "cifar100":
            num = 10
        if dataset in ('celeba', 'covtype', 'a9a', 'rcv1', 'SUSY'):
            num = 1
            K = 2
        elif dataset == 'cifar100':
            K = 100
        elif dataset == 'cifar10':
            K = 10
        else:
            assert False
            print("Choose Dataset in readme.")

        # -------------------------------------------#
        # Divide classes + num samples for each user #
        # -------------------------------------------#
        assert (num * n_parties) % K == 0, "equal classes appearance is needed"
        count_per_class = (num * n_parties) // K
        class_dict = {}
        for i in range(K):
            # sampling alpha_i_c
            probs = np.random.uniform(0.4, 0.6, size=count_per_class)
            # normalizing
            probs_norm = (probs / probs.sum()).tolist()
            class_dict[i] = {'count': count_per_class, 'prob': probs_norm}

        # -------------------------------------#
        # Assign each client with data indexes #
        # -------------------------------------#
        class_partitions = defaultdict(list)
        for i in range(n_parties):
            c = []
            for _ in range(num):
                class_counts = [class_dict[i]['count'] for i in range(K)]
                max_class_counts = np.where(np.array(class_counts) == max(class_counts))[0]
                c.append(np.random.choice(max_class_counts))
                class_dict[c[-1]]['count'] -= 1
            class_partitions['class'].append(c)
            class_partitions['prob'].append([class_dict[i]['prob'].pop() for i in c])

        # -------------------------- #
        # Create class index mapping #
        # -------------------------- #
        data_class_idx_train = {i: np.where(y_train == i)[0] for i in range(K)}
        data_class_idx_test = {i: np.where(y_test == i)[0] for i in range(K)}

        num_samples_train = {i: len(data_class_idx_train[i]) for i in range(K)}
        num_samples_test = {i: len(data_class_idx_test[i]) for i in range(K)}

        # --------- #
        # Shuffling #
        # --------- #
        for data_idx in data_class_idx_train.values():
            random.shuffle(data_idx)
        for data_idx in data_class_idx_test.values():
            random.shuffle(data_idx)

        # ------------------------------ #
        # Assigning samples to each user #
        # ------------------------------ #
        net_dataidx_map_train = {i: np.ndarray(0, dtype=np.int64) for i in range(n_parties)}
        net_dataidx_map_test = {i: np.ndarray(0, dtype=np.int64) for i in range(n_parties)}

        for usr_i in range(n_parties):
            for c, p in zip(class_partitions['class'][usr_i], class_partitions['prob'][usr_i]):
                end_idx_train = int(num_samples_train[c] * p)
                end_idx_test = int(num_samples_test[c] * p)

                net_dataidx_map_train[usr_i] = np.append(net_dataidx_map_train[usr_i],
                                                         data_class_idx_train[c][:end_idx_train])
                net_dataidx_map_test[usr_i] = np.append(net_dataidx_map_test[usr_i],
                                                        data_class_idx_test[c][:end_idx_test])
                data_class_idx_train[c] = data_class_idx_train[c][end_idx_train:]
                data_class_idx_test[c] = data_class_idx_test[c][end_idx_test:]

    elif partition == "noniid-labeldir":
        min_size = 0
        min_require_size = 10
        if dataset == 'cifar10' or dataset == 'fmnist':
            K = 10
        elif dataset == "cifar100":
            K = 100
        elif dataset in ('celeba', 'covtype', 'a9a', 'rcv1', 'SUSY'):
            K = 2
            # min_require_size = 100
        else:
            assert False
            print("Choose Dataset in readme.")

        N_train = y_train.shape[0]
        N_test = y_test.shape[0]
        net_dataidx_map_train = {}
        net_dataidx_map_test = {}

        while min_size < min_require_size:
            idx_batch_train = [[] for _ in range(n_parties)]
            idx_batch_test = [[] for _ in range(n_parties)]
            for k in range(K):
                train_idx_k = np.where(y_train == k)[0]
                test_idx_k = np.where(y_test == k)[0]

                np.random.shuffle(train_idx_k)
                np.random.shuffle(test_idx_k)

                proportions = np.random.dirichlet(np.repeat(beta, n_parties))
                proportions = np.array \
                    ([p * (len(idx_j) < N_train / n_parties) for p, idx_j in zip(proportions, idx_batch_train)])
                proportions = proportions / proportions.sum()
                proportions_train = (np.cumsum(proportions) * len(train_idx_k)).astype(int)[:-1]
                proportions_test = (np.cumsum(proportions) * len(test_idx_k)).astype(int)[:-1]
                idx_batch_train = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch_train, np.split(train_idx_k, proportions_train))]
                idx_batch_test = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch_test, np.split(test_idx_k, proportions_test))]

                min_size_train = min([len(idx_j) for idx_j in idx_batch_train])
                min_size_test = min([len(idx_j) for idx_j in idx_batch_test])
                min_size = min(min_size_train, min_size_test)

        for j in range(n_parties):
            np.random.shuffle(idx_batch_train[j])
            np.random.shuffle(idx_batch_test[j])
            net_dataidx_map_train[j] = idx_batch_train[j]
            net_dataidx_map_test[j] = idx_batch_test[j]

    elif partition == "noniid-labeldir100":
        seed = 12345
        alpha = 10
        n_fine_labels = 100
        n_coarse_labels = 20
        coarse_labels = \
            np.array([
                4, 1, 14, 8, 0, 6, 7, 7, 18, 3,
                3, 14, 9, 18, 7, 11, 3, 9, 7, 11,
                6, 11, 5, 10, 7, 6, 13, 15, 3, 15,
                0, 11, 1, 10, 12, 14, 16, 9, 11, 5,
                5, 19, 8, 8, 15, 13, 14, 17, 18, 10,
                16, 4, 17, 4, 2, 0, 17, 4, 18, 17,
                10, 3, 2, 12, 12, 16, 12, 1, 9, 19,
                2, 10, 0, 1, 16, 12, 9, 13, 15, 13,
                16, 19, 2, 4, 6, 19, 5, 5, 8, 19,
                18, 1, 2, 15, 6, 0, 17, 8, 14, 13
            ])

        rng_seed = (seed if (seed is not None and seed >= 0) else int(time.time()))
        rng = random.Random(rng_seed)
        np.random.seed(rng_seed)

        n_samples = y.shape[0]

        selected_indices = rng.sample(list(range(n_samples)), n_samples)

        n_samples_by_client = n_samples // n_parties

        indices_by_fine_labels = {k: list() for k in range(n_fine_labels)}
        indices_by_coarse_labels = {k: list() for k in range(n_coarse_labels)}

        for idx in selected_indices:
            fine_label = y[idx]
            coarse_label = coarse_labels[fine_label]

            indices_by_fine_labels[fine_label].append(idx)
            indices_by_coarse_labels[coarse_label].append(idx)

        available_coarse_labels = [ii for ii in range(n_coarse_labels)]

        fine_labels_by_coarse_labels = {k: list() for k in range(n_coarse_labels)}

        for fine_label, coarse_label in enumerate(coarse_labels):
            fine_labels_by_coarse_labels[coarse_label].append(fine_label)

        net_dataidx_map = [[] for i in range(n_parties)]

        for client_idx in range(n_parties):
            coarse_labels_weights = np.random.dirichlet(alpha=beta * np.ones(len(fine_labels_by_coarse_labels)))
            weights_by_coarse_labels = dict()

            for coarse_label, fine_labels in fine_labels_by_coarse_labels.items():
                weights_by_coarse_labels[coarse_label] = np.random.dirichlet(alpha=alpha * np.ones(len(fine_labels)))

            for ii in range(n_samples_by_client):
                coarse_label_idx = int(np.argmax(np.random.multinomial(1, coarse_labels_weights)))
                coarse_label = available_coarse_labels[coarse_label_idx]
                fine_label_idx = int(np.argmax(np.random.multinomial(1, weights_by_coarse_labels[coarse_label])))
                fine_label = fine_labels_by_coarse_labels[coarse_label][fine_label_idx]
                sample_idx = int(rng.choice(list(indices_by_fine_labels[fine_label])))

                net_dataidx_map[client_idx] = np.append(net_dataidx_map[client_idx], sample_idx)

                indices_by_fine_labels[fine_label].remove(sample_idx)
                indices_by_coarse_labels[coarse_label].remove(sample_idx)


                if len(indices_by_fine_labels[fine_label]) == 0:
                    fine_labels_by_coarse_labels[coarse_label].remove(fine_label)

                    weights_by_coarse_labels[coarse_label] = renormalize(weights_by_coarse_labels[coarse_label]
                                                                         ,fine_label_idx)

                    if len(indices_by_coarse_labels[coarse_label]) == 0:
                        fine_labels_by_coarse_labels.pop(coarse_label, None)
                        available_coarse_labels.remove(coarse_label)

                        coarse_labels_weights = renormalize(coarse_labels_weights, coarse_label_idx)

        random.shuffle(net_dataidx_map)
        net_dataidx_map_train = {i: np.ndarray(0, dtype=np.int64) for i in range(n_parties)}
        net_dataidx_map_test = {i: np.ndarray(0, dtype=np.int64) for i in range(n_parties)}
        for i, index in enumerate(net_dataidx_map):
            net_dataidx_map_train[i] = np.append(net_dataidx_map_train[i], index[index < 50_000]).astype(int)
            net_dataidx_map_test[i] = np.append(net_dataidx_map_test[i], index[index >= 50_000 ] -50000).astype(int)

    elif partition > "noniid-#label0" and partition <= "noniid-#label9":
        num = eval(partition[13:])
        if dataset in ('celeba', 'covtype', 'a9a', 'rcv1', 'SUSY'):
            num = 1
            K = 2
        elif dataset == 'cifar10':
            K = 10
        elif dataset == "cifar100":
            K = 100
        else:
            assert False
            print("Choose Dataset in readme.")

        if num == 10:
            net_dataidx_map_train ={i: np.ndarray(0, dtype=np.int64) for i in range(n_parties)}
            net_dataidx_map_train = {i: np.ndarray(0, dtype=np.int64) for i in range(n_parties)}
            for i in range(10):
                idx_k_train = np.where(y_train == i)[0]
                idx_k_test = np.where(y_test == i)[0]

                np.random.shuffle(idx_k_train)
                np.random.shuffle(idx_k_test)

                train_split = np.array_split(idx_k_train, n_parties)
                test_split = np.array_split(idx_k_test, n_parties)
                for j in range(n_parties):
                    net_dataidx_map_train[j] = np.append(net_dataidx_map_train[j], train_split[j])
                    net_dataidx_map_test[j] = np.append(net_dataidx_map_test[j], test_split[j])
        else:
            times = [0 for i in range(10)]
            contain = []
            for i in range(n_parties):
                current = [i % K]
                times[i % K] += 1
                j = 1
                while (j < num):
                    ind = random.randint(0, K - 1)
                    if (ind not in current):
                        j = j + 1
                        current.append(ind)
                        times[ind] += 1
                contain.append(current)
            net_dataidx_map_train = {i: np.ndarray(0, dtype=np.int64) for i in range(n_parties)}
            net_dataidx_map_test = {i: np.ndarray(0, dtype=np.int64) for i in range(n_parties)}

            for i in range(K):
                idx_k_train = np.where(y_train == i)[0]
                idx_k_test = np.where(y_test == i)[0]

                np.random.shuffle(idx_k_train)
                np.random.shuffle(idx_k_test)

                train_split = np.array_split(idx_k_train, times[i])
                test_split = np.array_split(idx_k_test, times[i])

                ids = 0
                for j in range(n_parties):
                    if i in contain[j]:
                        net_dataidx_map_train[j] = np.append(net_dataidx_map_train[j], train_split[ids])
                        net_dataidx_map_test[j] = np.append(net_dataidx_map_test[j], test_split[ids])
                        ids += 1

    traindata_cls_counts = record_net_data_stats(y_train, net_dataidx_map_train, logdir)
    testdata_cls_counts = record_net_data_stats(y_test, net_dataidx_map_test, logdir)

    return (data_train, data_test, lab2cname, classnames, net_dataidx_map_train, net_dataidx_map_test, traindata_cls_counts,
            testdata_cls_counts,y_train)


def partition_data_LT(
    dataset, datadir, partition, n_parties, imb_factor, imb_type, beta, logdir=None,
    head_client_ratio=0.9, tail_client_ratio=0.1,
    head_class_ratio=0.8, tail_class_ratio=0.2,
    *,
    specialization_lambda,
    intra_group_alpha,
    head_leakage_scale,
    controlled_tail_min_purity=0.8,
    split_seed=1,
):
    loaders = _data_loaders()
    if dataset == 'cifar10_LT':
        X_train, y_train, X_test, y_test, data_train, data_test, lab2cname, classnames = loaders["cifar10_LT"](datadir, imb_factor, imb_type)
        y = np.concatenate([y_train, y_test], axis=0)
    elif dataset == 'cifar100_LT':
        X_train, y_train, X_test, y_test, data_train, data_test, lab2cname, classnames = loaders["cifar100_LT"](datadir, imb_factor, imb_type)
        y = np.concatenate([y_train, y_test], axis=0)
    elif dataset == 'fmnist_LT':
        X_train, y_train, X_test, y_test, data_train, data_test, lab2cname, classnames = loaders["fmnist_LT"](datadir, imb_factor, imb_type)
        y = np.concatenate([y_train, y_test], axis=0)


    n_train = y_train.shape[0]
    n_test = y_test.shape[0]

    if partition == "homo":
        idxs_train = np.random.permutation(n_train)
        idxs_test = np.random.permutation(n_test)

        batch_idxs_train = np.array_split(idxs_train, n_parties)
        batch_idxs_test = np.array_split(idxs_test, n_parties)

        net_dataidx_map_train = {i: batch_idxs_train[i] for i in range(n_parties)}
        net_dataidx_map_test = {i: batch_idxs_test[i] for i in range(n_parties)}

    elif partition == "iid-label100":
        seed = 12345
        n_fine_labels = 100
        n_coarse_labels = 20
        coarse_labels = \
            np.array([
                4, 1, 14, 8, 0, 6, 7, 7, 18, 3,
                3, 14, 9, 18, 7, 11, 3, 9, 7, 11,
                6, 11, 5, 10, 7, 6, 13, 15, 3, 15,
                0, 11, 1, 10, 12, 14, 16, 9, 11, 5,
                5, 19, 8, 8, 15, 13, 14, 17, 18, 10,
                16, 4, 17, 4, 2, 0, 17, 4, 18, 17,
                10, 3, 2, 12, 12, 16, 12, 1, 9, 19,
                2, 10, 0, 1, 16, 12, 9, 13, 15, 13,
                16, 19, 2, 4, 6, 19, 5, 5, 8, 19,
                18, 1, 2, 15, 6, 0, 17, 8, 14, 13
            ])
        rng_seed = (seed if (seed is not None and seed >= 0) else int(time.time()))
        rng = random.Random(rng_seed)
        np.random.seed(rng_seed)

        n_samples_train = y_train.shape[0]
        n_samples_test = y_test.shape[0]

        selected_indices_train = rng.sample(list(range(n_samples_train)), n_samples_train)
        selected_indices_test = rng.sample(list(range(n_samples_test)), n_samples_test)

        n_samples_by_client_train = int((n_samples_train / n_parties) // 5)
        n_samples_by_client_test = int((n_samples_test / n_parties) // 5)

        indices_by_fine_labels_train = {k: list() for k in range(n_fine_labels)}
        indices_by_coarse_labels_train = {k: list() for k in range(n_coarse_labels)}

        indices_by_fine_labels_test = {k: list() for k in range(n_fine_labels)}
        indices_by_coarse_labels_test = {k: list() for k in range(n_coarse_labels)}

        for idx in selected_indices_train:
            fine_label = y_train[idx]
            coarse_label = coarse_labels[fine_label]

            indices_by_fine_labels_train[fine_label].append(idx)
            indices_by_coarse_labels_train[coarse_label].append(idx)

        for idx in selected_indices_test:
            fine_label = y_test[idx]
            coarse_label = coarse_labels[fine_label]

            indices_by_fine_labels_test[fine_label].append(idx)
            indices_by_coarse_labels_test[coarse_label].append(idx)

        fine_labels_by_coarse_labels = {k: list() for k in range(n_coarse_labels)}

        for fine_label, coarse_label in enumerate(coarse_labels):
            fine_labels_by_coarse_labels[coarse_label].append(fine_label)

        net_dataidx_map_train = {i: np.ndarray(0, dtype=np.int64) for i in range(n_parties)}
        net_dataidx_map_test = {i: np.ndarray(0, dtype=np.int64) for i in range(n_parties)}

        for client_idx in range(n_parties):
            coarse_idx = client_idx // 5
            fine_idx = fine_labels_by_coarse_labels[coarse_idx]
            for k in range(5):
                fine_label = fine_idx[k]
                sample_idx = rng.sample(list(indices_by_fine_labels_train[fine_label]), n_samples_by_client_train)
                net_dataidx_map_train[client_idx] = np.append(net_dataidx_map_train[client_idx], sample_idx)
                for idx in sample_idx:
                    indices_by_fine_labels_train[fine_label].remove(idx)

        for client_idx in range(n_parties):
            coarse_idx = client_idx // 5
            fine_idx = fine_labels_by_coarse_labels[coarse_idx]
            for k in range(5):
                fine_label = fine_idx[k]
                sample_idx = rng.sample(list(indices_by_fine_labels_test[fine_label]), n_samples_by_client_test)
                net_dataidx_map_test[client_idx] = np.append(net_dataidx_map_test[client_idx], sample_idx)
                for idx in sample_idx:
                    indices_by_fine_labels_test[fine_label].remove(idx)

    elif partition == "noniid-labeluni":
        if dataset == "cifar10" or dataset == "cifar10_LT":
            num = 2
        elif dataset == "cifar100" or dataset == "cifar100_LT":
            num = 10
        if dataset in ('celeba', 'covtype', 'a9a', 'rcv1', 'SUSY'):
            num = 1
            K = 2
        elif dataset == 'cifar100' or dataset == "LT_cifar100":
            K = 100
        elif dataset == 'cifar10' or dataset == "cifar10_LT":
            K = 10
        else:
            assert False
            print("Choose Dataset in readme.")

        # -------------------------------------------#
        # Divide classes + num samples for each user #
        # -------------------------------------------#
        assert (num * n_parties) % K == 0, "equal classes appearance is needed"
        count_per_class = (num * n_parties) // K
        class_dict = {}
        for i in range(K):
            # sampling alpha_i_c
            probs = np.random.uniform(0.4, 0.6, size=count_per_class)
            # normalizing
            probs_norm = (probs / probs.sum()).tolist()
            class_dict[i] = {'count': count_per_class, 'prob': probs_norm}

        # -------------------------------------#
        # Assign each client with data indexes #
        # -------------------------------------#
        class_partitions = defaultdict(list)
        for i in range(n_parties):
            c = []
            for _ in range(num):
                class_counts = [class_dict[i]['count'] for i in range(K)]
                max_class_counts = np.where(np.array(class_counts) == max(class_counts))[0]
                c.append(np.random.choice(max_class_counts))
                class_dict[c[-1]]['count'] -= 1
            class_partitions['class'].append(c)
            class_partitions['prob'].append([class_dict[i]['prob'].pop() for i in c])

        # -------------------------- #
        # Create class index mapping #
        # -------------------------- #
        data_class_idx_train = {i: np.where(y_train == i)[0] for i in range(K)}
        data_class_idx_test = {i: np.where(y_test == i)[0] for i in range(K)}

        num_samples_train = {i: len(data_class_idx_train[i]) for i in range(K)}
        num_samples_test = {i: len(data_class_idx_test[i]) for i in range(K)}

        # --------- #
        # Shuffling #
        # --------- #
        for data_idx in data_class_idx_train.values():
            random.shuffle(data_idx)
        for data_idx in data_class_idx_test.values():
            random.shuffle(data_idx)

        # ------------------------------ #
        # Assigning samples to each user #
        # ------------------------------ #
        net_dataidx_map_train = {i: np.ndarray(0, dtype=np.int64) for i in range(n_parties)}
        net_dataidx_map_test = {i: np.ndarray(0, dtype=np.int64) for i in range(n_parties)}

        for usr_i in range(n_parties):
            for c, p in zip(class_partitions['class'][usr_i], class_partitions['prob'][usr_i]):
                end_idx_train = int(num_samples_train[c] * p)
                end_idx_test = int(num_samples_test[c] * p)

                net_dataidx_map_train[usr_i] = np.append(net_dataidx_map_train[usr_i],
                                                         data_class_idx_train[c][:end_idx_train])
                net_dataidx_map_test[usr_i] = np.append(net_dataidx_map_test[usr_i],
                                                        data_class_idx_test[c][:end_idx_test])
                data_class_idx_train[c] = data_class_idx_train[c][end_idx_train:]
                data_class_idx_test[c] = data_class_idx_test[c][end_idx_test:]

    elif partition == "noniid-labeldir-fine":
        num_classes = _get_num_classes(dataset, y_train, y_test)
        net_dataidx_map_train, net_dataidx_map_test = partition_fine_class_dirichlet(
            y_train,
            y_test,
            n_parties,
            num_classes,
            beta,
            split_seed,
        )

    elif partition == "noniid-labeldir":
        if dataset == 'cifar100_LT':
            seed = 12345
            alpha = 10
            n_fine_labels = 100
            n_coarse_labels = 20
            coarse_labels = \
                np.array([
                    4, 1, 14, 8, 0, 6, 7, 7, 18, 3,
                    3, 14, 9, 18, 7, 11, 3, 9, 7, 11,
                    6, 11, 5, 10, 7, 6, 13, 15, 3, 15,
                    0, 11, 1, 10, 12, 14, 16, 9, 11, 5,
                    5, 19, 8, 8, 15, 13, 14, 17, 18, 10,
                    16, 4, 17, 4, 2, 0, 17, 4, 18, 17,
                    10, 3, 2, 12, 12, 16, 12, 1, 9, 19,
                    2, 10, 0, 1, 16, 12, 9, 13, 15, 13,
                    16, 19, 2, 4, 6, 19, 5, 5, 8, 19,
                    18, 1, 2, 15, 6, 0, 17, 8, 14, 13
                ])

            rng_seed = (seed if (seed is not None and seed >= 0) else int(time.time()))
            rng = random.Random(rng_seed)
            np.random.seed(rng_seed)

            n_samples = y.shape[0]

            selected_indices = rng.sample(list(range(n_samples)), n_samples)

            n_samples_by_client = n_samples // n_parties

            indices_by_fine_labels = {k: list() for k in range(n_fine_labels)}
            indices_by_coarse_labels = {k: list() for k in range(n_coarse_labels)}

            for idx in selected_indices:
                fine_label = y[idx]
                coarse_label = coarse_labels[fine_label]

                indices_by_fine_labels[fine_label].append(idx)
                indices_by_coarse_labels[coarse_label].append(idx)

            available_coarse_labels = [ii for ii in range(n_coarse_labels)]

            fine_labels_by_coarse_labels = {k: list() for k in range(n_coarse_labels)}

            for fine_label, coarse_label in enumerate(coarse_labels):
                fine_labels_by_coarse_labels[coarse_label].append(fine_label)

            net_dataidx_map = [[] for i in range(n_parties)]

            for client_idx in range(n_parties):
                coarse_labels_weights = np.random.dirichlet(alpha=beta * np.ones(len(fine_labels_by_coarse_labels)))
                weights_by_coarse_labels = dict()

                for coarse_label, fine_labels in fine_labels_by_coarse_labels.items():
                    weights_by_coarse_labels[coarse_label] = np.random.dirichlet(
                        alpha=alpha * np.ones(len(fine_labels)))

                for ii in range(n_samples_by_client):
                    coarse_label_idx = int(np.argmax(np.random.multinomial(1, coarse_labels_weights)))
                    coarse_label = available_coarse_labels[coarse_label_idx]
                    fine_label_idx = int(np.argmax(np.random.multinomial(1, weights_by_coarse_labels[coarse_label])))
                    fine_label = fine_labels_by_coarse_labels[coarse_label][fine_label_idx]
                    sample_idx = int(rng.choice(list(indices_by_fine_labels[fine_label])))

                    net_dataidx_map[client_idx] = np.append(net_dataidx_map[client_idx], sample_idx)

                    indices_by_fine_labels[fine_label].remove(sample_idx)
                    indices_by_coarse_labels[coarse_label].remove(sample_idx)

                    if len(indices_by_fine_labels[fine_label]) == 0:
                        fine_labels_by_coarse_labels[coarse_label].remove(fine_label)

                        weights_by_coarse_labels[coarse_label] = renormalize(weights_by_coarse_labels[coarse_label]
                                                                             , fine_label_idx)

                        if len(indices_by_coarse_labels[coarse_label]) == 0:
                            fine_labels_by_coarse_labels.pop(coarse_label, None)
                            available_coarse_labels.remove(coarse_label)

                            coarse_labels_weights = renormalize(coarse_labels_weights, coarse_label_idx)

            random.shuffle(net_dataidx_map)
            net_dataidx_map_train = {i: np.ndarray(0, dtype=np.int64) for i in range(n_parties)}
            net_dataidx_map_test = {i: np.ndarray(0, dtype=np.int64) for i in range(n_parties)}

            train_size = len(y_train)
            for i, index in enumerate(net_dataidx_map):
                net_dataidx_map_train[i] = np.append(net_dataidx_map_train[i], index[index < train_size]).astype(int)
                net_dataidx_map_test[i] = np.append(net_dataidx_map_test[i],
                                                    index[index >= train_size] - train_size).astype(int)
        else:
            min_size = 0
            min_require_size = 10
            if dataset == 'fmnist_LT' or dataset == 'cifar10_LT':
                K = 10
            elif dataset == 'cifar100_LT':
                K = 100
            elif dataset in ('celeba', 'covtype', 'a9a', 'rcv1', 'SUSY'):
                K = 2
            else:
                assert False

            N_train = y_train.shape[0]
            N_test = y_test.shape[0]
            net_dataidx_map_train = {}
            net_dataidx_map_test = {}

            while min_size < min_require_size:
                idx_batch_train = [[] for _ in range(n_parties)]
                idx_batch_test = [[] for _ in range(n_parties)]
                for k in range(K):
                    train_idx_k = np.where(y_train == k)[0]
                    test_idx_k = np.where(y_test == k)[0]

                    np.random.shuffle(train_idx_k)
                    np.random.shuffle(test_idx_k)

                    proportions = np.random.dirichlet(np.repeat(beta, n_parties))
                    proportions = np.array \
                        ([p * (len(idx_j) < N_train / n_parties) for p, idx_j in zip(proportions, idx_batch_train)])
                    proportions = proportions / proportions.sum()
                    proportions_train = (np.cumsum(proportions) * len(train_idx_k)).astype(int)[:-1]
                    proportions_test = (np.cumsum(proportions) * len(test_idx_k)).astype(int)[:-1]
                    idx_batch_train = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch_train, np.split(train_idx_k, proportions_train))]
                    idx_batch_test = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch_test, np.split(test_idx_k, proportions_test))]

                    min_size_train = min([len(idx_j) for idx_j in idx_batch_train])
                    min_size_test = min([len(idx_j) for idx_j in idx_batch_test])
                    min_size = min(min_size_train, min_size_test)

            for j in range(n_parties):
                np.random.shuffle(idx_batch_train[j])
                np.random.shuffle(idx_batch_test[j])
                net_dataidx_map_train[j] = idx_batch_train[j]
                net_dataidx_map_test[j] = idx_batch_test[j]

    # ----------------------------------------------------------
    # 客户端长尾划分（超简版）—— 触发方式：partition == "longtail-client"
    # ----------------------------------------------------------
    elif partition == "client-longtail":
        num_classes = _get_num_classes(dataset, y_train, y_test)
        train_rng = np.random.RandomState(int(split_seed))
        test_rng = np.random.RandomState(int(split_seed) + 1)
        net_dataidx_map_train = partition_client_longtail(
            y_train,
            n_parties,
            num_classes,
            head_client_ratio=head_client_ratio,
            tail_client_ratio=tail_client_ratio,
            head_class_ratio=head_class_ratio,
            tail_class_ratio=tail_class_ratio,
            specialization_lambda=specialization_lambda,
            intra_group_alpha=intra_group_alpha,
            head_leakage_scale=head_leakage_scale,
            rng=train_rng,
        )
        net_dataidx_map_test = partition_client_longtail(
            y_test,
            n_parties,
            num_classes,
            head_client_ratio=head_client_ratio,
            tail_client_ratio=tail_client_ratio,
            head_class_ratio=head_class_ratio,
            tail_class_ratio=tail_class_ratio,
            specialization_lambda=specialization_lambda,
            intra_group_alpha=intra_group_alpha,
            head_leakage_scale=head_leakage_scale,
            rng=test_rng,
        )

    elif partition == "client-longtail-controlled":
        num_classes = _get_num_classes(dataset, y_train, y_test)
        train_tail_class_ids = _bottom_classes_from_counts(
            y_train, num_classes, tail_class_ratio
        )
        train_rng = np.random.RandomState(int(split_seed))
        test_rng = np.random.RandomState(int(split_seed) + 1)
        net_dataidx_map_train = partition_client_longtail_controlled(
            y_train,
            n_parties,
            num_classes,
            head_client_ratio=head_client_ratio,
            tail_client_ratio=tail_client_ratio,
            tail_class_ratio=tail_class_ratio,
            intra_group_alpha=intra_group_alpha,
            tail_client_min_purity=controlled_tail_min_purity,
            tail_class_ids=train_tail_class_ids,
            rng=train_rng,
        )
        # The test set is balanced, so reuse the train-derived tail ids. The
        # purity ratio is applied to its own sample count; the train budget 38
        # is never hard-coded into test allocation.
        net_dataidx_map_test = partition_client_longtail_controlled(
            y_test,
            n_parties,
            num_classes,
            head_client_ratio=head_client_ratio,
            tail_client_ratio=tail_client_ratio,
            tail_class_ratio=tail_class_ratio,
            intra_group_alpha=intra_group_alpha,
            tail_client_min_purity=controlled_tail_min_purity,
            tail_class_ids=train_tail_class_ids,
            rng=test_rng,
        )

    elif partition == "longtail-client":
        net_dataidx_map_train, net_dataidx_map_test = partition_longtail_client(
            y_train, y_test, n_parties, seed=2025)

    elif partition == "noniid-labeldir100":
        seed = 12345
        alpha = 10
        n_fine_labels = 100
        n_coarse_labels = 20
        coarse_labels = \
            np.array([
                4, 1, 14, 8, 0, 6, 7, 7, 18, 3,
                3, 14, 9, 18, 7, 11, 3, 9, 7, 11,
                6, 11, 5, 10, 7, 6, 13, 15, 3, 15,
                0, 11, 1, 10, 12, 14, 16, 9, 11, 5,
                5, 19, 8, 8, 15, 13, 14, 17, 18, 10,
                16, 4, 17, 4, 2, 0, 17, 4, 18, 17,
                10, 3, 2, 12, 12, 16, 12, 1, 9, 19,
                2, 10, 0, 1, 16, 12, 9, 13, 15, 13,
                16, 19, 2, 4, 6, 19, 5, 5, 8, 19,
                18, 1, 2, 15, 6, 0, 17, 8, 14, 13
            ])

        rng_seed = (seed if (seed is not None and seed >= 0) else int(time.time()))
        rng = random.Random(rng_seed)
        np.random.seed(rng_seed)
        rng, used_seed = set_random_seed(1)

        n_samples = y.shape[0]

        selected_indices = rng.sample(list(range(n_samples)), n_samples)

        n_samples_by_client = n_samples // n_parties

        indices_by_fine_labels = {k: list() for k in range(n_fine_labels)}
        indices_by_coarse_labels = {k: list() for k in range(n_coarse_labels)}

        for idx in selected_indices:
            fine_label = y[idx]
            coarse_label = coarse_labels[fine_label]

            indices_by_fine_labels[fine_label].append(idx)
            indices_by_coarse_labels[coarse_label].append(idx)

        available_coarse_labels = [ii for ii in range(n_coarse_labels)]

        fine_labels_by_coarse_labels = {k: list() for k in range(n_coarse_labels)}

        for fine_label, coarse_label in enumerate(coarse_labels):
            fine_labels_by_coarse_labels[coarse_label].append(fine_label)

        net_dataidx_map = [[] for i in range(n_parties)]

        for client_idx in range(n_parties):
            coarse_labels_weights = np.random.dirichlet(alpha=beta * np.ones(len(fine_labels_by_coarse_labels)))
            weights_by_coarse_labels = dict()

            for coarse_label, fine_labels in fine_labels_by_coarse_labels.items():
                weights_by_coarse_labels[coarse_label] = np.random.dirichlet(alpha=alpha * np.ones(len(fine_labels)))

            for ii in range(n_samples_by_client):
                coarse_label_idx = int(np.argmax(np.random.multinomial(1, coarse_labels_weights)))
                coarse_label = available_coarse_labels[coarse_label_idx]
                fine_label_idx = int(np.argmax(np.random.multinomial(1, weights_by_coarse_labels[coarse_label])))
                fine_label = fine_labels_by_coarse_labels[coarse_label][fine_label_idx]
                sample_idx = int(rng.choice(list(indices_by_fine_labels[fine_label])))

                net_dataidx_map[client_idx] = np.append(net_dataidx_map[client_idx], sample_idx)

                indices_by_fine_labels[fine_label].remove(sample_idx)
                indices_by_coarse_labels[coarse_label].remove(sample_idx)


                if len(indices_by_fine_labels[fine_label]) == 0:
                    fine_labels_by_coarse_labels[coarse_label].remove(fine_label)

                    weights_by_coarse_labels[coarse_label] = renormalize(weights_by_coarse_labels[coarse_label]
                                                                         ,fine_label_idx)

                    if len(indices_by_coarse_labels[coarse_label]) == 0:
                        fine_labels_by_coarse_labels.pop(coarse_label, None)
                        available_coarse_labels.remove(coarse_label)

                        coarse_labels_weights = renormalize(coarse_labels_weights, coarse_label_idx)

        random.shuffle(net_dataidx_map)
        net_dataidx_map_train = {i: np.ndarray(0, dtype=np.int64) for i in range(n_parties)}
        net_dataidx_map_test = {i: np.ndarray(0, dtype=np.int64) for i in range(n_parties)}
        for i, index in enumerate(net_dataidx_map):
            net_dataidx_map_train[i] = np.append(net_dataidx_map_train[i], index[index < 50_000]).astype(int)
            net_dataidx_map_test[i] = np.append(net_dataidx_map_test[i], index[index >= 50_000 ] -50000).astype(int)

    elif partition == "noniid-labeldir100_LT":
        seed = 12345
        alpha = 10
        n_fine_labels = 100
        n_coarse_labels = 20
        coarse_labels = \
            np.array([
                4, 1, 14, 8, 0, 6, 7, 7, 18, 3,
                3, 14, 9, 18, 7, 11, 3, 9, 7, 11,
                6, 11, 5, 10, 7, 6, 13, 15, 3, 15,
                0, 11, 1, 10, 12, 14, 16, 9, 11, 5,
                5, 19, 8, 8, 15, 13, 14, 17, 18, 10,
                16, 4, 17, 4, 2, 0, 17, 4, 18, 17,
                10, 3, 2, 12, 12, 16, 12, 1, 9, 19,
                2, 10, 0, 1, 16, 12, 9, 13, 15, 13,
                16, 19, 2, 4, 6, 19, 5, 5, 8, 19,
                18, 1, 2, 15, 6, 0, 17, 8, 14, 13
            ])

        rng_seed = (seed if (seed is not None and seed >= 0) else int(time.time()))
        rng = random.Random(rng_seed)
        np.random.seed(rng_seed)

        n_samples = y.shape[0]

        selected_indices = rng.sample(list(range(n_samples)), n_samples)

        n_samples_by_client = n_samples // n_parties

        indices_by_fine_labels = {k: list() for k in range(n_fine_labels)}
        indices_by_coarse_labels = {k: list() for k in range(n_coarse_labels)}

        for idx in selected_indices:
            fine_label = y[idx]
            coarse_label = coarse_labels[fine_label]

            indices_by_fine_labels[fine_label].append(idx)
            indices_by_coarse_labels[coarse_label].append(idx)

        available_coarse_labels = [ii for ii in range(n_coarse_labels)]

        fine_labels_by_coarse_labels = {k: list() for k in range(n_coarse_labels)}

        for fine_label, coarse_label in enumerate(coarse_labels):
            fine_labels_by_coarse_labels[coarse_label].append(fine_label)

        net_dataidx_map = [[] for i in range(n_parties)]

        for client_idx in range(n_parties):
            coarse_labels_weights = np.random.dirichlet(alpha=beta * np.ones(len(fine_labels_by_coarse_labels)))
            weights_by_coarse_labels = dict()

            for coarse_label, fine_labels in fine_labels_by_coarse_labels.items():
                weights_by_coarse_labels[coarse_label] = np.random.dirichlet(alpha=alpha * np.ones(len(fine_labels)))

            for ii in range(n_samples_by_client):
                coarse_label_idx = int(np.argmax(np.random.multinomial(1, coarse_labels_weights)))
                coarse_label = available_coarse_labels[coarse_label_idx]
                fine_label_idx = int(np.argmax(np.random.multinomial(1, weights_by_coarse_labels[coarse_label])))
                fine_label = fine_labels_by_coarse_labels[coarse_label][fine_label_idx]
                sample_idx = int(rng.choice(list(indices_by_fine_labels[fine_label])))

                net_dataidx_map[client_idx] = np.append(net_dataidx_map[client_idx], sample_idx)

                indices_by_fine_labels[fine_label].remove(sample_idx)
                indices_by_coarse_labels[coarse_label].remove(sample_idx)


                if len(indices_by_fine_labels[fine_label]) == 0:
                    fine_labels_by_coarse_labels[coarse_label].remove(fine_label)

                    weights_by_coarse_labels[coarse_label] = renormalize(weights_by_coarse_labels[coarse_label]
                                                                         ,fine_label_idx)

                    if len(indices_by_coarse_labels[coarse_label]) == 0:
                        fine_labels_by_coarse_labels.pop(coarse_label, None)
                        available_coarse_labels.remove(coarse_label)

                        coarse_labels_weights = renormalize(coarse_labels_weights, coarse_label_idx)

        random.shuffle(net_dataidx_map)
        net_dataidx_map_train = {i: np.ndarray(0, dtype=np.int64) for i in range(n_parties)}
        net_dataidx_map_test = {i: np.ndarray(0, dtype=np.int64) for i in range(n_parties)}

        train_size = len(y_train)
        for i, index in enumerate(net_dataidx_map):
            net_dataidx_map_train[i] = np.append(net_dataidx_map_train[i], index[index < train_size]).astype(int)
            net_dataidx_map_test[i] = np.append(net_dataidx_map_test[i],
                                                index[index >= train_size] - train_size).astype(int)

    elif partition > "noniid-#label0" and partition <= "noniid-#label9":
        num = eval(partition[13:])
        if dataset in ('celeba', 'covtype', 'a9a', 'rcv1', 'SUSY'):
            num = 1
            K = 2
        elif dataset == 'cifar10':
            K = 10
        elif dataset == "cifar100":
            K = 100
        else:
            assert False
            print("Choose Dataset in readme.")

        if num == 10:
            net_dataidx_map_train ={i: np.ndarray(0, dtype=np.int64) for i in range(n_parties)}
            net_dataidx_map_train = {i: np.ndarray(0, dtype=np.int64) for i in range(n_parties)}
            for i in range(10):
                idx_k_train = np.where(y_train == i)[0]
                idx_k_test = np.where(y_test == i)[0]

                np.random.shuffle(idx_k_train)
                np.random.shuffle(idx_k_test)

                train_split = np.array_split(idx_k_train, n_parties)
                test_split = np.array_split(idx_k_test, n_parties)
                for j in range(n_parties):
                    net_dataidx_map_train[j] = np.append(net_dataidx_map_train[j], train_split[j])
                    net_dataidx_map_test[j] = np.append(net_dataidx_map_test[j], test_split[j])
        else:
            times = [0 for i in range(10)]
            contain = []
            for i in range(n_parties):
                current = [i % K]
                times[i % K] += 1
                j = 1
                while (j < num):
                    ind = random.randint(0, K - 1)
                    if (ind not in current):
                        j = j + 1
                        current.append(ind)
                        times[ind] += 1
                contain.append(current)
            net_dataidx_map_train = {i: np.ndarray(0, dtype=np.int64) for i in range(n_parties)}
            net_dataidx_map_test = {i: np.ndarray(0, dtype=np.int64) for i in range(n_parties)}

            for i in range(K):
                idx_k_train = np.where(y_train == i)[0]
                idx_k_test = np.where(y_test == i)[0]

                np.random.shuffle(idx_k_train)
                np.random.shuffle(idx_k_test)

                train_split = np.array_split(idx_k_train, times[i])
                test_split = np.array_split(idx_k_test, times[i])

                ids = 0
                for j in range(n_parties):
                    if i in contain[j]:
                        net_dataidx_map_train[j] = np.append(net_dataidx_map_train[j], train_split[ids])
                        net_dataidx_map_test[j] = np.append(net_dataidx_map_test[j], test_split[ids])
                        ids += 1


    traindata_cls_counts = record_net_data_stats(y_train, net_dataidx_map_train, logdir)
    testdata_cls_counts = record_net_data_stats(y_test, net_dataidx_map_test, logdir)

    return (data_train, data_test, lab2cname, classnames, net_dataidx_map_train, net_dataidx_map_test, traindata_cls_counts,
            testdata_cls_counts, y_train)
