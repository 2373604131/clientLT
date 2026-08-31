import numpy as np

from utils.datasplit import partition_fixed_marginal_dirichlet


def _counts(labels, net_map, num_clients, num_classes):
    matrix = np.zeros((num_clients, num_classes), dtype=np.int64)
    for client_id in range(num_clients):
        matrix[client_id] = np.bincount(
            labels[np.asarray(net_map[client_id], dtype=np.int64)],
            minlength=num_classes,
        )
    return matrix


def test_fixed_marginal_dirichlet_preserves_both_margins_and_is_deterministic():
    labels = np.repeat(np.arange(6), [31, 27, 23, 19, 13, 7])
    capacities = np.asarray([35, 29, 24, 18, 14], dtype=np.int64)

    first = partition_fixed_marginal_dirichlet(
        labels,
        capacities,
        num_classes=6,
        alpha=0.5,
        rng=np.random.RandomState(42),
    )
    second = partition_fixed_marginal_dirichlet(
        labels,
        capacities,
        num_classes=6,
        alpha=0.5,
        rng=np.random.RandomState(42),
    )

    first_counts = _counts(labels, first, len(capacities), 6)
    second_counts = _counts(labels, second, len(capacities), 6)
    assert np.array_equal(first_counts, second_counts)
    assert np.array_equal(first_counts.sum(axis=1), capacities)
    assert np.array_equal(
        first_counts.sum(axis=0), np.bincount(labels, minlength=6)
    )
    merged = np.concatenate([first[client_id] for client_id in range(len(capacities))])
    assert np.array_equal(np.sort(merged), np.arange(len(labels)))


def test_fixed_marginal_dirichlet_rejects_inconsistent_capacity():
    labels = np.repeat(np.arange(2), 5)
    try:
        partition_fixed_marginal_dirichlet(
            labels,
            [3, 3],
            num_classes=2,
            alpha=0.5,
            rng=np.random.RandomState(1),
        )
    except ValueError as error:
        assert "must sum" in str(error)
    else:
        raise AssertionError("inconsistent capacities were accepted")
