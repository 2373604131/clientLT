import inspect
import random
import sys
import types
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

stub_dataloader = types.ModuleType("utils.dataloader")
for name in (
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
):
    setattr(stub_dataloader, name, lambda *args, **kwargs: None)
sys.modules.setdefault("utils.dataloader", stub_dataloader)

stub_dataset = types.ModuleType("utils.dataset")
stub_dataset.mkdirs = lambda *args, **kwargs: None
sys.modules.setdefault("utils.dataset", stub_dataset)

from utils.datasplit import partition_fine_class_dirichlet


def _maps_equal(left, right):
    return all(np.array_equal(left[k], right[k]) for k in left)


def _assert_complete(labels, partition_map):
    merged = np.concatenate([np.asarray(partition_map[i], dtype=np.int64) for i in range(len(partition_map))])
    assert len(merged) == len(labels)
    assert len(np.unique(merged)) == len(labels)
    assert np.array_equal(np.sort(merged), np.arange(len(labels)))


def _class_counts(labels, partition_map, num_classes):
    merged = np.concatenate([np.asarray(partition_map[i], dtype=np.int64) for i in range(len(partition_map))])
    return np.bincount(np.asarray(labels)[merged], minlength=num_classes)[:num_classes]


def _np_state_equal(left, right):
    return (
        left[0] == right[0]
        and np.array_equal(left[1], right[1])
        and left[2:] == right[2:]
    )


def test_fine_dirichlet_reproducible_seed_and_different_seed():
    y_train = np.repeat(np.arange(100), 6)
    y_test = np.repeat(np.arange(100), 2)

    train_a, test_a = partition_fine_class_dirichlet(y_train, y_test, 5, 100, 0.8, 7)
    train_b, test_b = partition_fine_class_dirichlet(y_train, y_test, 5, 100, 0.8, 7)
    train_c, test_c = partition_fine_class_dirichlet(y_train, y_test, 5, 100, 0.8, 8)

    assert _maps_equal(train_a, train_b)
    assert _maps_equal(test_a, test_b)
    assert any(
        not np.array_equal(train_a[i], train_c[i]) or not np.array_equal(test_a[i], test_c[i])
        for i in range(5)
    )


def test_fine_dirichlet_coverage_class_counts_and_independent_index_spaces():
    y_train = np.repeat(np.arange(100), 6)
    y_test = np.repeat(np.arange(100), 3)

    train_map, test_map = partition_fine_class_dirichlet(y_train, y_test, 5, 100, 0.8, 11)

    _assert_complete(y_train, train_map)
    _assert_complete(y_test, test_map)
    assert np.array_equal(_class_counts(y_train, train_map, 100), np.bincount(y_train, minlength=100))
    assert np.array_equal(_class_counts(y_test, test_map, 100), np.bincount(y_test, minlength=100))
    assert max(np.max(v) for v in train_map.values() if len(v)) < len(y_train)
    assert max(np.max(v) for v in test_map.values() if len(v)) < len(y_test)


def test_fine_dirichlet_uses_fine_labels_without_coarse_mapping():
    source = inspect.getsource(partition_fine_class_dirichlet)
    assert "coarse_labels" not in source

    y_train = np.arange(100)
    y_test = np.arange(100)
    train_map, test_map = partition_fine_class_dirichlet(
        y_train,
        y_test,
        2,
        100,
        100.0,
        3,
        min_client_train_samples=1,
    )

    _assert_complete(y_train, train_map)
    _assert_complete(y_test, test_map)


def test_fine_dirichlet_invalid_beta_and_retry_failure():
    y_train = np.arange(8) % 2
    y_test = np.arange(4) % 2

    with pytest.raises(ValueError):
        partition_fine_class_dirichlet(y_train, y_test, 2, 2, 0, 1)
    with pytest.raises(ValueError):
        partition_fine_class_dirichlet(y_train, y_test, 2, 2, -0.5, 1)
    with pytest.raises(ValueError):
        partition_fine_class_dirichlet(y_train, y_test, 1, 2, 0.5, 1)
    with pytest.raises(ValueError):
        partition_fine_class_dirichlet(y_train, y_test, 2, 2, 0.5, 1, max_retries=0)

    with pytest.raises(RuntimeError, match="beta=.*n_parties=.*min_client_train_samples=.*max_retries"):
        partition_fine_class_dirichlet(
            y_train,
            y_test,
            4,
            2,
            0.5,
            1,
            min_client_train_samples=3,
            max_retries=3,
        )


def test_fine_dirichlet_does_not_pollute_global_rng_state():
    y_train = np.repeat(np.arange(100), 5)
    y_test = np.repeat(np.arange(100), 2)

    np.random.seed(123)
    random.seed(456)
    np_before = np.random.get_state()
    py_before = random.getstate()

    partition_fine_class_dirichlet(y_train, y_test, 5, 100, 1.0, 9)

    assert _np_state_equal(np_before, np.random.get_state())
    assert py_before == random.getstate()
