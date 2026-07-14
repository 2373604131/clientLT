import argparse
import csv
import importlib.util
import json
import random
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

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

stub_dassl_utils = types.ModuleType("Dassl.dassl.utils")
stub_dassl_utils.setup_logger = lambda *args, **kwargs: None
stub_dassl_utils.set_random_seed = lambda *args, **kwargs: None
stub_dassl_utils.Registry = lambda *args, **kwargs: None
stub_dassl_utils.MetricMeter = object
stub_dassl_utils.AverageMeter = object
stub_dassl_utils.tolist_if_not = lambda value: value if isinstance(value, list) else [value]
stub_dassl_utils.count_num_param = lambda *args, **kwargs: 0
stub_dassl_utils.load_checkpoint = lambda *args, **kwargs: {}
stub_dassl_utils.save_checkpoint = lambda *args, **kwargs: None
stub_dassl_utils.mkdir_if_missing = lambda *args, **kwargs: None
stub_dassl_utils.resume_from_checkpoint = lambda *args, **kwargs: 0
stub_dassl_utils.load_pretrained_weights = lambda *args, **kwargs: None
sys.modules.setdefault("Dassl.dassl.utils", stub_dassl_utils)

stub_dassl_config = types.ModuleType("Dassl.dassl.config")
stub_dassl_config.get_cfg_default = lambda: None
sys.modules.setdefault("Dassl.dassl.config", stub_dassl_config)

stub_dassl_engine = types.ModuleType("Dassl.dassl.engine")
stub_dassl_engine.__path__ = [str(ROOT / "Dassl" / "dassl" / "engine")]
stub_dassl_engine.build_trainer = lambda *args, **kwargs: None
sys.modules.setdefault("Dassl.dassl.engine", stub_dassl_engine)

stub_fed_utils = types.ModuleType("utils.fed_utils")
stub_fed_utils.average_weights = lambda *args, **kwargs: None
sys.modules.setdefault("utils.fed_utils", stub_fed_utils)

stub_prompt_loss = types.ModuleType("loss.prompt_loss")
stub_prompt_loss.PromptLoss = object
stub_prompt_loss.update_class_priors = lambda *args, **kwargs: None
sys.modules.setdefault("loss.prompt_loss", stub_prompt_loss)

stub_capt = types.ModuleType("trainers.capt")
stub_capt.MABScheduler = object
sys.modules.setdefault("trainers.capt", stub_capt)

from federated_main import add_expF_runtime_arguments, apply_federated_runtime_overrides, save_partition_summary
from utils.datasplit import _allocate_class_budgets, partition_client_longtail

cifar100_lt_spec = importlib.util.spec_from_file_location(
    "local_cifar100_LT",
    ROOT / "datasets" / "cifar100_LT.py",
)
cifar100_LT = importlib.util.module_from_spec(cifar100_lt_spec)
cifar100_lt_spec.loader.exec_module(cifar100_LT)


def _np_state_equal(left, right):
    return (
        left[0] == right[0]
        and np.array_equal(left[1], right[1])
        and left[2:] == right[2:]
    )


def _make_parser():
    parser = argparse.ArgumentParser()
    add_expF_runtime_arguments(parser)
    return parser


def _maps_equal(left, right):
    return all(np.array_equal(left[k], right[k]) for k in left)


def test_expF_parser_accepts_runtime_arguments_and_default_command():
    parser = _make_parser()

    args = parser.parse_args([
        "--local_epochs",
        "5",
        "--split_seed",
        "7",
        "--isolate_local_optimizer_state",
        "True",
        "--federated_single_scheduler_step",
        "True",
    ])

    assert args.local_epochs == 5
    assert args.split_seed == 7
    assert args.isolate_local_optimizer_state is True
    assert args.federated_single_scheduler_step is True

    default_args = parser.parse_args([])
    assert default_args.local_epochs is None
    assert default_args.split_seed == 1
    assert default_args.isolate_local_optimizer_state is False
    assert default_args.federated_single_scheduler_step is False


def test_save_partition_summary_writes_panelC_topology_and_budget_fields(tmp_path):
    client_class_counts = {
        0: torch.tensor([8, 0, 1, 0]),
        1: torch.tensor([2, 5, 2, 0]),
        2: torch.tensor([0, 5, 0, 4]),
    }
    args = SimpleNamespace(
        partition="client-longtail",
        trainer="PromptFL",
        seed=1,
        tail_class_ratio=0.5,
        tail_client_ratio=1 / 3,
        specialization_lambda=0.75,
        intra_group_alpha=0.5,
        head_leakage_scale=3.0,
    )

    save_partition_summary(tmp_path, client_class_counts, args, num_users=3, num_classes=4)

    summary = json.loads((tmp_path / "partition_summary.json").read_text(encoding="utf-8"))
    with (tmp_path / "class_topology.csv").open("r", newline="", encoding="utf-8") as f:
        topology_rows = list(csv.DictReader(f))

    assert {"top1_client_mass", "top2_client_mass", "effective_client_number", "num_support_clients"}.issubset(
        topology_rows[0].keys()
    )
    assert set(summary["tail_classes"]) == {2, 3}
    assert set(summary["head_classes"]) == {0, 1}
    assert summary["specialization_lambda"] == 0.75
    assert summary["intra_group_alpha"] == 0.5
    assert summary["head_leakage_scale"] == 3.0
    assert summary["tail_client_ids"] == [2]
    assert summary["tail_samples_in_tail_clients"] == 4.0
    assert summary["non_tail_samples_in_tail_clients"] == 5.0
    assert summary["tail_to_tail_budget"] == 4.0
    assert summary["non_tail_to_tail_budget"] == 5.0
    assert summary["actual_tail_client_purity"] == pytest.approx(4 / 9)
    assert summary["client_sample_min"] == 9.0
    assert summary["client_sample_max"] == 9.0
    assert summary["client_sample_cv"] == 0.0
    assert summary["tail_top1_client_mass_mean"] == pytest.approx((2 / 3 + 1.0) / 2)
    assert summary["tail_top2_client_mass_mean"] == pytest.approx(1.0)
    assert summary["tail_effective_client_number_mean"] == pytest.approx((1.8 + 1.0) / 2)


def test_local_epochs_override_validation_and_order():
    cfg = SimpleNamespace(
        DATASET=SimpleNamespace(SPLIT_SEED=99),
        OPTIM=SimpleNamespace(MAX_EPOCH=3),
    )

    apply_federated_runtime_overrides(cfg, SimpleNamespace(local_epochs=None, split_seed=7))
    assert cfg.OPTIM.MAX_EPOCH == 3
    assert cfg.DATASET.SPLIT_SEED == 7

    apply_federated_runtime_overrides(cfg, SimpleNamespace(local_epochs=5, split_seed=8))
    assert cfg.OPTIM.MAX_EPOCH == 5
    assert cfg.DATASET.SPLIT_SEED == 8

    with pytest.raises(ValueError):
        apply_federated_runtime_overrides(cfg, SimpleNamespace(local_epochs=0, split_seed=1))
    with pytest.raises(ValueError):
        apply_federated_runtime_overrides(cfg, SimpleNamespace(local_epochs=-1, split_seed=1))


def test_split_seed_reaches_cifar100lt_wrapper(monkeypatch):
    captured = {}

    def fake_partition_data_LT(*args, **kwargs):
        captured["split_seed"] = kwargs["split_seed"]
        return (
            ["train0", "train1"],
            ["test0"],
            {0: "zero", 1: "one"},
            ["zero", "one"],
            {0: np.asarray([0]), 1: np.asarray([1])},
            {0: np.asarray([0]), 1: np.asarray([], dtype=np.int64)},
            {0: {0: 1}, 1: {1: 1}},
            {0: {0: 1}, 1: {}},
            np.asarray([0, 1]),
        )

    monkeypatch.setattr(cifar100_LT, "partition_data_LT", fake_partition_data_LT)

    cfg = SimpleNamespace(
        DATASET=SimpleNamespace(
            ROOT=".",
            USERS=2,
            PARTITION="noniid-labeldir-fine",
            BETA=0.5,
            IMB_FACTOR=0.01,
            IMB_TYPE="exp",
            HEAD_CLIENT_RATIO=0.9,
            TAIL_CLIENT_RATIO=0.1,
            HEAD_CLASS_RATIO=0.8,
            TAIL_CLASS_RATIO=0.2,
            SPECIALIZATION_LAMBDA=1.0,
            INTRA_GROUP_ALPHA=0.1,
            HEAD_LEAKAGE_SCALE=3.0,
            SPLIT_SEED=7,
        )
    )

    dataset = cifar100_LT.Cifar100_LT(cfg)

    assert captured["split_seed"] == 7
    assert dataset.federated_train_x == [["train0"], ["train1"]]


def test_client_longtail_seed_reproducibility_variation_and_counts():
    labels = np.repeat(np.arange(10), 20)
    kwargs = dict(
        n_parties=5,
        num_classes=10,
        head_client_ratio=0.8,
        tail_client_ratio=0.2,
        head_class_ratio=0.8,
        tail_class_ratio=0.2,
        specialization_lambda=0.5,
        intra_group_alpha=0.7,
        head_leakage_scale=3.0,
    )

    map_a = partition_client_longtail(labels, rng=np.random.RandomState(7), **kwargs)
    map_b = partition_client_longtail(labels, rng=np.random.RandomState(7), **kwargs)
    map_c = partition_client_longtail(labels, rng=np.random.RandomState(8), **kwargs)

    assert _maps_equal(map_a, map_b)
    assert any(not np.array_equal(map_a[i], map_c[i]) for i in range(5))

    merged = np.concatenate([map_a[i] for i in range(5)])
    assert len(merged) == len(labels)
    assert len(np.unique(merged)) == len(labels)
    assert np.array_equal(np.bincount(labels[merged], minlength=10), np.bincount(labels, minlength=10))


def test_client_longtail_does_not_pollute_global_rng_and_formulas_hold():
    labels = np.repeat(np.arange(10), 20)

    np.random.seed(123)
    random.seed(456)
    np_before = np.random.get_state()
    py_before = random.getstate()

    partition_client_longtail(
        labels,
        n_parties=5,
        num_classes=10,
        head_client_ratio=0.8,
        tail_client_ratio=0.2,
        head_class_ratio=0.8,
        tail_class_ratio=0.2,
        specialization_lambda=0.5,
        intra_group_alpha=0.7,
        head_leakage_scale=3.0,
        rng=np.random.RandomState(9),
    )

    assert _np_state_equal(np_before, np.random.get_state())
    assert py_before == random.getstate()

    class_budgets = _allocate_class_budgets({8: 13, 9: 13}, 20)
    assert class_budgets == {8: 10, 9: 10}

    q_t = 0.1
    lambda_t = 0.75
    rho = 3.0
    n_tail = 26
    n_non_tail = 174
    tail_to_tail_budget = round(n_tail * (q_t + (1.0 - q_t) * lambda_t))
    non_tail_to_tail_budget = round(rho * n_tail * q_t * (1.0 - lambda_t))
    assert tail_to_tail_budget == 20
    assert min(max(non_tail_to_tail_budget, 0), n_non_tail) == 2
