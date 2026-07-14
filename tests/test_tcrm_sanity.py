import importlib
import sys
from pathlib import Path

import torch
from PIL import Image

from trainers.tcrm.classifier import build_tail_direction, tail_vs_head_margins
from trainers.tcrm.client_core import _tail_split
from trainers.tcrm.feature_cache import build_or_load_feature_cache
from trainers.tcrm.losses import hbs_loss, true_class_margin
from trainers.tcrm.server_core import (
    compute_pre_reliability,
    corroboration,
    direction_consistency,
    empty_sufficient_stats,
    update_core_state,
)
from trainers.tcrm.state import init_core_state, sanitize_residual
from trainers.tcrm.topology import compute_tail_topology
from tcrm_main import build_client_indices_from_allocation_csv, class_counts_from_client_indices


def test_zero_gate_returns_zero_shot_direction():
    z = torch.nn.functional.normalize(torch.randn(8), dim=0)
    rho = torch.randn(8)
    q = build_tail_direction(z, rho, gate=0.0)
    assert torch.allclose(q, z, atol=1e-6)


def test_perpendicular_residual():
    z = torch.nn.functional.normalize(torch.randn(8), dim=0)
    rho = sanitize_residual(torch.randn(8), z, 0.2)
    assert abs(float((rho * z).sum().item())) < 1e-6


def test_norm_projection():
    z = torch.nn.functional.normalize(torch.randn(8), dim=0)
    rho = sanitize_residual(torch.randn(8) * 100, z, 0.2)
    assert rho.norm().item() <= 0.2 + 1e-6


def test_tail_split_singleton_has_empty_holdout():
    adapt, holdout = _tail_split(torch.tensor([7]), holdout_ratio=0.2, holdout_min=1)
    assert adapt.tolist() == [7]
    assert holdout.numel() == 0


def test_tail_split_keeps_holdout_independent_when_possible():
    adapt, holdout = _tail_split(torch.tensor([3, 4]), holdout_ratio=0.2, holdout_min=1)
    assert adapt.tolist() == [3]
    assert holdout.tolist() == [4]


def test_true_class_margin():
    logits = torch.tensor([[1.0, 3.0, 0.0], [5.0, 4.0, 1.0]])
    labels = torch.tensor([1, 2])
    margins = true_class_margin(logits, labels)
    assert torch.allclose(margins, torch.tensor([2.0, -4.0]))


def test_hbs_uses_zero_shot_reference():
    labels = torch.tensor([2])
    zero = torch.tensor([[2.0, 0.0, 0.0]])
    same_hybrid = torch.tensor([[2.0, 0.0, 0.0]])
    worse_hybrid = torch.tensor([[3.0, 0.0, 0.0]])
    same_loss = hbs_loss(same_hybrid, zero, labels, tail_class_ids=[2], non_tail_class_ids=[0, 1])
    worse_loss = hbs_loss(worse_hybrid, zero, labels, tail_class_ids=[2], non_tail_class_ids=[0, 1])
    assert same_loss.item() == 0.0
    assert worse_loss.item() > 0.9


def test_tail_metrics_report_tail_sample_count_and_single_tail_margin():
    logits = torch.tensor([[0.1, 2.0, 1.0], [3.0, 2.0, 1.0]])
    labels = torch.tensor([1, 1])
    metrics = tail_vs_head_margins(logits, labels, {1}, non_tail_class_ids=[0, 2], tail_class_ids=[1])
    assert metrics["num_tail_samples"] == 2
    assert metrics["tail_to_head_error_rate"] == 0.5
    assert metrics["mean_tail_vs_tail_margin"] == 0.0


def test_correct_topology_direction():
    concentrated = torch.tensor([[50.0], [50.0]])
    tensors, _rows = compute_tail_topology(concentrated, [0], torch.tensor([100.0]))
    assert abs(tensors["D"][0].item() - 50.0) < 1e-5
    assert abs(tensors["C"][0].item() - 0.5) < 1e-5
    fragmented = torch.ones(100, 1)
    tensors, _rows = compute_tail_topology(fragmented, [0], torch.tensor([100.0]))
    assert abs(tensors["D"][0].item() - 1.0) < 1e-5
    assert abs(tensors["C"][0].item() - 0.01) < 1e-5


def test_controlled_allocation_csv_builds_client_indices(tmp_path):
    path = tmp_path / "allocation.csv"
    path.write_text(
        "client_id,class_0,class_1\n"
        "0,2,0\n"
        "1,0,2\n",
        encoding="utf-8",
    )
    labels = torch.tensor([0, 0, 0, 1, 1, 1])
    client_indices = build_client_indices_from_allocation_csv(path, labels, num_classes=2, seed=1)
    assert len(client_indices) == 2
    counts = class_counts_from_client_indices(client_indices, labels, num_classes=2)
    assert counts.tolist() == [2.0, 2.0]
    assert all(labels[idx].item() == 0 for idx in client_indices[0])
    assert all(labels[idx].item() == 1 for idx in client_indices[1])


def _state():
    z = torch.eye(4)
    topology = {"M": torch.tensor([10.0]), "C": torch.tensor([0.5]), "D": torch.tensor([5.0])}
    return init_core_state(
        {"ctx_delta": torch.zeros(1, 4)},
        z,
        topology,
        [2],
        [0, 1, 3],
        torch.ones(4) / 4,
        total_rounds=20,
        rho_norm_bound=0.2,
    )


def test_cold_start_reliability_positive():
    state = _state()
    r_pre = compute_pre_reliability(state.M, state.D, torch.zeros_like(state.age), state.m0, state.d0, state.stale_horizon)
    assert r_pre.item() > 0


def test_direction_consistency():
    same = direction_consistency(torch.tensor([[2.0, 0.0]]), torch.tensor([2.0]))
    opposite = direction_consistency(torch.tensor([[0.0, 0.0]]), torch.tensor([2.0]))
    assert same.item() > 0.99
    assert opposite.item() < 1e-6


def test_single_expert_soft_penalty():
    b = corroboration(torch.tensor([1.0]), nu0=2.0)
    assert 0.0 < b.item() < 1.0


def test_no_local_gain_no_write():
    state = _state()
    stats = empty_sufficient_stats(1, 4)
    stats["update_sum"][0] = torch.tensor([0.0, 0.0, 0.1, 0.0])
    stats["update_weight"][0] = 1.0
    stats["unit_direction_sum"][0] = torch.tensor([0.0, 0.0, 1.0, 0.0])
    stats["valid_count"][0] = 1.0
    stats["gain_sum"][0] = 0.0
    before = state.rho.clone()
    update_core_state(state, stats, variant="tcrm_core")
    assert state.last_write[0].item() == 0.0
    assert torch.allclose(state.rho, sanitize_residual((1.0 - state.last_decay).view(-1, 1) * before, state.zero_shot_text[state.tail_class_ids], state.rho_norm_bound))


def test_disable_write_sets_w_to_one_when_update_exists():
    state = _state()
    stats = empty_sufficient_stats(1, 4)
    stats["update_sum"][0] = torch.tensor([0.0, 0.0, 0.1, 0.0])
    stats["update_weight"][0] = 1.0
    update_core_state(state, stats, variant="tcrm_core", disable_write=True)
    assert state.last_write[0].item() == 1.0


class CountingClip(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def encode_image(self, images):
        self.calls += 1
        flat = images.float().view(images.shape[0], -1)
        return flat[:, :4] + 1.0


def test_feature_cache_hit_and_metadata_mismatch(tmp_path):
    data = [(Image.new("RGB", (2, 2), color=(i, 0, 0)), i % 2) for i in range(3)]

    def transform(image):
        return torch.as_tensor(list(image.getdata()), dtype=torch.float32).view(2, 2, 3).permute(2, 0, 1) / 255.0

    model = CountingClip()
    path = tmp_path / "cache.pt"
    f1, y1, _ = build_or_load_feature_cache(data, transform, model, path, "toy", "train", "fake", batch_size=2, device="cpu", dtype="float32", log_fn=None)
    assert model.calls > 0
    calls = model.calls
    f2, y2, _ = build_or_load_feature_cache(data, transform, model, path, "toy", "train", "fake", batch_size=2, device="cpu", dtype="float32", log_fn=None)
    assert model.calls == calls
    assert torch.allclose(f1, f2)
    _f3, _y3, _ = build_or_load_feature_cache(data, transform, model, path, "toy", "test", "fake", batch_size=2, device="cpu", dtype="float32", log_fn=None)
    assert model.calls > calls


def test_standalone_entrypoint_does_not_import_federated_main():
    sys.modules.pop("tcrm_main", None)
    sys.modules.pop("federated_main", None)
    importlib.import_module("tcrm_main")
    assert "federated_main" not in sys.modules
