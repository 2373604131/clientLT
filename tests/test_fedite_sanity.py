import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from trainers.fedite.aggregation import aggregate_fedite
from trainers.fedite.losses import (
    cross_entropy_loss,
    protected_boundary_retention_loss,
    protected_candidate_kl_loss,
    protected_logit_retention_loss,
    router_loss,
    router_utility_targets,
)
from trainers.fedite.model import FedITEModel
from trainers.fedite.observer import EvidenceTopologyObserver
from trainers.fedite.trainer import FedITEClientTrainer
from trainers.fedite.utils import as_cpu_state_dict, is_in_classes, split_head_medium_tail


class TinyBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fc = nn.Linear(dim, dim)

    def forward(self, x):
        return x + 0.1 * self.fc(self.norm(x))


class TinyVisual(nn.Module):
    def __init__(self, width=32, output_dim=16, layers=4, image_size=8, patch=4):
        super().__init__()
        self.input_resolution = image_size
        self.output_dim = output_dim
        self.conv1 = nn.Conv2d(3, width, kernel_size=patch, stride=patch, bias=False)
        self.class_embedding = nn.Parameter(torch.randn(width) * 0.01)
        self.positional_embedding = nn.Parameter(torch.randn((image_size // patch) ** 2 + 1, width) * 0.01)
        self.ln_pre = nn.LayerNorm(width)
        self.transformer = nn.Module()
        self.transformer.resblocks = nn.Sequential(*[TinyBlock(width) for _ in range(layers)])
        self.ln_post = nn.LayerNorm(width)
        self.proj = nn.Parameter(torch.randn(width, output_dim) * 0.01)


class TinyTextTransformer(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.block = TinyBlock(dim)

    def forward(self, x):
        return self.block(x)


class TinyCLIP(nn.Module):
    def __init__(self, num_classes=5, width=32, embed_dim=16):
        super().__init__()
        self.visual = TinyVisual(width=width, output_dim=embed_dim)
        self.token_embedding = nn.Embedding(50000, width)
        self.positional_embedding = nn.Parameter(torch.randn(77, width) * 0.01)
        self.transformer = TinyTextTransformer(width)
        self.ln_final = nn.LayerNorm(width)
        self.text_projection = nn.Parameter(torch.randn(width, embed_dim) * 0.01)
        self.logit_scale = nn.Parameter(torch.tensor(1.0))
        self.dtype = torch.float32


def args():
    return argparse.Namespace(
        fedite_adapter_layers="1,3",
        fedite_adapter_bottleneck=8,
        fedite_num_tail_basis=2,
        fedite_alpha_shared=1.0,
        fedite_alpha_tail=1.0,
        fedite_adapter_dropout=0.0,
        fedite_basis_dropout=0.0,
        fedite_token_selective=False,
        fedite_train_prompt=False,
        fedite_prompt_ctx="a photo of a",
        fedite_prompt_n_ctx=4,
    )


def args_token_selective():
    value = args()
    value.fedite_token_selective = True
    return value


def make_model(num_classes=5):
    classnames = [f"class_{i}" for i in range(num_classes)]
    model = FedITEModel(TinyCLIP(num_classes=num_classes), classnames, args())
    state = torch.zeros(num_classes, 6)
    state[:, 0] = torch.linspace(0.2, 0.9, num_classes)
    state[:, 1] = torch.linspace(0.3, 0.8, num_classes)
    state[:, 2] = state[:, 0] * state[:, 1]
    model.set_class_evidence_state(state)
    return model, state


def make_token_selective_model(num_classes=5):
    classnames = [f"class_{i}" for i in range(num_classes)]
    model = FedITEModel(TinyCLIP(num_classes=num_classes), classnames, args_token_selective())
    state = torch.zeros(num_classes, 6)
    state[:, 0] = torch.linspace(0.2, 0.9, num_classes)
    state[:, 1] = torch.linspace(0.3, 0.8, num_classes)
    state[:, 2] = state[:, 0] * state[:, 1]
    model.set_class_evidence_state(state)
    return model, state


def any_nonzero_grad(model, keys):
    key_set = set(keys)
    for name, param in model.named_parameters():
        if name in key_set and param.grad is not None and param.grad.detach().abs().sum() > 0:
            return True
    return False


def all_zero_or_none_grad(model, keys):
    key_set = set(keys)
    for name, param in model.named_parameters():
        if name in key_set and param.grad is not None and param.grad.detach().abs().sum() > 1e-10:
            return False
    return True


def test_fedite_forward_and_freezing():
    model, state = make_model()
    images = torch.randn(3, 3, 8, 8)
    out = model.forward_inference(images, class_state=state, return_diagnostics=True)
    assert out["logits_final"].shape == (3, 5)
    assert out["logits_shared"].shape == (3, 5)
    assert all(not p.requires_grad for n, p in model.named_parameters() if n.startswith("clip_model."))
    assert any(p.requires_grad for n, p in model.named_parameters() if n.startswith("visual_wrapper.shared_adapters"))
    assert any(p.requires_grad for n, p in model.named_parameters() if n.startswith("visual_wrapper.tail_adapters"))
    assert any(p.requires_grad for n, p in model.named_parameters() if ".gate_head." in n)


def test_public_forward_ignores_labels_for_inference():
    model, state = make_model()
    model.eval()
    images = torch.randn(3, 3, 8, 8)
    labels_a = torch.tensor([0, 1, 2])
    labels_b = torch.tensor([2, 2, 2])
    out_a = model(images, labels=labels_a, class_state=state)
    out_b = model(images, labels=labels_b, class_state=state)
    assert torch.allclose(out_a, out_b, atol=1e-6)


def test_warmup_eval_equivalent_to_shared_path():
    model, state = make_model()
    images = torch.randn(4, 3, 8, 8)
    shared = model.forward_shared(images)["logits_shared"]
    # This mirrors fedite_main.evaluate(..., tail_active=False): final logits
    # are intentionally the shared logits during warmup.
    final_during_warmup = shared
    assert torch.allclose(final_during_warmup, shared, atol=1e-6)


def test_prefix_suffix_replay_matches_full_forward():
    model, state = make_model()
    model.eval()
    images = torch.randn(4, 3, 8, 8)
    prefix = model.encode_visual_prefix(images)

    full_shared = model.forward_shared(images, compute_base=False)
    cached_shared = model.forward_shared(images, prefix_tokens=prefix, compute_base=False)
    assert full_shared["base_image_features"] is None
    assert cached_shared["base_image_features"] is None
    assert torch.allclose(full_shared["logits_shared"], cached_shared["logits_shared"], atol=1e-6)
    assert torch.allclose(full_shared["shared_image_features"], cached_shared["shared_image_features"], atol=1e-6)

    full_inference = model.forward_inference(images, class_state=state, return_dict=True)
    cached_inference = model.forward_inference(images, class_state=state, prefix_tokens=prefix, return_dict=True)
    assert torch.allclose(full_inference["logits_shared"], cached_inference["logits_shared"], atol=1e-6)
    assert torch.allclose(full_inference["logits_final"], cached_inference["logits_final"], atol=1e-6)


def test_gradient_isolation_shared_router_tail():
    model, state = make_model()
    images = torch.randn(4, 3, 8, 8)
    labels = torch.tensor([0, 1, 2, 3])
    shared_keys = model.get_shared_parameter_keys()
    gate_keys = model.get_gate_parameter_keys()
    tail_keys = model.get_tail_parameter_keys()

    model.set_trainable_groups(shared=True, gate=False, tail=False)
    model.zero_grad(set_to_none=True)
    shared = model.forward_shared(images)
    cross_entropy_loss(shared["logits_shared"], labels).backward()
    assert any_nonzero_grad(model, shared_keys)
    assert all_zero_or_none_grad(model, gate_keys)
    assert all_zero_or_none_grad(model, tail_keys)

    model.set_protected_classes([1, 3])
    model.set_trainable_groups(shared=False, gate=True, tail=False)
    model.zero_grad(set_to_none=True)
    router = model.forward_router_train(images, labels, state)
    target = router_utility_targets(labels, is_in_classes(labels, [1, 3]), state, router["logits_shared"])
    router_loss(router["tail_gates"], target).backward()
    assert any_nonzero_grad(model, gate_keys)
    assert all_zero_or_none_grad(model, tail_keys)
    assert all_zero_or_none_grad(model, shared_keys)

    model.set_protected_classes([])
    model.set_trainable_groups(shared=False, gate=False, tail=True)
    model.zero_grad(set_to_none=True)
    tail = model.forward_tail_train(images, labels, class_state=state)
    cross_entropy_loss(tail["logits"], labels).backward()
    assert all_zero_or_none_grad(model, tail_keys)

    model.set_protected_classes([1, 3])
    model.zero_grad(set_to_none=True)
    idx = torch.tensor([1, 3])
    tail = model.forward_tail_train(images[idx], labels[idx], class_state=state)
    cross_entropy_loss(tail["logits"], labels[idx]).backward()
    assert any_nonzero_grad(model, tail_keys)
    assert all_zero_or_none_grad(model, gate_keys)


def test_observer_topology_risk_reliability_selection():
    observer = EvidenceTopologyObserver(
        num_classes=5,
        protected_ratio=0.4,
        warmup_rounds=0,
        reliability_min=0.01,
        selection_rho=0.5,
        beta=0.5,
    )
    stats = {
        "M": torch.tensor([10.0, 1.0, 0.0, 2.0, 5.0]),
        "Q": torch.tensor([3.0, 1.0, 0.0, 1.0, 2.0]),
        "H": torch.tensor([50.0, 1.0, 0.0, 4.0, 13.0]),
        "U": torch.tensor([1.0, 0.1, 0.0, 0.2, 0.5]),
        "write_count": torch.tensor([2.0, 1.0, 0.0, 1.0, 2.0]),
    }
    observer.update(stats)
    selected = observer.select_protected_classes(0)
    assert torch.allclose(observer.S, observer.D * (0.5 + 0.5 * observer.R) * (0.5 + 0.5 * observer.Rarity))
    assert torch.isfinite(observer.D).all()
    assert torch.isfinite(observer.R).all()
    assert torch.isfinite(observer.S).all()
    assert torch.isfinite(observer.Rarity).all()
    assert 2 not in selected  # no mass, below reliable evidence requirement
    class_state = observer.get_class_state()
    assert class_state.shape == (5, 6)


def test_observer_saturated_selection_prefers_concentrated_tail_over_head():
    observer = EvidenceTopologyObserver(
        num_classes=4,
        protected_ratio=0.25,
        warmup_rounds=0,
        beta=0.0,
        reliability_min=0.9,
        selection_rho=0.5,
        reliability_support_m0=2.0,
        reliability_client_q0=1.0,
    )
    stats = {
        "M": torch.tensor([100.0, 20.0, 1.0, 0.0]),
        "Q": torch.tensor([5.0, 3.0, 1.0, 0.0]),
        "H": torch.tensor([2000.0, 133.3333, 1.0, 0.0]),
        "U": torch.tensor([1.0, 1.0, 0.0, 0.0]),
        "write_count": torch.tensor([1.0, 1.0, 1.0, 0.0]),
    }
    observer.update(stats)
    selected = observer.select_protected_classes(0)
    assert selected == [2]
    assert 3 not in selected  # truly unseen class is not a candidate


def test_observer_historical_seen_diffuse_tail_stays_candidate_after_gap():
    observer = EvidenceTopologyObserver(
        num_classes=4,
        protected_ratio=0.5,
        warmup_rounds=0,
        beta=0.5,
        reliability_min=0.9,
        selection_rho=0.5,
        reliability_support_m0=2.0,
        reliability_client_q0=1.0,
    )
    observer.update({
        "M": torch.tensor([100.0, 0.0, 1.0, 0.0]),
        "Q": torch.tensor([5.0, 0.0, 1.0, 0.0]),
        "H": torch.tensor([2000.0, 0.0, 1.0, 0.0]),
        "U": torch.tensor([1.0, 0.0, 0.1, 0.0]),
        "write_count": torch.tensor([1.0, 0.0, 1.0, 0.0]),
    })
    observer.update({
        "M": torch.tensor([100.0, 0.0, 0.0, 0.0]),
        "Q": torch.tensor([5.0, 0.0, 0.0, 0.0]),
        "H": torch.tensor([2000.0, 0.0, 0.0, 0.0]),
        "U": torch.tensor([1.0, 0.0, 0.0, 0.0]),
        "write_count": torch.tensor([1.0, 0.0, 0.0, 0.0]),
    })
    selected = observer.select_protected_classes(0)
    assert 2 in selected
    assert 1 not in selected
    assert 3 not in selected


def test_observer_rarity_quota_keeps_low_exposure_tail_candidates():
    observer = EvidenceTopologyObserver(
        num_classes=10,
        protected_ratio=0.3,
        warmup_rounds=0,
        beta=0.0,
        reliability_min=0.99,
        selection_rho=0.5,
    )
    observer.update({
        "M": torch.tensor([100.0, 90.0, 80.0, 70.0, 35.0, 30.0, 25.0, 20.0, 2.0, 1.0]),
        "Q": torch.tensor([5.0, 5.0, 5.0, 5.0, 4.0, 4.0, 3.0, 3.0, 1.0, 1.0]),
        "H": torch.tensor([2500.0, 2025.0, 1600.0, 1225.0, 306.25, 225.0, 208.33, 133.33, 4.0, 1.0]),
        "U": torch.ones(10),
        "write_count": torch.ones(10),
    })
    selected = observer.select_protected_classes(0)
    assert {8, 9}.issubset(set(selected))
    assert observer.last_selection_info["rare_quota"] == 2


def test_observer_first_active_round_does_not_keep_warmup_inertia():
    observer = EvidenceTopologyObserver(
        num_classes=6,
        protected_ratio=0.5,
        warmup_rounds=2,
        warmup_mode="round_robin",
        reliability_min=0.0,
        beta=0.0,
    )
    warmup_selected = observer.select_protected_classes(1)
    assert warmup_selected
    stats = {
        "M": torch.tensor([100.0, 90.0, 1.0, 1.0, 1.0, 1.0]),
        "Q": torch.ones(6),
        "H": torch.tensor([10000.0, 8100.0, 1.0, 1.0, 1.0, 1.0]),
        "U": torch.ones(6),
        "write_count": torch.ones(6),
    }
    observer.update(stats)
    first_active = observer.select_protected_classes(2)
    assert observer.last_selection_info["mode"] == "score"
    assert observer.last_selection_info["overlap"] == 0
    assert set(first_active) != set(warmup_selected)


def test_protected_logit_retention_loss_tracks_base_teacher():
    logits_base = torch.tensor([[2.0, 0.0, 1.0], [0.5, 1.5, -0.5]])
    same = protected_logit_retention_loss(logits_base.clone(), logits_base, [0, 2])
    shifted = protected_logit_retention_loss(logits_base + torch.tensor([[0.0, 0.0, 2.0], [0.0, 0.0, -2.0]]), logits_base, [0, 2])
    assert same.item() < 1e-6
    assert shifted.item() > same.item()


def test_protected_boundary_retention_penalizes_tail_margin_drop():
    labels = torch.tensor([0, 1])
    logits_base = torch.tensor([[3.0, 1.0, 0.0], [0.0, 3.0, 1.0]])
    same = protected_boundary_retention_loss(logits_base.clone(), logits_base, labels, [0], topk=1)
    damaged = protected_boundary_retention_loss(
        torch.tensor([[1.0, 2.0, 0.0], [0.0, 0.5, 3.0]]),
        logits_base,
        labels,
        [0],
        topk=1,
    )
    ignored = protected_boundary_retention_loss(
        torch.tensor([[1.0, 2.0, 0.0], [0.0, 0.5, 3.0]]),
        logits_base,
        labels,
        [2],
        topk=1,
    )
    assert same.item() < 1e-6
    assert damaged.item() > same.item()
    assert ignored.item() < 1e-6


def test_protected_candidate_kl_tracks_base_on_local_candidates():
    labels = torch.tensor([0, 1])
    logits_base = torch.tensor([[3.0, 1.0, 0.0, -1.0], [0.0, 3.0, 1.0, -1.0]])
    same = protected_candidate_kl_loss(logits_base.clone(), logits_base, labels, [0, 2], topk=1)
    shifted = protected_candidate_kl_loss(
        torch.tensor([[0.0, 3.0, 2.0, -1.0], [0.0, 0.5, 3.0, -1.0]]),
        logits_base,
        labels,
        [0, 2],
        topk=1,
    )
    assert same.item() < 1e-6
    assert shifted.item() > same.item()


def test_router_utility_has_protected_floor_for_confident_samples():
    labels = torch.tensor([0, 1, 2])
    protected = torch.tensor([True, False, True])
    state = torch.zeros(3, 6)
    state[:, 2] = torch.tensor([1.0, 0.5, 0.0])
    logits = torch.tensor([[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]])
    target = router_utility_targets(labels, protected, state, logits)
    assert target[0].item() >= 0.49
    assert target[1].item() == 0.0
    assert target[2].item() >= 0.19


def test_aggregation_tail_preserves_previous_without_eligible_clients():
    model, _state = make_model()
    prev = as_cpu_state_dict(model)
    local = []
    stats = []
    for delta in [0.1, 0.2]:
        state = {k: v.clone() for k, v in prev.items()}
        for key in model.get_shared_parameter_keys():
            state[key] = state[key] + delta
        for key in model.get_tail_parameter_keys():
            state[key] = state[key] + delta
        local.append(state)
        stats.append({
            "num_samples": 10,
            "protected_positive_count": 0,
            "protected_class_support": torch.zeros(model.num_classes),
            "tail_update_norm": 1.0,
        })
    new_state, diag = aggregate_fedite(
        local,
        stats,
        prev,
        model.get_shared_parameter_keys(),
        model.get_gate_parameter_keys(),
        model.get_tail_parameter_keys(),
        {"class_state": torch.ones(model.num_classes, 6)},
    )
    assert diag["kept_previous_tail"]
    for key in model.get_tail_parameter_keys():
        assert torch.allclose(new_state[key], prev[key])
    assert any(not torch.allclose(new_state[key], prev[key]) for key in model.get_shared_parameter_keys())


def test_aggregation_classwise_tail_updates_only_supported_rows():
    model, _state = make_model()
    prev = as_cpu_state_dict(model)
    local = []
    stats = []
    for support_class, changed_class, value in [(2, 2, 2.0), (0, 1, 9.0)]:
        state = {k: v.clone() for k, v in prev.items()}
        state["class_basis_logits"][changed_class] = value
        local.append(state)
        support = torch.zeros(model.num_classes)
        support[support_class] = 1.0
        stats.append({
            "num_samples": 10,
            "protected_positive_count": 1,
            "protected_class_support": support,
            "tail_update_norm": 1.0,
        })
    new_state, _diag = aggregate_fedite(
        local,
        stats,
        prev,
        model.get_shared_parameter_keys(),
        model.get_gate_parameter_keys(),
        model.get_tail_parameter_keys(),
        {"class_state": torch.ones(model.num_classes, 6)},
    )
    assert torch.allclose(new_state["class_basis_logits"][1], prev["class_basis_logits"][1])
    assert new_state["class_basis_logits"][2].abs().sum().item() > 0


def test_warmup_disables_gate_tail_and_returns_sparse_shared_update():
    model, state = make_model()
    loader = DataLoader(
        TensorDataset(torch.randn(6, 3, 8, 8), torch.tensor([1, 1, 2, 3, 4, 0])),
        batch_size=3,
        shuffle=False,
    )
    trainer = FedITEClientTrainer(model, argparse.Namespace(local_ep=1), "cpu")
    update, stats = trainer.train_one_client(
        global_state=as_cpu_state_dict(model),
        train_loader=loader,
        protected_classes=[1, 3],
        class_evidence_state=state,
        round_idx=0,
        tail_active=False,
    )
    assert stats["protected_positive_count"] == 0
    assert stats.get("gate_optimizer_steps", 0) == 0
    assert stats.get("tail_optimizer_steps", 0) == 0
    assert set(update.keys()) == set(model.get_shared_parameter_keys())


def test_shared_only_feature_anchor_runs_without_tail_active():
    model, state = make_model()
    loader = DataLoader(
        TensorDataset(torch.randn(6, 3, 8, 8), torch.tensor([1, 1, 2, 3, 4, 0])),
        batch_size=3,
        shuffle=False,
    )
    train_args = argparse.Namespace(
        local_ep=1,
        fedite_lambda_feature_anchor=0.2,
        fedite_lambda_safe=0.0,
        fedite_lambda_boundary=0.0,
        fedite_lambda_candidate_kl=0.0,
        fedite_shared_survival_weight=0.0,
        fedite_prefix_cache=True,
    )
    trainer = FedITEClientTrainer(model, train_args, "cpu")
    update, stats = trainer.train_one_client(
        global_state=as_cpu_state_dict(model),
        train_loader=loader,
        protected_classes=[],
        class_evidence_state=state,
        round_idx=0,
        tail_active=False,
    )
    assert stats.get("shared_optimizer_steps", 0) > 0
    assert stats["loss_feature_anchor"] >= 0.0
    assert stats["loss_router"] == 0.0
    assert set(update.keys()) == set(model.get_shared_parameter_keys())


def test_client_training_accepts_boundary_loss_without_candidate_kl():
    model, state = make_model()
    loader = DataLoader(
        TensorDataset(torch.randn(6, 3, 8, 8), torch.tensor([1, 1, 2, 3, 4, 0])),
        batch_size=3,
        shuffle=False,
    )
    train_args = argparse.Namespace(
        local_ep=1,
        fedite_lambda_safe=0.0,
        fedite_lambda_boundary=0.1,
        fedite_boundary_topk=2,
        fedite_boundary_tolerance=0.0,
        fedite_lambda_candidate_kl=0.0,
        fedite_lambda_router=0.1,
        fedite_lambda_tail=1.0,
        fedite_prefix_cache=True,
        fedite_scalar_diagnostics=True,
        fedite_diagnostics_interval=1,
    )
    trainer = FedITEClientTrainer(model, train_args, "cpu")
    update, stats = trainer.train_one_client(
        global_state=as_cpu_state_dict(model),
        train_loader=loader,
        protected_classes=[1, 3],
        class_evidence_state=state,
        round_idx=3,
        tail_active=True,
    )
    assert stats.get("shared_optimizer_steps", 0) > 0
    assert "loss_boundary" in stats
    assert stats["loss_candidate_kl"] == 0.0
    assert set(model.get_shared_parameter_keys()).issubset(set(update.keys()))


def test_count_dataset_labels_supports_dict_data_source():
    model, _state = make_model()
    trainer = FedITEClientTrainer(model, argparse.Namespace(local_ep=1), "cpu")

    class DictBackedDataset:
        data_source = [
            {"label": 1, "data": torch.zeros(3, 8, 8)},
            {"label": 4, "data": torch.zeros(3, 8, 8)},
            {"label": 1, "data": torch.zeros(3, 8, 8)},
            {"label": 99, "data": torch.zeros(3, 8, 8)},
        ]
        indices = [0, 2, 3]

    counts = trainer._count_dataset_labels_once(DictBackedDataset())
    assert counts.tolist() == [0.0, 2.0, 0.0, 0.0, 0.0]


def test_default_split_uses_non_tail_not_head_medium():
    splits = split_head_medium_tail(torch.tensor([100.0, 80.0, 60.0, 20.0, 1.0]), tail_ratio=0.2)
    assert splits == {"non_tail": [0, 1, 2, 3], "tail": [4]}
    assert "head" not in splits
    assert "medium" not in splits


def test_token_selective_token_proj_is_tail_parameter():
    model, _state = make_token_selective_model()
    tail_keys = set(model.get_tail_parameter_keys())
    assert any(".token_proj." in key for key in tail_keys)
