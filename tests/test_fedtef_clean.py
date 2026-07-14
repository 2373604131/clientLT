import copy
from types import SimpleNamespace

import torch

from trainers.fedtef_aggregation import (
    compute_tail_update_stats,
    fedavg_keys,
    fedtef_v10_evidence_preserving_tailagg,
    is_shared_stream_key,
    is_tail_stream_key,
)
from trainers.fedtef_loss import (
    build_positive_row_mask,
    compute_fedtef_loss,
    controlled_hard_negative_loss,
    mask_classwise_grad,
)
from trainers.fedtef_model import TailResidualStream
from trainers.fedtef_observer import TopologyExposureSurvivalObserver


def _loss_cfg():
    return SimpleNamespace(
        EXPOSURE_EPS=1e-6,
        LOSS_BASE_WEIGHT=1.0,
        V10_PRIOR_BASE_WEIGHT=0.2,
        LOSS_TAIL_WEIGHT=0.8,
        LOSS_FUSED_WEIGHT=0.2,
        LOSS_KEEP_KL_WEIGHT=0.05,
        V10_PRIOR_KAPPA=0.3,
        V10_PRIOR_W_MAX=2.0,
        V10_HARDNEG_TOPM=2,
        V10_HARDNEG_LAMBDA=0.5,
        V10_SAFE_CONF_THRESHOLD=0.7,
    )


def test_loss_has_all_clean_fedtef_terms_and_stays_finite():
    labels = torch.tensor([0, 2, 1])
    logits_base = torch.tensor(
        [[2.0, 1.0, 0.0], [0.0, 1.0, 2.0], [0.5, 1.5, 0.0]],
        requires_grad=True,
    )
    residual = torch.zeros_like(logits_base)
    residual[0, 0] = 1.0
    outputs = {
        "logits_base": logits_base,
        "residual_tail": residual,
        "gated_residual": 0.5 * residual,
        "logits": logits_base + 0.5 * residual,
    }
    loss, items = compute_fedtef_loss(
        outputs,
        labels,
        gate=torch.tensor([1.0, 0.0, 1.0]),
        tail_score=torch.tensor([2.0, 1.0, 2.0]),
        cfg=_loss_cfg(),
    )
    assert torch.isfinite(loss)
    assert {"loss_base", "loss_prior", "loss_res", "loss_fused", "loss_safe"}.issubset(items)


def test_hard_negative_loss_returns_zero_without_protected_samples():
    logits_base = torch.randn(2, 4)
    residual = torch.randn(2, 4)
    labels = torch.tensor([0, 1])
    loss = controlled_hard_negative_loss(
        logits_base,
        residual,
        labels,
        protected_label=torch.tensor([False, False]),
    )
    assert loss.item() == 0.0


def test_positive_row_gradient_mask_updates_only_protected_labels():
    param = torch.nn.Parameter(torch.ones(4, 3))
    param.grad = torch.ones_like(param)
    labels = torch.tensor([1, 2, 2])
    row_mask = build_positive_row_mask(labels, torch.tensor([0.0, 1.0, 0.0, 1.0]), 4)
    mask_classwise_grad(param, row_mask)
    assert torch.equal(row_mask, torch.tensor([False, True, False, False]))
    assert param.grad[1].abs().sum().item() > 0
    assert param.grad[0].abs().sum().item() == 0
    assert param.grad[2].abs().sum().item() == 0
    assert param.grad[3].abs().sum().item() == 0


def test_cosine_tail_stream_rejects_zero_residual_dead_init():
    try:
        TailResidualStream(
            feature_dim=4,
            num_classes=3,
            dtype=torch.float32,
            init_mode="zero_residual",
        )
    except ValueError as exc:
        assert "cannot use zero_residual" in str(exc)
    else:
        raise AssertionError("zero_residual should be rejected for cosine residual stream")


def test_topology_observer_tracks_exposure_difficulty_survival_and_age():
    observer = TopologyExposureSurvivalObserver(
        num_classes=5,
        exposure_budget=2,
        survival_budget=1,
        warmup_mode="none",
        rho=0.0,
        min_hold=2,
        w_exposure=1.0,
        w_age=1.0,
        w_survival=1.0,
    )
    observer.update(
        exposure_proxy=torch.tensor([10.0, 8.0, 0.1, 0.0, 0.0]),
        difficulty_proxy=torch.tensor([1.0, 1.0, 5.0, 4.0, 3.0]),
        survival_ratio=torch.tensor([1.0, 0.8, 0.2, 1.0, 1.0]),
    )
    gate, scores, mask = observer.compute_gate(current_round=1)
    ids = set(torch.nonzero(mask, as_tuple=False).view(-1).tolist())
    assert int(mask.sum().item()) == 3
    assert 2 in ids
    assert observer.age[3].item() == 1.0
    assert abs(observer.reliability[2].item() - 0.2) < 1e-6
    assert torch.equal(gate > 0, mask)
    assert torch.isfinite(scores).all()


def test_observer_preview_is_pure_and_commit_is_stateful():
    observer = TopologyExposureSurvivalObserver(
        num_classes=4,
        exposure_budget=2,
        survival_budget=0,
        warmup_mode="none",
        rho=0.0,
        min_hold=2,
    )
    observer.update(
        exposure_proxy=torch.tensor([10.0, 1.0, 0.5, 0.2]),
        difficulty_proxy=torch.ones(4),
        survival_ratio=torch.ones(4),
    )
    hold_before = observer.hold_counter.clone()
    life_before = observer.protected_lifetime.clone()
    observer.preview_gate(current_round=1)
    assert torch.equal(observer.hold_counter, hold_before)
    assert torch.equal(observer.protected_lifetime, life_before)
    observer.compute_gate(current_round=1)
    assert torch.equal(observer.hold_counter, hold_before)
    assert torch.equal(observer.protected_lifetime, life_before)

    observer.commit_gate(current_round=1)
    assert observer.protected_mask.any()
    assert observer.protected_lifetime.sum().item() > 0


def test_round_robin_warmup_rotates_with_round_index():
    observer = TopologyExposureSurvivalObserver(
        num_classes=10,
        exposure_budget=3,
        survival_budget=0,
        warmup_mode="round_robin",
        warmup_rounds=5,
        rho=0.0,
    )
    _, _, mask0 = observer.preview_gate(current_round=0)
    _, _, mask1 = observer.preview_gate(current_round=1)
    assert torch.nonzero(mask0, as_tuple=False).view(-1).tolist() == [0, 1, 2]
    assert torch.nonzero(mask1, as_tuple=False).view(-1).tolist() == [3, 4, 5]


def test_oracle_bottom20_selects_fixed_bottom_tail_classes():
    observer = TopologyExposureSurvivalObserver(
        num_classes=100,
        exposure_budget=20,
        survival_budget=10,
        warmup_mode="none",
        oracle_bottom20=True,
        oracle_bottomk=20,
    )
    gate, scores, mask = observer.preview_gate(current_round=0)
    ids = torch.nonzero(mask, as_tuple=False).view(-1).tolist()
    assert ids == list(range(80, 100))
    assert torch.equal(gate > 0, mask)
    assert scores[80:].sum().item() == 20.0
    assert scores[:80].sum().item() == 0.0


def test_tailagg_keeps_absent_and_gate_zero_rows_while_reporting_survival():
    global_state = {
        "tail_stream.weight": torch.zeros(3, 2),
        "tail_stream.bias": torch.zeros(3),
    }
    local_states = {
        0: copy.deepcopy(global_state),
        1: copy.deepcopy(global_state),
    }
    local_states[0]["tail_stream.weight"][2] = torch.tensor([2.0, 0.0])
    local_states[0]["tail_stream.bias"][2] = torch.tensor(2.0)
    local_states[1]["tail_stream.weight"][1] = torch.tensor([9.0, 0.0])

    updated, energy, diagnostics = fedtef_v10_evidence_preserving_tailagg(
        copy.deepcopy(global_state),
        local_states,
        [0, 1],
        gate=torch.tensor([0.0, 0.0, 1.0]),
        num_classes=3,
        survival_ratio=torch.ones(3),
        base_momentum=0.5,
        low_survival_momentum=0.5,
        return_diagnostics=True,
    )
    assert torch.equal(updated["tail_stream.weight"][0], global_state["tail_stream.weight"][0])
    assert torch.equal(updated["tail_stream.weight"][1], global_state["tail_stream.weight"][1])
    assert updated["tail_stream.weight"][2].abs().sum().item() > 0
    assert energy[2].item() > 0
    assert diagnostics["survival_ratio"][2].item() > 0.99


def test_tail_update_stats_detects_client_conflict_survival():
    global_state = {"tail_stream.weight": torch.zeros(2, 2)}
    local_states = {
        0: {"tail_stream.weight": torch.tensor([[1.0, 0.0], [1.0, 0.0]])},
        1: {"tail_stream.weight": torch.tensor([[1.0, 0.0], [-1.0, 0.0]])},
    }
    proxy, observed, survival = compute_tail_update_stats(global_state, local_states, [0, 1], 2)
    assert torch.allclose(proxy, torch.tensor([2.0, 2.0]))
    assert torch.equal(observed, torch.tensor([2.0, 2.0]))
    assert survival[0].item() > 0.99
    assert survival[1].item() < 1e-5


def test_key_groups_keep_shared_and_tail_streams_separate():
    assert is_shared_stream_key("prompt_learner.ctx")
    assert is_shared_stream_key("img_adap.net.0.weight", train_img_adap=True)
    assert not is_shared_stream_key("tail_stream.weight")
    assert is_tail_stream_key("tail_stream.weight")
    assert is_tail_stream_key("routed_prompt_delta")

    global_state = {"prompt_learner.ctx": torch.zeros(1, 2)}
    local_states = {
        0: {"prompt_learner.ctx": torch.ones(1, 2)},
        1: {"prompt_learner.ctx": torch.full((1, 2), 3.0)},
    }
    updated = fedavg_keys(global_state, local_states, [0, 1], [1, 1], ["prompt_learner.ctx"])
    assert torch.allclose(updated["prompt_learner.ctx"], torch.full((1, 2), 2.0))
