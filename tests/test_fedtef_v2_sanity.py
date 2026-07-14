import copy
import os
import sys
from types import SimpleNamespace

import torch
import torch.nn as nn

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from trainers.fedtef_v2_utils import (
    CosineResidualTailExpert,
    EvidenceMemoryTracker,
    ExposureTracker,
    GradientPriorTracker,
    LowExposureRouterTracker,
    SemanticResidualMemoryExpert,
    TailResidualExpert,
    TailNeedTracker,
    TopologyExposureSurvivalObserver,
    compute_controlled_hard_negative_loss,
    compute_tail_stream_positive_update_stats,
    compute_tail_stream_gradient_prior_proxy,
    fedavg_keys,
    fedtef_v2_tailagg,
    fedtef_v10_evidence_preserving_tailagg,
    is_shared_stream_key,
    is_tail_stream_key,
)
from utils.loralib.layers import PlainMultiheadAttentionLoRA
from utils.loralib.utils import apply_lora


def _fuse(logits_base, residual_tail, gate, fusion_lambda):
    return logits_base + fusion_lambda * gate.unsqueeze(0) * residual_tail


def test_zero_residual_initialization_equivalent_to_base():
    expert = TailResidualExpert(feature_dim=4, num_classes=3, dtype=torch.float32, hidden_dim=5)
    features = torch.randn(2, 4)
    logits_base = torch.randn(2, 3)
    gate = torch.rand(3)
    residual = expert(features)
    logits_fused = _fuse(logits_base, residual, gate, fusion_lambda=0.3)
    assert torch.max(torch.abs(residual)).item() == 0.0
    assert torch.allclose(logits_fused, logits_base, atol=1e-7)


def test_lambda_or_gate_zero_equivalent_to_base():
    logits_base = torch.randn(2, 3)
    residual_tail = torch.randn(2, 3)
    gate = torch.rand(3)
    assert torch.allclose(_fuse(logits_base, residual_tail, gate, 0.0), logits_base)
    assert torch.allclose(_fuse(logits_base, residual_tail, torch.zeros(3), 0.3), logits_base)


def test_cosine_tail_stream_can_be_zero_or_normal_residual():
    features = torch.randn(2, 4)
    zero_expert = CosineResidualTailExpert(
        feature_dim=4,
        num_classes=3,
        dtype=torch.float32,
        init_mode="zero_residual",
    )
    assert torch.allclose(zero_expert(features), torch.zeros(2, 3), atol=1e-7)

    normal_expert = CosineResidualTailExpert(
        feature_dim=4,
        num_classes=3,
        dtype=torch.float32,
        init_mode="normal_residual",
    )
    assert normal_expert.weight.shape == (3, 4)
    assert normal_expert(features).shape == (2, 3)


def test_semantic_memory_stream_starts_as_exact_zero_residual():
    expert = SemanticResidualMemoryExpert(
        feature_dim=4,
        num_classes=3,
        dtype=torch.float32,
    )
    features = torch.randn(2, 4)
    text_features = torch.randn(3, 4)
    residual = expert(features, text_features=text_features)
    assert torch.allclose(residual, torch.zeros_like(residual), atol=1e-7)

    with torch.no_grad():
        expert.memory[1, 0] = 0.5
    changed = expert(features, text_features=text_features)
    assert not torch.allclose(changed[:, 1], torch.zeros_like(changed[:, 1]))


def test_round_robin_warmup_uses_one_consistent_gate_and_mask():
    tracker = ExposureTracker(
        num_classes=10,
        tail_topk=3,
        warmup_mode="round_robin",
        warmup_rounds=5,
        seed=1,
    )
    gate, scores, mask = tracker.compute_gate(current_round=2)
    expected = [6, 7, 8]
    assert torch.nonzero(mask, as_tuple=False).view(-1).tolist() == expected
    assert torch.nonzero(gate, as_tuple=False).view(-1).tolist() == expected
    assert torch.nonzero(scores, as_tuple=False).view(-1).tolist() == expected


def test_oracle_bottom20_only_when_explicitly_enabled():
    tracker = ExposureTracker(
        num_classes=100,
        tail_topk=20,
        warmup_mode="oracle_bottom20",
        warmup_rounds=5,
        seed=1,
        dataset_name="cifar100_LT",
    )
    gate, _, mask = tracker.compute_gate(current_round=0)
    ids = torch.nonzero(mask, as_tuple=False).view(-1).tolist()
    assert ids == list(range(80, 100))
    assert torch.nonzero(gate, as_tuple=False).view(-1).tolist() == list(range(80, 100))


def test_default_tied_dynamic_gate_does_not_pick_head_prefix():
    tracker = ExposureTracker(
        num_classes=100,
        tail_topk=20,
        round0_tie_break="random",
        warmup_mode="none",
        seed=1,
        dataset_name="cifar100_LT",
    )
    _, _, mask = tracker.compute_gate(current_round=0)
    ids = torch.nonzero(mask, as_tuple=False).view(-1).tolist()
    assert ids != list(range(20))


def test_opportunity_count_changes_exposure_score():
    tracker = ExposureTracker(num_classes=4, tail_topk=2, warmup_mode="none")
    _, scores_before, _ = tracker.compute_gate(current_round=10)
    tracker.update_from_energy(torch.zeros(4), gate=torch.tensor([1.0, 1.0, 0.0, 0.0]))
    scores_after = tracker.compute_scores()
    assert tracker.opportunity_count.tolist() == [1.0, 1.0, 0.0, 0.0]
    assert scores_after[0].item() < scores_before[0].item()
    assert scores_after[2].item() > scores_after[0].item()


def test_soft_gate_is_sparse_and_matches_protected_mask():
    tracker = ExposureTracker(
        num_classes=6,
        tail_topk=2,
        gate_mode="soft",
        warmup_mode="none",
        seed=1,
    )
    tracker.exposure = torch.tensor([10.0, 9.0, 1.0, 0.5, 8.0, 7.0])
    gate, _, mask = tracker.compute_gate(current_round=10)
    assert torch.equal(gate > 0, mask)
    assert int((gate > 0).sum().item()) == 2

    tracker.update_from_energy(torch.zeros(6), gate=gate)
    assert torch.equal(tracker.opportunity_count > 0, mask)


def test_tail_need_gate_has_min_hold_persistence():
    tracker = TailNeedTracker(
        num_classes=6,
        tail_topk=2,
        gate_mode="hard_topk",
        warmup_mode="none",
        min_hold=3,
        beta=0.5,
        w_scarcity=0.0,
        w_residual=1.0,
    )
    tracker.update_from_energy(torch.tensor([5.0, 4.0, 0.0, 0.0, 0.0, 0.0]))
    _, _, mask0 = tracker.compute_gate(current_round=1)
    ids0 = set(torch.nonzero(mask0, as_tuple=False).view(-1).tolist())
    assert ids0 == {0, 1}

    tracker.update_from_energy(torch.tensor([0.0, 0.0, 5.0, 4.0, 0.0, 0.0]))
    _, _, mask1 = tracker.compute_gate(current_round=2)
    ids1 = set(torch.nonzero(mask1, as_tuple=False).view(-1).tolist())
    assert {0, 1}.issubset(ids1)
    assert tracker.last_jaccard > 0.0


def test_gradient_prior_gate_inverts_positive_row_update_proxy():
    tracker = GradientPriorTracker(
        num_classes=5,
        tail_topk=2,
        gate_mode="hard_topk",
        warmup_mode="none",
        rho=0.0,
        prior_floor=1e-3,
    )
    tracker.update_from_gradient_proxy(
        torch.tensor([10.0, 8.0, 1.0, 0.5, 0.2]),
        gate=torch.ones(5),
    )
    gate, scores, mask = tracker.compute_gate(current_round=1)
    ids = set(torch.nonzero(mask, as_tuple=False).view(-1).tolist())
    assert ids == {3, 4}
    assert scores[4].item() > scores[0].item()
    assert tracker.class_prior_proxy[0].item() > tracker.class_prior_proxy[4].item()
    assert torch.equal(gate > 0, mask)


def test_low_exposure_router_uses_all_positive_row_proxy_for_tail_discovery():
    tracker = LowExposureRouterTracker(
        num_classes=100,
        tail_topk=30,
        gate_mode="hard_topk",
        warmup_mode="none",
        rho=0.0,
        prior_floor=1e-3,
        score_power=1.0,
        update_all_rows=True,
    )
    proxy = torch.zeros(100)
    proxy[:80] = 1.0
    proxy[80:] = 0.01
    tracker.update_from_gradient_proxy(proxy, gate=torch.zeros(100))
    gate, scores, mask = tracker.compute_gate(current_round=1)
    assert tracker.score_mode == "low_exposure_router"
    assert int(mask.sum().item()) == 30
    assert int(mask[80:].sum().item()) == 20
    assert int(mask[:80].sum().item()) == 10
    assert scores[80].item() > scores[0].item()
    assert torch.equal(gate > 0, mask)


def test_topology_observer_tracks_exposure_difficulty_survival_and_age():
    observer = TopologyExposureSurvivalObserver(
        num_classes=5,
        tail_topk=3,
        exposure_budget=2,
        survival_budget=1,
        warmup_mode="none",
        gate_mode="hard_topk",
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
    assert observer.score_mode == "topology_observer"
    assert int(mask.sum().item()) == 3
    assert 2 in ids
    assert observer.age[3].item() == 1.0
    assert observer.age[0].item() == 0.0
    assert abs(observer.reliability[2].item() - 0.2) < 1e-6
    assert torch.equal(gate > 0, mask)

    observer.update(
        exposure_proxy=torch.tensor([10.0, 8.0, 0.1, 20.0, 20.0]),
        difficulty_proxy=torch.tensor([1.0, 1.0, 5.0, 4.0, 3.0]),
        survival_ratio=torch.ones(5),
    )
    _, _, mask_next = observer.compute_gate(current_round=2)
    assert ids.intersection(set(torch.nonzero(mask_next, as_tuple=False).view(-1).tolist()))


def test_topology_observer_replace_margin_blocks_weak_replacement():
    observer = TopologyExposureSurvivalObserver(
        num_classes=4,
        tail_topk=2,
        exposure_budget=2,
        survival_budget=0,
        warmup_mode="none",
        gate_mode="hard_topk",
        min_hold=0,
        replace_margin=1.5,
    )
    observer.topology_score = torch.tensor([10.0, 9.0, 13.0, 1.0])
    observer.protected_mask = torch.tensor([True, True, False, False])
    proposed = torch.tensor([True, False, True, False])
    selected = observer._apply_hysteresis(proposed)
    assert set(torch.nonzero(selected, as_tuple=False).view(-1).tolist()) == {0, 1}

    observer.topology_score = torch.tensor([10.0, 9.0, 14.0, 1.0])
    observer.protected_mask = torch.tensor([True, True, False, False])
    observer.hold_counter.zero_()
    selected = observer._apply_hysteresis(proposed)
    assert set(torch.nonzero(selected, as_tuple=False).view(-1).tolist()) == {0, 2}


def test_tail_stream_positive_update_stats_compute_survival_ratio():
    global_weights = {"tail_stream.weight": torch.zeros(2, 2)}
    local_weights = {
        0: {"tail_stream.weight": torch.tensor([[1.0, 0.0], [1.0, 0.0]])},
        1: {"tail_stream.weight": torch.tensor([[1.0, 0.0], [-1.0, 0.0]])},
    }
    proxy, observed, survival = compute_tail_stream_positive_update_stats(
        global_weights,
        local_weights,
        [0, 1],
        num_classes=2,
    )
    assert torch.allclose(proxy, torch.tensor([2.0, 2.0]))
    assert torch.equal(observed, torch.tensor([2.0, 2.0]))
    assert survival[0].item() > 0.99
    assert survival[1].item() < 1e-5


def test_gradient_prior_lock_holds_then_refreshes():
    tracker = GradientPriorTracker(
        num_classes=5,
        tail_topk=2,
        gate_mode="hard_topk",
        warmup_mode="none",
        rho=0.0,
        prior_floor=1e-3,
        lock_rounds=3,
        lock_mode="full_refresh",
    )
    tracker.class_prior_ema = torch.tensor([10.0, 8.0, 5.0, 0.5, 0.2])
    _, _, mask0 = tracker.compute_gate(current_round=1)
    ids0 = set(torch.nonzero(mask0, as_tuple=False).view(-1).tolist())
    assert ids0 == {3, 4}
    assert tracker.last_lock_active
    assert tracker.locked_until_round == 4

    tracker.class_prior_ema = torch.tensor([0.1, 0.2, 10.0, 9.0, 8.0])
    _, _, mask1 = tracker.compute_gate(current_round=2)
    ids1 = set(torch.nonzero(mask1, as_tuple=False).view(-1).tolist())
    assert ids1 == {3, 4}
    assert tracker.last_lock_active

    _, _, mask2 = tracker.compute_gate(current_round=4)
    ids2 = set(torch.nonzero(mask2, as_tuple=False).view(-1).tolist())
    assert ids2 == {0, 1}
    assert tracker.lock_source_round == 4
    assert tracker.locked_until_round == 7


def test_gradient_prior_anchor_refine_limits_refresh_churn():
    tracker = GradientPriorTracker(
        num_classes=6,
        tail_topk=3,
        gate_mode="hard_topk",
        warmup_mode="none",
        rho=0.0,
        prior_floor=1e-3,
        lock_rounds=3,
        lock_mode="anchor_refine",
        refine_max_swap=1,
        refine_margin=1.0,
    )
    tracker.class_prior_ema = torch.tensor([10.0, 9.0, 8.0, 0.4, 0.3, 0.2])
    _, _, mask0 = tracker.compute_gate(current_round=1)
    ids0 = set(torch.nonzero(mask0, as_tuple=False).view(-1).tolist())
    assert ids0 == {3, 4, 5}

    tracker.class_prior_ema = torch.tensor([0.1, 0.2, 0.3, 8.0, 9.0, 10.0])
    _, _, mask1 = tracker.compute_gate(current_round=4)
    ids1 = set(torch.nonzero(mask1, as_tuple=False).view(-1).tolist())
    assert len(ids0 - ids1) == 1
    assert len(ids1 - ids0) == 1
    assert tracker.last_refine_swaps == 1
    assert tracker.lock_source_round == 4


def test_gradient_prior_anchor_refine_margin_can_keep_anchor():
    tracker = GradientPriorTracker(
        num_classes=5,
        tail_topk=2,
        gate_mode="hard_topk",
        warmup_mode="none",
        rho=0.0,
        prior_floor=1e-3,
        lock_rounds=2,
        lock_mode="anchor_refine",
        refine_max_swap=2,
        refine_margin=10.0,
    )
    tracker.class_prior_ema = torch.tensor([10.0, 9.0, 8.0, 0.5, 0.4])
    _, _, mask0 = tracker.compute_gate(current_round=1)
    ids0 = set(torch.nonzero(mask0, as_tuple=False).view(-1).tolist())
    assert ids0 == {3, 4}

    tracker.class_prior_ema = torch.tensor([0.35, 0.36, 10.0, 0.5, 0.4])
    _, _, mask1 = tracker.compute_gate(current_round=3)
    ids1 = set(torch.nonzero(mask1, as_tuple=False).view(-1).tolist())
    assert ids1 == ids0
    assert tracker.last_refine_swaps == 0


def test_gradient_prior_proxy_uses_tail_stream_row_updates_only():
    global_weights = {
        "prompt_learner.ctx": torch.zeros(1, 2),
        "tail_stream.weight": torch.zeros(3, 2),
        "tail_stream.bias": torch.zeros(3),
        "tail_stream.logit_scale": torch.zeros(()),
    }
    local_weights = {
        0: copy.deepcopy(global_weights),
        1: copy.deepcopy(global_weights),
    }
    local_weights[0]["prompt_learner.ctx"] = torch.ones(1, 2) * 100.0
    local_weights[0]["tail_stream.weight"][0] = torch.tensor([3.0, 4.0])
    local_weights[0]["tail_stream.bias"][0] = torch.tensor(2.0)
    local_weights[1]["tail_stream.weight"][2] = torch.tensor([0.0, 6.0])

    proxy, observed = compute_tail_stream_gradient_prior_proxy(
        global_weights,
        local_weights,
        [0, 1],
        num_classes=3,
    )
    assert torch.allclose(proxy, torch.tensor([7.0, 0.0, 6.0]))
    assert torch.equal(observed, torch.tensor([1.0, 0.0, 1.0]))


def test_routed_prompt_is_tailagg_protected_but_not_prior_proxy_source():
    assert is_tail_stream_key("routed_prompt_delta")
    global_weights = {
        "tail_stream.weight": torch.zeros(3, 2),
        "routed_prompt_delta": torch.zeros(3, 2, 2),
    }
    local_weights = {
        0: copy.deepcopy(global_weights),
        1: copy.deepcopy(global_weights),
    }
    local_weights[0]["routed_prompt_delta"][2] = torch.ones(2, 2) * 10.0
    proxy, observed = compute_tail_stream_gradient_prior_proxy(
        global_weights,
        local_weights,
        [0, 1],
        num_classes=3,
    )
    assert torch.equal(proxy, torch.zeros(3))
    assert torch.equal(observed, torch.zeros(3))

    updated, energy = fedtef_v2_tailagg(
        copy.deepcopy(global_weights),
        local_weights,
        [0, 1],
        [1, 1],
        gate=torch.tensor([0.0, 0.0, 1.0]),
        num_classes=3,
    )
    assert updated["routed_prompt_delta"][2].abs().sum().item() > 0.0
    assert updated["routed_prompt_delta"][0].abs().sum().item() == 0.0
    assert energy[2].item() > 0.0


def test_evidence_memory_gate_is_sparse_and_prioritizes_sparse_evidence():
    tracker = EvidenceMemoryTracker(
        num_classes=3,
        rho=0.5,
        tail_topk=1,
        warmup_mode="none",
        gate_floor=0.05,
        residual_weight=0.0,
    )
    tracker.update_from_evidence(torch.tensor([10.0, 1.0, 0.0]))
    tracker.update_from_evidence(torch.tensor([10.0, 0.0, 0.0]))
    tracker.update_from_evidence(torch.tensor([10.0, 0.0, 0.0]))
    gate, scores, mask = tracker.compute_gate(current_round=4)
    assert mask.sum().item() == 1
    assert torch.equal(mask, gate > 0)
    assert scores[1].item() > scores[0].item()
    assert mask[1].item()
    assert gate[1].item() > 0
    assert gate[0].item() == 0
    assert gate[2].item() == 0


def test_evidence_memory_tailagg_updates_observed_rows_and_retains_absent_rows():
    global_weights = {
        "tail_stream.memory": torch.zeros(3, 2),
        "tail_stream.bias": torch.zeros(3),
    }
    local_weights = {
        0: copy.deepcopy(global_weights),
        1: copy.deepcopy(global_weights),
    }
    local_weights[0]["tail_stream.memory"][2] = torch.tensor([2.0, 0.0])
    local_weights[0]["tail_stream.bias"][2] = torch.tensor(2.0)
    updated, energy = fedtef_v2_tailagg(
        copy.deepcopy(global_weights),
        local_weights,
        [0, 1],
        [1, 1],
        gate=torch.zeros(3),
        num_classes=3,
        mode="evidence_memory",
        memory_momentum=0.5,
    )
    assert torch.allclose(updated["tail_stream.memory"][2], torch.tensor([1.0, 0.0]), atol=1e-5)
    assert torch.allclose(updated["tail_stream.bias"][2], torch.tensor(1.0), atol=1e-5)
    assert energy[2].item() > 0.0

    absent_local = {0: copy.deepcopy(updated), 1: copy.deepcopy(updated)}
    retained, _ = fedtef_v2_tailagg(
        copy.deepcopy(updated),
        absent_local,
        [0, 1],
        [1, 1],
        gate=torch.zeros(3),
        num_classes=3,
        mode="evidence_memory",
        memory_momentum=0.5,
    )
    assert torch.equal(retained["tail_stream.memory"], updated["tail_stream.memory"])
    assert torch.equal(retained["tail_stream.bias"], updated["tail_stream.bias"])


def test_v10_evidence_preserving_tailagg_keeps_absent_rows_and_updates_observed_only():
    global_weights = {
        "tail_stream.weight": torch.zeros(3, 2),
        "tail_stream.bias": torch.zeros(3),
    }
    local_weights = {
        0: copy.deepcopy(global_weights),
        1: copy.deepcopy(global_weights),
    }
    local_weights[0]["tail_stream.weight"][2] = torch.tensor([2.0, 0.0])
    local_weights[0]["tail_stream.bias"][2] = torch.tensor(2.0)
    updated, energy, diagnostics = fedtef_v10_evidence_preserving_tailagg(
        copy.deepcopy(global_weights),
        local_weights,
        [0, 1],
        gate=torch.ones(3),
        num_classes=3,
        survival_ratio=torch.tensor([1.0, 1.0, 1.0]),
        base_momentum=0.5,
        low_survival_momentum=0.5,
        return_diagnostics=True,
    )
    assert torch.equal(updated["tail_stream.weight"][0], global_weights["tail_stream.weight"][0])
    assert torch.allclose(updated["tail_stream.weight"][2], torch.tensor([1.0, 0.0]))
    assert diagnostics["mode"] == "evidence_preserving"
    assert diagnostics["updated_rows"] > 0
    assert diagnostics["updated_row_tensors"] == diagnostics["updated_rows"]
    assert diagnostics["updated_classes"] == 1
    assert energy[2].item() > 0


def test_v10_hard_negative_loss_uses_protected_samples_only():
    logits_base = torch.tensor([[0.0, 2.0, 1.0], [3.0, 0.0, 1.0]])
    residual = torch.zeros_like(logits_base)
    residual[0, 0] = 4.0
    labels = torch.tensor([0, 1])
    loss = compute_controlled_hard_negative_loss(
        logits_base,
        residual,
        labels,
        protected_label=torch.tensor([True, False]),
        topm=1,
        residual_lambda=1.0,
    )
    assert loss.item() < 0.2
    zero_loss = compute_controlled_hard_negative_loss(
        logits_base,
        residual,
        labels,
        protected_label=torch.tensor([False, False]),
        topm=1,
        residual_lambda=1.0,
    )
    assert zero_loss.item() == 0.0


def test_v10_hard_negative_loss_sanitizes_nonfinite_residuals():
    logits_base = torch.tensor([[0.0, 2.0, 1.0], [3.0, 0.0, 1.0]])
    residual = torch.zeros_like(logits_base)
    residual[0, 0] = float("nan")
    residual[0, 1] = float("inf")
    labels = torch.tensor([0, 1])
    loss = compute_controlled_hard_negative_loss(
        logits_base,
        residual,
        labels,
        protected_label=torch.tensor([True, False]),
        topm=1,
        residual_lambda=1.0,
    )
    assert torch.isfinite(loss)


def test_tailagg_diagnostics_compare_fedavg_and_memory_retention():
    global_weights = {
        "tail_stream.memory": torch.zeros(3, 2),
        "tail_stream.bias": torch.zeros(3),
    }
    local_weights = {
        0: copy.deepcopy(global_weights),
        1: copy.deepcopy(global_weights),
    }
    local_weights[0]["tail_stream.memory"][2] = torch.tensor([2.0, 0.0])
    local_weights[1]["tail_stream.memory"][2] = torch.tensor([1.0, 0.0])
    updated, energy, diagnostics = fedtef_v2_tailagg(
        copy.deepcopy(global_weights),
        local_weights,
        [0, 1],
        [1, 1],
        gate=torch.ones(3),
        num_classes=3,
        mode="evidence_memory",
        memory_momentum=0.5,
        return_diagnostics=True,
    )
    assert diagnostics["mode"] == "evidence_memory"
    assert diagnostics["observed_client_count"][2].item() == 2.0
    assert diagnostics["local_energy_sum"][2].item() > 0.0
    assert diagnostics["fedavg_row_energy"][2].item() > 0.0
    assert diagnostics["tailagg_row_energy"][2].item() > 0.0
    assert diagnostics["memory_row_norm"][2].item() > 0.0
    assert energy[2].item() > 0.0
    assert updated["tail_stream.memory"][2].norm().item() > 0.0


def test_tailagg_is_separate_from_shared_fedavg():
    global_weights = {
        "prompt_learner.ctx": torch.zeros(1, 2),
        "tail_stream.fc2.weight": torch.zeros(3, 2),
        "tail_stream.fc2.bias": torch.zeros(3),
    }
    local_weights = {
        0: copy.deepcopy(global_weights),
        1: copy.deepcopy(global_weights),
    }
    local_weights[0]["prompt_learner.ctx"] = torch.ones(1, 2)
    local_weights[1]["prompt_learner.ctx"] = torch.full((1, 2), 3.0)
    local_weights[0]["tail_stream.fc2.weight"][2] = torch.tensor([1.0, 0.0])
    local_weights[1]["tail_stream.fc2.weight"][2] = torch.tensor([3.0, 0.0])
    local_weights[0]["tail_stream.fc2.bias"][2] = torch.tensor(1.0)
    local_weights[1]["tail_stream.fc2.bias"][2] = torch.tensor(3.0)

    idxs_users = [0, 1]
    datanumber_client = [1, 1]
    shared = fedavg_keys(
        copy.deepcopy(global_weights),
        local_weights,
        idxs_users,
        datanumber_client,
        ["prompt_learner.ctx"],
    )
    assert torch.allclose(shared["prompt_learner.ctx"], torch.full((1, 2), 2.0))

    tailagg, energy = fedtef_v2_tailagg(
        copy.deepcopy(global_weights),
        local_weights,
        idxs_users,
        datanumber_client,
        gate=torch.tensor([0.0, 0.0, 1.0]),
        num_classes=3,
    )
    fedavg_tail = fedavg_keys(
        copy.deepcopy(global_weights),
        local_weights,
        idxs_users,
        datanumber_client,
        ["tail_stream.fc2.weight", "tail_stream.fc2.bias"],
    )
    assert not torch.allclose(
        tailagg["tail_stream.fc2.weight"][2],
        fedavg_tail["tail_stream.fc2.weight"][2],
    )
    assert energy[2].item() > 0.0


def test_lora_parameters_join_shared_fedavg_only_when_enabled():
    key = "image_encoder.transformer.resblocks.11.attn.q_proj.w_lora_A"
    assert not is_shared_stream_key(key, train_img_adap=False, train_lora=False)
    assert is_shared_stream_key(key, train_img_adap=False, train_lora=True)
    assert not key.startswith("tail_stream.")


class _DummyResidualBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=4, num_heads=1)


class _DummyTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.resblocks = nn.ModuleList([_DummyResidualBlock() for _ in range(12)])


class _DummyClip(nn.Module):
    def __init__(self):
        super().__init__()
        self.transformer = _DummyTransformer()
        self.visual = SimpleNamespace(transformer=_DummyTransformer())


def test_apply_lora_can_target_visual_stream_without_touching_text():
    cfg = SimpleNamespace(
        TRAINER=SimpleNamespace(
            CLIPLORA=SimpleNamespace(
                encoder="vision",
                position="top2",
                backbone="ViT-B/16",
                params=["q", "v"],
                r=2,
                alpha=1,
                dropout_rate=0.0,
            )
        )
    )
    clip_model = _DummyClip()
    layers = apply_lora(cfg, clip_model)
    assert len(layers) == 2
    assert isinstance(clip_model.transformer.resblocks[11].attn, nn.MultiheadAttention)
    assert isinstance(clip_model.visual.transformer.resblocks[9].attn, nn.MultiheadAttention)
    assert isinstance(clip_model.visual.transformer.resblocks[10].attn, PlainMultiheadAttentionLoRA)
    assert isinstance(clip_model.visual.transformer.resblocks[11].attn, PlainMultiheadAttentionLoRA)
    assert hasattr(clip_model.visual.transformer.resblocks[11].attn.q_proj, "w_lora_A")
    assert hasattr(clip_model.visual.transformer.resblocks[11].attn.v_proj, "w_lora_A")


def test_lora_attention_replacement_starts_equivalent_to_base_attention():
    torch.manual_seed(7)
    base_attention = nn.MultiheadAttention(embed_dim=4, num_heads=1, dropout=0.0)
    lora_attention = PlainMultiheadAttentionLoRA(
        copy.deepcopy(base_attention),
        enable_lora=["q", "v"],
        r=2,
        lora_alpha=1,
        dropout_rate=0.0,
    )
    base_attention.eval()
    lora_attention.eval()
    features = torch.randn(3, 2, 4)
    base_output, _ = base_attention(features, features, features, need_weights=False)
    lora_output, _ = lora_attention(features, features, features, need_weights=False)
    assert torch.allclose(lora_output, base_output, atol=1e-6)


if __name__ == "__main__":
    test_zero_residual_initialization_equivalent_to_base()
    test_lambda_or_gate_zero_equivalent_to_base()
    test_cosine_tail_stream_can_be_zero_or_normal_residual()
    test_semantic_memory_stream_starts_as_exact_zero_residual()
    test_round_robin_warmup_uses_one_consistent_gate_and_mask()
    test_oracle_bottom20_only_when_explicitly_enabled()
    test_default_tied_dynamic_gate_does_not_pick_head_prefix()
    test_opportunity_count_changes_exposure_score()
    test_soft_gate_is_sparse_and_matches_protected_mask()
    test_tail_need_gate_has_min_hold_persistence()
    test_gradient_prior_gate_inverts_positive_row_update_proxy()
    test_topology_observer_tracks_exposure_difficulty_survival_and_age()
    test_topology_observer_replace_margin_blocks_weak_replacement()
    test_tail_stream_positive_update_stats_compute_survival_ratio()
    test_gradient_prior_lock_holds_then_refreshes()
    test_gradient_prior_anchor_refine_limits_refresh_churn()
    test_gradient_prior_anchor_refine_margin_can_keep_anchor()
    test_gradient_prior_proxy_uses_tail_stream_row_updates_only()
    test_evidence_memory_gate_is_sparse_and_prioritizes_sparse_evidence()
    test_evidence_memory_tailagg_updates_observed_rows_and_retains_absent_rows()
    test_v10_evidence_preserving_tailagg_keeps_absent_rows_and_updates_observed_only()
    test_v10_hard_negative_loss_uses_protected_samples_only()
    test_v10_hard_negative_loss_sanitizes_nonfinite_residuals()
    test_tailagg_diagnostics_compare_fedavg_and_memory_retention()
    test_tailagg_is_separate_from_shared_fedavg()
    test_lora_parameters_join_shared_fedavg_only_when_enabled()
    test_apply_lora_can_target_visual_stream_without_touching_text()
    test_lora_attention_replacement_starts_equivalent_to_base_attention()
    print("FedTEF-v2 sanity checks passed")
