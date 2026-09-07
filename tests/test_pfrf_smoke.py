from types import SimpleNamespace

import pytest
import torch

from utils.lora_aggregation import (
    aggregate_effective_lora_state,
    factorize_effective_lora,
    inspect_model_lora_scaling,
)
from utils.pfrf import (
    PFRFClientSession,
    PFRFRuntime,
    PFRFTargetStore,
    additive_target,
    functional_gap,
    max_target,
    capture_rng_state,
    load_pfrf_checkpoint,
    save_pfrf_checkpoint,
    restore_rng_state,
)
from utils.stage3_methods import EvidenceBatch


class LookupModel(torch.nn.Module):
    def __init__(self, logits):
        super().__init__()
        self.logits = torch.nn.Parameter(torch.as_tensor(logits, dtype=torch.float32))

    def forward(self, images):
        return self.logits[: images.shape[0]]


class FakeTrainer:
    def __init__(self, model):
        self.model = model
        self.cleared = None

    def clear_pfrf_session(self, session):
        self.cleared = session


def test_max_state_initializes_raises_and_never_bootstraps_from_upload():
    store = PFRFTargetStore(condition="pfrf_max", global_seed=42, num_classes=2)
    runtime = SimpleNamespace(
        condition="pfrf_max",
        proposal_epochs=2,
        temperature=1.0,
        add_cap=1.0,
        targets=store,
    )
    evidence = EvidenceBatch(
        images=torch.zeros(2, 1), labels=torch.tensor([0, 0]), sample_ids=(1, 2)
    )
    model = LookupModel([[1.0, 0.0], [1.0, 0.0]])
    proposal_probability = float(torch.softmax(model.logits[0].detach(), dim=0)[0])
    session = PFRFClientSession(
        runtime=runtime,
        client_id=3,
        round_id=0,
        evidence=evidence,
        incoming={0: 0.2},
        target_before={0: 0.2},
    )
    assert store.client(3).targets == {}
    assert session.target_before[0] == pytest.approx(0.2)
    assert functional_gap(session.target_before[0], session.incoming[0]) == 0.0
    session.before_local_epoch(model, 2)
    assert store.client(3).targets[0] == pytest.approx(proposal_probability)

    with torch.no_grad():
        model.logits[:, 0] = 9.0
        model.logits[:, 1] = -9.0
    trainer = FakeTrainer(model)
    PFRFRuntime.finalize_client(runtime, trainer, session)
    assert store.client(3).targets[0] == pytest.approx(proposal_probability)
    assert session.upload[0] > proposal_probability
    assert trainer.cleared is session

    # A weaker future CE-only proposal cannot lower the historical target.
    assert max_target(store.client(3).targets[0], 0.45) == pytest.approx(
        proposal_probability
    )


def test_gap_can_disappear_and_reappear_after_absence():
    target = 0.5
    assert functional_gap(target, 0.3) == pytest.approx(0.2)
    assert functional_gap(target, 0.6) == 0.0
    # No update is made while the client is absent in rounds 2--4.
    restored = PFRFTargetStore.from_state_dict(
        {
            "condition": "pfrf_max",
            "global_seed": 42,
            "num_classes": 2,
            "clients": {
                "0": {
                    "targets": {"1": target},
                    "last_participation_round": 0,
                }
            },
        },
        expected_condition="pfrf_max",
        expected_global_seed=42,
        expected_num_classes=2,
    )
    assert restored.client(0).last_participation_round == 0
    assert functional_gap(restored.client(0).targets[1], 0.35) == pytest.approx(0.15)

    # The returning client observes the preserved target and advances only its
    # participation marker; an inferior proposal cannot overwrite H.
    runtime = SimpleNamespace(
        condition="pfrf_max",
        proposal_epochs=2,
        temperature=1.0,
        add_cap=1.0,
        targets=restored,
    )
    evidence = EvidenceBatch(
        images=torch.zeros(1, 1), labels=torch.tensor([1]), sample_ids=(8,)
    )
    returning_model = LookupModel([[0.4, 0.0]])
    returning = PFRFClientSession(
        runtime=runtime,
        client_id=0,
        round_id=4,
        evidence=evidence,
        incoming={1: 0.35},
        target_before={1: restored.client(0).targets[1]},
    )
    returning.before_local_epoch(returning_model, 2)
    assert restored.client(0).targets[1] == pytest.approx(0.5)
    assert restored.client(0).last_participation_round == 4


def test_add_and_max_have_distinct_semantics_and_add_is_capped():
    assert max_target(0.5, 0.5) == pytest.approx(0.5)
    target, old_gap, fresh, discarded = additive_target(0.5, 0.2, 0.5)
    assert (target, old_gap, fresh, discarded) == pytest.approx((0.8, 0.3, 0.3, 0.0))

    target, old_gap, fresh, discarded = additive_target(0.9, 0.1, 0.6)
    assert (target, old_gap, fresh, discarded) == pytest.approx((1.0, 0.8, 0.5, 0.4))


def test_target_hinge_is_active_only_below_frozen_target():
    store = PFRFTargetStore(condition="pfrf_max", global_seed=42, num_classes=2)
    runtime = SimpleNamespace(
        condition="pfrf_max",
        proposal_epochs=2,
        temperature=1.0,
        aux_lambda=1.0,
        add_cap=1.0,
        targets=store,
        client_class_counts={0: torch.tensor([2, 0])},
    )
    evidence = EvidenceBatch(
        images=torch.zeros(2, 1), labels=torch.tensor([0, 0]), sample_ids=(1, 2)
    )
    model = LookupModel([[0.0, 0.0], [0.0, 0.0]])
    session = PFRFClientSession(
        runtime=runtime,
        client_id=0,
        round_id=0,
        evidence=evidence,
        incoming={0: 0.2},
        target_before={0: 0.8},
        proposal={0: 0.5},
        correction_target={0: 0.8},
    )
    active, summary = session.auxiliary_loss(model)
    active.backward()
    assert active.item() > 0
    assert summary["pfrf_active_class_fraction"] == 1.0
    assert model.logits.grad is not None
    assert torch.linalg.vector_norm(model.logits.grad).item() > 0

    model.logits.grad = None
    session.correction_target = {0: 0.5}
    exact, summary = session.auxiliary_loss(model)
    exact.backward()
    assert exact.item() == pytest.approx(0.0, abs=1e-8)
    assert summary["pfrf_active_class_fraction"] == 0.0
    assert torch.count_nonzero(model.logits.grad).item() == 0

    with torch.no_grad():
        model.logits[:, 0] = 9.0
        model.logits[:, 1] = -9.0
    inactive, summary = session.auxiliary_loss(model)
    assert inactive.item() == pytest.approx(0.0, abs=1e-8)
    assert summary["pfrf_active_class_fraction"] == 0.0


def _advance_targets(store, observations):
    state = store.client(0)
    for round_id, (incoming, proposal) in enumerate(observations):
        previous = state.targets.get(0, incoming)
        state.targets[0] = max_target(previous, proposal)
        state.last_participation_round = round_id


def test_target_state_five_rounds_matches_two_plus_resumed_three():
    observations = [(0.2, 0.4), (0.3, 0.5), (0.55, 0.52), (0.45, 0.6), (0.58, 0.57)]
    continuous = PFRFTargetStore(condition="pfrf_max", global_seed=42, num_classes=1)
    _advance_targets(continuous, observations)

    split = PFRFTargetStore(condition="pfrf_max", global_seed=42, num_classes=1)
    _advance_targets(split, observations[:2])
    resumed = PFRFTargetStore.from_state_dict(
        split.state_dict(),
        expected_condition="pfrf_max",
        expected_global_seed=42,
        expected_num_classes=1,
    )
    state = resumed.client(0)
    for round_id, (incoming, proposal) in enumerate(observations[2:], start=2):
        state.targets[0] = max_target(state.targets[0], proposal)
        state.last_participation_round = round_id
    assert resumed.state_dict() == continuous.state_dict()


def test_effective_svd_respects_scaling_rank_and_optimal_tail_error():
    matrix = torch.tensor(
        [[3.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=torch.float64,
    )
    a, b, report = factorize_effective_lora(matrix, rank=2, scaling=0.5)
    reconstructed = 0.5 * (b @ a)
    assert torch.linalg.matrix_rank(reconstructed).item() <= 2
    assert torch.sum((matrix - reconstructed) ** 2).item() == pytest.approx(1.0)
    assert report["tail_singular_energy"] == pytest.approx(1.0)
    assert report["expected_projection_error_squared"] == pytest.approx(1.0)
    assert report["scaling"] == pytest.approx(0.5)


def test_effective_svd_aggregate_is_factorization_invariant_and_preserves_frozen_state():
    scaling = 0.5
    global_state = {
        "frozen": torch.tensor([7.0]),
        "block.w_lora_A": torch.zeros(2, 3),
        "block.w_lora_B": torch.zeros(3, 2),
    }
    a0 = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    b0 = torch.tensor([[2.0, 0.0], [0.0, 1.0], [0.0, 0.0]])
    transform = torch.tensor([[2.0, 0.0], [0.0, 0.5]])
    inverse = torch.linalg.inv(transform)
    local = [
        {"block.w_lora_A": a0, "block.w_lora_B": b0},
        {
            "block.w_lora_A": transform @ a0,
            "block.w_lora_B": b0 @ inverse,
        },
    ]
    aggregated, _ = aggregate_effective_lora_state(
        global_state,
        local,
        [0, 1],
        ["block.w_lora_A", "block.w_lora_B"],
        {0: 0.25, 1: 0.75},
        scaling=scaling,
    )
    assert torch.equal(aggregated["frozen"], global_state["frozen"])
    expected = scaling * (b0 @ a0)
    actual = scaling * (
        aggregated["block.w_lora_B"] @ aggregated["block.w_lora_A"]
    )
    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-6)


def test_effective_svd_single_client_reconstructs_a_nonsymmetric_adapter():
    scaling = 0.25
    local_a = torch.tensor([[1.0, -2.0, 0.5], [0.0, 1.0, 3.0]])
    local_b = torch.tensor(
        [[2.0, 0.0], [-1.0, 0.5], [0.25, 1.0], [3.0, -2.0]]
    )
    global_state = {
        "block.w_lora_A": torch.zeros_like(local_a),
        "block.w_lora_B": torch.zeros_like(local_b),
    }
    aggregated, reports = aggregate_effective_lora_state(
        global_state,
        [{"block.w_lora_A": local_a, "block.w_lora_B": local_b}],
        [0],
        ["block.w_lora_A", "block.w_lora_B"],
        {0: 1.0},
        scaling=scaling,
    )
    expected = scaling * (local_b @ local_a)
    actual = scaling * (
        aggregated["block.w_lora_B"] @ aggregated["block.w_lora_A"]
    )
    assert torch.linalg.matrix_rank(actual).item() <= 2
    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-6)
    assert reports[0]["projection_error_squared"] == pytest.approx(0.0, abs=1e-12)


def test_zero_effective_adapter_has_a_learnable_factor():
    scaling = 0.5
    a, b, report = factorize_effective_lora(
        torch.zeros(4, 3, dtype=torch.float64), rank=2, scaling=scaling
    )
    assert report["numerical_rank"] == 0
    a_parameter = torch.nn.Parameter(a)
    b_parameter = torch.nn.Parameter(b)
    desired = torch.ones(4, 3, dtype=torch.float64)
    loss = torch.sum((scaling * (b_parameter @ a_parameter) - desired) ** 2)
    loss.backward()
    assert torch.count_nonzero(a_parameter).item() > 0
    assert b_parameter.grad is not None
    assert torch.linalg.vector_norm(b_parameter.grad).item() > 0


def test_zero_effective_adapter_can_leave_zero_via_one_ce_step():
    scaling = 0.5
    a, b, _ = factorize_effective_lora(
        torch.zeros(2, 3, dtype=torch.float64), rank=2, scaling=scaling
    )
    a_parameter = torch.nn.Parameter(a)
    b_parameter = torch.nn.Parameter(b)
    optimizer = torch.optim.SGD([a_parameter, b_parameter], lr=0.1)
    images = torch.tensor(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=torch.float64
    )
    labels = torch.tensor([0, 1])
    before = scaling * (b_parameter @ a_parameter)
    assert torch.count_nonzero(before).item() == 0
    loss = torch.nn.functional.cross_entropy(images @ before.T, labels)
    optimizer.zero_grad()
    loss.backward()
    assert torch.linalg.vector_norm(b_parameter.grad).item() > 0
    optimizer.step()
    after = scaling * (b_parameter @ a_parameter)
    assert torch.linalg.vector_norm(after).item() > 0


class _ScaledLoRALinear(torch.nn.Module):
    def __init__(self, scaling):
        super().__init__()
        self.scaling = float(scaling)
        self.w_lora_A = torch.nn.Parameter(torch.ones(2, 3))
        self.w_lora_B = torch.nn.Parameter(torch.zeros(4, 2))


def test_effective_svd_reads_scaling_from_runtime_modules():
    model = torch.nn.Sequential(_ScaledLoRALinear(0.5), _ScaledLoRALinear(0.5))
    scaling, details = inspect_model_lora_scaling(model)
    assert scaling == pytest.approx(0.5)
    assert len(details) == 2
    assert {item["module"] for item in details} == {"0", "1"}

    model[1].scaling = 0.25
    with pytest.raises(ValueError, match="one common LoRA scaling"):
        inspect_model_lora_scaling(model)


def test_checkpoint_saves_only_lora_and_validates_scaling(tmp_path):
    runtime = SimpleNamespace(
        spec=SimpleNamespace(names=("block.w_lora_A", "block.w_lora_B")),
        aggregation_scaling=0.5,
        state_dict=lambda: {"schema_version": "runtime"},
    )
    state = {
        "frozen": torch.ones(100),
        "block.w_lora_A": torch.ones(1, 2),
        "block.w_lora_B": torch.ones(2, 1),
    }
    save_pfrf_checkpoint(
        tmp_path,
        completed_round=2,
        global_state=state,
        runtime=runtime,
        rng_state=capture_rng_state(),
        aggregation_scaling=0.5,
    )
    loaded = load_pfrf_checkpoint(tmp_path, expected_scaling=0.5)
    assert loaded["completed_round"] == 2
    assert set(loaded["global_lora_state"]) == {
        "block.w_lora_A",
        "block.w_lora_B",
    }
    with pytest.raises(ValueError, match="scaling mismatch"):
        load_pfrf_checkpoint(tmp_path, expected_scaling=1.0)


def test_rng_checkpoint_restores_the_next_draws():
    torch.manual_seed(42)
    state = capture_rng_state()
    expected = torch.rand(5)
    _ = torch.rand(17)
    restore_rng_state(state)
    assert torch.equal(torch.rand(5), expected)
