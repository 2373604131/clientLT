import copy
import random

import numpy as np
import pytest
import torch
from torch import nn

from utils.stage3_methods import (
    EvidenceBatch,
    compute_restore_direction,
    evaluate_and_select_proposals,
    evaluate_proposals_for_audit,
)
from utils.stage3_private_state import IncomingGlobalReference
from utils.stage3_proposals import (
    ClientProposalPayload,
    MixedProposal,
    PROPOSAL_BANK_SCHEMA,
)
from utils.stage3_vectors import build_model_lora_flat_spec, flatten_model


class ToyPrivateClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.image_encoder = nn.Module()
        self.image_encoder.q_lora_A = nn.Parameter(torch.zeros(2, 2))
        self.bn = nn.BatchNorm1d(2)
        self.bn.weight.requires_grad = False
        self.bn.bias.requires_grad = False

    def forward(self, images):
        return self.bn(images) @ self.image_encoder.q_lora_A


def _batch():
    return EvidenceBatch(
        images=torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
        labels=torch.tensor([0, 1]),
        sample_ids=(10, 11),
    )


def _proposal(proposal_id, matrix, spec, source_count=4):
    return MixedProposal(
        proposal_id=proposal_id,
        vector=torch.as_tensor(matrix, dtype=torch.float32).reshape(-1),
        source_count=source_count,
        source_round=0,
        spec_hash=spec.spec_hash,
    )


def _payload(spec, proposals):
    return ClientProposalPayload(
        schema_version=PROPOSAL_BANK_SCHEMA,
        condition="p_fcc_only",
        target_round=1,
        spec_hash=spec.spec_hash,
        proposals=tuple(proposals),
    )


def _rng_snapshot():
    return (
        random.getstate(),
        np.random.get_state(),
        torch.random.get_rng_state().clone(),
    )


def _assert_rng_equal(left, right):
    assert left[0] == right[0]
    assert left[1][0] == right[1][0]
    assert np.array_equal(left[1][1], right[1][1])
    assert left[1][2:] == right[1][2:]
    assert torch.equal(left[2], right[2])


def test_private_p_fcc_selects_only_positive_top_proposals():
    model = ToyPrivateClassifier()
    model.train()
    spec = build_model_lora_flat_spec(model)
    incoming = flatten_model(model, spec).cpu()
    positive = _proposal(2, [[1.0, -1.0], [-1.0, 1.0]], spec)
    negative = _proposal(1, [[-1.0, 1.0], [1.0, -1.0]], spec)
    payload = _payload(spec, [negative, positive])

    result = evaluate_and_select_proposals(
        model,
        spec,
        incoming,
        _batch(),
        payload,
        mode="private",
        global_seed=42,
        round_id=1,
        client_id=3,
    )

    utilities = {probe.proposal_id: probe.utility for probe in result.probes}
    assert utilities[2] > 0
    assert utilities[1] < 0
    assert result.selected_proposal_ids == (2,)
    assert result.selected_weights == pytest.approx((1.0,))
    assert torch.equal(result.direction, positive.vector)
    assert result.forward_count == 3


def test_probe_order_does_not_change_utility_or_mutate_state_rng_or_gradients():
    random.seed(8)
    np.random.seed(8)
    torch.manual_seed(8)
    model = ToyPrivateClassifier()
    model.train()
    spec = build_model_lora_flat_spec(model)
    incoming = flatten_model(model, spec).cpu()
    model.image_encoder.q_lora_A.grad = torch.full_like(
        model.image_encoder.q_lora_A, 7.0
    )
    parameter_before = incoming.clone()
    gradient_before = model.image_encoder.q_lora_A.grad.clone()
    buffers_before = {
        name: value.clone() for name, value in model.named_buffers()
    }
    rng_before = _rng_snapshot()
    proposals = [
        _proposal(0, [[1.0, -1.0], [-1.0, 1.0]], spec),
        _proposal(1, [[-1.0, 1.0], [1.0, -1.0]], spec),
    ]

    forward = evaluate_and_select_proposals(
        model,
        spec,
        incoming,
        _batch(),
        _payload(spec, proposals),
        mode="private",
        global_seed=42,
        round_id=1,
        client_id=3,
    )
    reverse = evaluate_and_select_proposals(
        model,
        spec,
        incoming,
        _batch(),
        _payload(spec, list(reversed(proposals))),
        mode="private",
        global_seed=42,
        round_id=1,
        client_id=3,
    )

    assert [(item.proposal_id, item.utility) for item in forward.probes] == pytest.approx(
        [(item.proposal_id, item.utility) for item in reverse.probes]
    )
    assert torch.equal(flatten_model(model, spec).cpu(), parameter_before)
    assert torch.equal(model.image_encoder.q_lora_A.grad, gradient_before)
    assert model.training is True
    assert all(
        torch.equal(dict(model.named_buffers())[name], value)
        for name, value in buffers_before.items()
    )
    _assert_rng_equal(_rng_snapshot(), rng_before)


def test_random_proposal_evaluates_all_then_selects_deterministically():
    model = ToyPrivateClassifier()
    spec = build_model_lora_flat_spec(model)
    incoming = flatten_model(model, spec).cpu()
    proposals = [
        _proposal(0, [[1.0, -1.0], [-1.0, 1.0]], spec),
        _proposal(1, [[-1.0, 1.0], [1.0, -1.0]], spec),
        _proposal(2, [[0.5, -0.5], [-0.5, 0.5]], spec),
    ]
    payload = _payload(spec, proposals)

    first = evaluate_and_select_proposals(
        model,
        spec,
        incoming,
        _batch(),
        payload,
        mode="random",
        global_seed=42,
        round_id=1,
        client_id=9,
    )
    second = evaluate_and_select_proposals(
        model,
        spec,
        incoming,
        _batch(),
        payload,
        mode="random",
        global_seed=42,
        round_id=1,
        client_id=9,
    )

    assert len(first.probes) == len(proposals)
    assert first.forward_count == 4
    assert first.selected_proposal_ids == second.selected_proposal_ids
    assert len(first.selected_proposal_ids) == 2
    assert first.selected_weights == (0.5, 0.5)
    expected = sum(
        (_proposal.vector for _proposal in proposals if _proposal.proposal_id in first.selected_proposal_ids),
        torch.zeros(spec.numel),
    ) / 2
    assert torch.equal(first.direction, expected)


def test_selection_independent_audit_never_returns_a_direction():
    model = ToyPrivateClassifier()
    spec = build_model_lora_flat_spec(model)
    incoming = flatten_model(model, spec).cpu()
    payload = _payload(
        spec, [_proposal(0, [[1.0, -1.0], [-1.0, 1.0]], spec)]
    )
    probes = evaluate_proposals_for_audit(
        model, spec, incoming, _batch(), payload
    )
    assert len(probes) == 1
    assert probes[0].proposal_id == 0


def test_restore_direction_uses_one_backward_without_optimizer_step_or_pollution():
    model = ToyPrivateClassifier()
    model.train()
    spec = build_model_lora_flat_spec(model)
    optimizer = torch.optim.Adam(
        [model.image_encoder.q_lora_A], lr=0.1
    )
    optimizer_before = copy.deepcopy(optimizer.state_dict())
    parameter_before = flatten_model(model, spec).cpu().clone()
    batch = _batch()
    reference = IncomingGlobalReference(
        reference_logits=torch.tensor([[4.0, -4.0], [-4.0, 4.0]]),
        reference_ce=0.01,
        reference_round=0,
        reference_update_count=1,
        memory_fingerprint="memory",
    )

    result = compute_restore_direction(
        model,
        spec,
        batch,
        reference,
        degradation=0.5,
    )

    assert result.backward_count == 1
    assert result.gradient_norm > 0
    assert result.loss > 0
    assert torch.equal(flatten_model(model, spec).cpu(), parameter_before)
    assert optimizer.state_dict() == optimizer_before
    assert model.image_encoder.q_lora_A.grad is None
    assert model.bn.weight.grad is None
    assert model.bn.bias.grad is None
    assert model.training is True


def test_zero_degradation_skips_restore_backward_strictly():
    model = ToyPrivateClassifier()
    spec = build_model_lora_flat_spec(model)
    reference = IncomingGlobalReference(
        reference_logits=torch.zeros(2, 2),
        reference_ce=1.0,
        reference_round=0,
        reference_update_count=1,
        memory_fingerprint="memory",
    )
    result = compute_restore_direction(
        model, spec, _batch(), reference, degradation=0.0
    )
    assert result.backward_count == 0
    assert result.gradient_norm == 0.0
    assert torch.equal(result.direction, torch.zeros(spec.numel))


def test_method_inputs_fail_closed_on_wrong_round_spec_or_reference_shape():
    model = ToyPrivateClassifier()
    spec = build_model_lora_flat_spec(model)
    incoming = flatten_model(model, spec).cpu()
    payload = _payload(spec, [])
    wrong_round = copy.copy(payload)
    object.__setattr__(wrong_round, "target_round", 2)
    with pytest.raises(ValueError, match="target round"):
        evaluate_and_select_proposals(
            model,
            spec,
            incoming,
            _batch(),
            wrong_round,
            mode="none",
            global_seed=42,
            round_id=1,
            client_id=0,
        )

    reference = IncomingGlobalReference(
        reference_logits=torch.zeros(3, 2),
        reference_ce=1.0,
        reference_round=0,
        reference_update_count=1,
        memory_fingerprint="memory",
    )
    with pytest.raises(ValueError, match="different sizes"):
        compute_restore_direction(
            model, spec, _batch(), reference, degradation=0.5
        )
