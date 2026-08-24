import copy

import pytest
import torch

from utils.stage3_private_state import (
    AUDIT_VIEW_SIZE,
    FUNCTIONAL_MEMORY_SIZE,
    ClientPrivateStateStore,
    build_client_evidence_view,
    observe_incoming_global,
)


def _balanced_data(per_class=25):
    sample_ids = []
    labels = []
    for class_id in range(4):
        for offset in range(per_class):
            sample_ids.append(class_id * 1000 + offset)
            labels.append(class_id)
    return sample_ids, labels


def test_evidence_views_are_deterministic_balanced_disjoint_and_order_independent():
    sample_ids, labels = _balanced_data()
    first = build_client_evidence_view(
        sample_ids, labels, global_seed=42, client_id=7
    )
    second = build_client_evidence_view(
        list(reversed(sample_ids)),
        list(reversed(labels)),
        global_seed=42,
        client_id=7,
    )

    assert first == second
    assert len(first.memory_sample_ids) == FUNCTIONAL_MEMORY_SIZE
    assert len(first.audit_sample_ids) == AUDIT_VIEW_SIZE
    assert set(first.memory_sample_ids).isdisjoint(first.audit_sample_ids)

    label_by_id = dict(zip(sample_ids, labels))
    memory_counts = [
        sum(label_by_id[item] == class_id for item in first.memory_sample_ids)
        for class_id in range(4)
    ]
    audit_counts = [
        sum(label_by_id[item] == class_id for item in first.audit_sample_ids)
        for class_id in range(4)
    ]
    assert max(memory_counts) - min(memory_counts) <= 1
    assert max(audit_counts) - min(audit_counts) <= 1


def test_small_client_uses_every_sample_for_memory_and_has_no_audit_reuse():
    view = build_client_evidence_view(
        list(range(10)), [0] * 5 + [1] * 5, global_seed=42, client_id=2
    )
    assert set(view.memory_sample_ids) == set(range(10))
    assert view.audit_sample_ids == ()


def test_evidence_selection_is_identical_across_experiment_conditions():
    sample_ids, labels = _balanced_data()
    views = []
    for condition in ("fedavg", "p_fcc_only", "d_rtc_only", "combined", "random"):
        store = ClientPrivateStateStore(
            global_seed=42,
            condition=condition,
            flatten_spec_hash="same-spec",
        )
        views.append(store.get_or_create_evidence(4, sample_ids, labels))
    assert all(view == views[0] for view in views[1:])


def test_evidence_rejects_transient_or_duplicate_ids():
    with pytest.raises(TypeError, match="globally stable"):
        build_client_evidence_view(
            [("loader", 0)], [0], global_seed=42, client_id=0
        )
    with pytest.raises(ValueError, match="duplicate"):
        build_client_evidence_view(
            [10, 10], [0, 0], global_seed=42, client_id=0
        )


def _store_with_client():
    sample_ids = list(range(40))
    labels = [index % 2 for index in sample_ids]
    store = ClientPrivateStateStore(
        global_seed=42,
        condition="p_fcc_only",
        flatten_spec_hash="spec-abc",
    )
    evidence = store.get_or_create_evidence(3, sample_ids, labels)
    memory_labels = torch.tensor(
        [labels[sample_id] for sample_id in evidence.memory_sample_ids]
    )
    return store, sample_ids, labels, memory_labels


def test_reference_initializes_only_from_incoming_and_tracks_strict_best():
    store, _, _, memory_labels = _store_with_client()
    count = memory_labels.numel()

    first_logits = torch.zeros(count, 2, requires_grad=True)
    first = store.observe_incoming_global(
        3, logits=first_logits, labels=memory_labels, round_id=0
    )
    assert first.initialized is True
    assert first.degradation == 0.0

    state = store.get(3)
    saved_logits = state.reference.reference_logits.clone()
    assert state.reference.reference_logits.dtype == torch.float32
    assert state.reference.reference_logits.device.type == "cpu"
    assert state.reference.reference_logits.requires_grad is False

    worse_logits = torch.zeros(count, 2)
    worse_logits[torch.arange(count), 1 - memory_labels] = 4.0
    worse = store.observe_incoming_global(
        3, logits=worse_logits, labels=memory_labels, round_id=1
    )
    assert worse.updated is False
    assert worse.degradation > 0.0
    assert torch.equal(store.get(3).reference.reference_logits, saved_logits)
    assert store.get(3).reference.reference_round == 0

    better_logits = torch.zeros(count, 2)
    better_logits[torch.arange(count), memory_labels] = 5.0
    better = store.observe_incoming_global(
        3, logits=better_logits, labels=memory_labels, round_id=2
    )
    assert better.updated is True
    assert better.degradation == 0.0
    assert store.get(3).reference.reference_round == 2
    assert store.get(3).reference.reference_update_count == 2

    equal = store.observe_incoming_global(
        3, logits=better_logits.clone(), labels=memory_labels, round_id=3
    )
    assert equal.updated is False  # strict '<', not '<='.
    assert equal.degradation == pytest.approx(0.0)
    assert store.get(3).reference.reference_round == 2


def test_absent_client_state_is_unchanged_and_missing_state_reinitializes():
    store, _, _, memory_labels = _store_with_client()
    other_ids = list(range(100, 140))
    other_labels = [item % 2 for item in other_ids]
    store.get_or_create_evidence(8, other_ids, other_labels)
    before = copy.deepcopy(store.get(8).as_dict())

    store.observe_incoming_global(
        3,
        logits=torch.zeros(memory_labels.numel(), 2),
        labels=memory_labels,
        round_id=0,
    )
    assert store.get(8).as_dict() == before
    assert store.get(8).reference is None

    fresh = ClientPrivateStateStore(
        global_seed=42,
        condition="p_fcc_only",
        flatten_spec_hash="spec-abc",
    )
    sample_ids = list(range(40))
    labels = [item % 2 for item in sample_ids]
    evidence = fresh.get_or_create_evidence(3, sample_ids, labels)
    fresh_labels = torch.tensor([labels[item] for item in evidence.memory_sample_ids])
    observation = fresh.observe_incoming_global(
        3,
        logits=torch.zeros(len(evidence.memory_sample_ids), 2),
        labels=fresh_labels,
        round_id=9,
    )
    assert observation.initialized is True
    assert observation.degradation == 0.0


def test_private_state_checkpoint_resume_is_exact_and_condition_bound(tmp_path):
    store, sample_ids, labels, memory_labels = _store_with_client()
    store.observe_incoming_global(
        3,
        logits=torch.zeros(memory_labels.numel(), 2),
        labels=memory_labels,
        round_id=0,
    )
    path = store.save(tmp_path / "stage3_client_state_latest.pt")
    resumed = ClientPrivateStateStore.load(
        path,
        expected_global_seed=42,
        expected_condition="p_fcc_only",
        expected_flatten_spec_hash="spec-abc",
    )

    assert resumed.state_dict()["schema_version"] == store.state_dict()["schema_version"]
    assert resumed.client_ids == (3,)
    assert resumed.get(3).evidence == store.get(3).evidence
    assert torch.equal(
        resumed.get(3).reference.reference_logits,
        store.get(3).reference.reference_logits,
    )
    assert resumed.get_or_create_evidence(3, sample_ids, labels) == store.get(3).evidence

    with pytest.raises(ValueError, match="condition mismatch"):
        ClientPrivateStateStore.load(path, expected_condition="combined")
    with pytest.raises(ValueError, match="flatten spec mismatch"):
        ClientPrivateStateStore.load(path, expected_flatten_spec_hash="wrong")


def test_existing_evidence_fails_closed_if_local_data_changes():
    store, sample_ids, labels, _ = _store_with_client()
    changed = labels.copy()
    changed[0] = 1 - changed[0]
    with pytest.raises(ValueError, match="changed after initialization"):
        store.get_or_create_evidence(3, sample_ids, changed)


def test_reference_requires_memory_shape_and_matching_fingerprint():
    store, _, _, memory_labels = _store_with_client()
    with pytest.raises(ValueError, match="memory size"):
        store.observe_incoming_global(
            3,
            logits=torch.zeros(memory_labels.numel() - 1, 2),
            labels=memory_labels[:-1],
            round_id=0,
        )

    store.observe_incoming_global(
        3,
        logits=torch.zeros(memory_labels.numel(), 2),
        labels=memory_labels,
        round_id=0,
    )
    with pytest.raises(ValueError, match="fingerprint changed"):
        observe_incoming_global(
            store.get(3).reference,
            logits=torch.zeros(memory_labels.numel(), 2),
            labels=memory_labels,
            round_id=1,
            memory_fingerprint="changed",
        )
