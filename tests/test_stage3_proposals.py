import math

import pytest
import torch

from utils.stage3_proposals import (
    MAX_PROTOTYPES,
    MIN_CLUSTER_SOURCES,
    ClientUpload,
    ProposalBank,
    assert_client_payload_is_private,
    build_proposal_bank,
    merge_small_clusters,
)
from utils.stage3_vectors import make_flat_spec


CONDITION = "p_fcc_only"
SOURCE_ROUND = 7


def _spec(dimension=6):
    state = {
        "image_encoder.q_lora_A": torch.zeros(dimension, dtype=torch.float32)
    }
    return make_flat_spec(state), state


def _upload(client_id, vector, spec, *, condition=CONDITION, round_id=SOURCE_ROUND, spec_hash=None):
    return ClientUpload(
        client_id=client_id,
        vector=torch.as_tensor(vector),
        spec_hash=spec.spec_hash if spec_hash is None else spec_hash,
        condition=condition,
        round_id=round_id,
    )


def _build(vectors, *, dimension=None, condition=CONDITION, source_round=SOURCE_ROUND):
    dimension = dimension or len(vectors[0]) if vectors else (dimension or 2)
    spec, _ = _spec(dimension)
    uploads = [
        _upload(
            client_id,
            vector,
            spec,
            condition=condition,
            round_id=source_round,
        )
        for client_id, vector in enumerate(vectors)
    ]
    return build_proposal_bank(
        uploads,
        spec=spec,
        global_seed=42,
        source_round=source_round,
        condition=condition,
    ), spec


@pytest.mark.parametrize("count", [0, 1, 2, 3])
def test_zero_to_three_valid_updates_produce_empty_bank(count):
    bank, spec = _build([[1.0, 0.0]] * count, dimension=2)
    assert bank.is_empty
    assert bank.valid_update_count == count
    assert bank.initial_cluster_count == 0
    assert bank.payload_for(99).proposals == ()
    assert bank.spec_hash == spec.spec_hash


def test_four_valid_updates_form_one_multi_source_prototype():
    bank, _ = _build([[1.0, 0.0]] * 4)
    assert bank.initial_cluster_count == 1
    assert len(bank.clusters) == 1
    assert bank.clusters[0].source_count == MIN_CLUSTER_SOURCES

    nonmember = bank.payload_for(99).proposals
    member = bank.payload_for(0).proposals
    assert len(nonmember) == len(member) == 1
    assert nonmember[0].source_count == 4
    assert member[0].source_count == 3
    assert nonmember[0].vector.norm().item() == pytest.approx(1.0)


def test_twenty_four_updates_can_retain_six_balanced_clusters():
    vectors = []
    for direction in range(6):
        basis = torch.zeros(6)
        basis[direction] = 1.0
        vectors.extend([basis.clone() for _ in range(4)])
    bank, _ = _build(vectors, dimension=6)

    assert bank.initial_cluster_count == MAX_PROTOTYPES == 6
    assert len(bank.clusters) == 6
    assert sorted(cluster.source_count for cluster in bank.clusters) == [4] * 6
    assert len(bank.payload_for(999).proposals) == 6
    assert all(
        proposal.vector.norm().item() == pytest.approx(1.0)
        for proposal in bank.payload_for(999).proposals
    )


@pytest.mark.parametrize("count", [4, 5, 7, 8, 12, 23, 24, 25, 60])
def test_random_update_counts_never_leave_an_undersized_cluster(count):
    generator = torch.Generator().manual_seed(1000 + count)
    vectors = torch.randn(count, 16, generator=generator)
    bank, _ = _build(vectors.tolist(), dimension=16)

    assert len(bank.clusters) <= min(MAX_PROTOTYPES, count // MIN_CLUSTER_SOURCES)
    assert all(cluster.source_count >= MIN_CLUSTER_SOURCES for cluster in bank.clusters)
    assert all(
        torch.isfinite(proposal.vector).all()
        for proposal in bank.payload_for(999).proposals
    )


def test_identical_directions_delete_empty_clusters_without_refill():
    bank_forward, _ = _build([[1.0, 0.0]] * 8)
    bank_reverse, spec = _build(list(reversed([[1.0, 0.0]] * 8)))

    assert bank_forward.initial_cluster_count == 2
    assert len(bank_forward.clusters) == 1
    assert bank_forward.clusters[0].cluster_id == 0
    assert bank_forward.clusters[0].source_count == 8
    assert bank_forward.clusters[0].member_client_ids == bank_reverse.clusters[0].member_client_ids
    assert torch.equal(
        bank_forward.payload_for(99).proposals[0].vector,
        bank_reverse.payload_for(99, expected_spec_hash=spec.spec_hash).proposals[0].vector,
    )


def test_opposite_directions_form_two_nonzero_prototypes():
    bank, _ = _build([[1.0, 0.0]] * 4 + [[-1.0, 0.0]] * 4)
    proposals = bank.payload_for(99).proposals

    assert len(bank.clusters) == 2
    assert sorted(cluster.source_count for cluster in bank.clusters) == [4, 4]
    assert len(proposals) == 2
    cosine = torch.dot(proposals[0].vector, proposals[1].vector) / (
        proposals[0].vector.norm() * proposals[1].vector.norm()
    )
    assert cosine.item() == pytest.approx(-1.0)


def test_small_cluster_is_merged_and_not_replenished():
    bank, _ = _build([[1.0, 0.0]] * 7 + [[0.0, 1.0]])
    assert bank.initial_cluster_count == 2
    assert len(bank.clusters) == 1
    assert bank.clusters[0].source_count == 8


def test_near_zero_full_mean_is_discarded_not_randomly_filled():
    bank, _ = _build(
        [[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]]
    )
    assert bank.valid_update_count == 4
    assert bank.initial_cluster_count == 1
    assert bank.clusters == ()
    assert bank.payload_for(99).proposals == ()


def test_invalid_updates_are_excluded_from_median_and_clustering():
    spec, _ = _spec(2)
    uploads = [_upload(index, [1.0, 0.0], spec) for index in range(4)]
    uploads.extend(
        [
            _upload(4, [float("nan"), 0.0], spec),
            _upload(5, [float("inf"), 0.0], spec),
            _upload(6, [0.0, 0.0], spec),
            _upload(7, [1.0, 0.0, 0.0], spec),
            _upload(8, [1.0, 0.0], spec, spec_hash="wrong"),
            _upload(9, torch.tensor([1, 0], dtype=torch.int64), spec),
        ]
    )
    bank = build_proposal_bank(
        uploads,
        spec=spec,
        global_seed=42,
        source_round=SOURCE_ROUND,
        condition=CONDITION,
    )

    assert bank.valid_update_count == 4
    assert bank.median_update_norm == pytest.approx(1.0)
    assert dict(bank.invalid_update_reasons) == {
        4: "nonfinite",
        5: "nonfinite",
        6: "near_zero_norm",
        7: "shape_or_numel_mismatch",
        8: "spec_hash_mismatch",
        9: "nonfloating_dtype",
    }
    assert bank.clusters[0].member_client_ids == (0, 1, 2, 3)


def test_median_norm_clipping_and_prototype_budget_are_exact():
    vectors = [[scale, 0.0] for scale in (1.0, 2.0, 3.0, 100.0)]
    bank, _ = _build(vectors)
    cluster = bank.clusters[0]
    clipped_norms = [float(vector.norm().item()) for _, vector in cluster.clipped_updates]

    assert bank.median_update_norm == pytest.approx(2.5)
    assert clipped_norms == pytest.approx([1.0, 2.0, 2.5, 2.5])
    assert bank.payload_for(99).proposals[0].vector.norm().item() == pytest.approx(2.5)


def test_leave_one_out_removes_own_clipped_contribution_exactly():
    root_three_over_two = math.sqrt(3.0) / 2.0
    vectors = [
        [1.0, 0.0],
        [1.0, 0.0],
        [-0.5, root_three_over_two],
        [-0.5, -root_three_over_two],
    ]
    bank, _ = _build(vectors)

    # Removing client 0 leaves an exactly near-zero mean, so no proposal is
    # sent.  A nonmember receives the full four-source prototype.
    assert bank.payload_for(0).proposals == ()
    assert bank.payload_for(99).proposals[0].source_count == 4

    proposal_for_one = bank.payload_for(2).proposals[0]
    cluster = bank.clusters[0]
    own = cluster.contribution(2)
    expected_sum = cluster.clipped_sum - own
    expected = bank.median_update_norm * expected_sum / expected_sum.norm()
    assert proposal_for_one.source_count == 3
    assert torch.allclose(proposal_for_one.vector, expected, atol=1e-7, rtol=1e-7)


def test_merge_destination_cosine_tie_uses_lower_cluster_index():
    rows = torch.tensor(
        [
            [1.0, 0.0, 0.0],  # one-source small cluster 0
            [0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],  # cluster 1
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],  # cluster 2
        ]
    )
    merged = merge_small_clusters({0: [0], 1: [1, 2, 3, 4], 2: [5, 6, 7, 8]}, rows)
    assert 0 not in merged
    assert 0 in merged[1]
    assert 0 not in merged[2]


def test_condition_round_and_duplicate_boundaries_fail_closed():
    spec, _ = _spec(2)
    good = [_upload(index, [1.0, 0.0], spec) for index in range(4)]
    with pytest.raises(ValueError, match="Cross-condition"):
        build_proposal_bank(
            good[:3] + [_upload(3, [1.0, 0.0], spec, condition="combined")],
            spec=spec,
            global_seed=42,
            source_round=SOURCE_ROUND,
            condition=CONDITION,
        )
    with pytest.raises(ValueError, match="Stale/future"):
        build_proposal_bank(
            good[:3] + [_upload(3, [1.0, 0.0], spec, round_id=6)],
            spec=spec,
            global_seed=42,
            source_round=SOURCE_ROUND,
            condition=CONDITION,
        )
    with pytest.raises(ValueError, match="Duplicate"):
        build_proposal_bank(
            good + [_upload(3, [1.0, 0.0], spec)],
            spec=spec,
            global_seed=42,
            source_round=SOURCE_ROUND,
            condition=CONDITION,
        )


def test_payload_contains_no_membership_or_private_functional_fields():
    bank, spec = _build([[1.0, 0.0]] * 4)
    payload = bank.payload_for(
        0,
        expected_condition=CONDITION,
        expected_target_round=SOURCE_ROUND + 1,
        expected_spec_hash=spec.spec_hash,
    ).as_dict()
    assert_client_payload_is_private(payload)
    text = repr(payload).lower()
    for forbidden in (
        "member_client_ids",
        "client_id",
        "utility",
        "label",
        "logit",
        "degradation",
        "accepted",
    ):
        assert forbidden not in text

    with pytest.raises(ValueError, match="condition mismatch"):
        bank.payload_for(0, expected_condition="combined")
    with pytest.raises(ValueError, match="target round mismatch"):
        bank.payload_for(0, expected_target_round=999)


def test_bank_checkpoint_roundtrip_preserves_client_specific_loo(tmp_path):
    bank, spec = _build(
        [[1.0, 0.0]] * 4 + [[-1.0, 0.0]] * 4
    )
    path = bank.save(tmp_path / "proposal_bank_latest.pt")
    loaded = ProposalBank.load(
        path,
        expected_global_seed=42,
        expected_condition=CONDITION,
        expected_source_round=SOURCE_ROUND,
        expected_spec_hash=spec.spec_hash,
    )

    assert loaded.diagnostics() == bank.diagnostics()
    for client_id in (0, 4, 99):
        expected = bank.payload_for(client_id).proposals
        actual = loaded.payload_for(client_id).proposals
        assert [item.proposal_id for item in actual] == [item.proposal_id for item in expected]
        assert [item.source_count for item in actual] == [item.source_count for item in expected]
        assert all(
            torch.equal(left.vector, right.vector)
            for left, right in zip(actual, expected)
        )

    with pytest.raises(ValueError, match="checkpoint condition mismatch"):
        ProposalBank.load(path, expected_condition="combined")
    with pytest.raises(ValueError, match="global seed mismatch"):
        ProposalBank.load(path, expected_global_seed=2026)


def test_determinism_is_independent_of_upload_input_order():
    spec, _ = _spec(3)
    directions = [
        [1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, 1.0],
        [-1.0, 0.0, 0.0],
        [-1.0, 0.0, 0.0],
    ]
    uploads = [_upload(index, vector, spec) for index, vector in enumerate(directions)]
    first = build_proposal_bank(
        uploads,
        spec=spec,
        global_seed=42,
        source_round=SOURCE_ROUND,
        condition=CONDITION,
    )
    second = build_proposal_bank(
        list(reversed(uploads)),
        spec=spec,
        global_seed=42,
        source_round=SOURCE_ROUND,
        condition=CONDITION,
    )
    assert [cluster.member_client_ids for cluster in first.clusters] == [
        cluster.member_client_ids for cluster in second.clusters
    ]
    assert all(
        torch.equal(left.vector, right.vector)
        for left, right in zip(
            first.payload_for(99).proposals,
            second.payload_for(99).proposals,
        )
    )


def test_identical_uploads_use_identical_clustering_across_conditions():
    spec, _ = _spec(5)
    generator = torch.Generator().manual_seed(91)
    vectors = torch.randn(20, 5, generator=generator)

    banks = []
    for condition in ("p_fcc_only", "combined", "random_proposal"):
        uploads = [
            _upload(
                client_id,
                vector,
                spec,
                condition=condition,
            )
            for client_id, vector in enumerate(vectors)
        ]
        banks.append(
            build_proposal_bank(
                uploads,
                spec=spec,
                global_seed=42,
                source_round=SOURCE_ROUND,
                condition=condition,
            )
        )

    expected_members = [cluster.member_client_ids for cluster in banks[0].clusters]
    expected_vectors = banks[0].payload_for(999).proposals
    for bank in banks[1:]:
        assert [cluster.member_client_ids for cluster in bank.clusters] == expected_members
        actual_vectors = bank.payload_for(999).proposals
        assert [item.proposal_id for item in actual_vectors] == [
            item.proposal_id for item in expected_vectors
        ]
        assert all(
            torch.equal(left.vector, right.vector)
            for left, right in zip(expected_vectors, actual_vectors)
        )
