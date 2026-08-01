import torch

from utils.functional_cusp import (
    candidate_hash_from_delta,
    client_disagreement_subspace,
    solve_safe_direction,
)


def test_disagreement_subspace_is_orthogonal_to_fedavg():
    delta_avg = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64)
    client_deltas = torch.tensor(
        [[1.0, 0.2, 0.0], [1.0, -0.2, 0.0], [1.0, 0.0, 0.3]],
        dtype=torch.float64,
    )
    weights = torch.tensor([1 / 3, 1 / 3, 1 / 3], dtype=torch.float64)
    q, report = client_disagreement_subspace(client_deltas, delta_avg, weights, rank_max=2)
    assert q is not None
    assert report["fallback"] is False
    assert torch.allclose(q.T @ q, torch.eye(q.shape[1], dtype=torch.float64), atol=1e-10)
    assert torch.allclose(q.T @ delta_avg, torch.zeros(q.shape[1], dtype=torch.float64), atol=1e-10)


def test_safe_projection_removes_negative_common_direction():
    class_utilities = torch.tensor(
        [[-1.0, 0.0], [0.0, 1.0], [0.0, 1.0]],
        dtype=torch.float64,
    )
    counts = torch.tensor([100, 3, 3], dtype=torch.float64)
    v, report = solve_safe_direction(class_utilities, counts)
    assert v is not None
    assert report["fallback"] is False
    assert report["safe_dot"] >= -1e-10


def test_identical_client_updates_fallback_to_fedavg_subspace():
    delta_avg = torch.tensor([0.1, 0.2], dtype=torch.float64)
    client_deltas = delta_avg.repeat(3, 1)
    weights = torch.tensor([0.2, 0.3, 0.5], dtype=torch.float64)
    q, report = client_disagreement_subspace(client_deltas, delta_avg, weights)
    assert q is None
    assert report["fallback"] is True


def test_candidate_hash_is_deterministic():
    delta = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
    assert candidate_hash_from_delta(delta) == candidate_hash_from_delta(delta.clone())
