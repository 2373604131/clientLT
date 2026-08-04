import math
import tempfile
from pathlib import Path
from types import SimpleNamespace

import torch

from utils.boundary_audit import load_boundary_round_dump, save_boundary_round_dump, diagnose_edges
from utils.boundary_gate import BoundaryGateConfig, build_boundary_candidates
from utils.boundary_metrics import (
    cap_repair_norm,
    support_counterfactual_delta,
)
from utils.boundary_repair import build_repair_candidates, solve_minimum_norm_repair


def test_support_counterfactual_preserves_or_renormalizes_fedavg_mass():
    deltas = torch.tensor([[2.0, 0.0], [0.0, 1.0]], dtype=torch.float64)
    weights = torch.tensor([0.2, 0.8], dtype=torch.float64)
    support = torch.tensor([True, False])
    actual, actual_report = support_counterfactual_delta(deltas, weights, support, normalized=False)
    normalized, normalized_report = support_counterfactual_delta(deltas, weights, support, normalized=True)
    assert torch.allclose(actual, torch.tensor([0.4, 0.0], dtype=torch.float64))
    assert torch.allclose(normalized, torch.tensor([2.0, 0.0], dtype=torch.float64))
    assert actual_report["support_mass"] == normalized_report["support_mass"] == 0.2
    assert actual_report["raw_norm"] < normalized_report["raw_norm"]


def test_boundary_dump_uses_explicit_trainable_keys_and_reconstructs_fedavg():
    before = {"w": torch.tensor([0.0, 0.0]), "frozen": torch.tensor([7], dtype=torch.int64)}
    locals_ = [
        {"w": torch.tensor([1.0, 0.0])},
        {"w": torch.tensor([0.0, 2.0])},
    ]
    after = {"w": torch.tensor([0.25, 1.5]), "frozen": torch.tensor([7], dtype=torch.int64)}
    with tempfile.TemporaryDirectory() as temporary_dir:
        directory = save_boundary_round_dump(
            output_dir=temporary_dir,
            args=SimpleNamespace(flag=True),
            cfg=SimpleNamespace(NAME="test"),
            epoch=0,
            global_before=before,
            global_after=after,
            local_weights=locals_,
            selected_clients=[0, 1],
            datanumber_client=[1, 3],
            client_class_counts={0: torch.tensor([1, 0]), 1: torch.tensor([0, 3])},
            global_class_counts=torch.tensor([1, 3]),
            trainable_keys={"w"},
        )
        payload, metadata = load_boundary_round_dump(directory)
    assert payload["trainable_keys"] == ["w"]
    assert "frozen" not in payload["global_before_trainable"]
    assert metadata["test_used_before_dump"] is False


def test_repair_is_capped_before_the_complete_update_is_norm_matched():
    fedavg = torch.tensor([3.0, 4.0], dtype=torch.float64)
    repair = torch.tensor([10.0, 0.0], dtype=torch.float64)
    capped, report = cap_repair_norm(repair, budget=5.0, repair_ratio=0.2)
    assert math.isclose(float(capped.norm().item()), 1.0)
    candidates, _ = build_repair_candidates(fedavg, repair, repair_ratio=0.2, alphas=(1.0, 0.5))
    for _, candidate, match_report in candidates:
        assert math.isclose(float(candidate.norm().item()), 5.0, rel_tol=0.0, abs_tol=1e-10)
        assert math.isclose(match_report["target_final_norm"], 5.0)


def test_minimum_norm_solver_reports_and_satisfies_simple_constraints():
    gradients = torch.eye(2, dtype=torch.float64)
    deficits = torch.tensor([0.5, 1.0], dtype=torch.float64)
    repair, report = solve_minimum_norm_repair(gradients, deficits, tolerance=1e-12)
    assert report["status"] in {"converged", "max_iterations"}
    assert report["max_linear_violation"] < 1e-8
    assert torch.all(repair >= deficits - 1e-8)


class _MarginModel:
    """Small offline model fixture with logits controlled by trainable state."""

    def logits_from_cached_features(self, features, trainable_state):
        weights = trainable_state["w"].to(dtype=torch.float64)
        values = torch.as_tensor(features, dtype=torch.float64)[:, :1]
        return values * weights.reshape(1, -1)

    def edge_gradient_from_cached_features(self, features, labels, negatives, trainable_state, trainable_keys):
        return 0.0, {"w": torch.tensor([1.0, -1.0])}

    def text_features_from_trainable_state(self, trainable_state):
        return torch.eye(2)


def test_four_model_diagnostics_decompose_dilution_and_interference():
    before = {"w": torch.tensor([0.0, 0.0])}
    payload = {
        "flatten_spec": {
            "keys": ["w"],
            "shapes": [[2]],
            "dtypes": ["torch.float32"],
            "offsets": [[0, 2]],
            "numel": 2,
        },
        "global_before_trainable": before,
        "local_trainable_states": [
            {"w": torch.tensor([2.0, 0.0])},
            {"w": torch.tensor([0.0, 1.0])},
        ],
        "selected_client_ids": [0, 1],
        "fedavg_weights": torch.tensor([0.2, 0.8], dtype=torch.float64),
        "client_class_counts": torch.tensor([[1, 0], [0, 1]]),
        "global_class_counts": torch.tensor([1, 1]),
        "num_classes": 2,
    }
    cache = {
        "features": torch.tensor([[1.0], [1.0]]),
        "labels": torch.tensor([0, 1]),
        "client_ids": torch.tensor([0, 1]),
        "fixed_hard_negatives": torch.tensor([[1], [0]]),
        "num_clients": 2,
    }
    edges = [{"edge_id": 0, "class_id": 0, "negative_id": 1, "num_support_clients": 1}]
    edge_counts = torch.tensor([[1], [0]])
    rows, fragile, _ = diagnose_edges(
        _MarginModel(), payload, cache, edges, edge_counts,
        gamma=0.5, tau=0.0, min_support_clients=1, max_fragile_edges_per_class=1, max_total_edges=5,
    )
    assert len(fragile) == 1
    row = rows[0]
    assert math.isclose(row["local_audit_gain"], 2.0)
    assert math.isclose(row["gain_support_normalized"], 2.0)
    assert math.isclose(row["gain_support_actual"], 0.4, abs_tol=1e-7)
    assert math.isclose(row["gain_all_fedavg"], -0.4, abs_tol=1e-7)
    assert math.isclose(row["dilution"], 1.6, abs_tol=1e-7)
    assert math.isclose(row["interference"], 0.8, abs_tol=1e-7)
    assert math.isclose(row["gain_support_normalized"] - row["gain_all_fedavg"], row["dilution"] + row["interference"], abs_tol=1e-7)
    assert row["local_audit_gain_by_client"] == {"0": 2.0}


def test_all_performance_candidates_match_the_fedavg_final_update_norm():
    payload = {
        "flatten_spec": {
            "keys": ["w"],
            "shapes": [[2]],
            "dtypes": ["torch.float32"],
            "offsets": [[0, 2]],
            "numel": 2,
        },
        "global_before_trainable": {"w": torch.tensor([0.0, 0.0])},
        "local_trainable_states": [
            {"w": torch.tensor([2.0, 0.0])},
            {"w": torch.tensor([0.0, 1.0])},
        ],
        "selected_client_ids": [0, 1],
        "fedavg_weights": torch.tensor([0.2, 0.8], dtype=torch.float64),
        "client_class_counts": torch.tensor([[1, 0], [0, 1]]),
        "global_class_counts": torch.tensor([1, 1]),
        "num_classes": 2,
    }
    cache = {
        "features": torch.tensor([[1.0], [1.0]]),
        "labels": torch.tensor([0, 1]),
        "client_ids": torch.tensor([0, 1]),
        "fixed_hard_negatives": torch.tensor([[1], [0]]),
        "num_clients": 2,
    }
    _, rows, diagnostics, context = build_boundary_candidates(
        _MarginModel(),
        payload,
        {"head_class_ids": [1], "tail_class_ids": [0]},
        cache,
        BoundaryGateConfig(min_support_clients=1, max_edges_per_class=1, max_total_edges=1),
    )
    budget = context["norm_budget"]
    assert any(row["fragile_selected"] for row in diagnostics)
    assert {row["method"] for row in rows} == {
        "fedavg",
        "inverse_support_mass",
        "classwise_aggregation",
        "cusp_minimal",
        "random_repair",
        "ordinary_audit_gradient",
        "edge_level_boundary_repair",
    }
    for row in rows:
        assert math.isclose(row["final_norm"], budget, rel_tol=0.0, abs_tol=1e-8)
        assert row["norm_relative_error"] < 1e-10
