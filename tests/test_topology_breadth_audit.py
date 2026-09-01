from __future__ import annotations

import ast
from pathlib import Path

import numpy as np

from tools.topology_breadth_audit.metrics import breadth_metrics, potential_pool_metrics
from tools.topology_breadth_audit.protocol import frozen_protocol


ROOT = Path(__file__).resolve().parents[1]


def test_protocol_uses_real_clients_fixed_margins_and_two_participation_views():
    protocol = frozen_protocol()
    assert protocol["local_updates"]["clients_per_topology"] == 30
    assert protocol["topologies"]["matched_dirichlet"]["row_margins"].startswith("exactly")
    assert protocol["A1_spatial"]["participation"] == "all 30 clients available"
    assert protocol["A2_temporal"]["frac"] == 0.4
    assert protocol["functional_evidence"]["test_split_accessed"] is False
    assert protocol["federated_deployment"]["server_deployable_method"] is False


def test_breadth_distinguishes_equal_strength_narrow_and_broad_vectors():
    narrow = breadth_metrics([4.0, 0.0, 0.0, 0.0])
    broad = breadth_metrics([1.0, 1.0, 1.0, 1.0])
    assert narrow["positive_strength"] == broad["positive_strength"] == 4.0
    assert narrow["effective_breadth"] == 1.0
    assert np.isclose(broad["effective_breadth"], 4.0)


def test_potential_pool_counts_positive_donors_without_negative_cancellation():
    result = potential_pool_metrics(
        np.asarray([[2.0, -3.0], [-1.0, 2.0]]), np.asarray([0.5, 0.5])
    )
    assert result["positive_donor_count"] == 2
    assert result["mean_positive_donors_per_boundary"] == 1.0
    assert np.isclose(result["potential_effective_breadth"], 2.0)


def test_phase2_analysis_does_not_call_client_training():
    source = (ROOT / "tools/topology_breadth_audit/analyze.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    names = {
        node.func.id for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert "_train_client" not in names
    assert ".backward(" not in source
    assert "test_labels" not in source

