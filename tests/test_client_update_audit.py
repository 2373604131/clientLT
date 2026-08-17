from __future__ import annotations

import numpy as np
import pandas as pd

from tools.client_update_audit.manifests import (
    _balanced_quotas,
    _matched_allocations,
    _tail_relevance_by_class,
)
from tools.client_update_audit.protocol import TAIL_CLASSES, TAIL_CLIENT_IDS, frozen_protocol
from tools.client_update_audit.summarize import _accuracy_matched_pairs, _weighted_tail


def test_e2_protocol_freezes_local_only_scope_and_clientlt_constraints():
    protocol = frozen_protocol()
    assert protocol["local_training"]["server_aggregation_permitted"] is False
    assert protocol["dataset"]["tail_classes"] == list(range(80, 100))
    assert protocol["dataset"]["tail_sample_count"] == 153
    assert protocol["e2a"]["topologies"]["clientlt"]["tail_client_ids"] == [27, 28, 29]
    assert protocol["e2a"]["topologies"]["clientlt"]["tail_leakage"] == 0
    assert protocol["e2b"]["primary_endpoint"] == "accuracy_matched_worst_neighbor_margin_gain"
    assert len(protocol["protocol_hash"]) == 64


def test_balanced_companion_quotas_are_exact():
    assert _balanced_quotas(11, 2) == [6, 5]
    assert _balanced_quotas(11, 8) == [2, 2, 2, 1, 1, 1, 1, 1]
    assert sum(_balanced_quotas(14, 8)) == 14


def test_tail_relevance_uses_only_present_ranked_non_tail_neighbors():
    counts = np.zeros(100, dtype=np.int64)
    counts[80] = 3
    counts[81] = 1
    neighbors = {class_id: list(range(10)) for class_id in TAIL_CLASSES}
    neighbors[80] = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
    neighbors[81] = [20, 21, 22, 23, 24, 25, 26, 27, 28, 29]
    relevance = _tail_relevance_by_class(counts, neighbors)
    assert np.isclose(relevance[10], 0.75)
    assert np.isclose(relevance[11], 0.75 / 2)
    assert np.isclose(relevance[20], 0.25)
    assert np.all(relevance[80:] == 0)


def test_matched_allocation_preserves_count_changes_only_width():
    relevance = np.zeros(100, dtype=np.float64)
    # Many tied positive classes make exact narrow/broad mean matching feasible.
    relevance[:40] = 0.2
    capacities = {class_id: 100 for class_id in range(80)}
    fine_to_coarse = np.repeat(np.arange(20), 5)
    result = _matched_allocations(
        relevance, capacities, 11, 42, TAIL_CLIENT_IDS[0], 0.03, fine_to_coarse
    )
    narrow = result["narrow_related"]
    broad = result["broad_related"]
    unrelated = result["broad_unrelated"]
    assert len(narrow[1]) == 2
    assert len(broad[1]) == 8
    assert len(unrelated[1]) == 8
    assert len({int(fine_to_coarse[c]) for c in broad[1]}) >= 6
    assert len({int(fine_to_coarse[c]) for c in unrelated[1]}) >= 6
    assert sum(narrow[2]) == sum(broad[2]) == sum(unrelated[2]) == 11
    assert abs(narrow[0] - broad[0]) <= 0.03
    assert broad[0] > unrelated[0]


def test_tail_mass_weighting_and_accuracy_matching_are_class_centric():
    rows = []
    for condition, accuracy, worst in (
        ("narrow_related", 0.50, -0.2),
        ("broad_related", 0.51, 0.1),
    ):
        for client_id, mass in ((27, 3), (28, 1)):
            rows.append({
                "stage": "e2b", "data_seed": 42, "topology": "clientlt",
                "condition": condition, "local_epoch": 1, "tail_class": 80,
                "tail_sample_count": mass, "accuracy": accuracy,
                "margin": 0.0, "accuracy_gain": accuracy, "margin_gain": 0.0,
                "target_vs_neighbor_pairwise_margin": worst + 0.2,
                "worst_neighbor_margin": worst,
                "positive_margin_neighbor_coverage": float(worst > 0),
                "target_vs_neighbor_pairwise_margin_gain": worst + 0.2,
                "worst_neighbor_margin_gain": worst,
                "positive_margin_neighbor_coverage_gain": float(worst > 0),
                "tail_neighbor_access_score": 2.0 if condition == "broad_related" else 1.0,
                "companion_class_count": 8 if condition == "broad_related" else 2,
                "client_id": client_id,
            })
    weighted = _weighted_tail(pd.DataFrame(rows))
    assert np.allclose(weighted.tail_mass_weight_sum, 1.0)
    pairs = _accuracy_matched_pairs(
        weighted, "broad_related", "narrow_related", topology="clientlt",
        tolerance=0.02, effect_prefix="effect",
    )
    assert len(pairs) == 1
    assert pairs[0]["accuracy_matched"] is True
    assert np.isclose(pairs[0]["effect_worst_neighbor_margin_gain"], 0.3)
