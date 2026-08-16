from __future__ import annotations

import json

import numpy as np
from PIL import Image

from tools.breadth_audit.artifacts import append_breadth_artifacts
from tools.breadth_audit.comparison import preregistered_direction_gate
from tools.breadth_audit.evaluator import evaluate_three_breadth_families
from tools.breadth_audit.inputs import load_preregistered_neighbors
from tools.breadth_audit.metrics import (
    multiview_robustness_metrics,
    neighbor_discrimination_metrics,
    visual_subgroup_metrics,
)
from tools.breadth_audit.protocol import frozen_protocol, write_frozen_protocol
from tools.breadth_audit.views import FROZEN_VIEW_NAMES, fixed_view


def _logits():
    labels = np.asarray([2, 2, 2, 2])
    logits = np.asarray([
        [0.0, 1.0, 3.0, 2.0],
        [0.0, 1.0, 3.0, 2.0],
        [0.0, 4.0, 3.0, 2.0],
        [0.0, 4.0, 3.0, 2.0],
    ])
    return logits, labels


def test_protocol_is_mechanism_only_and_refuses_drift(tmp_path):
    protocol = frozen_protocol()
    assert protocol["scope"] == "mechanism_validation_only"
    assert "sota_comparison" in protocol["scope_exclusions"]
    assert protocol["dataset"]["tail_classes"] == list(range(80, 100))
    assert protocol["dataset"]["tail_sample_count"] == 153
    assert protocol["fairness"]["optimizer_rule_equal"] is True
    assert protocol["fairness"]["realized_optimizer_steps_equal"] is False
    assert protocol["training"]["precision"] == "fp32"
    path = write_frozen_protocol(tmp_path)
    assert write_frozen_protocol(tmp_path) == path
    changed = json.loads(path.read_text(encoding="utf-8"))
    changed["training"]["learning_rate"] = 999
    path.write_text(json.dumps(changed), encoding="utf-8")
    try:
        write_frozen_protocol(tmp_path)
    except RuntimeError:
        pass
    else:
        raise AssertionError("protocol drift was not rejected")


def test_preregistered_neighbors_are_complete_and_non_tail_only():
    neighbors, metadata = load_preregistered_neighbors(range(80, 100))
    assert set(neighbors) == set(range(80, 100))
    assert metadata["neighbors_hash"]
    assert all(len(values) == 10 for values in neighbors.values())
    assert all(not set(values).intersection(range(80, 100)) for values in neighbors.values())


def test_visual_subgroup_metrics():
    logits, labels = _logits()
    rows = visual_subgroup_metrics(
        logits, labels, [0, 0, 1, 1], [2], recognized_accuracy_threshold=0.5
    )
    row = rows[0]
    assert row["cluster_balanced_accuracy"] == 0.5
    assert row["worst_cluster_accuracy"] == 0.0
    assert row["cluster_accuracy_std"] == 0.5
    assert row["recognized_cluster_count_at_50"] == 1


def test_multiview_metrics_and_frozen_view_set():
    clean, labels = _logits()
    corrupted = clean.copy()
    corrupted[:, 1] += 3.0
    views = {name: clean.copy() for name in FROZEN_VIEW_NAMES}
    views["blur"] = corrupted
    row = multiview_robustness_metrics(views, labels, [2])[0]
    assert row["clean_accuracy"] == 0.5
    assert row["worst_view_accuracy"] == 0.0
    assert row["prediction_consistency"] < 1.0
    assert row["clean_to_corruption_accuracy_drop"] > 0.0


def test_neighbor_discrimination_metrics():
    logits, labels = _logits()
    row = neighbor_discrimination_metrics(logits, labels, {2: [1, 3]}, [2])[0]
    expected = np.mean(logits[:, 2, None] - logits[:, [1, 3]])
    assert row["target_vs_neighbor_pairwise_margin"] == expected
    assert row["neighbor_count"] == 2
    assert 0.0 <= row["positive_margin_neighbor_coverage"] <= 1.0


def test_all_three_families_are_emitted_together():
    logits, labels = _logits()
    views = {name: logits.copy() for name in FROZEN_VIEW_NAMES}
    result = evaluate_three_breadth_families(
        logits_by_view=views, labels=labels, cluster_ids=[0, 0, 1, 1],
        neighbors_by_tail={2: [1, 3]}, tail_classes=[2],
    )
    assert set(result) == {
        "visual_subgroup_coverage",
        "multi_view_robustness",
        "neighbor_discrimination_breadth",
    }


def test_fixed_views_are_deterministic_and_keep_size():
    values = np.arange(32 * 32 * 3, dtype=np.uint8).reshape(32, 32, 3)
    image = Image.fromarray(values, mode="RGB")
    for view in FROZEN_VIEW_NAMES:
        first = np.asarray(fixed_view(image, view))
        second = np.asarray(fixed_view(image, view))
        assert first.shape == (32, 32, 3)
        assert np.array_equal(first, second)


def test_artifact_writer_requires_and_writes_all_families(tmp_path):
    logits, labels = _logits()
    views = {name: logits.copy() for name in FROZEN_VIEW_NAMES}
    result = evaluate_three_breadth_families(
        logits_by_view=views, labels=labels, cluster_ids=[0, 0, 1, 1],
        neighbors_by_tail={2: [1, 3]}, tail_classes=[2],
    )
    paths = append_breadth_artifacts(
        tmp_path, result, run_metadata={"seed": 42, "topology": "Dirichlet", "round": 0}
    )
    assert len(paths) == 3
    assert all(path.is_file() for path in paths)
    try:
        append_breadth_artifacts(
            tmp_path, {"visual_subgroup_coverage": result["visual_subgroup_coverage"]},
            run_metadata={"seed": 42, "topology": "Dirichlet", "round": 1},
        )
    except ValueError:
        pass
    else:
        raise AssertionError("partial metric-family output was accepted")


def test_preregistered_direction_gate_uses_every_primary_endpoint():
    def row(seed, round_id, tail_class, **values):
        return {"seed": seed, "round": round_id, "tail_class": tail_class, **values}

    dirichlet = {
        "visual_subgroup_coverage": [row(
            42, 1, 79, worst_cluster_accuracy=0.8,
            cluster_balanced_accuracy=0.9, recognized_cluster_fraction_at_50=1.0,
        )],
        "multi_view_robustness": [row(
            42, 1, 79, worst_view_accuracy=0.8, prediction_consistency=0.9,
            worst_view_margin=2.0, clean_to_corruption_accuracy_drop=0.1,
        )],
        "neighbor_discrimination_breadth": [row(
            42, 1, 79, target_vs_neighbor_pairwise_margin=2.0,
            worst_neighbor_margin=1.0, positive_margin_neighbor_coverage=1.0,
        )],
    }
    clientlt = {
        "visual_subgroup_coverage": [row(
            42, 1, 79, worst_cluster_accuracy=0.4,
            cluster_balanced_accuracy=0.6, recognized_cluster_fraction_at_50=0.5,
        )],
        "multi_view_robustness": [row(
            42, 1, 79, worst_view_accuracy=0.4, prediction_consistency=0.8,
            worst_view_margin=1.0, clean_to_corruption_accuracy_drop=0.2,
        )],
        "neighbor_discrimination_breadth": [row(
            42, 1, 79, target_vs_neighbor_pairwise_margin=1.0,
            worst_neighbor_margin=0.5, positive_margin_neighbor_coverage=0.5,
        )],
    }
    result = preregistered_direction_gate(dirichlet, clientlt)
    assert result["supporting_family_count"] == 3
    assert result["directional_gate_pass"] is True

    # One contradictory primary endpoint makes that whole family fail; it is
    # not legal to select only the favorable endpoints after seeing results.
    clientlt["multi_view_robustness"][0]["prediction_consistency"] = 0.95
    result = preregistered_direction_gate(dirichlet, clientlt)
    assert result["families"]["multi_view_robustness"]["directionally_consistent"] is False
    assert result["supporting_family_count"] == 2
