from __future__ import annotations

import ast
from pathlib import Path

import numpy as np

from tools.functional_breadth_feasibility.matching import (
    constraint_aware_shortlist,
    coverage_metrics,
    symmetric_relative_gap,
)
from tools.functional_breadth_feasibility.p0_audit import run as run_p0
from tools.functional_breadth_feasibility.protocol import frozen_protocol
from tools.functional_breadth_feasibility.sampling import select_head_safety_ids


ROOT = Path(__file__).resolve().parents[1]


def test_protocol_is_no_training_and_private_only():
    protocol = frozen_protocol()
    assert protocol["protocol_name"] == "FUNCTIONAL_BREADTH_FEASIBILITY_V3"
    assert protocol["merge"]["shortlist_contrasts_per_tail"] == 24
    assert protocol["training"]["allowed"] is False
    assert protocol["training"]["missing_state_policy"] == "fail_without_retraining"
    assert protocol["evidence"]["test_split_access_allowed"] is False
    assert protocol["feasibility_gate"]["test_metrics_used_for_selection"] is False
    assert protocol["federated_deployment"]["server_deployable_method"] is False
    assert protocol["federated_deployment"]["privacy_claim"] is False


def test_v3_preserves_preregistered_scientific_thresholds():
    matching = frozen_protocol()["matching"]
    assert matching == {
        "exact": ["donor_count", "candidate_sample_count", "optimizer_steps"],
        "actual_symmetric_relative_strength_gap_max": 0.20,
        "relative_update_norm_gap_max": 0.10,
        "absolute_head_margin_gain_gap_max": 0.10,
        "absolute_direct_tail_cosine_gap_max": 0.15,
        "minimum_actual_effective_breadth_gap": 1.0,
    }


def test_coverage_metrics_separates_strength_and_breadth():
    narrow = coverage_metrics([4.0, 0.0, 0.0, 0.0])
    broad = coverage_metrics([1.0, 1.0, 1.0, 1.0])
    assert narrow["positive_strength"] == broad["positive_strength"] == 4.0
    assert narrow["effective_breadth"] == 1.0
    assert np.isclose(broad["effective_breadth"], 4.0)
    assert symmetric_relative_gap(4.0, 4.0) == 0.0


def test_constraint_aware_shortlist_filters_before_ranking_without_quartiles():
    def row(a, b, strength, breadth, norm=1.0, head=0.0, cosine=0.0):
        return {
            "tail_class": 80, "candidate_a": a, "candidate_b": b,
            "donor_count": 2, "candidate_sample_count": 24, "optimizer_steps": 6,
            "predicted_positive_strength": strength,
            "predicted_effective_breadth": breadth,
            "predicted_update_l2": norm,
            "predicted_head_margin_gain": head,
            "predicted_direct_tail_cosine": cosine,
        }

    rows = [
        row(0, 1, strength=1.0, breadth=1.0),
        row(2, 3, strength=4.0, breadth=2.0),
        row(8, 9, strength=4.05, breadth=3.0),
        row(4, 5, strength=4.1, breadth=4.0),
        row(6, 7, strength=10.0, breadth=9.0),
    ]
    selected, audit = constraint_aware_shortlist(rows, count=2, protocol=frozen_protocol())

    assert len(selected) == 2
    assert audit["predicted_feasible_contrasts"] >= 1
    assert all(item["predicted_constraints_pass"] for item in selected)
    assert all(item["predicted_breadth_gap"] >= 1.0 for item in selected)
    assert all(item["predicted_strength_symmetric_relative_gap"] <= 0.20 for item in selected)
    # The extreme 10-vs-1 strength contrast must never survive merely because
    # it has the largest breadth difference.
    assert not any(
        item["broad_a"] == 6 and item["narrow_a"] == 0 for item in selected
    )


def test_runtime_has_no_training_or_test_store_access():
    source = (ROOT / "tools/functional_breadth_feasibility/runtime.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    called_names = {
        node.func.id for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert "_train_client" not in called_names
    assert ".backward(" not in source
    assert "torch.optim" not in source
    assert "test_labels" not in source
    assert 'split != "train"' in source


def test_head_safety_uses_raw_train_complement_when_lt_remainder_is_too_small():
    # Class 0 has five raw-train examples. Four are in the federated LT pool,
    # leaving only one. Class 1 supplies a larger outside-LT complement.
    labels = np.asarray([0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1], dtype=np.int64)
    selected = select_head_safety_ids(
        labels, lt_raw_ids=[0, 1], used_sample_ids=["train:2", "train:3"],
        class_ids=[0, 1], samples_per_class=1,
    )
    assert selected[0] == [4]
    assert selected[1][0] in {5, 6, 7, 8, 9, 10}
    assert not ({0, 1, 2, 3} & set(selected[0]))


def test_head_safety_allows_unused_lt_examples_when_a_head_class_saturates_raw_train():
    labels = np.asarray([0, 0, 0, 1, 1], dtype=np.int64)
    selected = select_head_safety_ids(
        labels, lt_raw_ids=[0, 1, 2], used_sample_ids=["train:0", "train:1"],
        class_ids=[0], samples_per_class=1,
    )
    assert selected[0] == [2]


def test_p0_parses_and_deduplicates_frac1_logs(tmp_path):
    command = (
        "python federated_main.py --trainer PromptFL --config-file vit_b16.yaml "
        "--seed 42 --split_seed 42 --num_users 30 --frac 1.0 --round 100 "
        "--local_epochs 3 --partition client-longtail --specialization_lambda 0.75 "
        "--intra_group_alpha 0.5 --head_leakage_scale 3.0 --tail_class_ratio 0.2\n"
    )
    body = command + "Tail accuracy (bottom 20 classes): 20.00%\nTail accuracy (bottom 20 classes): 18.00%\n"
    for name in ("one", "duplicate"):
        path = tmp_path / name / "run.log"
        path.parent.mkdir()
        path.write_text(body, encoding="utf-8")
    result = run_p0(tmp_path, tmp_path / "audit")
    assert result["eligible_unique_runs"] == 1
    inventory = (tmp_path / "audit/frac1_run_inventory.csv").read_text(encoding="utf-8")
    assert "2.0" in inventory
    assert "descriptive_frac1_only" in inventory
