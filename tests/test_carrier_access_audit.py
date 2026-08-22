from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np

from tools.carrier_access_audit.footprint import analyse
from tools.carrier_access_audit.protocol import frozen_protocol
from tools.carrier_access_audit.summarize import summarize_b, summarize_c
from tools.carrier_access_audit.statistics import (
    effective_count,
    normalized_positive_entropy,
    spearman,
    weighted_pairwise_cosine_diversity,
)
from tools.client_update_audit.protocol import frozen_protocol as e2_protocol
from tools.semantic_acquisition.common import write_csv, write_json


def test_protocol_freezes_single_seed_without_test_selection():
    protocol = frozen_protocol()
    assert protocol["data_seed"] == 42
    assert protocol["dataset"]["tail_classes"] == list(range(80, 100))
    assert protocol["experiment_b"]["candidate_train_samples_per_class"] == 12
    assert protocol["experiment_c"]["private_readapt"]["test_labels_used_for_selection"] is False
    assert protocol["experiment_c"]["fairness"]["optimizer_trajectory_is_treatment"] is True
    assert len(protocol["protocol_hash"]) == 64


def test_footprint_statistics_have_expected_geometry():
    assert np.isclose(effective_count([1, 1]), 2.0)
    assert np.isclose(effective_count([3, 1]), 1.6)
    assert np.isclose(normalized_positive_entropy([1, 1, 0, 0]), 0.5)
    diversity = weighted_pairwise_cosine_diversity([[1, 0], [0, 1]], [1, 1])
    assert np.isclose(diversity, 1.0)
    assert np.isclose(spearman([1, 2, 3], [3, 2, 1]), -1.0)


def test_runtime_source_keeps_private_and_test_evidence_separate():
    source = (Path(__file__).resolve().parents[1] / "tools" / "carrier_access_audit" / "runtime.py").read_text(encoding="utf-8")
    assert '"selection_used_test_metrics": False' in source
    assert "private_margin_gain" in source
    assert "test_margin_gain_in_b_audit_only" in source
    assert "(0.5 * F.cross_entropy(model(images), labels)).backward()" in source
    assert '"gradient_calls": gradient_calls' in source


def test_launcher_regenerates_only_a_missing_semantic_prior():
    root = Path(__file__).resolve().parents[1]
    source = (root / "scripts" / "run_carrier_access_audit.py").read_text(encoding="utf-8")
    shell = (root / "scripts" / "run_carrier_access_audit.sh").read_text(encoding="utf-8")
    assert "if not args.similarity_file.is_file()" in source
    assert '"tools.carrier_access_audit.semantic_prior"' in source
    assert '"$@"' in shell


def test_b_and_c_summarizers_use_preregistered_primary_endpoint():
    source = (Path(__file__).resolve().parents[1] / "tools" / "carrier_access_audit" / "summarize.py").read_text(encoding="utf-8")
    assert "values[:10]" in source
    assert "values[-10:]" in source
    assert 'primary = summaries[name]["test_margin_gain"]' in source
    assert "positive_count" in source


def test_experiment_a_complete_fixture_reports_broader_dirichlet_carriers():
    with tempfile.TemporaryDirectory() as source_name, tempfile.TemporaryDirectory() as output_name:
        source, output = Path(source_name), Path(output_name)
        write_json(source / "runtime_contract.json", {
            "stage": "e2a", "server_aggregation_called": False,
            "protocol": e2_protocol(), "theta0_hash": "fixture",
        })
        write_csv(source / "runtime_fairness.csv", [{"pass": True}])
        all_rows = []
        tail_rows = []
        for topology, client_id in (("dirichlet", 0), ("clientlt", 27)):
            for class_id in range(100):
                gain = 1.0 if topology == "dirichlet" or class_id < 50 else -1.0
                all_rows.append({
                    "data_seed": 42, "topology": topology, "condition": "natural",
                    "client_id": client_id, "local_epoch": 3, "class_id": class_id,
                    "margin_gain": gain, "locally_supported": class_id >= 80,
                })
            for tail_class in range(80, 100):
                tail_rows.append({
                    "data_seed": 42, "topology": topology, "condition": "natural",
                    "client_id": client_id, "local_epoch": 3, "tail_class": tail_class,
                    "tail_sample_count": 1,
                    "worst_neighbor_margin_gain": 1.0 if topology == "dirichlet" else -1.0,
                })
        write_csv(source / "local_all_class_footprints.csv", all_rows)
        write_csv(source / "local_tail_metrics.csv", tail_rows)
        result = analyse(source, output)
        assert result["verdict"] == "DIRICHLET_CARRIERS_FUNCTIONALLY_BROADER"
        assert result["descriptive_only"] is True


def test_experiment_b_and_c_complete_fixtures_follow_effect_direction():
    with tempfile.TemporaryDirectory() as b_name, tempfile.TemporaryDirectory() as c_name, tempfile.TemporaryDirectory() as out_name:
        b_dir, c_dir, output = Path(b_name), Path(c_name), Path(out_name)
        write_json(b_dir / "runtime_contract.json", {"stage": "B", "protocol": frozen_protocol()})
        write_csv(b_dir / "runtime_fairness.csv", [
            {"candidate_class": candidate, "pass": True} for candidate in range(80)
        ])
        matrix = []
        for tail_class in range(80, 100):
            for rank in range(1, 81):
                effect = 1.0 if rank <= 10 else -0.1
                matrix.append({
                    "tail_class": tail_class, "candidate_class": rank - 1,
                    "semantic_rank": rank, "cosine_similarity": 1.0 - rank / 100.0,
                    "private_margin_gain": effect, "test_margin_gain": effect,
                    "test_nll_gain": effect, "test_worst_neighbor_margin_gain": effect,
                })
        write_csv(b_dir / "transfer_matrix.csv", matrix)
        b_result = summarize_b(b_dir, output / "b")
        assert b_result["verdict"] == "SEMANTIC_PRIOR_ENRICHES_FUNCTIONAL_DONORS"

        write_json(c_dir / "runtime_contract.json", {"stage": "C", "protocol": frozen_protocol()})
        write_csv(c_dir / "runtime_fairness.csv", [
            {"tail_class": tail_class, "pass": True} for tail_class in range(80, 100)
        ])
        condition_values = {
            "tail_only": 0.1,
            "joint_related": 1.0,
            "separate_merge_related": 0.0,
            "separate_readapt_related": 0.5,
            "joint_unrelated": -0.2,
        }
        placement = []
        for tail_class in range(80, 100):
            for condition, value in condition_values.items():
                placement.append({
                    "tail_class": tail_class, "condition": condition,
                    "test_margin_gain": value, "test_nll_gain": value,
                    "test_worst_neighbor_margin_gain": value,
                    "test_accuracy_gain": value, "lora_update_l2": 1.0,
                    "chosen_lambda": 0.5 if condition == "separate_readapt_related" else 1.0,
                })
        write_csv(c_dir / "placement_metrics.csv", placement)
        c_result = summarize_c(c_dir, output / "c")
        assert c_result["verdict"] == "JOINT_AND_PRIVATE_READAPT_BOTH_SUPPORTED"
