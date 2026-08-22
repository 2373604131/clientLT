from __future__ import annotations

import json
import tempfile
from pathlib import Path

from tools.carrier_access_audit.rewrite_protocol import frozen_rewrite_protocol
from tools.carrier_access_audit.rewrite_summarize import summarize_d1, summarize_d2
from tools.semantic_acquisition.common import file_sha256, write_csv, write_json


def test_rewrite_protocol_freezes_private_split_and_replay_without_test_selection():
    protocol = frozen_rewrite_protocol()
    assert protocol["private_split"]["write_slots"] == [0, 1, 2]
    assert protocol["private_split"]["evidence_slots"] == [3, 4]
    assert protocol["candidate_updates"]["normalization"] == "common_median_l2_norm_across_80_candidate_deltas"
    assert protocol["candidate_updates"]["d1_alpha"] == 0.5
    assert protocol["d2"]["sequence_lengths"] == [5, 10, 20]
    assert protocol["d2"]["per_update_beta"] == 0.05
    assert protocol["d2"]["test_metrics_used_for_sequence_selection"] is False
    assert len(protocol["protocol_hash"]) == 64


def test_rewrite_runtime_source_normalizes_updates_and_keeps_test_out_of_selection():
    root = Path(__file__).resolve().parents[1]
    source = (root / "tools" / "carrier_access_audit" / "rewrite_runtime.py").read_text(encoding="utf-8")
    launcher = (root / "scripts" / "run_post_write_rewrite_audit.py").read_text(encoding="utf-8")
    shell = (root / "scripts" / "run_post_write_rewrite_audit.sh").read_text(encoding="utf-8")
    assert "float(target_norm) / float(candidate_norm)" in source
    assert "write = unique[unique.slot.isin([0, 1, 2])]" in source
    assert "evidence = unique[unique.slot.isin([3, 4])]" in source
    assert '"test_metrics_used_for_sequence_selection": False' in source
    assert 'sort_values(\n            ["private_post_margin_gain"' in source
    assert "stable_seed(\"d2-blind\"" in source
    assert 'choices=["all", "protocol", "d1", "summarize-d1", "d2", "summarize-d2"]' in launcher
    assert '"$@"' in shell


def _write_d1_fixture(input_dir: Path) -> None:
    write_csv(input_dir / "tail_writer_fairness.csv", [
        {"tail_class": tail, "pass": True} for tail in range(80, 100)
    ])
    write_csv(input_dir / "candidate_norm_fairness.csv", [
        {"candidate_class": candidate, "pass": True} for candidate in range(80)
    ])
    write_csv(input_dir / "tail_writer_metrics.csv", [
        {"tail_class": tail, "direct_test_margin_gain": 1.0} for tail in range(80, 100)
    ])
    pre, post = [], []
    for tail in range(80, 100):
        for candidate in range(80):
            if candidate < 20:
                pre_effect, post_effect, transition = 1.0, -1.0, "donor_to_rewriter"
            elif candidate < 40:
                pre_effect, post_effect, transition = -1.0, 1.0, "rewriter_to_donor"
            elif candidate < 60:
                pre_effect, post_effect, transition = 1.0, 1.0, "donor_to_donor"
            else:
                pre_effect, post_effect, transition = -1.0, -1.0, "rewriter_to_rewriter"
            pre.append({
                "tail_class": tail, "candidate_class": candidate,
                "test_pre_margin_gain": pre_effect,
            })
            post.append({
                "tail_class": tail, "candidate_class": candidate,
                "private_post_margin_gain": post_effect,
                "test_post_margin_gain": post_effect,
                "sign_transition": transition,
            })
    write_csv(input_dir / "matched_pre_effects.csv", pre)
    write_csv(input_dir / "post_write_effects.csv", post)
    names = (
        "tail_writer_metrics.csv", "tail_writer_fairness.csv", "candidate_norm_fairness.csv",
        "matched_pre_effects.csv", "post_write_effects.csv",
    )
    write_json(input_dir / "runtime_contract.json", {
        "stage": "D1", "protocol": frozen_rewrite_protocol(),
        "result_hashes": {name: file_sha256(input_dir / name) for name in names},
    })


def test_d1_complete_fixture_detects_signed_post_write_turnover():
    with tempfile.TemporaryDirectory() as input_name, tempfile.TemporaryDirectory() as output_name:
        input_dir, output_dir = Path(input_name), Path(output_name)
        _write_d1_fixture(input_dir)
        result = summarize_d1(input_dir, output_dir)
        assert result["valid_comparison"] is True
        assert result["gate_pass"] is True
        assert result["verdict"] == "POST_WRITE_TURNOVER_AND_PRIVATE_DETECTION_SUPPORTED"
        assert result["tail_class_counts"]["at_least_one_donor_to_rewriter"] == 20


def test_d2_complete_fixture_links_private_risk_to_blind_forgetting():
    with tempfile.TemporaryDirectory() as input_name, tempfile.TemporaryDirectory() as output_name:
        input_dir, output_dir = Path(input_name), Path(output_name)
        d1_summary_path = output_dir / "d1_summary.json"
        write_json(d1_summary_path, {
            "valid_comparison": True,
            "valid_tail_classes": list(range(80, 100)),
        })
        rows = []
        for tail in range(80, 100):
            for length in (5, 10, 20):
                rows.extend([
                    {
                        "tail_class": tail, "condition": "low_risk", "sequence_length": length,
                        "draw": -1, "predicted_private_rewrite_risk": 0.0,
                        "test_forgetting": 0.1, "test_retention": 0.9,
                    },
                    {
                        "tail_class": tail, "condition": "high_risk", "sequence_length": length,
                        "draw": -1, "predicted_private_rewrite_risk": 1.0,
                        "test_forgetting": 0.9, "test_retention": 0.1,
                    },
                ])
                for draw in range(5):
                    forgetting = 0.25 + 0.1 * draw
                    rows.append({
                        "tail_class": tail, "condition": "blind", "sequence_length": length,
                        "draw": draw, "predicted_private_rewrite_risk": float(draw),
                        "test_forgetting": forgetting, "test_retention": 1.0 - forgetting,
                    })
        write_csv(input_dir / "replay_metrics.csv", rows)
        write_csv(input_dir / "runtime_fairness.csv", [
            {"tail_class": tail, "pass": True} for tail in range(80, 100)
        ])
        write_json(input_dir / "runtime_contract.json", {
            "stage": "D2", "protocol": frozen_rewrite_protocol(),
            "valid_tail_classes": list(range(80, 100)),
            "d1_summary_hash": file_sha256(d1_summary_path),
            "result_hashes": {
                name: file_sha256(input_dir / name)
                for name in ("replay_metrics.csv", "runtime_fairness.csv")
            },
        })
        result = summarize_d2(input_dir, d1_summary_path, output_dir / "d2")
        assert result["gate_pass"] is True
        assert result["verdict"] == "REWRITE_RISK_PREDICTS_RETENTION"
        assert result["tail_class_counts"]["positive_blind_risk_forgetting_spearman"] == 20
