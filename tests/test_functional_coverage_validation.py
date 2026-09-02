import csv
import json
from pathlib import Path

from scripts.analyze_functional_coverage_validation import analyze
from utils.functional_coverage_validation import parse_validation_rounds


def _csv(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _make_run(root: Path, partition: str, coverage: float, tail_final: float, tail_best: float):
    run = root / partition
    protocol = {
        "schema_version": "functional_coverage_validation_v1",
        "selected_rounds": [1, 2],
        "tail_classes": list(range(80, 100)),
        "samples_per_tail_class": 2,
        "probe_manifest_hash": "probe",
        "common_lora_anchor_sha256": "theta0",
        "gain_epsilon": 0.0,
        "lora_keys": ["lora_a"],
    }
    (run / "functional_coverage").mkdir(parents=True, exist_ok=True)
    (run / "functional_coverage" / "protocol.json").write_text(
        json.dumps(protocol), encoding="utf-8"
    )
    _csv(
        run / "functional_coverage" / "frozen_boundary_weights.csv",
        [
            {
                "class_id": class_id,
                "competitor_class": competitor,
                "confusion_weight": 1.0 / 99.0,
            }
            for class_id in range(80, 100)
            for competitor in range(100)
            if competitor != class_id
        ],
    )
    _csv(
        run / "client_class_counts.csv",
        [
            {"client_id": client, **{f"class_{class_id}": 1 for class_id in range(100)}}
            for client in range(30)
        ],
    )
    _csv(
        run / "lora_aggregation_weights.csv",
        [
            {
                "epoch_index": round_id - 1,
                "communication_round": round_id,
                "partition": partition,
                "aggregation": "fedavg",
                "client_id": client,
                "client_num_samples": 100,
                "num_supported_tail_classes": 0,
                "aggregation_weight": 0.5,
            }
            for round_id in (1, 2)
            for client in (0, 1)
        ],
    )
    _csv(
        run / "round_metrics.csv",
        [
            {
                "epoch": epoch,
                "overall_acc": 50.0,
                "non_tail_acc": 60.0,
                "bottom20_tail_acc": tail,
                "macro_f1": 45.0,
            }
            for epoch, tail in ((0, tail_best), (1, tail_final))
        ],
    )
    for epoch, tail in ((0, tail_best), (1, tail_final)):
        _csv(
            run / f"per_class_accuracy_epoch_{epoch}.csv",
            [
                {"class_id": class_id, "per_class_acc": tail if class_id >= 80 else 60.0}
                for class_id in range(100)
            ],
        )
    _csv(
        run / "functional_coverage" / "coverage_per_class_round.csv",
        [
            {
                "communication_round": round_id,
                "partition": partition,
                "class_id": class_id,
                "selected_client_count": 2,
                "available_functional_coverage": coverage + 0.05,
                "realized_functional_coverage": coverage,
                "coverage_retention_ratio": coverage / (coverage + 0.05),
                "available_positive_boundary_count": 50,
                "realized_positive_boundary_count": 45,
            }
            for round_id in (1, 2)
            for class_id in range(80, 100)
        ],
    )


def test_parse_validation_rounds_requires_final_round():
    assert parse_validation_rounds("1,5,10", 10) == [1, 5, 10]


def test_direct_scorecard_full_chain(tmp_path):
    _make_run(tmp_path, "clientlt", coverage=0.40, tail_final=30.0, tail_best=40.0)
    _make_run(
        tmp_path,
        "matched_dirichlet",
        coverage=0.55,
        tail_final=45.0,
        tail_best=48.0,
    )
    result = analyze(tmp_path)
    assert result["verdict"] == "FULL_CHAIN_SUPPORTED"
    assert result["primary_results"]["matched_minus_clientlt_realized_coverage"] > 0
    assert result["primary_results"]["matched_minus_clientlt_final_tail_accuracy_pp"] > 0
    assert result["primary_results"]["clientlt_minus_matched_best_to_final_tail_drop_pp"] > 0

