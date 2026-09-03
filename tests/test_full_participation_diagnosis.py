import csv
import json
from pathlib import Path
from types import SimpleNamespace

from scripts.analyze_full_participation_diagnosis import analyze
from scripts.run_full_participation_diagnosis import _full_schedule, common_command


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _count_rows(matrix: list[list[int]]) -> list[dict]:
    return [
        {"client_id": client_id, **{f"class_{class_id}": value for class_id, value in enumerate(row)}}
        for client_id, row in enumerate(matrix)
    ]


def _make_run(path: Path, matrix: list[list[int]], final: float, best: float) -> None:
    _write_json(
        path / "cliplora_initialization_audit.json",
        {
            "common_init_seed": 424242,
            "initial_lora_sha256": "same-anchor",
            "global_local_initialization_equal": True,
        },
    )
    _write_csv(path / "client_class_counts.csv", _count_rows(matrix))
    weight_rows = []
    for round_id in range(1, 81):
        for client_id in range(30):
            weight_rows.append(
                {
                    "communication_round": round_id,
                    "client_id": client_id,
                    "aggregation_weight": 1.0 / 30.0,
                }
            )
    _write_csv(path / "lora_aggregation_weights.csv", weight_rows)
    metric_rows = []
    for epoch in range(80):
        value = best if epoch == 0 else final
        metric_rows.append({"epoch": epoch, "bottom20_tail_acc": value})
    _write_csv(path / "round_metrics.csv", metric_rows)


def test_different_topologies_with_equal_nc_nk_pass_and_keep_decisions_separate(tmp_path):
    output_root = tmp_path / "full"
    partial_root = tmp_path / "partial"
    _write_json(
        output_root / "frozen_protocol.json",
        {"equivalence_threshold_pp": 2.0},
    )

    clientlt = [[1 for _ in range(100)] for _ in range(30)]
    matched = [[1 for _ in range(100)] for _ in range(30)]
    clientlt[0][0], clientlt[0][1], clientlt[1][0], clientlt[1][1] = 2, 0, 0, 2
    matched[0][0], matched[0][1], matched[1][0], matched[1][1] = 0, 2, 2, 0
    assert clientlt != matched

    _make_run(output_root / "clientlt", clientlt, final=10.0, best=15.0)
    _make_run(output_root / "matched_dirichlet", matched, final=13.0, best=18.0)
    _write_json(
        partial_root / "analysis" / "validation_summary.json",
        {
            "raw_outcomes": {
                "clientlt": {
                    "final": {"tail": 8.0},
                    "best_to_final_tail_drop": 9.0,
                },
                "matched_dirichlet": {
                    "final": {"tail": 21.85},
                    "best_to_final_tail_drop": 3.0,
                },
            }
        },
    )
    _write_json(
        partial_root / "clientlt" / "functional_coverage" / "protocol.json",
        {"common_lora_anchor_sha256": "same-anchor"},
    )

    result = analyze(output_root, partial_root)

    assert result["verdict"] == "FINAL_GAP_WITHOUT_EXTRA_COLLAPSE"
    assert result["primary_results"]["frac1p0"]["final_tail_accuracy_gap_pp"] == 3.0
    assert result["primary_results"]["frac1p0"]["best_to_final_drop_gap_pp"] == 0.0
    assert result["required_audits"]["global_class_counts_equal"] is True
    assert result["required_audits"]["client_total_samples_equal"] is True


def test_full_runner_is_minimal_frac_one_plain_fedavg(tmp_path):
    args = SimpleNamespace(
        python_bin="python",
        data_root=Path("DATA"),
        output_root=tmp_path,
        lr=0.001,
        test_batch_size=100,
        common_init_seed=424242,
        num_workers=4,
    )
    command = common_command(
        args,
        "client-longtail",
        tmp_path / "clientlt",
        {"position": "top3", "rank": 2, "alpha": 1, "params": ["q", "v"]},
    )
    joined = " ".join(command)

    assert len(_full_schedule()) == 80
    assert all(round_clients == list(range(30)) for round_clients in _full_schedule())
    assert "--frac 1.0" in joined
    assert "--cliplora_aggregation fedavg" in joined
    assert "--functional_coverage_validation_enable False" in joined
    assert "--cliplora_common_init_seed 424242" in joined
    assert "retention_ratio" not in joined
