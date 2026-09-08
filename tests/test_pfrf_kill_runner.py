import csv
import json
from types import SimpleNamespace

import pytest

import scripts.run_pfrf_kill_test as kill_runner
from scripts.run_pfrf_kill_test import (
    KILL_ROUNDS,
    child_environment,
    commands,
    gpu_assignments,
    load_smoke_gate,
    prepare,
    run_directory,
    validate_roots,
    write_metric_summary,
)
from utils.pfrf import PFRF_CONDITIONS


def _args(tmp_path):
    return SimpleNamespace(
        python_bin="python",
        data_root=tmp_path / "data",
        output_root=tmp_path / "kill",
        smoke_root=tmp_path / "smoke",
        num_users=30,
        conditions=list(PFRF_CONDITIONS),
        gpu_ids=None,
    )


def _write_passing_smoke(root):
    root.mkdir(parents=True, exist_ok=True)
    (root / "smoke_report.json").write_text(
        json.dumps(
            {
                "schema_version": "pfrf_smoke_report_v1",
                "status": "pass",
                "seed": 42,
                "rounds": 5,
                "conditions": list(PFRF_CONDITIONS),
                "checks": [{"name": "gate", "passed": True}],
            }
        ),
        encoding="utf-8",
    )


def test_kill_runner_builds_six_fresh_100_round_commands(tmp_path):
    args = _args(tmp_path)
    built = commands(args)
    assert [item["label"] for item in built] == list(PFRF_CONDITIONS)
    for item in built:
        command = item["command"]
        assert command[command.index("--round") + 1] == str(KILL_ROUNDS)
        assert command[command.index("--frac") + 1] == "1.0"
        assert command[command.index("--seed") + 1] == "42"
        assert "--resume" not in command
        assert "rounds100" in command[command.index("--client_schedule_file") + 1]
        assert command.index("DATALOADER.NUM_WORKERS") == len(command) - 2


def test_six_conditions_map_to_six_distinct_physical_gpus(tmp_path):
    args = _args(tmp_path)
    args.gpu_ids = [0, 1, 2, 3, 4, 5]
    assert gpu_assignments(args) == dict(zip(PFRF_CONDITIONS, range(6)))

    args.gpu_ids = [0, 1]
    with pytest.raises(ValueError, match="one GPU id"):
        gpu_assignments(args)

    args.gpu_ids = [0, 1, 2, 3, 4, 4]
    with pytest.raises(ValueError, match="distinct"):
        gpu_assignments(args)

    environment = child_environment(5)
    assert environment["CUDA_VISIBLE_DEVICES"] == "5"
    assert environment["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"


def test_single_condition_cli_selects_one_shard(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        ["run_pfrf_kill_test.py", "--stage", "train", "--condition", "pfrf_max"],
    )
    args = kill_runner.parse_args()
    assert args.condition == "pfrf_max"
    assert args.conditions == ["pfrf_max"]


def test_prepare_requires_a_passed_smoke_gate_and_disjoint_root(tmp_path):
    args = _args(tmp_path)
    _write_passing_smoke(args.smoke_root)
    path = prepare(args)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    assert manifest["fresh_initialization"] is True
    assert manifest["resume_from_smoke"] is False
    assert manifest["rounds"] == 100

    report = json.loads((args.smoke_root / "smoke_report.json").read_text())
    report["status"] = "fail"
    (args.smoke_root / "smoke_report.json").write_text(json.dumps(report))
    with pytest.raises(ValueError, match="did not pass"):
        load_smoke_gate(args.smoke_root)

    with pytest.raises(ValueError, match="disjoint"):
        validate_roots(args.smoke_root / "kill", args.smoke_root)


def test_metric_summary_uses_round_100_and_late_81_to_100(tmp_path):
    args = _args(tmp_path)
    for condition_index, condition in enumerate(PFRF_CONDITIONS):
        run_dir = run_directory(args.output_root, condition)
        run_dir.mkdir(parents=True, exist_ok=True)
        path = run_dir / "round_metrics.csv"
        fieldnames = [
            "epoch",
            "overall_acc",
            "non_tail_acc",
            "bottom20_tail_acc",
            "macro_per_class_acc",
            "macro_f1",
        ]
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for epoch in range(100):
                value = float(condition_index + epoch)
                writer.writerow(
                    {
                        "epoch": epoch,
                        "overall_acc": value,
                        "non_tail_acc": value,
                        "bottom20_tail_acc": value,
                        "macro_per_class_acc": value,
                        "macro_f1": value,
                    }
                )
    path = write_metric_summary(args)
    with path.open(encoding="utf-8", newline="") as handle:
        rows = {row["condition"]: row for row in csv.DictReader(handle)}
    ordinary = rows["ordinary_ce"]
    assert float(ordinary["final_bottom20_tail_acc"]) == pytest.approx(99.0)
    assert float(ordinary["late81_100_mean_bottom20_tail_acc"]) == pytest.approx(89.5)
    screening = json.loads(
        (args.output_root / "analysis" / "screening_summary.json").read_text()
    )
    assert screening["scientific_verdict"] == "pending_external_probe_and_additional_seeds"
