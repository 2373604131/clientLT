import csv
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_g0_d1 import (
    CONFIGS,
    build_d1_command,
    build_g0_command,
    freeze_lora,
    summarize_d1,
)
from utils.experiment_d import (
    PER_CLASS_FIELDS,
    ROUND_SUMMARY_FIELDS,
    class_ids_from_tail_ratio,
    support_clients_for_class,
)
from utils.g0_lora_probe import effective_ba_delta_norm, select_probe_clients


def _launcher_args(tmp_path: Path):
    return SimpleNamespace(
        output_root=tmp_path,
        data_root=Path("DATA"),
        python_bin="python",
        lr=0.001,
        test_batch_size=128,
        num_workers=4,
        random_support_count=20,
    )


def _value_after(command, flag):
    return command[command.index(flag) + 1]


def test_probe_selection_uses_all_tail_clients_and_matched_unique_heads():
    counts = {client_id: torch.tensor([client_id + 1, 1]) for client_id in range(30)}
    heads, tails = select_probe_clients(counts, num_users=30, tail_client_ratio=0.1)

    assert tails == [27, 28, 29]
    assert len(heads) == 3
    assert len(set(heads)) == 3
    assert all(0 <= client_id < 27 for client_id in heads)


def test_effective_ba_norm_uses_executable_product():
    before = {
        "block.q_lora_A": torch.tensor([[1.0, 0.0]]),
        "block.q_lora_B": torch.zeros(2, 1),
    }
    after = {
        "block.q_lora_A": torch.tensor([[1.0, 0.0]]),
        "block.q_lora_B": torch.tensor([[3.0], [4.0]]),
    }

    assert effective_ba_delta_norm(before, after, scaling=0.5) == pytest.approx(2.5)


def test_g0_commands_are_exactly_old_and_moderate_candidate(tmp_path):
    args = _launcher_args(tmp_path)
    _, old = build_g0_command(args, "old_r2")
    _, candidate = build_g0_command(args, "candidate_r4")

    assert _value_after(old, "--cliplora_position") == "top3"
    assert _value_after(old, "--cliplora_rank") == "2"
    assert _value_after(old, "--cliplora_alpha") == "1"
    assert old[old.index("--cliplora_params") + 1 : old.index("--g0_probe_enable")] == ["q", "v"]
    assert _value_after(candidate, "--cliplora_position") == "up"
    assert _value_after(candidate, "--cliplora_rank") == "4"
    assert _value_after(candidate, "--cliplora_alpha") == "2"
    assert candidate[
        candidate.index("--cliplora_params") + 1 : candidate.index("--g0_probe_enable")
    ] == ["q", "k", "v"]
    assert _value_after(candidate, "--round") == "1"
    assert _value_after(candidate, "--g0_probe_enable") == "True"


def _write_probe_summary(root: Path, config_id: str, tail_gain: float, passed=True):
    path = root / "g0" / config_id / "g0_probe" / "g0_config_summary.json"
    path.parent.mkdir(parents=True)
    payload = {
        "client_count": 6,
        "tail_client_count": 3,
        "all_finite": passed,
        "mean_train_loss_relative_drop": 0.1 if passed else -0.1,
        "positive_tail_client_count": 3 if passed else 0,
        "mean_prediction_flip_rate": 0.1 if passed else 0.0,
        "mean_abs_logit_change": 0.2 if passed else 0.0,
        "mean_effective_ba_delta_norm": 1.0 if passed else 0.0,
        "median_tail_margin_gain": tail_gain,
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_freeze_selects_passing_config_with_larger_tail_margin(tmp_path):
    _write_probe_summary(tmp_path, "old_r2", 0.1)
    _write_probe_summary(tmp_path, "candidate_r4", 0.3)

    frozen = freeze_lora(tmp_path)

    assert frozen["verdict"] == "PASS"
    assert frozen["selected_config_id"] == "candidate_r4"
    assert frozen["selected_config"] == CONFIGS["candidate_r4"]
    assert (tmp_path / "lora_freeze.json").exists()


def test_d1_command_reads_frozen_lora_and_enables_three_round_audit(tmp_path):
    args = _launcher_args(tmp_path)
    frozen = {
        "selected_config_id": "candidate_r4",
        "selected_config": CONFIGS["candidate_r4"],
    }

    _, command = build_d1_command(args, frozen)

    assert _value_after(command, "--round") == "80"
    assert _value_after(command, "--experimentD_rounds") == "20,50,80"
    assert _value_after(command, "--experimentD_support_min_fraction") == "0.1"
    assert _value_after(command, "--experimentD_random_support_count") == "20"
    assert _value_after(command, "--cliplora_position") == "up"
    assert _value_after(command, "--cliplora_rank") == "4"


def test_support_threshold_matches_capt_strict_fraction_rule():
    counts = {
        0: torch.tensor([10, 90]),
        1: torch.tensor([11, 89]),
        2: torch.tensor([1, 0]),
    }

    assert support_clients_for_class(counts, [0, 1, 2], 0, 0.1) == [1, 2]
    assert support_clients_for_class(counts, [0, 1, 2], 0, 0.0) == [0, 1, 2]


def test_tail_split_prefers_larger_class_id_on_realized_count_tie():
    head, tail = class_ids_from_tail_ratio(torch.tensor([9, 5, 5, 1]), 0.5)

    assert tail == [3, 2]
    assert head == [0, 1]


def test_d1_schema_contains_non_support_random_and_head_safety_metrics():
    assert {
        "acc_non_support_actual",
        "dilution_gap",
        "tail_gain_support_normalized_vs_fedavg",
        "tail_gain_support_normalized_vs_random_p95",
        "head_damage_support_normalized_vs_fedavg",
        "h_gain_support_normalized_vs_fedavg",
    }.issubset(PER_CLASS_FIELDS)
    assert {
        "mean_tail_gain_support_normalized_vs_fedavg",
        "support_normalized_beats_random_p95_rate",
        "mean_head_damage_support_normalized_vs_fedavg",
        "mean_h_gain_support_normalized_vs_fedavg",
    }.issubset(ROUND_SUMMARY_FIELDS)


def test_d1_summary_requires_and_aggregates_exact_three_rounds(tmp_path):
    experiment_dir = tmp_path / "d1_seed42" / "experiment_d"
    path = experiment_dir / "experiment_d_round_summary.csv"
    experiment_dir.mkdir(parents=True)
    rows = []
    for communication_round in (20, 50, 80):
        rows.append(
            {
                "communication_round": communication_round,
                "mean_tail_gain_support_normalized_vs_fedavg": 2.0,
                "mean_head_damage_support_normalized_vs_fedavg": 0.2,
                "mean_h_gain_support_normalized_vs_fedavg": 0.5,
                "support_normalized_beats_random_p95_rate": 0.7,
                "valid_support_class_rate": 1.0,
            }
        )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    per_class = []
    for communication_round in (20, 50, 80):
        for class_id in range(80, 100):
            per_class.append({
                "communication_round": communication_round,
                "class_id": class_id,
                "support_valid": "True",
                "support_normalized_beats_random_p95": "True",
                "tail_gain_support_normalized_vs_fedavg": 2.0,
                "tail_gain_support_normalized_vs_random_p95": 1.0,
                "head_damage_support_normalized_vs_fedavg": 0.2,
                "h_gain_support_normalized_vs_fedavg": 0.5,
                "num_support_clients": 1,
            })
    with (experiment_dir / "experiment_d_per_class.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(per_class[0]))
        writer.writeheader()
        writer.writerows(per_class)

    report = summarize_d1(
        tmp_path,
        {
            "selected_config_id": "candidate_r4",
            "selected_config": CONFIGS["candidate_r4"],
        },
    )

    assert report["verdict"] == "D1_FULL_PASS"
    assert report["conditional_valid_class_beats_random_p95_rate"] == 1.0
    saved = json.loads((tmp_path / "d1_summary" / "d1_verdict.json").read_text())
    assert saved["rounds"] == [20, 50, 80]


def test_d1_summary_separates_supported_phenomenon_from_coverage_gap(tmp_path):
    experiment_dir = tmp_path / "d1_seed42" / "experiment_d"
    experiment_dir.mkdir(parents=True)
    summary = []
    classes = []
    for communication_round in (20, 50, 80):
        summary.append({
            "communication_round": communication_round,
            "mean_tail_gain_support_normalized_vs_fedavg": 20.0,
            "mean_head_damage_support_normalized_vs_fedavg": 3.0,
            "mean_h_gain_support_normalized_vs_fedavg": 5.0,
            "support_normalized_beats_random_p95_rate": 0.4,
            "valid_support_class_rate": 0.55,
        })
        for offset, class_id in enumerate(range(80, 100)):
            valid = offset < 11
            classes.append({
                "communication_round": communication_round,
                "class_id": class_id,
                "support_valid": str(valid),
                "support_normalized_beats_random_p95": str(valid and offset < 8),
                "tail_gain_support_normalized_vs_fedavg": 20.0 if valid else "nan",
                "tail_gain_support_normalized_vs_random_p95": 2.0 if valid else "nan",
                "head_damage_support_normalized_vs_fedavg": 3.0 if valid else "nan",
                "h_gain_support_normalized_vs_fedavg": 5.0 if valid else "nan",
                "num_support_clients": 1 if valid else 0,
            })
    for name, rows in (
        ("experiment_d_round_summary.csv", summary),
        ("experiment_d_per_class.csv", classes),
    ):
        with (experiment_dir / name).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    report = summarize_d1(
        tmp_path,
        {"selected_config_id": "candidate_r4", "selected_config": CONFIGS["candidate_r4"]},
    )

    assert report["phenomenon_pass"] is True
    assert report["support_rule_coverage_pass"] is False
    assert report["method_ready"] is False
    assert report["verdict"] == "D1_SUPPORTED_WITH_COVERAGE_GAP"
