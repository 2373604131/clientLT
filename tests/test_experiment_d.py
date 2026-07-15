import csv
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.experiment_d import (
    PER_CLASS_FIELDS,
    build_non_support_actual_state,
    build_support_actual_state,
    build_support_normalized_state,
    evaluate_state_per_class,
    fedavg_client_weights,
    parse_experiment_d_rounds,
    reconstruct_full_fedavg_state,
    should_log_experiment_d,
    support_clients_for_class,
    tail_specialist_clients,
    validate_full_participation,
    validate_support_decomposition,
    verify_fedavg_reconstruction,
)


SUMMARY_SCRIPT = REPO_ROOT / "scripts" / "summarize_experimentD_local_epochs.py"
PLOT_SCRIPT = REPO_ROOT / "scripts" / "plot_experimentD_local_epochs.py"
PILOT_SCRIPT = REPO_ROOT / "scripts" / "experimentD_local_epochs_pilot.sh"


def _state(value, *, key="p"):
    return {key: torch.as_tensor(value, dtype=torch.float32)}


def test_support_actual_uses_original_fedavg_denominator():
    before = _state(0.0)
    local_weights = [_state(2.0), _state(4.0)]
    weights = fedavg_client_weights([0, 1], [90, 10])

    support_actual = build_support_actual_state(before, local_weights, [0, 1], [1], weights)

    assert support_actual["p"].item() == pytest.approx(0.4)


def test_support_normalized_uses_support_denominator():
    before = _state(0.0)
    local_weights = [_state(2.0), _state(4.0)]

    normalized = build_support_normalized_state(before, local_weights, [1], [90, 10])

    assert normalized["p"].item() == pytest.approx(4.0)


def test_full_fedavg_reconstruction_matches_total_sample_weights():
    before = _state(0.0)
    local_weights = [_state(2.0), _state(4.0)]
    weights = fedavg_client_weights([0, 1], [90, 10])

    full = reconstruct_full_fedavg_state(before, local_weights, [0, 1], weights)

    assert full["p"].item() == pytest.approx(2.2)


def test_all_clients_support_actual_and_normalized_equal_full_fedavg():
    before = _state(0.0)
    local_weights = [_state(2.0), _state(4.0)]
    selected = [0, 1]
    datanumber_client = [90, 10]
    weights = fedavg_client_weights(selected, datanumber_client)

    full = reconstruct_full_fedavg_state(before, local_weights, selected, weights)
    support_actual = build_support_actual_state(before, local_weights, selected, selected, weights)
    normalized = build_support_normalized_state(before, local_weights, selected, datanumber_client)

    assert support_actual["p"].item() == pytest.approx(full["p"].item())
    assert normalized["p"].item() == pytest.approx(full["p"].item())


def test_support_weights_use_total_samples_not_class_sample_counts():
    before = _state(0.0)
    local_weights = [_state(2.0), _state(4.0)]
    selected = [0, 1]
    datanumber_client = [90, 10]
    weights = fedavg_client_weights(selected, datanumber_client)
    counts_a = {0: torch.tensor([3, 0]), 1: torch.tensor([2, 0])}
    counts_b = {0: torch.tensor([300, 0]), 1: torch.tensor([1, 0])}

    support_a = support_clients_for_class(counts_a, selected, 0)
    support_b = support_clients_for_class(counts_b, selected, 0)
    state_a = build_support_actual_state(before, local_weights, selected, support_a, weights)
    state_b = build_support_actual_state(before, local_weights, selected, support_b, weights)

    assert support_a == support_b == [0, 1]
    assert state_a["p"].item() == pytest.approx(state_b["p"].item())


def test_full_state_dict_is_complete_and_inputs_are_unmodified():
    before = {
        "prompt_learner.general_ctx": torch.tensor([0.0, 1.0]),
        "prompt_learner.class_aware_ctx": torch.tensor([[1.0], [2.0]]),
        "image_encoder.frozen": torch.tensor([5.0]),
        "num_batches_tracked": torch.tensor(7, dtype=torch.long),
    }
    local_weights = [
        {
            "prompt_learner.general_ctx": torch.tensor([2.0, 3.0]),
            "prompt_learner.class_aware_ctx": torch.tensor([[3.0], [5.0]]),
            "image_encoder.frozen": torch.tensor([9.0]),
            "num_batches_tracked": torch.tensor(99, dtype=torch.long),
        },
        {
            "prompt_learner.general_ctx": torch.tensor([4.0, 5.0]),
            "prompt_learner.class_aware_ctx": torch.tensor([[7.0], [11.0]]),
            "image_encoder.frozen": torch.tensor([13.0]),
            "num_batches_tracked": torch.tensor(100, dtype=torch.long),
        },
    ]
    before_snapshot = {key: value.clone() for key, value in before.items()}
    weights = fedavg_client_weights([0, 1], [1, 3])

    full = reconstruct_full_fedavg_state(before, local_weights, [0, 1], weights)

    assert set(full) == set(before)
    assert torch.allclose(full["prompt_learner.general_ctx"], torch.tensor([3.5, 4.5]))
    assert torch.allclose(full["prompt_learner.class_aware_ctx"], torch.tensor([[6.0], [9.5]]))
    assert torch.allclose(full["image_encoder.frozen"], torch.tensor([12.0]))
    assert full["num_batches_tracked"].item() == 7
    for key in before:
        assert torch.equal(before[key], before_snapshot[key])


def test_verify_fedavg_reconstruction_consistency():
    before = _state([0.0, 0.0])
    local_weights = [_state([2.0, 6.0]), _state([4.0, 8.0])]
    weights = fedavg_client_weights([0, 1], [90, 10])
    reconstructed = reconstruct_full_fedavg_state(before, local_weights, [0, 1], weights)
    expected = _state([2.2, 6.2])

    assert verify_fedavg_reconstruction(reconstructed, expected) == []


def test_support_plus_non_support_delta_decomposition():
    before = _state(torch.zeros(2))
    local_weights = [_state(torch.ones(2) * 2), _state(torch.ones(2) * 4)]
    selected = [0, 1]
    weights = fedavg_client_weights(selected, [90, 10])

    full = reconstruct_full_fedavg_state(before, local_weights, selected, weights)
    support = build_support_actual_state(before, local_weights, selected, [1], weights)
    non_support = build_non_support_actual_state(before, local_weights, selected, [1], weights)

    assert validate_support_decomposition(before, support, non_support, full) == []


def test_full_participation_validation_fails_when_frac_not_one():
    args = type("Args", (), {"frac": 0.5, "num_users": 2})()
    with pytest.raises(RuntimeError, match="full participation"):
        validate_full_participation(args, [0])


def test_tail_specialists_for_30_clients_ratio_point_one():
    args = type(
        "Args",
        (),
        {"partition": "client-longtail", "num_users": 30, "tail_client_ratio": 0.1},
    )()
    assert tail_specialist_clients(args) == [27, 28, 29]


def test_round_parser_uses_one_based_communication_rounds():
    args = type("Args", (), {"experimentD_enable": True, "experimentD_rounds": "5,10,20,30"})()
    assert parse_experiment_d_rounds(args.experimentD_rounds) == {5, 10, 20, 30}
    assert should_log_experiment_d(args, 4)
    assert should_log_experiment_d(args, 9)
    assert should_log_experiment_d(args, 19)
    assert should_log_experiment_d(args, 29)
    assert not should_log_experiment_d(args, 0)


class _TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([[1.0], [2.0]]))

    def forward(self, x):
        return x @ self.weight.t()


class _FailingTrainer:
    def __init__(self):
        self.model = _TinyModel()
        self.test_loader = [(torch.ones(1, 1), torch.tensor([0]))]

    def parse_batch_test(self, batch):
        return batch

    def model_inference(self, inputs):
        raise RuntimeError("synthetic eval failure")


def test_evaluate_state_restores_model_after_exception():
    trainer = _FailingTrainer()
    original = {key: value.clone() for key, value in trainer.model.state_dict().items()}
    target = {"weight": torch.tensor([[10.0], [20.0]])}

    with pytest.raises(RuntimeError, match="synthetic eval failure"):
        evaluate_state_per_class(trainer, target, [0])

    for key, value in trainer.model.state_dict().items():
        assert torch.equal(value, original[key])


def test_per_class_csv_fields_include_accuracy_gain_columns():
    required = {
        "acc_before",
        "acc_support_actual",
        "acc_support_normalized",
        "acc_all",
        "gain_support_actual",
        "gain_support_normalized",
        "gain_all",
        "offset_gap",
        "support_fedavg_weight",
    }
    assert required.issubset(set(PER_CLASS_FIELDS))


def _run_bash_dry_run(env):
    bash = shutil.which("bash")
    if not bash:
        pytest.skip("bash is unavailable")
    result = subprocess.run(
        [bash, str(PILOT_SCRIPT)],
        cwd=REPO_ROOT,
        env={**os.environ, **env},
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if result.returncode != 0 and "wslstore" in (result.stdout + result.stderr).lower():
        pytest.skip("bash/WSL is unavailable")
    return result


def test_experiment_d_pilot_dry_run_default_has_two_clientlt_runs():
    result = _run_bash_dry_run({
        "DRY_RUN": "1",
        "GPUS": "3 4",
        "INCLUDE_LOCAL_E5": "0",
        "DATA": "DATA/",
        "PYTHON_BIN": "python",
    })
    stdout = result.stdout.replace('"', "")
    assert result.returncode == 0, result.stdout + result.stderr
    assert "[ExperimentD Pilot 01/2]" in stdout
    assert "[ExperimentD Pilot 02/2]" in stdout
    assert "noniid-labeldir-fine" not in stdout
    assert "--partition client-longtail" in stdout
    assert "--round 30" in stdout
    assert "--local_epochs 1" in stdout
    assert "--local_epochs 3" in stdout
    assert "--local_epochs 5" not in stdout
    assert "--experimentD_rounds 5,10,20,30" in stdout


def test_experiment_d_pilot_dry_run_can_include_local_epochs_five():
    result = _run_bash_dry_run({
        "DRY_RUN": "1",
        "GPUS": "3 4 5",
        "INCLUDE_LOCAL_E5": "1",
        "DATA": "DATA/",
        "PYTHON_BIN": "python",
    })
    stdout = result.stdout.replace('"', "")
    assert result.returncode == 0, result.stdout + result.stderr
    assert "[ExperimentD Pilot 01/3]" in stdout
    assert "[ExperimentD Pilot 03/3]" in stdout
    assert "--local_epochs 5" in stdout


def test_summarize_missing_root_does_not_create_fake_outputs(tmp_path):
    missing_root = tmp_path / "missing"
    out = tmp_path / "summary"
    result = subprocess.run(
        [
            sys.executable,
            str(SUMMARY_SCRIPT),
            "--root",
            str(missing_root),
            "--output-dir",
            str(out),
        ],
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert result.returncode == 1
    assert "does not exist" in result.stdout
    assert not out.exists()


def test_plot_missing_input_does_not_create_fake_figure(tmp_path):
    out = tmp_path / "figures"
    result = subprocess.run(
        [
            sys.executable,
            str(PLOT_SCRIPT),
            "--all-runs",
            str(tmp_path / "missing.csv"),
            "--output-dir",
            str(out),
        ],
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert result.returncode == 1
    assert "does not exist" in result.stdout
    assert not out.exists()


def test_summarizer_writes_outputs_from_synthetic_run(tmp_path):
    root = tmp_path / "ExperimentD_LocalEpochPilot"
    run = root / "client-longtail_lambda=0.75_alpha=0.5_rho=3.0_localE=1_seed=1"
    expd = run / "experiment_d"
    expd.mkdir(parents=True)

    with (run / "round_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["epoch", "overall_acc", "non_tail_acc", "bottom20_tail_acc", "macro_per_class_acc"],
        )
        writer.writeheader()
        writer.writerow({"epoch": 0, "overall_acc": 10, "non_tail_acc": 11, "bottom20_tail_acc": 5, "macro_per_class_acc": 7})
        writer.writerow({"epoch": 29, "overall_acc": 20, "non_tail_acc": 21, "bottom20_tail_acc": 15, "macro_per_class_acc": 17})

    with (expd / "experiment_d_round_summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "communication_round",
                "mean_gain_support_actual",
                "mean_gain_support_normalized",
                "mean_gain_all",
                "mean_offset_gap",
                "support_actual_positive_rate",
                "support_normalized_positive_rate",
                "offset_observed_rate",
                "full_reversal_rate",
                "mean_support_fedavg_weight",
                "mean_num_support_clients",
                "mean_positive_sample_specialist_ratio",
            ],
        )
        writer.writeheader()
        writer.writerow({
            "communication_round": 5,
            "mean_gain_support_actual": 1,
            "mean_gain_support_normalized": 2,
            "mean_gain_all": 0.5,
            "mean_offset_gap": 0.5,
            "support_actual_positive_rate": 0.8,
            "support_normalized_positive_rate": 0.9,
            "offset_observed_rate": 0.7,
            "full_reversal_rate": 0.1,
            "mean_support_fedavg_weight": 0.2,
            "mean_num_support_clients": 3,
            "mean_positive_sample_specialist_ratio": 0.6,
        })

    with (expd / "client_update_norm_summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "communication_round",
                "mean_update_norm_all",
                "mean_update_norm_head_clients",
                "mean_update_norm_tail_specialists",
            ],
        )
        writer.writeheader()
        writer.writerow({
            "communication_round": 1,
            "mean_update_norm_all": 3,
            "mean_update_norm_head_clients": 2,
            "mean_update_norm_tail_specialists": 4,
        })

    with (expd / "runtime_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "communication_round",
                "local_training_seconds",
                "experimentD_diagnostic_seconds",
                "normal_global_eval_seconds",
                "round_total_seconds",
                "cumulative_seconds",
            ],
        )
        writer.writeheader()
        writer.writerow({
            "communication_round": 1,
            "local_training_seconds": 10,
            "experimentD_diagnostic_seconds": 1,
            "normal_global_eval_seconds": 2,
            "round_total_seconds": 13,
            "cumulative_seconds": 13,
        })

    out = tmp_path / "summary"
    result = subprocess.run(
        [
            sys.executable,
            str(SUMMARY_SCRIPT),
            "--root",
            str(root),
            "--output-dir",
            str(out),
        ],
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert (out / "experimentD_local_epochs_all_runs.csv").exists()
    assert (out / "experimentD_local_epochs_summary.csv").exists()
    assert (out / "experimentD_local_epochs_comparison.json").exists()
