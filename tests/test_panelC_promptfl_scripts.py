import os
import re
import subprocess
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_SH = REPO_ROOT / "scripts" / "panelC_promptfl.sh"
SCRIPT_3GPU = REPO_ROOT / "scripts" / "panelC_promptfl_3gpu.sh"
SCRIPT_BAT = REPO_ROOT / "scripts" / "panelC_promptfl.bat"
SCHEDULE_SCRIPT = REPO_ROOT / "scripts" / "create_client_schedule.py"


def _read(path):
    return path.read_text(encoding="utf-8")


def _assignments(text):
    values = {}
    for line in text.splitlines():
        match = re.match(r'^\s*(?:set\s+)?"?([A-Z0-9_]+)=(.*?)"?\s*$', line)
        if not match:
            continue
        key, value = match.groups()
        value = value.strip().strip('"')
        if value.startswith("${") or value.startswith("%"):
            continue
        values[key] = value
    return values


def _dry_run_lines(script=SCRIPT_SH):
    env = {**os.environ, "DRY_RUN": "1"}
    bash_result = subprocess.run(
        ["bash", str(script)],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if bash_result.returncode == 0:
        return bash_result.stdout.replace('"', "").splitlines()

    if os.name == "nt" and script == SCRIPT_SH:
        bat_result = subprocess.run(
            ["cmd", "/c", "set DRY_RUN=1&& scripts\\panelC_promptfl.bat"],
            cwd=REPO_ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
        )
        return bat_result.stdout.replace('"', "").splitlines()

    raise AssertionError(
        "PanelC dry run failed:\n"
        + bash_result.stdout
        + bash_result.stderr
    )


def _commands_from(lines):
    return [line for line in lines if " federated_main.py " in line]


def test_panelC_scripts_exist_and_call_federated_main_directly():
    assert SCRIPT_SH.exists()
    assert SCRIPT_3GPU.exists()
    assert SCRIPT_BAT.exists()

    for text in (_read(SCRIPT_SH), _read(SCRIPT_3GPU), _read(SCRIPT_BAT)):
        assert "federated_main.py" in text
        assert "run_" + "panelC.py" not in text
        assert "tail_" + "specialization_strength" not in text
        assert "tail_" + "aggregation_alpha" not in text
        assert "ExpF" not in text


def test_panelC_static_configuration_matches_design():
    sh_values = _assignments(_read(SCRIPT_SH))
    gpu_values = _assignments(_read(SCRIPT_3GPU))
    bat_values = _assignments(_read(SCRIPT_BAT))

    expected = {
        "MODEL": "fedavg",
        "TRAINER": "PromptFL",
        "USERS": "30",
        "FRAC": "1.0",
        "ROUND": "100",
        "LOCAL_EPOCHS": "3",
        "BATCH_SIZE": "32",
        "TEST_BATCH_SIZE": "64",
        "NUM_WORKERS": "8",
        "GLOBAL_EVAL_INTERVAL": "5",
        "UPDATE_RETENTION_INTERVAL": "5",
        "LOG_UPDATE_RETENTION": "False",
        "HEAD_CLIENT_RATIO": "0.9",
        "TAIL_CLIENT_RATIO": "0.1",
        "HEAD_CLASS_RATIO": "0.8",
        "TAIL_CLASS_RATIO": "0.2",
        "SPECIALIZATION_LAMBDA": "0.75",
        "HEAD_LEAKAGE_SCALE": "3.0",
        "ISOLATE_LOCAL_OPTIMIZER_STATE": "True",
        "FEDERATED_SINGLE_SCHEDULER_STEP": "True",
        "SEEDS": "1 42 2026",
        "ALPHAS": "0.1 0.25 0.5 0.75 1.0",
        "TOTAL_RUNS": "30",
    }
    for key, value in expected.items():
        assert sh_values[key] == value
        assert gpu_values[key] == value
        assert bat_values[key] == value

    assert sh_values["DATASETS"] == "cifar100_LT"
    assert bat_values["DATASETS"] == "cifar100_LT"
    assert gpu_values["DATASET"] == "cifar100_LT"

    assert "GPUS=\"${GPUS:-0 1 2}\"" in _read(SCRIPT_3GPU)
    assert "RERUN_FAILED=\"${RERUN_FAILED:-0}\"" in _read(SCRIPT_3GPU)
    assert "running.lock" in _read(SCRIPT_3GPU)
    assert "failed.flag" in _read(SCRIPT_3GPU)
    assert "(run_id - 1) % GPU_COUNT" in _read(SCRIPT_3GPU)
    assert "worker \"${GPU_ARRAY[slot]}\" \"${slot}\"" in _read(SCRIPT_3GPU)
    assert "prepare_shared_schedules" in _read(SCRIPT_3GPU)
    assert "scripts/create_client_schedule.py" in _read(SCRIPT_3GPU)


def test_panelC_single_gpu_dry_run_expands_exact_matrix():
    lines = _dry_run_lines()
    markers = [line for line in lines if line.startswith("[PanelC ")]
    commands = _commands_from(lines)

    assert len(markers) == 30
    assert markers[0] == "[PanelC 01/30]"
    assert markers[-1] == "[PanelC 30/30]"
    assert len(commands) == 30

    assert sum("--partition noniid-labeldir-fine" in line for line in commands) == 15
    assert sum("--partition client-longtail" in line for line in commands) == 15
    assert all("--num_users 30" in line for line in commands)
    assert all("--frac 1.0" in line for line in commands)
    assert all("--round 100" in line for line in commands)
    assert all("--local_epochs 3" in line for line in commands)
    assert all("/PanelC_users30_localE3/" in line for line in commands)
    assert all("--log_update_retention False" in line for line in commands)
    assert all("DATALOADER.NUM_WORKERS 8" in line for line in commands)

    for seed in ("1", "42", "2026"):
        schedule = f"output/panelC_shared_schedules/users30_frac1.0_round100_seed{seed}.json"
        seed_commands = [line for line in commands if f"--seed {seed} " in line]
        assert len(seed_commands) == 10
        assert all(f"--split_seed {seed}" in line for line in seed_commands)
        assert all(schedule in line for line in seed_commands)


def test_panelC_alpha_mapping_and_output_directories():
    commands = _commands_from(_dry_run_lines())
    alphas = ("0.1", "0.25", "0.5", "0.75", "1.0")
    seed1 = commands[:10]

    for idx, alpha in enumerate(alphas):
        line = seed1[idx]
        assert "--partition noniid-labeldir-fine" in line
        assert f"--beta {alpha}" in line
        assert "--intra_group_alpha" not in line
        assert f"partition=noniid-labeldir-fine_alpha={alpha}_IF=0.01_localE=3_seed=1" in line

    for idx, alpha in enumerate(alphas, start=5):
        line = seed1[idx]
        assert "--partition client-longtail" in line
        assert f"--intra_group_alpha {alpha}" in line
        assert "--beta" not in line
        assert "--specialization_lambda 0.75" in line
        assert "--head_leakage_scale 3.0" in line
        assert "--head_client_ratio 0.9" in line
        assert "--tail_client_ratio 0.1" in line
        assert "--head_class_ratio 0.8" in line
        assert "--tail_class_ratio 0.2" in line
        assert f"partition=client-longtail_lambda=0.75_alpha={alpha}_rho=3.0_IF=0.01_localE=3_seed=1" in line


def test_panelC_common_training_arguments_are_present_in_both_scripts():
    sh = _read(SCRIPT_SH)
    bat = _read(SCRIPT_BAT)
    key_args = [
        "--root",
        "--model",
        "--trainer",
        "--dataset",
        "--seed",
        "--split_seed",
        "--num_users",
        "--frac",
        "--round",
        "--local_epochs",
        "--isolate_local_optimizer_state",
        "--federated_single_scheduler_step",
        "--client_schedule_file",
        "--client_schedule_seed",
        "--log_update_retention",
        "--update_retention_interval",
        "--update_retention_param_key",
        "DATALOADER.NUM_WORKERS",
    ]
    for arg in key_args:
        assert arg in sh
        assert arg in bat


def test_panelC_3gpu_dry_run_assigns_each_task_once_when_bash_available():
    try:
        lines = _dry_run_lines(SCRIPT_3GPU)
    except AssertionError as exc:
        if os.name == "nt" and "wslstore" in str(exc).lower():
            pytest.skip("bash/WSL is unavailable on this Windows machine")
        raise
    markers = [line for line in lines if line.startswith("[PanelC ")]
    commands = _commands_from(lines)

    assert len(markers) == 30
    assert len(commands) == 30
    assert len(set(commands)) == 30
    assert any("gpu=0" in line for line in markers)
    assert any("gpu=1" in line for line in markers)
    assert any("gpu=2" in line for line in markers)


def test_create_client_schedule_outputs_valid_full_participation_schedule(tmp_path):
    schedule_path = tmp_path / "users30_frac1.0_round100_seed1.json"
    subprocess.run(
        [
            os.sys.executable,
            str(SCHEDULE_SCRIPT),
            "--path",
            str(schedule_path),
            "--num_rounds",
            "100",
            "--num_users",
            "30",
            "--frac",
            "1.0",
            "--seed",
            "1",
        ],
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )

    import json

    payload = json.loads(schedule_path.read_text(encoding="utf-8"))
    schedule = payload["schedule"]
    assert payload["num_rounds"] == 100
    assert payload["num_users"] == 30
    assert payload["clients_per_round"] == 30
    assert len(schedule) == 100
    for clients in schedule:
        assert len(clients) == 30
        assert len(set(clients)) == 30
        assert sorted(clients) == list(range(30))
