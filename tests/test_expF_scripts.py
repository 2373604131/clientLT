import os
import re
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_SH = REPO_ROOT / "scripts" / "expF_promptfl.sh"
SCRIPT_BAT = REPO_ROOT / "scripts" / "expF_promptfl.bat"


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


def _dry_run_lines():
    env = {**os.environ, "DRY_RUN": "1"}
    bash_result = subprocess.run(
        ["bash", str(SCRIPT_SH)],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if bash_result.returncode == 0:
        return bash_result.stdout.replace('"', "").splitlines()

    if os.name == "nt":
        bat_result = subprocess.run(
            ["cmd", "/c", "set DRY_RUN=1&& scripts\\expF_promptfl.bat"],
            cwd=REPO_ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
        )
        return bat_result.stdout.replace('"', "").splitlines()

    raise AssertionError(
        "Linux DRY_RUN failed and no Windows batch fallback is available:\n"
        + bash_result.stdout
        + bash_result.stderr
    )


def test_expF_scripts_exist_and_call_federated_main_directly():
    assert SCRIPT_SH.exists()
    assert SCRIPT_BAT.exists()

    sh = _read(SCRIPT_SH)
    bat = _read(SCRIPT_BAT)
    assert "federated_main.py" in sh
    assert "federated_main.py" in bat

    forbidden = [
        "run_" + "expF.py",
        "experiment" + " runner",
        "exp" + "D",
        "Exp" + "D",
        "matched" + " pair",
        "matched" + "_pairs",
        "approved" + "_pairs",
    ]
    for token in forbidden:
        assert token not in sh
        assert token not in bat


def test_expF_script_static_configuration():
    sh = _read(SCRIPT_SH)
    bat = _read(SCRIPT_BAT)
    sh_values = _assignments(sh)
    bat_values = _assignments(bat)

    expected = {
        "MODEL": "fedavg",
        "TRAINER": "PromptFL",
        "USERS": "50",
        "FRAC": "0.2",
        "ROUND": "100",
        "LOCAL_EPOCHS": "5",
        "BATCH_SIZE": "32",
        "TEST_BATCH_SIZE": "64",
        "GLOBAL_EVAL_INTERVAL": "5",
        "UPDATE_RETENTION_INTERVAL": "5",
        "INTRA_GROUP_ALPHA": "0.1",
        "HEAD_LEAKAGE_SCALE": "3.0",
        "ISOLATE_LOCAL_OPTIMIZER_STATE": "True",
        "FEDERATED_SINGLE_SCHEDULER_STEP": "True",
        "SEEDS": "1 2 3",
        "DIRICHLET_BETAS": "1.0 0.5 0.3 0.1",
        "CLIENTLT_LAMBDAS": "0.0 0.25 0.5 0.75 1.0",
        "TOTAL_RUNS": "27",
    }
    for key, value in expected.items():
        assert sh_values[key] == value
        assert bat_values[key] == value

    assert "set -euo pipefail" in sh
    assert "errorlevel 1" in bat
    assert "DRY_RUN" in sh
    assert "DRY_RUN" in bat
    assert "finished.flag" in sh
    assert "finished.flag" in bat
    assert "run.log" in sh
    assert "run.log" in bat


def test_expF_linux_dry_run_expands_exact_matrix_without_side_effect_paths():
    lines = _dry_run_lines()
    markers = [line for line in lines if line.startswith("[ExpF ")]
    commands = [line for line in lines if " federated_main.py " in line]

    assert len(markers) == 27
    assert markers[0] == "[ExpF 01/27]"
    assert markers[-1] == "[ExpF 27/27]"
    assert len(commands) == 27

    assert sum("--partition noniid-labeldir-fine" in line for line in commands) == 12
    assert sum("--partition client-longtail" in line for line in commands) == 15
    assert all("--local_epochs 5" in line for line in commands)
    assert all("--isolate_local_optimizer_state True" in line for line in commands)
    assert all("--federated_single_scheduler_step True" in line for line in commands)
    assert all("/ExpF/" in line for line in commands)

    for seed in ("1", "2", "3"):
        schedule = f"output/expF_shared_schedules/users50_frac0.2_round100_seed{seed}.json"
        seed_commands = [line for line in commands if f"--seed {seed} " in line]
        assert len(seed_commands) == 9
        assert all(schedule in line for line in seed_commands)
        assert all(f"--split_seed {seed}" in line for line in seed_commands)

    seed1 = commands[:9]
    assert "--partition noniid-labeldir-fine --beta 1.0" in seed1[0]
    assert "--partition noniid-labeldir-fine --beta 0.5" in seed1[1]
    assert "--partition noniid-labeldir-fine --beta 0.3" in seed1[2]
    assert "--partition noniid-labeldir-fine --beta 0.1" in seed1[3]
    assert "--partition client-longtail" in seed1[4]
    assert "--specialization_lambda 0.0" in seed1[4]
    assert "--specialization_lambda 1.0" in seed1[8]


def test_expF_protocol_specific_arguments_are_not_mixed():
    commands = [line for line in _dry_run_lines() if " federated_main.py " in line]
    dirichlet = [line for line in commands if "--partition noniid-labeldir-fine" in line]
    clientlt = [line for line in commands if "--partition client-longtail" in line]

    assert all("--specialization_lambda" not in line for line in dirichlet)
    assert all("--intra_group_alpha" not in line for line in dirichlet)
    assert all("--head_leakage_scale" not in line for line in dirichlet)
    assert all("--head_client_ratio" not in line for line in dirichlet)
    assert all("--tail_client_ratio" not in line for line in dirichlet)

    assert all("--intra_group_alpha 0.1" in line for line in clientlt)
    assert all("--head_leakage_scale 3.0" in line for line in clientlt)
    assert all("--head_client_ratio 0.9" in line for line in clientlt)
    assert all("--tail_client_ratio 0.1" in line for line in clientlt)
    assert all("--head_class_ratio 0.8" in line for line in clientlt)
    assert all("--tail_class_ratio 0.2" in line for line in clientlt)
    assert all("_beta=" not in line for line in clientlt)


def test_expF_linux_and_windows_key_arguments_match():
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
