import sys
from pathlib import Path
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_eri_closure import build_command, ensure_full_schedule, schedule_file


def _args(tmp_path):
    return SimpleNamespace(
        output_root=tmp_path, python_bin="python", data_root=Path("DATA"), num_users=30,
        rounds=100, local_epochs=3, lr=0.001, imb_factor=0.01, train_batch_size=32,
        test_batch_size=64, dirichlet_beta=0.5, specialization_lambda=0.75,
        intra_group_alpha=0.5, head_leakage_scale=3.0,
        audit_rounds="1,10,100", num_workers=8,
    )


def test_runner_uses_matched_dirichlet_and_frozen_eri_protocol(tmp_path):
    args = _args(tmp_path)
    command = build_command(args, "matched_dirichlet_fedavg", 42)
    assert command[command.index("--partition") + 1] == "matched-dirichlet"
    assert command[command.index("--eri_audit_enable") + 1] == "True"
    assert command[command.index("--cliplora_precision") + 1] == "fp32"
    assert command[command.index("--frac") + 1] == "1.0"


def test_schedule_is_full_and_stable(tmp_path):
    path = schedule_file(tmp_path, 42)
    ensure_full_schedule(path, rounds=3, users=4, seed=42)
    text = path.read_text(encoding="utf-8")
    ensure_full_schedule(path, rounds=3, users=4, seed=42)
    assert path.read_text(encoding="utf-8") == text
