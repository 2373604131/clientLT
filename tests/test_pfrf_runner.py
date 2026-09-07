from types import SimpleNamespace

import torch

from scripts.run_pfrf_smoke import _rng_state_equal, commands
from utils.pfrf import capture_rng_state


def test_smoke_runner_builds_six_full_runs_and_two_resume_pairs(tmp_path):
    args = SimpleNamespace(
        python_bin="python",
        data_root=tmp_path / "data",
        output_root=tmp_path / "output",
        num_users=30,
        conditions=[
            "ordinary_ce",
            "matched_memory_ce",
            "class_reweighted_ce",
            "instantaneous_target",
            "pfrf_max",
            "pfrf_add",
        ],
    )
    built = commands(args)
    labels = [item["label"] for item in built]
    assert len(built) == 10
    assert labels[:6] == [f"full:{condition}" for condition in args.conditions]
    assert labels[-4:] == [
        "resume-prefix:pfrf_max",
        "resume-suffix:pfrf_max",
        "resume-prefix:pfrf_add",
        "resume-suffix:pfrf_add",
    ]
    for item in built:
        command = item["command"]
        assert command[command.index("--frac") + 1] == "1.0"
        assert command[command.index("--local_epochs") + 1] == "3"
        assert command[command.index("--cliplora_aggregation") + 1] == "effective_svd"
        assert command[command.index("--pfrf_deterministic") + 1] == "True"


def test_resume_rng_comparison_checks_the_next_torch_stream():
    torch.manual_seed(42)
    left = capture_rng_state()
    assert _rng_state_equal(left, left)
    _ = torch.rand(1)
    right = capture_rng_state()
    assert not _rng_state_equal(left, right)
