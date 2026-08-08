import json
from pathlib import Path
from types import SimpleNamespace

from scripts.run_cliplora_support_normalized import (
    build_command,
    run_dir,
    verify_matched_class_marginals,
)


def _args(tmp_path):
    return SimpleNamespace(
        output_root=tmp_path,
        data_root=Path("DATA"),
        python_bin="python",
        dirichlet_beta=0.5,
        specialization_lambda=0.75,
        intra_group_alpha=0.5,
        head_leakage_scale=3.0,
        num_users=30,
        rounds=100,
        local_epochs=3,
        lr=0.001,
        imb_factor=0.01,
        train_batch_size=32,
        test_batch_size=64,
        global_eval_interval=1,
        num_workers=8,
    )


def _value_after(command, flag):
    return command[command.index(flag) + 1]


def test_runner_builds_matched_factorial_commands(tmp_path):
    args = _args(tmp_path)
    clientlt = build_command(args, "clientlt_support_normalized", 42)
    dirichlet = build_command(args, "dirichlet_support_normalized", 42)
    fedavg = build_command(args, "clientlt_fedavg", 42)

    assert _value_after(clientlt, "--partition") == "client-longtail"
    assert _value_after(dirichlet, "--partition") == "noniid-labeldir-fine"
    assert _value_after(clientlt, "--cliplora_aggregation") == "support_normalized"
    assert _value_after(dirichlet, "--cliplora_aggregation") == "support_normalized"
    assert _value_after(fedavg, "--cliplora_aggregation") == "fedavg"
    assert _value_after(clientlt, "--experimentD_enable") == "False"
    assert _value_after(fedavg, "--experimentD_enable") == "True"
    assert _value_after(clientlt, "--lr") == _value_after(dirichlet, "--lr") == "0.001"
    assert _value_after(clientlt, "--client_schedule_file") == _value_after(
        dirichlet, "--client_schedule_file"
    )


def test_runner_verifies_global_class_marginal_fingerprint(tmp_path):
    cases = ["clientlt_fedavg", "dirichlet_fedavg"]
    for case in cases:
        path = run_dir(tmp_path, case, 42)
        path.mkdir(parents=True)
        (path / "partition_summary.json").write_text(
            json.dumps(
                {
                    "global_lt_fingerprint": "same",
                    "global_class_counts": [100, 10, 1],
                }
            ),
            encoding="utf-8",
        )

    manifest = verify_matched_class_marginals(tmp_path, cases, [42])
    payload = json.loads(manifest.read_text(encoding="utf-8"))

    assert payload["class_marginal_match_verified"] is True
    assert len(payload["runs"]) == 2
