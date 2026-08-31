import csv
from argparse import Namespace

from scripts.analyze_sca_factorial import analyze
from scripts.run_online_sca import build_command, summarize


def _write_run(path, matrix, values, aggregation):
    path.mkdir(parents=True)
    with (path / "client_class_counts.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(["client_id", "class_0", "class_1"])
        for client_id, row in enumerate(matrix):
            writer.writerow([client_id, *row])
    with (path / "round_metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        fieldnames = [
            "epoch",
            "overall_acc",
            "non_tail_acc",
            "bottom20_tail_acc",
            "macro_per_class_acc",
            "macro_f1",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for epoch, value in enumerate(values):
            writer.writerow(
                {
                    "epoch": epoch,
                    "overall_acc": value,
                    "non_tail_acc": value + 10,
                    "bottom20_tail_acc": value - 10,
                    "macro_per_class_acc": value,
                    "macro_f1": value - 1,
                }
            )
    (path / "online_sca_protocol.json").write_text(
        '{"seed": 42, "split_seed": 42, "client_schedule_sha256": "same", '
        '"rounds": 2, "num_users": 2, "frac": 1.0, "local_epochs": 1, '
        '"residual_scale": 10.0, "residual_clamp": 3.0, '
        '"residual_lr_multiplier": 5.0, "residual_use_bias": false, '
        f'"tail_class_ids": [1], "residual_aggregation": "{aggregation}"}}',
        encoding="utf-8",
    )


def test_factorial_analysis_computes_did_and_checks_margins(tmp_path):
    clt = [[8, 2], [2, 8]]
    matched = [[5, 5], [5, 5]]
    dirs = {
        "clientlt_residual_fedavg": tmp_path / "clt_rf",
        "clientlt_sca": tmp_path / "clt_sca",
        "matched_residual_fedavg": tmp_path / "dir_rf",
        "matched_sca": tmp_path / "dir_sca",
    }
    _write_run(dirs["clientlt_residual_fedavg"], clt, [50, 52], "fedavg")
    _write_run(dirs["clientlt_sca"], clt, [54, 58], "class_separable")
    _write_run(dirs["matched_residual_fedavg"], matched, [60, 62], "fedavg")
    _write_run(dirs["matched_sca"], matched, [61, 64], "class_separable")
    args = Namespace(
        **{f"{key}_dir": value for key, value in dirs.items()},
        output_dir=tmp_path / "analysis",
        primary_metric="macro_per_class_acc",
    )

    payload = analyze(args)

    assert payload["topology_audit"]["passed"] is True
    assert payload["primary_final"]["delta_clientlt"] == 6.0
    assert payload["primary_final"]["delta_matched_dirichlet"] == 2.0
    assert payload["primary_final"]["difference_in_differences"] == 4.0
    assert (args.output_dir / "factorial_per_round.csv").exists()
    assert (args.output_dir / "factorial_best_common_round.csv").exists()


def test_factorial_commands_change_only_topology_and_residual_aggregation(tmp_path):
    args = Namespace(
        output_root=tmp_path,
        data_root=tmp_path / "data",
        python_bin="python",
        frac=0.4,
        rounds=80,
        lr=0.001,
        test_batch_size=256,
        eval_interval=1,
        num_workers=8,
        sca_scale=10.0,
        sca_clamp=3.0,
        sca_lr_mult=5.0,
        support_min_fraction=0.0,
        support_weighting="class_count",
        matched_beta=0.5,
    )
    config = {"position": "top3", "rank": 4, "alpha": 1, "params": ["q", "v"]}
    expected = {
        "residual_fedavg_clientlt": ("client-longtail", "fedavg"),
        "online_sca": ("client-longtail", "class_separable"),
        "residual_fedavg_matched_dirichlet": ("matched-dirichlet", "fedavg"),
        "online_sca_matched_dirichlet": ("matched-dirichlet", "class_separable"),
    }
    commands = {}
    for condition, pair in expected.items():
        _, command = build_command(args, condition, config)
        assert command[command.index("--partition") + 1] == pair[0]
        assert command[command.index("--cliplora_residual_aggregation") + 1] == pair[1]
        assert command[command.index("--cliplora_sca_enable") + 1] == "True"
        commands[condition] = command

    ignored = {
        "--output-dir",
        "--partition",
        "--cliplora_residual_aggregation",
        "--cliplora_sca_d4_enable",
    }

    def canonical(command):
        result = []
        index = 0
        while index < len(command):
            token = command[index]
            if token in ignored:
                index += 2
            else:
                result.append(token)
                index += 1
        return result

    reference = canonical(commands["residual_fedavg_clientlt"])
    assert all(canonical(command) == reference for command in commands.values())


def test_clientlt_screen_is_available_before_matched_runs(tmp_path):
    matrix = [[8, 2], [2, 8]]
    _write_run(
        tmp_path / "residual_fedavg_clientlt", matrix, [50, 52], "fedavg"
    )
    _write_run(tmp_path / "online_sca", matrix, [54, 58], "class_separable")
    summarize(Namespace(output_root=tmp_path, frac=0.4, rounds=2))

    screen_path = tmp_path / "clientlt_aggregation_screen.csv"
    assert screen_path.exists()
    with screen_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert float(rows[-1]["delta_macro_per_class_acc"]) == 6.0
