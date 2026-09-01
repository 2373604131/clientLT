import csv
import hashlib
import json
from argparse import Namespace

import numpy as np

from scripts.analyze_stage1_capt_gap import analyze as analyze_capt_gap
from scripts.analyze_stage1_topology_gate import analyze as analyze_topology_gate
from scripts.run_stage1_capt_dual_topology import build_command


CLIENTLT = np.asarray(
    [[2, 2, 2, 2, 0, 0, 0, 0]] * 15
    + [[0, 0, 0, 0, 2, 2, 2, 2]] * 15,
    dtype=np.int64,
)
MATCHED = np.ones((30, 8), dtype=np.int64)
SCHEDULE = [
    [int((epoch + offset) % 30) for offset in range(12)]
    for epoch in range(80)
]


def _write_matrix(path, matrix):
    path.mkdir(parents=True, exist_ok=True)
    with (path / "client_class_counts.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(["client_id", *[f"class_{value}" for value in range(matrix.shape[1])]])
        for client_id, row in enumerate(matrix):
            writer.writerow([client_id, *row.tolist()])


def _write_schedule_audit(path):
    with (path / "lora_aggregation_weights.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=["epoch_index", "client_id"])
        writer.writeheader()
        for epoch, clients in enumerate(SCHEDULE):
            for client_id in clients:
                writer.writerow({"epoch_index": epoch, "client_id": client_id})


def _write_per_class(path, method, topology):
    for epoch in range(80):
        with (path / f"per_class_accuracy_epoch_{epoch}.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=["class_id", "per_class_acc"])
            writer.writeheader()
            for class_id in range(8):
                base = 50.0 + class_id
                gain = 0.0
                if method == "sca" and class_id >= 6:
                    gain = 4.0 if topology == "clientlt" else -1.0
                    if epoch >= 50 and class_id == 7:
                        gain -= 2.0
                writer.writerow({"class_id": class_id, "per_class_acc": base + gain})


def _write_round_metrics(path, value):
    fieldnames = [
        "epoch",
        "overall_acc",
        "non_tail_acc",
        "bottom20_tail_acc",
        "macro_per_class_acc",
        "macro_f1",
    ]
    with (path / "round_metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for epoch in range(80):
            writer.writerow(
                {
                    "epoch": epoch,
                    "overall_acc": value,
                    "non_tail_acc": value,
                    "bottom20_tail_acc": value,
                    "macro_per_class_acc": value,
                    "macro_f1": value - 1,
                }
            )


def _prepare_sca_root(root):
    specs = {
        "residual_fedavg_clientlt": (CLIENTLT, "residual", "clientlt", 49.0),
        "online_sca": (CLIENTLT, "sca", "clientlt", 50.0),
        "residual_fedavg_matched_dirichlet": (
            MATCHED,
            "residual",
            "matched",
            59.0,
        ),
        "online_sca_matched_dirichlet": (MATCHED, "sca", "matched", 60.0),
    }
    for name, (matrix, method, topology, metric) in specs.items():
        path = root / name
        _write_matrix(path, matrix)
        _write_schedule_audit(path)
        _write_per_class(path, method, topology)
        _write_round_metrics(path, metric)


def _normalized_schedule_hash():
    encoded = json.dumps(
        [sorted(row) for row in SCHEDULE], separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _write_capt_protocol(path, condition):
    payload = {
        "schema_version": "stage1b_capt_dual_topology_v1",
        "condition": condition,
        "partition": "client-longtail" if condition == "clientlt" else "matched-dirichlet",
        "seed": 42,
        "split_seed": 42,
        "schedule_seed": 42,
        "client_schedule_sha256": _normalized_schedule_hash(),
        "num_users": 30,
        "frac": 0.4,
        "rounds": 80,
        "local_epochs": 3,
        "matched_beta": 0.5,
        "clientlt": {"same": True},
        "capt_protocol": {
            "fixed_global_aggregation_frequency": 1,
            "official_test_controls_future_training": False,
        },
    }
    (path / "stage1b_capt_protocol.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )


def _write_capt_selected_clients(path):
    with (path / "selected_clients.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["epoch_index", "client_id", "selection_order"],
        )
        writer.writeheader()
        for epoch, clients in enumerate(SCHEDULE):
            for order, client_id in enumerate(clients):
                writer.writerow(
                    {
                        "epoch_index": epoch,
                        "client_id": client_id,
                        "selection_order": order,
                    }
                )


def test_stage1a_builds_strict_class_round_table_without_agreement_proxy(tmp_path):
    root = tmp_path / "factorial"
    _prepare_sca_root(root)
    args = Namespace(
        output_root=root,
        output_dir=root / "stage1a",
        tail_class_ratio=0.25,
        null_samples=20,
        null_seed=7,
        permutation_samples=20,
        permutation_seed=9,
    )

    payload = analyze_topology_gate(args)

    assert payload["artifact_audit"]["all_four_actual_client_schedules_equal"] is True
    assert payload["update_agreement"]["available"] is False
    assert payload["update_agreement"]["proxy_used"] is False
    with (args.output_dir / "stage1a_class_round.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 2 * 80 * 8
    assert {row["stage"] for row in rows} == {"early", "middle", "late"}
    assert all(row["update_agreement_available"] == "False" for row in rows)


def test_stage1b_decomposition_closes_exactly_and_audits_protocol(tmp_path):
    sca_root = tmp_path / "factorial"
    capt_root = sca_root / "stage1b_capt"
    _prepare_sca_root(sca_root)
    for condition, matrix, value in (
        ("clientlt", CLIENTLT, 58.0),
        ("matched", MATCHED, 62.0),
    ):
        path = capt_root / f"capt_{condition}"
        _write_matrix(path, matrix)
        _write_round_metrics(path, value)
        _write_capt_protocol(path, condition)
        _write_capt_selected_clients(path)
    args = Namespace(
        sca_output_root=sca_root,
        capt_output_root=capt_root,
        output_dir=sca_root / "stage1b_gap",
        expected_rounds=80,
    )

    payload = analyze_capt_gap(args)

    primary = payload["primary_final"]
    assert abs(primary["closure_error"]) < 1e-12
    assert primary["base_gap"] == 2.0
    assert primary["ours_topology_penalty"] == 10.0
    assert primary["capt_topology_penalty"] == 4.0
    assert primary["topology_robustness_gap"] == 6.0
    assert primary["capt_total_advantage_on_clientlt"] == 8.0


def test_capt_runner_uses_exact_topology_and_nonleaky_fixed_schedule(tmp_path):
    args = Namespace(
        output_root=tmp_path / "capt",
        data_root=tmp_path / "data",
        python_bin="python",
        seed=42,
        split_seed=42,
        num_users=30,
        frac=0.4,
        rounds=80,
        local_epochs=3,
        schedule_seed=42,
        lr=0.001,
        test_batch_size=256,
        matched_beta=0.5,
        num_workers=8,
    )
    schedule = tmp_path / "schedule.json"
    clientlt = build_command(args, "clientlt", schedule)
    matched = build_command(args, "matched", schedule)

    assert clientlt[clientlt.index("--partition") + 1] == "client-longtail"
    assert matched[matched.index("--partition") + 1] == "matched-dirichlet"
    assert clientlt[clientlt.index("--capt_fixed_global_agg_freq") + 1] == "1"
    ignored = {"--partition", "--output-dir"}

    def canonical(command):
        result = []
        index = 0
        while index < len(command):
            if command[index] in ignored:
                index += 2
            else:
                result.append(command[index])
                index += 1
        return result

    assert canonical(clientlt) == canonical(matched)
