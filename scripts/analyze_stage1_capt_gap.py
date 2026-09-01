#!/usr/bin/env python3
"""Stage-1B: decompose CAPT's Client-LT advantage into two exact gaps."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np


METRICS = (
    "overall_acc",
    "head_acc",
    "tail_acc",
    "head_tail_h_mean",
    "macro_f1",
)
COMPARATORS = ("sca", "residual_fedavg")
STRICT_STAGE1_PROTOCOL = {
    "seed": 42,
    "split_seed": 42,
    "schedule_seed": 42,
    "num_users": 30,
    "frac": 0.4,
    "rounds": 80,
    "local_epochs": 3,
}


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _json_safe(value):
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, (float, np.floating)) and not math.isfinite(float(value)):
        return None
    if isinstance(value, np.integer):
        return int(value)
    return value


def _read_metrics(run_dir: Path) -> dict[int, dict[str, float]]:
    path = run_dir / "round_metrics.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    rows = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for raw in csv.DictReader(handle):
            epoch = int(raw["epoch"])
            if epoch < 0:
                continue
            if epoch in rows:
                raise ValueError(f"Duplicate epoch {epoch} in {path}")
            head = float(raw["non_tail_acc"])
            tail = float(raw["bottom20_tail_acc"])
            rows[epoch] = {
                "overall_acc": float(raw["overall_acc"]),
                "head_acc": head,
                "tail_acc": tail,
                "head_tail_h_mean": (
                    2.0 * head * tail / (head + tail) if head + tail > 0 else 0.0
                ),
                "macro_f1": float(raw["macro_f1"]),
            }
    if not rows:
        raise ValueError(f"No evaluated training rounds in {path}")
    return rows


def _read_matrix(run_dir: Path) -> np.ndarray:
    path = run_dir / "client_class_counts.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    rows = []
    ids = []
    with path.open(newline="", encoding="utf-8") as handle:
        for raw in csv.DictReader(handle):
            ids.append(int(raw["client_id"]))
            rows.append([int(raw[key]) for key in raw if key.startswith("class_")])
    if ids != list(range(len(ids))):
        raise ValueError(f"Non-canonical client ids in {path}")
    return np.asarray(rows, dtype=np.int64)


def _read_sca_schedule(run_dir: Path) -> list[list[int]]:
    path = run_dir / "lora_aggregation_weights.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    grouped = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for raw in csv.DictReader(handle):
            grouped.setdefault(int(raw["epoch_index"]), []).append(
                int(raw["client_id"])
            )
    return [sorted(grouped[epoch]) for epoch in sorted(grouped)]


def _read_capt_schedule(
    run_dir: Path, expected_rounds: int, expected_clients_per_round: int
) -> list[list[int]]:
    path = run_dir / "selected_clients.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing executed-client audit for strict CAPT comparison: {path}"
        )
    grouped: dict[int, list[int]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for raw in csv.DictReader(handle):
            grouped.setdefault(int(raw["epoch_index"]), []).append(
                int(raw["client_id"])
            )
    expected = list(range(expected_rounds))
    if sorted(grouped) != expected:
        raise ValueError(
            f"CAPT executed-client audit does not cover exactly {expected_rounds} "
            f"rounds: {path}"
        )
    for epoch, clients in grouped.items():
        if len(clients) != len(set(clients)):
            raise ValueError(f"Duplicate CAPT client in epoch {epoch}: {path}")
        if len(clients) != expected_clients_per_round:
            raise ValueError(
                f"CAPT epoch {epoch} selected {len(clients)} clients, expected "
                f"{expected_clients_per_round}: {path}"
            )
    return [sorted(grouped[epoch]) for epoch in expected]


def _schedule_hash(schedule: list[list[int]]) -> str:
    normalized = [sorted(int(value) for value in row) for row in schedule]
    encoded = json.dumps(normalized, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _read_protocol(run_dir: Path) -> dict:
    path = run_dir / "stage1b_capt_protocol.json"
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _decomposition_row(
    epoch: int,
    metric: str,
    comparator: str,
    runs: dict[str, dict[int, dict[str, float]]],
) -> dict:
    capt_clt = runs["capt_clientlt"][epoch][metric]
    capt_dir = runs["capt_matched"][epoch][metric]
    ours_clt = runs[f"ours_{comparator}_clientlt"][epoch][metric]
    ours_dir = runs[f"ours_{comparator}_matched"][epoch][metric]
    base_gap = capt_dir - ours_dir
    ours_penalty = ours_dir - ours_clt
    capt_penalty = capt_dir - capt_clt
    robustness_gap = ours_penalty - capt_penalty
    total_advantage = capt_clt - ours_clt
    return {
        "comparator": comparator,
        "metric": metric,
        "epoch_index": epoch,
        "communication_round": epoch + 1,
        "capt_clientlt": capt_clt,
        "capt_matched_dirichlet": capt_dir,
        "ours_clientlt": ours_clt,
        "ours_matched_dirichlet": ours_dir,
        "base_gap": base_gap,
        "ours_topology_penalty": ours_penalty,
        "capt_topology_penalty": capt_penalty,
        "topology_robustness_gap": robustness_gap,
        "capt_total_advantage_on_clientlt": total_advantage,
        "decomposition_sum": base_gap + robustness_gap,
        "closure_error": total_advantage - (base_gap + robustness_gap),
        "base_gap_fraction_of_total": (
            base_gap / total_advantage if abs(total_advantage) > 1e-12 else math.nan
        ),
        "topology_gap_fraction_of_total": (
            robustness_gap / total_advantage
            if abs(total_advantage) > 1e-12
            else math.nan
        ),
    }


def analyze(args) -> dict:
    directories = {
        "capt_clientlt": args.capt_output_root / "capt_clientlt",
        "capt_matched": args.capt_output_root / "capt_matched",
        "ours_sca_clientlt": args.sca_output_root / "online_sca",
        "ours_sca_matched": args.sca_output_root / "online_sca_matched_dirichlet",
        "ours_residual_fedavg_clientlt": (
            args.sca_output_root / "residual_fedavg_clientlt"
        ),
        "ours_residual_fedavg_matched": (
            args.sca_output_root / "residual_fedavg_matched_dirichlet"
        ),
    }
    runs = {name: _read_metrics(path) for name, path in directories.items()}
    common_epochs = set.intersection(*(set(rows) for rows in runs.values()))
    if not common_epochs:
        raise ValueError("CAPT/SCA runs have no common evaluated rounds")
    expected_epochs = set(range(args.expected_rounds))
    missing_by_run = {
        name: sorted(expected_epochs - set(rows)) for name, rows in runs.items()
    }
    if any(missing_by_run.values()):
        raise ValueError(
            "Strict Stage-1B requires every communication round in every run: "
            f"{missing_by_run}"
        )
    epochs = sorted(common_epochs)
    final_epoch = args.expected_rounds - 1

    matrices = {name: _read_matrix(path) for name, path in directories.items()}
    topology_pairs = {
        "clientlt": [
            "capt_clientlt",
            "ours_sca_clientlt",
            "ours_residual_fedavg_clientlt",
        ],
        "matched_dirichlet": [
            "capt_matched",
            "ours_sca_matched",
            "ours_residual_fedavg_matched",
        ],
    }
    for topology, names in topology_pairs.items():
        reference = matrices[names[0]]
        if any(not np.array_equal(reference, matrices[name]) for name in names[1:]):
            raise ValueError(f"CAPT/ours partition mismatch within {topology}")
    clientlt_matrix = matrices["capt_clientlt"]
    matched_matrix = matrices["capt_matched"]
    if not np.array_equal(clientlt_matrix.sum(axis=0), matched_matrix.sum(axis=0)):
        raise ValueError("CAPT topology class margins differ")
    if not np.array_equal(clientlt_matrix.sum(axis=1), matched_matrix.sum(axis=1)):
        raise ValueError("CAPT topology client margins differ")
    if np.array_equal(clientlt_matrix, matched_matrix):
        raise ValueError("CAPT topology coupling did not change")

    protocols = {
        "clientlt": _read_protocol(directories["capt_clientlt"]),
        "matched": _read_protocol(directories["capt_matched"]),
    }
    invariant_keys = (
        "seed",
        "split_seed",
        "schedule_seed",
        "client_schedule_sha256",
        "num_users",
        "frac",
        "rounds",
        "local_epochs",
        "matched_beta",
        "clientlt",
        "capt_protocol",
    )
    invariant_mismatches = {
        key: {condition: payload.get(key) for condition, payload in protocols.items()}
        for key in invariant_keys
        if protocols["clientlt"].get(key) != protocols["matched"].get(key)
    }
    if invariant_mismatches:
        raise ValueError(f"CAPT protocol mismatch: {invariant_mismatches}")
    strict_mismatches = {
        key: {
            "required": expected,
            "observed": protocols["clientlt"].get(key),
        }
        for key, expected in STRICT_STAGE1_PROTOCOL.items()
        if protocols["clientlt"].get(key) != expected
    }
    if strict_mismatches:
        raise ValueError(f"CAPT did not use the strict Stage-1B protocol: {strict_mismatches}")
    if args.expected_rounds != STRICT_STAGE1_PROTOCOL["rounds"]:
        raise ValueError(
            f"Strict Stage-1B requires {STRICT_STAGE1_PROTOCOL['rounds']} rounds"
        )
    for condition, payload in protocols.items():
        capt_protocol = payload.get("capt_protocol", {})
        if capt_protocol.get("fixed_global_aggregation_frequency") != 1:
            raise ValueError(f"CAPT {condition} did not aggregate every round")
        if capt_protocol.get("official_test_controls_future_training") is not False:
            raise ValueError(f"CAPT {condition} allows official test to control training")

    expected_clients_per_round = max(
        int(
            float(protocols["clientlt"]["frac"])
            * int(protocols["clientlt"]["num_users"])
        ),
        1,
    )
    sca_schedules = {
        "clientlt": _read_sca_schedule(directories["ours_sca_clientlt"]),
        "matched": _read_sca_schedule(directories["ours_sca_matched"]),
    }
    if sca_schedules["clientlt"] != sca_schedules["matched"]:
        raise ValueError("Existing SCA cells used different schedules")
    if len(sca_schedules["clientlt"]) != args.expected_rounds or any(
        len(clients) != expected_clients_per_round
        for clients in sca_schedules["clientlt"]
    ):
        raise ValueError("SCA executed-client audit has an invalid round/cardinality")
    actual_schedule_hash = _schedule_hash(sca_schedules["clientlt"])
    if any(
        payload["client_schedule_sha256"] != actual_schedule_hash
        for payload in protocols.values()
    ):
        raise ValueError("CAPT protocol schedule differs from actual SCA schedule")
    capt_schedules = {
        "clientlt": _read_capt_schedule(
            directories["capt_clientlt"],
            args.expected_rounds,
            expected_clients_per_round,
        ),
        "matched": _read_capt_schedule(
            directories["capt_matched"],
            args.expected_rounds,
            expected_clients_per_round,
        ),
    }
    if capt_schedules["clientlt"] != capt_schedules["matched"]:
        raise ValueError("CAPT cells executed different client schedules")
    if capt_schedules["clientlt"] != sca_schedules["clientlt"]:
        raise ValueError("CAPT and SCA executed-client schedules differ")

    per_round_rows = [
        _decomposition_row(epoch, metric, comparator, runs)
        for comparator in COMPARATORS
        for metric in METRICS
        for epoch in epochs
    ]
    final_rows = [
        _decomposition_row(final_epoch, metric, comparator, runs)
        for comparator in COMPARATORS
        for metric in METRICS
    ]
    best_common_rows = []
    independent_best_rows = []
    for comparator in COMPARATORS:
        four_names = (
            "capt_clientlt",
            "capt_matched",
            f"ours_{comparator}_clientlt",
            f"ours_{comparator}_matched",
        )
        for metric in METRICS:
            best_epoch = max(
                epochs,
                key=lambda epoch: (
                    sum(runs[name][epoch][metric] for name in four_names)
                    / len(four_names),
                    epoch,
                ),
            )
            row = _decomposition_row(best_epoch, metric, comparator, runs)
            row["selection_rule"] = (
                "max mean metric across CAPT/ours and both topologies at one shared round"
            )
            best_common_rows.append(row)
            for name in four_names:
                cell_epoch = max(
                    epochs, key=lambda epoch: (runs[name][epoch][metric], epoch)
                )
                independent_best_rows.append(
                    {
                        "comparator": comparator,
                        "metric": metric,
                        "cell": name,
                        "best_epoch_index": cell_epoch,
                        "best_communication_round": cell_epoch + 1,
                        "best_value": runs[name][cell_epoch][metric],
                        "eligible_for_gap_decomposition": False,
                    }
                )

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "stage1b_gap_per_round.csv", per_round_rows)
    _write_csv(output_dir / "stage1b_gap_final.csv", final_rows)
    _write_csv(output_dir / "stage1b_gap_best_common_round.csv", best_common_rows)
    _write_csv(
        output_dir / "stage1b_independent_best_descriptive.csv",
        independent_best_rows,
    )

    primary = next(
        row
        for row in final_rows
        if row["comparator"] == "sca" and row["metric"] == "head_tail_h_mean"
    )
    abs_base = abs(float(primary["base_gap"]))
    abs_topology = abs(float(primary["topology_robustness_gap"]))
    directional_diagnosis = (
        "BASE_SUBSTRATE_GAP_DOMINANT"
        if abs_base > abs_topology
        else "TOPOLOGY_ROBUSTNESS_GAP_DOMINANT"
        if abs_topology > abs_base
        else "BALANCED_OR_ZERO_GAPS"
    )
    payload = {
        "schema_version": "stage1b_capt_gap_decomposition_v1",
        "status": "COMPLETE_SINGLE_SEED_DIAGNOSTIC",
        "primary_comparator": "static SCA",
        "primary_metric": "head_tail_h_mean",
        "primary_final": primary,
        "directional_diagnosis": directional_diagnosis,
        "artifact_audit": {
            "all_runs_have_every_expected_round": True,
            "expected_rounds": args.expected_rounds,
            "within_topology_partitions_exactly_equal": True,
            "class_margins_equal_across_topologies": True,
            "client_margins_equal_across_topologies": True,
            "joint_coupling_changed": True,
            "actual_client_schedule_equal": True,
            "capt_executed_client_schedule_audited": True,
            "capt_fixed_every_round_aggregation": True,
            "official_test_controls_future_training": False,
            "protocol_invariant_mismatches": invariant_mismatches,
        },
        "interpretation": (
            "Directional seed-42 decomposition only. BaseGap measures CAPT-vs-ours "
            "on matched topology; TopologyRobustnessGap measures the difference in "
            "topology penalties. Independent per-cell maxima are not decomposed."
        ),
        "outputs": {
            "per_round": "stage1b_gap_per_round.csv",
            "final": "stage1b_gap_final.csv",
            "best_common": "stage1b_gap_best_common_round.csv",
            "independent_best": "stage1b_independent_best_descriptive.csv",
        },
    }
    safe_payload = _json_safe(payload)
    (output_dir / "stage1b_summary.json").write_text(
        json.dumps(safe_payload, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    report = [
        "# Stage-1B CAPT gap decomposition",
        "",
        f"Directional diagnosis: `{directional_diagnosis}`",
        "",
        "## Final round, static SCA comparator",
        "",
        "| metric | BaseGap | Ours topology penalty | CAPT topology penalty | TopologyRobustnessGap | Total CAPT advantage |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in final_rows:
        if row["comparator"] != "sca":
            continue
        report.append(
            f"| {row['metric']} | {row['base_gap']:.4f} | "
            f"{row['ours_topology_penalty']:.4f} | "
            f"{row['capt_topology_penalty']:.4f} | "
            f"{row['topology_robustness_gap']:.4f} | "
            f"{row['capt_total_advantage_on_clientlt']:.4f} |"
        )
    report.extend(
        [
            "",
            "CAPT was run with fixed every-round global aggregation. This preserves "
            "the repository's default CAPT path outside the explicit diagnostic flag "
            "and prevents official-test metrics from controlling future training.",
            "",
            "This is a single-seed directional decomposition, not inferential evidence.",
        ]
    )
    (output_dir / "stage1b_report.md").write_text(
        "\n".join(report) + "\n", encoding="utf-8"
    )
    print(json.dumps(safe_payload, indent=2, sort_keys=True, allow_nan=False))
    return safe_payload


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sca-output-root", type=Path, default=Path("output/online_sca_seed42_v2")
    )
    parser.add_argument(
        "--capt-output-root",
        type=Path,
        default=Path("output/online_sca_seed42_v2/stage1b_capt"),
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--expected-rounds", type=int, default=80)
    args = parser.parse_args()
    args.output_dir = args.output_dir or args.sca_output_root / "stage1b_capt_gap"
    return args


if __name__ == "__main__":
    analyze(parse_args())
