#!/usr/bin/env python
"""Audit and summarize the 2x2 topology x residual-aggregation experiment."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path


CELL_NAMES = (
    "clientlt_residual_fedavg",
    "clientlt_sca",
    "matched_residual_fedavg",
    "matched_sca",
)
BASE_METRICS = (
    "overall_acc",
    "non_tail_acc",
    "bottom20_tail_acc",
    "macro_per_class_acc",
    "macro_f1",
)
METRICS = BASE_METRICS + ("head_tail_h_mean",)


def _read_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _read_metrics(run_dir: Path) -> dict[int, dict[str, float]]:
    path = run_dir / "round_metrics.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    rows: dict[int, dict[str, float]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for raw in csv.DictReader(handle):
            epoch = int(raw["epoch"])
            if epoch < 0:
                continue
            if epoch in rows:
                raise ValueError(f"Duplicate epoch {epoch} in {path}")
            row = {metric: float(raw[metric]) for metric in BASE_METRICS}
            head = row["non_tail_acc"]
            tail = row["bottom20_tail_acc"]
            row["head_tail_h_mean"] = (
                2.0 * head * tail / (head + tail) if head + tail > 0 else 0.0
            )
            rows[epoch] = row
    if not rows:
        raise ValueError(f"No evaluated training rounds in {path}")
    return rows


def _read_count_matrix(run_dir: Path) -> list[list[int]]:
    path = run_dir / "client_class_counts.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    matrix = []
    client_ids = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            client_ids.append(int(row["client_id"]))
            matrix.append(
                [int(row[key]) for key in row if key.startswith("class_")]
            )
    if client_ids != list(range(len(client_ids))):
        raise ValueError(f"Non-canonical client ordering in {path}: {client_ids}")
    if not matrix or any(len(row) != len(matrix[0]) for row in matrix):
        raise ValueError(f"Malformed client-class matrix: {path}")
    return matrix


def _row_sums(matrix):
    return [sum(row) for row in matrix]


def _column_sums(matrix):
    return [sum(row[column] for row in matrix) for column in range(len(matrix[0]))]


def _matrix_hash(matrix) -> str:
    payload = json.dumps(matrix, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _topology_descriptives(matrix) -> dict:
    num_clients = len(matrix)
    num_classes = len(matrix[0])
    support_by_client = [sum(value > 0 for value in row) for row in matrix]
    support_by_class = [
        sum(matrix[client_id][class_id] > 0 for client_id in range(num_clients))
        for class_id in range(num_classes)
    ]
    class_totals = _column_sums(matrix)
    tail_count = max(1, int(round(0.2 * num_classes)))
    tail_ids = sorted(
        range(num_classes), key=lambda class_id: (class_totals[class_id], -class_id)
    )[:tail_count]
    effective_clients = []
    for class_id in tail_ids:
        values = [matrix[client_id][class_id] for client_id in range(num_clients)]
        denominator = sum(value * value for value in values)
        total = sum(values)
        effective_clients.append(total * total / denominator if denominator else 0.0)
    return {
        "zero_cell_ratio": sum(
            value == 0 for row in matrix for value in row
        )
        / float(num_clients * num_classes),
        "mean_local_supported_classes": sum(support_by_client) / float(num_clients),
        "mean_clients_per_class": sum(support_by_class) / float(num_classes),
        "tail_mean_clients_per_class": sum(support_by_class[class_id] for class_id in tail_ids)
        / float(len(tail_ids)),
        "tail_mean_effective_clients": sum(effective_clients) / float(len(effective_clients)),
    }


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _factorial_row(epoch: int, metric: str, cells) -> dict:
    clt_rf = cells["clientlt_residual_fedavg"][epoch][metric]
    clt_sca = cells["clientlt_sca"][epoch][metric]
    dir_rf = cells["matched_residual_fedavg"][epoch][metric]
    dir_sca = cells["matched_sca"][epoch][metric]
    delta_clientlt = clt_sca - clt_rf
    delta_matched = dir_sca - dir_rf
    return {
        "epoch_index": epoch,
        "communication_round": epoch + 1,
        "metric": metric,
        "clientlt_residual_fedavg": clt_rf,
        "clientlt_sca": clt_sca,
        "matched_residual_fedavg": dir_rf,
        "matched_sca": dir_sca,
        "delta_clientlt": delta_clientlt,
        "delta_matched_dirichlet": delta_matched,
        "difference_in_differences": delta_clientlt - delta_matched,
    }


def _protocol_audit(run_dirs: dict[str, Path]) -> tuple[dict, list[str]]:
    protocols = {}
    warnings = []
    for cell, run_dir in run_dirs.items():
        path = run_dir / "online_sca_protocol.json"
        if path.exists():
            protocols[cell] = _read_json(path)
        else:
            warnings.append(f"missing protocol metadata for {cell}: {path}")
    invariant_keys = (
        "seed",
        "split_seed",
        "client_schedule_sha256",
        "rounds",
        "num_users",
        "frac",
        "local_epochs",
        "residual_scale",
        "residual_clamp",
        "residual_lr_multiplier",
        "residual_use_bias",
        "tail_class_ids",
    )
    mismatches = {}
    for key in invariant_keys:
        present = {
            cell: payload[key] for cell, payload in protocols.items() if key in payload
        }
        if len(present) == len(run_dirs) and len({json.dumps(v, sort_keys=True) for v in present.values()}) > 1:
            mismatches[key] = present
        elif len(present) != len(run_dirs):
            warnings.append(f"protocol key {key!r} is unavailable in one or more cells")
    return {
        "protocols_found": sorted(protocols),
        "invariant_mismatches": mismatches,
        "passed": not mismatches,
    }, warnings


def analyze(args) -> dict:
    run_dirs = {
        "clientlt_residual_fedavg": args.clientlt_residual_fedavg_dir,
        "clientlt_sca": args.clientlt_sca_dir,
        "matched_residual_fedavg": args.matched_residual_fedavg_dir,
        "matched_sca": args.matched_sca_dir,
    }
    cells = {cell: _read_metrics(path) for cell, path in run_dirs.items()}
    round_sets = {cell: set(rows) for cell, rows in cells.items()}
    first_rounds = next(iter(round_sets.values()))
    if any(rounds != first_rounds for rounds in round_sets.values()):
        raise ValueError(
            "The four cells do not contain identical evaluated rounds: "
            + json.dumps({key: sorted(value) for key, value in round_sets.items()})
        )
    epochs = sorted(first_rounds)

    matrices = {cell: _read_count_matrix(path) for cell, path in run_dirs.items()}
    architecture_pair_equal = {
        "clientlt": matrices["clientlt_residual_fedavg"] == matrices["clientlt_sca"],
        "matched_dirichlet": (
            matrices["matched_residual_fedavg"] == matrices["matched_sca"]
        ),
    }
    if not all(architecture_pair_equal.values()):
        raise ValueError(
            "Aggregation cells within the same topology use different partitions: "
            f"{architecture_pair_equal}"
        )
    clientlt_matrix = matrices["clientlt_sca"]
    matched_matrix = matrices["matched_sca"]
    row_margins_equal = _row_sums(clientlt_matrix) == _row_sums(matched_matrix)
    class_margins_equal = _column_sums(clientlt_matrix) == _column_sums(matched_matrix)
    coupling_changed = clientlt_matrix != matched_matrix
    clientlt_descriptives = _topology_descriptives(clientlt_matrix)
    matched_descriptives = _topology_descriptives(matched_matrix)
    broader_tail_exposure = (
        matched_descriptives["tail_mean_effective_clients"]
        > clientlt_descriptives["tail_mean_effective_clients"]
        and matched_descriptives["tail_mean_clients_per_class"]
        > clientlt_descriptives["tail_mean_clients_per_class"]
    )
    topology_audit = {
        "within_topology_partition_equal": architecture_pair_equal,
        "client_margins_nk_equal": row_margins_equal,
        "class_margins_nc_equal": class_margins_equal,
        "joint_coupling_changed": coupling_changed,
        "clientlt_matrix_sha256": _matrix_hash(clientlt_matrix),
        "matched_dirichlet_matrix_sha256": _matrix_hash(matched_matrix),
        "clientlt_descriptives": clientlt_descriptives,
        "matched_dirichlet_descriptives": matched_descriptives,
        "expected_broader_tail_exposure": broader_tail_exposure,
        "passed": (
            all(architecture_pair_equal.values())
            and row_margins_equal
            and class_margins_equal
            and coupling_changed
        ),
    }
    if not topology_audit["passed"]:
        raise ValueError(f"Fixed-marginal topology audit failed: {topology_audit}")

    per_round = [
        _factorial_row(epoch, metric, cells)
        for epoch in epochs
        for metric in METRICS
    ]
    final_epoch = epochs[-1]
    final_rows = [
        _factorial_row(final_epoch, metric, cells) for metric in METRICS
    ]
    best_common_rows = []
    for metric in METRICS:
        best_epoch = max(
            epochs,
            key=lambda epoch: (
                sum(cells[cell][epoch][metric] for cell in CELL_NAMES) / len(CELL_NAMES),
                epoch,
            ),
        )
        row = _factorial_row(best_epoch, metric, cells)
        row["selection_rule"] = "max mean metric across all four cells at one common round"
        best_common_rows.append(row)

    best_per_cell_rows = []
    for metric in METRICS:
        for cell in CELL_NAMES:
            best_epoch = max(epochs, key=lambda epoch: (cells[cell][epoch][metric], epoch))
            best_per_cell_rows.append(
                {
                    "metric": metric,
                    "cell": cell,
                    "best_epoch_index": best_epoch,
                    "best_communication_round": best_epoch + 1,
                    "best_value": cells[cell][best_epoch][metric],
                    "causal_did_eligible": False,
                }
            )

    protocol_audit, warnings = _protocol_audit(run_dirs)
    if not broader_tail_exposure:
        warnings.append(
            "matched Dirichlet did not broaden both tail support count and tail "
            "effective-client number; tune --matched-beta before using DiD as a "
            "Client-LT-specific causal contrast"
        )
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "factorial_per_round.csv", per_round)
    _write_csv(output_dir / "factorial_final.csv", final_rows)
    _write_csv(output_dir / "factorial_best_common_round.csv", best_common_rows)
    _write_csv(output_dir / "factorial_best_per_cell_descriptive.csv", best_per_cell_rows)

    primary = next(
        row for row in final_rows if row["metric"] == args.primary_metric
    )
    payload = {
        "schema_version": "sca_factorial_analysis_v1",
        "run_dirs": {cell: str(path.resolve()) for cell, path in run_dirs.items()},
        "evaluated_epoch_indices": epochs,
        "final_epoch_index": final_epoch,
        "final_communication_round": final_epoch + 1,
        "primary_metric": args.primary_metric,
        "primary_final": primary,
        "aggregation_net_gain_on_clientlt": primary["delta_clientlt"] > 0,
        "topology_specific_amplification": primary["difference_in_differences"] > 0,
        "interpretation": (
            "descriptive single-run result; seed-level uncertainty is required for inference"
        ),
        "topology_audit": topology_audit,
        "protocol_audit": protocol_audit,
        "did_design_ready": (
            topology_audit["passed"]
            and broader_tail_exposure
            and protocol_audit["passed"]
        ),
        "warnings": warnings,
        "best_checkpoint_note": (
            "DiD is reported only at a shared selected round. Independent per-cell maxima "
            "are exported as descriptive values and must not be used as a causal DiD."
        ),
    }
    (output_dir / "factorial_summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return payload


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-root", type=Path, default=Path("output/online_sca_seed42_v2")
    )
    parser.add_argument("--clientlt-residual-fedavg-dir", type=Path)
    parser.add_argument("--clientlt-sca-dir", type=Path)
    parser.add_argument("--matched-residual-fedavg-dir", type=Path)
    parser.add_argument("--matched-sca-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--primary-metric", choices=METRICS, default="head_tail_h_mean")
    args = parser.parse_args()
    root = args.output_root
    args.clientlt_residual_fedavg_dir = (
        args.clientlt_residual_fedavg_dir or root / "residual_fedavg_clientlt"
    )
    args.clientlt_sca_dir = args.clientlt_sca_dir or root / "online_sca"
    args.matched_residual_fedavg_dir = (
        args.matched_residual_fedavg_dir
        or root / "residual_fedavg_matched_dirichlet"
    )
    args.matched_sca_dir = (
        args.matched_sca_dir or root / "online_sca_matched_dirichlet"
    )
    args.output_dir = args.output_dir or root / "factorial_analysis"
    return args


if __name__ == "__main__":
    analyze(parse_args())
