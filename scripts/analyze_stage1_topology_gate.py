#!/usr/bin/env python3
"""Stage-1A: test whether exposure topology supports an adaptive SCA gate.

This script is deliberately artifact-only. It reconstructs per-round class
support from the saved client-class matrices and the actually logged client
schedule. It never substitutes an aggregated row delta for unavailable
client-level update agreement.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path

import numpy as np


TOPOLOGY_RUNS = {
    "clientlt": ("residual_fedavg_clientlt", "online_sca"),
    "matched_dirichlet": (
        "residual_fedavg_matched_dirichlet",
        "online_sca_matched_dirichlet",
    ),
}
STAGES = (
    ("early", 1, 20),
    ("middle", 21, 50),
    ("late", 51, 80),
)


def _stage(round_id: int) -> str:
    for name, first, last in STAGES:
        if first <= round_id <= last:
            return name
    return "outside_prespecified_window"


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
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


def _fmt(value) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "N/A"
    return f"{numeric:.4f}" if math.isfinite(numeric) else "N/A"


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
        raise ValueError(f"Non-canonical client ids in {path}: {ids}")
    matrix = np.asarray(rows, dtype=np.int64)
    if matrix.ndim != 2 or matrix.size == 0 or np.any(matrix < 0):
        raise ValueError(f"Malformed client-class matrix: {path}")
    return matrix


def _read_schedule(run_dir: Path) -> dict[int, tuple[int, ...]]:
    path = run_dir / "lora_aggregation_weights.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing actual selected-client audit required by Stage-1A: {path}"
        )
    grouped: dict[int, list[int]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for raw in csv.DictReader(handle):
            epoch = int(raw["epoch_index"])
            grouped.setdefault(epoch, []).append(int(raw["client_id"]))
    if not grouped:
        raise ValueError(f"No selected clients in {path}")
    schedule = {}
    for epoch, clients in grouped.items():
        if len(clients) != len(set(clients)):
            raise ValueError(f"Duplicate clients in epoch {epoch} of {path}")
        schedule[epoch] = tuple(sorted(clients))
    return schedule


def _read_per_class_accuracy(run_dir: Path) -> dict[int, dict[int, float]]:
    pattern = re.compile(r"per_class_accuracy_epoch_(-?\d+)\.csv$")
    result = {}
    for path in run_dir.glob("per_class_accuracy_epoch_*.csv"):
        match = pattern.search(path.name)
        if not match:
            continue
        epoch = int(match.group(1))
        if epoch < 0:
            continue
        values = {}
        with path.open(newline="", encoding="utf-8") as handle:
            for raw in csv.DictReader(handle):
                values[int(raw["class_id"])] = float(raw["per_class_acc"])
        if epoch in result:
            raise ValueError(f"Duplicate per-class epoch {epoch} in {run_dir}")
        result[epoch] = values
    if not result:
        raise FileNotFoundError(f"No per-class accuracy files in {run_dir}")
    return result


def _tail_ids(class_totals: np.ndarray, ratio: float) -> list[int]:
    count = min(len(class_totals), max(1, int(round(len(class_totals) * ratio))))
    return sorted(
        range(len(class_totals)),
        key=lambda class_id: (int(class_totals[class_id]), -class_id),
    )[:count]


def _topology_metrics(matrix: np.ndarray) -> dict[str, np.ndarray]:
    totals = matrix.sum(axis=0).astype(np.float64)
    support = (matrix > 0).sum(axis=0).astype(np.int64)
    sorted_counts = np.sort(matrix, axis=0)[::-1]
    top1 = np.divide(
        sorted_counts[0], totals, out=np.zeros_like(totals), where=totals > 0
    )
    denominator = np.square(matrix.astype(np.float64)).sum(axis=0)
    effective = np.divide(
        np.square(totals),
        denominator,
        out=np.zeros_like(totals),
        where=denominator > 0,
    )
    return {"support": support, "top1": top1, "effective": effective}


def _fixed_margin_null(
    matrix: np.ndarray, *, samples: int, seed: int
) -> dict[str, np.ndarray]:
    """Random-coupling null conditional on exact client and class margins."""
    if samples < 1:
        raise ValueError("null samples must be positive")
    client_totals = matrix.sum(axis=1).astype(np.int64)
    class_totals = matrix.sum(axis=0).astype(np.int64)
    labels = np.repeat(np.arange(matrix.shape[1], dtype=np.int64), class_totals)
    if len(labels) != int(client_totals.sum()):
        raise RuntimeError("Fixed-margin null received inconsistent margins")
    boundaries = np.cumsum(client_totals)
    rng = np.random.default_rng(int(seed))
    effective_samples = np.empty((samples, matrix.shape[1]), dtype=np.float64)
    support_samples = np.empty((samples, matrix.shape[1]), dtype=np.float64)
    top1_samples = np.empty((samples, matrix.shape[1]), dtype=np.float64)
    for sample_index in range(samples):
        permuted = rng.permutation(labels)
        null_matrix = np.zeros_like(matrix)
        first = 0
        for client_id, last in enumerate(boundaries):
            null_matrix[client_id] = np.bincount(
                permuted[first:last], minlength=matrix.shape[1]
            )
            first = int(last)
        metrics = _topology_metrics(null_matrix)
        effective_samples[sample_index] = metrics["effective"]
        support_samples[sample_index] = metrics["support"]
        top1_samples[sample_index] = metrics["top1"]
    observed = _topology_metrics(matrix)
    effective_mean = effective_samples.mean(axis=0)
    rho = np.maximum(
        0.0,
        np.divide(
            effective_mean - observed["effective"],
            effective_mean,
            out=np.zeros_like(effective_mean),
            where=effective_mean > 0,
        ),
    )
    return {
        "effective_mean": effective_mean,
        "effective_q025": np.quantile(effective_samples, 0.025, axis=0),
        "effective_q975": np.quantile(effective_samples, 0.975, axis=0),
        "support_mean": support_samples.mean(axis=0),
        "top1_mean": top1_samples.mean(axis=0),
        "rho": rho,
        "concentration_p": (
            1.0 + (effective_samples <= observed["effective"]).sum(axis=0)
        )
        / float(samples + 1),
        "_effective_samples": effective_samples,
    }


def _rank(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda index: values[index])
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        average = (start + 1 + end) / 2.0
        for position in range(start, end):
            ranks[order[position]] = average
        start = end
    return ranks


def _pearson(x: list[float], y: list[float]) -> float:
    if len(x) != len(y) or len(x) < 2:
        return math.nan
    x_array = np.asarray(x, dtype=np.float64)
    y_array = np.asarray(y, dtype=np.float64)
    x_centered = x_array - x_array.mean()
    y_centered = y_array - y_array.mean()
    denominator = float(np.linalg.norm(x_centered) * np.linalg.norm(y_centered))
    return float(np.dot(x_centered, y_centered) / denominator) if denominator else math.nan


def _spearman(x: list[float], y: list[float]) -> float:
    return _pearson(_rank(x), _rank(y))


def _permutation_p(
    x: list[float], y: list[float], observed: float, *, samples: int, seed: int
) -> float:
    if not math.isfinite(observed) or samples < 1:
        return math.nan
    rng = np.random.default_rng(int(seed))
    y_array = np.asarray(y, dtype=np.float64)
    exceed = 0
    for _ in range(samples):
        candidate = _spearman(x, rng.permutation(y_array).tolist())
        if math.isfinite(candidate) and abs(candidate) >= abs(observed) - 1e-12:
            exceed += 1
    return (exceed + 1.0) / (samples + 1.0)


def _balanced_quartiles(class_rows: list[dict], key: str) -> dict[int, str]:
    ordered = sorted(class_rows, key=lambda row: (float(row[key]), int(row["class_id"])))
    result = {}
    for position, row in enumerate(ordered):
        quartile = min(4, int(4 * position / max(len(ordered), 1)) + 1)
        result[int(row["class_id"])] = f"Q{quartile}"
    return result


def _mean(rows: list[dict], key: str) -> float:
    values = [float(row[key]) for row in rows if math.isfinite(float(row[key]))]
    return sum(values) / len(values) if values else math.nan


def analyze(args) -> dict:
    expected_rounds = int(getattr(args, "expected_rounds", 80))
    if expected_rounds != STAGES[-1][2]:
        raise ValueError(
            f"Stage-1A is prespecified for rounds 1--{STAGES[-1][2]}, got "
            f"{expected_rounds}"
        )
    run_dirs = {
        topology: {
            "residual_fedavg": args.output_root / conditions[0],
            "sca": args.output_root / conditions[1],
        }
        for topology, conditions in TOPOLOGY_RUNS.items()
    }
    matrices = {}
    schedules = {}
    accuracies = {}
    for topology, methods in run_dirs.items():
        method_matrices = {method: _read_matrix(path) for method, path in methods.items()}
        if not np.array_equal(
            method_matrices["residual_fedavg"], method_matrices["sca"]
        ):
            raise ValueError(f"Within-topology partition mismatch: {topology}")
        matrices[topology] = method_matrices["sca"]
        method_schedules = {method: _read_schedule(path) for method, path in methods.items()}
        if method_schedules["residual_fedavg"] != method_schedules["sca"]:
            raise ValueError(f"Within-topology client schedule mismatch: {topology}")
        schedules[topology] = method_schedules["sca"]
        accuracies[topology] = {
            method: _read_per_class_accuracy(path) for method, path in methods.items()
        }
        if set(accuracies[topology]["residual_fedavg"]) != set(
            accuracies[topology]["sca"]
        ):
            raise ValueError(f"Within-topology evaluated-round mismatch: {topology}")
        if sorted(method_schedules["sca"]) != list(range(expected_rounds)):
            raise ValueError(
                f"Strict Stage-1A requires exactly epochs 0--{expected_rounds - 1}: "
                f"{topology}"
            )
        expected_class_ids = set(range(matrices[topology].shape[1]))
        for method, by_epoch in accuracies[topology].items():
            for epoch, values in by_epoch.items():
                if set(values) != expected_class_ids:
                    raise ValueError(
                        f"Per-class metric coverage mismatch: {topology}/{method}/"
                        f"epoch_{epoch}"
                    )
        for method, schedule in method_schedules.items():
            if any(
                client_id < 0 or client_id >= matrices[topology].shape[0]
                for clients in schedule.values()
                for client_id in clients
            ):
                raise ValueError(
                    f"Out-of-range client in schedule: {topology}/{method}"
                )

    if not np.array_equal(matrices["clientlt"].sum(axis=0), matrices["matched_dirichlet"].sum(axis=0)):
        raise ValueError("Client-LT and matched Dirichlet class margins differ")
    if not np.array_equal(matrices["clientlt"].sum(axis=1), matrices["matched_dirichlet"].sum(axis=1)):
        raise ValueError("Client-LT and matched Dirichlet client margins differ")
    if schedules["clientlt"] != schedules["matched_dirichlet"]:
        raise ValueError("The four cells did not use one common selected-client schedule")
    if matrices["clientlt"].shape != matrices["matched_dirichlet"].shape:
        raise ValueError("Topology matrices have different shapes")

    class_totals = matrices["clientlt"].sum(axis=0)
    tail_ids = _tail_ids(class_totals, args.tail_class_ratio)
    tail_set = set(tail_ids)
    null = _fixed_margin_null(
        matrices["clientlt"], samples=args.null_samples, seed=args.null_seed
    )
    topology_static = {}
    for topology, matrix in matrices.items():
        observed = _topology_metrics(matrix)
        rho = np.maximum(
            0.0,
            np.divide(
                null["effective_mean"] - observed["effective"],
                null["effective_mean"],
                out=np.zeros_like(null["effective_mean"]),
                where=null["effective_mean"] > 0,
            ),
        )
        concentration_p = (
            1.0
            + (null["_effective_samples"] <= observed["effective"]).sum(axis=0)
        ) / float(args.null_samples + 1)
        topology_static[topology] = {
            **observed,
            "rho": rho,
            "concentration_p": concentration_p,
        }

    class_round_rows = []
    for topology, matrix in matrices.items():
        sample_counts = matrix.sum(axis=1).astype(np.float64)
        absence_streak = np.zeros(matrix.shape[1], dtype=np.int64)
        epochs = sorted(accuracies[topology]["sca"])
        if epochs != sorted(schedules[topology]):
            raise ValueError(
                f"Evaluated rounds and schedule audit differ for {topology}: "
                f"metrics={epochs[:3]}...{epochs[-3:]} schedule="
                f"{sorted(schedules[topology])[:3]}...{sorted(schedules[topology])[-3:]}"
            )
        for epoch in epochs:
            selected = list(schedules[topology][epoch])
            selected_total_samples = float(sample_counts[selected].sum())
            supporter_mask = matrix[selected] > 0
            supporter_count = supporter_mask.sum(axis=0)
            supporter_class_mass = matrix[selected].sum(axis=0)
            scalar_mass = np.divide(
                (supporter_mask * sample_counts[selected, None]).sum(axis=0),
                selected_total_samples,
                out=np.zeros(matrix.shape[1], dtype=np.float64),
                where=selected_total_samples > 0,
            )
            absence_streak = np.where(supporter_count > 0, 0, absence_streak + 1)
            for class_id in range(matrix.shape[1]):
                sca_acc = accuracies[topology]["sca"][epoch][class_id]
                control_acc = accuracies[topology]["residual_fedavg"][epoch][class_id]
                supporters = [
                    selected[position]
                    for position in range(len(selected))
                    if supporter_mask[position, class_id]
                ]
                class_round_rows.append(
                    {
                        "topology": topology,
                        "epoch_index": epoch,
                        "communication_round": epoch + 1,
                        "stage": _stage(epoch + 1),
                        "class_id": class_id,
                        "class_group": "tail" if class_id in tail_set else "head",
                        "global_class_count": int(class_totals[class_id]),
                        "sca_accuracy": sca_acc,
                        "residual_fedavg_accuracy": control_acc,
                        "sca_minus_residual_fedavg": sca_acc - control_acc,
                        "supporter_count": int(supporter_count[class_id]),
                        "supporter_ids": ",".join(str(value) for value in supporters),
                        "supporter_class_sample_mass": int(supporter_class_mass[class_id]),
                        "supporter_global_class_fraction": float(
                            supporter_class_mass[class_id] / max(class_totals[class_id], 1)
                        ),
                        "scalar_fedavg_supporter_weight_mass": float(scalar_mass[class_id]),
                        "absent_this_round": int(supporter_count[class_id] == 0),
                        "absence_streak": int(absence_streak[class_id]),
                        "num_support_clients": int(
                            topology_static[topology]["support"][class_id]
                        ),
                        "effective_carrier": float(
                            topology_static[topology]["effective"][class_id]
                        ),
                        "top1_client_mass": float(
                            topology_static[topology]["top1"][class_id]
                        ),
                        "null_effective_carrier_mean": float(
                            null["effective_mean"][class_id]
                        ),
                        "null_effective_carrier_q025": float(
                            null["effective_q025"][class_id]
                        ),
                        "null_effective_carrier_q975": float(
                            null["effective_q975"][class_id]
                        ),
                        "null_support_clients_mean": float(null["support_mean"][class_id]),
                        "null_top1_client_mass_mean": float(null["top1_mean"][class_id]),
                        "excess_exposure_concentration_rho": float(
                            topology_static[topology]["rho"][class_id]
                        ),
                        "null_concentration_empirical_p": float(
                            topology_static[topology]["concentration_p"][class_id]
                        ),
                        "update_agreement": "",
                        "update_agreement_available": False,
                    }
                )

    grouped: dict[tuple[str, int, str], list[dict]] = {}
    for row in class_round_rows:
        grouped.setdefault(
            (row["topology"], int(row["class_id"]), row["stage"]), []
        ).append(row)
    class_stage_rows = []
    for (topology, class_id, stage), rows in sorted(grouped.items()):
        if stage == "outside_prespecified_window":
            continue
        class_stage_rows.append(
            {
                "topology": topology,
                "class_id": class_id,
                "class_group": rows[0]["class_group"],
                "stage": stage,
                "round_count": len(rows),
                "rho": rows[0]["excess_exposure_concentration_rho"],
                "effective_carrier": rows[0]["effective_carrier"],
                "top1_client_mass": rows[0]["top1_client_mass"],
                "mean_sca_accuracy": _mean(rows, "sca_accuracy"),
                "mean_residual_fedavg_accuracy": _mean(
                    rows, "residual_fedavg_accuracy"
                ),
                "mean_sca_minus_residual_fedavg": _mean(
                    rows, "sca_minus_residual_fedavg"
                ),
                "mean_supporter_count": _mean(rows, "supporter_count"),
                "mean_supporter_class_sample_mass": _mean(
                    rows, "supporter_class_sample_mass"
                ),
                "mean_scalar_fedavg_supporter_weight_mass": _mean(
                    rows, "scalar_fedavg_supporter_weight_mass"
                ),
                "absence_round_fraction": _mean(rows, "absent_this_round"),
                "max_absence_streak": max(int(row["absence_streak"]) for row in rows),
            }
        )

    paired_class_stage_rows = []
    stage_index = {
        (row["topology"], int(row["class_id"]), row["stage"]): row
        for row in class_stage_rows
    }
    for class_id in range(matrices["clientlt"].shape[1]):
        for stage, _, _ in STAGES:
            clientlt = stage_index[("clientlt", class_id, stage)]
            matched = stage_index[("matched_dirichlet", class_id, stage)]
            paired_class_stage_rows.append(
                {
                    "class_id": class_id,
                    "class_group": clientlt["class_group"],
                    "stage": stage,
                    "clientlt_rho": clientlt["rho"],
                    "matched_dirichlet_rho": matched["rho"],
                    "clientlt_minus_matched_rho": (
                        clientlt["rho"] - matched["rho"]
                    ),
                    "clientlt_aggregation_gain": clientlt[
                        "mean_sca_minus_residual_fedavg"
                    ],
                    "matched_dirichlet_aggregation_gain": matched[
                        "mean_sca_minus_residual_fedavg"
                    ],
                    "aggregation_gain_difference_in_differences": (
                        clientlt["mean_sca_minus_residual_fedavg"]
                        - matched["mean_sca_minus_residual_fedavg"]
                    ),
                    "clientlt_absence_round_fraction": clientlt[
                        "absence_round_fraction"
                    ],
                    "matched_absence_round_fraction": matched[
                        "absence_round_fraction"
                    ],
                    "absence_fraction_difference": (
                        clientlt["absence_round_fraction"]
                        - matched["absence_round_fraction"]
                    ),
                }
            )

    spearman_rows = []
    predictors = (
        "rho",
        "effective_carrier",
        "top1_client_mass",
        "mean_supporter_count",
        "mean_scalar_fedavg_supporter_weight_mass",
        "absence_round_fraction",
        "max_absence_streak",
    )
    for topology in TOPOLOGY_RUNS:
        for stage, _, _ in STAGES:
            for class_group in ("all", "head", "tail"):
                rows = [
                    row
                    for row in class_stage_rows
                    if row["topology"] == topology
                    and row["stage"] == stage
                    and (
                        class_group == "all"
                        or row["class_group"] == class_group
                    )
                ]
                for predictor in predictors:
                    x = [float(row[predictor]) for row in rows]
                    y = [float(row["mean_sca_minus_residual_fedavg"]) for row in rows]
                    correlation = _spearman(x, y)
                    spearman_rows.append(
                        {
                            "topology": topology,
                            "stage": stage,
                            "class_group": class_group,
                            "predictor": predictor,
                            "outcome": "mean_sca_minus_residual_fedavg",
                            "n_classes": len(rows),
                            "spearman_rho": correlation,
                            "two_sided_permutation_p_exploratory": _permutation_p(
                                x,
                                y,
                                correlation,
                                samples=args.permutation_samples,
                                seed=args.permutation_seed
                                + len(spearman_rows) * 1009,
                            ),
                            "inference_unit_warning": "single-seed class-level exploratory association",
                        }
                    )
    for stage, _, _ in STAGES:
        for class_group in ("all", "head", "tail"):
            rows = [
                row
                for row in paired_class_stage_rows
                if row["stage"] == stage
                and (class_group == "all" or row["class_group"] == class_group)
            ]
            x = [float(row["clientlt_minus_matched_rho"]) for row in rows]
            y = [
                float(row["aggregation_gain_difference_in_differences"])
                for row in rows
            ]
            correlation = _spearman(x, y)
            spearman_rows.append(
                {
                    "topology": "clientlt_minus_matched_dirichlet",
                    "stage": stage,
                    "class_group": class_group,
                    "predictor": "clientlt_minus_matched_rho",
                    "outcome": "aggregation_gain_difference_in_differences",
                    "n_classes": len(rows),
                    "spearman_rho": correlation,
                    "two_sided_permutation_p_exploratory": _permutation_p(
                        x,
                        y,
                        correlation,
                        samples=args.permutation_samples,
                        seed=args.permutation_seed + len(spearman_rows) * 1009,
                    ),
                    "inference_unit_warning": "paired single-seed class-level exploratory association",
                }
            )

    quartile_rows = []
    quartile_assignments = {}
    for topology in TOPOLOGY_RUNS:
        topology_tail = [
            row
            for row in class_stage_rows
            if row["topology"] == topology
            and row["stage"] == "early"
            and row["class_group"] == "tail"
        ]
        quartiles = _balanced_quartiles(topology_tail, "rho")
        quartile_assignments[topology] = quartiles
        for stage, _, _ in STAGES:
            stage_rows = [
                row
                for row in class_stage_rows
                if row["topology"] == topology
                and row["stage"] == stage
                and row["class_group"] == "tail"
            ]
            for quartile in ("Q1", "Q2", "Q3", "Q4"):
                rows = [
                    row
                    for row in stage_rows
                    if quartiles[int(row["class_id"])] == quartile
                ]
                quartile_rows.append(
                    {
                        "topology": topology,
                        "stage": stage,
                        "rho_quartile": quartile,
                        "interpretation": (
                            "broadest_or_lowest_excess_concentration"
                            if quartile == "Q1"
                            else "narrowest_or_highest_excess_concentration"
                            if quartile == "Q4"
                            else "intermediate"
                        ),
                        "n_classes": len(rows),
                        # A tiny synthetic dataset (or an unusually small tail)
                        # may not populate all four bins.  Keep the prespecified
                        # Q1--Q4 schema and mark an empty bin as unavailable
                        # instead of silently merging bins or inventing values.
                        "rho_min": (
                            min(float(row["rho"]) for row in rows)
                            if rows
                            else float("nan")
                        ),
                        "rho_max": (
                            max(float(row["rho"]) for row in rows)
                            if rows
                            else float("nan")
                        ),
                        "mean_sca_accuracy": _mean(rows, "mean_sca_accuracy"),
                        "mean_residual_fedavg_accuracy": _mean(
                            rows, "mean_residual_fedavg_accuracy"
                        ),
                        "mean_sca_minus_residual_fedavg": _mean(
                            rows, "mean_sca_minus_residual_fedavg"
                        ),
                        "mean_absence_round_fraction": _mean(
                            rows, "absence_round_fraction"
                        ),
                        "mean_max_absence_streak": _mean(rows, "max_absence_streak"),
                    }
                )

    contribution_rows = []
    for topology in TOPOLOGY_RUNS:
        assignments = quartile_assignments[topology]
        for stage, _, _ in STAGES:
            rows = [
                row
                for row in class_stage_rows
                if row["topology"] == topology
                and row["stage"] == stage
                and row["class_group"] == "tail"
            ]
            positive_total = sum(
                max(float(row["mean_sca_minus_residual_fedavg"]), 0.0)
                for row in rows
            )
            negative_total = sum(
                max(-float(row["mean_sca_minus_residual_fedavg"]), 0.0)
                for row in rows
            )
            for quartile in ("Q1", "Q2", "Q3", "Q4"):
                selected = [
                    row
                    for row in rows
                    if assignments[int(row["class_id"])] == quartile
                ]
                positive = sum(
                    max(float(row["mean_sca_minus_residual_fedavg"]), 0.0)
                    for row in selected
                )
                negative = sum(
                    max(-float(row["mean_sca_minus_residual_fedavg"]), 0.0)
                    for row in selected
                )
                contribution_rows.append(
                    {
                        "topology": topology,
                        "stage": stage,
                        "rho_quartile": quartile,
                        "n_classes": len(selected),
                        "signed_gain_sum": sum(
                            float(row["mean_sca_minus_residual_fedavg"])
                            for row in selected
                        ),
                        "positive_gain_sum": positive,
                        "positive_gain_share": (
                            positive / positive_total
                            if positive_total > 0
                            else float("nan")
                        ),
                        "negative_gain_magnitude_sum": negative,
                        "negative_gain_share": (
                            negative / negative_total
                            if negative_total > 0
                            else float("nan")
                        ),
                    }
                )

    absence_rows = []
    for topology in TOPOLOGY_RUNS:
        by_class = {
            class_id: {
                row["stage"]: row
                for row in class_stage_rows
                if row["topology"] == topology
                and int(row["class_id"]) == class_id
                and row["class_group"] == "tail"
            }
            for class_id in tail_ids
        }
        for class_id, stages in by_class.items():
            if not {"early", "late"}.issubset(stages):
                continue
            early = stages["early"]
            late = stages["late"]
            all_rounds = [
                row
                for row in class_round_rows
                if row["topology"] == topology
                and int(row["class_id"]) == class_id
            ]
            absence_rows.append(
                {
                    "topology": topology,
                    "class_id": class_id,
                    "rho": early["rho"],
                    "overall_absence_round_fraction": _mean(
                        all_rounds, "absent_this_round"
                    ),
                    "overall_max_absence_streak": max(
                        int(row["absence_streak"]) for row in all_rounds
                    ),
                    "early_sca_accuracy": early["mean_sca_accuracy"],
                    "late_sca_accuracy": late["mean_sca_accuracy"],
                    "sca_accuracy_late_minus_early": (
                        late["mean_sca_accuracy"] - early["mean_sca_accuracy"]
                    ),
                    "early_aggregation_gain": early[
                        "mean_sca_minus_residual_fedavg"
                    ],
                    "late_aggregation_gain": late[
                        "mean_sca_minus_residual_fedavg"
                    ],
                    "aggregation_gain_late_minus_early": (
                        late["mean_sca_minus_residual_fedavg"]
                        - early["mean_sca_minus_residual_fedavg"]
                    ),
                }
            )

    decay_association_rows = []
    for topology in TOPOLOGY_RUNS:
        rows = [row for row in absence_rows if row["topology"] == topology]
        for predictor in (
            "overall_absence_round_fraction",
            "overall_max_absence_streak",
            "rho",
        ):
            for outcome in (
                "sca_accuracy_late_minus_early",
                "aggregation_gain_late_minus_early",
            ):
                x = [float(row[predictor]) for row in rows]
                y = [float(row[outcome]) for row in rows]
                correlation = _spearman(x, y)
                decay_association_rows.append(
                    {
                        "topology": topology,
                        "class_group": "tail",
                        "predictor": predictor,
                        "outcome": outcome,
                        "n_classes": len(rows),
                        "spearman_rho": correlation,
                        "two_sided_permutation_p_exploratory": _permutation_p(
                            x,
                            y,
                            correlation,
                            samples=args.permutation_samples,
                            seed=args.permutation_seed
                            + 700001
                            + len(decay_association_rows) * 1009,
                        ),
                        "sign_interpretation": (
                            "a negative value means greater absence is associated "
                            "with stronger late-stage decline"
                        ),
                    }
                )

        ordered = sorted(
            rows,
            key=lambda row: (
                float(row["overall_max_absence_streak"]),
                int(row["class_id"]),
            ),
        )
        midpoint = len(ordered) // 2
        for group_name, selected in (
            ("shorter_absence_half", ordered[:midpoint]),
            ("longer_absence_half", ordered[midpoint:]),
        ):
            decay_association_rows.append(
                {
                    "topology": topology,
                    "class_group": "tail",
                    "predictor": "max_absence_streak_median_split",
                    "outcome": "aggregation_gain_late_minus_early",
                    "absence_group": group_name,
                    "n_classes": len(selected),
                    "mean_max_absence_streak": _mean(
                        selected, "overall_max_absence_streak"
                    ),
                    "mean_outcome": _mean(
                        selected, "aggregation_gain_late_minus_early"
                    ),
                    "sign_interpretation": (
                        "compare longer versus shorter halves; a more negative mean "
                        "indicates absence-localized gain decay"
                    ),
                }
            )

    evidence = {}
    for topology in TOPOLOGY_RUNS:
        stage_rho_rows = [
            row
            for row in spearman_rows
            if row["topology"] == topology
            and row["class_group"] == "tail"
            and row["predictor"] == "rho"
        ]
        late_rho = next(
            row
            for row in spearman_rows
            if row["topology"] == topology
            and row["stage"] == "late"
            and row["class_group"] == "tail"
            and row["predictor"] == "rho"
        )
        q1 = next(
            row
            for row in quartile_rows
            if row["topology"] == topology
            and row["stage"] == "late"
            and row["rho_quartile"] == "Q1"
        )
        q4 = next(
            row
            for row in quartile_rows
            if row["topology"] == topology
            and row["stage"] == "late"
            and row["rho_quartile"] == "Q4"
        )
        evidence[topology] = {
            "tail_rho_gain_spearman_by_stage": {
                row["stage"]: row["spearman_rho"] for row in stage_rho_rows
            },
            "late_rho_gain_spearman": late_rho["spearman_rho"],
            "late_q1_gain": q1["mean_sca_minus_residual_fedavg"],
            "late_q4_gain": q4["mean_sca_minus_residual_fedavg"],
            "late_q4_minus_q1_gain": (
                q4["mean_sca_minus_residual_fedavg"]
                - q1["mean_sca_minus_residual_fedavg"]
            ),
        }
    paired_late = next(
        row
        for row in spearman_rows
        if row["topology"] == "clientlt_minus_matched_dirichlet"
        and row["stage"] == "late"
        and row["class_group"] == "tail"
        and row["predictor"] == "clientlt_minus_matched_rho"
    )
    absence_decay = next(
        row
        for row in decay_association_rows
        if row["topology"] == "clientlt"
        and row["predictor"] == "overall_max_absence_streak"
        and row["outcome"] == "aggregation_gain_late_minus_early"
    )
    positive_stage_count = sum(
        math.isfinite(float(value)) and float(value) > 0
        for value in evidence["clientlt"]["tail_rho_gain_spearman_by_stage"].values()
    )
    clientlt_directional = (
        positive_stage_count >= 2
        and float(evidence["clientlt"]["late_q4_minus_q1_gain"]) > 0
    )

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "stage1a_class_round.csv", class_round_rows)
    _write_csv(output_dir / "stage1a_class_stage.csv", class_stage_rows)
    _write_csv(output_dir / "stage1a_paired_class_stage.csv", paired_class_stage_rows)
    _write_csv(output_dir / "stage1a_spearman.csv", spearman_rows)
    _write_csv(output_dir / "stage1a_rho_quartiles.csv", quartile_rows)
    _write_csv(output_dir / "stage1a_gain_contribution.csv", contribution_rows)
    _write_csv(output_dir / "stage1a_absence_analysis.csv", absence_rows)
    _write_csv(
        output_dir / "stage1a_decay_association.csv", decay_association_rows
    )

    payload = {
        "schema_version": "stage1a_topology_gate_v1",
        "status": (
            "DESCRIPTIVE_TOPOLOGY_GATE_SUPPORT"
            if clientlt_directional
            else "NO_DIRECTIONAL_TOPOLOGY_GATE_SUPPORT"
        ),
        "primary_scope": "bottom-tail classes activated by the residual head",
        "tail_class_ids": tail_ids,
        "prespecified_stages": [
            {"name": name, "first_round": first, "last_round": last}
            for name, first, last in STAGES
        ],
        "fixed_margin_null": {
            "definition": "uniform random label-to-client coupling conditional on exact n_k and n_c",
            "samples": args.null_samples,
            "seed": args.null_seed,
        },
        "artifact_audit": {
            "within_topology_partitions_equal": True,
            "class_margins_equal": True,
            "client_margins_equal": True,
            "all_four_actual_client_schedules_equal": True,
            "round_count": len(schedules["clientlt"]),
            "class_count": int(matrices["clientlt"].shape[1]),
        },
        "topology_gate_evidence": evidence,
        "paired_topology_evidence": {
            "late_spearman_delta_rho_vs_gain_did": paired_late["spearman_rho"],
            "interpretation": (
                "positive means classes made more concentrated by Client-LT also "
                "receive a larger SCA-vs-ResidualFedAvg gain amplification"
            ),
        },
        "late_decay_evidence": {
            "clientlt_spearman_max_absence_streak_vs_gain_decay": absence_decay[
                "spearman_rho"
            ],
            "interpretation": (
                "negative means the late SCA-gain decay is concentrated in classes "
                "with longer supporter-absence streaks"
            ),
        },
        "update_agreement": {
            "available": False,
            "reason": "client-level residual update vectors were not saved in the four existing runs",
            "proxy_used": False,
            "next_run_requirement": "log each supporter residual delta before server aggregation",
        },
        "inference_warning": (
            "One discovery seed; communication rounds and classes are not independent "
            "seed replicates. Permutation p-values are exploratory only."
        ),
        "outputs": {
            "class_round": "stage1a_class_round.csv",
            "class_stage": "stage1a_class_stage.csv",
            "paired_class_stage": "stage1a_paired_class_stage.csv",
            "spearman": "stage1a_spearman.csv",
            "rho_quartiles": "stage1a_rho_quartiles.csv",
            "gain_contribution": "stage1a_gain_contribution.csv",
            "absence": "stage1a_absence_analysis.csv",
            "decay_association": "stage1a_decay_association.csv",
        },
    }
    safe_payload = _json_safe(payload)
    (output_dir / "stage1a_summary.json").write_text(
        json.dumps(safe_payload, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    markdown = [
        "# Stage-1A topology-gate diagnostic",
        "",
        f"Verdict: `{safe_payload['status']}`",
        "",
        "The analysis uses a random fixed-margin null and the actually logged "
        "selected clients. No validation/test quantity controls training.",
        "",
        "## Late-stage directional evidence",
        "",
        "| topology | Spearman(rho, SCA-RF) | Q1 gain | Q4 gain | Q4-Q1 |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for topology in TOPOLOGY_RUNS:
        item = evidence[topology]
        markdown.append(
            f"| {topology} | {_fmt(item['late_rho_gain_spearman'])} | "
            f"{_fmt(item['late_q1_gain'])} | {_fmt(item['late_q4_gain'])} | "
            f"{_fmt(item['late_q4_minus_q1_gain'])} |"
        )
    markdown.extend(
        [
            "",
            "## Cross-topology and decay checks",
            "",
            "Paired Spearman(delta rho, gain DiD): "
            f"`{_fmt(paired_late['spearman_rho'])}`. "
            "Spearman(max absence streak, late-minus-early gain): "
            f"`{_fmt(absence_decay['spearman_rho'])}`.",
            "",
            "## Agreement limitation",
            "",
            "Client-level residual update vectors are unavailable. Agreement is "
            "therefore not estimated, and aggregated row deltas are not used as a proxy.",
            "",
            "This is descriptive seed-42 evidence, not seed-level inference.",
        ]
    )
    (output_dir / "stage1a_report.md").write_text(
        "\n".join(markdown) + "\n", encoding="utf-8"
    )
    print(json.dumps(safe_payload, indent=2, sort_keys=True, allow_nan=False))
    return safe_payload


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-root", type=Path, default=Path("output/online_sca_seed42_v2")
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--tail-class-ratio", type=float, default=0.2)
    parser.add_argument("--null-samples", type=int, default=500)
    parser.add_argument("--null-seed", type=int, default=42001)
    parser.add_argument("--permutation-samples", type=int, default=10000)
    parser.add_argument("--permutation-seed", type=int, default=42002)
    parser.add_argument("--expected-rounds", type=int, default=80)
    args = parser.parse_args()
    args.output_dir = args.output_dir or args.output_root / "stage1a_topology_gate"
    if not 0 < args.tail_class_ratio <= 1:
        parser.error("--tail-class-ratio must be in (0, 1]")
    if args.null_samples < 1:
        parser.error("--null-samples must be positive")
    if args.permutation_samples < 1:
        parser.error("--permutation-samples must be positive")
    return args


if __name__ == "__main__":
    analyze(parse_args())
