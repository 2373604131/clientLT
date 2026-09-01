"""GPU analysis for Phase-2 spatial and temporal client-level breadth."""

from __future__ import annotations

import csv
import json
import math
import traceback
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from tools.breadth_audit.inputs import load_preregistered_neighbors
from tools.client_update_audit.runtime import _prepare_model, _repository_cifar_transform
from tools.functional_breadth_feasibility.runtime import TrainOnlyCifarRawStore
from tools.semantic_acquisition.common import file_sha256, tensor_mapping_hash, write_csv, write_json
from tools.semantic_acquisition.runtime import _predict_ids, load_lora_state
from tools.topology_breadth_audit.metrics import breadth_metrics, potential_pool_metrics
from tools.topology_breadth_audit.protocol import frozen_protocol
from tools.topology_breadth_audit.runtime import _state_path


TAIL_CLASSES = list(range(80, 100))
TOPOLOGIES = ("clientlt", "matched_dirichlet")
POOLS = ("evidence_supporters", "all_clients")


def _read_csv(path: Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _truth(value) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def _verify(args):
    manifest_path = Path(args.manifest_dir) / "manifest_contract.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("protocol") != frozen_protocol():
        raise RuntimeError("Phase-2 analysis manifest protocol mismatch")
    if not manifest.get("class_margins_equal") or not manifest.get("client_margins_equal"):
        raise RuntimeError("Phase-2 strict fixed-margin design was not satisfied")
    runtimes = {}
    for topology, directory in (
        ("clientlt", Path(args.clientlt_dir)),
        ("matched_dirichlet", Path(args.matched_dir)),
    ):
        path = directory / "runtime_contract.json"
        if not path.is_file():
            raise FileNotFoundError(path)
        value = json.loads(path.read_text(encoding="utf-8"))
        if (
            value.get("topology") != topology
            or value.get("protocol") != frozen_protocol()
            or int(value.get("completed_clients", -1)) != 30
            or value.get("server_aggregation_called") is not False
            or value.get("test_split_accessed") is not False
            or value.get("manifest_contract_sha256") != file_sha256(manifest_path)
        ):
            raise RuntimeError(f"Invalid Phase-2 client runtime: {path}")
        for name, expected in value.get("result_hashes", {}).items():
            if file_sha256(directory / name) != expected:
                raise RuntimeError(f"Phase-2 runtime hash mismatch: {directory / name}")
        fairness = _read_csv(directory / "runtime_fairness.csv")
        if len(fairness) != 30 or any(not _truth(row["pass"]) for row in fairness):
            raise RuntimeError(f"Phase-2 runtime fairness failed: {directory}")
        for client_id in range(30):
            state_path = _state_path(directory, topology, client_id)
            if not state_path.is_file():
                raise FileNotFoundError(state_path)
            expected = value.get("state_hashes", {}).get(str(client_id))
            if expected and file_sha256(state_path) != expected:
                raise RuntimeError(f"Client state hash mismatch: {state_path}")
        runtimes[topology] = value
    if runtimes["clientlt"]["theta0_hash"] != runtimes["matched_dirichlet"]["theta0_hash"]:
        raise RuntimeError("The two Phase-2 topologies used different theta0 states")
    return manifest, runtimes


def _matrix(counts: pd.DataFrame, topology: str) -> np.ndarray:
    selected = counts[counts.topology == topology]
    matrix = np.zeros((30, 100), dtype=np.int64)
    for row in selected.itertuples(index=False):
        matrix[int(row.client_id), int(row.class_id)] = int(row.sample_count)
    return matrix


def _schedule(rows: pd.DataFrame) -> dict[int, tuple[int, ...]]:
    result = {
        int(epoch): tuple(sorted(group.client_id.astype(int).tolist()))
        for epoch, group in rows.groupby("epoch_index", sort=True)
    }
    if sorted(result) != list(range(80)) or any(len(value) != 12 for value in result.values()):
        raise RuntimeError("Phase-2 schedule is not the frozen 80x12 design")
    return result


def _merge(theta0, states: dict[int, dict], clients: list[int], weights: np.ndarray):
    result = {name: tensor.clone() for name, tensor in theta0.items()}
    if not clients:
        return result
    weights = np.asarray(weights, dtype=np.float64)
    if len(weights) != len(clients) or np.any(weights < 0) or float(weights.sum()) <= 0:
        raise ValueError("Invalid merge clients/weights")
    weights = weights / weights.sum()
    for client_id, weight in zip(clients, weights.tolist()):
        for name in result:
            result[name] += float(weight) * (states[client_id][name] - theta0[name])
    return result


def _actual_gain_vector(
    model, state, store, transform, probes: pd.DataFrame,
    baseline_by_tail: dict[int, torch.Tensor], neighbors: dict[int, list[int]], tail_class: int,
) -> list[dict]:
    rows = probes[probes.tail_class == tail_class].sort_values("slot")
    load_lora_state(model, state)
    logits, _ = _predict_ids(
        model, store, transform, rows.base_sample_id.tolist(), rows.label.astype(int).tolist(),
        batch_size=64,
    )
    before = baseline_by_tail[tail_class]
    output = []
    for rank, neighbor in enumerate(neighbors[tail_class], start=1):
        old = before[:, tail_class] - before[:, neighbor]
        new = logits[:, tail_class] - logits[:, neighbor]
        output.append({
            "boundary_neighbor_class": neighbor, "semantic_neighbor_rank": rank,
            "actual_boundary_gain": float((new - old).mean().item()),
        })
    return output


def _stage(epoch: int) -> str:
    round_id = epoch + 1
    if round_id <= 20:
        return "early"
    if round_id <= 50:
        return "middle"
    return "late"


def _available_and_weights(
    pool: str, available: tuple[int, ...], matrix: np.ndarray, tail_class: int
) -> tuple[list[int], np.ndarray]:
    clients = list(available)
    if pool == "evidence_supporters":
        clients = [client for client in clients if matrix[client, tail_class] > 0]
        weights = np.asarray([matrix[client, tail_class] for client in clients], dtype=np.float64)
    elif pool == "all_clients":
        weights = np.asarray([matrix[client].sum() for client in clients], dtype=np.float64)
    else:
        raise ValueError(pool)
    return clients, weights


def _summary_contrast(left: list[float], right: list[float]) -> dict:
    values = np.asarray(left, dtype=np.float64) - np.asarray(right, dtype=np.float64)
    return {
        "mean": float(values.mean()), "median": float(np.median(values)),
        "positive_tail_classes": int((values > 0).sum()),
        "negative_tail_classes": int((values < 0).sum()), "tail_class_count": len(values),
    }


def run(args) -> dict:
    manifest, runtime_contracts = _verify(args)
    args.output_dir = Path(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cache_contract_path = args.output_dir / "analysis_cache_contract.json"
    cache_contract = {
        "protocol_hash": frozen_protocol()["protocol_hash"],
        "manifest_contract_sha256": file_sha256(Path(args.manifest_dir) / "manifest_contract.json"),
        "clientlt_runtime_contract_sha256": file_sha256(Path(args.clientlt_dir) / "runtime_contract.json"),
        "matched_runtime_contract_sha256": file_sha256(Path(args.matched_dir) / "runtime_contract.json"),
    }
    if cache_contract_path.is_file():
        if json.loads(cache_contract_path.read_text(encoding="utf-8")) != cache_contract:
            raise RuntimeError("Existing Phase-2 analysis cache belongs to different inputs")
    else:
        write_json(cache_contract_path, cache_contract)
    counts = pd.read_csv(Path(args.manifest_dir) / "client_class_counts_long.csv")
    probes = pd.read_csv(Path(args.manifest_dir) / "heldout_tail_train_probes.csv")
    schedule = _schedule(pd.read_csv(Path(args.manifest_dir) / "frac0p4_schedule.csv"))
    matrices = {topology: _matrix(counts, topology) for topology in TOPOLOGIES}
    if not np.array_equal(matrices["clientlt"].sum(0), matrices["matched_dirichlet"].sum(0)):
        raise RuntimeError("Analysis detected unequal class margins")
    if not np.array_equal(matrices["clientlt"].sum(1), matrices["matched_dirichlet"].sum(1)):
        raise RuntimeError("Analysis detected unequal client margins")

    gain_lookup = {}
    for topology, directory in (
        ("clientlt", Path(args.clientlt_dir)),
        ("matched_dirichlet", Path(args.matched_dir)),
    ):
        frame = pd.read_csv(directory / "client_boundary_gains.csv")
        for (client, tail), group in frame.groupby(["client_id", "tail_class"], sort=True):
            ordered = group.sort_values("semantic_neighbor_rank")
            if ordered.semantic_neighbor_rank.astype(int).tolist() != list(range(1, 11)):
                raise RuntimeError(f"Incomplete client boundary vector: {topology}/{client}/{tail}")
            gain_lookup[(topology, int(client), int(tail))] = ordered.boundary_gain.to_numpy(np.float64)

    store = TrainOnlyCifarRawStore(args.data_dir)
    if not Path(args.theta0_file).is_file():
        raise FileNotFoundError(args.theta0_file)
    _, model, theta0, _ = _prepare_model(args, store)
    theta_hash = tensor_mapping_hash(theta0)
    if theta_hash != runtime_contracts["clientlt"]["theta0_hash"]:
        raise RuntimeError("Analysis theta0 differs from the local-update runtime")
    transform = _repository_cifar_transform()
    neighbors, _ = load_preregistered_neighbors(TAIL_CLASSES)
    load_lora_state(model, theta0)
    baseline_by_tail = {}
    for tail_class in TAIL_CLASSES:
        rows = probes[probes.tail_class == tail_class].sort_values("slot")
        logits, _ = _predict_ids(
            model, store, transform, rows.base_sample_id.tolist(), rows.label.astype(int).tolist(),
            batch_size=args.eval_batch_size,
        )
        baseline_by_tail[tail_class] = logits
    states = {}
    for topology, directory in (
        ("clientlt", Path(args.clientlt_dir)),
        ("matched_dirichlet", Path(args.matched_dir)),
    ):
        states[topology] = {
            client: torch.load(_state_path(directory, topology, client), map_location="cpu")["lora_state"]
            for client in range(30)
        }

    actual_path = args.output_dir / "actual_merged_boundary_gains.csv"
    existing = _read_csv(actual_path) if actual_path.is_file() else []
    actual_map: dict[tuple[str, str, str, int, int], list[dict]] = defaultdict(list)
    for row in existing:
        key = (
            row["phase"], row["pool"], row["topology"],
            int(row["epoch_index"]), int(row["tail_class"]),
        )
        actual_map[key].append(row)
    all_clients = tuple(range(30))
    specs = []
    for topology in TOPOLOGIES:
        for pool in POOLS:
            for tail_class in TAIL_CLASSES:
                specs.append(("A1", pool, topology, -1, tail_class, all_clients))
            for epoch in range(80):
                for tail_class in TAIL_CLASSES:
                    specs.append(("A2", pool, topology, epoch, tail_class, schedule[epoch]))
    completed_since_write = 0
    for index, (phase, pool, topology, epoch, tail_class, available) in enumerate(specs, start=1):
        key = (phase, pool, topology, epoch, tail_class)
        if len(actual_map.get(key, [])) == 10:
            continue
        clients, weights = _available_and_weights(pool, available, matrices[topology], tail_class)
        state = _merge(theta0, states[topology], clients, weights) if clients else theta0
        rows = _actual_gain_vector(
            model, state, store, transform, probes, baseline_by_tail, neighbors, tail_class
        )
        actual_map[key] = [{
            "phase": phase, "pool": pool, "topology": topology,
            "epoch_index": epoch, "communication_round": epoch + 1 if epoch >= 0 else 0,
            "stage": _stage(epoch) if epoch >= 0 else "all_clients",
            "tail_class": tail_class, "available_client_count": len(available),
            "merged_client_count": len(clients),
            "merged_client_ids": ",".join(str(value) for value in clients),
            **row,
        } for row in rows]
        completed_since_write += 1
        if completed_since_write >= 25:
            write_csv(actual_path, [
                row for map_key in sorted(actual_map) for row in sorted(
                    actual_map[map_key], key=lambda value: int(value["semantic_neighbor_rank"])
                )
            ])
            completed_since_write = 0
        print(json.dumps({
            "stage": "phase2-actual-merge", "phase": phase, "pool": pool,
            "topology": topology, "epoch": epoch, "tail_class": tail_class,
            "index": index, "total": len(specs),
        }))
    actual_rows = [
        row for key in sorted(actual_map) for row in sorted(
            actual_map[key], key=lambda value: int(value["semantic_neighbor_rank"])
        )
    ]
    write_csv(actual_path, actual_rows)

    metric_rows, streaks = [], defaultdict(int)
    a1_actual_breadth = {}
    for phase, pool, topology, epoch, tail_class, available in specs:
        clients, weights = _available_and_weights(pool, available, matrices[topology], tail_class)
        gain_matrix = np.stack([
            gain_lookup[(topology, client, tail_class)] for client in clients
        ]) if clients else np.zeros((0, 10), dtype=np.float64)
        potential = potential_pool_metrics(gain_matrix, weights)
        key = (phase, pool, topology, epoch, tail_class)
        actual_vector = np.asarray([
            float(row["actual_boundary_gain"])
            for row in sorted(actual_map[key], key=lambda row: int(row["semantic_neighbor_rank"]))
        ])
        actual = breadth_metrics(actual_vector)
        absent = pool == "evidence_supporters" and not clients
        streak_key = (pool, topology, tail_class)
        streaks[streak_key] = streaks[streak_key] + 1 if phase == "A2" and absent else 0
        row = {
            "phase": phase, "pool": pool, "topology": topology,
            "epoch_index": epoch, "communication_round": epoch + 1 if epoch >= 0 else 0,
            "stage": _stage(epoch) if epoch >= 0 else "all_clients",
            "tail_class": tail_class, "available_client_count": len(available),
            "merged_client_count": len(clients), "supporter_class_mass": int(
                matrices[topology][clients, tail_class].sum() if clients else 0
            ),
            "absent_this_round": bool(absent),
            "absence_streak": int(streaks[streak_key]) if phase == "A2" else 0,
            **potential,
            **{f"actual_{name}": value for name, value in actual.items()},
        }
        if phase == "A1":
            a1_actual_breadth[(pool, topology, tail_class)] = float(actual["effective_breadth"])
            row["low_breadth_vs_a1"] = False
        else:
            threshold = 0.5 * a1_actual_breadth[(pool, topology, tail_class)]
            row["low_breadth_vs_a1"] = bool(
                absent or (
                    a1_actual_breadth[(pool, topology, tail_class)] > 0
                    and float(actual["effective_breadth"]) < threshold
                )
            )
        metric_rows.append(row)
    write_csv(args.output_dir / "pool_round_metrics.csv", metric_rows)

    a1_rows = [row for row in metric_rows if row["phase"] == "A1"]
    a2_rows = [row for row in metric_rows if row["phase"] == "A2"]
    write_csv(args.output_dir / "a1_spatial_summary.csv", a1_rows)
    temporal_rows = []
    for (pool, topology, tail_class), values in sorted(
        _group(a2_rows, ("pool", "topology", "tail_class")).items()
    ):
        ordered = sorted(values, key=lambda row: int(row["epoch_index"]))
        breadth = np.asarray([float(row["actual_effective_breadth"]) for row in ordered])
        mean = float(breadth.mean())
        temporal_rows.append({
            "pool": pool, "topology": topology, "tail_class": int(tail_class),
            "breadth_auc": mean,
            "breadth_auc_normalized_by_max10": mean / 10.0,
            "breadth_cv": float(breadth.std() / mean) if mean > 0 else 0.0,
            "low_breadth_round_fraction": float(np.mean([bool(row["low_breadth_vs_a1"]) for row in ordered])),
            "no_support_round_fraction": float(np.mean([bool(row["absent_this_round"]) for row in ordered])),
            "maximum_absence_streak": max(int(row["absence_streak"]) for row in ordered),
            "early_breadth": float(np.mean([float(row["actual_effective_breadth"]) for row in ordered if row["stage"] == "early"])),
            "middle_breadth": float(np.mean([float(row["actual_effective_breadth"]) for row in ordered if row["stage"] == "middle"])),
            "late_breadth": float(np.mean([float(row["actual_effective_breadth"]) for row in ordered if row["stage"] == "late"])),
        })
    write_csv(args.output_dir / "a2_temporal_summary.csv", temporal_rows)

    a1_index = {(row["pool"], row["topology"], int(row["tail_class"])): row for row in a1_rows}
    temporal_index = {(row["pool"], row["topology"], int(row["tail_class"])): row for row in temporal_rows}
    contrast_rows = []
    for pool in POOLS:
        for tail_class in TAIL_CLASSES:
            matched_a1 = a1_index[(pool, "matched_dirichlet", tail_class)]
            clientlt_a1 = a1_index[(pool, "clientlt", tail_class)]
            matched_a2 = temporal_index[(pool, "matched_dirichlet", tail_class)]
            clientlt_a2 = temporal_index[(pool, "clientlt", tail_class)]
            contrast_rows.append({
                "pool": pool, "tail_class": tail_class,
                "matched_minus_clientlt_a1_potential_effective_breadth": float(matched_a1["potential_effective_breadth"]) - float(clientlt_a1["potential_effective_breadth"]),
                "matched_minus_clientlt_a1_actual_effective_breadth": float(matched_a1["actual_effective_breadth"]) - float(clientlt_a1["actual_effective_breadth"]),
                "matched_minus_clientlt_a1_actual_worst_boundary_gain": float(matched_a1["actual_worst_boundary_gain"]) - float(clientlt_a1["actual_worst_boundary_gain"]),
                "matched_minus_clientlt_a2_breadth_auc": float(matched_a2["breadth_auc"]) - float(clientlt_a2["breadth_auc"]),
                "clientlt_minus_matched_a2_low_breadth_fraction": float(clientlt_a2["low_breadth_round_fraction"]) - float(matched_a2["low_breadth_round_fraction"]),
                "clientlt_minus_matched_a2_no_support_fraction": float(clientlt_a2["no_support_round_fraction"]) - float(matched_a2["no_support_round_fraction"]),
            })
    write_csv(args.output_dir / "paired_topology_contrasts.csv", contrast_rows)

    primary = [row for row in contrast_rows if row["pool"] == "evidence_supporters"]
    contrast_summary = {}
    for name in (
        "matched_minus_clientlt_a1_potential_effective_breadth",
        "matched_minus_clientlt_a1_actual_effective_breadth",
        "matched_minus_clientlt_a1_actual_worst_boundary_gain",
        "matched_minus_clientlt_a2_breadth_auc",
        "clientlt_minus_matched_a2_low_breadth_fraction",
        "clientlt_minus_matched_a2_no_support_fraction",
    ):
        values = [float(row[name]) for row in primary]
        contrast_summary[name] = _summary_contrast(values, [0.0] * len(values))
    spatial = (
        contrast_summary["matched_minus_clientlt_a1_potential_effective_breadth"]["positive_tail_classes"] >= 12
        and contrast_summary["matched_minus_clientlt_a1_actual_effective_breadth"]["positive_tail_classes"] >= 12
    )
    temporal = (
        contrast_summary["matched_minus_clientlt_a2_breadth_auc"]["positive_tail_classes"] >= 12
        and contrast_summary["clientlt_minus_matched_a2_low_breadth_fraction"]["positive_tail_classes"] >= 12
    )
    verdict = (
        "BOTH" if spatial and temporal else "SPATIAL_ONLY" if spatial
        else "TEMPORAL_ONLY" if temporal else "NO_CONSISTENT_GAP"
    )
    result_names = (
        "analysis_cache_contract.json",
        "actual_merged_boundary_gains.csv", "pool_round_metrics.csv",
        "a1_spatial_summary.csv",
        "a2_temporal_summary.csv", "paired_topology_contrasts.csv",
    )
    summary = {
        "schema_version": "topology_breadth_analysis_v1", "verdict": verdict,
        "spatial_breadth_supported": spatial, "temporal_breadth_supported": temporal,
        "primary_pool": "evidence_supporters", "primary_contrasts": contrast_summary,
        "fixed_class_margins": True, "fixed_client_margins": True,
        "frac1_interpretation": "all-client availability; no participation sampling",
        "frac0p4_interpretation": "actual common 80-round schedule; 12 of 30 clients per round",
        "single_seed_descriptive": True, "test_split_accessed": False,
        "server_deployable_method": False,
        "result_hashes": {name: file_sha256(args.output_dir / name) for name in result_names},
        "interpretation": (
            "BOTH supports the topology-to-functional-breadth arrow under this frozen seed/substrate. "
            "It does not show that breadth improves accuracy; that remains Phase 3."
        ),
    }
    write_json(args.output_dir / "phase2_summary.json", summary)
    lines = [
        "# Phase 2 — client topology to Functional Breadth", "",
        f"- Verdict: **{verdict}**",
        f"- A1 spatial breadth supported: **{spatial}**",
        f"- A2 temporal accessible breadth supported: **{temporal}**",
        "- Fixed client/class margins: **yes / yes**",
        "- Test split accessed: **no**", "",
        "This result tests the topology→breadth arrow only. It is not yet a method or a breadth→accuracy result.",
    ]
    (args.output_dir / "phase2_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def _group(rows: list[dict], keys: tuple[str, ...]):
    output = defaultdict(list)
    for row in rows:
        output[tuple(row[key] for key in keys)].append(row)
    return output


def guarded_run(args) -> dict:
    failure = Path(args.output_dir) / "failure.json"
    if failure.is_file():
        failure.unlink()
    try:
        return run(args)
    except Exception as exc:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        write_json(failure, {
            "stage": "PHASE2_ANALYSIS", "error_type": type(exc).__name__,
            "error": str(exc), "traceback": traceback.format_exc(),
        })
        raise
