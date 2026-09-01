from __future__ import annotations

import csv
import importlib.util
import io
import json
import pickle
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np

from tools.breadth_audit.inputs import load_preregistered_neighbors
from tools.semantic_acquisition.common import (
    deterministic_choice,
    file_sha256,
    stable_hash,
    stable_seed,
    write_csv,
    write_json,
)
from tools.topology_breadth_audit.protocol import frozen_protocol, write_protocol
from utils.datasplit import partition_client_longtail, partition_fixed_marginal_dirichlet


ROOT = Path(__file__).resolve().parents[2]
TOPOLOGY_RUNS = {
    "clientlt": ("online_sca", "residual_fedavg_clientlt"),
    "matched_dirichlet": (
        "online_sca_matched_dirichlet", "residual_fedavg_matched_dirichlet"
    ),
}


def _load_train(data_dir: Path):
    with (Path(data_dir) / "train").open("rb") as handle:
        train = pickle.load(handle, encoding="latin1")
    labels = np.asarray(train["fine_labels"], dtype=np.int64)
    return labels


def _exact_lt_train_pool(data_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    raw_labels = _load_train(data_dir)
    spec = importlib.util.spec_from_file_location(
        "topology_breadth_long_tail", ROOT / "datasets" / "long_tail.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    by_class = [np.flatnonzero(raw_labels == class_id).tolist() for class_id in range(100)]
    with redirect_stdout(io.StringIO()):
        _, selected = module.train_long_tail(by_class, 100, 0.01, "exp")
    raw_ids = np.asarray(module.flatten_list(selected), dtype=np.int64)
    labels = raw_labels[raw_ids]
    if len(labels) != 10847 or int(np.isin(labels, list(range(80, 100))).sum()) != 153:
        raise RuntimeError("Unexpected CIFAR-100-LT train pool")
    return labels, raw_ids, raw_labels


def _counts(labels: np.ndarray, partitions: dict[int, np.ndarray]) -> np.ndarray:
    return np.stack([
        np.bincount(labels[np.asarray(partitions[client], dtype=np.int64)], minlength=100)
        for client in range(30)
    ]).astype(np.int64)


def _read_reference_counts(path: Path) -> np.ndarray:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    ids = [int(row["client_id"]) for row in rows]
    if ids != list(range(30)):
        raise RuntimeError(f"Reference counts have non-canonical clients: {path}")
    return np.asarray([
        [int(row[f"class_{class_id}"]) for class_id in range(100)] for row in rows
    ], dtype=np.int64)


def _read_schedule(path: Path) -> list[tuple[int, ...]]:
    grouped: dict[int, list[int]] = {}
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            grouped.setdefault(int(row["epoch_index"]), []).append(int(row["client_id"]))
    if sorted(grouped) != list(range(80)):
        raise RuntimeError(f"Schedule does not contain epochs 0--79: {path}")
    output = []
    for epoch in range(80):
        clients = tuple(sorted(grouped[epoch]))
        if len(clients) != 12 or len(set(clients)) != 12 or min(clients) < 0 or max(clients) >= 30:
            raise RuntimeError(f"Invalid frac=0.4 schedule at epoch {epoch}: {clients}")
        output.append(clients)
    return output


def _partitions(labels: np.ndarray) -> dict[str, dict[int, np.ndarray]]:
    protocol = frozen_protocol()["topologies"]
    spec = protocol["clientlt"]
    clientlt = partition_client_longtail(
        labels, 30, 100,
        head_client_ratio=spec["head_client_ratio"],
        tail_client_ratio=spec["tail_client_ratio"],
        head_class_ratio=spec["head_class_ratio"],
        tail_class_ratio=spec["tail_class_ratio"],
        specialization_lambda=spec["specialization_lambda"],
        intra_group_alpha=spec["intra_group_alpha"],
        head_leakage_scale=spec["head_leakage_scale"],
        rng=np.random.RandomState(42),
    )
    capacities = [len(clientlt[client]) for client in range(30)]
    matched = partition_fixed_marginal_dirichlet(
        labels, capacities, 100, protocol["matched_dirichlet"]["beta"],
        rng=np.random.RandomState(42 + 100003),
    )
    return {"clientlt": clientlt, "matched_dirichlet": matched}


def build(data_dir: Path, sca_output_root: Path, output_dir: Path) -> dict:
    data_dir, sca_output_root, output_dir = map(Path, (data_dir, sca_output_root, output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)
    protocol_path = write_protocol(output_dir)
    labels, raw_ids, raw_labels = _exact_lt_train_pool(data_dir)
    partitions = _partitions(labels)
    matrices = {name: _counts(labels, value) for name, value in partitions.items()}
    if not np.array_equal(matrices["clientlt"].sum(0), matrices["matched_dirichlet"].sum(0)):
        raise RuntimeError("Phase-2 topologies changed class margins n_c")
    if not np.array_equal(matrices["clientlt"].sum(1), matrices["matched_dirichlet"].sum(1)):
        raise RuntimeError("Phase-2 topologies changed client margins n_k")

    reference_hashes = {}
    schedules = []
    for topology, run_names in TOPOLOGY_RUNS.items():
        for run_name in run_names:
            count_path = sca_output_root / run_name / "client_class_counts.csv"
            schedule_path = sca_output_root / run_name / "lora_aggregation_weights.csv"
            if not count_path.is_file() or not schedule_path.is_file():
                raise FileNotFoundError(f"Missing strict SCA reference artifact under {run_name}")
            if not np.array_equal(_read_reference_counts(count_path), matrices[topology]):
                raise RuntimeError(
                    f"Rebuilt {topology} partition differs from completed SCA cell: {count_path}"
                )
            schedules.append(_read_schedule(schedule_path))
            reference_hashes[f"{run_name}/client_class_counts.csv"] = file_sha256(count_path)
            reference_hashes[f"{run_name}/lora_aggregation_weights.csv"] = file_sha256(schedule_path)
    if any(schedule != schedules[0] for schedule in schedules[1:]):
        raise RuntimeError("The four completed SCA cells did not use one common schedule")
    schedule = schedules[0]

    sample_rows, execution_rows, count_rows = [], [], []
    for topology, partition in partitions.items():
        matrix = matrices[topology]
        for client_id in range(30):
            indices = np.asarray(partition[client_id], dtype=np.int64)
            for class_id in range(100):
                count_rows.append({
                    "topology": topology, "client_id": client_id,
                    "class_id": class_id, "sample_count": int(matrix[client_id, class_id]),
                })
            for lt_index in indices.tolist():
                sample_rows.append({
                    "topology": topology, "client_id": client_id,
                    "lt_index": int(lt_index), "raw_train_index": int(raw_ids[lt_index]),
                    "base_sample_id": f"train:{int(raw_ids[lt_index])}",
                    "label": int(labels[lt_index]),
                })
            for epoch in (1, 2, 3):
                order = indices.copy()
                np.random.default_rng(stable_seed(
                    "phase2-client-order", 42, topology, client_id, epoch
                )).shuffle(order)
                for position, lt_index in enumerate(order.tolist()):
                    execution_rows.append({
                        "topology": topology, "client_id": client_id, "epoch": epoch,
                        "batch_index": position // 32, "position_in_batch": position % 32,
                        "lt_index": int(lt_index), "raw_train_index": int(raw_ids[lt_index]),
                        "base_sample_id": f"train:{int(raw_ids[lt_index])}",
                        "label": int(labels[lt_index]),
                        "augmentation_seed": stable_seed(
                            "phase2-client-augmentation", 42, topology, client_id,
                            epoch, f"train:{int(raw_ids[lt_index])}",
                        ),
                    })

    lt_raw_set = set(int(value) for value in raw_ids.tolist())
    probe_rows = []
    probe_count = int(frozen_protocol()["functional_evidence"]["samples_per_tail_class"])
    for tail_class in range(80, 100):
        pool = [
            int(value) for value in np.flatnonzero(raw_labels == tail_class).tolist()
            if int(value) not in lt_raw_set
        ]
        chosen = deterministic_choice(pool, probe_count, "phase2-heldout-tail", 42, tail_class)
        for slot, raw_id in enumerate(chosen):
            probe_rows.append({
                "tail_class": tail_class, "slot": slot, "raw_train_index": raw_id,
                "base_sample_id": f"train:{raw_id}", "label": tail_class,
                "excluded_from_federated_lt_pool": True,
            })
    neighbors, neighbor_metadata = load_preregistered_neighbors(list(range(80, 100)))
    boundary_rows = [
        {"tail_class": tail, "boundary_neighbor_class": neighbor, "semantic_neighbor_rank": rank}
        for tail in range(80, 100)
        for rank, neighbor in enumerate(neighbors[tail], start=1)
    ]
    schedule_rows = [
        {"epoch_index": epoch, "communication_round": epoch + 1, "client_id": client}
        for epoch, clients in enumerate(schedule) for client in clients
    ]
    files = {
        "partition_samples.csv": sample_rows,
        "local_execution.csv": execution_rows,
        "client_class_counts_long.csv": count_rows,
        "heldout_tail_train_probes.csv": probe_rows,
        "hard_boundaries.csv": boundary_rows,
        "frac0p4_schedule.csv": schedule_rows,
    }
    for name, rows in files.items():
        write_csv(output_dir / name, rows)
    contract = {
        "schema_version": "topology_breadth_manifests_v1",
        "protocol": frozen_protocol(), "protocol_file_sha256": file_sha256(protocol_path),
        "data_dir": str(data_dir.resolve()), "lt_sample_count": len(labels),
        "tail_probe_count": len(probe_rows), "schedule_rounds": 80,
        "class_margins_equal": True, "client_margins_equal": True,
        "clientlt_matrix_hash": stable_hash(matrices["clientlt"].tolist()),
        "matched_matrix_hash": stable_hash(matrices["matched_dirichlet"].tolist()),
        "schedule_hash": stable_hash([list(value) for value in schedule]),
        "neighbor_metadata": neighbor_metadata, "reference_hashes": reference_hashes,
        "manifest_hashes": {name: file_sha256(output_dir / name) for name in files},
        "test_split_accessed": False,
    }
    write_json(output_dir / "manifest_contract.json", contract)
    return {
        "status": "PASS", "output_dir": str(output_dir.resolve()),
        "partition_sample_rows": len(sample_rows), "execution_rows": len(execution_rows),
        "fixed_margins_verified": True, "schedule_verified_across_four_cells": True,
    }

