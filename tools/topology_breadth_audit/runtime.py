"""Train one common-anchor local update per real client and measure private boundaries."""

from __future__ import annotations

import json
import platform
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from tools.breadth_audit.inputs import load_preregistered_neighbors
from tools.client_update_audit.runtime import _prepare_model, _repository_cifar_transform
from tools.functional_breadth_feasibility.runtime import TrainOnlyCifarRawStore
from tools.semantic_acquisition.common import (
    file_sha256,
    stable_hash,
    tensor_mapping_hash,
    write_csv,
    write_json,
)
from tools.semantic_acquisition.runtime import (
    _predict_ids,
    _train_client,
    flatten_named,
    load_lora_state,
)
from tools.topology_breadth_audit.protocol import frozen_protocol


TAIL_CLASSES = list(range(80, 100))


def _verify_manifests(manifest_dir: Path) -> tuple[dict, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    manifest_dir = Path(manifest_dir)
    path = manifest_dir / "manifest_contract.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    contract = json.loads(path.read_text(encoding="utf-8"))
    if contract.get("protocol") != frozen_protocol():
        raise RuntimeError("Phase-2 manifest protocol mismatch")
    for name, expected in contract.get("manifest_hashes", {}).items():
        if file_sha256(manifest_dir / name) != expected:
            raise RuntimeError(f"Phase-2 manifest hash mismatch: {name}")
    samples = pd.read_csv(manifest_dir / "partition_samples.csv")
    execution = pd.read_csv(manifest_dir / "local_execution.csv")
    probes = pd.read_csv(manifest_dir / "heldout_tail_train_probes.csv")
    return contract, samples, execution, probes


def _state_path(output_dir: Path, topology: str, client_id: int) -> Path:
    return Path(output_dir) / "client_states" / f"{topology}_client_{int(client_id):02d}.pt"


def _boundary_rows(
    before: torch.Tensor, after: torch.Tensor, probes: pd.DataFrame,
    neighbors: dict[int, list[int]], topology: str, client_id: int,
) -> list[dict]:
    rows, offset = [], 0
    for tail_class in TAIL_CLASSES:
        count = int((probes.tail_class == tail_class).sum())
        baseline, updated = before[offset:offset + count], after[offset:offset + count]
        for rank, neighbor in enumerate(neighbors[tail_class], start=1):
            old = baseline[:, tail_class] - baseline[:, neighbor]
            new = updated[:, tail_class] - updated[:, neighbor]
            rows.append({
                "topology": topology, "client_id": client_id, "tail_class": tail_class,
                "boundary_neighbor_class": neighbor, "semantic_neighbor_rank": rank,
                "probe_sample_count": count,
                "baseline_boundary_margin": float(old.mean().item()),
                "updated_boundary_margin": float(new.mean().item()),
                "boundary_gain": float((new - old).mean().item()),
                "probe_split": "heldout_train",
            })
        offset += count
    return rows


def run(args) -> dict:
    if args.topology not in {"clientlt", "matched_dirichlet"}:
        raise ValueError(f"Unknown topology: {args.topology}")
    contract, samples, execution, probes = _verify_manifests(args.manifest_dir)
    samples = samples[samples.topology == args.topology].copy()
    execution = execution[execution.topology == args.topology].copy()
    if set(samples.client_id.astype(int)) != set(range(30)):
        raise RuntimeError(f"{args.topology} does not contain all 30 real clients")
    args.output_dir = Path(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if not Path(args.theta0_file).is_file():
        raise FileNotFoundError("Phase 2 requires the pre-existing common theta0")
    store = TrainOnlyCifarRawStore(args.data_dir)
    cfg, model, theta0, names = _prepare_model(args, store)
    transform = _repository_cifar_transform()
    theta_hash = tensor_mapping_hash(theta0)
    theta_vector = flatten_named(theta0, names)
    manifest_hash = file_sha256(Path(args.manifest_dir) / "manifest_contract.json")
    fairness_rows, state_hashes, update_arrays = [], {}, {}
    for client_id in range(30):
        state_path = _state_path(args.output_dir, args.topology, client_id)
        unit_execution = execution[execution.client_id == client_id].copy()
        client_samples = samples[samples.client_id == client_id]
        if state_path.is_file():
            saved = torch.load(state_path, map_location="cpu")
            if (
                saved.get("topology") != args.topology
                or int(saved.get("client_id", -1)) != client_id
                or saved.get("theta0_hash") != theta_hash
                or saved.get("manifest_contract_sha256") != manifest_hash
            ):
                raise RuntimeError(f"Stale Phase-2 client state: {state_path}")
            state, train_meta, status = saved["lora_state"], saved["train_meta"], "resumed"
        else:
            states, train_meta = _train_client(
                model, cfg, theta0, unit_execution, store, transform
            )
            state, status = states[3], "trained"
            state_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                "topology": args.topology, "client_id": client_id,
                "theta0_hash": theta_hash, "manifest_contract_sha256": manifest_hash,
                "lora_state": state, "train_meta": train_meta,
            }, state_path)
        vector = flatten_named(state, names) - theta_vector
        update_arrays[f"client_{client_id:02d}"] = vector.numpy()
        state_hashes[str(client_id)] = file_sha256(state_path)
        expected_steps = int(unit_execution.groupby(["epoch", "batch_index"]).ngroups)
        fairness_rows.append({
            "topology": args.topology, "client_id": client_id,
            "client_sample_count": int(len(client_samples)),
            "expected_optimizer_steps": expected_steps,
            "optimizer_steps_successful": int(train_meta["optimizer_steps_successful"]),
            "scheduler_steps": int(train_meta["scheduler_steps"]),
            "amp_overflow_count": int(train_meta["amp_overflow_count"]),
            "update_l2": float(vector.norm().item()),
            "common_theta0_hash": theta_hash, "server_aggregation_called": False,
            "pass": bool(
                expected_steps == int(train_meta["optimizer_steps_successful"])
                and int(train_meta["scheduler_steps"]) == 3
                and int(train_meta["amp_overflow_count"]) == 0
            ),
        })
        print(json.dumps({
            "stage": "phase2-local-update", "topology": args.topology,
            "client_id": client_id, "status": status,
        }))
    write_csv(args.output_dir / "runtime_fairness.csv", fairness_rows)
    np.savez_compressed(args.output_dir / "client_update_vectors.npz", **update_arrays)
    if not all(bool(row["pass"]) for row in fairness_rows):
        raise RuntimeError(f"{args.topology} local-update fairness failed")

    probe_ids = probes.sort_values(["tail_class", "slot"]).base_sample_id.tolist()
    probe_labels = probes.sort_values(["tail_class", "slot"]).label.astype(int).tolist()
    probes = probes.sort_values(["tail_class", "slot"]).reset_index(drop=True)
    neighbors, neighbor_metadata = load_preregistered_neighbors(TAIL_CLASSES)
    load_lora_state(model, theta0)
    baseline_logits, _ = _predict_ids(
        model, store, transform, probe_ids, probe_labels, batch_size=args.eval_batch_size
    )
    gain_rows = []
    for client_id in range(30):
        saved = torch.load(_state_path(args.output_dir, args.topology, client_id), map_location="cpu")
        load_lora_state(model, saved["lora_state"])
        logits, _ = _predict_ids(
            model, store, transform, probe_ids, probe_labels, batch_size=args.eval_batch_size
        )
        gain_rows.extend(_boundary_rows(
            baseline_logits, logits, probes, neighbors, args.topology, client_id
        ))
        print(json.dumps({
            "stage": "phase2-client-boundaries", "topology": args.topology,
            "client_id": client_id,
        }))
    write_csv(args.output_dir / "client_boundary_gains.csv", gain_rows)
    result_names = (
        "runtime_fairness.csv", "client_update_vectors.npz", "client_boundary_gains.csv"
    )
    runtime_contract = {
        "schema_version": "topology_breadth_client_runtime_v1",
        "topology": args.topology, "protocol": frozen_protocol(),
        "manifest_contract_sha256": manifest_hash, "theta0_hash": theta_hash,
        "completed_clients": 30, "boundary_gain_rows": len(gain_rows),
        "server_aggregation_called": False, "test_split_accessed": False,
        "neighbor_metadata": neighbor_metadata, "trainable_parameter_names": names,
        "state_hashes": state_hashes, "device": torch.cuda.get_device_name(torch.cuda.current_device()),
        "torch_version": torch.__version__, "python_version": platform.python_version(),
        "result_hashes": {
            name: file_sha256(args.output_dir / name) for name in result_names
        },
    }
    write_json(args.output_dir / "runtime_contract.json", runtime_contract)
    return {
        "status": "PASS", "topology": args.topology, "completed_clients": 30,
        "boundary_gain_rows": len(gain_rows), "output_dir": str(args.output_dir.resolve()),
    }


def guarded_run(args) -> dict:
    failure = Path(args.output_dir) / "failure.json"
    if failure.is_file():
        failure.unlink()
    try:
        return run(args)
    except Exception as exc:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        write_json(failure, {
            "stage": "PHASE2_CLIENT_RUNTIME", "topology": args.topology,
            "error_type": type(exc).__name__, "error": str(exc),
            "traceback": traceback.format_exc(),
        })
        raise

