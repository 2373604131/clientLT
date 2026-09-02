"""Run the final Compatibility-to-Retention Bridge experiment.

The runner consumes a completed boundary-evidence experiment without changing
it.  It deterministically reconstructs the frozen c+h and c+r local updates,
then applies the exact same real class-absent Client-LT FedAvg update to both.
The sole primary endpoint is the tail-class retention ratio

    R_c = G_c_post / G_c_local,

where both gains are measured against the common pre-federation theta0.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
import torch

from tools.boundary_evidence.core import class_cluster_summary
from tools.boundary_evidence.run import (
    ROOT,
    _build_runtime,
    _eval_pair,
    _load_inputs,
    _load_theta_payload,
    _read_contract as _read_boundary_contract,
)
from tools.compatibility_retention.core import (
    CONDITIONS,
    additive_post_state,
    sample_weights,
    tail_retention_rows,
)
from tools.semantic_acquisition.common import (
    file_sha256,
    stable_seed,
    tensor_mapping_hash,
    write_csv,
    write_json,
)
from tools.semantic_acquisition.manifests import DEFAULT_DATA
from tools.semantic_acquisition.runtime import (
    _aggregate_weighted,
    _train_client,
    load_lora_state,
    update_norm,
)
from utils.datasplit import partition_client_longtail_controlled


DEFAULT_SOURCE = ROOT / "output" / "boundary_evidence"
DEFAULT_OUTPUT = ROOT / "output" / "compatibility_retention_bridge"
BRIDGE_MANIFESTS = (
    "background_client_manifest.csv",
    "background_execution_manifest.csv",
    "background_selection_manifest.csv",
    "bridge_episode_manifest.csv",
)


def _clientlt_partition(inputs, seed: int, alpha: float, purity: float):
    return partition_client_longtail_controlled(
        inputs.labels,
        30,
        len(inputs.class_names),
        head_client_ratio=0.9,
        tail_client_ratio=0.1,
        tail_class_ratio=len(inputs.tail_classes) / len(inputs.class_names),
        intra_group_alpha=float(alpha),
        tail_client_min_purity=float(purity),
        tail_class_ids=inputs.tail_classes,
        rng=np.random.RandomState(int(seed)),
    )


def _counts_matrix(inputs, partition) -> np.ndarray:
    return np.stack([
        np.bincount(
            inputs.labels[np.asarray(partition[client_id], dtype=np.int64)],
            minlength=len(inputs.class_names),
        )
        for client_id in range(30)
    ])


def _verify_source_result(source: Path, contract: Mapping) -> None:
    metrics = source / "experiment_b_metrics.csv"
    summary_path = source / "summary.json"
    if not metrics.is_file() or not summary_path.is_file():
        raise FileNotFoundError("Completed Experiment B metrics and summary are required")
    rows = pd.read_csv(metrics)
    expected = (
        len(contract["data_seeds"])
        * len(contract["tail_classes"])
        * int(contract["hard_k"])
        * len(CONDITIONS)
    )
    if len(rows) != expected:
        raise RuntimeError(f"Source Experiment B is incomplete: expected {expected}, found {len(rows)}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    interpretation = summary.get("experiment_b", {}).get("target_side_interpretation")
    if interpretation != "OPPOSITE_SIDE_PRESERVATION_WITH_TARGET_SIDE_TRADEOFF":
        raise RuntimeError(
            "This bridge is conditional on the observed target/opposite-side trade-off; "
            f"source interpretation is {interpretation!r}"
        )


def _verify_reconstructed_topology(source: Path, inputs, seed: int, counts: np.ndarray) -> None:
    source_a = pd.read_csv(source / "experiment_a_coexposure.csv")
    source_a = source_a[
        (source_a.data_seed == int(seed)) & (source_a.topology == "clientlt")
    ]
    for row in source_a.itertuples():
        carriers = counts[:, int(row.tail_class)] > 0
        joint = carriers & (counts[:, int(row.hard_class)] > 0)
        observed_carriers = int(carriers.sum())
        observed_joint = int(joint.sum())
        observed_q = float(observed_joint / observed_carriers)
        if (
            observed_carriers != int(row.carrier_count)
            or observed_joint != int(row.joint_carrier_count)
            or not np.isclose(observed_q, float(row.q), atol=1e-15, rtol=0.0)
        ):
            raise RuntimeError(
                "Bridge Client-LT reconstruction differs from frozen Experiment A; "
                "check partition alpha and purity"
            )


def _build_background_manifests(source: Path, inputs, contract, alpha: float, purity: float):
    client_rows, execution_rows, selection_rows = [], [], []
    for seed in [int(value) for value in contract["data_seeds"]]:
        partition = _clientlt_partition(inputs, seed, alpha, purity)
        counts = _counts_matrix(inputs, partition)
        _verify_reconstructed_topology(source, inputs, seed, counts)
        sample_counts = {client: int(len(partition[client])) for client in range(30)}
        for client in range(30):
            local_indices = np.asarray(partition[client], dtype=np.int64)
            raw_ids = inputs.raw_train_ids[local_indices]
            labels = inputs.labels[local_indices]
            by_raw = {int(raw): int(label) for raw, label in zip(raw_ids, labels)}
            for raw_id in sorted(by_raw):
                client_rows.append({
                    "data_seed": seed,
                    "client_id": client,
                    "base_sample_id": f"train:{raw_id}",
                    "label": by_raw[raw_id],
                    "client_sample_count": sample_counts[client],
                })
            for epoch in (1, 2, 3):
                generator = np.random.default_rng(
                    stable_seed("compatibility-background-order", seed, client, epoch)
                )
                ordered = [int(value) for value in generator.permutation(raw_ids).tolist()]
                for position, raw_id in enumerate(ordered):
                    execution_rows.append({
                        "data_seed": seed,
                        "client_id": client,
                        "epoch": epoch,
                        "batch_index": position // 32,
                        "position_in_batch": position % 32,
                        "base_sample_id": f"train:{raw_id}",
                        "label": by_raw[raw_id],
                        "augmentation_seed": stable_seed(
                            "compatibility-background-augmentation",
                            seed,
                            client,
                            epoch,
                            position,
                        ),
                    })
        for tail_class in [int(value) for value in contract["tail_classes"]]:
            selected = [client for client in range(30) if int(counts[client, tail_class]) == 0]
            weights = sample_weights(sample_counts, selected)
            for client in range(30):
                selection_rows.append({
                    "data_seed": seed,
                    "tail_class": tail_class,
                    "client_id": client,
                    "client_sample_count": sample_counts[client],
                    "client_tail_count": int(counts[client, tail_class]),
                    "class_absent_selected": client in weights,
                    "fedavg_weight": float(weights.get(client, 0.0)),
                })
    return client_rows, execution_rows, selection_rows


def prepare(args) -> dict:
    source, output = Path(args.source_dir), Path(args.output_dir)
    source_contract = _read_boundary_contract(source)
    _verify_source_result(source, source_contract)
    inputs = _load_inputs(Path(args.data_dir))
    if [int(value) for value in source_contract["tail_classes"]] != inputs.tail_classes:
        raise RuntimeError("Source tail classes differ from the reconstructed CIFAR-100-LT pool")
    client_rows, execution_rows, selection_rows = _build_background_manifests(
        source,
        inputs,
        source_contract,
        args.partition_alpha,
        args.clientlt_purity,
    )
    bridge_rows = pd.read_csv(source / "experiment_b_metrics.csv").loc[
        :, ["data_seed", "tail_class", "hard_class", "hard_rank", "control_class", "condition"]
    ].to_dict("records")
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "background_client_manifest.csv", client_rows)
    write_csv(output / "background_execution_manifest.csv", execution_rows)
    write_csv(output / "background_selection_manifest.csv", selection_rows)
    write_csv(output / "bridge_episode_manifest.csv", bridge_rows)
    theta_payload = _load_theta_payload(source / "theta0.pt")
    experiment_contract = {
        "schema_version": 1,
        "name": "Compatibility-to-Retention Bridge",
        "source_boundary_dir": str(source.resolve()),
        "source_contract_sha256": file_sha256(source / "experiment_contract.json"),
        "source_summary_sha256": file_sha256(source / "summary.json"),
        "source_metrics_sha256": file_sha256(source / "experiment_b_metrics.csv"),
        "theta0_hash": source_contract["theta0_hash"],
        "model_init_seed": int(theta_payload.get("model_init_seed", 42)),
        "data_seeds": [int(value) for value in source_contract["data_seeds"]],
        "tail_classes": [int(value) for value in source_contract["tail_classes"]],
        "hard_k": int(source_contract["hard_k"]),
        "conditions": list(CONDITIONS),
        "background": {
            "topology": "controlled_clientlt",
            "clients": "all real clients with n_k,c=0",
            "local_training": "full client dataset, ordinary unweighted CE, 3 epochs, batch size 32",
            "aggregation": "sample-count FedAvg renormalized within class-absent clients",
            "partition_alpha": float(args.partition_alpha),
            "clientlt_purity": float(args.clientlt_purity),
            "rounds": 1,
        },
        "composition": "theta_post=theta0+(theta_local-theta0)+(theta_bg-theta0)",
        "tail_update_scale": 1.0,
        "background_update_scale": 1.0,
        "primary_endpoint": "R_c=G_post_c/G_local_c",
        "gain_reference": "theta0",
        "ratio_order": "average five hard-negative pairs and data seeds within tail class, then divide",
        "inference_unit": "20 tail classes",
        "directional_gate": "mean(R_hard-R_control)>0 and 95% tail-class bootstrap CI excludes zero",
        "manifest_hashes": {
            name: file_sha256(output / name) for name in BRIDGE_MANIFESTS
        },
        "implementation_hashes": {
            name: file_sha256(ROOT / name)
            for name in (
                "tools/compatibility_retention/core.py",
                "tools/compatibility_retention/run.py",
                "tools/boundary_evidence/core.py",
                "tools/boundary_evidence/run.py",
                "tools/semantic_acquisition/runtime.py",
                "utils/lora_aggregation.py",
                "utils/datasplit.py",
            )
        },
        "claim_boundary": (
            "Tests whether under-constrained c+r updates retain a smaller fraction of their local "
            "target gain under one identical real class-absent Client-LT FedAvg update. It does "
            "not attribute all of the observed 13.85pp final accuracy gap to this mechanism."
        ),
    }
    write_json(output / "experiment_contract.json", experiment_contract)
    return experiment_contract


def _read_contract(output: Path) -> dict:
    output = Path(output)
    contract = json.loads((output / "experiment_contract.json").read_text(encoding="utf-8"))
    source = Path(contract["source_boundary_dir"])
    # Re-run the source experiment's own provenance checks on every bridge
    # stage, rather than merely trusting the nested hashes in its JSON.
    _read_boundary_contract(source)
    expected_sources = {
        "experiment_contract.json": contract["source_contract_sha256"],
        "summary.json": contract["source_summary_sha256"],
        "experiment_b_metrics.csv": contract["source_metrics_sha256"],
    }
    for name, expected in expected_sources.items():
        if file_sha256(source / name) != expected:
            raise RuntimeError(f"Frozen source boundary artifact changed: {name}")
    for name, expected in contract["manifest_hashes"].items():
        if file_sha256(output / name) != expected:
            raise RuntimeError(f"Frozen bridge manifest changed: {name}")
    for name, expected in contract["implementation_hashes"].items():
        if file_sha256(ROOT / name) != expected:
            raise RuntimeError(f"Bridge implementation changed after preparation: {name}")
    if contract["tail_update_scale"] != 1.0 or contract["background_update_scale"] != 1.0:
        raise RuntimeError("The frozen bridge forbids tail/background scale sweeps")
    return contract


def _runtime(args, contract):
    source = Path(contract["source_boundary_dir"])
    runtime_args = SimpleNamespace(
        output_dir=source,
        data_dir=Path(args.data_dir),
        model_init_seed=int(contract["model_init_seed"]),
    )
    cfg, store, model, theta0, train_transform, eval_transform = _build_runtime(
        runtime_args, create_theta0=False
    )
    if tensor_mapping_hash(theta0) != contract["theta0_hash"]:
        raise RuntimeError("Bridge runtime theta0 differs from the frozen source experiment")
    return cfg, store, model, theta0, train_transform, eval_transform


def _state_path(output: Path, seed: int, client: int) -> Path:
    return output / "background_client_states" / f"seed_{seed}" / f"client_{client:02d}.pt"


def _background_path(output: Path, seed: int, tail_class: int) -> Path:
    return output / "background_aggregate_states" / f"seed_{seed}" / f"tail_{tail_class:03d}.pt"


def _save_state(path: Path, state: Mapping[str, torch.Tensor], **metadata) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"lora_state": dict(state), "state_hash": tensor_mapping_hash(state), **metadata}, path)


def _load_state(path: Path, theta0_hash: str) -> dict[str, torch.Tensor]:
    payload = _load_theta_payload(path)
    if payload.get("theta0_hash") != theta0_hash:
        raise RuntimeError(f"Cached state has the wrong theta0 provenance: {path}")
    state = {name: value.detach().cpu().clone() for name, value in payload["lora_state"].items()}
    if payload.get("state_hash") != tensor_mapping_hash(state):
        raise RuntimeError(f"Cached state hash mismatch: {path}")
    return state


def run_background(args) -> list[dict]:
    output = Path(args.output_dir)
    contract = _read_contract(output)
    cfg, store, model, theta0, train_transform, _ = _runtime(args, contract)
    execution = pd.read_csv(output / "background_execution_manifest.csv")
    client_manifest = pd.read_csv(output / "background_client_manifest.csv")
    client_metrics_path = output / "background_client_metrics.csv"
    client_rows = pd.read_csv(client_metrics_path).to_dict("records") if client_metrics_path.is_file() else []
    recorded = {(int(row["data_seed"]), int(row["client_id"])) for row in client_rows}
    local_states: dict[tuple[int, int], dict[str, torch.Tensor]] = {}
    for (seed, client), episode in execution.groupby(["data_seed", "client_id"], sort=True):
        key = (int(seed), int(client))
        state_file = _state_path(output, *key)
        if state_file.is_file():
            final_state = _load_state(state_file, contract["theta0_hash"])
        else:
            states, runtime = _train_client(model, cfg, theta0, episode, store, train_transform)
            final_state = states[3]
            _save_state(
                state_file,
                final_state,
                theta0_hash=contract["theta0_hash"],
                data_seed=key[0],
                client_id=key[1],
            )
        if key not in recorded:
            expected_steps = int(episode.groupby(["epoch", "batch_index"]).ngroups)
            client_rows.append({
                "data_seed": key[0],
                "client_id": key[1],
                "client_sample_count": int(
                    client_manifest[
                        (client_manifest.data_seed == key[0])
                        & (client_manifest.client_id == key[1])
                    ].client_sample_count.iloc[0]
                ),
                "update_norm_diagnostic": update_norm(theta0, final_state),
                "optimizer_steps": expected_steps,
                "scheduler_steps": 3,
                "precision": str(cfg.TRAINER.COOP.PREC),
                "state_hash": tensor_mapping_hash(final_state),
            })
            write_csv(client_metrics_path, client_rows)
            recorded.add(key)
        local_states[key] = final_state
        print(json.dumps({"stage": "background_client", "completed": key}, sort_keys=True))

    selection = pd.read_csv(output / "background_selection_manifest.csv")
    aggregate_path = output / "background_aggregate_metrics.csv"
    aggregate_rows = pd.read_csv(aggregate_path).to_dict("records") if aggregate_path.is_file() else []
    completed = {(int(row["data_seed"]), int(row["tail_class"])) for row in aggregate_rows}
    for (seed, tail_class), group in selection.groupby(["data_seed", "tail_class"], sort=True):
        key = (int(seed), int(tail_class))
        selected = group[group.class_absent_selected.astype(str).str.lower().isin(["true", "1"])]
        clients = [int(value) for value in selected.client_id.tolist()]
        if not clients or (selected.client_tail_count.astype(int) != 0).any():
            raise RuntimeError(f"Invalid class-absent selection for {key}")
        weights = {int(row.client_id): float(row.fedavg_weight) for row in selected.itertuples()}
        if not np.isclose(sum(weights.values()), 1.0, atol=1e-12, rtol=1e-12):
            raise RuntimeError(f"Background FedAvg weights do not sum to one for {key}")
        states = {client: local_states[(key[0], client)] for client in clients}
        background_state = _aggregate_weighted(model, theta0, states, weights)
        state_file = _background_path(output, *key)
        state_hash = tensor_mapping_hash(background_state)
        if state_file.is_file():
            cached = _load_state(state_file, contract["theta0_hash"])
            if tensor_mapping_hash(cached) != state_hash:
                raise RuntimeError(f"Cached background aggregate differs for {key}")
        else:
            _save_state(
                state_file,
                background_state,
                theta0_hash=contract["theta0_hash"],
                data_seed=key[0],
                tail_class=key[1],
            )
        if key not in completed:
            aggregate_rows.append({
                "data_seed": key[0],
                "tail_class": key[1],
                "absent_client_count": len(clients),
                "absent_sample_count": int(selected.client_sample_count.sum()),
                "fedavg_weight_sum": float(sum(weights.values())),
                "background_update_norm_diagnostic": update_norm(theta0, background_state),
                "background_state_hash": state_hash,
            })
            write_csv(aggregate_path, aggregate_rows)
            completed.add(key)
        print(json.dumps({"stage": "background_aggregate", "completed": key}, sort_keys=True))
    return aggregate_rows


def run_bridge(args) -> list[dict]:
    output = Path(args.output_dir)
    contract = _read_contract(output)
    source = Path(contract["source_boundary_dir"])
    cfg, store, model, theta0, train_transform, eval_transform = _runtime(args, contract)
    execution = pd.read_csv(source / "execution_manifest.csv")
    source_metrics = pd.read_csv(source / "experiment_b_metrics.csv")
    source_index = {
        (int(row.data_seed), int(row.tail_class), int(row.hard_class), str(row.condition)): row
        for row in source_metrics.itertuples()
    }
    metrics_path = output / "bridge_metrics.csv"
    rows = pd.read_csv(metrics_path).to_dict("records") if metrics_path.is_file() else []
    completed = {
        (int(row["data_seed"]), int(row["tail_class"]), int(row["hard_class"]), str(row["condition"]))
        for row in rows
    }
    groups = execution.groupby(["data_seed", "tail_class", "hard_class", "condition"], sort=True)
    for (seed, c, h, condition), episode in groups:
        key = (int(seed), int(c), int(h), str(condition))
        if key in completed:
            continue
        background_file = _background_path(output, key[0], key[1])
        if not background_file.is_file():
            raise FileNotFoundError(f"Run --stage background first; missing {background_file}")
        background_state = _load_state(background_file, contract["theta0_hash"])
        background_hash = tensor_mapping_hash(background_state)
        source_row = source_index[key]
        load_lora_state(model, theta0)
        states, runtime = _train_client(model, cfg, theta0, episode, store, train_transform)
        local_state = states[3]
        load_lora_state(model, local_state)
        local_metrics = _eval_pair(model, store, eval_transform, key[1], key[2])
        reproduction_error = abs(float(local_metrics["m_c"]) - float(source_row.after_m_c))
        if reproduction_error > 1e-6:
            raise RuntimeError(
                f"Reconstructed Experiment B update does not reproduce after_m_c for {key}: "
                f"absolute error={reproduction_error}"
            )
        post_state = additive_post_state(theta0, local_state, background_state)
        load_lora_state(model, post_state)
        post_metrics = _eval_pair(model, store, eval_transform, key[1], key[2])
        theta0_m_c = float(source_row.before_m_c)
        g_local = float(local_metrics["m_c"] - theta0_m_c)
        g_post = float(post_metrics["m_c"] - theta0_m_c)
        rows.append({
            "data_seed": key[0],
            "tail_class": key[1],
            "hard_class": key[2],
            "hard_rank": int(source_row.hard_rank),
            "control_class": int(source_row.control_class),
            "condition": key[3],
            "theta0_m_c": theta0_m_c,
            "local_m_c": float(local_metrics["m_c"]),
            "post_m_c": float(post_metrics["m_c"]),
            "g_local": g_local,
            "g_post": g_post,
            "background_state_hash": background_hash,
            "local_state_hash": tensor_mapping_hash(local_state),
            "post_state_hash": tensor_mapping_hash(post_state),
            "source_reproduction_abs_error": reproduction_error,
            "optimizer_steps": int(runtime["optimizer_steps_successful"]),
            "scheduler_steps": int(runtime["scheduler_steps"]),
            "precision": str(runtime["precision"]),
        })
        write_csv(metrics_path, rows)
        completed.add(key)
        print(json.dumps({
            "stage": "bridge",
            "completed": key,
            "g_local": g_local,
            "g_post": g_post,
        }, sort_keys=True))
    return rows


def summarize(args) -> dict:
    output = Path(args.output_dir)
    contract = _read_contract(output)
    metrics_path = output / "bridge_metrics.csv"
    if not metrics_path.is_file():
        raise FileNotFoundError("Bridge metrics are missing; run --stage bridge first")
    rows = pd.read_csv(metrics_path).to_dict("records")
    expected = (
        len(contract["data_seeds"])
        * len(contract["tail_classes"])
        * int(contract["hard_k"])
        * len(CONDITIONS)
    )
    if len(rows) != expected:
        raise RuntimeError(f"Bridge is incomplete: expected {expected} rows, found {len(rows)}")
    frame = pd.DataFrame(rows)
    for (seed, tail_class), group in frame.groupby(["data_seed", "tail_class"], sort=True):
        if group.background_state_hash.nunique() != 1:
            raise RuntimeError(
                f"Paired conditions did not receive one identical background update: {(seed, tail_class)}"
            )
    class_rows = tail_retention_rows(rows)
    if len(class_rows) != len(contract["tail_classes"]):
        raise RuntimeError("The bridge does not contain exactly one primary ratio per tail class")
    write_csv(output / "bridge_per_tail_class.csv", class_rows)
    fields = {
        "hard_competitor": "hard_retention_ratio",
        "matched_control": "control_retention_ratio",
        "hard_competitor_minus_matched_control": "hard_minus_control_retention_ratio",
    }
    summaries = {}
    for index, (name, field) in enumerate(fields.items()):
        values = {int(row["tail_class"]): [float(row[field])] for row in class_rows}
        summaries[name] = class_cluster_summary(values, seed=20260903 + index)
    write_csv(output / "bridge_main_results.csv", [
        {"condition_or_contrast": name, "retention_ratio": value["mean"],
         "ci_low": value["ci_low"], "ci_high": value["ci_high"],
         "tail_class_count": value["tail_class_count"]}
        for name, value in summaries.items()
    ])
    contrast = summaries["hard_competitor_minus_matched_control"]
    gate = bool(contrast["mean"] > 0.0 and contrast["ci_low"] > 0.0)
    result = {
        "schema_version": 1,
        "verdict": (
            "COMPATIBILITY_TO_RETENTION_BRIDGE_SUPPORTED"
            if gate else "COMPATIBILITY_TO_RETENTION_BRIDGE_NOT_SUPPORTED"
        ),
        "primary_endpoint": "R_c=G_post_c/G_local_c",
        "by_condition": {
            "hard_competitor": summaries["hard_competitor"],
            "matched_control": summaries["matched_control"],
        },
        "hard_competitor_minus_matched_control": contrast,
        "directional_gate": {
            "R_c_plus_r_lower_than_R_c_plus_h_with_95pct_CI": gate,
        },
        "inference_unit": (
            "20 tail classes; G_local and G_post are first averaged over five hard-negative "
            "pairs and data seeds within class, then R_c is formed"
        ),
        "claim_boundary": contract["claim_boundary"],
    }
    write_json(output / "summary.json", result)
    return result


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage", choices=("prepare", "background", "bridge", "summarize", "all"), required=True
    )
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--partition-alpha", type=float, default=0.5)
    parser.add_argument("--clientlt-purity", type=float, default=0.8)
    args = parser.parse_args(argv)
    if args.partition_alpha <= 0:
        parser.error("partition-alpha must be positive")
    if not 0 < args.clientlt_purity <= 1:
        parser.error("clientlt-purity must be in (0,1]")
    return args


def main(argv=None):
    args = parse_args(argv)
    if args.stage in ("prepare", "all"):
        prepare(args)
    if args.stage in ("background", "all"):
        run_background(args)
    if args.stage in ("bridge", "all"):
        run_bridge(args)
    if args.stage in ("summarize", "all"):
        result = summarize(args)
        print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
