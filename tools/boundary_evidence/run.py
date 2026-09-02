"""Run the minimal Client-LT boundary-evidence experiment.

Scientific scope:
  A. training-free hard-negative co-exposure under two matched topologies;
  B. two-condition local adaptation, c+h versus c+r.

The shared reference is always the topology-independent pre-federation theta0.
Hard-negative selection, local training, and evaluation use three disjoint
sample pools: P_select, D_local, and P_eval.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
import torch

from tools.boundary_evidence.core import (
    choose_matched_control,
    class_cluster_summary,
    coexposure_rate,
    hard_negative_ranking,
    metric_deltas,
    pairwise_boundary_metrics,
)
from tools.semantic_acquisition.common import (
    deterministic_choice,
    file_sha256,
    stable_seed,
    tensor_mapping_hash,
    write_csv,
    write_json,
)
from tools.semantic_acquisition.manifests import DEFAULT_DATA
from tools.semantic_acquisition.runtime import (
    CifarRawStore,
    _load_cliplora_api,
    _predict_ids,
    _set_determinism,
    _train_client,
    build_experiment_cfg,
    load_lora_state,
    lora_state,
    trainable_named,
    update_norm,
)
from utils.datasplit import (
    partition_client_longtail_controlled,
    partition_fixed_marginal_dirichlet,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "output" / "boundary_evidence"
CONDITIONS = ("hard_competitor", "matched_control")
PRIMARY_METRICS = ("delta_m_c", "delta_m_h", "delta_pair_accuracy")


@dataclass(frozen=True)
class ExperimentInputs:
    labels: np.ndarray
    raw_train_ids: np.ndarray
    global_counts: np.ndarray
    tail_classes: list[int]
    class_names: list[str]


def _load_inputs(data_dir: Path) -> ExperimentInputs:
    """Rebuild the exact CIFAR-100-LT pool without any semantic artifact."""
    data_dir = Path(data_dir)
    with (data_dir / "train").open("rb") as handle:
        train = pickle.load(handle, encoding="latin1")
    with (data_dir / "meta").open("rb") as handle:
        meta = pickle.load(handle, encoding="latin1")
    raw_labels = np.asarray(train["fine_labels"], dtype=np.int64)
    generator = np.random.RandomState(1)
    selected = []
    image_max = float(len(raw_labels)) / 100.0
    for class_id in range(100):
        count = int(image_max * 0.01 ** (float(class_id) / 99.0))
        indices = np.flatnonzero(raw_labels == class_id).copy()
        generator.shuffle(indices)
        selected.extend(int(value) for value in indices[:count].tolist())
    raw_train_ids = np.asarray(selected, dtype=np.int64)
    labels = raw_labels[raw_train_ids]
    global_counts = np.bincount(labels, minlength=100)
    # Larger class ids are later in the exponential LT schedule and win
    # integer-count ties at the bottom-20 boundary.
    tail_classes = sorted(
        range(100), key=lambda class_id: (int(global_counts[class_id]), -class_id)
    )[:20]
    class_names = [str(value).replace("_", " ") for value in meta["fine_label_names"]]
    return ExperimentInputs(labels, raw_train_ids, global_counts, tail_classes, class_names)


def _parse_ints(value: str | Sequence[int]) -> list[int]:
    raw = value if isinstance(value, (list, tuple)) else str(value).split(",")
    result = [int(item) for item in raw]
    if not result or len(set(result)) != len(result):
        raise ValueError("Seeds must be a non-empty unique list")
    return result


def _counts_matrix(labels: np.ndarray, partition: Mapping[int, np.ndarray], classes: int) -> np.ndarray:
    return np.stack(
        [
            np.bincount(labels[np.asarray(partition[client_id], dtype=np.int64)], minlength=classes)
            for client_id in sorted(partition)
        ],
        axis=0,
    )


def _build_partitions(inputs, seed: int, alpha: float, purity: float):
    clientlt = partition_client_longtail_controlled(
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
    capacities = [len(clientlt[client_id]) for client_id in range(30)]
    matched = partition_fixed_marginal_dirichlet(
        inputs.labels,
        capacities,
        len(inputs.class_names),
        float(alpha),
        rng=np.random.RandomState(int(seed) + 100003),
    )
    clientlt_counts = _counts_matrix(inputs.labels, clientlt, len(inputs.class_names))
    matched_counts = _counts_matrix(inputs.labels, matched, len(inputs.class_names))
    if not np.array_equal(clientlt_counts.sum(axis=0), matched_counts.sum(axis=0)):
        raise RuntimeError("Matched Dirichlet changed the global class marginal n_c")
    if not np.array_equal(clientlt_counts.sum(axis=1), matched_counts.sum(axis=1)):
        raise RuntimeError("Matched Dirichlet changed the client capacity marginal n_k")
    return {"clientlt": clientlt_counts, "matched_dirichlet": matched_counts}


def _load_theta_payload(path: Path) -> dict:
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    if isinstance(payload, Mapping) and "lora_state" in payload:
        state = payload["lora_state"]
    else:
        state = payload
        payload = {"lora_state": state}
    if not isinstance(state, Mapping) or not state:
        raise TypeError(f"theta0 is not a non-empty LoRA tensor mapping: {path}")
    if any(not isinstance(value, torch.Tensor) for value in state.values()):
        raise TypeError("theta0 contains non-tensor LoRA values")
    return dict(payload)


def _build_runtime(args, *, create_theta0: bool):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    if not torch.cuda.is_available():
        raise RuntimeError("Boundary-evidence model stages require CUDA")
    cfg = build_experiment_cfg(Path(args.output_dir))
    _set_determinism(args.model_init_seed)
    store = CifarRawStore(Path(args.data_dir))
    from Dassl.dassl.data.transforms import build_transform

    build_model, _, _ = _load_cliplora_api()
    train_transform = build_transform(cfg, is_train=True)
    eval_transform = build_transform(cfg, is_train=False)
    model = build_model(cfg, store.class_names).cuda()
    constructed = lora_state(model)
    theta_path = Path(args.output_dir) / "theta0.pt"
    if create_theta0:
        # Always construct theta0 before any topology is instantiated.  An
        # arbitrary checkpoint option is intentionally absent: accepting one
        # would allow a Client-LT/Dirichlet-trained state to contaminate H_c.
        theta = constructed
        payload = {"lora_state": theta, "model_init_seed": int(args.model_init_seed)}
        theta_path.parent.mkdir(parents=True, exist_ok=True)
        if theta_path.is_file():
            existing = _load_theta_payload(theta_path)["lora_state"]
            if tensor_mapping_hash(existing) != tensor_mapping_hash(theta):
                raise RuntimeError(
                    "Output already contains a different theta0; use a new output directory"
                )
        else:
            torch.save(payload, theta_path)
    elif not theta_path.is_file():
        raise FileNotFoundError(f"Prepared common theta0 is missing: {theta_path}")
    payload = _load_theta_payload(theta_path)
    theta = {name: value.detach().cpu().clone() for name, value in payload["lora_state"].items()}
    if set(theta) != set(constructed):
        raise RuntimeError("Prepared theta0 LoRA keys do not match the configured model")
    load_lora_state(model, theta)
    if sorted(name for name, _ in trainable_named(model)) != sorted(theta):
        raise RuntimeError("Trainable scope differs from the common theta0 LoRA state")
    return cfg, store, model, theta, train_transform, eval_transform


def _selection_ids(store: CifarRawStore, inputs, class_id: int, count: int) -> list[str]:
    local_raw = set(int(value) for value in inputs.raw_train_ids.tolist())
    candidates = [
        int(raw_id)
        for raw_id in np.flatnonzero(store.train_labels == int(class_id)).tolist()
        if int(raw_id) not in local_raw
    ]
    chosen = deterministic_choice(
        candidates,
        min(int(count), len(candidates)),
        "boundary-evidence-p-select",
        int(class_id),
    )
    if not chosen:
        raise RuntimeError(f"No P_select samples remain outside D_local for class {class_id}")
    return [f"train:{raw_id}" for raw_id in chosen]


def _predict_class(model, store, transform, class_id: int, sample_ids: Sequence[str]):
    return _predict_ids(
        model,
        store,
        transform,
        list(sample_ids),
        [int(class_id)] * len(sample_ids),
    )[0]


def prepare(args) -> dict:
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    inputs = _load_inputs(Path(args.data_dir))
    cfg, store, model, theta0, _, eval_transform = _build_runtime(args, create_theta0=True)
    selection: dict[int, list[str]] = {}
    hard_rows, pair_rows = [], []
    for class_id in inputs.tail_classes:
        sample_ids = _selection_ids(store, inputs, class_id, args.selection_samples)
        selection[int(class_id)] = sample_ids
        logits = _predict_class(model, store, eval_transform, class_id, sample_ids)
        mean_logits = logits.double().mean(dim=0)
        ranking = hard_negative_ranking(mean_logits, class_id)
        top20 = ranking[:20]
        for rank, hard_class in enumerate(ranking, start=1):
            hard_rows.append({
                "tail_class": int(class_id),
                "competitor_class": int(hard_class),
                "rank": int(rank),
                "selection_margin": float((mean_logits[class_id] - mean_logits[hard_class]).item()),
                "selected_top5": rank <= int(args.hard_k),
                "excluded_control_top20": rank <= 20,
            })
        for hard_rank, hard_class in enumerate(ranking[: int(args.hard_k)], start=1):
            control = choose_matched_control(
                inputs.global_counts, class_id, hard_class, top20
            )
            pair_rows.append({
                "tail_class": int(class_id),
                "hard_class": int(hard_class),
                "hard_rank": int(hard_rank),
                "control_class": int(control),
                "hard_global_count": int(inputs.global_counts[hard_class]),
                "control_global_count": int(inputs.global_counts[control]),
                "frequency_gap": int(abs(inputs.global_counts[hard_class] - inputs.global_counts[control])),
                "control_outside_top20": bool(control not in set(top20)),
            })
    write_csv(output / "hard_negative_ranking.csv", hard_rows)
    write_csv(output / "pair_manifest.csv", pair_rows)

    # P_select uses train examples outside the exact LT pool; P_eval is the
    # official test split. D_local is the exact LT pool, so all three are
    # disjoint by construction and verified explicitly below.
    probe_rows = []
    for class_id, sample_ids in sorted(selection.items()):
        probe_rows.extend(
            {"pool": "P_select", "class_id": class_id, "sample_id": sample_id}
            for sample_id in sample_ids
        )
    eval_classes = sorted(
        {int(row["tail_class"]) for row in pair_rows}
        | {int(row["hard_class"]) for row in pair_rows}
    )
    for class_id in eval_classes:
        probe_rows.extend(
            {"pool": "P_eval", "class_id": class_id, "sample_id": f"test:{int(raw_id)}"}
            for raw_id in np.flatnonzero(store.test_labels == class_id).tolist()
        )
    write_csv(output / "probe_manifest.csv", probe_rows)
    select_ids = {row["sample_id"] for row in probe_rows if row["pool"] == "P_select"}
    eval_ids = {row["sample_id"] for row in probe_rows if row["pool"] == "P_eval"}
    local_ids = {f"train:{int(raw_id)}" for raw_id in inputs.raw_train_ids.tolist()}
    disjoint = not (select_ids & eval_ids or select_ids & local_ids or eval_ids & local_ids)
    if not disjoint:
        raise RuntimeError("P_select, D_local, and P_eval are not mutually disjoint")

    coexposure_rows = []
    for seed in _parse_ints(args.data_seeds):
        topologies = _build_partitions(inputs, seed, args.partition_alpha, args.clientlt_purity)
        for topology, counts in topologies.items():
            for pair in pair_rows:
                result = coexposure_rate(counts, pair["tail_class"], pair["hard_class"])
                coexposure_rows.append({
                    "data_seed": int(seed),
                    "topology": topology,
                    "tail_class": int(pair["tail_class"]),
                    "hard_class": int(pair["hard_class"]),
                    "hard_rank": int(pair["hard_rank"]),
                    **result,
                })
    write_csv(output / "experiment_a_coexposure.csv", coexposure_rows)

    local_rows, execution_rows = _build_local_manifests(inputs, pair_rows, _parse_ints(args.data_seeds))
    write_csv(output / "local_manifest.csv", local_rows)
    write_csv(output / "execution_manifest.csv", execution_rows)
    manifest_names = (
        "hard_negative_ranking.csv",
        "pair_manifest.csv",
        "probe_manifest.csv",
        "experiment_a_coexposure.csv",
        "local_manifest.csv",
        "execution_manifest.csv",
    )
    contract = {
        "schema_version": 1,
        "scientific_scope": "Experiment A co-exposure plus Experiment B c+h versus c+r",
        "theta": "topology-independent pre-federation theta0",
        "theta0_hash": tensor_mapping_hash(theta0),
        "theta0_path": str((output / "theta0.pt").resolve()),
        "hard_negative_rule": "ascending E_Pselect[z_c-z_h], top-5 primary",
        "hard_k": int(args.hard_k),
        "control_rule": "frequency-nearest class outside frozen Top20(c), class-id tie break",
        "pools": {
            "P_select": "CIFAR-100 train samples excluded from exact federated LT pool",
            "D_local": "exact federated CIFAR-100-LT train pool",
            "P_eval": "CIFAR-100 test split",
            "mutually_disjoint": disjoint,
        },
        "conditions": list(CONDITIONS),
        "primary_metrics": list(PRIMARY_METRICS),
        "tail_classes": [int(value) for value in inputs.tail_classes],
        "update_norm": "diagnostic_only_not_matched",
        "aggregation_unit": "mean five hard-negative pairs within tail class, then infer over 20 tail classes",
        "data_seeds": _parse_ints(args.data_seeds),
        "training": {
            "trainer": "ClipLora", "encoder": "vision", "rank": 2,
            "position": "top3", "params": ["q", "v"], "local_epochs": 3,
            "batch_size": 32, "optimizer": "sgd", "lr": 0.002,
            "precision": str(cfg.TRAINER.COOP.PREC), "fedavg": False,
        },
        "manifest_hashes": {name: file_sha256(output / name) for name in manifest_names},
        "implementation_hashes": {
            name: file_sha256(ROOT / name)
            for name in (
                "tools/boundary_evidence/core.py",
                "tools/boundary_evidence/run.py",
                "tools/semantic_acquisition/runtime.py",
                "trainers/cliplora.py",
                "utils/datasplit.py",
            )
        },
    }
    write_json(output / "experiment_contract.json", contract)
    return contract


def _build_local_manifests(inputs, pair_rows: Sequence[Mapping], seeds: Sequence[int]):
    pools = {
        class_id: [
            int(raw_id)
            for raw_id, label in zip(inputs.raw_train_ids.tolist(), inputs.labels.tolist())
            if int(label) == class_id
        ]
        for class_id in range(len(inputs.class_names))
    }
    base_rows, execution_rows = [], []
    for seed in seeds:
        for pair in pair_rows:
            c, h, r = (int(pair[name]) for name in ("tail_class", "hard_class", "control_class"))
            c_ids = sorted(pools[c])
            companion_count = min(len(c_ids), len(pools[h]), len(pools[r]))
            if not c_ids or companion_count <= 0:
                raise RuntimeError(f"Insufficient D_local samples for pair {(c, h, r)}")
            h_ids = deterministic_choice(pools[h], companion_count, "boundary-h", seed, c, h)
            r_ids = deterministic_choice(pools[r], companion_count, "boundary-r", seed, c, h, r)
            for condition, companion_class, companion_ids in (
                ("hard_competitor", h, h_ids),
                ("matched_control", r, r_ids),
            ):
                samples = [("tail", c, raw_id) for raw_id in c_ids] + [
                    ("companion", companion_class, raw_id) for raw_id in companion_ids
                ]
                if len(samples) > 32:
                    raise RuntimeError("A local episode exceeds the preregistered single batch")
                for position, (role, label, raw_id) in enumerate(samples):
                    base_rows.append({
                        "data_seed": int(seed), "tail_class": c, "hard_class": h,
                        "hard_rank": int(pair["hard_rank"]), "control_class": r,
                        "condition": condition, "position_in_batch": position,
                        "slot_role": role, "base_sample_id": f"train:{raw_id}", "label": label,
                        "tail_sample_count": len(c_ids), "companion_sample_count": companion_count,
                        "total_sample_count": len(samples),
                    })
                    for epoch in (1, 2, 3):
                        execution_rows.append({
                            "data_seed": int(seed), "tail_class": c, "hard_class": h,
                            "hard_rank": int(pair["hard_rank"]), "control_class": r,
                            "condition": condition, "epoch": epoch, "batch_index": 0,
                            "position_in_batch": position, "slot_role": role,
                            "base_sample_id": f"train:{raw_id}", "label": label,
                            # Binding augmentation to the paired slot makes c identical and
                            # gives h/r companion slots the same stochastic transform seed.
                            "augmentation_seed": stable_seed(
                                "boundary-evidence-augmentation", seed, c, h, epoch, position
                            ),
                        })
    frame = pd.DataFrame(base_rows)
    for (seed, c, h), group in frame.groupby(["data_seed", "tail_class", "hard_class"]):
        left = group[group.condition == "hard_competitor"]
        right = group[group.condition == "matched_control"]
        if len(left) != len(right):
            raise RuntimeError(f"Condition sample counts differ for {(seed, c, h)}")
        if left[left.slot_role == "tail"].base_sample_id.tolist() != right[right.slot_role == "tail"].base_sample_id.tolist():
            raise RuntimeError(f"Tail images differ across conditions for {(seed, c, h)}")
        if int(left.companion_sample_count.iloc[0]) != int(right.companion_sample_count.iloc[0]):
            raise RuntimeError(f"Companion counts differ for {(seed, c, h)}")
    return base_rows, execution_rows


def _read_contract(output: Path) -> dict:
    contract = json.loads((output / "experiment_contract.json").read_text(encoding="utf-8"))
    for name, expected in contract["manifest_hashes"].items():
        if file_sha256(output / name) != expected:
            raise RuntimeError(f"Prepared manifest changed after freezing: {name}")
    for name, expected in contract["implementation_hashes"].items():
        if file_sha256(ROOT / name) != expected:
            raise RuntimeError(f"Implementation changed after preparation: {name}")
    if contract["hard_k"] != 5 or contract["conditions"] != list(CONDITIONS):
        raise RuntimeError("Prepared contract is not the approved top-5/two-condition design")
    return contract


def _eval_pair(model, store, transform, c: int, h: int) -> dict[str, float]:
    c_ids = [f"test:{int(value)}" for value in np.flatnonzero(store.test_labels == c).tolist()]
    h_ids = [f"test:{int(value)}" for value in np.flatnonzero(store.test_labels == h).tolist()]
    logits_c = _predict_class(model, store, transform, c, c_ids)
    logits_h = _predict_class(model, store, transform, h, h_ids)
    return pairwise_boundary_metrics(logits_c, logits_h, c, h)


def run_local(args) -> list[dict]:
    output = Path(args.output_dir)
    contract = _read_contract(output)
    cfg, store, model, theta0, train_transform, eval_transform = _build_runtime(args, create_theta0=False)
    if tensor_mapping_hash(theta0) != contract["theta0_hash"]:
        raise RuntimeError("Local runtime theta0 differs from the frozen preparation model")
    execution = pd.read_csv(output / "execution_manifest.csv")
    metrics_path = output / "experiment_b_metrics.csv"
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
        load_lora_state(model, theta0)
        before = _eval_pair(model, store, eval_transform, int(c), int(h))
        states, runtime = _train_client(model, cfg, theta0, episode, store, train_transform)
        final_state = states[3]
        load_lora_state(model, final_state)
        after = _eval_pair(model, store, eval_transform, int(c), int(h))
        delta = metric_deltas(before, after)
        rows.append({
            "data_seed": int(seed), "tail_class": int(c), "hard_class": int(h),
            "hard_rank": int(episode.hard_rank.iloc[0]),
            "control_class": int(episode.control_class.iloc[0]),
            "condition": str(condition),
            **{f"before_{name}": value for name, value in before.items()},
            **{f"after_{name}": value for name, value in after.items()},
            **delta,
            "update_norm_diagnostic": update_norm(theta0, final_state),
            "optimizer_steps": int(runtime["optimizer_steps_successful"]),
            "scheduler_steps": int(runtime["scheduler_steps"]),
            "precision": str(runtime["precision"]),
        })
        write_csv(metrics_path, rows)
        print(json.dumps({"stage": "local", "completed": key, **delta}, sort_keys=True))
    return rows


def _cluster(rows: Sequence[Mapping], field: str, *, seed_offset: int = 0):
    values: dict[int, list[float]] = {}
    for row in rows:
        values.setdefault(int(row["tail_class"]), []).append(float(row[field]))
    return class_cluster_summary(values, seed=20260903 + int(seed_offset))


def summarize(args) -> dict:
    output = Path(args.output_dir)
    contract = _read_contract(output)
    a_rows = pd.read_csv(output / "experiment_a_coexposure.csv").to_dict("records")
    a_by_topology = {
        topology: _cluster([row for row in a_rows if row["topology"] == topology], "q", seed_offset=index)
        for index, topology in enumerate(("clientlt", "matched_dirichlet"))
    }
    a_index = {
        (int(row["data_seed"]), int(row["tail_class"]), int(row["hard_class"])): row
        for row in a_rows if row["topology"] == "clientlt"
    }
    a_contrasts = []
    for row in a_rows:
        if row["topology"] != "matched_dirichlet":
            continue
        key = (int(row["data_seed"]), int(row["tail_class"]), int(row["hard_class"]))
        a_contrasts.append({**row, "dirichlet_minus_clientlt_q": float(row["q"] - a_index[key]["q"])})
    a_gap = _cluster(a_contrasts, "dirichlet_minus_clientlt_q", seed_offset=10)
    a_class_rows = []
    a_frame = pd.DataFrame(a_rows)
    for class_id, group in a_frame.groupby("tail_class", sort=True):
        clientlt_q = float(group[group.topology == "clientlt"].q.mean())
        dirichlet_q = float(group[group.topology == "matched_dirichlet"].q.mean())
        a_class_rows.append({
            "tail_class": int(class_id),
            "clientlt_q": clientlt_q,
            "matched_dirichlet_q": dirichlet_q,
            "dirichlet_minus_clientlt_q": dirichlet_q - clientlt_q,
        })
    write_csv(output / "experiment_a_per_tail_class.csv", a_class_rows)
    write_csv(output / "experiment_a_summary.csv", [
        {
            "topology_or_contrast": "clientlt",
            **a_by_topology["clientlt"],
        },
        {
            "topology_or_contrast": "matched_dirichlet",
            **a_by_topology["matched_dirichlet"],
        },
        {
            "topology_or_contrast": "matched_dirichlet_minus_clientlt",
            **a_gap,
        },
    ])

    metrics_path = output / "experiment_b_metrics.csv"
    if not metrics_path.is_file():
        raise FileNotFoundError("Experiment B metrics are missing; run --stage local first")
    b_rows = pd.read_csv(metrics_path).to_dict("records")
    expected = (
        len(contract["data_seeds"])
        * len(contract["tail_classes"])
        * int(contract["hard_k"])
        * len(CONDITIONS)
    )
    if len(b_rows) != expected:
        raise RuntimeError(f"Experiment B is incomplete: expected {expected} rows, found {len(b_rows)}")
    index = {
        (int(row["data_seed"]), int(row["tail_class"]), int(row["hard_class"]), str(row["condition"])): row
        for row in b_rows
    }
    contrast_rows = []
    for seed, c, h, condition in sorted(index):
        if condition != "hard_competitor":
            continue
        hard = index[(seed, c, h, "hard_competitor")]
        control = index[(seed, c, h, "matched_control")]
        for name in ("m_c", "m_h", "pair_accuracy"):
            if float(hard[f"before_{name}"]) != float(control[f"before_{name}"]):
                raise RuntimeError(
                    f"Paired conditions did not start from identical theta0 metrics: {(seed, c, h, name)}"
                )
        contrast_rows.append({
            "data_seed": seed, "tail_class": c, "hard_class": h,
            **{f"contrast_{metric}": float(hard[metric] - control[metric]) for metric in PRIMARY_METRICS},
        })
    write_csv(output / "experiment_b_paired_contrasts.csv", contrast_rows)
    condition_summary = {}
    for condition_index, condition in enumerate(CONDITIONS):
        condition_rows = [row for row in b_rows if row["condition"] == condition]
        condition_summary[condition] = {
            metric: _cluster(condition_rows, metric, seed_offset=100 + condition_index * 10 + metric_index)
            for metric_index, metric in enumerate(PRIMARY_METRICS)
        }
    contrast_summary = {
        metric: _cluster(contrast_rows, f"contrast_{metric}", seed_offset=200 + index)
        for index, metric in enumerate(PRIMARY_METRICS)
    }
    b_class_rows = []
    b_frame = pd.DataFrame(b_rows)
    contrast_frame = pd.DataFrame(contrast_rows)
    for class_id in sorted(int(value) for value in contract["tail_classes"]):
        for condition in CONDITIONS:
            group = b_frame[(b_frame.tail_class == class_id) & (b_frame.condition == condition)]
            b_class_rows.append({
                "tail_class": class_id,
                "condition": condition,
                **{metric: float(group[metric].mean()) for metric in PRIMARY_METRICS},
            })
        group = contrast_frame[contrast_frame.tail_class == class_id]
        b_class_rows.append({
            "tail_class": class_id,
            "condition": "hard_competitor_minus_matched_control",
            **{metric: float(group[f"contrast_{metric}"].mean()) for metric in PRIMARY_METRICS},
        })
    write_csv(output / "experiment_b_per_tail_class.csv", b_class_rows)
    main_rows = []
    for condition in CONDITIONS:
        main_rows.append({
            "condition": condition,
            **{
                metric: float(condition_summary[condition][metric]["mean"])
                for metric in PRIMARY_METRICS
            },
            **{
                f"{metric}_ci_low": float(condition_summary[condition][metric]["ci_low"])
                for metric in PRIMARY_METRICS
            },
            **{
                f"{metric}_ci_high": float(condition_summary[condition][metric]["ci_high"])
                for metric in PRIMARY_METRICS
            },
        })
    main_rows.append({
        "condition": "hard_competitor_minus_matched_control",
        **{metric: float(contrast_summary[metric]["mean"]) for metric in PRIMARY_METRICS},
        **{f"{metric}_ci_low": float(contrast_summary[metric]["ci_low"]) for metric in PRIMARY_METRICS},
        **{f"{metric}_ci_high": float(contrast_summary[metric]["ci_high"]) for metric in PRIMARY_METRICS},
    })
    write_csv(output / "experiment_b_main_results.csv", main_rows)
    target = contrast_summary["delta_m_c"]
    if target["ci_high"] < 0:
        target_interpretation = "OPPOSITE_SIDE_PRESERVATION_WITH_TARGET_SIDE_TRADEOFF"
    elif target["ci_low"] > 0:
        target_interpretation = "HARD_COMPETITOR_IMPROVES_BOTH_SIDES"
    else:
        target_interpretation = "TARGET_SIDE_COMPARABLE_WITHIN_CLASS_CLUSTER_UNCERTAINTY"
    gates = {
        "clientlt_coexposure_lower": a_gap["mean"] > 0,
        "hard_competitor_preserves_h_side_more": contrast_summary["delta_m_h"]["mean"] > 0,
        "hard_competitor_improves_pair_accuracy_more": contrast_summary["delta_pair_accuracy"]["mean"] > 0,
    }
    if not all(gates.values()):
        verdict = "BOUNDARY_EVIDENCE_ASYMMETRY_NOT_SUPPORTED"
    elif target_interpretation == "OPPOSITE_SIDE_PRESERVATION_WITH_TARGET_SIDE_TRADEOFF":
        verdict = "BOUNDARY_PRESERVATION_WITH_TARGET_SIDE_TRADEOFF"
    else:
        verdict = "BOUNDARY_EVIDENCE_ASYMMETRY_SUPPORTED"
    summary = {
        "schema_version": 1,
        "verdict": verdict,
        "experiment_a": {"by_topology": a_by_topology, "dirichlet_minus_clientlt": a_gap},
        "experiment_b": {
            "by_condition": condition_summary,
            "hard_competitor_minus_matched_control": contrast_summary,
            "target_side_interpretation": target_interpretation,
        },
        "directional_gates": gates,
        "inference_unit": "20 tail classes; five hard-negative pairs and data seeds averaged within class",
        "claim_boundary": (
            "Experiment B tests opposite-side preservation during matched local tail adaptation. "
            "Delta M_c is not a gate, but determines whether the result supports comparable "
            "tail adaptation or only a target-side/opposite-side trade-off. Experiment B does "
            "not by itself establish rewrite causality."
        ),
    }
    write_json(output / "summary.json", summary)
    return summary


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("prepare", "local", "summarize", "all"), required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--model-init-seed", type=int, default=42)
    parser.add_argument("--data-seeds", default="42,2026")
    parser.add_argument("--hard-k", type=int, default=5)
    parser.add_argument("--selection-samples", type=int, default=64)
    parser.add_argument("--partition-alpha", type=float, default=0.5)
    parser.add_argument("--clientlt-purity", type=float, default=0.8)
    args = parser.parse_args(argv)
    if args.hard_k != 5:
        parser.error("The approved primary design fixes hard-k at 5")
    if args.selection_samples < 1:
        parser.error("selection-samples must be positive")
    _parse_ints(args.data_seeds)
    return args


def main(argv=None):
    args = parse_args(argv)
    if args.stage in ("prepare", "all"):
        prepare(args)
    if args.stage in ("local", "all"):
        run_local(args)
    if args.stage in ("summarize", "all"):
        result = summarize(args)
        print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
