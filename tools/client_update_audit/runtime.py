from __future__ import annotations

import argparse
import json
import platform
import sys
import traceback
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from torchvision import transforms as T

from tools.breadth_audit.inputs import load_preregistered_neighbors
from tools.breadth_audit.metrics import neighbor_discrimination_metrics
from tools.client_update_audit.protocol import TAIL_CLASSES, frozen_protocol
from tools.semantic_acquisition.common import (
    file_sha256,
    stable_hash,
    tensor_mapping_hash,
    write_csv,
    write_json,
)
from tools.semantic_acquisition.metrics import classification_metrics
from tools.semantic_acquisition.runtime import (
    CifarRawStore,
    _load_cliplora_api,
    _predict_ids,
    _set_determinism,
    _train_client,
    build_experiment_cfg,
    flatten_named,
    load_lora_state,
    lora_state,
    trainable_named,
)


ROOT = Path(__file__).resolve().parents[2]


def _cfg(output_dir: Path):
    """Build the E1-equivalent FP32 vision-LoRA local-training config."""
    cfg = build_experiment_cfg(output_dir)
    cfg.defrost()
    cfg.DATASET.NAME = "Cifar100_LT"
    cfg.TRAINER.COOP.N_CTX = 4
    # federated_main's ClipLora path keeps COOP.CSC=False; the CLI --csc flag
    # belongs to PromptFL and does not alter this visual-LoRA baseline.
    cfg.TRAINER.COOP.CSC = False
    cfg.TRAINER.COOP.PREC = "fp32"
    cfg.OPTIM.LR = 0.001
    cfg.OPTIM.MAX_EPOCH = 3
    cfg.OPTIM.LR_SCHEDULER = "single_step"
    cfg.OPTIM.STEPSIZE = 3
    cfg.OPTIM.GAMMA = 1.0
    cfg.OPTIM.WARMUP_EPOCH = -1
    cfg.DATALOADER.TRAIN_X.BATCH_SIZE = 32
    cfg.DATALOADER.TEST.BATCH_SIZE = 64
    cfg.freeze()
    return cfg


def _repository_cifar_transform():
    # This exactly mirrors DatasetCifar100.__getitem__ in the formal E1 path.
    return T.Compose([
        T.ToTensor(),
        T.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),
        T.Resize([224, 224]),
    ])


def _verify_manifests(manifest_dir: Path) -> tuple[dict, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    manifest_dir = Path(manifest_dir)
    contract = json.loads((manifest_dir / "manifest_contract.json").read_text(encoding="utf-8"))
    if contract.get("protocol") != frozen_protocol():
        raise RuntimeError("E2 manifest protocol differs from the checked-in frozen protocol")
    for name, expected in contract.get("manifest_hashes", {}).items():
        observed = file_sha256(manifest_dir / name)
        if observed != expected:
            raise RuntimeError(f"E2 manifest hash mismatch for {name}: {observed} != {expected}")
    samples = pd.read_csv(manifest_dir / "partition_sample_manifest.csv")
    support = pd.read_csv(manifest_dir / "client_tail_support_manifest.csv")
    execution = pd.read_csv(manifest_dir / "local_execution_manifest.csv")
    return contract, samples, support, execution


def _prepare_model(args, store: CifarRawStore):
    if not torch.cuda.is_available() or str(args.device).lower() != "cuda":
        raise RuntimeError("Formal E2 local-update audit requires a CUDA compute node")
    cfg = _cfg(args.output_dir)
    _set_determinism(args.model_init_seed)
    build_cliplora_model, _, _ = _load_cliplora_api()
    model = build_cliplora_model(cfg, store.class_names).cuda()
    candidate = lora_state(model)
    theta_path = Path(args.theta0_file)
    theta_path.parent.mkdir(parents=True, exist_ok=True)
    if theta_path.exists():
        saved = torch.load(theta_path, map_location="cpu")
        if int(saved.get("model_seed", -1)) != int(args.model_init_seed):
            raise RuntimeError("E2 theta0 was created with a different model seed")
        theta0 = saved["lora_state"]
    else:
        theta0 = candidate
        torch.save({
            "model_seed": int(args.model_init_seed),
            "lora_state": theta0,
            "lora_hash": tensor_mapping_hash(theta0),
        }, theta_path)
    load_lora_state(model, theta0)
    if tensor_mapping_hash(lora_state(model)) != tensor_mapping_hash(theta0):
        raise RuntimeError("E2 theta0 did not load tensor-exactly")
    names = [name for name, _ in trainable_named(model)]
    if names != sorted(theta0):
        raise RuntimeError("E2 trainable LoRA keys differ from theta0 keys")
    return cfg, model, theta0, names


def _class_metrics(logits: torch.Tensor, labels: torch.Tensor, class_ids: Sequence[int]) -> dict[int, dict]:
    output = {}
    for class_id in class_ids:
        mask = labels == int(class_id)
        output[int(class_id)] = classification_metrics(logits[mask], labels[mask], int(class_id))
    return output


def _neighbor_metrics(logits: torch.Tensor, labels: torch.Tensor, neighbors) -> dict[int, dict]:
    rows = neighbor_discrimination_metrics(logits.numpy(), labels.numpy(), neighbors, TAIL_CLASSES)
    return {int(row["tail_class"]): row for row in rows}


def _tail_metric_rows(
    unit: Mapping,
    epoch: int,
    metrics: Mapping[int, Mapping],
    baseline: Mapping[int, Mapping],
    neighbor: Mapping[int, Mapping],
    baseline_neighbor: Mapping[int, Mapping],
    support_rows: pd.DataFrame,
) -> list[dict]:
    support_by_class = {int(row.tail_class): row for row in support_rows.itertuples()}
    output = []
    for class_id in TAIL_CLASSES:
        current, before = metrics[class_id], baseline[class_id]
        current_neighbor, before_neighbor = neighbor[class_id], baseline_neighbor[class_id]
        support = support_by_class[class_id]
        output.append({
            **unit, "local_epoch": int(epoch), "tail_class": class_id,
            "tail_sample_count": int(support.tail_sample_count),
            "supports_tail_class": int(support.supports_tail_class),
            "client_size": int(support.client_size),
            "client_tail_count": int(support.client_tail_count),
            "client_companion_count": int(support.client_companion_count),
            "tail_purity": float(support.tail_purity),
            "companion_class_count": int(support.companion_class_count),
            "tail_neighbor_access_score": float(support.tail_neighbor_access_score),
            "accuracy": float(current["accuracy"]),
            "margin": float(current["margin"]),
            "nll": float(current["nll"]),
            "accuracy_gain": float(current["accuracy"] - before["accuracy"]),
            "margin_gain": float(current["margin"] - before["margin"]),
            "nll_gain": float(before["nll"] - current["nll"]),
            "target_vs_neighbor_pairwise_margin": float(current_neighbor["target_vs_neighbor_pairwise_margin"]),
            "worst_neighbor_margin": float(current_neighbor["worst_neighbor_margin"]),
            "positive_margin_neighbor_coverage": float(current_neighbor["positive_margin_neighbor_coverage"]),
            "target_vs_neighbor_pairwise_margin_gain": float(
                current_neighbor["target_vs_neighbor_pairwise_margin"]
                - before_neighbor["target_vs_neighbor_pairwise_margin"]
            ),
            "worst_neighbor_margin_gain": float(
                current_neighbor["worst_neighbor_margin"] - before_neighbor["worst_neighbor_margin"]
            ),
            "positive_margin_neighbor_coverage_gain": float(
                current_neighbor["positive_margin_neighbor_coverage"]
                - before_neighbor["positive_margin_neighbor_coverage"]
            ),
        })
    return output


def _all_class_rows(
    unit: Mapping,
    metrics: Mapping[int, Mapping],
    baseline: Mapping[int, Mapping],
    class_counts: np.ndarray,
) -> list[dict]:
    rows = []
    for class_id in range(100):
        current, before = metrics[class_id], baseline[class_id]
        rows.append({
            **unit, "local_epoch": 3, "class_id": class_id,
            "class_group": "tail" if class_id in TAIL_CLASSES else "non_tail",
            "local_sample_count": int(class_counts[class_id]),
            "locally_supported": int(class_counts[class_id] > 0),
            "accuracy_before": float(before["accuracy"]),
            "accuracy_after": float(current["accuracy"]),
            "accuracy_gain": float(current["accuracy"] - before["accuracy"]),
            "margin_before": float(before["margin"]),
            "margin_after": float(current["margin"]),
            "margin_gain": float(current["margin"] - before["margin"]),
            "nll_before": float(before["nll"]),
            "nll_after": float(current["nll"]),
            "nll_gain": float(before["nll"] - current["nll"]),
        })
    return rows


def run(args) -> dict:
    contract, samples, support, execution = _verify_manifests(args.manifest_dir)
    samples = samples[samples.stage == args.stage].copy()
    support = support[support.stage == args.stage].copy()
    execution = execution[execution.stage == args.stage].copy()
    if samples.empty or support.empty or execution.empty:
        raise RuntimeError(f"No manifested units exist for stage={args.stage}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    store = CifarRawStore(args.data_dir)
    cfg, model, theta0, trainable_names = _prepare_model(args, store)
    transform = _repository_cifar_transform()
    neighbors, neighbor_metadata = load_preregistered_neighbors(TAIL_CLASSES)

    all_test_ids = [f"test:{index}" for index in range(len(store.test_labels))]
    all_test_labels = store.test_labels.tolist()
    baseline_logits, baseline_labels = _predict_ids(
        model, store, transform, all_test_ids, all_test_labels, batch_size=args.eval_batch_size
    )
    baseline_all = _class_metrics(baseline_logits, baseline_labels, range(100))
    tail_mask = torch.as_tensor(np.isin(store.test_labels, TAIL_CLASSES), dtype=torch.bool)
    tail_ids = [all_test_ids[index] for index in np.flatnonzero(np.isin(store.test_labels, TAIL_CLASSES))]
    tail_labels_list = store.test_labels[np.isin(store.test_labels, TAIL_CLASSES)].tolist()
    baseline_tail_logits = baseline_logits[tail_mask]
    baseline_tail_labels = baseline_labels[tail_mask]
    baseline_tail = _class_metrics(baseline_tail_logits, baseline_tail_labels, TAIL_CLASSES)
    baseline_neighbor = _neighbor_metrics(baseline_tail_logits, baseline_tail_labels, neighbors)

    unit_columns = ["data_seed", "topology", "condition", "client_id"]
    units = support[unit_columns].drop_duplicates().sort_values(unit_columns)
    tail_rows, all_rows, summary_rows, fairness_rows, difficulty_rows = [], [], [], [], []
    update_arrays = {}
    theta_vector = flatten_named(theta0, trainable_names)

    for row in units.itertuples(index=False):
        unit = {
            "stage": args.stage, "data_seed": int(row.data_seed),
            "topology": str(row.topology), "condition": str(row.condition),
            "client_id": int(row.client_id),
        }
        unit_filter = (
            (execution.data_seed == unit["data_seed"])
            & (execution.topology == unit["topology"])
            & (execution.condition == unit["condition"])
            & (execution.client_id == unit["client_id"])
        )
        unit_execution = execution[unit_filter].copy()
        unit_support = support[
            (support.data_seed == unit["data_seed"])
            & (support.topology == unit["topology"])
            & (support.condition == unit["condition"])
            & (support.client_id == unit["client_id"])
        ].copy()
        unit_samples = samples[
            (samples.data_seed == unit["data_seed"])
            & (samples.topology == unit["topology"])
            & (samples.condition == unit["condition"])
            & (samples.client_id == unit["client_id"])
        ]
        if len(unit_support) != len(TAIL_CLASSES):
            raise RuntimeError(f"Unit lacks 20 tail support rows: {unit}")
        companion_samples = unit_samples[~unit_samples.label.isin(TAIL_CLASSES)].sort_values("base_sample_id")
        companion_nll_values = []
        if not companion_samples.empty:
            load_lora_state(model, theta0)
            companion_logits, companion_labels = _predict_ids(
                model, store, transform,
                companion_samples.base_sample_id.tolist(),
                companion_samples.label.astype(int).tolist(),
                batch_size=args.eval_batch_size,
            )
            per_sample_nll = F.cross_entropy(companion_logits, companion_labels, reduction="none")
            probabilities = companion_logits.softmax(dim=1)
            correct_confidence = probabilities[
                torch.arange(companion_labels.numel()), companion_labels
            ]
            companion_nll_values = per_sample_nll.numpy().tolist()
            for sample, nll, confidence in zip(
                companion_samples.itertuples(), companion_nll_values,
                correct_confidence.numpy().tolist(),
            ):
                difficulty_rows.append({
                    **unit, "base_sample_id": str(sample.base_sample_id),
                    "label": int(sample.label), "theta0_nll": float(nll),
                    "theta0_correct_class_confidence": float(confidence),
                })
        states, train_meta = _train_client(model, cfg, theta0, unit_execution, store, transform)

        tail_rows.extend(_tail_metric_rows(
            unit, 0, baseline_tail, baseline_tail, baseline_neighbor, baseline_neighbor, unit_support
        ))
        final_all_metrics = None
        for epoch in (1, 2, 3):
            load_lora_state(model, states[epoch])
            if epoch < 3:
                logits, labels = _predict_ids(
                    model, store, transform, tail_ids, tail_labels_list,
                    batch_size=args.eval_batch_size,
                )
                metrics = _class_metrics(logits, labels, TAIL_CLASSES)
                neighbor_metrics = _neighbor_metrics(logits, labels, neighbors)
            else:
                logits_all, labels_all = _predict_ids(
                    model, store, transform, all_test_ids, all_test_labels,
                    batch_size=args.eval_batch_size,
                )
                final_all_metrics = _class_metrics(logits_all, labels_all, range(100))
                logits, labels = logits_all[tail_mask], labels_all[tail_mask]
                metrics = {class_id: final_all_metrics[class_id] for class_id in TAIL_CLASSES}
                neighbor_metrics = _neighbor_metrics(logits, labels, neighbors)
            tail_rows.extend(_tail_metric_rows(
                unit, epoch, metrics, baseline_tail, neighbor_metrics, baseline_neighbor, unit_support
            ))

        assert final_all_metrics is not None
        class_counts = np.bincount(unit_samples.label.to_numpy(dtype=np.int64), minlength=100)
        all_rows.extend(_all_class_rows(unit, final_all_metrics, baseline_all, class_counts))
        final_state_vector = flatten_named(states[3], trainable_names)
        update = (final_state_vector - theta_vector).numpy()
        update_key = (
            f"seed{unit['data_seed']}__{unit['stage']}__{unit['topology']}__"
            f"{unit['condition']}__client{unit['client_id']}"
        )
        update_arrays[update_key] = update
        own = [c for c in TAIL_CLASSES if class_counts[c] > 0]
        unseen = [c for c in TAIL_CLASSES if class_counts[c] == 0]
        final_tail_unit = [x for x in tail_rows if all(
            x[key] == value for key, value in unit.items()
        ) and x["local_epoch"] == 3]
        by_tail = {int(x["tail_class"]): x for x in final_tail_unit}
        positive_class_fraction = float(np.mean([
            final_all_metrics[c]["margin"] - baseline_all[c]["margin"] > 0 for c in range(100)
        ]))
        baseline_accuracy = float(np.mean([baseline_all[c]["accuracy"] for c in range(100)]))
        final_accuracy = float(np.mean([final_all_metrics[c]["accuracy"] for c in range(100)]))
        summary_rows.append({
            **unit, "client_size": int(class_counts.sum()),
            "client_tail_count": int(class_counts[TAIL_CLASSES].sum()),
            "supported_tail_class_count": len(own),
            "own_tail_margin_gain": float(np.mean([by_tail[c]["margin_gain"] for c in own])),
            "own_tail_accuracy_gain": float(np.mean([by_tail[c]["accuracy_gain"] for c in own])),
            "unseen_tail_margin_gain": float(np.mean([by_tail[c]["margin_gain"] for c in unseen])) if unseen else 0.0,
            "unseen_tail_accuracy_gain": float(np.mean([by_tail[c]["accuracy_gain"] for c in unseen])) if unseen else 0.0,
            "own_tail_worst_neighbor_margin_gain": float(np.mean([by_tail[c]["worst_neighbor_margin_gain"] for c in own])),
            "all_class_positive_margin_gain_fraction": positive_class_fraction,
            "all_class_macro_accuracy_before": baseline_accuracy,
            "all_class_macro_accuracy_after": final_accuracy,
            "all_class_macro_accuracy_gain": final_accuracy - baseline_accuracy,
            "lora_update_l2": float(np.linalg.norm(update)),
            "companion_theta0_nll_mean": float(np.mean(companion_nll_values)) if companion_nll_values else 0.0,
            "companion_theta0_nll_std": float(np.std(companion_nll_values)) if companion_nll_values else 0.0,
        })
        fairness_rows.append({
            **unit, "theta0_hash": tensor_mapping_hash(theta0),
            "expected_optimizer_steps": int(train_meta["expected_optimizer_steps"]),
            "optimizer_steps_successful": int(train_meta["optimizer_steps_successful"]),
            "scheduler_steps": int(train_meta["scheduler_steps"]),
            "amp_overflow_count": int(train_meta["amp_overflow_count"]),
            "precision": str(train_meta["precision"]),
            "server_aggregation_called": False,
            "pass": bool(
                train_meta["expected_optimizer_steps"] == train_meta["optimizer_steps_successful"]
                and train_meta["scheduler_steps"] == 3
                and train_meta["amp_overflow_count"] == 0
            ),
        })
        print(json.dumps({**unit, "status": "complete", "optimizer_steps": train_meta["optimizer_steps_successful"]}))

    write_csv(args.output_dir / "local_tail_metrics.csv", tail_rows)
    write_csv(args.output_dir / "local_all_class_footprints.csv", all_rows)
    write_csv(args.output_dir / "local_client_summaries.csv", summary_rows)
    write_csv(args.output_dir / "runtime_fairness.csv", fairness_rows)
    write_csv(args.output_dir / "companion_initial_difficulty.csv", difficulty_rows)
    np.savez_compressed(args.output_dir / "local_lora_update_vectors.npz", **update_arrays)
    metadata = {
        "stage": args.stage,
        "protocol": frozen_protocol(),
        "manifest_contract_hash": stable_hash(contract),
        "theta0_file": str(Path(args.theta0_file).resolve()),
        "theta0_hash": tensor_mapping_hash(theta0),
        "baseline_logits_hash": stable_hash(baseline_logits.numpy().tobytes().hex()),
        "neighbor_metadata": neighbor_metadata,
        "trainable_keys": trainable_names,
        "trainable_parameter_count": int(sum(p.numel() for _, p in trainable_named(model))),
        "device": torch.cuda.get_device_name(torch.cuda.current_device()),
        "torch_version": torch.__version__, "python_version": platform.python_version(),
        "server_aggregation_called": False,
        "result_hashes": {
            name: file_sha256(args.output_dir / name) for name in (
                "local_tail_metrics.csv", "local_all_class_footprints.csv",
                "local_client_summaries.csv", "runtime_fairness.csv",
                "companion_initial_difficulty.csv", "local_lora_update_vectors.npz",
            )
        },
    }
    write_json(args.output_dir / "runtime_contract.json", metadata)
    return {
        "stage": args.stage, "completed_clients": len(units),
        "tail_metric_rows": len(tail_rows), "all_class_rows": len(all_rows),
        "output_dir": str(args.output_dir.resolve()),
        "valid_local_only_comparison": all(bool(row["pass"]) for row in fairness_rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["e2a", "e2b"], required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("DATA/cifar-100/cifar-100-python"))
    parser.add_argument("--manifest-dir", type=Path, default=Path("output/e2_client_update_audit/manifests"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--theta0-file", type=Path, default=Path("output/e1_strength_breadth/protocol_v2/theta0_seed42.pt"))
    parser.add_argument("--model-init-seed", type=int, default=42)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    try:
        print(json.dumps(run(args)))
    except Exception as exc:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        write_json(args.output_dir / "failure.json", {
            "stage": args.stage, "error_type": type(exc).__name__, "error": str(exc),
            "traceback": traceback.format_exc(),
        })
        raise


if __name__ == "__main__":
    main()
