from __future__ import annotations

import csv
import hashlib
import json
import pickle
from pathlib import Path
from typing import Mapping

import numpy as np
import torch
from PIL import Image
from torchvision import transforms as T

from tools.breadth_audit.artifacts import append_breadth_artifacts
from tools.breadth_audit.evaluator import (
    evaluate_three_breadth_families,
    predict_fixed_views,
)
from tools.breadth_audit.inputs import (
    load_dino_clusters,
    load_preregistered_neighbors,
)
from tools.breadth_audit.protocol import TAIL_CLASSES, frozen_protocol
from tools.semantic_acquisition.common import file_sha256, tensor_mapping_hash, write_json


def _load_cifar_test(data_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load raw CIFAR test images without importing the pandas-heavy V2 runtime."""
    test_path = Path(data_dir) / "test"
    if not test_path.is_file():
        raise FileNotFoundError(f"raw CIFAR-100 test pickle is missing: {test_path}")
    with test_path.open("rb") as handle:
        payload = pickle.load(handle, encoding="latin1")
    images = np.asarray(payload["data"], dtype=np.uint8).reshape(-1, 3, 32, 32)
    images = images.transpose(0, 2, 3, 1)
    labels = np.asarray(payload["fine_labels"], dtype=np.int64)
    if images.shape != (10000, 32, 32, 3) or labels.shape != (10000,):
        raise RuntimeError(
            f"unexpected CIFAR-100 test shapes: images={images.shape}, labels={labels.shape}"
        )
    return images, labels


def _lora_state(model) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
        if "lora_" in name
    }


def _load_lora_state(model, state: Mapping[str, torch.Tensor]) -> None:
    expected = set(_lora_state(model))
    if set(state) != expected:
        raise RuntimeError(
            f"E1 LoRA keys differ: missing={sorted(expected - set(state))}, "
            f"extra={sorted(set(state) - expected)}"
        )
    result = model.load_state_dict(state, strict=False)
    if result.unexpected_keys:
        raise RuntimeError(f"unexpected E1 theta0 keys: {result.unexpected_keys}")


def _append_csv(path: Path, row: Mapping) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(row)
    exists = path.is_file()
    if exists:
        with path.open("r", encoding="utf-8", newline="") as handle:
            existing = next(csv.reader(handle), None)
        if existing != fields:
            raise RuntimeError(f"existing E1 CSV schema differs: {path}")
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def _array_hash(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.tobytes())
    return digest.hexdigest()


def _true_margins(logits: np.ndarray, labels: np.ndarray) -> np.ndarray:
    rows = np.arange(len(labels))
    true = logits[rows, labels]
    other = logits.copy()
    other[rows, labels] = -np.inf
    return true - other.max(axis=1)


def _cross_entropy(logits: np.ndarray, labels: np.ndarray) -> np.ndarray:
    maximum = logits.max(axis=1, keepdims=True)
    logsumexp = maximum[:, 0] + np.log(np.exp(logits - maximum).sum(axis=1))
    return logsumexp - logits[np.arange(len(labels)), labels]


def _strength_rows(logits: np.ndarray, labels: np.ndarray) -> list[dict]:
    predictions = logits.argmax(axis=1)
    margins = _true_margins(logits, labels)
    losses = _cross_entropy(logits, labels)
    rows = []
    for class_id in TAIL_CLASSES:
        mask = labels == class_id
        rows.append({
            "tail_class": class_id,
            "sample_count": int(mask.sum()),
            "tail_accuracy": float(np.mean(predictions[mask] == labels[mask])),
            "tail_margin": float(np.mean(margins[mask])),
            "tail_loss": float(np.mean(losses[mask])),
        })
    return rows


def _representation_rows(
    features: np.ndarray,
    reference: np.ndarray,
    labels: np.ndarray,
) -> list[dict]:
    if features.shape != reference.shape:
        raise RuntimeError("E1 representation reference shape changed")
    rows = []
    for class_id in TAIL_CLASSES:
        mask = labels == class_id
        current = features[mask]
        initial = reference[mask]
        sample_cosine = np.sum(current * initial, axis=1)
        current_centroid = current.mean(axis=0)
        initial_centroid = initial.mean(axis=0)
        current_centroid /= max(float(np.linalg.norm(current_centroid)), 1e-12)
        initial_centroid /= max(float(np.linalg.norm(initial_centroid)), 1e-12)
        centroid_cosine = float(np.dot(current_centroid, initial_centroid))
        rows.append({
            "tail_class": class_id,
            "sample_count": int(mask.sum()),
            "mean_sample_cosine_drift_from_round0": float(np.mean(1.0 - sample_cosine)),
            "centroid_cosine_drift_from_round0": float(1.0 - centroid_cosine),
        })
    return rows


def _repository_test_transform():
    """Match DatasetCifar100's actual deterministic evaluation preprocessing."""
    return T.Compose([
        T.ToTensor(),
        T.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),
        T.Resize([224, 224]),
    ])


def _topology_name(partition: str) -> str:
    mapping = {
        "noniid-labeldir-fine": "dirichlet",
        "client-longtail-controlled": "clientlt_controlled",
    }
    if partition not in mapping:
        raise ValueError(f"E1 does not accept partition={partition}")
    return mapping[partition]


def _validate_protocol_and_args(args, cfg) -> tuple[dict, str]:
    protocol_path = Path(args.e1_protocol_file)
    if not protocol_path.is_file():
        raise FileNotFoundError(f"E1 protocol is missing: {protocol_path}")
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    expected = frozen_protocol()
    if protocol != expected:
        raise RuntimeError(
            "E1 protocol differs from the current V2 contract. Generate a new "
            "protocol_v2 directory before training; do not reuse the superseded V1 file."
        )
    topology = _topology_name(str(args.partition))
    checks = {
        "trainer": (str(args.trainer), "ClipLora"),
        "model": (str(args.model), "fedavg"),
        "seed": (int(args.seed), 42),
        "split_seed": (int(args.split_seed), 42),
        "num_users": (int(args.num_users), 30),
        "rounds": (int(args.round), 100),
        "local_epochs": (int(cfg.OPTIM.MAX_EPOCH), 3),
        "train_batch_size": (int(cfg.DATALOADER.TRAIN_X.BATCH_SIZE), 32),
        "test_batch_size": (int(cfg.DATALOADER.TEST.BATCH_SIZE), 64),
        "precision": (str(args.cliplora_precision), "fp32"),
        "aggregation": (str(args.cliplora_aggregation), "fedavg"),
        "encoder": (str(args.encoder), "vision"),
        "rank": (int(args.cliplora_rank), 2),
        "alpha": (int(args.cliplora_alpha), 1),
        "position": (str(args.cliplora_position), "top3"),
        "dropout": (float(args.cliplora_dropout_rate), 0.0),
        "lr_policy": (str(args.cliplora_lr_policy), "constant"),
        "global_eval_interval": (int(args.global_eval_interval), 1),
        "optimizer": (str(cfg.OPTIM.NAME).lower(), "sgd"),
        "weight_decay": (float(cfg.OPTIM.WEIGHT_DECAY), 0.0005),
        "momentum": (float(cfg.OPTIM.MOMENTUM), 0.9),
        "sgd_dampening": (float(cfg.OPTIM.SGD_DAMPNING), 0.0),
        "sgd_nesterov": (bool(cfg.OPTIM.SGD_NESTEROV), False),
    }
    for name, (observed, wanted) in checks.items():
        if observed != wanted:
            raise ValueError(f"E1 frozen {name} differs: {observed!r} != {wanted!r}")
    if abs(float(args.frac) - 1.0) > 1e-12 or abs(float(args.lr) - 0.001) > 1e-12:
        raise ValueError("E1 requires frac=1.0 and lr=0.001")
    if list(args.cliplora_params) != ["q", "v"]:
        raise ValueError("E1 requires ClipLora parameters q and v")
    if not bool(args.isolate_local_optimizer_state) or not bool(args.federated_single_scheduler_step):
        raise ValueError("E1 requires isolated local optimizers and one scheduler step per local epoch")
    if topology == "dirichlet" and abs(float(args.beta) - 0.5) > 1e-12:
        raise ValueError("E1 Dirichlet requires beta=0.5")
    if topology == "clientlt_controlled":
        if abs(float(args.intra_group_alpha) - 0.5) > 1e-12:
            raise ValueError("E1 controlled Client-LT requires intra_group_alpha=0.5")
        if abs(float(args.controlled_tail_min_purity) - 0.8) > 1e-12:
            raise ValueError("E1 controlled Client-LT requires min purity=0.8")
    return protocol, topology


def _validate_realized_partition(args, topology: str) -> dict:
    path = Path(args.output_dir) / "partition_summary.json"
    if not path.is_file():
        raise FileNotFoundError(f"E1 partition summary was not written: {path}")
    summary = json.loads(path.read_text(encoding="utf-8"))
    counts = [int(value) for value in summary.get("global_class_counts", [])]
    if len(counts) != 100 or counts[80:] != [12, 11, 11, 10, 10, 9, 9, 8, 8, 7, 7, 7, 6, 6, 6, 6, 5, 5, 5, 5]:
        raise RuntimeError("E1 global CIFAR-100-LT marginal or index-defined tail changed")
    if sum(counts[80:]) != 153:
        raise RuntimeError("E1 tail sample count is not 153")
    if topology == "clientlt_controlled":
        if int(round(float(summary["tail_samples_in_tail_clients"]))) != 153:
            raise RuntimeError("controlled Client-LT did not place all 153 tail samples in tail clients")
        if int(round(float(summary["tail_samples_in_head_clients"]))) != 0:
            raise RuntimeError("controlled Client-LT leaked tail samples to ordinary clients")
        companions = int(round(float(summary["non_tail_samples_in_tail_clients"])))
        if companions > 38:
            raise RuntimeError(f"controlled Client-LT companion budget exceeded 38: {companions}")
        purities = [float(value) for value in summary["per_tail_client_purity"].values()]
        if len(purities) != 3 or min(purities) + 1e-12 < 0.8:
            raise RuntimeError(f"controlled Client-LT per-client purity failed: {purities}")
    return summary


class E1RoundEvaluator:
    def __init__(self, args, cfg, model, *, protocol: dict, topology: str, theta0_hash: str):
        self.args = args
        self.cfg = cfg
        self.output_dir = Path(args.output_dir)
        self.topology = topology
        self.protocol = protocol
        self.theta0_hash = theta0_hash
        self.batch_size = int(args.e1_eval_batch_size)
        arrays, dino_meta = load_dino_clusters(Path(args.e1_dino_artifact))
        observed_dino_hash = file_sha256(Path(args.e1_dino_artifact))
        expected_dino_hash = protocol["breadth_audit"]["visual_subgroups"]["artifact_sha256"]
        if observed_dino_hash != expected_dino_hash:
            raise RuntimeError(
                f"E1 DINO artifact differs from the frozen seed-42 artifact: "
                f"{observed_dino_hash} != {expected_dino_hash}"
            )
        self.labels = arrays["labels"].astype(np.int64)
        self.cluster_ids = arrays["cluster_ids"].astype(np.int64)
        raw_test_images, raw_test_labels = _load_cifar_test(Path(args.e1_data_dir))
        if not np.array_equal(
            raw_test_labels[arrays["raw_test_indices"].astype(np.int64)],
            self.labels,
        ):
            raise RuntimeError("DINO artifact raw test indices do not match the CIFAR labels")
        self.images = [
            Image.fromarray(raw_test_images[int(index)])
            for index in arrays["raw_test_indices"]
        ]
        self.neighbors, neighbor_meta = load_preregistered_neighbors(TAIL_CLASSES)
        expected_neighbor_hash = protocol["breadth_audit"]["neighbor_discrimination"]["neighbors_hash"]
        if neighbor_meta["neighbors_hash"] != expected_neighbor_hash:
            raise RuntimeError("E1 semantic-neighbor hash differs from the frozen protocol")
        self.transform = _repository_test_transform()
        self.reference_features = None
        self.evaluated_rounds: set[int] = set()
        contract = {
            "protocol_name": protocol["protocol_name"],
            "protocol_hash": protocol["protocol_hash"],
            "seed": int(args.seed),
            "topology": topology,
            "partition": str(args.partition),
            "theta0_hash": theta0_hash,
            "dino_artifact": str(Path(args.e1_dino_artifact).resolve()),
            "dino_artifact_sha256": file_sha256(Path(args.e1_dino_artifact)),
            "dino_metadata": dino_meta,
            "neighbor_metadata": neighbor_meta,
            "test_preprocessing": "repository_DatasetCifar100_deterministic_cifar_norm_then_resize224",
        }
        write_json(self.output_dir / "e1_contract.json", contract)

    def _save_checkpoint(self, model, round_id: int) -> tuple[Path, str]:
        state = _lora_state(model)
        state_hash = tensor_mapping_hash(state)
        path = self.output_dir / "e1_checkpoints" / f"round_{round_id:03d}.pt"
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            saved = torch.load(path, map_location="cpu")
            if int(saved["round"]) != round_id or tensor_mapping_hash(saved["lora_state"]) != state_hash:
                raise RuntimeError(f"refusing to overwrite a different E1 checkpoint: {path}")
        else:
            torch.save({
                "round": round_id,
                "seed": int(self.args.seed),
                "topology": self.topology,
                "theta0_hash": self.theta0_hash,
                "lora_state": state,
            }, path)
        return path, state_hash

    def evaluate(self, model, round_id: int) -> None:
        round_id = int(round_id)
        if round_id in self.evaluated_rounds:
            raise RuntimeError(f"E1 round {round_id} was evaluated twice")
        checkpoint_path, checkpoint_hash = self._save_checkpoint(model, round_id)
        logits_by_view, features = predict_fixed_views(
            model,
            self.images,
            self.transform,
            batch_size=self.batch_size,
            return_clean_features=True,
        )
        if self.reference_features is None:
            if round_id != 0:
                raise RuntimeError("E1 representation drift requires round 0 first")
            self.reference_features = features.copy()
        breadth = evaluate_three_breadth_families(
            logits_by_view=logits_by_view,
            labels=self.labels,
            cluster_ids=self.cluster_ids,
            neighbors_by_tail=self.neighbors,
            tail_classes=TAIL_CLASSES,
        )
        metadata = {
            "seed": int(self.args.seed),
            "topology": self.topology,
            "round": round_id,
            "theta0_hash": self.theta0_hash,
        }
        append_breadth_artifacts(self.output_dir, breadth, run_metadata=metadata)
        for row in _strength_rows(logits_by_view["clean"], self.labels):
            _append_csv(self.output_dir / "tail_strength.csv", {**metadata, **row})
        for row in _representation_rows(features, self.reference_features, self.labels):
            _append_csv(self.output_dir / "representation_drift.csv", {**metadata, **row})
        _append_csv(self.output_dir / "e1_round_manifest.csv", {
            **metadata,
            "checkpoint_path": str(checkpoint_path),
            "checkpoint_lora_hash": checkpoint_hash,
            "clean_logits_hash": _array_hash(logits_by_view["clean"]),
            "clean_features_hash": _array_hash(features),
        })
        self.evaluated_rounds.add(round_id)
        print(json.dumps({
            "stage": "E1",
            "seed": int(self.args.seed),
            "topology": self.topology,
            "evaluated_round": round_id,
        }))

    def record_optimizer_steps(
        self,
        *,
        round_id: int,
        client_id: int,
        client_samples: int,
        optimizer_steps: int,
        scheduler_steps: int,
    ) -> None:
        _append_csv(self.output_dir / "e1_optimizer_steps.csv", {
            "seed": int(self.args.seed),
            "topology": self.topology,
            "round": int(round_id),
            "client_id": int(client_id),
            "client_samples": int(client_samples),
            "optimizer_steps": int(optimizer_steps),
            "scheduler_steps": int(scheduler_steps),
        })


def prepare_e1_run(args, cfg, global_model, local_model) -> E1RoundEvaluator:
    protocol, topology = _validate_protocol_and_args(args, cfg)
    partition_summary = _validate_realized_partition(args, topology)
    theta_path = Path(args.e1_theta0_file)
    theta_path.parent.mkdir(parents=True, exist_ok=True)
    candidate = _lora_state(global_model)
    if theta_path.exists():
        saved = torch.load(theta_path, map_location="cpu")
        theta = saved["lora_state"]
        if int(saved.get("model_seed", -1)) != int(args.e1_model_seed):
            raise RuntimeError("shared E1 theta0 was created with a different model seed")
    else:
        theta = candidate
        torch.save({
            "model_seed": int(args.e1_model_seed),
            "lora_state": theta,
            "lora_hash": tensor_mapping_hash(theta),
        }, theta_path)
    _load_lora_state(global_model, theta)
    _load_lora_state(local_model, theta)
    observed_global = tensor_mapping_hash(_lora_state(global_model))
    observed_local = tensor_mapping_hash(_lora_state(local_model))
    expected = tensor_mapping_hash(theta)
    if observed_global != expected or observed_local != expected:
        raise RuntimeError("E1 theta0 was not loaded exactly into both trainers")
    evaluator = E1RoundEvaluator(
        args,
        cfg,
        global_model,
        protocol=protocol,
        topology=topology,
        theta0_hash=expected,
    )
    write_json(Path(args.output_dir) / "e1_partition_gate.json", {
        "pass": True,
        "topology": topology,
        "global_lt_fingerprint": partition_summary["global_lt_fingerprint"],
        "tail_samples": int(sum(int(value) for value in partition_summary["global_class_counts"][80:])),
        "tail_samples_in_tail_clients": partition_summary["tail_samples_in_tail_clients"],
        "tail_samples_in_head_clients": partition_summary["tail_samples_in_head_clients"],
        "companion_samples_in_tail_clients": partition_summary["non_tail_samples_in_tail_clients"],
    })
    return evaluator
