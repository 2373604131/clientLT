"""Read-only functional-coverage logging for the dual-topology validation.

The diagnostic is intentionally small: at a few frozen communication rounds it
asks whether the *actual selected client updates* collectively improve more of
the tail class's currently difficult decision boundaries, and whether ordinary
FedAvg preserves those improvements.  Probe images come only from CIFAR-100's
training split and are excluded from the federated long-tail training pool.
"""

from __future__ import annotations

import csv
import json
import pickle
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch
from PIL import Image

from tools.semantic_acquisition.common import (
    deterministic_choice,
    stable_hash,
    tensor_mapping_hash,
)


SCHEMA_VERSION = "functional_coverage_validation_v1"


def parse_validation_rounds(value, total_rounds: int) -> list[int]:
    if isinstance(value, (list, tuple, set)):
        raw = value
    else:
        raw = [item.strip() for item in str(value).split(",") if item.strip()]
    rounds = sorted({int(item) for item in raw})
    if not rounds:
        raise ValueError("Functional-coverage validation requires at least one round")
    if rounds[0] < 1 or rounds[-1] > int(total_rounds):
        raise ValueError(
            f"Functional-coverage rounds must be within [1, {int(total_rounds)}]: {rounds}"
        )
    if int(total_rounds) not in rounds:
        raise ValueError("Functional-coverage rounds must include the final round")
    return rounds


def _locate_cifar100(root: Path) -> Path:
    root = Path(root)
    choices = (
        root,
        root / "cifar-100-python",
        root / "cifar-100" / "cifar-100-python",
    )
    for candidate in choices:
        if (candidate / "train").is_file() and (candidate / "meta").is_file():
            return candidate
    raise FileNotFoundError(f"Cannot locate CIFAR-100 train/meta; checked {list(choices)}")


def _append_rows(path: Path, rows: Sequence[Mapping]) -> None:
    rows = list(rows)
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    fieldnames = list(rows[0].keys())
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="raise")
        if not exists:
            writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def _lora_state_from_payload(path: Path) -> dict[str, torch.Tensor]:
    payload = torch.load(Path(path), map_location="cpu")
    if isinstance(payload, Mapping) and "lora_state" in payload:
        payload = payload["lora_state"]
    if not isinstance(payload, Mapping):
        raise TypeError(f"Frozen LoRA anchor is not a tensor mapping: {path}")
    state = {
        str(key): value.detach().cpu().clone()
        for key, value in payload.items()
        if isinstance(value, torch.Tensor) and "lora_" in str(key)
    }
    if not state:
        raise ValueError(f"Frozen LoRA anchor contains no LoRA tensors: {path}")
    return state


def load_common_lora_anchor(models, path: Path) -> tuple[dict[str, torch.Tensor], str]:
    """Load one frozen trainable initialization into both global/local models."""
    anchor = _lora_state_from_payload(path)
    for model in models:
        model_state = model.state_dict()
        model_keys = {key for key in model_state if "lora_" in key}
        if model_keys != set(anchor):
            missing = sorted(model_keys - set(anchor))
            extra = sorted(set(anchor) - model_keys)
            raise RuntimeError(
                "Frozen LoRA anchor does not match the current architecture: "
                f"missing={missing}, extra={extra}"
            )
        shape_mismatches = {
            key: {"model": tuple(model_state[key].shape), "anchor": tuple(anchor[key].shape)}
            for key in sorted(model_keys)
            if tuple(model_state[key].shape) != tuple(anchor[key].shape)
        }
        if shape_mismatches:
            raise RuntimeError(
                "Frozen LoRA anchor tensor shapes do not match the current architecture: "
                f"{shape_mismatches}"
            )
    for model in models:
        result = model.load_state_dict(anchor, strict=False)
        unexpected = [key for key in result.unexpected_keys if "lora_" in key]
        if unexpected:
            raise RuntimeError(f"Unexpected frozen LoRA keys: {unexpected}")
    return anchor, tensor_mapping_hash(anchor)


def current_common_lora_anchor(models) -> tuple[dict[str, torch.Tensor], str]:
    """Verify that independently built global/local models share one LoRA init."""
    states = []
    for model in models:
        state = {
            str(key): value.detach().cpu().clone()
            for key, value in model.state_dict().items()
            if "lora_" in str(key)
        }
        if not state:
            raise RuntimeError("Current ClipLora model exposes no LoRA tensors")
        states.append(state)
    hashes = [tensor_mapping_hash(state) for state in states]
    if len(set(hashes)) != 1:
        raise RuntimeError(
            "Deterministic common initialization failed: global/local LoRA hashes differ"
        )
    return states[0], hashes[0]


class _TrainOnlyCifar100:
    def __init__(self, data_dir: Path):
        with (Path(data_dir) / "train").open("rb") as handle:
            train = pickle.load(handle, encoding="latin1")
        values = np.asarray(train["data"], dtype=np.uint8).reshape(-1, 3, 32, 32)
        self.images = values.transpose(0, 2, 3, 1)
        self.labels = np.asarray(train["fine_labels"], dtype=np.int64)

    def image(self, raw_id: int) -> Image.Image:
        return Image.fromarray(self.images[int(raw_id)])


def _exact_lt_raw_ids(labels: np.ndarray, imbalance_factor: float, imbalance_type: str):
    # This is the exact, side-effect-free equivalent of datasets/long_tail.py's
    # CIFAR-100 exp branch (which otherwise resets process-global RNG states).
    if str(imbalance_type) != "exp":
        raise ValueError("Functional coverage currently supports imb_type=exp only")
    generator = np.random.RandomState(1)
    selected = []
    image_max = float(len(labels)) / 100.0
    for class_id in range(100):
        count = int(
            image_max
            * float(imbalance_factor) ** (float(class_id) / (100.0 - 1.0))
        )
        indices = np.flatnonzero(labels == class_id).copy()
        generator.shuffle(indices)
        selected.extend(int(value) for value in indices[:count].tolist())
    return np.asarray(selected, dtype=np.int64)


class FunctionalCoverageDiagnostic:
    """Measure available and FedAvg-realized functional boundary coverage."""

    def __init__(
        self,
        *,
        output_dir,
        data_root,
        model,
        initial_state,
        lora_keys: Sequence[str],
        anchor_hash: str,
        anchor_source: str,
        tail_classes: Sequence[int],
        selected_rounds: Sequence[int],
        samples_per_class: int,
        imbalance_factor: float,
        imbalance_type: str,
        seed: int,
        partition: str,
        gain_epsilon: float = 0.0,
        eval_batch_size: int = 100,
    ):
        self.root = Path(output_dir) / "functional_coverage"
        self.root.mkdir(parents=True, exist_ok=True)
        self.partition = str(partition)
        self.seed = int(seed)
        self.tail_classes = [int(value) for value in tail_classes]
        self.rounds = {int(value) for value in selected_rounds}
        self.samples_per_class = int(samples_per_class)
        self.gain_epsilon = float(gain_epsilon)
        self.eval_batch_size = int(eval_batch_size)
        self.lora_keys = sorted(str(value) for value in lora_keys)
        if self.samples_per_class < 1:
            raise ValueError("samples_per_class must be positive")
        if not self.tail_classes:
            raise ValueError("Functional coverage requires non-empty tail classes")

        data_dir = _locate_cifar100(Path(data_root))
        self.store = _TrainOnlyCifar100(data_dir)
        lt_ids = set(
            int(value)
            for value in _exact_lt_raw_ids(
                self.store.labels, imbalance_factor, imbalance_type
            ).tolist()
        )
        self.probe_ids = {}
        probe_rows = []
        for class_id in self.tail_classes:
            pool = [
                int(value)
                for value in np.flatnonzero(self.store.labels == class_id).tolist()
                if int(value) not in lt_ids
            ]
            chosen = deterministic_choice(
                pool,
                self.samples_per_class,
                "functional-coverage-validation",
                self.seed,
                class_id,
            )
            self.probe_ids[class_id] = chosen
            for slot, raw_id in enumerate(chosen):
                probe_rows.append(
                    {
                        "class_id": class_id,
                        "slot": slot,
                        "raw_train_index": raw_id,
                        "excluded_from_federated_lt_pool": 1,
                    }
                )
        _append_rows(self.root / "probe_manifest.csv", probe_rows)

        from torchvision import transforms as T

        self.transform = T.Compose(
            [
                T.ToTensor(),
                T.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),
                T.Resize([224, 224]),
            ]
        )
        self.flat_ids = [
            raw_id for class_id in self.tail_classes for raw_id in self.probe_ids[class_id]
        ]
        self.class_slices = {
            class_id: slice(
                position * self.samples_per_class,
                (position + 1) * self.samples_per_class,
            )
            for position, class_id in enumerate(self.tail_classes)
        }

        initial_logits = self._predict_state(model, initial_state)
        probabilities = initial_logits.softmax(dim=1)
        self.boundary_weights = {}
        weight_rows = []
        for class_id in self.tail_classes:
            mean_prob = probabilities[self.class_slices[class_id]].mean(dim=0)
            mean_prob[class_id] = 0.0
            denominator = float(mean_prob.sum().item())
            if denominator <= 0:
                weights = torch.ones_like(mean_prob)
                weights[class_id] = 0.0
                weights /= weights.sum()
            else:
                weights = mean_prob / denominator
            self.boundary_weights[class_id] = weights.cpu()
            for competitor in range(int(weights.numel())):
                if competitor != class_id:
                    weight_rows.append(
                        {
                            "class_id": class_id,
                            "competitor_class": competitor,
                            "confusion_weight": float(weights[competitor].item()),
                        }
                    )
        _append_rows(self.root / "frozen_boundary_weights.csv", weight_rows)

        protocol = {
            "schema_version": SCHEMA_VERSION,
            "claim": (
                "fixed-margin Client-LT narrows available and FedAvg-realized functional "
                "coverage, accompanying worse tail accuracy and retention"
            ),
            "partition": self.partition,
            "seed": self.seed,
            "selected_rounds": sorted(self.rounds),
            "tail_classes": self.tail_classes,
            "samples_per_tail_class": self.samples_per_class,
            "probe_source": "CIFAR-100 train samples excluded from the federated LT pool",
            "test_split_accessed_by_coverage_diagnostic": False,
            "boundary_bank": "all non-target classes weighted by frozen theta0 confusion",
            "available_coverage": "sum_h w_ch * 1[max_selected_client gain(c,h) > epsilon]",
            "realized_coverage": "sum_h w_ch * 1[FedAvg gain(c,h) > epsilon]",
            "selected_client_pool": "all selected clients, including class-absent donors",
            "gain_epsilon": self.gain_epsilon,
            "common_lora_anchor_sha256": anchor_hash,
            "common_lora_anchor_source": str(anchor_source),
            "lora_keys": self.lora_keys,
            "probe_manifest_hash": stable_hash(probe_rows),
            "boundary_weight_hash": stable_hash(weight_rows),
            "diagnostic_controls_training": False,
        }
        _write_json(self.root / "protocol.json", protocol)

    def is_selected(self, communication_round: int) -> bool:
        return int(communication_round) in self.rounds

    def _load_lora(self, model, state) -> None:
        subset = {key: state[key] for key in self.lora_keys}
        result = model.load_state_dict(subset, strict=False)
        unexpected = [key for key in result.unexpected_keys if "lora_" in key]
        if unexpected:
            raise RuntimeError(f"Unexpected LoRA keys while evaluating coverage: {unexpected}")

    @torch.no_grad()
    def _predict_state(self, model, state) -> torch.Tensor:
        self._load_lora(model, state)
        model.eval()
        device = next(model.parameters()).device
        chunks = []
        for start in range(0, len(self.flat_ids), self.eval_batch_size):
            tensors = [
                self.transform(self.store.image(raw_id))
                for raw_id in self.flat_ids[start : start + self.eval_batch_size]
            ]
            logits = model(torch.stack(tensors).to(device))
            chunks.append(logits.detach().float().cpu())
        return torch.cat(chunks, dim=0)

    def _boundary_gain(self, before: torch.Tensor, after: torch.Tensor, class_id: int):
        class_slice = self.class_slices[class_id]
        before_rows = before[class_slice]
        after_rows = after[class_slice]
        before_margin = before_rows[:, class_id : class_id + 1] - before_rows
        after_margin = after_rows[:, class_id : class_id + 1] - after_rows
        return (after_margin - before_margin).mean(dim=0)

    def record_round(
        self,
        *,
        model,
        communication_round: int,
        pre_state,
        local_states,
        post_state,
        selected_clients: Sequence[int],
    ) -> None:
        round_id = int(communication_round)
        if not self.is_selected(round_id):
            return
        selected = [int(value) for value in selected_clients]
        if not selected:
            raise ValueError("Functional coverage received an empty selected-client set")

        before = self._predict_state(model, pre_state)
        local_logits = {
            client_id: self._predict_state(model, local_states[client_id])
            for client_id in selected
        }
        after = self._predict_state(model, post_state)
        # Restore the real server state before returning to the training loop.
        self._load_lora(model, post_state)

        class_rows = []
        for class_id in self.tail_classes:
            individual = torch.stack(
                [
                    self._boundary_gain(before, local_logits[client_id], class_id)
                    for client_id in selected
                ],
                dim=0,
            )
            available_gain = individual.max(dim=0).values
            realized_gain = self._boundary_gain(before, after, class_id)
            weights = self.boundary_weights[class_id]
            competitor_mask = torch.ones(weights.numel(), dtype=torch.bool)
            competitor_mask[class_id] = False
            available_positive = available_gain > self.gain_epsilon
            realized_positive = realized_gain > self.gain_epsilon
            available_coverage = float(
                weights[competitor_mask][available_positive[competitor_mask]].sum().item()
            )
            realized_coverage = float(
                weights[competitor_mask][realized_positive[competitor_mask]].sum().item()
            )
            class_rows.append(
                {
                    "communication_round": round_id,
                    "partition": self.partition,
                    "class_id": class_id,
                    "selected_client_count": len(selected),
                    "available_functional_coverage": available_coverage,
                    "realized_functional_coverage": realized_coverage,
                    "coverage_retention_ratio": (
                        realized_coverage / available_coverage
                        if available_coverage > 0
                        else 0.0
                    ),
                    "available_positive_boundary_count": int(
                        available_positive[competitor_mask].sum().item()
                    ),
                    "realized_positive_boundary_count": int(
                        realized_positive[competitor_mask].sum().item()
                    ),
                }
            )
        _append_rows(self.root / "coverage_per_class_round.csv", class_rows)
        _append_rows(
            self.root / "coverage_round_summary.csv",
            [
                {
                    "communication_round": round_id,
                    "partition": self.partition,
                    "tail_class_count": len(class_rows),
                    "available_functional_coverage": float(
                        np.mean([row["available_functional_coverage"] for row in class_rows])
                    ),
                    "realized_functional_coverage": float(
                        np.mean([row["realized_functional_coverage"] for row in class_rows])
                    ),
                    "coverage_retention_ratio": float(
                        np.mean([row["coverage_retention_ratio"] for row in class_rows])
                    ),
                }
            ],
        )
        print(
            "Functional coverage round {}: available={:.6f} realized={:.6f}".format(
                round_id,
                np.mean([row["available_functional_coverage"] for row in class_rows]),
                np.mean([row["realized_functional_coverage"] for row in class_rows]),
            )
        )
