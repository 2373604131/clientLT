"""Shared, frozen utilities for the D2/D3 ClipLoRA diagnostics."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from utils.cusp_minimal import FlatSpec, flatten_state, unflatten_state


D23_SCHEMA_VERSION = "d23_diagnostic_v1"


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: str | Path, payload: Mapping) -> None:
    def safe(value):
        if isinstance(value, Mapping):
            return {str(key): safe(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [safe(item) for item in value]
        if isinstance(value, np.generic):
            return safe(value.item())
        if isinstance(value, torch.Tensor):
            return safe(value.detach().cpu().tolist())
        if isinstance(value, float) and not math.isfinite(value):
            return None
        return value

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(safe(payload), indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )


def write_csv(path: str | Path, rows: Sequence[Mapping], fields: Sequence[str] | None = None) -> None:
    rows = list(rows)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = []
        for row in rows:
            for key in row:
                if key not in fields:
                    fields.append(key)
    with target.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields or ["status"]), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def load_dump(dump_dir: str | Path) -> tuple[dict, dict]:
    directory = Path(dump_dir)
    state_path = directory / "round_state.pt"
    metadata_path = directory / "metadata.json"
    if not state_path.is_file() or not metadata_path.is_file():
        raise FileNotFoundError(f"Incomplete D2/D3 dump: {directory}")
    payload = torch.load(state_path, map_location="cpu", weights_only=False)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if bool(metadata.get("validation_used_before_dump", False)) or bool(
        metadata.get("test_used_before_dump", False)
    ):
        raise RuntimeError("D2/D3 requires a dump created before validation/test access")
    return payload, metadata


def validate_dump(payload: Mapping, metadata: Mapping) -> None:
    required = {
        "flatten_spec",
        "global_before_trainable",
        "global_after_fedavg_trainable",
        "local_trainable_states",
        "selected_client_ids",
        "fedavg_weights",
        "client_class_counts",
        "global_class_counts",
    }
    missing = sorted(required.difference(payload))
    if missing:
        raise KeyError(f"D2/D3 dump is missing fields: {missing}")
    selected = list(payload["selected_client_ids"])
    if len(selected) != 30 or sorted(int(value) for value in selected) != list(range(30)):
        raise ValueError("D2/D3 is frozen to full participation by all 30 clients")
    args = metadata.get("resolved_args", {})
    if str(args.get("partition")) != "client-longtail":
        raise ValueError("D2/D3 is frozen to the normal client-longtail partition")
    if int(args.get("seed", -1)) != 42 or int(args.get("split_seed", -1)) != 42:
        raise ValueError("D2/D3 is frozen to seed=split_seed=42")
    config = {
        "position": str(args.get("cliplora_position")),
        "rank": int(args.get("cliplora_rank", -1)),
        "alpha": int(args.get("cliplora_alpha", -1)),
        "params": list(args.get("cliplora_params", [])),
    }
    expected = {"position": "up", "rank": 4, "alpha": 2, "params": ["q", "k", "v"]}
    if config != expected:
        raise ValueError(f"D2/D3 dump does not use the frozen LoRA config: {config}")


def class_split(global_class_counts, tail_ratio: float = 0.2) -> tuple[list[int], list[int]]:
    counts = torch.as_tensor(global_class_counts, dtype=torch.float64).reshape(-1)
    tail_count = max(1, int(round(len(counts) * float(tail_ratio))))
    tail = sorted(
        range(len(counts)), key=lambda class_id: (float(counts[class_id]), -class_id)
    )[:tail_count]
    tail_set = set(tail)
    head = sorted(
        (class_id for class_id in range(len(counts)) if class_id not in tail_set),
        key=lambda class_id: (-float(counts[class_id]), class_id),
    )
    return head, tail


def build_trainer(metadata: Mapping, output_dir: str | Path, eval_batch_size: int = 256):
    from Dassl.dassl.engine import build_trainer
    from federated_main import setup_cfg

    args = SimpleNamespace(**metadata["resolved_args"])
    args.output_dir = str(output_dir)
    args.test_batch_size = int(eval_batch_size)
    # Offline analysis must not recursively activate training-time diagnostics.
    args.v0_dump_enable = False
    args.experimentD_enable = False
    args.g0_probe_enable = False
    cfg = setup_cfg(args)
    trainer = build_trainer(cfg)
    trainer.fed_before_train(is_global=True)
    return cfg, trainer


def compact_state_from_vector(vector: torch.Tensor, spec: FlatSpec) -> dict[str, torch.Tensor]:
    return unflatten_state(torch.as_tensor(vector, dtype=torch.float64), spec)


def load_compact_state(model, state: Mapping[str, torch.Tensor]) -> None:
    incompatible = model.load_state_dict(state, strict=False)
    unexpected = list(getattr(incompatible, "unexpected_keys", []))
    if unexpected:
        raise RuntimeError(f"Unexpected trainable state keys: {unexpected}")


@torch.no_grad()
def collect_logits(trainer, state: Mapping[str, torch.Tensor], loader=None) -> tuple[torch.Tensor, torch.Tensor]:
    model = trainer.model
    load_compact_state(model, state)
    loader = loader if loader is not None else trainer.test_loader
    was_training = model.training
    logits, labels = [], []
    try:
        model.eval()
        for batch in loader:
            images, target = trainer.parse_batch_test(batch)
            output = trainer.model_inference(images)
            if isinstance(output, (tuple, list)):
                output = output[0]
            logits.append(output.detach().float().cpu())
            labels.append(target.detach().long().cpu())
    finally:
        model.train(was_training)
    if not logits:
        raise RuntimeError("D2/D3 encountered an empty evaluation loader")
    return torch.cat(logits), torch.cat(labels)


@torch.no_grad()
def collect_features_and_logits(
    trainer,
    state: Mapping[str, torch.Tensor],
    loader,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    model = trainer.model
    load_compact_state(model, state)
    was_training = model.training
    features, logits, labels = [], [], []
    try:
        model.eval()
        prompts = model.prompt_learner()
        text_features = model.text_encoder(prompts, model.tokenized_prompts)
        text_features = F.normalize(text_features.float(), dim=-1)
        scale = model.logit_scale.detach().float().exp()
        for batch in loader:
            images, target = trainer.parse_batch_test(batch)
            image_features = model.image_encoder(images.type(model.dtype))
            image_features = F.normalize(image_features.float(), dim=-1)
            output = scale * image_features @ text_features.t()
            features.append(image_features.detach().cpu())
            logits.append(output.detach().cpu())
            labels.append(target.detach().long().cpu())
    finally:
        model.train(was_training)
    if not features:
        raise RuntimeError("D3 encountered an empty feature loader")
    return torch.cat(features), torch.cat(logits), torch.cat(labels)


def build_global_train_eval_loader(cfg, trainer):
    from Dassl.dassl.data.data_manager import build_data_loader
    from Dassl.dassl.data.transforms import build_transform

    source = list(getattr(trainer.dm.dataset, "train_x", []) or [])
    if not source:
        raise RuntimeError("D3 could not find the global federated training source")
    return build_data_loader(
        cfg,
        sampler_type="SequentialSampler",
        data_source=source,
        batch_size=cfg.DATALOADER.TEST.BATCH_SIZE,
        tfm=build_transform(cfg, is_train=False),
        is_train=False,
        dataset_wrapper=None,
        class_names=trainer.dm.dataset.classnames,
        drop_last=False,
    )


def per_class_metrics(logits: torch.Tensor, labels: torch.Tensor) -> list[dict]:
    logits = torch.as_tensor(logits).float().cpu()
    labels = torch.as_tensor(labels).long().cpu()
    losses = F.cross_entropy(logits, labels, reduction="none")
    predictions = logits.argmax(dim=1)
    rows = []
    for class_id in range(logits.shape[1]):
        mask = labels == class_id
        count = int(mask.sum().item())
        if count == 0:
            rows.append({"class_id": class_id, "count": 0, "accuracy": math.nan, "margin": math.nan, "nll": math.nan})
            continue
        selected = logits[mask]
        target = labels[mask]
        true_logits = selected.gather(1, target[:, None]).squeeze(1)
        other = selected.clone()
        other.scatter_(1, target[:, None], -torch.inf)
        rows.append({
            "class_id": class_id,
            "count": count,
            "accuracy": 100.0 * float((predictions[mask] == target).float().mean().item()),
            "margin": float((true_logits - other.max(dim=1).values).mean().item()),
            "nll": float(losses[mask].mean().item()),
        })
    return rows


def aggregate_metrics(logits: torch.Tensor, labels: torch.Tensor, head: Sequence[int], tail: Sequence[int]) -> dict:
    rows = per_class_metrics(logits, labels)

    def mean(ids, field):
        values = [float(rows[int(class_id)][field]) for class_id in ids]
        values = [value for value in values if math.isfinite(value)]
        return float(sum(values) / len(values)) if values else math.nan

    head_acc = mean(head, "accuracy")
    tail_acc = mean(tail, "accuracy")
    harmonic = (
        2.0 * head_acc * tail_acc / (head_acc + tail_acc)
        if head_acc > 0.0 and tail_acc > 0.0 else 0.0
    )
    return {
        "overall_accuracy": 100.0 * float((logits.argmax(1) == labels).float().mean().item()),
        "balanced_accuracy": mean(range(len(rows)), "accuracy"),
        "head_accuracy": head_acc,
        "tail_accuracy": tail_acc,
        "head_tail_harmonic": harmonic,
        "head_margin": mean(head, "margin"),
        "tail_margin": mean(tail, "margin"),
        "head_nll": mean(head, "nll"),
        "tail_nll": mean(tail, "nll"),
    }


def stratified_fit_calibration_split(labels: torch.Tensor, seed: int = 42, calibration_fraction: float = 0.2):
    labels = torch.as_tensor(labels).long().cpu()
    rng = np.random.default_rng(int(seed))
    fit, calibration = [], []
    for class_id in sorted(int(value) for value in labels.unique().tolist()):
        indices = torch.nonzero(labels == class_id, as_tuple=False).reshape(-1).tolist()
        order = rng.permutation(indices).tolist()
        count = max(1, min(len(order) - 1, int(round(len(order) * float(calibration_fraction)))))
        calibration.extend(order[:count])
        fit.extend(order[count:])
    return torch.tensor(sorted(fit), dtype=torch.long), torch.tensor(sorted(calibration), dtype=torch.long)
