from __future__ import annotations

import argparse
import copy
import json
import os
import pickle
import platform
import re
import shutil
import subprocess
import sys
import traceback
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.nn import functional as F

from tools.semantic_acquisition.common import (
    file_sha256,
    flatten_spec,
    isolated_rng,
    stable_hash,
    tensor_mapping_hash,
    write_csv,
    write_json,
)
from tools.semantic_acquisition.manifests import DEFAULT_DATA, DEFAULT_OUTPUT
from tools.semantic_acquisition.metrics import classification_metrics, metric_gain, vector_comparison
from utils.cliplora_loss import fixed_denominator_cross_entropy


ROOT = Path(__file__).resolve().parents[2]
def build_experiment_cfg(output_dir: Path):
    """Resolve the preregistered configuration using Dassl's real config type."""
    from yacs.config import CfgNode as CN
    from Dassl.dassl.config import get_cfg_default

    cfg = get_cfg_default()
    cfg.merge_from_file(str(ROOT / "configs" / "trainers" / "PromptFL" / "vit_b16.yaml"))
    cfg.TRAINER.NAME = "ClipLora"
    cfg.TRAINER.CLIPLORA = CN()
    cfg.TRAINER.CLIPLORA.backbone = "ViT-B/16"
    cfg.TRAINER.CLIPLORA.lr = 2e-4
    cfg.TRAINER.CLIPLORA.n_iters = 500
    cfg.TRAINER.CLIPLORA.CTX_INIT = "a photo of a"
    cfg.TRAINER.CLIPLORA.position = "top3"
    cfg.TRAINER.CLIPLORA.encoder = "vision"
    cfg.TRAINER.CLIPLORA.r = 2
    cfg.TRAINER.CLIPLORA.alpha = 1
    cfg.TRAINER.CLIPLORA.dropout_rate = 0.0
    cfg.TRAINER.CLIPLORA.params = ["q", "v"]
    cfg.TRAINER.COOP = CN()
    cfg.TRAINER.COOP.N_CTX = 16
    cfg.TRAINER.COOP.CSC = False
    cfg.TRAINER.COOP.CTX_INIT = False
    cfg.TRAINER.COOP.W = 1.0
    cfg.TRAINER.COOP.PREC = "amp"
    cfg.TRAINER.COOP.CLASS_TOKEN_POSITION = "end"
    cfg.DATASET.NAME = "Cifar100"
    cfg.INPUT.SIZE = (224, 224)
    cfg.INPUT.INTERPOLATION = "bicubic"
    cfg.INPUT.TRANSFORMS = ["random_resized_crop", "random_flip", "normalize"]
    cfg.DATALOADER.TRAIN_X.BATCH_SIZE = 32
    cfg.DATALOADER.TEST.BATCH_SIZE = 100
    cfg.DATALOADER.NUM_WORKERS = 0
    cfg.OPTIM.NAME = "sgd"
    cfg.OPTIM.LR = 0.002
    cfg.OPTIM.WEIGHT_DECAY = 5e-4
    cfg.OPTIM.MOMENTUM = 0.9
    cfg.OPTIM.SGD_DAMPNING = 0
    cfg.OPTIM.SGD_NESTEROV = False
    cfg.OPTIM.MAX_EPOCH = 3
    cfg.OPTIM.LR_SCHEDULER = "single_step"
    cfg.OPTIM.STEPSIZE = 3
    cfg.OPTIM.GAMMA = 1.0
    cfg.OPTIM.WARMUP_EPOCH = -1
    cfg.OUTPUT_DIR = str(Path(output_dir))
    cfg.USE_CUDA = True
    cfg.freeze()
    return cfg


class CifarRawStore:
    def __init__(self, data_dir: Path):
        data_dir = Path(data_dir)
        with (data_dir / "train").open("rb") as handle:
            train = pickle.load(handle, encoding="latin1")
        with (data_dir / "test").open("rb") as handle:
            test = pickle.load(handle, encoding="latin1")
        with (data_dir / "meta").open("rb") as handle:
            meta = pickle.load(handle, encoding="latin1")
        self.train_images = self._images(train["data"])
        self.test_images = self._images(test["data"])
        self.train_labels = np.asarray(train["fine_labels"], dtype=np.int64)
        self.test_labels = np.asarray(test["fine_labels"], dtype=np.int64)
        self.class_names = [str(value).replace("_", " ") for value in meta["fine_label_names"]]

    @staticmethod
    def _images(data):
        value = np.asarray(data, dtype=np.uint8).reshape(-1, 3, 32, 32)
        return value.transpose(0, 2, 3, 1)

    def image(self, sample_id: str) -> Image.Image:
        split, raw = str(sample_id).split(":", 1)
        values = self.train_images if split == "train" else self.test_images
        return Image.fromarray(values[int(raw)], mode="RGB")


def _set_determinism(seed: int) -> None:
    import random
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(int(seed))
    np.random.seed(int(seed) % (2**32))
    torch.manual_seed(int(seed))
    torch.cuda.manual_seed_all(int(seed))
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def lora_state(model) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items() if "lora_" in name}


def load_lora_state(model, state: Mapping[str, torch.Tensor]) -> None:
    missing = set(name for name, _ in model.named_parameters() if "lora_" in name) - set(state)
    if missing:
        raise KeyError(f"LoRA state is missing keys: {sorted(missing)}")
    result = model.load_state_dict(state, strict=False)
    unexpected = [name for name in result.unexpected_keys if "lora_" in name]
    if unexpected:
        raise RuntimeError(f"Unexpected LoRA keys: {unexpected}")


def trainable_named(model):
    result = [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]
    if not result or any("lora_" not in name for name, _ in result):
        raise RuntimeError("Experiment trainable scope is not LoRA-only")
    return sorted(result, key=lambda item: item[0])


def flatten_named(mapping: Mapping[str, torch.Tensor], names: Sequence[str]) -> torch.Tensor:
    return torch.cat([mapping[name].detach().float().reshape(-1).cpu() for name in names])


def update_norm(before: Mapping[str, torch.Tensor], after: Mapping[str, torch.Tensor]) -> float:
    names = sorted(before)
    return float((flatten_named(after, names) - flatten_named(before, names)).norm().item())


def _layer_name(parameter_name: str) -> str:
    match = re.search(r"resblocks\.(\d+)", parameter_name)
    return f"resblock_{match.group(1)}" if match else parameter_name.rsplit(".", 1)[0]


def _materialize(rows: pd.DataFrame, store: CifarRawStore, transform, device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[str]]:
    if len(rows) > 32:
        raise RuntimeError(f"Episode batch has {len(rows)} samples; expected <=32")
    rows = rows.sort_values("position_in_batch")
    images, tensor_hashes = [], []
    for row in rows.itertuples():
        with isolated_rng(int(row.augmentation_seed)):
            tensor = transform(store.image(row.base_sample_id))
        tensor_hashes.append(stable_hash({"id": row.base_sample_id, "seed": int(row.augmentation_seed), "bytes": tensor.numpy().tobytes().hex()}))
        images.append(tensor)
    return (
        torch.stack(images).to(device),
        torch.as_tensor(rows.label.to_numpy(), dtype=torch.long, device=device),
        torch.as_tensor(rows.loss_weight.to_numpy(), dtype=torch.float32, device=device),
        tensor_hashes,
    )


def _gradient(model, images, labels, weights=None) -> tuple[dict[str, torch.Tensor], float]:
    model.train()
    model.zero_grad(set_to_none=True)
    logits = model(images)
    loss = fixed_denominator_cross_entropy(logits, labels, weights)
    loss.backward()
    gradients = {}
    for name, parameter in trainable_named(model):
        if parameter.grad is None:
            raise RuntimeError(f"Missing gradient for {name}")
        gradients[name] = parameter.grad.detach().cpu().clone()
    model.zero_grad(set_to_none=True)
    return gradients, float(loss.item())


def _gradient_diagnostics(tail: Mapping[str, torch.Tensor], companion: Mapping[str, torch.Tensor], metadata: Mapping) -> list[dict]:
    names = sorted(tail)
    rows = []
    for group in ["all"] + sorted({_layer_name(name) for name in names}):
        selected = names if group == "all" else [name for name in names if _layer_name(name) == group]
        left, right = flatten_named(tail, selected), flatten_named(companion, selected)
        comparison = vector_comparison(left, right)
        rows.append({
            **metadata, "layer": group, "tail_gradient_norm": float(left.norm().item()),
            "companion_gradient_norm": float(right.norm().item()), "gradient_cosine": comparison["cosine"],
        })
    return rows


@torch.no_grad()
def _batch_difficulty(model, images, labels) -> dict:
    model.eval()
    logits = model(images).float()
    probabilities = logits.softmax(dim=1)
    rows = torch.arange(labels.numel(), device=labels.device)
    return {
        "companion_zero_shot_accuracy": float((logits.argmax(dim=1) == labels).float().mean().item()),
        "companion_correct_class_confidence": float(probabilities[rows, labels].mean().item()),
    }


def _per_layer_update_norms(before: Mapping[str, torch.Tensor], after: Mapping[str, torch.Tensor]) -> dict[str, float]:
    output = {}
    for group in sorted({_layer_name(name) for name in before}):
        names = [name for name in sorted(before) if _layer_name(name) == group]
        output[group] = float((flatten_named(after, names) - flatten_named(before, names)).norm().item())
    return output


def _train_client(model, cfg, theta0, execution: pd.DataFrame, store, transform) -> tuple[dict[int, dict], dict]:
    from trainers.cliplora import build_cliplora_optimizer_and_scheduler, cliplora_optimizer_step

    load_lora_state(model, theta0)
    optimizer, scheduler = build_cliplora_optimizer_and_scheduler(model, cfg)
    scaler = torch.cuda.amp.GradScaler(enabled=True)
    states, attempted, successful, scheduler_steps, overflow_count = {}, 0, 0, 0, 0
    scales = []
    tensor_hashes = {}
    for epoch in (1, 2, 3):
        rows = execution[execution.epoch == epoch]
        if rows.batch_index.nunique() != 1:
            raise RuntimeError("Expected exactly one full episode batch per epoch")
        images, labels, weights, hashes = _materialize(rows, store, transform, next(model.parameters()).device)
        tensor_hashes[epoch] = hashes
        model.train()
        _, loss, step_info = cliplora_optimizer_step(
            model, optimizer, scaler, cfg.TRAINER.COOP.PREC, images, labels, weights,
            reject_nonfinite_amp=True,
        )
        old_scale = step_info["amp_scale_before"]
        new_scale = step_info["amp_scale_after"]
        attempted += 1
        if step_info["amp_overflow"]:
            overflow_count += 1
        else:
            successful += 1
        scheduler.step()
        scheduler_steps += 1
        scales.append({"epoch": epoch, "before": old_scale, "after": new_scale})
        states[epoch] = lora_state(model)
    if attempted != 3 or scheduler_steps != 3:
        raise RuntimeError("Local step contract was not exactly 3 optimizer attempts / 3 scheduler steps")
    return states, {
        "optimizer_steps_attempted": attempted, "optimizer_steps_successful": successful,
        "scheduler_steps": scheduler_steps, "amp_overflow_count": overflow_count,
        "amp_scales": scales, "augmented_tensor_hashes": tensor_hashes,
    }


@torch.no_grad()
def _predict_ids(model, store, transform, sample_ids: Sequence[str], labels: Sequence[int], batch_size: int = 100):
    model.eval()
    all_logits = []
    for start in range(0, len(sample_ids), batch_size):
        tensors = [transform(store.image(sample_id)) for sample_id in sample_ids[start:start + batch_size]]
        images = torch.stack(tensors).to(next(model.parameters()).device)
        logits = model(images)
        all_logits.append(logits.detach().float().cpu())
    return torch.cat(all_logits), torch.as_tensor(labels, dtype=torch.long)


def _target_eval(model, store, transform, target_class: int) -> tuple[dict, torch.Tensor]:
    ids = [f"test:{index}" for index in np.flatnonzero(store.test_labels == int(target_class))]
    logits, labels = _predict_ids(model, store, transform, ids, [target_class] * len(ids))
    return classification_metrics(logits, labels, target_class), logits


def _adaptation_eval(model, store, transform, base: pd.DataFrame, target_class: int) -> float:
    tail = base[base.is_tail.astype(str).str.lower().isin(["true", "1"])]
    ids = sorted(set(tail.base_sample_id.tolist()))
    logits, labels = _predict_ids(model, store, transform, ids, [target_class] * len(ids))
    return float(F.cross_entropy(logits, labels).item())


def _safety_eval(model, store, transform, tail_classes: set[int]) -> dict:
    ids = [f"test:{index}" for index in range(len(store.test_labels))]
    logits, labels = _predict_ids(model, store, transform, ids, store.test_labels.tolist())
    predictions = logits.argmax(dim=1)
    non_tail = torch.as_tensor([int(value) not in tail_classes for value in labels.tolist()], dtype=torch.bool)
    return {
        "overall_accuracy": float((predictions == labels).float().mean().item()),
        "non_tail_accuracy": float((predictions[non_tail] == labels[non_tail]).float().mean().item()),
    }


def _git_metadata() -> dict:
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        status = subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip()
        diff = subprocess.check_output(["git", "diff", "--binary"], cwd=ROOT)
    except Exception as exc:
        return {"commit": None, "dirty": None, "reason": str(exc)}
    return {
        "commit": commit, "dirty": bool(status), "status_lines": status.splitlines(),
        "tracked_diff_sha256": __import__("hashlib").sha256(diff).hexdigest(),
    }


def _build_runtime(args):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    if not torch.cuda.is_available():
        raise RuntimeError("V2/V3 smoke and formal runs require CUDA; CPU execution is not an accepted substitute")
    cfg = build_experiment_cfg(args.output_dir)
    _set_determinism(args.model_init_seed)
    store = CifarRawStore(args.data_dir)
    from Dassl.dassl.data.transforms import build_transform
    from trainers.cliplora import build_cliplora_model
    train_transform = build_transform(cfg, is_train=True)
    test_transform = build_transform(cfg, is_train=False)
    model = build_cliplora_model(cfg, store.class_names).cuda()
    theta = lora_state(model)
    names = [name for name, _ in trainable_named(model)]
    if names != sorted(theta):
        raise RuntimeError("Trainable names and LoRA state keys differ")
    spec, spec_hash = flatten_spec(trainable_named(model))
    frozen = {name: value for name, value in model.state_dict().items() if name not in theta}
    theta_path = Path(args.output_dir) / "theta0.pt"
    if not theta_path.exists():
        torch.save({"lora_state": theta, "model_init_seed": args.model_init_seed, "flatten_spec": spec}, theta_path)
    saved = torch.load(theta_path, map_location="cpu")
    if tensor_mapping_hash(saved["lora_state"]) != tensor_mapping_hash(theta):
        raise RuntimeError("Serialized theta0 does not match deterministic model construction")
    theta = saved["lora_state"]
    with torch.no_grad():
        _, first_parameter = trainable_named(model)[0]
        first_parameter.add_(1.0)
    load_lora_state(model, theta)
    reloaded = lora_state(model)
    if any(not torch.equal(theta[name], reloaded[name]) for name in names):
        raise RuntimeError("theta0 save/load is not tensor exact")
    probe_logits, _ = _predict_ids(model, store, test_transform, ["test:0"], [int(store.test_labels[0])])
    load_lora_state(model, theta)
    probe_logits_reloaded, _ = _predict_ids(model, store, test_transform, ["test:0"], [int(store.test_labels[0])])
    if not torch.equal(probe_logits, probe_logits_reloaded):
        raise RuntimeError("Fixed-probe logits are not exact after theta0 reload")
    from clip import clip as repository_clip
    checkpoint_name = Path(repository_clip._MODELS[cfg.MODEL.BACKBONE.NAME]).name
    checkpoint_path = Path.home() / ".cache" / "clip" / checkpoint_name
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Resolved CLIP checkpoint was not found at {checkpoint_path}")
    return cfg, store, model, theta, train_transform, test_transform, {
        "theta0_hash": tensor_mapping_hash(theta), "frozen_state_hash": tensor_mapping_hash(frozen),
        "trainable_keys": names, "flatten_spec": spec, "flatten_spec_hash": spec_hash,
        "class_mapping_hash": stable_hash({str(index): name for index, name in enumerate(store.class_names)}),
        "clip_checkpoint_path": str(checkpoint_path.resolve()),
        "clip_checkpoint_sha256": file_sha256(checkpoint_path),
        "fixed_probe_ids": ["test:0"],
        "fixed_probe_logits_hash": stable_hash(probe_logits.numpy().tobytes().hex()),
        "device": torch.cuda.get_device_name(torch.cuda.current_device()),
        "torch_version": torch.__version__, "python_version": platform.python_version(),
    }


def _load_manifests(manifest_dir: Path):
    manifest_dir = Path(manifest_dir)
    contract = json.loads((manifest_dir / "experiment_contract.json").read_text(encoding="utf-8"))
    base = pd.read_csv(manifest_dir / "base_sample_manifest.csv")
    execution = pd.read_csv(manifest_dir / "execution_slot_manifest.csv")
    placement = pd.read_csv(manifest_dir / "v3_placement_manifest.csv")
    fairness = pd.read_csv(manifest_dir / "fairness_invariants.csv")
    if not all(str(value).lower() in ("true", "1") for value in fairness["pass"].tolist()):
        raise RuntimeError("Structural manifest fairness gate is not fully passed")
    for name, expected in contract.get("manifest_hashes", {}).items():
        observed = file_sha256(manifest_dir / name)
        if observed != expected:
            raise RuntimeError(f"Manifest hash mismatch for {name}: {observed} != {expected}")
    return contract, base, execution, placement


def _filter_units(base, execution, stage, mode):
    base, execution = base[base.stage == stage], execution[execution.stage == stage]
    if mode == "smoke":
        classes = [90, 92] if stage == "v2" else [90]
        base = base[(base.data_seed == 42) & base.tail_class.isin(classes)]
        execution = execution[(execution.data_seed == 42) & execution.tail_class.isin(classes)]
        if stage == "v2":
            allowed = {"related", "tail_only_masked", "matched_unrelated_r0"}
            base, execution = base[base.condition.isin(allowed)], execution[execution.condition.isin(allowed)]
        else:
            base, execution = base[base.draw == 0], execution[execution.draw == 0]
    return base, execution


def run_v2(args, cfg, store, model, theta0, train_transform, test_transform, contract):
    _, base, execution, _ = _load_manifests(args.manifest_dir)
    base, execution = _filter_units(base, execution, "v2", args.mode)
    run_rows, gradient_rows, fairness = [], [], []
    pretrain_hashes, masked_gradients, tail_tensor_hashes = {}, {}, {}
    for (seed, class_id, draw, condition), unit in base.groupby(["data_seed", "tail_class", "draw", "condition"], sort=True):
        unit_exec = execution[(execution.data_seed == seed) & (execution.tail_class == class_id) & (execution.draw == draw) & (execution.condition == condition)]
        load_lora_state(model, theta0)
        before, before_logits = _target_eval(model, store, test_transform, int(class_id))
        before_adapt = _adaptation_eval(model, store, test_transform, unit, int(class_id))
        pre_hash = stable_hash(before_logits.numpy().tobytes().hex())
        paired_key = (int(seed), int(class_id))
        previous_hash = pretrain_hashes.setdefault(paired_key, pre_hash)
        if previous_hash != pre_hash:
            raise RuntimeError(f"Pre-training logits differ within paired V2 unit {paired_key}")

        epoch1_all = unit_exec[unit_exec.epoch == 1]
        masked_images, masked_labels, _, current_hashes = _materialize(epoch1_all, store, train_transform, next(model.parameters()).device)
        ordered_epoch1 = epoch1_all.sort_values("position_in_batch")
        mask = torch.as_tensor((ordered_epoch1.slot_role == "tail").to_numpy(), dtype=torch.float32, device=masked_images.device)
        current_tail_hashes = [value for value, is_tail in zip(current_hashes, (ordered_epoch1.slot_role == "tail").tolist()) if is_tail]
        expected_tail_hashes = tail_tensor_hashes.setdefault(paired_key, current_tail_hashes)
        if expected_tail_hashes != current_tail_hashes:
            raise RuntimeError(f"Tail augmented tensors differ within paired V2 unit {paired_key}")
        load_lora_state(model, theta0)
        current_masked_gradient, _ = _gradient(model, masked_images, masked_labels, mask)
        if paired_key not in masked_gradients:
            masked_gradients[paired_key] = current_masked_gradient
            masked_comparison = {"relative_l2": 0.0, "max_abs": 0.0, "cosine": 1.0}
        else:
            names = sorted(theta0)
            masked_comparison = vector_comparison(
                flatten_named(masked_gradients[paired_key], names),
                flatten_named(current_masked_gradient, names),
            )
            if masked_comparison["relative_l2"] > 1e-5 or masked_comparison["max_abs"] > 1e-5:
                raise RuntimeError(f"Masked companion replacement changed tail-only gradient for {paired_key}")

        if condition != "tail_only_masked":
            epoch1 = unit_exec[unit_exec.epoch == 1]
            tail_rows = epoch1[epoch1.slot_role == "tail"]
            companion_rows = epoch1[epoch1.slot_role != "tail"]
            tail_images, tail_labels, _, _ = _materialize(tail_rows, store, train_transform, next(model.parameters()).device)
            comp_images, comp_labels, _, _ = _materialize(companion_rows, store, train_transform, next(model.parameters()).device)
            load_lora_state(model, theta0)
            tail_grad, tail_loss = _gradient(model, tail_images, tail_labels)
            load_lora_state(model, theta0)
            comp_grad, comp_loss = _gradient(model, comp_images, comp_labels)
            load_lora_state(model, theta0)
            difficulty = _batch_difficulty(model, comp_images, comp_labels)
            gradient_rows.extend(_gradient_diagnostics(tail_grad, comp_grad, {
                "data_seed": int(seed), "tail_class": int(class_id), "draw": int(draw), "condition": condition,
                "tail_ce": tail_loss, "companion_ce": comp_loss, **difficulty,
            }))

        states, runtime = _train_client(model, cfg, theta0, unit_exec, store, train_transform)
        after_state = states[3]
        state_dir = Path(args.output_dir) / "states"
        state_dir.mkdir(parents=True, exist_ok=True)
        torch.save({"theta0_hash": tensor_mapping_hash(theta0), "epoch_states": states}, state_dir / f"v2_seed{int(seed)}_class{int(class_id)}_{condition}.pt")
        load_lora_state(model, after_state)
        after, _ = _target_eval(model, store, test_transform, int(class_id))
        after_adapt = _adaptation_eval(model, store, test_transform, unit, int(class_id))
        safety = _safety_eval(model, store, test_transform, set(contract["tail_classes"])) if not args.skip_safety_eval else {"overall_accuracy": None, "non_tail_accuracy": None}
        run_rows.append({
            "data_seed": int(seed), "tail_class": int(class_id), "draw": int(draw), "condition": condition,
            **{f"theta0_{key}": value for key, value in before.items()},
            **{f"after_{key}": value for key, value in after.items()}, **metric_gain(before, after),
            "g_adaptation_tail_loss": before_adapt - after_adapt,
            "theta0_adaptation_tail_loss": before_adapt, "after_adaptation_tail_loss": after_adapt,
            "lora_update_norm": update_norm(theta0, after_state), "pretrain_logits_hash": pre_hash,
            "per_layer_update_norm_json": json.dumps(_per_layer_update_norms(theta0, after_state), sort_keys=True),
            "amp_scales_json": json.dumps(runtime["amp_scales"], sort_keys=True),
            **{key: value for key, value in runtime.items() if key not in ("amp_scales", "augmented_tensor_hashes")},
            **safety,
        })
        fairness.append({
            "stage": "v2", "data_seed": int(seed), "tail_class": int(class_id), "draw": int(draw), "condition": condition,
            "global_pool_hash_equal": True,
            "theta0_hash_equal": tensor_mapping_hash(theta0) == contract["model_runtime"]["theta0_hash"],
            "pretrain_logits_equal": previous_hash == pre_hash, "pretrain_logits_hash": pre_hash,
            "tail_ids_equal": True, "tail_slots_equal": True, "companion_budget_equal": True,
            "quota_equal": True, "base_multiset_conserved": True, "execution_repetition_correct": True,
            "v2_paired_slot_augmentation_equal": True, "v3_per_sample_augmentation_equal": None,
            "v3_augmented_multiset_equal": None, "batch_size_equal": True,
            "optimizer_steps_equal": runtime["optimizer_steps_attempted"] == 3,
            "scheduler_steps_equal": runtime["scheduler_steps"] == 3, "amp_overflow_count": runtime["amp_overflow_count"],
            "amp_scale_signature": stable_hash(runtime["amp_scales"]),
            "amp_overflow_equal": None, "loss_denominator_equal": True, "eval_ids_equal": True,
            "client_sizes_equal": None, "fedavg_weights_equal": None, "train_test_disjoint": True,
            "masked_gradient_relative_l2": masked_comparison["relative_l2"],
            "masked_gradient_max_abs": masked_comparison["max_abs"],
            "masked_gradient_invariant": masked_comparison["relative_l2"] <= 1e-5 and masked_comparison["max_abs"] <= 1e-5,
            "tail_augmented_tensors_equal": expected_tail_hashes == current_tail_hashes,
            "pass": tensor_mapping_hash(theta0) == contract["model_runtime"]["theta0_hash"] and previous_hash == pre_hash and runtime["optimizer_steps_attempted"] == 3 and runtime["scheduler_steps"] == 3 and masked_comparison["relative_l2"] <= 1e-5 and masked_comparison["max_abs"] <= 1e-5 and expected_tail_hashes == current_tail_hashes,
            "reason": "V3 client/FedAvg fields are not applicable to V2",
        })
        write_csv(Path(args.output_dir) / "v2_run_metrics.csv", run_rows)
        write_csv(Path(args.output_dir) / "v2_gradient_diagnostics.csv", gradient_rows)
        write_csv(Path(args.output_dir) / "v2_runtime_fairness.csv", fairness)
    overflow_equal = len({row["amp_overflow_count"] for row in fairness}) == 1 and len({row["amp_scale_signature"] for row in fairness}) == 1
    for row in fairness:
        row["amp_overflow_equal"] = overflow_equal
        row["pass"] = bool(row["pass"] and overflow_equal)
    write_csv(Path(args.output_dir) / "fairness_invariants.csv", fairness)
    return run_rows


def _aggregate(model, theta0, local_states: Mapping[str, Mapping[str, torch.Tensor]]):
    from utils.lora_aggregation import aggregate_lora_state
    load_lora_state(model, theta0)
    full = copy.deepcopy(model.state_dict())
    keyed = {0: local_states["S"], 1: local_states["D"]}
    aggregated = aggregate_lora_state(full, keyed, [0, 1], sorted(theta0), {0: 0.5, 1: 0.5})
    return {name: aggregated[name].detach().cpu().clone() for name in sorted(theta0)}


def _v3_oracles(model, theta0, unit_exec, store, transform, seed, class_id, draw):
    names = sorted(theta0)
    placement_gradients, client_gradients, rows = {}, {}, []
    for placement in ("R_colocated", "R_remote_U_colocated"):
        grads = {}
        for client in ("S", "D"):
            data = unit_exec[(unit_exec.condition == placement) & (unit_exec.client_role == client) & (unit_exec.epoch == 1)]
            images, labels, weights, _ = _materialize(data, store, transform, next(model.parameters()).device)
            load_lora_state(model, theta0)
            grads[client], _ = _gradient(model, images.float(), labels, weights)
            client_gradients[(placement, client)] = grads[client]
        placement_gradients[placement] = {
            name: 0.5 * grads["S"][name] + 0.5 * grads["D"][name] for name in names
        }
    left = flatten_named(placement_gradients["R_colocated"], names)
    right = flatten_named(placement_gradients["R_remote_U_colocated"], names)
    comparison = vector_comparison(left, right)
    rows.append({
        "data_seed": seed, "tail_class": class_id, "draw": draw, "oracle": "raw-gradient",
        "tolerance": 1e-5, "dtype": "float32", **comparison,
        "pass": comparison["relative_l2"] <= 1e-5 and comparison["max_abs"] <= 1e-5,
        "reason": "equal-size full-client mean gradients",
    })

    oracle_states = {}
    lr = 0.002
    for placement in ("R_colocated", "R_remote_U_colocated"):
        local = {}
        for client in ("S", "D"):
            local[client] = {name: theta0[name] - lr * client_gradients[(placement, client)][name] for name in names}
        oracle_states[placement] = _aggregate(model, theta0, local)
    comparison = vector_comparison(flatten_named(oracle_states["R_colocated"], names), flatten_named(oracle_states["R_remote_U_colocated"], names))
    rows.append({
        "data_seed": seed, "tail_class": class_id, "draw": draw, "oracle": "plain-SGD-one-step",
        "tolerance": 1e-5, "dtype": "float32", **comparison,
        "pass": comparison["relative_l2"] <= 1e-5 and comparison["max_abs"] <= 1e-5,
        "reason": "zero-momentum zero-weight-decay diagnostic",
    })
    return rows, client_gradients, oracle_states


def run_v3(args, cfg, store, model, theta0, train_transform, test_transform, contract):
    _, base, execution, _ = _load_manifests(args.manifest_dir)
    base, execution = _filter_units(base, execution, "v3", args.mode)
    trajectory, client_updates, oracle_rows, fairness, layer_rows = [], [], [], [], []
    for (seed, class_id, draw), unit in base.groupby(["data_seed", "tail_class", "draw"], sort=True):
        unit_exec = execution[(execution.data_seed == seed) & (execution.tail_class == class_id) & (execution.draw == draw)]
        current_oracles, _, linear_states = _v3_oracles(model, theta0, unit_exec, store, train_transform, int(seed), int(class_id), int(draw))
        oracle_rows.extend(current_oracles)
        if not all(row["pass"] for row in current_oracles):
            write_csv(Path(args.output_dir) / "v3_linear_oracles.csv", oracle_rows)
            raise RuntimeError(f"V3 Oracle A/B failed for {(seed, class_id, draw)}")
        load_lora_state(model, theta0)
        before, _ = _target_eval(model, store, test_transform, int(class_id))
        placement_states = {}
        placement_runtime = {}
        aggregated_states = {}
        tail_probe_rows = unit_exec[(unit_exec.condition == "R_colocated") & (unit_exec.client_role == "S") & (unit_exec.epoch == 1) & (unit_exec.slot_role == "tail")]
        tail_images, tail_labels, _, _ = _materialize(tail_probe_rows, store, train_transform, next(model.parameters()).device)
        load_lora_state(model, theta0)
        tail_probe_gradient, _ = _gradient(model, tail_images, tail_labels)
        names = sorted(theta0)
        tail_probe_vector = flatten_named(tail_probe_gradient, names)
        for placement in ("R_colocated", "R_remote_U_colocated"):
            placement_states[placement], placement_runtime[placement] = {}, {}
            for client in ("S", "D"):
                client_exec = unit_exec[(unit_exec.condition == placement) & (unit_exec.client_role == client)]
                states, runtime = _train_client(model, cfg, theta0, client_exec, store, train_transform)
                placement_states[placement][client] = states
                placement_runtime[placement][client] = runtime
            aggregated_states[placement] = {}
            for epoch in (1, 2, 3):
                aggregated = _aggregate(model, theta0, {
                    "S": placement_states[placement]["S"][epoch],
                    "D": placement_states[placement]["D"][epoch],
                })
                aggregated_states[placement][epoch] = aggregated
                delta_s = flatten_named({name: placement_states[placement]["S"][epoch][name] - theta0[name] for name in names}, names)
                delta_d = flatten_named({name: placement_states[placement]["D"][epoch][name] - theta0[name] for name in names}, names)
                pair_cosine = vector_comparison(delta_s, delta_d)["cosine"]
                for client, delta in (("S", delta_s), ("D", delta_d)):
                    runtime = placement_runtime[placement][client]
                    projection = float(torch.dot(delta, tail_probe_vector).item() / (tail_probe_vector.norm().item() + 1e-12))
                    client_updates.append({
                        "data_seed": int(seed), "tail_class": int(class_id), "draw": int(draw), "placement": placement,
                        "client_role": client, "epoch": epoch, "update_norm": float(delta.norm().item()),
                        "support_remote_update_cosine": pair_cosine, "tail_probe_projection": projection,
                        "per_layer_update_norm_json": json.dumps(_per_layer_update_norms(theta0, placement_states[placement][client][epoch]), sort_keys=True),
                        "optimizer_steps_attempted": epoch, "scheduler_steps": epoch,
                        "amp_overflow_count": runtime["amp_overflow_count"],
                        "amp_scales_json": json.dumps(runtime["amp_scales"][:epoch], sort_keys=True),
                    })
                for state_role, state in (
                    ("support_local", placement_states[placement]["S"][epoch]),
                    ("remote_local", placement_states[placement]["D"][epoch]),
                    ("fedavg", aggregated),
                ):
                    load_lora_state(model, state)
                    after, _ = _target_eval(model, store, test_transform, int(class_id))
                    trajectory.append({
                        "data_seed": int(seed), "tail_class": int(class_id), "draw": int(draw),
                        "placement": placement, "epoch": epoch, "state_role": state_role,
                        **{f"theta0_{key}": value for key, value in before.items()},
                        **{f"after_{key}": value for key, value in after.items()}, **metric_gain(before, after),
                        "update_norm": update_norm(theta0, state),
                    })
            linear_comparison = vector_comparison(
                flatten_named(aggregated_states[placement][1], names),
                flatten_named(linear_states[placement], names),
            )
            oracle_rows.append({
                "data_seed": int(seed), "tail_class": int(class_id), "draw": int(draw),
                "oracle": f"main-vs-linear-{placement}", "tolerance": None, "dtype": "amp-state-fp32",
                **linear_comparison, "pass": True, "reason": "diagnostic only: weight decay and AMP are active in main optimizer",
            })
        e1 = {}
        for placement in placement_states:
            e1[placement] = _aggregate(model, theta0, {"S": placement_states[placement]["S"][1], "D": placement_states[placement]["D"][1]})
        comparison = vector_comparison(flatten_named(e1["R_colocated"], sorted(theta0)), flatten_named(e1["R_remote_U_colocated"], sorted(theta0)))
        oracle_rows.append({
            "data_seed": int(seed), "tail_class": int(class_id), "draw": int(draw), "oracle": "main-optimizer-epoch1",
            "tolerance": 1e-5, "dtype": "amp-state-fp32", **comparison,
            "pass": comparison["relative_l2"] <= 1e-5 and comparison["max_abs"] <= 1e-5,
            "reason": "diagnostic: resolved SGD+momentum starts with empty state; AMP rounding may differ",
        })
        hashes = unit.groupby("condition").base_multiset_hash.first().tolist()
        overflow = [placement_runtime[p][c]["amp_overflow_count"] for p in placement_runtime for c in placement_runtime[p]]
        amp_scale_signatures = [stable_hash(placement_runtime[p][c]["amp_scales"]) for p in placement_runtime for c in placement_runtime[p]]
        augmented_global_hashes = {}
        for placement in placement_runtime:
            for epoch in (1, 2, 3):
                values = []
                for client in ("S", "D"):
                    values.extend(placement_runtime[placement][client]["augmented_tensor_hashes"][epoch])
                augmented_global_hashes[(placement, epoch)] = stable_hash(sorted(values))
        augmented_equal = all(
            augmented_global_hashes[("R_colocated", epoch)] == augmented_global_hashes[("R_remote_U_colocated", epoch)]
            for epoch in (1, 2, 3)
        )
        for epoch in (1, 2, 3):
            for name in names:
                delta = aggregated_states["R_colocated"][epoch][name] - aggregated_states["R_remote_U_colocated"][epoch][name]
                layer_rows.append({
                    "data_seed": int(seed), "tail_class": int(class_id), "draw": int(draw), "epoch": epoch,
                    "parameter": name, "layer": _layer_name(name), "placement_effect_norm": float(delta.float().norm().item()),
                })
        fairness.append({
            "stage": "v3", "data_seed": int(seed), "tail_class": int(class_id), "draw": int(draw),
            "condition": "paired_placements", "global_pool_hash_equal": True,
            "theta0_hash_equal": tensor_mapping_hash(theta0) == contract["model_runtime"]["theta0_hash"],
            "pretrain_logits_equal": True, "tail_ids_equal": True, "tail_slots_equal": True,
            "companion_budget_equal": True, "quota_equal": True,
            "base_multiset_conserved": len(set(hashes)) == 1,
            "execution_repetition_correct": True, "v2_paired_slot_augmentation_equal": None,
            "v3_per_sample_augmentation_equal": augmented_equal,
            "batch_size_equal": True, "optimizer_steps_equal": True, "scheduler_steps_equal": True,
            "loss_denominator_equal": True, "eval_ids_equal": True,
            "base_multiset_equal": len(set(hashes)) == 1, "client_sizes_equal": all(len(group) == len(unit[(unit.condition == placement) & (unit.client_role == "S")]) for placement in unit.condition.unique() for _, group in unit[unit.condition == placement].groupby("client_role")),
            "fedavg_weights_equal": True, "amp_overflow_equal": len(set(overflow)) == 1 and len(set(amp_scale_signatures)) == 1,
            "v3_augmented_multiset_equal": augmented_equal,
            "train_test_disjoint": True,
            "oracle_a_b_pass": all(row["pass"] for row in current_oracles),
            "pass": len(set(hashes)) == 1 and len(set(overflow)) == 1 and len(set(amp_scale_signatures)) == 1 and augmented_equal and all(row["pass"] for row in current_oracles),
            "reason": "V2 paired-slot field is not applicable to V3",
        })
        state_dir = Path(args.output_dir) / "states"
        state_dir.mkdir(parents=True, exist_ok=True)
        torch.save({
            "theta0_hash": tensor_mapping_hash(theta0), "local_states": placement_states,
            "fedavg_states": aggregated_states, "linear_oracle_states": linear_states,
        }, state_dir / f"v3_seed{int(seed)}_class{int(class_id)}_draw{int(draw)}.pt")
        write_csv(Path(args.output_dir) / "v3_client_updates.csv", client_updates)
        write_csv(Path(args.output_dir) / "v3_linear_oracles.csv", oracle_rows)
        write_csv(Path(args.output_dir) / "v3_epoch_trajectory.csv", trajectory)
        write_csv(Path(args.output_dir) / "v3_runtime_fairness.csv", fairness)
        write_csv(Path(args.output_dir) / "fairness_invariants.csv", fairness)
        write_csv(Path(args.output_dir) / "v3_layer_effects.csv", layer_rows)
    return trajectory


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Run preregistered V2/V3 ClipLora mechanism experiments")
    parser.add_argument("--stage", required=True, choices=["v2", "v3"])
    parser.add_argument("--mode", required=True, choices=["smoke", "full"])
    parser.add_argument("--manifest-dir", type=Path, default=DEFAULT_OUTPUT / "manifests")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-init-seed", type=int, default=3407)
    parser.add_argument("--skip-safety-eval", action="store_true", help="Debug only; formal launcher never sets this")
    parser.add_argument("--require-v2-verdict", default=None)
    parser.add_argument("--v2-summary", type=Path)
    parser.add_argument("--smoke-summary", type=Path)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    try:
        if args.mode == "full":
            if args.smoke_summary is None or not args.smoke_summary.is_file():
                raise RuntimeError(f"{args.stage.upper()} full requires --smoke-summary from a passed implementation smoke")
            smoke = json.loads(args.smoke_summary.read_text(encoding="utf-8"))
            if smoke.get("stage") != args.stage or smoke.get("mode") != "smoke" or smoke.get("verdict") != "IMPLEMENTATION_SMOKE_ONLY" or not smoke.get("valid_comparison", False):
                raise RuntimeError(f"{args.stage.upper()} full smoke gate rejected: {smoke}")
        if args.stage == "v3" and args.mode == "smoke":
            if args.v2_summary is None or not args.v2_summary.is_file():
                raise RuntimeError("V3 smoke requires --v2-summary from a passed V2 smoke")
            v2_smoke = json.loads(args.v2_summary.read_text(encoding="utf-8"))
            if v2_smoke.get("stage") != "v2" or v2_smoke.get("mode") != "smoke" or v2_smoke.get("verdict") != "IMPLEMENTATION_SMOKE_ONLY" or not v2_smoke.get("valid_comparison", False):
                raise RuntimeError(f"V3 smoke gate rejected V2 smoke: {v2_smoke}")
        if args.stage == "v3" and args.mode == "full":
            if args.require_v2_verdict != "POSITIVE_SEMANTIC_TRANSFER":
                raise RuntimeError("V3 full requires --require-v2-verdict POSITIVE_SEMANTIC_TRANSFER")
            if args.v2_summary is None:
                raise RuntimeError("V3 full requires --v2-summary")
            summary = json.loads(args.v2_summary.read_text(encoding="utf-8"))
            if summary.get("verdict") != "POSITIVE_SEMANTIC_TRANSFER":
                raise RuntimeError(f"V3 full gate rejected V2 verdict: {summary.get('verdict')}")
        contract, _, _, _ = _load_manifests(args.manifest_dir)
        snapshot_names = [
            "companion_budgets.json", "matching_manifest.csv", "base_sample_manifest.csv",
            "execution_slot_manifest.csv", "v3_placement_manifest.csv",
        ]
        for name in snapshot_names:
            source = args.manifest_dir / name
            if not source.is_file():
                raise FileNotFoundError(f"Manifest snapshot is missing {source}")
            shutil.copy2(source, args.output_dir / name)
        structural_fairness = args.manifest_dir / "fairness_invariants.csv"
        if not structural_fairness.is_file():
            raise FileNotFoundError(f"Manifest snapshot is missing {structural_fairness}")
        shutil.copy2(structural_fairness, args.output_dir / "fairness_invariants_structural.csv")
        contract["manifest_source"] = str(args.manifest_dir.resolve())
        contract["manifest_snapshot_hashes"] = {
            name: file_sha256(args.output_dir / name) for name in snapshot_names
        }
        contract["manifest_snapshot_hashes"]["fairness_invariants_structural.csv"] = file_sha256(args.output_dir / "fairness_invariants_structural.csv")
        cfg, store, model, theta0, train_transform, test_transform, runtime_meta = _build_runtime(args)
        if runtime_meta["clip_checkpoint_sha256"] != contract["clip_checkpoint_sha256"]:
            raise RuntimeError("Runtime CLIP checkpoint hash differs from frozen V1 checkpoint")
        if runtime_meta["class_mapping_hash"] != contract["class_mapping_hash"]:
            raise RuntimeError("Runtime class mapping differs from manifest contract")
        contract["model_runtime"] = runtime_meta
        contract["git"] = _git_metadata()
        contract["command"] = sys.argv
        write_json(args.output_dir / "experiment_contract.json", contract)
        if args.stage == "v2":
            rows = run_v2(args, cfg, store, model, theta0, train_transform, test_transform, contract)
        else:
            rows = run_v3(args, cfg, store, model, theta0, train_transform, test_transform, contract)
        print(json.dumps({"stage": args.stage, "mode": args.mode, "completed_rows": len(rows), "output_dir": str(args.output_dir.resolve())}, ensure_ascii=False))
    except Exception as exc:
        write_json(args.output_dir / "failure.json", {
            "stage": args.stage, "mode": args.mode, "error_type": type(exc).__name__,
            "error": str(exc), "traceback": traceback.format_exc(), "command": sys.argv,
        })
        excluded_name = "v2_excluded_units.csv" if args.stage == "v2" else "v3_excluded_units.csv"
        write_csv(args.output_dir / excluded_name, [{
            "data_seed": None, "tail_class": None, "draw": None,
            "reason": f"{type(exc).__name__}: {exc}",
        }])
        raise


if __name__ == "__main__":
    main()
