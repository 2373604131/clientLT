import copy
import json
import math
import os
import random
from dataclasses import asdict, is_dataclass
from typing import Dict, Iterable, List, Mapping, Sequence

import numpy as np
import torch


def parse_int_list(value, default=None):
    if value is None:
        return list(default or [])
    if isinstance(value, (list, tuple)):
        return [int(v) for v in value]
    value = str(value).strip()
    if not value:
        return list(default or [])
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def str2bool(value):
    if isinstance(value, bool):
        return value
    value = str(value).strip().lower()
    if value in {"1", "true", "yes", "y", "on"}:
        return True
    if value in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"Cannot parse boolean value: {value}")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def robust_normalize(x, eps=1e-12, neutral=0.5):
    x = torch.as_tensor(x, dtype=torch.float32)
    x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    if x.numel() == 0:
        return x
    lo = x.min()
    hi = x.max()
    if torch.abs(hi - lo) < eps:
        return torch.full_like(x, float(neutral))
    out = (x - lo) / (hi - lo + eps)
    return torch.nan_to_num(out.clamp(0.0, 1.0), nan=float(neutral))


def tensor_stats(x):
    x = torch.as_tensor(x, dtype=torch.float32)
    if x.numel() == 0:
        return {"min": 0.0, "max": 0.0, "mean": 0.0}
    return {
        "min": float(torch.nan_to_num(x).min().item()),
        "max": float(torch.nan_to_num(x).max().item()),
        "mean": float(torch.nan_to_num(x).mean().item()),
    }


def as_cpu_state_dict(model_or_state):
    state = model_or_state.state_dict() if hasattr(model_or_state, "state_dict") else model_or_state
    return {k: v.detach().cpu().clone() for k, v in state.items()}


def weighted_average_tensors(values, weights, eps=1e-12):
    weights = torch.as_tensor(weights, dtype=torch.float64)
    total = weights.sum().clamp_min(float(eps))
    out = torch.zeros_like(values[0].detach().cpu(), dtype=torch.float32)
    for value, weight in zip(values, weights):
        out += (weight / total).float() * value.detach().cpu().float()
    return out.to(dtype=values[0].dtype)


def group_update_norm(old_state, new_state, keys):
    total = torch.tensor(0.0)
    for key in keys:
        if key not in old_state or key not in new_state:
            continue
        delta = new_state[key].detach().cpu().float() - old_state[key].detach().cpu().float()
        total = total + delta.pow(2).sum()
    return float(total.sqrt().item())


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


def to_jsonable(obj):
    if is_dataclass(obj):
        return to_jsonable(asdict(obj))
    if isinstance(obj, torch.Tensor):
        if obj.numel() == 1:
            return float(obj.detach().cpu().item())
        return obj.detach().cpu().tolist()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, Mapping):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    return obj


def write_json(path, payload):
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(to_jsonable(payload), handle, indent=2, ensure_ascii=False)


def append_jsonl(path, payload):
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(to_jsonable(payload), ensure_ascii=False) + "\n")
        handle.flush()


def class_counts_from_labels(labels, num_classes):
    labels = torch.as_tensor(labels, dtype=torch.long)
    if labels.numel() == 0:
        return torch.zeros(num_classes, dtype=torch.float32)
    return torch.bincount(labels.cpu(), minlength=num_classes).float()


def is_in_classes(labels, classes):
    labels = torch.as_tensor(labels, dtype=torch.long)
    if classes is None or len(classes) == 0:
        return torch.zeros_like(labels, dtype=torch.bool)
    class_tensor = torch.as_tensor(sorted(set(int(c) for c in classes)), device=labels.device, dtype=labels.dtype)
    return (labels[:, None] == class_tensor[None, :]).any(dim=1)


def split_head_medium_tail(train_class_counts, tail_ratio=0.2, head_ratio=None):
    del head_ratio
    counts = torch.as_tensor(train_class_counts, dtype=torch.float32)
    order = torch.argsort(counts, descending=True).tolist()
    num_classes = int(counts.numel())
    n_tail = max(1, int(round(num_classes * tail_ratio)))
    n_tail = min(n_tail, num_classes)
    tail = set(order[-n_tail:])
    non_tail = set(order[:-n_tail])
    return {"non_tail": sorted(non_tail), "tail": sorted(tail)}


def parameter_manifest(model, shared_keys, gate_keys, tail_keys, frozen_keys):
    shared_keys = set(shared_keys)
    gate_keys = set(gate_keys)
    tail_keys = set(tail_keys)
    frozen_keys = set(frozen_keys)
    rows = []
    for name, param in model.named_parameters():
        if name in shared_keys:
            group = "shared"
        elif name in gate_keys:
            group = "gate"
        elif name in tail_keys:
            group = "tail"
        elif name in frozen_keys:
            group = "frozen"
        else:
            group = "unclassified"
        rows.append({
            "name": name,
            "shape": list(param.shape),
            "numel": int(param.numel()),
            "requires_grad": bool(param.requires_grad),
            "group": group,
        })
    return rows
