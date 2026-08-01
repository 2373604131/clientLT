"""Small utilities for the simplified CUSP minimal experiment.

This module intentionally avoids CLIP construction and official-test access.
It only saves a compact trainable-parameter dump and builds equal-norm
candidate states from that dump.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch


SCHEMA_VERSION = "cusp_minimal_v1"
METHODS = ("fedavg", "random_reweight", "classwise_weighting", "oracle_cusp")
NUM_RANDOM = 10
RANDOM_SEED = 42


def now_stamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def jsonable(value):
    if isinstance(value, torch.Tensor):
        if value.ndim == 0:
            return value.item()
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return value


def write_json(path: str | Path, payload: Mapping) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jsonable(payload), indent=2, ensure_ascii=False), encoding="utf-8")


def write_csv(path: str | Path, rows: Sequence[Mapping], fields: Sequence[str] | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    if fields is None:
        fields = []
        for row in rows:
            for key in row:
                if key not in fields:
                    fields.append(key)
        fields = fields or ["method"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def sha256_json(payload: Mapping) -> str:
    text = json.dumps(jsonable(payload), sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def trainable_state_dict_to_cpu(model) -> dict[str, torch.Tensor]:
    return {
        name: param.detach().cpu().clone()
        for name, param in model.named_parameters()
        if param.requires_grad and torch.is_floating_point(param)
    }


@dataclass(frozen=True)
class FlatSpec:
    keys: tuple[str, ...]
    shapes: tuple[tuple[int, ...], ...]
    dtypes: tuple[str, ...]
    offsets: tuple[tuple[int, int], ...]

    @property
    def numel(self) -> int:
        return self.offsets[-1][1] if self.offsets else 0

    def as_dict(self) -> dict:
        return {
            "keys": list(self.keys),
            "shapes": [list(shape) for shape in self.shapes],
            "dtypes": list(self.dtypes),
            "offsets": [list(offset) for offset in self.offsets],
            "numel": self.numel,
        }

    @classmethod
    def from_dict(cls, data: Mapping) -> "FlatSpec":
        return cls(
            keys=tuple(data["keys"]),
            shapes=tuple(tuple(shape) for shape in data["shapes"]),
            dtypes=tuple(data["dtypes"]),
            offsets=tuple(tuple(offset) for offset in data["offsets"]),
        )


def make_flat_spec(state: Mapping[str, torch.Tensor], keys: Sequence[str] | None = None) -> FlatSpec:
    keys = tuple(keys or sorted(state.keys()))
    offsets, shapes, dtypes = [], [], []
    cursor = 0
    for key in keys:
        tensor = state[key]
        start = cursor
        cursor += tensor.numel()
        offsets.append((start, cursor))
        shapes.append(tuple(tensor.shape))
        dtypes.append(str(tensor.dtype))
    return FlatSpec(keys=keys, shapes=tuple(shapes), dtypes=tuple(dtypes), offsets=tuple(offsets))


def flatten_state(state: Mapping[str, torch.Tensor], spec: FlatSpec) -> torch.Tensor:
    chunks = [state[key].detach().cpu().to(torch.float64).reshape(-1) for key in spec.keys]
    return torch.cat(chunks) if chunks else torch.empty(0, dtype=torch.float64)


def unflatten_state(vector: torch.Tensor, spec: FlatSpec) -> dict[str, torch.Tensor]:
    vector = torch.as_tensor(vector, dtype=torch.float64).reshape(-1)
    state = {}
    for key, shape, dtype, (start, end) in zip(spec.keys, spec.shapes, spec.dtypes, spec.offsets):
        torch_dtype = getattr(torch, dtype.rsplit(".", 1)[-1])
        state[key] = vector[start:end].reshape(shape).to(torch_dtype)
    return state


def _fedavg_weights(selected: Sequence[int], sample_counts: Sequence[int]) -> torch.Tensor:
    counts = torch.tensor([float(sample_counts[int(client_id)]) for client_id in selected], dtype=torch.float64)
    return counts / counts.sum()


def _lt_class_groups(global_class_counts: Sequence[float], tail_ratio: float) -> tuple[list[int], list[int]]:
    counts = torch.as_tensor(global_class_counts, dtype=torch.float64)
    num_tail = max(1, int(round(len(counts) * float(tail_ratio))))
    order = torch.argsort(counts, descending=True).tolist()
    tail_ids = sorted(order[-num_tail:])
    head_ids = sorted(order[:-num_tail])
    return head_ids, tail_ids


def save_cusp_minimal_dump(
    *,
    output_dir: str | Path,
    args,
    cfg,
    epoch: int,
    global_before: Mapping[str, torch.Tensor],
    global_after: Mapping[str, torch.Tensor],
    local_weights: Sequence[Mapping[str, torch.Tensor]],
    selected_clients: Sequence[int],
    datanumber_client: Sequence[int],
    client_class_counts: Mapping[int, torch.Tensor],
    global_class_counts: torch.Tensor,
    trainer=None,
) -> Path:
    selected = [int(client_id) for client_id in selected_clients]
    run_dir = Path(output_dir) / "cusp_minimal" / f"round_{int(epoch) + 1:03d}"
    run_dir.mkdir(parents=True, exist_ok=True)

    keys = sorted(global_before.keys())
    spec = make_flat_spec(global_before, keys)
    local_states = [{key: local_weights[client_id][key].detach().cpu().clone() for key in keys} for client_id in selected]
    weights = _fedavg_weights(selected, datanumber_client)
    counts = torch.stack([client_class_counts[client_id].detach().cpu().long() for client_id in selected], dim=0)
    head_ids, tail_ids = _lt_class_groups(global_class_counts, args.tail_class_ratio)

    payload = {
        "schema_version": SCHEMA_VERSION,
        "flatten_spec": spec.as_dict(),
        "trainable_keys": keys,
        "global_before_trainable": {key: global_before[key].detach().cpu().clone() for key in keys},
        "global_after_fedavg_trainable": {key: global_after[key].detach().cpu().clone() for key in keys},
        "local_trainable_states": local_states,
        "selected_client_ids": selected,
        "fedavg_weights": weights,
        "client_sample_counts": [int(datanumber_client[client_id]) for client_id in selected],
        "client_class_counts": counts,
        "global_class_counts": global_class_counts.detach().cpu().long(),
        "num_classes": int(len(global_class_counts)),
    }
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "created_at": now_stamp(),
        "communication_round": int(epoch) + 1,
        "candidate_methods": list(METHODS),
        "num_random": NUM_RANDOM,
        "accuracy_scale": "percent",
        "head_class_ids": head_ids,
        "tail_class_ids": tail_ids,
        "resolved_args": vars(args),
        "resolved_config": str(cfg),
        "test_used_before_dump": False,
    }
    torch.save(payload, run_dir / "round_state.pt")
    write_json(run_dir / "metadata.json", metadata)
    print(f"Saved CUSP minimal dump: {run_dir}")
    return run_dir


def load_cusp_minimal_dump(dump_dir: str | Path) -> tuple[dict, dict]:
    dump_dir = Path(dump_dir)
    payload = torch.load(dump_dir / "round_state.pt", map_location="cpu", weights_only=False)
    metadata = json.loads((dump_dir / "metadata.json").read_text(encoding="utf-8"))
    return payload, metadata


def _scale_to_budget(delta: torch.Tensor, budget: float) -> tuple[torch.Tensor, dict]:
    delta = torch.as_tensor(delta, dtype=torch.float64)
    raw_norm = float(delta.norm().item())
    if raw_norm <= 1e-12 or not math.isfinite(raw_norm):
        return torch.zeros_like(delta), {"raw_norm": raw_norm, "final_norm": 0.0, "scale_factor": 0.0}
    scale = float(budget) / raw_norm
    final = delta * scale
    return final, {"raw_norm": raw_norm, "final_norm": float(final.norm().item()), "scale_factor": scale}


def _fedavg_state(local_states: Sequence[Mapping[str, torch.Tensor]], weights: torch.Tensor, spec: FlatSpec) -> dict[str, torch.Tensor]:
    state = {}
    for key in spec.keys:
        stacked = torch.stack([item[key].detach().cpu().to(torch.float64) for item in local_states], dim=0)
        view = [weights.numel()] + [1] * (stacked.ndim - 1)
        state[key] = (stacked * weights.reshape(*view)).sum(dim=0).to(local_states[0][key].dtype)
    return state


def _classwise_delta(payload: Mapping, spec: FlatSpec, theta0: torch.Tensor, budget: float) -> tuple[torch.Tensor, dict]:
    local_states = payload["local_trainable_states"]
    weights = torch.as_tensor(payload["fedavg_weights"], dtype=torch.float64)
    counts = torch.as_tensor(payload["client_class_counts"], dtype=torch.float64)
    state = _fedavg_state(local_states, weights, spec)
    key = "prompt_learner.class_aware_ctx"
    fallback = []
    if key in spec.keys:
        rows = torch.stack([item[key].detach().cpu().to(torch.float64) for item in local_states], dim=0)
        merged = state[key].detach().cpu().to(torch.float64).clone()
        for class_id in range(int(payload["num_classes"])):
            support = counts[:, class_id] > 0
            if not bool(support.any()):
                fallback.append(class_id)
                continue
            class_weights = counts[support, class_id]
            class_weights = class_weights / class_weights.sum()
            selected_rows = rows[support, class_id]
            view = [class_weights.numel()] + [1] * (selected_rows.ndim - 1)
            merged[class_id] = (selected_rows * class_weights.reshape(*view)).sum(dim=0)
        state[key] = merged.to(state[key].dtype)
    delta, report = _scale_to_budget(flatten_state(state, spec) - theta0, budget)
    return delta, {"fallback_class_count": len(fallback), **report}


def _subspace(client_deltas: torch.Tensor, fedavg_delta: torch.Tensor) -> torch.Tensor:
    u, singular, _ = torch.linalg.svd(client_deltas, full_matrices=False)
    keep = int((singular > 1e-12).sum().item())
    q = u[:, :max(keep, 1)]
    residual = fedavg_delta - q @ (q.T @ fedavg_delta)
    if float(residual.norm().item()) > 1e-10:
        q = torch.cat([q, residual[:, None] / residual.norm()], dim=1)
        q, _ = torch.linalg.qr(q, mode="reduced")
    return q


def _oracle_cusp_delta(payload: Mapping, metadata: Mapping, client_deltas: torch.Tensor, fedavg_delta: torch.Tensor, budget: float) -> tuple[torch.Tensor, dict]:
    q = _subspace(client_deltas, fedavg_delta)
    projected_clients = (q.T @ client_deltas).T
    projected_clients = projected_clients / projected_clients.norm(dim=1, keepdim=True).clamp_min(1e-12)

    counts = torch.as_tensor(payload["client_class_counts"], dtype=torch.float64).T
    class_weights = counts / counts.sum(dim=1, keepdim=True).clamp_min(1.0)
    class_utility = class_weights @ projected_clients

    head_ids = [int(x) for x in metadata["head_class_ids"]]
    tail_ids = [int(x) for x in metadata["tail_class_ids"]]
    tail_direction = class_utility[tail_ids].mean(dim=0)
    head_direction = class_utility[head_ids].mean(dim=0) if head_ids else torch.zeros_like(tail_direction)
    fedavg_direction = q.T @ fedavg_delta
    fedavg_direction = fedavg_direction / fedavg_direction.norm().clamp_min(1e-12)

    # A deliberately simple train-only utility direction:
    # move toward tail-supported client updates, stay near FedAvg, mildly avoid
    # directions dominated by head-supported updates.
    u_star = tail_direction + 0.5 * fedavg_direction - 0.25 * head_direction
    raw = q @ u_star
    delta, report = _scale_to_budget(raw, budget)
    report.update({
        "solver": "closed_form_support_proxy",
        "utility_source": "train_client_class_counts",
        "subspace_rank": int(q.shape[1]),
    })
    return delta, report


def build_cusp_candidates(payload: Mapping, metadata: Mapping) -> tuple[dict[str, dict[str, torch.Tensor]], list[dict], dict]:
    spec = FlatSpec.from_dict(payload["flatten_spec"])
    theta0 = flatten_state(payload["global_before_trainable"], spec)
    local_vectors = torch.stack([flatten_state(state, spec) for state in payload["local_trainable_states"]], dim=0)
    weights = torch.as_tensor(payload["fedavg_weights"], dtype=torch.float64)
    client_deltas = (local_vectors - theta0.unsqueeze(0)).T
    fedavg_delta = ((local_vectors - theta0.unsqueeze(0)) * weights[:, None]).sum(dim=0)
    budget = float(fedavg_delta.norm().item())
    if budget <= 1e-12:
        raise RuntimeError("CUSP minimal cannot build equal-norm candidates because the FedAvg update norm is zero")

    candidates: list[tuple[str, str, torch.Tensor, dict]] = [
        ("fedavg", "fedavg", fedavg_delta, {"raw_norm": budget, "final_norm": budget, "scale_factor": 1.0}),
    ]

    rng = np.random.default_rng(RANDOM_SEED)
    for index in range(NUM_RANDOM):
        alpha = torch.tensor(rng.dirichlet(np.ones(local_vectors.shape[0])), dtype=torch.float64)
        delta, report = _scale_to_budget(client_deltas @ alpha, budget)
        report["coefficient_hash"] = sha256_json({"alpha": [round(float(x), 12) for x in alpha.tolist()]})
        candidates.append((f"random_reweight_{index:03d}", "random_reweight", delta, report))

    classwise_delta, classwise_report = _classwise_delta(payload, spec, theta0, budget)
    candidates.append(("classwise_weighting", "classwise_weighting", classwise_delta, classwise_report))

    cusp_delta, cusp_report = _oracle_cusp_delta(payload, metadata, client_deltas, fedavg_delta, budget)
    candidates.append(("oracle_cusp", "oracle_cusp", cusp_delta, cusp_report))

    states = {}
    rows = []
    for candidate_id, method, delta, report in candidates:
        states[candidate_id] = unflatten_state(theta0 + delta, spec)
        rows.append({
            "candidate_id": candidate_id,
            "method": method,
            "raw_norm": float(report["raw_norm"]),
            "final_norm": float(report["final_norm"]),
            "scale_factor": float(report["scale_factor"]),
            "candidate_hash": sha256_json({"delta": [round(float(x), 12) for x in delta.tolist()]}),
            **{key: value for key, value in report.items() if key not in {"raw_norm", "final_norm", "scale_factor"}},
        })
    context = {
        "schema_version": SCHEMA_VERSION,
        "norm_budget": budget,
        "candidate_methods": list(METHODS),
        "num_random": NUM_RANDOM,
        "candidate_frozen_at": now_stamp(),
    }
    return states, rows, context


def freeze_cusp_candidates(output_dir: str | Path, states: Mapping[str, Mapping[str, torch.Tensor]], rows: Sequence[Mapping], context: Mapping) -> dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(dict(states), output_dir / "candidate_states.pt")
    write_csv(output_dir / "candidate_manifest.csv", rows)
    manifest = {**dict(context), "candidate_count": len(rows), "test_accessed": False}
    write_json(output_dir / "candidate_manifest.json", manifest)
    return manifest


def summarize_values(values: Sequence[float]) -> dict:
    values = np.asarray([float(v) for v in values if math.isfinite(float(v))], dtype=float)
    if values.size == 0:
        return {"mean": math.nan, "std": math.nan, "min": math.nan, "p25": math.nan, "median": math.nan, "p75": math.nan, "max": math.nan}
    return {
        "mean": float(values.mean()),
        "std": float(values.std()),
        "min": float(values.min()),
        "p25": float(np.percentile(values, 25)),
        "median": float(np.percentile(values, 50)),
        "p75": float(np.percentile(values, 75)),
        "max": float(values.max()),
    }
