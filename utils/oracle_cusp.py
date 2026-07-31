"""Small, auditable building blocks for the offline Round-1 Oracle CUSP.

This module deliberately contains no data-loader or CLIP construction.  The
training entry point owns extraction of frozen training features; the offline
script owns model replay.  Keeping the linear algebra here makes it possible
to test the scientific invariants without a GPU or CIFAR download.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Sequence

import numpy as np
import torch


SCHEMA_VERSION = "cusp_round1_v1"
ROUND1_METHODS = ("fedavg", "random_reweight", "classwise_weighting", "oracle_cusp")


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
            "keys": list(self.keys), "shapes": [list(x) for x in self.shapes],
            "dtypes": list(self.dtypes), "offsets": [list(x) for x in self.offsets],
            "numel": self.numel,
        }

    @classmethod
    def from_dict(cls, payload: Mapping) -> "FlatSpec":
        return cls(
            keys=tuple(payload["keys"]),
            shapes=tuple(tuple(x) for x in payload["shapes"]),
            dtypes=tuple(payload["dtypes"]),
            offsets=tuple(tuple(x) for x in payload["offsets"]),
        )


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_json(payload: Mapping) -> str:
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def trainable_float_keys(model) -> list[str]:
    """Return the stable complete PromptFL trainable floating-point key set."""
    return sorted(
        name for name, parameter in model.named_parameters()
        if parameter.requires_grad and (torch.is_floating_point(parameter) or torch.is_complex(parameter))
    )


def make_flat_spec(state: Mapping[str, torch.Tensor], keys: Sequence[str]) -> FlatSpec:
    keys = tuple(str(key) for key in keys)
    if not keys:
        raise ValueError("Oracle CUSP requires at least one trainable floating-point key")
    offsets, shapes, dtypes, cursor = [], [], [], 0
    for key in keys:
        value = state.get(key)
        if not isinstance(value, torch.Tensor) or not torch.is_floating_point(value):
            raise ValueError(f"Oracle CUSP trainable key is missing or non-floating: {key}")
        start, cursor = cursor, cursor + value.numel()
        offsets.append((start, cursor))
        shapes.append(tuple(value.shape))
        dtypes.append(str(value.dtype))
    return FlatSpec(keys, tuple(shapes), tuple(dtypes), tuple(offsets))


def flatten_state(state: Mapping[str, torch.Tensor], spec: FlatSpec) -> torch.Tensor:
    chunks = []
    for key, shape, dtype in zip(spec.keys, spec.shapes, spec.dtypes):
        value = state.get(key)
        if not isinstance(value, torch.Tensor) or tuple(value.shape) != shape or str(value.dtype) != dtype:
            raise ValueError(f"Oracle CUSP state mismatch for key={key}: expected shape={shape}, dtype={dtype}")
        chunks.append(value.detach().cpu().to(torch.float64).reshape(-1))
    return torch.cat(chunks) if chunks else torch.empty(0, dtype=torch.float64)


def unflatten_state(vector: torch.Tensor, spec: FlatSpec) -> dict[str, torch.Tensor]:
    vector = torch.as_tensor(vector, dtype=torch.float64).reshape(-1)
    if vector.numel() != spec.numel:
        raise ValueError(f"Oracle CUSP vector has {vector.numel()} values, expected {spec.numel}")
    result = {}
    for key, shape, dtype, (start, end) in zip(spec.keys, spec.shapes, spec.dtypes, spec.offsets):
        target_dtype = getattr(torch, dtype.rsplit(".", 1)[-1])
        result[key] = vector[start:end].reshape(shape).to(target_dtype)
    return result


def fedavg_delta(global_before: torch.Tensor, local_states: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    weights = torch.as_tensor(weights, dtype=torch.float64).reshape(-1)
    if local_states.ndim != 2 or local_states.shape[0] != weights.numel():
        raise ValueError("local_states must be [clients, parameters] and match fedavg_weights")
    if not torch.isclose(weights.sum(), torch.tensor(1.0, dtype=torch.float64), atol=1e-10):
        raise ValueError(f"FedAvg weights do not sum to one: {weights.sum().item()}")
    return ((local_states - global_before.unsqueeze(0)) * weights.unsqueeze(1)).sum(dim=0)


def verify_round_state(payload: Mapping) -> None:
    spec = FlatSpec.from_dict(payload["flatten_spec"])
    client_ids = [int(x) for x in payload["selected_client_ids"]]
    if client_ids != sorted(client_ids):
        raise RuntimeError("Oracle dump selected_client_ids must be sorted")
    if len(client_ids) != len(set(client_ids)):
        raise RuntimeError("Oracle dump selected_client_ids contain duplicates")
    flatten_state(payload["global_before_trainable"], spec)
    flatten_state(payload["global_after_fedavg_trainable"], spec)
    for state in payload["local_trainable_states"]:
        flatten_state(state, spec)
    weights = torch.as_tensor(payload["fedavg_weights"], dtype=torch.float64)
    if len(payload["local_trainable_states"]) != len(client_ids) or weights.numel() != len(client_ids):
        raise RuntimeError("Oracle dump client/state/weight lengths are not aligned")


def save_round_dump(directory: str | Path, payload: Mapping, metadata: Mapping) -> Path:
    """Save a trusted internal dump only after all structural invariants pass."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    verify_round_state(payload)
    if int(metadata["communication_round"]) < 1:
        raise ValueError("communication_round must be 1-based")
    path = directory / "round_state.pt"
    torch.save(dict(payload), path)
    metadata = {"round_state_sha256": sha256_file(path), **dict(metadata)}
    (directory / "metadata.json").write_text(
        json.dumps({"schema_version": SCHEMA_VERSION, **metadata}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return path


def load_round_dump(directory: str | Path) -> tuple[dict, dict]:
    directory = Path(directory)
    payload = torch.load(directory / "round_state.pt", map_location="cpu", weights_only=False)
    metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
    if metadata.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported Oracle CUSP dump schema")
    verify_round_state(payload)
    return payload, metadata


def validate_train_feature_cache(cache: Mapping, expected_counts: Sequence[int] | torch.Tensor | None = None) -> None:
    if cache.get("source") != "train" or bool(cache.get("test_used_for_utility", True)):
        raise RuntimeError("Oracle utility cache must be source=train and test_used_for_utility=false")
    for key in ("features", "labels", "sample_identity", "class_counts"):
        if key not in cache:
            raise RuntimeError(f"Oracle utility cache missing key: {key}")
    features = torch.as_tensor(cache["features"])
    labels = torch.as_tensor(cache["labels"])
    identity = torch.as_tensor(cache["sample_identity"])
    if features.ndim != 2 or labels.ndim != 1 or identity.ndim != 2 or identity.shape[1] != 2:
        raise RuntimeError("Oracle utility cache has invalid feature/label/identity shape")
    if features.shape[0] != labels.numel() or labels.numel() != identity.shape[0]:
        raise RuntimeError("Oracle utility cache feature/label/identity lengths differ")
    identities = [tuple(int(x) for x in row) for row in identity.cpu().tolist()]
    if identities != sorted(identities):
        raise RuntimeError("Oracle utility cache sample_identity must be sorted")
    if len(identities) != len(set(identities)):
        raise RuntimeError("Oracle utility cache sample_identity contains duplicates")
    class_counts = torch.as_tensor(cache["class_counts"], dtype=torch.long).cpu()
    observed = torch.bincount(labels.to(torch.long).cpu(), minlength=class_counts.numel())
    if not torch.equal(class_counts, observed):
        raise RuntimeError("Oracle utility cache class_counts do not match labels")
    if expected_counts is not None:
        expected = torch.as_tensor(expected_counts, dtype=torch.long).cpu()
        if not torch.equal(class_counts, expected):
            raise RuntimeError("Oracle utility cache class_counts do not match expected global train counts")


def subspace_from_updates(deltas: torch.Tensor, energy: float = 0.999, eps: float = 1e-12) -> dict:
    """Return Q and append a normalized FedAvg residual if truncation loses it."""
    deltas = torch.as_tensor(deltas, dtype=torch.float64)
    if deltas.ndim != 2:
        raise ValueError("deltas must be [parameters, clients]")
    u, singular, _ = torch.linalg.svd(deltas, full_matrices=False)
    energy_values = singular.square()
    total = float(energy_values.sum().item())
    if total <= eps:
        raise ValueError("all client updates are zero")
    rank = int(torch.searchsorted(torch.cumsum(energy_values, 0) / total, torch.tensor(energy)).item()) + 1
    return {"Q": u[:, :rank], "singular_values": singular, "rank": rank,
            "energy_retained": float(energy_values[:rank].sum().item() / total), "fedavg_direction_appended": False}


def ensure_fedavg_in_subspace(info: dict, delta_fedavg: torch.Tensor, eps: float = 1e-10) -> dict:
    q = info["Q"]
    residual = delta_fedavg - q @ (q.T @ delta_fedavg)
    relative_error = float(residual.norm().item() / max(delta_fedavg.norm().item(), eps))
    if relative_error > 1e-8 and residual.norm() > eps:
        q = torch.cat([q, residual.unsqueeze(1) / residual.norm()], dim=1)
        q, _ = torch.linalg.qr(q, mode="reduced")
        info = {**info, "Q": q, "rank": q.shape[1], "fedavg_direction_appended": True}
    info["fedavg_projection_relative_error"] = relative_error
    return info


def finite_difference_utility(margin_fn: Callable[[torch.Tensor], torch.Tensor], theta0: torch.Tensor, q: torch.Tensor,
                              epsilon: float = 1e-3) -> tuple[torch.Tensor, dict]:
    """Central difference A[c,k]; margin_fn must only close over train cache."""
    columns, half_columns = [], []
    for direction in q.T:
        plus, minus = margin_fn(theta0 + epsilon * direction), margin_fn(theta0 - epsilon * direction)
        half_plus = margin_fn(theta0 + epsilon * 0.5 * direction)
        half_minus = margin_fn(theta0 - epsilon * 0.5 * direction)
        columns.append((plus - minus) / (2 * epsilon))
        half_columns.append((half_plus - half_minus) / epsilon)
    a = torch.stack(columns, dim=1).to(torch.float64)
    a_half = torch.stack(half_columns, dim=1).to(torch.float64)
    finite = torch.isfinite(a).all(dim=1)
    sign_agreement = float(((torch.sign(a[finite]) == torch.sign(a_half[finite])).double().mean()).item()) if finite.any() else 0.0
    relative = float((a - a_half).norm().item() / max(a.norm().item(), 1e-12))
    return a, {"epsilon": epsilon, "sign_agreement": sign_agreement, "relative_difference": relative,
               "sign_agreement_threshold": 0.95, "relative_difference_threshold": 0.10,
               "stable": sign_agreement >= 0.95 and math.isfinite(relative) and relative <= 0.10}


def scale_to_budget(delta: torch.Tensor, budget: float, eps: float = 1e-12) -> tuple[torch.Tensor | None, dict]:
    """Scale a valid nonzero direction to exactly the FedAvg update norm."""
    delta = torch.as_tensor(delta, dtype=torch.float64)
    raw_norm = float(delta.norm().item())
    valid = bool(math.isfinite(raw_norm) and raw_norm > eps and math.isfinite(float(budget)) and float(budget) > eps)
    if not valid:
        return None, {"valid": False, "raw_norm": raw_norm, "final_norm": math.nan, "scale_factor": math.nan}
    scale = float(budget) / raw_norm
    final = delta * scale
    return final, {"valid": True, "raw_norm": raw_norm, "final_norm": float(final.norm().item()), "scale_factor": scale}


def random_reweight(deltas: torch.Tensor, budget: float, count: int = 10, seed: int = 42) -> list[dict]:
    rng = np.random.default_rng(seed)
    out = []
    for index in range(count):
        alpha = torch.tensor(rng.dirichlet(np.ones(deltas.shape[1])), dtype=torch.float64)
        delta, norm_report = scale_to_budget(deltas @ alpha, budget)
        out.append({"index": index, "alpha": alpha, "delta": delta, **norm_report,
                    "coefficient_hash": hashlib.sha256(alpha.numpy().tobytes()).hexdigest()})
    return out


def summarize_values(values: Sequence[float]) -> dict:
    array = np.asarray([float(x) for x in values], dtype=float)
    if array.size == 0:
        return {"mean": math.nan, "std": math.nan, "min": math.nan, "p25": math.nan, "median": math.nan, "p75": math.nan, "max": math.nan}
    return {
        "mean": float(np.mean(array)),
        "std": float(np.std(array)),
        "min": float(np.min(array)),
        "p25": float(np.percentile(array, 25)),
        "median": float(np.percentile(array, 50)),
        "p75": float(np.percentile(array, 75)),
        "max": float(np.max(array)),
    }


def classwise_weighting_delta(
    global_before: Mapping[str, torch.Tensor],
    local_states: Sequence[Mapping[str, torch.Tensor]],
    weights: Sequence[float] | torch.Tensor,
    client_class_counts: torch.Tensor,
    spec: FlatSpec,
    num_classes: int,
    budget: float,
    classwise_key: str = "prompt_learner.class_aware_ctx",
) -> tuple[torch.Tensor, dict]:
    """Build the Round-1 class-wise support-client baseline.

    Only rows of ``prompt_learner.class_aware_ctx`` are reweighted class-wise.
    ``general_ctx`` and any other trainable keys remain the ordinary FedAvg
    state. Missing support for a class falls back to global-before for that row.
    """
    weights = torch.as_tensor(weights, dtype=torch.float64).reshape(-1)
    counts = torch.as_tensor(client_class_counts, dtype=torch.float64)
    if counts.shape[0] != weights.numel() or counts.shape[1] != int(num_classes):
        raise ValueError("client_class_counts must be [clients, num_classes]")
    if len(local_states) != weights.numel():
        raise ValueError("local_states length must match fedavg weights")

    before = {key: value.detach().cpu().clone() for key, value in global_before.items()}
    fedavg_state = {}
    for key in spec.keys:
        stacked = torch.stack([state[key].detach().cpu().to(torch.float64) for state in local_states])
        fedavg_state[key] = (stacked * weights.reshape([-1] + [1] * (stacked.ndim - 1))).sum(dim=0).to(before[key].dtype)

    fallback_classes: list[int] = []
    if classwise_key in spec.keys:
        before_rows = before[classwise_key].detach().cpu().to(torch.float64)
        if before_rows.ndim < 1 or before_rows.shape[0] != int(num_classes):
            raise ValueError(f"{classwise_key} first dimension must equal num_classes")
        classwise_rows = before_rows.clone()
        local_rows = torch.stack([state[classwise_key].detach().cpu().to(torch.float64) for state in local_states])
        for class_id in range(int(num_classes)):
            support = counts[:, class_id] > 0
            if not bool(support.any()):
                fallback_classes.append(class_id)
                continue
            support_weights = weights[support]
            support_weights = support_weights / support_weights.sum()
            selected_rows = local_rows[support, class_id]
            view_shape = [support_weights.numel()] + [1] * (selected_rows.ndim - 1)
            classwise_rows[class_id] = (selected_rows * support_weights.reshape(*view_shape)).sum(dim=0)
        fedavg_state[classwise_key] = classwise_rows.to(before[classwise_key].dtype)

    delta = flatten_state(fedavg_state, spec) - flatten_state(before, spec)
    scaled, norm_report = scale_to_budget(delta, float(budget))
    return scaled, {
        "classwise_key": classwise_key,
        "fallback_class_ids": fallback_classes,
        "fallback_class_count": len(fallback_classes),
        **norm_report,
    }


def solve_cusp(a: torch.Tensor, u_fedavg: torch.Tensor, head_mask: torch.Tensor, *, lam=0.1, mu=10.0, tolerance=0.05):
    """Optional CVXPY solve. No hidden fallback is ever substituted for CUSP."""
    try:
        import cvxpy as cp
    except ImportError:
        return None, {"status": "dependency_missing", "failure_reason": "cvxpy is not installed"}

    a_np, u_np = a.detach().cpu().numpy(), u_fedavg.detach().cpu().numpy()
    n, r = a_np.shape
    u, tau, slack = cp.Variable(r), cp.Variable(), cp.Variable(n, nonneg=True)
    constraints = [a_np @ u + slack >= tau, cp.norm(u, 2) <= 1, tau <= 1]
    if bool(head_mask.any()):
        constraints.append(cp.sum(a_np[head_mask.cpu().numpy()] @ u) / int(head_mask.sum()) >=
                           float((a_np[head_mask.cpu().numpy()] @ u_np).mean() - tolerance))
    constraints.append(cp.sum(a_np @ u) / n >= float((a_np @ u_np).mean() - tolerance))
    problem = cp.Problem(cp.Maximize(tau - lam * cp.sum_squares(u - u_np) - mu * cp.sum(slack) / n), constraints)
    try:
        problem.solve()
    except cp.SolverError as exc:
        return None, {"status": "solver_failed", "failure_reason": str(exc)}
    except Exception as exc:
        return None, {"status": "solver_failed", "failure_reason": str(exc)}

    report = {"status": problem.status, "objective": problem.value, "tau": None if tau.value is None else float(tau.value),
              "slack_sum": None if slack.value is None else float(np.sum(slack.value)), "lambda": lam, "mu": mu, "utility_tolerance": tolerance}
    if problem.status in {"infeasible", "unbounded", "infeasible_inaccurate", "unbounded_inaccurate"}:
        return None, {**report, "failure_reason": problem.status}
    if problem.status not in {"optimal", "optimal_inaccurate"} or u.value is None:
        return None, {**report, "failure_reason": "solver did not return a usable vector"}
    if not np.all(np.isfinite(u.value)):
        return None, {**report, "status": "numerical_exception", "failure_reason": "non-finite solution"}
    return torch.tensor(u.value, dtype=torch.float64), report


def write_json(path: str | Path, payload: Mapping) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
