"""On-trajectory ERI round dumps and test-curve logging."""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import Mapping, Sequence

import torch

from utils.cusp_minimal import make_flat_spec


SCHEMA_VERSION = "eri_round_dump_v1"


def _json_default(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    raise TypeError(type(value).__name__)


def _copy_state(state: Mapping[str, torch.Tensor], keys: Sequence[str]) -> dict[str, torch.Tensor]:
    return {key: state[key].detach().cpu().clone() for key in keys}


def _exact_aggregate(
    before: Mapping[str, torch.Tensor],
    local_states: Sequence[Mapping[str, torch.Tensor]],
    keys: Sequence[str],
    weights: Sequence[float],
) -> dict[str, torch.Tensor]:
    result = _copy_state(before, keys)
    for key in keys:
        accumulator = torch.zeros_like(before[key], dtype=torch.float32)
        for state, weight in zip(local_states, weights):
            accumulator.add_(state[key].detach().to(dtype=torch.float32), alpha=float(weight))
        result[key] = accumulator.to(dtype=before[key].dtype).cpu()
    return result


def save_eri_round_dump(
    *,
    output_dir: str | Path,
    args,
    cfg,
    epoch: int,
    global_before: Mapping[str, torch.Tensor],
    global_after: Mapping[str, torch.Tensor],
    local_weights,
    selected_clients: Sequence[int],
    client_sample_counts: Sequence[int | float],
    client_class_counts,
    global_class_counts: torch.Tensor,
    trainable_keys: Sequence[str],
    server_weights: Mapping[int, float],
    aggregation_details: Mapping,
) -> Path:
    """Persist the exact tensors necessary for a post-hoc ERI decomposition.

    The order of all client-aligned arrays is the frozen selected-client order,
    never dictionary order.  A reconstruction check is stored with the dump.
    """
    selected = [int(client_id) for client_id in selected_clients]
    keys = sorted(str(key) for key in trainable_keys)
    if not selected or not keys:
        raise ValueError("ERI dump requires selected clients and non-empty trainable keys")
    local = [_copy_state(local_weights[client_id], keys) for client_id in selected]
    weights = [float(server_weights[client_id]) for client_id in selected]
    if abs(sum(weights) - 1.0) > 1e-6:
        raise RuntimeError(f"ERI server weights do not sum to one: {sum(weights)}")
    before = _copy_state(global_before, keys)
    after = _copy_state(global_after, keys)
    reconstructed = _exact_aggregate(before, local, keys, weights)
    max_abs_error = max(
        float((reconstructed[key].to(torch.float64) - after[key].to(torch.float64)).abs().max().item())
        for key in keys
    )
    run_dir = Path(output_dir) / "eri_closure" / "dumps" / f"round_{int(epoch) + 1:03d}"
    run_dir.mkdir(parents=True, exist_ok=True)
    counts = torch.stack(
        [torch.as_tensor(client_class_counts[client_id]).detach().cpu().long() for client_id in selected]
    )
    payload = {
        "schema_version": SCHEMA_VERSION,
        "flatten_spec": make_flat_spec(before, keys).as_dict(),
        "trainable_keys": keys,
        "global_before_trainable": before,
        "global_after_trainable": after,
        "local_trainable_states": local,
        "selected_client_ids": selected,
        "server_weights": torch.tensor(weights, dtype=torch.float64),
        # Alias keeps generic legacy readers interoperable without claiming FedAvg.
        "fedavg_weights": torch.tensor(weights, dtype=torch.float64),
        "client_sample_counts": [float(client_sample_counts[client_id]) for client_id in selected],
        "client_class_counts": counts,
        "global_class_counts": torch.as_tensor(global_class_counts).detach().cpu().long(),
        "num_classes": int(torch.as_tensor(global_class_counts).numel()),
    }
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "communication_round": int(epoch) + 1,
        "created_at_unix": time.time(),
        "aggregation": str(getattr(args, "cliplora_aggregation", "fedavg")),
        "partition": str(getattr(args, "partition", "")),
        "resolved_args": vars(args),
        "resolved_config": str(cfg),
        "aggregation_details": dict(aggregation_details),
        "reconstruction": {
            "method": "ordered float32 server aggregation",
            "max_abs_error": max_abs_error,
            "tolerance": 1e-5,
            "passed": max_abs_error <= 1e-5,
        },
        "test_used_before_dump": False,
    }
    if not metadata["reconstruction"]["passed"]:
        raise RuntimeError(f"ERI dump reconstruction failed: {metadata['reconstruction']}")
    torch.save(payload, run_dir / "round_state.pt")
    (run_dir / "metadata.json").write_text(
        json.dumps(metadata, default=_json_default, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return run_dir


def load_eri_round_dump(dump_dir: str | Path) -> tuple[dict, dict]:
    root = Path(dump_dir)
    payload = torch.load(root / "round_state.pt", map_location="cpu", weights_only=False)
    metadata = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
    return payload, metadata


def append_eri_test_metrics(output_dir: str | Path, *, epoch: int, class_accuracy, class_log_odds) -> None:
    """Append official-test outcomes only after the normal global evaluation.

    They are outcomes for BFD/retention analysis, never an ERI gradient or
    replay input.  The ERI functional metric remains train-probe only.
    """
    path = Path(output_dir) / "eri_closure" / "test_per_class_metrics.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "communication_round": int(epoch) + 1,
            "class_id": int(class_id),
            "accuracy_percent": float(class_accuracy.get(class_id, float("nan"))),
            "mean_true_log_odds": float(class_log_odds.get(class_id, float("nan"))),
        }
        for class_id in sorted(set(class_accuracy) | set(class_log_odds))
    ]
    if not rows:
        return
    exists = path.exists()
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        if not exists:
            writer.writeheader()
        writer.writerows(rows)
