"""Aggregation policies for the standalone federated ClipLoRA experiments.

The support-normalized policy is an oracle baseline derived directly from
Experiment D.  Experiment D creates one support-normalized counterfactual per
tail class.  End-to-end training can keep only one global LoRA state, so we
average those per-class client-weight distributions before aggregating the
single global update.
"""

from __future__ import annotations

import copy
import csv
import math
from pathlib import Path
from typing import Mapping, Sequence

import torch


LORA_AGGREGATION_MODES = ("fedavg", "support_normalized")


def _selected_client_ids(selected_clients: Sequence[int]) -> list[int]:
    selected = [int(client_id) for client_id in selected_clients]
    if not selected:
        raise ValueError("LoRA aggregation requires at least one selected client")
    if len(set(selected)) != len(selected):
        raise ValueError(f"selected_clients contains duplicates: {selected}")
    return selected


def _sample_count(datanumber_client: Sequence[float], client_id: int) -> float:
    count = float(datanumber_client[int(client_id)])
    if not math.isfinite(count) or count < 0:
        raise ValueError(f"Invalid sample count for client {client_id}: {count}")
    return count


def _class_count(client_class_counts, client_id: int, class_id: int) -> float:
    if isinstance(client_class_counts, Mapping):
        row = client_class_counts[int(client_id)]
    else:
        row = client_class_counts[int(client_id)]
    return float(torch.as_tensor(row)[int(class_id)].item())


def sample_weighted_client_weights(
    selected_clients: Sequence[int],
    datanumber_client: Sequence[float],
) -> dict[int, float]:
    """Return ordinary sample-weighted FedAvg coefficients."""
    selected = _selected_client_ids(selected_clients)
    total = sum(_sample_count(datanumber_client, client_id) for client_id in selected)
    if total <= 0:
        raise ValueError("Selected clients have zero total samples")
    return {
        client_id: _sample_count(datanumber_client, client_id) / total
        for client_id in selected
    }


def support_normalized_client_weights(
    selected_clients: Sequence[int],
    datanumber_client: Sequence[float],
    client_class_counts,
    tail_class_ids: Sequence[int],
) -> tuple[dict[int, float], dict]:
    """Build one end-to-end weight vector from class-wise support weights.

    For every tail class ``c``, clients containing ``c`` are normalized by
    their total local sample counts, exactly as in Experiment D.  The final
    single-model coefficient is the mean of these class-wise distributions:

        w_i = mean_c 1[i supports c] * n_i / sum_j 1[j supports c] * n_j

    Each tail class therefore receives one equal vote.  Classes absent from a
    partially participating round are reported and skipped; with the intended
    full-participation protocol every tail class must be covered.
    """
    selected = _selected_client_ids(selected_clients)
    tail_classes = [int(class_id) for class_id in tail_class_ids]
    if not tail_classes:
        raise ValueError("support_normalized aggregation requires tail classes")
    if len(set(tail_classes)) != len(tail_classes):
        raise ValueError(f"tail_class_ids contains duplicates: {tail_classes}")

    weights = {client_id: 0.0 for client_id in selected}
    support_counts = {client_id: 0 for client_id in selected}
    covered_classes = []
    uncovered_classes = []

    for class_id in tail_classes:
        support = [
            client_id
            for client_id in selected
            if _class_count(client_class_counts, client_id, class_id) > 0
        ]
        denominator = sum(
            _sample_count(datanumber_client, client_id) for client_id in support
        )
        if denominator <= 0:
            uncovered_classes.append(class_id)
            continue

        covered_classes.append(class_id)
        for client_id in support:
            support_counts[client_id] += 1
            weights[client_id] += _sample_count(datanumber_client, client_id) / denominator

    if not covered_classes:
        raise ValueError(
            "No selected client supports any requested tail class; "
            "cannot run support-normalized aggregation"
        )

    divisor = float(len(covered_classes))
    weights = {client_id: value / divisor for client_id, value in weights.items()}
    weight_sum = sum(weights.values())
    if not math.isclose(weight_sum, 1.0, rel_tol=1e-9, abs_tol=1e-9):
        raise RuntimeError(f"support-normalized weights sum to {weight_sum}, expected 1")

    return weights, {
        "tail_class_count": len(tail_classes),
        "covered_tail_class_count": len(covered_classes),
        "covered_tail_classes": covered_classes,
        "uncovered_tail_classes": uncovered_classes,
        "client_supported_tail_classes": support_counts,
    }


def compute_lora_aggregation_weights(
    mode: str,
    selected_clients: Sequence[int],
    datanumber_client: Sequence[float],
    *,
    client_class_counts=None,
    tail_class_ids: Sequence[int] | None = None,
) -> tuple[dict[int, float], dict]:
    """Resolve a named ClipLoRA aggregation policy."""
    mode = str(mode).lower()
    if mode not in LORA_AGGREGATION_MODES:
        raise ValueError(
            f"Unknown ClipLoRA aggregation mode {mode!r}; "
            f"choose from {LORA_AGGREGATION_MODES}"
        )

    selected = _selected_client_ids(selected_clients)
    if mode == "fedavg":
        weights = sample_weighted_client_weights(selected, datanumber_client)
        support_counts = {client_id: 0 for client_id in selected}
        if client_class_counts is not None and tail_class_ids is not None:
            support_counts = {
                client_id: sum(
                    _class_count(client_class_counts, client_id, class_id) > 0
                    for class_id in tail_class_ids
                )
                for client_id in selected
            }
        return weights, {
            "tail_class_count": len(tail_class_ids or ()),
            "covered_tail_class_count": len(tail_class_ids or ()),
            "covered_tail_classes": [int(x) for x in (tail_class_ids or ())],
            "uncovered_tail_classes": [],
            "client_supported_tail_classes": support_counts,
        }

    if client_class_counts is None:
        raise ValueError("support_normalized aggregation requires client_class_counts")
    if tail_class_ids is None:
        raise ValueError("support_normalized aggregation requires tail_class_ids")
    return support_normalized_client_weights(
        selected,
        datanumber_client,
        client_class_counts,
        tail_class_ids,
    )


def aggregate_lora_state(
    global_state: Mapping[str, torch.Tensor],
    local_states,
    selected_clients: Sequence[int],
    trainable_keys: Sequence[str],
    client_weights: Mapping[int, float],
) -> dict:
    """Aggregate only LoRA tensors and preserve the frozen CLIP state."""
    selected = _selected_client_ids(selected_clients)
    weights = {client_id: float(client_weights[client_id]) for client_id in selected}
    if any(not math.isfinite(weight) or weight < 0 for weight in weights.values()):
        raise ValueError(f"Invalid LoRA aggregation weights: {weights}")
    weight_sum = sum(weights.values())
    if not math.isclose(weight_sum, 1.0, rel_tol=1e-7, abs_tol=1e-7):
        raise ValueError(f"LoRA aggregation weights sum to {weight_sum}, expected 1")

    aggregated = copy.deepcopy(global_state)
    for key in sorted(trainable_keys):
        if "lora_" not in key:
            raise ValueError(f"ClipLora aggregation received a non-LoRA key: {key}")
        if key not in aggregated:
            raise KeyError(f"Global ClipLora state is missing trainable key: {key}")

        reference = aggregated[key]
        accumulator = torch.zeros_like(reference, dtype=torch.float32)
        for client_id in selected:
            if key not in local_states[client_id]:
                raise KeyError(f"Client {client_id} ClipLora state is missing key: {key}")
            client_value = local_states[client_id][key].detach().to(
                device=reference.device,
                dtype=torch.float32,
            )
            accumulator.add_(client_value, alpha=weights[client_id])
        aggregated[key] = accumulator.to(dtype=reference.dtype)
    return aggregated


def append_lora_aggregation_diagnostics(
    output_dir,
    *,
    epoch: int,
    partition: str,
    mode: str,
    selected_clients: Sequence[int],
    datanumber_client: Sequence[float],
    client_weights: Mapping[int, float],
    details: Mapping,
) -> None:
    """Write auditable per-round client weights for the aggregation baseline."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    selected = _selected_client_ids(selected_clients)
    support_counts = details.get("client_supported_tail_classes", {})

    weight_rows = [
        {
            "epoch_index": int(epoch),
            "communication_round": int(epoch) + 1,
            "partition": str(partition),
            "aggregation": str(mode),
            "client_id": client_id,
            "client_num_samples": _sample_count(datanumber_client, client_id),
            "num_supported_tail_classes": int(support_counts.get(client_id, 0)),
            "aggregation_weight": float(client_weights[client_id]),
        }
        for client_id in selected
    ]
    _append_csv_rows(output_path / "lora_aggregation_weights.csv", weight_rows)

    values = [float(client_weights[client_id]) for client_id in selected]
    active = [value for value in values if value > 0]
    effective_clients = 1.0 / sum(value * value for value in values)
    summary_row = {
        "epoch_index": int(epoch),
        "communication_round": int(epoch) + 1,
        "partition": str(partition),
        "aggregation": str(mode),
        "num_selected_clients": len(selected),
        "num_active_clients": len(active),
        "weight_sum": sum(values),
        "effective_num_clients": effective_clients,
        "max_weight": max(values),
        "min_positive_weight": min(active) if active else math.nan,
        "tail_class_count": int(details.get("tail_class_count", 0)),
        "covered_tail_class_count": int(details.get("covered_tail_class_count", 0)),
        "uncovered_tail_classes": ",".join(
            str(x) for x in details.get("uncovered_tail_classes", [])
        ),
    }
    _append_csv_rows(output_path / "lora_aggregation_summary.csv", [summary_row])


def _append_csv_rows(path: Path, rows: Sequence[Mapping]) -> None:
    if not rows:
        return
    exists = path.exists()
    fieldnames = list(rows[0].keys())
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerows(rows)
