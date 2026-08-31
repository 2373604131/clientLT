"""Online class-separable aggregation and D4-A diagnostics.

The method implemented here deliberately has two independent server paths:

* shared LoRA tensors use ordinary sample-weighted FedAvg;
* each class-residual row is aggregated only from clients that observed that
  class in their local training partition.

No validation/test utility is consumed by the aggregation rule.  D4-A is a
read-only diagnostic logger and never changes the model or client weights.
"""

from __future__ import annotations

import copy
import csv
import math
from pathlib import Path
from typing import Mapping, Sequence

import torch


SCA_WEIGHT_KEY = "class_residual.weight"
SCA_BIAS_KEY = "class_residual.bias"
RESIDUAL_AGGREGATION_MODES = ("class_separable", "fedavg")


def _selected_ids(selected_clients: Sequence[int]) -> list[int]:
    selected = [int(client_id) for client_id in selected_clients]
    if not selected:
        raise ValueError("Class-separable aggregation requires selected clients")
    if len(selected) != len(set(selected)):
        raise ValueError(f"Duplicate selected clients: {selected}")
    return selected


def _count(client_class_counts, client_id: int, class_id: int) -> float:
    row = client_class_counts[int(client_id)]
    value = float(torch.as_tensor(row)[int(class_id)].item())
    if not math.isfinite(value) or value < 0:
        raise ValueError(
            f"Invalid class count for client={client_id}, class={class_id}: {value}"
        )
    return value


def class_supporters(
    selected_clients: Sequence[int],
    client_class_counts,
    class_id: int,
    min_fraction: float = 0.0,
) -> list[int]:
    """Return selected clients allowed to upload one class row.

    ``min_fraction=0`` means positive support (at least one local example).
    A positive threshold additionally requires the class to occupy at least
    that fraction of the client's local data.
    """
    if not 0.0 <= float(min_fraction) < 1.0:
        raise ValueError("min_fraction must be in [0, 1)")
    supporters = []
    for client_id in _selected_ids(selected_clients):
        row = torch.as_tensor(client_class_counts[int(client_id)])
        class_count = _count(client_class_counts, client_id, class_id)
        total = float(row.sum().item())
        fraction = class_count / total if total > 0 else 0.0
        if class_count > 0 and fraction >= float(min_fraction):
            supporters.append(client_id)
    return supporters


def aggregate_class_residual_rows(
    state_after_shared_fedavg: Mapping[str, torch.Tensor],
    previous_global_state: Mapping[str, torch.Tensor],
    local_states,
    selected_clients: Sequence[int],
    client_class_counts,
    class_ids: Sequence[int],
    *,
    min_fraction: float = 0.0,
    weighting: str = "class_count",
    weight_key: str = SCA_WEIGHT_KEY,
    bias_key: str = SCA_BIAS_KEY,
) -> tuple[dict, list[dict]]:
    """Aggregate class rows online and retain the previous row if unsupported.

    ``class_count`` weighting is the class-wise analogue of FedAvg:
    ``n_kc / sum_j(n_jc)``. ``uniform`` is retained as a clean CAPT-like
    ablation.  Both rules depend only on local training support metadata.
    """
    selected = _selected_ids(selected_clients)
    weighting = str(weighting).lower()
    if weighting not in {"class_count", "uniform"}:
        raise ValueError("weighting must be 'class_count' or 'uniform'")
    if weight_key not in previous_global_state or weight_key not in state_after_shared_fedavg:
        raise KeyError(f"Missing class residual tensor: {weight_key}")

    aggregated = copy.deepcopy(state_after_shared_fedavg)
    previous_weight = previous_global_state[weight_key]
    if previous_weight.ndim != 2:
        raise ValueError(f"{weight_key} must be [classes, feature_dim]")
    has_bias = bias_key in previous_global_state
    diagnostics = []

    for class_id in [int(value) for value in class_ids]:
        if class_id < 0 or class_id >= previous_weight.shape[0]:
            raise IndexError(f"Class id {class_id} is outside residual table")
        supporters = class_supporters(
            selected, client_class_counts, class_id, min_fraction=min_fraction
        )
        retained = not supporters
        coefficients: dict[int, float] = {}
        if supporters:
            if weighting == "uniform":
                coefficients = {
                    client_id: 1.0 / len(supporters) for client_id in supporters
                }
            else:
                denominator = sum(
                    _count(client_class_counts, client_id, class_id)
                    for client_id in supporters
                )
                if denominator <= 0:
                    raise RuntimeError("Positive supporters have zero class-count mass")
                coefficients = {
                    client_id: _count(client_class_counts, client_id, class_id)
                    / denominator
                    for client_id in supporters
                }

            reference = previous_weight[class_id]
            row = torch.zeros_like(reference, dtype=torch.float32)
            for client_id, coefficient in coefficients.items():
                local_value = local_states[client_id][weight_key][class_id]
                row.add_(
                    local_value.detach().to(reference.device, torch.float32),
                    alpha=float(coefficient),
                )
            aggregated[weight_key][class_id] = row.to(reference.dtype)

            if has_bias:
                reference_bias = previous_global_state[bias_key][class_id]
                value = torch.zeros_like(reference_bias, dtype=torch.float32)
                for client_id, coefficient in coefficients.items():
                    value.add_(
                        local_states[client_id][bias_key][class_id]
                        .detach()
                        .to(reference_bias.device, torch.float32),
                        alpha=float(coefficient),
                    )
                aggregated[bias_key][class_id] = value.to(reference_bias.dtype)
        else:
            # Persistent-by-construction, but not yet a distillation component:
            # an unsupported row simply does not receive a destructive update.
            aggregated[weight_key][class_id] = previous_global_state[weight_key][class_id]
            if has_bias:
                aggregated[bias_key][class_id] = previous_global_state[bias_key][class_id]

        row_delta = (
            aggregated[weight_key][class_id].detach().float().cpu()
            - previous_global_state[weight_key][class_id].detach().float().cpu()
        )
        diagnostics.append(
            {
                "class_id": class_id,
                "supporter_count": len(supporters),
                "supporter_ids": ",".join(str(value) for value in supporters),
                "support_mass": sum(
                    _count(client_class_counts, client_id, class_id)
                    for client_id in supporters
                ),
                "retained_previous_row": retained,
                "row_delta_norm": float(row_delta.norm().item()),
                "max_supporter_weight": max(coefficients.values()) if coefficients else 0.0,
            }
        )

    return aggregated, diagnostics


def aggregate_class_residual_fedavg_rows(
    state_after_shared_fedavg: Mapping[str, torch.Tensor],
    previous_global_state: Mapping[str, torch.Tensor],
    local_states,
    selected_clients: Sequence[int],
    client_class_counts,
    class_ids: Sequence[int],
    *,
    client_weights: Mapping[int, float],
    weight_key: str = SCA_WEIGHT_KEY,
    bias_key: str = SCA_BIAS_KEY,
) -> tuple[dict, list[dict]]:
    """Aggregate the same residual rows with ordinary scalar FedAvg.

    This is the architecture-matched control for class-separable aggregation:
    local model, residual head, active classes, gradient mask, optimizer, and
    client schedule are identical.  The only intervention is that every
    selected client's residual table receives its ordinary sample-count
    FedAvg coefficient, irrespective of class support.
    """
    selected = _selected_ids(selected_clients)
    weights = {client_id: float(client_weights[client_id]) for client_id in selected}
    if any(not math.isfinite(value) or value < 0 for value in weights.values()):
        raise ValueError(f"Invalid residual FedAvg weights: {weights}")
    weight_sum = sum(weights.values())
    if not math.isclose(weight_sum, 1.0, rel_tol=1e-7, abs_tol=1e-7):
        raise ValueError(
            f"Residual FedAvg weights sum to {weight_sum}, expected 1"
        )
    if weight_key not in previous_global_state or weight_key not in state_after_shared_fedavg:
        raise KeyError(f"Missing class residual tensor: {weight_key}")

    aggregated = copy.deepcopy(state_after_shared_fedavg)
    previous_weight = previous_global_state[weight_key]
    if previous_weight.ndim != 2:
        raise ValueError(f"{weight_key} must be [classes, feature_dim]")
    has_bias = bias_key in previous_global_state
    diagnostics = []

    for class_id in [int(value) for value in class_ids]:
        if class_id < 0 or class_id >= previous_weight.shape[0]:
            raise IndexError(f"Class id {class_id} is outside residual table")
        reference = previous_weight[class_id]
        row = torch.zeros_like(reference, dtype=torch.float32)
        for client_id in selected:
            if weight_key not in local_states[client_id]:
                raise KeyError(
                    f"Client {client_id} state is missing {weight_key}"
                )
            row.add_(
                local_states[client_id][weight_key][class_id]
                .detach()
                .to(reference.device, torch.float32),
                alpha=weights[client_id],
            )
        aggregated[weight_key][class_id] = row.to(reference.dtype)

        if has_bias:
            reference_bias = previous_global_state[bias_key][class_id]
            value = torch.zeros_like(reference_bias, dtype=torch.float32)
            for client_id in selected:
                value.add_(
                    local_states[client_id][bias_key][class_id]
                    .detach()
                    .to(reference_bias.device, torch.float32),
                    alpha=weights[client_id],
                )
            aggregated[bias_key][class_id] = value.to(reference_bias.dtype)

        supporters = class_supporters(selected, client_class_counts, class_id)
        row_delta = (
            aggregated[weight_key][class_id].detach().float().cpu()
            - previous_global_state[weight_key][class_id].detach().float().cpu()
        )
        diagnostics.append(
            {
                "class_id": class_id,
                "supporter_count": len(supporters),
                "supporter_ids": ",".join(str(value) for value in supporters),
                "support_mass": sum(
                    _count(client_class_counts, client_id, class_id)
                    for client_id in supporters
                ),
                # Ordinary FedAvg has no special no-support branch. If all
                # local rows remained unchanged this equality occurs only by
                # arithmetic, not by a server retention rule.
                "retained_previous_row": False,
                "row_delta_norm": float(row_delta.norm().item()),
                "max_supporter_weight": max(
                    (weights[client_id] for client_id in supporters), default=0.0
                ),
            }
        )

    return aggregated, diagnostics


def _append_csv(path: Path, rows: Sequence[Mapping]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists() and path.stat().st_size > 0
    fieldnames = list(rows[0].keys())
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerows(rows)


class D4ATracker:
    """Stateful, read-only logger for supporter absence and tail degradation."""

    def __init__(self, output_dir, class_ids: Sequence[int], source="official_test_diagnostic"):
        self.output_dir = Path(output_dir)
        self.class_ids = [int(value) for value in class_ids]
        self.source = str(source)
        self.absence_streak = {class_id: 0 for class_id in self.class_ids}
        self.best_accuracy = {class_id: -math.inf for class_id in self.class_ids}
        self.best_margin = {class_id: -math.inf for class_id in self.class_ids}

    def record(
        self,
        epoch: int,
        aggregation_rows: Sequence[Mapping],
        per_class_metrics: Mapping[int, Mapping[str, float]] | None = None,
    ) -> None:
        by_class = {int(row["class_id"]): row for row in aggregation_rows}
        metrics = per_class_metrics or {}
        rows = []
        for class_id in self.class_ids:
            aggregate = by_class[class_id]
            supporter_count = int(aggregate["supporter_count"])
            if supporter_count > 0:
                self.absence_streak[class_id] = 0
            else:
                self.absence_streak[class_id] += 1

            item = metrics.get(class_id, {})
            accuracy = float(item.get("accuracy", math.nan))
            margin = float(item.get("margin", math.nan))
            accuracy_drop = math.nan
            margin_drop = math.nan
            if math.isfinite(accuracy):
                previous_best = self.best_accuracy[class_id]
                accuracy_drop = max(previous_best - accuracy, 0.0) if math.isfinite(previous_best) else 0.0
                self.best_accuracy[class_id] = max(previous_best, accuracy)
            if math.isfinite(margin):
                previous_best = self.best_margin[class_id]
                margin_drop = max(previous_best - margin, 0.0) if math.isfinite(previous_best) else 0.0
                self.best_margin[class_id] = max(previous_best, margin)

            rows.append(
                {
                    "epoch_index": int(epoch),
                    "communication_round": int(epoch) + 1,
                    "class_id": class_id,
                    "supporter_count": supporter_count,
                    "supporter_ids": aggregate["supporter_ids"],
                    "support_mass": aggregate["support_mass"],
                    "absence_streak": self.absence_streak[class_id],
                    "retained_previous_row": bool(aggregate["retained_previous_row"]),
                    "row_delta_norm": aggregate["row_delta_norm"],
                    "evaluated": math.isfinite(accuracy) and math.isfinite(margin),
                    "accuracy": accuracy,
                    "true_class_margin": margin,
                    "accuracy_drop_from_historical_best": accuracy_drop,
                    "margin_drop_from_historical_best": margin_drop,
                    "metric_source": self.source,
                    "used_by_aggregation": False,
                }
            )
        _append_csv(self.output_dir / "d4a" / "d4a_per_class_round.csv", rows)


def evaluate_per_class_accuracy_margin(trainer, num_classes: int) -> dict[int, dict[str, float]]:
    """Measure class accuracy and true-vs-best-other margin without model mutation."""
    trainer.set_model_mode("eval")
    correct = torch.zeros(num_classes, dtype=torch.float64)
    total = torch.zeros(num_classes, dtype=torch.float64)
    margin_sum = torch.zeros(num_classes, dtype=torch.float64)
    with torch.no_grad():
        for batch in trainer.test_loader:
            images, labels = trainer.parse_batch_test(batch)
            logits = trainer.model_inference(images).detach().float()
            labels = labels.long()
            predictions = logits.argmax(dim=1)
            true_logits = logits.gather(1, labels.view(-1, 1)).squeeze(1)
            masked = logits.clone()
            masked.scatter_(1, labels.view(-1, 1), -torch.inf)
            margins = true_logits - masked.max(dim=1).values
            for class_id in labels.unique().tolist():
                mask = labels == int(class_id)
                total[class_id] += int(mask.sum().item())
                correct[class_id] += int((predictions[mask] == labels[mask]).sum().item())
                margin_sum[class_id] += margins[mask].double().sum().cpu()
    return {
        class_id: {
            "accuracy": float(100.0 * correct[class_id] / total[class_id]) if total[class_id] else math.nan,
            "margin": float(margin_sum[class_id] / total[class_id]) if total[class_id] else math.nan,
        }
        for class_id in range(num_classes)
    }
