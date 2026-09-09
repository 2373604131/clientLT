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


LORA_AGGREGATION_MODES = (
    "fedavg",
    "support_normalized",
    "effective_svd",
    "la_aggregation",
)


def inspect_model_lora_scaling(model) -> tuple[float, list[dict]]:
    """Read and validate the scaling used by every trainable LoRA A/B pair.

    Effective-matrix aggregation must use the value in the actual forward
    modules, not a duplicate alpha/rank formula in the experiment driver.
    The current ClipLoRA configuration is intentionally restricted to one
    common scaling value across all aggregated matrices.
    """
    modules = dict(model.named_modules())
    parameter_names = {
        str(name) for name, parameter in model.named_parameters()
        if bool(parameter.requires_grad) and "lora_" in str(name)
    }
    pairs = _lora_factor_pairs(sorted(parameter_names))
    details = []
    for a_key, b_key in pairs:
        module_name, _, parameter_name = a_key.rpartition(".")
        b_module_name, _, b_parameter_name = b_key.rpartition(".")
        if module_name != b_module_name:
            raise ValueError(f"LoRA A/B factors belong to different modules: {a_key}, {b_key}")
        module = modules.get(module_name)
        if module is None:
            raise ValueError(f"Could not resolve LoRA module for {a_key}")
        if parameter_name not in module._parameters or b_parameter_name not in module._parameters:
            raise ValueError(f"LoRA factors are not direct parameters of module {module_name}")
        value = float(getattr(module, "scaling", math.nan))
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"Invalid runtime LoRA scaling for {module_name}: {value}")
        details.append(
            {
                "module": module_name,
                "a_key": a_key,
                "b_key": b_key,
                "scaling": value,
            }
        )
    scaling = details[0]["scaling"]
    if any(
        not math.isclose(item["scaling"], scaling, rel_tol=0.0, abs_tol=1e-12)
        for item in details[1:]
    ):
        observed = {item["module"]: item["scaling"] for item in details}
        raise ValueError(
            "Effective SVD currently requires one common LoRA scaling; "
            f"observed {observed}"
        )
    return scaling, details


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


def la_aggregation_client_weights(
    selected_clients: Sequence[int],
    datanumber_client: Sequence[float],
    client_class_counts,
    *,
    temperature: float = 1.0,
    regularization: float = 1.0,
    max_iter: int = 100,
) -> tuple[dict[int, float], dict]:
    """Fit client weights whose induced label prior matches an LA target.

    Let ``p_k`` be client k's empirical label distribution and ``a_k`` its
    ordinary sample-weighted FedAvg coefficient.  FedAvg induces the prior
    ``pi = sum_k a_k p_k``.  We use the logit-adjustment path

        q_c(tau) proportional to pi_c ** (1 - tau)

    and solve

        min_alpha KL(q || sum_k alpha_k p_k) + lambda KL(alpha || a)

    over the probability simplex.  ``tau=0`` recovers the FedAvg target and
    ``tau=1`` requests a uniform prior over classes present in the round.
    The learned scalar client coefficients are subsequently used for effective
    LoRA-matrix aggregation and rank-r SVD recompression.
    """
    selected = _selected_client_ids(selected_clients)
    temperature = float(temperature)
    regularization = float(regularization)
    max_iter = int(max_iter)
    if not math.isfinite(temperature) or not 0.0 <= temperature <= 1.0:
        raise ValueError("LA aggregation temperature must be in [0, 1]")
    if not math.isfinite(regularization) or regularization < 0.0:
        raise ValueError("LA aggregation regularization must be non-negative")
    if max_iter < 1:
        raise ValueError("LA aggregation max_iter must be positive")
    if client_class_counts is None:
        raise ValueError("la_aggregation requires client_class_counts")

    rows = []
    for client_id in selected:
        if isinstance(client_class_counts, Mapping):
            row = client_class_counts[client_id]
        else:
            row = client_class_counts[client_id]
        row = torch.as_tensor(row, dtype=torch.float64, device="cpu").flatten()
        if row.numel() == 0 or not bool(torch.isfinite(row).all()) or bool((row < 0).any()):
            raise ValueError(f"Invalid class counts for client {client_id}")
        total = float(row.sum().item())
        if total <= 0.0:
            raise ValueError(f"Client {client_id} has zero class-count total")
        rows.append(row / total)
    num_classes = int(rows[0].numel())
    if any(int(row.numel()) != num_classes for row in rows):
        raise ValueError("All clients must have the same number of classes")
    client_priors = torch.stack(rows, dim=0)

    fedavg_dict = sample_weighted_client_weights(selected, datanumber_client)
    fedavg = torch.tensor(
        [fedavg_dict[client_id] for client_id in selected], dtype=torch.float64
    )
    source_prior = fedavg @ client_priors
    covered = source_prior > 0
    if not bool(covered.any()):
        raise ValueError("Selected clients cover no classes")

    target_prior = torch.zeros_like(source_prior)
    target_values = source_prior[covered].pow(1.0 - temperature)
    target_prior[covered] = target_values / target_values.sum()

    # Softmax parameterization keeps alpha strictly inside the simplex, which
    # makes both KL terms finite.  This is a tiny deterministic CPU problem
    # (30 variables in the canonical experiment), so L-BFGS overhead is
    # negligible relative to local training.
    logits = torch.log(fedavg).detach().clone().requires_grad_(True)
    optimizer = torch.optim.LBFGS(
        [logits],
        lr=1.0,
        max_iter=max_iter,
        tolerance_grad=1e-12,
        tolerance_change=1e-14,
        line_search_fn="strong_wolfe",
    )
    tiny = torch.finfo(torch.float64).tiny

    def terms():
        alpha = torch.softmax(logits, dim=0)
        achieved = alpha @ client_priors
        target_kl = torch.sum(
            target_prior[covered]
            * (
                torch.log(target_prior[covered].clamp_min(tiny))
                - torch.log(achieved[covered].clamp_min(tiny))
            )
        )
        base_kl = torch.sum(
            alpha
            * (
                torch.log(alpha.clamp_min(tiny))
                - torch.log(fedavg.clamp_min(tiny))
            )
        )
        return target_kl + regularization * base_kl, alpha, achieved, target_kl, base_kl

    def closure():
        optimizer.zero_grad()
        objective, _, _, _, _ = terms()
        objective.backward()
        return objective

    optimizer.step(closure)
    with torch.no_grad():
        objective, alpha, achieved, target_kl, base_kl = terms()

    values = [float(value) for value in alpha.tolist()]
    normalizer = sum(values)
    weights = {
        client_id: values[index] / normalizer
        for index, client_id in enumerate(selected)
    }
    covered_classes = torch.nonzero(covered, as_tuple=False).flatten().tolist()
    uncovered_classes = torch.nonzero(~covered, as_tuple=False).flatten().tolist()
    return weights, {
        "tail_class_count": 0,
        "covered_tail_class_count": 0,
        "covered_tail_classes": [],
        "uncovered_tail_classes": [],
        "client_supported_tail_classes": {client_id: 0 for client_id in selected},
        "covered_classes": [int(value) for value in covered_classes],
        "uncovered_classes": [int(value) for value in uncovered_classes],
        "la_temperature": temperature,
        "la_regularization": regularization,
        "la_max_iter": max_iter,
        "la_objective": float(objective.item()),
        "la_target_kl": float(target_kl.item()),
        "la_fedavg_kl": float(base_kl.item()),
        "la_source_prior": [float(value) for value in source_prior.tolist()],
        "la_target_prior": [float(value) for value in target_prior.tolist()],
        "la_achieved_prior": [float(value) for value in achieved.tolist()],
    }


def compute_lora_aggregation_weights(
    mode: str,
    selected_clients: Sequence[int],
    datanumber_client: Sequence[float],
    *,
    client_class_counts=None,
    tail_class_ids: Sequence[int] | None = None,
    la_temperature: float = 1.0,
    la_regularization: float = 1.0,
    la_max_iter: int = 100,
) -> tuple[dict[int, float], dict]:
    """Resolve a named ClipLoRA aggregation policy."""
    mode = str(mode).lower()
    if mode not in LORA_AGGREGATION_MODES:
        raise ValueError(
            f"Unknown ClipLoRA aggregation mode {mode!r}; "
            f"choose from {LORA_AGGREGATION_MODES}"
        )

    selected = _selected_client_ids(selected_clients)
    if mode in {"fedavg", "effective_svd"}:
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

    if mode == "la_aggregation":
        weights, details = la_aggregation_client_weights(
            selected,
            datanumber_client,
            client_class_counts,
            temperature=la_temperature,
            regularization=la_regularization,
            max_iter=la_max_iter,
        )
        tail_classes = [int(value) for value in (tail_class_ids or ())]
        covered_tail = [
            class_id
            for class_id in tail_classes
            if any(
                _class_count(client_class_counts, client_id, class_id) > 0
                for client_id in selected
            )
        ]
        details.update(
            {
                "tail_class_count": len(tail_classes),
                "covered_tail_class_count": len(covered_tail),
                "covered_tail_classes": covered_tail,
                "uncovered_tail_classes": sorted(set(tail_classes) - set(covered_tail)),
                "client_supported_tail_classes": {
                    client_id: sum(
                        _class_count(client_class_counts, client_id, class_id) > 0
                        for class_id in tail_classes
                    )
                    for client_id in selected
                },
            }
        )
        return weights, details

    if client_class_counts is None:
        raise ValueError("support_normalized aggregation requires client_class_counts")
    if tail_class_ids is None:
        raise ValueError("support_normalized aggregation requires tail_class_ids")
    try:
        return support_normalized_client_weights(
            selected,
            datanumber_client,
            client_class_counts,
            tail_class_ids,
        )
    except ValueError as error:
        # Under partial participation an entire round can contain no tail
        # evidence. There is then no support-conditioned direction to apply;
        # ordinary FedAvg is the uniquely auditable no-information fallback.
        if "No selected client supports any requested tail class" not in str(error):
            raise
        weights = sample_weighted_client_weights(selected, datanumber_client)
        return weights, {
            "tail_class_count": len(tail_class_ids),
            "covered_tail_class_count": 0,
            "covered_tail_classes": [],
            "uncovered_tail_classes": [int(item) for item in tail_class_ids],
            "client_supported_tail_classes": {client_id: 0 for client_id in selected},
            "fallback": "fedavg_no_selected_tail_support",
        }


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


def _lora_factor_pairs(trainable_keys: Sequence[str]) -> list[tuple[str, str]]:
    """Resolve complete A/B pairs without relying on module-name conventions."""
    keys = set(str(key) for key in trainable_keys)
    pairs = []
    consumed = set()
    for key in sorted(keys):
        if not key.endswith("_lora_A"):
            continue
        b_key = key[:-1] + "B"
        if b_key not in keys:
            raise ValueError(f"Effective LoRA aggregation is missing pair for {key}")
        pairs.append((key, b_key))
        consumed.update((key, b_key))
    unexpected = sorted(keys - consumed)
    if unexpected:
        raise ValueError(
            "Effective LoRA aggregation received unpaired trainable keys: "
            f"{unexpected}"
        )
    if not pairs:
        raise ValueError("Effective LoRA aggregation found no A/B factor pairs")
    return pairs


def factorize_effective_lora(
    effective: torch.Tensor,
    *,
    rank: int,
    scaling: float,
    zero_tolerance: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor, dict]:
    """Return factors whose ``scaling * B @ A`` is the best rank-r projection.

    A deterministic non-zero A row is retained for every zero singular slot so
    a zero effective adapter does not become a two-factor gradient dead point.
    Validation compares effective matrices, since singular-vector signs and
    bases inside repeated singular subspaces are not identifiable.
    """
    matrix = torch.as_tensor(effective)
    if matrix.ndim != 2 or matrix.numel() == 0:
        raise ValueError("effective must be a non-empty matrix")
    if not bool(torch.isfinite(matrix).all()):
        raise ValueError("effective contains NaN or Inf")
    rank = int(rank)
    if rank < 1 or rank > min(matrix.shape):
        raise ValueError(f"rank={rank} is invalid for shape {tuple(matrix.shape)}")
    scaling = float(scaling)
    if not math.isfinite(scaling) or scaling <= 0:
        raise ValueError("scaling must be finite and positive")

    # CPU SVD gives the smoke/resume path a stable implementation independent
    # of which CUDA device is assigned. The result is returned on the input
    # device and in the input dtype.
    work = matrix.detach().to(device="cpu", dtype=torch.float64)
    u, singular, vh = torch.linalg.svd(work, full_matrices=False)
    leading = singular[:rank]
    if zero_tolerance is None:
        zero_tolerance = (
            torch.finfo(work.dtype).eps
            * float(max(work.shape))
            * max(float(singular[0].item()), 1.0)
        )
    tolerance = float(zero_tolerance)
    if not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError("zero_tolerance must be finite and non-negative")
    positive = leading > tolerance

    a = torch.zeros((rank, work.shape[1]), dtype=work.dtype)
    b = torch.zeros((work.shape[0], rank), dtype=work.dtype)
    for slot in range(rank):
        if bool(positive[slot]):
            root = torch.sqrt(leading[slot] / scaling)
            b[:, slot] = u[:, slot] * root
            a[slot, :] = vh[slot, :] * root
        else:
            # Unit A with zero B preserves an exact zero contribution while
            # leaving d(B@A)/dB non-zero for the next local optimizer step.
            a[slot, slot % work.shape[1]] = 1.0

    projected = (scaling * (b @ a)).to(dtype=work.dtype)
    kept = leading * positive.to(leading.dtype)
    optimal = (u[:, :rank] * kept.unsqueeze(0)) @ vh[:rank, :]
    projection_norm_error = float(
        torch.linalg.vector_norm(projected - optimal).item()
        / max(float(torch.linalg.vector_norm(optimal).item()), 1e-12)
    )
    tail_energy = float(torch.sum(singular[rank:] ** 2).item())
    thresholded_energy = float(torch.sum(leading[~positive] ** 2).item())
    optimal_error = tail_energy + thresholded_energy
    error = float(torch.sum((work - projected) ** 2).item())
    diagnostics = {
        "rank": rank,
        "numerical_rank": int(positive.sum().item()),
        "scaling": scaling,
        "zero_tolerance": tolerance,
        "projection_error_squared": error,
        "tail_singular_energy": tail_energy,
        "thresholded_singular_energy": thresholded_energy,
        "expected_projection_error_squared": optimal_error,
        "projection_norm_error": projection_norm_error,
    }
    return (
        a.to(device=matrix.device, dtype=matrix.dtype),
        b.to(device=matrix.device, dtype=matrix.dtype),
        diagnostics,
    )


def aggregate_effective_lora_state(
    global_state: Mapping[str, torch.Tensor],
    local_states,
    selected_clients: Sequence[int],
    trainable_keys: Sequence[str],
    client_weights: Mapping[int, float],
    *,
    scaling: float,
) -> tuple[dict, list[dict]]:
    """Average absolute effective adapters and recompress every matrix to rank r."""
    selected = _selected_client_ids(selected_clients)
    weights = {client_id: float(client_weights[client_id]) for client_id in selected}
    if any(not math.isfinite(value) or value < 0 for value in weights.values()):
        raise ValueError(f"Invalid LoRA aggregation weights: {weights}")
    if not math.isclose(sum(weights.values()), 1.0, rel_tol=1e-7, abs_tol=1e-7):
        raise ValueError("Effective LoRA aggregation weights must sum to one")

    aggregated = copy.deepcopy(global_state)
    diagnostics = []
    for a_key, b_key in _lora_factor_pairs(trainable_keys):
        reference_a = global_state[a_key]
        reference_b = global_state[b_key]
        if reference_a.ndim != 2 or reference_b.ndim != 2:
            raise ValueError(f"LoRA factors must be matrices: {a_key}, {b_key}")
        if reference_b.shape[1] != reference_a.shape[0]:
            raise ValueError(f"Incompatible LoRA factor shapes: {a_key}, {b_key}")
        effective = torch.zeros(
            (reference_b.shape[0], reference_a.shape[1]),
            dtype=torch.float64,
            device="cpu",
        )
        for client_id in selected:
            local_a = local_states[client_id][a_key].detach().to("cpu", torch.float64)
            local_b = local_states[client_id][b_key].detach().to("cpu", torch.float64)
            if local_a.shape != reference_a.shape or local_b.shape != reference_b.shape:
                raise ValueError(f"Client {client_id} LoRA shape mismatch for {a_key}")
            effective.add_(local_b @ local_a, alpha=float(scaling) * weights[client_id])

        a, b, report = factorize_effective_lora(
            effective,
            rank=int(reference_a.shape[0]),
            scaling=float(scaling),
        )
        aggregated[a_key] = a.to(device=reference_a.device, dtype=reference_a.dtype)
        aggregated[b_key] = b.to(device=reference_b.device, dtype=reference_b.dtype)
        diagnostics.append({"a_key": a_key, "b_key": b_key, **report})
    return aggregated, diagnostics


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
        "la_temperature": details.get("la_temperature", ""),
        "la_regularization": details.get("la_regularization", ""),
        "la_objective": details.get("la_objective", ""),
        "la_target_kl": details.get("la_target_kl", ""),
        "la_fedavg_kl": details.get("la_fedavg_kl", ""),
    }
    _append_csv_rows(output_path / "lora_aggregation_summary.csv", [summary_row])


def append_effective_svd_diagnostics(
    output_dir,
    *,
    epoch: int,
    diagnostics: Sequence[Mapping],
) -> None:
    rows = [
        {
            "epoch_index": int(epoch),
            "communication_round": int(epoch) + 1,
            **dict(item),
        }
        for item in diagnostics
    ]
    if rows:
        _append_csv_rows(Path(output_dir) / "effective_svd_diagnostics.csv", rows)


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
