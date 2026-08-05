"""Audit-cache construction and causal edge diagnostics for Boundary Repair.

All image features in an audit cache originate from deterministic views of the
federated training split.  They are deliberately labelled as audit data, not
as held-out evaluation data.  Official-test loading is intentionally absent.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Mapping, Sequence

import torch

from utils.boundary_metrics import (
    EPS,
    edge_sample_weights,
    finite_mean,
    finite_median,
    support_counterfactual_delta,
)
from utils.cusp_minimal import (
    FlatSpec,
    flatten_state,
    make_flat_spec,
    unflatten_state,
    write_json,
)
from utils.functional_cusp import fedavg_delta_from_payload


BOUNDARY_SCHEMA_VERSION = "visual_semantic_boundary_v1"


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save_boundary_round_dump(
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
    trainable_keys: Sequence[str],
    artifact_root: str = "boundary_gate",
) -> Path:
    """Persist the minimum state needed by an offline boundary Gate.

    This function is independent from CUSP so that both methods remain valid
    controls.  It saves only trainable floating-point parameters.
    """
    selected = [int(client_id) for client_id in selected_clients]
    keys = tuple(sorted(str(key) for key in trainable_keys))
    if not keys:
        raise ValueError("boundary dump received no trainable parameter keys")
    missing = [key for key in keys if key not in global_before or key not in global_after]
    if missing:
        raise KeyError(f"boundary dump trainable keys are missing from global state: {missing}")
    trainable_before = {
        key: global_before[key].detach().cpu().clone()
        for key in keys
    }
    spec = make_flat_spec(trainable_before)
    local_states = [
        {key: local_weights[client_id][key].detach().cpu().clone() for key in spec.keys}
        for client_id in selected
    ]
    total = sum(float(datanumber_client[client_id]) for client_id in selected)
    if total <= 0:
        raise ValueError("selected clients have zero total sample count")
    weights = torch.tensor([float(datanumber_client[client_id]) / total for client_id in selected], dtype=torch.float64)
    counts = torch.stack([client_class_counts[client_id].detach().cpu().long() for client_id in selected], dim=0)
    # Reproduce utils.fed_utils.average_weights exactly: keep the local tensor
    # dtype and add clients in selected-client order.  A float64 vectorized
    # reconstruction is mathematically equivalent but can differ from the
    # actual FP32 server state by ~1e-4 after many additions.
    mismatches = []
    for key in spec.keys:
        reconstructed = local_states[0][key].clone() * float(weights[0].item())
        for state, weight in zip(local_states[1:], weights[1:]):
            reconstructed += state[key] * float(weight.item())
        expected = global_after[key].detach().cpu().to(reconstructed.dtype)
        if not torch.equal(reconstructed, expected):
            mismatches.append({
                "key": key,
                "max_abs_error": float((reconstructed - expected).abs().max().item()),
            })
    reconstruction_report = {
        "ordered_fp32_exact_match": not mismatches,
        "mismatches": mismatches,
        "authoritative_state": "fedavg_candidate_trainable",
    }
    payload = {
        "schema_version": BOUNDARY_SCHEMA_VERSION,
        "flatten_spec": spec.as_dict(),
        "trainable_keys": list(spec.keys),
        "global_before_trainable": trainable_before,
        "fedavg_candidate_trainable": {key: global_after[key].detach().cpu().clone() for key in spec.keys},
        "local_trainable_states": local_states,
        "selected_client_ids": selected,
        "fedavg_weights": weights,
        "client_sample_counts": [int(datanumber_client[client_id]) for client_id in selected],
        "client_class_counts": counts,
        "global_class_counts": global_class_counts.detach().cpu().long(),
        "num_classes": int(global_class_counts.numel()),
    }
    metadata = {
        "schema_version": BOUNDARY_SCHEMA_VERSION,
        "communication_round": int(epoch) + 1,
        "resolved_args": vars(args),
        "resolved_config": str(cfg),
        "test_used_before_dump": False,
        "fedavg_reconstruction_check": reconstruction_report,
    }
    run_dir = Path(output_dir) / str(artifact_root) / f"round_{int(epoch) + 1:03d}"
    run_dir.mkdir(parents=True, exist_ok=True)
    torch.save(payload, run_dir / "round_state.pt")
    write_json(run_dir / "metadata.json", metadata)
    return run_dir


def load_boundary_round_dump(dump_dir: str | Path) -> tuple[dict, dict]:
    dump_dir = Path(dump_dir)
    payload = torch.load(dump_dir / "round_state.pt", map_location="cpu", weights_only=False)
    metadata = json.loads((dump_dir / "metadata.json").read_text(encoding="utf-8"))
    return payload, metadata


def attach_audit_cache_to_round_dump(
    cfg,
    trainer,
    dump_dir: str | Path,
    *,
    max_edges_per_class: int = 3,
    batch_size: int = 256,
) -> tuple[dict, str]:
    """Build the audit cache and persist its fixed edge catalogue into a dump."""
    dump_dir = Path(dump_dir)
    payload, metadata = load_boundary_round_dump(dump_dir)
    cache, cache_hash = build_audit_cache(cfg, trainer, payload, dump_dir, batch_size=batch_size)
    edges, edge_counts = build_edge_catalog(cache, max_edges_per_class=max_edges_per_class)
    payload["edge_catalog"] = edges
    payload["client_edge_counts"] = edge_counts
    torch.save(payload, dump_dir / "round_state.pt")
    metadata.update({
        "audit_cache_hash": cache_hash,
        "audit_edge_rule": "round_before_promptfl_top2_false_plus_zeroshot_top1_false",
        "max_edges_per_class": int(max_edges_per_class),
        "official_test_used": False,
    })
    write_json(dump_dir / "metadata.json", metadata)
    return cache, cache_hash


def _logits_from_theta(model, spec: FlatSpec, theta: torch.Tensor, features: torch.Tensor, batch_size: int) -> torch.Tensor:
    state = unflatten_state(theta, spec)
    chunks = []
    with torch.no_grad():
        for start in range(0, int(features.shape[0]), int(batch_size)):
            logits = model.logits_from_cached_features(features[start:start + batch_size], state)
            chunks.append(logits.detach().cpu().to(torch.float64))
    return torch.cat(chunks, dim=0)


def build_audit_cache(cfg, trainer, payload: Mapping, output_dir: str | Path, *, batch_size: int = 256) -> tuple[dict, str]:
    """Encode deterministic audit views for all selected clients.

    The deterministic evaluation transform is intentionally different from
    PromptFL's stochastic local-training transform.  No test loader is used.
    """
    from Dassl.dassl.data.data_manager import build_data_loader
    from Dassl.dassl.data.transforms import build_transform

    selected = [int(client_id) for client_id in payload["selected_client_ids"]]
    transform = build_transform(cfg, is_train=False)
    model = trainer.model
    was_training = model.training
    features, labels, client_ids, sample_ids = [], [], [], []
    try:
        model.eval()
        with torch.no_grad():
            for client_id in selected:
                data_source = trainer.dm.dataset.federated_train_x[client_id]
                loader = build_data_loader(
                    cfg,
                    sampler_type="SequentialSampler",
                    data_source=data_source,
                    batch_size=int(batch_size),
                    tfm=transform,
                    is_train=False,
                    dataset_wrapper=None,
                    class_names=trainer.dm.dataset.classnames,
                )
                offset = 0
                for batch in loader:
                    images = batch["img"].to(model.logit_scale.device)
                    batch_labels = batch["label"].detach().long().cpu()
                    encoded = model.encode_audit_images(images).detach().float().cpu()
                    features.append(encoded)
                    labels.append(batch_labels)
                    client_ids.append(torch.full_like(batch_labels, client_id))
                    sample_ids.extend(f"{client_id}:{offset + item}" for item in range(batch_labels.numel()))
                    offset += batch_labels.numel()
    finally:
        model.train(was_training)
    cache = {
        "schema_version": BOUNDARY_SCHEMA_VERSION,
        "source": "train_audit_view",
        "official_test_used": False,
        "audit_transform": {"mode": "deterministic_eval_transform", "repr": repr(transform)},
        "audit_seed": int(getattr(cfg, "SEED", -1)),
        "features": torch.cat(features, dim=0),
        "labels": torch.cat(labels, dim=0),
        "client_ids": torch.cat(client_ids, dim=0),
        "sample_ids": sample_ids,
        "num_clients": int(max(selected) + 1),
        "num_classes": int(payload["num_classes"]),
    }
    validate_audit_cache(cache, payload)
    spec = FlatSpec.from_dict(payload["flatten_spec"])
    theta_before = flatten_state(payload["global_before_trainable"], spec)
    cache["fixed_hard_negatives"] = model.compute_fixed_hard_negatives_from_cached_features(
        cache["features"], cache["labels"], unflatten_state(theta_before, spec), batch_size=int(batch_size)
    ).detach().cpu().long()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "audit_cache.pt"
    torch.save(cache, path)
    return cache, sha256_file(path)


def validate_audit_cache(cache: Mapping, payload: Mapping) -> None:
    required = {"features", "labels", "client_ids", "sample_ids", "official_test_used"}
    missing = required - set(cache)
    if missing:
        raise ValueError(f"audit cache is missing keys: {sorted(missing)}")
    if bool(cache["official_test_used"]):
        raise ValueError("audit cache must not use official test")
    size = int(torch.as_tensor(cache["labels"]).numel())
    if int(torch.as_tensor(cache["features"]).shape[0]) != size or int(torch.as_tensor(cache["client_ids"]).numel()) != size:
        raise ValueError("audit cache feature/label/client lengths differ")
    if len(cache["sample_ids"]) != size or len(set(cache["sample_ids"])) != size:
        raise ValueError("audit cache sample_ids must be unique and aligned")
    selected = [int(value) for value in payload["selected_client_ids"]]
    expected = torch.as_tensor(payload["client_class_counts"], dtype=torch.long)
    labels = torch.as_tensor(cache["labels"], dtype=torch.long)
    client_ids = torch.as_tensor(cache["client_ids"], dtype=torch.long)
    observed = torch.zeros_like(expected)
    for row, client_id in enumerate(selected):
        observed[row] = torch.bincount(labels[client_ids == client_id], minlength=expected.shape[1])[:expected.shape[1]]
    if not torch.equal(observed, expected):
        raise ValueError("audit cache no longer matches the stored selected-client class counts")


def build_edge_catalog(cache: Mapping, *, max_edges_per_class: int = 3) -> tuple[list[dict], torch.Tensor]:
    """Choose deterministic class-pair edges from frozen per-sample negatives."""
    labels = torch.as_tensor(cache["labels"], dtype=torch.long)
    client_ids = torch.as_tensor(cache["client_ids"], dtype=torch.long)
    negatives = torch.as_tensor(cache["fixed_hard_negatives"], dtype=torch.long)
    if negatives.ndim != 2 or negatives.shape[0] != labels.numel():
        raise ValueError("fixed_hard_negatives must be [num_samples, max_negatives]")
    num_clients = int(cache.get("num_clients", int(client_ids.max().item()) + 1))
    counts: dict[tuple[int, int], int] = {}
    for label, row in zip(labels.tolist(), negatives.tolist()):
        for negative in sorted({int(item) for item in row if int(item) >= 0 and int(item) != int(label)}):
            counts[(int(label), negative)] = counts.get((int(label), negative), 0) + 1
    edges = []
    class_ids = sorted({class_id for class_id, _ in counts})
    for class_id in class_ids:
        choices = [(negative, count) for (value, negative), count in counts.items() if value == class_id]
        choices.sort(key=lambda item: (-item[1], item[0]))
        for negative, count in choices[: int(max_edges_per_class)]:
            edges.append({"class_id": class_id, "negative_id": negative, "audit_sample_count": int(count)})
    edges.sort(key=lambda row: (int(row["class_id"]), int(row["negative_id"])))
    edge_counts = torch.zeros((num_clients, len(edges)), dtype=torch.long)
    for edge_id, edge in enumerate(edges):
        mask = (labels == int(edge["class_id"])) & (negatives == int(edge["negative_id"])).any(dim=1)
        for client_id in torch.unique(client_ids[mask]).tolist():
            edge_counts[int(client_id), edge_id] = int((mask & (client_ids == int(client_id))).sum().item())
        edge["edge_id"] = edge_id
        edge["num_support_clients"] = int((edge_counts[:, edge_id] > 0).sum().item())
    return edges, edge_counts


def edge_sample_mask(cache: Mapping, edge: Mapping) -> torch.Tensor:
    labels = torch.as_tensor(cache["labels"], dtype=torch.long)
    negatives = torch.as_tensor(cache["fixed_hard_negatives"], dtype=torch.long)
    return (labels == int(edge["class_id"])) & (negatives == int(edge["negative_id"])).any(dim=1)


def edge_margins_from_logits(logits: torch.Tensor, cache: Mapping, edges: Sequence[Mapping]) -> torch.Tensor:
    """Return [global_client_id, edge] audit mean margins, using NaN for no support."""
    logits = torch.as_tensor(logits, dtype=torch.float64)
    labels = torch.as_tensor(cache["labels"], dtype=torch.long)
    client_ids = torch.as_tensor(cache["client_ids"], dtype=torch.long)
    num_clients = int(cache.get("num_clients", int(client_ids.max().item()) + 1))
    values = torch.full((num_clients, len(edges)), torch.nan, dtype=torch.float64)
    for edge_id, edge in enumerate(edges):
        mask = edge_sample_mask(cache, edge)
        if not bool(mask.any()):
            continue
        margin = logits[mask, int(edge["class_id"])] - logits[mask, int(edge["negative_id"])]
        selected_clients = client_ids[mask]
        for client_id in torch.unique(selected_clients).tolist():
            values[int(client_id), edge_id] = margin[selected_clients == int(client_id)].mean()
    return values


def evaluate_state_edge_margins(model, spec: FlatSpec, theta: torch.Tensor, cache: Mapping, edges: Sequence[Mapping], *, batch_size: int = 2048) -> torch.Tensor:
    logits = _logits_from_theta(model, spec, theta, torch.as_tensor(cache["features"], dtype=torch.float32), batch_size)
    return edge_margins_from_logits(logits, cache, edges)


def evaluate_single_edge_margins(
    model,
    spec: FlatSpec,
    theta: torch.Tensor,
    cache: Mapping,
    edge: Mapping,
    *,
    batch_size: int = 2048,
) -> torch.Tensor:
    """Evaluate one state only on samples contributing to one frozen edge."""
    mask = edge_sample_mask(cache, edge)
    client_ids = torch.as_tensor(cache["client_ids"], dtype=torch.long)
    num_clients = int(cache.get("num_clients", int(client_ids.max().item()) + 1))
    values = torch.full((num_clients, 1), torch.nan, dtype=torch.float64)
    if not bool(mask.any()):
        return values
    features = torch.as_tensor(cache["features"], dtype=torch.float32)[mask]
    logits = _logits_from_theta(model, spec, theta, features, batch_size)
    margin = logits[:, int(edge["class_id"])] - logits[:, int(edge["negative_id"])]
    selected_clients = client_ids[mask]
    for client_id in torch.unique(selected_clients).tolist():
        values[int(client_id), 0] = margin[selected_clients == int(client_id)].mean()
    return values


def evaluate_local_edge_margins(model, spec: FlatSpec, payload: Mapping, cache: Mapping, edges: Sequence[Mapping], *, batch_size: int = 2048) -> torch.Tensor:
    """Evaluate each local model only on that client's audit samples."""
    selected = [int(value) for value in payload["selected_client_ids"]]
    client_ids = torch.as_tensor(cache["client_ids"], dtype=torch.long)
    num_clients = int(cache.get("num_clients", int(client_ids.max().item()) + 1))
    values = torch.full((num_clients, len(edges)), torch.nan, dtype=torch.float64)
    for state, client_id in zip(payload["local_trainable_states"], selected):
        mask = client_ids == client_id
        if not bool(mask.any()):
            continue
        theta = flatten_state(state, spec)
        local_cache = {
            "features": torch.as_tensor(cache["features"])[mask],
            "labels": torch.as_tensor(cache["labels"])[mask],
            "client_ids": client_ids[mask],
            "fixed_hard_negatives": torch.as_tensor(cache["fixed_hard_negatives"])[mask],
            "num_clients": num_clients,
        }
        local_values = evaluate_state_edge_margins(model, spec, theta, local_cache, edges, batch_size=batch_size)
        values[client_id] = local_values[client_id]
    return values


def _edge_gain(values: torch.Tensor, before: torch.Tensor, weights: torch.Tensor, edge_id: int) -> float:
    valid = torch.isfinite(values[:, edge_id]) & torch.isfinite(before[:, edge_id]) & (weights > 0)
    if not bool(valid.any()):
        return math.nan
    return float((weights[valid] * (values[valid, edge_id] - before[valid, edge_id])).sum().item())


def diagnose_edges(
    model,
    payload: Mapping,
    cache: Mapping,
    edges: Sequence[Mapping],
    edge_counts: torch.Tensor,
    *,
    gamma: float,
    tau: float,
    min_support_clients: int,
    max_fragile_edges_per_class: int,
    max_total_edges: int,
    batch_size: int = 2048,
) -> tuple[list[dict], list[dict], dict]:
    """Run local/full diagnostics and support counterfactuals for fragile edges."""
    spec = FlatSpec.from_dict(payload["flatten_spec"])
    theta_before, _, client_deltas, fedavg_delta = fedavg_delta_from_payload(payload, spec)
    selected = [int(value) for value in payload["selected_client_ids"]]
    selected_tensor = torch.tensor(selected, dtype=torch.long)
    weights = torch.as_tensor(payload["fedavg_weights"], dtype=torch.float64)
    before = evaluate_state_edge_margins(model, spec, theta_before, cache, edges, batch_size=batch_size)
    full = evaluate_state_edge_margins(model, spec, theta_before + fedavg_delta, cache, edges, batch_size=batch_size)
    local = evaluate_local_edge_margins(model, spec, payload, cache, edges, batch_size=batch_size)
    edge_counts = torch.as_tensor(edge_counts, dtype=torch.float64)

    rows = []
    eligible = []
    for edge_id, edge in enumerate(edges):
        sample_weights = edge_sample_weights(edge_counts, edge_id)
        support_global = edge_counts[:, edge_id] > 0
        support_selected = support_global[selected_tensor]
        local_gain = _edge_gain(local, before, sample_weights, edge_id)
        full_gain = _edge_gain(full, before, sample_weights, edge_id)
        deficit = max(float(gamma) * local_gain - full_gain - float(tau), 0.0) if math.isfinite(local_gain) and math.isfinite(full_gain) else math.nan
        local_values = [float((local[client_id, edge_id] - before[client_id, edge_id]).item()) for client_id in selected if bool(support_global[client_id]) and torch.isfinite(local[client_id, edge_id]) and torch.isfinite(before[client_id, edge_id])]
        local_by_client = {
            str(client_id): float((local[client_id, edge_id] - before[client_id, edge_id]).item())
            for client_id in selected
            if bool(support_global[client_id])
            and torch.isfinite(local[client_id, edge_id])
            and torch.isfinite(before[client_id, edge_id])
        }
        pooled_sign = math.copysign(1.0, local_gain) if math.isfinite(local_gain) and local_gain != 0.0 else 0.0
        sign_agreement = (
            float(sum(math.copysign(1.0, value) == pooled_sign for value in local_values if value != 0.0) /
                  sum(value != 0.0 for value in local_values))
            if pooled_sign and any(value != 0.0 for value in local_values)
            else math.nan
        )
        row = {
            **dict(edge),
            "support_mass": float(weights[support_selected].sum().item()),
            "local_audit_gain": local_gain,
            "local_audit_gain_mean": finite_mean(local_values),
            "local_audit_gain_median": finite_median(local_values),
            "local_audit_gain_min": min(local_values) if local_values else math.nan,
            "local_audit_positive_rate": float(sum(value > 0 for value in local_values) / len(local_values)) if local_values else math.nan,
            "local_audit_sign_agreement": sign_agreement,
            "local_audit_gain_by_client": local_by_client,
            "gain_all_fedavg": full_gain,
            "visibility_deficit": deficit,
            "gain_support_normalized": math.nan,
            "gain_support_actual": math.nan,
            "dilution": math.nan,
            "interference": math.nan,
            "support_diagnostics_evaluated": False,
            "fragile_selected": False,
        }
        rows.append(row)
        if (
            int(edge["num_support_clients"]) >= int(min_support_clients)
            and math.isfinite(local_gain)
            and math.isfinite(deficit)
            and local_gain > 0.0
            and deficit > 0.0
        ):
            eligible.append(row)

    # Preserve the strongest deficit per class first, then apply a global cap.
    selected_rows = []
    by_class: dict[int, list[dict]] = {}
    for row in eligible:
        by_class.setdefault(int(row["class_id"]), []).append(row)
    for class_id in sorted(by_class):
        selected_rows.extend(
            sorted(
                by_class[class_id],
                key=lambda row: (-float(row["visibility_deficit"]), int(row["negative_id"])),
            )[: int(max_fragile_edges_per_class)]
        )
    selected_rows = sorted(selected_rows, key=lambda row: (-float(row["visibility_deficit"]), int(row["class_id"]), int(row["negative_id"])))[: int(max_total_edges)]
    selected_ids = {int(row["edge_id"]) for row in selected_rows}

    for row in rows:
        edge_id = int(row["edge_id"])
        support_global = edge_counts[:, edge_id] > 0
        support_selected = support_global[selected_tensor]
        actual_delta, actual_report = support_counterfactual_delta(client_deltas, weights, support_selected, normalized=False)
        norm_delta, norm_report = support_counterfactual_delta(client_deltas, weights, support_selected, normalized=True)
        actual_values = evaluate_single_edge_margins(model, spec, theta_before + actual_delta, cache, row, batch_size=batch_size)
        norm_values = evaluate_single_edge_margins(model, spec, theta_before + norm_delta, cache, row, batch_size=batch_size)
        sample_weights = edge_sample_weights(edge_counts, edge_id)
        actual_gain = _edge_gain(actual_values, before[:, edge_id:edge_id + 1], sample_weights, 0)
        norm_gain = _edge_gain(norm_values, before[:, edge_id:edge_id + 1], sample_weights, 0)
        row.update({
            "gain_support_actual": actual_gain,
            "gain_support_normalized": norm_gain,
            "dilution": norm_gain - actual_gain if math.isfinite(norm_gain) and math.isfinite(actual_gain) else math.nan,
            "interference": actual_gain - row["gain_all_fedavg"] if math.isfinite(actual_gain) and math.isfinite(row["gain_all_fedavg"]) else math.nan,
            "support_actual_update_norm": actual_report.get("raw_norm", math.nan),
            "support_normalized_update_norm": norm_report.get("raw_norm", math.nan),
            "support_diagnostics_evaluated": True,
            "fragile_selected": edge_id in selected_ids,
        })

    context = {
        "theta_before": theta_before,
        "fedavg_delta": fedavg_delta,
        "spec": spec,
        "before_margins": before,
        "fedavg_margins": full,
        "edge_counts": edge_counts,
        "fragile_edge_ids": sorted(selected_ids),
    }
    fragile_edges = [dict(row) for row in rows if bool(row["fragile_selected"])]
    return rows, fragile_edges, context


def edge_gradient(model, spec: FlatSpec, theta: torch.Tensor, cache: Mapping, edge: Mapping) -> tuple[torch.Tensor, dict]:
    """Compute one mean-margin gradient in the shared trainable Prompt space."""
    mask = edge_sample_mask(cache, edge)
    if not bool(mask.any()):
        raise ValueError("cannot differentiate an edge with no audit samples")
    features = torch.as_tensor(cache["features"], dtype=torch.float32)[mask]
    labels = torch.full((features.shape[0],), int(edge["class_id"]), dtype=torch.long)
    negatives = torch.full((features.shape[0],), int(edge["negative_id"]), dtype=torch.long)
    state = unflatten_state(theta, spec)
    margin, gradients = model.edge_gradient_from_cached_features(features, labels, negatives, state, list(spec.keys))
    flat = flatten_state(gradients, spec)
    return flat, {"edge_margin_at_fedavg": float(margin), "edge_gradient_norm": float(flat.norm().item()), "edge_audit_sample_count": int(features.shape[0])}
