#!/usr/bin/env python
"""D2: test whether update conflict predicts real tail-class damage.

Geometry is computed and frozen before the official test loader is iterated.
The test set is then used only as an offline diagnostic label: removing one
client's *actual FedAvg mass* measures that client's realized contribution.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.cusp_minimal import FlatSpec, flatten_state
from utils.d23_common import (
    D23_SCHEMA_VERSION,
    aggregate_metrics,
    build_trainer,
    class_split,
    collect_logits,
    compact_state_from_vector,
    load_dump,
    per_class_metrics,
    sha256_file,
    validate_dump,
    write_csv,
    write_json,
)


def safe_cosine(left: torch.Tensor, right: torch.Tensor, eps: float = 1e-12) -> float:
    denominator = float(left.norm().item() * right.norm().item())
    if denominator <= eps:
        return math.nan
    return float(torch.dot(left, right).item() / denominator)


def sign_disagreement(left: torch.Tensor, right: torch.Tensor, eps: float = 1e-12) -> float:
    active = (left.abs() > eps) & (right.abs() > eps)
    count = int(active.sum().item())
    if count == 0:
        return math.nan
    return float(((left[active] * right[active]) < 0).double().mean().item())


def _finite_pairs(x: Sequence[float], y: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
    pairs = [(float(a), float(b)) for a, b in zip(x, y) if math.isfinite(float(a)) and math.isfinite(float(b))]
    if not pairs:
        return np.asarray([]), np.asarray([])
    left, right = zip(*pairs)
    return np.asarray(left, dtype=np.float64), np.asarray(right, dtype=np.float64)


def _rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1) + 1.0
        start = end
    return ranks


def spearman(x: Sequence[float], y: Sequence[float]) -> float:
    left, right = _finite_pairs(x, y)
    if len(left) < 3:
        return math.nan
    left, right = _rankdata(left), _rankdata(right)
    left = left - left.mean()
    right = right - right.mean()
    denominator = float(np.sqrt(np.dot(left, left) * np.dot(right, right)))
    if denominator <= 0.0:
        return math.nan
    return float(np.dot(left, right) / denominator)


def binary_auc(labels: Sequence[bool], scores: Sequence[float]) -> float:
    usable = [(bool(label), float(score)) for label, score in zip(labels, scores) if math.isfinite(float(score))]
    positives = [score for label, score in usable if label]
    negatives = [score for label, score in usable if not label]
    if not positives or not negatives:
        return math.nan
    wins = sum((p > n) + 0.5 * (p == n) for p in positives for n in negatives)
    return float(wins / (len(positives) * len(negatives)))


def _mean(values: Sequence[float]) -> float:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    return float(sum(finite) / len(finite)) if finite else math.nan


def _dump_dir(root: Path, communication_round: int) -> Path:
    direct = root / f"round_{communication_round:03d}"
    nested = root / "v0_oracle" / f"round_{communication_round:03d}"
    if direct.is_dir():
        return direct
    return nested


def compute_geometry(payload: dict, metadata: dict) -> tuple[list[dict], dict]:
    spec = FlatSpec.from_dict(payload["flatten_spec"])
    before = flatten_state(payload["global_before_trainable"], spec)
    after = flatten_state(payload["global_after_fedavg_trainable"], spec)
    local = torch.stack([flatten_state(state, spec) for state in payload["local_trainable_states"]])
    deltas = local - before
    weights = torch.as_tensor(payload["fedavg_weights"], dtype=torch.float64).reshape(-1)
    counts = torch.as_tensor(payload["client_class_counts"], dtype=torch.float64)
    client_ids = [int(value) for value in payload["selected_client_ids"]]
    expected = before + torch.sum(weights[:, None] * deltas, dim=0)
    verification_error = float((expected - after).norm().item())
    verification_relative = verification_error / max(float((after - before).norm().item()), 1e-12)
    if verification_relative > 1e-5:
        raise RuntimeError(f"D2 dump does not reconstruct FedAvg (relative error={verification_relative:.3e})")

    fedavg_delta = after - before
    _, tail = class_split(payload["global_class_counts"])
    rows = []
    for class_id in tail:
        class_counts = counts[:, class_id]
        totals = counts.sum(dim=1).clamp_min(1.0)
        fractions = class_counts / totals
        any_support = class_counts > 0
        strict_support = fractions > 0.1
        for row_index, client_id in enumerate(client_ids):
            delta = deltas[row_index]
            # Exclude the target client from its reference direction. Including
            # it would mechanically inflate agreement, and a sole supporter
            # does not define an independent class consensus.
            peer_support = any_support.clone()
            peer_support[row_index] = False
            peer_support_mass = float(weights[peer_support].sum().item())
            if peer_support_mass > 0.0:
                support_direction = torch.sum(
                    weights[peer_support, None] * deltas[peer_support], dim=0
                ) / peer_support_mass
            else:
                support_direction = torch.zeros_like(fedavg_delta)
            remaining_mass = 1.0 - float(weights[row_index].item())
            peer_fedavg = (
                (fedavg_delta - weights[row_index] * delta) / remaining_mass
                if remaining_mass > 1e-12
                else torch.zeros_like(fedavg_delta)
            )
            rows.append({
                "communication_round": int(metadata["communication_round"]),
                "class_id": int(class_id),
                "client_id": client_id,
                "fedavg_weight": float(weights[row_index].item()),
                "client_class_count": int(class_counts[row_index].item()),
                "client_class_fraction": float(fractions[row_index].item()),
                "any_support": bool(any_support[row_index].item()),
                "strict_support": bool(strict_support[row_index].item()),
                "peer_support_count": int(peer_support.sum().item()),
                "peer_support_mass": peer_support_mass,
                "peer_support_available": peer_support_mass > 0.0,
                "client_delta_norm": float(delta.norm().item()),
                "support_direction_norm": float(support_direction.norm().item()),
                "cosine_to_support_direction": safe_cosine(delta, support_direction),
                "sign_disagreement_to_support_direction": sign_disagreement(delta, support_direction),
                "cosine_to_fedavg": safe_cosine(delta, fedavg_delta),
                "sign_disagreement_to_fedavg": sign_disagreement(delta, fedavg_delta),
                "cosine_to_peer_fedavg": safe_cosine(delta, peer_fedavg),
                "sign_disagreement_to_peer_fedavg": sign_disagreement(delta, peer_fedavg),
            })
    audit = {
        "communication_round": int(metadata["communication_round"]),
        "parameter_space": "uploaded_lora_parameter_delta",
        "num_clients": len(client_ids),
        "tail_class_ids": tail,
        "fedavg_reconstruction_error": verification_error,
        "fedavg_reconstruction_relative_error": verification_relative,
    }
    return rows, audit


def summarize_rows(rows: list[dict], communication_round: int) -> list[dict]:
    summaries = []
    for scope, predicate in (
        ("all", lambda row: True),
        ("all_with_reference", lambda row: bool(row["peer_support_available"])),
        ("any_support_with_peer", lambda row: bool(row["any_support"]) and bool(row["peer_support_available"])),
        ("strict_support_with_peer", lambda row: bool(row["strict_support"]) and bool(row["peer_support_available"])),
    ):
        selected = [row for row in rows if predicate(row)]
        utility = [row["tail_margin_contribution"] for row in selected]
        harmful = [value < 0.0 for value in utility]
        sign_score = [row["sign_disagreement_to_support_direction"] for row in selected]
        cosine = [row["cosine_to_support_direction"] for row in selected]
        peer_sign_score = [row["sign_disagreement_to_peer_fedavg"] for row in selected]
        peer_cosine = [row["cosine_to_peer_fedavg"] for row in selected]
        harmful_losses = [max(0.0, -float(value)) for value in utility]
        summaries.append({
            "communication_round": communication_round,
            "scope": scope,
            "pair_count": len(selected),
            "harmful_pair_count": sum(harmful),
            "harmful_pair_rate": sum(harmful) / len(harmful) if harmful else math.nan,
            "false_protection_rate": sum(harmful) / len(harmful) if scope in {"any_support_with_peer", "strict_support_with_peer"} and harmful else math.nan,
            "mean_harmful_margin_loss": _mean([value for value in harmful_losses if value > 0.0]),
            "total_harmful_margin_loss": float(sum(harmful_losses)),
            "spearman_cosine_vs_utility": spearman(cosine, utility),
            "spearman_sign_disagreement_vs_utility": spearman(sign_score, utility),
            "auc_negative_cosine_for_harm": binary_auc(harmful, [-float(value) for value in cosine]),
            "auc_sign_disagreement_for_harm": binary_auc(harmful, sign_score),
            "spearman_peer_fedavg_cosine_vs_utility": spearman(peer_cosine, utility),
            "spearman_peer_fedavg_sign_disagreement_vs_utility": spearman(peer_sign_score, utility),
            "auc_peer_fedavg_negative_cosine_for_harm": binary_auc(
                harmful, [-float(value) for value in peer_cosine]
            ),
            "auc_peer_fedavg_sign_disagreement_for_harm": binary_auc(
                harmful, peer_sign_score
            ),
        })
    return summaries


def analyze_round(trainer, payload: dict, metadata: dict, geometry_rows: list[dict]) -> tuple[list[dict], list[dict]]:
    spec = FlatSpec.from_dict(payload["flatten_spec"])
    before = flatten_state(payload["global_before_trainable"], spec)
    after = flatten_state(payload["global_after_fedavg_trainable"], spec)
    local = torch.stack([flatten_state(state, spec) for state in payload["local_trainable_states"]])
    deltas = local - before
    weights = torch.as_tensor(payload["fedavg_weights"], dtype=torch.float64).reshape(-1)
    client_ids = [int(value) for value in payload["selected_client_ids"]]
    head, tail = class_split(payload["global_class_counts"])

    print(f"D2 round {metadata['communication_round']}: evaluate FedAvg baseline", flush=True)
    baseline_logits, labels = collect_logits(trainer, compact_state_from_vector(after, spec))
    baseline_classes = per_class_metrics(baseline_logits, labels)
    baseline_global = aggregate_metrics(baseline_logits, labels, head, tail)
    removal = {}
    for index, client_id in enumerate(client_ids):
        print(f"  remove client {index + 1:02d}/{len(client_ids)} (id={client_id})", flush=True)
        removed_state = after - weights[index] * deltas[index]
        logits, removed_labels = collect_logits(
            trainer, compact_state_from_vector(removed_state, spec)
        )
        if not torch.equal(labels, removed_labels):
            raise RuntimeError("D2 test-loader order changed between counterfactual evaluations")
        removal[client_id] = {
            "classes": per_class_metrics(logits, labels),
            "global": aggregate_metrics(logits, labels, head, tail),
        }

    merged = []
    for geometry in geometry_rows:
        class_id = int(geometry["class_id"])
        client_id = int(geometry["client_id"])
        base = baseline_classes[class_id]
        counterfactual = removal[client_id]["classes"][class_id]
        removed_global = removal[client_id]["global"]
        merged.append({
            **geometry,
            "tail_accuracy_contribution": float(base["accuracy"] - counterfactual["accuracy"]),
            "tail_margin_contribution": float(base["margin"] - counterfactual["margin"]),
            "tail_nll_benefit": float(counterfactual["nll"] - base["nll"]),
            "head_accuracy_contribution": float(
                baseline_global["head_accuracy"] - removed_global["head_accuracy"]
            ),
            "is_harmful_by_margin": bool(base["margin"] - counterfactual["margin"] < 0.0),
        })
    return merged, summarize_rows(merged, int(metadata["communication_round"]))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dump-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--rounds", default="20,50,80")
    parser.add_argument("--eval-batch-size", type=int, default=256)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rounds = [int(value) for value in args.rounds.split(",") if value.strip()]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    loaded = []
    all_geometry, audits = [], []
    for communication_round in rounds:
        directory = _dump_dir(args.dump_root, communication_round)
        payload, metadata = load_dump(directory)
        validate_dump(payload, metadata)
        if int(metadata["communication_round"]) != communication_round:
            raise RuntimeError(f"Round mismatch in {directory}")
        geometry, audit = compute_geometry(payload, metadata)
        all_geometry.extend(geometry)
        audits.append({**audit, "dump_sha256": sha256_file(directory / "round_state.pt")})
        loaded.append((payload, metadata))

    geometry_path = args.output_dir / "d2_geometry_frozen.csv"
    write_csv(geometry_path, all_geometry)
    manifest = {
        "schema_version": D23_SCHEMA_VERSION,
        "diagnostic": "D2_update_conflict_alignment",
        "seed": 42,
        "rounds": rounds,
        "geometry_frozen_before_test": True,
        "geometry_sha256": sha256_file(geometry_path),
        "test_accessed": False,
        "round_audits": audits,
    }
    write_json(args.output_dir / "d2_manifest.json", manifest)
    print(f"D2 geometry frozen before test access: {geometry_path}", flush=True)

    _, trainer = build_trainer(loaded[0][1], args.output_dir / "eval_runtime", args.eval_batch_size)
    all_utility, all_summary = [], []
    offset = 0
    per_round_count = len(all_geometry) // len(rounds)
    for payload, metadata in loaded:
        geometry = all_geometry[offset:offset + per_round_count]
        offset += per_round_count
        utility, summary = analyze_round(trainer, payload, metadata, geometry)
        all_utility.extend(utility)
        all_summary.extend(summary)

    write_csv(args.output_dir / "d2_client_class_utility.csv", all_utility)
    write_csv(args.output_dir / "d2_round_summary.csv", all_summary)
    supported = [row for row in all_summary if row["scope"] == "all"]
    def best_finite_auc(row):
        values = [
            float(row["auc_peer_fedavg_negative_cosine_for_harm"]),
            float(row["auc_peer_fedavg_sign_disagreement_for_harm"]),
        ]
        values = [value for value in values if math.isfinite(value)]
        return max(values) if values else math.nan
    round_proxy = [
        best_finite_auc(row)
        for row in supported
        if math.isfinite(best_finite_auc(row))
    ]
    strong_rounds = sum(value >= 0.6 for value in round_proxy)
    proxy_pass = len(round_proxy) == len(rounds) and _mean(round_proxy) >= 0.65 and strong_rounds >= 2
    verdict = {
        **manifest,
        "test_accessed": True,
        "utility_definition": (
            "FedAvg metric minus metric after subtracting client_i's actual FedAvg mass; "
            "positive means the uploaded client update was beneficial."
        ),
        "primary_scope": "all client-tail pairs",
        "primary_proxy": (
            "best of two predeclared class-label-free leave-one-client-out peer-FedAvg "
            "signals: negative cosine and sign disagreement"
        ),
        "primary_round_harm_auc": round_proxy,
        "primary_mean_harm_auc": _mean(round_proxy),
        "rounds_with_auc_at_least_0p6": strong_rounds,
        "proxy_pass": proxy_pass,
        "verdict": "D2_CONFLICT_PROXY_SUPPORTED" if proxy_pass else "D2_CONFLICT_PROXY_NOT_SUPPORTED",
        "method_ready": False,
        "note": "D2 is a seed-42 diagnostic and does not itself define a deployable conflict resolver.",
    }
    write_json(args.output_dir / "d2_verdict.json", verdict)
    print(json.dumps(verdict, indent=2), flush=True)


if __name__ == "__main__":
    main()
