#!/usr/bin/env python
"""D2b: test whether scalar client aggregation can express client-class utility.

This is a deliberately non-deployable, train-selected oracle gate for CCAR.
It uses the frozen D2/D3 dumps without additional federated training. Candidate
weights are learned on a train-only fit split, selected on a disjoint train
calibration split, frozen to disk, and only then evaluated on official test.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.analyze_d3_boundary import apply_logit_adjustment, select_tau
from utils.cusp_minimal import FlatSpec, flatten_state
from utils.d23_common import (
    D23_SCHEMA_VERSION,
    aggregate_metrics,
    build_global_train_eval_loader,
    build_trainer,
    class_split,
    collect_logits,
    compact_state_from_vector,
    load_dump,
    per_class_metrics,
    sha256_file,
    stratified_fit_calibration_split,
    validate_dump,
    write_csv,
    write_json,
)


METHODS = (
    "fedavg",
    "fedavg_la",
    "scalar_oracle",
    "scalar_oracle_la",
    "class_conditional_oracle",
    "class_conditional_oracle_la",
)


def _dump_dir(root: Path, communication_round: int) -> Path:
    direct = root / f"round_{communication_round:03d}"
    nested = root / "v0_oracle" / f"round_{communication_round:03d}"
    return direct if direct.is_dir() else nested


def read_utility_rows(path: Path) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(
            f"D2b requires D2 client-class utility labels: {path}"
        )
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def utility_interaction_report(
    rows: Sequence[Mapping], communication_round: int
) -> dict:
    selected = [
        row for row in rows
        if int(float(row["communication_round"])) == int(communication_round)
    ]
    clients = sorted({int(float(row["client_id"])) for row in selected})
    classes = sorted({int(float(row["class_id"])) for row in selected})
    if len(clients) != 30 or len(classes) != 20 or len(selected) != 600:
        raise RuntimeError(
            f"D2b expected a 20x30 utility matrix at round {communication_round}; "
            f"got classes={len(classes)}, clients={len(clients)}, rows={len(selected)}"
        )
    client_index = {client_id: index for index, client_id in enumerate(clients)}
    class_index = {class_id: index for index, class_id in enumerate(classes)}
    matrix = torch.empty(len(classes), len(clients), dtype=torch.float64)
    for row in selected:
        matrix[
            class_index[int(float(row["class_id"]))],
            client_index[int(float(row["client_id"]))],
        ] = float(row["tail_margin_contribution"])

    grand = matrix.mean()
    additive = matrix.mean(dim=1, keepdim=True) + matrix.mean(dim=0, keepdim=True) - grand
    interaction = matrix - additive
    centered_energy = float(torch.sum((matrix - grand) ** 2).item())
    interaction_energy = float(torch.sum(interaction ** 2).item())
    singular_values = torch.linalg.svdvals(interaction)
    squared = singular_values.square()
    effective_rank = (
        float(squared.sum().square().item() / squared.square().sum().item())
        if float(squared.square().sum().item()) > 0.0 else 0.0
    )
    client_sign_flip = [
        bool((matrix[:, index] > 0).any() and (matrix[:, index] < 0).any())
        for index in range(len(clients))
    ]
    class_sign_flip = [
        bool((matrix[index] > 0).any() and (matrix[index] < 0).any())
        for index in range(len(classes))
    ]
    return {
        "communication_round": int(communication_round),
        "class_count": len(classes),
        "client_count": len(clients),
        "interaction_energy_ratio": interaction_energy / max(centered_energy, 1e-18),
        "additive_client_plus_class_explained_ratio": 1.0 - interaction_energy / max(centered_energy, 1e-18),
        "interaction_effective_rank": effective_rank,
        "interaction_rank1_energy_fraction": (
            float(squared[0].item() / squared.sum().item())
            if len(squared) and float(squared.sum().item()) > 0.0 else math.nan
        ),
        "clients_with_both_beneficial_and_harmful_classes_rate": sum(client_sign_flip) / len(client_sign_flip),
        "classes_with_both_beneficial_and_harmful_clients_rate": sum(class_sign_flip) / len(class_sign_flip),
        "utility_positive_rate": float((matrix > 0).double().mean().item()),
        "utility_matrix_frobenius_norm": float(matrix.norm().item()),
    }


def client_vectors(payload: Mapping) -> tuple[FlatSpec, torch.Tensor, torch.Tensor, torch.Tensor]:
    spec = FlatSpec.from_dict(payload["flatten_spec"])
    before = flatten_state(payload["global_before_trainable"], spec)
    local = torch.stack(
        [flatten_state(state, spec) for state in payload["local_trainable_states"]]
    )
    deltas = local - before
    weights = torch.as_tensor(payload["fedavg_weights"], dtype=torch.float64).reshape(-1)
    return spec, before, deltas, weights


def vector_from_weights(
    before: torch.Tensor, deltas: torch.Tensor, weights: torch.Tensor
) -> torch.Tensor:
    weights = torch.as_tensor(weights, dtype=torch.float64).reshape(-1)
    return before + torch.sum(weights[:, None] * deltas, dim=0)


def compose_scalar_logits(
    baseline: torch.Tensor,
    responses: torch.Tensor,
    fedavg_weights: torch.Tensor,
    candidate_weights: torch.Tensor,
) -> torch.Tensor:
    ratio = (
        torch.as_tensor(candidate_weights).float()
        - torch.as_tensor(fedavg_weights).float()
    ) / torch.as_tensor(fedavg_weights).float().clamp_min(1e-12)
    return baseline + torch.einsum("k,knc->nc", ratio, responses)


def compose_class_logits(
    baseline: torch.Tensor,
    responses: torch.Tensor,
    fedavg_weights: torch.Tensor,
    candidate_weights: torch.Tensor,
    tail: Sequence[int],
) -> torch.Tensor:
    result = baseline.clone()
    ratio = (
        torch.as_tensor(candidate_weights).float()
        - torch.as_tensor(fedavg_weights).float()[:, None]
    ) / torch.as_tensor(fedavg_weights).float().clamp_min(1e-12)[:, None]
    tail_index = torch.as_tensor(list(tail), dtype=torch.long)
    result[:, tail_index] += torch.einsum(
        "kt,knt->nt", ratio, responses[:, :, tail_index]
    )
    return result


def collect_functional_responses(
    trainer,
    spec: FlatSpec,
    before: torch.Tensor,
    deltas: torch.Tensor,
    weights: torch.Tensor,
    loader,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    fedavg_vector = vector_from_weights(before, deltas, weights)
    baseline, labels = collect_logits(
        trainer, compact_state_from_vector(fedavg_vector, spec), loader
    )
    responses = []
    for index in range(len(weights)):
        print(f"  train response {index + 1:02d}/{len(weights)}", flush=True)
        removed = fedavg_vector - weights[index] * deltas[index]
        logits, observed_labels = collect_logits(
            trainer, compact_state_from_vector(removed, spec), loader
        )
        if not torch.equal(labels, observed_labels):
            raise RuntimeError("D2b train loader order changed across response evaluations")
        responses.append(baseline - logits)
    return baseline, labels, torch.stack(responses)


def build_train_subset_loader(cfg, trainer, indices: torch.Tensor):
    """Build a deterministic evaluation loader for selected global-train rows."""
    from Dassl.dassl.data.data_manager import build_data_loader
    from Dassl.dassl.data.transforms import build_transform

    source = list(getattr(trainer.dm.dataset, "train_x", []) or [])
    selected = [source[int(index)] for index in torch.as_tensor(indices).tolist()]
    return build_data_loader(
        cfg,
        sampler_type="SequentialSampler",
        data_source=selected,
        batch_size=cfg.DATALOADER.TEST.BATCH_SIZE,
        tfm=build_transform(cfg, is_train=False),
        is_train=False,
        dataset_wrapper=None,
        class_names=trainer.dm.dataset.classnames,
        drop_last=False,
    )


def _balanced_batch_indices(
    labels: torch.Tensor,
    rng: np.random.Generator,
    samples_per_class: int,
) -> torch.Tensor:
    selected = []
    labels_np = torch.as_tensor(labels).cpu().numpy()
    for class_id in sorted(np.unique(labels_np).tolist()):
        pool = np.flatnonzero(labels_np == class_id)
        selected.extend(
            rng.choice(pool, size=int(samples_per_class), replace=len(pool) < samples_per_class).tolist()
        )
    rng.shuffle(selected)
    return torch.as_tensor(selected, dtype=torch.long)


def optimize_weight_distributions(
    baseline: torch.Tensor,
    responses: torch.Tensor,
    labels: torch.Tensor,
    fedavg_weights: torch.Tensor,
    tail: Sequence[int],
    *,
    steps: int,
    samples_per_class: int,
    learning_rate: float,
    kl_weight: float,
    seed: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, list[dict]]:
    """Learn alternative scalar and per-tail-class distributions at gamma=1."""
    base_weights = torch.as_tensor(fedavg_weights).float().to(device)
    scalar_logits = base_weights.clamp_min(1e-12).log().detach().clone().requires_grad_(True)
    class_logits = (
        base_weights.clamp_min(1e-12).log()[:, None]
        .repeat(1, len(tail)).detach().clone().requires_grad_(True)
    )
    optimizer = torch.optim.Adam([scalar_logits, class_logits], lr=float(learning_rate))
    rng = np.random.default_rng(int(seed))
    trace = []
    tail_index = torch.as_tensor(list(tail), dtype=torch.long, device=device)
    log_base = base_weights.clamp_min(1e-12).log()
    for step in range(int(steps)):
        batch_index = _balanced_batch_indices(labels, rng, samples_per_class)
        batch_logits = baseline[batch_index].to(device)
        batch_response = responses[:, batch_index].to(device)
        batch_labels = labels[batch_index].to(device)
        scalar_weights = torch.softmax(scalar_logits, dim=0)
        class_weights = torch.softmax(class_logits, dim=0)
        scalar_ratio = (scalar_weights - base_weights) / base_weights.clamp_min(1e-12)
        class_ratio = (
            class_weights - base_weights[:, None]
        ) / base_weights.clamp_min(1e-12)[:, None]
        scalar_output = batch_logits + torch.einsum(
            "k,knc->nc", scalar_ratio, batch_response
        )
        class_output = batch_logits.clone()
        class_output[:, tail_index] += torch.einsum(
            "kt,knt->nt", class_ratio, batch_response[:, :, tail_index]
        )
        scalar_kl = torch.sum(
            scalar_weights * (scalar_weights.clamp_min(1e-12).log() - log_base)
        )
        class_kl = torch.mean(torch.sum(
            class_weights * (
                class_weights.clamp_min(1e-12).log() - log_base[:, None]
            ), dim=0
        ))
        scalar_loss = F.cross_entropy(scalar_output, batch_labels) + float(kl_weight) * scalar_kl
        class_loss = F.cross_entropy(class_output, batch_labels) + float(kl_weight) * class_kl
        loss = scalar_loss + class_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step == 0 or (step + 1) % 50 == 0 or step + 1 == int(steps):
            row = {
                "step": step + 1,
                "scalar_loss": float(scalar_loss.detach().cpu().item()),
                "class_loss": float(class_loss.detach().cpu().item()),
                "scalar_kl": float(scalar_kl.detach().cpu().item()),
                "class_kl": float(class_kl.detach().cpu().item()),
            }
            trace.append(row)
            print(
                f"  optimize {step + 1:04d}/{steps}: "
                f"scalar={row['scalar_loss']:.5f} class={row['class_loss']:.5f}",
                flush=True,
            )
    return (
        torch.softmax(scalar_logits.detach(), dim=0).cpu(),
        torch.softmax(class_logits.detach(), dim=0).cpu(),
        trace,
    )


def select_gamma(
    baseline: torch.Tensor,
    responses: torch.Tensor,
    labels: torch.Tensor,
    fedavg_weights: torch.Tensor,
    alternative_weights: torch.Tensor,
    head: Sequence[int],
    tail: Sequence[int],
    gammas: Sequence[float],
    *,
    class_conditional: bool,
    max_head_damage: float,
) -> tuple[float, torch.Tensor, list[dict]]:
    baseline_metrics = aggregate_metrics(baseline, labels, head, tail)
    rows = []
    candidates = {}
    base = torch.as_tensor(fedavg_weights).float()
    if class_conditional:
        base = base[:, None]
    for gamma in gammas:
        candidate = (1.0 - float(gamma)) * base + float(gamma) * alternative_weights
        logits = (
            compose_class_logits(baseline, responses, fedavg_weights, candidate, tail)
            if class_conditional
            else compose_scalar_logits(baseline, responses, fedavg_weights, candidate)
        )
        metrics = aggregate_metrics(logits, labels, head, tail)
        head_damage = baseline_metrics["head_accuracy"] - metrics["head_accuracy"]
        row = {
            "gamma": float(gamma),
            "class_conditional": bool(class_conditional),
            "head_damage_vs_fedavg": head_damage,
            "head_safe": head_damage <= float(max_head_damage),
            **metrics,
        }
        rows.append(row)
        candidates[float(gamma)] = candidate
    safe = [row for row in rows if row["head_safe"]]
    pool = safe or rows
    selected = max(
        pool,
        key=lambda row: (
            float(row["head_tail_harmonic"]),
            float(row["balanced_accuracy"]),
            -float(row["head_damage_vs_fedavg"]),
            -float(row["gamma"]),
        ),
    )
    gamma = float(selected["gamma"])
    return gamma, candidates[gamma], rows


def select_exact_scalar_gamma(
    trainer,
    spec: FlatSpec,
    before: torch.Tensor,
    deltas: torch.Tensor,
    fedavg_weights: torch.Tensor,
    alternative_weights: torch.Tensor,
    loader,
    labels: torch.Tensor,
    baseline_logits: torch.Tensor,
    head: Sequence[int],
    tail: Sequence[int],
    gammas: Sequence[float],
    max_head_damage: float,
) -> tuple[float, torch.Tensor, torch.Tensor, list[dict]]:
    """Give the scalar ceiling an exact-model gamma selection advantage."""
    baseline_metrics = aggregate_metrics(baseline_logits, labels, head, tail)
    rows, candidates, logits_by_gamma = [], {}, {}
    base = torch.as_tensor(fedavg_weights).float()
    for gamma in gammas:
        candidate = (1.0 - float(gamma)) * base + float(gamma) * alternative_weights
        if abs(float(gamma)) <= 1e-12:
            logits = baseline_logits
        else:
            print(f"  exact scalar gamma={float(gamma):g}", flush=True)
            logits, observed_labels = exact_scalar_logits(
                trainer, spec, before, deltas, candidate, loader
            )
            if not torch.equal(labels, observed_labels):
                raise RuntimeError("D2b calibration loader order changed across scalar gammas")
        metrics = aggregate_metrics(logits, labels, head, tail)
        head_damage = baseline_metrics["head_accuracy"] - metrics["head_accuracy"]
        rows.append({
            "gamma": float(gamma),
            "class_conditional": False,
            "head_damage_vs_fedavg": head_damage,
            "head_safe": head_damage <= float(max_head_damage),
            **metrics,
        })
        candidates[float(gamma)] = candidate
        logits_by_gamma[float(gamma)] = logits
    safe = [row for row in rows if row["head_safe"]]
    selected = max(
        safe or rows,
        key=lambda row: (
            float(row["head_tail_harmonic"]),
            float(row["balanced_accuracy"]),
            -float(row["head_damage_vs_fedavg"]),
            -float(row["gamma"]),
        ),
    )
    gamma = float(selected["gamma"])
    return gamma, candidates[gamma], logits_by_gamma[gamma], rows


def exact_scalar_logits(
    trainer, spec, before, deltas, weights, loader
) -> tuple[torch.Tensor, torch.Tensor]:
    vector = vector_from_weights(before, deltas, weights)
    return collect_logits(trainer, compact_state_from_vector(vector, spec), loader)


def exact_class_conditional_logits(
    trainer,
    spec: FlatSpec,
    before: torch.Tensor,
    deltas: torch.Tensor,
    class_weights: torch.Tensor,
    tail: Sequence[int],
    loader,
    baseline_logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    split_name: str,
) -> torch.Tensor:
    result = baseline_logits.clone()
    for tail_index, class_id in enumerate(tail):
        print(
            f"  exact {split_name} class state {tail_index + 1:02d}/{len(tail)} "
            f"(class={class_id})",
            flush=True,
        )
        vector = vector_from_weights(before, deltas, class_weights[:, tail_index])
        logits, observed_labels = collect_logits(
            trainer, compact_state_from_vector(vector, spec), loader
        )
        if not torch.equal(labels, observed_labels):
            raise RuntimeError(f"D2b {split_name} loader order changed across class states")
        result[:, int(class_id)] = logits[:, int(class_id)]
    return result


def _group_accuracy(per_class: Sequence[Mapping], class_ids: Sequence[int]) -> float:
    values = [float(per_class[int(class_id)]["accuracy"]) for class_id in class_ids]
    values = [value for value in values if math.isfinite(value)]
    return float(sum(values) / len(values)) if values else math.nan


def tail_coverage_groups(payload: Mapping, tail: Sequence[int]) -> tuple[list[int], list[int]]:
    counts = torch.as_tensor(payload["client_class_counts"], dtype=torch.float64)
    fractions = counts / counts.sum(dim=1, keepdim=True).clamp_min(1.0)
    covered = [int(class_id) for class_id in tail if bool((fractions[:, class_id] > 0.1).any())]
    covered_set = set(covered)
    uncovered = [int(class_id) for class_id in tail if int(class_id) not in covered_set]
    return covered, uncovered


def _method_metrics(
    round_id: int,
    method: str,
    logits: torch.Tensor,
    labels: torch.Tensor,
    head: Sequence[int],
    tail: Sequence[int],
    covered: Sequence[int],
    uncovered: Sequence[int],
) -> dict:
    per_class = per_class_metrics(logits, labels)
    return {
        "communication_round": int(round_id),
        "method": method,
        **aggregate_metrics(logits, labels, head, tail),
        "covered_tail_accuracy": _group_accuracy(per_class, covered),
        "uncovered_tail_accuracy": _group_accuracy(per_class, uncovered),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dump-root", type=Path, required=True)
    parser.add_argument("--d2-utility", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--rounds", default="20,50,80")
    parser.add_argument("--eval-batch-size", type=int, default=256)
    parser.add_argument("--calibration-fraction", type=float, default=0.2)
    parser.add_argument("--gammas", default="0,0.1,0.25,0.5,0.75,1.0")
    parser.add_argument("--taus", default="0,0.25,0.5,0.75,1.0,1.5,2.0")
    parser.add_argument("--optimization-steps", type=int, default=300)
    parser.add_argument("--samples-per-class", type=int, default=4)
    parser.add_argument("--optimization-lr", type=float, default=0.05)
    parser.add_argument("--kl-weight", type=float, default=1e-3)
    parser.add_argument("--max-head-damage", type=float, default=2.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rounds = [int(value) for value in args.rounds.split(",") if value.strip()]
    gammas = [float(value) for value in args.gammas.split(",") if value.strip()]
    taus = [float(value) for value in args.taus.split(",") if value.strip()]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    utility_rows = read_utility_rows(args.d2_utility)
    interaction_rows = [utility_interaction_report(utility_rows, round_id) for round_id in rounds]
    write_csv(args.output_dir / "d2b_utility_interaction.csv", interaction_rows)

    loaded = []
    for round_id in rounds:
        directory = _dump_dir(args.dump_root, round_id)
        payload, metadata = load_dump(directory)
        validate_dump(payload, metadata)
        if int(metadata["communication_round"]) != round_id:
            raise RuntimeError(f"Round mismatch in {directory}")
        loaded.append((directory, payload, metadata))

    cfg, trainer = build_trainer(
        loaded[0][2], args.output_dir / "eval_runtime", args.eval_batch_size
    )
    train_loader = build_global_train_eval_loader(cfg, trainer)
    device = trainer.device if torch.cuda.is_available() else torch.device("cpu")
    gamma_rows, tau_rows, trace_rows, frozen_rows, cached = [], [], [], [], []

    for directory, payload, metadata in loaded:
        round_id = int(metadata["communication_round"])
        print(f"D2b round {round_id}: collect train-only functional responses", flush=True)
        spec, before, deltas, fedavg_weights = client_vectors(payload)
        train_baseline, train_labels, responses = collect_functional_responses(
            trainer, spec, before, deltas, fedavg_weights, train_loader
        )
        fit_index, calibration_index = stratified_fit_calibration_split(
            train_labels, seed=42, calibration_fraction=args.calibration_fraction
        )
        head, tail = class_split(payload["global_class_counts"])
        calibration_loader = build_train_subset_loader(
            cfg, trainer, calibration_index
        )
        alternative_scalar, alternative_class, trace = optimize_weight_distributions(
            train_baseline[fit_index],
            responses[:, fit_index],
            train_labels[fit_index],
            fedavg_weights,
            tail,
            steps=args.optimization_steps,
            samples_per_class=args.samples_per_class,
            learning_rate=args.optimization_lr,
            kl_weight=args.kl_weight,
            seed=42000 + round_id,
            device=device,
        )
        trace_rows.extend({"communication_round": round_id, **row} for row in trace)
        scalar_gamma_approx, _, scalar_grid = select_gamma(
            train_baseline[calibration_index],
            responses[:, calibration_index],
            train_labels[calibration_index],
            fedavg_weights,
            alternative_scalar,
            head,
            tail,
            gammas,
            class_conditional=False,
            max_head_damage=args.max_head_damage,
        )
        class_gamma, class_weights, class_grid = select_gamma(
            train_baseline[calibration_index],
            responses[:, calibration_index],
            train_labels[calibration_index],
            fedavg_weights,
            alternative_class,
            head,
            tail,
            gammas,
            class_conditional=True,
            max_head_damage=args.max_head_damage,
        )
        gamma_rows.extend(
            {
                "communication_round": round_id,
                "oracle": "scalar",
                "selection_space": "functional_approximation",
                "selected": row["gamma"] == scalar_gamma_approx,
                **row,
            }
            for row in scalar_grid
        )
        gamma_rows.extend(
            {
                "communication_round": round_id,
                "oracle": "class_conditional",
                "selection_space": "functional_approximation",
                "selected": row["gamma"] == class_gamma,
                **row,
            }
            for row in class_grid
        )

        print(f"D2b round {round_id}: exact train-calibration oracle evaluation", flush=True)
        calibration_labels = train_labels[calibration_index]
        scalar_gamma, scalar_weights, scalar_calibration, exact_scalar_grid = (
            select_exact_scalar_gamma(
                trainer,
                spec,
                before,
                deltas,
                fedavg_weights,
                alternative_scalar,
                calibration_loader,
                calibration_labels,
                train_baseline[calibration_index],
                head,
                tail,
                gammas,
                args.max_head_damage,
            )
        )
        gamma_rows.extend(
            {
                "communication_round": round_id,
                "oracle": "scalar",
                "selection_space": "exact_model",
                "selected": row["gamma"] == scalar_gamma,
                **row,
            }
            for row in exact_scalar_grid
        )
        class_calibration = exact_class_conditional_logits(
            trainer,
            spec,
            before,
            deltas,
            class_weights,
            tail,
            calibration_loader,
            train_baseline[calibration_index],
            calibration_labels,
            split_name="train-calibration",
        )
        priors = torch.bincount(
            train_labels[fit_index], minlength=train_baseline.shape[1]
        ).float()
        priors /= priors.sum()
        exact_sources = {
            "fedavg": train_baseline[calibration_index],
            "scalar_oracle": scalar_calibration,
            "class_conditional_oracle": class_calibration,
        }
        selected_taus = {}
        for method, logits in exact_sources.items():
            tau, rows = select_tau(
                logits,
                calibration_labels,
                priors,
                taus,
                head,
                tail,
            )
            selected_taus[method] = tau
            tau_rows.extend({
                "communication_round": round_id,
                "source_method": method,
                "selected": float(row["tau"]) == tau,
                **row,
            } for row in rows)

        approx_scalar = compose_scalar_logits(
            train_baseline[calibration_index],
            responses[:, calibration_index],
            fedavg_weights,
            scalar_weights,
        )
        approx_class = compose_class_logits(
            train_baseline[calibration_index],
            responses[:, calibration_index],
            fedavg_weights,
            class_weights,
            tail,
        )
        artifact_path = args.output_dir / f"d2b_frozen_round_{round_id:03d}.pt"
        torch.save({
            "communication_round": round_id,
            "scalar_gamma": scalar_gamma,
            "class_gamma": class_gamma,
            "scalar_weights": scalar_weights,
            "class_weights": class_weights,
            "selected_taus": selected_taus,
            "priors": priors,
            "tail_class_ids": tail,
            "fit_indices": fit_index,
            "calibration_indices": calibration_index,
        }, artifact_path)
        frozen_rows.append({
            "communication_round": round_id,
            "scalar_gamma": scalar_gamma,
            "class_gamma": class_gamma,
            "fedavg_tau": selected_taus["fedavg"],
            "scalar_tau": selected_taus["scalar_oracle"],
            "class_tau": selected_taus["class_conditional_oracle"],
            "scalar_approximation_mae_on_calibration": float(
                (approx_scalar - scalar_calibration).abs().mean().item()
            ),
            "class_approximation_mae_on_calibration": float(
                (approx_class - class_calibration).abs().mean().item()
            ),
            "artifact": str(artifact_path),
            "artifact_sha256": sha256_file(artifact_path),
            "dump_sha256": sha256_file(directory / "round_state.pt"),
        })
        cached.append((payload, metadata, artifact_path))
        del (
            train_baseline,
            train_labels,
            responses,
            scalar_calibration,
            class_calibration,
        )

    write_csv(args.output_dir / "d2b_optimization_trace.csv", trace_rows)
    write_csv(args.output_dir / "d2b_gamma_selection.csv", gamma_rows)
    write_csv(args.output_dir / "d2b_tau_selection.csv", tau_rows)
    write_csv(args.output_dir / "d2b_frozen_choices.csv", frozen_rows)
    frozen_manifest = {
        "schema_version": D23_SCHEMA_VERSION,
        "diagnostic": "D2b_scalar_aggregation_ceiling",
        "seed": 42,
        "rounds": rounds,
        "oracle_status": "non_deployable_train_selected_upper_bound",
        "fit_source": "deterministic global-train fit split",
        "selection_source": "disjoint deterministic global-train calibration split",
        "choices_frozen_before_test": True,
        "candidate_selection_used_official_test": False,
        "interaction_report_uses_precomputed_d2_test_utility": True,
        "new_test_inference_accessed": False,
        "utility_csv_sha256": sha256_file(args.d2_utility),
        "frozen_choices_sha256": sha256_file(args.output_dir / "d2b_frozen_choices.csv"),
    }
    write_json(args.output_dir / "d2b_manifest.json", frozen_manifest)
    print("D2b scalar/class choices and LA parameters frozen before test", flush=True)

    method_rows, contrast_rows = [], []
    for payload, metadata, artifact_path in cached:
        round_id = int(metadata["communication_round"])
        artifact = torch.load(artifact_path, map_location="cpu", weights_only=False)
        spec, before, deltas, fedavg_weights = client_vectors(payload)
        head, tail = class_split(payload["global_class_counts"])
        covered, uncovered = tail_coverage_groups(payload, tail)
        print(f"D2b round {round_id}: exact official-test oracle evaluation", flush=True)
        fedavg_test, test_labels = exact_scalar_logits(
            trainer, spec, before, deltas, fedavg_weights, trainer.test_loader
        )
        scalar_test, scalar_labels = exact_scalar_logits(
            trainer, spec, before, deltas, artifact["scalar_weights"], trainer.test_loader
        )
        if not torch.equal(test_labels, scalar_labels):
            raise RuntimeError("D2b test loader order changed for exact scalar state")
        class_test = exact_class_conditional_logits(
            trainer,
            spec,
            before,
            deltas,
            artifact["class_weights"],
            tail,
            trainer.test_loader,
            fedavg_test,
            test_labels,
            split_name="test",
        )
        priors = artifact["priors"]
        taus_by_source = artifact["selected_taus"]
        predictions = {
            "fedavg": fedavg_test,
            "fedavg_la": apply_logit_adjustment(fedavg_test, priors, taus_by_source["fedavg"]),
            "scalar_oracle": scalar_test,
            "scalar_oracle_la": apply_logit_adjustment(
                scalar_test, priors, taus_by_source["scalar_oracle"]
            ),
            "class_conditional_oracle": class_test,
            "class_conditional_oracle_la": apply_logit_adjustment(
                class_test, priors, taus_by_source["class_conditional_oracle"]
            ),
        }
        round_metrics = {
            method: _method_metrics(
                round_id, method, logits, test_labels, head, tail, covered, uncovered
            )
            for method, logits in predictions.items()
        }
        method_rows.extend(round_metrics.values())
        scalar = round_metrics["scalar_oracle"]
        class_only = round_metrics["class_conditional_oracle"]
        scalar_la = round_metrics["scalar_oracle_la"]
        class_la = round_metrics["class_conditional_oracle_la"]
        fedavg_la = round_metrics["fedavg_la"]
        contrast_rows.append({
            "communication_round": round_id,
            "class_minus_scalar_tail_gain": class_only["tail_accuracy"] - scalar["tail_accuracy"],
            "class_minus_scalar_balanced_gain": class_only["balanced_accuracy"] - scalar["balanced_accuracy"],
            "class_minus_scalar_harmonic_gain": class_only["head_tail_harmonic"] - scalar["head_tail_harmonic"],
            "class_minus_scalar_uncovered_tail_gain": class_only["uncovered_tail_accuracy"] - scalar["uncovered_tail_accuracy"],
            "class_la_minus_scalar_la_tail_gain": class_la["tail_accuracy"] - scalar_la["tail_accuracy"],
            "class_la_minus_scalar_la_balanced_gain": class_la["balanced_accuracy"] - scalar_la["balanced_accuracy"],
            "class_la_minus_scalar_la_harmonic_gain": class_la["head_tail_harmonic"] - scalar_la["head_tail_harmonic"],
            "class_la_minus_fedavg_la_tail_gain": class_la["tail_accuracy"] - fedavg_la["tail_accuracy"],
            "class_la_minus_fedavg_la_balanced_gain": class_la["balanced_accuracy"] - fedavg_la["balanced_accuracy"],
            "class_la_minus_fedavg_la_harmonic_gain": class_la["head_tail_harmonic"] - fedavg_la["head_tail_harmonic"],
        })

    write_csv(args.output_dir / "d2b_method_metrics.csv", method_rows)
    write_csv(args.output_dir / "d2b_round_contrasts.csv", contrast_rows)
    interaction_pass = sum(
        float(row["interaction_energy_ratio"]) >= 0.5
        and float(row["clients_with_both_beneficial_and_harmful_classes_rate"]) >= 0.5
        for row in interaction_rows
    ) >= 2
    class_ceiling_rounds = sum(
        float(row["class_minus_scalar_tail_gain"]) >= 2.0
        and float(row["class_minus_scalar_harmonic_gain"]) >= 1.0
        for row in contrast_rows
    )
    post_la_rounds = sum(
        float(row["class_la_minus_scalar_la_balanced_gain"]) >= 1.0
        and float(row["class_la_minus_scalar_la_harmonic_gain"]) >= 1.0
        for row in contrast_rows
    )
    class_ceiling_pass = class_ceiling_rounds >= 2
    post_la_pass = post_la_rounds >= 2
    necessity_pass = interaction_pass and class_ceiling_pass and post_la_pass
    if necessity_pass:
        verdict_name = "D2B_CCAR_NECESSITY_SUPPORTED"
    elif interaction_pass and class_ceiling_pass:
        verdict_name = "D2B_CLASS_GAP_REMOVED_BY_LOGIT_CALIBRATION"
    elif interaction_pass:
        verdict_name = "D2B_INTERACTION_EXISTS_BUT_SCALAR_CEILING_NOT_BROKEN"
    else:
        verdict_name = "D2B_SCALAR_AGGREGATION_NOT_REJECTED"
    verdict = {
        **frozen_manifest,
        "new_test_inference_accessed": True,
        "interaction_pass": interaction_pass,
        "class_ceiling_pass": class_ceiling_pass,
        "post_logit_adjustment_increment_pass": post_la_pass,
        "class_ceiling_positive_rounds": class_ceiling_rounds,
        "post_logit_adjustment_positive_rounds": post_la_rounds,
        "necessity_pass": necessity_pass,
        "verdict": verdict_name,
        "method_ready": False,
        "decision_rule": (
            "CCAR necessity requires non-additive client-class utility, class-conditional "
            "tail/H gains over the best train-selected scalar aggregation in >=2 rounds, "
            "and >=1pp balanced/H gains after independently selected logit adjustment."
        ),
    }
    write_json(args.output_dir / "d2b_verdict.json", verdict)
    print(json.dumps(verdict, indent=2), flush=True)


if __name__ == "__main__":
    main()
