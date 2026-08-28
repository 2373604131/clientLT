#!/usr/bin/env python
"""Run the V0 label-oracle aggregation headroom experiment.

The runner consumes a compact final-round dump, uses a validation (or explicit
optimistic train) split to construct candidates, freezes every candidate and
its hash, and only then iterates the official test loader.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.cusp_minimal import FlatSpec, flatten_state, unflatten_state, write_csv, write_json
from utils.functional_cusp import candidate_hash_from_delta, client_disagreement_subspace, fedavg_delta_from_payload
from utils.v0_oracle import (
    V0_SCHEMA_VERSION,
    class_groups_from_counts,
    gap_closure,
    maximum_trust_angle,
    metrics_from_logits,
    oracle_objective,
    optimize_span_oracle,
    random_span_candidates,
    sphere_candidate_from_coordinates,
    support_normalized_deltas,
    weighted_disagreement_scale,
)


def now_stamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_dump(dump_dir: Path) -> tuple[dict, dict]:
    payload = torch.load(dump_dir / "round_state.pt", map_location="cpu", weights_only=False)
    metadata = json.loads((dump_dir / "metadata.json").read_text(encoding="utf-8"))
    if bool(metadata.get("test_used_before_dump", False)):
        raise RuntimeError("V0 dump is invalid: official test was used before the dump")
    return payload, metadata


def build_trainer(metadata: dict, output_dir: Path, eval_batch_size: int | None = None):
    from Dassl.dassl.engine import build_trainer
    from federated_main import setup_cfg

    args = SimpleNamespace(**metadata["resolved_args"])
    args.output_dir = str(output_dir)
    if eval_batch_size is not None:
        if int(eval_batch_size) < 1:
            raise ValueError("eval_batch_size must be positive")
        args.test_batch_size = int(eval_batch_size)
    cfg = setup_cfg(args)
    trainer = build_trainer(cfg)
    trainer.fed_before_train(is_global=True)
    return cfg, trainer


def _item_label(item) -> int:
    if hasattr(item, "label"):
        return int(item.label)
    if isinstance(item, dict) and "label" in item:
        return int(item["label"])
    if isinstance(item, (list, tuple)) and len(item) >= 2:
        return int(item[1])
    raise TypeError(f"Cannot recover a label from selection item type {type(item)!r}")


def stratified_half_split(data_source, seed: int) -> tuple[list, list]:
    buckets: dict[int, list] = {}
    for item in data_source:
        buckets.setdefault(_item_label(item), []).append(item)
    rng = np.random.default_rng(int(seed))
    opt, safe = [], []
    for class_id in sorted(buckets):
        items = list(buckets[class_id])
        order = rng.permutation(len(items)).tolist()
        items = [items[index] for index in order]
        if len(items) < 2:
            raise RuntimeError(f"Class {class_id} has fewer than two validation examples")
        split = max(1, min(len(items) - 1, len(items) // 2))
        opt.extend(items[:split])
        safe.extend(items[split:])
    return opt, safe


def stratified_cap(data_source, per_class: int | None, seed: int) -> list:
    """Take a deterministic class-balanced search subset without replacement."""
    if per_class is None or int(per_class) <= 0:
        return list(data_source)
    buckets: dict[int, list] = {}
    for item in data_source:
        buckets.setdefault(_item_label(item), []).append(item)
    rng = np.random.default_rng(int(seed))
    selected = []
    for class_id in sorted(buckets):
        items = list(buckets[class_id])
        order = rng.permutation(len(items)).tolist()
        selected.extend(items[index] for index in order[: int(per_class)])
    return selected


def build_selection_loaders(
    cfg,
    trainer,
    source_name: str,
    seed: int,
    allow_optimistic: bool,
    opt_per_class: int | None = None,
):
    from Dassl.dassl.data.data_manager import build_data_loader
    from Dassl.dassl.data.transforms import build_transform

    dataset = trainer.dm.dataset
    if source_name == "val":
        source = list(dataset.val or [])
        if not source:
            raise RuntimeError(
                "The dataset exposes no validation split. Add a real validation split, or pass "
                "--selection-source train --allow-optimistic-selection for a non-formal pilot."
            )
        leakage = False
    else:
        if not allow_optimistic:
            raise RuntimeError("Train-based oracle selection requires --allow-optimistic-selection")
        source = list(dataset.train_x or [])
        leakage = True
    opt_source, safe_source = stratified_half_split(source, seed)
    opt_full_count = len(opt_source)
    opt_source = stratified_cap(opt_source, opt_per_class, seed + 17)
    transform = build_transform(cfg, is_train=False)

    def loader(items):
        return build_data_loader(
            cfg,
            sampler_type="SequentialSampler",
            data_source=items,
            batch_size=cfg.DATALOADER.TEST.BATCH_SIZE,
            tfm=transform,
            is_train=False,
            dataset_wrapper=None,
            class_names=dataset.classnames,
            drop_last=False,
        )

    return loader(opt_source), loader(safe_source), {
        "selection_source": source_name,
        "optimistic_train_leakage": leakage,
        "opt_count": len(opt_source),
        "opt_full_count": opt_full_count,
        "opt_per_class_cap": int(opt_per_class or 0),
        "safe_count": len(safe_source),
        "split_seed": int(seed),
    }


class StateEvaluator:
    def __init__(self, trainer, spec: FlatSpec, theta_t: torch.Tensor, groups: dict[str, list[int]]):
        self.trainer = trainer
        self.model = trainer.model
        self.spec = spec
        self.theta_t = theta_t
        self.groups = groups

    def state_from_delta(self, delta: torch.Tensor) -> dict[str, torch.Tensor]:
        return unflatten_state(self.theta_t + torch.as_tensor(delta, dtype=torch.float64), self.spec)

    def evaluate_state(self, state: dict[str, torch.Tensor], loader) -> tuple[dict, list[dict]]:
        self.model.load_state_dict(state, strict=False)
        was_training = self.model.training
        logits, labels = [], []
        try:
            self.model.eval()
            with torch.no_grad():
                for batch in loader:
                    images, batch_labels = self.trainer.parse_batch_test(batch)
                    output = self.model(images)
                    if isinstance(output, (tuple, list)):
                        output = output[0]
                    logits.append(output.detach().float().cpu())
                    labels.append(batch_labels.detach().long().cpu())
        finally:
            self.model.train(was_training)
        return metrics_from_logits(torch.cat(logits), torch.cat(labels), self.groups)

    def evaluate_delta(self, delta: torch.Tensor, loader) -> tuple[dict, list[dict]]:
        return self.evaluate_state(self.state_from_delta(delta), loader)


def _safe(metrics: dict, baseline: dict, args: argparse.Namespace) -> bool:
    return (
        float(metrics["head_acc"]) >= float(baseline["head_acc"]) - float(args.max_head_drop)
        and float(metrics["mid_acc"]) >= float(baseline["mid_acc"]) - float(args.max_mid_drop)
        and float(metrics["overall_acc"]) >= float(baseline["overall_acc"]) - float(args.max_overall_drop)
    )


def _gamma_id(gamma: float) -> str:
    return f"{float(gamma):.4f}".replace(".", "p")


def freeze_candidates(output_dir: Path, states: dict, rows: list[dict], support_states: dict, manifest: dict) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(states, output_dir / "candidate_states.pt")
    torch.save(support_states, output_dir / "support_only_states.pt")
    write_csv(output_dir / "candidate_manifest.csv", rows)
    frozen = {
        **manifest,
        "candidate_count": len(rows),
        "candidate_names": [row["candidate_id"] for row in rows],
        "candidate_hashes": {row["candidate_id"]: row["candidate_hash"] for row in rows},
        "support_only_class_ids": sorted(int(value) for value in support_states),
        "candidates_frozen": True,
        "candidate_frozen_at": now_stamp(),
        "test_accessed": False,
    }
    write_json(output_dir / "v0_manifest.json", frozen)
    return frozen


def build_candidates(payload: dict, metadata: dict, evaluator: StateEvaluator, opt_loader, safe_loader, args):
    spec = evaluator.spec
    theta_t, _, client_deltas, delta_avg = fedavg_delta_from_payload(payload, spec)
    weights = torch.as_tensor(payload["fedavg_weights"], dtype=torch.float64)
    disagreement = weighted_disagreement_scale(client_deltas, delta_avg, weights)
    basis, basis_report = client_disagreement_subspace(
        client_deltas, delta_avg, weights, rank_max=args.rank_max
    )
    if basis is None:
        raise RuntimeError(f"V0 client disagreement subspace is unavailable: {basis_report}")

    evaluation_cache: dict[tuple[str, str], dict] = {}

    def cached_metrics(delta: torch.Tensor, loader, split: str) -> dict:
        key = (str(split), candidate_hash_from_delta(delta))
        if key not in evaluation_cache:
            evaluation_cache[key] = evaluator.evaluate_delta(delta, loader)[0]
        return dict(evaluation_cache[key])

    fedavg_opt = cached_metrics(delta_avg, opt_loader, "opt")
    fedavg_safe = cached_metrics(delta_avg, safe_loader, "safe")
    states, rows = {}, []

    def add(candidate_id: str, method: str, delta: torch.Tensor, gamma: float, report: dict, opt_metrics: dict, safe_metrics: dict):
        states[candidate_id] = evaluator.state_from_delta(delta)
        rows.append({
            "candidate_id": candidate_id,
            "method": method,
            "gamma": float(gamma),
            "candidate_hash": candidate_hash_from_delta(delta),
            "update_norm": float(delta.norm().item()),
            "trust_distance": float((delta - delta_avg).norm().item()),
            "val_opt_tail_acc": float(opt_metrics["tail_acc"]),
            "val_safe_overall_acc": float(safe_metrics["overall_acc"]),
            "val_safe_head_acc": float(safe_metrics["head_acc"]),
            "val_safe_mid_acc": float(safe_metrics["mid_acc"]),
            "val_safe_tail_acc": float(safe_metrics["tail_acc"]),
            "val_safe": _safe(safe_metrics, fedavg_safe, args),
            **report,
        })

    add("fedavg", "fedavg", delta_avg, 0.0, {"fallback": False}, fedavg_opt, fedavg_safe)
    rng = np.random.default_rng(int(args.random_seed))
    convex_alpha_rows = [weights.numpy(), np.ones(client_deltas.shape[0]) / client_deltas.shape[0]]
    convex_alpha_rows.extend(
        rng.dirichlet(np.ones(client_deltas.shape[0]), size=args.convex_random_count)
    )

    counts = torch.as_tensor(payload["client_class_counts"], dtype=torch.float64)
    support_coefficients = torch.zeros(client_deltas.shape[0], dtype=torch.float64)
    covered = 0
    for class_id in evaluator.groups["tail"]:
        support = counts[:, int(class_id)] > 0
        mass = float(weights[support].sum().item())
        if mass <= 1e-12:
            continue
        support_coefficients[support] += weights[support] / mass
        covered += 1
    support_raw = None
    if covered:
        support_coefficients /= float(covered)
        support_raw = (client_deltas * support_coefficients[:, None]).sum(dim=0)
    equal_raw = client_deltas.mean(dim=0)

    for gamma_index, gamma in enumerate(args.gammas):
        gamma = float(gamma)
        if gamma == 0.0:
            continue

        angle_cap = maximum_trust_angle(float(delta_avg.norm().item()), gamma * disagreement)

        def project_raw_candidate(raw_delta: torch.Tensor):
            coordinates = basis.T @ (raw_delta - delta_avg)
            coordinate_norm = float(coordinates.norm().item())
            if coordinate_norm > 1e-12:
                coordinates = coordinates / coordinate_norm * angle_cap
            delta, report = sphere_candidate_from_coordinates(
                delta_avg, basis, coordinates, trust_radius=gamma * disagreement
            )
            return delta, report, coordinates

        start_candidates = [{
            "name": "fedavg",
            "coordinates": torch.zeros(basis.shape[1], dtype=torch.float64),
            "delta": delta_avg,
            "opt": fedavg_opt,
            "safe": fedavg_safe,
        }]

        random_entries = []
        for random_delta, random_report in random_span_candidates(
            delta_avg,
            basis,
            gamma=gamma,
            disagreement_scale=disagreement,
            count=args.random_count,
            seed=int(args.random_seed) + 1009 * gamma_index,
        ):
            opt_metrics = cached_metrics(random_delta, opt_loader, "opt")
            safe_metrics = cached_metrics(random_delta, safe_loader, "safe")
            random_id = f"random_span_g{_gamma_id(gamma)}_{int(random_report['random_index']):03d}"
            add(random_id, "random_span", random_delta, gamma, random_report, opt_metrics, safe_metrics)
            random_entries.append({
                "name": f"best_random_{int(random_report['random_index']):03d}",
                "coordinates": basis.T @ (random_delta - delta_avg),
                "delta": random_delta,
                "opt": opt_metrics,
                "safe": safe_metrics,
            })

        equal_delta, equal_report, equal_coordinates = project_raw_candidate(equal_raw)
        equal_opt = cached_metrics(equal_delta, opt_loader, "opt")
        equal_safe = cached_metrics(equal_delta, safe_loader, "safe")
        add(
            f"equal_client_g{_gamma_id(gamma)}", "equal_client", equal_delta, gamma,
            equal_report, equal_opt, equal_safe,
        )
        if "equal" in args.oracle_starts:
            start_candidates.append({
                "name": "equal",
                "coordinates": equal_coordinates,
                "delta": equal_delta,
                "opt": equal_opt,
                "safe": equal_safe,
            })

        if support_raw is not None:
            support_delta, support_report, support_coordinates = project_raw_candidate(support_raw)
            support_opt = cached_metrics(support_delta, opt_loader, "opt")
            support_safe = cached_metrics(support_delta, safe_loader, "safe")
            add(
                f"support_weighting_g{_gamma_id(gamma)}", "support_weighting", support_delta, gamma,
                {**support_report, "covered_tail_classes": covered}, support_opt, support_safe,
            )
            if "support" in args.oracle_starts:
                start_candidates.append({
                    "name": "support",
                    "coordinates": support_coordinates,
                    "delta": support_delta,
                    "opt": support_opt,
                    "safe": support_safe,
                })

        if "best_random" in args.oracle_starts:
            start_candidates.extend(random_entries)

        enabled_names = set(args.oracle_starts)
        if "fedavg" not in enabled_names:
            start_candidates = [item for item in start_candidates if item["name"] != "fedavg"]
        if not start_candidates:
            raise RuntimeError("V0b has no usable oracle initialization candidates")

        # Keep every feasible initialization in the final pool as well as
        # refining from the objective-best one. This makes the safe-selection
        # oracle auditable: it can never discard a stronger support/equal/
        # random candidate merely because local coordinate refinement moved
        # away from it.
        span_pool = [
            (
                SpanFallback(
                    item["delta"],
                    item["opt"],
                    report={
                        "fallback": False,
                        "initialization": item["name"],
                        "initialization_only": True,
                        "multi_start_count": len(start_candidates),
                        "safe_start_count": sum(
                            _safe(candidate["safe"], fedavg_safe, args)
                            for candidate in start_candidates
                        ),
                        "evaluation_count": 0,
                        "objective_improvement_from_start": 0.0,
                    },
                ),
                item["safe"],
            )
            for item in start_candidates
        ]
        for lambda_head in args.lambda_head:
            for lambda_mid in args.lambda_mid:
                safe_starts = [
                    item for item in start_candidates if _safe(item["safe"], fedavg_safe, args)
                ]
                eligible_starts = safe_starts or start_candidates
                initial = min(
                    eligible_starts,
                    key=lambda item: oracle_objective(item["opt"], lambda_head, lambda_mid),
                )
                initial_objective = oracle_objective(initial["opt"], lambda_head, lambda_mid)
                result = optimize_span_oracle(
                    lambda delta: cached_metrics(delta, opt_loader, "opt"),
                    delta_avg,
                    basis,
                    gamma=gamma,
                    disagreement_scale=disagreement,
                    lambda_head=lambda_head,
                    lambda_mid=lambda_mid,
                    iterations=args.solver_iterations,
                    probe_angle=args.probe_angle,
                    initial_coordinates=initial["coordinates"],
                    initialization=initial["name"],
                )
                if float(result.report["objective"]) > float(initial_objective) + 1e-9:
                    raise RuntimeError(
                        "V0b solver violated initialization dominance: "
                        f"gamma={gamma} lambda_head={lambda_head} lambda_mid={lambda_mid} "
                        f"start={initial['name']} initial={initial_objective} "
                        f"result={result.report['objective']}"
                    )
                safe_metrics = cached_metrics(result.delta, safe_loader, "safe")
                report = {
                    **result.report,
                    "multi_start_count": len(start_candidates),
                    "safe_start_count": len(safe_starts),
                    "initial_objective": float(initial_objective),
                    "objective_improvement_from_start": float(initial_objective - result.report["objective"]),
                    "evaluation_cache_entries": len(evaluation_cache),
                }
                wrapped = SpanFallback(result.delta, result.metrics, report=report)
                span_pool.append((wrapped, safe_metrics))
        safe_span = [item for item in span_pool if _safe(item[1], fedavg_safe, args)]
        chosen, chosen_safe = max(
            safe_span or [(SpanFallback(delta_avg, fedavg_opt), fedavg_safe)],
            key=lambda item: (float(item[1]["tail_acc"]), float(item[1]["h3"])),
        )
        chosen_report = dict(chosen.report)
        if support_raw is not None:
            safe_tail_regret = float(support_safe["tail_acc"]) - float(chosen_safe["tail_acc"])
            chosen_report.update({
                "support_val_safe_tail_regret": safe_tail_regret,
                "support_val_opt_tail_regret": (
                    float(support_opt["tail_acc"]) - float(chosen.metrics["tail_acc"])
                ),
                "support_initialization_safe": _safe(support_safe, fedavg_safe, args),
            })
            if _safe(support_safe, fedavg_safe, args) and safe_tail_regret > 1e-9:
                raise RuntimeError(
                    "V0b safe-selection oracle failed to dominate its support initialization: "
                    f"gamma={gamma} regret={safe_tail_regret}"
                )
        span_id = f"oracle_span_g{_gamma_id(gamma)}"
        add(span_id, "oracle_span", chosen.delta, gamma, chosen_report, dict(chosen.metrics), chosen_safe)

        convex_pool = []
        for alpha in convex_alpha_rows:
            alpha_tensor = torch.as_tensor(alpha, dtype=torch.float64)
            raw = (client_deltas * alpha_tensor[:, None]).sum(dim=0)
            convex_delta, convex_report, _ = project_raw_candidate(raw)
            opt_metrics = cached_metrics(convex_delta, opt_loader, "opt")
            safe_metrics = cached_metrics(convex_delta, safe_loader, "safe")
            convex_pool.append((convex_delta, convex_report, opt_metrics, safe_metrics))
        safe_convex = [item for item in convex_pool if _safe(item[3], fedavg_safe, args)]
        convex_delta, convex_report, convex_opt, convex_safe = max(
            safe_convex or [(delta_avg, {"fallback": True, "fallback_reason": "no_safe_convex"}, fedavg_opt, fedavg_safe)],
            key=lambda item: (float(item[3]["tail_acc"]), float(item[3]["h3"])),
        )
        add(
            f"oracle_convex_g{_gamma_id(gamma)}",
            "oracle_convex_search",
            convex_delta,
            gamma,
            convex_report,
            convex_opt,
            convex_safe,
        )

    theta0 = flatten_state(payload["global_before_trainable"], spec)
    support_states = {
        int(class_id): unflatten_state(theta0 + delta, spec)
        for class_id, delta in support_normalized_deltas(payload).items()
        if int(class_id) in set(int(value) for value in evaluator.groups["tail"])
    }
    context = {
        "theta_t": theta_t,
        "delta_avg": delta_avg,
        "basis_report": basis_report,
        "disagreement_scale": disagreement,
        "fedavg_opt": fedavg_opt,
        "fedavg_safe": fedavg_safe,
        "evaluation_cache_entries": len(evaluation_cache),
    }
    return states, rows, support_states, context


def build_candidates_pooled(payload: dict, metadata: dict, evaluator: StateEvaluator, opt_loader, safe_loader, args):
    """Fast two-stage oracle search with one candidate bank shared by all lambdas.

    Stage one evaluates a deterministic candidate bank on the small balanced
    optimization split. Stage two evaluates only a shortlist on the full safe
    split. The official test remains inaccessible until candidates are frozen.
    """
    del metadata  # Kept in the signature to match the refined builder.
    spec = evaluator.spec
    theta_t, _, client_deltas, delta_avg = fedavg_delta_from_payload(payload, spec)
    weights = torch.as_tensor(payload["fedavg_weights"], dtype=torch.float64)
    disagreement = weighted_disagreement_scale(client_deltas, delta_avg, weights)
    basis, basis_report = client_disagreement_subspace(
        client_deltas, delta_avg, weights, rank_max=args.rank_max
    )
    if basis is None:
        raise RuntimeError(f"V0 pooled client disagreement subspace is unavailable: {basis_report}")

    evaluation_cache: dict[tuple[str, str], dict] = {}
    evaluation_counts = {"opt": 0, "safe": 0}

    def cached_metrics(delta: torch.Tensor, loader, split: str) -> dict:
        key = (str(split), candidate_hash_from_delta(delta))
        if key not in evaluation_cache:
            evaluation_cache[key] = evaluator.evaluate_delta(delta, loader)[0]
            evaluation_counts[str(split)] += 1
        return dict(evaluation_cache[key])

    def metric_or_nan(metrics: dict | None, key: str) -> float:
        return float(metrics[key]) if metrics is not None else math.nan

    fedavg_opt = cached_metrics(delta_avg, opt_loader, "opt")
    fedavg_safe = cached_metrics(delta_avg, safe_loader, "safe")
    states: dict[str, dict[str, torch.Tensor]] = {}
    rows: list[dict] = []

    def add_frozen(
        candidate_id: str,
        method: str,
        delta: torch.Tensor,
        gamma: float,
        report: dict,
        opt_metrics: dict,
        safe_metrics: dict | None,
    ) -> None:
        states[candidate_id] = evaluator.state_from_delta(delta)
        rows.append({
            "candidate_id": candidate_id,
            "method": method,
            "gamma": float(gamma),
            "candidate_hash": candidate_hash_from_delta(delta),
            "update_norm": float(delta.norm().item()),
            "trust_distance": float((delta - delta_avg).norm().item()),
            "val_opt_tail_acc": float(opt_metrics["tail_acc"]),
            "val_safe_overall_acc": metric_or_nan(safe_metrics, "overall_acc"),
            "val_safe_head_acc": metric_or_nan(safe_metrics, "head_acc"),
            "val_safe_mid_acc": metric_or_nan(safe_metrics, "mid_acc"),
            "val_safe_tail_acc": metric_or_nan(safe_metrics, "tail_acc"),
            "val_safe_evaluated": safe_metrics is not None,
            "val_safe": _safe(safe_metrics, fedavg_safe, args) if safe_metrics is not None else False,
            **report,
        })

    add_frozen("fedavg", "fedavg", delta_avg, 0.0, {"fallback": False}, fedavg_opt, fedavg_safe)

    counts = torch.as_tensor(payload["client_class_counts"], dtype=torch.float64)
    support_coefficients = torch.zeros(client_deltas.shape[0], dtype=torch.float64)
    covered = 0
    for class_id in evaluator.groups["tail"]:
        support = counts[:, int(class_id)] > 0
        mass = float(weights[support].sum().item())
        if mass <= 1e-12:
            continue
        support_coefficients[support] += weights[support] / mass
        covered += 1
    support_raw = None
    if covered:
        support_coefficients /= float(covered)
        support_raw = (client_deltas * support_coefficients[:, None]).sum(dim=0)
    equal_raw = client_deltas.mean(dim=0)

    rng = np.random.default_rng(int(args.random_seed))
    convex_alpha_rows = rng.dirichlet(
        np.ones(client_deltas.shape[0]), size=max(0, int(args.convex_random_count))
    )
    total_pool_count = 0
    total_shortlist_count = 0

    for gamma_index, gamma_value in enumerate(args.gammas):
        gamma = float(gamma_value)
        if gamma == 0.0:
            continue
        print(f"[V0 pooled] gamma={gamma:g}: constructing shared candidate bank", flush=True)
        angle_cap = maximum_trust_angle(float(delta_avg.norm().item()), gamma * disagreement)
        entries: list[dict] = []
        entries_by_hash: dict[str, dict] = {}

        def add_entry(name: str, family: str, delta: torch.Tensor, report: dict) -> dict:
            candidate_hash = candidate_hash_from_delta(delta)
            if candidate_hash in entries_by_hash:
                return entries_by_hash[candidate_hash]
            opt_metrics = cached_metrics(delta, opt_loader, "opt")
            entry = {
                "name": name,
                "family": family,
                "delta": delta,
                "report": report,
                "opt": opt_metrics,
                "safe": None,
                "candidate_hash": candidate_hash,
            }
            entries.append(entry)
            entries_by_hash[candidate_hash] = entry
            if len(entries) % max(1, int(args.progress_every)) == 0:
                print(
                    f"[V0 pooled] gamma={gamma:g}: opt-evaluated {len(entries)} candidates",
                    flush=True,
                )
            return entry

        def project_raw_candidate(raw_delta: torch.Tensor):
            coordinates = basis.T @ (raw_delta - delta_avg)
            coordinate_norm = float(coordinates.norm().item())
            if coordinate_norm > 1e-12:
                coordinates = coordinates / coordinate_norm * angle_cap
            delta, report = sphere_candidate_from_coordinates(
                delta_avg, basis, coordinates, trust_radius=gamma * disagreement
            )
            return delta, report

        random_entries = []
        for random_delta, random_report in random_span_candidates(
            delta_avg,
            basis,
            gamma=gamma,
            disagreement_scale=disagreement,
            count=args.random_count,
            seed=int(args.random_seed) + 1009 * gamma_index,
        ):
            random_index = int(random_report["random_index"])
            random_entries.append(add_entry(
                f"random_span_g{_gamma_id(gamma)}_{random_index:03d}",
                "random_span",
                random_delta,
                random_report,
            ))

        equal_delta, equal_report = project_raw_candidate(equal_raw)
        equal_entry = add_entry("equal", "equal_client", equal_delta, equal_report)
        support_entry = None
        if support_raw is not None:
            support_delta, support_report = project_raw_candidate(support_raw)
            support_entry = add_entry(
                "support",
                "support_weighting",
                support_delta,
                {**support_report, "covered_tail_classes": covered},
            )

        for scale in args.axis_scales:
            scale = float(scale)
            if scale <= 0.0:
                continue
            for direction_id in range(int(basis.shape[1])):
                for sign in (-1.0, 1.0):
                    coordinates = torch.zeros(basis.shape[1], dtype=torch.float64)
                    coordinates[direction_id] = sign * scale * angle_cap
                    axis_delta, axis_report = sphere_candidate_from_coordinates(
                        delta_avg,
                        basis,
                        coordinates,
                        trust_radius=gamma * disagreement,
                    )
                    sign_name = "p" if sign > 0 else "m"
                    add_entry(
                        f"axis_{direction_id:02d}_{sign_name}_s{scale:g}",
                        "axis_probe",
                        axis_delta,
                        {**axis_report, "axis": direction_id, "axis_sign": sign, "axis_scale": scale},
                    )

        convex_entries = []
        for convex_index, alpha in enumerate(convex_alpha_rows):
            alpha_tensor = torch.as_tensor(alpha, dtype=torch.float64)
            raw_delta = (client_deltas * alpha_tensor[:, None]).sum(dim=0)
            convex_delta, convex_report = project_raw_candidate(raw_delta)
            convex_entries.append(add_entry(
                f"convex_{convex_index:03d}",
                "convex",
                convex_delta,
                {**convex_report, "convex_index": convex_index},
            ))

        shortlist_hashes: set[str] = set()

        def shortlist(entry: dict | None) -> None:
            if entry is not None:
                shortlist_hashes.add(str(entry["candidate_hash"]))

        shortlist(equal_entry)
        shortlist(support_entry)
        ranked_tail = sorted(
            entries,
            key=lambda item: (float(item["opt"]["tail_acc"]), float(item["opt"]["h3"])),
            reverse=True,
        )
        for entry in ranked_tail[: max(1, int(args.safe_top_k))]:
            shortlist(entry)
        for lambda_head in args.lambda_head:
            for lambda_mid in args.lambda_mid:
                shortlist(min(
                    entries,
                    key=lambda item: oracle_objective(item["opt"], lambda_head, lambda_mid),
                ))

        ranked_convex = sorted(
            convex_entries,
            key=lambda item: (float(item["opt"]["tail_acc"]), float(item["opt"]["h3"])),
            reverse=True,
        )
        for entry in ranked_convex[: max(2, int(args.safe_top_k) // 2)]:
            shortlist(entry)
        for lambda_head in args.lambda_head:
            for lambda_mid in args.lambda_mid:
                shortlist(min(
                    convex_entries,
                    key=lambda item: oracle_objective(item["opt"], lambda_head, lambda_mid),
                ))

        shortlisted = [entry for entry in entries if entry["candidate_hash"] in shortlist_hashes]
        print(
            f"[V0 pooled] gamma={gamma:g}: pool={len(entries)}, "
            f"full-safe shortlist={len(shortlisted)}",
            flush=True,
        )
        for safe_index, entry in enumerate(shortlisted, start=1):
            entry["safe"] = cached_metrics(entry["delta"], safe_loader, "safe")
            if safe_index % max(1, int(args.progress_every)) == 0 or safe_index == len(shortlisted):
                print(
                    f"[V0 pooled] gamma={gamma:g}: safe-evaluated "
                    f"{safe_index}/{len(shortlisted)}",
                    flush=True,
                )

        safe_entries = [
            entry for entry in shortlisted if _safe(entry["safe"], fedavg_safe, args)
        ]
        chosen = max(
            safe_entries or [{
                "name": "fedavg_fallback",
                "family": "fedavg",
                "delta": delta_avg,
                "report": {"fallback": True, "fallback_reason": "no_safe_pooled_candidate"},
                "opt": fedavg_opt,
                "safe": fedavg_safe,
            }],
            key=lambda item: (float(item["safe"]["tail_acc"]), float(item["safe"]["h3"])),
        )

        for entry in random_entries:
            add_frozen(
                entry["name"], "random_span", entry["delta"], gamma,
                entry["report"], entry["opt"], entry["safe"],
            )
        add_frozen(
            f"equal_client_g{_gamma_id(gamma)}", "equal_client", equal_entry["delta"], gamma,
            equal_entry["report"], equal_entry["opt"], equal_entry["safe"],
        )
        if support_entry is not None:
            add_frozen(
                f"support_weighting_g{_gamma_id(gamma)}", "support_weighting",
                support_entry["delta"], gamma, support_entry["report"],
                support_entry["opt"], support_entry["safe"],
            )

        chosen_report = {
            **chosen["report"],
            "search_mode": "pooled",
            "pooled_source": chosen["name"],
            "pooled_family": chosen["family"],
            "pool_count": len(entries),
            "safe_shortlist_count": len(shortlisted),
            "shared_lambda_count": len(args.lambda_head) * len(args.lambda_mid),
        }
        if support_entry is not None and support_entry["safe"] is not None:
            chosen_report.update({
                "support_val_safe_tail_regret": (
                    float(support_entry["safe"]["tail_acc"]) - float(chosen["safe"]["tail_acc"])
                ),
                "support_val_opt_tail_regret": (
                    float(support_entry["opt"]["tail_acc"]) - float(chosen["opt"]["tail_acc"])
                ),
                "support_initialization_safe": _safe(support_entry["safe"], fedavg_safe, args),
            })
        add_frozen(
            f"oracle_span_g{_gamma_id(gamma)}", "oracle_span", chosen["delta"], gamma,
            chosen_report, chosen["opt"], chosen["safe"],
        )

        safe_convex = [
            entry for entry in convex_entries
            if entry["safe"] is not None and _safe(entry["safe"], fedavg_safe, args)
        ]
        chosen_convex = max(
            safe_convex or [{
                "name": "fedavg_fallback",
                "family": "fedavg",
                "delta": delta_avg,
                "report": {"fallback": True, "fallback_reason": "no_safe_pooled_convex"},
                "opt": fedavg_opt,
                "safe": fedavg_safe,
            }],
            key=lambda item: (float(item["safe"]["tail_acc"]), float(item["safe"]["h3"])),
        )
        add_frozen(
            f"oracle_convex_g{_gamma_id(gamma)}", "oracle_convex_search",
            chosen_convex["delta"], gamma,
            {
                **chosen_convex["report"],
                "search_mode": "pooled",
                "pooled_source": chosen_convex["name"],
                "pool_count": len(convex_entries),
                "safe_shortlist_count": sum(entry["safe"] is not None for entry in convex_entries),
            },
            chosen_convex["opt"], chosen_convex["safe"],
        )
        total_pool_count += len(entries)
        total_shortlist_count += len(shortlisted)

    theta0 = flatten_state(payload["global_before_trainable"], spec)
    support_states = {
        int(class_id): unflatten_state(theta0 + delta, spec)
        for class_id, delta in support_normalized_deltas(payload).items()
        if int(class_id) in set(int(value) for value in evaluator.groups["tail"])
    }
    context = {
        "theta_t": theta_t,
        "delta_avg": delta_avg,
        "basis_report": basis_report,
        "disagreement_scale": disagreement,
        "fedavg_opt": fedavg_opt,
        "fedavg_safe": fedavg_safe,
        "evaluation_cache_entries": len(evaluation_cache),
        "evaluation_counts": evaluation_counts,
        "pooled_candidate_count": total_pool_count,
        "safe_shortlist_count": total_shortlist_count,
    }
    print(
        f"[V0 pooled] candidate construction complete: opt forwards={evaluation_counts['opt']}, "
        f"safe forwards={evaluation_counts['safe']}",
        flush=True,
    )
    return states, rows, support_states, context


class SpanFallback:
    def __init__(self, delta: torch.Tensor, metrics: dict, report: dict | None = None):
        self.delta = delta
        self.metrics = metrics
        self.report = report or {"fallback": True, "fallback_reason": "no_safe_span_candidate"}


def evaluate_test(
    output_dir: Path,
    evaluator: StateEvaluator,
    test_loader,
    groups: dict[str, list[int]],
    progress_every: int = 10,
):
    manifest_path = output_dir / "v0_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not manifest.get("candidates_frozen", False):
        raise RuntimeError("V0 candidates must be frozen before official-test access")
    first_access = now_stamp()
    states = torch.load(output_dir / "candidate_states.pt", map_location="cpu", weights_only=False)
    candidate_rows = list(csv.DictReader((output_dir / "candidate_manifest.csv").open(encoding="utf-8")))
    metric_rows, per_class_rows = [], []
    print(f"[V0 test] evaluating {len(candidate_rows)} frozen candidates", flush=True)
    for candidate_index, row in enumerate(candidate_rows, start=1):
        metrics, per_class = evaluator.evaluate_state(states[row["candidate_id"]], test_loader)
        metric_rows.append({
            "candidate_id": row["candidate_id"],
            "method": row["method"],
            "gamma": float(row["gamma"]),
            **metrics,
        })
        per_class_rows.extend({"candidate_id": row["candidate_id"], "method": row["method"], **item} for item in per_class)
        if candidate_index % max(1, int(progress_every)) == 0 or candidate_index == len(candidate_rows):
            print(f"[V0 test] candidates {candidate_index}/{len(candidate_rows)}", flush=True)

    support_states = torch.load(output_dir / "support_only_states.pt", map_location="cpu", weights_only=False)
    support_rows = []
    print(f"[V0 test] evaluating {len(support_states)} support-only ceilings", flush=True)
    for support_index, (class_id, state) in enumerate(support_states.items(), start=1):
        _, per_class = evaluator.evaluate_state(state, test_loader)
        row = next(item for item in per_class if int(item["class_id"]) == int(class_id))
        support_rows.append({"class_id": int(class_id), "support_only_class_acc": float(row["class_acc"])})
        if support_index % max(1, int(progress_every)) == 0 or support_index == len(support_states):
            print(f"[V0 test] support ceilings {support_index}/{len(support_states)}", flush=True)
    support_ceiling = (
        float(np.mean([row["support_only_class_acc"] for row in support_rows])) if support_rows else math.nan
    )
    fedavg = next(row for row in metric_rows if row["method"] == "fedavg")
    for row in metric_rows:
        row["tail_gain"] = float(row["tail_acc"]) - float(fedavg["tail_acc"])
        row["head_damage"] = float(fedavg["head_acc"]) - float(row["head_acc"])
        row["mid_damage"] = float(fedavg["mid_acc"]) - float(row["mid_acc"])
        row["gap_closure"] = gap_closure(row["tail_acc"], fedavg["tail_acc"], support_ceiling)
        row["support_only_tail_ceiling"] = support_ceiling
    support_by_gamma = {
        float(row["gamma"]): row for row in metric_rows if row["method"] == "support_weighting"
    }
    random_by_gamma = {}
    for row in metric_rows:
        if row["method"] == "random_span":
            random_by_gamma.setdefault(float(row["gamma"]), []).append(float(row["tail_gain"]))
    for row in metric_rows:
        support = support_by_gamma.get(float(row["gamma"]))
        row["support_regret"] = (
            float(support["tail_gain"]) - float(row["tail_gain"])
            if support is not None else math.nan
        )
        random_values = random_by_gamma.get(float(row["gamma"]), [])
        row["random_tail_gain_p95"] = (
            float(np.percentile(random_values, 95)) if random_values else math.nan
        )
        row["beats_random_p95"] = (
            bool(float(row["tail_gain"]) > row["random_tail_gain_p95"])
            if math.isfinite(float(row["random_tail_gain_p95"])) else False
        )
    write_csv(output_dir / "test_metrics.csv", metric_rows)
    write_csv(output_dir / "per_class_metrics.csv", per_class_rows)
    write_csv(output_dir / "support_only_ceiling.csv", support_rows)
    write_csv(output_dir / "pareto_points.csv", metric_rows)
    manifest.update({
        "test_accessed": True,
        "test_first_accessed_at": first_access,
        "candidate_frozen_before_test": True,
    })
    write_json(manifest_path, manifest)
    return metric_rows, support_ceiling


def unit_verdict(metric_rows: list[dict]) -> dict:
    span = [row for row in metric_rows if row["method"] == "oracle_span"]
    comparisons = []
    for row in span:
        random_p95 = float(row.get("random_tail_gain_p95", math.nan))
        passed = (
            float(row["tail_gain"]) > 0.0
            and float(row["head_damage"]) <= 0.5
            and (math.isnan(random_p95) or float(row["tail_gain"]) > random_p95)
            and (math.isnan(float(row["gap_closure"])) or float(row["gap_closure"]) >= 0.1)
        )
        comparisons.append({**row, "random_tail_gain_p95": random_p95, "single_unit_pass": passed})
    support_comparisons = []
    for row in [item for item in metric_rows if item["method"] == "support_weighting"]:
        support_comparisons.append({
            **row,
            "single_unit_signal": (
                float(row["tail_gain"]) > 0.0
                and float(row["head_damage"]) <= 0.5
                and bool(row.get("beats_random_p95", False))
            ),
        })
    return {
        "verdict": "PASS_SINGLE_UNIT" if any(row["single_unit_pass"] for row in comparisons) else "FAIL_SINGLE_UNIT",
        "warning": "A formal V0 verdict requires aggregation across three seeds and three dump rounds.",
        "span_comparisons": comparisons,
        "support_comparisons": support_comparisons,
    }


def run(args: argparse.Namespace) -> None:
    payload, metadata = load_dump(args.dump_dir)
    cfg, trainer = build_trainer(
        metadata,
        args.output_dir / "model_build",
        eval_batch_size=args.eval_batch_size,
    )
    groups = class_groups_from_counts(payload["global_class_counts"])
    opt_loader, safe_loader, selection_report = build_selection_loaders(
        cfg,
        trainer,
        args.selection_source,
        args.selection_seed,
        args.allow_optimistic_selection,
        opt_per_class=args.opt_per_class,
    )
    spec = FlatSpec.from_dict(payload["flatten_spec"])
    theta_t = flatten_state(payload["global_before_trainable"], spec)
    evaluator = StateEvaluator(trainer, spec, theta_t, groups)
    candidate_builder = build_candidates_pooled if args.search_mode == "pooled" else build_candidates
    states, rows, support_states, context = candidate_builder(
        payload, metadata, evaluator, opt_loader, safe_loader, args
    )
    manifest = freeze_candidates(args.output_dir, states, rows, support_states, {
        "schema_version": V0_SCHEMA_VERSION,
        "dump_hash": sha256_file(args.dump_dir / "round_state.pt"),
        "partition": metadata.get("resolved_args", {}).get("partition", ""),
        "seed": metadata.get("resolved_args", {}).get("seed", ""),
        "round": metadata.get("communication_round", ""),
        "groups": groups,
        "selection": selection_report,
        "gammas": [float(value) for value in args.gammas],
        "lambda_head": [float(value) for value in args.lambda_head],
        "lambda_mid": [float(value) for value in args.lambda_mid],
        "rank_max": int(args.rank_max),
        "basis_report": context["basis_report"],
        "disagreement_scale": context["disagreement_scale"],
        "oracle_starts": list(args.oracle_starts),
        "search_mode": str(args.search_mode),
        "opt_per_class": int(args.opt_per_class),
        "axis_scales": [float(value) for value in args.axis_scales],
        "safe_top_k": int(args.safe_top_k),
        "eval_batch_size_override": args.eval_batch_size,
        "selection_evaluation_cache_entries": int(context["evaluation_cache_entries"]),
        "selection_evaluation_counts": dict(context.get("evaluation_counts", {})),
        "pooled_candidate_count": int(context.get("pooled_candidate_count", 0)),
        "safe_shortlist_count": int(context.get("safe_shortlist_count", 0)),
        "selection_only_before_freeze": True,
    })
    validation_rows = [{key: value for key, value in row.items() if key != "candidate_hash"} for row in rows]
    write_csv(args.output_dir / "validation_metrics.csv", validation_rows)
    metric_rows, support_ceiling = evaluate_test(
        args.output_dir,
        evaluator,
        trainer.test_loader,
        groups,
        progress_every=args.progress_every,
    )
    verdict = unit_verdict(metric_rows)
    write_json(args.output_dir / "v0_verdict.json", {
        **verdict,
        "support_only_tail_ceiling": support_ceiling,
        "manifest": manifest,
    })
    print(f"V0 oracle finished: {args.output_dir}")


def synthetic_smoke(output_dir: Path) -> None:
    fedavg = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64)
    basis = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=torch.float64)

    def evaluate(delta):
        tail_loss = float(torch.square(delta[1] - 0.5).item())
        return {
            "tail_loss": tail_loss,
            "head_loss": float(torch.square(delta[0] - 1.0).item()),
            "mid_loss": float(torch.square(delta[2]).item()),
            "tail_acc": 100.0 - 10.0 * tail_loss,
            "head_acc": 100.0,
            "mid_acc": 100.0,
            "overall_acc": 100.0,
            "h3": 100.0,
        }

    result = optimize_span_oracle(
        evaluate, fedavg, basis, gamma=0.5, disagreement_scale=1.0,
        lambda_head=1.0, lambda_mid=1.0, iterations=6, probe_angle=0.02,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "synthetic_smoke.json", {
        "status": "PASS" if result.metrics["tail_loss"] < evaluate(fedavg)["tail_loss"] else "FAIL",
        "metrics": result.metrics,
        "report": result.report,
    })
    if result.metrics["tail_loss"] >= evaluate(fedavg)["tail_loss"]:
        raise RuntimeError("V0 synthetic oracle failed to improve its tail objective")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["run", "synthetic"], default="run")
    parser.add_argument("--dump-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--selection-source", choices=["val", "train"], default="val")
    parser.add_argument("--allow-optimistic-selection", action="store_true")
    parser.add_argument("--selection-seed", type=int, default=2026)
    parser.add_argument(
        "--search-mode",
        choices=["refine", "pooled"],
        default="refine",
        help="refine is the exhaustive V0b optimizer; pooled is the fast shared-bank V0c audit.",
    )
    parser.add_argument(
        "--opt-per-class",
        type=int,
        default=0,
        help="Class-balanced cap for the optimization split; 0 keeps the full split.",
    )
    parser.add_argument("--gammas", type=float, nargs="+", default=[0.0, 0.05, 0.1, 0.2, 0.4, 0.8, 1.0])
    parser.add_argument("--lambda-head", type=float, nargs="+", default=[0.0, 0.25, 1.0, 4.0, 16.0])
    parser.add_argument("--lambda-mid", type=float, nargs="+", default=[0.0, 0.25, 1.0, 4.0, 16.0])
    parser.add_argument("--rank-max", type=int, default=8)
    parser.add_argument("--solver-iterations", type=int, default=4)
    parser.add_argument("--probe-angle", type=float, default=0.02)
    parser.add_argument(
        "--oracle-starts",
        nargs="+",
        choices=["fedavg", "support", "equal", "best_random"],
        default=["fedavg", "support", "equal", "best_random"],
        help=(
            "Feasible initializations audited before local span refinement. best_random chooses "
            "the objective-best member of the frozen random pool for each lambda pair."
        ),
    )
    parser.add_argument("--random-count", type=int, default=20)
    parser.add_argument("--convex-random-count", type=int, default=32)
    parser.add_argument("--random-seed", type=int, default=2026)
    parser.add_argument("--axis-scales", type=float, nargs="+", default=[0.5, 1.0])
    parser.add_argument(
        "--safe-top-k",
        type=int,
        default=8,
        help="Number of top pooled candidates promoted to full-safe evaluation per gamma.",
    )
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument("--max-head-drop", type=float, default=0.5)
    parser.add_argument("--max-mid-drop", type=float, default=0.5)
    parser.add_argument("--max-overall-drop", type=float, default=0.25)
    parser.add_argument(
        "--eval-batch-size",
        type=int,
        default=None,
        help="Optional offline-only validation/test batch-size override; it does not change training.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.stage == "synthetic":
        synthetic_smoke(args.output_dir)
        return
    if args.dump_dir is None:
        raise SystemExit("--dump-dir is required for --stage run")
    if args.opt_per_class < 0:
        raise SystemExit("--opt-per-class must be non-negative")
    if args.safe_top_k < 1:
        raise SystemExit("--safe-top-k must be positive")
    if args.progress_every < 1:
        raise SystemExit("--progress-every must be positive")
    run(args)


if __name__ == "__main__":
    main()
