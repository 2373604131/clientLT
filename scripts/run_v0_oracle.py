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


def build_trainer(metadata: dict, output_dir: Path):
    from Dassl.dassl.engine import build_trainer
    from federated_main import setup_cfg

    args = SimpleNamespace(**metadata["resolved_args"])
    args.output_dir = str(output_dir)
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


def build_selection_loaders(cfg, trainer, source_name: str, seed: int, allow_optimistic: bool):
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

    fedavg_opt, _ = evaluator.evaluate_delta(delta_avg, opt_loader)
    fedavg_safe, _ = evaluator.evaluate_delta(delta_avg, safe_loader)
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
    for gamma_index, gamma in enumerate(args.gammas):
        gamma = float(gamma)
        if gamma == 0.0:
            continue
        span_pool = []
        for lambda_head in args.lambda_head:
            for lambda_mid in args.lambda_mid:
                result = optimize_span_oracle(
                    lambda delta: evaluator.evaluate_delta(delta, opt_loader)[0],
                    delta_avg,
                    basis,
                    gamma=gamma,
                    disagreement_scale=disagreement,
                    lambda_head=lambda_head,
                    lambda_mid=lambda_mid,
                    iterations=args.solver_iterations,
                    probe_angle=args.probe_angle,
                )
                safe_metrics, _ = evaluator.evaluate_delta(result.delta, safe_loader)
                span_pool.append((result, safe_metrics))
        safe_span = [item for item in span_pool if _safe(item[1], fedavg_safe, args)]
        chosen, chosen_safe = max(
            safe_span or [(SpanFallback(delta_avg, fedavg_opt), fedavg_safe)],
            key=lambda item: (float(item[1]["tail_acc"]), float(item[1]["h3"])),
        )
        span_id = f"oracle_span_g{_gamma_id(gamma)}"
        add(span_id, "oracle_span", chosen.delta, gamma, dict(chosen.report), dict(chosen.metrics), chosen_safe)

        for random_delta, random_report in random_span_candidates(
            delta_avg,
            basis,
            gamma=gamma,
            disagreement_scale=disagreement,
            count=args.random_count,
            seed=int(args.random_seed) + 1009 * gamma_index,
        ):
            opt_metrics, _ = evaluator.evaluate_delta(random_delta, opt_loader)
            safe_metrics, _ = evaluator.evaluate_delta(random_delta, safe_loader)
            random_id = f"random_span_g{_gamma_id(gamma)}_{int(random_report['random_index']):03d}"
            add(random_id, "random_span", random_delta, gamma, random_report, opt_metrics, safe_metrics)

        angle_cap = maximum_trust_angle(float(delta_avg.norm().item()), gamma * disagreement)

        def project_raw_candidate(raw_delta: torch.Tensor):
            coordinates = basis.T @ (raw_delta - delta_avg)
            coordinate_norm = float(coordinates.norm().item())
            if coordinate_norm > 1e-12:
                coordinates = coordinates / coordinate_norm * angle_cap
            return sphere_candidate_from_coordinates(
                delta_avg, basis, coordinates, trust_radius=gamma * disagreement
            )

        equal_delta, equal_report = project_raw_candidate(client_deltas.mean(dim=0))
        equal_opt, _ = evaluator.evaluate_delta(equal_delta, opt_loader)
        equal_safe, _ = evaluator.evaluate_delta(equal_delta, safe_loader)
        add(
            f"equal_client_g{_gamma_id(gamma)}", "equal_client", equal_delta, gamma,
            equal_report, equal_opt, equal_safe,
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
        if covered:
            support_coefficients /= float(covered)
            support_raw = (client_deltas * support_coefficients[:, None]).sum(dim=0)
            support_delta, support_report = project_raw_candidate(support_raw)
            support_opt, _ = evaluator.evaluate_delta(support_delta, opt_loader)
            support_safe, _ = evaluator.evaluate_delta(support_delta, safe_loader)
            add(
                f"support_weighting_g{_gamma_id(gamma)}", "support_weighting", support_delta, gamma,
                {**support_report, "covered_tail_classes": covered}, support_opt, support_safe,
            )

        convex_pool = []
        for alpha in convex_alpha_rows:
            alpha_tensor = torch.as_tensor(alpha, dtype=torch.float64)
            raw = (client_deltas * alpha_tensor[:, None]).sum(dim=0)
            convex_delta, convex_report = project_raw_candidate(raw)
            opt_metrics, _ = evaluator.evaluate_delta(convex_delta, opt_loader)
            safe_metrics, _ = evaluator.evaluate_delta(convex_delta, safe_loader)
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
    }
    return states, rows, support_states, context


class SpanFallback:
    def __init__(self, delta: torch.Tensor, metrics: dict):
        self.delta = delta
        self.metrics = metrics
        self.report = {"fallback": True, "fallback_reason": "no_safe_span_candidate"}


def evaluate_test(output_dir: Path, evaluator: StateEvaluator, test_loader, groups: dict[str, list[int]]):
    manifest_path = output_dir / "v0_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not manifest.get("candidates_frozen", False):
        raise RuntimeError("V0 candidates must be frozen before official-test access")
    first_access = now_stamp()
    states = torch.load(output_dir / "candidate_states.pt", map_location="cpu", weights_only=False)
    candidate_rows = list(csv.DictReader((output_dir / "candidate_manifest.csv").open(encoding="utf-8")))
    metric_rows, per_class_rows = [], []
    for row in candidate_rows:
        metrics, per_class = evaluator.evaluate_state(states[row["candidate_id"]], test_loader)
        metric_rows.append({
            "candidate_id": row["candidate_id"],
            "method": row["method"],
            "gamma": float(row["gamma"]),
            **metrics,
        })
        per_class_rows.extend({"candidate_id": row["candidate_id"], "method": row["method"], **item} for item in per_class)

    support_states = torch.load(output_dir / "support_only_states.pt", map_location="cpu", weights_only=False)
    support_rows = []
    for class_id, state in support_states.items():
        _, per_class = evaluator.evaluate_state(state, test_loader)
        row = next(item for item in per_class if int(item["class_id"]) == int(class_id))
        support_rows.append({"class_id": int(class_id), "support_only_class_acc": float(row["class_acc"])})
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
    random_rows = [row for row in metric_rows if row["method"] == "random_span"]
    comparisons = []
    for row in span:
        same_gamma = [float(item["tail_gain"]) for item in random_rows if float(item["gamma"]) == float(row["gamma"])]
        random_p95 = float(np.percentile(same_gamma, 95)) if same_gamma else math.nan
        passed = (
            float(row["tail_gain"]) > 0.0
            and float(row["head_damage"]) <= 0.5
            and (math.isnan(random_p95) or float(row["tail_gain"]) > random_p95)
            and (math.isnan(float(row["gap_closure"])) or float(row["gap_closure"]) >= 0.1)
        )
        comparisons.append({**row, "random_tail_gain_p95": random_p95, "single_unit_pass": passed})
    return {
        "verdict": "PASS_SINGLE_UNIT" if any(row["single_unit_pass"] for row in comparisons) else "FAIL_SINGLE_UNIT",
        "warning": "A formal V0 verdict requires aggregation across three seeds and three dump rounds.",
        "span_comparisons": comparisons,
    }


def run(args: argparse.Namespace) -> None:
    payload, metadata = load_dump(args.dump_dir)
    cfg, trainer = build_trainer(metadata, args.output_dir / "model_build")
    groups = class_groups_from_counts(payload["global_class_counts"])
    opt_loader, safe_loader, selection_report = build_selection_loaders(
        cfg, trainer, args.selection_source, args.selection_seed, args.allow_optimistic_selection
    )
    spec = FlatSpec.from_dict(payload["flatten_spec"])
    theta_t = flatten_state(payload["global_before_trainable"], spec)
    evaluator = StateEvaluator(trainer, spec, theta_t, groups)
    states, rows, support_states, context = build_candidates(
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
        "selection_only_before_freeze": True,
    })
    validation_rows = [{key: value for key, value in row.items() if key != "candidate_hash"} for row in rows]
    write_csv(args.output_dir / "validation_metrics.csv", validation_rows)
    metric_rows, support_ceiling = evaluate_test(args.output_dir, evaluator, trainer.test_loader, groups)
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
    parser.add_argument("--gammas", type=float, nargs="+", default=[0.0, 0.05, 0.1, 0.2, 0.4, 0.8, 1.0])
    parser.add_argument("--lambda-head", type=float, nargs="+", default=[0.0, 0.25, 1.0, 4.0, 16.0])
    parser.add_argument("--lambda-mid", type=float, nargs="+", default=[0.0, 0.25, 1.0, 4.0, 16.0])
    parser.add_argument("--rank-max", type=int, default=8)
    parser.add_argument("--solver-iterations", type=int, default=4)
    parser.add_argument("--probe-angle", type=float, default=0.02)
    parser.add_argument("--random-count", type=int, default=20)
    parser.add_argument("--convex-random-count", type=int, default=32)
    parser.add_argument("--random-seed", type=int, default=2026)
    parser.add_argument("--max-head-drop", type=float, default=0.5)
    parser.add_argument("--max-mid-drop", type=float, default=0.5)
    parser.add_argument("--max-overall-drop", type=float, default=0.25)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.stage == "synthetic":
        synthetic_smoke(args.output_dir)
        return
    if args.dump_dir is None:
        raise SystemExit("--dump-dir is required for --stage run")
    run(args)


if __name__ == "__main__":
    main()
