#!/usr/bin/env python
"""D3: separate representation quality from a head-biased classifier boundary.

All classifiers and logit-adjustment hyperparameters are fitted or selected on
a deterministic train-only fit/calibration split. The official test set is
iterated only after those choices have been frozen to disk.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Sequence

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.cusp_minimal import FlatSpec, flatten_state
from utils.d23_common import (
    D23_SCHEMA_VERSION,
    aggregate_metrics,
    build_global_train_eval_loader,
    build_trainer,
    class_split,
    collect_features_and_logits,
    compact_state_from_vector,
    load_dump,
    sha256_file,
    stratified_fit_calibration_split,
    validate_dump,
    write_csv,
    write_json,
)


def _dump_dir(root: Path, communication_round: int) -> Path:
    direct = root / f"round_{communication_round:03d}"
    nested = root / "v0_oracle" / f"round_{communication_round:03d}"
    return direct if direct.is_dir() else nested


def apply_logit_adjustment(logits: torch.Tensor, priors: torch.Tensor, tau: float) -> torch.Tensor:
    priors = torch.as_tensor(priors, dtype=logits.dtype, device=logits.device).clamp_min(1e-12)
    return logits - float(tau) * priors.log()[None, :]


def nearest_centroid_weights(features: torch.Tensor, labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    features = F.normalize(torch.as_tensor(features).float(), dim=1)
    labels = torch.as_tensor(labels).long()
    centroids = torch.zeros(num_classes, features.shape[1], dtype=features.dtype)
    centroids.index_add_(0, labels, features)
    counts = torch.bincount(labels, minlength=num_classes).float().clamp_min(1.0)
    centroids = centroids / counts[:, None]
    return F.normalize(centroids, dim=1)


def centroid_logits(features: torch.Tensor, centroids: torch.Tensor) -> torch.Tensor:
    return F.normalize(torch.as_tensor(features).float(), dim=1) @ centroids.t()


def fit_balanced_ridge(
    features: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
    ridge: float,
) -> torch.Tensor:
    """Return [feature_dim + bias, classes] class-balanced ridge weights."""
    x = torch.as_tensor(features).double().cpu()
    labels = torch.as_tensor(labels).long().cpu()
    x = torch.cat([x, torch.ones(len(x), 1, dtype=x.dtype)], dim=1)
    counts = torch.bincount(labels, minlength=num_classes).double().clamp_min(1.0)
    sample_weight = 1.0 / counts[labels]
    weighted_x = x * sample_weight.sqrt()[:, None]
    target = torch.zeros(len(x), num_classes, dtype=x.dtype)
    target[torch.arange(len(x)), labels] = sample_weight.sqrt()
    gram = weighted_x.t() @ weighted_x
    penalty = torch.eye(gram.shape[0], dtype=gram.dtype) * float(ridge)
    penalty[-1, -1] = 0.0
    rhs = weighted_x.t() @ target
    try:
        solution = torch.linalg.solve(gram + penalty, rhs)
    except torch.linalg.LinAlgError:
        solution = torch.linalg.pinv(gram + penalty) @ rhs
    return solution.float()


def ridge_logits(features: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    x = torch.as_tensor(features).float().cpu()
    x = torch.cat([x, torch.ones(len(x), 1, dtype=x.dtype)], dim=1)
    return x @ weights


def select_tau(
    calibration_logits: torch.Tensor,
    calibration_labels: torch.Tensor,
    priors: torch.Tensor,
    taus: Sequence[float],
    head: Sequence[int],
    tail: Sequence[int],
) -> tuple[float, list[dict]]:
    rows = []
    for tau in taus:
        metrics = aggregate_metrics(
            apply_logit_adjustment(calibration_logits, priors, tau),
            calibration_labels,
            head,
            tail,
        )
        rows.append({"tau": float(tau), **metrics})
    # Harmonic mean is primary; balanced accuracy and then smaller tau break ties.
    best = max(
        rows,
        key=lambda row: (
            float(row["head_tail_harmonic"]),
            float(row["balanced_accuracy"]),
            -float(row["tau"]),
        ),
    )
    return float(best["tau"]), rows


def _metric_row(round_id: int, method: str, metrics: dict) -> dict:
    return {"communication_round": round_id, "method": method, **metrics}


def _mean(values: Sequence[float]) -> float:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    return float(sum(finite) / len(finite)) if finite else math.nan


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dump-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--rounds", default="20,50,80")
    parser.add_argument("--eval-batch-size", type=int, default=256)
    parser.add_argument("--calibration-fraction", type=float, default=0.2)
    parser.add_argument("--taus", default="0,0.25,0.5,0.75,1.0,1.5,2.0")
    parser.add_argument("--ridge", type=float, default=1e-2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rounds = [int(value) for value in args.rounds.split(",") if value.strip()]
    taus = [float(value) for value in args.taus.split(",") if value.strip()]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    loaded = []
    for round_id in rounds:
        directory = _dump_dir(args.dump_root, round_id)
        payload, metadata = load_dump(directory)
        validate_dump(payload, metadata)
        if int(metadata["communication_round"]) != round_id:
            raise RuntimeError(f"Round mismatch in {directory}")
        loaded.append((directory, payload, metadata))

    cfg, trainer = build_trainer(loaded[0][2], args.output_dir / "eval_runtime", args.eval_batch_size)
    train_loader = build_global_train_eval_loader(cfg, trainer)
    frozen_rows, calibration_rows = [], []
    cached_train = []
    for directory, payload, metadata in loaded:
        round_id = int(metadata["communication_round"])
        spec = FlatSpec.from_dict(payload["flatten_spec"])
        state = compact_state_from_vector(
            flatten_state(payload["global_after_fedavg_trainable"], spec), spec
        )
        print(f"D3 round {round_id}: extract train-only features", flush=True)
        train_features, train_logits, train_labels = collect_features_and_logits(
            trainer, state, train_loader
        )
        fit_index, calibration_index = stratified_fit_calibration_split(
            train_labels,
            seed=42,
            calibration_fraction=args.calibration_fraction,
        )
        num_classes = int(train_logits.shape[1])
        head, tail = class_split(payload["global_class_counts"])
        fit_labels = train_labels[fit_index]
        priors = torch.bincount(fit_labels, minlength=num_classes).float()
        priors = priors / priors.sum()
        selected_tau, tau_rows = select_tau(
            train_logits[calibration_index],
            train_labels[calibration_index],
            priors,
            taus,
            head,
            tail,
        )
        for row in tau_rows:
            calibration_rows.append({
                "communication_round": round_id,
                "selected": float(row["tau"]) == selected_tau,
                **row,
            })
        centroids = nearest_centroid_weights(
            train_features[fit_index], fit_labels, num_classes
        )
        ridge = fit_balanced_ridge(
            train_features[fit_index], fit_labels, num_classes, args.ridge
        )
        artifact_path = args.output_dir / f"d3_frozen_round_{round_id:03d}.pt"
        torch.save({
            "communication_round": round_id,
            "selected_tau": selected_tau,
            "priors": priors,
            "centroids": centroids,
            "ridge_weights": ridge,
            "fit_indices": fit_index,
            "calibration_indices": calibration_index,
        }, artifact_path)
        frozen_rows.append({
            "communication_round": round_id,
            "selected_tau": selected_tau,
            "fit_count": len(fit_index),
            "calibration_count": len(calibration_index),
            "ridge": float(args.ridge),
            "artifact": str(artifact_path),
            "artifact_sha256": sha256_file(artifact_path),
            "dump_sha256": sha256_file(directory / "round_state.pt"),
        })
        cached_train.append((payload, metadata, state, selected_tau, priors, centroids, ridge))

    write_csv(args.output_dir / "d3_calibration_grid.csv", calibration_rows)
    write_csv(args.output_dir / "d3_frozen_choices.csv", frozen_rows)
    frozen_manifest = {
        "schema_version": D23_SCHEMA_VERSION,
        "diagnostic": "D3_representation_vs_classifier_boundary",
        "seed": 42,
        "rounds": rounds,
        "calibration_source": "deterministic class-stratified global train split",
        "calibration_fraction": float(args.calibration_fraction),
        "choices_frozen_before_test": True,
        "test_accessed": False,
        "frozen_choices_sha256": sha256_file(args.output_dir / "d3_frozen_choices.csv"),
    }
    write_json(args.output_dir / "d3_manifest.json", frozen_manifest)
    print("D3 calibration/probe choices frozen before test access", flush=True)

    method_rows, contrast_rows = [], []
    for payload, metadata, state, selected_tau, priors, centroids, ridge in cached_train:
        round_id = int(metadata["communication_round"])
        head, tail = class_split(payload["global_class_counts"])
        print(f"D3 round {round_id}: evaluate frozen choices on official test", flush=True)
        test_features, native_logits, test_labels = collect_features_and_logits(
            trainer, state, trainer.test_loader
        )
        predictions = {
            "native_clip_logits": native_logits,
            "train_selected_logit_adjustment": apply_logit_adjustment(
                native_logits, priors, selected_tau
            ),
            "nearest_class_centroid": centroid_logits(test_features, centroids),
            "class_balanced_ridge_probe": ridge_logits(test_features, ridge),
        }
        metrics = {
            method: aggregate_metrics(logits, test_labels, head, tail)
            for method, logits in predictions.items()
        }
        for method, values in metrics.items():
            method_rows.append(_metric_row(round_id, method, values))
        native = metrics["native_clip_logits"]
        contrast = {"communication_round": round_id, "selected_tau": selected_tau}
        for method in (
            "train_selected_logit_adjustment",
            "nearest_class_centroid",
            "class_balanced_ridge_probe",
        ):
            contrast[f"{method}_tail_gain"] = metrics[method]["tail_accuracy"] - native["tail_accuracy"]
            contrast[f"{method}_balanced_gain"] = metrics[method]["balanced_accuracy"] - native["balanced_accuracy"]
            contrast[f"{method}_harmonic_gain"] = metrics[method]["head_tail_harmonic"] - native["head_tail_harmonic"]
            contrast[f"{method}_head_damage"] = native["head_accuracy"] - metrics[method]["head_accuracy"]
        contrast_rows.append(contrast)

    write_csv(args.output_dir / "d3_method_metrics.csv", method_rows)
    write_csv(args.output_dir / "d3_round_contrasts.csv", contrast_rows)
    representation_rounds = []
    calibration_rounds = []
    for row in contrast_rows:
        representation_rounds.append(any(
            float(row[f"{method}_tail_gain"]) >= 5.0
            and float(row[f"{method}_harmonic_gain"]) >= 0.0
            for method in ("nearest_class_centroid", "class_balanced_ridge_probe")
        ))
        calibration_rounds.append(
            float(row["train_selected_logit_adjustment_tail_gain"]) >= 2.0
            and float(row["train_selected_logit_adjustment_harmonic_gain"]) >= 0.0
        )
    representation_pass = sum(representation_rounds) >= 2
    calibration_pass = sum(calibration_rounds) >= 2
    if representation_pass and calibration_pass:
        verdict_name = "D3_REPRESENTATION_AND_CALIBRATION_SUPPORTED"
    elif representation_pass:
        verdict_name = "D3_REPRESENTATION_EXISTS_BOUNDARY_FIX_UNRESOLVED"
    elif calibration_pass:
        verdict_name = "D3_LOGIT_CALIBRATION_SUPPORTED_WITHOUT_PROBE_GAP"
    else:
        verdict_name = "D3_BOUNDARY_HYPOTHESIS_NOT_SUPPORTED"
    verdict = {
        **frozen_manifest,
        "test_accessed": True,
        "representation_pass": representation_pass,
        "calibration_pass": calibration_pass,
        "representation_positive_rounds": sum(representation_rounds),
        "calibration_positive_rounds": sum(calibration_rounds),
        "verdict": verdict_name,
        "method_ready": False,
        "note": (
            "The centroid and ridge classifiers are offline probes. They diagnose whether "
            "features contain tail information; they are not proposed federated methods."
        ),
    }
    write_json(args.output_dir / "d3_verdict.json", verdict)
    print(json.dumps(verdict, indent=2), flush=True)


if __name__ == "__main__":
    main()
