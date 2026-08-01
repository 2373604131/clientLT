#!/usr/bin/env python
"""Unified launcher/evaluator for the simplified CUSP minimal experiment.

Typical Linux run:
  CUDA_VISIBLE_DEVICES=0 OUTPUT_ROOT=/data/yzh/cusp_minimal_$(date +%Y%m%d_%H%M%S) \
  DRY_RUN=0 STAGE=all bash scripts/run_cusp_minimal.sh

The experiment has two real stages:
  train: run PromptFL/FedAvg for 10 rounds and save a compact trainable dump
  eval : freeze 13 equal-norm candidates, then read official test once
"""

from __future__ import annotations

import argparse
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.cusp_minimal import (
    METHODS,
    SCHEMA_VERSION,
    build_cusp_candidates,
    freeze_cusp_candidates,
    load_cusp_minimal_dump,
    summarize_values,
    write_csv,
    write_json,
)


def bool_from_env(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "y"}


def build_paths(output_root: Path) -> dict[str, Path]:
    train_dir = output_root / "client-longtail_seed42_round10"
    return {
        "output_root": output_root,
        "train_dir": train_dir,
        "eval_dir": output_root / "cusp_eval_client-longtail_seed42_round10",
        "schedule_file": output_root / "shared_client_schedule_seed42_round10.json",
        "dump_dir": train_dir / "cusp_minimal" / "round_010",
    }


def build_train_command(python_bin: str, data_root: str, paths: dict[str, Path]) -> list[str]:
    return [
        python_bin, "federated_main.py",
        "--model", "fedavg",
        "--trainer", "PromptFL",
        "--dataset", "cifar100_LT",
        "--partition", "client-longtail",
        "--config-file", "configs/trainers/PromptFL/vit_b16.yaml",
        "--dataset-config-file", "configs/datasets/cifar100_LT.yaml",
        "--root", data_root,
        "--output-dir", str(paths["train_dir"]),
        "--num_users", "30",
        "--frac", "1.0",
        "--round", "10",
        "--local_epochs", "3",
        "--seed", "42",
        "--split_seed", "42",
        "--client_schedule_seed", "42",
        "--client_schedule_file", str(paths["schedule_file"]),
        "--global_eval_interval", "999999",
        "--train_batch_size", "32",
        "--test_batch_size", "64",
        "--lr", "0.001",
        "--imb_factor", "0.01",
        "--head_client_ratio", "0.9",
        "--tail_client_ratio", "0.1",
        "--head_class_ratio", "0.8",
        "--tail_class_ratio", "0.2",
        "--specialization_lambda", "0.75",
        "--intra_group_alpha", "0.5",
        "--head_leakage_scale", "3.0",
        "--n_ctx", "4",
        "--num_prompt", "1",
        "--avg_prompt", "1",
        "--ctx_init", "False",
        "--csc", "True",
        "--experimentD_enable", "False",
        "--log_update_retention", "False",
        "--isolate_local_optimizer_state", "True",
        "--federated_single_scheduler_step", "True",
        "--cusp_minimal_enable", "True",
        "--cusp_minimal_round", "10",
    ]


def run_command(command: list[str], env: dict[str, str]) -> None:
    print("Running:")
    print(" ".join(str(part) for part in command))
    subprocess.run(command, cwd=REPO_ROOT, env=env, check=True)


def print_command(command: list[str]) -> None:
    print(" ".join(str(part) for part in command))


def candidate_group_ids(metadata: dict, num_classes: int) -> tuple[set[int], set[int]]:
    head = set(int(x) for x in metadata["head_class_ids"])
    tail = set(int(x) for x in metadata["tail_class_ids"])
    missing = set(range(num_classes)) - head - tail
    if missing:
        head |= missing
    return head, tail


def build_test_cache(trainer, output_dir: Path) -> dict:
    model = trainer.model
    was_training = model.training
    model.eval()
    features, labels = [], []
    try:
        with torch.no_grad():
            for batch in trainer.test_loader:
                images, batch_labels = trainer.parse_batch_train(batch)
                image_features = model.image_encoder(images.type(model.dtype))
                image_features = image_features / image_features.norm(dim=-1, keepdim=True).clamp_min(1e-12)
                features.append(image_features.detach().float().cpu())
                labels.append(batch_labels.detach().long().cpu())
    finally:
        model.train(was_training)
    cache = {
        "schema_version": SCHEMA_VERSION,
        "source": "official_test",
        "features": torch.cat(features, dim=0),
        "labels": torch.cat(labels, dim=0),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(cache, output_dir / "official_test_cache.pt")
    return cache


def compute_accuracy(logits: torch.Tensor, labels: torch.Tensor, head_ids: set[int], tail_ids: set[int]) -> tuple[dict, list[dict]]:
    logits = logits.detach().cpu()
    labels = labels.detach().cpu().long()
    preds = logits.argmax(dim=1)
    num_classes = logits.shape[1]
    per_class = []
    for class_id in range(num_classes):
        mask = labels == class_id
        total = int(mask.sum().item())
        correct = int((preds[mask] == labels[mask]).sum().item()) if total else 0
        acc = 100.0 * correct / total if total else math.nan
        per_class.append({
            "class_id": class_id,
            "group": "tail" if class_id in tail_ids else "head",
            "test_count": total,
            "correct_count": correct,
            "class_acc": acc,
        })

    def mean_for(ids: set[int]) -> float:
        values = [row["class_acc"] for row in per_class if row["class_id"] in ids and math.isfinite(row["class_acc"])]
        return float(sum(values) / len(values)) if values else math.nan

    finite = [row["class_acc"] for row in per_class if math.isfinite(row["class_acc"])]
    metrics = {
        "overall_acc": 100.0 * float((preds == labels).double().mean().item()),
        "macro_acc": float(sum(finite) / len(finite)) if finite else math.nan,
        "head_acc": mean_for(head_ids),
        "tail_acc": mean_for(tail_ids),
    }
    return metrics, per_class


def evaluate_dump(paths: dict[str, Path]) -> None:
    payload, metadata = load_cusp_minimal_dump(paths["dump_dir"])
    states, rows, context = build_cusp_candidates(payload, metadata)
    manifest = freeze_cusp_candidates(paths["eval_dir"], states, rows, context)

    from Dassl.dassl.engine import build_trainer
    from federated_main import setup_cfg

    train_args = SimpleNamespace(**metadata["resolved_args"])
    train_args.output_dir = str(paths["eval_dir"] / "model_eval")
    cfg = setup_cfg(train_args)
    trainer = build_trainer(cfg)
    trainer.fed_before_train(is_global=True)

    test_first_accessed_at = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    test_cache = build_test_cache(trainer, paths["eval_dir"])
    labels = test_cache["labels"]
    head_ids, tail_ids = candidate_group_ids(metadata, int(payload["num_classes"]))

    result_rows = []
    per_class_rows = []
    for row in rows:
        candidate_id = row["candidate_id"]
        with torch.no_grad():
            logits = trainer.model.logits_from_cached_features(test_cache["features"], states[candidate_id])
        metrics, per_class = compute_accuracy(logits, labels, head_ids, tail_ids)
        result_rows.append({"candidate_id": candidate_id, "method": row["method"], **metrics, "final_norm": row["final_norm"]})
        for item in per_class:
            per_class_rows.append({"candidate_id": candidate_id, "method": row["method"], **item})

    random_tail = [row["tail_acc"] for row in result_rows if row["method"] == "random_reweight"]
    summary_rows = []
    for method in METHODS:
        method_rows = [row for row in result_rows if row["method"] == method]
        if method == "random_reweight":
            stats = {f"tail_acc_{key}": value for key, value in summarize_values(random_tail).items()}
            summary_rows.append({"method": method, "num_candidates": len(method_rows), **stats})
        elif method_rows:
            summary_rows.append({"method": method, "num_candidates": 1, **method_rows[0]})

    fedavg = next(row for row in result_rows if row["method"] == "fedavg")
    classwise = next(row for row in result_rows if row["method"] == "classwise_weighting")
    cusp = next(row for row in result_rows if row["method"] == "oracle_cusp")
    random_median = summarize_values(random_tail)["median"]
    verdict = (
        "PASS"
        if cusp["tail_acc"] > fedavg["tail_acc"]
        and cusp["tail_acc"] > classwise["tail_acc"]
        and cusp["tail_acc"] > random_median
        and cusp["head_acc"] >= fedavg["head_acc"] - 0.5
        and cusp["overall_acc"] >= fedavg["overall_acc"] - 0.5
        else "FAIL"
    )

    write_csv(paths["eval_dir"] / "oracle_results.csv", summary_rows)
    write_csv(paths["eval_dir"] / "oracle_all_candidates.csv", result_rows)
    write_csv(paths["eval_dir"] / "oracle_per_class.csv", per_class_rows)
    write_csv(paths["eval_dir"] / "random_reweight_distribution.csv", [row for row in result_rows if row["method"] == "random_reweight"])
    write_json(paths["eval_dir"] / "oracle_metadata.json", {
        "schema_version": SCHEMA_VERSION,
        "minimal_pilot_status": verdict,
        "candidate_frozen_at": manifest["candidate_frozen_at"],
        "test_first_accessed_at": test_first_accessed_at,
        "candidate_frozen_before_test": manifest["candidate_frozen_at"] <= test_first_accessed_at,
        "candidate_methods": list(METHODS),
        "num_concrete_candidates": len(result_rows),
        "accuracy_scale": "percent",
        "norm_budget": context["norm_budget"],
        "test_access_policy": "official test is encoded only after candidate_states.pt and candidate_manifest are written",
    })
    (paths["eval_dir"] / "oracle_summary.md").write_text(
        "\n".join([
            "# CUSP Minimal Pilot",
            "",
            f"Status: {verdict}",
            f"FedAvg tail acc: {fedavg['tail_acc']:.4f}",
            f"Classwise tail acc: {classwise['tail_acc']:.4f}",
            f"Random median tail acc: {random_median:.4f}",
            f"CUSP tail acc: {cusp['tail_acc']:.4f}",
            "",
        ]),
        encoding="utf-8",
    )


def synthetic_smoke(output_dir: Path) -> None:
    before = {
        "prompt_learner.class_aware_ctx": torch.zeros(3, 2),
        "prompt_learner.general_ctx": torch.zeros(1, 2),
    }
    locals_ = []
    for value in [0.2, -0.1, 0.3, -0.2]:
        locals_.append({
            "prompt_learner.class_aware_ctx": torch.randn(3, 2, generator=torch.Generator().manual_seed(int((value + 1) * 100))) * 0.1 + value,
            "prompt_learner.general_ctx": torch.ones(1, 2) * value,
        })
    payload = {
        "schema_version": SCHEMA_VERSION,
        "flatten_spec": {
            "keys": ["prompt_learner.class_aware_ctx", "prompt_learner.general_ctx"],
            "shapes": [[3, 2], [1, 2]],
            "dtypes": ["torch.float32", "torch.float32"],
            "offsets": [[0, 6], [6, 8]],
            "numel": 8,
        },
        "global_before_trainable": before,
        "global_after_fedavg_trainable": before,
        "local_trainable_states": locals_,
        "selected_client_ids": [0, 1, 2, 3],
        "fedavg_weights": torch.tensor([0.4, 0.3, 0.2, 0.1], dtype=torch.float64),
        "client_sample_counts": [4, 3, 2, 1],
        "client_class_counts": torch.tensor([[3, 1, 0], [1, 2, 0], [0, 1, 2], [0, 0, 1]]),
        "global_class_counts": torch.tensor([4, 4, 3]),
        "num_classes": 3,
    }
    metadata = {"head_class_ids": [0, 1], "tail_class_ids": [2]}
    states, rows, context = build_cusp_candidates(payload, metadata)
    freeze_cusp_candidates(output_dir, states, rows, context)
    write_json(output_dir / "oracle_metadata.json", {
        "schema_version": SCHEMA_VERSION,
        "synthetic": True,
        "candidate_count": len(rows),
        "methods": list(METHODS),
        "status": "PASS",
    })


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["all", "train", "eval", "synthetic"], default=os.environ.get("STAGE", "all"))
    parser.add_argument("--output-root", type=Path, default=Path(os.environ.get("OUTPUT_ROOT", "output/cusp_minimal_seed42")))
    parser.add_argument("--data", default=os.environ.get("DATA", "DATA"))
    parser.add_argument("--python-bin", default=os.environ.get("PYTHON_BIN", sys.executable))
    parser.add_argument("--cuda-visible-devices", default=os.environ.get("CUDA_VISIBLE_DEVICES"))
    parser.add_argument("--run", action="store_true", help="actually run; otherwise obey DRY_RUN env")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dry_run = args.dry_run or (not args.run and bool_from_env("DRY_RUN", True))
    paths = build_paths(args.output_root)
    train_command = build_train_command(args.python_bin, args.data, paths)
    env = os.environ.copy()
    if args.cuda_visible_devices is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(args.cuda_visible_devices)

    print("CUSP minimal refactor")
    print(f"Stage: {args.stage}")
    print(f"Dry run: {dry_run}")
    print(f"Output root: {args.output_root}")

    if dry_run:
        if args.stage in {"all", "train"}:
            print("Training command:")
            print_command(train_command)
        if args.stage in {"all", "eval"}:
            print("Eval will read dump:")
            print(paths["dump_dir"])
        if args.stage == "synthetic":
            print("Synthetic output:")
            print(args.output_root / "synthetic_smoke")
        return

    os.chdir(REPO_ROOT)
    if args.stage == "synthetic":
        synthetic_smoke(args.output_root / "synthetic_smoke")
        print(f"Synthetic smoke passed: {args.output_root / 'synthetic_smoke'}")
        return

    paths["output_root"].mkdir(parents=True, exist_ok=True)
    if args.stage in {"all", "train"}:
        run_command(train_command, env)
    if args.stage in {"all", "eval"}:
        evaluate_dump(paths)
        print(f"CUSP eval finished: {paths['eval_dir']}")


if __name__ == "__main__":
    main()
