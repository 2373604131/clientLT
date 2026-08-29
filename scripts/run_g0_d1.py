#!/usr/bin/env python
"""Foreground launcher for the frozen G0 -> D1 ClientLT diagnosis.

G0 runs two local-only ClipLoRA configurations on the same six clients and
writes ``lora_freeze.json``.  D1 refuses to run without that file, then uses
the frozen configuration for one seed-42, 80-round FedAvg trajectory with
counterfactual audits at rounds 20, 50, and 80.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Mapping

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.g0_lora_probe import G0_LORA_CONFIGS


CONFIGS = G0_LORA_CONFIGS


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _common_command(args, output_dir: Path, *, rounds: int) -> list[str]:
    schedule = args.output_root / "schedules" / f"full_users30_rounds{rounds}_seed42.json"
    return [
        args.python_bin,
        "-u",
        "federated_main.py",
        "--root", str(args.data_root),
        "--model", "fedavg",
        "--trainer", "ClipLora",
        "--dataset", "cifar100_LT",
        "--seed", "42",
        "--split_seed", "42",
        "--num_users", "30",
        "--frac", "1.0",
        "--round", str(rounds),
        "--local_epochs", "3",
        "--client_schedule_seed", "42",
        "--client_schedule_file", str(schedule),
        "--isolate_local_optimizer_state", "True",
        "--federated_single_scheduler_step", "True",
        "--lr", str(args.lr),
        "--gamma", "1",
        "--n_ctx", "4",
        "--n_general", "1",
        "--ctx_init", "False",
        "--csc", "True",
        "--dataset-config-file", "configs/datasets/cifar100_LT.yaml",
        "--config-file", "configs/trainers/PromptFL/vit_b16.yaml",
        "--output-dir", str(output_dir),
        "--imb_factor", "0.01",
        "--imb_type", "exp",
        "--train_batch_size", "32",
        "--test_batch_size", str(args.test_batch_size),
        "--global_eval_interval", "10",
        "--num_classes", "100",
        "--tail_class_ratio", "0.2",
        "--head_client_ratio", "0.9",
        "--tail_client_ratio", "0.1",
        "--head_class_ratio", "0.8",
        "--partition", "client-longtail",
        "--beta", "0.5",
        "--specialization_lambda", "0.75",
        "--intra_group_alpha", "0.5",
        "--head_leakage_scale", "3.0",
        "--encoder", "vision",
        "--cliplora_dropout_rate", "0.0",
        "--cliplora_lr_policy", "constant",
        "--cliplora_precision", "fp32",
        "--cliplora_aggregation", "fedavg",
    ]


def _append_lora(command: list[str], config: Mapping) -> list[str]:
    return command + [
        "--cliplora_position", str(config["position"]),
        "--cliplora_rank", str(config["rank"]),
        "--cliplora_alpha", str(config["alpha"]),
        "--cliplora_params", *[str(value) for value in config["params"]],
    ]


def build_g0_command(args, config_id: str) -> tuple[Path, list[str]]:
    config = CONFIGS[config_id]
    output_dir = args.output_root / "g0" / config_id
    command = _append_lora(_common_command(args, output_dir, rounds=1), config)
    command += [
        "--g0_probe_enable", "True",
        "--g0_probe_config_id", config_id,
        "--experimentD_enable", "False",
        "DATALOADER.NUM_WORKERS", str(args.num_workers),
    ]
    return output_dir, command


def build_d1_command(args, frozen: Mapping) -> tuple[Path, list[str]]:
    selected_id = str(frozen["selected_config_id"])
    if selected_id not in CONFIGS:
        raise ValueError(f"Unknown frozen LoRA configuration: {selected_id}")
    config = frozen["selected_config"]
    if config != CONFIGS[selected_id]:
        raise ValueError(
            "Frozen LoRA payload does not match the immutable launcher configuration "
            f"for {selected_id}"
        )
    output_dir = args.output_root / "d1_seed42"
    command = _append_lora(_common_command(args, output_dir, rounds=80), config)
    command += [
        "--g0_probe_enable", "False",
        "--experimentD_enable", "True",
        "--experimentD_rounds", "20,50,80",
        "--experimentD_include_normalized", "True",
        "--experimentD_support_min_fraction", "0.1",
        "--experimentD_random_support_count", str(args.random_support_count),
        "--experimentD_log_update_norm", "True",
        "--experimentD_require_full_participation", "True",
        "--experimentD_verify_fedavg", "True",
        "--experimentD_eval_mode", "class_filtered",
        "DATALOADER.NUM_WORKERS", str(args.num_workers),
    ]
    return output_dir, command


def _command_text(command: list[str]) -> str:
    return subprocess.list2cmdline(command) if os.name == "nt" else shlex.join(command)


def _run(command: list[str], *, gpu: str, dry_run: bool) -> None:
    print("\n" + "=" * 78, flush=True)
    print(_command_text(command), flush=True)
    print("=" * 78, flush=True)
    if dry_run:
        return
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    subprocess.run(command, cwd=REPO_ROOT, env=env, check=True)


def _is_nonempty(path: Path) -> bool:
    return path.exists() and any(path.iterdir())


def _probe_summary_path(output_dir: Path) -> Path:
    return output_dir / "g0_probe" / "g0_config_summary.json"


def _summary_number(summary: Mapping, key: str, default: float) -> float:
    try:
        value = float(summary.get(key, default))
    except (TypeError, ValueError):
        return float(default)
    return value if math.isfinite(value) else float(default)


def probe_pass(summary: Mapping) -> tuple[bool, list[str]]:
    reasons = []
    if int(summary.get("client_count", 0)) != 6:
        reasons.append("client_count != 6")
    if int(summary.get("tail_client_count", 0)) != 3:
        reasons.append("tail_client_count != 3")
    if not bool(summary.get("all_finite", False)):
        reasons.append("non-finite metric detected")
    if _summary_number(summary, "mean_train_loss_relative_drop", -math.inf) <= 0.0:
        reasons.append("mean train loss did not decrease")
    if int(summary.get("positive_tail_client_count", 0)) < 2:
        reasons.append("fewer than 2/3 tail clients improved held-out tail margin")
    if _summary_number(summary, "mean_prediction_flip_rate", 0.0) <= 1e-4:
        reasons.append("prediction flip rate is effectively zero")
    if _summary_number(summary, "mean_abs_logit_change", 0.0) <= 1e-6:
        reasons.append("mean logit change is effectively zero")
    if _summary_number(summary, "mean_effective_ba_delta_norm", 0.0) <= 1e-8:
        reasons.append("effective BA update is effectively zero")
    return not reasons, reasons


def freeze_lora(output_root: Path) -> dict:
    summaries = {}
    decisions = {}
    hashes = {}
    for config_id in CONFIGS:
        path = _probe_summary_path(output_root / "g0" / config_id)
        if not path.exists():
            raise FileNotFoundError(f"Missing G0 result: {path}")
        summary = json.loads(path.read_text(encoding="utf-8"))
        passed, reasons = probe_pass(summary)
        summaries[config_id] = summary
        decisions[config_id] = {"passed": passed, "failure_reasons": reasons}
        hashes[config_id] = _sha256(path)

    passed_ids = [config_id for config_id, item in decisions.items() if item["passed"]]
    selected_id = None
    if len(passed_ids) == 1:
        selected_id = passed_ids[0]
    elif len(passed_ids) > 1:
        old_gain = _summary_number(summaries["old_r2"], "median_tail_margin_gain", -math.inf)
        candidate_gain = _summary_number(
            summaries["candidate_r4"], "median_tail_margin_gain", -math.inf
        )
        selected_id = "candidate_r4" if candidate_gain > old_gain else "old_r2"

    payload = {
        "schema_version": "g0_lora_freeze_v1",
        "seed": 42,
        "verdict": "PASS" if selected_id is not None else "FAIL",
        "selection_rule": (
            "Select the sole passing configuration; if both pass, select the one with "
            "larger median held-out tail-margin gain (ties choose the smaller old_r2)."
        ),
        "selected_config_id": selected_id,
        "selected_config": CONFIGS[selected_id] if selected_id is not None else None,
        "decisions": decisions,
        "summary_sha256": hashes,
        "g0_has_no_server_aggregation": True,
    }
    freeze_path = output_root / "lora_freeze.json"
    freeze_path.parent.mkdir(parents=True, exist_ok=True)
    freeze_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(f"\nG0 freeze written: {freeze_path}", flush=True)
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)
    if selected_id is None:
        raise RuntimeError("G0 failed: D1 is blocked; inspect lora_freeze.json")
    return payload


def _read_csv(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _mean(values) -> float:
    values = [float(value) for value in values if math.isfinite(float(value))]
    return sum(values) / len(values) if values else math.nan


def _json_safe(value):
    if isinstance(value, Mapping):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _csv_bool(value) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def summarize_d1(output_root: Path, frozen: Mapping) -> dict:
    run_dir = output_root / "d1_seed42"
    experiment_dir = run_dir / "experiment_d"
    summary_rows = _read_csv(experiment_dir / "experiment_d_round_summary.csv")
    per_class_rows = _read_csv(experiment_dir / "experiment_d_per_class.csv")
    wanted = {20, 50, 80}
    rows = [row for row in summary_rows if int(float(row["communication_round"])) in wanted]
    observed = {int(float(row["communication_round"])) for row in rows}
    if observed != wanted:
        raise RuntimeError(f"D1 is incomplete: expected rounds {sorted(wanted)}, got {sorted(observed)}")
    rows.sort(key=lambda row: int(float(row["communication_round"])))

    class_rows = [
        row for row in per_class_rows
        if int(float(row["communication_round"])) in wanted
    ]
    class_rounds = {int(float(row["communication_round"])) for row in class_rows}
    if class_rounds != wanted:
        raise RuntimeError(
            "D1 per-class output is incomplete: "
            f"expected rounds {sorted(wanted)}, got {sorted(class_rounds)}"
        )
    by_round = {
        communication_round: [
            row for row in class_rows
            if int(float(row["communication_round"])) == communication_round
        ]
        for communication_round in sorted(wanted)
    }
    if any(len(items) != 20 for items in by_round.values()):
        sizes = {key: len(value) for key, value in by_round.items()}
        raise RuntimeError(f"D1 requires exactly 20 tail-class rows per round, got {sizes}")

    conditional_rows = []
    uncovered_by_round = {}
    for communication_round, items in by_round.items():
        valid = [row for row in items if _csv_bool(row.get("support_valid", False))]
        uncovered = sorted(
            int(float(row["class_id"]))
            for row in items
            if not _csv_bool(row.get("support_valid", False))
        )
        beats_random = [
            _csv_bool(row.get("support_normalized_beats_random_p95", False))
            for row in valid
        ]
        beneficial_supporters = [
            float(row["tail_gain_support_normalized_vs_fedavg"]) > 0.0
            for row in valid
        ]
        conditional_rows.append({
            "communication_round": communication_round,
            "tail_class_count": len(items),
            "valid_support_class_count": len(valid),
            "valid_support_class_rate": len(valid) / len(items),
            "conditional_beats_random_p95_rate": (
                sum(beats_random) / len(beats_random) if beats_random else math.nan
            ),
            "conditional_beats_fedavg_rate": (
                sum(beneficial_supporters) / len(beneficial_supporters)
                if beneficial_supporters else math.nan
            ),
            "conditional_mean_tail_gain_vs_fedavg": _mean(
                [row["tail_gain_support_normalized_vs_fedavg"] for row in valid]
            ),
            "conditional_mean_tail_gain_vs_random_p95": _mean(
                [row["tail_gain_support_normalized_vs_random_p95"] for row in valid]
            ),
            "conditional_mean_head_damage_vs_fedavg": _mean(
                [row["head_damage_support_normalized_vs_fedavg"] for row in valid]
            ),
            "conditional_mean_h_gain_vs_fedavg": _mean(
                [row["h_gain_support_normalized_vs_fedavg"] for row in valid]
            ),
            "mean_strict_supporter_count_all_tail_classes": _mean(
                [row["num_support_clients"] for row in items]
            ),
            "mean_strict_supporter_count_valid_classes": _mean(
                [row["num_support_clients"] for row in valid]
            ),
            "uncovered_tail_class_ids": uncovered,
        })
        uncovered_by_round[str(communication_round)] = uncovered

    tail_gains = [row["conditional_mean_tail_gain_vs_fedavg"] for row in conditional_rows]
    head_damage = [row["conditional_mean_head_damage_vs_fedavg"] for row in conditional_rows]
    h_gains = [row["conditional_mean_h_gain_vs_fedavg"] for row in conditional_rows]
    random_rates = [float(row["support_normalized_beats_random_p95_rate"]) for row in rows]
    valid_rates = [float(row["valid_support_class_rate"]) for row in rows]
    conditional_random_rates = [
        row["conditional_beats_random_p95_rate"] for row in conditional_rows
    ]
    conditional_fedavg_rates = [
        row["conditional_beats_fedavg_rate"] for row in conditional_rows
    ]
    positive_rounds = sum(value > 0.0 for value in tail_gains)
    phenomenon_checks = {
        "positive_at_least_two_rounds": positive_rounds >= 2,
        "conditional_mean_tail_gain_at_least_1pp": _mean(tail_gains) >= 1.0,
        "conditional_beats_random_p95_rate_at_least_0p6": (
            _mean(conditional_random_rates) >= 0.6
        ),
        "head_safe_or_h_improves": _mean(head_damage) <= 0.5 or _mean(h_gains) >= 0.0,
    }
    coverage_checks = {
        "strict_support_covers_at_least_0p8_of_tail_classes": _mean(valid_rates) >= 0.8,
    }
    phenomenon_pass = all(phenomenon_checks.values())
    coverage_pass = all(coverage_checks.values())
    if phenomenon_pass and coverage_pass:
        verdict = "D1_FULL_PASS"
    elif phenomenon_pass:
        verdict = "D1_SUPPORTED_WITH_COVERAGE_GAP"
    else:
        verdict = "D1_PHENOMENON_NOT_SUPPORTED"
    report = {
        "schema_version": "d1_support_counterfactual_v2",
        "seed": 42,
        "rounds": [20, 50, 80],
        "frozen_config_id": frozen["selected_config_id"],
        "frozen_config": frozen["selected_config"],
        "support_definition": "client class fraction > 0.1",
        "mean_tail_gain_support_normalized_vs_fedavg": _mean(tail_gains),
        "mean_head_damage_support_normalized_vs_fedavg": _mean(head_damage),
        "mean_h_gain_support_normalized_vs_fedavg": _mean(h_gains),
        "legacy_all_tail_beats_random_p95_rate": _mean(random_rates),
        "conditional_valid_class_beats_random_p95_rate": _mean(conditional_random_rates),
        "conditional_valid_class_beats_fedavg_rate": _mean(conditional_fedavg_rates),
        "mean_valid_support_class_rate": _mean(valid_rates),
        "positive_round_count": positive_rounds,
        "valid_class_conditioning_note": (
            "Tail/head gains and random-p95 comparisons are defined only for classes "
            "with at least one strict supporter. Uncovered classes are a support-rule "
            "coverage failure, not negative random-control trials."
        ),
        "phenomenon_checks": phenomenon_checks,
        "coverage_checks": coverage_checks,
        "phenomenon_pass": phenomenon_pass,
        "support_rule_coverage_pass": coverage_pass,
        "method_ready": phenomenon_pass and coverage_pass,
        "verdict": verdict,
        "conditional_round_rows": conditional_rows,
        "uncovered_tail_classes_by_round": uncovered_by_round,
        "round_rows": rows,
    }
    summary_dir = output_root / "d1_summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    path = summary_dir / "d1_verdict.json"
    safe_report = _json_safe(report)
    path.write_text(
        json.dumps(safe_report, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    print(f"\nD1 summary written: {path}", flush=True)
    print(json.dumps(safe_report, indent=2, sort_keys=True, allow_nan=False), flush=True)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage", choices=["g0", "d1", "d1-summary", "all"], default="all"
    )
    parser.add_argument("--output-root", type=Path, default=Path("output/g0_d1_seed42"))
    parser.add_argument("--data-root", type=Path, default=Path("DATA"))
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--test-batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--random-support-count", type=int, default=20)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-completed", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_root = args.output_root.resolve()
    args.data_root = args.data_root.resolve()
    if not args.dry_run:
        (args.output_root / "schedules").mkdir(parents=True, exist_ok=True)

    frozen = None
    if args.stage == "d1-summary":
        freeze_path = args.output_root / "lora_freeze.json"
        if not freeze_path.exists():
            raise FileNotFoundError(f"D1 summary requires {freeze_path}")
        frozen = json.loads(freeze_path.read_text(encoding="utf-8"))
        summarize_d1(args.output_root, frozen)
        return
    if args.stage in {"g0", "all"}:
        for config_id in CONFIGS:
            output_dir, command = build_g0_command(args, config_id)
            summary_path = _probe_summary_path(output_dir)
            if summary_path.exists() and args.skip_completed:
                print(f"Skip completed G0 config: {summary_path}", flush=True)
                continue
            if _is_nonempty(output_dir) and not args.dry_run:
                raise FileExistsError(
                    f"Refusing to overwrite non-empty G0 directory: {output_dir}. "
                    "Use a fresh --output-root or --skip-completed."
                )
            _run(command, gpu=args.gpu, dry_run=args.dry_run)
        if not args.dry_run:
            frozen = freeze_lora(args.output_root)

    if args.stage in {"d1", "all"}:
        freeze_path = args.output_root / "lora_freeze.json"
        if frozen is None:
            if not freeze_path.exists():
                if args.dry_run:
                    frozen = {
                        "selected_config_id": "candidate_r4",
                        "selected_config": CONFIGS["candidate_r4"],
                    }
                else:
                    raise FileNotFoundError(
                        f"D1 requires the G0 freeze artifact: {freeze_path}"
                    )
            else:
                frozen = json.loads(freeze_path.read_text(encoding="utf-8"))
        if frozen.get("verdict", "PASS") != "PASS" or not frozen.get("selected_config"):
            raise RuntimeError("D1 refuses to run because G0 did not pass")
        output_dir, command = build_d1_command(args, frozen)
        verdict_path = args.output_root / "d1_summary" / "d1_verdict.json"
        if verdict_path.exists() and args.skip_completed:
            print(f"Skip completed D1: {verdict_path}", flush=True)
            return
        if _is_nonempty(output_dir) and args.skip_completed and not args.dry_run:
            print(
                "D1 run directory already exists; attempting summary-only recovery "
                f"from {output_dir}",
                flush=True,
            )
            summarize_d1(args.output_root, frozen)
            return
        if _is_nonempty(output_dir) and not args.dry_run:
            raise FileExistsError(
                f"Refusing to overwrite non-empty D1 directory: {output_dir}. "
                "Use a fresh --output-root or --skip-completed."
            )
        _run(command, gpu=args.gpu, dry_run=args.dry_run)
        if not args.dry_run:
            summarize_d1(args.output_root, frozen)


if __name__ == "__main__":
    main()
