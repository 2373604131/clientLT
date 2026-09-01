"""Read-only temporal diagnostics for the ClipLoRA class residual stream.

The runtime in this module never returns a model state to the optimizer or the
server aggregation code.  It only snapshots already-computed states and runs
test-set evaluations after preserving the process RNG state.
"""

from __future__ import annotations

import csv
import json
import math
import os
import random
from pathlib import Path

import numpy as np
import torch


SCHEMA_VERSION = "stage2c_temporal_misalignment_v1"


def parse_stage2c_rounds(value, total_rounds):
    """Parse unique, increasing, one-based communication rounds."""
    if isinstance(value, (list, tuple, set)):
        pieces = list(value)
    else:
        pieces = [piece.strip() for piece in str(value).split(",") if piece.strip()]
    rounds = sorted(set(int(piece) for piece in pieces))
    if not rounds:
        raise ValueError("Stage 2-C requires at least one checkpoint round")
    if rounds[0] < 1 or rounds[-1] > int(total_rounds):
        raise ValueError("Stage 2-C checkpoint rounds must be within [1, --round]")
    return rounds


def _cpu_clone(tensor):
    return tensor.detach().cpu().clone()


def split_model_state(state_dict, shared_keys, residual_keys):
    """Split a complete model state into fixed, shared-LoRA, and residual parts."""
    shared_keys = set(shared_keys)
    residual_keys = set(residual_keys)
    overlap = shared_keys.intersection(residual_keys)
    if overlap:
        raise ValueError("Shared and residual state keys overlap: {}".format(sorted(overlap)))
    missing = (shared_keys.union(residual_keys)).difference(state_dict)
    if missing:
        raise KeyError("Diagnostic state keys are missing: {}".format(sorted(missing)))
    fixed = {}
    shared = {}
    residual = {}
    for key, value in state_dict.items():
        if key in shared_keys:
            shared[key] = _cpu_clone(value)
        elif key in residual_keys:
            residual[key] = _cpu_clone(value)
        else:
            fixed[key] = _cpu_clone(value)
    return fixed, shared, residual


def combine_model_state(fixed_state, shared_state, residual_state, zero_residual=False):
    """Reconstruct a strict full state, optionally zeroing residual tensors."""
    combined = {}
    for source in (fixed_state, shared_state):
        for key, value in source.items():
            if key in combined:
                raise ValueError("Duplicate state key: {}".format(key))
            combined[key] = _cpu_clone(value)
    for key, value in residual_state.items():
        if key in combined:
            raise ValueError("Duplicate state key: {}".format(key))
        combined[key] = torch.zeros_like(value) if zero_residual else _cpu_clone(value)
    return combined


def _atomic_torch_save(payload, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(str(temporary), str(path))


def _write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(str(temporary), str(path))


def _write_csv(path, rows, fieldnames=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    if fieldnames is None:
        fieldnames = list(rows[0]) if rows else []
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        if fieldnames:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    os.replace(str(temporary), str(path))


def _capture_rng_state():
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def _hmean(head, tail):
    denominator = float(head) + float(tail)
    return 2.0 * float(head) * float(tail) / denominator if denominator > 0 else 0.0


def _class_groups(class_counts, tail_ratio):
    counts = [float(value) for value in class_counts]
    tail_count = max(1, int(round(len(counts) * float(tail_ratio))))
    tail_count = min(tail_count, len(counts))
    # Match federated_main.get_lt_class_splits_from_counts exactly, including
    # the intended larger-class-id tie break at the tail boundary.
    tail = set(sorted(
        range(len(counts)), key=lambda class_id: (counts[class_id], -class_id)
    )[:tail_count])
    order = set(range(len(counts)))
    return tail, set(order).difference(tail)


def metrics_from_test_result(result, margins, class_counts, tail_ratio):
    per_class = result[3]
    tail_ids, head_ids = _class_groups(class_counts, tail_ratio)

    def mean_for(class_ids):
        values = [float(per_class.get(class_id, 0.0)) for class_id in class_ids]
        return float(np.mean(values)) if values else 0.0

    head = mean_for(head_ids)
    tail = mean_for(tail_ids)
    macro = mean_for(range(len(class_counts)))
    summary = {
        "overall_acc": float(result[0]),
        "head_acc": head,
        "tail_acc": tail,
        "macro_per_class_acc": macro,
        "macro_f1": float(result[2]) if len(result) > 2 else math.nan,
        "head_tail_h_mean": _hmean(head, tail),
    }
    class_rows = []
    for class_id in range(len(class_counts)):
        class_rows.append({
            "class_id": int(class_id),
            "class_group": "tail" if class_id in tail_ids else "head",
            "global_class_count": float(class_counts[class_id]),
            "accuracy": float(per_class.get(class_id, 0.0)),
            "margin": float(margins.get(class_id, math.nan)),
        })
    return summary, class_rows


def diagnose_route(cross_rows, early_round, late_round, substantive_drop=2.0):
    """Produce a transparent, post-training routing suggestion from H-mean."""
    lookup = {
        (int(row["shared_round"]), str(row["residual_round"])): row
        for row in cross_rows
    }

    def metric(shared_round, residual_round):
        return float(lookup[(int(shared_round), str(residual_round))]["head_tail_h_mean"])

    early_matched = metric(early_round, early_round)
    late_matched = metric(late_round, late_round)
    old_on_new = metric(late_round, early_round)
    new_on_old = metric(early_round, late_round)
    zero_early = metric(early_round, "zero")
    zero_late = metric(late_round, "zero")
    old_on_new_drop = late_matched - old_on_new
    new_on_old_drop = early_matched - new_on_old
    shared_only_drop = zero_early - zero_late
    threshold = float(substantive_drop)
    old_bad = old_on_new_drop >= threshold
    new_bad = new_on_old_drop >= threshold
    shared_bad = shared_only_drop >= threshold
    if shared_bad:
        route = "shared_substrate_degradation"
    elif old_bad and new_bad:
        route = "shared_residual_coadaptation"
    elif old_bad:
        route = "temporal_residual_transport_alignment"
    elif new_bad:
        route = "residual_aggregation_or_stability"
    else:
        route = "no_clear_temporal_misalignment"
    return {
        "primary_metric": "head_tail_h_mean",
        "substantive_drop_threshold_pp": threshold,
        "early_round": int(early_round),
        "late_round": int(late_round),
        "early_matched": early_matched,
        "late_matched": late_matched,
        "shared_late_residual_early": old_on_new,
        "shared_early_residual_late": new_on_old,
        "shared_early_zero_residual": zero_early,
        "shared_late_zero_residual": zero_late,
        "old_residual_on_new_shared_drop_pp": old_on_new_drop,
        "new_residual_on_old_shared_drop_pp": new_on_old_drop,
        "shared_only_early_to_late_drop_pp": shared_only_drop,
        "recommended_route": route,
        "interpretation": (
            "Descriptive seed-42 diagnostic only. The threshold selects a follow-up "
            "route after training and never controls optimization or aggregation."
        ),
    }


class Stage2CTemporalDiagnostic(object):
    """Checkpoint, aggregation-stage, and cross-swap diagnostic manager."""

    STAGES = ("pre_aggregation", "after_shared", "after_full")

    def __init__(
        self,
        output_dir,
        checkpoint_rounds,
        shared_keys,
        residual_keys,
        class_counts,
        tail_ratio,
        protocol,
        substantive_drop=2.0,
    ):
        self.root = Path(output_dir) / "stage2c"
        self.checkpoint_dir = self.root / "checkpoints"
        self.rounds = tuple(sorted(int(value) for value in checkpoint_rounds))
        self.shared_keys = tuple(sorted(shared_keys))
        self.residual_keys = tuple(sorted(residual_keys))
        self.class_counts = [float(value) for value in class_counts]
        self.tail_ratio = float(tail_ratio)
        self.substantive_drop = float(substantive_drop)
        self.stage_records = {}
        self.saved_rounds = set()
        if self.root.exists() and any(self.root.iterdir()):
            raise FileExistsError(
                "Refusing to mix a new Stage 2-C run with existing diagnostics: {}".format(
                    self.root
                )
            )
        self.root.mkdir(parents=True, exist_ok=True)
        payload = dict(protocol)
        payload.update({
            "schema_version": SCHEMA_VERSION,
            "checkpoint_rounds": list(self.rounds),
            "shared_keys": list(self.shared_keys),
            "residual_keys": list(self.residual_keys),
            "test_metrics_control_training": False,
            "cross_swap_runs_after_training": True,
            "substantive_drop_threshold_pp": self.substantive_drop,
        })
        _write_json(self.root / "protocol.json", payload)

    def is_selected(self, communication_round):
        return int(communication_round) in self.rounds

    def save_checkpoint(self, communication_round, state_dict):
        communication_round = int(communication_round)
        if not self.is_selected(communication_round):
            raise ValueError("Round {} was not selected".format(communication_round))
        fixed_path = self.checkpoint_dir / "fixed_state.pt"
        if not fixed_path.exists():
            fixed, shared, residual = split_model_state(
                state_dict, self.shared_keys, self.residual_keys
            )
            _atomic_torch_save(
                {"schema_version": SCHEMA_VERSION, "state_dict": fixed}, fixed_path
            )
        else:
            missing = set(self.shared_keys).union(self.residual_keys).difference(state_dict)
            if missing:
                raise KeyError("Diagnostic state keys are missing: {}".format(sorted(missing)))
            shared = {
                key: _cpu_clone(state_dict[key]) for key in self.shared_keys
            }
            residual = {
                key: _cpu_clone(state_dict[key]) for key in self.residual_keys
            }
        checkpoint_path = self.checkpoint_dir / "round_{:03d}.pt".format(communication_round)
        _atomic_torch_save({
            "schema_version": SCHEMA_VERSION,
            "communication_round": communication_round,
            "shared_state": shared,
            "residual_state": residual,
        }, checkpoint_path)
        self.saved_rounds.add(communication_round)

    def evaluate_state(self, trainer, state_dict, communication_round, label):
        """Evaluate without allowing the diagnostic pass to advance RNG state."""
        rng_state = _capture_rng_state()
        try:
            trainer.model.load_state_dict(state_dict, strict=True)
            result = trainer.global_test(
                is_global=True, current_epoch=int(communication_round) - 1
            )
            margins = getattr(trainer, "last_global_test_class_margins", {})
            summary, class_rows = metrics_from_test_result(
                result, margins, self.class_counts, self.tail_ratio
            )
        finally:
            _restore_rng_state(rng_state)
        summary.update({
            "communication_round": int(communication_round),
            "evaluation": str(label),
        })
        for row in class_rows:
            row.update({
                "communication_round": int(communication_round),
                "evaluation": str(label),
            })
        return result, summary, class_rows

    def record_stage(self, communication_round, stage, result, margins):
        if stage not in self.STAGES:
            raise ValueError("Unknown Stage 2-C aggregation stage: {}".format(stage))
        summary, class_rows = metrics_from_test_result(
            result, margins, self.class_counts, self.tail_ratio
        )
        self.stage_records[(int(communication_round), str(stage))] = (
            summary, class_rows
        )

    def _write_stage_outputs(self):
        summary_rows = []
        class_rows = []
        metric_names = (
            "overall_acc", "head_acc", "tail_acc", "macro_per_class_acc",
            "macro_f1", "head_tail_h_mean",
        )
        for communication_round in self.rounds:
            missing = [
                stage for stage in self.STAGES
                if (communication_round, stage) not in self.stage_records
            ]
            if missing:
                raise RuntimeError(
                    "Round {} is missing aggregation stages: {}".format(
                        communication_round, missing
                    )
                )
            summaries = {
                stage: self.stage_records[(communication_round, stage)][0]
                for stage in self.STAGES
            }
            wide = {"communication_round": communication_round}
            for metric_name in metric_names:
                pre = float(summaries["pre_aggregation"][metric_name])
                shared = float(summaries["after_shared"][metric_name])
                full = float(summaries["after_full"][metric_name])
                wide["pre_{}".format(metric_name)] = pre
                wide["after_shared_{}".format(metric_name)] = shared
                wide["after_full_{}".format(metric_name)] = full
                wide["shared_delta_{}".format(metric_name)] = shared - pre
                wide["residual_delta_{}".format(metric_name)] = full - shared
                wide["total_delta_{}".format(metric_name)] = full - pre
                wide["decomposition_error_{}".format(metric_name)] = (
                    (shared - pre) + (full - shared) - (full - pre)
                )
            summary_rows.append(wide)

            per_stage = {
                stage: {
                    int(row["class_id"]): row
                    for row in self.stage_records[(communication_round, stage)][1]
                }
                for stage in self.STAGES
            }
            for class_id in range(len(self.class_counts)):
                pre_row = per_stage["pre_aggregation"][class_id]
                shared_row = per_stage["after_shared"][class_id]
                full_row = per_stage["after_full"][class_id]
                row = {
                    "communication_round": communication_round,
                    "class_id": class_id,
                    "class_group": pre_row["class_group"],
                    "global_class_count": pre_row["global_class_count"],
                }
                for metric_name in ("accuracy", "margin"):
                    pre = float(pre_row[metric_name])
                    shared = float(shared_row[metric_name])
                    full = float(full_row[metric_name])
                    row["pre_{}".format(metric_name)] = pre
                    row["after_shared_{}".format(metric_name)] = shared
                    row["after_full_{}".format(metric_name)] = full
                    row["shared_delta_{}".format(metric_name)] = shared - pre
                    row["residual_delta_{}".format(metric_name)] = full - shared
                    row["total_delta_{}".format(metric_name)] = full - pre
                    row["decomposition_error_{}".format(metric_name)] = (
                        (shared - pre) + (full - shared) - (full - pre)
                    )
                class_rows.append(row)
        _write_csv(self.root / "aggregation_stage_summary.csv", summary_rows)
        _write_csv(self.root / "aggregation_stage_per_class.csv", class_rows)
        return summary_rows, class_rows

    def _load_checkpoint_parts(self, communication_round):
        path = self.checkpoint_dir / "round_{:03d}.pt".format(int(communication_round))
        payload = torch.load(str(path), map_location="cpu")
        return payload["shared_state"], payload["residual_state"]

    def run_cross_swap(self, trainer, restore_state):
        expected = set(self.rounds)
        if self.saved_rounds and self.saved_rounds != expected:
            raise RuntimeError("Not all selected Stage 2-C checkpoints were saved")
        fixed_payload = torch.load(
            str(self.checkpoint_dir / "fixed_state.pt"), map_location="cpu"
        )
        fixed_state = fixed_payload["state_dict"]
        parts = {}
        for communication_round in self.rounds:
            parts[communication_round] = self._load_checkpoint_parts(communication_round)

        summary_rows = []
        per_class_rows = []
        try:
            for shared_round in self.rounds:
                shared_state = parts[shared_round][0]
                residual_choices = list(self.rounds) + ["zero"]
                for residual_round in residual_choices:
                    source_round = self.rounds[0] if residual_round == "zero" else residual_round
                    residual_state = parts[source_round][1]
                    combined = combine_model_state(
                        fixed_state,
                        shared_state,
                        residual_state,
                        zero_residual=(residual_round == "zero"),
                    )
                    _, summary, class_rows = self.evaluate_state(
                        trainer,
                        combined,
                        communication_round=shared_round,
                        label="shared_{}_residual_{}".format(shared_round, residual_round),
                    )
                    summary.update({
                        "shared_round": int(shared_round),
                        "residual_round": str(residual_round),
                        "matched_time": bool(residual_round == shared_round),
                        "zero_residual": bool(residual_round == "zero"),
                    })
                    summary_rows.append(summary)
                    for row in class_rows:
                        row.update({
                            "shared_round": int(shared_round),
                            "residual_round": str(residual_round),
                            "matched_time": bool(residual_round == shared_round),
                            "zero_residual": bool(residual_round == "zero"),
                        })
                        per_class_rows.append(row)
        finally:
            trainer.model.load_state_dict(restore_state, strict=True)

        zero_summary = {
            int(row["shared_round"]): row
            for row in summary_rows if row["zero_residual"]
        }
        metric_names = (
            "overall_acc", "head_acc", "tail_acc", "macro_per_class_acc",
            "macro_f1", "head_tail_h_mean",
        )
        for row in summary_rows:
            zero = zero_summary[int(row["shared_round"])]
            for metric_name in metric_names:
                row["residual_net_{}".format(metric_name)] = (
                    float(row[metric_name]) - float(zero[metric_name])
                )

        zero_class = {
            (int(row["shared_round"]), int(row["class_id"])): row
            for row in per_class_rows if row["zero_residual"]
        }
        for row in per_class_rows:
            zero = zero_class[(int(row["shared_round"]), int(row["class_id"]))]
            row["residual_net_accuracy"] = float(row["accuracy"]) - float(zero["accuracy"])
            row["residual_net_margin"] = float(row["margin"]) - float(zero["margin"])

        _write_csv(self.root / "cross_swap_summary.csv", summary_rows)
        _write_csv(self.root / "cross_swap_per_class.csv", per_class_rows)
        stage_summary, _ = self._write_stage_outputs()
        route = diagnose_route(
            summary_rows,
            early_round=self.rounds[0],
            late_round=self.rounds[-1],
            substantive_drop=self.substantive_drop,
        )
        final_payload = {
            "schema_version": SCHEMA_VERSION,
            "checkpoint_rounds": list(self.rounds),
            "num_cross_swap_evaluations": len(self.rounds) * len(self.rounds),
            "num_zero_residual_evaluations": len(self.rounds),
            "aggregation_stage_rows": len(stage_summary),
            "route": route,
            "files": {
                "aggregation_stage_summary": "aggregation_stage_summary.csv",
                "aggregation_stage_per_class": "aggregation_stage_per_class.csv",
                "cross_swap_summary": "cross_swap_summary.csv",
                "cross_swap_per_class": "cross_swap_per_class.csv",
            },
            "causal_scope": (
                "All evaluations are post-update diagnostics. Test metrics are never "
                "used to select a checkpoint, gate, hyperparameter, or server update."
            ),
        }
        _write_json(self.root / "stage2c_summary.json", final_payload)
        report = [
            "# Stage 2-C temporal misalignment diagnostic",
            "",
            "- Recommended next route: `{}`".format(route["recommended_route"]),
            "- Old residual on new shared drop (H-mean): {:.4f} pp".format(
                route["old_residual_on_new_shared_drop_pp"]
            ),
            "- New residual on old shared drop (H-mean): {:.4f} pp".format(
                route["new_residual_on_old_shared_drop_pp"]
            ),
            "- Shared-only early-to-late drop (H-mean): {:.4f} pp".format(
                route["shared_only_early_to_late_drop_pp"]
            ),
            "",
            "This is a descriptive seed-42 diagnostic. Test results did not control training.",
        ]
        (self.root / "stage2c_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
        return final_payload
