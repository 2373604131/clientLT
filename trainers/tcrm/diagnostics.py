import csv
import json
from pathlib import Path

import torch


def append_csv(path, row, fieldnames):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow({key: row.get(key, "") for key in fieldnames})


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def tail_diagnostic_rows(round_idx, state, metrics=None):
    metrics = metrics or {}
    class_acc = metrics.get("class_acc", None)
    zero_shot_class_acc = metrics.get("zero_shot_class_acc", None)
    class_tail_vs_head_margin = metrics.get("class_tail_vs_head_margin", None)
    class_tail_vs_tail_margin = metrics.get("class_tail_vs_tail_margin", None)
    rows = []
    for tail_index, class_id in enumerate(state.tail_class_ids):
        tail_acc = float(class_acc[class_id].item()) if class_acc is not None else 0.0
        zero_shot_tail_acc = float(zero_shot_class_acc[class_id].item()) if zero_shot_class_acc is not None else 0.0
        rows.append({
            "round": int(round_idx),
            "class_id": int(class_id),
            "tail_index": int(tail_index),
            "M_k": float(state.M[tail_index].item()),
            "C_k": float(state.C[tail_index].item()),
            "D_k": float(state.D[tail_index].item()),
            "age_k": float(state.age[tail_index].item()),
            "R_pre_k": float(state.r_pre[tail_index].item()),
            "width_gate_k": float(state.width_gate[tail_index].item()),
            "valid_contributors_nu": float(state.last_num_valid_contributors[tail_index].item()),
            "candidate_skip_count_k": float(state.last_candidate_skip_count[tail_index].item()),
            "corroboration_B_k": float(state.last_corroboration[tail_index].item()),
            "direction_consistency_Cdir_k": float(state.last_direction_consistency[tail_index].item()),
            "local_gain_Gpost_k": float(state.last_local_gain[tail_index].item()),
            "write_weight_W_k": float(state.last_write[tail_index].item()),
            "rho_norm_k": float(state.rho[tail_index].float().norm().item()),
            "survival_decay_k": float(state.last_decay[tail_index].item()),
            "tail_accuracy_k": tail_acc,
            "zero_shot_tail_accuracy_k": zero_shot_tail_acc,
            "tail_gain_over_zero_shot_k": tail_acc - zero_shot_tail_acc,
            "tail_vs_head_margin_k": float(class_tail_vs_head_margin[class_id].item()) if class_tail_vs_head_margin is not None else metrics.get("mean_tail_vs_head_margin", 0.0),
            "tail_vs_tail_margin_k": float(class_tail_vs_tail_margin[class_id].item()) if class_tail_vs_tail_margin is not None else metrics.get("mean_tail_vs_tail_margin", 0.0),
        })
    return rows


ROUND_FIELDS = [
    "round", "overall_acc", "macro_acc", "head_acc", "tail_acc",
    "zero_shot_overall_acc", "zero_shot_head_acc", "zero_shot_tail_acc",
    "hybrid_tail_acc", "tail_gain_over_zero_shot",
    "tail_to_head_error_rate", "tail_to_tail_error_rate",
    "mean_tail_vs_head_margin", "mean_tail_vs_tail_margin",
    "hbs_loss_mean", "prompt_grad_norm", "prompt_delta_norm",
    "number_of_prompt_contributing_clients", "rho_grad_norm_mean", "rho_grad_norm_max",
    "mean_rho_norm", "max_rho_norm", "mean_r_pre", "mean_write_weight",
    "mean_direction_consistency", "mean_local_admission_gain", "mean_corroboration",
    "mean_candidate_skip_count", "mean_effective_age", "round_time",
]


TAIL_FIELDS = [
    "round", "class_id", "tail_index", "M_k", "C_k", "D_k", "age_k",
    "R_pre_k", "width_gate_k", "valid_contributors_nu", "candidate_skip_count_k", "corroboration_B_k",
    "direction_consistency_Cdir_k", "local_gain_Gpost_k", "write_weight_W_k",
    "rho_norm_k", "survival_decay_k", "tail_accuracy_k",
    "zero_shot_tail_accuracy_k", "tail_gain_over_zero_shot_k", "tail_vs_head_margin_k",
    "tail_vs_tail_margin_k",
]
