import csv
import json
import math
import os
import time
from collections.abc import Mapping
from pathlib import Path

import torch


PER_CLASS_FIELDS = [
    "epoch_index",
    "communication_round",
    "seed",
    "partition",
    "num_users",
    "frac",
    "local_epochs",
    "class_id",
    "class_group",
    "global_class_count",
    "num_support_clients",
    "support_client_fraction",
    "support_fedavg_weight",
    "support_normalized_denominator",
    "class_positive_samples",
    "positive_samples_in_tail_specialists",
    "positive_sample_specialist_ratio",
    "num_tail_specialists",
    "num_support_tail_specialists",
    "acc_before",
    "acc_support_actual",
    "acc_support_normalized",
    "acc_all",
    "gain_support_actual",
    "gain_support_normalized",
    "gain_all",
    "offset_gap",
    "support_actual_positive",
    "support_normalized_positive",
    "offset_observed",
    "full_reversal",
    "specialization_lambda",
    "intra_group_alpha",
    "head_leakage_scale",
    "beta",
]

ROUND_SUMMARY_FIELDS = [
    "epoch_index",
    "communication_round",
    "seed",
    "partition",
    "local_epochs",
    "num_tail_classes",
    "mean_gain_support_actual",
    "median_gain_support_actual",
    "mean_gain_support_normalized",
    "median_gain_support_normalized",
    "mean_gain_all",
    "median_gain_all",
    "mean_offset_gap",
    "median_offset_gap",
    "support_actual_positive_rate",
    "support_normalized_positive_rate",
    "offset_observed_rate",
    "full_reversal_rate",
    "mean_support_fedavg_weight",
    "mean_num_support_clients",
    "mean_positive_sample_specialist_ratio",
]

UPDATE_NORM_FIELDS = [
    "epoch_index",
    "communication_round",
    "seed",
    "partition",
    "local_epochs",
    "client_id",
    "client_role",
    "client_num_samples",
    "update_norm",
    "is_finite",
]

UPDATE_NORM_SUMMARY_FIELDS = [
    "epoch_index",
    "communication_round",
    "seed",
    "partition",
    "local_epochs",
    "mean_update_norm_all",
    "median_update_norm_all",
    "mean_update_norm_head_clients",
    "mean_update_norm_tail_specialists",
    "max_update_norm",
    "min_update_norm",
    "nan_count",
    "inf_count",
]

RUNTIME_FIELDS = [
    "epoch_index",
    "communication_round",
    "local_epochs",
    "local_training_seconds",
    "experimentD_diagnostic_seconds",
    "normal_global_eval_seconds",
    "round_total_seconds",
    "cumulative_seconds",
]


def parse_experiment_d_rounds(rounds):
    if rounds is None:
        return set()
    if isinstance(rounds, (list, tuple, set)):
        values = rounds
    else:
        text = str(rounds).strip()
        if not text:
            return set()
        values = [x.strip() for x in text.split(",")]

    parsed = set()
    for value in values:
        if value == "":
            continue
        round_id = int(value)
        if round_id <= 0:
            raise ValueError("Experiment D rounds are 1-based and must be positive")
        parsed.add(round_id)
    return parsed


def should_log_experiment_d(args, epoch):
    if not bool(getattr(args, "experimentD_enable", False)):
        return False
    communication_round = int(epoch) + 1
    return communication_round in parse_experiment_d_rounds(
        getattr(args, "experimentD_rounds", "")
    )


def experiment_d_output_dir(output_dir):
    path = Path(output_dir) / "experiment_d"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _append_csv(path, row, fieldnames):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerow({key: row.get(key, "") for key in fieldnames})


def _append_many_csv(path, rows, fieldnames):
    for row in rows:
        _append_csv(path, row, fieldnames)


def _is_float_or_complex_tensor(value):
    return isinstance(value, torch.Tensor) and (
        torch.is_floating_point(value) or torch.is_complex(value)
    )


def clone_state_dict(state_dict):
    cloned = {}
    for key, value in state_dict.items():
        if isinstance(value, torch.Tensor):
            cloned[key] = value.detach().clone()
        else:
            cloned[key] = value
    return cloned


def fedavg_client_weights(selected_clients, datanumber_client):
    selected = [int(x) for x in selected_clients]
    total = sum(float(datanumber_client[idx]) for idx in selected)
    if total <= 0:
        raise ValueError("selected clients have zero total samples")
    return {idx: float(datanumber_client[idx]) / total for idx in selected}


def _weighted_delta_state(global_before, local_weights, client_weight_items):
    with torch.no_grad():
        out = {}
        for key, before_value in global_before.items():
            if not isinstance(before_value, torch.Tensor):
                out[key] = before_value
                continue

            if not _is_float_or_complex_tensor(before_value):
                out[key] = before_value.detach().clone()
                continue

            target = before_value.detach().clone()
            for client_id, weight in client_weight_items:
                local_value = local_weights[int(client_id)][key]
                target = target + (local_value.detach().to(before_value.device) - before_value.detach()) * float(weight)
            out[key] = target.to(dtype=before_value.dtype)
        return out


def reconstruct_full_fedavg_state(global_before, local_weights, selected_clients, client_weights):
    items = [(int(client_id), float(client_weights[int(client_id)])) for client_id in selected_clients]
    return _weighted_delta_state(global_before, local_weights, items)


def build_support_actual_state(
    global_before,
    local_weights,
    selected_clients,
    support_clients,
    client_weights,
):
    selected_set = {int(x) for x in selected_clients}
    support = [int(x) for x in support_clients if int(x) in selected_set]
    items = [(client_id, float(client_weights[int(client_id)])) for client_id in support]
    return _weighted_delta_state(global_before, local_weights, items)


def build_non_support_actual_state(
    global_before,
    local_weights,
    selected_clients,
    support_clients,
    client_weights,
):
    support_set = {int(x) for x in support_clients}
    items = [
        (int(client_id), float(client_weights[int(client_id)]))
        for client_id in selected_clients
        if int(client_id) not in support_set
    ]
    return _weighted_delta_state(global_before, local_weights, items)


def build_support_normalized_state(
    global_before,
    local_weights,
    support_clients,
    datanumber_client,
):
    support = [int(x) for x in support_clients]
    total = sum(float(datanumber_client[idx]) for idx in support)
    if total <= 0:
        return clone_state_dict(global_before)
    items = [(idx, float(datanumber_client[idx]) / total) for idx in support]
    return _weighted_delta_state(global_before, local_weights, items)


def max_state_abs_diff(left, right):
    max_diff = 0.0
    max_key = ""
    for key, left_value in left.items():
        right_value = right[key]
        if not isinstance(left_value, torch.Tensor) or not isinstance(right_value, torch.Tensor):
            continue
        if not _is_float_or_complex_tensor(left_value):
            if not torch.equal(left_value.detach().cpu(), right_value.detach().cpu()):
                return math.inf, key
            continue
        diff = (left_value.detach().float().cpu() - right_value.detach().float().cpu()).abs()
        current = float(diff.max().item()) if diff.numel() else 0.0
        if current > max_diff:
            max_diff = current
            max_key = key
    return max_diff, max_key


def verify_fedavg_reconstruction(
    reconstructed,
    theta_all_from_average_weights,
    *,
    global_before=None,
    atol=1e-6,
    rtol=1e-5,
    raise_on_error=True,
):
    def _dtype_tolerance(dtype):
        if dtype == torch.float16:
            return max(float(atol), 5e-3), max(float(rtol), 5e-3)
        if dtype == torch.bfloat16:
            return max(float(atol), 1e-2), max(float(rtol), 1e-2)
        return float(atol), float(rtol)

    mismatches = []
    for key, expected in theta_all_from_average_weights.items():
        if global_before is not None:
            before_value = global_before.get(key)
            if isinstance(before_value, torch.Tensor) and not _is_float_or_complex_tensor(before_value):
                continue
        actual = reconstructed.get(key)
        if actual is None:
            mismatches.append({"key": key, "reason": "missing"})
            continue
        if not isinstance(expected, torch.Tensor):
            continue
        if not _is_float_or_complex_tensor(expected):
            if not torch.equal(actual.detach().cpu(), expected.detach().cpu()):
                mismatches.append({"key": key, "reason": "non_float_buffer_mismatch"})
            continue
        key_atol, key_rtol = _dtype_tolerance(expected.dtype)
        if not torch.allclose(actual.detach().cpu(), expected.detach().cpu(), atol=key_atol, rtol=key_rtol):
            abs_diff = (actual.detach().float().cpu() - expected.detach().float().cpu()).abs()
            denom = expected.detach().float().cpu().abs().clamp_min(1e-12)
            rel_diff = abs_diff / denom
            mismatches.append(
                {
                    "key": key,
                    "max_abs": float(abs_diff.max().item()) if abs_diff.numel() else 0.0,
                    "max_rel": float(rel_diff.max().item()) if rel_diff.numel() else 0.0,
                    "shape": tuple(expected.shape),
                    "dtype": str(expected.dtype),
                    "atol": key_atol,
                    "rtol": key_rtol,
                }
            )
    if mismatches and raise_on_error:
        first = mismatches[0]
        raise RuntimeError(f"Experiment D FedAvg reconstruction mismatch: {first}")
    return mismatches


def fedavg_reconstruction_error_report(reconstructed, theta_all_from_average_weights, global_before=None):
    report = {"key": "", "max_abs": 0.0, "max_rel": 0.0, "dtype": ""}
    for key, expected in theta_all_from_average_weights.items():
        if global_before is not None:
            before_value = global_before.get(key)
            if isinstance(before_value, torch.Tensor) and not _is_float_or_complex_tensor(before_value):
                continue
        actual = reconstructed.get(key)
        if actual is None or not _is_float_or_complex_tensor(expected):
            continue
        abs_diff = (actual.detach().float().cpu() - expected.detach().float().cpu()).abs()
        if not abs_diff.numel():
            continue
        denom = expected.detach().float().cpu().abs().clamp_min(1e-12)
        rel_diff = abs_diff / denom
        max_abs = float(abs_diff.max().item())
        if max_abs >= report["max_abs"]:
            report = {
                "key": key,
                "max_abs": max_abs,
                "max_rel": float(rel_diff.max().item()),
                "dtype": str(expected.dtype),
            }
    return report


def validate_support_decomposition(global_before, support_state, non_support_state, full_state, *, atol=1e-6, rtol=1e-5):
    mismatches = []
    for key, before_value in global_before.items():
        if not _is_float_or_complex_tensor(before_value):
            continue
        support_delta = support_state[key].detach().cpu() - before_value.detach().cpu()
        non_support_delta = non_support_state[key].detach().cpu() - before_value.detach().cpu()
        full_delta = full_state[key].detach().cpu() - before_value.detach().cpu()
        if not torch.allclose(support_delta + non_support_delta, full_delta, atol=atol, rtol=rtol):
            diff = (support_delta + non_support_delta - full_delta).abs()
            mismatches.append({"key": key, "max_abs": float(diff.max().item())})
    return mismatches


def get_trainable_state_keys(model):
    return {name for name, param in model.named_parameters() if param.requires_grad}


def client_counts_to_matrix(client_class_counts, num_users, num_classes):
    counts = torch.zeros(num_users, num_classes, dtype=torch.float32)
    for client_id, values in client_class_counts.items():
        idx = int(client_id)
        if 0 <= idx < num_users:
            counts[idx] = torch.as_tensor(values, dtype=torch.float32).cpu()
    return counts


def class_ids_from_tail_ratio(global_class_counts, tail_class_ratio):
    counts = torch.as_tensor(global_class_counts, dtype=torch.float32)
    num_classes = int(counts.numel())
    num_tail = max(1, int(round(num_classes * float(tail_class_ratio))))
    num_tail = min(num_tail, num_classes)
    order = torch.argsort(counts, descending=True)
    tail = [int(x) for x in order[-num_tail:].tolist()]
    head = [int(x) for x in order[:-num_tail].tolist()]
    return head, tail


def tail_specialist_clients(args):
    if str(getattr(args, "partition", "")) != "client-longtail":
        return []
    num_users = int(getattr(args, "num_users"))
    num_tail_clients = int(round(num_users * float(getattr(args, "tail_client_ratio", 0.0))))
    num_tail_clients = min(max(num_tail_clients, 0), num_users)
    return list(range(num_users - num_tail_clients, num_users))


def client_role_for_experiment_d(client_id, args):
    tail_clients = set(tail_specialist_clients(args))
    if not tail_clients:
        return ""
    return "tail_specialist" if int(client_id) in tail_clients else "head_client"


def validate_full_participation(args, selected_clients):
    selected = {int(x) for x in selected_clients}
    expected = set(range(int(getattr(args, "num_users"))))
    frac = float(getattr(args, "frac", 0.0))
    if abs(frac - 1.0) > 1e-12 or selected != expected:
        raise RuntimeError(
            "Experiment D requires full participation when "
            "--experimentD_require_full_participation True: "
            f"frac={frac}, selected={sorted(selected)}, expected={sorted(expected)}"
        )


def support_clients_for_class(client_class_counts, selected_clients, class_id):
    support = []
    for client_id in selected_clients:
        counts = client_class_counts[int(client_id)]
        if float(torch.as_tensor(counts)[int(class_id)].item()) > 0:
            support.append(int(client_id))
    return support


def _mean(values):
    valid = [float(x) for x in values if x is not None and not math.isnan(float(x))]
    return float(sum(valid) / len(valid)) if valid else math.nan


def _median(values):
    valid = sorted(float(x) for x in values if x is not None and not math.isnan(float(x)))
    if not valid:
        return math.nan
    mid = len(valid) // 2
    if len(valid) % 2:
        return valid[mid]
    return (valid[mid - 1] + valid[mid]) / 2.0


def _rate(values):
    valid = [bool(x) for x in values]
    return float(sum(1 for x in valid if x) / len(valid)) if valid else math.nan


def _as_bool_text(value):
    return "True" if bool(value) else "False"


def _parse_batch(trainer, batch):
    if hasattr(trainer, "parse_batch_test"):
        return trainer.parse_batch_test(batch)
    if isinstance(batch, Mapping):
        return batch["img"], batch["label"]
    if isinstance(batch, (list, tuple)) and len(batch) >= 2:
        return batch[0], batch[1]
    raise TypeError("Cannot parse evaluation batch")


def _model_inference(trainer, inputs):
    if hasattr(trainer, "model_inference"):
        return trainer.model_inference(inputs)
    return trainer.model(inputs)


def _get_default_eval_loader(trainer):
    for name in ("test_loader", "test_loader_x", "val_loader"):
        loader = getattr(trainer, name, None)
        if loader is not None:
            return loader
    raise RuntimeError("Experiment D could not find an evaluation loader on the trainer")


def evaluate_state_per_class(global_trainer, state_dict, class_ids, eval_loader_or_loaders=None):
    class_ids = [int(x) for x in class_ids]
    target_classes = set(class_ids)
    model = global_trainer.model
    original_state = clone_state_dict(model.state_dict())
    was_training = bool(getattr(model, "training", False))

    correct = {class_id: 0 for class_id in class_ids}
    total = {class_id: 0 for class_id in class_ids}

    try:
        model.load_state_dict(state_dict, strict=True)
        model.eval()
        loaders_by_class = eval_loader_or_loaders if isinstance(eval_loader_or_loaders, Mapping) else None

        with torch.no_grad():
            if loaders_by_class is not None:
                items = [(class_id, loaders_by_class[class_id]) for class_id in class_ids]
            else:
                loader = eval_loader_or_loaders or _get_default_eval_loader(global_trainer)
                items = [(None, loader)]

            for forced_class_id, loader in items:
                if loader is None:
                    continue
                for batch in loader:
                    inputs, labels = _parse_batch(global_trainer, batch)
                    outputs = _model_inference(global_trainer, inputs)
                    preds = outputs.argmax(dim=1)
                    labels_cpu = labels.detach().cpu()
                    preds_cpu = preds.detach().cpu()
                    for label, pred in zip(labels_cpu, preds_cpu):
                        label_id = int(label.item())
                        if forced_class_id is not None and label_id != int(forced_class_id):
                            continue
                        if label_id not in target_classes:
                            continue
                        total[label_id] += 1
                        if label_id == int(pred.item()):
                            correct[label_id] += 1

        return {
            class_id: (float(correct[class_id]) / max(float(total[class_id]), 1.0)) * 100.0
            for class_id in class_ids
        }
    finally:
        model.load_state_dict(original_state, strict=True)
        model.train(was_training)


def _dataset_item_label(item):
    if isinstance(item, Mapping):
        return int(item["label"])
    return int(getattr(item, "label"))


def build_class_filtered_eval_loaders(global_trainer, class_ids):
    try:
        from torch.utils.data import DataLoader, Subset
    except Exception:
        return None

    base_loader = _get_default_eval_loader(global_trainer)
    dataset = getattr(base_loader, "dataset", None)
    data_source = getattr(dataset, "data_source", None)
    if dataset is None or data_source is None:
        return None

    by_class = {}
    class_ids = [int(x) for x in class_ids]
    for class_id in class_ids:
        indices = [idx for idx, item in enumerate(data_source) if _dataset_item_label(item) == class_id]
        if not indices:
            return None
        by_class[class_id] = DataLoader(
            Subset(dataset, indices),
            batch_size=getattr(base_loader, "batch_size", 64) or 64,
            shuffle=False,
            num_workers=0,
            collate_fn=getattr(base_loader, "collate_fn", None),
            pin_memory=bool(getattr(base_loader, "pin_memory", False)),
        )
    return by_class


def compute_client_update_norms(global_before, local_weights, selected_clients, datanumber_client, trainable_keys, args):
    rows = []
    for client_id in [int(x) for x in selected_clients]:
        sq_sum = 0.0
        finite = True
        for key in trainable_keys:
            if key not in global_before or key not in local_weights[client_id]:
                continue
            before = global_before[key]
            local = local_weights[client_id][key]
            if not _is_float_or_complex_tensor(before):
                continue
            delta = local.detach().float().cpu() - before.detach().float().cpu()
            norm_sq = float(torch.sum(delta * delta).item())
            if not math.isfinite(norm_sq):
                finite = False
            sq_sum += norm_sq
        update_norm = math.sqrt(max(sq_sum, 0.0))
        rows.append(
            {
                "epoch_index": int(getattr(args, "_experiment_d_epoch", -1)),
                "communication_round": int(getattr(args, "_experiment_d_epoch", -1)) + 1,
                "seed": int(getattr(args, "seed", -1)),
                "partition": getattr(args, "partition", ""),
                "local_epochs": int(getattr(args, "local_epochs", -1)),
                "client_id": client_id,
                "client_role": client_role_for_experiment_d(client_id, args),
                "client_num_samples": float(datanumber_client[client_id]),
                "update_norm": update_norm,
                "is_finite": _as_bool_text(finite and math.isfinite(update_norm)),
            }
        )
    return rows


def summarize_client_update_norms(rows, args):
    has_clientlt_roles = bool(tail_specialist_clients(args))
    values_all = [float(row["update_norm"]) for row in rows if row["is_finite"] == "True"]
    tail_values = [
        float(row["update_norm"])
        for row in rows
        if has_clientlt_roles and row["is_finite"] == "True" and row["client_role"] == "tail_specialist"
    ]
    head_values = [
        float(row["update_norm"])
        for row in rows
        if has_clientlt_roles and row["is_finite"] == "True" and row["client_role"] == "head_client"
    ]
    nan_count = sum(1 for row in rows if math.isnan(float(row["update_norm"])))
    inf_count = sum(1 for row in rows if math.isinf(float(row["update_norm"])))
    epoch = int(getattr(args, "_experiment_d_epoch", -1))
    return {
        "epoch_index": epoch,
        "communication_round": epoch + 1,
        "seed": int(getattr(args, "seed", -1)),
        "partition": getattr(args, "partition", ""),
        "local_epochs": int(getattr(args, "local_epochs", -1)),
        "mean_update_norm_all": _mean(values_all),
        "median_update_norm_all": _median(values_all),
        "mean_update_norm_head_clients": _mean(head_values),
        "mean_update_norm_tail_specialists": _mean(tail_values),
        "max_update_norm": max(values_all) if values_all else math.nan,
        "min_update_norm": min(values_all) if values_all else math.nan,
        "nan_count": nan_count,
        "inf_count": inf_count,
    }


def append_client_update_norms(
    output_dir,
    args,
    epoch,
    global_before,
    local_weights,
    selected_clients,
    datanumber_client,
    trainable_keys,
):
    setattr(args, "_experiment_d_epoch", int(epoch))
    rows = compute_client_update_norms(
        global_before,
        local_weights,
        selected_clients,
        datanumber_client,
        trainable_keys,
        args,
    )
    out_dir = experiment_d_output_dir(output_dir)
    _append_many_csv(out_dir / "client_update_norms.csv", rows, UPDATE_NORM_FIELDS)
    _append_csv(
        out_dir / "client_update_norm_summary.csv",
        summarize_client_update_norms(rows, args),
        UPDATE_NORM_SUMMARY_FIELDS,
    )


def append_runtime_metrics(
    output_dir,
    epoch,
    local_epochs,
    local_training_seconds,
    experimentD_diagnostic_seconds,
    normal_global_eval_seconds,
    round_total_seconds,
    cumulative_seconds,
):
    out_dir = experiment_d_output_dir(output_dir)
    _append_csv(
        out_dir / "runtime_metrics.csv",
        {
            "epoch_index": int(epoch),
            "communication_round": int(epoch) + 1,
            "local_epochs": int(local_epochs),
            "local_training_seconds": float(local_training_seconds),
            "experimentD_diagnostic_seconds": float(experimentD_diagnostic_seconds),
            "normal_global_eval_seconds": float(normal_global_eval_seconds),
            "round_total_seconds": float(round_total_seconds),
            "cumulative_seconds": float(cumulative_seconds),
        },
        RUNTIME_FIELDS,
    )


def _support_normalized_denominator(support_clients, datanumber_client):
    return float(sum(float(datanumber_client[int(idx)]) for idx in support_clients))


def _write_experiment_d_metadata(out_dir, args):
    path = out_dir / "experiment_d_metadata.json"
    if path.exists():
        return
    with path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "description": (
                    "Experiment D counterfactual evaluation. Training remains ordinary "
                    "sample-weighted FedAvg. Diagnostics construct temporary full "
                    "state_dict counterfactual models for target tail classes and "
                    "measure per-class accuracy gains."
                ),
                "accuracy_unit": "percentage_points",
                "core_model": "support_actual uses original FedAvg weights over support clients only; no renormalization.",
                "rounds": getattr(args, "experimentD_rounds", ""),
                "include_normalized": bool(getattr(args, "experimentD_include_normalized", False)),
            },
            f,
            indent=2,
        )


def run_experiment_d_round(
    output_dir,
    args,
    epoch,
    global_trainer,
    global_before,
    theta_all,
    local_weights,
    selected_clients,
    datanumber_client,
    client_class_counts,
    num_classes,
):
    if bool(getattr(args, "experimentD_require_full_participation", False)):
        validate_full_participation(args, selected_clients)

    selected = [int(x) for x in selected_clients]
    client_weights = fedavg_client_weights(selected, datanumber_client)
    reconstructed = reconstruct_full_fedavg_state(
        global_before,
        local_weights,
        selected,
        client_weights,
    )
    if bool(getattr(args, "experimentD_verify_fedavg", False)):
        verify_fedavg_reconstruction(
            reconstructed,
            theta_all,
            global_before=global_before,
            raise_on_error=True,
        )
        report = fedavg_reconstruction_error_report(
            reconstructed,
            theta_all,
            global_before=global_before,
        )
        print(
            "Experiment D FedAvg reconstruction verified: "
            f"max_abs={report['max_abs']:.6g} "
            f"max_rel={report['max_rel']:.6g} "
            f"key={report['key']} dtype={report['dtype']}"
        )

    counts = client_counts_to_matrix(client_class_counts, int(getattr(args, "num_users")), num_classes)
    global_counts = counts.sum(dim=0)
    head_classes, tail_classes = class_ids_from_tail_ratio(
        global_counts,
        getattr(args, "tail_class_ratio", 0.2),
    )
    head_set = set(head_classes)
    tail_set = set(tail_classes)

    eval_mode = str(getattr(args, "experimentD_eval_mode", "class_filtered"))
    eval_loaders = None
    if eval_mode == "class_filtered":
        eval_loaders = build_class_filtered_eval_loaders(global_trainer, tail_classes)
        if eval_loaders is None:
            print(
                "Experiment D warning: class_filtered eval loader unavailable; "
                "falling back to the full global test loader for counterfactual diagnostics."
            )
    elif eval_mode != "full":
        raise ValueError(f"Unknown --experimentD_eval_mode: {eval_mode}")

    acc_before = evaluate_state_per_class(
        global_trainer,
        global_before,
        tail_classes,
        eval_loaders,
    )
    acc_all = evaluate_state_per_class(
        global_trainer,
        theta_all,
        tail_classes,
        eval_loaders,
    )

    tail_specialists = tail_specialist_clients(args)
    tail_specialist_set = set(tail_specialists)
    per_class_rows = []

    for class_id in tail_classes:
        support = support_clients_for_class(client_class_counts, selected, class_id)
        support_state = build_support_actual_state(
            global_before,
            local_weights,
            selected,
            support,
            client_weights,
        )
        acc_support_actual = evaluate_state_per_class(
            global_trainer,
            support_state,
            [class_id],
            {class_id: eval_loaders[class_id]} if isinstance(eval_loaders, Mapping) else eval_loaders,
        )[class_id]

        acc_support_normalized = math.nan
        if bool(getattr(args, "experimentD_include_normalized", False)):
            support_normalized_state = build_support_normalized_state(
                global_before,
                local_weights,
                support,
                datanumber_client,
            )
            acc_support_normalized = evaluate_state_per_class(
                global_trainer,
                support_normalized_state,
                [class_id],
                {class_id: eval_loaders[class_id]} if isinstance(eval_loaders, Mapping) else eval_loaders,
            )[class_id]

        gain_support_actual = float(acc_support_actual) - float(acc_before[class_id])
        gain_support_normalized = (
            float(acc_support_normalized) - float(acc_before[class_id])
            if not math.isnan(float(acc_support_normalized))
            else math.nan
        )
        gain_all = float(acc_all[class_id]) - float(acc_before[class_id])
        offset_gap = gain_support_actual - gain_all

        class_positive_samples = float(global_counts[class_id].item())
        positive_samples_in_tail_specialists = (
            float(counts[tail_specialists, class_id].sum().item()) if tail_specialists else 0.0
        )
        positive_sample_specialist_ratio = (
            positive_samples_in_tail_specialists / class_positive_samples
            if class_positive_samples > 0
            else 0.0
        )
        support_fedavg_weight = sum(float(client_weights[idx]) for idx in support)
        num_support_tail_specialists = sum(1 for idx in support if idx in tail_specialist_set)

        per_class_rows.append(
            {
                "epoch_index": int(epoch),
                "communication_round": int(epoch) + 1,
                "seed": int(getattr(args, "seed", -1)),
                "partition": getattr(args, "partition", ""),
                "num_users": int(getattr(args, "num_users", len(datanumber_client))),
                "frac": float(getattr(args, "frac", 0.0)),
                "local_epochs": int(getattr(args, "local_epochs", -1)),
                "class_id": int(class_id),
                "class_group": "tail" if class_id in tail_set else ("head" if class_id in head_set else ""),
                "global_class_count": class_positive_samples,
                "num_support_clients": int(len(support)),
                "support_client_fraction": float(len(support) / max(len(selected), 1)),
                "support_fedavg_weight": support_fedavg_weight,
                "support_normalized_denominator": _support_normalized_denominator(support, datanumber_client),
                "class_positive_samples": class_positive_samples,
                "positive_samples_in_tail_specialists": positive_samples_in_tail_specialists,
                "positive_sample_specialist_ratio": positive_sample_specialist_ratio,
                "num_tail_specialists": int(len(tail_specialists)),
                "num_support_tail_specialists": int(num_support_tail_specialists),
                "acc_before": float(acc_before[class_id]),
                "acc_support_actual": float(acc_support_actual),
                "acc_support_normalized": acc_support_normalized,
                "acc_all": float(acc_all[class_id]),
                "gain_support_actual": gain_support_actual,
                "gain_support_normalized": gain_support_normalized,
                "gain_all": gain_all,
                "offset_gap": offset_gap,
                "support_actual_positive": _as_bool_text(gain_support_actual > 0.0),
                "support_normalized_positive": _as_bool_text(
                    (not math.isnan(gain_support_normalized)) and gain_support_normalized > 0.0
                ),
                "offset_observed": _as_bool_text(gain_support_actual > 0.0 and offset_gap > 0.0),
                "full_reversal": _as_bool_text(gain_support_actual > 0.0 and gain_all <= 0.0),
                "specialization_lambda": getattr(args, "specialization_lambda", ""),
                "intra_group_alpha": getattr(args, "intra_group_alpha", ""),
                "head_leakage_scale": getattr(args, "head_leakage_scale", ""),
                "beta": getattr(args, "beta", ""),
            }
        )

    summary = {
        "epoch_index": int(epoch),
        "communication_round": int(epoch) + 1,
        "seed": int(getattr(args, "seed", -1)),
        "partition": getattr(args, "partition", ""),
        "local_epochs": int(getattr(args, "local_epochs", -1)),
        "num_tail_classes": int(len(per_class_rows)),
        "mean_gain_support_actual": _mean([row["gain_support_actual"] for row in per_class_rows]),
        "median_gain_support_actual": _median([row["gain_support_actual"] for row in per_class_rows]),
        "mean_gain_support_normalized": _mean([row["gain_support_normalized"] for row in per_class_rows]),
        "median_gain_support_normalized": _median([row["gain_support_normalized"] for row in per_class_rows]),
        "mean_gain_all": _mean([row["gain_all"] for row in per_class_rows]),
        "median_gain_all": _median([row["gain_all"] for row in per_class_rows]),
        "mean_offset_gap": _mean([row["offset_gap"] for row in per_class_rows]),
        "median_offset_gap": _median([row["offset_gap"] for row in per_class_rows]),
        "support_actual_positive_rate": _rate([row["gain_support_actual"] > 0.0 for row in per_class_rows]),
        "support_normalized_positive_rate": _rate(
            [
                (not math.isnan(float(row["gain_support_normalized"]))) and row["gain_support_normalized"] > 0.0
                for row in per_class_rows
            ]
        ),
        "offset_observed_rate": _rate(
            [row["gain_support_actual"] > 0.0 and row["offset_gap"] > 0.0 for row in per_class_rows]
        ),
        "full_reversal_rate": _rate(
            [row["gain_support_actual"] > 0.0 and row["gain_all"] <= 0.0 for row in per_class_rows]
        ),
        "mean_support_fedavg_weight": _mean([row["support_fedavg_weight"] for row in per_class_rows]),
        "mean_num_support_clients": _mean([row["num_support_clients"] for row in per_class_rows]),
        "mean_positive_sample_specialist_ratio": _mean(
            [row["positive_sample_specialist_ratio"] for row in per_class_rows]
        ),
    }

    out_dir = experiment_d_output_dir(output_dir)
    _write_experiment_d_metadata(out_dir, args)
    _append_many_csv(out_dir / "experiment_d_per_class.csv", per_class_rows, PER_CLASS_FIELDS)
    _append_csv(out_dir / "experiment_d_round_summary.csv", summary, ROUND_SUMMARY_FIELDS)

    global_trainer.model.load_state_dict(theta_all, strict=True)
    max_diff, max_key = max_state_abs_diff(global_trainer.model.state_dict(), theta_all)
    if max_diff > 1e-6:
        raise RuntimeError(
            f"Experiment D failed to restore theta_all after diagnostics: key={max_key}, max_abs={max_diff}"
        )
    return summary


def timed_experiment_d_round(*args, **kwargs):
    start = time.time()
    summary = run_experiment_d_round(*args, **kwargs)
    return summary, time.time() - start
