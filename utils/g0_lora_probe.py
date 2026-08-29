"""Local ClipLoRA capacity probe used before the D1 aggregation diagnosis.

G0 deliberately performs no server aggregation.  Every selected client starts
from the same incoming global state, trains locally once, and is evaluated on
deterministic views of its own train and federated-test split.  This separates
"can the carrier learn?" from D1's question of whether aggregation destroys a
useful client update.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Callable, Mapping, Sequence

import torch
import torch.nn.functional as F


G0_PER_CLIENT_FIELDS = [
    "config_id",
    "client_id",
    "client_role",
    "client_num_samples",
    "heldout_num_samples",
    "present_class_count",
    "tail_present_class_count",
    "train_loss_before",
    "train_loss_after",
    "train_loss_relative_drop",
    "heldout_acc_before",
    "heldout_acc_after",
    "heldout_acc_gain",
    "present_margin_before",
    "present_margin_after",
    "present_margin_gain",
    "tail_margin_before",
    "tail_margin_after",
    "tail_margin_gain",
    "prediction_flip_rate",
    "mean_abs_logit_change",
    "kl_before_after",
    "lora_parameter_delta_norm",
    "effective_ba_delta_norm",
    "optimizer_steps",
    "scheduler_steps",
    "all_finite",
]

G0_LORA_CONFIGS = {
    "old_r2": {
        "position": "top3",
        "rank": 2,
        "alpha": 1,
        "params": ["q", "v"],
    },
    "candidate_r4": {
        "position": "up",
        "rank": 4,
        "alpha": 2,
        "params": ["q", "k", "v"],
    },
}


def validate_g0_protocol(args) -> None:
    """Reject runs that would be mislabeled as the frozen G0 protocol."""

    if int(args.seed) != 42 or int(args.split_seed) != 42:
        raise ValueError("G0 is frozen to seed=split_seed=42")
    if str(args.partition) != "client-longtail":
        raise ValueError("G0 is frozen to the client-longtail partition")
    if int(args.num_users) != 30:
        raise ValueError("G0 is frozen to 30 clients")
    if abs(float(args.tail_client_ratio) - 0.1) > 1e-12:
        raise ValueError("G0 is frozen to exactly three tail specialists")
    if int(args.local_epochs) != 3:
        raise ValueError("G0 is frozen to three local epochs")

    config_id = str(args.g0_probe_config_id)
    if config_id not in G0_LORA_CONFIGS:
        raise ValueError(f"Unknown frozen G0 config id: {config_id}")
    observed = {
        "position": str(args.cliplora_position),
        "rank": int(args.cliplora_rank),
        "alpha": int(args.cliplora_alpha),
        "params": list(args.cliplora_params),
    }
    if observed != G0_LORA_CONFIGS[config_id]:
        raise ValueError(
            f"G0 config {config_id} was altered: expected "
            f"{G0_LORA_CONFIGS[config_id]}, got {observed}"
        )


def _finite_mean(values: Sequence[float]) -> float:
    items = [float(value) for value in values if math.isfinite(float(value))]
    return float(sum(items) / len(items)) if items else math.nan


def _finite_median(values: Sequence[float]) -> float:
    items = sorted(float(value) for value in values if math.isfinite(float(value)))
    if not items:
        return math.nan
    middle = len(items) // 2
    if len(items) % 2:
        return items[middle]
    return (items[middle - 1] + items[middle]) / 2.0


def select_probe_clients(
    client_class_counts: Mapping[int, torch.Tensor],
    *,
    num_users: int,
    tail_client_ratio: float,
) -> tuple[list[int], list[int]]:
    """Use every tail specialist and sample-count-matched head clients.

    Matching is deterministic and fixed before any model evaluation.  It
    cannot cherry-pick clients by utility.
    """

    tail_count = int(round(int(num_users) * float(tail_client_ratio)))
    tail_count = min(max(tail_count, 1), int(num_users) - 1)
    head_ids = list(range(0, int(num_users) - tail_count))
    tail_ids = list(range(int(num_users) - tail_count, int(num_users)))
    totals = {
        int(client_id): int(torch.as_tensor(client_class_counts[int(client_id)]).sum().item())
        for client_id in range(int(num_users))
    }
    available = set(head_ids)
    matched = []
    for tail_id in tail_ids:
        if not available:
            raise RuntimeError("G0 cannot match every tail client to a distinct head client")
        chosen = min(
            available,
            key=lambda head_id: (abs(totals[head_id] - totals[tail_id]), head_id),
        )
        matched.append(int(chosen))
        available.remove(chosen)
    return matched, tail_ids


def _build_eval_loader(cfg, trainer, data_source):
    from Dassl.dassl.data.data_manager import build_data_loader
    from Dassl.dassl.data.transforms import build_transform

    return build_data_loader(
        cfg,
        sampler_type="SequentialSampler",
        data_source=list(data_source),
        batch_size=cfg.DATALOADER.TEST.BATCH_SIZE,
        tfm=build_transform(cfg, is_train=False),
        is_train=False,
        dataset_wrapper=None,
        class_names=trainer.dm.dataset.classnames,
        drop_last=False,
    )


@torch.no_grad()
def _collect_logits(trainer, loader) -> tuple[torch.Tensor, torch.Tensor]:
    was_training = trainer.model.training
    logits, labels = [], []
    try:
        trainer.model.eval()
        for batch in loader:
            images, target = trainer.parse_batch_test(batch)
            output = trainer.model_inference(images)
            if isinstance(output, (tuple, list)):
                output = output[0]
            logits.append(output.detach().float().cpu())
            labels.append(target.detach().long().cpu())
    finally:
        trainer.model.train(was_training)
    if not logits:
        raise RuntimeError("G0 encountered an empty evaluation loader")
    return torch.cat(logits, dim=0), torch.cat(labels, dim=0)


def _mask_for_classes(labels: torch.Tensor, class_ids: Sequence[int] | None) -> torch.Tensor:
    if class_ids is None:
        return torch.ones_like(labels, dtype=torch.bool)
    mask = torch.zeros_like(labels, dtype=torch.bool)
    for class_id in class_ids:
        mask |= labels == int(class_id)
    return mask


def _metrics(logits: torch.Tensor, labels: torch.Tensor, class_ids=None) -> dict[str, float]:
    mask = _mask_for_classes(labels, class_ids)
    if not bool(mask.any()):
        return {"count": 0, "loss": math.nan, "accuracy": math.nan, "margin": math.nan}
    selected_logits = logits[mask]
    selected_labels = labels[mask]
    true_logits = selected_logits.gather(1, selected_labels[:, None]).squeeze(1)
    other = selected_logits.clone()
    other.scatter_(1, selected_labels[:, None], -torch.inf)
    margins = true_logits - other.max(dim=1).values
    return {
        "count": int(selected_labels.numel()),
        "loss": float(F.cross_entropy(selected_logits, selected_labels).item()),
        "accuracy": float((selected_logits.argmax(dim=1) == selected_labels).float().mean().item() * 100.0),
        "margin": float(margins.mean().item()),
    }


def _state_delta_norm(before: Mapping[str, torch.Tensor], after: Mapping[str, torch.Tensor]) -> float:
    squared = 0.0
    for key, before_value in before.items():
        if "lora_" not in key or key not in after:
            continue
        delta = after[key].detach().float().cpu() - before_value.detach().float().cpu()
        squared += float(torch.sum(delta * delta).item())
    return math.sqrt(max(squared, 0.0))


def effective_ba_delta_norm(
    before: Mapping[str, torch.Tensor],
    after: Mapping[str, torch.Tensor],
    *,
    scaling: float,
) -> float:
    """Norm of the executable low-rank weight change, not raw A/B drift."""

    squared = 0.0
    pair_count = 0
    for a_key, a_before in before.items():
        if not a_key.endswith("_lora_A"):
            continue
        b_key = a_key[:-1] + "B"
        if b_key not in before or a_key not in after or b_key not in after:
            continue
        ba_before = before[b_key].detach().float().cpu() @ a_before.detach().float().cpu()
        ba_after = after[b_key].detach().float().cpu() @ after[a_key].detach().float().cpu()
        delta = float(scaling) * (ba_after - ba_before)
        squared += float(torch.sum(delta * delta).item())
        pair_count += 1
    if pair_count == 0:
        raise RuntimeError("G0 found no paired *_lora_A/*_lora_B parameters")
    return math.sqrt(max(squared, 0.0))


def _append_csv(path: Path, rows: Sequence[Mapping]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=G0_PER_CLIENT_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in G0_PER_CLIENT_FIELDS})


def _write_json(path: Path, payload: Mapping) -> None:
    def sanitize(value):
        if isinstance(value, Mapping):
            return {key: sanitize(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [sanitize(item) for item in value]
        if isinstance(value, float) and not math.isfinite(value):
            return None
        return value

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(sanitize(payload), indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )


def summarize_probe_rows(rows: Sequence[Mapping], config: Mapping) -> dict:
    tail_rows = [row for row in rows if row["client_role"] == "tail_specialist"]
    positive_tail = [
        row for row in tail_rows
        if math.isfinite(float(row["tail_margin_gain"])) and float(row["tail_margin_gain"]) > 0.0
    ]
    return {
        "schema_version": "g0_local_capacity_v1",
        "config": dict(config),
        "client_count": len(rows),
        "tail_client_count": len(tail_rows),
        "all_finite": all(bool(row["all_finite"]) for row in rows),
        "mean_train_loss_relative_drop": _finite_mean(
            [row["train_loss_relative_drop"] for row in rows]
        ),
        "median_heldout_acc_gain": _finite_median([row["heldout_acc_gain"] for row in rows]),
        "median_tail_margin_gain": _finite_median([row["tail_margin_gain"] for row in tail_rows]),
        "positive_tail_client_count": len(positive_tail),
        "positive_tail_client_rate": len(positive_tail) / max(len(tail_rows), 1),
        "mean_prediction_flip_rate": _finite_mean([row["prediction_flip_rate"] for row in rows]),
        "mean_abs_logit_change": _finite_mean([row["mean_abs_logit_change"] for row in rows]),
        "mean_lora_parameter_delta_norm": _finite_mean(
            [row["lora_parameter_delta_norm"] for row in rows]
        ),
        "mean_effective_ba_delta_norm": _finite_mean(
            [row["effective_ba_delta_norm"] for row in rows]
        ),
        "rows": [dict(row) for row in rows],
    }


def run_g0_local_probe(
    *,
    output_dir: str | Path,
    args,
    cfg,
    trainer,
    global_weights: Mapping[str, torch.Tensor],
    client_class_counts: Mapping[int, torch.Tensor],
    global_class_counts: torch.Tensor,
    train_client: Callable[[int], Mapping[str, int | float]],
) -> dict:
    output_dir = Path(output_dir) / "g0_probe"
    output_dir.mkdir(parents=True, exist_ok=True)
    head_clients, tail_clients = select_probe_clients(
        client_class_counts,
        num_users=int(args.num_users),
        tail_client_ratio=float(args.tail_client_ratio),
    )
    counts_for_split = torch.as_tensor(global_class_counts, dtype=torch.float32)
    tail_count = max(1, int(round(len(counts_for_split) * float(args.tail_class_ratio))))
    # Match federated_main.get_lt_class_splits_from_counts exactly.  Realized
    # CIFAR-100-LT counts can tie at the 79/80 boundary; larger class ids are
    # the intended tail classes under the monotone LT generator.
    tail_classes = sorted(
        range(len(counts_for_split)),
        key=lambda class_id: (float(counts_for_split[class_id]), -class_id),
    )[:tail_count]
    selected = head_clients + tail_clients
    scaling = float(args.cliplora_alpha) / math.sqrt(float(args.cliplora_rank))
    config = {
        "config_id": str(args.g0_probe_config_id),
        "seed": int(args.seed),
        "split_seed": int(args.split_seed),
        "encoder": str(args.encoder),
        "position": str(args.cliplora_position),
        "rank": int(args.cliplora_rank),
        "alpha": int(args.cliplora_alpha),
        "params": list(args.cliplora_params),
        "dropout": float(args.cliplora_dropout_rate),
        "precision": str(args.cliplora_precision),
        "local_epochs": int(args.local_epochs),
        "lr": float(args.lr),
        "implementation_scaling_alpha_over_sqrt_rank": scaling,
        "selected_head_clients": head_clients,
        "selected_tail_clients": tail_clients,
        "tail_class_ids": [int(value) for value in tail_classes],
    }

    original = {key: value.detach().cpu().clone() for key, value in global_weights.items()}
    rows = []
    try:
        for client_id in selected:
            print(f"G0 probe client {client_id}: load common incoming global", flush=True)
            trainer.model.load_state_dict(global_weights, strict=True)
            train_source = trainer.dm.dataset.federated_train_x[int(client_id)]
            train_loader = _build_eval_loader(cfg, trainer, train_source)
            test_loader = trainer.fed_test_loader_x_dict[int(client_id)]
            train_logits_before, train_labels = _collect_logits(trainer, train_loader)
            test_logits_before, test_labels = _collect_logits(trainer, test_loader)
            before_state = {
                key: value.detach().cpu().clone() for key, value in trainer.model.state_dict().items()
            }
            lifecycle = dict(train_client(int(client_id)))
            after_state = {
                key: value.detach().cpu().clone() for key, value in trainer.model.state_dict().items()
            }
            train_logits_after, train_labels_after = _collect_logits(trainer, train_loader)
            test_logits_after, test_labels_after = _collect_logits(trainer, test_loader)
            if not torch.equal(train_labels, train_labels_after) or not torch.equal(test_labels, test_labels_after):
                raise RuntimeError("G0 deterministic evaluation loader changed label order")

            counts = torch.as_tensor(client_class_counts[int(client_id)])
            present = torch.nonzero(counts > 0, as_tuple=False).reshape(-1).tolist()
            tail_present = sorted(set(int(value) for value in present).intersection(tail_classes))
            train_before = _metrics(train_logits_before, train_labels)
            train_after = _metrics(train_logits_after, train_labels)
            heldout_before = _metrics(test_logits_before, test_labels)
            heldout_after = _metrics(test_logits_after, test_labels)
            present_before = _metrics(test_logits_before, test_labels, present)
            present_after = _metrics(test_logits_after, test_labels, present)
            tail_before = _metrics(test_logits_before, test_labels, tail_present)
            tail_after = _metrics(test_logits_after, test_labels, tail_present)

            before_probs = torch.softmax(test_logits_before, dim=1)
            kl = F.kl_div(
                torch.log_softmax(test_logits_after, dim=1),
                before_probs,
                reduction="batchmean",
            )
            prediction_flip_rate = float(
                (test_logits_before.argmax(1) != test_logits_after.argmax(1))
                .float()
                .mean()
                .item()
            )
            mean_abs_logit_change = float(
                (test_logits_after - test_logits_before).abs().mean().item()
            )
            lora_parameter_delta_norm = _state_delta_norm(before_state, after_state)
            executable_delta_norm = effective_ba_delta_norm(
                before_state, after_state, scaling=scaling
            )
            values = [
                train_before["loss"], train_after["loss"], heldout_before["accuracy"],
                heldout_after["accuracy"], present_before["margin"], present_after["margin"],
                float(kl.item()), prediction_flip_rate, mean_abs_logit_change,
                lora_parameter_delta_norm, executable_delta_norm,
            ]
            if client_id in tail_clients:
                values.extend([tail_before["margin"], tail_after["margin"]])
            row = {
                "config_id": str(args.g0_probe_config_id),
                "client_id": int(client_id),
                "client_role": "tail_specialist" if client_id in tail_clients else "head_client",
                "client_num_samples": int(counts.sum().item()),
                "heldout_num_samples": int(test_labels.numel()),
                "present_class_count": len(present),
                "tail_present_class_count": len(tail_present),
                "train_loss_before": train_before["loss"],
                "train_loss_after": train_after["loss"],
                "train_loss_relative_drop": (
                    (train_before["loss"] - train_after["loss"]) / max(train_before["loss"], 1e-12)
                ),
                "heldout_acc_before": heldout_before["accuracy"],
                "heldout_acc_after": heldout_after["accuracy"],
                "heldout_acc_gain": heldout_after["accuracy"] - heldout_before["accuracy"],
                "present_margin_before": present_before["margin"],
                "present_margin_after": present_after["margin"],
                "present_margin_gain": present_after["margin"] - present_before["margin"],
                "tail_margin_before": tail_before["margin"],
                "tail_margin_after": tail_after["margin"],
                "tail_margin_gain": tail_after["margin"] - tail_before["margin"],
                "prediction_flip_rate": prediction_flip_rate,
                "mean_abs_logit_change": mean_abs_logit_change,
                "kl_before_after": float(kl.item()),
                "lora_parameter_delta_norm": lora_parameter_delta_norm,
                "effective_ba_delta_norm": executable_delta_norm,
                "optimizer_steps": int(lifecycle.get("optimizer_steps", -1)),
                "scheduler_steps": int(lifecycle.get("scheduler_steps", -1)),
                "all_finite": bool(all(math.isfinite(float(value)) for value in values)),
            }
            rows.append(row)
            print(
                "G0 result: "
                f"config={row['config_id']} client={client_id} role={row['client_role']} "
                f"loss_drop={row['train_loss_relative_drop']:.6f} "
                f"heldout_gain={row['heldout_acc_gain']:.4f} "
                f"tail_margin_gain={row['tail_margin_gain']:.6f} "
                f"flip={row['prediction_flip_rate']:.6f} "
                f"BA_delta={row['effective_ba_delta_norm']:.6g}",
                flush=True,
            )
    finally:
        trainer.model.load_state_dict(original, strict=True)

    summary = summarize_probe_rows(rows, config)
    _append_csv(output_dir / "g0_per_client.csv", rows)
    _write_json(output_dir / "g0_config_summary.json", summary)
    _write_json(output_dir / "g0_manifest.json", config)
    print(f"G0 config summary: {output_dir / 'g0_config_summary.json'}", flush=True)
    return summary
