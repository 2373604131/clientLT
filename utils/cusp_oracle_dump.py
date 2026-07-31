"""Training-side dump utilities for the minimal Oracle CUSP pilot.

The training entry point should only decide *when* to dump.  This module owns
the small amount of bookkeeping needed to save the round-10 trainable PromptFL
states and a deterministic train-feature cache.  It deliberately avoids global
test evaluation and does not require a second CLIP trainer.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Mapping, Sequence

import torch

from utils.oracle_cusp import (
    make_flat_spec,
    save_round_dump,
    sha256_file,
    sha256_json,
    trainable_float_keys,
    validate_train_feature_cache,
)


def trainable_state_dict_to_cpu(model) -> dict[str, torch.Tensor]:
    """Return only trainable parameters as detached CPU tensors."""
    state = model.state_dict()
    return {
        name: state[name].detach().cpu().clone()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }


def oracle_round_dir(output_dir: str | os.PathLike[str], communication_round: int) -> str:
    return os.path.join(
        str(output_dir),
        "oracle_cusp",
        f"round_{int(communication_round):03d}",
    )


def _class_splits_from_counts(global_class_counts, tail_class_ratio: float) -> dict[str, list[int]]:
    counts = torch.as_tensor(global_class_counts, dtype=torch.float32)
    sorted_classes = torch.argsort(counts, descending=True).tolist()
    tail_count = max(1, int(round(len(sorted_classes) * float(tail_class_ratio))))
    tail_count = min(tail_count, len(sorted_classes))
    return {
        "head": [int(x) for x in sorted_classes[:-tail_count]],
        "tail": [int(x) for x in sorted_classes[-tail_count:]],
        "all": [int(x) for x in sorted_classes],
    }


def _client_train_sources(local_trainer, num_users: int) -> dict[int, list[tuple[int, object]]]:
    train_x = list(getattr(local_trainer.dm.dataset, "train_x", []) or [])
    federated_train_x = getattr(local_trainer.dm.dataset, "federated_train_x", None)
    if not train_x or not federated_train_x:
        raise RuntimeError("Oracle CUSP requires dataset.train_x and dataset.federated_train_x")

    index_by_object = {id(item): idx for idx, item in enumerate(train_x)}
    sources: dict[int, list[tuple[int, object]]] = {}
    seen = set()

    for client_id in range(int(num_users)):
        rows = []
        for item in federated_train_x[client_id]:
            dataset_index = index_by_object.get(id(item))
            if dataset_index is None:
                raise RuntimeError(f"Client {client_id} has a train item not present in dataset.train_x")
            if dataset_index in seen:
                raise RuntimeError(f"Train sample {dataset_index} appears in more than one client")
            seen.add(dataset_index)
            rows.append((int(dataset_index), item))
        sources[int(client_id)] = sorted(rows, key=lambda pair: pair[0])

    if len(seen) != len(train_x):
        raise RuntimeError(f"Federated train split covers {len(seen)} samples, expected {len(train_x)}")
    return sources


def _cache_train_features(
    local_trainer,
    model,
    cfg,
    num_users: int,
    expected_counts: Sequence[int],
    output_path: str | os.PathLike[str],
    max_per_class: int = 0,
) -> None:
    """Cache normalized image features for train samples only."""
    from Dassl.dassl.data.data_manager import build_data_loader
    from Dassl.dassl.data.transforms import build_transform

    expected = torch.as_tensor(expected_counts, dtype=torch.long).cpu()
    saved_per_class = torch.zeros_like(expected)
    max_per_class = int(max_per_class)
    client_sources = _client_train_sources(local_trainer, int(num_users))
    eval_transform = build_transform(cfg, is_train=False)

    features = []
    labels = []
    identities = []
    was_training = model.training
    model.eval()

    try:
        with torch.no_grad():
            for client_id in range(int(num_users)):
                indexed_items = client_sources[int(client_id)]
                data_source = [item for _, item in indexed_items]
                dataset_indices = [idx for idx, _ in indexed_items]
                loader = build_data_loader(
                    cfg,
                    sampler_type=cfg.DATALOADER.TEST.SAMPLER,
                    data_source=data_source,
                    batch_size=cfg.DATALOADER.TEST.BATCH_SIZE,
                    tfm=eval_transform,
                    is_train=False,
                )

                cursor = 0
                for batch in loader:
                    inputs, batch_labels = local_trainer.parse_batch_train(batch)
                    image_features = model.image_encoder(inputs.type(model.dtype))
                    image_features = image_features / image_features.norm(dim=-1, keepdim=True).clamp_min(1e-12)

                    for row, label_value in enumerate(batch_labels.detach().cpu().tolist()):
                        label = int(label_value)
                        dataset_index = int(dataset_indices[cursor])
                        cursor += 1
                        if max_per_class and saved_per_class[label] >= max_per_class:
                            continue
                        features.append(image_features[row].detach().float().cpu())
                        labels.append(label)
                        identities.append([int(client_id), dataset_index])
                        saved_per_class[label] += 1

                if cursor != len(dataset_indices):
                    raise RuntimeError(f"Feature-cache loader length mismatch for client {client_id}")
    finally:
        model.train(was_training)

    if not features:
        raise RuntimeError("Oracle CUSP train feature cache is empty")
    if max_per_class == 0 and not torch.equal(saved_per_class, expected):
        raise RuntimeError(
            "Oracle CUSP train cache class counts differ from partition counts: "
            f"cached={saved_per_class.tolist()} expected={expected.tolist()}"
        )

    sample_identity = torch.tensor(identities, dtype=torch.long)
    order = torch.argsort(sample_identity[:, 0] * (len(local_trainer.dm.dataset.train_x) + 1) + sample_identity[:, 1])
    cache = {
        "schema_version": "cusp_round1_v1",
        "source": "train",
        "test_used_for_utility": False,
        "sample_identity_kind": "client_id,dataset_index",
        "features": torch.stack(features)[order],
        "labels": torch.tensor(labels, dtype=torch.long)[order],
        "sample_identity": sample_identity[order],
        "class_counts": saved_per_class,
        "feature_dtype": "torch.float32",
    }
    validate_train_feature_cache(cache, expected if max_per_class == 0 else None)
    torch.save(cache, output_path)


def save_oracle_round_dump(
    *,
    output_dir,
    args,
    cfg,
    epoch: int,
    global_before: Mapping[str, torch.Tensor],
    global_after: Mapping[str, torch.Tensor],
    local_weights,
    selected_clients,
    datanumber_client,
    client_class_counts,
    global_class_counts,
    trainer,
) -> str:
    """Save the final-round Oracle CUSP dump and return its directory."""
    communication_round = int(epoch) + 1
    selected = sorted(int(client_id) for client_id in selected_clients)
    if len(selected) != int(args.num_users):
        raise RuntimeError(f"Oracle CUSP requires all {args.num_users} clients, got {len(selected)}")

    keys = trainable_float_keys(trainer.model)
    spec = make_flat_spec(global_before, keys)
    sample_total = sum(float(datanumber_client[client_id]) for client_id in selected)
    fedavg_weights = [float(datanumber_client[client_id]) / sample_total for client_id in selected]
    class_counts = torch.stack([torch.as_tensor(client_class_counts[client_id]).cpu() for client_id in selected])
    global_counts = [int(x) for x in torch.as_tensor(global_class_counts).tolist()]
    splits = _class_splits_from_counts(global_class_counts, args.tail_class_ratio)

    run_dir = oracle_round_dir(output_dir, communication_round)
    if os.path.exists(run_dir):
        raise RuntimeError(f"Oracle CUSP dump already exists: {run_dir}")
    os.makedirs(run_dir, exist_ok=False)

    payload = {
        "trainable_keys": keys,
        "flatten_spec": spec.as_dict(),
        "global_before_trainable": {key: global_before[key].detach().cpu().clone() for key in keys},
        "local_trainable_states": [
            {key: local_weights[client_id][key].detach().cpu().clone() for key in keys}
            for client_id in selected
        ],
        "global_after_fedavg_trainable": {key: global_after[key].detach().cpu().clone() for key in keys},
        "selected_client_ids": selected,
        "fedavg_weights": fedavg_weights,
        "client_sample_counts": [int(datanumber_client[client_id]) for client_id in selected],
        "client_class_counts": class_counts,
        "num_classes": int(len(global_counts)),
    }

    trainable_parameters = [
        {"key": key, "shape": list(shape), "dtype": dtype, "numel": int(end - start)}
        for key, shape, dtype, (start, end) in zip(spec.keys, spec.shapes, spec.dtypes, spec.offsets)
    ]
    split_fingerprint_path = Path(output_dir) / "client_split_fingerprint.json"
    split_fingerprint = (
        json.loads(split_fingerprint_path.read_text(encoding="utf-8"))
        if split_fingerprint_path.exists()
        else {}
    )

    metadata = {
        "dataset": args.dataset,
        "trainer": args.trainer,
        "model": args.model,
        "seed": int(args.seed),
        "split_seed": int(args.split_seed),
        "client_schedule_seed": int(args.client_schedule_seed),
        "client_schedule_file": getattr(args, "client_schedule_file", ""),
        "client_schedule_sha256": sha256_file(args.client_schedule_file) if getattr(args, "client_schedule_file", "") else None,
        "communication_round": communication_round,
        "internal_epoch": int(epoch),
        "num_users": int(args.num_users),
        "frac": float(args.frac),
        "local_epochs": int(args.local_epochs),
        "partition": args.partition,
        "specialization_lambda": float(args.specialization_lambda),
        "intra_group_alpha": float(args.intra_group_alpha),
        "head_leakage_scale": float(args.head_leakage_scale),
        "head_client_ratio": float(args.head_client_ratio),
        "tail_client_ratio": float(args.tail_client_ratio),
        "head_class_ratio": float(args.head_class_ratio),
        "tail_class_ratio": float(args.tail_class_ratio),
        "global_class_counts": global_counts,
        "head_class_ids": splits["head"],
        "tail_class_ids": splits["tail"],
        "utility_data_source": "train",
        "test_used_for_utility": False,
        "dataset_config_file": args.dataset_config_file,
        "trainer_config_file": args.config_file,
        "resolved_args": vars(args),
        "resolved_config": cfg.dump(),
        "trainable_parameters": trainable_parameters,
        "client_ids": selected,
        "client_sample_counts": payload["client_sample_counts"],
        "fedavg_weights": fedavg_weights,
        "fedavg_weight_sum": float(sum(fedavg_weights)),
        "global_lt_fingerprint": sha256_json({"global_class_counts": global_counts}),
        "client_split_fingerprint": split_fingerprint,
    }

    if bool(args.oracle_cusp_cache_train_features):
        cache_path = os.path.join(run_dir, "train_feature_cache.pt")
        _cache_train_features(
            local_trainer=trainer,
            model=trainer.model,
            cfg=cfg,
            num_users=args.num_users,
            expected_counts=global_class_counts,
            output_path=cache_path,
            max_per_class=args.oracle_cusp_max_train_samples_per_class,
        )
        metadata["train_feature_cache_sha256"] = sha256_file(cache_path)
    else:
        metadata["train_feature_cache_sha256"] = None

    save_round_dump(run_dir, payload, metadata)
    print(f"Saved Oracle CUSP dump: {run_dir}")
    return run_dir
