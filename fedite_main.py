import argparse
import csv
import os
import random
import time
from typing import Dict, List

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from clip import clip
from trainers.fedite.aggregation import aggregate_fedite, aggregate_round_stats
from trainers.fedite.model import FedITEModel
from trainers.fedite.observer import EvidenceTopologyObserver
from trainers.fedite.trainer import FedITEClientTrainer
from trainers.fedite.utils import (
    append_jsonl,
    as_cpu_state_dict,
    ensure_dir,
    parameter_manifest,
    set_seed,
    split_head_medium_tail,
    str2bool,
    write_json,
)
from utils.datasplit import partition_data_LT


class FedITEDatumDataset(Dataset):
    def __init__(self, data_source, indices=None, transform=None):
        self.data_source = data_source
        self.indices = list(range(len(data_source))) if indices is None else [int(i) for i in indices]
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        item = self.data_source[self.indices[index]]
        if isinstance(item, dict):
            if "label" not in item:
                raise KeyError("FedITE sample dict is missing 'label'")
            label = int(item["label"])
            if "data" in item:
                image = item["data"]
            elif "impath" in item:
                image = Image.open(item["impath"]).convert("RGB")
            else:
                raise KeyError("FedITE sample dict has neither 'data' nor 'impath'")
        elif hasattr(item, "data") and hasattr(item, "label"):
            image = item.data
            label = int(item.label)
        elif hasattr(item, "impath") and hasattr(item, "label"):
            image = Image.open(item.impath).convert("RGB")
            label = int(item.label)
        else:
            image, label = item[0], int(item[1])
        if isinstance(image, torch.Tensor):
            if image.ndim == 3:
                image = image.permute(1, 2, 0).cpu().numpy()
            else:
                image = image.cpu().numpy()
        if not isinstance(image, Image.Image):
            image = Image.fromarray(np.asarray(image).astype(np.uint8))
        if self.transform is not None:
            image = self.transform(image)
        return image, label


def load_clip_to_cpu(backbone_name):
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)
    try:
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None
    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")
    design_details = {
        "trainer": "FedITE",
        "vision_depth": 0,
        "language_depth": 0,
        "vision_ctx": 0,
        "language_ctx": 0,
    }
    model = clip.build_model(state_dict or model.state_dict(), design_details)
    return model


def build_fedite_data(args, preprocess):
    if args.dataset != "cifar100_lt":
        raise ValueError("FedITE standalone main currently supports --dataset cifar100_lt")
    data_dir = os.path.join(args.data_root, "cifar-100")
    partition_name = "homo" if args.partition == "iid" else args.partition
    outputs = partition_data_LT(
        "cifar100_LT",
        data_dir,
        partition_name,
        args.num_users,
        imb_factor=args.imb_factor,
        imb_type=args.imb_type,
        beta=args.beta,
        logdir=None,
        head_client_ratio=args.head_client_ratio,
        tail_client_ratio=args.tail_client_ratio,
        head_class_ratio=args.head_class_ratio,
        tail_class_ratio=args.tail_class_ratio,
        specialization_lambda=args.specialization_lambda,
        intra_group_alpha=args.intra_group_alpha,
        head_leakage_scale=args.head_leakage_scale,
    )
    (
        data_train,
        data_test,
        lab2cname,
        classnames,
        net_train,
        _net_test,
        train_counts,
        _test_counts,
        y_train,
    ) = outputs
    client_loaders = []
    for client_id in range(args.num_users):
        ds = FedITEDatumDataset(data_train, net_train[client_id], transform=preprocess)
        client_loaders.append(DataLoader(ds, batch_size=args.local_bs, shuffle=True, num_workers=args.num_workers, drop_last=False))
    test_ds = FedITEDatumDataset(data_test, None, transform=preprocess)
    test_loader = DataLoader(test_ds, batch_size=args.test_bs, shuffle=False, num_workers=args.num_workers, drop_last=False)
    train_class_counts = torch.bincount(torch.as_tensor(y_train, dtype=torch.long), minlength=len(classnames)).float()
    return client_loaders, test_loader, classnames, train_class_counts, train_counts


def append_csv(path, rows, fieldnames):
    exists = os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})
        handle.flush()


@torch.no_grad()
def evaluate(model, test_loader, class_state, splits, protected_classes, args, device, tail_active=True):
    model.eval()
    model.set_class_evidence_state(class_state)
    model.set_protected_classes(protected_classes)
    class_total = torch.zeros(model.num_classes)
    class_final = torch.zeros(model.num_classes)
    class_shared = torch.zeros(model.num_classes)
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device).long()
        prefix_tokens = model.encode_visual_prefix(images) if bool(getattr(args, "fedite_prefix_cache", True)) else None
        if not tail_active:
            out_shared = model.forward_shared(images, prefix_tokens=prefix_tokens, compute_base=False)
            logits_shared = out_shared["logits_shared"]
            logits_final = logits_shared
        else:
            out = model.forward_inference(
                images,
                class_state=class_state.to(device),
                inference_topk=args.fedite_inference_topk,
                inference_temperature=args.fedite_inference_temperature,
                return_diagnostics=False,
                prefix_tokens=prefix_tokens,
                return_dict=True,
            )
            logits_shared = out["logits_shared"]
            logits_final = out["logits_final"]
        pred_final = logits_final.argmax(dim=1)
        pred_shared = logits_shared.argmax(dim=1)
        for cls in labels.detach().cpu().unique().tolist():
            mask = labels.detach().cpu() == int(cls)
            class_total[int(cls)] += mask.sum()
            class_final[int(cls)] += (pred_final.detach().cpu()[mask] == int(cls)).sum()
            class_shared[int(cls)] += (pred_shared.detach().cpu()[mask] == int(cls)).sum()
    class_acc_final = class_final / class_total.clamp_min(1.0)
    class_acc_shared = class_shared / class_total.clamp_min(1.0)

    def split_acc(names):
        ids = splits.get(names, [])
        if not ids:
            return 0.0
        total = class_total[ids].sum().clamp_min(1.0)
        return float(class_final[ids].sum().item() / total.item())

    def split_acc_shared(names):
        ids = splits.get(names, [])
        if not ids:
            return 0.0
        total = class_total[ids].sum().clamp_min(1.0)
        return float(class_shared[ids].sum().item() / total.item())

    protected = sorted(set(int(c) for c in protected_classes))
    nonprotected = [c for c in range(model.num_classes) if c not in protected]
    metrics = {
        "overall_acc": float(class_final.sum().item() / class_total.sum().clamp_min(1.0).item()),
        "shared_overall_acc": float(class_shared.sum().item() / class_total.sum().clamp_min(1.0).item()),
        "non_tail_acc": split_acc("non_tail"),
        "tail_acc": split_acc("tail"),
        "shared_tail_acc": split_acc_shared("tail"),
        "macro_acc": float(class_acc_final.mean().item()),
        "protected_acc": float(class_final[protected].sum().item() / class_total[protected].sum().clamp_min(1.0).item()) if protected else 0.0,
        "nonprotected_acc": float(class_final[nonprotected].sum().item() / class_total[nonprotected].sum().clamp_min(1.0).item()) if nonprotected else 0.0,
        "class_acc_final": class_acc_final,
        "class_acc_shared": class_acc_shared,
        "class_total": class_total,
    }
    metrics["final_minus_shared_tail"] = metrics["tail_acc"] - metrics["shared_tail_acc"]
    return metrics


def per_class_metric_rows(round_idx, metrics, observer, train_class_counts):
    if not metrics:
        return []
    protected = set(observer.protected_classes)
    rows = []
    final = metrics["class_acc_final"]
    shared = metrics["class_acc_shared"]
    total = metrics["class_total"]
    for class_id in range(int(final.numel())):
        rows.append({
            "round": int(round_idx),
            "class_id": int(class_id),
            "global_train_count": float(train_class_counts[class_id].item()),
            "test_support": float(total[class_id].item()),
            "final_accuracy": float(final[class_id].item()),
            "shared_accuracy": float(shared[class_id].item()),
            "final_minus_shared": float((final[class_id] - shared[class_id]).item()),
            "D": float(observer.D[class_id].item()),
            "R": float(observer.R[class_id].item()),
            "S": float(observer.S[class_id].item()),
            "Rarity": float(observer.Rarity[class_id].item()),
            "is_protected": int(class_id in protected),
            "EMA_M": float(observer.EMA_M[class_id].item()),
            "EMA_N_eff": float(observer.EMA_N_eff[class_id].item()),
            "EMA_U": float(observer.EMA_U[class_id].item()),
            "Gap": float(observer.Gap[class_id].item()),
        })
    return rows


def format_class_ids(class_ids):
    return ";".join(str(int(c)) for c in class_ids)


def selection_diagnostics(round_idx, observer, train_class_counts, splits):
    protected = sorted(set(int(c) for c in observer.protected_classes))
    tail = sorted(set(int(c) for c in splits.get("tail", [])))
    tail_set = set(tail)
    protected_tail = sorted(tail_set.intersection(protected))
    missed_tail = sorted(tail_set.difference(protected))
    budget = max(len(protected), len(tail), 1)
    top_d = torch.topk(observer.D.detach().cpu(), k=min(budget, observer.num_classes)).indices.tolist()
    top_s = torch.topk(observer.S.detach().cpu(), k=min(budget, observer.num_classes)).indices.tolist()
    top_rarity = torch.topk(observer.Rarity.detach().cpu(), k=min(budget, observer.num_classes)).indices.tolist()
    counts = torch.as_tensor(train_class_counts, dtype=torch.float32)
    protected_avg = float(counts[protected].mean().item()) if protected else 0.0
    return {
        "round": int(round_idx),
        "protected_tail_overlap": int(len(protected_tail)),
        "protected_tail_overlap_classes": protected_tail,
        "protected_avg_global_count": protected_avg,
        "topD_classes": [int(c) for c in top_d],
        "topS_classes": [int(c) for c in top_s],
        "topRarity_classes": [int(c) for c in top_rarity],
        "missed_tail_classes": missed_tail,
    }


def make_output_dir(args):
    run_name = args.run_name or f"fedite_{time.strftime('%Y%m%d_%H%M%S')}"
    return ensure_dir(os.path.join(args.output_dir, "fedite", args.dataset, args.partition, run_name))


def build_arg_parser():
    parser = argparse.ArgumentParser(description="FedITE standalone federated main")
    parser.add_argument("--method", type=str, default="fedite")
    parser.add_argument("--dataset", type=str, default="cifar100_lt")
    parser.add_argument("--partition", type=str, default="client-longtail", choices=["client-longtail", "noniid-labeldir", "iid"])
    parser.add_argument("--data_root", type=str, default="DATA")
    parser.add_argument("--output_dir", type=str, default="output")
    parser.add_argument("--run_name", type=str, default="")
    parser.add_argument("--num_users", type=int, default=20)
    parser.add_argument("--frac", type=float, default=0.4)
    parser.add_argument("--rounds", type=int, default=100)
    parser.add_argument("--local_ep", type=int, default=1)
    parser.add_argument("--local_bs", type=int, default=32)
    parser.add_argument("--test_bs", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--beta", type=float, default=0.05)
    parser.add_argument("--imb_factor", type=float, default=0.01)
    parser.add_argument("--imb_type", type=str, default="exp")
    parser.add_argument("--head_client_ratio", type=float, default=0.9)
    parser.add_argument("--tail_client_ratio", type=float, default=0.1)
    parser.add_argument("--head_class_ratio", type=float, default=0.8)
    parser.add_argument("--tail_class_ratio", type=float, default=0.2)
    parser.add_argument(
        "--specialization_lambda",
        type=float,
        default=1.0,
        help="Client-LT lambda. Controls how strongly tail classes concentrate on tail clients.",
    )
    parser.add_argument(
        "--intra_group_alpha",
        type=float,
        default=0.1,
        help="Client-LT Dirichlet concentration inside client groups.",
    )
    parser.add_argument(
        "--head_leakage_scale",
        type=float,
        default=0.0,
        help="Client-LT scale for non-tail samples entering tail clients.",
    )

    parser.add_argument("--clip_backbone", type=str, default="ViT-B/16")
    parser.add_argument("--fedite_adapter_layers", type=str, default="10,11")
    parser.add_argument("--fedite_adapter_bottleneck", type=int, default=64)
    parser.add_argument("--fedite_num_tail_basis", type=int, default=2)
    parser.add_argument("--fedite_class_basis_scale", type=float, default=1.0)
    parser.add_argument("--fedite_alpha_shared", type=float, default=1.0)
    parser.add_argument("--fedite_alpha_tail", type=float, default=0.2)
    parser.add_argument("--fedite_adapter_dropout", type=float, default=0.0)
    parser.add_argument("--fedite_basis_dropout", type=float, default=0.0)
    parser.add_argument("--fedite_token_selective", type=str2bool, default=False)
    parser.add_argument("--fedite_train_prompt", type=str2bool, default=False)
    parser.add_argument("--fedite_prompt_ctx", type=str, default="a photo of a")
    parser.add_argument("--fedite_prompt_n_ctx", type=int, default=4)
    parser.add_argument("--fedite_shared_only", type=str2bool, default=False)

    parser.add_argument("--fedite_protected_ratio", type=float, default=0.2)
    parser.add_argument("--fedite_exploration_ratio", type=float, default=0.0)
    parser.add_argument("--fedite_warmup_rounds", type=int, default=5)
    parser.add_argument("--fedite_warmup_mode", type=str, default="round_robin", choices=["round_robin", "random", "none"])
    parser.add_argument("--fedite_reliability_min", type=float, default=0.05)
    parser.add_argument("--fedite_observer_beta", type=float, default=0.9)
    parser.add_argument("--fedite_tau_enter", type=float, default=None)
    parser.add_argument("--fedite_tau_exit", type=float, default=None)
    parser.add_argument("--fedite_exit_patience", type=int, default=2)
    parser.add_argument("--fedite_observer_we", type=float, default=0.25)
    parser.add_argument("--fedite_observer_wk", type=float, default=0.25)
    parser.add_argument("--fedite_observer_wg", type=float, default=0.25)
    parser.add_argument("--fedite_observer_wu", type=float, default=0.25)
    parser.add_argument("--fedite_reliability_wm", type=float, default=1 / 3)
    parser.add_argument("--fedite_reliability_wk", type=float, default=1 / 3)
    parser.add_argument("--fedite_reliability_ws", type=float, default=1 / 3)
    parser.add_argument("--fedite_stability_temperature", type=float, default=1.0)
    parser.add_argument("--fedite_reliability_support_m0", type=float, default=2.0)
    parser.add_argument("--fedite_reliability_client_q0", type=float, default=1.0)
    parser.add_argument("--fedite_selection_rho", type=float, default=0.5)
    parser.add_argument("--fedite_support_clip_max", type=float, default=20.0)
    parser.add_argument("--fedite_write_clip_max", type=float, default=20.0)
    parser.add_argument("--fedite_tailagg_momentum", type=float, default=1.0)

    parser.add_argument("--fedite_lambda_tail", type=float, default=1.0)
    parser.add_argument("--fedite_lambda_router", type=float, default=0.1)
    parser.add_argument("--fedite_lambda_feature_anchor", type=float, default=0.0)
    parser.add_argument("--fedite_lambda_safe", type=float, default=0.0)
    parser.add_argument("--fedite_safe_type", type=str, default="cosine")
    parser.add_argument("--fedite_lambda_boundary", type=float, default=0.0)
    parser.add_argument("--fedite_boundary_topk", type=int, default=5)
    parser.add_argument("--fedite_boundary_tolerance", type=float, default=0.0)
    parser.add_argument("--fedite_lambda_candidate_kl", type=float, default=0.0)
    parser.add_argument("--fedite_candidate_kl_topk", type=int, default=5)
    parser.add_argument("--fedite_candidate_kl_temperature", type=float, default=2.0)
    parser.add_argument("--fedite_tail_safe_weight", type=float, default=0.0)
    parser.add_argument("--fedite_shared_survival_weight", type=float, default=0.0)
    parser.add_argument("--fedite_tail_survival_weight", type=float, default=0.0)
    parser.add_argument("--fedite_tail_hardneg_weight", type=float, default=0.0)
    parser.add_argument("--fedite_tail_hardneg_topm", type=int, default=5)
    parser.add_argument("--fedite_tail_margin_weight", type=float, default=0.0)
    parser.add_argument("--fedite_tail_margin", type=float, default=0.5)
    parser.add_argument("--fedite_shared_lr", type=float, default=1e-3)
    parser.add_argument("--fedite_gate_lr", type=float, default=1e-3)
    parser.add_argument("--fedite_tail_lr", type=float, default=1e-3)
    parser.add_argument("--fedite_weight_decay", type=float, default=0.0)
    parser.add_argument("--fedite_inference_topk", type=int, default=5)
    parser.add_argument("--fedite_inference_temperature", type=float, default=2.0)
    parser.add_argument("--fedite_prefix_cache", type=str2bool, default=True)
    parser.add_argument("--fedite_cache_text_features", type=str2bool, default=True)
    parser.add_argument("--fedite_eval_prev_state", type=str2bool, default=False)
    parser.add_argument("--fedite_module_diagnostics", type=str2bool, default=False)
    parser.add_argument("--fedite_scalar_diagnostics", type=str2bool, default=True)
    parser.add_argument("--fedite_diagnostics_interval", type=int, default=10)
    parser.add_argument("--fedite_per_class_metrics", type=str2bool, default=False)
    parser.add_argument("--fedite_eval_interval", type=int, default=5)
    parser.add_argument("--fedite_checkpoint_interval", type=int, default=10)
    return parser


def main(args):
    if args.method.lower() != "fedite":
        raise ValueError("fedite_main.py only implements --method fedite")
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = make_output_dir(args)
    checkpoint_dir = ensure_dir(os.path.join(output_dir, "checkpoints"))
    write_json(os.path.join(output_dir, "config.json"), vars(args))
    train_log_path = os.path.join(output_dir, "train.log")

    def log(message):
        print(message)
        with open(train_log_path, "a", encoding="utf-8") as handle:
            handle.write(str(message) + "\n")
            handle.flush()

    log("FedITE method initialized")
    log("Topology definition: class-client evidence topology, not physical communication topology")
    log("Round protocol: local training and TailAgg use observer state from round t-1; evaluation logs updated-state deployment metrics.")
    log(
        "Main-experiment fast path: "
        f"prefix_cache={args.fedite_prefix_cache}, "
        f"text_cache={args.fedite_cache_text_features}, "
        f"prev_eval={args.fedite_eval_prev_state}, "
        f"module_diagnostics={args.fedite_module_diagnostics}, "
        f"per_class_metrics={args.fedite_per_class_metrics}"
    )
    log(f"Loading CLIP backbone: {args.clip_backbone}")
    clip_model = load_clip_to_cpu(args.clip_backbone)
    clip_model.float()
    preprocess = clip._transform(clip_model.visual.input_resolution)

    client_loaders, test_loader, classnames, train_class_counts, _train_counts = build_fedite_data(args, preprocess)
    model = FedITEModel(clip_model, classnames, args).to(device)
    log(f"adapter layers: {args.fedite_adapter_layers}")
    log(f"parameter counts: {model.parameter_counts()}")
    write_json(
        os.path.join(output_dir, "parameter_manifest.json"),
        parameter_manifest(
            model,
            model.get_shared_parameter_keys(),
            model.get_gate_parameter_keys(),
            model.get_tail_parameter_keys(),
            model.get_frozen_parameter_keys(),
        ),
    )

    observer = EvidenceTopologyObserver(
        num_classes=len(classnames),
        protected_ratio=args.fedite_protected_ratio,
        exploration_ratio=args.fedite_exploration_ratio,
        beta=args.fedite_observer_beta,
        reliability_min=args.fedite_reliability_min,
        warmup_rounds=args.fedite_warmup_rounds,
        warmup_mode=args.fedite_warmup_mode,
        tau_enter=args.fedite_tau_enter,
        tau_exit=args.fedite_tau_exit,
        exit_patience=args.fedite_exit_patience,
        w_e=args.fedite_observer_we,
        w_k=args.fedite_observer_wk,
        w_g=args.fedite_observer_wg,
        w_u=args.fedite_observer_wu,
        r_m=args.fedite_reliability_wm,
        r_k=args.fedite_reliability_wk,
        r_s=args.fedite_reliability_ws,
        stability_temperature=args.fedite_stability_temperature,
        reliability_support_m0=args.fedite_reliability_support_m0,
        reliability_client_q0=args.fedite_reliability_client_q0,
        selection_rho=args.fedite_selection_rho,
        seed=args.seed,
    )
    trainer = FedITEClientTrainer(model, args, device)
    global_state = as_cpu_state_dict(model)
    splits = split_head_medium_tail(train_class_counts, tail_ratio=args.tail_class_ratio)
    rng = random.Random(args.seed)
    best_tail = -1.0
    best_round = -1

    round_fields = [
        "round", "overall_acc", "non_tail_acc", "tail_acc", "macro_acc",
        "shared_overall_acc", "shared_tail_acc", "final_minus_shared_tail",
        "protected_acc", "nonprotected_acc", "protected_count", "protected_overlap",
        "protected_tail_overlap", "protected_avg_global_count",
        "topD_classes", "topS_classes", "topRarity_classes", "missed_tail_classes",
        "eligible_tail_clients", "protected_positive_samples", "loss_shared",
        "loss_feature_anchor", "loss_safe", "loss_boundary", "loss_candidate_kl",
        "loss_router", "loss_tail", "gate_mean",
        "gate_protected_mean", "gate_nonprotected_mean",
        "gate_low_saturation_ratio", "gate_high_saturation_ratio",
        "tail_weight_mean", "mean_reliability_eligible",
        "tail_optimizer_steps", "tail_skip_count", "tail_write_norm",
        "shared_update_norm", "gate_update_norm", "tail_update_norm",
        "tail_retention_ratio", "prev_state_overall_acc", "prev_state_tail_acc",
        "round_time",
    ]
    observer_fields = [
        "round", "class_id", "EMA_M", "EMA_Q", "EMA_H", "EMA_N_eff", "EMA_E",
        "EMA_U", "EMA_U2", "Var_U", "Gap", "SeenCount", "D", "R", "S", "Rarity",
        "is_protected", "is_exploration",
    ]

    for round_idx in range(args.rounds):
        start = time.time()
        shared_only = bool(args.fedite_shared_only)
        tail_active = (not shared_only) and round_idx >= int(args.fedite_warmup_rounds)
        m = max(1, int(round(args.num_users * args.frac)))
        selected = sorted(rng.sample(list(range(args.num_users)), m))
        if shared_only:
            protected_classes = []
            observer.protected_classes = []
            observer.exploration_classes = []
            observer.last_selection_info = {
                "mode": "shared_only",
                "new": [],
                "removed": [],
                "overlap": 0,
            }
        else:
            protected_classes = observer.select_protected_classes(round_idx)
        selection_diag = selection_diagnostics(round_idx, observer, train_class_counts, splits)
        append_jsonl(os.path.join(output_dir, "selection_diagnostics.jsonl"), selection_diag)
        class_state = observer.get_class_state()
        log(f"Round {round_idx}: selected={selected}, protected={protected_classes}, tail_active={tail_active}")

        local_states = []
        local_stats = []
        local_client_ids = []
        for client_id in selected:
            if len(client_loaders[client_id].dataset) == 0:
                continue
            state, stats = trainer.train_one_client(
                global_state=global_state,
                train_loader=client_loaders[client_id],
                protected_classes=protected_classes,
                class_evidence_state=class_state,
                round_idx=round_idx,
                tail_active=tail_active,
            )
            local_states.append(state)
            local_stats.append(stats)
            local_client_ids.append(client_id)
            append_jsonl(
                os.path.join(output_dir, "client_scalar_diagnostics.jsonl"),
                {
                    "round": round_idx,
                    "client_id": client_id,
                    "tail_active": bool(stats.get("tail_active", tail_active)),
                    "num_samples": int(stats.get("num_samples", 0)),
                    "protected_positive_count": int(stats.get("protected_positive_count", 0)),
                    "loss_shared": float(stats.get("loss_shared", 0.0)),
                    "loss_feature_anchor": float(stats.get("loss_feature_anchor", 0.0)),
                    "loss_safe": float(stats.get("loss_safe", 0.0)),
                    "loss_boundary": float(stats.get("loss_boundary", 0.0)),
                    "loss_candidate_kl": float(stats.get("loss_candidate_kl", 0.0)),
                    "loss_router": float(stats.get("loss_router", 0.0)),
                    "loss_tail": float(stats.get("loss_tail", 0.0)),
                    "tail_optimizer_steps": int(stats.get("tail_optimizer_steps", 0)),
                    "tail_skip_count": int(stats.get("tail_skip_count", 0)),
                    "shared_update_norm": float(stats.get("shared_update_norm", 0.0)),
                    "gate_update_norm": float(stats.get("gate_update_norm", 0.0)),
                    "tail_update_norm": float(stats.get("tail_update_norm", 0.0)),
                    "tail_write_norm": float(stats.get("tail_write_norm", 0.0)),
                },
            )

        round_summary = aggregate_round_stats(local_stats, len(classnames))
        global_state, agg_diag = aggregate_fedite(
            local_states=local_states,
            local_stats=local_stats,
            previous_global_state=global_state,
            shared_keys=model.get_shared_parameter_keys(),
            gate_keys=model.get_gate_parameter_keys(),
            tail_keys=model.get_tail_parameter_keys(),
            observer_state={"class_state": class_state},
            tail_active=tail_active,
            tail_momentum=args.fedite_tailagg_momentum,
        )
        observer_diag = observer.update(round_summary)

        model.load_state_dict(global_state, strict=False)
        current_state = observer.get_class_state()
        metrics = {}
        prev_metrics = {}
        if round_idx % max(int(args.fedite_eval_interval), 1) == 0:
            if bool(args.fedite_eval_prev_state):
                prev_metrics = evaluate(
                    model,
                    test_loader,
                    class_state,
                    splits,
                    protected_classes,
                    args,
                    device,
                    tail_active=tail_active,
                )
            metrics = evaluate(
                model,
                test_loader,
                current_state,
                splits,
                observer.protected_classes,
                args,
                device,
                tail_active=tail_active,
            )
            if tail_active and metrics["tail_acc"] > best_tail:
                best_tail = metrics["tail_acc"]
                best_round = round_idx
                torch.save(
                    {"round": round_idx, "global_state": global_state, "observer": observer.state_dict(), "args": vars(args)},
                    os.path.join(checkpoint_dir, "best.pt"),
                )

        loss_avg = {
            key: sum(float(stats.get(key, 0.0)) for stats in local_stats) / max(len(local_stats), 1)
            for key in [
                "loss_shared", "loss_feature_anchor", "loss_safe", "loss_boundary", "loss_candidate_kl",
                "loss_router", "loss_tail", "gate_mean",
                "gate_protected_mean", "gate_nonprotected_mean",
                "gate_low_saturation_ratio", "gate_high_saturation_ratio",
                "tail_write_norm",
            ]
        }
        step_avg = {
            key: sum(int(stats.get(key, 0)) for stats in local_stats)
            for key in ["tail_optimizer_steps", "tail_skip_count"]
        }
        row = {
            "round": round_idx,
            **{k: metrics.get(k, 0.0) for k in [
                "overall_acc", "non_tail_acc", "tail_acc", "macro_acc",
                "shared_overall_acc", "shared_tail_acc", "final_minus_shared_tail",
                "protected_acc", "nonprotected_acc",
            ]},
            "protected_count": len(observer.protected_classes),
            "protected_overlap": observer.last_selection_info.get("overlap", 0),
            "protected_tail_overlap": selection_diag["protected_tail_overlap"],
            "protected_avg_global_count": selection_diag["protected_avg_global_count"],
            "topD_classes": format_class_ids(selection_diag["topD_classes"]),
            "topS_classes": format_class_ids(selection_diag["topS_classes"]),
            "topRarity_classes": format_class_ids(selection_diag["topRarity_classes"]),
            "missed_tail_classes": format_class_ids(selection_diag["missed_tail_classes"]),
            "eligible_tail_clients": agg_diag["eligible_tail_clients"],
            "protected_positive_samples": agg_diag["protected_positive_samples"],
            **loss_avg,
            **step_avg,
            "tail_weight_mean": agg_diag["tail_weight_mean"],
            "mean_reliability_eligible": agg_diag["mean_reliability_eligible"],
            "shared_update_norm": agg_diag["shared_global_update_norm"],
            "gate_update_norm": agg_diag["gate_global_update_norm"],
            "tail_update_norm": agg_diag["tail_global_update_norm"],
            "tail_retention_ratio": agg_diag["tail_retention_ratio"],
            "prev_state_overall_acc": prev_metrics.get("overall_acc", 0.0),
            "prev_state_tail_acc": prev_metrics.get("tail_acc", 0.0),
            "round_time": time.time() - start,
        }
        append_csv(os.path.join(output_dir, "round_metrics.csv"), [row], round_fields)
        append_csv(os.path.join(output_dir, "observer_class_state.csv"), observer.class_state_rows(round_idx), observer_fields)
        per_class_fields = [
            "round", "class_id", "global_train_count", "test_support",
            "final_accuracy", "shared_accuracy", "final_minus_shared",
            "D", "R", "S", "Rarity", "is_protected", "EMA_M", "EMA_N_eff", "EMA_U", "Gap",
        ]
        if bool(args.fedite_per_class_metrics) and metrics:
            append_csv(
                os.path.join(output_dir, "per_class_metrics.csv"),
                per_class_metric_rows(round_idx, metrics, observer, train_class_counts),
                per_class_fields,
            )
        if bool(args.fedite_module_diagnostics):
            for client_id, stats in zip(local_client_ids, local_stats):
                for module_row in stats.get("module_diagnostics", []):
                    payload = {
                        "round": round_idx,
                        "client_id": client_id,
                        "tail_active": tail_active,
                        **module_row,
                    }
                    append_jsonl(os.path.join(output_dir, "module_diagnostics.jsonl"), payload)

        if round_idx % max(int(args.fedite_checkpoint_interval), 1) == 0:
            torch.save(
                {"round": round_idx, "global_state": global_state, "observer": observer.state_dict(), "args": vars(args)},
                os.path.join(checkpoint_dir, f"round_{round_idx:03d}.pt"),
            )
        log(
            "FedITE Observer: "
            f"M={observer_diag['M']}",
        )
        log(
            "FedITE Observer state: "
            f"N_eff={observer_diag['N_eff']} "
            f"D={observer_diag['D']} "
            f"R={observer_diag['R']} "
            f"S={observer_diag['S']}"
        )
        log(
            "FedITE Selection: "
            f"tail overlap={selection_diag['protected_tail_overlap']}/{len(splits.get('tail', []))} "
            f"avg_global_count={selection_diag['protected_avg_global_count']:.2f} "
            f"topD={selection_diag['topD_classes']} "
            f"topS={selection_diag['topS_classes']} "
            f"topRarity={selection_diag['topRarity_classes']} "
            f"missed_tail={selection_diag['missed_tail_classes']}"
        )
        log(
            "FedITE TailAgg: "
            f"eligible clients={agg_diag['eligible_tail_clients']}/{len(local_stats)} "
            f"protected positive samples={agg_diag['protected_positive_samples']} "
            f"tail weight mean={agg_diag['tail_weight_mean']:.6f}"
        )
        if metrics:
            log(
                f"Round {round_idx} acc: overall={metrics['overall_acc']:.4f}, "
                f"non_tail={metrics['non_tail_acc']:.4f}, "
                f"tail={metrics['tail_acc']:.4f}, macro={metrics['macro_acc']:.4f}"
            )

    torch.save(
        {"round": args.rounds - 1, "global_state": global_state, "observer": observer.state_dict(), "args": vars(args)},
        os.path.join(checkpoint_dir, "last.pt"),
    )
    write_json(
        os.path.join(output_dir, "final_summary.json"),
        {
            "best_round": best_round,
            "best_tail": best_tail,
            "last_round": args.rounds - 1,
            "protected_classes": observer.protected_classes,
            "parameter_counts": model.parameter_counts(),
        },
    )
    log(f"FedITE run complete. Outputs written to: {output_dir}")


if __name__ == "__main__":
    main(build_arg_parser().parse_args())
