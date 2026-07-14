import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import torch


def split_tail_non_tail(train_class_counts, tail_ratio=0.2):
    counts = torch.as_tensor(train_class_counts, dtype=torch.float32)
    num_classes = int(counts.numel())
    num_tail = max(1, int(round(num_classes * float(tail_ratio))))
    ordered = sorted(range(num_classes), key=lambda c: (float(counts[c].item()), int(c)))
    tail_class_ids = sorted(ordered[:num_tail])
    tail_set = set(tail_class_ids)
    non_tail_class_ids = [c for c in range(num_classes) if c not in tail_set]
    return tail_class_ids, non_tail_class_ids, {int(c): i for i, c in enumerate(tail_class_ids)}


def client_class_count_matrix(client_indices: Sequence[Sequence[int]], labels, num_classes: int):
    labels = torch.as_tensor(labels, dtype=torch.long)
    out = torch.zeros(len(client_indices), int(num_classes), dtype=torch.float32)
    for client_id, indices in enumerate(client_indices):
        if len(indices) == 0:
            continue
        y = labels[torch.as_tensor(indices, dtype=torch.long)]
        out[client_id] = torch.bincount(y.cpu(), minlength=num_classes).float()[:num_classes]
    return out


def compute_tail_topology(client_class_counts, tail_class_ids, global_train_counts=None, eps=1e-12):
    counts = torch.as_tensor(client_class_counts, dtype=torch.float32)
    tail_ids = [int(c) for c in tail_class_ids]
    tail_counts = counts[:, tail_ids] if tail_ids else counts.new_zeros(counts.shape[0], 0)
    s1 = tail_counts.sum(dim=0)
    s2 = tail_counts.pow(2).sum(dim=0)
    concentration = s2 / (s1.pow(2) + float(eps))
    depth = s2 / (s1 + float(eps))
    n_eff = s1.pow(2) / (s2 + float(eps))
    holders = (tail_counts > 0).sum(dim=0).float()
    sorted_counts = torch.sort(tail_counts, dim=0, descending=True).values
    top1 = torch.where(s1 > 0, sorted_counts[0] / s1.clamp_min(float(eps)), torch.zeros_like(s1))
    if sorted_counts.shape[0] >= 2:
        top2_mass = sorted_counts[:2].sum(dim=0)
    else:
        top2_mass = sorted_counts[:1].sum(dim=0)
    top2 = torch.where(s1 > 0, top2_mass / s1.clamp_min(float(eps)), torch.zeros_like(s1))
    global_counts = torch.as_tensor(global_train_counts, dtype=torch.float32) if global_train_counts is not None else counts.sum(dim=0)
    rows = []
    for tail_index, class_id in enumerate(tail_ids):
        rows.append({
            "class_id": int(class_id),
            "tail_index": int(tail_index),
            "global_train_count": float(global_counts[class_id].item()),
            "M_k": float(s1[tail_index].item()),
            "C_k": float(concentration[tail_index].item()),
            "D_k": float(depth[tail_index].item()),
            "N_eff_k": float(n_eff[tail_index].item()),
            "num_holders": float(holders[tail_index].item()),
            "top1_client_mass": float(top1[tail_index].item()),
            "top2_client_mass": float(top2[tail_index].item()),
        })
    tensors = {
        "M": s1.float(),
        "C": concentration.float(),
        "D": depth.float(),
        "N_eff": n_eff.float(),
        "num_holders": holders.float(),
        "top1_client_mass": top1.float(),
        "top2_client_mass": top2.float(),
    }
    return tensors, rows


def partial_participation_estimator(selected_client_counts, participation_prob, eps=1e-12):
    counts = torch.as_tensor(selected_client_counts, dtype=torch.float32)
    pi = max(float(participation_prob), float(eps))
    s1_hat = counts.sum(dim=0) / pi
    s2_hat = counts.pow(2).sum(dim=0) / pi
    return {
        "S1_hat": s1_hat,
        "S2_hat": s2_hat,
        "C_hat": s2_hat / (s1_hat.pow(2) + float(eps)),
        "D_hat": s2_hat / (s1_hat + float(eps)),
    }


def write_topology_report(rows: List[Dict], tensors: Dict[str, torch.Tensor], csv_path, json_path):
    csv_path = Path(csv_path)
    json_path = Path(json_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    summary = {
        "num_tail": len(rows),
        "M_mean": float(tensors["M"].mean().item()) if tensors["M"].numel() else 0.0,
        "C_mean": float(tensors["C"].mean().item()) if tensors["C"].numel() else 0.0,
        "D_mean": float(tensors["D"].mean().item()) if tensors["D"].numel() else 0.0,
        "N_eff_mean": float(tensors["N_eff"].mean().item()) if tensors["N_eff"].numel() else 0.0,
        "rows": rows,
    }
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    return summary
