from typing import Dict, Iterable, List

import torch
import torch.nn.functional as F

from .state import sanitize_residual


def build_tail_direction(z, rho, gate, radius=None):
    z = F.normalize(z.float(), dim=-1)
    rho = rho.float()
    if radius is not None:
        rho = sanitize_residual(rho, z, radius)
    else:
        rho = rho - z * (z * rho).sum(dim=-1, keepdim=True)
    gate = torch.as_tensor(gate, dtype=rho.dtype, device=rho.device)
    while gate.ndim < rho.ndim:
        gate = gate.unsqueeze(-1)
    return F.normalize(z + gate * rho, dim=-1)


def assemble_text_bank(num_classes, zero_shot_text, non_tail_class_ids, non_tail_text, tail_class_ids, tail_text):
    text = torch.empty(num_classes, zero_shot_text.shape[1], dtype=torch.float32, device=zero_shot_text.device)
    text[:] = zero_shot_text.float()
    if len(non_tail_class_ids) > 0:
        text[torch.as_tensor(non_tail_class_ids, dtype=torch.long, device=text.device)] = non_tail_text.float()
    if len(tail_class_ids) > 0:
        text[torch.as_tensor(tail_class_ids, dtype=torch.long, device=text.device)] = tail_text.float()
    return F.normalize(text, dim=-1)


def compute_logits(features, text_bank, logit_scale):
    return float(logit_scale) * F.normalize(features.float(), dim=-1) @ F.normalize(text_bank.float(), dim=-1).t()


def tail_vs_head_margins(logits, labels, tail_set, non_tail_class_ids, tail_class_ids):
    labels = labels.long()
    tail_mask = torch.as_tensor([int(y) in tail_set for y in labels.detach().cpu().tolist()], device=labels.device)
    if not bool(tail_mask.any()):
        return {
            "tail_to_head_error_rate": 0.0,
            "tail_to_tail_error_rate": 0.0,
            "mean_tail_vs_head_margin": 0.0,
            "mean_tail_vs_tail_margin": 0.0,
            "num_tail_samples": 0,
        }
    logits_t = logits[tail_mask]
    labels_t = labels[tail_mask]
    pred_t = logits_t.argmax(dim=1)
    pred_tail = torch.as_tensor([int(y) in tail_set for y in pred_t.detach().cpu().tolist()], device=labels.device)
    correct = pred_t == labels_t
    tail_to_head = (~pred_tail).float().mean().item()
    tail_to_tail = (pred_tail & (~correct)).float().mean().item()
    true = logits_t.gather(1, labels_t.view(-1, 1)).squeeze(1)
    head_ids = torch.as_tensor(non_tail_class_ids, dtype=torch.long, device=logits.device)
    tail_ids = torch.as_tensor(tail_class_ids, dtype=torch.long, device=logits.device)
    head_margin = (true - logits_t.index_select(1, head_ids).max(dim=1).values).mean().item() if head_ids.numel() else 0.0
    other_tail_logits = logits_t.index_select(1, tail_ids).clone() if tail_ids.numel() > 1 else logits_t.new_zeros(logits_t.shape[0], 0)
    if other_tail_logits.numel():
        tail_pos = {int(c): i for i, c in enumerate(tail_class_ids)}
        for row, y in enumerate(labels_t.detach().cpu().tolist()):
            if int(y) in tail_pos:
                other_tail_logits[row, tail_pos[int(y)]] = -1e4
        tail_margin = (true - other_tail_logits.max(dim=1).values).mean().item()
    else:
        tail_margin = 0.0
    return {
        "tail_to_head_error_rate": float(tail_to_head),
        "tail_to_tail_error_rate": float(tail_to_tail),
        "mean_tail_vs_head_margin": float(head_margin),
        "mean_tail_vs_tail_margin": float(tail_margin),
        "num_tail_samples": int(labels_t.numel()),
    }
