import torch

from .classifier import assemble_text_bank, build_tail_direction, compute_logits, tail_vs_head_margins


def _split_acc(class_correct, class_total, class_ids):
    ids = torch.as_tensor(class_ids, dtype=torch.long)
    if ids.numel() == 0:
        return 0.0
    return float(class_correct[ids].sum().item() / class_total[ids].sum().clamp_min(1.0).item())


@torch.no_grad()
def evaluate_tcrm(features, labels, prompt_learner, state, logit_scale, batch_size=512, device="cpu"):
    prompt_learner.eval().to(device)
    features = features.float()
    labels = labels.long()
    num_classes = int(state.zero_shot_text.shape[0])
    tail_z = state.zero_shot_text[state.tail_class_ids].to(device)
    tail_text = build_tail_direction(tail_z, state.rho.to(device), state.width_gate.to(device), state.rho_norm_bound)
    non_tail_ids = torch.as_tensor(state.non_tail_class_ids, dtype=torch.long, device=device)
    non_tail_text = prompt_learner(non_tail_ids).detach().float()
    text_bank = assemble_text_bank(
        num_classes,
        state.zero_shot_text.to(device),
        state.non_tail_class_ids,
        non_tail_text,
        state.tail_class_ids,
        tail_text,
    )
    zero_text = state.zero_shot_text.to(device)
    class_total = torch.zeros(num_classes)
    class_correct = torch.zeros(num_classes)
    zs_correct = torch.zeros(num_classes)
    tail_metric_keys = ["tail_to_head_error_rate", "tail_to_tail_error_rate", "mean_tail_vs_head_margin", "mean_tail_vs_tail_margin"]
    tail_metric_sums = {key: 0.0 for key in tail_metric_keys}
    tail_metric_count = 0
    class_tail_vs_head_margin_sum = torch.zeros(num_classes)
    class_tail_vs_tail_margin_sum = torch.zeros(num_classes)
    class_tail_margin_count = torch.zeros(num_classes)
    tail_set = set(state.tail_class_ids)
    non_tail_ids = torch.as_tensor(state.non_tail_class_ids, dtype=torch.long, device=device)
    tail_ids = torch.as_tensor(state.tail_class_ids, dtype=torch.long, device=device)
    for start in range(0, len(features), int(batch_size)):
        x = features[start:start + int(batch_size)].to(device)
        y = labels[start:start + int(batch_size)].to(device)
        logits = compute_logits(x, text_bank, logit_scale)
        logits_zs = compute_logits(x, zero_text, logit_scale)
        pred = logits.argmax(dim=1)
        pred_zs = logits_zs.argmax(dim=1)
        for cls in y.detach().cpu().unique().tolist():
            mask = y.detach().cpu() == int(cls)
            class_total[int(cls)] += mask.sum()
            class_correct[int(cls)] += (pred.detach().cpu()[mask] == int(cls)).sum()
            zs_correct[int(cls)] += (pred_zs.detach().cpu()[mask] == int(cls)).sum()
        batch_tail_metrics = tail_vs_head_margins(logits, y, tail_set, state.non_tail_class_ids, state.tail_class_ids)
        tail_count = int(batch_tail_metrics.get("num_tail_samples", 0))
        if tail_count > 0:
            tail_metric_count += tail_count
            for key in tail_metric_keys:
                tail_metric_sums[key] += float(batch_tail_metrics[key]) * tail_count
            tail_batch_mask = torch.as_tensor([int(label) in tail_set for label in y.detach().cpu().tolist()], dtype=torch.bool, device=device)
            logits_tail = logits[tail_batch_mask]
            labels_tail = y[tail_batch_mask]
            true = logits_tail.gather(1, labels_tail.view(-1, 1)).squeeze(1)
            if non_tail_ids.numel() > 0:
                head_margin_samples = true - logits_tail.index_select(1, non_tail_ids).max(dim=1).values
            else:
                head_margin_samples = logits_tail.new_zeros(logits_tail.shape[0])
            if tail_ids.numel() > 1:
                other_tail_logits = logits_tail.index_select(1, tail_ids).clone()
                tail_pos = {int(class_id): idx for idx, class_id in enumerate(state.tail_class_ids)}
                for row, label in enumerate(labels_tail.detach().cpu().tolist()):
                    other_tail_logits[row, tail_pos[int(label)]] = -1e4
                tail_margin_samples = true - other_tail_logits.max(dim=1).values
            else:
                tail_margin_samples = logits_tail.new_zeros(logits_tail.shape[0])
            for cls in labels_tail.detach().cpu().unique().tolist():
                cls_mask = labels_tail == int(cls)
                class_tail_vs_head_margin_sum[int(cls)] += head_margin_samples[cls_mask].sum().detach().cpu()
                class_tail_vs_tail_margin_sum[int(cls)] += tail_margin_samples[cls_mask].sum().detach().cpu()
                class_tail_margin_count[int(cls)] += cls_mask.sum().detach().cpu()
    class_acc = class_correct / class_total.clamp_min(1.0)
    zs_acc = zs_correct / class_total.clamp_min(1.0)
    tail_metrics = {key: float(tail_metric_sums[key] / max(tail_metric_count, 1)) for key in tail_metric_keys}
    class_tail_vs_head_margin = class_tail_vs_head_margin_sum / class_tail_margin_count.clamp_min(1.0)
    class_tail_vs_tail_margin = class_tail_vs_tail_margin_sum / class_tail_margin_count.clamp_min(1.0)
    overall = float(class_correct.sum().item() / class_total.sum().clamp_min(1.0).item())
    zs_overall = float(zs_correct.sum().item() / class_total.sum().clamp_min(1.0).item())
    tail_acc = _split_acc(class_correct, class_total, state.tail_class_ids)
    zs_tail = _split_acc(zs_correct, class_total, state.tail_class_ids)
    head_acc = _split_acc(class_correct, class_total, state.non_tail_class_ids)
    zs_head = _split_acc(zs_correct, class_total, state.non_tail_class_ids)
    metrics = {
        "overall_acc": overall,
        "macro_acc": float(class_acc.mean().item()),
        "head_acc": head_acc,
        "tail_acc": tail_acc,
        "zero_shot_overall_acc": zs_overall,
        "zero_shot_head_acc": zs_head,
        "zero_shot_tail_acc": zs_tail,
        "hybrid_tail_acc": tail_acc,
        "tail_gain_over_zero_shot": tail_acc - zs_tail,
        "class_acc": class_acc,
        "zero_shot_class_acc": zs_acc,
        "class_total": class_total,
        "class_tail_vs_head_margin": class_tail_vs_head_margin,
        "class_tail_vs_tail_margin": class_tail_vs_tail_margin,
        **tail_metrics,
    }
    return metrics
