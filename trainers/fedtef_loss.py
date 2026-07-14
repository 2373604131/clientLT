from types import SimpleNamespace

import torch
import torch.nn.functional as F


def _cfg_value(cfg, name, default):
    return getattr(cfg, name, default) if cfg is not None else default


def assert_finite_tensor(name, tensor):
    if not torch.isfinite(tensor).all():
        safe = tensor.detach().float().nan_to_num()
        print(f"[FedTEF NaN DEBUG] {name} has NaN/Inf")
        print(
            "[FedTEF NaN DEBUG] "
            f"min={safe.min().item():.6f}, "
            f"max={safe.max().item():.6f}, "
            f"mean={safe.mean().item():.6f}"
        )
        raise FloatingPointError(f"{name} is NaN/Inf")


def accumulate_difficulty(logits_base, labels, margin_target=1.0):
    logits_base = logits_base.detach().float()
    labels = labels.detach()
    _, num_classes = logits_base.shape

    true_logits = logits_base.gather(1, labels.view(-1, 1)).squeeze(1)
    masked_logits = logits_base.clone()
    masked_logits.scatter_(1, labels.view(-1, 1), -1e4)
    max_negative = masked_logits.max(dim=1).values

    violation = F.relu(float(margin_target) - (true_logits - max_negative))
    difficulty_sum = torch.zeros(num_classes, device=logits_base.device)
    difficulty_count = torch.zeros(num_classes, device=logits_base.device)
    difficulty_sum.scatter_add_(0, labels, violation)
    difficulty_count.scatter_add_(0, labels, torch.ones_like(violation))
    return difficulty_sum, difficulty_count


def build_positive_row_mask(labels, gate, num_classes, eps=1e-6):
    mask = torch.zeros(num_classes, dtype=torch.bool, device=labels.device)
    unique_labels = labels.unique()
    active = gate.to(labels.device)[unique_labels] > float(eps)
    mask[unique_labels[active]] = True
    return mask


def mask_classwise_grad(param, row_mask):
    if param is None or param.grad is None:
        return
    view_shape = [row_mask.shape[0]] + [1] * (param.grad.ndim - 1)
    mask = row_mask.view(*view_shape).to(param.grad.device, param.grad.dtype)
    param.grad.mul_(mask)


def compute_prior_base_loss(logits_base, labels, tail_score, kappa=0.3, w_max=2.0, eps=1e-6):
    ce_each = F.cross_entropy(logits_base, labels, reduction="none")
    score = tail_score.detach().to(device=logits_base.device, dtype=torch.float32)
    positive = score[score > float(eps)]
    ref = positive.mean() if positive.numel() else score.mean()
    score = score / ref.clamp_min(float(eps))
    class_weight = (1.0 + float(kappa) * score).clamp(1.0, float(w_max))
    sample_weight = class_weight.to(device=logits_base.device, dtype=logits_base.dtype)[labels]
    return (sample_weight.detach() * ce_each).mean()


def controlled_hard_negative_loss(
    logits_base,
    residual_tail,
    labels,
    protected_label,
    topm=5,
    residual_lambda=0.5,
):
    active = torch.nonzero(protected_label, as_tuple=False).view(-1)
    if active.numel() == 0:
        return residual_tail.sum() * 0.0

    logits_base = torch.nan_to_num(logits_base.float(), nan=0.0, posinf=80.0, neginf=-80.0)
    residual_tail = torch.nan_to_num(residual_tail.float(), nan=0.0, posinf=80.0, neginf=-80.0)
    num_classes = logits_base.shape[1]
    topm = max(1, min(int(topm), num_classes - 1))
    logits_obj = logits_base.detach() + float(residual_lambda) * residual_tail

    losses = []
    for idx in active.tolist():
        label = int(labels[idx].item())
        base_row = logits_base[idx].detach().clone()
        base_row[label] = -1e4
        hard_ids = torch.topk(base_row, k=topm).indices
        class_ids = torch.cat(
            [torch.tensor([label], device=hard_ids.device, dtype=hard_ids.dtype), hard_ids]
        )
        local_logits = logits_obj[idx, class_ids].view(1, -1)
        local_target = torch.zeros(1, dtype=torch.long, device=local_logits.device)
        losses.append(F.cross_entropy(local_logits, local_target))

    return torch.stack(losses).mean()


def compute_safe_kl(logits_base, logits_fused, labels, gate, conf_threshold=0.7, eps=1e-6):
    with torch.no_grad():
        base_probs = F.softmax(logits_base.detach().float(), dim=-1)
        confidence = base_probs.max(dim=1).values
        non_protected = gate.to(labels.device)[labels] <= float(eps)
        high_conf = confidence > float(conf_threshold)
        safe_mask = non_protected | high_conf

    kl_each = F.kl_div(
        F.log_softmax(logits_fused.float(), dim=-1),
        F.softmax(logits_base.detach().float(), dim=-1),
        reduction="none",
    ).sum(dim=1)

    if safe_mask.any():
        return kl_each[safe_mask].mean()
    return logits_fused.sum() * 0.0


def make_fedtef_loss_config(cfg=None):
    return SimpleNamespace(
        eps=float(_cfg_value(cfg, "EXPOSURE_EPS", 1e-6)),
        w_base=float(_cfg_value(cfg, "LOSS_BASE_WEIGHT", 1.0)),
        w_prior=float(_cfg_value(cfg, "V10_PRIOR_BASE_WEIGHT", 0.2)),
        w_res=float(_cfg_value(cfg, "LOSS_TAIL_WEIGHT", 0.8)),
        w_fused=float(_cfg_value(cfg, "LOSS_FUSED_WEIGHT", 0.2)),
        w_safe=float(_cfg_value(cfg, "LOSS_KEEP_KL_WEIGHT", 0.05)),
        prior_kappa=float(_cfg_value(cfg, "V10_PRIOR_KAPPA", 0.3)),
        prior_w_max=float(_cfg_value(cfg, "V10_PRIOR_W_MAX", 2.0)),
        hardneg_topm=int(_cfg_value(cfg, "V10_HARDNEG_TOPM", 5)),
        hardneg_lambda=float(_cfg_value(cfg, "V10_HARDNEG_LAMBDA", 0.5)),
        safe_conf_threshold=float(_cfg_value(cfg, "V10_SAFE_CONF_THRESHOLD", 0.7)),
    )


def compute_fedtef_loss(outputs, labels, gate, tail_score, cfg=None):
    cfg = make_fedtef_loss_config(cfg)
    logits_base = outputs["logits_base"]
    logits_fused = outputs.get("logits", outputs.get("logits_fused"))
    residual_tail = outputs["residual_tail"]

    for name, tensor in {
        "logits_base": logits_base,
        "residual_tail": residual_tail,
        "logits": logits_fused,
    }.items():
        assert_finite_tensor(name, tensor)
    if "gated_residual" in outputs:
        assert_finite_tensor("gated_residual", outputs["gated_residual"])

    loss_base = F.cross_entropy(logits_base, labels)
    loss_prior = compute_prior_base_loss(
        logits_base,
        labels,
        tail_score,
        kappa=cfg.prior_kappa,
        w_max=cfg.prior_w_max,
        eps=cfg.eps,
    )
    protected_label = gate.to(labels.device)[labels] > cfg.eps
    loss_res = controlled_hard_negative_loss(
        logits_base=logits_base,
        residual_tail=residual_tail,
        labels=labels,
        protected_label=protected_label,
        topm=cfg.hardneg_topm,
        residual_lambda=cfg.hardneg_lambda,
    )
    loss_fused = F.cross_entropy(logits_fused, labels)
    loss_safe = compute_safe_kl(
        logits_base=logits_base,
        logits_fused=logits_fused,
        labels=labels,
        gate=gate,
        conf_threshold=cfg.safe_conf_threshold,
        eps=cfg.eps,
    )
    loss = (
        cfg.w_base * loss_base
        + cfg.w_prior * loss_prior
        + cfg.w_res * loss_res
        + cfg.w_fused * loss_fused
        + cfg.w_safe * loss_safe
    )
    assert_finite_tensor("loss", loss)
    return loss, {
        "loss": float(loss.detach().item()),
        "loss_base": float(loss_base.detach().item()),
        "loss_prior": float(loss_prior.detach().item()),
        "loss_res": float(loss_res.detach().item()),
        "loss_fused": float(loss_fused.detach().item()),
        "loss_safe": float(loss_safe.detach().item()),
        "protected_samples": float(protected_label.float().sum().detach().item()),
    }
