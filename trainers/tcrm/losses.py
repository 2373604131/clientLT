import torch
import torch.nn.functional as F


def logit_adjustment(class_prior, tau=1.0, device=None):
    prior = torch.as_tensor(class_prior, dtype=torch.float32, device=device).clamp_min(1e-12)
    prior = prior / prior.sum().clamp_min(1e-12)
    return float(tau) * prior.log()


def cross_entropy_with_optional_adjustment(logits, labels, class_prior=None, tau=0.0):
    if class_prior is not None and float(tau) != 0.0:
        logits = logits.float() + logit_adjustment(class_prior, tau=tau, device=logits.device).view(1, -1)
    return F.cross_entropy(logits.float(), labels.long())


def true_class_margin(logits, labels):
    logits = logits.float()
    labels = labels.long()
    true = logits.gather(1, labels.view(-1, 1)).squeeze(1)
    negatives = logits.clone()
    negatives.scatter_(1, labels.view(-1, 1), -1e4)
    hardest = negatives.max(dim=1).values
    return true - hardest


def hbs_loss(hybrid_logits, zero_shot_logits, labels, tail_class_ids, non_tail_class_ids, epsilon=0.0):
    tail_set = set(int(c) for c in tail_class_ids)
    mask = torch.as_tensor([int(y) in tail_set for y in labels.detach().cpu().tolist()], dtype=torch.bool, device=labels.device)
    if not bool(mask.any()) or not non_tail_class_ids:
        return hybrid_logits.sum() * 0.0
    hybrid_t = hybrid_logits.float()[mask]
    zero_t = zero_shot_logits.detach().float().to(hybrid_logits.device)[mask]
    labels_t = labels.long()[mask]
    head_ids = torch.as_tensor(non_tail_class_ids, dtype=torch.long, device=hybrid_logits.device)
    zero_tail = zero_t.gather(1, labels_t.view(-1, 1)).squeeze(1)
    zero_head = zero_t.index_select(1, head_ids).max(dim=1).values
    hybrid_tail = hybrid_t.gather(1, labels_t.view(-1, 1)).squeeze(1).detach()
    hybrid_head = hybrid_t.index_select(1, head_ids).max(dim=1).values
    margin_zero = zero_tail - zero_head
    margin_hybrid = hybrid_tail - hybrid_head
    return F.relu(margin_zero - margin_hybrid - float(epsilon)).mean()
