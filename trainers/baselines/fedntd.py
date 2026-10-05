"""FedNTD loss, adapted from the pinned MIT-licensed official implementation.

See third_party/paper_baselines/fedntd/algorithms/fedntd/{criterion,utils}.py.
The teacher is the frozen round-start global model, not zero-shot CLIP.
"""
import torch
from torch.nn import functional as F


def not_true_logits(logits, labels):
    mask = torch.ones_like(logits, dtype=torch.bool)
    mask.scatter_(1, labels[:, None], False)
    return logits[mask].reshape(logits.shape[0], logits.shape[1] - 1)


def loss(logits, labels, teacher_logits, temperature=3., beta=1.):
    if temperature <= 0 or beta < 0 or logits.shape[1] < 2:
        raise ValueError('Invalid FedNTD parameters')
    student = F.log_softmax(not_true_logits(logits, labels) / temperature, dim=1)
    teacher = F.softmax(not_true_logits(teacher_logits.detach(), labels) / temperature, dim=1)
    distillation = temperature ** 2 * F.kl_div(student, teacher, reduction='batchmean')
    return F.cross_entropy(logits, labels) + beta * distillation
