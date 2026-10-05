"""FedPuReL GLOBAL stage on matched CLIP-LoRA (not personalized FedPuReL).

Ports purelLoraGP.py entropy matching and TrainerBase.GradPur_backward_and_update
from the pinned official snapshot. The projection is PER PARAMETER, not on one
concatenated gradient. Zero-norm gradients explicitly keep the CE gradient.
"""
import torch
from torch.nn import functional as F


def entropy(logits):
    logits = logits.float()
    return -(F.softmax(logits, -1) * F.log_softmax(logits, -1)).sum(-1)


@torch.no_grad()
def temperature_for_entropy(logits, target, iterations=20, low=.05, high=20.):
    lo = torch.full_like(target, low, dtype=torch.float32)
    hi = torch.full_like(target, high, dtype=torch.float32)
    for _ in range(iterations):
        mid = (lo + hi) * .5
        left = entropy(logits / mid[:, None]) > target
        hi = torch.where(left, mid, hi)
        lo = torch.where(left, lo, mid)
    return (lo + hi) * .5


def losses(student, teacher, labels):
    with torch.no_grad():
        teacher = teacher.detach()
        target = .5 * (entropy(teacher) + entropy(student.detach()))
        teacher_tau = temperature_for_entropy(teacher, target)
        student_tau = temperature_for_entropy(student.detach(), target)
        pt = F.softmax(teacher / teacher_tau[:, None], -1)
    ps = F.softmax(student / student_tau[:, None], -1)
    kl = (pt * (pt.clamp_min(1e-12).log() - ps.clamp_min(1e-12).log())).sum(-1).mean()
    return F.cross_entropy(student, labels), kl


def project_gradient(ce_gradient, keep_gradient, weight=1.):
    dot = (ce_gradient * keep_gradient).sum()
    norm_sq = keep_gradient.square().sum()
    if dot < 0 and norm_sq > 0:
        return ce_gradient - weight * dot / norm_sq * keep_gradient
    return ce_gradient.clone()


def backward(ce, kl, parameters, weight=1.):
    parameters = list(parameters)
    keep = torch.autograd.grad(kl, parameters, retain_graph=True, allow_unused=True)
    task = torch.autograd.grad(ce, parameters, allow_unused=True)
    conflicts = 0
    for parameter, ga, gb in zip(parameters, task, keep):
        if ga is None:
            parameter.grad = None
        elif gb is None:
            parameter.grad = ga.detach()
        else:
            conflicts += int(((ga * gb).sum() < 0).item())
            parameter.grad = project_gradient(ga, gb, weight).detach()
    return conflicts
