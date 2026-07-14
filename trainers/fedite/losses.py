import torch
import torch.nn.functional as F


def cross_entropy_loss(logits, labels):
    return F.cross_entropy(logits.float(), labels.long())


def weighted_cross_entropy_loss(logits, labels, sample_weight=None):
    losses = F.cross_entropy(logits.float(), labels.long(), reduction="none")
    if sample_weight is None:
        return losses.mean()
    weight = sample_weight.to(logits.device, dtype=losses.dtype).detach()
    return (losses * weight).sum() / weight.sum().clamp_min(1e-6)


def hard_negative_ce_loss(logits, labels, reference_logits, topm=5):
    if logits.numel() == 0:
        return logits.sum() * 0.0
    logits = logits.float()
    reference_logits = reference_logits.detach().float()
    labels = labels.long()
    num_classes = logits.shape[1]
    if num_classes <= 1:
        return logits.sum() * 0.0
    topm = max(1, min(int(topm), num_classes - 1))
    ref = reference_logits.clone()
    ref.scatter_(1, labels.view(-1, 1), -1e4)
    hard_ids = torch.topk(ref, k=topm, dim=1).indices
    class_ids = torch.cat([labels.view(-1, 1), hard_ids], dim=1)
    local_logits = logits.gather(1, class_ids)
    local_target = torch.zeros(labels.shape[0], dtype=torch.long, device=labels.device)
    return F.cross_entropy(local_logits, local_target)


def margin_ranking_loss(logits, labels, margin=0.5):
    if logits.numel() == 0:
        return logits.sum() * 0.0
    logits = logits.float()
    labels = labels.long()
    true_logits = logits.gather(1, labels.view(-1, 1)).squeeze(1)
    negative_logits = logits.clone()
    negative_logits.scatter_(1, labels.view(-1, 1), -1e4)
    hardest_negative = negative_logits.max(dim=1).values
    return F.relu(float(margin) - (true_logits - hardest_negative)).mean()


def semantic_safety_loss(shared_image_features, base_image_features):
    shared = F.normalize(shared_image_features.float(), dim=-1)
    base = F.normalize(base_image_features.detach().float(), dim=-1)
    return (1.0 - (shared * base).sum(dim=-1)).mean()


def protected_logit_retention_loss(logits_shared, logits_base, protected_classes, temperature=2.0):
    if logits_base is None or protected_classes is None or len(protected_classes) == 0:
        return logits_shared.sum() * 0.0
    class_ids = torch.as_tensor(
        sorted(set(int(c) for c in protected_classes)),
        dtype=torch.long,
        device=logits_shared.device,
    )
    class_ids = class_ids[(class_ids >= 0) & (class_ids < logits_shared.shape[1])]
    if class_ids.numel() == 0:
        return logits_shared.sum() * 0.0
    temperature = max(float(temperature), 1e-6)
    shared = logits_shared.float().index_select(1, class_ids) / temperature
    base = logits_base.detach().float().to(logits_shared.device).index_select(1, class_ids) / temperature
    return F.kl_div(
        F.log_softmax(shared, dim=1),
        F.softmax(base, dim=1),
        reduction="batchmean",
    ) * (temperature ** 2)


def _protected_sample_mask(labels, protected_classes, num_classes, device):
    if protected_classes is None or len(protected_classes) == 0:
        return torch.zeros_like(labels, dtype=torch.bool, device=device)
    class_ids = torch.as_tensor(
        sorted(set(int(c) for c in protected_classes)),
        dtype=torch.long,
        device=device,
    )
    class_ids = class_ids[(class_ids >= 0) & (class_ids < num_classes)]
    if class_ids.numel() == 0:
        return torch.zeros_like(labels, dtype=torch.bool, device=device)
    class_mask = torch.zeros(num_classes, dtype=torch.bool, device=device)
    class_mask[class_ids] = True
    return class_mask[labels.long().clamp(0, num_classes - 1)]


def _hard_negative_ids(logits, labels, topk):
    num_classes = logits.shape[1]
    if num_classes <= 1:
        return torch.empty(labels.shape[0], 0, dtype=torch.long, device=logits.device)
    k = max(1, min(int(topk), num_classes - 1))
    scores = logits.detach().float().clone()
    scores.scatter_(1, labels.long().view(-1, 1), -1e4)
    return torch.topk(scores, k=k, dim=1).indices


def protected_boundary_retention_loss(
    logits_shared,
    logits_base,
    labels,
    protected_classes,
    topk=5,
    tolerance=0.0,
):
    if logits_base is None or logits_shared.numel() == 0:
        return logits_shared.sum() * 0.0
    labels = labels.long().to(logits_shared.device)
    num_classes = logits_shared.shape[1]
    if num_classes <= 1:
        return logits_shared.sum() * 0.0
    protected_mask = _protected_sample_mask(labels, protected_classes, num_classes, logits_shared.device)
    if not bool(protected_mask.any()):
        return logits_shared.sum() * 0.0

    shared = logits_shared.float()
    base = logits_base.detach().float().to(logits_shared.device)
    hard_shared = _hard_negative_ids(shared, labels, topk)
    hard_base = _hard_negative_ids(base, labels, topk)
    hard_ids = torch.cat([hard_shared, hard_base], dim=1)

    idx = torch.where(protected_mask)[0]
    y = labels.index_select(0, idx)
    h = hard_ids.index_select(0, idx)
    shared_y = shared.index_select(0, idx).gather(1, y.view(-1, 1))
    base_y = base.index_select(0, idx).gather(1, y.view(-1, 1))
    shared_h = shared.index_select(0, idx).gather(1, h)
    base_h = base.index_select(0, idx).gather(1, h)
    base_margin = base_y - base_h
    shared_margin = shared_y - shared_h
    margin_drop = base_margin - shared_margin
    return F.relu(margin_drop - float(tolerance)).mean()


def protected_candidate_kl_loss(
    logits_shared,
    logits_base,
    labels,
    protected_classes,
    topk=5,
    temperature=2.0,
):
    if logits_base is None or protected_classes is None or len(protected_classes) == 0:
        return logits_shared.sum() * 0.0
    labels = labels.long().to(logits_shared.device)
    num_classes = logits_shared.shape[1]
    protected_mask = _protected_sample_mask(labels, protected_classes, num_classes, logits_shared.device)
    if not bool(protected_mask.any()):
        return logits_shared.sum() * 0.0

    protected_ids = torch.as_tensor(
        sorted(set(int(c) for c in protected_classes)),
        dtype=torch.long,
        device=logits_shared.device,
    )
    protected_ids = protected_ids[(protected_ids >= 0) & (protected_ids < num_classes)]
    if protected_ids.numel() == 0:
        return logits_shared.sum() * 0.0

    shared = logits_shared.float()
    base = logits_base.detach().float().to(logits_shared.device)
    hard_shared = _hard_negative_ids(shared, labels, topk)
    hard_base = _hard_negative_ids(base, labels, topk)
    temperature = max(float(temperature), 1e-6)
    losses = []
    for row in torch.where(protected_mask)[0].tolist():
        class_ids = torch.unique(torch.cat([
            labels[row].view(1),
            hard_shared[row],
            hard_base[row],
            protected_ids,
        ]))
        shared_local = shared[row].index_select(0, class_ids) / temperature
        base_local = base[row].index_select(0, class_ids) / temperature
        losses.append(F.kl_div(
            F.log_softmax(shared_local, dim=0),
            F.softmax(base_local, dim=0),
            reduction="sum",
        ) * (temperature ** 2))
    if not losses:
        return logits_shared.sum() * 0.0
    return torch.stack(losses).mean()


def router_utility_targets(labels, protected_mask, class_state, logits_reference):
    probs = F.softmax(logits_reference.detach().float(), dim=-1)
    uncertainty = 1.0 - probs.max(dim=1).values
    labels = labels.long()
    state = class_state.to(logits_reference.device, dtype=torch.float32)
    evidence_score = state[labels, 2].clamp(0.0, 1.0)
    gate_floor = 0.2 + 0.3 * evidence_score
    # Protected samples get a reliability-controlled minimum write target; the
    # uncertainty term still opens the gate further for hard samples.
    target = protected_mask.float() * (gate_floor + (1.0 - gate_floor) * uncertainty)
    return target.clamp(0.0, 1.0)


def router_loss(gates, utility_target, positive_weight=1.0):
    if not gates:
        return utility_target.new_tensor(0.0)
    losses = []
    target = utility_target.float()
    weight = 1.0 + max(float(positive_weight) - 1.0, 0.0) * target.detach()
    for gate in gates:
        losses.append(F.binary_cross_entropy(
            gate.float().clamp(1e-6, 1.0 - 1e-6),
            target,
            weight=weight,
        ))
    return torch.stack(losses).mean()
