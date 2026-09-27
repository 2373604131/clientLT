"""Label-absent donor selection with fixed class-conditioned source mixtures."""
import math
import torch


PURPOSE = ('功能保持能够改善已有尾类能力的维护，但直接监督来源仍然有限。进一步分析表明，'
           '缺少目标尾类标签的客户端也能提供正向帮助，因此我们通过 B 的定向知识迁移，'
           '增强尾类能够获得的有效学习支持。')


def resolve_tail_clients(args):
    n = int(args.num_users)
    explicit = getattr(args, 'b_directed_tail_clients', '')
    if explicit:
        ids = [int(x.strip()) for x in explicit.split(',')]
    else:
        if args.partition not in ('client-longtail', 'client-longtail-controlled'):
            raise ValueError('Directed B needs protocol tail clients; supply explicit tail-client IDs for this partition')
        ratio = float(args.head_client_ratio)
        if not math.isfinite(ratio) or not 0 < ratio < 1:
            raise ValueError('Invalid protocol head-client ratio')
        ids = list(range(int(n * ratio), n))
    if not ids or len(ids) != len(set(ids)) or any(i < 0 or i >= n for i in ids) or len(ids) == n:
        raise ValueError('Tail clients must be a unique, nonempty proper subset of clients')
    return sorted(ids)


def directed_config(base, args):
    if base.get('mode') != 'shared' or getattr(args, 'sfra_b_aggregation', 'sample') != 'sample':
        raise ValueError('Directed B requires shared mode and ordinary sample-weighted FedAvg')
    if getattr(args, 'b_problem2_variant', 'off') != 'off':
        raise ValueError('Directed B and problem2 are distinct methods')
    if getattr(args, 'b_transfer_non_tail_sampling', 'sample') != 'sample' or getattr(args, 'b_transfer_tail_weight', .5) != .5:
        raise ValueError('Directed B has no non-tail sampling or group-weight sweep')
    k, threshold = int(getattr(args, 'b_directed_topk', 3)), float(getattr(args, 'b_directed_min_gain', 1e-6))
    if k < 1 or not math.isfinite(threshold) or threshold < 0:
        raise ValueError('Invalid directed donor budget or gain threshold')
    config = dict(base)
    config.update(schema_version='donor_b_tail_directed_v1', calibration_profile='tail_directed',
        purpose=PURPOSE, target_clients=resolve_tail_clients(args), donors_per_class=k, min_gain=threshold,
        donor_threshold=threshold, probe_per_tail_class=None,
        donor_rule='external_to_target_group; absent_target_class; positive_class_macro_probe_gain; per_class_topk',
        donor_pool='fixed class-conditioned positive-gain mixtures; union is bookkeeping only',
        c_scope='one rank-by-rank C per supported target class and module; no free donor C',
        source_weights='positive probe gain normalized within selected donors of each class; frozen during calibration',
        feedback_clients='protocol tail-client group with target-tail samples',
        calibration_tail_cap=None, calibration_non_tail_cap=0, tail_only_cap=None,
        calibration_objective='target-class macro LA over all target-client tail training samples plus one C penalty',
        calibration_sampling='full target-tail training pool on every step; client-local minibatches',
        non_tail_sampling='none', tail_weight=1.,
        initialization='zero_each_event', steps=2,
        residual_normalization='divide by all observed target classes, including unsupported classes',
        calibration_probe_overlap='same full training pool; not held-out validation',
        commit_rule='fixed_second_C_step; no accuracy-based gate or checkpoint selection')
    return config


def select_sources(gains, counts, target_clients, topk, threshold):
    """gains[c][j] is class sample-mean LA reduction on target clients."""
    if topk < 1 or not math.isfinite(threshold) or threshold < 0:
        raise ValueError('Invalid source-selection settings')
    targets = set(target_clients)
    mixtures = {}
    for c, scores in sorted(gains.items()):
        valid = []
        for j, gain in scores.items():
            if j in targets or int(counts[j, c]) != 0:
                raise ValueError('Source violates external label-absent constraint')
            if not math.isfinite(gain):
                raise ValueError('Nonfinite donor probe gain')
            if gain > threshold:
                valid.append((j, gain))
        valid.sort(key=lambda pair: (-pair[1], pair[0]))
        chosen = valid[:topk]
        if chosen:
            total = sum(g for _, g in chosen)
            mixtures[c] = {j: g/total for j, g in chosen}
    return mixtures


def build_class_bases(deltas, mixtures, keys, counts, target_clients):
    """Retain class->donor edges in the forward model, not just in logs."""
    classes = sorted(mixtures)
    if not classes:
        return classes, {}
    for c in classes:
        weights = mixtures[c]
        if not weights or not math.isclose(sum(weights.values()), 1., abs_tol=1e-7):
            raise ValueError('Invalid class source weights')
        for j, weight in weights.items():
            if j in target_clients or int(counts[j, c]) != 0 or j not in deltas:
                raise ValueError('Invalid class donor edge')
            if not math.isfinite(weight) or weight <= 0:
                raise ValueError('Source weights must be finite and positive')
    bases = {key: torch.stack([sum(deltas[j][key].detach()*w for j, w in mixtures[c].items())
                              for c in classes]).detach() for key in keys}
    if not all(torch.isfinite(v).all() for v in bases.values()):
        raise ValueError('Nonfinite class source basis')
    return classes, bases


def reconstruct_class_residual(bases, matrices, target_class_count):
    if target_class_count < 1 or bases.keys() != matrices.keys():
        raise ValueError('Invalid directed residual')
    return {key: torch.bmm(basis, matrices[key]).sum(0)/target_class_count for key, basis in bases.items()}


def class_macro_loss(losses, labels, class_totals):
    """Local contribution to sample-mean-per-class, then equal class mean."""
    if losses.ndim != 1 or losses.shape != labels.shape or not class_totals or any(n <= 0 for n in class_totals.values()):
        raise ValueError('Invalid target feedback')
    if any(int(c) not in class_totals for c in labels.unique()):
        raise ValueError('Non-target label entered directed B feedback')
    return sum(losses[labels == c].sum()/(len(class_totals)*class_totals[c]) for c in class_totals)
