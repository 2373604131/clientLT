"""CAPT with fixed per-round global aggregation and isolated local optimization.

Prompt architecture is reused from trainers/capt.py. Loss and clustering follow
the pinned official source; the loss uses equivalent stable log-space math.
The upstream post-optimizer gradient mask has no effect on the update; we do
not silently introduce a new pre-optimizer mask.
"""
import numpy as np
import torch
from torch.nn import functional as F


def loss(general, specific, features, logits, labels, counts, temperature=.1):
    prior = counts.to(device=logits.device, dtype=logits.dtype)
    if bool((prior <= 0).any()):
        raise ValueError('CAPT requires the global positive class counts, not local zero counts')
    log_prior = (prior / prior.sum()).log()
    general = F.normalize(general, dim=-1)
    specific = F.normalize(specific, dim=-1)
    features = F.normalize(features, dim=-1)
    gs = features @ general.t() / temperature
    general_loss = torch.logsumexp(gs.reshape(-1), 0) - gs.mean()
    if specific.ndim == 3:
        specific = specific.mean(1)
    cs = features @ specific.t() / temperature + log_prior
    class_loss = F.cross_entropy(cs, labels)
    return general_loss + class_loss + F.cross_entropy(logits + log_prior, labels)


def clusters(proportions, n_clusters=4):
    from scipy.cluster.hierarchy import linkage, fcluster
    from sklearn.cluster import KMeans
    p = np.asarray(proportions, dtype=np.float64)
    count, classes = p.shape
    n_clusters = min(n_clusters, count)
    js = np.zeros((count, count))
    complementarity = np.zeros_like(js)
    order = np.argsort(p.sum(0))[::-1]
    head, tail = order[:int(classes * .8)], order[int(classes * .8):]
    def kl(a, b):
        a, b = np.clip(a, 1e-10, 1), np.clip(b, 1e-10, 1)
        return np.sum(a * np.log(a / b))
    for i in range(count):
        for j in range(i + 1, count):
            mid = .5 * (p[i] + p[j])
            js[i, j] = js[j, i] = .5 * (kl(p[i], mid) + kl(p[j], mid))
            comp = np.sum(p[i, head] * (1 - p[j, head])) + np.sum(p[i, tail] * (1 - p[j, tail]))
            complementarity[i, j] = complementarity[j, i] = comp
    similarity = KMeans(n_clusters=n_clusters, n_init=10).fit_predict(1 - js)
    if count < 2 or complementarity.max() <= 0:
        return similarity, np.ones(count, dtype=int)
    distances = 1 - complementarity / complementarity.max()
    dissimilarity = fcluster(linkage(distances[np.triu_indices(count, 1)], method='complete'),
                            n_clusters, criterion='maxclust')
    return similarity, dissimilarity


def aggregate(local_states, client_ids, counts, global_state, n_clusters=4):
    """Upstream cluster averaging, >0.1 class support rule, then weighted FedAvg."""
    from trainers.baselines.common import weighted_average
    sizes = np.asarray([sum(counts[i]) for i in client_ids], dtype=np.float64)
    proportions = np.asarray([counts[i] for i in client_ids], dtype=np.float64) / sizes[:, None]
    similar, dissimilar = clusters(proportions, n_clusters)
    states = {i: {k: v.clone() for k, v in local_states[i].items()} for i in client_ids}
    specific, general = 'prompt_learner.class_aware_ctx', 'prompt_learner.general_ctx'
    for grouping, key in ((similar, specific), (dissimilar, general)):
        for group in set(grouping):
            members = [client_ids[j] for j in np.flatnonzero(grouping == group)]
            average = torch.stack([states[i][key] for i in members]).mean(0)
            for i in members:
                states[i][key] = average.clone()
    class_prompt = global_state[specific].clone()
    for c in range(class_prompt.shape[0]):
        members = [client_ids[j] for j in np.flatnonzero(proportions[:, c] > .1)]
        if members:
            class_prompt[c] = torch.stack([states[i][specific][c] for i in members]).mean(0)
    for i in client_ids:
        states[i][specific] = class_prompt.clone()
    return weighted_average(states, client_ids, sizes.tolist())
