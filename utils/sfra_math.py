"""FP32, graph-free state and algebra for SFRA and its classification regularizer."""
import torch


def require_finite(**values):
    for name, value in values.items():
        if not torch.isfinite(value).all():
            raise FloatingPointError(f'SFRA nonfinite {name}')


def flatten(state, keys):
    return torch.cat([state[k].detach().float().reshape(-1) for k in keys])


def unflatten(vector, template, keys):
    result, offset = {}, 0
    for key in keys:
        n = template[key].numel()
        result[key] = vector[offset:offset+n].reshape_as(template[key]).detach().clone()
        offset += n
    assert offset == vector.numel()
    return result


def proposal_coordinates(deltas):
    """Columns are clients in a fixed order; no rank truncation or reweighting."""
    require_finite(deltas=deltas)
    q, c = torch.linalg.qr(deltas.float(), mode='reduced')
    radius = deltas.float().square().sum(0).mean().sqrt()
    return q, c, radius


def source_statistics(responses, cache, sample_weights):
    """responses: token x view x client. Cache is updated only with support."""
    require_finite(responses=responses)
    conservative = responses.amin(1)
    positive = torch.where(conservative > 1e-6, conservative, 0.)
    count = (positive > 0).sum(1)
    supported = count > 0
    total = positive.sum(1)
    u = total / count.clamp_min(1)
    effective = torch.zeros_like(total)
    effective[supported] = total[supported].square() / positive[supported].square().sum(1)
    updated = cache.clone()
    updated[supported] = effective[supported].reciprocal()
    predicted = responses @ sample_weights
    return dict(supported=supported, positive_count=count, u=u, n_eff=effective,
                cache=updated, predicted=predicted)


def make_targets(scores, source, history, history_valid, variant):
    current = (scores + .5 * source['u'][:, None]).clamp_max(2.)
    has_current = source['supported']
    has_history = history_valid if variant != 'current' else torch.zeros_like(history_valid)
    active = has_current | has_history
    # Inactive entries are storage placeholders, never baseline-maintenance targets.
    target = torch.zeros_like(scores)
    target[has_current] = current[has_current]
    only_history = has_history & ~has_current
    target[only_history] = history[only_history, None]
    both = has_current & has_history
    target[both] = torch.maximum(target[both], history[both, None])
    raw = torch.ones_like(history) if variant == 'flat' else source['cache'].clone()
    raw[~active] = 0
    weights = raw / raw.sum() if active.any() else raw
    return dict(current=current, target=target, active=active, weights=weights)


def functional_loss(scores, target, sigma, weights):
    """sum_q omega_q / (2 V) sum_v positive((T-F)/sigma)^2."""
    return (weights * .5 * ((target-scores)/sigma).clamp_min(0).square().mean(-1)).sum()


def classification_preservation(loss, reference, scale):
    """One-sided GLOBAL LA loss increase; inputs are detached round statistics."""
    gap = max(float(loss) - float(reference), 0.)
    return .5 * (gap / scale)**2, gap / scale**2


def projected_step(z, radius, gradient_a, strength, step_size=.1, auxiliary_gradient=None):
    gradient_z = z + strength * radius * gradient_a
    if auxiliary_gradient is not None:
        gradient_z = gradient_z + radius * auxiliary_gradient
    require_finite(gradient_z=gradient_z)
    updated = z - step_size * gradient_z
    return (updated / updated.norm().clamp_min(1.)).detach()


class FunctionalHistory:
    def __init__(self, initial_scores, clients):
        self.reference = initial_scores.amin(1).detach().cpu().clone()
        self.level = torch.zeros_like(self.reference)
        self.valid = torch.zeros_like(self.reference, dtype=torch.bool)
        self.source_cache = torch.full_like(self.reference, 1. / clients)
        self.block_min = torch.full_like(self.reference, 2.)
        self.block_count = 0

    def commit(self, scores, round_id):
        require_finite(committed_scores=scores)
        self.block_min = torch.minimum(self.block_min, scores.cpu().amin(1))
        self.block_count += 1
        changed = torch.zeros_like(self.valid)
        if round_id % 5 == 0:
            assert self.block_count == 5
            baseline = torch.where(self.valid, self.level, self.reference)
            changed = self.block_min >= baseline + .001
            self.level[changed] = self.block_min[changed]
            self.valid |= changed
            self.block_min.fill_(2.)
            self.block_count = 0
        return changed

    def state_dict(self):
        return {key: value.clone() if torch.is_tensor(value) else value
                for key, value in vars(self).items()}

    def load_state_dict(self, state):
        for key, value in state.items():
            setattr(self, key, value.clone() if torch.is_tensor(value) else value)
