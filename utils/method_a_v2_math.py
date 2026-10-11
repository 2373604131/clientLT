"""Fixed source-memory-v2 contract; no test-driven choices or optimizer state."""
import copy

import torch

from utils.sfra_math import require_finite

VARIANT = 'source-memory-v2'
CONFIG = dict(schema_version=VARIANT, variant=VARIANT, retention_weight=10.,
    correction_steps=3, correction_step_size=.1, witness_per_local_class=8, witness_views=2,
    positive_response_threshold=1e-6, history_window=5, history_improvement=0.,
    sigma_floor=.001, radius='unweighted_client_proposal_RMS',
    refresh_rounds=list(range(1, 91)), history_stop_round=90,
    history='sample_positive_nonoverlapping_window_min_bound_to_source_snapshot',
    source='threshold_raw_then_actual_aggregation_weight_then_window_sum',
    normalization='class_balanced_valid_history_training_mass_no_second_weight',
    classification_objective='three_group_class_macro_original_LA_all_fixed_witnesses',
    learning_loss_fraction=.1, candidate_fractions=[1., .5],
    comparison='x <= y + 1e-7 * max(1, abs(y))',
    commit_rule='first_candidate_passing_all_three_CE_and_J_else_exact_ordinary',
    precision='fp32', source_coordinates='untruncated_reduced_QR')


def leq(x, y):
    x, y = torch.as_tensor(x), torch.as_tensor(y)
    require_finite(comparison_left=x, comparison_right=y)
    return x <= y + 1e-7 * torch.maximum(torch.ones_like(y), y.abs())


def positive_support(responses, actual_weights):
    """Q x view x source, with sources already aligned by client ID."""
    responses = torch.as_tensor(responses).detach()
    w = torch.as_tensor(actual_weights, dtype=responses.dtype, device=responses.device)
    require_finite(source_responses=responses, source_weights=w)
    if responses.ndim != 3 or responses.shape[1] != 2 or w.shape != responses.shape[2:]:
        raise ValueError('Source client/view dimensions differ')
    if (w < 0).any() or not torch.isclose(w.sum(), w.new_tensor(1.), atol=1e-6, rtol=0):
        raise ValueError('Invalid actual aggregation weights')
    raw = responses.amin(1)
    return torch.where(raw > 1e-6, raw, 0.) * w


class SampleSourceHistory:
    """Unknown pi/kappa are NaN storage sentinels, never operands in the loss."""
    def __init__(self, initial_margins, sample_manifest, token_count, clients):
        margins = torch.as_tensor(initial_margins).detach().float().cpu()
        require_finite(initial_margins=margins)
        self.manifest = copy.deepcopy(sample_manifest)
        if margins.shape != (len(self.manifest), 2):
            raise ValueError('Witness identity/shape mismatch')
        self.token_ids = torch.tensor([s['token_id'] for s in self.manifest])
        self.class_ids = torch.tensor([s['class_id'] for s in self.manifest])
        self.H = margins.amin(1).clamp_min(0).clone()
        self.valid = (margins > 0).all(1)
        self.known = torch.zeros_like(self.valid)
        self.reason = ['pretrained_unknown' if v else 'invalid' for v in self.valid.tolist()]
        self.pi = torch.full((len(margins), clients), float('nan'))
        self.kappa = torch.full((len(margins),), float('nan'))
        self.formation_start = torch.full((len(margins),), -1, dtype=torch.long)
        self.formation_end = self.formation_start.clone()
        self.registered_round = torch.where(self.valid, 0, -1)
        self.block_min = torch.full_like(self.H, float('inf'))
        self.sum_a = torch.zeros(token_count, clients)
        self.block_count = self.last_committed_round = 0

    def commit(self, margins, support, rnd):
        if rnd != self.last_committed_round + 1 or not 1 <= rnd <= 90:
            raise ValueError('History must advance exactly once per submitted A round')
        margins = torch.as_tensor(margins).detach().float().cpu()
        support = torch.as_tensor(support).detach().float().cpu()
        require_finite(submitted_margins=margins, submitted_support=support)
        if margins.shape != (len(self.H), 2) or support.shape != self.sum_a.shape or (support < 0).any():
            raise ValueError('History observation shape/sign mismatch')
        self.block_min = torch.minimum(self.block_min, margins.amin(1))
        self.sum_a += support
        require_finite(window_support=self.sum_a)
        self.block_count += 1
        changed = torch.zeros_like(self.valid)
        if rnd % 5 == 0:
            if self.block_count != 5:
                raise ValueError('Incomplete five-round formation window')
            changed = (self.block_min > 0) & (~self.valid | (self.block_min > self.H))
            total = self.sum_a.sum(1)
            supported = total > 0  # No second threshold, even with tiny joint weights.
            pi = torch.zeros_like(self.sum_a)
            pi[supported] = self.sum_a[supported] / total[supported, None]
            known = supported[self.token_ids]
            self.H[changed] = self.block_min[changed]
            self.valid |= changed
            self.known[changed] = known[changed]
            self.pi[changed] = float('nan')
            self.kappa[changed] = float('nan')
            registered_known = changed & known
            self.pi[registered_known] = pi[self.token_ids[registered_known]].clone()
            self.kappa[registered_known] = self.pi[registered_known].square().sum(1)
            self.formation_start[changed] = rnd - 4
            self.formation_end[changed] = rnd
            self.registered_round[changed] = rnd
            for i in changed.nonzero().flatten().tolist():
                self.reason[i] = 'window_support' if bool(known[i]) else 'no_positive_support'
            self.sum_a.zero_()
            self.block_min.fill_(float('inf'))
            self.block_count = 0
        self.last_committed_round = rnd
        return changed

    def state_dict(self):
        return dict(schema=VARIANT, **copy.deepcopy(vars(self)))

    @classmethod
    def restore(cls, state, manifest, token_count, clients):
        if state.get('schema') != VARIANT or state.get('manifest') != manifest:
            raise ValueError('Av2 checkpoint schema or fixed witness identities differ')
        result = cls(torch.zeros(len(manifest), 2), manifest, token_count, clients)
        for name in ('token_ids','class_ids'):
            if not torch.equal(state[name],getattr(result,name)):
                raise ValueError('Av2 checkpoint sample mapping differs')
        for name in ('H','valid','known','kappa','formation_start','formation_end','registered_round','block_min'):
            if state[name].shape != (len(manifest),):
                raise ValueError('Av2 checkpoint sample state shape differs: '+name)
        for name, value in state.items():
            if name != 'schema':
                setattr(result, name, copy.deepcopy(value))
        if (result.sum_a.shape != (token_count, clients) or result.pi.shape != (len(manifest), clients)
                or not 0 <= result.last_committed_round <= 90
                or result.block_count != result.last_committed_round % 5):
            raise ValueError('Av2 checkpoint window dimensions/count differ')
        require_finite(history=result.H, source_sum=result.sum_a,
                       known_kappa=result.kappa[result.known], known_pi=result.pi[result.known])
        return result


def make_targets(history, base_mass, token_norms, radius):
    """Ordinary margins/targets are filled during its single gradient forward."""
    valid = history.valid.clone()
    omega = torch.zeros_like(history.H)
    mass = torch.as_tensor(base_mass).float().cpu()
    classes = history.class_ids[valid].unique()
    for c in classes:
        mask = valid & (history.class_ids == c)
        omega[mask] = mass[mask] / mass[mask].sum() / len(classes)
    effective = torch.ones_like(history.H)
    known = valid & history.known
    effective[known] = history.kappa[known]
    sigma = (float(radius) * token_norms.cpu()).clamp_min(.001)[history.token_ids]
    require_finite(effective_source=effective, sigma=sigma, history_weights=omega)
    return dict(H=history.H.clone(), effective=effective, valid=valid, omega=omega,
                sigma=sigma, target=torch.zeros(len(valid), 2))


def history_terms(margins, target, sigma, omega):
    return .25 * omega * ((target - margins).clamp_min(0) / sigma).square()


def candidate_reasons(group_ce, bounds, loss, beta, z, j0):
    j = .5 * beta**2 * float(z.square().sum()) + 10 * float(loss)
    passed = leq(group_ce, bounds).tolist()
    reasons = [name for name, ok in zip(('H', 'M', 'T'), passed) if not ok]
    if not bool(leq(j, j0)):
        reasons.append('J')
    return j, '|'.join(reasons)
