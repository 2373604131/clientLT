"""Necessary positive-margin retention. No source attribution or candidate gate."""
import copy
import math

import torch

from utils.sfra_math import require_finite

VARIANT = 'margin-keep-v1'
ARM = 'method_a_margin'
DEFAULTS = dict(epsilon=.01, margin_scale=.01, keep_weight=1., learn_weight=1.)


def algorithm_config(**parameters):
    if set(parameters) - set(DEFAULTS):
        raise ValueError('Unknown necessary-margin parameter')
    values = dict(DEFAULTS, **parameters)
    if any(not math.isfinite(v) or v <= 0 for v in values.values()) or values['epsilon'] > 2:
        raise ValueError('Margin parameters must be finite and positive; epsilon must be <= 2')
    return dict(schema_version=VARIANT, variant=VARIANT, **values,
        retention_weight=values['keep_weight'], correction_steps=3, correction_step_size=.1,
        witness_per_local_class=8, witness_views=2, history_window=5, history_improvement=0.,
        history='sample_positive_nonoverlapping_window_min', history_stop_round=90,
        target='min(positive_history, epsilon)', normalization='fixed_shared_margin_scale',
        history_weights='class_balanced_valid_history_training_mass',
        classification_objective='all_class_macro_original_LA_all_fixed_witnesses',
        classification_group_weights='group_class_count / total_class_count',
        source_feedback=False, token_gradient_norms=False, candidate_checks=False,
        commit_rule='fixed_third_step_without_acceptance_or_rollback',
        trigger='positive_unweighted_keep_loss_at_ordinary',
        radius='unweighted_client_proposal_RMS', refresh_rounds=list(range(1, 91)),
        precision='fp32', parameter_status='exploratory_fixed_before_run_not_test_selected')


class SampleMarginHistory:
    """Reliable submitted sample margins; the current round cannot use its own history."""
    def __init__(self, initial_margins, manifest):
        margins = torch.as_tensor(initial_margins).detach().float().cpu()
        require_finite(initial_margins=margins)
        self.manifest = copy.deepcopy(manifest)
        if margins.shape != (len(manifest), 2):
            raise ValueError('Margin history witness identities/shape differ')
        self.class_ids = torch.tensor([s['class_id'] for s in manifest], dtype=torch.long)
        self.valid = (margins > 0).all(1)
        self.H = torch.where(self.valid, margins.amin(1), 0.)
        self.registered_round = torch.where(self.valid, 0, -1)
        self.block_min = torch.full_like(self.H, float('inf'))
        self.block_count = self.last_committed_round = 0

    def commit(self, margins, rnd):
        if rnd != self.last_committed_round + 1 or not 1 <= rnd <= 90:
            raise ValueError('History must advance exactly once per submitted A round')
        margins = torch.as_tensor(margins).detach().float().cpu()
        if margins.shape != (len(self.H), 2):
            raise ValueError('Submitted sample margins have wrong shape')
        require_finite(submitted_margins=margins)
        self.block_min = torch.minimum(self.block_min, margins.amin(1))
        self.block_count += 1
        changed = torch.zeros_like(self.valid)
        if rnd % 5 == 0:
            if self.block_count != 5:
                raise ValueError('Incomplete five-round history window')
            changed = (self.block_min > 0) & (~self.valid | (self.block_min > self.H))
            self.H[changed] = self.block_min[changed]
            self.valid |= changed
            self.registered_round[changed] = rnd
            self.block_min.fill_(float('inf'))
            self.block_count = 0
        self.last_committed_round = rnd
        return changed

    def state_dict(self):
        return dict(schema=VARIANT, **copy.deepcopy(vars(self)))

    @classmethod
    def restore(cls, state, manifest):
        if state.get('schema') != VARIANT or state.get('manifest') != manifest:
            raise ValueError('Margin history schema or fixed witness identities differ')
        result = cls(torch.zeros(len(manifest), 2), manifest)
        if set(state) != set(result.state_dict()):
            raise ValueError('Margin history checkpoint fields differ')
        for name in ('H', 'valid', 'class_ids', 'registered_round', 'block_min'):
            if state[name].shape != (len(manifest),) or state[name].dtype != getattr(result, name).dtype:
                raise ValueError('Margin history tensor shape/dtype differs: '+name)
        if not torch.equal(state['class_ids'], result.class_ids):
            raise ValueError('Margin history class identities differ')
        for name, value in state.items():
            if name != 'schema':
                setattr(result, name, copy.deepcopy(value))
        if (not 0 <= result.last_committed_round <= 90
                or result.block_count != result.last_committed_round % 5):
            raise ValueError('Margin history window/round mismatch')
        require_finite(history=result.H)
        if ((result.H[result.valid] <= 0).any() or (result.H[~result.valid] != 0).any()
                or (result.registered_round[result.valid] < 0).any()
                or (result.registered_round > result.last_committed_round).any()
                or (result.registered_round[~result.valid] != -1).any()):
            raise ValueError('Invalid reliable margin history')
        if result.block_count:
            require_finite(window_minimum=result.block_min)
        elif not torch.isposinf(result.block_min).all():
            raise ValueError('Empty history window must have reset minima')
        return result


def make_targets(history, base_mass, epsilon, margin_scale):
    algorithm_config(epsilon=epsilon, margin_scale=margin_scale)
    mass = torch.as_tensor(base_mass).detach().float().cpu()
    if mass.shape != history.H.shape or (mass <= 0).any():
        raise ValueError('Each fixed witness needs positive training mass')
    require_finite(witness_mass=mass)
    valid = history.valid.clone()
    weights = torch.zeros_like(history.H)
    classes = history.class_ids[valid].unique()
    for c in classes:
        mask = valid & (history.class_ids == c)
        weights[mask] = mass[mask] / mass[mask].sum() / len(classes)
    target = torch.where(valid, history.H.clamp_max(epsilon), 0.)
    return dict(valid=valid, target=target[:, None].expand(-1, 2).clone(),
                omega=weights, sigma=torch.tensor(float(margin_scale)))


def keep_terms(margins, target, sigma, omega):
    return .25 * omega * ((target - margins).clamp_min(0) / sigma).square()
