"""Read-only measurements around the unchanged margin-A correction."""
import copy
import json
import math
from pathlib import Path

import torch

def norm(value):
    return float(value.detach().double().norm())


def difference(left, right, keys):
    return torch.cat([(left[k].detach().cpu().double()-right[k].detach().cpu().double()).flatten()
                      for k in keys])


def transitions(before, after):
    def counts(a, b):
        return dict(count=a.numel(), correct_to_wrong=int((a & ~b).sum()),
                    wrong_to_correct=int((~a & b).sum()))
    return dict(views=counts(before, after), original_view=counts(before[:, 0], after[:, 0]),
                both_views_correct=counts(before.all(1), after.all(1)))


class AUpdateDiagnostic:
    def __init__(self, output):
        self.output = Path(output)
        self.result = dict(schema='margin_a_update_diagnostic_v1', steps=[])

    def begin(self, runtime, middle, ordinary, radius, rnd, targets, before):
        self.rt, self.middle, self.ordinary = runtime, middle, ordinary
        self.radius, self.targets, self.before = radius, targets, before
        self.lr = runtime.sfra_config['correction_step_size']
        self.result.update(round=rnd, history_last_used=runtime.history.last_committed_round,
            valid_history_count=int(runtime.history.valid.sum()), violated_count=before['deficit_count'],
            L_keep=before['keep_loss'], C_label=before['class_loss'], R_t=radius,
            lambda_keep=runtime.sfra_config['keep_weight'], lambda_learn=runtime.sfra_config['learn_weight'],
            optimizer='manual projected gradient descent, no momentum/Adam/clipping',
            parameterization='A = A_ord + R_t*z', objective='0.5*||z||^2 + lambda_keep*L_keep + lambda_learn*C_label',
            step_size=self.lr, first_step_effective_A_learning_rate=self.lr*radius**2,
            ordinary_delta_norm=norm(difference(ordinary, middle, runtime.a_keys)),
            z_initial_norm=0., gradient_space='weighted A gradients; z gradients = R_t * A gradients',
            target_weight_sum=float(targets['omega'].sum()),
            witness_count=len(runtime.margin_feedback.manifest))

    def step(self, index, previous, actual, measurement, submitted):
        g = measurement['gradient']
        keep, learn = (measurement['component_gradients'][k] for k in ('keep', 'learn'))
        nk, nl = norm(keep), norm(learn)
        cosine = float(torch.dot(keep.double(), learn.double())/(nk*nl)) if nk and nl else None
        if index == 0:
            self.first_gradients = {name: value.detach().cpu().double().clone()
                                    for name, value in (('keep', keep), ('learn', learn), ('total', g))}
        gz = previous + self.radius*g
        raw = previous - self.lr*gz
        predicted = raw/raw.norm().clamp_min(1.)
        residual = norm(g-keep-learn)
        row = dict(step=index+1, L_keep=measurement['keep_loss'], C_label=measurement['class_loss'],
            grad_keep_norm_A=nk, grad_learn_norm_A=nl, grad_total_norm_A=norm(g), grad_cosine=cosine,
            grad_angle_degrees=math.degrees(math.acos(max(-1., min(1., cosine)))) if cosine is not None else None,
            total_to_sum_component_norm_ratio=norm(g)/(nk+nl) if nk+nl else None,
            keep_directional_derivative_along_negative_total=-float(torch.dot(keep.double(), g.double())),
            learn_directional_derivative_along_negative_total=-float(torch.dot(learn.double(), g.double())),
            grad_keep_norm_z=abs(self.radius)*nk, grad_learn_norm_z=abs(self.radius)*nl,
            grad_total_norm_z=norm(gz), proximity_gradient_norm=norm(previous),
            component_sum_error_norm=residual, component_sum_relative_error=residual/max(norm(g), 1e-30),
            raw_z_norm=norm(raw), projection_triggered=float(raw.norm()) > 1.,
            predicted_z_norm=norm(predicted), actual_z_norm=norm(actual),
            predicted_actual_z_error=norm(predicted-actual),
            predicted_increment_A_norm=norm(self.radius*(predicted-previous)),
            actual_increment_A_ideal_norm=norm(self.radius*(actual-previous)),
            actual_cumulative_A_norm=norm(difference(submitted, self.ordinary, self.rt.a_keys)),
            diagnostic_extra_backward_calls=measurement['diagnostic_extra_backward_calls'],
            predicted_first_SGD_A_norm=norm(-self.lr*self.radius**2*g) if index == 0 else None)
        if index == 0:
            actual_delta = difference(submitted, self.ordinary, self.rt.a_keys)
            g_cpu = self.first_gradients['total']
            row.update(first_actual_A_vs_predicted_error_norm=norm(actual_delta+self.lr*self.radius**2*g_cpu),
                first_actual_effective_A_step=(-float(torch.dot(actual_delta, g_cpu))/norm(g_cpu)**2
                                              if norm(g_cpu) else None))
        self.result['steps'].append(row)

    def finish(self, submitted, z, final, summary):
        keys = self.rt.a_keys
        delta = difference(submitted, self.ordinary, keys)
        intended = (self.radius*z).detach().cpu().double()
        actual = self.rt.trainer.model.state_dict()
        write_error = max(float((actual[k].detach().cpu().double()-submitted[k].detach().cpu().double()).abs().max())
                          for k in keys)
        layers = []
        modules = dict(self.rt.trainer.model.named_modules())
        for key in keys:
            module = modules[key.rsplit('.', 1)[0]]
            scale = float(module.scaling)
            bkey = key[:-1]+'B'
            if not torch.equal(self.middle[bkey], self.ordinary[bkey]) or not torch.equal(self.ordinary[bkey], submitted[bkey]):
                raise ValueError('B changed during A-only correction')
            b = self.ordinary[bkey].detach().cpu().double()
            dr = submitted[key].detach().cpu().double()-self.ordinary[key].detach().cpu().double()
            do = self.ordinary[key].detach().cpu().double()-self.middle[key].detach().cpu().double()
            layers.append(dict(layer=key, scaling=scale, repair_effective_norm=norm(scale*(b@dr)),
                               ordinary_effective_norm=norm(scale*(b@do))))
        repair_w = math.sqrt(sum(x['repair_effective_norm']**2 for x in layers))
        ordinary_w = math.sqrt(sum(x['ordinary_effective_norm']**2 for x in layers))
        ratio = lambda a, b: a/b if b else None
        self.result.update(delta_A_norm=norm(delta), ideal_R_z_norm=norm(intended),
            writeback_rounding_error_norm=norm(delta-intended), loaded_model_writeback_max_abs_error=write_error,
            update_ratio=ratio(norm(delta), self.result['ordinary_delta_norm']),
            correction_z_norm=norm(z), correction_steps=summary['correction_steps'], skip_reason=summary['skip_reason'],
            B_unchanged=True, effective_weight_layers=layers, repair_effective_weight_norm=repair_w,
            ordinary_effective_weight_norm=ordinary_w, effective_weight_ratio=ratio(repair_w, ordinary_w),
            L_keep_after=final['keep_loss'], C_label_after=final['class_loss'],
            violated_count_after=final['deficit_count'],
            L_keep_change=final['keep_loss']-self.before['keep_loss'],
            C_label_change=final['class_loss']-self.before['class_loss'],
            group_classification={name: dict(before=float(old), after=float(new), change=float(new)-float(old))
                for name, old, new in zip(('Head', 'Middle', 'Tail'), self.before['group_ce'], final['group_ce'])},
            raw_classifier_transitions=transitions(self.before['correct'], final['correct']),
            LA_classifier_transitions=transitions(self.before['la_correct'], final['la_correct']))
        valid = self.targets['valid'][:, None].expand(-1, 2)
        gap_summary = {}
        for phase, obs in (('before', self.before), ('after', final)):
            gaps = (self.targets['target']-obs['sample_margins']).clamp_min(0)[valid]
            gap_summary[phase] = dict(mean=float(gaps.mean()) if gaps.numel() else 0.,
                                      max=float(gaps.max()) if gaps.numel() else 0., violated_views=int((gaps>0).sum()))
        self.result['history_gaps'] = gap_summary
        if hasattr(self, 'first_gradients'):
            self.result['first_order_loss_changes_for_actual_final_delta'] = {
                name: float(torch.dot(gradient, delta)) for name, gradient in self.first_gradients.items()}
            self.result['final_delta_cosine_with_negative_initial_total_gradient'] = (
                -float(torch.dot(self.first_gradients['total'], delta))/(norm(self.first_gradients['total'])*norm(delta))
                if norm(self.first_gradients['total'])*norm(delta) else None)
        self.output.mkdir(parents=True, exist_ok=True)
        (self.output/'diagnosis.json').write_text(json.dumps(self.result, ensure_ascii=False, indent=2,
                                                            allow_nan=False)+'\n', encoding='utf-8')
        torch.save(dict(round=self.result['round'], ordinary_lora={k:self.ordinary[k].detach().cpu() for k in self.rt.keys},
                        submitted_lora={k:submitted[k].detach().cpu() for k in self.rt.keys}, z=z.detach().cpu()),
                   self.output/'diagnosed_commit.pt')


def diagnose_prepared(runtime, middle, ordinary, radius, rnd, output):
    """Run only the original fixed correction; preserve the supplied history/RNG.

    `middle` is post-B/pre-ordinary-A. History must be from round rnd-1.
    The caller provides the production model, witness bank and exact saved tensors.
    No local optimizer, checkpoint, publish or training-loop entry is invoked.
    """
    from utils.cliplora_a_refresh import isolated_rng
    from utils.cliplora_functional_feedback import snapshot
    if runtime.history.last_committed_round != rnd-1:
        raise ValueError('Diagnosis requires the immediately preceding history, never final-round history')
    if runtime.job['aggregation'].get('rule', 'joint') != 'joint' or runtime.b_transfer is not None:
        raise ValueError('Diagnosis requires joint + margin-A and no new B')
    if not math.isfinite(radius) or radius < 0:
        raise ValueError('Invalid saved proposal radius')
    diagnostic = AUpdateDiagnostic(output)
    history, saved = runtime.history, snapshot(runtime.trainer.model)
    costs = len(runtime.costs)
    runtime.history = copy.deepcopy(history)
    try:
        with isolated_rng():
            submitted, row = runtime.correct_ordinary(middle, ordinary, radius, rnd, diagnostic)
        return submitted, row, diagnostic.result
    finally:
        runtime.history = history
        runtime.load_training_state(saved)
        del runtime.costs[costs:]
