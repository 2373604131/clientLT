"""Independent new A: ordinary learning, gap-triggered joint updates, direct commit."""
import time

import torch

from utils.cliplora_a_refresh import train_only
from utils.cliplora_bridge_audit import write_json
from utils.cliplora_functional_feedback import snapshot
from utils.cliplora_sfra import SFRARuntime
from utils.method_a_margin_feedback import MarginFeedback
from utils.method_a_margin_math import VARIANT, DEFAULTS, algorithm_config, SampleMarginHistory, make_targets
from utils.sfra_math import flatten, unflatten, projected_step, require_finite


class MethodAMarginRuntime(SFRARuntime):
    def configure_algorithm(self):
        config = algorithm_config(
            correction_step_size=getattr(self.args, 'a_margin_correction_step_size', .1),
            **{k: getattr(self.args, 'a_margin_'+k, v) for k, v in DEFAULTS.items()})
        if (self.variant != VARIANT or self.strength != config['keep_weight']
                or getattr(self.args, 'b_transfer_enable', False)
                or getattr(self.args, 'sfra_b_aggregation', 'sample') != 'sample'
                or not getattr(self.args, 'sfra_fast_execution_v2', False)):
            raise ValueError('New A requires its independent joint worker, registered weights/cache, and no method B')
        self.sfra_config = dict(config, base_training_config=self.config,
            seed=self.args.seed, protocol_seed=self.args.split_seed, partition=self.args.partition,
            witness_batch_size=self.args.sfra_witness_batch_size, a_parameter_order=self.a_keys)

    def initialize_feedback(self):
        self.margin_feedback = MarginFeedback(self.bank, self.audit.counts, (self.head, self.middle, self.tail),
                                              self.trainer.training_logit_adjustment)
        if self.resume_payload and self.resume_payload.get('schema_version') != VARIANT:
            raise ValueError('New A cannot restore an old Full-CP/Av2 checkpoint')
        write_json(self.root/'margin_witness_manifest.json', self.margin_feedback.manifest)

    def history_from_state(self, state):
        history = SampleMarginHistory.restore(state, self.margin_feedback.manifest)
        if history.last_committed_round != min(self.resume_payload['completed_round'], 90):
            raise ValueError('Margin history and submitted model rounds differ')
        return history

    def observe_margin(self, state, rnd, phase, **kwargs):
        self.load_training_state(state)
        print(f'MARGIN-A round={rnd} phase={phase}', flush=True)
        observed = self.margin_feedback.evaluate((rnd, phase),
            keep_weight=self.sfra_config['keep_weight'], learn_weight=self.sfra_config['learn_weight'], **kwargs)
        a_bytes = sum(state[k].numel()*4 for k in self.a_keys)
        self.costs.append(dict(round=rnd, phase=phase, seconds=observed['seconds'],
            forward_images=observed['forward_images'], backward_images=observed['backward_images'],
            backward_calls=observed['backward_calls'], batches=observed['batches'],
            downlink_bytes=len(self.sizes)*a_bytes,
            upload_bytes=len(self.margin_feedback.manifest)*2*4 + len(self.sizes)*3*4
                + (len(self.sizes)*a_bytes if observed['gradient'] is not None else 0)))
        return observed

    def refresh(self, middle, rnd):
        ordinary, deltas = self.train_phase(middle, rnd, 'A', True, return_deltas=True)
        self.events[-1]['state_role'] = 'ordinary_A_proposal_before_margin_correction'
        if sorted(deltas) != list(range(len(self.sizes))):
            raise ValueError('New A requires all registered clients')
        # One global RMS radius. No QR, source responses or per-token backward passes.
        radius_tensor = torch.stack([flatten(deltas[j], self.a_keys).square().sum()
                                     for j in sorted(deltas)]).mean().sqrt()
        require_finite(proposal_radius=radius_tensor)
        radius = float(radius_tensor)
        return self.correct_ordinary(middle, ordinary, radius, rnd)

    def correct_ordinary(self, middle, ordinary, radius, rnd, diagnostic=None):
        """The production correction, also callable on a saved proposal without training."""
        targets = make_targets(self.history, self.margin_feedback.base_mass,
                               self.sfra_config['epsilon'], self.sfra_config['margin_scale'])
        started = time.perf_counter()
        extra = dict(diagnostics=True) if diagnostic is not None else {}
        before = self.observe_margin(ordinary, rnd, 'ordinary', targets=targets, **extra)
        if diagnostic is not None:
            diagnostic.begin(self, middle, ordinary, radius, rnd, targets, before)
        skip = ('no_valid_history' if not self.history.valid.any() else
                'zero_proposal_radius' if radius == 0 else
                'no_history_deficit' if before['deficit_count'] == 0 else '')
        row = dict(round=rnd, variant=VARIANT, aggregation_rule=self.job['aggregation'].get('rule', 'joint'),
            valid_history_count=int(self.history.valid.sum()), history_last_used=self.history.last_committed_round,
            radius=radius, skip_reason=skip, correction_started=not bool(skip), correction_steps=0,
            submitted_step=0, commit_rule=self.sfra_config['commit_rule'],
            ordinary_keep=before['keep_loss'], ordinary_class=before['class_loss'],
            ordinary_deficit_count=before['deficit_count'], J_ordinary=before['loss'])
        ordinary_a = flatten(ordinary, self.a_keys).to(self.trainer.device)
        z = torch.zeros_like(ordinary_a)

        def at(value):
            state = dict(ordinary)
            state.update({k: v.to(ordinary[k]) for k, v in unflatten(value, ordinary, self.a_keys).items()})
            return state

        submitted, final = ordinary, before
        if not skip:
            for step in range(self.sfra_config['correction_steps']):
                measurement = self.observe_margin(at(ordinary_a+radius*z), rnd, f'update_z{step}',
                                                  targets=targets, gradient=True, **extra)
                row[f'J_z{step}'] = .5*float(z.square().sum()) + measurement['loss']
                # The feedback gradient already includes both loss coefficients. Chain R exactly once.
                previous_z = z
                z = projected_step(z, radius, measurement['gradient'], 1., self.sfra_config['correction_step_size'])
                if diagnostic is not None:
                    diagnostic.step(step, previous_z, z, measurement, at(ordinary_a+radius*z))
            submitted = at(ordinary_a+radius*z)
            # This observation updates history/logs only. It never accepts/rejects a model.
            final = self.observe_margin(submitted, rnd, 'submitted', targets=targets, **extra)
            row.update(correction_steps=3, submitted_step=3)
        row.update(repair_seconds=time.perf_counter()-started, correction_z_norm=float(z.norm()),
            nonzero_correction=any(not torch.equal(submitted[k], ordinary[k]) for k in self.a_keys),
            final_keep=final['keep_loss'], final_class=final['class_loss'], final_deficit_count=final['deficit_count'],
            J_submitted=.5*float(z.square().sum())+final['loss'])
        for g, old, new in zip(('H', 'M', 'T'), before['group_ce'].tolist(), final['group_ce'].tolist()):
            row.update({f'ordinary_CE_{g}': old, f'final_CE_{g}': new})
        if diagnostic is not None:
            diagnostic.finish(submitted, z, final, row)
        changed = self.history.commit(final['sample_margins'], rnd)
        row.update(history_registered=int(changed.sum()), history_count_after=int(self.history.valid.sum()),
                   history_last_committed=self.history.last_committed_round, window_count=self.history.block_count)
        return submitted, row

    def run(self, initial_state):
        last = self.args.sfra_stop_after_round or 100
        if self.resume_payload:
            state, completed = self.restore()
        else:
            self.load_training_state(initial_state)
            state, completed = snapshot(self.trainer.model), 0
            initial = self.observe_margin(state, 0, 'initial')
            self.history = SampleMarginHistory(initial['sample_margins'], self.margin_feedback.manifest)
            self.publish(state, 0, 0)
            self.checkpoint(state, 0)
        for rnd in range(completed+1, last+1):
            state = self.train_phase(state, rnd, 'B')
            if rnd <= 90:
                state, row = self.refresh(state, rnd)
            else:
                row = dict(round=rnd, variant=VARIANT, correction_steps=0, submitted_step=0,
                    correction_started=False, nonzero_correction=False, skip_reason='B_only_schedule',
                    history_last_committed=self.history.last_committed_round,
                    commit_rule=self.sfra_config['commit_rule'])
            require_finite(**{k: state[k] for k in self.keys})
            self.summaries.append(row)
            self.load_training_state(state)
            train_only(self.trainer.model, 'B')
            self.publish(state, rnd, rnd)
            self.checkpoint(state, rnd)
            print(f'MARGIN-A COMMITTED {rnd}/100: steps={row["correction_steps"]}, '
                  f'history={int(self.history.valid.sum())}', flush=True)
