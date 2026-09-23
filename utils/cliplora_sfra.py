"""Source-aware functional retention of A, on the unchanged LA-control S schedule."""
import csv
import json
from pathlib import Path
import time

import numpy as np
import torch

from utils.cliplora_a_refresh import train_only, product_norm_squared
from utils.cliplora_bridge_audit import write_csv, write_json
from utils.cliplora_functional_feedback import snapshot
from utils.cliplora_la_control import LAControlRuntime
from utils.pfrf import capture_rng_state, restore_rng_state
from utils.sfra_feedback import WitnessBank
from utils.b_aggregation import B_TRANSFER_ROUNDS, phase_weights
from utils.sfra_math import (FunctionalHistory, flatten, unflatten, proposal_coordinates,
                             source_statistics, make_targets, projected_step, require_finite, functional_loss,
                             classification_preservation)


class SFRARuntime(LAControlRuntime):
    def __init__(self, trainer, global_trainer, cfg, args, schedule, normal_train):
        assert args.lac_method == 's'
        super().__init__(trainer, global_trainer, cfg, args, schedule, normal_train)
        self.variant = args.sfra_variant
        self.strength = args.sfra_retention_weight
        self.classification_strength = args.sfra_classification_weight if self.variant.endswith('-cp') else 0.
        assert self.strength >= 0 and np.isfinite(self.strength)
        assert args.sfra_witness_batch_size > 0
        self.method = 'sfra_' + self.variant
        self.sfra_config = dict(schema_version='sfra_v1', variant=self.variant,
            retention_weight=self.strength, correction_steps=3, correction_step_size=.1,
            witness_per_local_class=8, witness_views=2, positive_response_threshold=1e-6,
            current_gain_fraction=.5, history_window=5, history_improvement=.001,
            sigma_floor=.001, radius='unweighted_client_proposal_RMS',
            source_coordinates='untruncated_reduced_QR', a_parameter_order=self.a_keys,
            base_training_config=self.config, seed=args.seed, protocol_seed=args.split_seed,
            partition=args.partition, witness_batch_size=args.sfra_witness_batch_size,
            feedback='training-side mean unscaled all-class cosine margin; deterministic view + flip',
            normalization='one global active-token normalization; no further client weighting',
            privacy='sequential FL simulation with local diagnostics; no DP or secure aggregation',
            precision='fp32', commit_rule='fixed_third_step', refresh_rounds=list(range(1,91)))
        if self.variant.endswith('-cp'):
            self.sfra_config.update(schema_version='sfra_cp_v1', classification_weight=self.classification_strength,
                classification_objective='global sample-weighted LA on all witness tokens and both views',
                classification_reference='same-round ordinary A proposal with fixed post-B model',
                classification_penalty='0.5 * positive((C-C0)/scale)^2 after global loss aggregation',
                classification_scale='max(0.001, R * norm(global LA gradient at ordinary A))',
                classification_scale_floor=.001, classification_gradient_upload='separate from functional gradient')
        if getattr(args, 'sfra_b_aggregation', 'sample') == 'uniform-transfer-rounds':
            self.sfra_config['b_aggregation'] = dict(mode='uniform-transfer-rounds',
                rounds=list(B_TRANSFER_ROUNDS), scope='whole local B, including transfer residual if enabled',
                other_B_rounds='sample_weighted', A_aggregation='sample_weighted',
                classification_preservation_weights='original_sample_weighted')
            self.method += '_bagg_uniform8'
        if getattr(args, 'b_transfer_enable', False):
            from utils.cliplora_b_transfer import transfer_config
            self.sfra_config['b_transfer'] = transfer_config(args)
            self.method += '_b_transfer'
        self.resume_payload = None
        if args.sfra_resume:
            self.resume_payload = torch.load(args.sfra_resume, map_location='cpu', weights_only=False)
            # The launcher replays the original command. Do not silently change a method on resume.
            assert self.resume_payload['sfra_config'] == self.sfra_config
            self.base_state = torch.load(self.root/'checkpoints/base_model.pt', map_location='cpu', weights_only=False)
            self.initial = {k:self.base_state[k].clone() for k in self.keys}
        write_json(self.root/'sfra_config.json', self.sfra_config)
        self.costs, self.summaries = [], []
        self.elapsed_before = 0.
        self.execution_config = None
        self._execution_loader = None
        if getattr(args, 'sfra_fast_execution', False):
            from utils.sfra_execution import FAST_EXECUTION_CONFIG, LoRAStateLoader, enable_model_execution
            self.execution_config = dict(FAST_EXECUTION_CONFIG)
            enable_model_execution(trainer.model)
            enable_model_execution(global_trainer.model)
            self._execution_loader = LoRAStateLoader(trainer.model, self.keys)
            write_json(self.root/'execution_config.json', self.execution_config)
        if self.resume_payload:
            assert self.resume_payload.get('execution_config') == self.execution_config, 'Resume execution mode differs'
        self.history, self.bank = None, None
        if self.variant != 's':
            tokens = self.resume_payload['witness_tokens'] if self.resume_payload else None
            self.bank = WitnessBank(trainer, cfg, self.a_keys, args.seed,
                                    args.sfra_witness_batch_size, tokens=tokens)
            if self.execution_config is not None:
                from utils.sfra_fast_feedback import FastWitnessFeedback
                self.bank._fast_feedback = FastWitnessFeedback(
                    self.bank, self.execution_config['feedback_forward_batch_size'])
            if self.variant.endswith('-cp'):
                self.bank.configure_classification(self.audit.counts, trainer.training_logit_adjustment)
            # Preserve both local positions and original image identities in the experiment artifact.
            with (self.root/'partition_manifest.csv').open(encoding='utf-8', newline='') as stream:
                identity = {(int(r['client_id']), int(r['local_position'])):int(r['raw_sample_id'])
                            for r in csv.DictReader(stream)}
            for token in self.bank.tokens:
                token['raw_sample_ids'] = [identity[token['client_id'], j] for j in token['local_positions']]
            write_json(self.root/'private_witness_manifest.json', self.bank.tokens)
        self.b_transfer = None
        if 'b_transfer' in self.sfra_config:
            from utils.cliplora_b_transfer import DonorBTransfer
            self.b_transfer = DonorBTransfer(self)
        label = f', classification mu={self.classification_strength:g}' if self.variant.endswith('-cp') else ''
        print(f'SFRA: {self.variant}, retention lambda={self.strength:g}{label}, A rounds=1..90', flush=True)
        if self.b_transfer is not None:
            print(f'B-transfer enabled: C lr={args.b_transfer_lr:g}, rounds=30,40,...,100', flush=True)
        if 'b_aggregation' in self.sfra_config:
            print('B aggregation: uniform over all 30 clients at rounds 30,40,...,100 only; '
                  'all other B rounds and every A proposal stay sample-weighted.', flush=True)
        if self.execution_config is not None:
            print('SFRA fast execution: FP32 caches, LoRA-only restores, low-rank forward, '
                  'client-local feedback batching. Algorithm and training budgets unchanged.', flush=True)

    def load_training_state(self, state):
        loader = getattr(self, '_execution_loader', None)
        if loader is None:
            super().load_training_state(state)
        else:
            loader.load(state)

    def phase_aggregation_weights(self, selected, rnd, factor, extra=False, branch='main'):
        mode = self.sfra_config.get('b_aggregation', {}).get('mode', 'sample')
        return phase_weights(self.q, selected, rnd, mode, factor, extra, branch)

    def prepare_b_aggregation(self, state, local_states, deltas, selected, rnd):
        if self.b_transfer is None:
            return local_states, deltas
        return self.b_transfer.apply(state, local_states, deltas, selected, rnd)

    def functional(self, rnd, phase, **kwargs):
        print(f'SFRA round={rnd} phase={phase}', flush=True)
        result = self.bank.evaluate(**kwargs)
        self.costs.append(dict(round=rnd, phase=phase, seconds=result['seconds'],
            forward_images=result['forward_images'], backward_images=result['backward_images'],
            downlink_bytes=0, upload_bytes=0))
        if 'classification_loss' in result:
            self.costs[-1].update(classification_forward_images=result['classification_forward_images'],
                                 classification_backward_images=result['classification_backward_images'])
        for key in ('execution_forward_batches', 'execution_zero_gradient_batches'):
            if key in result:
                self.costs[-1][key] = result[key]
        return result

    def effective_norms(self, before, ordinary, committed):
        rows = []
        for key in self.a_keys:
            b = before[key[:-1]+'B'].double()
            d0 = ordinary[key].double()-before[key].double()
            da = committed[key].double()-before[key].double()
            rows.append(dict(layer=key, proposal_effective_norm=(self.scaling**2*product_norm_squared(b,d0))**.5,
                             committed_effective_norm=(self.scaling**2*product_norm_squared(b,da))**.5))
        return rows

    def refresh(self, middle, rnd, folder):
        classification = self.variant.endswith('-cp')
        ordinary, deltas = self.train_phase(middle, rnd, 'A', True, return_deltas=True)
        self.events[-1]['state_role'] = 'ordinary_A_proposal_before_functional_correction'
        device = self.trainer.device
        client_ids = sorted(deltas)
        matrix = torch.stack([flatten(deltas[k], self.a_keys) for k in client_ids], 1).to(device)
        qr_started = time.perf_counter()
        basis, coordinates, radius = proposal_coordinates(matrix)
        radius_cpu = radius.cpu()
        self.costs.append(dict(round=rnd, phase='proposal_QR', seconds=time.perf_counter()-qr_started,
            forward_images=0, backward_images=0,
            downlink_bytes=len(client_ids)*basis.numel()*basis.element_size(), upload_bytes=0))
        self.load_training_state(middle)
        source_measurement = self.functional(rnd, 'source_at_post_B', basis=basis)
        n_tokens = len(self.bank.tokens)
        # h is the compact client-to-server message; never replace sigma by ||h||.
        responses = source_measurement['coordinates'] @ coordinates.cpu()
        self.costs[-1]['upload_bytes'] = n_tokens*2*(basis.shape[1]+2)*4  # h, F, full gradient norm
        p = torch.tensor([self.q[k] for k in client_ids], dtype=torch.float32)
        source = source_statistics(responses, self.history.source_cache, p)
        self.costs.append(dict(round=rnd, phase='source_response_return', seconds=0.,
            forward_images=0, backward_images=0,
            downlink_bytes=responses.numel()*4+len(client_ids)*4, upload_bytes=0))
        self.history.source_cache = source['cache'].clone()
        history_before = self.history.level.clone()
        valid_before = self.history.valid.clone()
        targets = make_targets(source_measurement['scores'], source, history_before, valid_before, self.variant)
        targets['sigma'] = (radius_cpu*source_measurement['gradient_norms']).clamp_min(.001)
        require_finite(target=targets['target'], sigma=targets['sigma'], weights=targets['weights'])
        self.costs.append(dict(round=rnd, phase='global_weight_normalization', seconds=0.,
            forward_images=0, backward_images=0, downlink_bytes=4*len(client_ids), upload_bytes=4*len(client_ids)))
        z = torch.zeros(matrix.shape[0], device=device)
        ordinary_a = flatten(ordinary, self.a_keys).to(device)
        before_a = flatten(middle, self.a_keys).to(device)
        d0 = ordinary_a-before_a
        skip = 'no_active_tokens' if not targets['active'].any() else ('zero_proposal_radius' if radius <= 1e-12 else '')
        steps = []
        committed = dict(ordinary)
        classification_reference = classification_scale = classification_reference_gradient_norm = None
        proposal_classification_scores = proposal_classification_client_losses = None
        for step in range(0 if skip else 3):
            self.load_training_state(committed)
            measurement = self.functional(rnd, f'correction_{step+1}', targets=targets,
                                          all_scores=(step == 0), classification=classification)
            if step == 0:
                proposal_scores = measurement['scores']
            self.costs[-1]['downlink_bytes'] = len(client_ids)*matrix.shape[0]*4
            active_clients = {t['client_id'] for t,a in zip(self.bank.tokens, targets['active']) if a}
            self.costs[-1]['upload_bytes'] = len(active_clients)*matrix.shape[0]*4
            proximity = float(.5*z.square().sum())
            steps.append(dict(step=step+1, functional_loss=measurement['loss'], proximity=proximity,
                              objective=proximity+self.strength*measurement['loss'], z_before_norm=float(z.norm())))
            auxiliary_gradient = None
            if classification:
                if step == 0:
                    classification_reference = measurement['classification_loss']
                    classification_reference_gradient_norm = float(measurement['classification_gradient'].norm())
                    classification_scale = max(.001, float(radius) * classification_reference_gradient_norm)
                    proposal_classification_scores = measurement['classification_scores']
                    proposal_classification_client_losses = measurement['classification_client_losses']
                penalty, coefficient = classification_preservation(
                    measurement['classification_loss'], classification_reference, classification_scale)
                auxiliary_gradient = (self.classification_strength * coefficient) * measurement['classification_gradient']
                functional_term = self.strength * radius * measurement['gradient']
                classification_term = radius * auxiliary_gradient
                fn, cn = float(functional_term.norm()), float(classification_term.norm())
                alignment = (float(torch.dot(functional_term, classification_term) / (fn * cn))
                             if fn > 0 and cn > 0 else None)
                steps[-1].update(classification_weight=self.classification_strength,
                    classification_reference_loss=classification_reference,
                    classification_loss=measurement['classification_loss'],
                    classification_loss_increase=measurement['classification_loss']-classification_reference,
                    classification_scale=classification_scale,
                    classification_scale_floor_active=classification_scale == .001,
                    classification_reference_gradient_norm=classification_reference_gradient_norm,
                    classification_penalty=penalty, classification_gradient_coefficient=coefficient,
                    functional_z_gradient_norm=fn, classification_z_gradient_norm=cn,
                    functional_classification_gradient_cosine=alignment,
                    classification_active=coefficient > 0)
                steps[-1]['objective'] += self.classification_strength * penalty
                # Each client returns its LA loss and A gradient, separate from functional feedback.
                self.costs[-1]['upload_bytes'] += len(client_ids) * (matrix.shape[0]+1) * 4
            z = projected_step(z, radius, measurement['gradient'], self.strength,
                               auxiliary_gradient=auxiliary_gradient)
            require_finite(z=z)
            committed.update(unflatten((ordinary_a+radius*z).cpu(), middle, self.a_keys))
            steps[-1]['z_after_norm'] = float(z.norm())
            label = (f' LA={measurement["classification_loss"]:.6g} '
                     f'LA_increase={steps[-1]["classification_loss_increase"]:.6g} '
                     f'cls_penalty={penalty:.6g} |g_func|={fn:.5g} |g_cls|={cn:.5g} '
                     f'mu={self.classification_strength:g}' if classification else '')
            print(f'SFRA round={rnd} step={step+1}/3 loss={measurement["loss"]:.6g} |z|={z.norm():.5g}{label}', flush=True)
        self.load_training_state(committed)
        final_measurement = self.functional(rnd, 'committed_A', classification=classification,
                                            classification_gradient=False)
        self.costs[-1]['downlink_bytes'] = len(client_ids)*matrix.shape[0]*4
        if skip:
            proposal_scores = final_measurement['scores']
        after_a = flatten(committed, self.a_keys).to(device)
        actual_delta = after_a-before_a
        effective = self.effective_norms(middle, ordinary, committed)
        cosine = float(torch.nn.functional.cosine_similarity(d0, actual_delta, dim=0))
        summary = dict(round=rnd, variant=self.variant, retention_weight=self.strength,
            active_tokens=int(targets['active'].sum()), supported_tokens=int(source['supported'].sum()),
            history_tokens_before=int(valid_before.sum()), correction_steps=len(steps), skip_reason=skip,
            projection_boundary_steps=sum(s['z_after_norm'] >= .99999 for s in steps),
            radius=float(radius), proposal_a_norm=float(d0.norm()), correction_a_norm=float((after_a-ordinary_a).norm()),
            committed_a_norm=float(actual_delta.norm()), proposal_committed_cosine=cosine,
            proposal_effective_norm=sum(r['proposal_effective_norm']**2 for r in effective)**.5,
            committed_effective_norm=sum(r['committed_effective_norm']**2 for r in effective)**.5,
            mean_post_B_margin=float(source_measurement['scores'].mean()),
            mean_proposal_margin=float(proposal_scores.mean()), mean_committed_margin=float(final_measurement['scores'].mean()),
            proposal_functional_loss=float(functional_loss(proposal_scores, targets['target'], targets['sigma'], targets['weights'])),
            committed_functional_loss=float(functional_loss(final_measurement['scores'], targets['target'], targets['sigma'], targets['weights'])))
        arrays = dict(F_post_B=source_measurement['scores'], F_proposal=proposal_scores,
            F_committed=final_measurement['scores'], responses=responses,
            predicted_sample_weighted_response=source['predicted'], positive_sources=source['positive_count'],
            supported=source['supported'], u=source['u'], n_eff=source['n_eff'], source_cache=source['cache'],
            current_target=targets['current'], target=targets['target'], active=targets['active'],
            weights=targets['weights'], sigma=targets['sigma'], full_gradient_norm=source_measurement['gradient_norms'],
            history_before=history_before, history_valid_before=valid_before)
        if classification:
            self.costs[-1]['upload_bytes'] += len(client_ids) * 4  # Final classification loss, no gradient.
            if skip:
                # The ordinary proposal is committed unchanged, so the classification increase is zero.
                classification_reference = final_measurement['classification_loss']
                proposal_classification_scores = final_measurement['classification_scores']
                proposal_classification_client_losses = final_measurement['classification_client_losses']
                penalty = 0.
            else:
                penalty, _ = classification_preservation(
                    final_measurement['classification_loss'], classification_reference, classification_scale)
            summary.update(classification_weight=self.classification_strength,
                classification_reference_loss=classification_reference,
                committed_classification_loss=final_measurement['classification_loss'],
                committed_classification_loss_increase=final_measurement['classification_loss']-classification_reference,
                classification_scale=classification_scale,
                classification_scale_floor_active=classification_scale == .001 if steps else None,
                classification_reference_gradient_norm=classification_reference_gradient_norm,
                committed_classification_penalty=penalty,
                committed_objective=float(.5*z.square().sum())+self.strength*summary['committed_functional_loss']
                                    +self.classification_strength*penalty,
                classification_active_steps=sum(s['classification_active'] for s in steps),
                functional_z_gradient_norm_mean=sum(s['functional_z_gradient_norm'] for s in steps)/len(steps) if steps else 0.,
                classification_z_gradient_norm_mean=sum(s['classification_z_gradient_norm'] for s in steps)/len(steps) if steps else 0.)
            client_index = torch.tensor([t['client_id'] for t in self.bank.tokens], device=device)
            arrays.update(classification_proposal_scores=proposal_classification_scores,
                classification_committed_scores=final_measurement['classification_scores'],
                classification_proposal_client_losses=proposal_classification_client_losses,
                classification_committed_client_losses=final_measurement['classification_client_losses'],
                classification_client_weights=self.bank.classification_client_weights.cpu(),
                classification_global_token_weights=(self.bank.classification_token_weights
                    * self.bank.classification_client_weights[client_index]).cpu())
            print(f'SFRA classification committed: reference={classification_reference:.6g}, '
                  f'LA={final_measurement["classification_loss"]:.6g}, penalty={penalty:.6g}', flush=True)
        write_csv(folder/'correction_steps.csv', steps) if steps else None
        write_csv(folder/'effective_updates.csv', effective)
        # Store actual committed A/B separately: the inherited refresh dump is only the ordinary proposal.
        torch.save(dict(round=rnd, before_lora={k:ordinary[k] for k in self.keys},
                        committed_lora={k:committed[k] for k in self.keys}), folder/'commit.pt')
        return committed, final_measurement['scores'], arrays, summary

    def checkpoint(self, state, completed):
        payload = dict(schema_version='sfra_cp_v1' if self.variant.endswith('-cp') else 'sfra_v1',
            completed_round=completed, sfra_config=self.sfra_config,
            state_dict_overrides=self.compressed_state(state), rng_state=capture_rng_state(),
            witness_tokens=None if self.bank is None else self.bank.tokens,
            functional_history=None if self.history is None else self.history.state_dict(),
            events=self.events, budget=self.budget, evaluations=self.evaluations,
            functional_costs=self.costs, round_summaries=self.summaries,
            elapsed_seconds=self.elapsed_before+time.perf_counter()-self.started)
        if self.b_transfer is not None:
            payload['b_transfer_state'] = self.b_transfer.state_dict()
        if getattr(self, 'execution_config', None) is not None:
            payload['execution_config'] = self.execution_config
        temporary = self.root/'checkpoints/sfra_last.tmp'
        torch.save(payload, temporary)
        temporary.replace(self.root/'checkpoints/sfra_last.pt')
        self.flush_records(completed)

    def flush_records(self, completed):
        write_csv(self.root/'event_manifest.csv', self.events)
        write_csv(self.root/'budget.csv', self.budget)
        write_csv(self.root/'evaluation_budget.csv', self.evaluations)
        write_csv(self.root/'sfra_costs.csv', self.costs)
        write_csv(self.root/'sfra_rounds.csv', self.summaries)
        normal = sum(r['optimizer_steps'] for r in self.budget if r['phase']=='normal_B')
        extra = sum(r['optimizer_steps'] for r in self.budget if r['phase']=='refresh_A')
        progress = dict(completed_round=completed, variant=self.variant, retention_weight=self.strength,
            normal_optimizer_steps=normal, extra_optimizer_steps=extra,
            total_local_optimizer_steps=normal+extra,
            functional_correction_steps=sum(r['correction_steps'] for r in self.summaries),
            official_test_passes=len(self.evaluations),
            elapsed_seconds=self.elapsed_before+time.perf_counter()-self.started)
        if self.variant.endswith('-cp'):
            progress['classification_weight'] = self.classification_strength
        if 'b_aggregation' in self.sfra_config:
            progress['b_aggregation'] = self.sfra_config['b_aggregation']['mode']
            progress['uniform_B_rounds_completed'] = sum(r <= completed for r in B_TRANSFER_ROUNDS)
        if self.b_transfer is not None:
            self.b_transfer.flush()
            progress.update(self.b_transfer.progress())
            progress['total_optimizer_steps_including_transfer'] = (
                normal+extra+progress['functional_correction_steps']+progress['b_transfer_optimizer_steps'])
        write_json(self.root/'progress.json', progress)
        if completed == 100:
            assert normal == self.config['normal_steps_expected'] and extra == self.config['extra_steps_expected']
            assert len(self.evaluations) == 101
            write_json(self.root/'completion.json', progress)

    def restore(self):
        saved = self.resume_payload
        state = dict(self.base_state)
        state.update(saved['state_dict_overrides'])
        self.events, self.budget, self.evaluations = saved['events'], saved['budget'], saved['evaluations']
        self.costs, self.summaries = saved['functional_costs'], saved['round_summaries']
        if self.b_transfer is not None:
            self.b_transfer.load_state_dict(saved['b_transfer_state'])
        self.elapsed_before = saved['elapsed_seconds']
        if self.bank is not None:
            self.history = FunctionalHistory(torch.zeros(len(self.bank.tokens),2), len(self.sizes))
            self.history.load_state_dict(saved['functional_history'])
        completed = saved['completed_round']
        # An interrupted uncommitted round may already have appended a test row.
        path = self.root/'round_metrics.csv'
        with path.open(encoding='utf-8', newline='') as stream:
            rows = [r for r in csv.DictReader(stream) if int(r['round']) <= completed]
        write_csv(path, rows)
        self.load_training_state(state)
        train_only(self.trainer.model, 'B')
        restore_rng_state(saved['rng_state'])  # Restore AFTER reconstructing data/model/witnesses.
        self.resume_payload = None
        self.flush_records(completed)
        print(f'SFRA resumed after completed round {completed}', flush=True)
        return state, completed

    def run(self, initial_state):
        if self.resume_payload:
            state, completed = self.restore()
        else:
            self.load_training_state(initial_state)
            state, completed = snapshot(self.trainer.model), 0
            if self.bank is not None:
                initial = self.functional(0, 'initial_reference')['scores']
                self.history = FunctionalHistory(initial, len(self.sizes))
            self.publish(state, 0, 0)
            self.checkpoint(state, 0)
        for rnd in range(completed+1, 101):
            try:
                state = self.train_phase(state, rnd, 'B')
                if self.b_transfer is not None:
                    self.b_transfer.observe_global(state, rnd)
                require_finite(**{k:state[k] for k in self.keys})
                if self.variant == 's':
                    if rnd <= 90:
                        state = self.train_phase(state, rnd, 'A', True)
                    require_finite(**{k:state[k] for k in self.keys})
                else:
                    folder = self.root/'sfra_rounds'/f'r{rnd:03d}'
                    folder.mkdir(parents=True, exist_ok=True)
                    if rnd <= 90:
                        state, scores, arrays, summary = self.refresh(state, rnd, folder)
                    else:
                        # No A proposals, QR or correction after round 90.
                        scores = self.functional(rnd, 'post_B_only')['scores']
                        arrays = dict(F_post_B=scores, F_committed=scores,
                                      history_before=self.history.level.clone(), history_valid_before=self.history.valid.clone())
                        summary = dict(round=rnd, variant=self.variant, retention_weight=self.strength,
                                       correction_steps=0, skip_reason='B_only_schedule')
                    if self.variant.endswith('-cp'):
                        summary['classification_weight'] = self.classification_strength
                    changed = self.history.commit(scores, rnd)  # Only now, effective next round.
                    arrays.update(history_after=self.history.level, history_valid_after=self.history.valid,
                                  history_registered=changed,
                                  history_gap_before=(arrays['history_before'][:,None]-scores).clamp_min(0),
                                  history_gap_after_registration=(self.history.level[:,None]-scores).clamp_min(0))
                    np.savez_compressed(folder/'tokens.npz', **{k:v.cpu().numpy() for k,v in arrays.items()})
                    summary['history_tokens_after'] = int(self.history.valid.sum())
                    summary['history_registered'] = int(changed.sum())
                    valid = arrays['history_valid_before']
                    if valid.any():
                        summary['mean_previous_history_gap'] = float((arrays['history_before'][valid,None]-scores[valid]).clamp_min(0).mean())
                    write_json(folder/'summary.json', summary)
                    self.summaries.append(summary)
                self.load_training_state(state)
                train_only(self.trainer.model, 'B')
                if self.b_transfer is not None:
                    self.b_transfer.observe_global(state, rnd, committed=True)
                self.publish(state, rnd, rnd)
                self.checkpoint(state, rnd)
                label = f', mu={self.classification_strength:g}' if self.variant.endswith('-cp') else ''
                if self.b_transfer is not None:
                    label += f', B-transfer lr={self.b_transfer.config["learning_rate"]:g}'
                print(f'SFRA COMMITTED {rnd}/100: {self.variant}, lambda={self.strength:g}{label}', flush=True)
            except Exception as error:
                # Fail visibly, including NaN/Inf; never submit a substitute model.
                torch.save(dict(round=rnd, error=repr(error), last_committed_checkpoint='checkpoints/sfra_last.pt',
                                current_lora={k:v.detach().cpu() for k,v in self.trainer.model.state_dict().items() if k in self.keys},
                                rng_state=capture_rng_state()), self.root/'failure_diagnostic.pt')
                raise
