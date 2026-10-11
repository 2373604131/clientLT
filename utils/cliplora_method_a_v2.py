"""Source-memory-v2 orchestration, reusing the ordinary SFRA/LA training runtime."""
import time

import torch

from utils.cliplora_a_refresh import train_only
from utils.cliplora_bridge_audit import write_json
from utils.cliplora_functional_feedback import snapshot
from utils.cliplora_sfra import SFRARuntime
from utils.method_a_v2_feedback import SampleFeedback
from utils.method_a_v2_math import (CONFIG, VARIANT, SampleSourceHistory, make_targets,
                                    positive_support, candidate_reasons)
from utils.sfra_math import flatten, unflatten, proposal_coordinates, projected_step, require_finite


class MethodAV2Runtime(SFRARuntime):
    def configure_algorithm(self):
        if (self.variant != VARIANT or self.strength != 10.
                or getattr(self.args, 'b_transfer_enable', False)
                or getattr(self.args, 'sfra_b_aggregation', 'sample') != 'sample'
                or not getattr(self.args, 'sfra_fast_execution_v2', False)):
            raise ValueError('Av2 requires its fixed contract, production cache and no B/C transfer')
        self.sfra_config = dict(CONFIG, base_training_config=self.config,
            seed=self.args.seed, protocol_seed=self.args.split_seed, partition=self.args.partition,
            witness_batch_size=self.args.sfra_witness_batch_size, a_parameter_order=self.a_keys)

    def initialize_feedback(self):
        self.av2 = SampleFeedback(self.bank, self.audit.counts, (self.head,self.middle,self.tail),
                                  self.trainer.training_logit_adjustment)
        if self.resume_payload and self.resume_payload.get('schema_version') != VARIANT:
            raise ValueError('Av2 cannot restore a legacy token-history checkpoint')
        write_json(self.root/'av2_witness_manifest.json', self.av2.manifest)

    def history_from_state(self, state):
        history = SampleSourceHistory.restore(state,self.av2.manifest,len(self.bank.tokens),len(self.sizes))
        if history.last_committed_round != min(self.resume_payload['completed_round'],90):
            raise ValueError('Av2 history and submitted model rounds differ')
        return history

    def observe_av2(self, state, rnd, phase, **kwargs):
        self.load_training_state(state)
        print(f'AV2 round={rnd} phase={phase}',flush=True)
        result = self.av2.evaluate((rnd,phase),**kwargs)
        source = kwargs.get('basis')
        a_bytes = sum(state[k].numel()*4 for k in self.a_keys)
        source_bytes = (len(self.bank.tokens)*2*(source.shape[1]+1)*4 if source is not None else 0)
        self.costs.append(dict(round=rnd,phase=phase,seconds=result['seconds'],
            forward_images=result['forward_images'],backward_images=result['backward_images'],
            downlink_bytes=(len(self.sizes)*source.numel()*4 if source is not None else len(self.sizes)*a_bytes),
            upload_bytes=source_bytes+len(self.av2.manifest)*2*4+len(self.sizes)*3*4
                +(len(self.sizes)*a_bytes if result['gradient'] is not None and result['backward_images'] else 0)))
        return result

    def refresh(self, middle, rnd):
        # Deltas are needed for QR. Source gradients are then evaluated at the
        # saved common post-B state, never at a locally trained or ordinary state.
        ordinary, deltas = self.train_phase(middle,rnd,'A',True,return_deltas=True)
        self.events[-1]['state_role'] = 'ordinary_A_proposal_before_functional_correction'
        clients = sorted(deltas)
        if clients != list(range(len(self.sizes))):
            raise ValueError('Av2 requires all registered source clients')
        matrix = torch.stack([flatten(deltas[j],self.a_keys) for j in clients],1).to(self.trainer.device)
        radius = matrix.square().sum(0).mean().sqrt()
        require_finite(radius=radius, local_A_proposals=matrix)
        started = time.perf_counter()
        if float(radius) > 0:
            basis, coordinates, radius = proposal_coordinates(matrix)
            pre = self.observe_av2(middle,rnd,'post_b_source',basis=basis)
            response = pre['coordinates'] @ coordinates.cpu()
            weights = [self.joint_weights[j] for j in clients]
            support = positive_support(response,weights)
            self.costs[-1]['downlink_bytes'] += response.numel()*4 + len(clients)*4
        else:
            pre = self.observe_av2(middle,rnd,'post_b_zero_source')
            support = torch.zeros(len(self.bank.tokens),len(clients))
        source_seconds = time.perf_counter()-started
        targets = make_targets(self.history,self.av2.base_mass,pre['gradient_norms'],radius)
        repair_started = time.perf_counter()
        ordinary_feedback = self.observe_av2(ordinary,rnd,'ordinary',targets=targets,
                                             initialize_target=True,gradient=float(radius)>0 and bool(self.history.valid.any()))
        bounds = ordinary_feedback['group_ce']+.1*(pre['group_ce']-ordinary_feedback['group_ce']).clamp_min(0)
        ordinary_a = flatten(ordinary,self.a_keys).to(self.trainer.device)
        z = torch.zeros_like(ordinary_a)
        j0 = 10*ordinary_feedback['loss']
        skip = ('no_valid_history' if not self.history.valid.any() else
                'zero_proposal_radius' if float(radius)==0 else
                'no_history_deficit' if ordinary_feedback['loss']==0 else '')
        summary = dict(round=rnd,aggregation_rule=self.job['aggregation'].get('rule','joint'),variant=VARIANT,
            valid_history_count=int(self.history.valid.sum()), history_last_used=self.history.last_committed_round,
            had_history_deficit=ordinary_feedback['loss']>0,candidate_attempted=not bool(skip),
            selected_beta=0.,nonzero_correction=False,full_reject='',half_reject='',
            J0=j0,J_final=j0,correction_steps=0,skip_reason=skip,radius=float(radius),
            source_seconds=source_seconds,repair_seconds=0.,candidate_check_seconds=0.)
        for label, ids in zip(('H','M','T'),self.av2.groups):
            mask = self.history.valid & torch.isin(self.history.class_ids,torch.tensor(ids))
            summary['known_'+label] = int((mask & self.history.known).sum())
            for reason in ('pretrained_unknown','no_positive_support'):
                summary[reason+'_'+label] = sum(bool(mask[i]) and value==reason for i,value in enumerate(self.history.reason))
        def at(value):
            candidate = dict(ordinary)
            candidate.update(unflatten(value,ordinary,self.a_keys))
            return candidate
        accepted, feedback = ordinary, ordinary_feedback
        if not skip:
            measurement = ordinary_feedback  # Gradient at z0, from the ordinary forward.
            for step in range(3):
                if step:
                    measurement = self.observe_av2(at(ordinary_a+radius*z),rnd,f'repair_z{step}',
                                                   targets=targets,gradient=True)
                z = projected_step(z,radius,measurement['gradient'],10.,.1)
            summary['correction_steps'] = 3
        summary['repair_seconds'] = time.perf_counter()-repair_started
        checking = time.perf_counter()
        if not skip:
            for beta, name in ((1.,'full'),(.5,'half')):
                candidate = at(ordinary_a+beta*radius*z)
                observed = self.observe_av2(candidate,rnd,name,targets=targets)
                objective,reasons = candidate_reasons(observed['group_ce'],bounds,observed['loss'],beta,z,j0)
                summary[name+'_reject'] = reasons
                summary[name+'_J'] = objective
                for g,c in zip(('H','M','T'),observed['group_ce'].tolist()):
                    summary[name+'_CE_'+g] = c
                if not reasons:
                    accepted,feedback = candidate,observed
                    summary.update(selected_beta=beta,J_final=objective)
                    break
        summary['candidate_check_seconds'] = time.perf_counter()-checking
        summary['nonzero_correction'] = any(not torch.equal(accepted[k],ordinary[k]) for k in self.a_keys)
        for g,before,proposal,bound,final in zip(('H','M','T'),pre['group_ce'].tolist(),
                ordinary_feedback['group_ce'].tolist(),bounds.tolist(),feedback['group_ce'].tolist()):
            summary.update({f'pre_CE_{g}':before,f'ordinary_CE_{g}':proposal,f'limit_CE_{g}':bound,f'final_CE_{g}':final})
        # Only the selected, fully observed model advances the history window.
        changed = self.history.commit(feedback['sample_margins'],support,rnd)
        summary.update(history_registered=int(changed.sum()),history_count_after=int(self.history.valid.sum()),
                       history_last_committed=self.history.last_committed_round,window_count=self.history.block_count)
        return accepted,summary

    def run(self, initial_state):
        last = self.args.sfra_stop_after_round or 100
        if self.resume_payload:
            state,completed = self.restore()
        else:
            self.load_training_state(initial_state)
            state,completed = snapshot(self.trainer.model),0
            initial = self.observe_av2(state,0,'initial')
            self.history = SampleSourceHistory(initial['sample_margins'],self.av2.manifest,
                                               len(self.bank.tokens),len(self.sizes))
            self.publish(state,0,0)
            self.checkpoint(state,0)
        for rnd in range(completed+1,last+1):
            state = self.train_phase(state,rnd,'B')
            if rnd <= 90:
                state,summary = self.refresh(state,rnd)
            else:
                summary = dict(round=rnd,variant=VARIANT,correction_steps=0,skip_reason='B_only_schedule',
                    selected_beta=0.,nonzero_correction=False,history_last_committed=self.history.last_committed_round)
            require_finite(**{k:state[k] for k in self.keys})
            self.summaries.append(summary)
            self.load_training_state(state)
            train_only(self.trainer.model,'B')
            self.publish(state,rnd,rnd)
            self.checkpoint(state,rnd)
            print(f'AV2 COMMITTED {rnd}/100: beta={summary["selected_beta"]}, '
                  f'history={int(self.history.valid.sum())}, window_count={self.history.block_count}',flush=True)
