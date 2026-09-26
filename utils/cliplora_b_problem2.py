"""Opt-in all-source shared C: independent class weighting and harm controls."""
from collections import defaultdict
import math
import time

import numpy as np
import torch

from utils.b_problem2_math import calibration_weights, class_means, feedback_loss, summarize_changes
from utils.cliplora_a_refresh import isolated_rng
from utils.cliplora_b_shared_transfer import SharedDonorBTransfer, shared_calibration_batches
from utils.cliplora_b_transfer import differentiable_b_residual, reconstruct_residual
from utils.cliplora_bridge_audit import write_csv, write_json


class Problem2DonorBTransfer(SharedDonorBTransfer):
    def _prepare_weights(self, manifests):
        positions = {k: {p: c for c, values in self.groups[k].items() for p in values}
                     for k in self.recipients}
        self.p2_weights, self.p2_labels = [], []
        rows = []
        for step in range(self.config['steps']):
            labels = {}
            for k in self.recipients:
                batch = manifests[k]['batches'][step]
                if not batch['tail_positions']:
                    raise ValueError('Empty tail batch')
                if any(positions[k][p] not in self.tail for p in batch['tail_positions']):
                    raise ValueError('Non-tail image in the tail batch')
                if any(positions[k][p] in self.tail for p in batch['non_tail_positions']):
                    raise ValueError('Tail image in the non-tail batch')
                labels[k] = [positions[k][p] for p in batch['tail_positions']+batch['non_tail_positions']]
            weights, mass = calibration_weights(labels, self.tail, self.config['tail_weight'],
                                                self.config['class_averaging'])
            self.p2_weights.append(weights)
            self.p2_labels.append(labels)
            rows.extend(dict(round=self.pending['round'], step=step+1, client_id=k, class_id=c,
                             weight=w, samples=labels[k].count(c), tail_mass=mass[True],
                             non_tail_mass=mass[False]) for (k, c), w in sorted(weights.items()))
        write_csv(self.pending['folder']/'class_weights.csv', rows)

    @torch.no_grad()
    def _trace(self, tensors, matrices, manifests, start, step, ordinary_norm):
        residual = {key: torch.bmm(tensors[key], matrices[key]).mean(0) for key in self.keys}
        means, rows, batch_losses, batch_harms = {}, [], [], []
        image_count = 0
        with isolated_rng(), differentiable_b_residual(self.modules, residual):
            for batch_index in range(self.config['steps']):
                means[batch_index] = {}
                classification, harm = 0., 0.
                for k in self.recipients:
                    sampled = manifests[k]['batches'][batch_index]
                    x, y = self.batch(k, sampled['tail_positions']+sampled['non_tail_positions'])
                    # Every arm pays the same C=0 algorithm reference budget.
                    losses, logits, _ = self.forward(x, y, diagnostic=step != 0)
                    if not torch.isfinite(losses).all():
                        raise ValueError('Nonfinite problem-2 reference/diagnostic loss')
                    values = {c: float(v) for c, v in class_means(losses, y).items()}
                    means[batch_index][k] = values
                    for c, value in values.items():
                        baseline = value if step == 0 else self.p2_baselines[batch_index][k][c]
                        weight = self.p2_weights[batch_index][k, c]
                        classification += weight*value
                        harm += weight*max(value-baseline, 0.)
                        mask = y == c
                        rows.append(dict(round=self.pending['round'], variant=self.config['problem2_variant'],
                            c_step=step, batch= batch_index+1, client_id=k, class_id=c,
                            group='tail' if c in self.tail else 'non_tail', samples=int(mask.sum()),
                            baseline_la=baseline, la=value, gain=baseline-value, weight=weight,
                            accuracy=100*float((logits.argmax(1)[mask] == y[mask]).float().mean()),
                            scope='fixed_calibration; not held-out'))
                    image_count += len(y)
                batch_losses.append(classification)
                batch_harms.append(harm)
        if step == 0:
            self.p2_baselines = means
        penalty = self.config['regularization']*sum(float(c.square().sum()) for c in matrices.values())/(
            len(next(iter(matrices.values())))*len(self.keys))
        loss, harm = sum(batch_losses)/len(batch_losses), sum(batch_harms)/len(batch_harms)
        norm = self.effective_norm(residual, start)
        trace = dict(round=self.pending['round'], variant=self.config['problem2_variant'], c_step=step,
            fixed_pool_la=loss, fixed_pool_harm=harm,
            fixed_pool_objective=loss+self.config['harm_beta']*harm+penalty,
            regularization_penalty=penalty, effective_transfer_norm=norm,
            transfer_to_ordinary_ratio=norm/ordinary_norm if ordinary_norm else None,
            images=image_count, scope='same calibration batches; no step selection',
            **{f'batch{i+1}_la': v for i, v in enumerate(batch_losses)},
            **{f'batch{i+1}_harm': v for i, v in enumerate(batch_harms)})
        self.pending['loss_trace'].append(trace)
        self.pending['class_trace'].extend(rows)
        write_csv(self.pending['folder']/'c_loss_trace.csv', self.pending['loss_trace'])
        write_csv(self.pending['folder']/'class_loss_trace.csv', self.pending['class_trace'])
        return trace

    def calibrate_shared(self, donor_ids, manifests, start):
        if not donor_ids or not self.recipients:
            raise ValueError('Problem 2 needs nonempty donors and feedback')
        self._prepare_weights(manifests)
        m, layers = len(donor_ids), len(self.keys)
        tensors = {key: torch.stack([self.original_deltas[j][key] for j in donor_ids]).detach().to(self.device)
                   for key in self.keys}
        matrices = {key: torch.nn.Parameter(torch.zeros(m, self.ranks[key], self.ranks[key],
                     dtype=tensors[key].dtype, device=self.device)) for key in self.keys}
        optimizer = torch.optim.Adam(list(matrices.values()), lr=self.config['learning_rate'],
                                     betas=(.9, .999), eps=1e-8, weight_decay=0.)
        summary = self.pending['summary']
        summary['learned_c_parameters'] = sum(c.numel() for c in matrices.values())
        parameter_bytes = sum(c.numel()*c.element_size() for c in matrices.values())
        ordinary_norm = self.effective_norm(
            {key: self.parameters[key].detach().cpu()-start[key].detach().cpu() for key in self.keys}, start)
        trace = self._trace(tensors, matrices, manifests, start, 0, ordinary_norm)
        self.last_matrix_steps = [{key: c.detach().cpu().clone() for key, c in matrices.items()}]
        for step in range(self.config['steps']):
            optimizer.zero_grad(set_to_none=True)
            data_loss, harm_loss = 0., 0.
            for k in self.recipients:
                sampled = manifests[k]['batches'][step]
                x, y = self.batch(k, sampled['tail_positions']+sampled['non_tail_positions'])
                if y.tolist() != self.p2_labels[step][k]:
                    raise ValueError('Actual image labels differ from the weight manifest')
                residual = {key: torch.bmm(tensors[key], matrices[key]).mean(0) for key in self.keys}
                with differentiable_b_residual(self.modules, residual):
                    losses, _, _ = self.forward(x, y)
                    objective, classification, harm = feedback_loss(losses, y, k, self.p2_weights[step],
                        self.p2_baselines[step][k], self.config['harm_beta'])
                    if not torch.isfinite(objective):
                        raise ValueError('Nonfinite shared-C objective')
                    objective.backward()
                data_loss += float(classification.detach())
                harm_loss += float(harm.detach())
                summary['algorithm_backward_images'] += len(y)
                summary['client_backward_batches'] += 1
                self.pending['feedback'].append(dict(round=self.pending['round'], step=step+1,
                    client_id=k, shared_c_version=step, weighted_la=float(classification.detach()),
                    weighted_harm=float(harm.detach()), tail_samples=len(sampled['tail_positions']),
                    non_tail_samples=len(sampled['non_tail_positions'])))
            regularizer = sum(c.square().sum() for c in matrices.values())/(m*layers)
            (self.config['regularization']*regularizer).backward()
            if not all(c.grad is not None and torch.isfinite(c.grad).all() for c in matrices.values()):
                raise ValueError('Nonfinite or missing C gradient')
            gradient_norm = math.sqrt(sum(float(c.grad.square().sum()) for c in matrices.values()))
            objective_before = data_loss+self.config['harm_beta']*harm_loss+float(regularizer.detach())*self.config['regularization']
            optimizer.step()
            if not all(torch.isfinite(c).all() for c in matrices.values()):
                raise ValueError('Nonfinite C update')
            self.last_matrix_steps.append({key: c.detach().cpu().clone() for key, c in matrices.items()})
            summary['optimizer_steps'] += 1
            summary['feedback_synchronizations'] += 1
            summary['extra_downlink_bytes'] += len(self.recipients)*parameter_bytes
            summary['extra_upload_bytes'] += len(self.recipients)*(parameter_bytes+8)
            after = self._trace(tensors, matrices, manifests, start, step+1, ordinary_norm)
            self.pending['steps'].append(dict(round=self.pending['round'], variant=self.config['problem2_variant'],
                step=step+1, la_before=data_loss, harm_before=harm_loss, objective_before=objective_before,
                same_batch_la_after=after[f'batch{step+1}_la'], same_batch_harm_after=after[f'batch{step+1}_harm'],
                same_batch_objective_after=after[f'batch{step+1}_la']+
                    self.config['harm_beta']*after[f'batch{step+1}_harm']+after['regularization_penalty'],
                fixed_pool_objective_before=trace['fixed_pool_objective'],
                fixed_pool_objective_after=after['fixed_pool_objective'], c_gradient_norm=gradient_norm,
                learning_rate=self.config['learning_rate']))
            print(f'B-problem2 {self.config["problem2_variant"]} r{self.pending["round"]:03d} '
                  f'C step={step+1} fixed objective={after["fixed_pool_objective"]:.6f}', flush=True)
            trace = after
        return {key: c.detach().cpu().clone() for key, c in matrices.items()}

    def apply_shared(self, start, ordinary, rnd):
        started = time.perf_counter()
        folder = self.runtime.root/'b_transfer_rounds'/f'r{rnd:03d}'
        folder.mkdir(parents=True, exist_ok=True)
        donors = sorted(self.selected)
        if not donors or len(set(donors)) != len(donors):
            raise ValueError('Invalid all-source donor list')
        summary = dict(round=rnd, mode='shared', problem2_variant=self.config['problem2_variant'],
            calibration_profile='problem2', tail_weight=self.config['tail_weight'],
            non_tail_sampling=self.config['non_tail_sampling'], harm_beta=self.config['harm_beta'],
            learning_rate=self.config['learning_rate'], aggregation=self.config['aggregation'],
            recipients=len(self.recipients), unique_donors=len(donors), donor_rule='all_selected',
            optimizer_steps=0, client_backward_batches=0, feedback_synchronizations=0,
            learned_c_parameters=0, algorithm_forward_images=0, algorithm_backward_images=0,
            diagnostic_forward_images=0, extra_downlink_bytes=0, extra_upload_bytes=0, seconds=0.)
        self.pending = dict(round=rnd, folder=folder, summary=summary, metrics=[], steps=[],
                            receivers=[], feedback=[], loss_trace=[], class_trace=[])
        manifests = getattr(self, 'replay_manifests', None)
        if manifests is None:
            manifests = {k: dict(donor_ids=donors, batches=shared_calibration_batches(
                self.groups[k], self.tail, self.runtime.args.seed, rnd, k, self.config['non_tail_sampling']))
                         for k in self.recipients}
        if set(manifests) != set(self.recipients):
            raise ValueError('Replay recipients differ from current recipients')
        if any(len(manifests[k]['batches']) != self.config['steps'] for k in self.recipients):
            raise ValueError('Replay batch count differs')
        delta_bytes = sum(ordinary[key].numel()*ordinary[key].element_size() for key in self.keys)
        summary['extra_downlink_bytes'] += len(self.recipients)*(delta_bytes*(1+len(donors))+4*len(donors))
        # Labels/counts suffice for global class weights; raw images remain local in FL semantics.
        weight_pairs = sum(len({c for c, positions in self.groups[k].items()
            if any(p in batch['tail_positions']+batch['non_tail_positions'] for p in positions)})
            for k in self.recipients for batch in manifests[k]['batches'])
        summary['extra_upload_bytes'] += weight_pairs*8  # (class ID, count), int32
        summary['extra_downlink_bytes'] += weight_pairs*8  # (class ID, weight), FP32/int32
        with self.session():
            self.copy_parameters(ordinary)
            before = self.measure_shared('ordinary_global_B')
            matrices = self.calibrate_shared(donors, manifests, start)
            residual = reconstruct_residual(self.original_deltas, donors, matrices, self.keys)
            committed = dict(ordinary)
            committed.update({key: ordinary[key]+residual[key] for key in self.keys})
            self.copy_parameters(committed)
            after = self.measure_shared('transferred_global_B')
        self.last_matrices, self.last_residual = matrices, residual
        norm = self.effective_norm(residual, start)
        summary.update(effective_global_transfer_norm=norm,
            ordinary_B_effective_update_norm=self.effective_norm(
                {key: ordinary[key]-start[key] for key in self.keys}, start))
        for k in self.recipients:
            self.pending['receivers'].append(dict(round=rnd, client_id=k, measurement_scope='shared_B_before_after',
                shared_donors=len(donors), effective_transfer_norm=norm,
                **{'before_'+name: v for name, v in before[k].items()},
                **{'after_'+name: v for name, v in after[k].items()}))
        indexed = defaultdict(dict)
        for row in self.pending['metrics']:
            indexed[row['phase']][row['client_id'], row['class_id']] = row
        changes = []
        for (k, c), b in indexed['ordinary_global_B'].items():
            a = indexed['transferred_global_B'][k, c]
            fitted = {p for batch in manifests[k]['batches'] for group in ('tail_positions','non_tail_positions')
                      for p in batch[group]}
            monitored = self.monitors[k]['tail_positions']+self.monitors[k]['non_tail_positions']
            local_class = set(self.groups[k][c])
            changes.append(dict(round=rnd, variant=self.config['problem2_variant'], client_id=k, class_id=c,
                group='tail' if c in self.tail else 'non_tail', baseline_la=b['la'], la=a['la'],
                gain=b['la']-a['la'], samples=b['samples'],
                calibration_overlap=len(set(monitored) & fitted & local_class),
                accuracy_change=a['accuracy']-b['accuracy'], scope='training_monitor; overlap recorded'))
        write_csv(folder/'class_changes.csv', changes)
        write_csv(folder/'class_change_summary.csv', [{**r, 'round':rnd,
            'variant':self.config['problem2_variant']} for r in summarize_changes(changes)])
        write_json(folder/'calibration_manifest.json', dict(mode='shared', module_order=self.keys,
            shared_donor_ids=donors, feedback_clients=self.recipients, recipients=manifests))
        np.savez_compressed(folder/'matrices.npz', donor_ids=np.asarray(donors, dtype=np.int64),
                            **{f'module_{i:02d}':matrices[key].numpy() for i, key in enumerate(self.keys)})
        torch.save(dict(round=rnd, donor_ids=donors, ordinary_lora={key:ordinary[key] for key in self.runtime.keys},
            committed_lora={key:committed[key] for key in self.runtime.keys}, transfer_residual=residual), folder/'commit.pt')
        summary['seconds'] = time.perf_counter()-started
        self.write_pending()
        self.original_deltas = None
        return committed
