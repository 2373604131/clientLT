"""Tail-only shared transfer using frozen, label-absent class donor mixtures.

The donor union is metadata, never the optimization basis. Only class mixtures
have trainable C blocks; every loss sees the same finally committed model.
"""
from collections import defaultdict
import math
import time

import numpy as np
import torch

from utils.b_directed_math import (select_sources, build_class_bases,
                                  reconstruct_class_residual, class_macro_loss)
from utils.cliplora_b_shared_transfer import SharedDonorBTransfer
from utils.cliplora_b_transfer import differentiable_b_residual
from utils.cliplora_bridge_audit import write_csv, write_json


class DirectedDonorBTransfer(SharedDonorBTransfer):
    def __init__(self, runtime):
        super().__init__(runtime)
        self.target_clients = list(self.config['target_clients'])
        for client, groups in enumerate(self.groups):
            if any(int(self.counts[client, c]) != len(groups.get(c, [])) for c in self.tail):
                raise ValueError('Class-presence table differs from actual local training labels')
        self.recipients = [k for k in self.target_clients if set(self.groups[k]) & self.tail]
        if not self.recipients:
            raise ValueError('Protocol tail clients have no target-tail training data')
        self.probes = {k: {c: list(p) for c, p in self.groups[k].items() if c in self.tail}
                       for k in self.recipients}
        self.class_totals = {c: sum(len(self.probes[k].get(c, [])) for k in self.recipients)
                             for c in sorted(self.tail)}
        self.class_totals = {c: n for c, n in self.class_totals.items() if n}
        self.monitors = {k: dict(tail_positions=[p for c in sorted(self.probes[k]) for p in self.probes[k][c]],
                                non_tail_positions=[]) for k in self.recipients}
        self.feedback_batch_size = int(getattr(runtime.args, 'sfra_feedback_batch_size', self.bank.batch_size))
        if self.feedback_batch_size < 1:
            raise ValueError('Feedback batch size must be positive')
        write_json(runtime.root/'b_transfer_manifest.json', dict(
            tail_ids=sorted(self.tail), target_clients=self.target_clients, recipients=self.recipients,
            target_class_totals=self.class_totals, unobserved_tail_classes=sorted(self.tail-set(self.class_totals)),
            class_presence=(self.counts > 0).tolist(), probes=self.probes, monitors=self.monitors,
            data_scope='full training-side target pool; not an independent validation set'))

    def _class_average(self, scores):
        sums, nums = defaultdict(float), defaultdict(int)
        for per_class in scores.values():
            for c, row in per_class.items():
                if row['samples'] <= 0 or not math.isfinite(row['la']):
                    raise ValueError('Invalid directed probe loss/count')
                sums[c] += row['la']*row['samples']
                nums[c] += row['samples']
        return {c: sums[c]/nums[c] for c in sums}

    def measure_shared(self, phase):
        receivers = super().measure_shared(phase)
        rows = [r for r in self.pending['metrics'] if r['phase'] == phase]
        for metric in ('la', 'accuracy'):
            sums, counts = defaultdict(float), defaultdict(int)
            for row in rows:
                if not math.isfinite(row[metric]) or row['samples'] <= 0:
                    raise ValueError('Invalid directed shared-model measurement')
                sums[row['class_id']] += row[metric]*row['samples']
                counts[row['class_id']] += row['samples']
            self.pending['summary'][phase+'_tail_'+metric] = sum(sums[c]/counts[c] for c in sums)/len(sums)
        return receivers

    @torch.no_grad()
    def screen_sources(self, ordinary_b):
        baseline = self._class_average({k: self.score_positions(k, self.monitors[k]['tail_positions'], False)
                                        for k in self.recipients})
        gains = {c: {} for c in self.class_totals}
        candidates = [j for j in sorted(self.selected) if j not in self.target_clients]
        rows = []
        delta_bytes = sum(v.numel()*v.element_size() for v in ordinary_b.values())
        summary = self.pending['summary']
        for donor in candidates:
            eligible = {c for c in self.class_totals if int(self.counts[donor, c]) == 0}
            if not eligible:
                continue
            proposal = {key: ordinary_b[key]+self.config['probe_step']*self.original_deltas[donor][key]
                        for key in self.keys}
            if not all(torch.isfinite(v).all() for v in proposal.values()):
                raise ValueError('Nonfinite directed donor proposal')
            self.copy_parameters(proposal)
            scores = {}
            for k in self.recipients:
                positions = [p for c in sorted(eligible) for p in self.probes[k].get(c, [])]
                if positions:
                    scores[k] = self.score_positions(k, positions, False)
                    summary['extra_downlink_bytes'] += delta_bytes
                    summary['extra_upload_bytes'] += len(scores[k])*12  # class ID, loss sum, count
            after = self._class_average(scores)
            for c in sorted(eligible):
                gain = baseline[c]-after[c]
                gains[c][donor] = gain
                rows.append(dict(round=self.pending['round'], class_id=c, donor=donor,
                    donor_class_count=int(self.counts[donor, c]), baseline_la=baseline[c],
                    injected_la=after[c], gain=gain, positive=gain > self.config['min_gain']))
        self.copy_parameters(ordinary_b)
        mixtures = select_sources(gains, self.counts, self.target_clients,
                                  self.config['donors_per_class'], self.config['min_gain'])
        for row in rows:
            row['source_weight'] = mixtures.get(row['class_id'], {}).get(row['donor'], 0.)
            row['selected'] = row['source_weight'] > 0
        summary['candidate_pairs'] = len(rows)
        return mixtures, rows

    def calibrate_bases(self, bases):
        matrices = {key: torch.nn.Parameter(torch.zeros(basis.shape[0], self.ranks[key], self.ranks[key],
                    device=self.device, dtype=basis.dtype)) for key, basis in bases.items()}
        optimizer = torch.optim.Adam(list(matrices.values()), lr=self.config['learning_rate'],
                                     betas=(.9, .999), eps=1e-8, weight_decay=0.)
        summary = self.pending['summary']
        summary['learned_c_parameters'] = sum(v.numel() for v in matrices.values())
        parameter_bytes = sum(v.numel()*v.element_size() for v in matrices.values())
        for step in range(self.config['steps']):
            optimizer.zero_grad(set_to_none=True)
            data_loss = 0.
            for k in self.recipients:
                positions = self.monitors[k]['tail_positions']
                local_loss, local_images = 0., 0
                for begin in range(0, len(positions), self.feedback_batch_size):
                    x, y = self.batch(k, positions[begin:begin+self.feedback_batch_size])
                    residual = reconstruct_class_residual(bases, matrices, len(self.class_totals))
                    with differentiable_b_residual(self.modules, residual):
                        losses, _, _ = self.forward(x, y)
                        objective = class_macro_loss(losses, y, self.class_totals)
                        if not torch.isfinite(objective):
                            raise ValueError('Nonfinite directed tail loss')
                        objective.backward()
                    local_loss += float(objective.detach())
                    local_images += len(y)
                    summary['client_backward_batches'] += 1
                data_loss += local_loss
                summary['algorithm_backward_images'] += local_images
                self.pending['feedback'].append(dict(round=self.pending['round'], step=step+1, client_id=k,
                    shared_c_version=step, weighted_tail_la=local_loss, tail_samples=local_images,
                    non_tail_samples=0, objective='global sample-mean per target class, then class macro'))
            penalty = sum(c.square().sum() for c in matrices.values())/(len(next(iter(matrices.values())))*len(self.keys))
            (self.config['regularization']*penalty).backward()
            if not all(c.grad is not None and torch.isfinite(c.grad).all() for c in matrices.values()):
                raise ValueError('Nonfinite or missing directed C gradient')
            gradient_norm = math.sqrt(sum(float(c.grad.square().sum()) for c in matrices.values()))
            penalty_value = self.config['regularization']*float(penalty.detach())
            optimizer.step()
            if not all(torch.isfinite(c).all() for c in matrices.values()):
                raise ValueError('Nonfinite directed C matrix')
            summary['optimizer_steps'] += 1
            summary['feedback_synchronizations'] += 1
            summary['extra_downlink_bytes'] += len(self.recipients)*parameter_bytes
            summary['extra_upload_bytes'] += len(self.recipients)*(parameter_bytes+4)
            self.pending['steps'].append(dict(round=self.pending['round'], step=step+1,
                la_before=data_loss, regularization_penalty_before=penalty_value,
                objective_before=data_loss+penalty_value, c_gradient_norm=gradient_norm,
                supported_classes=len(next(iter(matrices.values()))), target_classes=len(self.class_totals)))
            print(f'B-directed r{self.pending["round"]:03d} step={step+1} tail LA={data_loss:.6f}', flush=True)
        return {key: c.detach().clone() for key, c in matrices.items()}

    def apply_shared(self, start, ordinary, rnd):
        started = time.perf_counter()
        folder = self.runtime.root/'b_transfer_rounds'/f'r{rnd:03d}'
        folder.mkdir(parents=True, exist_ok=True)
        summary = dict(round=rnd, mode='shared', calibration_profile='tail_directed',
            learning_rate=self.config['learning_rate'], aggregation=self.config['aggregation'],
            recipients=len(self.recipients), target_classes=len(self.class_totals), non_tail_samples=0,
            unique_donors=0, donor_links=0, candidate_pairs=0, supported_classes=0,
            optimizer_steps=0, client_backward_batches=0, feedback_synchronizations=0,
            learned_c_parameters=0, algorithm_forward_images=0, algorithm_backward_images=0,
            diagnostic_forward_images=0, extra_downlink_bytes=0, extra_upload_bytes=0, seconds=0.)
        self.pending = dict(round=rnd, folder=folder, summary=summary, metrics=[], steps=[], receivers=[], feedback=[])
        ordinary_b = {key: ordinary[key] for key in self.keys}
        if len(self.selected) != len(set(self.selected)) or any(j not in self.original_deltas for j in self.selected):
            raise ValueError('Invalid cached donor updates')
        with self.session():
            self.copy_parameters(ordinary)
            before = self.measure_shared('ordinary_global_B')
            mixtures, donor_rows = self.screen_sources(ordinary_b)
            classes, bases = build_class_bases(self.original_deltas, mixtures, self.keys,
                                              self.counts, self.target_clients)
            bases = {key: v.to(self.device) for key, v in bases.items()}
            if classes:
                summary['extra_downlink_bytes'] += len(self.recipients)*sum(v.numel()*v.element_size() for v in bases.values())
                matrices = self.calibrate_bases(bases)
                residual = {key: v.detach().cpu() for key, v in
                    reconstruct_class_residual(bases, matrices, len(self.class_totals)).items()}
            else:
                matrices = {}
                residual = {key: torch.zeros_like(value).cpu() for key, value in ordinary_b.items()}
            committed = dict(ordinary)
            if not all(torch.isfinite(value).all() for value in residual.values()):
                raise ValueError('Nonfinite directed residual')
            committed.update({key: ordinary_b[key]+residual[key].to(ordinary_b[key]) for key in self.keys})
            self.copy_parameters(committed)
            after = self.measure_shared('transferred_global_B')
        union = sorted({j for w in mixtures.values() for j in w})
        summary.update(unique_donors=len(union), donor_links=sum(map(len, mixtures.values())),
            supported_classes=len(classes), unsupported_classes=len(self.class_totals)-len(classes),
            effective_global_transfer_norm=self.effective_norm(residual, start),
            ordinary_B_effective_update_norm=self.effective_norm(
                {key: ordinary[key]-start[key] for key in self.keys}, start))
        summary['extra_downlink_bytes'] += len(self.recipients)*sum(v.numel()*v.element_size() for v in ordinary_b.values())
        # Explicit class loss weights and class-presence metadata, once per event.
        units = sum(len(v) for v in self.probes.values())
        summary['extra_upload_bytes'] += units*8
        summary['extra_downlink_bytes'] += units*8
        if rnd == self.config['rounds'][0]:
            summary['extra_downlink_bytes'] += self.counts.numel()
        for k in self.recipients:
            self.pending['receivers'].append(dict(round=rnd, client_id=k, measurement_scope='shared_B_before_after',
                shared_donors=len(union), effective_transfer_norm=summary['effective_global_transfer_norm'],
                **{'before_'+name: value for name, value in before[k].items()},
                **{'after_'+name: value for name, value in after[k].items()}))
        self.last_mixtures, self.last_bases = mixtures, {key: v.detach().cpu() for key, v in bases.items()}
        self.last_matrices = {key: v.detach().cpu() for key, v in matrices.items()}
        self.last_residual = residual
        write_csv(folder/'donor_scores.csv', donor_rows)
        source_rows = [dict(round=rnd, class_id=c, donor=j, weight=w, donor_class_count=int(self.counts[j,c]))
                       for c, weights in mixtures.items() for j, w in weights.items()]
        write_csv(folder/'source_weights.csv', source_rows)
        write_json(folder/'calibration_manifest.json', dict(mode='shared', profile='tail_directed',
            target_clients=self.target_clients, target_class_totals=self.class_totals,
            feedback_clients=self.recipients, source_mixtures=mixtures, basis_class_ids=classes,
            shared_donor_ids=union, union_role='bookkeeping only; C indexes classes, not donors',
            module_order=self.keys, recipients={k:dict(tail_positions=self.monitors[k]['tail_positions'], non_tail_positions=[])
                                               for k in self.recipients}))
        weights = np.asarray([[mixtures[c].get(j, 0.) for j in union] for c in classes], dtype=np.float64).reshape(len(classes), len(union))
        np.savez_compressed(folder/'matrices.npz', basis_class_ids=np.asarray(classes, dtype=np.int64),
            donor_ids=np.asarray(union, dtype=np.int64), source_weights=weights,
            **{f'module_{i:02d}': self.last_matrices[key].numpy() for i, key in enumerate(self.keys) if key in matrices})
        torch.save(dict(round=rnd, profile='tail_directed', source_mixtures=mixtures, basis_class_ids=classes,
            class_bases=self.last_bases, class_matrices=self.last_matrices,
            ordinary_lora={k:ordinary[k] for k in self.runtime.keys},
            committed_lora={k:committed[k] for k in self.runtime.keys}, transfer_residual=residual), folder/'commit.pt')
        summary['seconds'] = time.perf_counter()-started
        self.write_pending()
        self.original_deltas = None
        return committed
