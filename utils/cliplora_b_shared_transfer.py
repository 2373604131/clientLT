"""Joint donor C calibration on the actual post-FedAvg B, with fixed A.

The original recipient list, calibration batches and local LA loss are reused.
All recipients see the SAME C at each step; Adam steps only after every backward.
"""
import math
import time

import numpy as np
import torch

from utils.cliplora_b_transfer import (DonorBTransfer, calibration_batches,
                                      differentiable_b_residual, reconstruct_residual,
                                      transfer_config)
from utils.cliplora_bridge_audit import write_csv, write_json


def shared_transfer_config(args):
    if getattr(args, 'sfra_b_aggregation', 'sample') != 'sample':
        raise ValueError('Shared donor C uses ordinary sample-weighted B FedAvg.')
    config = transfer_config(args)
    config.update(schema_version='donor_b_shared_v1', mode='shared',
        donor_rule='source_missing_class_and_positive_LA_gain_at_ordinary_shared_B',
        donor_pool='union of positive donors across original recipients; fixed during calibration',
        c_scope='one full rank-by-rank matrix per donor and module, shared across recipients',
        feedback_clients='original tail-present recipients only; no extra non-tail clients',
        calibration_anchor='actual ordinary post-FedAvg B with round-start A',
        calibration_objective='equal recipient mean of original local group-balanced LA, plus one C penalty',
        calibration_update='two synchronous global Adam steps; all client gradients before each step',
        aggregation='ordinary_sample_weighted_B_then_one_shared_residual_without_recipient_weights',
        diagnostic_anchor='ordinary_shared_B versus transferred_shared_B; not local before/after',
        communication='modeled dense FP32 donor/C/gradient messages; two feedback synchronizations',
        commit_rule='fixed_second_C_step; no evaluation-based selection')
    return config


class SharedDonorBTransfer(DonorBTransfer):
    def cache_updates(self, deltas, selected):
        # Cache only RAW ordinary local updates, never transformed/recipient updates.
        self.original_deltas = deltas
        self.selected = list(selected)

    def measure_shared(self, phase):
        rows = {k: self.measure_client(k, phase) for k in self.recipients}
        for key in ('tail_la', 'non_tail_la', 'tail_accuracy', 'non_tail_accuracy'):
            values = [r[key] for r in rows.values() if r[key] is not None]
            self.pending['summary'][phase+'_'+key] = float(np.mean(values)) if values else None
        return rows

    def calibrate_shared(self, donor_ids, manifests, start):
        m, layers, clients = len(donor_ids), len(self.keys), len(self.recipients)
        donor_tensors = {
            key: torch.stack([self.original_deltas[j][key] for j in donor_ids]).detach().to(self.device)
            for key in self.keys}
        matrices = {key: torch.nn.Parameter(torch.zeros(
            m, self.ranks[key], self.ranks[key], device=self.device, dtype=donor_tensors[key].dtype))
            for key in self.keys}
        optimizer = torch.optim.Adam(list(matrices.values()), lr=self.config['learning_rate'],
                                     betas=(.9, .999), eps=1e-8, weight_decay=0.)
        summary = self.pending['summary']
        parameter_bytes = sum(c.numel()*c.element_size() for c in matrices.values())
        summary['learned_c_parameters'] = sum(c.numel() for c in matrices.values())
        for step in range(1, self.config['steps']+1):
            optimizer.zero_grad(set_to_none=True)
            mean_loss, tail_samples, non_tail_samples = 0., 0, 0
            # Rebuild the small residual graph per client, releasing each CLIP graph
            # after backward. C and the shared base B NEVER change inside this loop.
            for client in self.recipients:
                sampled = manifests[client]['batches'][step-1]
                images, labels = self.batch(client, sampled['tail_positions']+sampled['non_tail_positions'])
                count_tail = len(sampled['tail_positions'])
                residual = {key: torch.bmm(donor_tensors[key], matrices[key]).mean(0) for key in self.keys}
                with differentiable_b_residual(self.modules, residual):
                    losses, _, _ = self.forward(images, labels)
                    parts = [losses[:count_tail].mean()]
                    if count_tail < len(labels):
                        parts.append(losses[count_tail:].mean())
                    local_loss = torch.stack(parts).mean()
                    (local_loss/clients).backward()
                mean_loss += float(local_loss.detach())/clients
                tail_samples += count_tail
                non_tail_samples += len(labels)-count_tail
                summary['algorithm_backward_images'] += len(labels)
                summary['client_backward_batches'] += 1
                self.pending['feedback'].append(dict(
                    round=self.pending['round'], step=step, client_id=client,
                    shared_c_version=step-1, client_weight=1/clients,
                    tail_samples=count_tail, non_tail_samples=len(labels)-count_tail,
                    local_la=float(local_loss.detach()),
                    tail_la=float(parts[0].detach()),
                    non_tail_la=float(parts[1].detach()) if len(parts) == 2 else None))
            # One regularizer for the shared C, not one full regularizer per client.
            regularizer = sum(c.square().sum() for c in matrices.values())/(m*layers)
            (self.config['regularization']*regularizer).backward()
            grad_norm = math.sqrt(sum(float(c.grad.square().sum()) for c in matrices.values()))
            optimizer.step()
            summary['optimizer_steps'] += 1
            summary['feedback_synchronizations'] += 1
            summary['extra_downlink_bytes'] += clients*parameter_bytes
            summary['extra_upload_bytes'] += clients*(parameter_bytes+4)  # dL/dC and scalar LA
            with torch.no_grad():
                updated = {key: torch.bmm(donor_tensors[key], matrices[key]).mean(0) for key in self.keys}
                norm = self.effective_norm(updated, start)
            self.pending['steps'].append(dict(
                round=self.pending['round'], step=step, scope='shared', feedback_clients=clients,
                donors=m, tail_samples=tail_samples, non_tail_samples=non_tail_samples,
                la_before=mean_loss, regularizer_before=float(regularizer.detach()),
                objective_before=mean_loss+self.config['regularization']*float(regularizer.detach()),
                c_gradient_norm=grad_norm,
                c_norm_after=math.sqrt(sum(float(c.detach().square().sum()) for c in matrices.values())),
                effective_transfer_norm_after=norm))
            print(f'B-shared r{self.pending["round"]:03d} step={step}/2 '
                  f'clients={clients} donors={m} LA={mean_loss:.6f} |sRA|={norm:.6g}', flush=True)
        return {key: c.detach().cpu().clone() for key, c in matrices.items()}

    def apply_shared(self, start, ordinary, rnd):
        started = time.perf_counter()
        folder = self.runtime.root/'b_transfer_rounds'/f'r{rnd:03d}'
        folder.mkdir(parents=True, exist_ok=True)
        summary = dict(round=rnd, mode='shared', learning_rate=self.config['learning_rate'],
            aggregation=self.config['aggregation'], recipients=len(self.recipients),
            receivers_with_donors=0, donor_links=0, unique_donors=0, candidate_pairs=0,
            optimizer_steps=0, client_backward_batches=0, feedback_synchronizations=0,
            learned_c_parameters=0, algorithm_forward_images=0, algorithm_backward_images=0,
            diagnostic_forward_images=0, extra_downlink_bytes=0, extra_upload_bytes=0, seconds=0.)
        self.pending = dict(round=rnd, folder=folder, summary=summary, metrics=[],
                            steps=[], receivers=[], feedback=[])
        donor_rows, manifests, candidate_sets, pool = [], {}, {}, set()
        ordinary_b = {key: ordinary[key] for key in self.keys}
        delta_bytes = sum(ordinary[key].numel()*ordinary[key].element_size() for key in self.keys)
        summary['extra_downlink_bytes'] += len(self.recipients)*delta_bytes  # shared base B
        with self.session():
            self.copy_parameters(ordinary)
            before = self.measure_shared('ordinary_global_B')
            for client in self.recipients:
                self.copy_parameters(ordinary_b)
                positions = [p for values in self.probes[client].values() for p in values]
                baseline = self.score_positions(client, positions, False)
                candidates = [j for j in self.selected if j != client and
                              any(self.counts[j, c] == 0 for c in self.probes[client])]
                candidate_sets[client] = set(candidates)
                donors = []
                for donor in candidates:
                    proposal = {key: ordinary_b[key]+self.config['probe_step']*self.original_deltas[donor][key]
                                for key in self.keys}
                    self.copy_parameters(proposal)  # Never cumulative, never local B.
                    measured = self.score_positions(client, positions, False)
                    accepted = False
                    for c in sorted(self.probes[client]):
                        if self.counts[donor, c] != 0:
                            continue
                        gain = baseline[c]['la']-measured[c]['la']
                        accepted |= gain > 0.
                        donor_rows.append(dict(round=rnd, receiver=client, donor=donor, class_id=c,
                            anchor='ordinary_shared_B', baseline_la=baseline[c]['la'],
                            injected_la=measured[c]['la'], gain=gain, positive=gain > 0.))
                    if accepted:
                        donors.append(donor)
                donors.sort()
                pool.update(donors)
                manifests[client] = dict(donor_ids=donors, batches=calibration_batches(
                    self.groups[client], self.tail, self.runtime.args.seed, rnd, client))
                summary['candidate_pairs'] += len(candidates)
                summary['receivers_with_donors'] += bool(donors)
                summary['donor_links'] += len(donors)
                summary['extra_downlink_bytes'] += len(candidates)*delta_bytes
                if rnd == self.config['rounds'][0]:
                    summary['extra_downlink_bytes'] += self.counts.numel()  # cached uint8 presence table
                summary['extra_upload_bytes'] += 4*(len(donors)+1)  # accepted IDs plus count
                print(f'B-shared r{rnd:03d} probe client={client}: '
                      f'{len(donors)}/{len(candidates)} donors', flush=True)
            donor_ids = sorted(pool)
            summary['unique_donors'] = len(donor_ids)
            self.copy_parameters(ordinary_b)
            if donor_ids:
                # Every recipient evaluates the whole union, not just its own donors.
                # Reuse screened donor tensors, send only missing union members.
                summary['extra_downlink_bytes'] += len(self.recipients)*4*(len(donor_ids)+1)
                summary['extra_downlink_bytes'] += sum(
                    len(pool-candidate_sets[k])*delta_bytes for k in self.recipients)
                returned_c = self.calibrate_shared(donor_ids, manifests, start)
                residual = reconstruct_residual(self.original_deltas, donor_ids, returned_c, self.keys)
            else:
                returned_c = {}
                residual = {key: torch.zeros_like(ordinary_b[key]) for key in self.keys}
            committed = dict(ordinary)
            committed.update({key: ordinary_b[key]+residual[key] for key in self.keys})
            self.copy_parameters(committed)
            after = self.measure_shared('transferred_global_B')
        norm = self.effective_norm(residual, start)
        summary['effective_global_transfer_norm'] = norm
        summary['ordinary_B_effective_update_norm'] = self.effective_norm(
            {key: ordinary_b[key]-start[key] for key in self.keys}, start)
        for client in self.recipients:
            self.pending['receivers'].append(dict(round=rnd, client_id=client,
                measurement_scope='shared_B_before_after', candidates=len(candidate_sets[client]),
                donors=len(manifests[client]['donor_ids']), shared_donors=len(donor_ids),
                feedback_weight=1/len(self.recipients), sample_weight=self.runtime.q[client],
                effective_transfer_norm=norm,
                **{'before_'+k: v for k, v in before[client].items()},
                **{'after_'+k: v for k, v in after[client].items()}))
        write_csv(folder/'donor_scores.csv', donor_rows)
        write_json(folder/'calibration_manifest.json', dict(
            mode='shared', module_order=self.keys, shared_donor_ids=donor_ids,
            feedback_clients=self.recipients, recipients=manifests))
        np.savez_compressed(folder/'matrices.npz', donor_ids=np.asarray(donor_ids, dtype=np.int64),
            **{f'module_{i:02d}': returned_c[key].numpy() for i, key in enumerate(self.keys) if key in returned_c})
        # Keep the raw ordinary-FedAvg bridge dump intact; this is its shared commit.
        torch.save(dict(round=rnd, donor_ids=donor_ids,
                        ordinary_lora={k: ordinary[k] for k in self.runtime.keys},
                        committed_lora={k: committed[k] for k in self.runtime.keys},
                        transfer_residual=residual), folder/'commit.pt')
        summary['seconds'] += time.perf_counter()-started
        self.write_pending()
        self.original_deltas = None
        return committed

    def observe_global(self, state, rnd, committed=False):
        if not self.scheduled(rnd) or not committed:
            # apply_shared already measured the exact transferred shared state.
            return
        started = time.perf_counter()
        with self.session():
            self.copy_parameters(state)
            self.measure_shared('committed_global')
        summary = self.pending['summary']
        summary['seconds'] += time.perf_counter()-started
        summary['A_updated_this_round'] = rnd <= 90
        summary['tail_la_gain_after_B_aggregation'] = (summary['ordinary_global_B_tail_la']
                                                      - summary['transferred_global_B_tail_la'])
        summary['tail_la_change_during_A'] = (summary['committed_global_tail_la']
                                            - summary['transferred_global_B_tail_la'])
        summary['shared_receiver_tail_la_gain_mean'] = float(np.mean([
            r['before_tail_la']-r['after_tail_la'] for r in self.pending['receivers']]))
        self.records.append(dict(summary))
        self.write_pending()
        self.pending = None

    def write_pending(self):
        super().write_pending()
        write_csv(self.pending['folder']/'client_feedback_steps.csv', self.pending['feedback'])

    def progress(self):
        return dict(super().progress(), b_transfer_mode='shared',
                    b_transfer_client_backward_batches=sum(r['client_backward_batches'] for r in self.records),
                    b_transfer_feedback_synchronizations=sum(r['feedback_synchronizations'] for r in self.records))
