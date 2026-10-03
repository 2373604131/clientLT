"""Opt-in donor-C guards; all legacy training modules remain unchanged."""
import math

import numpy as np
import torch

from utils.b_class_guard_math import guard_losses
from utils.cliplora_a_refresh import isolated_rng
from utils.cliplora_b_shared_transfer import SharedDonorBTransfer, shared_calibration_loss
from utils.cliplora_b_transfer import differentiable_b_residual
from utils.cliplora_bridge_audit import write_csv


def tensor_norm(values):
    return math.sqrt(sum(float(x.detach().square().sum()) for x in values))


class GuardedSharedDonorBTransfer(SharedDonorBTransfer):
    def selected_guard(self, values):
        return values[self.config['guard_variant']]

    def checked_reference(self, client, index, sampled, labels):
        saved = self._guard_references[client, index]
        positions = sampled['tail_positions'] + sampled['non_tail_positions']
        if positions != saved['positions'] or not torch.equal(labels.detach().cpu(), saved['labels']):
            raise ValueError('Guard reference sample identities/order changed within an event')
        return saved['losses']

    @torch.no_grad()
    def guard_trace(self, donors, matrices, manifests, start, step, ordinary_norm):
        clients, steps = len(self.recipients), self.config['steps']
        residual = {k: torch.bmm(donors[k], matrices[k]).mean(0) for k in self.keys}
        la, harm = [0.]*steps, [0.]*steps
        tail_la, other_la, tail_acc, other_acc = [], [], [], []
        images_used = 0
        with isolated_rng(), differentiable_b_residual(self.modules, residual):
            for client in self.recipients:
                for index, sampled in enumerate(manifests[client]['batches']):
                    positions = sampled['tail_positions']+sampled['non_tail_positions']
                    images, labels = self.batch(client, positions)
                    losses, logits, _ = self.forward(images, labels, diagnostic=True)
                    n = len(sampled['tail_positions'])
                    if step == 0:
                        self._guard_references[client, index] = dict(positions=list(positions),
                            labels=labels.detach().cpu().clone(), losses=losses.detach().cpu().clone())
                    reference = self.checked_reference(client, index, sampled, labels)
                    guards = guard_losses(losses, reference, labels, n, self.config['tail_weight'])
                    local, parts = shared_calibration_loss(losses, n, self.config['tail_weight'])
                    la[index] += float(local)/clients
                    harm[index] += float(self.selected_guard(guards))/clients
                    correct = (logits.argmax(1) == labels).float()
                    tail_la.append(float(parts[0])); tail_acc.append(100*float(correct[:n].mean()))
                    if len(parts) == 2:
                        other_la.append(float(parts[1])); other_acc.append(100*float(correct[n:].mean()))
                    info = dict(round=self.pending['round'], c_step=step, batch=index+1, client_id=client,
                                client_weight=1/clients, guard_variant=self.config['guard_variant'])
                    self.pending['guard_cells'].extend({**info, **cell} for cell in guards['cells'])
                    self.pending['guard_clients'].append(dict(info, client_guard=float(guards['client']),
                        class_guard=float(guards['class']), selected_guard=float(self.selected_guard(guards)),
                        active_classes=sum(x['active'] for x in guards['cells']), observed_classes=len(guards['cells'])))
                    images_used += len(labels)
        raw_reg = sum(float(c.square().sum()) for c in matrices.values())/(len(next(iter(matrices.values())))*len(self.keys))
        penalty = self.config['regularization']*raw_reg
        beta = self.config['guard_weight']
        norm = self.effective_norm(residual, start)
        row = dict(round=self.pending['round'], c_step=step, guard_variant=self.config['guard_variant'],
            guard_weight=beta, learning_rate=self.config['learning_rate'], scope='same_two_calibration_batches',
            tail_weight=self.config['tail_weight'], fixed_pool_la=sum(la)/steps, fixed_pool_guard=sum(harm)/steps,
            fixed_pool_objective=(sum(la)+beta*sum(harm))/steps+penalty,
            regularizer=raw_reg, regularization_penalty=penalty, c_norm=tensor_norm(matrices.values()),
            effective_transfer_norm=norm, ordinary_B_effective_update_norm=ordinary_norm,
            transfer_to_ordinary_ratio=norm/ordinary_norm if ordinary_norm else None,
            tail_la=float(np.mean(tail_la)), non_tail_la=float(np.mean(other_la)) if other_la else None,
            tail_accuracy=float(np.mean(tail_acc)), non_tail_accuracy=float(np.mean(other_acc)) if other_acc else None,
            diagnostic_forward_images=images_used,
            **{f'batch{i+1}_la': x for i,x in enumerate(la)},
            **{f'batch{i+1}_guard': x for i,x in enumerate(harm)})
        self.pending['loss_trace'].append(row)
        folder = self.pending['folder']
        write_csv(folder/'c_loss_trace.csv', self.pending['loss_trace'])
        write_csv(folder/'guard_class_trace.csv', self.pending['guard_cells'])
        write_csv(folder/'guard_client_trace.csv', self.pending['guard_clients'])
        if step == 0:
            arrays = {}
            for (client,index), saved in self._guard_references.items():
                prefix = f'client{client:02d}_batch{index+1}'
                arrays[prefix+'_positions'] = np.asarray(saved['positions'], dtype=np.int64)
                arrays[prefix+'_labels'] = saved['labels'].numpy()
                arrays[prefix+'_losses'] = saved['losses'].numpy()
            np.savez_compressed(folder/'guard_references.npz', **arrays)
        return row

    def calibrate_shared(self, donor_ids, manifests, start):
        # Exact legacy path when disabled, including the old gradient arithmetic.
        if self.config['guard_variant'] == 'off':
            return super().calibrate_shared(donor_ids, manifests, start)
        m, layers, clients = len(donor_ids), len(self.keys), len(self.recipients)
        if not m or not clients or self.config['steps'] != 2:
            raise ValueError('Guard calibration needs donors, recipients and two fixed steps')
        donors = {k: torch.stack([self.original_deltas[j][k] for j in donor_ids]).detach().to(self.device)
                  for k in self.keys}
        matrices = {k: torch.nn.Parameter(torch.zeros(m,self.ranks[k],self.ranks[k],
                    device=self.device,dtype=donors[k].dtype)) for k in self.keys}
        parameters = tuple(matrices.values())
        optimizer = torch.optim.Adam(parameters, lr=self.config['learning_rate'], betas=(.9,.999), eps=1e-8, weight_decay=0.)
        summary = self.pending['summary']
        parameter_bytes = sum(c.numel()*c.element_size() for c in parameters)
        summary.update(learned_c_parameters=sum(c.numel() for c in parameters),
                       guard_diagnostic_backward_images=0, guard_gradient_diagnostic_batches=0)
        self.pending.update(loss_trace=[], guard_cells=[], guard_clients=[])
        self._guard_references = {}
        ordinary_norm = self.effective_norm({k:self.parameters[k].detach().cpu()-start[k].detach().cpu() for k in self.keys},start)
        trace = self.guard_trace(donors,matrices,manifests,start,0,ordinary_norm)
        beta = self.config['guard_weight']
        for step in (1,2):
            optimizer.zero_grad(set_to_none=True)
            guard_grads = [torch.zeros_like(c) for c in parameters]
            mean_la, mean_guard, tail_samples, other_samples = 0.,0.,0,0
            active, observed = 0,0
            for client in self.recipients:
                sampled = manifests[client]['batches'][step-1]
                images, labels = self.batch(client,sampled['tail_positions']+sampled['non_tail_positions'])
                n = len(sampled['tail_positions'])
                residual = {k:torch.bmm(donors[k],matrices[k]).mean(0) for k in self.keys}
                with differentiable_b_residual(self.modules,residual):
                    losses,_,_ = self.forward(images,labels)
                    base,parts = shared_calibration_loss(losses,n,self.config['tail_weight'])
                    ref = self.checked_reference(client,step-1,sampled,labels)
                    guards = guard_losses(losses,ref,labels,n,self.config['tail_weight'])
                    h = self.selected_guard(guards)
                    # Read-only extra traversal for attribution; it does not update .grad.
                    if float(h.detach()) > 0:
                        hg = torch.autograd.grad(beta*h/clients,parameters,retain_graph=True)
                        for dst,src in zip(guard_grads,hg):
                            dst.add_(src.detach())
                        summary['guard_diagnostic_backward_images'] += len(labels)
                        summary['guard_gradient_diagnostic_batches'] += 1
                    ((base+beta*h)/clients).backward()
                mean_la += float(base.detach())/clients
                mean_guard += float(h.detach())/clients
                tail_samples += n; other_samples += len(labels)-n
                count_active = sum(cell['active'] for cell in guards['cells'])
                active += count_active; observed += len(guards['cells'])
                summary['algorithm_backward_images'] += len(labels)
                summary['client_backward_batches'] += 1
                self.pending['feedback'].append(dict(round=self.pending['round'],step=step,client_id=client,
                    shared_c_version=step-1,client_weight=1/clients,tail_samples=n,non_tail_samples=len(labels)-n,
                    local_la=float(base.detach()),tail_la=float(parts[0].detach()),
                    non_tail_la=float(parts[1].detach()) if len(parts)==2 else None,
                    tail_loss_weight=self.config['tail_weight'] if len(parts)==2 else 1.,
                    non_tail_loss_weight=1-self.config['tail_weight'] if len(parts)==2 else 0.,
                    guard_variant=self.config['guard_variant'],guard_loss=float(h.detach()),
                    client_guard=float(guards['client'].detach()),class_guard=float(guards['class'].detach()),
                    active_classes=count_active,observed_classes=len(guards['cells'])))
            base_norm = tensor_norm(c.grad-hg for c,hg in zip(parameters,guard_grads))
            guard_norm = tensor_norm(guard_grads)
            regularizer = sum(c.square().sum() for c in parameters)/(m*layers)
            penalty_before = self.config['regularization']*float(regularizer.detach())
            (self.config['regularization']*regularizer).backward()
            if not all(torch.isfinite(c.grad).all() for c in parameters):
                raise ValueError('Nonfinite shared-C gradient')
            total_norm = tensor_norm(c.grad for c in parameters)
            previous = {k:c.detach().clone() for k,c in matrices.items()}
            optimizer.step()
            if not all(torch.isfinite(c).all() for c in parameters):
                raise ValueError('Nonfinite shared-C candidate')
            summary['optimizer_steps'] += 1; summary['feedback_synchronizations'] += 1
            summary['extra_downlink_bytes'] += clients*parameter_bytes
            summary['extra_upload_bytes'] += clients*(parameter_bytes+4)
            after = self.guard_trace(donors,matrices,manifests,start,step,ordinary_norm)
            objective_before = mean_la+beta*mean_guard+penalty_before
            objective_after = after[f'batch{step}_la']+beta*after[f'batch{step}_guard']+after['regularization_penalty']
            self.pending['steps'].append(dict(round=self.pending['round'],step=step,scope='shared',
                guard_variant=self.config['guard_variant'],guard_weight=beta,feedback_clients=clients,donors=m,
                tail_samples=tail_samples,non_tail_samples=other_samples,la_before=mean_la,guard_before=mean_guard,
                regularizer_before=float(regularizer.detach()),objective_before=objective_before,
                base_la_gradient_norm=base_norm,guard_gradient_norm=guard_norm,
                guard_to_base_gradient_ratio=guard_norm/base_norm if base_norm else None,
                active_classes=active,observed_classes=observed,c_gradient_norm=total_norm,
                c_norm_after=after['c_norm'],c_update_norm=tensor_norm(c.detach()-previous[k] for k,c in matrices.items()),
                effective_transfer_norm_after=after['effective_transfer_norm'],
                transfer_to_ordinary_ratio=after['transfer_to_ordinary_ratio'],
                same_batch_la_after=after[f'batch{step}_la'],same_batch_la_change=after[f'batch{step}_la']-mean_la,
                same_batch_guard_after=after[f'batch{step}_guard'],objective_after_same_batch=objective_after,
                objective_change_same_batch=objective_after-objective_before,
                fixed_pool_la_before=trace['fixed_pool_la'],fixed_pool_la_after=after['fixed_pool_la'],
                fixed_pool_objective_before=trace['fixed_pool_objective'],fixed_pool_objective_after=after['fixed_pool_objective']))
            print(f'B-guard {self.config["guard_variant"]} r{self.pending["round"]:03d} step={step}/2 '
                  f'LA={mean_la:.6f} H={mean_guard:.6g} |gradH|={guard_norm:.6g} '
                  f'|sRA|={after["effective_transfer_norm"]:.6g}',flush=True)
            trace = after
        self._guard_references = {}
        return {k:c.detach().cpu().clone() for k,c in matrices.items()}

    def write_pending(self):
        self.pending['summary'].update(guard_variant=self.config['guard_variant'],guard_weight=self.config['guard_weight'])
        self.pending['summary'].setdefault('guard_diagnostic_backward_images',0)
        super().write_pending()

    def progress(self):
        return dict(super().progress(),guard_variant=self.config['guard_variant'],guard_weight=self.config['guard_weight'])
