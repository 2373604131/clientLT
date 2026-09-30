"""Extra classification calibration without donor bases; training-norm matched.

This is a reference-dependent control, not a deployable replacement for B.
The paired AB supplies only training-side residual norms and sample identities.
"""
import math
import time

import torch

from tools.sfra.calibration_reference import protocol_signature
from utils.cliplora_b_shared_transfer import (SharedDonorBTransfer, shared_calibration_batches,
                                              shared_calibration_loss)
from utils.cliplora_b_transfer import differentiable_b_residual
from utils.cliplora_bridge_audit import write_csv, write_json


class DirectCalibrationControl(SharedDonorBTransfer):
    def __init__(self, runtime):
        super().__init__(runtime)
        reference = self.config['norm_reference']
        if protocol_signature(runtime.root) != reference['signature']:
            raise ValueError('Calibration reference differs in initialization, partition, witnesses or execution')
        write_json(runtime.root/'calibration_reference.json', reference)

    def cache_updates(self, deltas, selected):
        # No individual or aggregated donor update enters this control's basis.
        self.original_deltas = None

    @torch.no_grad()
    def project(self, residual, target, start):
        before = self.effective_norm(residual, start)
        if not math.isfinite(before) or not math.isfinite(target) or target < 0:
            raise ValueError('Nonfinite/negative direct calibration norm')
        if target == 0:
            for value in residual.values():
                value.zero_()
            scale = 0.
        else:
            if before <= 1e-15:
                raise ValueError('Cannot match a positive norm using a zero effective update')
            scale = target / before
            for value in residual.values():
                value.mul_(scale)
        actual = self.effective_norm(residual, start)
        if not math.isclose(actual, target, rel_tol=1e-5, abs_tol=1e-10):
            raise ValueError('Direct calibration effective norm mismatch')
        return before, scale, actual

    @torch.no_grad()
    def trace(self, residual, manifests, step):
        losses = [0., 0.]
        with differentiable_b_residual(self.modules, residual):
            for client in self.recipients:
                for index, batch in enumerate(manifests[client]['batches']):
                    x,y = self.batch(client, batch['tail_positions']+batch['non_tail_positions'])
                    per_image,_,_ = self.forward(x,y,diagnostic=True)
                    loss,_ = shared_calibration_loss(per_image,len(batch['tail_positions']),.35)
                    losses[index] += float(loss)/len(self.recipients)
        row = dict(round=self.pending['round'], c_step=step, fixed_pool_la=sum(losses)/2,
            fixed_pool_objective=sum(losses)/2, regularization_penalty=0.,
            batch1_la=losses[0], batch2_la=losses[1], profile='direct_norm_matched')
        self.pending['loss_trace'].append(row)
        return row

    def calibrate(self, ordinary, start, manifests, norms):
        residual = {k:torch.nn.Parameter(torch.zeros_like(ordinary[k],device=self.device)) for k in self.keys}
        optimizer = torch.optim.Adam(list(residual.values()),lr=.3,betas=(.9,.999),eps=1e-8,weight_decay=0.)
        summary = self.pending['summary']
        parameter_bytes = sum(v.numel()*v.element_size() for v in residual.values())
        summary['learned_direct_parameters'] = sum(v.numel() for v in residual.values())
        self.trace(residual,manifests,0)
        for step in (1,2):
            optimizer.zero_grad(set_to_none=True)
            total = 0.
            for client in self.recipients:
                batch = manifests[client]['batches'][step-1]
                x,y = self.batch(client,batch['tail_positions']+batch['non_tail_positions'])
                with differentiable_b_residual(self.modules,residual):
                    per_image,_,_ = self.forward(x,y)
                    loss,parts = shared_calibration_loss(per_image,len(batch['tail_positions']),.35)
                    if not torch.isfinite(loss):
                        raise ValueError('Nonfinite direct calibration loss')
                    (loss/len(self.recipients)).backward()
                total += float(loss.detach())/len(self.recipients)
                summary['algorithm_backward_images'] += len(y)
                summary['client_backward_batches'] += 1
                self.pending['feedback'].append(dict(round=self.pending['round'],step=step,client_id=client,
                    tail_samples=len(batch['tail_positions']),non_tail_samples=len(batch['non_tail_positions']),
                    client_weight=1/len(self.recipients),local_la=float(loss.detach()),
                    tail_loss_weight=.35 if len(parts)==2 else 1.,non_tail_loss_weight=.65 if len(parts)==2 else 0.))
            if not all(v.grad is not None and torch.isfinite(v.grad).all() for v in residual.values()):
                raise ValueError('Nonfinite/absent direct calibration gradient')
            optimizer.step()
            if not all(torch.isfinite(v).all() for v in residual.values()):
                raise ValueError('Nonfinite direct calibration candidate')
            before,scale,actual = self.project(residual,norms[step-1],start)
            trace = self.trace(residual,manifests,step)
            summary['optimizer_steps'] += 1
            summary['feedback_synchronizations'] += 1
            summary['extra_downlink_bytes'] += len(self.recipients)*parameter_bytes
            summary['extra_upload_bytes'] += len(self.recipients)*(parameter_bytes+4)
            self.pending['steps'].append(dict(round=self.pending['round'],step=step,la_before=total,
                objective_before=total,reference_effective_norm=norms[step-1],unprojected_effective_norm=before,
                projection_scale=scale,effective_transfer_norm_after=actual,
                fixed_pool_objective_after=trace['fixed_pool_objective'],profile='direct_norm_matched'))
        return {k:v.detach().cpu().clone() for k,v in residual.items()}

    def apply_shared(self, start, ordinary, rnd):
        started = time.perf_counter()
        folder = self.runtime.root/'b_transfer_rounds'/f'r{rnd:03d}'
        folder.mkdir(parents=True,exist_ok=True)
        reference = self.config['norm_reference']['events'][str(rnd)]
        manifests = {k:dict(donor_ids=[],batches=shared_calibration_batches(
            self.groups[k],self.tail,self.runtime.args.seed,rnd,k,'class-cyclic')) for k in self.recipients}
        if self.recipients != reference['feedback_clients'] or {str(k):v['batches'] for k,v in manifests.items()} != reference['batches']:
            raise ValueError('Direct control batches differ from paired AB; do not silently rematch')
        summary = dict(round=rnd,mode='shared',calibration_profile='direct_norm_matched',learning_rate=.3,
            recipients=len(self.recipients),unique_donors=0,donor_links=0,candidate_pairs=0,
            learned_c_parameters=0,learned_direct_parameters=0,optimizer_steps=0,client_backward_batches=0,
            feedback_synchronizations=0,algorithm_forward_images=0,algorithm_backward_images=0,
            diagnostic_forward_images=0,extra_downlink_bytes=0,extra_upload_bytes=0,seconds=0.)
        self.pending = dict(round=rnd,folder=folder,summary=summary,metrics=[],steps=[],receivers=[],feedback=[],loss_trace=[])
        summary['extra_downlink_bytes'] += len(self.recipients)*sum(ordinary[k].numel()*ordinary[k].element_size() for k in self.keys)
        with self.session():
            self.copy_parameters(ordinary)
            before = self.measure_shared('ordinary_global_B')
            residual = (self.calibrate(ordinary,start,manifests,reference['norms']) if reference['norms'] else
                        {k:torch.zeros_like(ordinary[k]) for k in self.keys})
            committed = dict(ordinary)
            committed.update({k:ordinary[k]+residual[k].to(ordinary[k]) for k in self.keys})
            self.copy_parameters(committed)
            after = self.measure_shared('transferred_global_B')
        for client in self.recipients:
            self.pending['receivers'].append(dict(round=rnd,client_id=client,
                **{'before_'+k:v for k,v in before[client].items()},**{'after_'+k:v for k,v in after[client].items()}))
        summary.update(effective_global_transfer_norm=self.effective_norm(residual,start),
            ordinary_B_effective_update_norm=self.effective_norm({k:ordinary[k]-start[k] for k in self.keys},start))
        write_json(folder/'calibration_manifest.json',dict(profile='direct_norm_matched',module_order=self.keys,
            feedback_clients=self.recipients,shared_donor_ids=[],recipients=manifests,reference_norms=reference['norms']))
        write_csv(folder/'c_loss_trace.csv',self.pending['loss_trace'])
        torch.save(dict(round=rnd,profile='direct_norm_matched',transfer_residual=residual,
            ordinary_lora=ordinary,committed_lora=committed),folder/'commit.pt')
        summary['seconds'] = time.perf_counter()-started
        self.write_pending()
        return committed
