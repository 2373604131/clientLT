"""Common effective-coordinate optimizer; legacy B implementations are untouched."""
import math
import time

import torch

from utils.b_route_math import EffectiveCoordinates, RouteProjector, norm, projected_step, rotated_sources
from utils.cliplora_b_shared_transfer import SharedDonorBTransfer, shared_calibration_batches
from utils.cliplora_b_transfer import differentiable_b_residual
from utils.cliplora_bridge_audit import write_csv, write_json


def route_config(base, settings):
    value = dict(base)
    value.update(schema_version='b_routes_transfer_v1', calibration_profile='routes',
                 route_arm=settings['arm'], route_rho=settings.get('rho', 1.),
                 route_null_index=settings.get('null_index', 0), regularization=0.,
                 optimizer='normalized_effective_gradient_then_projection', learning_rate=0.,
                 calibration_update='synchronous gradients in X; normalized step and exact space projection',
                 c_scope='arm-specific feasible effective update space; C recovered only for audits',
                 donor_pool='all ordinary participating clients; no screening',
                 donor_rule='none', steps=2, non_tail_sampling='class-cyclic', tail_weight=.35,
                 norm_rule='rho times current ordinary effective B update',
                 calibration_objective='equal recipient group-balanced LA; no C penalty',
                 commit_rule='fixed second projected step',
                 missing_non_tail_rule='mean of whichever groups exist; skip empty clients',
                 probe_usage='diagnostic only; never screens donors or controls the update',
                 communication='dense FP32 effective-residual gradients; sources stay at server')
    for unused in ('betas', 'adam_eps', 'weight_decay'):
        value.pop(unused, None)
    return value


def group_loss(losses, count_tail, weight=.35):
    if not len(losses):
        raise ValueError('Empty feedback batch')
    if count_tail == 0:
        return losses.mean()
    if count_tail == len(losses):
        return losses.mean()
    return weight*losses[:count_tail].mean()+(1-weight)*losses[count_tail:].mean()


class RouteTransfer(SharedDonorBTransfer):
    def cache_updates(self, deltas, selected):
        super().cache_updates(deltas, sorted(selected))

    def begin(self, folder, rnd, arm):
        folder.mkdir(parents=True, exist_ok=True)
        summary = dict(round=rnd, mode='shared', route_arm=arm, recipients=len(self.recipients),
                       optimizer_steps=0, client_backward_batches=0, feedback_synchronizations=0,
                       algorithm_forward_images=0, algorithm_backward_images=0, diagnostic_forward_images=0,
                       extra_downlink_bytes=0, extra_upload_bytes=0, seconds=0., learning_rate=0.,
                       unique_donors=0, learned_c_parameters=0, diagnostic_seconds=0.,
                       primary_two_steps_optimization_seconds=0., diagnostic_extra_steps_optimization_seconds=0.)
        self.pending = dict(round=rnd, folder=folder, summary=summary, metrics=[], steps=[],
                            receivers=[], feedback=[], loss_trace=[])
        self.sample_rows, self.group_rows, self.geometry_rows, self.coefficient_rows = [], [], [], []
        self.saved_residuals = {}
        self.objective_rows = []
        if self.device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(self.device)
            summary['initial_allocated_bytes'] = torch.cuda.memory_allocated(self.device)

    def measure_shared(self, phase):
        started = time.perf_counter()
        result = super().measure_shared(phase)
        self.pending['summary']['diagnostic_seconds'] += time.perf_counter()-started
        return result

    def manifests(self, rnd, groups=None):
        groups = self.groups if groups is None else groups
        return {k: dict(donor_ids=list(getattr(self, 'selected', [])), batches=shared_calibration_batches(
            groups[k], self.tail, self.runtime.args.seed, rnd, k, 'class-cyclic')) for k in self.recipients}

    @staticmethod
    def fit_positions(manifests):
        return {k: sorted({p for b in v['batches'] for p in b['tail_positions']+b['non_tail_positions']})
                for k, v in manifests.items()}

    @torch.no_grad()
    def evaluate(self, residual, positions, split, step, baseline=None, beta=1.):
        started = time.perf_counter()
        classes, sample_map = {}, {}
        with differentiable_b_residual(self.modules, {k: beta*v.to(self.parameters[k]) for k, v in residual.items()}):
            for client, ids in positions.items():
                for offset in range(0, len(ids), self.bank.batch_size):
                    chosen = ids[offset:offset+self.bank.batch_size]
                    x, y = self.batch(client, chosen)
                    losses, logits, _ = self.forward(x, y, diagnostic=True)
                    if not torch.isfinite(losses).all() or not torch.isfinite(logits).all():
                        raise ValueError('Nonfinite route diagnostic')
                    wrong = logits.clone()
                    wrong.scatter_(1, y[:, None], -torch.inf)
                    margins = logits.gather(1, y[:, None]).flatten()-wrong.max(1).values
                    correct = logits.argmax(1) == y
                    for i, p in enumerate(chosen):
                        c, ok = int(y[i]), bool(correct[i])
                        identity = (client, p)
                        previous = baseline.get(identity) if baseline is not None else None
                        row = dict(round=self.pending['round'], step=step, split=split, beta=beta,
                                   client_id=client, local_position=p, class_id=c, is_tail=c in self.tail,
                                   la=float(losses[i]), accuracy=100.*ok, margin=float(margins[i]),
                                   wrong_to_correct=int(previous is False and ok),
                                   correct_to_wrong=int(previous is True and not ok),
                                   unchanged_correct=int(previous is True and ok),
                                   unchanged_wrong=int(previous is False and not ok))
                        self.sample_rows.append(row)
                        classes.setdefault(c, []).append(row)
                        sample_map[identity] = ok
        for name, ids in [('tail', [c for c in classes if c in self.tail]),
                          ('non_tail', [c for c in classes if c not in self.tail])]:
            row = dict(round=self.pending['round'], step=step, split=split, beta=beta, group=name,
                       covered_classes=len(ids), samples=sum(len(classes[c]) for c in ids))
            for metric in ('la', 'accuracy', 'margin'):
                row[metric] = (sum(sum(x[metric] for x in classes[c])/len(classes[c]) for c in ids)/len(ids)
                               if ids else None)
            for metric in ('wrong_to_correct', 'correct_to_wrong', 'unchanged_correct', 'unchanged_wrong'):
                row[metric] = sum(x[metric] for c in ids for x in classes[c])
            self.group_rows.append(row)
        self.pending['summary']['diagnostic_seconds'] += time.perf_counter()-started
        return sample_map

    @torch.no_grad()
    def trace_objective(self, residual, manifests, step):
        started = time.perf_counter()
        values = []
        with differentiable_b_residual(self.modules, {k: v.to(self.parameters[k]) for k, v in residual.items()}):
            for index in range(2):
                active = []
                for client, item in manifests.items():
                    batch = item['batches'][index]
                    positions = batch['tail_positions']+batch['non_tail_positions']
                    if positions:
                        images, labels = self.batch(client, positions)
                        losses, _, _ = self.forward(images, labels, diagnostic=True)
                        active.append(float(group_loss(losses, len(batch['tail_positions']))))
                values.append(sum(active)/len(active) if active else None)
        self.objective_rows.append(dict(step=step, batch1_la=values[0], batch2_la=values[1],
            fixed_pool_la=sum(v for v in values if v is not None)/sum(v is not None for v in values)
                if any(v is not None for v in values) else None, regularization_penalty=0.))
        self.pending['summary']['diagnostic_seconds'] += time.perf_counter()-started

    def optimize(self, ordinary, start, manifests, arm, rho, steps=2, diagnostics=None, null_index=0):
        """Base parameters and original donors stay fixed; only X receives gradients."""
        if steps not in (2, 8) or not math.isfinite(rho) or rho <= 0:
            raise ValueError('Routes require 2/8 steps and positive finite rho')
        optimizer_started = time.perf_counter()
        diagnostic_before = self.pending['summary']['diagnostic_seconds']
        a = {k: start[k[:-1]+'A'] for k in self.keys}
        coordinates = EffectiveCoordinates(a, {k: self.modules[k].scaling for k in self.keys})
        encoded_ordinary = coordinates.encode({k: ordinary[k]-start[k] for k in self.keys})
        nu, radius = norm(encoded_ordinary.values()), rho*norm(encoded_ordinary.values())
        zero_r = {k: torch.zeros_like(ordinary[k], device=self.device) for k in self.keys}
        x = {k: torch.nn.Parameter(v) for k, v in coordinates.encode(zero_r).items()}
        donors = {} if arm in ('N', 'F') else {
            k: torch.stack([self.original_deltas[j][k] for j in self.selected]).double().cpu() for k in self.keys}
        rotation_checks = []
        if arm.endswith('-rot'):
            donors, rotation_checks = rotated_sources(donors, self.runtime.args.seed,
                                                     self.pending['round'], null_index)
        projector = RouteProjector(arm, coordinates, donors, {k: v.shape for k, v in x.items()})
        summary = self.pending['summary']
        summary.update(ordinary_B_effective_update_norm=nu, radius=radius, rho=rho,
                       null_index=null_index, unique_donors=len(donors.get(self.keys[0], [])),
                       effective_space_dimension=projector.dimension,
                       learned_direct_parameters=sum(v.numel() for v in x.values()),
                       gradient_payload='dense effective coordinates, not small C')
        self.geometry_rows.extend(dict(kind='A_coordinates', **r) for r in coordinates.info)
        self.geometry_rows.extend(dict(kind='source_rotation', **r) for r in rotation_checks)
        self.geometry_rows.extend(dict(kind='C_source_rank', module=k, source_column_rank=q.shape[1],
            effective_input_rank=x[k].shape[1]) for k, q in projector.source_bases.items())
        eval_sets = dict(fit=self.fit_positions(manifests))
        if diagnostics is not None:
            eval_sets['diagnostic'] = diagnostics
        summary['optimizer_setup_seconds'] = time.perf_counter()-optimizer_started
        baselines = {name: self.evaluate(zero_r, ids, name, 0) for name, ids in eval_sets.items()}
        self.trace_objective(zero_r, manifests, 0)
        self.saved_residuals[0] = {k: v.cpu() for k, v in zero_r.items()}
        parameter_bytes = sum(v.numel()*v.element_size() for v in x.values())
        if arm != 'N':
            summary['extra_downlink_bytes'] += len(self.recipients)*sum(ordinary[k].numel()*4 for k in self.runtime.keys)
        if arm == 'N':
            steps = 0
        for step in range(1, steps+1):
            step_started = time.perf_counter()
            for v in x.values():
                v.grad = None
            batch_index = (step-1) % 2
            active = [(k, v['batches'][batch_index]) for k, v in manifests.items()
                      if v['batches'][batch_index]['tail_positions']+v['batches'][batch_index]['non_tail_positions']]
            if not active:
                raise ValueError('No fit data in this fold/batch')
            before = 0.
            for client, batch in active:
                images, labels = self.batch(client, batch['tail_positions']+batch['non_tail_positions'])
                with differentiable_b_residual(self.modules, coordinates.decode(x)):
                    losses, _, _ = self.forward(images, labels)
                    loss = group_loss(losses, len(batch['tail_positions']))
                    if not torch.isfinite(loss):
                        raise ValueError('Nonfinite route classification loss')
                    (loss/len(active)).backward()
                before += float(loss.detach())/len(active)
                summary['algorithm_backward_images'] += len(labels)
                summary['client_backward_batches'] += 1
                self.pending['feedback'].append(dict(round=self.pending['round'], step=step,
                    client_id=client, tail_samples=len(batch['tail_positions']),
                    non_tail_samples=len(batch['non_tail_positions']), client_weight=1/len(active),
                    local_la=float(loss.detach())))
            gradients = {k: (v.grad if v.grad is not None else torch.zeros_like(v)) for k, v in x.items()}
            if step == 1:
                projected = projector.project({k: -g for k, g in gradients.items()})
                gnorm = norm(gradients.values())
                self.geometry_rows.append(dict(kind='first_gradient', space_dimension=projector.dimension,
                    capture_fraction=(norm(projected.values())/gnorm)**2 if gnorm > 1e-12 else None))
            updated, grad_norm = projected_step(x, gradients, projector, radius)
            with torch.no_grad():
                for k, v in x.items():
                    v.copy_(updated[k])
            summary['optimizer_steps'] += 1
            summary['feedback_synchronizations'] += 1
            summary['extra_downlink_bytes'] += len(active)*parameter_bytes
            summary['extra_upload_bytes'] += len(active)*(parameter_bytes+4)
            actual = norm(x.values())
            step_seconds = time.perf_counter()-step_started
            time_field = ('primary_two_steps' if step <= 2 else 'diagnostic_extra_steps')+'_optimization_seconds'
            summary[time_field] += step_seconds
            active_coefficients = list(projector.last_coefficients.values())
            active_sources = int(torch.stack([v.abs()>1e-10 for v in active_coefficients]).any(0).sum()) if active_coefficients else None
            self.pending['steps'].append(dict(round=self.pending['round'], step=step, arm=arm,
                la_before=before, gradient_norm=grad_norm, radius=radius,
                effective_transfer_norm_after=actual, radius_usage=actual/radius if radius else 0.,
                transfer_to_ordinary_ratio=actual/nu if nu else None,
                nnls_kkt_error=projector.kkt_error, gradient_zero=grad_norm <= 1e-12,
                optimization_seconds=step_seconds,
                active_sources=active_sources,
                active_source_module_pairs=sum(int((v.abs()>1e-10).sum()) for v in active_coefficients) if active_coefficients else None,
                cost_scope='primary_two_steps' if step <= 2 else 'diagnostic_extra_steps'))
            for module, values in projector.last_coefficients.items():
                self.coefficient_rows.extend(dict(step=step, module=module, donor=j, coefficient=float(value))
                                             for j, value in zip(self.selected, values))
            if step in (2, 8):
                residual = {k: v.detach().cpu() for k, v in coordinates.decode(x).items()}
                self.saved_residuals[step] = residual
                for name, ids in eval_sets.items():
                    self.evaluate(residual, ids, name, step, baselines[name])
                self.trace_objective(residual, manifests, step)
                torch.save(dict(step=step, residual=residual), self.pending['folder']/f'residual_step{step}.pt')
                for field in ('algorithm_forward_images', 'algorithm_backward_images', 'extra_downlink_bytes',
                              'extra_upload_bytes', 'client_backward_batches', 'feedback_synchronizations'):
                    if step == 2:
                        summary['primary_two_steps_'+field] = summary[field]
                    else:
                        summary['diagnostic_extra_steps_'+field] = summary[field]-summary['primary_two_steps_'+field]
                if arm.removesuffix('-rot') == 'C':
                    for simpler in ('P', 'S'):
                        other = RouteProjector(simpler, coordinates, donors, {k: v.shape for k, v in x.items()})
                        projection = other.project(x)
                        distance = norm((x[k]-projection[k] for k in x))
                        self.geometry_rows.append(dict(kind='distance_to_'+simpler, step=step,
                            residual_distance=distance, residual_energy=distance**2,
                            relative_distance=distance/norm(x.values()) if norm(x.values()) else None))
            self.write_route_logs()
        result = {k: v.detach().cpu() for k, v in coordinates.decode(x).items()}
        if arm.removesuffix('-rot') == 'C':
            recovered = projector.reconstruct_c(result)
            torch.save(recovered, self.pending['folder']/'reconstructed_c.pt')
            self.geometry_rows.extend(dict(kind='C_reconstruction', **r) for r in projector.reconstruction_errors)
        summary['effective_global_transfer_norm'] = norm(x.values())
        summary['optimizer_wall_seconds_excluding_forwards_for_diagnostics'] = (
            time.perf_counter()-optimizer_started-(summary['diagnostic_seconds']-diagnostic_before))
        self.write_route_logs()
        return result

    def write_route_logs(self):
        folder = self.pending['folder']
        for filename, rows in [('sample_metrics', self.sample_rows), ('group_metrics', self.group_rows),
                               ('source_geometry', self.geometry_rows), ('coefficients', self.coefficient_rows),
                               ('objective_trace', self.objective_rows)]:
            write_csv(folder/(filename+'.csv'), rows)
        if self.device.type == 'cuda':
            self.pending['summary']['peak_allocated_bytes'] = torch.cuda.max_memory_allocated(self.device)
        self.write_pending()

    def apply_shared(self, start, ordinary, rnd):
        started = time.perf_counter()
        arm = self.config['route_arm']
        self.begin(self.runtime.root/'b_transfer_rounds'/f'r{rnd:03d}', rnd, arm)
        manifests = self.manifests(rnd)
        with self.session():
            self.copy_parameters(ordinary)
            before = self.measure_shared('ordinary_global_B')
            residual = self.optimize(ordinary, start, manifests, arm, self.config['route_rho'],
                                     null_index=self.config['route_null_index'])
            committed = dict(ordinary)
            committed.update({k: ordinary[k]+residual[k].to(ordinary[k]) for k in self.keys})
            self.copy_parameters(committed)
            after = self.measure_shared('transferred_global_B')
        for client in self.recipients:
            self.pending['receivers'].append(dict(round=rnd, client_id=client,
                **{'before_'+k: v for k, v in before[client].items()},
                **{'after_'+k: v for k, v in after[client].items()}))
        write_json(self.pending['folder']/'calibration_manifest.json', dict(
            profile='routes', arm=arm, module_order=self.keys, feedback_clients=self.recipients,
            shared_donor_ids=[] if arm == 'F' else self.selected, recipients=manifests))
        torch.save(dict(round=rnd, ordinary_lora={k: ordinary[k] for k in self.runtime.keys},
                        committed_lora={k: committed[k] for k in self.runtime.keys},
                        transfer_residual=residual), self.pending['folder']/'commit.pt')
        self.pending['summary']['seconds'] = time.perf_counter()-started
        self.write_route_logs()
        self.original_deltas = None
        return committed
