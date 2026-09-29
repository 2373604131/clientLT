"""Tail response supervision with independent donor C on one shared model.

Teacher outputs stay detached. Only C is optimized; A and ordinary B are
fixed throughout an event. Local minibatches contribute to one global step.
"""
import math
import time

import numpy as np
import torch

from utils.b_directed_math import class_macro_loss, select_sources
from utils.b_response_math import (build_response_targets, decision_margins,
    donor_residual, image_view, pairwise_margins, response_losses)
from utils.cliplora_b_directed import DirectedDonorBTransfer
from utils.cliplora_b_transfer import differentiable_b_residual
from utils.cliplora_bridge_audit import write_csv, write_json


class ResponseDonorBTransfer(DirectedDonorBTransfer):
    @torch.no_grad()
    def collect_outputs(self, diagnostic=False):
        outputs = {}
        for client in self.recipients:
            positions = self.monitors[client]['tail_positions']
            all_logits, all_losses, labels = [[], []], [[], []], []
            for begin in range(0, len(positions), self.feedback_batch_size):
                images, y = self.batch(client, positions[begin:begin+self.feedback_batch_size])
                labels.append(y.detach().cpu())
                for view in (0, 1):
                    loss, logits, _ = self.forward(image_view(images, view), y, diagnostic)
                    if not torch.isfinite(loss).all() or not torch.isfinite(logits).all():
                        raise ValueError('Nonfinite response probe output')
                    all_logits[view].append(logits.detach().cpu())
                    all_losses[view].append(loss.detach().cpu())
            outputs[client] = dict(labels=torch.cat(labels), positions=list(positions),
                logits=torch.stack([torch.cat(v) for v in all_logits]),
                losses=torch.stack([torch.cat(v) for v in all_losses]))
        return outputs

    def class_margin_means(self, outputs):
        sums = {c: torch.zeros(2, dtype=torch.float64) for c in self.class_totals}
        counts = {c: 0 for c in self.class_totals}
        for row in outputs.values():
            margins = torch.stack([decision_margins(v, row['labels']) for v in row['logits']])
            for c in row['labels'].unique().tolist():
                mask = row['labels'] == c
                sums[c] += margins[:, mask].double().sum(1)
                counts[c] += int(mask.sum())
        if counts != self.class_totals:
            raise ValueError('Response feedback pool differs from global class counts')
        return {c: sums[c]/counts[c] for c in sums}

    @torch.no_grad()
    def screen_responses(self, ordinary_b, baseline):
        base_means = self.class_margin_means(baseline)
        gains, rows, teachers = {c: {} for c in self.class_totals}, [], {}
        summary = self.pending['summary']
        delta_bytes = sum(v.numel()*v.element_size() for v in ordinary_b.values())
        for donor in sorted(self.selected):
            if donor in self.target_clients:
                continue
            eligible = [c for c in self.class_totals if int(self.counts[donor, c]) == 0]
            if not eligible:
                continue
            proposal = {key: ordinary_b[key]+self.config['probe_step']*self.original_deltas[donor][key]
                        for key in self.keys}
            if not all(torch.isfinite(v).all() for v in proposal.values()):
                raise ValueError('Nonfinite response donor proposal')
            self.copy_parameters(proposal)
            teachers[donor] = self.collect_outputs()
            after = self.class_margin_means(teachers[donor])
            summary['extra_downlink_bytes'] += len(self.recipients)*delta_bytes
            # One class/count plus two float32 margin sums per eligible local class.
            summary['extra_upload_bytes'] += 16*sum(
                sum(c in self.probes[k] for c in eligible) for k in self.recipients)
            for c in eligible:
                per_view = after[c]-base_means[c]
                gain = float(per_view.min())
                gains[c][donor] = gain
                rows.append(dict(round=self.pending['round'], class_id=c, donor=donor,
                    donor_class_count=int(self.counts[donor, c]), score='min_view_class_mean_raw_margin_gain',
                    baseline_margin_view0=float(base_means[c][0]), baseline_margin_view1=float(base_means[c][1]),
                    gain_view0=float(per_view[0]), gain_view1=float(per_view[1]), gain=gain,
                    positive=gain > self.config['min_gain']))
        self.copy_parameters(ordinary_b)
        mixtures = select_sources(gains, self.counts, self.target_clients,
                                  self.config['donors_per_class'], self.config['min_gain'])
        for row in rows:
            row['source_weight'] = mixtures.get(row['class_id'], {}).get(row['donor'], 0.)
            row['selected'] = row['source_weight'] > 0
        summary['candidate_pairs'] = len(rows)
        return mixtures, rows, teachers

    def make_targets(self, baseline, teachers, mixtures):
        targets = {}
        for client, base in baseline.items():
            local_teachers = {j: values[client]['logits'] for j, values in teachers.items()}
            for values in teachers.values():
                if not torch.equal(values[client]['labels'], base['labels']):
                    raise ValueError('Donor teacher sample order changed')
            targets[client] = build_response_targets(base['logits'], local_teachers,
                base['labels'], mixtures, self.config['response_variant'])
        return targets

    def penalty(self, matrices):
        return sum(c.square().sum() for c in matrices.values())/(len(next(iter(matrices.values())))*len(self.keys))

    def calibrate_responses(self, donor_tensors, targets, baseline):
        matrices = {key: torch.nn.Parameter(torch.zeros(
            len(value), self.ranks[key], self.ranks[key], device=self.device, dtype=value.dtype))
            for key, value in donor_tensors.items()}
        optimizer = torch.optim.Adam(list(matrices.values()), lr=self.config['learning_rate'],
                                     betas=(.9, .999), eps=1e-8, weight_decay=0.)
        summary = self.pending['summary']
        parameter_bytes = sum(c.numel()*c.element_size() for c in matrices.values())
        summary['learned_c_parameters'] = sum(c.numel() for c in matrices.values())
        strength = self.config['response_weight']
        for step in range(1, self.config['steps']+1):
            optimizer.zero_grad(set_to_none=True)
            la_total, response_total = 0., 0.
            for client in self.recipients:
                positions = self.monitors[client]['tail_positions']
                local_la, local_response = 0., 0.
                for begin in range(0, len(positions), self.feedback_batch_size):
                    end = begin+self.feedback_batch_size
                    images, y = self.batch(client, positions[begin:end])
                    if not torch.equal(y.cpu(), baseline[client]['labels'][begin:end]):
                        raise ValueError('Calibration/teacher label order differs')
                    for view in (0, 1):
                        target = targets[client]['target_margins'][view, begin:end].to(self.device)
                        weights = targets[client]['competitor_weights'][view, begin:end].to(self.device)
                        residual = donor_residual(donor_tensors, matrices)
                        with differentiable_b_residual(self.modules, residual):
                            losses, logits, _ = self.forward(image_view(images, view), y)
                            la = class_macro_loss(losses, y, self.class_totals)/2
                            response = class_macro_loss(response_losses(logits, y, target, weights),
                                                        y, self.class_totals)/2
                            objective = la+strength*response
                            if not torch.isfinite(objective):
                                raise ValueError('Nonfinite response-C objective')
                            objective.backward()
                        local_la += float(la.detach())
                        local_response += float(response.detach())
                        summary['algorithm_backward_images'] += len(y)
                        summary['client_backward_batches'] += 1
                la_total += local_la
                response_total += local_response
                self.pending['feedback'].append(dict(round=self.pending['round'], step=step,
                    client_id=client, shared_c_version=step-1, tail_samples=len(positions), views=2,
                    non_tail_samples=0, weighted_tail_la=local_la, weighted_response_loss=local_response,
                    response_weight=strength, response_variant=self.config['response_variant']))
            regularizer = self.penalty(matrices)
            (self.config['regularization']*regularizer).backward()
            if not all(c.grad is not None and torch.isfinite(c.grad).all() for c in matrices.values()):
                raise ValueError('Nonfinite or absent independent-donor C gradient')
            penalty = self.config['regularization']*float(regularizer.detach())
            gradient_norm = math.sqrt(sum(float(c.grad.square().sum()) for c in matrices.values()))
            self.pending['loss_trace'].append(dict(round=self.pending['round'], c_step=step-1,
                fixed_pool_la=la_total, response_loss=response_total, response_weight=strength,
                regularization_penalty=penalty, fixed_pool_objective=la_total+strength*response_total+penalty))
            optimizer.step()
            if not all(torch.isfinite(c).all() for c in matrices.values()):
                raise ValueError('Nonfinite independent-donor C')
            summary['optimizer_steps'] += 1
            summary['feedback_synchronizations'] += 1
            summary['extra_downlink_bytes'] += len(self.recipients)*parameter_bytes
            summary['extra_upload_bytes'] += len(self.recipients)*(parameter_bytes+8)
            self.pending['steps'].append(dict(round=self.pending['round'], step=step,
                la_before=la_total, response_before=response_total, regularization_penalty_before=penalty,
                objective_before=la_total+strength*response_total+penalty, c_gradient_norm=gradient_norm,
                c_norm_after=math.sqrt(sum(float(c.detach().square().sum()) for c in matrices.values())),
                donor_count=len(next(iter(matrices.values()))), response_variant=self.config['response_variant']))
            print(f'B-response r{self.pending["round"]:03d} C step={step}/2 '
                  f'tail_LA={la_total:.6f} response={response_total:.6g}', flush=True)
        return {key: value.detach() for key, value in matrices.items()}

    def response_metrics(self, outputs, targets, phase):
        totals = {c: dict(samples=0, la=0., response_loss=0., positive_pairs=0,
                         attained_positive_pairs=0, margin_gain_sum=0.) for c in self.class_totals}
        for client, row in outputs.items():
            target = targets[client]
            positive = target['positive_increment'] > 0
            margins = torch.stack([pairwise_margins(v, row['labels']) for v in row['logits']])
            losses = torch.stack([response_losses(row['logits'][v], row['labels'],
                target['target_margins'][v], target['competitor_weights'][v]) for v in (0, 1)])
            gain = (margins-target['baseline_margins']).min(0).values
            attained = positive & (gain >= target['positive_increment']-1e-6)
            for c in row['labels'].unique().tolist():
                mask = row['labels'] == c
                values = totals[c]
                values['samples'] += int(mask.sum())
                values['la'] += float(row['losses'][:, mask].double().mean(0).sum())
                values['response_loss'] += float(losses[:, mask].double().mean(0).sum())
                values['positive_pairs'] += int(positive[mask].sum())
                values['attained_positive_pairs'] += int(attained[mask].sum())
                values['margin_gain_sum'] += float(gain[mask][positive[mask]].double().sum())
        result = []
        for c, values in totals.items():
            n, pairs = values['samples'], values['positive_pairs']
            result.append(dict(round=self.pending['round'], phase=phase, class_id=c, samples=n,
                la=values['la']/n, response_loss=values['response_loss']/n, positive_pairs=pairs,
                attained_positive_pairs=values['attained_positive_pairs'],
                positive_pair_attainment=values['attained_positive_pairs']/pairs if pairs else None,
                measured_positive_margin_gain=values['margin_gain_sum']/pairs if pairs else None,
                response_variant=self.config['response_variant']))
        return result

    def apply_shared(self, start, ordinary, rnd):
        started = time.perf_counter()
        folder = self.runtime.root/'b_transfer_rounds'/f'r{rnd:03d}'
        summary = dict(round=rnd, mode='shared', calibration_profile='tail_response',
            response_variant=self.config['response_variant'], response_weight=self.config['response_weight'],
            learning_rate=self.config['learning_rate'], aggregation=self.config['aggregation'],
            recipients=len(self.recipients), target_classes=len(self.class_totals), non_tail_samples=0,
            unique_donors=0, donor_links=0, candidate_pairs=0, supported_classes=0,
            optimizer_steps=0, client_backward_batches=0, feedback_synchronizations=0,
            learned_c_parameters=0, algorithm_forward_images=0, algorithm_backward_images=0,
            diagnostic_forward_images=0, extra_downlink_bytes=0, extra_upload_bytes=0, seconds=0.)
        self.pending = dict(round=rnd, folder=folder, summary=summary, metrics=[], steps=[],
                            receivers=[], feedback=[], loss_trace=[])
        ordinary_b = {key: ordinary[key] for key in self.keys}
        if len(self.selected) != len(set(self.selected)) or any(j not in self.original_deltas for j in self.selected):
            raise ValueError('Invalid cached response donor updates')
        # Ordinary B is broadcast once; raw donor deltas sent during screening
        # are assumed cached by receivers until this event ends.
        summary['extra_downlink_bytes'] += len(self.recipients)*sum(v.numel()*v.element_size() for v in ordinary_b.values())
        with self.session():
            self.copy_parameters(ordinary)
            before = self.measure_shared('ordinary_global_B')
            baseline = self.collect_outputs()
            mixtures, donor_rows, teachers = self.screen_responses(ordinary_b, baseline)
            union = sorted({j for values in mixtures.values() for j in values})
            targets = self.make_targets(baseline, teachers, mixtures)
            positive_pairs = sum(int((v['positive_increment'] > 0).sum()) for v in targets.values())
            donor_tensors = {key: torch.stack([self.original_deltas[j][key].detach() for j in union]).to(self.device)
                             for key in self.keys} if union else {}
            matrices = {key: torch.zeros(len(union), self.ranks[key], self.ranks[key],
                device=self.device, dtype=value.dtype) for key, value in donor_tensors.items()}
            # Both variants use the measured response for eligibility; zero only
            # removes it from supervision, never silently changes the C budget.
            if positive_pairs:
                matrices = self.calibrate_responses(donor_tensors, targets, baseline)
            residual = ({key: v.detach().cpu() for key, v in donor_residual(donor_tensors, matrices).items()}
                        if union else {key: torch.zeros_like(v).cpu() for key, v in ordinary_b.items()})
            if not all(torch.isfinite(v).all() for v in residual.values()):
                raise ValueError('Nonfinite response-C residual')
            committed = dict(ordinary)
            committed.update({key: ordinary_b[key]+residual[key].to(ordinary_b[key]) for key in self.keys})
            self.copy_parameters(committed)
            after = self.measure_shared('transferred_global_B')
            final_outputs = self.collect_outputs(diagnostic=True)
        metrics = self.response_metrics(baseline, targets, 'ordinary_global_B')
        final_metrics = self.response_metrics(final_outputs, targets, 'transferred_global_B')
        final_la = sum(r['la'] for r in final_metrics)/len(final_metrics)
        final_response = sum(r['response_loss'] for r in final_metrics)/len(final_metrics)
        penalty = self.config['regularization']*float(self.penalty(matrices)) if matrices else 0.
        self.pending['loss_trace'].append(dict(round=rnd, c_step=summary['optimizer_steps'],
            fixed_pool_la=final_la, response_loss=final_response, response_weight=self.config['response_weight'],
            regularization_penalty=penalty,
            fixed_pool_objective=final_la+self.config['response_weight']*final_response+penalty))
        summary.update(unique_donors=len(union), donor_links=sum(map(len, mixtures.values())),
            supported_classes=len(mixtures), unsupported_classes=len(self.class_totals)-len(mixtures),
            positive_response_pairs=positive_pairs, skipped_no_positive_response=not bool(positive_pairs),
            final_response_loss=final_response,
            effective_global_transfer_norm=self.effective_norm(residual, start),
            ordinary_B_effective_update_norm=self.effective_norm({key:ordinary[key]-start[key] for key in self.keys}, start))
        units = sum(len(v) for v in self.probes.values())
        summary['extra_upload_bytes'] += units*16  # baseline class/count/two-view margin sums
        summary['extra_downlink_bytes'] += units*8  # global class counts
        summary['extra_downlink_bytes'] += sum(len(v)*12*len(self.recipients) for v in mixtures.values())
        if summary['optimizer_steps']:
            # Final C is needed for the post-transfer local measurement.
            summary['extra_downlink_bytes'] += len(self.recipients)*sum(v.numel()*v.element_size() for v in matrices.values())
        if rnd == self.config['rounds'][0]:
            summary['extra_downlink_bytes'] += self.counts.numel()
        for client in self.recipients:
            self.pending['receivers'].append(dict(round=rnd, client_id=client,
                measurement_scope='shared_B_before_after', shared_donors=len(union),
                **{'before_'+k:v for k,v in before[client].items()},
                **{'after_'+k:v for k,v in after[client].items()}))
        self.last_mixtures, self.last_targets = mixtures, targets
        self.last_matrices = {key:v.detach().cpu() for key,v in matrices.items()}
        self.last_donors, self.last_residual = union, residual
        write_csv(folder/'donor_scores.csv', donor_rows)
        write_csv(folder/'source_weights.csv', [dict(round=rnd, class_id=c, donor=j, weight=w,
            donor_class_count=int(self.counts[j,c])) for c, values in mixtures.items() for j,w in values.items()])
        write_csv(folder/'response_metrics.csv', metrics+final_metrics)
        write_csv(folder/'c_loss_trace.csv', self.pending['loss_trace'])
        write_json(folder/'calibration_manifest.json', dict(mode='shared', profile='tail_response',
            target_clients=self.target_clients, target_class_totals=self.class_totals,
            feedback_clients=self.recipients, source_mixtures=mixtures, shared_donor_ids=union,
            c_axis='donor IDs; each donor independently trainable', response_variant=self.config['response_variant'],
            union_role='C parameter basis; class source relations separately define fixed response supervision',
            response_views=self.config['response_views'], module_order=self.keys,
            recipients={k:dict(tail_positions=self.monitors[k]['tail_positions'], non_tail_positions=[])
                        for k in self.recipients}))
        classes = sorted(self.class_totals)
        weights = np.asarray([[mixtures.get(c, {}).get(j, 0.) for j in union] for c in classes],
                             dtype=np.float64).reshape(len(classes), len(union))
        np.savez_compressed(folder/'matrices.npz', class_ids=np.asarray(classes, dtype=np.int64),
            donor_ids=np.asarray(union, dtype=np.int64), source_weights=weights,
            **{f'module_{i:02d}':self.last_matrices[key].numpy() for i,key in enumerate(self.keys) if key in matrices})
        archive = {}
        for client, values in baseline.items():
            prefix = f'client_{client:03d}_'
            archive[prefix+'labels'] = values['labels'].numpy()
            archive[prefix+'positions'] = np.asarray(values['positions'], dtype=np.int64)
            archive[prefix+'baseline_logits'] = values['logits'].numpy()
            archive[prefix+'student_logits'] = final_outputs[client]['logits'].numpy()
            archive.update({prefix+name:value.numpy() for name,value in targets[client].items()})
            for donor in union:
                archive[prefix+f'donor_{donor:03d}_logits'] = teachers[donor][client]['logits'].numpy()
        np.savez_compressed(folder/'response_targets.npz', **archive)
        torch.save(dict(round=rnd, profile='tail_response', source_mixtures=mixtures, donor_ids=union,
            donor_updates={key:value.detach().cpu() for key,value in donor_tensors.items()},
            donor_matrices=self.last_matrices, ordinary_lora={k:ordinary[k] for k in self.runtime.keys},
            committed_lora={k:committed[k] for k in self.runtime.keys}, transfer_residual=residual), folder/'commit.pt')
        summary['seconds'] = time.perf_counter()-started
        self.write_pending()
        self.original_deltas = None
        return committed
