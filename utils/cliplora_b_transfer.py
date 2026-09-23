"""Periodic missing-class donor transfer in the common-A B phase.

Only C is trained. Original donor deltas never change during a transfer round.
All measurements use training images; diagnostic scores never gate a commit.
"""
from collections import defaultdict
from contextlib import contextmanager
import math
import time

import numpy as np
import torch
from torch.nn import functional as F

from utils.cliplora_a_refresh import isolated_rng, product_norm_squared, train_only
from utils.cliplora_bridge_audit import write_csv, write_json
from utils.b_aggregation import B_TRANSFER_ROUNDS


def transfer_config(args):
    return dict(schema_version='donor_b_v1', rounds=list(B_TRANSFER_ROUNDS),
                warmup=20, interval=10, probe_step=args.b_transfer_probe_step,
                learning_rate=args.b_transfer_lr, regularization=args.b_transfer_reg,
                optimizer='Adam', betas=[.9, .999], adam_eps=1e-8, weight_decay=0.,
                steps=2, initialization='zero_each_event', rank='read_from_each_B_tensor',
                probe_per_tail_class=8, probe_views=1, donor_threshold=0.,
                donor_rule='source_missing_class_and_positive_LA_forward_gain',
                calibration_tail_cap=16, calibration_non_tail_cap=16, tail_only_cap=32,
                calibration_objective='equal mean of tail and non-tail LA means; existing groups only',
                calibration_sampling='class-cyclic tail coverage over two steps; no repeats within a step',
                calibration_transform='deterministic evaluation transform; autograd enabled',
                calibration_probe_overlap='allowed training-side proxy',
                aggregation=('uniform_whole_B_at_transfer_rounds'
                             if getattr(args, 'sfra_b_aggregation', 'sample') == 'uniform-transfer-rounds'
                             else 'unchanged_sample_weighted_B_FedAvg'),
                privacy='sequential FL simulation; class presence and donor identities visible')


def calibration_batches(groups, tail_ids, seed, round_id, client):
    """Two reproducible batches, independent of donor screening and learning rate."""
    rng = np.random.default_rng(np.random.SeedSequence([seed, round_id, client, 314159]))
    labels = sorted(set(groups) & set(tail_ids))
    order = list(map(int, rng.permutation(labels)))
    non_tail = [p for c, positions in groups.items() if c not in tail_ids for p in positions]
    total_tail = sum(len(groups[c]) for c in labels)
    cap = min(16 if non_tail else 32, total_tail)
    batches, cursor = [], 0
    for _ in range(2):
        bags = {c: list(map(int, rng.permutation(groups[c]))) for c in labels}
        tail = []
        while len(tail) < cap:
            label = order[cursor % len(order)]
            cursor += 1
            if bags[label]:
                tail.append(bags[label].pop())
        other = list(map(int, rng.choice(non_tail, min(16, len(non_tail)), replace=False)))
        batches.append(dict(tail_positions=tail, non_tail_positions=other))
    return batches


def reconstruct_residual(deltas, donor_ids, matrices, keys):
    """Server reconstruction from cached ORIGINAL updates and returned small C."""
    return {key: torch.bmm(torch.stack([deltas[j][key] for j in donor_ids]), matrices[key]).mean(0)
            for key in keys}


@contextmanager
def differentiable_b_residual(modules, residuals):
    """Add s R A x without overwriting Parameters or detaching the graph to C."""
    handles = []
    try:
        for key, module in modules.items():
            def add_residual(layer, inputs, output, residual=residuals[key]):
                latent = F.linear(inputs[0], layer.w_lora_A)
                return output + layer.scaling * F.linear(latent, residual)
            handles.append(module.register_forward_hook(add_residual))
        yield
    finally:
        for handle in handles:
            handle.remove()


class DonorBTransfer:
    def __init__(self, runtime):
        self.runtime, self.bank = runtime, runtime.bank
        self.config = runtime.sfra_config['b_transfer']
        self.model, self.core = self.bank.model, self.bank.core
        self.device, self.keys = self.bank.device, runtime.b_keys
        self.parameters = dict(self.model.named_parameters())
        modules = dict(self.model.named_modules())
        self.modules = {key: modules[key.rsplit('.', 1)[0]] for key in self.keys}
        self.ranks = {key: self.parameters[key].shape[1] for key in self.keys}
        self.tail = set(map(int, runtime.tail))
        self.counts = runtime.audit.counts.cpu()
        self.adjustment = runtime.trainer.training_logit_adjustment.detach().to(self.device)
        self.groups = []
        for source in runtime.trainer.dm.dataset.federated_train_x:
            groups = defaultdict(list)
            for position, item in enumerate(source):
                groups[int(item.label)].append(position)
            self.groups.append(dict(groups))
        self.probes = defaultdict(dict)
        for token in self.bank.tokens:
            if token['class_id'] in self.tail:
                self.probes[token['client_id']][token['class_id']] = token['local_positions']
        self.recipients = sorted(self.probes)
        self.monitors = {}
        for k in self.recipients:
            rng = np.random.default_rng(np.random.SeedSequence([runtime.args.seed, k, 271828]))
            other = [p for c, positions in self.groups[k].items() if c not in self.tail for p in positions]
            self.monitors[k] = dict(
                tail_positions=[p for positions in self.probes[k].values() for p in positions],
                non_tail_positions=list(map(int, rng.choice(other, min(16, len(other)), replace=False))))
        self.records, self.pending = [], None
        write_json(runtime.root/'b_transfer_config.json', {**self.config, 'module_ranks':self.ranks})
        write_json(runtime.root/'b_transfer_manifest.json', dict(
            tail_ids=sorted(self.tail), recipients=self.recipients,
            class_presence=(self.counts > 0).tolist(), probes=dict(self.probes), monitors=self.monitors))

    def scheduled(self, rnd):
        return rnd in self.config['rounds']

    @contextmanager
    def session(self):
        flags = [(p, p.requires_grad) for p in self.model.parameters()]
        modes = [(m, m.training) for m in self.model.modules()]
        saved = {k: self.parameters[k].detach().clone() for k in self.runtime.keys}
        with isolated_rng():
            try:
                train_only(self.model, '')
                self.model.eval()
                yield
            finally:
                self.copy_parameters(saved)
                for p, flag in flags:
                    p.requires_grad_(flag)
                for module, mode in modes:
                    module.training = mode

    def copy_parameters(self, state):
        with torch.no_grad():
            for key in self.runtime.keys:
                if key in state:
                    self.parameters[key].copy_(state[key])

    def batch(self, client, positions):
        items = [self.bank.datasets[client][p] for p in positions]
        images = torch.stack([item['img'] for item in items]).to(self.device)
        labels = torch.tensor([int(item['label']) for item in items], device=self.device)
        return images, labels

    def forward(self, images, labels, diagnostic=False):
        features = self.core.image_encoder(images.type(self.core.dtype))
        features = features / features.norm(dim=-1, keepdim=True)
        cosine = features @ self.bank.text_features.T
        logits = (self.core.logit_scale.exp().detach() * features) @ self.bank.text_features.T
        losses = F.cross_entropy(logits + self.adjustment, labels, reduction='none')
        field = 'diagnostic_forward_images' if diagnostic else 'algorithm_forward_images'
        self.pending['summary'][field] += len(labels)
        return losses, logits, cosine

    @torch.no_grad()
    def score_positions(self, client, positions, diagnostic):
        results = defaultdict(lambda: dict(count=0, loss=0., correct=0., margin=0.))
        for start in range(0, len(positions), self.bank.batch_size):
            images, labels = self.batch(client, positions[start:start+self.bank.batch_size])
            losses, logits, cosine = self.forward(images, labels, diagnostic)
            wrong = cosine.clone()
            wrong.scatter_(1, labels[:, None], -torch.inf)
            margins = cosine.gather(1, labels[:, None]).flatten() - wrong.max(1).values
            correct = logits.argmax(1) == labels  # Same raw-logit rule as official inference.
            for c in labels.unique().tolist():
                mask = labels == c
                row = results[c]
                row['count'] += int(mask.sum())
                row['loss'] += float(losses[mask].sum())
                row['correct'] += float(correct[mask].sum())
                row['margin'] += float(margins[mask].sum())
        return {c: dict(samples=r['count'], la=r['loss']/r['count'],
                        accuracy=100*r['correct']/r['count'], margin=r['margin']/r['count'])
                for c, r in results.items()}

    def measure_client(self, client, phase):
        monitor = self.monitors[client]
        scores = self.score_positions(client, monitor['tail_positions']+monitor['non_tail_positions'], True)
        for c, values in sorted(scores.items()):
            self.pending['metrics'].append(dict(round=self.pending['round'], phase=phase,
                client_id=client, class_id=c, is_tail=c in self.tail, **values))
        result = {}
        for name, classes in [('tail', [c for c in scores if c in self.tail]),
                              ('non_tail', [c for c in scores if c not in self.tail])]:
            for metric in ('la', 'accuracy', 'margin'):
                result[name+'_'+metric] = (float(np.mean([scores[c][metric] for c in classes]))
                                           if classes else None)
        return result

    def measure_all(self, phase):
        rows = [self.measure_client(k, phase) for k in self.recipients]
        for key in ('tail_la', 'non_tail_la', 'tail_accuracy', 'non_tail_accuracy'):
            values = [r[key] for r in rows if r[key] is not None]
            self.pending['summary'][phase+'_'+key] = float(np.mean(values)) if values else None

    def effective_norm(self, residual, start):
        return math.sqrt(sum(self.modules[key].scaling**2 * product_norm_squared(
            value.detach().double().cpu(), start[key[:-1]+'A'].double().cpu())
            for key, value in residual.items()))

    def calibrate(self, client, donor_ids, original_deltas, batches, start):
        m, layers = len(donor_ids), len(self.keys)
        donor_tensors = {key: torch.stack([original_deltas[j][key] for j in donor_ids]).to(self.device)
                         for key in self.keys}
        matrices = {key: torch.nn.Parameter(torch.zeros(m, self.ranks[key], self.ranks[key], device=self.device))
                    for key in self.keys}
        optimizer = torch.optim.Adam(list(matrices.values()), lr=self.config['learning_rate'],
                                     betas=(.9, .999), eps=1e-8, weight_decay=0.)
        for step, sampled in enumerate(batches, 1):
            images, labels = self.batch(client, sampled['tail_positions']+sampled['non_tail_positions'])
            count_tail = len(sampled['tail_positions'])
            optimizer.zero_grad(set_to_none=True)
            residual = {key: torch.bmm(donor_tensors[key], matrices[key]).mean(0) for key in self.keys}
            with differentiable_b_residual(self.modules, residual):
                losses, _, _ = self.forward(images, labels)
                parts = [losses[:count_tail].mean()]
                if count_tail < len(labels):
                    parts.append(losses[count_tail:].mean())
                calibration_loss = torch.stack(parts).mean()
                regularizer = sum(c.square().sum() for c in matrices.values()) / (m*layers)
                loss = calibration_loss + self.config['regularization']*regularizer
                loss.backward()
            grad_norm = math.sqrt(sum(float(c.grad.square().sum()) for c in matrices.values()))
            optimizer.step()
            self.pending['summary']['algorithm_backward_images'] += len(labels)
            self.pending['summary']['optimizer_steps'] += 1
            with torch.no_grad():
                updated = {key: torch.bmm(donor_tensors[key], matrices[key]).mean(0) for key in self.keys}
                norm = self.effective_norm(updated, start)
            self.pending['steps'].append(dict(round=self.pending['round'], client_id=client, step=step,
                donors=m, tail_samples=count_tail, non_tail_samples=len(labels)-count_tail,
                la_before=float(calibration_loss.detach()), regularizer_before=float(regularizer.detach()),
                objective_before=float(loss.detach()), c_gradient_norm=grad_norm,
                c_norm_after=math.sqrt(sum(float(c.detach().square().sum()) for c in matrices.values())),
                effective_transfer_norm_after=norm))
            print(f'B-transfer r{self.pending["round"]:03d} client={client} step={step}/2 '
                  f'donors={m} LA={float(calibration_loss.detach()):.6f} |sRA|={norm:.6g}', flush=True)
        return {key: c.detach().cpu().clone() for key, c in matrices.items()}

    def apply(self, start, local_states, deltas, selected, rnd):
        if not self.scheduled(rnd):
            return local_states, deltas
        started = time.perf_counter()
        folder = self.runtime.root/'b_transfer_rounds'/f'r{rnd:03d}'
        folder.mkdir(parents=True, exist_ok=True)
        summary = dict(round=rnd, learning_rate=self.config['learning_rate'], recipients=len(self.recipients),
            receivers_with_donors=0, donor_links=0, candidate_pairs=0, optimizer_steps=0,
            algorithm_forward_images=0, algorithm_backward_images=0, diagnostic_forward_images=0,
            extra_downlink_bytes=0, extra_upload_bytes=0, seconds=0.)
        self.pending = dict(round=rnd, folder=folder, summary=summary, metrics=[], steps=[], receivers=[])
        candidate_rows, manifests, matrices_archive = [], {}, {}
        weights = self.runtime.phase_aggregation_weights(selected, rnd, 'B')
        summary['aggregation'] = self.config['aggregation']
        ordinary_b = {}
        for key in self.keys:
            ordinary_b[key] = torch.zeros_like(start[key])
            for k in selected:
                ordinary_b[key].add_(local_states[k][key], alpha=weights[k])
        new_states = dict(local_states)  # Original local states and deltas remain immutable.
        with self.session():
            self.copy_parameters(start)
            self.copy_parameters(ordinary_b)
            self.measure_all('ordinary_global_B')
            for client in self.recipients:
                self.copy_parameters(local_states[client])
                before = self.measure_client(client, 'local_before')
                positions = [p for values in self.probes[client].values() for p in values]
                baseline = self.score_positions(client, positions, False)
                candidates = [j for j in selected if j != client and
                              any(self.counts[j, c] == 0 for c in self.probes[client])]
                donors = []
                for j in candidates:
                    proposal = {key: local_states[client][key]+self.config['probe_step']*deltas[j][key]
                                for key in self.keys}
                    self.copy_parameters(proposal)  # Always local B + one ORIGINAL delta.
                    measured = self.score_positions(client, positions, False)
                    accepted = False
                    for c in sorted(self.probes[client]):
                        if self.counts[j, c] != 0:
                            continue
                        gain = baseline[c]['la'] - measured[c]['la']
                        accepted |= gain > 0.
                        candidate_rows.append(dict(round=rnd, receiver=client, donor=j, class_id=c,
                            baseline_la=baseline[c]['la'], injected_la=measured[c]['la'],
                            gain=gain, positive=gain > 0.))
                    if accepted:
                        donors.append(j)
                donors.sort()
                self.copy_parameters(local_states[client])
                batches = calibration_batches(self.groups[client], self.tail, self.runtime.args.seed, rnd, client)
                manifests[client] = dict(donor_ids=donors, batches=batches)
                summary['candidate_pairs'] += len(candidates)
                summary['extra_downlink_bytes'] += len(candidates)*sum(deltas[client][k].numel()*4 for k in self.keys)
                if rnd == self.config['rounds'][0]:
                    summary['extra_downlink_bytes'] += self.counts.numel()  # Cached uint8 class-presence table.
                if donors:
                    returned_c = self.calibrate(client, donors, deltas, batches, start)
                    # Simulated server receives only IDs and C, reconstructs from its raw cache.
                    residual = reconstruct_residual(deltas, donors, returned_c, self.keys)
                    new_states[client] = {key: local_states[client][key]+residual[key] for key in self.keys}
                    self.copy_parameters(new_states[client])
                    summary['receivers_with_donors'] += 1
                    summary['donor_links'] += len(donors)
                    summary['extra_upload_bytes'] += 4*len(donors)+sum(c.numel()*4 for c in returned_c.values())
                    for i, key in enumerate(self.keys):
                        matrices_archive[f'client_{client:03d}_module_{i:02d}'] = returned_c[key].numpy()
                    effective_norm = self.effective_norm(residual, start)
                else:
                    effective_norm = 0.
                after = self.measure_client(client, 'local_after')
                self.pending['receivers'].append(dict(round=rnd, client_id=client, candidates=len(candidates),
                    donors=len(donors), sample_weight=self.runtime.q[client], aggregation_weight=weights[client],
                    effective_transfer_norm=effective_norm,
                    **{'before_'+k:v for k,v in before.items()}, **{'after_'+k:v for k,v in after.items()}))
                print(f'B-transfer r{rnd:03d} client={client}: {len(donors)}/{len(candidates)} donors, '
                      f'tail LA {before["tail_la"]:.6f} -> {after["tail_la"]:.6f}', flush=True)
        global_residual = {key: torch.zeros_like(start[key]) for key in self.keys}
        for client in selected:
            for key in self.keys:
                global_residual[key].add_(new_states[client][key]-local_states[client][key], alpha=weights[client])
        summary['effective_global_transfer_norm'] = self.effective_norm(global_residual, start)
        summary['seconds'] += time.perf_counter()-started
        write_csv(folder/'donor_scores.csv', candidate_rows)
        write_json(folder/'calibration_manifest.json', dict(module_order=self.keys, recipients=manifests))
        np.savez_compressed(folder/'matrices.npz', **matrices_archive)
        self.write_pending()
        updated_deltas = {k: {key: value-start[key] for key, value in local.items()}
                          for k, local in new_states.items()}
        return new_states, updated_deltas

    def observe_global(self, state, rnd, committed=False):
        if not self.scheduled(rnd):
            return
        started = time.perf_counter()
        phase = 'committed_global' if committed else 'transferred_global_B'
        with self.session():
            self.copy_parameters(state)
            self.measure_all(phase)
        summary = self.pending['summary']
        summary['seconds'] += time.perf_counter()-started
        if committed:
            summary['A_updated_this_round'] = rnd <= 90
            summary['tail_la_gain_after_B_aggregation'] = (summary['ordinary_global_B_tail_la']
                                                          - summary['transferred_global_B_tail_la'])
            summary['tail_la_change_during_A'] = (summary['committed_global_tail_la']
                                                - summary['transferred_global_B_tail_la'])
            summary['local_tail_la_gain_mean'] = float(np.mean([
                r['before_tail_la']-r['after_tail_la'] for r in self.pending['receivers']]))
            self.records.append(dict(summary))
        self.write_pending()
        if committed:
            self.pending = None

    def write_pending(self):
        folder = self.pending['folder']
        write_csv(folder/'probe_metrics.csv', self.pending['metrics'])
        write_csv(folder/'optimization_steps.csv', self.pending['steps'])
        write_csv(folder/'receiver_summary.csv', self.pending['receivers'])
        write_json(folder/'summary.json', self.pending['summary'])

    def flush(self):
        write_csv(self.runtime.root/'b_transfer_rounds.csv', self.records)

    def state_dict(self):
        return dict(round_summaries=self.records)

    def load_state_dict(self, state):
        self.records = state['round_summaries']

    def progress(self):
        return dict(b_transfer_events=len(self.records),
                    b_transfer_optimizer_steps=sum(r['optimizer_steps'] for r in self.records),
                    b_transfer_learning_rate=self.config['learning_rate'])
