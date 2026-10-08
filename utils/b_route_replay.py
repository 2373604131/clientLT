"""Same-state route diagnostics using saved raw ordinary B events."""
import copy
import math
from pathlib import Path
import time

import torch
import numpy as np

from tools.sfra.b_routes import branch_path, file_hash, replay_jobs
from tools.sfra.maintext import PAIR_HASHES, digest, load_json
from tools.sfra.summary import read_csv
from utils.b_route_sampling import image_identity, make_folds
from utils.cliplora_a_refresh import state_hash
from utils.b_route_math import EffectiveCoordinates, norm
from utils.cliplora_b_routes import RouteTransfer, group_loss, route_config
from utils.cliplora_b_shared_transfer import SharedDonorBTransfer, shared_transfer_config
from utils.cliplora_b_transfer import differentiable_b_residual, reconstruct_residual
from utils.cliplora_bridge_audit import write_csv, write_json
from utils.cliplora_sfra import SFRARuntime


def branch_name(fold, arm, rho, null_index=0):
    return branch_path(fold, arm, rho, null_index)


def legacy_optimize(transfer, start, ordinary, manifests, diagnostics):
    """Old Adam/C objective, with the common existing-group rule on sparse folds."""
    donors = {k: torch.stack([transfer.original_deltas[j][k] for j in transfer.selected]).to(transfer.device)
              for k in transfer.keys}
    m = len(transfer.selected)
    matrices = {k: torch.nn.Parameter(torch.zeros(m, d.shape[2], d.shape[2], device=d.device)) for k, d in donors.items()}
    optimizer = torch.optim.Adam(list(matrices.values()), lr=transfer.config['learning_rate'],
                                 betas=(.9, .999), eps=1e-8, weight_decay=0.)
    residual = lambda: {k: torch.bmm(donors[k], v).mean(0) for k, v in matrices.items()}
    penalty = lambda: transfer.config['regularization']*sum(v.square().sum() for v in matrices.values())/(m*len(matrices))
    sets = dict(fit=transfer.fit_positions(manifests), diagnostic=diagnostics)
    baselines = {name: transfer.evaluate(residual(), ids, name, 0) for name, ids in sets.items()}
    transfer.trace_objective(residual(), manifests, 0)
    transfer.objective_rows[-1]['regularization_penalty'] = float(penalty().detach())
    summary = transfer.pending['summary']
    summary.update(learning_rate=transfer.config['learning_rate'], optimizer='Adam_C',
                   ordinary_B_effective_update_norm=transfer.effective_norm({k: ordinary[k]-start[k] for k in transfer.keys}, start))
    cbytes = sum(v.numel()*v.element_size() for v in matrices.values())
    summary['extra_downlink_bytes'] += len(transfer.recipients)*(sum(ordinary[k].numel()*4 for k in transfer.runtime.keys)+
                                                               sum(v.numel()*4 for v in donors.values()))
    for step in (1, 2):
        optimizer.zero_grad(set_to_none=True)
        active = [(client, item['batches'][step-1]) for client, item in manifests.items()
                  if item['batches'][step-1]['tail_positions']+item['batches'][step-1]['non_tail_positions']]
        if not active:
            raise ValueError('No active legacy replay clients')
        mean_loss = 0.
        for client, batch in active:
            images, labels = transfer.batch(client, batch['tail_positions']+batch['non_tail_positions'])
            with differentiable_b_residual(transfer.modules, residual()):
                losses, _, _ = transfer.forward(images, labels)
                loss = group_loss(losses, len(batch['tail_positions']), transfer.config['tail_weight'])
                (loss/len(active)).backward()
            mean_loss += float(loss.detach())/len(active)
            summary['algorithm_backward_images'] += len(labels)
            summary['client_backward_batches'] += 1
            transfer.pending['feedback'].append(dict(step=step, client_id=client, local_la=float(loss.detach()),
                tail_samples=len(batch['tail_positions']), non_tail_samples=len(batch['non_tail_positions']), client_weight=1/len(active)))
        regularizer = penalty()
        regularizer.backward()
        optimizer.step()
        summary['optimizer_steps'] += 1
        summary['feedback_synchronizations'] += 1
        summary['extra_downlink_bytes'] += len(active)*cbytes
        summary['extra_upload_bytes'] += len(active)*(cbytes+4)
        length = transfer.effective_norm(residual(), start)
        transfer.pending['steps'].append(dict(step=step, arm='O', la_before=mean_loss,
            objective_before=mean_loss+float(regularizer.detach()), effective_transfer_norm_after=length))
        transfer.trace_objective(residual(), manifests, step)
        transfer.objective_rows[-1]['regularization_penalty'] = float(penalty().detach())
    result = {k: v.detach().cpu() for k, v in residual().items()}
    transfer.saved_residuals = {2: result}
    for name, ids in sets.items():
        transfer.evaluate(result, ids, name, 2, baselines[name])
    summary['effective_global_transfer_norm'] = length
    torch.save(dict(step=2, residual=result), transfer.pending['folder']/'residual_step2.pt')
    transfer.write_route_logs()
    return {k: v.detach().cpu() for k, v in matrices.items()}


def completed_branch(folder, identity):
    path = folder/'branch_completion.json'
    if not path.is_file():
        return False
    result = load_json(path)
    if result.get('identity') != identity:
        raise ValueError('Completed branch belongs to a different route contract')
    for name, expected in result['artifacts'].items():
        if file_hash(folder/name) != expected:
            raise ValueError('Completed replay artifact changed: '+str(folder/name))
    return True


def finish_branch(folder, identity, summary):
    artifacts = {p.name: file_hash(p) for p in folder.iterdir()
                 if p.is_file() and p.name != 'branch_completion.json'}
    write_json(folder/'branch_completion.json', dict(identity=identity, artifacts=artifacts, costs=summary))


def replay_runtime_class(spec):
    class ReplayRuntime(SFRARuntime):
        def configure_experiment(self):
            source_cfg = load_json(Path(spec['source_run'])/'sfra_config.json')
            # The dedicated worker has the source training flags. New routes
            # register metadata through a hook, so reproduce only that hook here.
            if 'b_route_experiment' in source_cfg:
                self.sfra_config['b_route_experiment'] = source_cfg['b_route_experiment']
                if 'b_transfer' in source_cfg:
                    self.sfra_config['b_transfer'] = source_cfg['b_transfer']

        def __init__(self, *args, **kwargs):
            # Skip writing another full CLIP base copy for a read-only event.
            worker_args = args[3] if len(args) > 3 else kwargs['args']
            marker = worker_args.method_a_diagnostic_manifest
            worker_args.method_a_diagnostic_manifest = 'route_replay_skip_base_copy'
            try:
                super().__init__(*args, **kwargs)
            finally:
                worker_args.method_a_diagnostic_manifest = marker
            self.source = Path(spec['source_run'])
            self.event_root = self.root
            if self.args.sfra_resume or self.sfra_config != load_json(self.source/'sfra_config.json'):
                raise ValueError('Replay source/runtime configuration differs')
            self.source_meta = load_json(self.source/'bridge_metadata.json')
            if any(self.audit.meta[k] != self.source_meta[k] for k in PAIR_HASHES):
                raise ValueError('Replay model/data/initialization identity differs')
            if self.bank.tokens != load_json(self.source/'private_witness_manifest.json'):
                raise ValueError('Replay witness identities differ')
            if self.execution_config != load_json(self.source/'execution_config.json'):
                raise ValueError('Replay execution mode differs')

        def load_event(self):
            rnd = spec['round']
            base = torch.load(self.source/'checkpoints/base_model.pt', map_location='cpu', weights_only=False)
            if (state_hash(base, self.keys) != self.source_meta['initial_lora_sha256'] or
                state_hash(base, sorted(set(base)-set(self.keys))) != self.source_meta['frozen_model_sha256']):
                raise ValueError('Saved base model hash differs')
            self.trainer.model.load_state_dict(base, strict=True)
            if self.global_trainer.model is not self.trainer.model:
                self.global_trainer.model.load_state_dict(base, strict=True)
            folder = self.source/f'events/r{rnd:03d}_c000_main_normal_B'
            event = torch.load(folder/'state.pt', map_location='cpu', weights_only=False)
            meta = load_json(folder/'event.json')
            if event['round'] != rnd or event['phase'] != 'normal_B':
                raise ValueError('Wrong source event')
            start, ordinary = event['anchor_lora_state'], event['actual_after_lora_state']
            for state, field in [(start, 'before_lora_sha256'), (ordinary, 'after_lora_sha256')]:
                if set(state) != set(self.keys) or state_hash(state, self.keys) != meta[field]:
                    raise ValueError('Source event hash/parameter order differs')
            selected = list(map(int, event['selected_client_ids']))
            if sorted(selected) != list(range(30)) or len(selected) != len(event['local_factor_deltas']):
                raise ValueError('Source donor list invalid')
            deltas = dict(zip(selected, event['local_factor_deltas']))
            if not torch.equal(event['client_class_counts'], self.audit.counts[selected]):
                raise ValueError('Source class counts differ')
            weights = torch.as_tensor(event['server_weights'], dtype=torch.float64)
            expected_weights = self.audit.counts[selected].sum(1).double()
            expected_weights /= expected_weights.sum()
            torch.testing.assert_close(weights, expected_weights, atol=1e-7, rtol=1e-6)
            for k in self.b_keys:
                expected = start[k]+sum(deltas[j][k]*float(w) for j, w in zip(selected, event['server_weights']))
                torch.testing.assert_close(ordinary[k], expected, atol=1e-5, rtol=1e-5)
            for k in self.a_keys:
                if not torch.equal(ordinary[k], start[k]):
                    raise ValueError('Ordinary B event changed A')
            return start, ordinary, deltas, sorted(selected)

        def new_transfer(self, folder, settings, legacy=False):
            self.root = folder
            self.root.mkdir(parents=True, exist_ok=True)
            base = shared_transfer_config(self.args)
            base.update(non_tail_sampling='class-cyclic', tail_weight=.35, steps=2,
                        calibration_profile='coverage_tradeoff', learning_rate=.3, regularization=.001)
            self.sfra_config['b_transfer'] = base if legacy else route_config(base, settings)
            if settings.get('arm') == 'O':
                self.sfra_config['b_transfer'] = dict(base, calibration_profile='legacy_sparse_replay')
            return SharedDonorBTransfer(self) if legacy else RouteTransfer(self)

        def legacy(self, start, ordinary, deltas, selected, manifest, folder, verify_saved=False):
            transfer = self.new_transfer(folder, {}, legacy=True)
            transfer.cache_updates(deltas, selected)
            donors = list(map(int, manifest['shared_donor_ids']))
            manifests = {int(k): v for k, v in manifest['recipients'].items()}
            if any(not b['tail_positions'] for v in manifests.values() for b in v['batches']):
                raise ValueError('Original legacy manifest has an empty tail group')
            if (not donors or sorted(donors) != donors or not set(donors) <= set(selected)
                    or sorted(manifests) != transfer.recipients):
                raise ValueError('Original legacy donor/recipient identity differs')
            transfer.pending = dict(round=spec['round'], folder=folder, metrics=[], steps=[], receivers=[], feedback=[],
                                    loss_trace=[], summary=dict(optimizer_steps=0, algorithm_forward_images=0,
                                    algorithm_backward_images=0, diagnostic_forward_images=0, client_backward_batches=0,
                                    feedback_synchronizations=0, extra_downlink_bytes=0, extra_upload_bytes=0))
            with transfer.session():
                transfer.copy_parameters(ordinary)
                matrices = transfer.calibrate_shared(donors, manifests, start)
            residual = reconstruct_residual(deltas, donors, matrices, self.b_keys)
            if verify_saved:
                saved = torch.load(self.source/f'b_transfer_rounds/r{spec["round"]:03d}/commit.pt',
                                   map_location='cpu', weights_only=False)
                for k in self.b_keys:
                    torch.testing.assert_close(ordinary[k], saved['ordinary_lora'][k])
                    torch.testing.assert_close(residual[k], saved['transfer_residual'][k], atol=1e-6, rtol=1e-4)
                    torch.testing.assert_close(ordinary[k]+residual[k], saved['committed_lora'][k], atol=1e-6, rtol=1e-4)
                source_folder = self.source/f'b_transfer_rounds/r{spec["round"]:03d}'
                with np.load(source_folder/'matrices.npz', allow_pickle=False) as packed:
                    if packed['donor_ids'].tolist() != donors or set(manifest['module_order']) != set(self.b_keys):
                        raise ValueError('Legacy saved matrix ordering differs')
                    for i, key in enumerate(manifest['module_order']):
                        torch.testing.assert_close(matrices[key], torch.from_numpy(packed[f'module_{i:02d}'].copy()), atol=1e-5, rtol=1e-4)
                measured = {}
                with transfer.session():
                    for phase, state in [('ordinary_global_B', ordinary), ('transferred_global_B', {
                            **ordinary, **{k: ordinary[k]+residual[k] for k in self.b_keys}})]:
                        transfer.copy_parameters(state)
                        for client in transfer.recipients:
                            monitor = transfer.monitors[client]
                            rows = transfer.score_positions(client, monitor['tail_positions']+monitor['non_tail_positions'], True)
                            measured.update({(phase, client, c): r for c, r in rows.items()})
                checked = set()
                for row in read_csv(source_folder/'probe_metrics.csv'):
                    key = row['phase'], int(row['client_id']), int(row['class_id'])
                    if key not in measured:
                        continue
                    observed = measured[key]
                    if int(row['samples']) != observed['samples'] or not math.isclose(float(row['la']), observed['la'], abs_tol=1e-4, rel_tol=1e-4):
                        raise ValueError('Legacy forward-output equivalence failed')
                    checked.add(key)
                if checked != set(measured):
                    raise ValueError('Legacy saved forward observations are incomplete')
                write_json(folder/'equivalence.json', dict(passed=True, residual_atol=1e-6, residual_rtol=1e-4,
                                                          forward_atol=1e-4, forward_rtol=1e-4))
            transfer.write_pending()
            torch.save(residual, folder/'legacy_residual.pt')

        def probe(self, transfer, start, ordinary, deltas, selected, probes, folder):
            rows, baseline = [], {}
            with transfer.session():
                transfer.copy_parameters(ordinary)
                for split, clients in probes.items():
                    baseline[split] = {k: transfer.score_positions(k, [p for ps in groups.values() for p in ps], True)
                                       for k, groups in clients.items()}
                ordinary_norm = transfer.effective_norm({k: ordinary[k]-start[k] for k in self.b_keys}, start)
                for donor in selected:
                    length = transfer.effective_norm(deltas[donor], start)
                    for label, scale in [('raw_0.1', .1),
                                         ('matched_effective_norm', .1*ordinary_norm/length if length else 0.)]:
                        transfer.copy_parameters({k: ordinary[k]+scale*deltas[donor][k] for k in self.b_keys})
                        for split, clients in probes.items():
                            for client, groups in clients.items():
                                scores = transfer.score_positions(client, [p for ps in groups.values() for p in ps], True)
                                for c, row in scores.items():
                                    rows.append(dict(round=spec['round'], split=split, client_id=client,
                                        class_id=c, donor=donor, source_missing_class=bool(self.audit.counts[donor, c] == 0),
                                        source_is_receiver=donor == client, scale_kind=label, scale=scale,
                                        effective_norm=scale*length, samples=row['samples'],
                                        la_gain=baseline[split][client][c]['la']-row['la']))
            write_csv(folder/'source_probe.csv', rows)
            write_json(folder/'probe_costs.json', transfer.pending['summary'])

        def run(self, initial_state):
            started = time.perf_counter()
            original_cfg = copy.deepcopy(self.sfra_config)
            start, ordinary, deltas, selected = self.load_event()
            identities = image_identity(read_csv(self.source/'partition_manifest.csv'))
            base = self.event_root
            try:
                if spec['include_legacy']:
                    source_manifest = load_json(self.source/f'b_transfer_rounds/r{spec["round"]:03d}/calibration_manifest.json')
                    self.legacy(start, ordinary, deltas, selected, source_manifest, base/'legacy_equivalence', True)
                template = self.new_transfer(base/'sampling', dict(arm='F', rho=1.))
                template.cache_updates(deltas, selected)
                for fold in (0, 1):
                    fit, diagnostic, probes, sample_manifest = make_folds(template.groups, identities,
                        template.recipients, template.tail, self.args.seed, spec['round'], fold)
                    manifests = template.manifests(spec['round'], fit)
                    sample_manifest.update(recipients=manifests, diagnostic_positions=diagnostic)
                    for client, item in manifests.items():
                        item['raw_sample_ids'] = [[identities[client, p] for p in b['tail_positions']+b['non_tail_positions']]
                                                  for b in item['batches']]
                    fold_root = base/f'fold{fold}'
                    write_json(fold_root/'sample_manifest.json', sample_manifest)
                    # JSON stringifies integer client keys; hash the serialized
                    # form so clients 2/10 have the same canonical order on resume.
                    sample_manifest = load_json(fold_root/'sample_manifest.json')
                    if spec['source_probe']:
                        probe_transfer = self.new_transfer(fold_root/'probe', dict(arm='F', rho=1.))
                        probe_transfer.begin(fold_root/'probe', spec['round'], 'diagnostic_probe')
                        self.probe(probe_transfer, start, ordinary, deltas, selected, probes, fold_root/'probe')
                    for arm, rho, null_index in replay_jobs(spec):
                        folder = base/branch_name(fold, arm, rho, null_index)
                        identity = dict(spec_digest=digest(spec), samples=digest(sample_manifest),
                                        arm=arm, rho=rho, null_index=null_index)
                        if completed_branch(folder, identity):
                            print('SKIP completed route branch', folder, flush=True)
                            continue
                        settings = dict(arm=arm, rho=rho, null_index=null_index)
                        transfer = self.new_transfer(folder, settings)
                        transfer.cache_updates(deltas, selected)
                        transfer.begin(folder, spec['round'], arm)
                        branch_started = time.perf_counter()
                        with transfer.session():
                            transfer.copy_parameters(ordinary)
                            if arm == 'O':
                                legacy_optimize(transfer, start, ordinary, manifests, diagnostic)
                            else:
                                transfer.optimize(ordinary, start, manifests, arm, rho, steps=8,
                                                  diagnostics=diagnostic, null_index=null_index)
                            # Forward-only curves never choose a committed step/scale.
                            for step, residual in transfer.saved_residuals.items():
                                if step:
                                    for beta in (0., .25, .5):
                                        transfer.evaluate(residual, diagnostic, 'diagnostic_amplitude', step, beta=beta)
                        transfer.pending['summary']['seconds'] = time.perf_counter()-branch_started
                        transfer.write_route_logs()
                        finish_branch(folder, identity, transfer.pending['summary'])
                        print(f'Route replay r{spec["round"]} fold{fold} {arm} rho={rho:g} complete', flush=True)
                    directions = []
                    coordinates = EffectiveCoordinates({k: start[k[:-1]+'A'] for k in self.b_keys},
                                                        {k: template.modules[k].scaling for k in self.b_keys})
                    for arm, rho, null_index in replay_jobs(spec):
                        if arm in ('N', 'O'):
                            continue
                        for step in (2, 8):
                            folder = base/branch_name(fold, arm, rho, null_index)
                            free_path = base/branch_name(fold, 'F', rho)/f'residual_step{step}.pt'
                            if free_path.is_file():
                                x = coordinates.encode(torch.load(folder/f'residual_step{step}.pt', weights_only=False)['residual'])
                                f = coordinates.encode(torch.load(free_path, weights_only=False)['residual'])
                                nx, nf = norm(x.values()), norm(f.values())
                                cosine = sum(float((x[k].double()*f[k].double()).sum()) for k in x)/(nx*nf) if nx and nf else None
                                directions.append(dict(arm=arm, rho=rho, null_index=null_index, step=step, cosine_to_F=cosine))
                    write_csv(fold_root/'directions.csv', directions)
                auxiliary = [f'fold{f}/{n}' for f in (0, 1) for n in ('sample_manifest.json', 'directions.csv')]
                if spec['source_probe']:
                    auxiliary += [f'fold{f}/probe/{n}' for f in (0, 1) for n in ('source_probe.csv', 'probe_costs.json')]
                if spec['include_legacy']:
                    auxiliary += ['legacy_equivalence/equivalence.json']
                write_json(base/'replay_completion.json', dict(spec_digest=digest(spec), complete=True,
                    seconds=time.perf_counter()-started, artifacts={n: file_hash(base/n) for n in auxiliary}))
            finally:
                self.root, self.sfra_config = base, original_cfg

    return ReplayRuntime
