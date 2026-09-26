"""Read-only source-event replay; no local training, A update, or test-set selection."""
import copy
from pathlib import Path
import time

import numpy as np
import torch

from tools.sfra.b_problem2 import SCHEMA, VARIANTS, read_json, read_csv, write_json, write_csv
from utils.b_problem2_math import problem2_config, summarize_changes
from utils.cliplora_a_refresh import state_hash, isolated_rng
from utils.cliplora_b_problem2 import Problem2DonorBTransfer
from utils.cliplora_b_transfer import reconstruct_residual
from utils.cliplora_sfra import SFRARuntime


def without_donor(residual, deltas, matrices, donor_ids, donor, keys):
    """Hold C, other terms and original denominator fixed."""
    if donor not in donor_ids:
        return {key:value.clone() for key, value in residual.items()}
    index, size = donor_ids.index(donor), len(donor_ids)
    return {key:residual[key] - deltas[donor][key] @ matrices[key][index] / size for key in keys}


def changed_rows(before, after, tail):
    if before.keys() != after.keys():
        raise ValueError('Counterfactual monitor identities differ')
    return [dict(client_id=k, class_id=c, group='tail' if c in tail else 'non_tail',
                 gain=b['la']-after[k,c]['la'], samples=b['samples'])
            for (k,c), b in before.items()]


def measure_residual(transfer, ordinary, residual):
    result = {}
    with transfer.session():
        transfer.copy_parameters({**ordinary, **{key:ordinary[key]+residual[key] for key in transfer.keys}})
        for k in transfer.recipients:
            monitor = transfer.monitors[k]
            rows = transfer.score_positions(k, monitor['tail_positions']+monitor['non_tail_positions'], True)
            result.update({(k,c):values for c,values in rows.items()})
    return result


class Problem2ReplayRuntime(SFRARuntime):
    def __init__(self, trainer, global_trainer, cfg, args, schedule, normal_train):
        self.job = read_json(args.b_problem2_replay_manifest)
        if (self.job['schema_version'] != SCHEMA or self.job['round'] not in self.job['rounds']
                or not self.job['variants'] or any(v not in VARIANTS for v in self.job['variants'])
                or 'E00' not in self.job['variants']):
            raise ValueError('Invalid B replay protocol')
        if args.sfra_resume or args.b_problem2_variant != 'off' or args.method_a_diagnostic_manifest:
            raise ValueError('Replay is an isolated source job, not a training resume')
        super().__init__(trainer, global_trainer, cfg, args, schedule, normal_train)
        self.source = Path(self.job['source'])
        if self.root.resolve() == self.source.resolve() or self.source.resolve() in self.root.resolve().parents:
            raise ValueError('Replay cannot write inside its source run')
        if self.sfra_config != read_json(self.source/'sfra_config.json'):
            raise ValueError('Reconstructed training configuration differs from source')
        source_meta = read_json(self.source/'bridge_metadata.json')
        for key in ('pool_sha256','test_sha256','schedule_sha256','frozen_model_sha256',
                    'initial_lora_sha256','probe_images_sha256','probe_manifest_sha256'):
            if self.audit.meta[key] != source_meta[key]:
                raise ValueError(f'Replay identity mismatch: {key}')
        if self.bank.tokens != read_json(self.source/'private_witness_manifest.json'):
            raise ValueError('Replay witness identities differ')
        if read_json(self.root/'b_transfer_manifest.json') != read_json(self.source/'b_transfer_manifest.json'):
            raise ValueError('Replay probe/monitor identities differ')
        execution = read_json(self.source/'execution_config.json') if (self.source/'execution_config.json').exists() else None
        if self.execution_config != execution:
            raise ValueError('Replay execution mode differs')
        self.source_meta = source_meta

    def load_event(self):
        rnd = self.job['round']
        base = torch.load(self.source/'checkpoints/base_model.pt', map_location='cpu', weights_only=False)
        if state_hash(base, self.keys) != self.source_meta['initial_lora_sha256']:
            raise ValueError('Saved base LoRA identity differs')
        if state_hash(base, sorted(set(base)-set(self.keys))) != self.source_meta['frozen_model_sha256']:
            raise ValueError('Saved frozen model identity differs')
        self.base_state = base
        self.trainer.model.load_state_dict(base, strict=True)
        if self.global_trainer.model is not self.trainer.model:
            self.global_trainer.model.load_state_dict(base, strict=True)
        event = self.source/f'events/r{rnd:03d}_c000_main_normal_B'
        payload = torch.load(event/'state.pt', map_location='cpu', weights_only=False)
        meta = read_json(event/'event.json')
        if payload['round'] != rnd or payload['phase'] != 'normal_B':
            raise ValueError('Wrong replay B event')
        start, ordinary = payload['anchor_lora_state'], payload['actual_after_lora_state']
        for state, field in [(start,'before_lora_sha256'), (ordinary,'after_lora_sha256')]:
            if set(state) != set(self.keys) or state_hash(state,self.keys) != meta[field]:
                raise ValueError('Event LoRA state hash/order differs')
        selected = list(map(int,payload['selected_client_ids']))
        if len(selected) != len(set(selected)) or len(payload['local_factor_deltas']) != len(selected):
            raise ValueError('Invalid event donor list')
        deltas = dict(zip(selected,payload['local_factor_deltas']))
        if not torch.equal(payload['client_class_counts'], self.audit.counts[selected]):
            raise ValueError('Event donor class counts differ')
        for key in self.b_keys:
            expected = start[key] + sum(deltas[j][key]*float(w) for j,w in zip(selected,payload['server_weights']))
            torch.testing.assert_close(ordinary[key], expected, atol=1e-5, rtol=1e-5)
        for key in self.a_keys:
            if not torch.equal(start[key], ordinary[key]):
                raise ValueError('A changed during the ordinary B event')
        folder = self.source/f'b_transfer_rounds/r{rnd:03d}'
        saved = torch.load(folder/'commit.pt', map_location='cpu', weights_only=False)
        if state_hash(saved['ordinary_lora'],self.keys) != state_hash(ordinary,self.keys):
            raise ValueError('Transfer and ordinary event anchors differ')
        manifest = read_json(folder/'calibration_manifest.json')
        donors = list(map(int,manifest['shared_donor_ids']))
        if sorted(donors) != donors or len(set(donors)) != len(donors) or not set(donors) <= set(selected):
            raise ValueError('Saved donor pool is invalid')
        if donors != list(map(int,saved['donor_ids'])) or manifest['feedback_clients'] != self.b_transfer.recipients:
            raise ValueError('Saved donor/recipient identity differs')
        matrices = {}
        with np.load(folder/'matrices.npz', allow_pickle=False) as packed:
            if packed['donor_ids'].tolist() != donors or set(manifest['module_order']) != set(self.b_keys):
                raise ValueError('Matrix donor/module order differs')
            for i,key in enumerate(manifest['module_order']):
                matrices[key] = torch.from_numpy(packed[f'module_{i:02d}'].copy())
        if not donors:
            raise ValueError('Replay D0/D1 requires a nonempty recorded source pool')
        residual = reconstruct_residual(deltas,donors,matrices,self.b_keys)
        for key in self.b_keys:
            torch.testing.assert_close(residual[key],saved['transfer_residual'][key])
            torch.testing.assert_close(ordinary[key]+residual[key],saved['committed_lora'][key])
        return start, ordinary, deltas, selected, manifest, matrices, residual

    def run(self, initial_state):
        started = time.perf_counter()
        rnd, event_root = self.job['round'], self.root
        start, ordinary, deltas, selected, manifest, source_c, source_residual = self.load_event()
        source_config = dict(self.sfra_config['b_transfer'])
        results, profiles = {}, {}
        try:
            for variant in self.job['variants']:
                self.root = event_root/variant
                self.root.mkdir(parents=True, exist_ok=True)
                self.sfra_config['b_transfer'] = problem2_config(source_config,variant,self.job['harm_beta'])
                transfer = Problem2DonorBTransfer(self)
                transfer.replay_manifests = {int(k):dict(copy.deepcopy(v), donor_ids=sorted(selected))
                                             for k,v in manifest['recipients'].items()}
                transfer.cache_updates(deltas,selected)
                with isolated_rng():
                    committed = transfer.apply_shared(start,ordinary,rnd)
                results[variant] = {key:value.clone() for key,value in transfer.last_residual.items()}
                profiles[variant] = transfer
                write_json(self.root/'replay_variant.json', dict(variant=variant, source=str(self.source),
                    round=rnd, config=self.sfra_config['b_transfer'], scope='same-state; no local training or A update'))
        finally:
            self.root = event_root
            self.sfra_config['b_transfer'] = source_config
        reference = profiles['E00']
        diagnostic_started = time.perf_counter()
        all_pool = sorted(selected)
        equal_pool = manifest['shared_donor_ids'] == all_pool
        equivalence = [dict(round=rnd, pool_equal=equal_pool,
            scope='recorded original union versus replay all-source E00',
            c_max_abs=max(float((reference.last_matrices[key]-source_c[key]).abs().max()) for key in self.b_keys)
                if equal_pool else None,
            residual_max_abs=max(float((results['E00'][key]-source_residual[key]).abs().max()) for key in self.b_keys),
            c_close=all(torch.allclose(reference.last_matrices[key],source_c[key],atol=1e-5,rtol=1e-4)
                        for key in self.b_keys) if equal_pool else None,
            residual_close=all(torch.allclose(results['E00'][key],source_residual[key],atol=1e-6,rtol=1e-4)
                               for key in self.b_keys) if equal_pool else None)]
        zeros = {key:torch.zeros_like(ordinary[key]) for key in self.b_keys}
        before = measure_residual(reference,ordinary,zeros)
        source_after = measure_residual(reference,ordinary,source_residual)
        saved_scores = read_csv(self.source/f'b_transfer_rounds/r{rnd:03d}/probe_metrics.csv')
        loss_errors = []
        for row in saved_scores:
            scores = {'ordinary_global_B':before,'transferred_global_B':source_after}.get(row['phase'])
            if scores is not None:
                key = int(row['client_id']),int(row['class_id'])
                if scores[key]['samples'] != int(row['samples']):
                    raise ValueError('Saved probe sample count differs')
                loss_errors.append(abs(scores[key]['la']-float(row['la'])))
        equivalence[0]['source_probe_max_la_error'] = max(loss_errors)
        equivalence[0]['c_atol'], equivalence[0]['c_rtol'] = 1e-5, 1e-4
        equivalence[0]['residual_atol'], equivalence[0]['residual_rtol'] = 1e-6, 1e-4
        write_csv(event_root/'equivalence.csv', equivalence)
        donor_rows = []
        if self.job['donor_diagnostics']:
            raw = read_csv(self.source/f'b_transfer_rounds/r{rnd:03d}/donor_scores.csv')
            for donor in sorted(selected):
                removed = without_donor(source_residual,deltas,source_c,manifest['shared_donor_ids'],donor,self.b_keys)
                measured = measure_residual(reference,ordinary,removed) if donor in manifest['shared_donor_ids'] else source_after
                for row in raw:
                    if int(row['donor']) != donor:
                        continue
                    k,c = int(row['receiver']),int(row['class_id'])
                    donor_rows.append(dict(round=rnd,receiver=k,class_id=c,donor=donor,
                        used=donor in manifest['shared_donor_ids'], raw_gain=float(row['gain']),
                        conditional_gain=measured[k,c]['la']-source_after[k,c]['la'],
                        scope='fixed original C, other donors and denominator; common measured relations only'))
        write_csv(event_root/'donor_contributions.csv', donor_rows)
        baseline_norm = reference.effective_norm(results['E00'],start)
        matched_rows = []
        for variant,residual in results.items():
            if variant == 'E00':
                continue
            norm = reference.effective_norm(residual,start)
            if baseline_norm == 0 or norm >= baseline_norm:
                matched_rows.append(dict(round=rnd,variant=variant,status='not_smaller_or_zero_reference',
                                         reference_norm=baseline_norm,target_norm=norm))
                continue
            scale = norm/baseline_norm
            matched = {key:value*scale for key,value in results['E00'].items()}
            after = measure_residual(reference,ordinary,matched)
            matched_rows.extend(dict(row,round=rnd,variant=variant,status='E00_norm_matched',
                reference_norm=baseline_norm,target_norm=norm,scale=scale)
                for row in summarize_changes(changed_rows(before,after,reference.tail)))
        write_csv(event_root/'norm_matched.csv', matched_rows)
        reference.write_pending()
        write_json(event_root/'replay_completion.json', dict(schema_version=SCHEMA,round=rnd,
            variants=self.job['variants'], equivalence=equivalence[0],
            scope='training-side repeated measurements; no test gate or full-trajectory claim',
            seconds=time.perf_counter()-started, extra_diagnostic_seconds=time.perf_counter()-diagnostic_started,
            total_optimizer_steps=sum(p.pending['summary']['optimizer_steps'] for p in profiles.values()),
            total_algorithm_forward_images=sum(p.pending['summary']['algorithm_forward_images'] for p in profiles.values()),
            total_algorithm_backward_images=sum(p.pending['summary']['algorithm_backward_images'] for p in profiles.values()),
            total_diagnostic_forward_images=sum(p.pending['summary']['diagnostic_forward_images'] for p in profiles.values())))
