"""Paired five-round diagnostic branches; never entered by ordinary SFRA runs."""
import copy
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from tools.sfra.diagnostics import (SCHEMA, ANCHORS, HORIZON, BRANCHES, class_groups,
                                    read_json, write_json, write_csv, file_hash)
from utils.cliplora_a_refresh import isolated_rng, state_hash, train_only
from utils.cliplora_functional_feedback import observational_model
from utils.cliplora_sfra import SFRARuntime
from utils.sfra_math import FunctionalHistory


def effective_norm(before, after, a_keys, scaling):
    squared = 0.
    for key in a_keys:
        delta = after[key].double() - before[key].double()
        product = scaling * (before[key[:-1]+'B'].double() @ delta)
        squared += float(product.square().sum())
    return squared ** .5


def norm_matched_candidate(before, ordinary, target_norm, a_keys, scaling, epsilon=1e-12):
    """One scalar in A-space matches the concatenated effective sBΔA norm."""
    denominator = effective_norm(before, ordinary, a_keys, scaling)
    if not np.isfinite(target_norm) or target_norm < 0:
        raise ValueError('Invalid target norm')
    if not np.isfinite(denominator) or denominator <= epsilon:
        raise ValueError('Ordinary effective update is zero/too small: cannot match; no fallback')
    alpha = target_norm / denominator
    result = dict(before)
    for key in a_keys:
        result[key] = (before[key].double() + alpha * (ordinary[key].double()-before[key].double())).to(before[key].dtype)
    actual = effective_norm(before, result, a_keys, scaling)
    error = abs(actual-target_norm)
    # Casting the committed A back to FP32 introduces cancellation at very small steps.
    if error > max(1e-8, 2e-4*target_norm):
        raise ValueError(f'Effective norm matching failed: {actual} != {target_norm}')
    return result, dict(alpha=alpha, ordinary_effective_norm=denominator, target_effective_norm=target_norm,
                        actual_effective_norm=actual, norm_absolute_error=error,
                        norm_relative_error=error/target_norm if target_norm else None,
                        update_kind='enlarged' if alpha > 1 else 'reduced')


def recover_history(initial_scores, source, anchor, clients):
    """Replay stored committed scores, not newly generated historical measurements."""
    history = FunctionalHistory(initial_scores, clients)
    for rnd in range(1, anchor+1):
        with np.load(Path(source)/f'sfra_rounds/r{rnd:03d}/tokens.npz', allow_pickle=False) as z:
            history.commit(torch.from_numpy(z['F_committed'].copy()), rnd)
            if not torch.equal(history.valid, torch.from_numpy(z['history_valid_after'].copy())):
                raise ValueError(f'History validity replay differs at round {rnd}')
            if not torch.allclose(history.level, torch.from_numpy(z['history_after'].copy()), atol=1e-7, rtol=0):
                raise ValueError(f'History level replay differs at round {rnd}')
            history.source_cache = torch.from_numpy(z['source_cache'].copy())
    if history.block_count != 0:
        raise ValueError('Expected a five-round history block boundary')
    return history


class MethodADiagnosticRuntime(SFRARuntime):
    def __init__(self, trainer, global_trainer, cfg, args, schedule, normal_train):
        self.job = read_json(args.method_a_diagnostic_manifest)
        if (self.job['schema_version'] != SCHEMA or self.job['anchor'] not in ANCHORS
                or self.job['branch'] not in BRANCHES or self.job['horizon'] != HORIZON):
            raise ValueError('Unsupported diagnostic protocol')
        if args.sfra_resume or args.sfra_variant != 'full-cp' or getattr(args, 'b_transfer_enable', False):
            raise ValueError('Diagnostics require A-only full-cp; ordinary --resume is not a branch operation')
        super().__init__(trainer, global_trainer, cfg, args, schedule, normal_train)
        self.source = Path(self.job['source'])
        self.anchor = self.job['anchor']
        self.branch_name = self.job['branch']
        self.groups = class_groups(self.audit.counts.sum(0).numpy(), self.tail)
        self.metrics, self.witness_metrics, self.norms = [], [], []
        self.evaluation_rows = []
        source_meta = read_json(self.source/'bridge_metadata.json')
        for key in ('pool_sha256', 'test_sha256', 'schedule_sha256', 'frozen_model_sha256',
                    'initial_lora_sha256', 'probe_images_sha256', 'probe_manifest_sha256'):
            if self.audit.meta[key] != source_meta[key]:
                raise ValueError(f'Diagnostic reference mismatch: {key}')
        if self.bank.tokens != read_json(self.source/'private_witness_manifest.json'):
            raise ValueError('Witness identities differ from the reference')
        source_execution = read_json(self.source/'execution_config.json') if (self.source/'execution_config.json').exists() else None
        if self.execution_config != source_execution:
            raise ValueError('Execution mode differs from the reference')
        self.source_meta = source_meta

    def activate_execution_override(self, state):
        """Switch only the NEW paired continuation, never historical replay."""
        requested = self.job.get('execution_override')
        if requested is None:
            return
        from utils.sfra_resident_feedback import execution_config_v2, ResidentWitnessFeedback
        from utils.sfra_execution import enable_model_execution, LoRAStateLoader
        expected = execution_config_v2(requested['feedback_forward_batch_size'], requested['device_cache_gib'])
        if requested != expected:
            raise ValueError('Unsupported diagnostic execution override')
        reference_execution = copy.deepcopy(self.execution_config)
        # Release a possible reference-mode device cache before building v2.
        if hasattr(self.bank, '_fast_feedback'):
            del self.bank._fast_feedback
        enable_model_execution(self.trainer.model)
        enable_model_execution(self.global_trainer.model)
        self._execution_loader = LoRAStateLoader(self.trainer.model, self.keys)
        self.load_training_state(state)
        self.bank._fast_feedback = ResidentWitnessFeedback(
            self.bank, requested['feedback_forward_batch_size'], requested['device_cache_gib'],
            requested['device_cache_reserve_gib'])
        observed = self.bank.evaluate()['scores'].cpu().numpy()
        with np.load(self.source/f'sfra_rounds/r{self.anchor:03d}/tokens.npz', allow_pickle=False) as archive:
            error = float(np.max(np.abs(observed - archive['F_committed'])))
        if not np.isfinite(error) or error > 5e-6:
            raise ValueError(f'Accelerated anchor witness reproduction error: {error}')
        self.execution_config = expected
        write_json(self.root/'execution_config.json', expected)
        write_json(self.root/'diagnostic_execution_audit.json', dict(
            reference_execution=reference_execution, continuation_execution=expected,
            switch_after_reference_anchor_and_history_audits=True,
            anchor_margin_max_error=error, original_historical_scores_preserved=True,
            scope='new paired five-round continuation; not bitwise replay of the original trajectory'))
        print(f'Diagnostic v2 enabled after anchor audit: feedback batch='
              f'{requested["feedback_forward_batch_size"]}, cache <= '
              f'{requested["device_cache_gib"]:g} GiB', flush=True)

    def load_anchor(self):
        base = torch.load(self.source/'checkpoints/base_model.pt', map_location='cpu', weights_only=False)
        if state_hash(base, self.keys) != self.source_meta['initial_lora_sha256']:
            raise ValueError('Saved base LoRA hash differs')
        frozen = sorted(set(base)-set(self.keys))
        if state_hash(base, frozen) != self.source_meta['frozen_model_sha256']:
            raise ValueError('Saved frozen base hash differs')
        self.base_state = base
        self.load_training_state(base)
        initial = self.bank.evaluate()['scores']
        self.history = recover_history(initial, self.source, self.anchor, len(self.sizes))
        saved = torch.load(self.source/f'sfra_rounds/r{self.anchor:03d}/commit.pt',
                           map_location='cpu', weights_only=False)
        if saved['round'] != self.anchor or set(saved['committed_lora']) != set(self.keys):
            raise ValueError('Anchor commit keys or round differ')
        state = dict(base)
        state.update(saved['committed_lora'])
        expected = read_json(self.source/f'events/r{self.anchor+1:03d}_c000_main_normal_B/event.json')
        if state_hash(state, self.keys) != expected['before_lora_sha256']:
            raise ValueError('Anchor commit hash differs from the next B event input')
        self.load_training_state(state)
        measured = self.bank.evaluate()['scores'].cpu().numpy()
        with np.load(self.source/f'sfra_rounds/r{self.anchor:03d}/tokens.npz', allow_pickle=False) as z:
            error = float(np.max(np.abs(measured-z['F_committed'])))
        if error > 5e-6:
            raise ValueError(f'Anchor witness reproduction error: {error}')
        write_json(self.root/'anchor_audit.json', dict(anchor=self.anchor, anchor_lora_sha256=state_hash(state, self.keys),
                   anchor_margin_max_error=error, history_replayed_rounds=self.anchor,
                   reference_state_file=str(self.source/f'sfra_rounds/r{self.anchor:03d}/commit.pt'),
                   rng_protocol='new paired per-phase stream; original historical RNG is not assumed'))
        return state

    def evaluate_candidate(self, state, rnd, candidate, kind):
        """Read-only witness feedback and fixed-order test predictions; never used to choose updates."""
        started = time.perf_counter()
        self.load_training_state(state)
        with isolated_rng():
            observed = self.bank.evaluate(classification=True, classification_gradient=False)
        labels = np.array([t['class_id'] for t in self.bank.tokens])
        witness_dir = self.root/'witness_tokens'
        witness_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(witness_dir/f'r{rnd:03d}_{candidate}_{kind}.npz',
            class_ids=labels, client_ids=np.array([t['client_id'] for t in self.bank.tokens]),
            scores=observed['scores'].cpu().numpy(),
            classification_scores=observed['classification_scores'].cpu().numpy())
        loss = observed['classification_scores'].cpu().numpy().mean(1)
        margin = observed['scores'].cpu().numpy().mean(1)
        for group, ids in self.groups.items():
            self.witness_metrics.append(dict(round=rnd, candidate=candidate, kind=kind, group=group,
                la_loss=float(np.mean([loss[labels==c].mean() for c in ids])),
                margin=float(np.mean([margin[labels==c].mean() for c in ids]))))
        sample_rows = []
        correct, counts = np.zeros(100), np.zeros(100)
        ce_sum, la_sum, margin_sum = np.zeros(100), np.zeros(100), np.zeros(100)
        trainer = self.global_trainer
        if not isinstance(trainer.test_loader.sampler, torch.utils.data.SequentialSampler):
            raise ValueError('Sample IDs require a sequential fixed test loader')
        with observational_model(trainer.model):
            trainer.model.load_state_dict(state, strict=True)
            core = trainer.model.module if hasattr(trainer.model, 'module') else trainer.model
            scale = float(core.logit_scale.exp())
            prior = self.trainer.training_logit_adjustment.to(trainer.device)
            for batch in trainer.test_loader:
                images, target = trainer.parse_batch_test(batch)
                logits = trainer.model_inference(images)
                prediction = logits.argmax(1)
                ce = F.cross_entropy(logits, target, reduction='none')
                la = F.cross_entropy(logits+prior, target, reduction='none')
                true = logits.gather(1, target[:, None]).squeeze(1)
                wrong = logits.clone().scatter_(1, target[:, None], -torch.inf).max(1).values
                margins = (true-wrong)/scale
                for c, p, e, a, m in zip(target.tolist(), prediction.tolist(), ce.tolist(), la.tolist(), margins.tolist()):
                    row = dict(sample_id=len(sample_rows), class_id=c, prediction=p, correct=int(c==p),
                               ce_loss=e, la_loss=a, margin=m)
                    sample_rows.append(row)
                    counts[c] += 1
                    correct[c] += c==p
                    ce_sum[c] += e
                    la_sum[c] += a
                    margin_sum[c] += m
        if (counts == 0).any() or len(sample_rows) != len(trainer.test_loader.dataset):
            raise ValueError('Test coverage differs from the fixed evaluation set')
        path = self.root/'predictions'/f'r{rnd:03d}_{candidate}_{kind}.csv'
        write_csv(path, sample_rows)
        for group, ids in self.groups.items():
            acc = float(correct.sum()/counts.sum()*100) if group=='Overall' else float(np.mean(correct[ids]/counts[ids])*100)
            self.metrics.append(dict(round=rnd, candidate=candidate, kind=kind, group=group,
                accuracy=acc, ce_loss=float(np.mean(ce_sum[ids]/counts[ids])),
                la_loss=float(np.mean(la_sum[ids]/counts[ids])), margin=float(np.mean(margin_sum[ids]/counts[ids]))))
        self.evaluation_rows.append(dict(round=rnd, candidate=candidate, kind=kind,
            seconds=time.perf_counter()-started, witness_forward_images=observed['forward_images'],
            test_images=len(sample_rows), purpose='offline_diagnostic_only'))
        return observed['scores']

    def no_cp_probe(self, middle, prepared, rnd, history_before):
        after = copy.deepcopy(self.history.state_dict())
        weight = self.classification_strength
        cost_start = len(self.costs)
        try:
            self.history.load_state_dict(history_before)
            self.classification_strength = 0.
            folder = self.root/'no_cp_probe'/f'r{rnd:03d}'
            folder.mkdir(parents=True, exist_ok=True)
            with isolated_rng():
                state, scores, arrays, summary = self.refresh(middle, rnd, folder, prepared=prepared)
            np.savez_compressed(folder/'tokens.npz', **{k:v.cpu().numpy() for k,v in arrays.items()})
            write_json(folder/'summary.json', summary)
            return state, arrays
        finally:
            for row in self.costs[cost_start:]:
                row['diagnostic_role'] = 'mu0_probe_only'
            self.classification_strength = weight
            self.history.load_state_dict(after)

    def flush_diagnostic(self, state, completed):
        write_csv(self.root/'candidate_metrics.csv', self.metrics)
        write_csv(self.root/'witness_metrics.csv', self.witness_metrics)
        write_csv(self.root/'norms.csv', self.norms)
        write_csv(self.root/'diagnostic_evaluation_costs.csv', self.evaluation_rows)
        write_csv(self.root/'functional_costs.csv', self.costs)
        write_csv(self.root/'local_training_budget.csv', self.budget)
        write_csv(self.root/'event_manifest.csv', self.events)
        payload = dict(job=self.job, completed_round=completed, state_dict_overrides=self.compressed_state(state),
            history=self.history.state_dict(), metrics=self.metrics, witness_metrics=self.witness_metrics,
            norms=self.norms, evaluation_rows=self.evaluation_rows, costs=self.costs, budget=self.budget, events=self.events)
        temporary = self.root/'diagnostic_last.tmp'
        torch.save(payload, temporary)
        temporary.replace(self.root/'diagnostic_last.pt')
        write_json(self.root/'diagnostic_progress.json', dict(completed_round=completed, anchor=self.anchor,
                   branch=self.branch_name, target_round=self.anchor+HORIZON))

    def run(self, initial_state):
        started = time.perf_counter()
        state = self.load_anchor()
        self.activate_execution_override(state)
        completed = self.anchor
        checkpoint = self.root/'diagnostic_last.pt'
        if self.job.get('resume') and checkpoint.is_file():
            saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
            if {k:v for k,v in saved['job'].items() if k!='resume'} != {k:v for k,v in self.job.items() if k!='resume'}:
                raise ValueError('Resume job differs')
            state = dict(self.base_state)
            state.update(saved['state_dict_overrides'])
            self.history.load_state_dict(saved['history'])
            completed = saved['completed_round']
            for key in ('metrics', 'witness_metrics', 'norms', 'evaluation_rows', 'costs', 'budget', 'events'):
                setattr(self, key, saved[key])
        else:
            self.evaluate_candidate(state, self.anchor, self.branch_name, 'anchor')
            self.flush_diagnostic(state, self.anchor)
        reference = {}
        if self.branch_name == 'norm-matched':
            ref = Path(self.job['full_branch'])
            done = read_json(ref/'diagnostic_completion.json')
            if done['anchor'] != self.anchor or done['completed_round'] != self.anchor+HORIZON:
                raise ValueError('Full-CP reference branch is incomplete')
            from tools.sfra.diagnostics import read_csv
            reference = {int(r['round']):float(r['actual_effective_norm']) for r in read_csv(ref/'norms.csv')}
        for rnd in range(completed+1, self.anchor+HORIZON+1):
            # Explicitly fork a new common random stream at each phase. Historical RNG is not fabricated.
            with isolated_rng(self.args.seed + 1000003*rnd + 271):
                middle = self.train_phase(state, rnd, 'B')
            with isolated_rng(self.args.seed + 1000003*rnd + 719):
                ordinary, deltas = self.train_phase(middle, rnd, 'A', True, return_deltas=True)
            self.events[-1]['state_role'] = 'ordinary_A_proposal_before_diagnostic_intervention'
            prepared = ordinary, deltas
            norm_info = {}
            if self.branch_name == 'full-cp':
                history_before = copy.deepcopy(self.history.state_dict())
                folder = self.root/'sfra_rounds'/f'r{rnd:03d}'
                folder.mkdir(parents=True, exist_ok=True)
                state, scores, arrays, summary = self.refresh(middle, rnd, folder, prepared=prepared)
                norm = effective_norm(middle, state, self.a_keys, self.scaling)
                if rnd == self.anchor+1:
                    probe, probe_arrays = self.no_cp_probe(middle, prepared, rnd, history_before)
                    for key in ('target', 'weights', 'sigma', 'responses', 'history_before'):
                        torch.testing.assert_close(arrays[key], probe_arrays[key], rtol=0, atol=0)
                    matched, match = norm_matched_candidate(middle, ordinary, norm, self.a_keys, self.scaling)
                    write_json(self.root/'first_event_match.json', match)
                    for name, candidate in [('pre-A', middle), ('ordinary', ordinary), ('full-cp', state),
                                             ('no-cp', probe), ('norm-matched', matched)]:
                        self.evaluate_candidate(candidate, rnd, name, 'same-state')
                self.history.commit(scores, rnd)
                arrays.update(history_after=self.history.level, history_valid_after=self.history.valid)
                np.savez_compressed(folder/'tokens.npz', **{k:v.cpu().numpy() for k,v in arrays.items()})
                write_json(folder/'summary.json', summary)
            elif self.branch_name == 'ordinary':
                state = ordinary
            else:
                state, norm_info = norm_matched_candidate(middle, ordinary, reference[rnd], self.a_keys, self.scaling)
            self.norms.append(dict(round=rnd, branch=self.branch_name,
                actual_effective_norm=effective_norm(middle, state, self.a_keys, self.scaling),
                ordinary_effective_norm=effective_norm(middle, ordinary, self.a_keys, self.scaling),
                post_b_lora_sha256=state_hash(middle, self.keys), ordinary_lora_sha256=state_hash(ordinary, self.keys),
                **{k:v for k,v in norm_info.items() if k not in ('actual_effective_norm', 'ordinary_effective_norm')}))
            if self.branch_name != 'full-cp' and rnd == self.anchor+1:
                from tools.sfra.diagnostics import read_csv
                ref_row = read_csv(Path(self.job['full_branch'])/'norms.csv')[0]
                for key in ('post_b_lora_sha256', 'ordinary_lora_sha256'):
                    if self.norms[-1][key] != ref_row[key]:
                        raise ValueError(f'Branches do not share the first ordinary proposal: {key}')
            self.evaluate_candidate(state, rnd, self.branch_name, 'committed')
            self.load_training_state(state)
            train_only(self.trainer.model, 'B')
            self.flush_diagnostic(state, rnd)
            print(f'METHOD A DIAGNOSTIC anchor={self.anchor} branch={self.branch_name} round={rnd}', flush=True)
        write_json(self.root/'diagnostic_completion.json', dict(schema_version=SCHEMA, anchor=self.anchor,
            branch=self.branch_name, completed_round=self.anchor+HORIZON,
            seconds_this_process=time.perf_counter()-started,
            limitation='single-seed five-round intervention; norm-matched uses Full-CP reference amplitudes'))
