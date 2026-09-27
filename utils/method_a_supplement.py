"""Four fixed seed42 controls. Official predictions are observational only."""
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from tools.sfra.supplement import (SCHEMA, METHODS, REFERENCE_ROUNDS, read_json, read_csv,
                                    class_groups, prediction_metrics, reference_state_path)
from utils.cliplora_a_refresh import append_rows
from utils.cliplora_bridge_audit import write_csv, write_json
from utils.cliplora_functional_feedback import observational_model
from utils.cliplora_sfra import SFRARuntime
from utils.sfra_math import make_targets, require_finite


class MethodASupplementRuntime(SFRARuntime):
    def __init__(self, trainer, global_trainer, cfg, args, schedule, normal_train):
        self.job = read_json(args.method_a_supplement_manifest)
        self.experiment = self.job['experiment']
        self.source = Path(self.job['source_run'])
        expected = 's' if self.experiment == 'norm-matched' else (
            'current-cp' if self.experiment == 'current-cp' else 'full-cp')
        if (self.job['schema_version'] != SCHEMA or self.experiment not in METHODS or args.sfra_variant != expected
                or args.seed != 42 or args.split_seed != 42 or args.sfra_stop_after_round != 0
                or args.b_transfer_enable or args.sfra_b_aggregation != 'sample'
                or args.sfra_retention_weight != 10 or args.sfra_classification_weight != 1):
            raise ValueError('Use the frozen seed42 supplement launcher and configuration')
        # Direct state loading, no hashes or reconstruction of an assumed random initialization.
        initial = torch.load(self.source/'checkpoints/base_model.pt', map_location='cpu', weights_only=False)
        trainer.model.load_state_dict(initial, strict=True)
        global_trainer.model.load_state_dict(initial, strict=True)
        super().__init__(trainer, global_trainer, cfg, args, schedule, normal_train)
        self.method = self.experiment
        self.groups = class_groups(self.audit.counts.sum(0).tolist(), self.tail)
        if self.config != read_json(self.source/'sfra_config.json')['base_training_config']:
            raise ValueError('Local training schedule/optimizer/LA settings differ from Full-CP')
        reference_execution = (read_json(self.source/'execution_config.json')
                               if (self.source/'execution_config.json').is_file() else None)
        if self.execution_config != reference_execution:
            raise ValueError('Execution implementation/settings differ from Full-CP')
        source_partition = read_csv(self.source/'partition_manifest.csv')
        if source_partition != read_csv(self.root/'partition_manifest.csv'):
            raise ValueError('Actual client partition or sample order differs from Full-CP')
        if self.bank is not None:
            reference_tokens = read_json(self.source/'private_witness_manifest.json')
            if self.bank.tokens != reference_tokens:
                raise ValueError('Witness identities/order differ from Full-CP')
        self.shuffle_orders = {}
        if self.experiment == 'shuffle-cp':
            # Separate RNG, drawn ONCE. An active subset inherits this fixed cyclic order.
            rng = np.random.default_rng(self.job['shuffle_seed'])
            labels = np.array([t['class_id'] for t in self.bank.tokens])
            for c in range(100):
                self.shuffle_orders[c] = rng.permutation(np.flatnonzero(labels == c)).tolist()
            write_json(self.root/'shuffle_orders.json', self.shuffle_orders)
        self.norm_targets = self.job['full_effective_norms']
        write_json(self.root/'experiment_definition.json', dict(
            **self.job, actual_execution_config=self.execution_config,
            evaluation='fixed sequential test set; raw logits; training RNG restored',
            initial_b_is_zero=all(torch.count_nonzero(initial[k]).item() == 0 for k in self.b_keys),
            inference_note='Round 0 is task initialization; not automatically a zero-shot CLIP claim'))

    def configure_experiment(self):
        # Keep the internal CP dispatch unchanged, but never label a new control as Full-CP in its manifest.
        self.sfra_config.update(variant=self.experiment, implementation_variant=self.variant)
        self.sfra_config['supplement'] = dict(schema_version=SCHEMA, experiment=self.experiment,
            shuffle_seed=self.job['shuffle_seed'], target_rule=self.job['target_rule'],
            norm_rule='one scalar; global Frobenius norm of scaling * post-B B * delta_A',
            full_effective_norms=self.job['full_effective_norms'] if self.experiment == 'norm-matched' else None)
        if self.experiment == 'plain-hold':
            self.sfra_config.update(source_coordinates='none; full gradient norms only',
                                    current_gain_fraction=0., normalization='uniform over all witness tokens')
        elif self.experiment == 'norm-matched':
            self.sfra_config.update(source_coordinates='none', correction_steps=0,
                                    current_gain_fraction=0., normalization='no functional targets',
                                    commit_rule='ordinary proposal scaled by one global scalar')

    def measure_refresh_source(self, middle, rnd, matrix, client_ids):
        if self.experiment != 'plain-hold':
            return super().measure_refresh_source(middle, rnd, matrix, client_ids)
        require_finite(proposals=matrix)
        radius = matrix.float().square().sum(0).mean().sqrt()
        self.load_training_state(middle)
        # Zero columns request FULL independent gradients for sigma, without QR or source responses.
        measured = self.functional(rnd, 'plain_hold_at_post_B', basis=matrix.new_empty((matrix.shape[0], 0)))
        count = len(self.bank.tokens)
        self.costs[-1]['upload_bytes'] = count * 2 * 2 * 4  # F and full gradient norm, both views.
        zero = torch.zeros(count)
        # Compatibility placeholders are removed from the saved Plain-Hold arrays below.
        source = dict(supported=torch.zeros(count, dtype=torch.bool), positive_count=zero.long(),
                      u=zero, n_eff=zero, cache=self.history.source_cache.clone(), predicted=torch.zeros(count, 2))
        return radius, measured, source, torch.empty(count, 2, 0)

    def build_refresh_targets(self, scores, source, history, valid, rnd):
        if self.experiment == 'plain-hold':
            target = scores.clone()
            target[valid] = torch.maximum(target[valid], history[valid, None])
            return dict(current=scores.clone(), target=target, active=torch.ones_like(valid),
                        weights=torch.full_like(history, 1. / len(history)))
        target = make_targets(scores, source, history, valid, self.variant)
        if self.experiment != 'shuffle-cp':
            return target
        weights = target['weights'].clone()
        donor = torch.arange(len(weights))
        self.shuffle_class_rows = []
        for c, order in self.shuffle_orders.items():
            ids = [q for q in order if bool(target['active'][q])]
            if len(ids) > 1:
                donor[ids] = torch.tensor(ids[1:] + ids[:1])
            changed = int((weights[ids] != weights[donor[ids]]).sum()) if ids else 0
            before, after = float(weights[ids].double().sum()), float(weights[donor[ids]].double().sum())
            if not math.isclose(before, after, rel_tol=1e-12, abs_tol=1e-12):
                raise ValueError('Shuffle changed a class total weight')
            self.shuffle_class_rows.append(dict(round=rnd, class_id=c, active_tokens=len(ids),
                swappable_tokens=len(ids) if len(ids) > 1 else 0, changed_weight_tokens=changed,
                class_weight_before=before, class_weight_after=after))
        target['weights'] = weights[donor]
        if not torch.equal(weights.sort().values, target['weights'].sort().values):
            raise ValueError('Shuffle changed the weight distribution')
        self.shuffle_original_weights, self.shuffle_donors = weights, donor
        return target

    def refresh(self, middle, rnd, folder, prepared=None):
        committed, scores, arrays, summary = super().refresh(middle, rnd, folder, prepared)
        summary.update(variant=self.experiment, experiment=self.experiment,
                       history_target_enabled=self.experiment != 'current-cp',
                       source_measurement_enabled=self.experiment != 'plain-hold')
        if self.experiment == 'current-cp':
            if not torch.equal(arrays['active'], arrays['supported']):
                raise ValueError('History activated a Current-CP target')
            if not torch.equal(arrays['target'][arrays['active']], arrays['current_target'][arrays['active']]):
                raise ValueError('History altered a Current-CP target')
        if self.experiment == 'plain-hold':
            for key in ('responses', 'predicted_sample_weighted_response', 'positive_sources',
                        'supported', 'u', 'n_eff', 'source_cache'):
                del arrays[key]
            summary.pop('supported_tokens')
        if self.experiment == 'shuffle-cp':
            arrays.update(weights_before_shuffle=self.shuffle_original_weights, weight_donor_token=self.shuffle_donors)
            write_csv(folder/'shuffle_by_class.csv', self.shuffle_class_rows)
            active = summary['active_tokens']
            summary.update(swappable_tokens=sum(r['swappable_tokens'] for r in self.shuffle_class_rows),
                           changed_weight_tokens=sum(r['changed_weight_tokens'] for r in self.shuffle_class_rows))
            summary['changed_weight_fraction'] = summary['changed_weight_tokens'] / active if active else 0.
        return committed, scores, arrays, summary

    def train_phase(self, state, rnd, factor='B', extra=False, branch='main', candidate=0, return_deltas=False):
        result = super().train_phase(state, rnd, factor, extra, branch, candidate, return_deltas)
        if self.experiment != 'norm-matched' or factor != 'A' or not extra:
            return result
        ordinary = result[0] if return_deltas else result
        proposed = self.effective_norms(state, ordinary, ordinary)
        denominator = math.sqrt(sum(r['proposal_effective_norm']**2 for r in proposed))
        target = float(self.norm_targets[str(rnd)])
        if denominator == 0 and target != 0:
            raise ValueError(f'Round {rnd}: zero ordinary effective update cannot match positive target {target}')
        alpha = target / denominator if denominator else 0.
        if not math.isfinite(alpha):
            raise FloatingPointError('Nonfinite matching scalar')
        committed = dict(ordinary)
        for key in self.a_keys:
            committed[key] = (state[key].double() + alpha * (ordinary[key].double()-state[key].double())).to(state[key].dtype)
        require_finite(**{k:committed[k] for k in self.a_keys})
        effective = self.effective_norms(state, ordinary, committed)
        actual = math.sqrt(sum(r['committed_effective_norm']**2 for r in effective))
        error = abs(actual-target)
        if error > max(1e-8, 2e-4*target):
            raise ValueError(f'Round {rnd}: effective norm mismatch: actual={actual}, target={target}')
        row = dict(round=rnd, experiment=self.experiment, variant=self.experiment, correction_steps=0,
                   ordinary_effective_norm=denominator, target_effective_norm=target,
                   committed_effective_norm=actual, absolute_error=error,
                   relative_error=error/target if target else 0., scalar=alpha, enlarged=alpha > 1,
                   zero_effective_proposal=denominator == 0,
                   reference_round=rnd, reference_dependency='Full-CP training norm; never test scores')
        folder = self.root/'sfra_rounds'/f'r{rnd:03d}'
        write_json(folder/'summary.json', row)
        write_csv(folder/'effective_updates.csv', effective)
        torch.save(dict(round=rnd, committed_lora={k:committed[k] for k in self.keys}), folder/'commit.pt')
        self.summaries.append(row)
        self.events[-1].update(state_role='ordinary_A_proposal_before_norm_matching',
                              supplement_committed_state_path=f'sfra_rounds/r{rnd:03d}/commit.pt')
        self.load_training_state(committed)
        print(f'NORM MATCH round={rnd} scalar={alpha:.6g} target={target:.8g} actual={actual:.8g}', flush=True)
        return (committed, result[1]) if return_deltas else committed

    def predict(self, state, path):
        trainer = self.global_trainer
        if not isinstance(trainer.test_loader.sampler, torch.utils.data.SequentialSampler):
            raise ValueError('Per-sample evaluation requires SequentialSampler')
        parts = {k:[] for k in ('class_id', 'prediction', 'ce_loss', 'la_loss', 'margin')}
        with observational_model(trainer.model):
            trainer.model.load_state_dict(state, strict=True)
            core = trainer.model.module if hasattr(trainer.model, 'module') else trainer.model
            scale = core.logit_scale.exp()
            prior = self.trainer.training_logit_adjustment.to(trainer.device)
            for batch in trainer.test_loader:
                images, target = trainer.parse_batch_test(batch)
                logits = trainer.model_inference(images)
                require_finite(logits=logits)
                values = dict(class_id=target, prediction=logits.argmax(1),
                              ce_loss=F.cross_entropy(logits, target, reduction='none'),
                              la_loss=F.cross_entropy(logits+prior, target, reduction='none'),
                              margin=(logits.gather(1, target[:, None]).squeeze(1) -
                                      logits.clone().scatter_(1, target[:, None], -torch.inf).amax(1))/scale)
                for key, value in values.items():
                    parts[key].append(value.detach().cpu().numpy())
        arrays = {key:np.concatenate(value) for key, value in parts.items()}
        arrays['sample_id'] = np.arange(len(arrays['class_id']))
        arrays['correct'] = arrays['class_id'] == arrays['prediction']
        # Dataset order is the raw CIFAR100 test-file order. Store labels as well as positions.
        labels = np.array([int(x.label) for x in trainer.dm.dataset.data_test])
        if (len(labels) != 10000 or not np.array_equal(labels, arrays['class_id'])
                or not np.all(np.bincount(labels, minlength=100) == 100)):
            raise ValueError('Official test sample identities/order changed')
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix('.tmp')
        with temporary.open('wb') as stream:
            np.savez_compressed(stream, **arrays)
        temporary.replace(path)
        return arrays

    def publish(self, state, rnd, decision_round):
        started = time.perf_counter()
        arrays = self.predict(state, self.root/'predictions'/f'r{rnd:03d}.npz')
        rows = prediction_metrics(arrays, self.groups, rnd)
        values = {r['group']:r['accuracy'] for r in rows}
        counts = np.bincount(arrays['class_id'], minlength=100)
        correct = np.bincount(arrays['class_id'], weights=arrays['correct'], minlength=100)
        per_class = correct/counts*100
        non_tail = np.setdiff1d(np.arange(100), self.tail)
        append_rows(self.root/'round_metrics.csv', [dict(epoch=rnd-1, round=rnd, decision_round=decision_round,
            method=self.experiment, partition=self.args.partition, seed=self.args.seed,
            overall_acc=values['Overall'], non_tail_acc=float(per_class[non_tail].mean()),
            head20_acc=float(per_class[self.head].mean()), middle60_acc=float(per_class[self.middle].mean()),
            bottom20_tail_acc=values['Tail20'], macro_per_class_acc=float(per_class.mean()),
            many35_acc=values['Many35'], medium35_acc=values['Medium35'], few30_acc=values['Few30'])])
        write_csv(self.root/f'per_class_accuracy_epoch_{rnd-1}.csv',
                  [dict(class_id=c, per_class_acc=float(per_class[c])) for c in range(100)])
        write_csv(self.root/'prediction_metrics'/f'r{rnd:03d}.csv', rows)
        # Including B-only rounds gives a directly evaluable formal model at every round.
        folder = self.root/'formal_lora'
        folder.mkdir(exist_ok=True)
        torch.save(dict(round=rnd, committed_lora={k:state[k] for k in self.keys}), folder/f'r{rnd:03d}.pt')
        self.evaluations.append(dict(label=f'official_round_{rnd}', kind='official_test',
            seconds=time.perf_counter()-started, sample_presentations=len(arrays['sample_id']),
            upload_bytes=0, modeled_downlink_bytes=0))
        print(f'{self.experiment} round={rnd}: Overall={values["Overall"]:.3f} '
              f'Medium35={values["Medium35"]:.3f} Tail20={values["Tail20"]:.3f}', flush=True)

    def export_reference(self):
        """A few saved Full-CP models, forward evaluation only; never another training run."""
        root = self.root/'reference_eval'
        costs = []
        for rnd in REFERENCE_ROUNDS:
            path = root/'predictions'/f'r{rnd:03d}.npz'
            if path.is_file() and (root/'prediction_metrics'/f'r{rnd:03d}.csv').is_file():
                continue  # This directory belongs to the immutable registered job.
            state = dict(self.base_state)
            if rnd:
                saved = torch.load(reference_state_path(self.source, rnd), map_location='cpu', weights_only=False)
                if int(saved['round']) != rnd:
                    raise ValueError(f'Wrong reference checkpoint at round {rnd}')
                state.update(saved['committed_lora'] if rnd <= 90 else saved['actual_after_lora_state'])
            started = time.perf_counter()
            arrays = self.predict(state, path)
            write_csv(root/'prediction_metrics'/f'r{rnd:03d}.csv', prediction_metrics(arrays, self.groups, rnd))
            costs.append(dict(round=rnd, seconds=time.perf_counter()-started, test_images=10000, training_steps=0))
        if costs:
            append_rows(root/'costs.csv', costs)
        write_json(root/'completion.json', dict(rounds=list(REFERENCE_ROUNDS), source=str(self.source), training_steps=0))

    def run(self, initial_state):
        if self.experiment == 'norm-matched':
            self.export_reference()
        # Do not use federated_main's earlier global_weights after loading the actual reference initialization.
        super().run(self.base_state)

    def flush_records(self, completed):
        if completed > 90 and self.experiment != 'norm-matched' and self.summaries:
            row = self.summaries[-1]
            if row['round'] == completed:
                row.update(variant=self.experiment, experiment=self.experiment)
                write_json(self.root/'sfra_rounds'/f'r{completed:03d}'/'summary.json', row)
        super().flush_records(completed)
        for name in ('progress.json', 'completion.json') if completed == 100 else ('progress.json',):
            value = read_json(self.root/name)
            value.update(variant=self.experiment, implementation_variant=self.variant,
                         experiment=self.experiment, experiment_schema=SCHEMA)
            write_json(self.root/name, value)
