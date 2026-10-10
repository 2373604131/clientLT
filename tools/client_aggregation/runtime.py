"""Opt-in runtime; legacy SFRA, its objectives, and its launchers remain untouched."""
import time
from pathlib import Path

import numpy as np
import torch

from scripts.run_ab_validation import write_json
from tools.client_aggregation.protocol import (SCHEMA, DIAGNOSTIC_ROUNDS, check_aggregation,
    load_json, read_csv, partition_signature, write_table, write_weight_tables, aggregation_name, validate_arm)
from utils.cliplora_a_refresh import aggregate_refresh_deltas, append_rows, state_hash, train_only
from utils.cliplora_functional_feedback import observational_model, snapshot
from utils.cliplora_sfra import SFRARuntime
from utils.lora_aggregation import aggregate_lora_state
from utils.sfra_math import require_finite


def runtime_class(job):
    class JointAggregationRuntime(SFRARuntime):
        def __init__(self, *args, **kwargs):
            from tools.client_aggregation.numerics import apply_policy, policy_for
            # federated_main turns cuDNN benchmarking back on during setup.
            # Restore our registered policy before runtime feedback/evaluation.
            apply_policy(policy_for(job))
            super().__init__(*args, **kwargs)
            self.client_pool = None
            if job.get('settings', {}).get('client_concurrency', 1) > 1:
                from tools.client_aggregation.parallel import LoRAClientPool
                self.client_pool = LoRAClientPool(self)
            if self.b_transfer is not None:
                original = self.b_transfer.apply_shared

                def measured_transfer(start, ordinary, rnd):
                    if self.diagnostic_round(rnd):
                        self.stage_evaluate(ordinary, rnd, 'ordinary_B')
                    result = original(start, ordinary, rnd)
                    if self.diagnostic_round(rnd):
                        self.stage_evaluate(result, rnd, 'after_B_transfer')
                    return result

                self.b_transfer.apply_shared = measured_transfer
            print('CLIENT AGGREGATION:', aggregation_name(job['aggregation']), job['arm'],
                  '; frozen A has no refresh' if self.is_frozen else
                  '; ordinary A rounds1..90; no correction or transfer' if job['arm'] == 'plain_a' else
                  '; Full-CP + original shared C', flush=True)

        def configure_experiment(self):
            validate_arm(job)
            self.job = job
            self.is_frozen = job['arm'] == 'frozen'
            plain = job['arm'] == 'plain_a'
            self.joint_weights = dict(enumerate(job['aggregation']['weights']))
            check_aggregation(job['aggregation'])
            actual = self.audit.counts.cpu().numpy()
            if not np.array_equal(actual, np.asarray(job['aggregation']['matrix'])):
                raise ValueError('Runtime class counts differ from the frozen aggregation inputs')
            if partition_signature(read_csv(self.root / 'partition_manifest.csv')) != job['aggregation']['partition_sha256']:
                raise ValueError('Runtime sample allocation differs from the frozen partition')
            if self.sfra_config.get('b_aggregation'):
                raise ValueError('Do not combine joint aggregation with other aggregation overrides')
            if (self.variant != ('full-cp' if job['arm'] == 'ab' else 's')
                    or bool(self.sfra_config.get('b_transfer')) != (job['arm'] == 'ab')):
                raise ValueError('Wrong frozen/AB runtime variant')
            self.config['aggregation'] = aggregation_name(job['aggregation'])
            if self.is_frozen:
                self.rounds = []
                self.periodic = False
                self.config.update(candidate_rounds=[], extra_trainable_factor=None, extra_steps_expected=0)
                self.sfra_config['refresh_rounds'] = []
            elif plain:
                self.rounds = list(range(1, 91))
                self.periodic = True
                self.config.update(candidate_rounds=self.rounds, extra_trainable_factor='A', extra_steps_expected=31680)
                self.sfra_config.update(refresh_rounds=self.rounds, retention_weight=0., correction_steps=0,
                                        commit_rule='ordinary_A_without_correction')
            else:
                transfer = self.sfra_config['b_transfer']
                transfer.update(aggregation='ordinary_joint_class_weighted_B_then_one_shared_residual',
                    calibration_anchor='actual ordinary joint-class-weighted B with round-start A')
                if job['mode'] == 'smoke':
                    transfer['rounds'] = [1]  # Separate disposable run, never formal evidence.
            self.sfra_config['joint_aggregation'] = dict(schema=SCHEMA, arm=job['arm'], mode=job['mode'],
                lambda_value=job['aggregation']['lambda_value'], weights_sha256=job['aggregation']['weights_sha256'])
            from tools.client_aggregation.protocol import client_execution_config
            self.sfra_config['client_execution'] = client_execution_config(job)
            from tools.client_aggregation.numerics import policy_for, verify_active
            if policy_for(job) == 'deterministic':
                write_json(self.root / 'cuda_numerics.json', verify_active('deterministic'))
            write_json(self.root / 'client_execution.json', self.sfra_config['client_execution'])
            self.method = 'joint_' + job['arm']
            write_json(self.root / 'control_config.json', self.config)
            self.audit.meta['control_config'] = self.config
            write_json(self.root / 'bridge_metadata.json', self.audit.meta)
            write_json(self.root / 'aggregation_solution.json', job['aggregation'])
            write_weight_tables(self.root, job['aggregation'])

        def diagnostic_round(self, rnd):
            return rnd in DIAGNOSTIC_ROUNDS or (job['mode'] == 'smoke' and rnd == 1)

        def phase_aggregation_weights(self, selected, rnd, factor, extra=False, branch='main'):
            if (sorted(selected) != list(range(30)) or branch != 'main'
                    or not (factor == 'B' and not extra and 1 <= rnd <= 100
                            or factor == 'A' and extra and not self.is_frozen and 1 <= rnd <= 90)):
                raise ValueError('Unexpected aggregation phase')
            write_table(self.root / 'aggregation_weights' / f'r{rnd:03d}_{factor}.csv',
                [dict(round=rnd, factor=factor, client_id=j, original_weight=self.q[j],
                      weight=self.joint_weights[j]) for j in sorted(selected)])
            return self.joint_weights

        def prepare_b_aggregation(self, state, local_states, deltas, selected, rnd):
            if self.diagnostic_round(rnd):
                # Identical local updates and round-start state; this branch is never committed.
                counterfactual = aggregate_lora_state(state, local_states, selected, self.b_keys, self.q)
                self.stage_evaluate(counterfactual, rnd, 'sample_weighted_B_counterfactual')
            return super().prepare_b_aggregation(state, local_states, deltas, selected, rnd)

        def train_phase(self, state, rnd, factor='B', extra=False, branch='main', candidate=0,
                        return_deltas=False):
            from tools.client_aggregation.numerics import policy_for, verify_active
            if policy_for(job) == 'deterministic':
                verify_active('deterministic')
            if self.is_frozen and (factor != 'B' or extra):
                raise ValueError('Frozen arm must never train A or execute extra local training')
            if getattr(self, 'client_pool', None) is not None:
                from tools.client_aggregation.parallel import parallel_phase
                after, deltas = parallel_phase(self, state, rnd, factor, extra, branch, candidate)
            else:
                after, deltas = super().train_phase(state, rnd, factor, extra, branch, candidate, return_deltas=True)
            if self.diagnostic_round(rnd):
                if factor == 'A':
                    self.stage_evaluate(after, rnd, 'ordinary_A')
                    reference = aggregate_refresh_deltas(state, deltas, self.q, self.a_keys)
                    self.stage_evaluate(reference, rnd, 'sample_weighted_A_counterfactual')
                elif self.b_transfer is None or not self.b_transfer.scheduled(rnd):
                    self.stage_evaluate(after, rnd, 'ordinary_B')
            return (after, deltas) if return_deltas else after

        def evaluate_predictions(self, state, rnd, stage, folder):
            started = time.perf_counter()
            trainer = self.global_trainer
            if not isinstance(trainer.test_loader.sampler, torch.utils.data.SequentialSampler):
                raise ValueError('Diagnostics require the immutable sequential test ordering')
            labels, predictions, margins = [], [], []
            with observational_model(trainer.model):
                trainer.model.load_state_dict(state, strict=True)
                for batch in trainer.test_loader:
                    images, target = trainer.parse_batch_test(batch)
                    logits = trainer.model_inference(images)
                    require_finite(logits=logits)
                    correct_logit = logits.gather(1, target[:, None]).squeeze(1)
                    competitors = logits.clone()
                    competitors.scatter_(1, target[:, None], -float('inf'))
                    margins.append((correct_logit - competitors.max(1).values).cpu().numpy())
                    labels.append(target.cpu().numpy())
                    predictions.append(logits.argmax(1).cpu().numpy())
            labels, predictions, margins = map(np.concatenate, (labels, predictions, margins))
            expected = np.array([int(item.label) for item in trainer.dm.dataset.data_test])
            if (len(labels) != 10000 or not np.array_equal(labels, expected)
                    or not np.all(np.bincount(labels, minlength=100) == 100)):
                raise ValueError('Official test labels/identities differ')
            correct = labels == predictions
            class_acc = np.bincount(labels, weights=correct, minlength=100).astype(float)
            groups = dict(overall_acc=np.arange(100), head20_acc=self.head,
                middle60_acc=self.middle, bottom20_tail_acc=self.tail,
                non_tail_acc=np.setdiff1d(np.arange(100), self.tail))
            metrics = {key: float(class_acc[ids].mean()) for key, ids in groups.items()}
            folder.mkdir(parents=True, exist_ok=True)
            path = folder / f'r{rnd:03d}.npz'
            temporary = path.with_suffix('.tmp')
            with temporary.open('wb') as stream:
                np.savez_compressed(stream, sample_id=np.arange(10000), class_id=labels,
                    prediction=predictions, correct=correct, logit_margin=margins)
            temporary.replace(path)
            write_json(folder / f'r{rnd:03d}.json', dict(round=rnd, stage=stage, **metrics,
                seconds=time.perf_counter() - started, sample_presentations=len(labels),
                a_sha256=state_hash(state, self.a_keys), b_sha256=state_hash(state, self.b_keys),
                scope='read-only full official test; no training decisions'))
            return metrics, class_acc

        def stage_evaluate(self, state, rnd, stage):
            self.evaluate_predictions(state, rnd, stage, self.root / 'stage_predictions' / stage)

        def publish(self, state, rnd, decision_round):
            if self.is_frozen and state_hash(state, self.a_keys) != state_hash(self.initial, self.a_keys):
                raise ValueError('Frozen LoRA A changed')
            metrics, per_class = self.evaluate_predictions(state, rnd, 'committed', self.root / 'predictions')
            append_rows(self.root / 'round_metrics.csv', [dict(epoch=rnd-1, round=rnd,
                decision_round=decision_round, method=self.method, partition=self.args.partition,
                seed=self.args.seed, macro_per_class_acc=metrics['overall_acc'], **metrics)])
            write_table(self.root / f'per_class_accuracy_epoch_{rnd-1}.csv',
                [dict(class_id=c, per_class_acc=float(per_class[c])) for c in range(100)])
            record = load_json(self.root / 'predictions' / f'r{rnd:03d}.json')
            self.evaluations.append(dict(label=f'official_round_{rnd}', kind='official_test',
                seconds=record['seconds'], sample_presentations=10000, upload_bytes=0, modeled_downlink_bytes=0))
            print(f'joint/{job["arm"]} round={rnd}: Overall={metrics["overall_acc"]:.4f} '
                  f'Tail20={metrics["bottom20_tail_acc"]:.4f}', flush=True)

        def flush_records(self, completed):
            super().flush_records(completed)
            stages = [load_json(p) for p in (self.root / 'stage_predictions').glob('*/*.json')]
            diagnostic_seconds = sum(row['seconds'] for row in stages if row['round'] <= completed)
            for filename in ('progress.json', 'completion.json') if completed == 100 else ('progress.json',):
                value = load_json(self.root / filename)
                value.update(experiment=SCHEMA, arm=job['arm'], mode=job['mode'],
                    weights_sha256=job['aggregation']['weights_sha256'],
                    stage_diagnostic_seconds=diagnostic_seconds,
                    stage_diagnostic_images=sum(row['sample_presentations'] for row in stages if row['round'] <= completed))
                write_json(self.root / filename, value)
            if torch.cuda.is_available():
                write_json(self.root / 'resource_usage.json', dict(
                    peak_allocated_bytes=int(torch.cuda.max_memory_allocated(self.trainer.device)),
                    scope='process CUDA allocated high-water mark, including diagnostics'))

        def run(self, initial_state):
            if not self.is_frozen:
                return super().run(initial_state)
            last = getattr(self.args, 'sfra_stop_after_round', 0) or 100
            if self.resume_payload:
                state, completed = self.restore()
            else:
                self.load_training_state(initial_state)
                state, completed = snapshot(self.trainer.model), 0
                self.publish(state, 0, 0)
                self.checkpoint(state, 0)
            for rnd in range(completed + 1, last + 1):
                state = self.train_phase(state, rnd, 'B')
                require_finite(**{k: state[k] for k in self.keys})
                self.load_training_state(state)
                train_only(self.trainer.model, 'B')
                self.publish(state, rnd, rnd)
                self.checkpoint(state, rnd)

    return JointAggregationRuntime
