"""Isolated seed42 TailRW/Cover-CP runtimes; the frozen SFRA defaults are untouched."""
import time

import numpy as np
import torch

from tools.sfra.simple_controls import (SCHEMA, CONTRACT, GAMMA, control_config, implementation_variant,
    load_json, partition_counts, read_csv, static_tables, tail_weights, write_table)
from utils.cliplora_a_refresh import append_rows
from utils.cliplora_bridge_audit import write_json
from utils.cliplora_functional_feedback import observational_model
from utils.cliplora_sfra import SFRARuntime
from utils.sfra_math import require_finite


class MethodASimpleControlsRuntime(SFRARuntime):
    def configure_experiment(self):
        job = load_json(self.args.method_a_simple_control_manifest)
        if job.get('contract') != CONTRACT or job.get('seed') != 42 or self.args.seed != 42:
            raise ValueError('Simple controls require the frozen seed42 contract')
        self.control_method = job['method']
        if self.variant != implementation_variant(self.control_method):
            raise ValueError('Control and SFRA implementation variant disagree')
        if self.sfra_config.get('b_transfer') or self.sfra_config.get('b_aggregation'):
            raise ValueError('Simple controls cannot mix with B transfer/aggregation controls')
        if (self.args.split_seed != 42 or self.strength != 10.
                or not getattr(self.args, 'sfra_fast_execution_v2', False)
                or self.args.sfra_feedback_batch_size != 128 or self.args.sfra_feedback_cache_gib != 4.
                or self.variant.endswith('-cp') and self.classification_strength != 1.):
            raise ValueError('Simple control training/execution contract changed')
        self.control_counts = partition_counts(read_csv(self.root/'partition_manifest.csv'), self.tail.tolist())
        if self.control_counts != job['counts'] or self.control_counts['sizes'] != self.sizes:
            raise ValueError('Actual training partition differs from the registered counts')
        self.control_weights = tail_weights(self.sizes, self.control_counts['tail_counts'], GAMMA[self.control_method])
        self.sfra_config['simple_control'] = control_config(self.control_method)
        clients, classes = static_tables(self.control_counts)
        write_table(self.root/'client_weights.csv', clients)
        write_table(self.root/'class_coverage.csv', classes)

    def phase_aggregation_weights(self, selected, rnd, factor, extra=False, branch='main'):
        if (sorted(selected) != list(range(30)) or branch != 'main'
                or not (factor == 'B' and not extra and 1 <= rnd <= 100
                        or factor == 'A' and extra and 1 <= rnd <= 90)):
            raise ValueError('Unexpected phase in fixed simple-control schedule')
        weights = (dict(enumerate(self.control_weights)) if self.control_method.startswith('tailrw-')
                   else super().phase_aggregation_weights(selected, rnd, factor, extra, branch))
        write_table(self.root/'aggregation_weights'/f'r{rnd:03d}_{factor}.csv',
            [dict(round=rnd, factor=factor, client_id=j, original_weight=self.q[j], weight=weights[j])
             for j in sorted(selected)])
        return weights

    def build_refresh_targets(self, scores, source, history, valid, rnd):
        targets = super().build_refresh_targets(scores, source, history, valid, rnd)
        if self.control_method == 'cover-cp':
            coverage = scores.new_tensor([self.control_counts['coverage'][q['class_id']] for q in self.bank.tokens])
            if (coverage < 1).any():
                raise ValueError('Nonempty functional token has zero class coverage')
            raw = coverage.reciprocal()
            raw[~targets['active']] = 0.
            targets['weights'] = raw/raw.sum() if targets['active'].any() else raw
        return targets

    def refresh(self, middle, rnd, folder, prepared=None):
        state, scores, arrays, summary = super().refresh(middle, rnd, folder, prepared)
        coverage = torch.tensor([self.control_counts['coverage'][q['class_id']] for q in self.bank.tokens])
        arrays['class_coverage'] = coverage
        write_table(self.root/'functional_priority'/f'r{rnd:03d}.csv',
            [dict(round=rnd, token_id=i, client_id=q['client_id'], class_id=q['class_id'],
                  holders=int(coverage[i]), active=bool(arrays['active'][i]),
                  source_rho=float(self.history.source_cache[i]),
                  actual_rho=(1./int(coverage[i]) if self.control_method == 'cover-cp'
                              else float(self.history.source_cache[i])),
                  weight=float(arrays['weights'][i]),
                  history_only=bool(arrays['history_valid_before'][i] and not arrays['supported'][i]))
             for i, q in enumerate(self.bank.tokens)])
        return state, scores, arrays, summary

    def publish(self, state, rnd, decision_round):
        """One observational official pass, including sample IDs; never used for training."""
        started = time.perf_counter()
        trainer = self.global_trainer
        if not isinstance(trainer.test_loader.sampler, torch.utils.data.SequentialSampler):
            raise ValueError('Per-sample evaluation requires fixed SequentialSampler')
        labels, predictions = [], []
        with observational_model(trainer.model):
            trainer.model.load_state_dict(state, strict=True)
            for batch in trainer.test_loader:
                images, target = trainer.parse_batch_test(batch)
                logits = trainer.model_inference(images)
                require_finite(logits=logits)
                labels.append(target.detach().cpu().numpy())
                predictions.append(logits.argmax(1).detach().cpu().numpy())
        labels, predictions = np.concatenate(labels), np.concatenate(predictions)
        expected = np.array([int(x.label) for x in trainer.dm.dataset.data_test])
        if len(labels) != 10000 or not np.array_equal(labels, expected) or not np.all(np.bincount(labels, minlength=100) == 100):
            raise ValueError('Official test identities/order changed')
        correct = labels == predictions
        path = self.root/'predictions'/f'r{rnd:03d}.npz'
        path.parent.mkdir(parents=True, exist_ok=True)
        temp = path.with_suffix('.tmp')
        with temp.open('wb') as stream:
            np.savez_compressed(stream, sample_id=np.arange(len(labels)), class_id=labels,
                                prediction=predictions, correct=correct)
        temp.replace(path)
        per_class = np.bincount(labels, weights=correct, minlength=100)/np.bincount(labels, minlength=100)*100
        non_tail = np.setdiff1d(np.arange(100), self.tail)
        metrics = dict(overall_acc=float(correct.mean()*100), head20_acc=float(per_class[self.head].mean()),
            middle60_acc=float(per_class[self.middle].mean()), bottom20_tail_acc=float(per_class[self.tail].mean()),
            non_tail_acc=float(per_class[non_tail].mean()), macro_per_class_acc=float(per_class.mean()))
        append_rows(self.root/'round_metrics.csv', [dict(epoch=rnd-1, round=rnd, decision_round=decision_round,
            method=self.control_method, partition=self.args.partition, seed=42, **metrics)])
        write_table(self.root/f'per_class_accuracy_epoch_{rnd-1}.csv',
                    [dict(class_id=c, per_class_acc=float(per_class[c])) for c in range(100)])
        self.evaluations.append(dict(label=f'official_round_{rnd}', kind='official_test',
            seconds=time.perf_counter()-started, sample_presentations=len(labels), upload_bytes=0, modeled_downlink_bytes=0))
        print(f'{self.control_method} round={rnd}: Overall={metrics["overall_acc"]:.3f} '
              f'Tail20={metrics["bottom20_tail_acc"]:.3f}', flush=True)

    def flush_records(self, completed):
        super().flush_records(completed)
        for name in ('progress.json', 'completion.json') if completed == 100 else ('progress.json',):
            value = load_json(self.root/name)
            value.update(experiment=self.control_method, experiment_schema=SCHEMA,
                         gamma=GAMMA[self.control_method])
            device = self.trainer.device
            peak = (int(torch.cuda.max_memory_allocated(device)) if torch.cuda.is_available()
                    and torch.device(device).type == 'cuda' else None)
            previous = load_json(self.root/'resource_usage.json') if (self.root/'resource_usage.json').is_file() else {}
            if peak is not None:
                peak = max(peak, previous.get('peak_allocated_bytes') or 0)
            write_json(self.root/'resource_usage.json', dict(peak_allocated_bytes=peak,
                scope='process CUDA allocated high-water mark; includes initialization/evaluation',
                device=str(device)))
            write_json(self.root/name, value)
