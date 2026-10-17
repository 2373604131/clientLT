"""Frozen joint + old Full-CP A + old shared-C B, using original method code.

Only the CIFAR-specific base initialization, image wrapper, ordinary client loop
and observational reporting are adapted. SFRARuntime.refresh/run, WitnessBank,
FunctionalHistory and SharedDonorBTransfer keep their existing implementations.
"""
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import time
import numpy as np
import torch

from tools.benchmarks.common import read_json, write_json, write_csv, digest
from tools.multidataset.data import ImageDataset
from tools.multidataset.model import evaluate
from tools.multidataset.protocol import metrics
from utils.cliplora_a_refresh import lora_keys, train_only, isolated_rng, aggregate_refresh_deltas, append_rows
from utils.cliplora_functional_feedback import snapshot
from utils.cliplora_la_control import LAControlRuntime
from utils.cliplora_sfra import SFRARuntime
from utils.lora_aggregation import aggregate_lora_state


class DatasetBaseRuntime(LAControlRuntime):
    """Alternative superclass init reached by SFRARuntime's cooperative super()."""
    def __init__(self, trainer, global_trainer, cfg, args, schedule, normal_train):
        self.trainer, self.global_trainer, self.cfg, self.args = trainer, global_trainer, cfg, args
        self.schedule, self.normal_train = schedule, normal_train
        self.root = Path(args.output_dir)
        self.method, self.control, self.periodic, self.normal_factor = 's', False, True, 'B'
        self.tau, self.h, self.eps_hist = 1., 0, 0.
        self.rounds = list(range(1, 91))
        self.a_keys, self.b_keys = lora_keys(trainer.model, 'A'), lora_keys(trainer.model, 'B')
        self.keys = sorted(self.a_keys + self.b_keys)
        self.scaling = .5
        self.base_state = snapshot(trainer.model)
        self.initial = {k: self.base_state[k].clone() for k in self.keys}
        self.failure_count = self.accepted = self.attempts = 0
        self.frozen, self.freeze_round = False, None
        self.best_valid_anchor = self.root_reference = None
        self.events, self.budget, self.evaluations, self.decisions = [], [], [], []
        self.started = time.perf_counter()
        matrix = torch.tensor(self.job['protocol']['matrix'], dtype=torch.long)
        self.audit = SimpleNamespace(counts=matrix, sizes=matrix.sum(1), meta={})
        self.sizes = matrix.sum(1).tolist()
        self.q = {c: size / sum(self.sizes) for c, size in enumerate(self.sizes)}
        group = self.job['protocol']['groups']
        self.tail, self.head, self.middle = [np.array(group[k], dtype=int) for k in ('tail_ids', 'head_ids', 'middle_ids')]
        self.prior = matrix.sum(0).double() / matrix.sum()
        trainer.training_logit_adjustment = self.prior.log().float().to(trainer.device)
        self.steps_per_epoch = sum((n+31)//32 for n in self.sizes)
        self.config = dict(schema_version='multidataset_old_ab_base_v1', method='s',
            seed=args.seed, protocol_seed=42, topology='client-longtail', la_tau=1., a_lr_mult=1.,
            extra_a_lr=.001, extra_b_lr=.001, normal_trainable_factor='B',
            normal_b_lr=float(cfg.OPTIM.LR), normal_b_weight_decay=float(cfg.OPTIM.WEIGHT_DECAY),
            extra_weight_decay=0., aggregation='joint_class_weighted', control_enabled=False,
            extra_trainable_factor='A', candidate_rounds=self.rounds,
            head_ids=self.head.tolist(), middle_ids=self.middle.tolist(), tail_ids=self.tail.tolist(),
            steps_per_epoch=self.steps_per_epoch, normal_steps_expected=100*3*self.steps_per_epoch,
            extra_steps_expected=90*self.steps_per_epoch,
            primary_endpoint='last20_committed_logical_rounds', protocol_sha256=digest(self.job['protocol']))
        write_json(self.root / 'control_config.json', self.config)
        (self.root / 'checkpoints').mkdir(parents=True, exist_ok=True)
        if not args.sfra_resume:
            torch.save(self.base_state, self.root / 'checkpoints/base_model.pt')


class FixedOldAB(SFRARuntime, DatasetBaseRuntime):
    def __init__(self, job, trainer, cfg, test_loader, resume, stop_after):
        self.job, self.test_loader = job, test_loader
        root = Path(job['run_root'])
        rows = read_json(Path(job['protocol_root']) / 'train.json')
        write_csv(root / 'partition_manifest.csv', [dict(raw_sample_id=r['raw_sample_id'], class_id=r['class_id'],
            client_id=r['client_id'], local_position=r['local_position']) for r in rows])
        last = root / 'checkpoints/sfra_last.pt'
        args = SimpleNamespace(output_dir=str(root), seed=job['seed'], split_seed=42,
            partition='client-longtail', lac_method='s', sfra_variant='full-cp',
            sfra_retention_weight=10., sfra_classification_weight=1., sfra_witness_batch_size=8,
            sfra_b_aggregation='sample', b_transfer_enable=True, b_transfer_mode='shared',
            b_transfer_probe_step=.1, b_transfer_lr=.3, b_transfer_reg=.001,
            b_transfer_non_tail_sampling='class-cyclic', b_transfer_tail_weight=.35,
            sfra_resume=str(last) if resume and last.exists() else '',
            sfra_stop_after_round=2 if job['smoke'] else stop_after,
            sfra_fast_execution_v2=True, sfra_feedback_batch_size=128, sfra_feedback_cache_gib=4.)
        # WitnessBank's historical image wrapper was hard-coded to CIFAR.
        # This process-local substitution only changes image decoding/transforms.
        from importlib import import_module
        manager = import_module('Dassl.dassl.data.data_manager')
        with patch.object(manager, 'DatasetCifar100', ImageDataset):
            super().__init__(trainer, trainer, cfg, args, job['protocol']['schedule'], None)
        from tools.multidataset.feedback import BoundedWitnessFeedback
        self.bank._fast_feedback = BoundedWitnessFeedback(self.bank)

    def configure_experiment(self):
        self.sfra_config['joint_aggregation'] = dict(strength=.1,
            weights=self.job['protocol']['joint_solution']['weights'])
        self.sfra_config['b_transfer'].update(
            aggregation='ordinary_joint_class_weighted_B_then_one_shared_residual',
            calibration_anchor='actual ordinary joint-class-weighted B with round-start A')
        if self.job['smoke']:
            self.sfra_config['b_transfer']['rounds'] = [1]
        self.sfra_config['multidataset_job_id'] = digest(self.job)
        self.sfra_config['cache_limits_gib'] = dict(images=.5, cpu_prefix=4., cuda_prefix=4.)

    def train_phase(self, state, rnd, factor='B', extra=False, branch='main', candidate=0, return_deltas=False):
        from trainers.cliplora import build_cliplora_optimizer_and_scheduler, cliplora_optimizer_step
        if (factor, extra) not in (('B', False), ('A', True)) or branch != 'main':
            raise ValueError('Frozen method permits only ordinary B and old-A refresh phases')
        started = time.perf_counter()
        phase = 'refresh_A' if extra else 'normal_B'
        event_id = f'r{rnd:03d}_{phase}'
        selected = self.schedule[rnd-1]
        keys = self.a_keys if extra else self.b_keys
        local, deltas = {}, {}
        for position, client in enumerate(selected, 1):
            write_json(self.root / 'progress.json', dict(completed_round=rnd-1, active_round=rnd,
                phase=phase, client_position=position, clients_this_round=len(selected), client_id=client,
                total_rounds=2 if self.job['smoke'] else 100))
            self.load_training_state(state)
            train_only(self.trainer.model, factor)
            if extra:
                optimizer = torch.optim.SGD([p for p in self.trainer.model.parameters() if p.requires_grad],
                    lr=.001, momentum=.9, weight_decay=0.)
                scheduler = None
            else:
                optimizer, scheduler = build_cliplora_optimizer_and_scheduler(self.trainer.model, self.cfg)
            steps = samples = epochs = 0
            with isolated_rng(self.args.seed + 1000003*rnd + 1009*client):
                self.trainer.model.train()
                for epoch in range(1 if extra or self.job['smoke'] else 3):
                    for batch in self.trainer.fed_train_loader_x_dict[client]:
                        images, labels = self.trainer.parse_batch_train(batch)
                        cliplora_optimizer_step(self.trainer.model, optimizer, None, 'fp32', images, labels,
                            logit_adjustment=self.trainer.training_logit_adjustment)
                        steps += 1
                        samples += labels.numel()
                        if self.job['smoke']:
                            break
                    if scheduler is not None:
                        scheduler.step()
                    epochs += 1
            local[client] = {k: self.trainer.model.state_dict()[k].detach().cpu().clone() for k in keys}
            deltas[client] = {k: local[client][k] - state[k] for k in keys}
            self.budget.append(dict(event_id=event_id, round=rnd, phase=phase, client_id=client,
                optimizer_steps=steps, scheduler_steps=epochs if scheduler else 0, sample_presentations=samples,
                a_optimizer_steps=steps if extra else 0, b_optimizer_steps=0 if extra else steps,
                upload_bytes=sum(v.numel()*v.element_size() for v in local[client].values()),
                modeled_downlink_bytes=sum(state[k].numel()*state[k].element_size() for k in self.keys)))
            print(f'Round {rnd} {phase} client {position}/{len(selected)} id={client} steps={steps}', flush=True)
        weights = dict(enumerate(self.job['protocol']['joint_solution']['weights']))
        after = (aggregate_refresh_deltas(state, deltas, weights, keys) if extra
                 else aggregate_lora_state(state, local, selected, keys, weights))
        self.events.append(dict(event_id=event_id, round=rnd, phase=phase, committed=True,
            seconds=time.perf_counter()-started, server_weights=[weights[c] for c in selected]))
        if not extra and self.b_transfer.scheduled(rnd):
            self.b_transfer.cache_updates(deltas, selected)
            after = self.b_transfer.apply_shared(state, after, rnd)
        self.load_training_state(after)
        train_only(self.trainer.model, 'B')
        return (after, deltas) if return_deltas else after

    def publish(self, state, rnd, decision_round):
        self.load_training_state(state)
        started = time.perf_counter()
        correct, total = evaluate(self.trainer.model, self.test_loader, self.trainer.device, self.job['protocol']['classes'])
        values = metrics(correct, total, self.job['protocol']['groups'])
        write_json(self.root / 'rounds' / f'r{rnd:03d}.json', dict(round=rnd, job_id=digest(self.job),
            correct=correct, total=total, metrics=values))
        append_rows(self.root / 'round_metrics.csv', [dict(round=rnd, **values)])
        self.evaluations.append(dict(round=rnd, kind='official_test', seconds=time.perf_counter()-started,
            sample_presentations=sum(total)))
        print(f'COMMITTED evaluation {rnd}: joint + old A + old B {values}', flush=True)

    def checkpoint(self, state, completed):
        from tools.multidataset.runner import verify_source
        verify_source(self.job)
        super().checkpoint(state, completed)
        if completed == (2 if self.job['smoke'] else 100):
            write_json(self.root / 'completion.json', dict(job_id=digest(self.job),
                completed_round=completed, smoke=self.job['smoke']))


def train(job, model, cfg, loaders, test_loader, clients, resume=False, stop_after=0):
    device = next(model.parameters()).device
    dataset = SimpleNamespace(classnames=job['protocol']['classnames'], federated_train_x=clients)
    trainer = SimpleNamespace(model=model, device=device, dm=SimpleNamespace(dataset=dataset),
        fed_train_loader_x_dict=loaders,
        parse_batch_train=lambda batch: (batch['img'].to(device), batch['label'].to(device)))
    runtime = FixedOldAB(job, trainer, cfg, test_loader, resume, stop_after)
    runtime.run(runtime.base_state)
