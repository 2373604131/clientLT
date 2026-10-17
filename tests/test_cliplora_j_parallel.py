"""CPU-only checks for J's AB optimizer, ordered queue, and runtime integration."""
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import torch
from torch.utils.data import DataLoader, default_collate

from tests.test_joint_client_parallel import Data, Model, canonical_step
from tools.client_aggregation.parallel import (
    clone_models, local_optimizer, ordered_queue, plan_phase, train_local,
)
from utils.cliplora_a_refresh import train_only
from utils.cliplora_functional_feedback import snapshot
from utils.cliplora_la_control import LAControlRuntime
from utils.lora_aggregation import aggregate_lora_state


class TestJParallel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def model_and_runtime(self):
        torch.manual_seed(9)
        model = Model()
        # Match J's original dense forward, without SFRA execution changes.
        model.linear._sfra_low_rank_execution = False
        keys = sorted(n for n, _ in model.named_parameters() if '_lora_' in n)
        rt = SimpleNamespace(keys=keys, a_keys=[k for k in keys if k.endswith('_A')],
                             b_keys=[k for k in keys if k.endswith('_B')], cfg=object(),
                             config=dict(normal_a_lr=.003, normal_a_weight_decay=0.,
                                         normal_b_lr=.01, normal_b_weight_decay=.0005,
                                         extra_a_lr=.002))
        return model, rt

    @staticmethod
    def standard_builder(model, cfg, param_groups=None):
        opt = torch.optim.SGD(param_groups, lr=.01, momentum=.9, weight_decay=.0005)
        return opt, torch.optim.lr_scheduler.StepLR(opt, 3, gamma=1.)

    def test_AB_plans_match_three_original_epochs_and_main_rng(self):
        clients = list(reversed(range(30)))
        loaders = {c: DataLoader(Data(1 + c % 9, c), batch_size=4, shuffle=True) for c in clients}
        before = torch.get_rng_state().clone()
        expected = {c: [[b['index'].tolist() for b in loaders[c]] for _ in range(3)] for c in clients}
        after = torch.get_rng_state().clone()
        torch.set_rng_state(before)
        actual = plan_phase(loaders, clients, 'AB', 42, 20)
        self.assertEqual(actual, expected)
        self.assertTrue(torch.equal(torch.get_rng_state(), after))

    def test_AB_optimizer_keeps_distinct_factor_groups_and_A_refresh(self):
        model, rt = self.model_and_runtime()
        train_only(model, 'AB')
        opt, sched = local_optimizer(model, 'AB', rt, self.standard_builder)
        self.assertEqual([g['lr'] for g in opt.param_groups], [.003, .01])
        self.assertEqual([g['weight_decay'] for g in opt.param_groups], [0., .0005])
        for group, keys in zip(opt.param_groups, [rt.a_keys, rt.b_keys]):
            self.assertEqual({id(p) for p in group['params']},
                             {id(dict(model.named_parameters())[k]) for k in keys})
        self.assertIsNotNone(sched)
        train_only(model, 'A')
        opt, sched = local_optimizer(model, 'A', rt, self.standard_builder)
        self.assertEqual(opt.param_groups[0]['lr'], .002)
        self.assertEqual(opt.param_groups[0]['weight_decay'], 0.)
        self.assertIsNone(sched)

    def test_AB_and_A_queue_match_independent_serial_updates_for_30_clients(self):
        model, rt = self.model_and_runtime()
        initial = snapshot(model)
        replicas = clone_models(model, rt.keys, 8)
        clients = list(reversed(range(30)))
        loaders = {c: DataLoader(Data(1 + c % 9, c), batch_size=4, shuffle=True) for c in clients}
        step = canonical_step()
        adjustment = torch.tensor([-.2, -.4])
        factory = lambda m, factor: local_optimizer(m, factor, rt, self.standard_builder)
        for factor in ('AB', 'A'):
            plans = plan_phase(loaders, clients, factor, 42, 20)
            active = rt.keys if factor == 'AB' else rt.a_keys
            expected = {}
            for c in clients:
                serial = copy.deepcopy(model)
                serial.load_state_dict(initial)
                train_only(serial, factor)
                # Independent reference reproduces LAControlRuntime's groups.
                params = dict(serial.named_parameters())
                if factor == 'AB':
                    groups = [{'params': [params[k] for k in rt.a_keys], 'lr': .003, 'weight_decay': 0.},
                              {'params': [params[k] for k in rt.b_keys], 'lr': .01, 'weight_decay': .0005}]
                    opt = torch.optim.SGD(groups, lr=.01, momentum=.9, weight_decay=.0005)
                    sched = torch.optim.lr_scheduler.StepLR(opt, 3, gamma=1.)
                else:
                    opt = torch.optim.SGD([params[k] for k in rt.a_keys], lr=.002, momentum=.9, weight_decay=0.)
                    sched = None
                serial.train()
                for epoch in plans[c]:
                    for indices in epoch:
                        batch = default_collate([loaders[c].dataset[i] for i in indices])
                        step(serial, opt, None, 'fp32', batch['img'], batch['label'], logit_adjustment=adjustment)
                    if sched is not None:
                        sched.step()
                expected[c] = {k: serial.state_dict()[k].detach().clone() for k in active}
            before_rng = torch.get_rng_state().clone()
            results = ordered_queue(clients, 8, lambda slot, c: train_local(
                replicas[slot], initial, rt.keys, loaders[c].dataset, plans[c], factor,
                'cpu', factory, step, adjustment))
            self.assertEqual(list(results), clients)
            self.assertTrue(torch.equal(before_rng, torch.get_rng_state()))
            for c in clients:
                self.assertEqual(set(results[c][0]), set(active))
                for key in active:
                    torch.testing.assert_close(results[c][0][key], expected[c][key], rtol=0, atol=0)
                self.assertEqual(results[c][1]['sample_presentations'], len(loaders[c].dataset) * (3 if factor == 'AB' else 1))
                self.assertEqual(results[c][1]['scheduler_steps'], 3 if factor == 'AB' else 0)
            weights = {c: len(loaders[c].dataset) / sum(len(loader.dataset) for loader in loaders.values()) for c in clients}
            actual = aggregate_lora_state(initial, {c: results[c][0] for c in clients}, clients, active, weights)
            reference = aggregate_lora_state(initial, expected, clients, active, weights)
            for key in initial:
                torch.testing.assert_close(actual[key], reference[key], rtol=0, atol=0)
                torch.testing.assert_close(model.state_dict()[key], initial[key], rtol=0, atol=0)

    def test_J_runtime_waits_for_all_clients_and_records_both_factor_budgets(self):
        model, config = self.model_and_runtime()
        with tempfile.TemporaryDirectory() as folder:
            rt = object.__new__(LAControlRuntime)
            rt.root = Path(folder)
            rt.trainer = SimpleNamespace(model=model)
            rt.args = SimpleNamespace(seed=42)
            rt.keys, rt.a_keys, rt.b_keys = config.keys, config.a_keys, config.b_keys
            rt.q = {c: (c + 1) / 465 for c in range(30)}
            rt.schedule = [list(reversed(range(30)))] * 100
            rt.events, rt.budget = [], []
            rt.prepare_b_aggregation = Mock(side_effect=AssertionError('AB must not call the B-only hook'))
            initial = snapshot(model)
            audit_calls = []
            audit = SimpleNamespace(event_context={})

            def save(before, after, local, selected, weights, rnd, factor=None):
                self.assertEqual(selected, rt.schedule[rnd - 1])
                self.assertEqual(set(local), set(range(30)))
                self.assertEqual(len(rt.budget), 30 * (len(audit_calls) + 1))
                audit_calls.append(factor)
                path = rt.root / 'events' / audit.event_context['event_id'] / 'event.json'
                path.parent.mkdir(parents=True)
                path.write_text('{}', encoding='utf-8')

            audit.normal = save
            audit.save = lambda *a: save(*a[:-1], factor='A')
            rt.audit = audit

            def run(state, clients, plans, factor):
                self.assertEqual(clients, rt.schedule[9])
                keys = rt.keys if factor == 'AB' else rt.a_keys
                steps = 3 if factor == 'AB' else 1
                result = {c: ({k: state[k] + (c + 1) * .001 for k in keys},
                              dict(optimizer_steps=steps, scheduler_steps=3 if factor == 'AB' else 0,
                                   sample_presentations=steps),
                              dict(client_id=c, slot=c % 8, batch_plan_sha256='test')) for c in reversed(clients)}
                return result, dict(slots=8, clients=len(clients))

            rt.client_pool = SimpleNamespace(
                loaders={c: DataLoader(Data(1, c), batch_size=32, shuffle=True) for c in range(30)}, run=run)
            after = rt.train_phase(initial, 10, 'AB')
            expected_local = {c: {k: initial[k] + (c + 1) * .001 for k in rt.keys} for c in range(30)}
            expected = aggregate_lora_state(initial, expected_local, rt.schedule[9], rt.keys, rt.q)
            for key in initial:
                torch.testing.assert_close(after[key], expected[key], rtol=0, atol=0)
            self.assertEqual(audit_calls, ['AB'])
            self.assertEqual(sum(b['a_optimizer_steps'] for b in rt.budget), 90)
            self.assertEqual(sum(b['b_optimizer_steps'] for b in rt.budget), 90)
            after_a = rt.train_phase(after, 10, 'A', True)
            for key in rt.b_keys:
                torch.testing.assert_close(after_a[key], after[key], rtol=0, atol=0)
            self.assertEqual(audit_calls, ['AB', 'A'])
            self.assertEqual(sum(b['a_optimizer_steps'] for b in rt.budget), 120)
            self.assertEqual(sum(b['b_optimizer_steps'] for b in rt.budget), 90)
            execution = json.loads((rt.root / 'parallel_execution/r010_AB.json').read_text(encoding='utf-8'))
            self.assertEqual(execution['slots'], 8)
            self.assertEqual(execution['selected_client_ids'], rt.schedule[9])
            # No partial aggregation when a worker returns incomplete results.
            rt.client_pool.run = lambda *a: ({}, dict(slots=8, clients=0))
            with self.assertRaisesRegex(RuntimeError, 'aggregation is forbidden'):
                rt.train_phase(after_a, 11, 'AB')
            self.assertEqual(len(audit_calls), 2)
            self.assertEqual(len(rt.budget), 60)


if __name__ == '__main__':
    unittest.main()
