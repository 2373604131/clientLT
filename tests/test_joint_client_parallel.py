"""CPU checks for queue tails, independent LoRA, exact batches and update parity."""
import ast
import copy
from pathlib import Path
from types import SimpleNamespace
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from scripts import run_cliplora_joint_aggregation as launcher
from scripts.run_ab_validation import write_json
from tools.client_aggregation.parallel import (ordered_queue, plan_phase, clone_models,
    train_local, compare_states, plan_batches)
from tools.client_aggregation.runtime import runtime_class
from utils.cliplora_a_refresh import isolated_rng, train_only
from utils.cliplora_functional_feedback import snapshot
from utils.cliplora_loss import fixed_denominator_cross_entropy
from utils.lora_aggregation import aggregate_lora_state
from utils.loralib.layers import LinearLoRA

REPO = Path(__file__).resolve().parents[1]


def canonical_step():
    # Execute the actual pure step definition without importing registry/GPU data dependencies.
    source = ast.parse((REPO/'trainers/cliplora.py').read_text(encoding='utf-8'))
    node = next(n for n in source.body if isinstance(n, ast.FunctionDef) and n.name == 'cliplora_optimizer_step')
    namespace = dict(torch=torch, fixed_denominator_cross_entropy=fixed_denominator_cross_entropy)
    exec(compile(ast.Module(body=[node], type_ignores=[]), 'trainers/cliplora.py', 'exec'), namespace)
    return namespace['cliplora_optimizer_step']


class Data(Dataset):
    def __init__(self, count, client=0):
        self.count, self.client = count, client

    def __len__(self):
        return self.count

    def __getitem__(self, i):
        return dict(img=torch.tensor([(i+1)/10., (self.client+1)/10., -.2]), label=(i+self.client)%2, index=i)


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = LinearLoRA(nn.Linear(3, 2), r=2, lora_alpha=1, dropout_rate=0.)
        self.linear._sfra_low_rank_execution = True
        with torch.no_grad():
            self.linear.w_lora_B.fill_(.1)
        train_only(self, 'B')

    def forward(self, x):
        return self.linear(x)


class TestParallelClients(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def test_work_queue_accepts_remainders_and_returns_schedule_order(self):
        for slots in (4, 6, 8):
            for count in (0, 1, 2, 3, 4, 5, 6, 7, 8, 30, 31, 35):
                seen, lock = [], threading.Lock()
                def work(slot, client):
                    time.sleep(.001*(client%3))
                    with lock:
                        seen.append(client)
                    self.assertLess(slot, slots)
                    return client*2
                result = ordered_queue(list(range(count)), slots, work)
                self.assertEqual(list(result), list(range(count)))
                self.assertEqual(sorted(seen), list(range(count)))
                self.assertEqual(list(result.values()), [c*2 for c in range(count)])

    def test_failure_is_propagated_without_partial_aggregation(self):
        def work(slot, client):
            if client == 4:
                raise RuntimeError('client failure')
            return client
        with self.assertRaisesRegex(RuntimeError, 'client failure'):
            ordered_queue(range(6), 4, work)
        with self.assertRaises(ValueError):
            ordered_queue([0, 0], 4, work)

    def test_B_plans_match_original_loader_orders_and_rng(self):
        loaders = {i: DataLoader(Data(n, i), batch_size=4, shuffle=True, drop_last=False)
                   for i, n in enumerate((1, 4, 6, 9, 2, 5))}
        torch.manual_seed(29)
        before = torch.get_rng_state().clone()
        expected = {c: [[batch['index'].tolist() for batch in loaders[c]] for _ in range(3)] for c in loaders}
        expected_rng = torch.get_rng_state().clone()
        torch.set_rng_state(before)
        actual = plan_phase(loaders, list(loaders), 'B', 42, 1)
        self.assertEqual(actual, expected)
        self.assertTrue(torch.equal(expected_rng, torch.get_rng_state()))

    def test_A_plans_reproduce_existing_isolated_seed_and_preserve_rng(self):
        loaders = {i: DataLoader(Data(7+i), batch_size=4, shuffle=True, drop_last=False) for i in range(3)}
        before = torch.get_rng_state().clone()
        actual = plan_phase(loaders, list(loaders), 'A', 42, 50)
        self.assertTrue(torch.equal(before, torch.get_rng_state()))
        for c in loaders:
            with isolated_rng(42+1000003*50+1009*c):
                expected = [[batch['index'].tolist() for batch in loaders[c]]]
            self.assertEqual(actual[c], expected)

    def test_private_A_even_when_frozen_and_backbone_shared(self):
        model = Model()
        keys = [n for n, _ in model.named_parameters() if '_lora_' in n]
        replicas = clone_models(model, keys, 8)
        for name, p in model.named_parameters():
            pointers = [dict(m.named_parameters())[name].data_ptr() for m in replicas]
            if name in keys:
                self.assertEqual(len(set([p.data_ptr()]+pointers)), 9)
            else:
                self.assertEqual(set(pointers), {p.data_ptr()})
        model.dropout = nn.Dropout(.1)
        with self.assertRaisesRegex(ValueError, 'dropout'):
            clone_models(model, keys, 4)

    def test_parallel_and_serial_match_canonical_steps_for_A_B_and_short_queue(self):
        torch.manual_seed(1)
        model = Model()
        keys = sorted(n for n, _ in model.named_parameters() if '_lora_' in n)
        replicas = clone_models(model, keys, 8)
        initial = snapshot(model)
        sizes = (2, 6, 9, 3, 1, 7, 5, 8)
        loaders = {i: DataLoader(Data(sizes[i % len(sizes)], i), batch_size=4, shuffle=True, drop_last=False)
                   for i in range(30)}
        step = canonical_step()
        def optimizer(m, factor):
            opt = torch.optim.SGD([p for p in m.parameters() if p.requires_grad],
                                  lr=.01, momentum=.9, weight_decay=.0005 if factor=='B' else 0.)
            return opt, torch.optim.lr_scheduler.StepLR(opt, 3, gamma=1.) if factor=='B' else None
        for factor in ('B', 'A'):
            plans = plan_phase(loaders, list(loaders), factor, 42, 1)
            reference = {}
            for c in loaders:
                # Independent copy and explicit original-style step loop.
                m = copy.deepcopy(model);m.load_state_dict(initial);train_only(m, factor);m.train()
                opt, sched = optimizer(m, factor)
                for epoch in plans[c]:
                    for indices in epoch:
                        batch = torch.utils.data.default_collate([loaders[c].dataset[i] for i in indices])
                        step(m, opt, None, 'fp32', batch['img'], batch['label'], logit_adjustment=torch.tensor([-.2, -.4]))
                    if sched is not None:
                        sched.step()
                reference[c] = {k: v.detach().clone() for k, v in m.state_dict().items() if k.endswith('_lora_'+factor)}
            before_rng = torch.get_rng_state().clone()
            def work(slot, c):
                return train_local(replicas[slot], initial, keys, loaders[c].dataset, plans[c], factor,
                                   'cpu', optimizer, step, torch.tensor([-.2, -.4]))
            for slots in (4, 6, 8):
                for clients in (list(loaders), [4, 5], [5], list(range(5)), list(range(6))):
                    results = ordered_queue(clients, slots, work)
                    for c in clients:
                        self.assertTrue(compare_states(reference[c], results[c][0])['bitwise_equal'])
                        self.assertEqual(results[c][1]['sample_presentations'], len(loaders[c].dataset)*(3 if factor=='B' else 1))
            self.assertTrue(torch.equal(before_rng, torch.get_rng_state()))
            for key, value in model.state_dict().items():
                self.assertTrue(torch.equal(value, initial[key]))
            selected = list(loaders)
            weights = {c:(c+1)/sum(j+1 for j in selected) for c in selected}
            actual = aggregate_lora_state(initial, {c:work(0,c)[0] for c in selected}, selected, list(reference[0]), weights)
            expected = aggregate_lora_state(initial, reference, selected, list(reference[0]), weights)
            self.assertTrue(compare_states(expected, actual)['bitwise_equal'])

    def test_server_gpu_mapping_and_smoke_before_run(self):
        with tempfile.TemporaryDirectory() as temp, patch('builtins.print'):
            args = launcher.parse_args(['--stage','server','--output-root',temp])
            self.assertEqual(args.gpus, [2, 3]);self.assertEqual(args.client_concurrency, 4)
            calls, lock = [], threading.Lock()
            def run(command, **kwargs):
                with lock:
                    calls.append((command, kwargs['env']))
                return type('Result', (), dict(returncode=0))()
            with patch.object(launcher.subprocess, 'run', side_effect=run):
                launcher.run_server(args)
            for arm, gpu in (('frozen', '2'), ('ab', '3')):
                arm_calls = [(cmd, env) for cmd, env in calls if cmd[cmd.index('--arms')+1] == arm]
                self.assertEqual([cmd[cmd.index('--stage')+1] for cmd, _ in arm_calls], ['smoke', 'run'])
                self.assertTrue(all(env['CUDA_VISIBLE_DEVICES'] == gpu for _, env in arm_calls))

    def test_parallel_runtime_aggregates_all_clients_once_before_shared_transfer(self):
        with tempfile.TemporaryDirectory() as temp, patch('builtins.print'):
            rt = object.__new__(runtime_class(dict(arm='ab', mode='formal')))
            rt.root = Path(temp);rt.job = dict(arm='ab', mode='formal');rt.is_frozen = False
            rt.trainer = SimpleNamespace(model=Model());rt.args = SimpleNamespace(seed=42)
            rt.keys = sorted(n for n, _ in rt.trainer.model.named_parameters() if '_lora_' in n)
            rt.a_keys = [k for k in rt.keys if k.endswith('_A')];rt.b_keys = [k for k in rt.keys if k.endswith('_B')]
            rt.q = {c:1/30 for c in range(30)};rt.joint_weights = {c:(c+1)/465 for c in range(30)}
            rt.schedule = [list(reversed(range(30)))]*100;rt.events = [];rt.budget = []
            rt.sfra_config = dict(b_transfer=dict(mode='shared'));rt.stage_evaluate = lambda *a: None
            start = snapshot(rt.trainer.model);transfer_calls = [];cached = []
            def shared(before, ordinary, rnd):
                self.assertEqual(before.keys(), start.keys())
                for k in rt.keys:
                    self.assertTrue(torch.equal(rt.trainer.model.state_dict()[k], ordinary[k]))
                transfer_calls.append(rnd)
                return dict(ordinary, **{k:ordinary[k]+.2 for k in rt.b_keys})
            rt.b_transfer = SimpleNamespace(scheduled=lambda r: r==30, apply_shared=shared,
                cache_updates=lambda d,s: cached.append((d,s)))
            local_calls = []
            def local_run(state, clients, plans, factor):
                local_calls.append((list(clients), factor))
                keys = rt.a_keys if factor=='A' else rt.b_keys
                results = {c:({k:state[k]+(c+1)*.001 for k in keys},
                    dict(optimizer_steps=len(plans[c]),sample_presentations=len(plans[c]),
                         scheduler_steps=0 if factor=='A' else len(plans[c])),
                    dict(client_id=c,slot=c%4,batch_plan_sha256='test')) for c in clients}
                return results, dict(slots=4,clients=len(clients))
            rt.client_pool = SimpleNamespace(loaders={c:DataLoader(Data(1,c),shuffle=True,batch_size=32) for c in range(30)},run=local_run)
            audit = SimpleNamespace(event_context={})
            def save(before, after, updates, selected, weights, rnd, factor=None):
                self.assertEqual(len(updates),30)
                self.assertEqual(selected,rt.schedule[rnd-1])
                write_json(rt.root/'events'/audit.event_context['event_id']/'event.json',{})
            audit.normal = save
            audit.save = lambda before,after,deltas,selected,weights,rnd,phase:save(before,after,deltas,selected,weights,rnd)
            rt.audit = audit
            committed, deltas = rt.train_phase(start,30,'B',return_deltas=True)
            ordinary = aggregate_lora_state(start,{c:{k:start[k]+(c+1)*.001 for k in rt.b_keys} for c in rt.schedule[29]},
                                           rt.schedule[29],rt.b_keys,rt.joint_weights)
            for k in rt.b_keys:
                torch.testing.assert_close(committed[k],ordinary[k]+.2,rtol=0,atol=0)
            self.assertEqual(transfer_calls,[30]);self.assertEqual(len(cached),1);self.assertEqual(len(rt.budget),30)
            after_A = rt.train_phase(committed,30,'A',True)
            self.assertEqual(transfer_calls,[30]);self.assertEqual(len(rt.budget),60)
            for k in rt.b_keys:
                self.assertTrue(torch.equal(after_A[k],committed[k]))
            self.assertEqual(len(rt.events),2)
            # Smoke must perform exactly one ordinary phase; there is no benchmark method.
            rt.job['mode'] = 'smoke'
            rt.train_phase(after_A, 1, 'B')
            self.assertEqual(len(local_calls), 3)
            self.assertEqual(local_calls[-1], (rt.schedule[0], 'B'))
            self.assertEqual(len(rt.budget), 90)
            self.assertFalse((rt.root/'parallel_benchmark').exists())


if __name__ == '__main__':
    unittest.main()
