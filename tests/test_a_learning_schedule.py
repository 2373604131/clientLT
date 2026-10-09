"""CPU tests for the actual paired training engine, continuation graph and reports."""
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from tools.a_learning_schedule.protocol import (
    ARMS, PRIMARY, actions, check_source, final_nodes, paired_probes, phase_seed, plan,
    read_csv, read_json, source_event, write_csv, write_json,
)
from tools.a_learning_schedule.runtime import Worker, aggregate, effective_norm, prediction_metrics, load_torch
from tools.a_learning_schedule.summary import primary_means, summarize
from utils.cliplora_a_refresh import state_hash


class ProtocolTests(unittest.TestCase):
    def test_equal_budgets_and_intervention_times(self):
        expected = {'BB': [(0, 'B'), (5, 'B')], 'AB': [(0, 'A'), (5, 'B')],
                    'AA_short': [(0, 'A'), (1, 'A')], 'AA_long': [(0, 'A'), (5, 'A')]}
        for arm in ARMS:
            seq = actions(arm)
            self.assertEqual([(h, f) for h, k, f in seq if k == 'extra'], expected[arm])
            self.assertEqual(sum(3 if k == 'normal' else 1 for _, k, _ in seq), 32)
            self.assertEqual([h for h, k, _ in seq if k == 'normal'], list(range(1, 11)))
            self.assertEqual(sorted(final_nodes(arm)), list(range(11)))
        self.assertEqual(final_nodes('AA_short')[1]['kind'], 'extra')
        self.assertEqual(final_nodes('AA_long')[5]['kind'], 'extra')

    def test_shared_parents_and_phase_rng(self):
        for a, b in paired_probes():
            self.assertEqual(a['parent'], b['parent'])
            self.assertEqual(phase_seed(42, 20, a, 7), phase_seed(42, 20, b, 7))
        short = next(n for n in plan('AA_short') if n['h'] == 1 and n['kind'] == 'extra')
        long = next(n for n in plan('AA_long') if n['h'] == 5 and n['kind'] == 'extra')
        self.assertEqual(phase_seed(42, 20, short, 0), phase_seed(42, 20, long, 0))
        nodes = {n['id']: n for arm in ARMS for n in plan(arm)}
        self.assertEqual(len(nodes), 40)
        self.assertEqual(sum(n['kind'] == 'normal' for n in nodes.values()), 34)

    def test_aggregations_preserve_inactive_factor(self):
        before = {'x_lora_A': torch.tensor([[1., 2.]]), 'x_lora_B': torch.tensor([[3.], [4.]])}
        local = {0: {'x_lora_A': before['x_lora_A'] + 2}, 1: {'x_lora_A': before['x_lora_A'] - 1}}
        for kind in ('normal', 'extra'):
            after = aggregate(before, local, [1, 0], [.25, .75], ['x_lora_A'], kind)
            torch.testing.assert_close(after['x_lora_A'], before['x_lora_A'] - .25)
            self.assertTrue(torch.equal(after['x_lora_B'], before['x_lora_B']))
            expected = torch.linalg.vector_norm(.5 * before['x_lora_B'].double() @ (after['x_lora_A'] - before['x_lora_A']).double())
            self.assertAlmostEqual(effective_norm(before, after, ['x_lora_A'], {'x_lora_A': .5}), float(expected), places=12)

    def test_learning_and_forgetting_are_not_confused(self):
        labels = np.array([0, 0, 1, 1])
        anchor = dict(labels=labels, prediction=np.array([0, 1, 1, 0]))
        current = dict(labels=labels, prediction=np.array([1, 0, 1, 1]),
                       ce=np.ones(4), la=np.ones(4), margin=np.zeros(4))
        result = prediction_metrics(current, {'overall': [0, 1]}, anchor)
        self.assertEqual(result['overall'], 75)
        self.assertEqual(result['overall_new_correct'], 2)
        self.assertEqual(result['overall_forgotten'], 1)

    def test_primary_endpoint_does_not_pick_peak(self):
        rows = [dict(arm=arm, h=h, test_tail20=100 if h == 1 else h) for arm in ARMS for h in range(11)]
        result = primary_means(rows)
        self.assertEqual(result['AA_long']['test_tail20'], 9)
        with self.assertRaises(ValueError):
            primary_means(rows[:-1])

    def test_preflight_rejects_lightweight_archive(self):
        root = Path(__file__).resolve().parents[1] / 'output/la_control_js_topology_analysis/la_control/seed42/client-longtail/e2/tau1_a1_protocol42'
        if not root.exists():
            self.skipTest('Local reference archive unavailable')
        source = check_source(root, 'e2', 42, [20, 50, 80], require_weights=False)
        self.assertEqual(sum(source['sizes']), 10847)
        if source['missing_weights']:
            with self.assertRaisesRegex(FileNotFoundError, 'SERVER checkpoints'):
                check_source(root, 'e2', 42, [20, 50, 80])
        self.assertIn('extra_B', str(source_event(root, 'e2', 20)))
        with self.assertRaises(ValueError):
            check_source(root, 'e3', 42, [20], require_weights=False)

    def test_launcher_preserves_environment_python_and_one_anchor_arguments(self):
        import sys
        from scripts.run_cliplora_a_learning_schedule import parse_args, child_command
        args = parse_args(['--stage', 'run', '--gpus', '0', '1', '2', '3', '4', '5'])
        command = child_command(args, dict(origin='e3', anchor_round=50, source_run='/source/e3'), 'frozen')
        self.assertEqual(command[0], sys.executable)
        self.assertEqual(command[command.index('--origin') + 1], 'e3')
        self.assertEqual(command[command.index('--anchor-round') + 1], '50')
        self.assertEqual(command[command.index('--worker-mode') + 1], 'run')
        self.assertNotIn('--gpus', command)
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                parse_args(['--gpus', '0', '0'])

    def test_six_gpu_launcher_assigns_one_anchor_per_gpu(self):
        from scripts.run_cliplora_a_learning_schedule import parse_args, launch
        jobs = [dict(origin=origin, anchor_round=rnd, source_run='/source/' + origin)
                for origin in ('e2', 'e3') for rnd in (20, 50, 80)]

        class FinishedProcess:
            pid = 123
            def poll(self):
                return 0

        with tempfile.TemporaryDirectory() as temp:
            args = parse_args(['--stage', 'run', '--gpus', '0', '1', '2', '3', '4', '5', '--output-root', temp])
            with patch('scripts.run_cliplora_a_learning_schedule.subprocess.Popen', return_value=FinishedProcess()) as popen:
                with contextlib.redirect_stdout(io.StringIO()):
                    launch(args, jobs, 'test')
            self.assertEqual(popen.call_count, 6)
            observed = [(call.kwargs['env']['CUDA_VISIBLE_DEVICES'],
                         call.args[0][call.args[0].index('--origin') + 1],
                         call.args[0][call.args[0].index('--anchor-round') + 1]) for call in popen.call_args_list]
            self.assertEqual(observed, [('0', 'e2', '20'), ('1', 'e2', '50'), ('2', 'e2', '80'),
                                        ('3', 'e3', '20'), ('4', 'e3', '50'), ('5', 'e3', '80')])


class TinyLoRA(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj_lora_A = nn.Parameter(torch.randn(2, 5) * .1)
        self.proj_lora_B = nn.Parameter(torch.randn(100, 2) * .1)
        self.base = nn.Parameter(torch.randn(100, 5) * .1, requires_grad=False)
        self.logit_scale = nn.Parameter(torch.tensor(.7), requires_grad=False)

    def forward(self, images):
        weight = self.base + .5 * (self.proj_lora_B @ self.proj_lora_A)
        return self.logit_scale.exp() * F.linear(images, weight)


def toy_worker(root):
    torch.manual_seed(123)
    worker = Worker.__new__(Worker)
    worker.root, worker.origin, worker.round = Path(root), 'e2', 20
    worker.args = SimpleNamespace(seed=42)
    worker.device = torch.device('cpu')
    worker.model = TinyLoRA()
    worker.a_keys, worker.b_keys = ['proj_lora_A'], ['proj_lora_B']
    worker.keys = sorted(worker.a_keys + worker.b_keys)
    worker.anchor = {k: worker.model.state_dict()[k].detach().clone() for k in worker.keys}
    worker.scales = {'proj_lora_A': .5}
    worker.frozen_hash = state_hash(worker.model.state_dict(), ['base', 'logit_scale'])
    worker.source = {'meta': {'schedule': [[0, 1] for _ in range(100)]}}
    worker.weights, worker.steps_epoch = [.5, .5], 2
    worker.adjustment = torch.full((100,), -np.log(100), dtype=torch.float32)
    labels = [0, 20, 80, 1, 21, 81]
    examples = [dict(img=torch.randn(5), label=y) for y in labels]
    clients = [examples[:3], examples[3:]]

    def loader(items, training):
        return torch.utils.data.DataLoader(items, batch_size=3, shuffle=training, drop_last=False)

    worker.data = dict(clients=clients, loader=loader, test=loader(examples, False),
                       feedback=loader(examples, False), local_feedback=[loader(c, False) for c in clients],
                       feedback_indices=[[0, 1, 2], [3, 4, 5]],
                       groups={'overall': list(range(100)), 'head20': list(range(20)),
                               'middle60': list(range(20, 80)), 'tail20': list(range(80, 100))})

    def step(model, optimizer, scaler, precision, images, y, logit_adjustment):
        output = model(images)
        loss = F.cross_entropy(output + logit_adjustment, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return output, loss, {}

    worker.step = step
    worker.root.mkdir(parents=True, exist_ok=True)
    write_json(worker.root / 'anchor_info.json', {'toy': True})
    return worker


class EngineTests(unittest.TestCase):
    def test_full_graph_resume_and_summary_on_cpu(self):
        old_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            with tempfile.TemporaryDirectory() as temp:
                root = Path(temp) / 'runs/seed42/e2/round020'
                worker = toy_worker(root)
                with contextlib.redirect_stdout(io.StringIO()):
                    worker.run()
                complete = read_json(root / 'complete.json')
                self.assertEqual(complete['unique_training_nodes'], 40)
                self.assertEqual(complete['actual_optimizer_steps'], 216)
                self.assertEqual(complete['logical_optimizer_steps_per_arm'], 64)
                self.assertEqual(len(read_csv(root / 'curves.csv')), 44)
                self.assertEqual(len(read_csv(root / 'norm_probes.csv')), 4)
                self.assertTrue(all(r['comparable'] == 'True' for r in read_csv(root / 'norm_probes.csv')))
                self.assertEqual(len(read_csv(root / 'local_feedback.csv')), 36)
                for a, b in paired_probes():
                    ap, bp = [load_torch(root / 'nodes' / (n['id'] + '.pt')) for n in (a, b)]
                    self.assertEqual(ap['before_sha256'], bp['before_sha256'])
                original_curves = (root / 'curves.csv').read_bytes()
                with patch.object(worker, 'train_node', side_effect=AssertionError('Resume retrained a completed node')):
                    with contextlib.redirect_stdout(io.StringIO()):
                        worker.run()
                self.assertEqual(original_curves, (root / 'curves.csv').read_bytes())
                statuses = summarize(temp, [42], ['e2'], [20], plots=False)
                self.assertEqual(statuses[0]['status'], 'complete')
                pairs = read_csv(Path(temp) / 'analysis/paired.csv')
                self.assertEqual(len(pairs), 3)
                cross = read_csv(Path(temp) / 'analysis/paired_across_seeds.csv')
                self.assertEqual(cross[0]['seed_count'], '1')
                self.assertEqual(cross[0]['tail20_std_pp'], '')
        finally:
            torch.set_num_threads(old_threads)

    def test_observation_restores_rng_and_training_mode(self):
        with tempfile.TemporaryDirectory() as temp:
            worker = toy_worker(temp)
            worker.model.train()
            before = torch.get_rng_state().clone()
            worker.evaluate_loader(worker.data['test'])
            self.assertTrue(torch.equal(before, torch.get_rng_state()))
            self.assertTrue(worker.model.training)


if __name__ == '__main__':
    unittest.main()
