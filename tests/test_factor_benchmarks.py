"""Frozen-factor, sparse-update, SVD parity and committed-round resume tests."""
import ast
import copy
import io
from pathlib import Path
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from trainers.baselines import factor_freezing as factors
from trainers.baselines.common import trainable_state, load_trainable
from tools.benchmarks import common, runtime, collect
from scripts.run_paper_benchmarks import parser_for_cli, command_for
from tests.test_paper_benchmarks import fake_reference


class TinyLoRA(nn.Module):
    def __init__(self, out=100):
        super().__init__()
        self.register_buffer('base', torch.randn(out, 5) * .1)
        self.w_lora_A = nn.Parameter(torch.randn(4, 5) * .1)
        self.w_lora_B = nn.Parameter(torch.zeros(out, 4))
    def forward(self, x):
        return F.linear(x, self.base) + .5 * F.linear(F.linear(x, self.w_lora_A), self.w_lora_B)


def options():
    return {**common.read_json(common.CONFIG), 'factor_baselines': common.read_json(common.FACTOR_CONFIG)}


class FactorMechanismTests(unittest.TestCase):
    def test_active_factor_and_frozen_factor_exact_under_sgd(self):
        batch = {'img': torch.randn(12, 5), 'label': torch.arange(12)}
        for method in factors.METHODS:
            torch.manual_seed(3)
            model = TinyLoRA()
            initial = trainable_state(model)
            one, cost, audit = factors.local_train(model, [batch], method, options(), 'cpu', 1)
            self.assertTrue(torch.equal(initial['w_lora_A'], one['w_lora_A']))
            self.assertFalse(torch.equal(initial['w_lora_B'], one['w_lora_B']))
            self.assertEqual(cost['optimizer_steps'], 4 if method == 'lora-a2' else 3)
            self.assertEqual(audit['active_factor'], 'B')
            self.assertTrue(all(p.requires_grad for p in model.parameters()))
            two, _, audit = factors.local_train(model, [batch], method, options(), 'cpu', 2)
            frozen = 'w_lora_B' if method in ('rolora', 'lora-a2') else 'w_lora_A'
            self.assertTrue(torch.equal(one[frozen], two[frozen]))
            self.assertFalse(torch.equal(one['w_lora_A' if frozen.endswith('B') else 'w_lora_B'],
                                         two['w_lora_A' if frozen.endswith('B') else 'w_lora_B']))

    def test_unselected_deltas_survive_nonzero_base_momentum_and_decay(self):
        model = TinyLoRA()
        with torch.no_grad():
            model.w_lora_B.normal_()
        before = trainable_state(model)
        batch = {'img': torch.randn(10, 5), 'label': torch.arange(10)}
        for rnd, key in ((1, 'w_lora_B'), (2, 'w_lora_A')):
            load_trainable(model, before)
            after, cost, audit = factors.local_train(model, [batch] * 3, 'lora-a2', options(), 'cpu', rnd)
            chosen = audit['rank_indices'][key]
            self.assertEqual(len(chosen), 2)
            unselected = [i for i in range(4) if i not in chosen]
            old = before[key][:, unselected] if rnd == 1 else before[key][unselected, :]
            new = after[key][:, unselected] if rnd == 1 else after[key][unselected, :]
            self.assertTrue(torch.equal(old, new))
            self.assertEqual(cost['selection_optimizer_steps'], 3)
            self.assertEqual(cost['optimizer_steps'], 12)

    def test_rank_selection_is_global_and_uses_both_factors(self):
        before = {'x_lora_A': torch.tensor([[10., 0.], [0., 1.]]),
                  'x_lora_B': torch.zeros(2, 2),
                  'y_lora_A': torch.tensor([[2., 0.], [0., .1]]),
                  'y_lora_B': torch.zeros(2, 2)}
        probe = copy.deepcopy(before)
        probe['x_lora_B'] = torch.eye(2)
        probe['y_lora_B'] = torch.eye(2)
        _, indices = factors.rank_masks(before, probe, 'B', 1)
        self.assertEqual(indices, {'x_lora_B': [0], 'y_lora_B': [0]})
        probe['x_lora_B'][:, 1] *= 4
        _, indices = factors.rank_masks(before, probe, 'B', 1)
        self.assertEqual(indices, {'x_lora_B': [0, 1], 'y_lora_B': []})
        # Same outer-product identity for A-active rounds.
        before = {k: torch.ones_like(v) for k, v in before.items()}
        probe = copy.deepcopy(before); probe['x_lora_A'][0] += 5
        masks, _ = factors.rank_masks(before, probe, 'A', 1)
        self.assertTrue(bool(masks['x_lora_A'][0].all()))

    def test_active_aggregation_matches_dense_effective_average(self):
        before = trainable_state(TinyLoRA(out=6))
        for rnd in (1, 2):
            if rnd == 2:
                before['w_lora_B'].normal_()
            key = 'w_lora_B' if rnd == 1 else 'w_lora_A'
            states = {i: copy.deepcopy(before) for i in (0, 1)}
            states[0][key].add_(.3); states[1][key].sub_(.4)
            averaged, _ = factors.aggregate('rolora', before, states, [0, 1], [3, 1], rnd)
            expected = .75 * (states[0]['w_lora_B'] @ states[0]['w_lora_A']) + .25 * (states[1]['w_lora_B'] @ states[1]['w_lora_A'])
            torch.testing.assert_close(averaged['w_lora_B'] @ averaged['w_lora_A'], expected)
            other = 'w_lora_A' if rnd == 1 else 'w_lora_B'
            states[0][other].add_(1)
            with self.assertRaises(ValueError):
                factors.aggregate('rolora', before, states, [0, 1], [3, 1], rnd)

    def test_sparse_zero_delta_does_not_erase_previous_global_slots(self):
        before = trainable_state(TinyLoRA(out=6)); before['w_lora_B'].fill_(2)
        states = {i: copy.deepcopy(before) for i in (0, 1)}
        states[0]['w_lora_B'][:, 0] += 1
        states[1]['w_lora_B'][:, 1] += 4
        result, _ = factors.aggregate('lora-a2', before, states, [0, 1], [3, 1], 1)
        torch.testing.assert_close(result['w_lora_B'][:, 0], torch.full((6,), 2.75))
        torch.testing.assert_close(result['w_lora_B'][:, 1], torch.full((6,), 3.))
        self.assertTrue(torch.equal(result['w_lora_B'][:, 2:], before['w_lora_B'][:, 2:]))

    def test_fedsvd_preserves_effective_matrix_and_orthonormal_rows(self):
        for mode in ('full', 'deficient', 'zero'):
            state = trainable_state(TinyLoRA(out=6))
            if mode != 'zero':
                state['w_lora_B'].normal_()
            if mode == 'deficient':
                state['w_lora_B'][:, 1:] = 0
            result, error = factors.svd_reparameterize(state)
            torch.testing.assert_close(result['w_lora_B'] @ result['w_lora_A'], state['w_lora_B'] @ state['w_lora_A'], atol=2e-6, rtol=1e-5)
            torch.testing.assert_close(result['w_lora_A'] @ result['w_lora_A'].T, torch.eye(4), atol=1e-6, rtol=1e-6)
            self.assertLess(error, 1e-5)

    def test_fedsvd_against_pinned_official_reinit(self):
        path = common.REPO / 'third_party/paper_baselines/fedsvd/misc/utils.py'
        tree = ast.parse(path.read_text(encoding='utf-8'))
        definition = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'reinit_lora')
        class FakeLoraLayer(nn.Module):
            active_adapters = ['default']
        env = {'torch': torch, 'LoraLayer': FakeLoraLayer}
        exec(compile(ast.Module(body=[definition], type_ignores=[]), str(path), 'exec'), env)
        layer = FakeLoraLayer()
        layer.lora_A = nn.ModuleDict({'default': nn.Linear(5, 4, bias=False)})
        layer.lora_B = nn.ModuleDict({'default': nn.Linear(4, 6, bias=False)})
        # Use float64 on both paths to compare the exact same SVD convention.
        layer.double()
        state = {'w_lora_A': layer.lora_A['default'].weight.detach().clone(),
                 'w_lora_B': layer.lora_B['default'].weight.detach().clone()}
        result, _ = factors.svd_reparameterize(state)
        env['reinit_lora'](layer)
        torch.testing.assert_close(result['w_lora_A'], layer.lora_A['default'].weight)
        torch.testing.assert_close(result['w_lora_B'], layer.lora_B['default'].weight)

    def test_factor_flags_restore_on_failure(self):
        model = TinyLoRA()
        batch = {'img': torch.full((2, 5), float('nan')), 'label': torch.zeros(2, dtype=torch.long)}
        with self.assertRaises(FloatingPointError):
            factors.local_train(model, [batch], 'rolora', options(), 'cpu', 2)
        self.assertTrue(all(p.requires_grad for p in model.parameters()))


class FactorIntegrationTests(unittest.TestCase):
    def test_first_four_jobs_can_register_concurrently_without_collision(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(common, 'source_hashes', return_value={'fixture': '1'}):
            root = Path(tmp); fake_reference(root / 'ref')
            jobs = [common.make_job(method, root / 'ref', root, root / 'out')
                    for method in ('fedavg-lora', 'capt', 'fedpurel', 'fedntd')]
            with ThreadPoolExecutor(max_workers=4) as pool:
                list(pool.map(common.register, jobs))
            self.assertEqual(len({common.run_path(job) for job in jobs}), 4)
            self.assertEqual(len(list((root / 'out/jobs').glob('*.json'))), 4)
            for job in jobs:
                self.assertEqual(common.read_json(common.job_path(root / 'out', job['method'])), job)
                cmd = command_for(job)
                self.assertEqual(Path(cmd[cmd.index('--job') + 1]).resolve(), common.job_path(root / 'out', job['method']).resolve())

    def test_formal_a2_collection_counts_probe_compute(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(common, 'source_hashes', return_value={}):
            root = Path(tmp); fake_reference(root / 'ref')
            job = common.make_job('lora-a2', root / 'ref', root, root / 'out')
            common.register(job); run = common.run_path(job); identity = common.job_id(job)
            common.write_json(run / 'run.json', dict(job_id=identity, job=job, trainable_parameters=420))
            common.write_json(run / 'completion.json', dict(job_id=identity, completed_round=100,
                official_test_passes=101, smoke=False, source_unchanged=True, optimizer_steps=12000))
            for rnd in range(101):
                metric, _ = collect.metrics([70] * 100, [100] * 100)
                common.write_json(run / 'rounds' / f'r{rnd:03d}.json', dict(round=rnd, job_id=identity,
                    correct=[70] * 100, total=[100] * 100, metrics=metric))
            rows = [dict(round=rnd, optimizer_steps=120, selection_optimizer_steps=30,
                selection_forward_images=300, train_seconds=1., upload_bytes=100, downlink_bytes=200,
                student_forward_images=1200, teacher_forward_images=0, backward_images=1200) for rnd in range(1, 101)]
            common.write_csv(run / 'costs.csv', rows)
            result = collect.collect(root / 'out')[0]
            self.assertEqual(result['last20_bottom20_tail_acc'], 70.)
            self.assertEqual(result['selection_optimizer_steps'], 3000)
            rows[-1]['selection_optimizer_steps'] = 0
            common.write_csv(run / 'costs.csv', rows)
            with self.assertRaises(ValueError):
                collect.collect(root / 'out')

    def test_separate_default_suites_and_shared_model_only(self):
        first, second = parser_for_cli().parse_args([]), parser_for_cli('factors').parse_args([])
        self.assertEqual(tuple(first.methods), common.METHODS)
        self.assertEqual(tuple(second.methods), common.FACTOR_METHODS)
        self.assertNotEqual(first.output_root, second.output_root)
        self.assertEqual(second.seed, 42)
        self.assertNotIn('fedsa-lora', common.ALL_METHODS)
        hashes = common.source_hashes()
        self.assertIn('third_party/paper_baselines/fedsvd/UPSTREAM.json', hashes)
        self.assertIn('configs/benchmarks/factor_baselines_seed42.json', hashes)

    def test_resume_keeps_both_factors_and_alternation_phase(self):
        for method in common.FACTOR_METHODS:
            with tempfile.TemporaryDirectory() as tmp, patch.object(common, 'source_hashes', return_value={'fixture': '1'}), \
                    patch.object(runtime, 'source_hashes', return_value={'fixture': '1'}), patch('sys.stdout', new_callable=io.StringIO):
                root = Path(tmp); fake_reference(root / 'ref')
                torch.manual_seed(9)
                batch = {'img': torch.randn(8, 5), 'label': torch.arange(8)}
                test = {'img': torch.randn(10000, 5), 'label': torch.arange(100).repeat_interleave(100)}
                def data(*args):
                    return {i: [batch] for i in range(30)}, [test], np.ones((30, 100), dtype=np.int64), 'images'
                def model(*args):
                    torch.manual_seed(42)
                    return TinyLoRA(), None
                full = common.make_job(method, root / 'ref', root, root / 'full', 0, smoke=True)
                split = common.make_job(method, root / 'ref', root, root / 'split', 0, smoke=True)
                common.register(full); common.register(split)
                with patch.object(runtime, 'build_config', return_value='test'), patch.object(runtime, 'build_data', side_effect=data), \
                        patch.object(runtime, 'build_model', side_effect=model):
                    runtime.train(full, device_override='cpu')
                    runtime.train(split, stop_after=1, device_override='cpu')
                    runtime.train(split, resume=True, device_override='cpu')
                left = torch.load(common.run_path(full) / 'checkpoint_last.pt', weights_only=False)
                right = torch.load(common.run_path(split) / 'checkpoint_last.pt', weights_only=False)
                self.assertEqual(len(right['state']), 2)
                for key in left['state']:
                    torch.testing.assert_close(left['state'][key], right['state'][key], atol=0, rtol=0)
                self.assertEqual(collect.collect(root / 'split'), [])
                self.assertEqual(common.read_csv(root / 'split/analysis/status.csv')[0]['status'], 'smoke_only')
                archive = collect.pack(root / 'split')
                self.assertTrue(archive.exists())


if __name__ == '__main__':
    unittest.main()
