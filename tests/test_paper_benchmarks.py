"""Numerical upstream parity, protocol integrity, resume, and reporting checks."""
import ast
import copy
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from trainers.baselines import capt, fedntd, fedpurel
from trainers.baselines.common import load_trainable, trainable_state, weighted_average
from tools.benchmarks import common, collect, runtime
from scripts.run_paper_benchmarks import command_for, parser_for_cli

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT / 'third_party/paper_baselines'


def extracted(path, names, globals_):
    tree = ast.parse(path.read_text(encoding='utf-8'))
    nodes = [node for node in tree.body if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in names]
    if len(nodes) != len(names):
        raise AssertionError('Upstream definitions changed')
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), globals_)
    return globals_


class OfficialParityTests(unittest.TestCase):
    def test_fedntd_official_value_and_gradient(self):
        folder = UPSTREAM / 'fedntd/algorithms/fedntd'
        env = dict(torch=torch, nn=nn, F=F)
        extracted(folder / 'utils.py', ['refine_as_not_true'], env)
        extracted(folder / 'criterion.py', ['NTD_Loss'], env)
        torch.manual_seed(9)
        logits = torch.randn(7, 6, requires_grad=True)
        teacher = torch.randn(7, 6, requires_grad=True)
        labels = torch.arange(7) % 6
        ours = fedntd.loss(logits, labels, teacher)
        expected = env['NTD_Loss'](6, 3, 1)(logits, labels, teacher)
        torch.testing.assert_close(ours, expected)
        torch.testing.assert_close(torch.autograd.grad(ours, logits)[0], torch.autograd.grad(expected, logits)[0])
        self.assertIsNone(torch.autograd.grad(fedntd.loss(logits, labels, teacher), teacher, allow_unused=True)[0])

    def test_ntd_excludes_true_teacher_logit(self):
        logits, teacher = torch.randn(3, 4), torch.randn(3, 4)
        labels = torch.tensor([0, 1, 2])
        altered = teacher.clone(); altered[torch.arange(3), labels] += 100
        torch.testing.assert_close(fedntd.loss(logits, labels, teacher), fedntd.loss(logits, labels, altered))

    def test_fedpurel_entropy_matching_against_official_helpers(self):
        path = UPSTREAM / 'fedpurel/trainers/purelLoraGP.py'
        tree = ast.parse(path.read_text(encoding='utf-8'))
        helpers = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
                   and node.name in ('entropy_from_logits', 'search_temperature_per_sample')]
        env = dict(torch=torch, F=F)
        exec(compile(ast.Module(body=helpers, type_ignores=[]), str(path), 'exec'), env)
        logits = torch.randn(5, 10) * 4
        target = torch.tensor([.1, .5, 1., 1.5, 2.])
        torch.testing.assert_close(fedpurel.temperature_for_entropy(logits, target),
                                  env['search_temperature_per_sample'](logits, target))

    def test_fedpurel_projection_matches_actual_official_optimizer(self):
        path = UPSTREAM / 'fedpurel/Dassl/dassl/engine/trainer.py'
        tree = ast.parse(path.read_text(encoding='utf-8'))
        fn = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == 'GradPur_backward_and_update')
        env = dict(torch=torch)
        exec(compile(ast.Module(body=[fn], type_ignores=[]), str(path), 'exec'), env)
        left = nn.Linear(3, 2)
        right = copy.deepcopy(left)
        x, y = torch.randn(8, 3), torch.arange(8) % 2
        teacher = torch.randn(8, 2)
        optim_l, optim_r = torch.optim.SGD(left.parameters(), lr=.01), torch.optim.SGD(right.parameters(), lr=.01)
        trainer = types.SimpleNamespace(_models={'lora': right}, _optims={'lora': optim_r},
            model_zero_grad=lambda names: optim_r.zero_grad(), get_model_names=lambda names: names,
            detect_anomaly=lambda loss: None)
        ce_l, kl_l = fedpurel.losses(left(x), teacher, y)
        ce_r, kl_r = fedpurel.losses(right(x), teacher, y)
        optim_l.zero_grad(); fedpurel.backward(ce_l, kl_l, left.parameters()); optim_l.step()
        env['GradPur_backward_and_update'](trainer, ce_r, kl_r, 1., ['lora'])
        for a, b in zip(left.parameters(), right.parameters()):
            torch.testing.assert_close(a, b, atol=1e-7, rtol=1e-6)

    def test_projection_zero_and_conflict(self):
        a, b = torch.tensor([-1., 2.]), torch.tensor([1., 0.])
        torch.testing.assert_close(fedpurel.project_gradient(a, b), torch.tensor([0., 2.]))
        torch.testing.assert_close(fedpurel.project_gradient(a, torch.zeros_like(a)), a)
        torch.testing.assert_close(fedpurel.project_gradient(b, b), b)

    def test_capt_loss_value_and_gradients_match_official(self):
        env = dict(torch=torch, nn=nn, F=F)
        extracted(UPSTREAM / 'capt/loss/prompt_loss.py', ['PromptLoss'], env)
        torch.manual_seed(11)
        general = torch.randn(1, 8, requires_grad=True)
        specific = torch.randn(6, 3, 8, requires_grad=True)
        features = torch.randn(5, 8, requires_grad=True)
        logits = torch.randn(5, 6, requires_grad=True)
        labels, counts = torch.arange(5), torch.arange(1., 7.)
        ours = capt.loss(general, specific, features, logits, labels, counts)
        # Official hardcoded CUDA constructor is mapped to CPU for parity only.
        with patch.object(torch.cuda, 'FloatTensor', side_effect=lambda x: torch.as_tensor(x, dtype=torch.float32)):
            expected, _, _ = env['PromptLoss'](6)(general, specific, features, labels,
                                                counts / counts.sum(), logits, counts)
        torch.testing.assert_close(ours, expected, atol=3e-6, rtol=1e-6)
        for a, b in zip(torch.autograd.grad(ours, [general, specific, features, logits]),
                        torch.autograd.grad(expected, [general, specific, features, logits])):
            torch.testing.assert_close(a, b, atol=2e-6, rtol=2e-5)

    def test_capt_aggregation_matches_official(self):
        from scipy.cluster.hierarchy import linkage, fcluster
        from sklearn.cluster import KMeans
        names = ['js_divergence', 'kl_divergence', 'similarity_clustering', 'dissimilarity_clustering',
                 'aggregate_class_aware_prompts', 'communicate_within_cluster_similarity', 'communicate_within_cluster_dissimilarity']
        env = dict(np=np, torch=torch, linkage=linkage, fcluster=fcluster, KMeans=KMeans)
        extracted(UPSTREAM / 'capt/federated_main.py', names, env)
        counts = np.array([[10, 2, 1, 1, 1], [1, 20, 1, 1, 1], [1, 1, 30, 1, 1], [1, 1, 1, 40, 1]])
        specific, general = 'prompt_learner.class_aware_ctx', 'prompt_learner.general_ctx'
        states = {i: {specific: torch.randn(5, 2, 4), general: torch.randn(1, 4), 'coupling': torch.randn(3, 4)} for i in range(4)}
        base = {k: v.clone() for k, v in states[0].items()}
        copied = copy.deepcopy(states)
        ids = [3, 0, 2, 1]
        p = counts[ids] / counts[ids].sum(1)[:, None]
        np.random.seed(42)
        ours = capt.aggregate(states, ids, counts, base, 2)
        np.random.seed(42)
        s, d = env['similarity_clustering'](p, 2), env['dissimilarity_clustering'](p, 2)
        for grouping, operation in ((s, 'communicate_within_cluster_similarity'), (d, 'communicate_within_cluster_dissimilarity')):
            for group in set(grouping):
                env[operation]([ids[j] for j in np.flatnonzero(grouping == group)], copied)
        prompt = env['aggregate_class_aware_prompts'](p, copied, ids, 5, base[specific])
        for i in ids:
            copied[i][specific] = prompt
        expected = weighted_average(copied, ids, counts[ids].sum(1))
        for k in expected:
            torch.testing.assert_close(ours[k], expected[k])


class StateAndTrainingTests(unittest.TestCase):
    def test_joint_parameters_train_and_teacher_stays_fixed(self):
        class Tiny(nn.Module):
            def __init__(self):
                super().__init__()
                self.lora_A = nn.Parameter(torch.randn(3, 2))
                self.lora_B = nn.Parameter(torch.randn(2, 5))
            def forward(self, x):
                return x @ self.lora_A @ self.lora_B
        for method in ('fedavg-lora', 'fedntd', 'fedpurel'):
            torch.manual_seed(7)
            model = Tiny()
            teacher = copy.deepcopy(model).requires_grad_(False)
            if method == 'fedpurel':
                with torch.no_grad():
                    teacher.lora_B.zero_()
            before, tea_before = trainable_state(model), copy.deepcopy(teacher.state_dict())
            batch = {'img': torch.randn(4, 3), 'label': torch.tensor([0, 1, 2, 3])}
            options = common.read_json(common.CONFIG)
            state, cost = runtime.local_train(model, teacher if method != 'fedavg-lora' else None,
                [batch], method, options, torch.ones(5), torch.device('cpu'))
            self.assertEqual(cost['optimizer_steps'], 3)
            self.assertTrue(all(not torch.equal(state[k], before[k]) for k in state))
            for key in tea_before:
                torch.testing.assert_close(teacher.state_dict()[key], tea_before[key])
            self.assertTrue(all(p.grad is None for p in teacher.parameters()))

    def test_optimizer_reset_reproduces_independent_client(self):
        model = nn.Linear(3, 5)
        state = trainable_state(model)
        batch = {'img': torch.randn(4, 3), 'label': torch.tensor([0, 1, 2, 3])}
        settings = common.read_json(common.CONFIG)
        first, _ = runtime.local_train(model, None, [batch], 'fedavg-lora', settings, torch.ones(5), torch.device('cpu'))
        load_trainable(model, state)
        second, _ = runtime.local_train(model, None, [batch], 'fedavg-lora', settings, torch.ones(5), torch.device('cpu'))
        for key in first:
            torch.testing.assert_close(first[key], second[key], atol=0, rtol=0)

    def test_aggregation_weights_and_invalid_state(self):
        result = weighted_average({0: {'p': torch.tensor([1.])}, 1: {'p': torch.tensor([5.])}}, [0, 1], [3, 1])
        self.assertEqual(result['p'].item(), 2.)
        with self.assertRaises(ValueError):
            load_trainable(nn.Linear(1, 1), {'wrong': torch.tensor([1.])})

    def test_round_resume_matches_uninterrupted_training(self):
        # Exercise real optimization, teacher refresh, checkpoint, RNG restore,
        # atomic commits and collector on a CPU model; replace only CLIP/data.
        for method in ('fedntd', 'fedpurel'):
            with tempfile.TemporaryDirectory() as tmp, patch.object(common, 'source_hashes', return_value={'fixture': '1'}), \
                    patch.object(runtime, 'source_hashes', return_value={'fixture': '1'}), patch('sys.stdout', new_callable=io.StringIO):
                base = Path(tmp); fake_reference(base / 'ref')
                torch.manual_seed(91)
                batch = {'img': torch.randn(8, 3), 'label': torch.arange(8)}
                test = {'img': torch.randn(10000, 3), 'label': torch.arange(100).repeat_interleave(100)}
                def data(*args):
                    return {i: [batch] for i in range(30)}, [test], np.ones((30, 100), dtype=np.int64), 'images'
                def model(job, cfg, meta, device):
                    torch.manual_seed(42)
                    student = nn.Linear(3, 100)
                    teacher = copy.deepcopy(student).requires_grad_(False).eval()
                    if job['method'] == 'fedpurel':
                        with torch.no_grad():
                            teacher.weight.zero_(); teacher.bias.zero_()
                    return student, teacher
                full = common.make_job(method, base / 'ref', base, base / 'full', 0, smoke=True)
                split = common.make_job(method, base / 'ref', base, base / 'split', 0, smoke=True)
                common.register(full); common.register(split)
                with patch.object(runtime, 'build_config', return_value='test'), patch.object(runtime, 'build_data', side_effect=data), \
                        patch.object(runtime, 'build_model', side_effect=model):
                    runtime.train(full, device_override='cpu')
                    runtime.train(split, stop_after=1, device_override='cpu')
                    self.assertFalse((common.run_path(split) / 'completion.json').exists())
                    runtime.train(split, resume=True, device_override='cpu')
                a = torch.load(common.run_path(full) / 'checkpoint_last.pt', weights_only=False)
                b = torch.load(common.run_path(split) / 'checkpoint_last.pt', weights_only=False)
                for name in a['state']:
                    torch.testing.assert_close(a['state'][name], b['state'][name], atol=0, rtol=0)
                self.assertEqual(collect.collect(base / 'split'), [])  # smoke never enters main table
                self.assertEqual(common.read_csv(base / 'split/analysis/status.csv')[0]['status'], 'smoke_only')


def fake_reference(root):
    root.mkdir(parents=True)
    rows = [dict(raw_sample_id=i, class_id=i % 100, client_id=i % 30, local_position=i // 30) for i in range(300)]
    common.write_csv(root / 'partition_manifest.csv', rows)
    schedule = [list(range(30)) for _ in range(100)]
    common.write_json(root / 'protocol/full_schedule.json', schedule)
    common.write_json(root / 'protocol/eri_protocol.json', {})
    common.write_csv(root / 'protocol/probe_manifest.csv', [dict(raw_train_index=301)])
    meta = dict(topology='client-longtail', classnames=[str(i) for i in range(100)], resolved_config='fixture',
        schedule_sha256=common.digest(schedule), pool_sha256=common.digest(list(range(300))), test_sha256='fixture')
    common.write_json(root / 'bridge_metadata.json', meta)


class ProtocolTests(unittest.TestCase):
    def test_seed42_default_and_cannot_select_seed0(self):
        args = parser_for_cli().parse_args([])
        self.assertEqual(args.seed, 42)
        self.assertEqual(tuple(args.methods), common.METHODS)
        with self.assertRaises(SystemExit):
            parser_for_cli().parse_args(['--seed', '0'])

    def test_duplicate_samples_and_schedule_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            ref = Path(tmp) / 'ref'; fake_reference(ref)
            common.reference_contract(ref)
            rows = common.read_csv(ref / 'partition_manifest.csv')
            rows[-1]['raw_sample_id'] = rows[0]['raw_sample_id']
            common.write_csv(ref / 'partition_manifest.csv', rows)
            with self.assertRaises(ValueError):
                common.reference_contract(ref)
        with tempfile.TemporaryDirectory() as tmp:
            ref = Path(tmp) / 'ref'; fake_reference(ref)
            common.write_json(ref / 'protocol/full_schedule.json', [list(range(29)) for _ in range(100)])
            with self.assertRaises(ValueError):
                common.reference_contract(ref)

    def test_freeze_source_and_ab_command(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(common, 'source_hashes', return_value={'test.py': 'hash'}):
            root = Path(tmp); fake_reference(root / 'ref')
            job = common.make_job('ab', root / 'ref', root / 'data', root / 'out')
            common.register(job); common.register(job)
            command = command_for(job)
            self.assertEqual(command[command.index('--seeds') + 1], '42')
            self.assertIn('run_ab_validation.py', command[2])
            changed = copy.deepcopy(job); changed['source_hashes']['test.py'] = 'changed'
            with self.assertRaises(ValueError):
                common.register(changed)

    def test_smoke_cannot_overwrite_formal_suite(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(common, 'source_hashes', return_value={}):
            root = Path(tmp); fake_reference(root / 'ref')
            formal = common.make_job('fedntd', root / 'ref', root / 'data', root / 'out')
            common.register(formal)
            with self.assertRaises(ValueError):
                common.register({**formal, 'smoke': True})

    def test_group_metrics_and_nonfinite_rejected(self):
        result, _ = collect.metrics(list(range(100)), [100] * 100)
        self.assertEqual(result['overall_acc'], 49.5)
        self.assertEqual(result['bottom20_tail_acc'], 89.5)
        with self.assertRaises(ValueError):
            collect.metrics([float('nan')] * 100, [100] * 100)

    def test_incomplete_not_zero_or_winner(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(common, 'source_hashes', return_value={}):
            root = Path(tmp); fake_reference(root / 'ref')
            job = common.make_job('fedntd', root / 'ref', root / 'data', root / 'out')
            common.register(job)
            self.assertEqual(collect.collect(root / 'out'), [])
            self.assertEqual(common.read_csv(root / 'out/analysis/status.csv')[0]['status'], 'not_started')

    def test_complete_collection_rejects_corrupt_group_metric(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(common, 'source_hashes', return_value={}):
            root = Path(tmp); fake_reference(root / 'ref')
            job = common.make_job('fedntd', root / 'ref', root / 'data', root / 'out')
            common.register(job); run = common.run_path(job)
            identity = common.job_id(job)
            common.write_json(run / 'run.json', dict(job_id=identity, job=job, trainable_parameters=5))
            common.write_json(run / 'completion.json', dict(job_id=identity, completed_round=100,
                official_test_passes=101, smoke=False, source_unchanged=True, optimizer_steps=9000))
            costs = []
            for rnd in range(101):
                correct = [70] * 100
                metric, _ = collect.metrics(correct, [100] * 100)
                common.write_json(run / 'rounds' / f'r{rnd:03d}.json', dict(round=rnd, job_id=identity,
                    correct=correct, total=[100] * 100, metrics=metric))
                if rnd:
                    costs.append(dict(round=rnd, optimizer_steps=90, train_seconds=1., upload_bytes=1,
                        downlink_bytes=1, student_forward_images=900, teacher_forward_images=900, backward_images=900))
            common.write_csv(run / 'costs.csv', costs)
            result = collect.collect(root / 'out')
            self.assertEqual(result[0]['last20_bottom20_tail_acc'], 70.)
            common.write_json(run / 'rounds/r100.json', dict(round=100, job_id=identity,
                correct=[70] * 100, total=[100] * 100, metrics={**metric, 'bottom20_tail_acc': 99.}))
            with self.assertRaises(ValueError):
                collect.collect(root / 'out')
            self.assertEqual(common.read_csv(root / 'out/analysis/performance.csv'), [])


if __name__ == '__main__':
    unittest.main()
