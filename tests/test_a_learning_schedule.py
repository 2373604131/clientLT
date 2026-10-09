"""CPU tests for the actual paired training engine, continuation graph and reports."""
import contextlib
from concurrent.futures import ThreadPoolExecutor
import io
import json
import os
import subprocess
import sys
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
from tools.a_learning_schedule.runtime import (
    Worker, aggregate, effective_norm, prediction_metrics, load_torch, register_request,
)
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


class ShardTests(unittest.TestCase):
    def test_shards_cover_each_anchor_once_and_reject_conflicting_options(self):
        from scripts.run_cliplora_a_learning_schedule import parse_args
        selected = []
        for shard in (1, 2, 3):
            args = parse_args(['--stage', 'run', '--shard', str(shard)])
            selected.extend((origin, rnd) for origin in args.origins for rnd in args.rounds)
        self.assertEqual(sorted(selected), [(origin, rnd) for origin in ('e2', 'e3') for rnd in (20, 50, 80)])
        for flags in (['--rounds', '20'], ['--origins', 'e2'], ['--gpus', '0', '1'],
                      ['--stage', 'collect']):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                parse_args(['--shard', '1'] + flags)

    def test_shard_preserves_allocated_gpu_and_rejects_disabled_or_multiple_devices(self):
        from scripts.run_cliplora_a_learning_schedule import parse_args, selected_gpus
        args = parse_args(['--shard', '1'])
        for token in ('0', '3', 'GPU-allocated-uuid', 'MIG-allocated-uuid'):
            with patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': token}):
                self.assertEqual(selected_gpus(args), [token])
        for mask in ('', '-1', '0,1'):
            with patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': mask}), self.assertRaises(ValueError):
                selected_gpus(args)
        with patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': '1,3'}):
            explicit = parse_args(['--shard', '1', '--gpus', '3'])
            self.assertEqual(selected_gpus(explicit), ['3'])

    def test_three_launchers_run_two_anchors_serially_on_their_own_device(self):
        from scripts.run_cliplora_a_learning_schedule import parse_args, launch
        observed = []
        with tempfile.TemporaryDirectory() as temp:
            for shard, token in enumerate(('3', 'GPU-node2', '0'), 1):
                args = parse_args(['--stage', 'run', '--shard', str(shard), '--output-root', temp])
                jobs = [dict(origin=origin, anchor_round=args.rounds[0], source_run='/source/' + origin)
                        for origin in args.origins]
                live = set()

                class SimulatedProcess:
                    pid = 123
                    def __init__(self, command, **kwargs):
                        if live:
                            raise AssertionError('Two anchors were launched concurrently on one shard GPU')
                        self.pending = True
                        live.add(self)
                        observed.append((kwargs['env']['CUDA_VISIBLE_DEVICES'],
                                         command[command.index('--origin') + 1],
                                         int(command[command.index('--anchor-round') + 1])))

                    def poll(self):
                        if self.pending:
                            self.pending = False
                            return None
                        live.discard(self)
                        return 0

                with patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': token}), \
                     patch('scripts.run_cliplora_a_learning_schedule.subprocess.Popen', side_effect=SimulatedProcess), \
                     patch('scripts.run_cliplora_a_learning_schedule.time.sleep'), \
                     contextlib.redirect_stdout(io.StringIO()):
                    launch(args, jobs, 'test')
                self.assertFalse(live)
        self.assertEqual(observed, [('3', 'e2', 20), ('3', 'e3', 20),
                                    ('GPU-node2', 'e2', 50), ('GPU-node2', 'e3', 50),
                                    ('0', 'e2', 80), ('0', 'e3', 80)])

    def test_concurrent_preflight_and_reports_do_not_overwrite_other_shards(self):
        from scripts.run_cliplora_a_learning_schedule import (
            parse_args, preflight, preflight_path, analysis_directory,
        )
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            data = root / 'data'
            data.mkdir()
            for name in ('train', 'test', 'meta'):
                (data / name).touch()
            args = [parse_args(['--shard', str(i), '--output-root', temp, '--data-root', str(data)])
                    for i in (1, 2, 3)]

            def source_check(source, *unused):
                return dict(path=str(source), rows=[], meta={'schedule': [], 'resolved_config':
                    "INPUT:\n  SIZE: (224, 224)\n  TRANSFORMS: ('normalize',)\n"})

            def run_reports(arg):
                preflight(arg)
                return summarize(root, [42], arg.origins, arg.rounds, plots=False,
                                 destination=analysis_directory(arg))

            with patch('scripts.run_cliplora_a_learning_schedule.discover_source',
                       side_effect=lambda root, seed, origin, override: '/source/' + origin), \
                 patch('scripts.run_cliplora_a_learning_schedule.check_source', side_effect=source_check), \
                 patch('scripts.run_cliplora_a_learning_schedule.code_fingerprint', return_value='test'), \
                 contextlib.redirect_stdout(io.StringIO()):
                with ThreadPoolExecutor(max_workers=3) as pool:
                    results = list(pool.map(run_reports, args))
            self.assertTrue(all(len(rows) == 2 for rows in results))
            for arg in args:
                report = read_json(preflight_path(arg))
                self.assertEqual([j['anchor_round'] for j in report['jobs']], arg.rounds * 2)
                summary = read_json(analysis_directory(arg) / 'summary.json')
                self.assertEqual(summary['rounds'], arg.rounds)
                self.assertEqual(summary['expected'], 2)
            self.assertFalse((root / 'analysis/summary.json').exists())
            summarize(root, [42], ['e2', 'e3'], [20, 50, 80], plots=False)
            self.assertEqual(read_json(root / 'analysis/summary.json')['expected'], 6)
            self.assertTrue(all(read_json(analysis_directory(arg) / 'summary.json')['expected'] == 2 for arg in args))


class StartupTests(unittest.TestCase):
    def test_launcher_prints_the_current_failure_without_old_appended_errors(self):
        from scripts.run_cliplora_a_learning_schedule import parse_args, launch
        with tempfile.TemporaryDirectory() as temp:
            args = parse_args(['--stage', 'smoke', '--shard', '1', '--gpus', '0', '--output-root', temp])
            log = Path(temp) / 'launcher_logs/seed42/e2_r020_smoke.log'
            log.parent.mkdir(parents=True)
            log.write_text('old-error-that-must-not-be-repeated\n', encoding='utf-8')

            class FailedProcess:
                pid = 123
                def __init__(self, command, **kwargs):
                    # Popen inherits this descriptor after the attempt header.
                    kwargs['stdout'].write('Traceback (most recent call last):\nValueError: current-error\n')
                    kwargs['stdout'].flush()

                def poll(self):
                    return 1

            output = io.StringIO()
            with patch('scripts.run_cliplora_a_learning_schedule.subprocess.Popen', side_effect=FailedProcess), \
                 contextlib.redirect_stdout(output), self.assertRaisesRegex(ValueError, '1 anchor jobs failed'):
                launch(args, [dict(origin='e2', anchor_round=20, source_run='/source/e2')], 'test')
            self.assertIn('ValueError: current-error', output.getvalue())
            self.assertNotIn('old-error-that-must-not-be-repeated', output.getvalue())
            self.assertIn('old-error-that-must-not-be-repeated', log.read_text(encoding='utf-8'))

    def test_dassl_registration_first_breaks_the_reported_import_cycle(self):
        # Reproduce the observed registration graph in clean interpreters. The
        # production bootstrap is exercised; GPU/model dependencies are omitted.
        repo = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            modules = {
                'Dassl/__init__.py': '', 'Dassl/dassl/__init__.py': '', 'trainers/__init__.py': '',
                'Dassl/dassl/engine/__init__.py': (repo / 'Dassl/dassl/engine/__init__.py').read_text(encoding='utf-8'),
                'Dassl/dassl/engine/trainer.py': '\n'.join('class %s: pass' % name for name in
                    ('TrainerX', 'TrainerXU', 'TrainerBase', 'SimpleTrainer', 'SimpleNet')),
                'Dassl/dassl/engine/build.py': 'from trainers.clip import CLIP\n'
                    'from trainers.cliplora import ClipLora\nTRAINER_REGISTRY = {}\n'
                    'def build_trainer(): return ClipLora()\n',
                'trainers/clip.py': 'from Dassl.dassl.engine.trainer import TrainerX\n'
                    'class CLIP(TrainerX): pass\n',
                'trainers/cliplora.py': 'from Dassl.dassl.engine.trainer import TrainerX\n'
                    'class ClipLora(TrainerX): pass\n'
                    'def build_cliplora_model(): return ClipLora()\n'
                    'def cliplora_optimizer_step(): return "step-ready"\n',
            }
            for name, source in modules.items():
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(source, encoding='utf-8')
            setup = 'import sys\nsys.path.insert(0, sys.argv[1])\n'
            broken = subprocess.run([sys.executable, '-c', setup + 'from trainers.cliplora import ClipLora', temp],
                                    cwd=repo, capture_output=True, text=True, timeout=30)
            self.assertNotEqual(broken.returncode, 0)
            self.assertIn('partially initialized module', broken.stderr)
            fixed = subprocess.run([sys.executable, '-c', setup +
                                    'from tools.a_learning_schedule.runtime import load_training_api\n'
                                    'build, step = load_training_api()\n'
                                    'assert type(build()).__name__ == "ClipLora"\n'
                                    'assert step() == "step-ready"\n', temp],
                                   cwd=repo, capture_output=True, text=True, timeout=30)
            self.assertEqual(fixed.returncode, 0, fixed.stderr)

    def test_failed_startup_request_can_recover_with_audited_code_change(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            previous = dict(seed=42, source='same-source', code_fingerprint='old')
            current = dict(previous, code_fingerprint='fixed')
            write_json(root / 'request.json', previous)
            (root / 'worker.lock').touch()
            with contextlib.redirect_stdout(io.StringIO()):
                register_request(root, current)
            self.assertEqual(read_json(root / 'request.json'), current)
            backups = list(root.glob('request_before_startup_fix_*.json'))
            self.assertEqual(len(backups), 1)
            self.assertEqual(read_json(backups[0]), previous)
            register_request(root, current)
            self.assertEqual(list(root.glob('request_before_startup_fix_*.json')), backups)

    def test_code_change_cannot_reuse_saved_work_or_change_experiment_inputs(self):
        previous = dict(seed=42, source='same-source', code_fingerprint='old')
        for artifact in ('anchor_info.json', 'feedback_manifest.csv', 'smoke.json',
                         'nodes/phase.pt', 'evaluations/anchor/metrics.json', 'complete.json'):
            with self.subTest(artifact=artifact), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                write_json(root / 'request.json', previous)
                path = root / artifact
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b'existing-work')
                with self.assertRaisesRegex(ValueError, 'different protocol/code/source'):
                    register_request(root, dict(previous, code_fingerprint='fixed'))
                self.assertEqual(read_json(root / 'request.json'), previous)
                self.assertEqual(path.read_bytes(), b'existing-work')
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            write_json(root / 'request.json', previous)
            for changed in (dict(previous, source='other-source'), dict(previous, seed=0, code_fingerprint='fixed')):
                with self.assertRaises(ValueError):
                    register_request(root, changed)
            self.assertEqual(read_json(root / 'request.json'), previous)


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
