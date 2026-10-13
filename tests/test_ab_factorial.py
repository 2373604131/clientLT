"""Four-GPU dispatch, pairing guards and actual tiny A/B joint trajectories."""
import copy
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from scripts import run_cliplora_ab_factorial as launcher
from scripts import run_cliplora_joint_aggregation as single
from scripts.run_ab_validation import write_json
from tools.client_aggregation import ab_factorial as protocol
from tools.client_aggregation.protocol import load_json, read_csv, validate_arm, write_table
from tools.client_aggregation.runtime import runtime_class
from utils.cliplora_ab_factorial import transfer_config
from utils.cliplora_b_routes import RouteTransfer
from utils.cliplora_sfra import SFRARuntime
from utils.cliplora_functional_feedback import snapshot
from tests import test_method_a_margin as margin_fixture


class TestFactorial(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def test_fixed_registration_and_four_cell_commands(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = launcher.parse_args(['--output-root', tmp])
            suite = launcher.register(args)
            self.assertEqual(suite, launcher.register(args))
            hashes = set()
            for cell, (arm, b) in protocol.CELLS.items():
                root = Path(suite['cells'][cell]['root'])
                plan = load_json(root/'experiment_plan.json')
                job = single.make_job(plan, arm)
                validate_arm(job)
                hashes.add(job['aggregation']['weights_sha256'])
                self.assertEqual(protocol.has_new_b(job), b)
                self.assertEqual(job['settings']['client_concurrency'], 8)
                smoke = single.make_job(plan, arm, 'smoke')
                command = single.command_for(smoke, root)
                self.assertEqual(command[command.index('--sfra_stop_after_round')+1], '6')
                self.assertNotIn('--b_transfer_enable', command)  # New B is injected only by validated joint spec.
                bad = copy.deepcopy(job); bad['settings']['client_concurrency'] = 4
                with self.assertRaises(ValueError): validate_arm(bad)
            self.assertEqual(len(hashes), 1)
            args.epsilon = .02
            with self.assertRaisesRegex(ValueError, 'changed'): launcher.register(args)
        for bad in (['--gpus', '0', '0', '1', '2'], ['--client-concurrency', '4'], ['--margin-scale', '0']):
            with self.assertRaises(SystemExit), patch('sys.stderr'): launcher.parse_args(bad)

    def test_four_processes_overlap_with_separate_gpus_and_smoke_first(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = launcher.parse_args(['--stage', 'server', '--output-root', tmp, '--gpus', '4', '5', '6', '7'])
            suite = launcher.register(args)
            barrier = threading.Barrier(4)
            calls, lock = [], threading.Lock()
            def run(command, **kwargs):
                stage = command[command.index('--stage')+1]
                cell = command[command.index('--cells')+1]
                with lock:
                    calls.append((cell, stage, kwargs['env']['CUDA_VISIBLE_DEVICES']))
                if stage == 'smoke': barrier.wait(timeout=10)
                return SimpleNamespace(returncode=0)
            with patch.object(launcher.subprocess, 'run', side_effect=run), patch('builtins.print'):
                launcher.dispatch(args, suite)
            self.assertEqual(len(calls), 8)
            for i, cell in enumerate(protocol.CELLS):
                self.assertEqual([(s,g) for c,s,g in calls if c == cell], [('smoke', str(i+4)), ('run', str(i+4))])
            suite['cells']['plain_joint']['reused'] = True
            self.assertEqual(len(launcher.selected_jobs(args, suite)), 3)
            args.cells = ['new_a_joint', 'new_a_new_b_joint']
            self.assertEqual(launcher.selected_jobs(args, suite), [('new_a_joint',5),('new_a_new_b_joint',6)])

    def test_failed_smoke_never_starts_its_formal_run_or_reduces_concurrency(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = launcher.parse_args(['--stage', 'server', '--cells', 'new_a_new_b_joint', '--output-root', tmp])
            suite = launcher.register(args)
            with patch.object(launcher.subprocess, 'run', return_value=SimpleNamespace(returncode=1)) as run, patch('builtins.print'):
                with self.assertRaisesRegex(ValueError, 'failed'): launcher.dispatch(args, suite)
            self.assertEqual(run.call_count, 1)
            self.assertEqual(run.call_args.args[0][run.call_args.args[0].index('--stage')+1], 'smoke')

    def test_real_feedback_coverage_and_formal_schedule(self):
        report = protocol.feedback_preflight('references/full10_clientlt')
        self.assertEqual(report['tail_classes'], 20)
        self.assertEqual(report['checked_rounds'], [1,6,30,40,50,60,70,80,90,100])
        cfg = transfer_config('formal')
        self.assertEqual(cfg['rounds'], list(range(30,101,10)))
        self.assertEqual(cfg['route_arm'], 'F-tail-equal')
        self.assertEqual(cfg['commit_rule'], 'fixed second projected step')
        self.assertEqual(cfg['feedback_gradient_reduction'], 'sum_global_coefficients_no_second_client_weight')

    def make_tiny(self, folder, cell, last):
        rt = margin_fixture.TestMethodAMargin().make_runtime(folder, last)
        arm, b = protocol.CELLS[cell]
        job = dict(arm=arm, mode='smoke', settings=dict(factorial_experiment=protocol.EXPERIMENT,
            factorial_cell=cell), aggregation={'counts': {'tail_ids': [2]}})
        rt.__class__ = runtime_class(job)
        rt.job = job
        rt.is_frozen = False
        rt.history = None
        rt.method = cell
        rt.args.seed, rt.args.partition = 42, 'client-longtail'
        rt.tail = [2]
        rt.trainer.training_logit_adjustment = rt.margin_feedback.adjustment
        source = [[None]*len(items) for items in rt.bank.datasets]
        rows = []
        for t in rt.bank.tokens:
            for p in t['local_positions']:
                source[t['client_id']][p] = SimpleNamespace(label=t['class_id'])
                rt.bank.datasets[t['client_id']][p]['label'] = t['class_id']
                rows.append(dict(client_id=t['client_id'], local_position=p, class_id=t['class_id']))
        write_table(rt.root/'partition_manifest.csv', rows)
        rt.trainer.dm = SimpleNamespace(dataset=SimpleNamespace(federated_train_x=source))
        rt.audit = SimpleNamespace(counts=rt.margin_feedback.counts.cpu())
        if arm == 'plain_a':
            rt.variant = 's'
        if b:
            rt.sfra_config['b_transfer'] = transfer_config('smoke')
            rt.b_transfer = RouteTransfer(rt)
        # Keep the production publish hook and checkpoint/restore, with tiny official metrics.
        del rt.publish
        def evaluate(state, rnd, stage, folder):
            metrics = dict(overall_acc=0., head20_acc=0., middle60_acc=0., bottom20_tail_acc=0., non_tail_acc=0.)
            write_json(folder/f'r{rnd:03d}.json', dict(seconds=0., **metrics))
            return metrics, [0.]*100
        rt.evaluate_predictions = evaluate
        rt.flush_records = SFRARuntime.flush_records.__get__(rt)
        original = rt.train_phase
        def phase(state, rnd, factor='B', extra=False, return_deltas=False):
            after, deltas = original(state, rnd, factor, extra, return_deltas=True)
            if factor == 'B' and b and rt.b_transfer.scheduled(rnd):
                rt.b_transfer.cache_updates(deltas, list(deltas))
                after = rt.b_transfer.apply_shared(state, after, rnd)
            return (after, deltas) if return_deltas else after
        rt.train_phase = phase
        return rt

    def test_real_new_A_new_B_and_plain_A_new_B_resume_without_extra_A_history(self):
        for cell in ('new_a_new_b_joint', 'plain_new_b_joint'):
            with self.subTest(cell=cell), tempfile.TemporaryDirectory() as tmp, patch('builtins.print'):
                rt = self.make_tiny(Path(tmp)/'full', cell, 6)
                rt.run(copy.deepcopy(rt.base_state))
                final = snapshot(rt.trainer.model)
                self.assertEqual([r['round'] for r in rt.b_transfer.records], [1,6])
                self.assertIsNone(rt.b_transfer.pending)
                progress = load_json(rt.root/'progress.json')
                self.assertEqual(progress['b_transfer_optimizer_steps'], 4)
                protocol.audit_b(rt.root, rt.job, progress, 6)
                if cell == 'plain_new_b_joint':
                    self.assertIsNone(rt.history)
                    self.assertEqual(progress['functional_correction_steps'], 0)
                    self.assertEqual(rt.costs, [])
                else:
                    self.assertEqual(rt.summaries[5]['history_last_used'], 5)
                for rnd in (1,6):
                    payload = torch.load(rt.root/f'b_transfer_rounds/r{rnd:03d}/commit.pt', weights_only=False)
                    for k in rt.a_keys:
                        torch.testing.assert_close(payload['ordinary_lora'][k], payload['committed_lora'][k], atol=0, rtol=0)
                paused = self.make_tiny(Path(tmp)/'resume', cell, 4)
                paused.run(copy.deepcopy(paused.base_state))
                resumed = self.make_tiny(paused.root, cell, 6)
                resumed.resume_payload = torch.load(paused.root/'checkpoints/sfra_last.pt', weights_only=False)
                resumed.run(copy.deepcopy(resumed.base_state))
                for k, value in final.items():
                    torch.testing.assert_close(value, resumed.trainer.model.state_dict()[k], atol=0, rtol=0)
                self.assertEqual([r['round'] for r in resumed.b_transfer.records], [1,6])
                self.assertIsNone(resumed.b_transfer.pending)


if __name__ == '__main__':
    unittest.main()
