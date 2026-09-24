"""Numerical checks of v2 against the original individual witness gradients."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

import torch

import test_sfra_execution as fixtures
from scripts import run_cliplora_sfra as launcher
from scripts.compare_sfra_timing import compare, load_timings, verify_comparable
from utils.sfra_math import proposal_coordinates
from utils.sfra_resident_feedback import ResidentWitnessFeedback, execution_config_v2


def bank_v2(batch_size=2, forward_batch_size=64, cache_gib=4.):
    bank = fixtures.visual_bank(True, batch_size, forward_batch_size)
    bank._fast_feedback = ResidentWitnessFeedback(bank, forward_batch_size, cache_gib)
    return bank


class TestResidentFeedback(unittest.TestCase):
    compare_feedback = fixtures.TestSFRAExecution.compare_feedback

    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def test_outputs_full_independent_source_gradients_and_cp_match(self):
        for batch_size in (1, 2, 8):
            for forward in (8, 16, 64, 128):
                with self.subTest(microbatch=batch_size, forward=forward):
                    old, new = fixtures.visual_bank(batch_size=batch_size), bank_v2(batch_size, forward)
                    torch.manual_seed(51)
                    basis, _, _ = proposal_coordinates(torch.randn(old.numel, 4) * .01)
                    self.compare_feedback(old.evaluate(basis=basis), new.evaluate(basis=basis))
                    for cp in (False, True):
                        self.compare_feedback(old.evaluate(targets=fixtures.targets(), classification=cp),
                                              new.evaluate(targets=fixtures.targets(), classification=cp))
                    self.compare_feedback(old.evaluate(classification=True, classification_gradient=False),
                                          new.evaluate(classification=True, classification_gradient=False))

    def test_inactive_tokens_unequal_sizes_and_zero_coefficients(self):
        old, new = fixtures.visual_bank(), bank_v2()
        t = fixtures.targets()
        t['active'][1] = False
        t['weights'][1] = 0
        # An unused target must never enter any arithmetic.
        t['sigma'][1] = 0
        t['target'][1] = float('nan')
        for cp in (False, True):
            self.compare_feedback(old.evaluate(targets=t, all_scores=False, classification=cp),
                                  new.evaluate(targets=t, all_scores=False, classification=cp))
        t = fixtures.targets()
        t['target'].fill_(-3.)
        with patch('torch.autograd.grad', wraps=torch.autograd.grad) as grad:
            result = new.evaluate(targets=t, classification=True)
            self.assertEqual(grad.call_count, 2 * result['execution_forward_batches'])
        self.assertEqual(result['execution_zero_gradient_batches'], 0)
        self.assertEqual(result['execution_zero_coefficient_batches'], result['execution_forward_batches'])
        self.compare_feedback(old.evaluate(targets=t, classification=True), result)
        t['active'].fill_(False)
        t['weights'].zero_()
        self.compare_feedback(old.evaluate(targets=t, all_scores=False),
                              new.evaluate(targets=t, all_scores=False))

    def test_cache_reuse_invalidation_and_no_state_or_rng_changes(self):
        b = bank_v2(cache_gib=0.)
        fast = b._fast_feedback
        state = copy.deepcopy(b.model.state_dict())
        rng = torch.get_rng_state().clone()
        flags = [p.requires_grad for p in b.model.parameters()]
        modes = [m.training for m in b.model.modules()]
        with patch.object(fast.prefix, 'forward_prefix', wraps=fast.prefix.forward_prefix) as encode:
            b.evaluate(targets=fixtures.targets(), classification=True)
            count = encode.call_count
            size = fast.cpu_cache_bytes
            self.assertGreater(size, 0)
            self.assertEqual(fast.device_cache_bytes, 0)
            b.evaluate(targets=fixtures.targets(), classification=True)
            self.assertEqual(encode.call_count, count)
            self.assertEqual(size, fast.cpu_cache_bytes)
            self.assertEqual(flags, [p.requires_grad for p in b.model.parameters()])
            self.assertEqual(modes, [m.training for m in b.model.modules()])
            self.assertTrue(torch.equal(rng, torch.get_rng_state()))
            for key, value in b.model.state_dict().items():
                self.assertTrue(torch.equal(state[key], value))
            with torch.no_grad():
                b.core.image_encoder.conv1.weight.add_(.01)
            b.evaluate()
            self.assertEqual(encode.call_count, count * 2)
            self.assertEqual(size, fast.cpu_cache_bytes)

    def test_nonfinite_source_rejected_before_return_and_exception_restores_state(self):
        b = bank_v2()
        basis, _, _ = proposal_coordinates(torch.randn(b.numel, 4))
        with torch.no_grad():
            b.core.image_encoder.conv1.weight.fill_(float('nan'))
        flags = [p.requires_grad for p in b.model.parameters()]
        modes = [m.training for m in b.model.modules()]
        rng = torch.get_rng_state().clone()
        for kwargs in ({'basis': basis}, {'targets': fixtures.targets(), 'classification': True}):
            with self.assertRaises(FloatingPointError):
                b.evaluate(**kwargs)
            self.assertEqual(flags, [p.requires_grad for p in b.model.parameters()])
            self.assertEqual(modes, [m.training for m in b.model.modules()])
            self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        b = bank_v2()
        with patch.object(b._fast_feedback.prefix, 'forward_suffix', side_effect=RuntimeError('failure')):
            with self.assertRaisesRegex(RuntimeError, 'failure'):
                b.evaluate(targets=fixtures.targets())
        self.assertEqual(flags, [p.requires_grad for p in b.model.parameters()])
        self.assertEqual(modes, [m.training for m in b.model.modules()])

    def test_real_runtime_three_corrections_and_history_match_original(self):
        original = fixtures.visual_bank

        def substitute(fast=False, batch_size=2, forward_batch_size=12):
            bank = original(fast, batch_size, forward_batch_size)
            if fast:
                bank._fast_feedback = ResidentWitnessFeedback(bank, 64)
            return bank

        audit = fixtures.TestSFRAExecution()
        with patch.object(fixtures, 'visual_bank', substitute):
            audit.test_actual_runtime_refresh_and_execution_checkpoint_metadata()
            audit.test_full_three_corrections_and_history_decisions_match()

    def test_launcher_separates_v2_tuning_and_replays_saved_settings(self):
        with tempfile.TemporaryDirectory() as tmp:
            def prepare(args, run):
                (run / 'protocol').mkdir(parents=True)
                return run / 'protocol/full_schedule.json', ''

            argv = ['run_cliplora_sfra.py', '--method', 'full-cp', '--output-root', tmp,
                    '--fast-execution-v2', '--feedback-batch-size', '128', '--feedback-cache-gib', '4',
                    '--stop-after-round', '3']
            with patch('sys.argv', argv), patch.object(launcher, 'prepare_protocol', prepare), \
                    patch.object(launcher.subprocess, 'run') as execute, patch('builtins.print'):
                launcher.main()
            command = execute.call_args.args[0]
            self.assertIn('--sfra_fast_execution_v2', command)
            self.assertEqual(command[command.index('--sfra_feedback_batch_size')+1], '128')
            self.assertEqual(command[command.index('--train_batch_size')+1], '32')
            run = Path(command[command.index('--output-dir')+1])
            self.assertTrue(run.name.endswith('_fast_v2_f128_c4'))
            self.assertEqual(command, json.loads((run/'command.json').read_text()))
            (run/'checkpoints').mkdir()
            (run/'checkpoints/sfra_last.pt').touch()
            with patch('sys.argv', argv+['--resume']), patch.object(launcher.subprocess, 'run') as execute, \
                    patch('builtins.print'):
                launcher.main()
            expected = list(command)
            expected[expected.index('--sfra_resume')+1] = str(run/'checkpoints/sfra_last.pt')
            self.assertEqual(execute.call_args.args[0], expected)
            with patch('sys.argv', argv+['--resume', '--stop-after-round', '0']), \
                    patch.object(launcher.subprocess, 'run') as execute, patch('builtins.print'):
                launcher.main()
            expected[expected.index('--sfra_stop_after_round')+1] = '0'
            self.assertEqual(execute.call_args.args[0], expected)
        self.assertEqual(execution_config_v2()['device_cache_gib'], 4.)
        for batch, cache in ((0, 4), (4, 4), (64, -1), (64, float('nan')), (64, 4.01), (64, 8)):
            with self.assertRaises(ValueError):
                execution_config_v2(batch, cache)

    def test_short_run_pauses_at_checkpoint_and_resume_preserves_round_numbers(self):
        rt = object.__new__(fixtures.SFRARuntime)
        rt.args = SimpleNamespace(sfra_stop_after_round=3)
        rt.trainer = SimpleNamespace(model=torch.nn.Linear(2, 1))
        rt.keys = ['weight', 'bias']
        rt.bank = rt.b_transfer = rt.resume_payload = None
        rt.variant, rt.strength = 's', 10.
        checkpoints, phases = [], []
        rt.load_training_state = lambda state: None
        rt.publish = lambda *args: None
        rt.checkpoint = lambda state, rnd: checkpoints.append(rnd)
        def train(state, rnd, factor, extra=False):
            phases.append((rnd, factor))
            return state
        rt.train_phase = train
        with patch('builtins.print'):
            rt.run(rt.trainer.model.state_dict())
        self.assertEqual(checkpoints, [0, 1, 2, 3])
        self.assertEqual(phases, [(r, f) for r in range(1, 4) for f in ('B', 'A')])
        rt.resume_payload = {'completed_round': 3}
        rt.restore = lambda: (rt.trainer.model.state_dict(), 3)
        rt.args.sfra_stop_after_round = 5
        with patch('builtins.print'):
            rt.run(None)
        self.assertEqual(checkpoints, [0, 1, 2, 3, 4, 5])
        self.assertEqual(phases[-4:], [(4, 'B'), (4, 'A'), (5, 'B'), (5, 'A')])

    def test_timing_comparison_matches_rounds_and_rejects_different_protocols(self):
        with tempfile.TemporaryDirectory() as tmp:
            left, right = Path(tmp)/'old', Path(tmp)/'new'
            for root, divisor, completed in ((left, 1, 4), (right, 2, 3)):
                (root/'protocol').mkdir(parents=True)
                (root/'sfra_config.json').write_text('{"seed":42}')
                (root/'partition_manifest.csv').write_text('same partition')
                (root/'protocol/full_schedule.json').write_text('[]')
                (root/'bridge_metadata.json').write_text(json.dumps(dict(
                    initial_lora_sha256='a', frozen_model_sha256='b', test_sha256='c')))
                (root/'progress.json').write_text(json.dumps(dict(completed_round=completed, elapsed_seconds=100)))
                (root/'event_manifest.csv').write_text('round,phase,seconds\n' + ''.join(
                    f'{r},normal_B,{20/divisor}\n' for r in range(1, 5)))
                (root/'sfra_costs.csv').write_text('round,phase,seconds\n' + ''.join(
                    f'{r},correction_1,{10/divisor}\n' for r in range(1, 5)))
                (root/'evaluation_budget.csv').write_text('label,seconds\n' + ''.join(
                    f'official_round_{r},{2/divisor}\n' for r in range(5)))
            self.assertNotIn(4, load_timings(right)[1])
            with patch('builtins.print'):
                totals = compare(left, right)
            self.assertEqual(totals, [(10, 5), (32, 16)])
            (right/'protocol/full_schedule.json').write_text('[1]')
            with self.assertRaisesRegex(ValueError, 'protocol differs'):
                verify_comparable(left, right)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA cache placement requires a GPU')
    def test_cuda_cache_budget_mixed_fallback_and_numerical_equivalence(self):
        banks = [bank_v2(cache_gib=v) for v in (0., 700 / 1024**3, 1.)]
        results = []
        for b in banks:
            b.device = torch.device('cuda')
            b.model.to(b.device)
            b.parameters = [dict(b.model.named_parameters())[k] for k in b.keys]
            b.text_features = b.text_features.to(b.device)
            b.configure_classification(torch.tensor([[50, 0, 2], [0, 3, 0]]), torch.tensor([-.1, -.7, -.8]))
            b._fast_feedback.reserve_bytes = 0
            results.append(b.evaluate(targets=fixtures.targets(), classification=True))
        for result in results[1:]:
            self.compare_feedback(results[0], result)
        self.assertEqual(banks[0]._fast_feedback.device_cache_bytes, 0)
        mixed = banks[1]._fast_feedback
        self.assertGreater(mixed.device_cache_bytes, 0)
        self.assertLessEqual(mixed.device_cache_bytes, 700)
        self.assertGreater(mixed.cpu_cache_bytes, 0)
        self.assertEqual(banks[2]._fast_feedback.cpu_cache_bytes, 0)


if __name__ == '__main__':
    unittest.main()
