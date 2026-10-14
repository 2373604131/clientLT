"""Numerical equivalence and no-training guards for the one-round observer."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from tests import test_method_a_margin as fixtures
from utils.cliplora_functional_feedback import snapshot
from utils.method_a_margin_math import SampleMarginHistory, make_targets
from utils.method_a_update_diagnostic import diagnose_prepared, transitions
from scripts.diagnose_method_a_round80 import replay, restore_history, worker_arguments


class TestAUpdateDiagnostic(unittest.TestCase):
    def test_recorded_python_flags_and_script_are_not_trainer_arguments(self):
        expected = ['--root', '/data/DATA', 'OPTIM.LR', '0.001']
        for prefix in (['python'], ['python', '-u'], ['python', '-u', '-B']):
            self.assertEqual(worker_arguments(prefix+['/repo/scripts/train_cliplora_joint_aggregation.py']+expected), expected)

    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def test_components_match_independent_backwards_and_do_not_change_joint(self):
        bank, feedback = fixtures.feedback_fixture(overlap=True)
        history = SampleMarginHistory(torch.ones(len(feedback.manifest), 2)*.2, feedback.manifest)
        targets = make_targets(history, feedback.base_mass, .01, .2)
        ordinary = feedback.evaluate('normal', targets, True, 1.7, .4)
        rng = torch.get_rng_state().clone()
        observed = feedback.evaluate('observed', targets, True, 1.7, .4, diagnostics=True)
        torch.testing.assert_close(ordinary['gradient'], observed['gradient'], atol=0, rtol=0)
        for name, weights in (('keep', (1.7, 0.)), ('learn', (0., .4))):
            independent = feedback.evaluate(name, targets, True, *weights)
            torch.testing.assert_close(observed['component_gradients'][name], independent['gradient'])
        torch.testing.assert_close(sum(observed['component_gradients'].values()), observed['gradient'], atol=1e-5, rtol=1e-5)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertEqual(observed['correct'].shape, (len(feedback.manifest), 2))

    def test_same_three_step_commit_real_writeback_effective_norms_and_preserved_state(self):
        with tempfile.TemporaryDirectory() as temp, patch('builtins.print'):
            rt = fixtures.TestMethodAMargin().make_runtime(Path(temp)/'run', 80)
            rt.history = SampleMarginHistory(torch.ones(len(rt.margin_feedback.manifest), 2)*.2,
                                             rt.margin_feedback.manifest)
            rt.history.last_committed_round = 79
            rt.history.block_count = 4
            rt.history.block_min.fill_(.2)
            original_history = copy.deepcopy(rt.history.state_dict())
            middle = snapshot(rt.trainer.model)
            ordinary = copy.deepcopy(middle)
            for k in rt.a_keys:
                ordinary[k] += .0001
            with patch.object(rt, 'train_phase', side_effect=AssertionError('No training')):
                expected, _ = rt.correct_ordinary(middle, ordinary, .011882410384714603, 80)
                rt.history = SampleMarginHistory.restore(original_history, rt.margin_feedback.manifest)
                rt.load_training_state(middle)
                rng = torch.get_rng_state().clone()
                actual, _, result = diagnose_prepared(rt, middle, ordinary, .011882410384714603, 80, Path(temp)/'diag')
            for key in rt.keys:
                torch.testing.assert_close(actual[key], expected[key], atol=0, rtol=0)
                torch.testing.assert_close(rt.trainer.model.state_dict()[key], middle[key], atol=0, rtol=0)
            self.assertTrue(torch.equal(rng, torch.get_rng_state()))
            self.assertEqual(rt.history.last_committed_round, 79)
            torch.testing.assert_close(rt.history.H, original_history['H'], atol=0, rtol=0)
            self.assertEqual(len(result['steps']), 3)
            self.assertGreater(result['delta_A_norm'], 0)
            self.assertGreater(result['repair_effective_weight_norm'], 0)
            self.assertEqual(result['loaded_model_writeback_max_abs_error'], 0)
            self.assertTrue(result['B_unchanged'])
            self.assertTrue(all(s['predicted_actual_z_error']==0 for s in result['steps']))
            self.assertTrue((Path(temp)/'diag/diagnosed_commit.pt').is_file())
            json.loads((Path(temp)/'diag/diagnosis.json').read_text())
            rt.history.last_committed_round = 90
            with self.assertRaisesRegex(ValueError, 'preceding history'):
                diagnose_prepared(rt, middle, ordinary, .01, 80, Path(temp)/'bad')

    def test_transitions_distinguish_views_from_samples(self):
        before = torch.tensor([[True, True], [False, True], [True, False]])
        after = torch.tensor([[False, True], [True, True], [False, True]])
        result = transitions(before, after)
        self.assertEqual(result['views']['correct_to_wrong'], 2)
        self.assertEqual(result['views']['wrong_to_correct'], 2)
        self.assertEqual(result['both_views_correct']['wrong_to_correct'], 1)

    def test_history_reconstruction_uses_next_round_anchor_without_training(self):
        with tempfile.TemporaryDirectory() as temp, patch('builtins.print'):
            rt = fixtures.TestMethodAMargin().make_runtime(Path(temp)/'run', 6)
            n = len(rt.margin_feedback.manifest)
            margins = [torch.ones(n, 2)*(.02+.001*r) for r in range(6)]
            reference = SampleMarginHistory(margins[0], rt.margin_feedback.manifest)
            logs = []
            for rnd in range(1, 6):
                valid = int(reference.valid.sum())
                changed = reference.commit(margins[rnd], rnd)
                logs.append(dict(round=str(rnd), valid_history_count=str(valid),
                                 history_count_after=str(int(reference.valid.sum())),
                                 history_registered=str(int(changed.sum()))))
            with patch('scripts.diagnose_method_a_round80.rows', return_value=logs), \
                 patch('scripts.diagnose_method_a_round80.saved_event', return_value={'anchor_lora_state':{}}) as events, \
                 patch.object(rt, 'observe_margin', side_effect=[{'sample_margins':m} for m in margins]):
                recovered, source = restore_history(rt, Path(temp), 6)
            self.assertEqual([c.args[1] for c in events.call_args_list], list(range(1, 7)))
            self.assertEqual(recovered.last_committed_round, 5)
            torch.testing.assert_close(recovered.H, reference.H, atol=0, rtol=0)
            self.assertIn('forward_only', source)

    def test_missing_tensor_inputs_stop_before_worker_initialization(self):
        from types import SimpleNamespace
        with tempfile.TemporaryDirectory() as temp:
            args = SimpleNamespace(run=Path(temp)/'missing', output=Path(temp)/'out', round=80,
                                   history_checkpoint=None)
            with self.assertRaisesRegex(ValueError, 'No training started'):
                replay(args, {})

    def test_projection_observer_reports_projected_actual_not_raw_sgd(self):
        from types import SimpleNamespace
        from utils.method_a_update_diagnostic import AUpdateDiagnostic
        from utils.sfra_math import projected_step
        observer = AUpdateDiagnostic('unused')
        observer.radius, observer.lr = 3., .1
        observer.rt = SimpleNamespace(a_keys=['a'])
        observer.ordinary = {'a': torch.zeros(2)}
        z, g = torch.zeros(2), torch.ones(2)*100
        actual = projected_step(z, 3., g, 1., .1)
        observer.step(0, z, actual, dict(gradient=g, component_gradients={'keep':g, 'learn':g*0},
                      keep_loss=1., class_loss=0., diagnostic_extra_backward_calls=2), {'a':3*actual})
        result = observer.result['steps'][0]
        self.assertTrue(result['projection_triggered'])
        self.assertIsNone(result['grad_cosine'])
        self.assertEqual(result['predicted_actual_z_error'], 0)
        self.assertAlmostEqual(result['actual_increment_A_ideal_norm'], 3., places=5)
        self.assertGreater(result['predicted_first_SGD_A_norm'], result['actual_increment_A_ideal_norm'])


if __name__ == '__main__':
    unittest.main()
