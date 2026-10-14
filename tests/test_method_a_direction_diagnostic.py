import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from tests import test_method_a_margin as fixtures
from utils.cliplora_functional_feedback import snapshot
from utils.method_a_margin_math import SampleMarginHistory, make_targets
from scripts.diagnose_method_a_direction import capture_feedback, summarize_candidate, transition_summary


class TestDirectionDiagnostic(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def test_per_sample_observation_preserves_original_joint_gradient(self):
        with tempfile.TemporaryDirectory() as temp, patch('builtins.print'):
            rt = fixtures.TestMethodAMargin().make_runtime(Path(temp), 80)
            state = snapshot(rt.trainer.model)
            history = SampleMarginHistory(torch.ones(len(rt.margin_feedback.manifest), 2)*.02,
                                          rt.margin_feedback.manifest)
            targets = make_targets(history, rt.margin_feedback.base_mass, .01, .01)
            expected = rt.observe_margin(state, 80, 'reference', targets=targets, gradient=True, diagnostics=True)
            observed = capture_feedback(rt, state, targets, 'capture', gradient=True)
            torch.testing.assert_close(expected['gradient'], observed['gradient'], atol=0, rtol=0)
            self.assertEqual(expected['class_loss'], observed['class_loss'])
            self.assertEqual(expected['keep_loss'], observed['keep_loss'])
            self.assertAlmostEqual(observed['stable_class_loss'], expected['class_loss'], places=5)
            torch.testing.assert_close(torch.tensor(observed['stable_keep_loss']), torch.tensor(expected['keep_loss']), rtol=1e-6, atol=1e-8)
            self.assertTrue(torch.equal(observed['la_margins']>0, observed['la_correct']))
            self.assertTrue(torch.isfinite(observed['sample_ce']).all())
            middle = copy.deepcopy(state)
            for key in rt.a_keys:
                middle[key] -= .01
            row = summarize_candidate(rt, observed, observed, targets, state, middle, state, 0.)
            self.assertEqual(row['delta_A_norm'], 0.)
            self.assertEqual(row['effective_delta_norm'], 0.)
            self.assertEqual(row['new_deficits'], 0)
            self.assertEqual(row['repaired_deficits'], 0)
            self.assertEqual(row['flip_details'], [])

    def test_historical_transitions_include_currently_wrong_samples(self):
        # A valid historical sample may already be wrong at ordinary A; count its repair.
        old = torch.tensor([[False, True], [True, True], [True, False]])
        new = torch.tensor([[True, True], [False, True], [False, True]])
        valid = torch.tensor([True, True, False])
        row = transition_summary(old, new, valid)
        self.assertEqual(row['original_view']['correct_to_wrong'], 1)
        self.assertEqual(row['original_view']['wrong_to_correct'], 1)
        self.assertEqual(row['views']['count'], 4)
        self.assertEqual(row['both_views_correct']['before_correct'], 1)
        self.assertEqual(row['both_views_correct']['after_correct'], 1)


if __name__ == '__main__':
    unittest.main()
