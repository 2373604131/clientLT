"""Check independent A history, original C execution, and exact restart continuity."""
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from scripts import run_cliplora_joint_aggregation as single
from scripts import run_cliplora_new_a_old_b as launcher
from scripts import run_cliplora_ab_factorial as factorial_launcher
from scripts.run_ab_validation import write_json
from tests import test_ab_factorial as factorial_fixture
from tools.client_aggregation import new_a_old_b as protocol
from tools.client_aggregation.analysis import expected_stages
from tools.client_aggregation.protocol import load_json, read_csv, validate_arm, code_hashes
from tools.client_aggregation.runtime import runtime_class
from utils.cliplora_b_shared_transfer import SharedDonorBTransfer
from utils.cliplora_functional_feedback import snapshot


class TestNewAOldB(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def reference_plan(self):
        with tempfile.TemporaryDirectory() as tmp:
            suite = factorial_launcher.register(factorial_launcher.parse_args(['--output-root', tmp]))
            return load_json(Path(suite['cells']['new_a_joint']['root'])/'experiment_plan.json')

    def test_plan_preserves_A_and_joint_and_does_not_enable_legacy_A(self):
        reference = self.reference_plan()
        original = copy.deepcopy(reference)
        plan = protocol.candidate_plan(reference, code_hashes(Path.cwd()))
        self.assertEqual(reference, original)
        self.assertEqual(plan['aggregation'], reference['aggregation'])
        self.assertEqual(plan['contract']['method_a_margin'], reference['contract']['method_a_margin'])
        self.assertEqual(plan['contract']['method_a_margin']['correction_step_size'], .1)
        self.assertNotIn('factorial_cell', plan['settings'])
        for mode in ('formal', 'smoke'):
            job = single.make_job(plan, 'method_a_margin', mode)
            validate_arm(job)
            command = single.command_for(job, Path('tmp/disposable_old_b'))
            self.assertNotIn('--b_transfer_enable', command)
            self.assertEqual(command[command.index('--sfra_variant')+1], 'margin-keep-v1')
            self.assertTrue(protocol.enabled(job))
        bad = copy.deepcopy(job); bad['contract']['old_B']['learning_rate'] = .6
        with self.assertRaises(ValueError): validate_arm(bad)
        bad = copy.deepcopy(job); bad['settings']['factorial_experiment'] = 'ab_factorial_joint_v1'
        with self.assertRaises(ValueError): validate_arm(bad)

    def test_scheduled_stage_is_after_transfer_including_B_only_final_round(self):
        job = dict(arm='method_a_margin', mode='formal',
                   settings=dict(b_comparison_experiment=protocol.EXPERIMENT))
        stages = set(expected_stages(job, 100))
        self.assertIn((30, 'after_B_transfer'), stages)
        self.assertIn((100, 'after_B_transfer'), stages)
        self.assertNotIn((20, 'after_B_transfer'), stages)
        self.assertNotIn((100, 'ordinary_A'), stages)
        cfg = protocol.transfer_config()
        self.assertEqual(cfg['rounds'], list(range(30, 101, 10)))
        self.assertEqual((cfg['steps'], cfg['learning_rate'], cfg['regularization']), (2, .3, .001))
        self.assertEqual(cfg['commit_rule'], 'fixed_second_C_step; no evaluation-based selection')
        self.assertNotIn('route_arm', cfg)

    def make_tiny(self, folder, last):
        rt = factorial_fixture.TestFactorial().make_tiny(folder, 'new_a_new_b_joint', last)
        job = dict(arm='method_a_margin', mode='formal',
                   settings=dict(b_comparison_experiment=protocol.EXPERIMENT),
                   aggregation={'rule': 'joint', 'counts': {'tail_ids': [0]}})
        rt.__class__ = runtime_class(job)
        rt.job = job
        rt.tail = [0]  # Deterministic toy case with a genuinely positive missing-class donor.
        rt.q = {i: n/sum(rt.sizes) for i, n in enumerate(rt.sizes)}
        rt.config.update(normal_steps_expected=600, extra_steps_expected=180)
        rt.sfra_config['b_transfer'] = protocol.transfer_config('formal')
        rt.b_transfer = SharedDonorBTransfer(rt)
        return rt

    def test_actual_old_C_new_A_and_resume_are_identical(self):
        with tempfile.TemporaryDirectory() as tmp, patch('builtins.print'):
            rt = self.make_tiny(Path(tmp)/'full', 100)
            rt.run(copy.deepcopy(rt.base_state))
            final = snapshot(rt.trainer.model)
            self.assertIs(type(rt.b_transfer), SharedDonorBTransfer)
            self.assertEqual([r['round'] for r in rt.b_transfer.records], list(range(30, 101, 10)))
            self.assertIsNone(rt.b_transfer.pending)
            progress = load_json(rt.root/'progress.json')
            protocol.audit_b(rt.root, rt.job, progress, 100)
            self.assertGreater(progress['b_transfer_optimizer_steps'], 0)
            self.assertEqual(rt.summaries[89]['history_last_used'], 89)
            self.assertTrue(all(r['skip_reason'] == 'B_only_schedule' for r in rt.summaries[90:]))
            self.assertEqual(rt.history.last_committed_round, 90)
            for rnd in range(30, 101, 10):
                payload = torch.load(rt.root/f'b_transfer_rounds/r{rnd:03d}/commit.pt', weights_only=False)
                for k in rt.a_keys:
                    torch.testing.assert_close(payload['ordinary_lora'][k], payload['committed_lora'][k], atol=0, rtol=0)
            paused = self.make_tiny(Path(tmp)/'resume', 89)
            paused.run(copy.deepcopy(paused.base_state))
            resumed = self.make_tiny(paused.root, 100)
            resumed.resume_payload = torch.load(paused.root/'checkpoints/sfra_last.pt', weights_only=False)
            resumed.run(copy.deepcopy(resumed.base_state))
            for key, value in final.items():
                torch.testing.assert_close(value, resumed.trainer.model.state_dict()[key], atol=0, rtol=0)
            self.assertEqual([r['round'] for r in resumed.b_transfer.records], list(range(30, 101, 10)))
            self.assertIsNone(resumed.b_transfer.pending)

    def test_summary_consumes_actual_correction_summary_without_retraining(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            plan = protocol.candidate_plan(self.reference_plan(), {})
            write_json(root/'experiment_plan.json', plan)
            write_json(root/'reference_audit.json', {'references': []})
            result = {'curves': [dict(round=r, overall_acc=70., head20_acc=71.,
                        middle60_acc=69., bottom20_tail_acc=72.) for r in range(81, 101)], 'progress': {}}
            records = [{'repair_seconds': 2., 'seconds': 3., 'effective_global_transfer_norm': .2}]
            norms = {'actual_A_correction_norm_mean_rounds1_90': .01}
            with patch('tools.client_aggregation.analysis.audit_run', return_value=result), \
                 patch.object(launcher, 'startup_audit'), \
                 patch('scripts.run_cliplora_ab_parameter_search.actual_corrections', return_value=norms), \
                 patch.object(launcher, 'read_csv', return_value=records), patch('builtins.print'):
                launcher.summarize(root)
            report = load_json(root/'analysis/report.json')
            self.assertEqual(report['A_corrections'], norms)
            self.assertEqual(report['B_effective_LoRA_transfer_norm_mean'], .2)
            self.assertEqual(report['A_repair_seconds_total'], 2.)
            self.assertTrue(load_json(root/'finalization_receipt.json')['completed_and_audited'])


if __name__ == '__main__':
    unittest.main()
