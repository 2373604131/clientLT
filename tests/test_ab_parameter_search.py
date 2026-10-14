"""The search changes only authorized coefficients and uses complete endpoints."""
import copy
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import torch

from scripts import run_cliplora_ab_parameter_search as search
from scripts import run_cliplora_ab_factorial as factorial
from scripts import run_cliplora_joint_aggregation as single
from utils.cliplora_method_a_margin import MethodAMarginRuntime
from utils.method_a_margin_math import DEFAULTS, VARIANT, algorithm_config
from utils.sfra_math import projected_step


class TestParameterSearch(unittest.TestCase):
    def test_six_full_method_commands_preserve_ordinary_A_B_and_aggregation(self):
        with tempfile.TemporaryDirectory() as tmp:
            suite = factorial.register(factorial.parse_args(['--output-root', tmp]))
            baseline = search.load_json(Path(suite['cells']['new_a_new_b_joint']['root'])/'experiment_plan.json')
            self.assertEqual(search.PRIORITY[:3], ('S2', 'S5', 'S4'))
            for group, (keep, learn) in search.GRID.items():
                plan = search.candidate_plan(baseline, group, baseline['code_sha256'])
                self.assertEqual(plan['aggregation'], baseline['aggregation'])
                self.assertEqual(plan['contract']['new_B'], baseline['contract']['new_B'])
                self.assertEqual(plan['contract']['A_lr'], .001)
                command = single.command_for(single.make_job(plan, 'method_a_margin'), Path(tmp)/group)
                for key, expected in [('keep_weight', keep), ('learn_weight', learn), ('correction_step_size', 1.)]:
                    self.assertEqual(float(command[command.index('--a_margin_'+key)+1]), expected)
                self.assertEqual(command[command.index('--sfra_resume')+1], '')
                self.assertEqual(command[command.index('--sfra_stop_after_round')+1], '0')
            self.assertEqual(baseline['settings']['margin_parameters'], DEFAULTS)

    def test_config_reaches_runtime_z_update_and_preserves_legacy_default(self):
        self.assertEqual(algorithm_config()['correction_step_size'], .1)
        rt = object.__new__(MethodAMarginRuntime)
        rt.args = SimpleNamespace(a_margin_correction_step_size=1., sfra_fast_execution_v2=True,
                                 seed=42, split_seed=42, partition='client-longtail', sfra_witness_batch_size=8)
        rt.variant, rt.strength, rt.config, rt.a_keys = VARIANT, 1., {}, []
        rt.configure_algorithm()
        self.assertEqual(rt.sfra_config['correction_step_size'], 1.)
        z, grad = torch.tensor([.2, -.1]), torch.tensor([.3, .2])
        # With mu=1 and eta_z=1, the old z term cancels in an interior projected step.
        torch.testing.assert_close(projected_step(z, .01, grad, 1., rt.sfra_config['correction_step_size']), -.01*grad)
        for invalid in (0., -1., float('nan'), float('inf')):
            with self.assertRaises(ValueError): algorithm_config(correction_step_size=invalid)

    def test_endpoint_rejects_missing_duplicate_and_nonfinite_rounds(self):
        rows = [dict(round=r, **{m: float(r) for m in search.METRICS[:4]}) for r in range(101)]
        self.assertEqual(search.endpoint(rows)['overall_acc'], 90.5)
        for bad in (rows[:-1], rows+[rows[-1]]):
            with self.assertRaises(ValueError): search.endpoint(bad)
        bad = copy.deepcopy(rows); bad[-1]['overall_acc'] = float('nan')
        with self.assertRaises(ValueError): search.endpoint(bad)
        base = dict(overall_acc=70., bottom20_tail_acc=60.)
        self.assertFalse(search.dominates_baseline(base, base))
        self.assertFalse(search.dominates_baseline(dict(overall_acc=69., bottom20_tail_acc=65.), base))
        self.assertTrue(search.dominates_baseline(dict(overall_acc=70., bottom20_tail_acc=61.), base))

    def test_worker_refuses_existing_history_instead_of_resuming(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root/'S2'/search.RUN).mkdir(parents=True)
            with self.assertRaisesRegex(ValueError, 'fresh'):
                search.worker(root, 'S2')

    def test_actual_norms_use_saved_tensors_and_recorded_model_scaling(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            search.save(run/'sfra_config.json', dict(a_parameter_order=['w_lora_A'], base_training_config={}))
            search.save(run/'bridge_metadata.json', dict(resolved_args=dict(cliplora_alpha=1, cliplora_rank=4)))
            a, b = torch.ones(1, 2), torch.tensor([[2.], [3.]])
            before = dict(w_lora_A=torch.zeros_like(a), w_lora_B=b)
            ordinary = dict(w_lora_A=a, w_lora_B=b)
            submitted = dict(w_lora_A=1.5*a, w_lora_B=b)
            for rnd in range(1, 91):
                for r, phase, payload in [(rnd, 'refresh_A', dict(anchor_lora_state=before, actual_after_lora_state=ordinary)),
                                           (rnd+1, 'normal_B', dict(anchor_lora_state=submitted))]:
                    folder = run/'events'/f'r{r:03d}_c000_main_{phase}'
                    folder.mkdir(parents=True)
                    torch.save(payload, folder/'state.pt')
            search.write_table(run/'sfra_rounds.csv', [dict(round=r, radius=1., correction_z_norm=.7) for r in range(1,91)])
            result = search.actual_corrections(run, run/'norms.csv')
            self.assertAlmostEqual(result['correction_to_ordinary_A_mean_rounds1_90'], .5)
            self.assertAlmostEqual(result['effective_LoRA_correction_norm_mean_rounds1_90'], float((.5*(b @ (.5*a))).double().norm()))


if __name__ == '__main__':
    unittest.main()
