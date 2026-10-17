"""Protect selected-A wiring and automatic three-way reporting for an absent B."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

import numpy as np

from scripts import run_selected_a_b_comparison as report
from scripts import run_cliplora_joint_aggregation as single
from scripts.run_cliplora_ab_parameter_search import candidate_plan, save
from scripts.run_cliplora_new_a_old_b import IDENTITIES
from tests.test_new_a_old_b import TestNewAOldB
from tools.client_aggregation.protocol import code_hashes, load_json


class TestSelectedComparison(unittest.TestCase):
    def setUp(self):
        report.load_json = load_json
        report.save = save

    def test_disabling_B_preserves_selected_S4_and_wires_actual_command(self):
        original = TestNewAOldB().reference_plan()
        from tools.client_aggregation.ab_factorial import contract_for, has_new_b
        original['settings']['factorial_cell'] = 'new_a_new_b_joint'
        original['contract'] = contract_for('new_a_new_b_joint', original['settings']['margin_parameters'])
        selected = candidate_plan(original, 'S4', code_hashes(Path.cwd()))
        plan = report.no_b_plan(selected, selected['code_sha256'])
        job = single.make_job(plan, 'method_a_margin')
        self.assertFalse(has_new_b(job))
        self.assertEqual(plan['contract']['method_a_margin'], selected['contract']['method_a_margin'])
        self.assertEqual(plan['aggregation'], selected['aggregation'])
        command = single.command_for(job, Path('tmp/test_selected_no_B'))
        self.assertEqual(float(command[command.index('--a_margin_learn_weight')+1]), 3.)
        self.assertEqual(float(command[command.index('--a_margin_correction_step_size')+1]), 1.)
        self.assertNotIn('--b_transfer_enable', command)

    def test_startup_accepts_absent_B_and_rejects_changed_predictions(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, ref = Path(tmp)/'new', Path(tmp)/'ref'
            plan = {'contract': {'method_a_margin': {'learn_weight': 3.}}}
            meta = {k: k for k in IDENTITIES}
            save(root/'experiment_plan.json', plan)
            save(root/report.RUN/'bridge_metadata.json', meta)
            save(root/report.RUN/'command.json', ['--sfra_resume', ''])
            save(root/'reference_audit.json', {'references': [{'root': str(ref), 'identity': meta}]})
            values = {k: np.arange(3) for k in ('sample_id', 'class_id', 'prediction', 'correct')}
            for folder in (root, ref):
                for name in ('private_witness_manifest.json', 'margin_witness_manifest.json'):
                    save(folder/report.RUN/name, {'samples': [1, 2]})
                p = folder/report.RUN/'predictions'
                p.mkdir()
                np.savez(p/'r000.npz', **values)
            with patch.object(single, 'make_job', return_value={}), \
                 patch('tools.client_aggregation.protocol.audit_runtime_config', return_value={}), patch('builtins.print'):
                value = report.startup_audit(root)
                self.assertIsNone(value['B'])
                self.assertTrue(value['passed'])
                np.savez(root/report.RUN/'predictions/r000.npz', **dict(values, prediction=np.zeros(3)))
                with self.assertRaisesRegex(ValueError, 'predictions mismatch'):
                    report.startup_audit(root)

    def test_complete_summary_handles_no_B_cost_and_retains_all_three(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            save(root/'comparison_plan.json', dict(reused_current_B_root=str(root/'current'),
                evaluation='development only', selection='fixed prior rule', endpoint='81..100'))
            for cell in (*report.CELLS, 'current'):
                save(root/cell/'experiment_plan.json', {})
            result = dict(curves=[dict(round=r, overall_acc=70., head20_acc=71., middle60_acc=69.,
                bottom20_tail_acc=72.) for r in range(81, 101)], progress={'elapsed_seconds': 100.})
            def records(path):
                if path.name == 'sfra_rounds.csv':
                    return [{'repair_seconds': 2.}]
                return [] if 'new_a_joint' in path.parts else [{'seconds': 3., 'effective_global_transfer_norm': .2}]
            pair = dict(status='eligible', mismatches=[])
            with patch.object(single, 'make_job', return_value={}), patch.object(report, 'startup_audit'), \
                 patch('tools.client_aggregation.analysis.audit_run', return_value=result), \
                 patch('tools.client_aggregation.protocol.read_csv', side_effect=records), \
                 patch('tools.client_aggregation.new_a_old_b.pair_check', side_effect=lambda *_: dict(pair)), \
                 patch('scripts.run_cliplora_ab_parameter_search.actual_corrections', return_value={'actual_A_norm': .01}):
                report.summarize(SimpleNamespace(output_root=root))
            output = load_json(root/'analysis/report.json')
            self.assertEqual(len(output['performance']), 3)
            self.assertEqual(len(output['comparisons']), 3)
            no_B = next(r for r in output['performance'] if r['method'] == 'new_a_joint')
            self.assertEqual(no_B['B_transfer_seconds_total'], 0.)
            self.assertTrue(load_json(root/'finalization_receipt.json')['all_three_completed_and_audited'])


if __name__ == '__main__':
    unittest.main()
