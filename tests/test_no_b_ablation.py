"""Check frozen/no-B launch wiring and reports against mismatched references."""
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from scripts import run_no_b_ablation as launcher
from scripts import run_cliplora_joint_aggregation as single
from scripts.run_cliplora_ab_parameter_search import IDENTITIES


class TestNoBAblation(unittest.TestCase):
    def reference(self):
        return dict(settings=dict(seed=42, lambda_value=.1, client_concurrency=8,
            parallel_validation='off', reference_run='references/full10_clientlt', data_root='DATA', num_workers=8,
            factorial_experiment='ab_factorial_joint_v1', factorial_cell='new_a_joint',
            margin_parameters=dict(keep_weight=1.,learn_weight=3.,correction_step_size=1.)),
            contract={}, code_sha256={'unchanged.py':'abc'}, aggregation={'rule':'joint'})

    def test_frozen_command_cannot_activate_A_or_either_extra_B(self):
        ref = self.reference()
        original = copy.deepcopy(ref)
        plan = launcher.frozen_plan(ref, ref['code_sha256'])
        self.assertEqual(ref,original)
        job = single.make_job(plan,'frozen')
        from tools.client_aggregation.runtime import runtime_class
        from utils.cliplora_sfra import SFRARuntime
        self.assertTrue(issubclass(runtime_class(job),SFRARuntime))
        cmd = single.command_for(job,Path('tmp/no_b_test'))
        self.assertEqual(cmd[cmd.index('--sfra_variant')+1],'s')
        self.assertEqual(cmd[cmd.index('--sfra_resume')+1],'')
        self.assertNotIn('--b_transfer_enable',cmd)
        self.assertNotIn('factorial_experiment',plan['settings'])
        self.assertNotIn('a_schedule_experiment',plan['settings'])
        with self.assertRaisesRegex(ValueError,'exact reference'):
            launcher.frozen_plan(ref,{'unchanged.py':'changed'})
        ref['settings']['old_b_experiment']='accidental'
        with self.assertRaisesRegex(ValueError,'Unexpected settings'):
            launcher.frozen_plan(ref,ref['code_sha256'])

    def test_startup_requires_matching_B_data_and_predictions_without_A_witnesses(self):
        with tempfile.TemporaryDirectory() as temp:
            root, ref = Path(temp)/'frozen', Path(temp)/'newA'
            plan = self.reference()
            meta = {k:'same' for k in IDENTITIES}
            for folder,suffix in ((root,launcher.FROZEN_RUN),(ref,launcher.A_RUN)):
                launcher.save(folder/'experiment_plan.json',plan)
                launcher.save(folder/suffix/'bridge_metadata.json',meta)
                launcher.save(folder/suffix/'command.json',['--sfra_resume',''])
                p = folder/suffix/'predictions';p.mkdir()
                np.savez(p/'r000.npz',**{k:np.arange(4) for k in ('sample_id','class_id','prediction','correct')})
            cfg = [dict(refresh_rounds=[],client_execution={'max_concurrent_clients':8},base_training_config={'normal_b_lr':.001}),
                   dict(refresh_rounds=list(range(1,91)),client_execution={'max_concurrent_clients':8},base_training_config={'normal_b_lr':.001})]
            with patch.object(single,'make_job',return_value={}):
                with patch('tools.client_aggregation.protocol.audit_runtime_config',side_effect=cfg):
                    self.assertTrue(launcher.compare_startup(root,ref)['passed'])
                cfg[1]['base_training_config']['normal_b_lr']=.002
                with patch('tools.client_aggregation.protocol.audit_runtime_config',side_effect=cfg):
                    with self.assertRaisesRegex(ValueError,'Ordinary training'):
                        launcher.compare_startup(root,ref)
                cfg[1]['base_training_config']['normal_b_lr']=.001
                cfg[1]['b_transfer']={'enabled':True}
                with patch('tools.client_aggregation.protocol.audit_runtime_config',side_effect=cfg):
                    with self.assertRaisesRegex(ValueError,'Extra B'):
                        launcher.compare_startup(root,ref)

    def test_report_keeps_both_runs_and_frozen_zero_cost_without_B_csv(self):
        with tempfile.TemporaryDirectory() as temp:
            root, ref = Path(temp)/'frozen', Path(temp)/'newA'
            launcher.save(root/'ablation_plan.json',dict(reference_root=str(ref),evaluation='development',
                interpretation='A training plus correction',endpoint='81..100'))
            for folder in (root,ref):launcher.save(folder/'experiment_plan.json',{})
            result = dict(curves=[dict(round=r,overall_acc=70.,head20_acc=71.,middle60_acc=69.,bottom20_tail_acc=72.)
                for r in range(81,101)],progress=dict(normal_optimizer_steps=105600,extra_optimizer_steps=0,
                    functional_correction_steps=0,elapsed_seconds=100.))
            with patch.object(single,'make_job',return_value={}),patch.object(launcher,'compare_startup',return_value={'passed':True}), \
                 patch('tools.client_aggregation.analysis.audit_run',return_value=result), \
                 patch('tools.client_aggregation.protocol.read_csv',return_value=[]), \
                 patch('scripts.run_cliplora_ab_parameter_search.actual_corrections',return_value={'actual_A_norm':.01}):
                launcher.summarize(root)
            report=launcher.load(root/'analysis/report.json')
            self.assertEqual(len(report['performance']),2)
            self.assertEqual(report['performance'][0]['A_repair_seconds_total'],0.)
            self.assertTrue(launcher.load(root/'finalization_receipt.json')['completed_and_audited'])


if __name__=='__main__':unittest.main()
