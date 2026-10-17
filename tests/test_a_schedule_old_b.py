"""Real optimization across schedule boundaries, history windows and old-B events."""
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from tests import test_new_a_old_b as fixture
from scripts import run_cliplora_joint_aggregation as single
from tools.client_aggregation import a_schedule as schedule
from tools.client_aggregation.analysis import expected_stages
from tools.client_aggregation.protocol import code_hashes, load_json, read_csv
from tools.client_aggregation.runtime import runtime_class
from utils.cliplora_functional_feedback import snapshot
from utils.cliplora_sfra import SFRARuntime
from utils.cliplora_b_shared_transfer import SharedDonorBTransfer


class TestAScheduleOldB(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads=torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def test_plans_preserve_selected_A_and_old_B_and_phase_budgets(self):
        reference=fixture.TestNewAOldB().reference_plan()
        reference['settings']['margin_parameters'].update(learn_weight=3.,correction_step_size=1.)
        hashes=code_hashes(Path.cwd())
        for cell,rounds in schedule.SCHEDULES.items():
            plan=schedule.candidate_plan(reference,cell,hashes)
            job=single.make_job(plan,'method_a_margin')
            schedule.validate(job)
            self.assertEqual(schedule.rounds_for(job),rounds)
            self.assertEqual(plan['contract']['extra_A_steps'],352*len(rounds))
            self.assertEqual(plan['contract']['old_B']['learning_rate'],.3)
            self.assertEqual(plan['contract']['old_B']['rounds'],list(range(30,101,10)))
            stages=set(expected_stages(job,100))
            self.assertIn((100,'after_B_transfer'),stages)
            self.assertEqual((70,'ordinary_A') in stages,70 in rounds)
            cmd=single.command_for(job,Path('tmp/test_schedule'))
            self.assertEqual(cmd[cmd.index('--a_margin_correction_step_size')+1],'1.0')
            self.assertNotIn('--b_transfer_enable',cmd)
            bad=copy.deepcopy(job);bad['contract']['old_B']['learning_rate']=.1
            with self.assertRaises(ValueError):schedule.validate(bad)

    def make_tiny(self,folder,cell,last):
        rt=fixture.TestNewAOldB().make_tiny(folder,last)
        job=dict(arm='method_a_margin',mode='formal',
            settings=dict(a_schedule_experiment=schedule.EXPERIMENT,a_schedule_cell=cell),
            aggregation={'rule':'joint','counts':{'tail_ids':[0]}})
        rt.__class__=runtime_class(job)
        rt.job=job
        rt.sfra_config.update(learn_weight=3.,correction_step_size=1.)
        schedule.configure(rt,job)
        return rt

    def test_actual_100_rounds_freeze_exactly_and_preserve_old_B(self):
        for cell,rounds in schedule.SCHEDULES.items():
            with self.subTest(cell=cell),tempfile.TemporaryDirectory() as tmp,patch('builtins.print'):
                rt=self.make_tiny(Path(tmp)/'full',cell,100)
                rt.run(copy.deepcopy(rt.base_state))
                progress=load_json(rt.root/'progress.json')
                self.assertIs(type(rt.b_transfer),SharedDonorBTransfer)
                self.assertEqual([r['round'] for r in rt.b_transfer.records],list(range(30,101,10)))
                fixture.protocol.audit_b(rt.root,rt.job,progress,100)
                self.assertEqual(progress['normal_optimizer_steps'],600)
                self.assertEqual(progress['extra_optimizer_steps'],2*len(rounds))
                schedule.audit_rounds(read_csv(rt.root/'sfra_rounds.csv'),progress,100,rt.job)
                seen=read_csv(rt.root/'a_schedule_audit.csv')
                self.assertEqual([int(r['round']) for r in seen if r['A_open']=='True'],rounds)
                for row in seen:
                    if int(row['round']) not in rounds:
                        self.assertEqual(row['A_before_sha256'],row['A_after_sha256'])
                if rounds:
                    self.assertEqual(rt.history.last_committed_round,max(rounds))
                    if cell=='a_every10':
                        self.assertEqual(rt.summaries[9]['history_last_used'],9)
                        self.assertEqual(rt.summaries[19]['history_last_used'],19)
                        self.assertEqual(len([c for c in rt.costs if c['phase']=='history_only']),81)
                else:
                    self.assertIsNone(rt.history)
                    self.assertEqual(rt.costs,[])
                    for key in rt.a_keys:
                        torch.testing.assert_close(rt.base_state[key],rt.trainer.model.state_dict()[key],rtol=0,atol=0)

    def test_resume_across_60_10_and_first_B_transfer_boundaries(self):
        for cell,pause in (('a_until60',59),('a_every10',9),('a_frozen',29)):
            with self.subTest(cell=cell),tempfile.TemporaryDirectory() as tmp,patch('builtins.print'):
                full=self.make_tiny(Path(tmp)/'full',cell,100)
                full.run(copy.deepcopy(full.base_state))
                final=snapshot(full.trainer.model)
                partial=self.make_tiny(Path(tmp)/'resume',cell,pause)
                partial.run(copy.deepcopy(partial.base_state))
                resumed=self.make_tiny(partial.root,cell,100)
                resumed.resume_payload=torch.load(partial.root/'checkpoints/sfra_last.pt',weights_only=False)
                resumed.run(copy.deepcopy(resumed.base_state))
                for key,value in final.items():
                    torch.testing.assert_close(value,resumed.trainer.model.state_dict()[key],atol=0,rtol=0)
                self.assertEqual(len(read_csv(resumed.root/'a_schedule_audit.csv')),100)
                self.assertEqual([r['round'] for r in resumed.b_transfer.records],list(range(30,101,10)))
                schedule.audit_rounds(read_csv(resumed.root/'sfra_rounds.csv'),load_json(resumed.root/'progress.json'),100,resumed.job)


class TestScheduleReports(unittest.TestCase):
    def test_actual_correction_reader_handles_sparse_and_fully_frozen_A(self):
        from scripts.run_cliplora_a_schedule import actual_corrections, RUN
        from scripts.run_cliplora_ab_parameter_search import save
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            save(root/RUN/'sfra_config.json',{'a_parameter_order':['x.lora_A']})
            save(root/RUN/'bridge_metadata.json',{'resolved_args':{'cliplora_alpha':2,'cliplora_rank':4}})
            for rnd in range(10,91,10):
                before={'x.lora_A':torch.zeros(2,3),'x.lora_B':torch.ones(4,2)}
                after=dict(before,**{'x.lora_A':torch.ones(2,3)*.1})
                submitted=dict(before,**{'x.lora_A':torch.ones(2,3)*.2})
                for r,phase,value in ((rnd,'refresh_A',dict(anchor_lora_state=before,actual_after_lora_state=after)),
                                      (rnd+1,'normal_B',dict(anchor_lora_state=submitted))):
                    path=root/RUN/'events'/f'r{r:03d}_c000_main_{phase}'/'state.pt'
                    path.parent.mkdir(parents=True,exist_ok=True)
                    torch.save(value,path)
            for cell,count in (('a_every10',9),('a_frozen',0)):
                job=dict(arm='method_a_margin',settings={'a_schedule_experiment':schedule.EXPERIMENT,'a_schedule_cell':cell})
                result=actual_corrections(root,job,root/(cell+'.csv'))
                self.assertEqual(result['A_update_rounds'],count)
                self.assertAlmostEqual(result['correction_to_ordinary_A_mean_active_rounds'],1. if count else 0.)
                self.assertEqual(len(read_csv(root/(cell+'.csv'))),100)
                self.assertAlmostEqual(result['effective_LoRA_correction_norm_mean_active_rounds'],(0.48**.5) if count else 0.,places=6)

    def test_three_schedules_and_reference_summary_keeps_intentional_budget_differences(self):
        import numpy as np
        from scripts import run_cliplora_a_schedule as launcher
        from scripts.run_cliplora_ab_parameter_search import save,IDENTITIES
        from tools.client_aggregation.protocol import PAIR_HASHES
        from tools.client_aggregation.new_a_old_b import candidate_plan as old_plan
        ref=fixture.TestNewAOldB().reference_plan()
        ref['settings']['margin_parameters'].update(learn_weight=3.,correction_step_size=1.)
        from tools.client_aggregation.ab_factorial import contract_for
        ref['contract']=contract_for('new_a_joint',ref['settings']['margin_parameters'])
        ref=old_plan(ref,code_hashes(Path.cwd()))
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            meta={k:'same' for k in set(IDENTITIES)|set(PAIR_HASHES)}
            save(root/'schedule_plan.json',dict(reference_root=str(root/'reference'),reference_identity={k:meta[k] for k in IDENTITIES},
                endpoint='81..100',exploratory_schedule_choice='development',budgets='A differs'))
            sources={c:schedule.candidate_plan(ref,c,ref['code_sha256']) for c in schedule.SCHEDULES}
            sources['reference']=ref
            for cell,plan in sources.items():
                save(root/cell/'experiment_plan.json',plan)
            save(root/'reference'/launcher.RUN/'completion.json',{})
            save(root/'reference'/launcher.RUN/'joint_receipt.json',{})
            def audited(run,job):
                rounds=schedule.rounds_for(job)
                return dict(job=job,metadata=meta,config=dict(variant='margin-keep-v1',refresh_rounds=rounds,
                    client_execution={},b_transfer=ref['contract']['old_B'],base_training_config=dict(
                        candidate_rounds=rounds,extra_steps_expected=352*len(rounds),extra_trainable_factor='A' if rounds else None)),
                    curves=[dict(round=r,overall_acc=70.,head20_acc=71.,middle60_acc=69.,bottom20_tail_acc=72.) for r in range(101)],
                    progress=dict(normal_optimizer_steps=105600,extra_optimizer_steps=352*len(rounds),functional_correction_steps=3*len(rounds),elapsed_seconds=100.),
                    predictions={0:{k:np.array([0]) for k in ('sample_id','class_id','prediction','correct')}})
            with patch('tools.client_aggregation.analysis.audit_run',side_effect=audited), \
                 patch.object(launcher,'startup_audit'),patch.object(launcher,'actual_corrections',return_value={}), \
                 patch.object(launcher,'read_csv',return_value=[{'seconds':3.,'repair_seconds':2.}]):
                result=launcher.summarize(root,require_reference=True)
            self.assertEqual(len(result['performance']),4)
            self.assertEqual(len(result['comparisons']),3)
            self.assertTrue(all(p['status']=='eligible' and not p['budget_matched'] for p in result['comparisons']))
            self.assertTrue(load_json(root/'finalization_receipt.json')['full_reference_comparison_complete'])


if __name__=='__main__':
    unittest.main()
