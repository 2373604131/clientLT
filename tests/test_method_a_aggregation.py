"""CPU checks of the Method-A-only integration; fixtures are not experiments."""
import copy
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from scripts import run_cliplora_method_a_aggregation as launcher
from scripts import run_cliplora_plain_a_aggregation as suite
from scripts import run_cliplora_joint_aggregation as single
from scripts.run_ab_validation import write_json
from tests import test_joint_aggregation as fixtures, test_sfra_v1 as sfra_fixtures
from tests.test_joint_aggregation import TinyModel
from tools.client_aggregation.analysis import audit_run, stage_comparisons, pack_results
from tools.client_aggregation.method_a import compare_plain, summarize
from tools.client_aggregation.protocol import METHOD_A_CONTRACT, PAIR_HASHES, load_json, write_table
from tools.client_aggregation.runtime import runtime_class
from utils.cliplora_functional_feedback import snapshot
from utils.sfra_math import FunctionalHistory


class TestMethodAAggregation(unittest.TestCase):
    def test_registered_three_rules_use_full_cp_without_B_transfer(self):
        self.assertEqual(launcher.parse_args(['--client-concurrency','8']).output_root.name,
                         'method_a_aggregation_v1_parallel8')
        with tempfile.TemporaryDirectory() as temp:
            args=launcher.parse_args(['--output-root',temp,'--client-concurrency','8'])
            plans=suite.register(args)
            self.assertEqual(plans,suite.register(args))
            for rule,plan in plans.items():
                self.assertEqual(plan['contract'],METHOD_A_CONTRACT)
                job=single.make_job(plan,'method_a')
                command=single.command_for(job,Path(temp)/rule)
                self.assertNotIn('--b_transfer_enable',command)
                for key,value in {'--sfra_variant':'full-cp','--sfra_retention_weight':'10',
                                  '--sfra_classification_weight':'1','--cliplora_freeze_a':'True'}.items():
                    self.assertEqual(command[command.index(key)+1],value)
                self.assertIn('run_cliplora_method_a_aggregation.py',suite.child_command(args,rule)[2])
                self.assertEqual(plan['settings']['client_concurrency'],8)
                self.assertNotEqual(single.make_job(plan,'method_a','smoke')['run'],job['run'])
            broken=copy.deepcopy(plans['fedavg']);broken['contract']['source_C_transfer']=True
            with self.assertRaisesRegex(ValueError,'Full-CP-only'):single.make_job(broken,'method_a')
            with self.assertRaisesRegex(ValueError,'suite changed'):
                suite.register(suite.parse_args(['--output-root',temp,'--client-concurrency','8']))

    def test_audit_accepts_correction_rejects_transfer_skip_and_B_change(self):
        for rule in suite.METHODS:
            with tempfile.TemporaryDirectory() as temp:
                root=Path(temp);job=fixtures.TestJointAggregation().audit_fixture(root,'method_a',rule)
                result=audit_run(root,job,1)
                operations={r['operation'] for r in stage_comparisons(result)}
                self.assertIn('A_functional_correction',operations)
                self.assertNotIn('B_shared_transfer',operations)
                progress=load_json(root/'progress.json')
                write_json(root/'progress.json',dict(progress,b_transfer_optimizer_steps=2))
                with self.assertRaisesRegex(ValueError,'source B transfer'):audit_run(root,job,1)
                write_json(root/'progress.json',dict(progress,functional_correction_steps=0))
                write_table(root/'sfra_rounds.csv',[dict(round=1,correction_steps=0,skip_reason='disabled')])
                with self.assertRaisesRegex(ValueError,'permitted reason'):audit_run(root,job,1)
                write_table(root/'sfra_rounds.csv',[dict(round=1,correction_steps=0,skip_reason='no_active_tokens')])
                audit_run(root,job,1)
                path=root/'predictions/r001.json';record=load_json(path)
                write_json(path,dict(record,b_sha256='changed-by-correction'))
                with self.assertRaisesRegex(ValueError,'altered B'):audit_run(root,job,1)

    def test_actual_full_cp_correction_preserves_B_and_original_CP_weights(self):
        # Actual autograd, source measurement and projected corrections, using a tiny
        # witness model. Exercise the new subclass, not a simulated correction.
        for weights in ({0:2/3,1:1/3},{0:.2,1:.8},{0:.4,1:.6}):
            bank=sfra_fixtures.TestSFRAClassification().bank()
            rt=object.__new__(runtime_class(dict(arm='method_a',mode='formal')))
            rt.trainer=SimpleNamespace(model=bank.model,device=bank.device)
            rt.bank,rt.a_keys=bank,bank.keys
            rt.b_keys=['image_encoder.proj_lora_B'];rt.keys=sorted(rt.a_keys+rt.b_keys)
            rt.variant,rt.strength,rt.scaling='full-cp',10.,.5
            rt.classification_strength=1.;rt.costs=[];rt.events=[{}];rt.q={0:2/3,1:1/3}
            rt.history=FunctionalHistory(bank.evaluate()['scores'],2)
            rt.history.valid[:]=True;rt.history.level[:]=1.5
            middle=copy.deepcopy(bank.model.state_dict())
            deltas={j:{bank.keys[0]:torch.randn_like(middle[bank.keys[0]])*.01} for j in weights}
            ordinary=copy.deepcopy(middle)
            ordinary[bank.keys[0]]+=sum(weights[j]*deltas[j][bank.keys[0]] for j in weights)
            rt.train_phase=lambda *a,**kw:(ordinary,deltas)
            with tempfile.TemporaryDirectory() as temp,patch('builtins.print'):
                committed,_,arrays,summary=rt.refresh(middle,1,Path(temp))
            self.assertEqual(summary['correction_steps'],3)
            self.assertEqual(summary['classification_weight'],1.)
            self.assertTrue(torch.equal(committed[rt.b_keys[0]],middle[rt.b_keys[0]]))
            torch.testing.assert_close(arrays['classification_global_token_weights'],torch.tensor([.6,2/30,1/3]))
            self.assertFalse(torch.equal(committed[bank.keys[0]],ordinary[bank.keys[0]]))

    def test_100_round_loop_and_resume_freeze_A_after_90(self):
        rt=object.__new__(runtime_class(dict(arm='method_a',mode='formal')))
        rt.is_frozen=False;rt.variant='full-cp';rt.strength=10.;rt.classification_strength=1.
        rt.b_transfer=None;rt.bank=SimpleNamespace();rt.sizes=[1];rt.summaries=[]
        rt.args=SimpleNamespace(sfra_stop_after_round=0);rt.resume_payload=None
        rt.trainer=SimpleNamespace(model=TinyModel());rt.keys=['proj_lora_A','proj_lora_B']
        rt.load_training_state=lambda state:rt.trainer.model.load_state_dict(state)
        rt.functional=lambda *a,**kw:dict(scores=torch.zeros(1,2))
        states,calls,histories={},[],{}
        rt.publish=lambda state,rnd,decision:states.update({rnd:copy.deepcopy(state)})
        rt.checkpoint=lambda state,rnd:histories.update({rnd:rt.history.state_dict()})
        def train(state,rnd,factor,extra=False):
            calls.append((rnd,factor));result=copy.deepcopy(state);result['proj_lora_'+factor]+=1
            return result
        rt.train_phase=train
        def refresh(state,rnd,folder):
            result=train(state,rnd,'A');result['proj_lora_A']+=.1
            return result,torch.zeros(1,2),dict(history_before=rt.history.level.clone(),
                history_valid_before=rt.history.valid.clone()),dict(round=rnd,correction_steps=3)
        rt.refresh=refresh
        with tempfile.TemporaryDirectory() as temp,patch('builtins.print'):
            rt.root=Path(temp);rt.run(snapshot(rt.trainer.model))
            self.assertEqual(len(calls),190)
            self.assertEqual(sum(r['correction_steps'] for r in rt.summaries),270)
            torch.testing.assert_close(states[90]['proj_lora_A'],states[100]['proj_lora_A'],rtol=0,atol=0)
            final=copy.deepcopy(states[100]);saved=copy.deepcopy(states[89]);calls.clear()
            rt.history.load_state_dict(histories[89])
            rt.resume_payload={'completed_round':89};rt.restore=lambda:(saved,89)
            rt.run(saved)
            self.assertEqual(calls[:2],[(90,'B'),(90,'A')])
            for key in final:torch.testing.assert_close(states[100][key],final[key],rtol=0,atol=0)

    def test_within_rule_comparison_requires_same_core_training_and_weights(self):
        with tempfile.TemporaryDirectory() as temp:
            results=[]
            for arm in ('method_a','plain_a'):
                root=Path(temp)/arm;job=fixtures.TestJointAggregation().audit_fixture(root,arm,'tailrw16')
                r=audit_run(root,job,1)
                r['curves']=[dict(r['curves'][0],round=i) for i in range(101)]
                r['predictions']={i:r['predictions'][0] for i in range(101)}
                r['metadata'].update({k:'synthetic-same' for k in PAIR_HASHES})
                r['job']['code_sha256']={'utils/cliplora_sfra.py':'same'}
                results.append(r)
            method,plain=results
            pair=compare_plain(method,plain)
            self.assertEqual(pair['status'],'eligible');self.assertFalse(pair['budget_matched'])
            self.assertTrue(pair['local_SGD_budget_matched'])
            method['job']['code_sha256']['tools/client_aggregation/method_a.py']='new'
            self.assertEqual(compare_plain(method,plain)['status'],'integration_source_differs')
            method['job']['code_sha256']['utils/cliplora_sfra.py']='changed'
            self.assertEqual(compare_plain(method,plain)['status'],'mismatch')
            method['job']['code_sha256']['utils/cliplora_sfra.py']='same'
            method['job']['aggregation']['weights'][0]+=.01
            self.assertIn('aggregation.weights',compare_plain(method,plain)['mismatches'])

    def test_summary_and_pack_include_three_A_only_runs_and_plain_contrasts(self):
        with tempfile.TemporaryDirectory() as temp,patch('builtins.print'):
            root=Path(temp)/'method';plain_root=Path(temp)/'plain';results={}
            for arm,folder in (('method_a',root),('plain_a',plain_root)):
                for rule in suite.METHODS:
                    run=folder/rule/'runs/seed42'/arm
                    job=fixtures.TestJointAggregation().audit_fixture(run,arm,rule)
                    r=audit_run(run,job,1)
                    r['curves']=[dict(r['curves'][0],round=i) for i in range(101)]
                    r['predictions']={i:r['predictions'][0] for i in range(101)}
                    r['progress'].update(completed_round=100,elapsed_seconds=1.,stage_diagnostic_seconds=0.,
                        normal_optimizer_steps=105600,extra_optimizer_steps=31680,
                        functional_correction_steps=270 if arm=='method_a' else 0)
                    r['metadata'].update({k:'synthetic-same' for k in PAIR_HASHES})
                    write_json(run/'completion.json',r['progress']);results[arm,rule]=r
            def load(path,rule,arm):return results[arm,rule]
            with patch('tools.client_aggregation.plain_a.load_a_run',side_effect=load), \
                 patch('tools.client_aggregation.method_a.load_a_run',side_effect=load), \
                 patch('tools.client_aggregation.method_a.load_plain',side_effect=lambda path,rule:results['plain_a',rule]):
                statuses,pairs=summarize(root,plain_root)
            self.assertEqual([s['status'] for s in statuses],['complete']*3)
            self.assertEqual([p['status'] for p in pairs],['eligible']*3)
            comparison=load_json(root/'analysis/method_a_vs_plain_a.json')['comparisons']
            self.assertEqual([p['status'] for p in comparison],['eligible']*3)
            self.assertTrue(all(p['delta_pp']['bottom20_tail_acc']==0. for p in comparison))
            self.assertIn('Full-CP',(root/'analysis/report.md').read_text(encoding='utf-8'))
            self.assertIn('A_functional_correction',(root/'analysis/stage_effects.csv').read_text(encoding='utf-8'))
            self.assertTrue(pack_results(root).is_file())


if __name__ == '__main__':
    unittest.main()
