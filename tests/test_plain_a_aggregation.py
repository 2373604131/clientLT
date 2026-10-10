"""CPU validation only; synthetic artifacts are never experiment results."""
import copy
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from scripts import run_cliplora_plain_a_aggregation as suite
from scripts import run_cliplora_joint_aggregation as single
from scripts.run_ab_validation import write_json
from tests import test_joint_aggregation as fixtures
from tests.test_joint_aggregation import TinyModel
from tools.client_aggregation.analysis import audit_run, stage_comparisons, pack_results
from tools.client_aggregation.plain_a import summarize
from tools.client_aggregation.protocol import (PLAIN_A_CONTRACT, PAIR_HASHES, load_json, write_table,
    client_execution_config, digest)
from tools.client_aggregation.runtime import runtime_class
from utils.cliplora_functional_feedback import snapshot

REPO = Path(__file__).resolve().parents[1]


class TestPlainAAggregation(unittest.TestCase):
    def test_eight_slots_propagate_to_child_and_both_phase_audits(self):
        self.assertEqual(suite.parse_args(['--client-concurrency','8']).output_root.name,
                         'plain_a_aggregation_v1_parallel8')
        self.assertEqual(suite.parse_args([]).output_root.name, 'plain_a_aggregation_v1_parallel6')
        with tempfile.TemporaryDirectory() as temp:
            args=suite.parse_args(['--client-concurrency','8','--output-root',temp])
            plans=suite.register(args)
            for rule,plan in plans.items():
                self.assertEqual(plan['settings']['client_concurrency'],8)
                cmd=suite.child_command(args,rule)
                self.assertEqual(cmd[cmd.index('--client-concurrency')+1],'8')
            run=Path(temp)/'synthetic'
            job=fixtures.TestJointAggregation().audit_fixture(run,'plain_a','fedavg')
            job['settings']=dict(client_concurrency=8)
            write_json(run/'joint_job.json',job)
            receipt=load_json(run/'joint_receipt.json');receipt['job_digest']=digest(job)
            write_json(run/'joint_receipt.json',receipt)
            cfg=load_json(run/'sfra_config.json');cfg['client_execution']=client_execution_config(job)
            self.assertEqual(cfg['client_execution']['max_concurrent_clients'],8)
            write_json(run/'sfra_config.json',cfg);write_json(run/'client_execution.json',cfg['client_execution'])
            for factor in ('B','A'):
                write_json(run/'parallel_execution'/f'r001_{factor}.json',dict(
                    round=1,factor=factor,slots=8,clients=30,selected_client_ids=list(range(30)),
                    client_audits=[dict(client_id=j,slot=j%8) for j in range(30)]))
            result=audit_run(run,job,1)
            self.assertEqual(len(result['budgets']),60)
            record=load_json(run/'parallel_execution/r001_A.json')
            record['client_audits'][-1]['slot']=8
            write_json(run/'parallel_execution/r001_A.json',record)
            with self.assertRaisesRegex(ValueError,'parallel client execution'):
                audit_run(run,job,1)

    def test_three_registered_policies_and_no_transfer_or_correction(self):
        with tempfile.TemporaryDirectory() as temp:
            args = suite.parse_args(['--output-root', temp])
            plans = suite.register(args)
            self.assertEqual(plans, suite.register(args))
            self.assertEqual(set(plans), {'fedavg', 'tailrw16', 'joint'})
            for rule, plan in plans.items():
                self.assertEqual(plan['contract'], PLAIN_A_CONTRACT)
                job = single.make_job(plan, 'plain_a')
                command = single.command_for(job, Path(temp)/rule)
                self.assertNotIn('--b_transfer_enable', command)
                self.assertEqual(command[command.index('--sfra_variant')+1], 's')
                self.assertEqual(command[command.index('--sfra_retention_weight')+1], '0')
                # Initial freeze is intentionally True; train_only switches each phase.
                self.assertEqual(command[command.index('--cliplora_freeze_a')+1], 'True')
                smoke = single.make_job(plan, 'plain_a', 'smoke')
                self.assertNotEqual(smoke['run'], job['run'])
                smoke_command=single.command_for(smoke, temp)
                self.assertEqual(smoke_command[smoke_command.index('--sfra_stop_after_round')+1], '1')
                self.assertEqual(sum((n+31)//32 for n in job['aggregation']['counts']['sizes'])*390, 137280)
                broken = copy.deepcopy(plan); broken['contract']['A_rounds'] = list(range(1,101))
                with self.assertRaisesRegex(ValueError, 'fixed-schedule'):
                    single.make_job(broken, 'plain_a')
            self.assertNotEqual(plans['joint']['aggregation']['weights'], plans['tailrw16']['aggregation']['weights'])
            args.client_concurrency = 4
            with self.assertRaisesRegex(ValueError, 'suite changed'):
                suite.register(args)

    def test_100_round_schedule_and_resume_across_round90(self):
        rt = object.__new__(runtime_class(dict(arm='plain_a', mode='formal')))
        rt.is_frozen=False; rt.variant='s'; rt.strength=0.; rt.bank=None; rt.b_transfer=None
        rt.args=SimpleNamespace(sfra_stop_after_round=0)
        rt.trainer=SimpleNamespace(model=TinyModel())
        rt.keys=['proj_lora_A', 'proj_lora_B']; rt.resume_payload=None
        rt.load_training_state=lambda state: rt.trainer.model.load_state_dict(state)
        states, calls = {}, []
        rt.publish=lambda state,rnd,decision: states.update({rnd:copy.deepcopy(state)})
        rt.checkpoint=lambda state,rnd: None
        def train(state,rnd,factor,extra=False):
            calls.append((rnd,factor,extra))
            result=copy.deepcopy(state)
            result['proj_lora_'+factor] += 1
            return result
        rt.train_phase=train
        with patch('builtins.print'):
            rt.run(snapshot(rt.trainer.model))
        self.assertEqual(len(calls),190)
        self.assertEqual(calls[:4],[(1,'B',False),(1,'A',True),(2,'B',False),(2,'A',True)])
        self.assertEqual(calls[-10:],[(r,'B',False) for r in range(91,101)])
        self.assertEqual(float(states[90]['proj_lora_A']),92.)
        torch.testing.assert_close(states[90]['proj_lora_A'],states[100]['proj_lora_A'],rtol=0,atol=0)
        final=copy.deepcopy(states[100]); saved=copy.deepcopy(states[89]); calls.clear()
        rt.resume_payload={'completed_round':89};rt.restore=lambda:(saved,89)
        with patch('builtins.print'):
            rt.run(saved)
        self.assertEqual(calls[:2],[(90,'B',False),(90,'A',True)])
        for key in final:
            torch.testing.assert_close(states[100][key],final[key],rtol=0,atol=0)

    def test_real_A_sgd_and_aggregation_preserve_post_B_state(self):
        """Actual autograd and SGD in the inherited local-A training phase."""
        # Load lazy optimizer dependencies before patch.dict snapshots sys.modules.
        # Otherwise its cleanup can unload modules with persistent C++ registrations.
        import torch._dynamo
        for rule in suite.METHODS:
            with tempfile.TemporaryDirectory() as temp:
                plan=suite.register(suite.parse_args(['--output-root',temp]))[rule]
                rt=object.__new__(runtime_class(single.make_job(plan,'plain_a')))
                rt.root=Path(temp)/'actual';rt.is_frozen=False
                model=TinyModel();model.proj_lora_B.data.fill_(7.)
                rt.a_keys=['proj_lora_A'];rt.b_keys=['proj_lora_B'];rt.keys=rt.a_keys+rt.b_keys
                rt.args=SimpleNamespace(seed=42);rt.sizes=[1]*30;rt.schedule=[list(range(30))]*100
                rt.q={j:1/30 for j in range(30)};rt.joint_weights=dict(enumerate(plan['aggregation']['weights']))
                rt.config=dict(extra_a_lr=.001);rt.b_transfer=None;rt.sfra_config={};rt.events=[];rt.budget=[]
                rt.trainer=SimpleNamespace(model=model, training_logit_adjustment=None,
                    fed_train_loader_x_dict={j:[(torch.tensor([1.]),torch.tensor([float(j+1)]))] for j in range(30)},
                    parse_batch_train=lambda batch:batch)
                audit=SimpleNamespace(event_context={})
                def save(before,after,deltas,selected,weights,rnd,phase):
                    torch.testing.assert_close(before['proj_lora_B'],after['proj_lora_B'],rtol=0,atol=0)
                    self.assertEqual(weights,rt.joint_weights)
                    write_json(rt.root/'events'/audit.event_context['event_id']/'event.json',{})
                audit.save=save;rt.audit=audit
                def step(model,optimizer,scaler,precision,images,labels,**kwargs):
                    self.assertTrue(model.proj_lora_A.requires_grad)
                    self.assertFalse(model.proj_lora_B.requires_grad)
                    optimizer.zero_grad()
                    (model.proj_lora_A-labels).square().sum().backward()
                    optimizer.step()
                with patch.dict('sys.modules', {'trainers.cliplora':SimpleNamespace(cliplora_optimizer_step=step)}), patch('builtins.print'):
                    after=rt.train_phase(snapshot(model),2,'A',True)
                expected=sum(w*(2-.002*(2-(j+1))) for j,w in rt.joint_weights.items())
                self.assertAlmostEqual(float(after['proj_lora_A']),expected,places=5)
                self.assertEqual(float(after['proj_lora_B']),7.)
                self.assertEqual(sum(b['optimizer_steps'] for b in rt.budget),30)

    def test_artifact_audit_rejects_functional_correction_or_transfer(self):
        for rule in suite.METHODS:
            with tempfile.TemporaryDirectory() as temp:
                root=Path(temp)
                job=fixtures.TestJointAggregation().audit_fixture(root,'plain_a',rule)
                result=audit_run(root,job,1)
                self.assertEqual(len(result['budgets']),60)
                self.assertEqual(result['progress']['extra_optimizer_steps'],352)
                self.assertEqual({r['operation'] for r in stage_comparisons(result)},
                    {'ordinary_B_learning','B_aggregation_same_local_updates','ordinary_A_learning','A_aggregation_same_local_updates'})
                cfg=load_json(root/'sfra_config.json')
                broken=copy.deepcopy(cfg);broken['base_training_config']['extra_a_lr']=.01
                write_json(root/'sfra_config.json',broken)
                with self.assertRaisesRegex(ValueError,'extra_a_lr'):audit_run(root,job,1)
                write_json(root/'sfra_config.json',cfg)
                progress=load_json(root/'progress.json');progress['functional_correction_steps']=3
                write_json(root/'progress.json',progress)
                with self.assertRaisesRegex(ValueError,'functional correction'):audit_run(root,job,1)
                progress['functional_correction_steps']=0;progress['b_transfer_events']=1
                write_json(root/'progress.json',progress)
                with self.assertRaisesRegex(ValueError,'source B transfer'):audit_run(root,job,1)

    def test_gpu_queue_runs_all_three_with_at_most_one_job_per_gpu(self):
        with tempfile.TemporaryDirectory() as temp,patch('builtins.print'):
            args=suite.parse_args(['--output-root',temp,'--gpus','2','3'])
            calls=[];active=set();lock=threading.Lock()
            class Process:
                def __init__(self,cmd,**kwargs):
                    self.gpu=kwargs['env']['CUDA_VISIBLE_DEVICES']
                    self.rule=cmd[cmd.index('--methods')+1]
                    with lock:
                        if self.gpu in active:raise AssertionError('overlapping GPU jobs')
                        active.add(self.gpu);calls.append((self.rule,self.gpu))
                def wait(self):
                    with lock:active.remove(self.gpu)
                    return int(self.rule=='fedavg')
            with patch.object(suite.subprocess,'Popen',Process),self.assertRaisesRegex(ValueError,'fedavg'):
                suite.launch(args)
            self.assertEqual({r for r,gpu in calls},set(suite.METHODS))
            self.assertTrue(all(gpu in ('2','3') for r,gpu in calls))

    def test_summary_and_archive_do_not_mix_frozen_or_full_AB(self):
        with tempfile.TemporaryDirectory() as temp,patch('builtins.print'):
            root=Path(temp); results={}
            for rule in suite.METHODS:
                run=root/rule/'runs/seed42/plain_a'
                job=fixtures.TestJointAggregation().audit_fixture(run,'plain_a',rule)
                r=audit_run(run,job,1)
                r['curves']=[dict(r['curves'][0],round=i) for i in range(101)]
                r['predictions']={i:r['predictions'][0] for i in range(101)}
                r['progress'].update(completed_round=100,elapsed_seconds=1.,stage_diagnostic_seconds=0.,
                                     normal_optimizer_steps=105600,extra_optimizer_steps=31680)
                r['metadata'].update({k:'same-synthetic-identity' for k in PAIR_HASHES})
                r['metadata']['environment']=dict(gpu='synthetic')
                write_json(run/'completion.json',r['progress']);results[rule]=r
            with patch('tools.client_aggregation.plain_a.load_plain',side_effect=lambda path,rule:results[rule]):
                status,pairs=summarize(root)
            self.assertEqual([s['status'] for s in status],['complete']*3)
            self.assertEqual([p['status'] for p in pairs],['eligible']*3)
            self.assertTrue(all(p['delta_pp']['overall_acc']==0. for p in pairs))
            self.assertIn('137280',(root/'analysis/report.md').read_text(encoding='utf-8'))
            self.assertNotIn('A_functional_correction',(root/'analysis/stage_effects.csv').read_text(encoding='utf-8'))
            results['joint']['job']['code_sha256']={'changed':'source'}
            with patch('tools.client_aggregation.plain_a.load_plain',side_effect=lambda path,rule:results[rule]):
                _,pairs=summarize(root)
            self.assertEqual([p['status'] for p in pairs],['eligible','mismatch','mismatch'])
            self.assertTrue(pack_results(root).is_file())


if __name__=='__main__':
    unittest.main()
