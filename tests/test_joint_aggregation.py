"""CPU audits for the solver, real phase aggregation, observation and two-arm protocol."""
import copy
import json
import random
import shutil
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from scripts import run_cliplora_joint_aggregation as launcher
from scripts.run_ab_validation import write_json
from tools.client_aggregation.analysis import audit_run, expected_stages, metrics_from_prediction, prediction, sample_dynamics, stage_comparisons, summarize, pack_results
from tools.client_aggregation.protocol import (CONTRACT, SCHEMA, aggregation_spec, check_aggregation,
    count_matrix, digest, load_json, write_table, audit_runtime_config, PLAIN_A_CONTRACT)
from tools.sfra.maintext import FROZEN, PAIR_HASHES
from tools.client_aggregation.runtime import runtime_class
from tools.client_aggregation.weights import audit_weights, objective_gradient, solve_weights
from utils.cliplora_functional_feedback import snapshot
from utils.cliplora_sfra import SFRARuntime

REPO = Path(__file__).resolve().parents[1]


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.proj_lora_A = torch.nn.Parameter(torch.tensor([2.]))
        self.proj_lora_B = torch.nn.Parameter(torch.tensor([1.]))
        self.register_buffer('fixed', torch.tensor([7.]))
        self.dropout = torch.nn.Dropout(.5)

    def forward(self, x):
        return self.dropout(x) + self.proj_lora_A + self.proj_lora_B


class TestJointAggregation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def test_gradient_matches_finite_difference(self):
        counts = [[90, 10, 0], [0, 3, 7], [2, 0, 1]]
        w = np.array([.4, .3, .3]); eps = 1e-6
        _, gradient = objective_gradient(w, counts, .1)
        numeric = []
        for j in range(3):
            step = np.eye(3)[j] * eps
            numeric.append((objective_gradient(w+step, counts, .1)[0]-objective_gradient(w-step, counts, .1)[0])/(2*eps))
        np.testing.assert_allclose(gradient, numeric, rtol=1e-6, atol=1e-7)

    def test_iid_reduces_to_sample_weights(self):
        solution = solve_weights([[90, 9, 1], [180, 18, 2], [270, 27, 3]])
        np.testing.assert_allclose(solution['weights'], [1/6, 2/6, 3/6], atol=1e-8)

    def test_minimum_agrees_with_dense_two_client_search(self):
        counts = [[100, 0], [0, 5]]
        solution = solve_weights(counts, .1)
        x = np.linspace(.0001, .9999, 10001)
        values = [objective_gradient([v, 1-v], counts, .1)[0] for v in x]
        self.assertAlmostEqual(solution['weights'][0], x[int(np.argmin(values))], delta=1e-4)
        self.assertGreater(solution['weights'][1], 5/105)

    def test_permutation_and_invalid_inputs(self):
        counts = np.array([[8, 1, 0], [0, 2, 1], [1, 0, 7]])
        w = solve_weights(counts)['weights']
        np.testing.assert_allclose(solve_weights(counts[[2,0,1]][:,[1,2,0]])['weights'], np.array(w)[[2,0,1]], atol=1e-6)
        for bad in ([[1,0],[2,0]], [[0,0],[1,2]], [[1,-1],[1,2]], [[1,float('nan')]]):
            with self.assertRaises(ValueError): solve_weights(bad)
        for bad in (0, -1, float('nan'), float('inf')):
            with self.assertRaises(ValueError): solve_weights(counts, bad)
        with self.assertRaises(ValueError): audit_weights([.8,.1,.1], counts, .1)

    def test_real_partition_and_frozen_weights_tamper_detection(self):
        spec = aggregation_spec(REPO/'references/full10_clientlt', .1)
        self.assertEqual(sum(spec['counts']['sizes']), 10847)
        self.assertAlmostEqual(sum(spec['weights']), 1.)
        self.assertLess(check_aggregation(spec)['kkt_max_error'], 1e-4)
        broken = copy.deepcopy(spec); broken['weights'][0] += .01
        with self.assertRaises(ValueError): check_aggregation(broken)

    def test_plan_is_shared_and_smoke_cannot_be_formal(self):
        with tempfile.TemporaryDirectory() as temp, patch('builtins.print'):
            args = launcher.parse_args(['--output-root',temp,'--reference-run',str(REPO/'references/full10_clientlt')])
            plan = launcher.register(args)
            self.assertEqual(plan, launcher.register(args))
            for arm in ('frozen','ab'):
                job=launcher.make_job(plan,arm)
                cmd=launcher.command_for(job,temp)
                self.assertEqual(cmd[cmd.index('--sfra_variant')+1], 's' if arm=='frozen' else 'full-cp')
                self.assertEqual('--b_transfer_enable' in cmd,arm=='ab')
                self.assertIn('train_cliplora_joint_aggregation.py', ' '.join(cmd))
                smoke=launcher.make_job(plan,arm,'smoke')
                smoke_cmd=launcher.command_for(smoke,temp)
                self.assertNotEqual(job['run'],smoke['run'])
                self.assertEqual(smoke_cmd[smoke_cmd.index('--sfra_stop_after_round')+1],'1')
                if arm=='ab': self.assertIn((1,'after_B_transfer'),list(expected_stages(smoke,1)))
            args.aggregation_lambda=.2
            with self.assertRaises(ValueError): launcher.register(args)

    def test_actual_normal_B_training_uses_new_weights_and_preserves_A(self):
        with tempfile.TemporaryDirectory() as temp, patch('builtins.print'):
            cls=runtime_class(dict(arm='frozen',mode='formal'))
            rt=object.__new__(cls); rt.root=Path(temp); rt.is_frozen=True
            model=TinyModel()
            rt.a_keys=['proj_lora_A'];rt.b_keys=['proj_lora_B'];rt.keys=rt.a_keys+rt.b_keys
            rt.args=SimpleNamespace(seed=42)
            rt.sizes=[2]*30;rt.schedule=[list(range(30))]*100
            rt.q={j:1/30 for j in range(30)}
            rt.joint_weights={j:(j+1)/465 for j in range(30)}
            original_q=dict(rt.q);rt.b_transfer=None;rt.sfra_config={};rt.events=[];rt.budget=[]
            trainer=SimpleNamespace(model=model,_writer=None)
            def reset(): trainer.optimizer=torch.optim.SGD([p for p in model.parameters() if p.requires_grad],lr=.01)
            trainer.reset_optimizer_and_scheduler=reset
            rt.trainer=trainer
            def train(trainer,client,rnd,args,epochs):
                for _ in range(3):
                    trainer.optimizer.zero_grad()
                    (model.proj_lora_B-(client+1)).square().sum().backward()
                    trainer.optimizer.step()
                return None,None,1,3
            rt.normal_train=train
            audit=SimpleNamespace(event_context={})
            def normal(before,after,local,selected,weights,rnd,factor):
                self.assertEqual(weights,rt.joint_weights)
                write_json(rt.root/'events'/audit.event_context['event_id']/'event.json',{})
            audit.normal=normal;rt.audit=audit
            state=snapshot(model)
            after,deltas=rt.train_phase(state,2,return_deltas=True)
            expected=sum(rt.joint_weights[j]*(state['proj_lora_B']+deltas[j]['proj_lora_B']) for j in range(30))
            torch.testing.assert_close(after['proj_lora_B'],expected)
            torch.testing.assert_close(after['proj_lora_A'],state['proj_lora_A'],rtol=0,atol=0)
            self.assertEqual(rt.q,original_q)
            self.assertEqual(len(rt.budget),30)
            with self.assertRaises(ValueError): rt.train_phase(state,2,'A',True)

    def test_A_weight_hook_does_not_change_CP_sample_weights(self):
        with tempfile.TemporaryDirectory() as temp:
            rt=object.__new__(runtime_class(dict(arm='ab',mode='formal')))
            rt.root=Path(temp);rt.is_frozen=False;rt.q={j:1/30 for j in range(30)}
            rt.joint_weights={j:(j+1)/465 for j in range(30)}
            result=rt.phase_aggregation_weights(list(range(30)),50,'A',True)
            self.assertEqual(result,rt.joint_weights)
            self.assertEqual(rt.q,{j:1/30 for j in range(30)})
            with self.assertRaises(ValueError): rt.phase_aggregation_weights(list(range(30)),91,'A',True)

    def test_frozen_loop_and_resume_never_call_A(self):
        with patch('builtins.print'):
            rt=object.__new__(runtime_class(dict(arm='frozen',mode='formal')))
            rt.is_frozen=True;rt.args=SimpleNamespace(sfra_stop_after_round=2)
            rt.trainer=SimpleNamespace(model=TinyModel());rt.keys=['proj_lora_A','proj_lora_B']
            rt.resume_payload=None;calls=[];published=[];checkpoints=[]
            rt.load_training_state=lambda state:rt.trainer.model.load_state_dict(state)
            def train(state,rnd,factor):
                calls.append((rnd,factor));out=dict(state);out['proj_lora_B']=state['proj_lora_B']+1;return out
            rt.train_phase=train;rt.publish=lambda state,rnd,decision:published.append(rnd)
            rt.checkpoint=lambda state,rnd:checkpoints.append(rnd)
            rt.run(snapshot(rt.trainer.model))
            self.assertEqual(calls,[(1,'B'),(2,'B')]);self.assertEqual(published,[0,1,2])
            rt.resume_payload={'completed_round':1};rt.restore=lambda:(snapshot(rt.trainer.model),1)
            calls.clear();published.clear();rt.run(snapshot(rt.trainer.model))
            self.assertEqual(calls,[(2,'B')]);self.assertEqual(published,[2])

    def test_AB_keeps_original_run_loop(self):
        rt=object.__new__(runtime_class(dict(arm='ab',mode='formal')));rt.is_frozen=False
        with patch.object(SFRARuntime,'run',return_value='original') as run:
            self.assertEqual(rt.run({'state':1}),'original');run.assert_called_once_with({'state':1})

    def test_official_diagnostics_restore_model_modes_and_rng(self):
        with tempfile.TemporaryDirectory() as temp:
            rt=object.__new__(runtime_class(dict(arm='frozen',mode='formal')))
            model=TinyModel();model.train()
            y=torch.arange(100).repeat_interleave(100)
            loader=torch.utils.data.DataLoader(torch.utils.data.TensorDataset(torch.nn.functional.one_hot(y,100).float(),y),batch_size=256)
            rt.global_trainer=SimpleNamespace(model=model,test_loader=loader,parse_batch_test=lambda batch:batch,
                model_inference=model,dm=SimpleNamespace(dataset=SimpleNamespace(data_test=[SimpleNamespace(label=int(v)) for v in y])))
            rt.head=np.arange(20);rt.middle=np.arange(20,80);rt.tail=np.arange(80,100)
            rt.a_keys=['proj_lora_A'];rt.b_keys=['proj_lora_B']
            original=snapshot(model);other=copy.deepcopy(original);other['proj_lora_A']+=10
            rng=torch.get_rng_state().clone();np_rng=np.random.get_state();py_rng=random.getstate()
            metrics,_=rt.evaluate_predictions(other,0,'test',Path(temp))
            self.assertEqual(metrics['overall_acc'],100)
            for k,v in model.state_dict().items():torch.testing.assert_close(v,original[k],rtol=0,atol=0)
            self.assertTrue(model.training and model.dropout.training)
            self.assertTrue(torch.equal(rng,torch.get_rng_state()))
            self.assertEqual(py_rng,random.getstate());np.testing.assert_array_equal(np_rng[1],np.random.get_state()[1])
            stored=prediction(Path(temp)/'r000.npz');self.assertEqual(int(stored['correct'].sum()),10000)

    def test_sample_and_stage_changes_are_measured_not_inferred(self):
        labels=np.arange(4);groups=dict(overall_acc=[0,1,2,3],bottom20_tail_acc=[2,3])
        def sample(hit):return dict(class_id=labels,correct=np.array(hit,dtype=bool),logit_margin=np.where(hit,1.,-1.))
        initial=sample([1,0,1,0]);after=sample([0,1,1,0])
        result=dict(job=dict(arm='frozen'),predictions={0:initial,1:after},groups=groups,
            stages={(1,'ordinary_B'):after,(1,'sample_weighted_B_counterfactual'):initial})
        rows=sample_dynamics(result)
        row=next(r for r in rows if r['round']==1 and r['group']=='overall_acc')
        self.assertEqual((row['retained_initial'],row['learned_initial_wrong'],row['never_correct_so_far']),(1,1,1))
        effects=stage_comparisons(result)
        row=next(r for r in effects if r['operation']=='B_aggregation_same_local_updates' and r['group']=='overall_acc')
        self.assertEqual((row['corrected_examples'],row['damaged_examples'],row['delta_accuracy']),(1,1,0))

    def audit_fixture(self, root, arm, rule='joint'):
        """Synthetic one-round artifact fixture, not model training or research evidence."""
        root=Path(root);root.mkdir(parents=True,exist_ok=True)
        job=dict(schema=SCHEMA,arm=arm,mode='smoke',contract=PLAIN_A_CONTRACT if arm=='plain_a' else CONTRACT,
            aggregation=aggregation_spec(REPO/'references/full10_clientlt',.1,rule))
        cls=runtime_class(job);rt=object.__new__(cls)
        rt.root=root;rt.variant='full-cp' if arm=='ab' else 's'
        sizes=job['aggregation']['counts']['sizes']
        rt.q={j:n/10847 for j,n in enumerate(sizes)}
        rt.config=dict(seed=42,protocol_seed=42,topology='client-longtail',la_tau=1.,
            aggregation='sample',normal_steps_expected=105600,extra_steps_expected=31680)
        if arm=='plain_a':
            rt.config.update(normal_trainable_factor='B', control_enabled=False,
                a_lr_mult=1., extra_a_lr=.001, extra_weight_decay=0., normal_b_lr=.001)
        rt.sfra_config=dict(FROZEN,variant=rt.variant,base_training_config=rt.config)
        if arm=='ab':rt.sfra_config.update(classification_weight=1.,b_transfer=dict(CONTRACT['B'],mode='shared'))
        rt.audit=SimpleNamespace(counts=torch.tensor(job['aggregation']['matrix']),meta={})
        shutil.copyfile(REPO/'references/full10_clientlt/partition_manifest.csv',root/'partition_manifest.csv')
        rt.configure_experiment()
        write_json(root/'sfra_config.json',rt.sfra_config)
        write_json(root/'execution_config.json',dict(version=2,feedback_forward_batch_size=128,device_cache_gib=4.))
        write_json(root/'joint_job.json',job)
        write_json(root/'joint_receipt.json',dict(schema=SCHEMA,job_digest=digest(job),code_unchanged=True,exit_code=0))
        counts=np.asarray(job['aggregation']['matrix']).sum(0)
        write_json(root/'class_prior.json',dict(counts=counts.tolist(),prior=(counts/counts.sum()).tolist(),tail_ids=list(range(80,100))))
        steps=sum((n+31)//32 for n in sizes)
        progress=dict(completed_round=1,official_test_passes=2,weights_sha256=job['aggregation']['weights_sha256'],
            arm=arm,mode='smoke',normal_optimizer_steps=steps*3,extra_optimizer_steps=0 if arm=='frozen' else steps,
            functional_correction_steps=3 if arm=='ab' else 0,b_transfer_events=1 if arm=='ab' else 0)
        write_json(root/'progress.json',progress)
        budget=[]
        for factor,phase,epochs in ([('B','normal_B',3),('A','refresh_A',1)] if arm!='frozen' else [('B','normal_B',3)]):
            for j,n in enumerate(sizes):
                budget.append(dict(round=1,phase=phase,client_id=j,optimizer_steps=epochs*((n+31)//32),sample_presentations=epochs*n))
            rt.phase_aggregation_weights(list(range(30)),1,factor,factor=='A')
            write_json(root/'events'/('r001_c000_main_'+phase)/'event.json',dict(reconstruction_passed=True,
                frozen_factor_max_abs_error=0,selected_client_ids=list(range(30)),server_weights=job['aggregation']['weights']))
        write_table(root/'budget.csv',budget)
        write_table(root/'sfra_rounds.csv',[dict(round=1,correction_steps=3)] if arm=='ab' else [])
        if arm=='ab':
            write_table(root/'b_transfer_rounds.csv',[dict(round=1,optimizer_steps=0)])
            write_json(root/'b_transfer_rounds/r001/calibration_manifest.json',dict(shared_donor_ids=[]))
        labels=np.repeat(np.arange(100),100)
        arrays=dict(sample_id=np.arange(10000),class_id=labels,prediction=labels.copy(),correct=np.ones(10000,dtype=bool),logit_margin=np.ones(10000))
        groups=dict(overall_acc=list(range(100)),head20_acc=list(range(20)),middle60_acc=list(range(20,80)),bottom20_tail_acc=list(range(80,100)),non_tail_acc=list(range(80)))
        metrics=metrics_from_prediction(arrays,groups)
        (root/'predictions').mkdir()
        for rnd in (0,1):
            np.savez_compressed(root/'predictions'/f'r{rnd:03d}.npz',**arrays)
            write_json(root/'predictions'/f'r{rnd:03d}.json',dict(a_sha256='initial' if arm=='frozen' or rnd==0 else 'updated',
                **(dict(b_sha256='synthetic_B') if arm=='plain_a' else {}), **metrics))
            write_table(root/f'per_class_accuracy_epoch_{rnd-1}.csv',[dict(class_id=c,per_class_acc=100.) for c in range(100)])
        write_table(root/'round_metrics.csv',[dict(round=rnd,**metrics) for rnd in (0,1)])
        for rnd,stage in expected_stages(job,1):
            folder=root/'stage_predictions'/stage;folder.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(folder/f'r{rnd:03d}.npz',**arrays)
            write_json(folder/f'r{rnd:03d}.json',dict(metrics,
                **(dict(a_sha256='updated', b_sha256='synthetic_B') if arm=='plain_a' else {})))
        return job

    def test_tailrw16_uses_exact_historical_weights_without_convex_solver(self):
        from tools.sfra.simple_controls import tail_weights
        with patch('tools.client_aggregation.protocol.solve_weights', side_effect=AssertionError('must not solve')):
            spec=aggregation_spec(REPO/'references/full10_clientlt',.1,'tailrw16')
        expected=tail_weights(spec['counts']['sizes'],spec['counts']['tail_counts'],16.)
        self.assertEqual(spec['weights'],expected)
        self.assertIsNone(spec['lambda_value'])
        self.assertAlmostEqual(sum(spec['weights']),1.)
        self.assertTrue(all(w>0 for w in spec['weights']))
        check_aggregation(spec)
        broken=copy.deepcopy(spec)
        broken['weights'][0]+=.001;broken['weights'][1]-=.001
        broken['weights_sha256']=digest(broken['weights'])
        with self.assertRaisesRegex(ValueError,'count formula'):check_aggregation(broken)

    def test_tailrw16_launcher_runs_only_frozen_and_propagates_policy(self):
        args=launcher.parse_args(['--aggregation-rule','tailrw16'])
        self.assertEqual(args.arms,['frozen'])
        self.assertEqual(args.output_root.name,'frozen_tailrw16_v1')
        for cmd in launcher.server_commands(args,'frozen'):
            self.assertEqual(cmd[cmd.index('--aggregation-rule')+1],'tailrw16')
        with self.assertRaises(SystemExit),patch('sys.stderr'):
            launcher.parse_args(['--aggregation-rule','tailrw16','--arms','ab'])
        with tempfile.TemporaryDirectory() as temp:
            args.output_root=Path(temp)
            plan=launcher.register(args)
            self.assertEqual(plan,launcher.register(args))
            self.assertEqual(plan['contract']['arms'],['frozen'])
            job=launcher.make_job(plan,'frozen')
            cmd=launcher.command_for(job,temp)
            self.assertNotIn('--b_transfer_enable',cmd)
            self.assertEqual(cmd[cmd.index('--sfra_variant')+1],'s')
            self.assertEqual(cmd[cmd.index('--cliplora_freeze_a')+1],'True')
            with self.assertRaises(ValueError):launcher.make_job(plan,'ab')
            args.aggregation_rule='joint'
            with self.assertRaises(ValueError):launcher.register(args)

    def test_tailrw16_runtime_and_pairing_reject_wrong_budgets(self):
        from tools.client_aggregation.frozen_tailrw import paired_audit
        with tempfile.TemporaryDirectory() as temp,patch('builtins.print'):
            results=[]
            for rule in ('tailrw16','joint'):
                run=Path(temp)/rule
                job=self.audit_fixture(run,'frozen',rule)
                r=audit_run(run,job,1)
                self.assertEqual(r['config']['refresh_rounds'],[])
                self.assertEqual(r['config']['base_training_config']['extra_steps_expected'],0)
                self.assertFalse(r['config'].get('b_transfer'))
                r['metadata'].update({k:'synthetic-same' for k in PAIR_HASHES})
                results.append(r)
            self.assertEqual(paired_audit(*results)['status'],'eligible')
            results[1]['progress']['normal_optimizer_steps']+=1
            self.assertEqual(paired_audit(*results)['status'],'mismatch')
            path=Path(temp)/'tailrw16/progress.json'
            progress=load_json(path);progress['b_transfer_events']=1;write_json(path,progress)
            with self.assertRaisesRegex(ValueError,'source B transfer'):
                audit_run(Path(temp)/'tailrw16',results[0]['job'],1)

    def test_tailrw16_summary_reports_single_run_and_read_only_pair(self):
        from tools.client_aggregation.frozen_tailrw import summarize as summarize_tailrw
        with tempfile.TemporaryDirectory() as temp,patch('builtins.print'):
            roots={rule:Path(temp)/rule for rule in ('tailrw16','joint')}
            results={}
            for rule,root in roots.items():
                run=root/'runs/seed42/frozen'
                job=self.audit_fixture(run,'frozen',rule)
                r=audit_run(run,job,1)
                r['metadata'].update({k:'synthetic-same' for k in PAIR_HASHES})
                r['predictions']={i:r['predictions'][0] for i in range(101)}
                r['curves']=[dict(r['curves'][0],round=i) for i in range(101)]
                r['progress'].update(completed_round=100,elapsed_seconds=1.,stage_diagnostic_seconds=0.)
                write_json(run/'completion.json',r['progress'])
                results[rule]=r
            with patch('tools.client_aggregation.frozen_tailrw.load_frozen',side_effect=lambda root,rule:results[rule]):
                rows=summarize_tailrw(roots['tailrw16'],roots['joint'])
            self.assertEqual(len(rows),1)
            self.assertEqual(rows[0]['status'],'complete')
            pair=load_json(roots['tailrw16']/'analysis/pair_audit.json')
            self.assertEqual(pair['status'],'eligible')
            self.assertEqual(pair['tailrw16_minus_joint']['overall_acc'],0.)
            self.assertFalse((roots['joint']/'analysis').exists())
            report=(roots['tailrw16']/'analysis/report.md').read_text(encoding='utf-8')
            self.assertIn('105600',report)
            self.assertIn('tailrw16_frozen',report)
            self.assertIn('joint_frozen',report)

    def test_runtime_configuration_and_complete_one_round_artifact_audit(self):
        with tempfile.TemporaryDirectory() as temp:
            for arm in ('frozen','ab'):
                run=Path(temp)/arm;job=self.audit_fixture(run,arm)
                result=audit_run(run,job,1)
                self.assertEqual(result['progress']['completed_round'],1)
                if arm=='frozen':
                    self.assertEqual(result['config']['refresh_rounds'],[])
                    self.assertEqual(result['config']['base_training_config']['extra_steps_expected'],0)
                else:
                    self.assertEqual(result['config']['b_transfer']['rounds'],[1])
                    self.assertIn('joint-class-weighted',result['config']['b_transfer']['calibration_anchor'])

    def test_audit_rejects_changed_frozen_A_and_changed_runtime_weights(self):
        with tempfile.TemporaryDirectory() as temp:
            run=Path(temp);job=self.audit_fixture(run,'frozen')
            path=run/'predictions/r001.json';original=load_json(path)
            write_json(path,dict(original,a_sha256='changed'))
            with self.assertRaisesRegex(ValueError,'frozen A'):audit_run(run,job,1)
            write_json(path,original)
            path=run/'events/r001_c000_main_normal_B/event.json';event=load_json(path)
            event['server_weights']=[1/30]*30;write_json(path,event)
            with self.assertRaisesRegex(ValueError,'different aggregation weights'):audit_run(run,job,1)

    def test_full_summary_and_pack_with_synthetic_trajectories(self):
        """Exercise 101-round reporting; synthetic perfect predictions are never real results."""
        with tempfile.TemporaryDirectory() as temp, patch('builtins.print'):
            root=Path(temp)/'synthetic_suite'
            args=launcher.parse_args(['--output-root',str(root),'--reference-run',str(REPO/'references/full10_clientlt')])
            plan=launcher.register(args)
            results={}
            for arm in ('frozen','ab'):
                run=root/'runs/seed42'/arm
                job=self.audit_fixture(run,arm)
                result=audit_run(run,job,1)
                result['predictions']={r:result['predictions'][0] for r in range(101)}
                result['curves']=[dict(result['curves'][0],round=r) for r in range(101)]
                result['metadata'].update({k:'same-synthetic-identity' for k in PAIR_HASHES})
                result['metadata']['environment']=dict(gpu='synthetic',visible_devices='0' if arm=='frozen' else '1')
                result['progress'].update(completed_round=100,elapsed_seconds=1000.,stage_diagnostic_seconds=20.)
                for row in result['budgets']:row['upload_bytes']=100
                write_json(run/'completion.json',result['progress'])
                write_table(run/'evaluation_budget.csv',[dict(seconds=10.)])
                write_table(run/'sfra_costs.csv',[])
                results[arm]=result
            with patch('tools.client_aggregation.analysis.audit_run',side_effect=lambda run,job:results[job['arm']]):
                rows=summarize(root)
            self.assertEqual([r['status'] for r in rows],['complete','complete'])
            pair=load_json(root/'analysis/pair_audit.json')
            self.assertEqual(pair['status'],'eligible');self.assertTrue(pair['same_recorded_environment'])
            self.assertEqual(pair['ab_minus_frozen']['overall_acc'],0.)
            self.assertFalse(pair['budget_matched'])
            figures=load_json(root/'analysis/figure_manifest.json')
            if figures['files']:
                self.assertEqual(len(figures['files']),6)
                self.assertTrue(all((root/'analysis'/name).stat().st_size > 1000 for name in figures['files']))
            report=(root/'analysis/report.md').read_text(encoding='utf-8')
            self.assertIn('阶段即时变化',report);self.assertIn('始终未答对',report)
            (root/'runs/seed42/ab/checkpoints').mkdir()
            (root/'runs/seed42/ab/checkpoints/sfra_last.pt').write_bytes(b'synthetic checkpoint excluded from archive')
            archive=pack_results(root)
            import tarfile
            with tarfile.open(archive) as packed:
                names=packed.getnames()
                self.assertTrue(any(n.endswith('/analysis/report.md') for n in names))
                self.assertFalse(any('checkpoints/' in n for n in names))


if __name__=='__main__':
    unittest.main()
