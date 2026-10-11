"""CPU integration checks. Tiny-model runs and synthetic artifacts are not CIFAR results."""
import copy
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch.utils.data import DataLoader

from scripts import run_cliplora_joint_floor as launcher
from scripts import run_cliplora_plain_a_aggregation as plain_launcher
from scripts import run_cliplora_joint_aggregation as single
from scripts.run_ab_validation import write_json
from tests import test_joint_aggregation as fixtures, test_joint_client_parallel as tiny
from tools.client_aggregation import floor16
from tools.client_aggregation.analysis import audit_run, metrics_from_prediction, pack_results
from tools.client_aggregation.parallel import clone_models, ordered_queue, train_local
from tools.client_aggregation.protocol import (PLAIN_A_CONTRACT, PAIR_HASHES, load_json, digest,
    check_aggregation, code_hashes, write_table, read_csv)
from tools.client_aggregation.runtime import runtime_class
from tools.client_aggregation.weights import (floor16_parameters, audit_weights, audit_floor16,
    solve_floor16, objective_gradient)
from utils.cliplora_functional_feedback import snapshot

REPO=Path(__file__).resolve().parents[1]


def spec():return load_json(REPO/floor16.REFERENCE)


class TestFloor16(unittest.TestCase):
    def test_solver_matches_offline_candidate_and_full_precision_order(self):
        s=spec(); p=floor16_parameters(s['matrix'],s['counts']['tail_ids'])
        self.assertEqual((p['N'],p['N_tail'],p['denominator']),(10847,153,13295))
        np.testing.assert_allclose(p['lower_bounds'],np.array(s['counts']['sizes'])/13295,rtol=0,atol=1e-15)
        solution=solve_floor16(s['matrix'],s['counts']['tail_ids'],s['solver']['joint_reference_weights'])
        np.testing.assert_allclose(solution['weights'],s['weights'],rtol=0,atol=1e-8)
        a=check_aggregation(s)
        self.assertLess(a['kkt_max_error'],1e-4)
        self.assertEqual(len(a['bound_clients']),22)
        self.assertEqual(len(a['free_clients']),8)
        self.assertEqual(a['active_tol'],1e-8)
        for name,w in [('joint',s['solver']['joint_reference_weights']),('joint_floor16',s['weights']),
                       ('fedavg',a['sample_weights']),('tailrw16',a['tailrw_weights'])]:
            self.assertEqual(a['objectives'][name],objective_gradient(w,s['matrix'],.1)[0])

    def test_feasibility_then_correct_one_sided_KKT_and_no_mixing(self):
        s=spec();w=np.array(s['weights']);lower=np.array(s['lower_bounds'])
        a=audit_weights(w,s['matrix'],.1,lower_bounds=lower)
        _,g=objective_gradient(w,s['matrix'],.1)
        self.assertGreater(float(np.max(g[a['bound_clients']]-a['kappa'])),.01)
        bad=w.copy();bad[0]-=1e-6;bad[1]+=1e-6
        with self.assertRaisesRegex(ValueError,'feasibility'):audit_weights(bad,s['matrix'],.1,lower)
        with self.assertRaisesRegex(ValueError,'normalization'):audit_weights(w+1e-5,s['matrix'],.1,lower)
        mixed=lower+(1-s['fedavg_floor_fraction'])*np.array(s['solver']['joint_reference_weights'])
        with self.assertRaisesRegex(ValueError,'KKT'):audit_weights(mixed,s['matrix'],.1,lower)
        with self.assertRaisesRegex(ValueError,'0.1'):audit_floor16(w,s['matrix'],s['counts']['tail_ids'],s['solver']['joint_reference_weights'],.2)
        broken=copy.deepcopy(s);broken['lower_bounds'][0]+=.001
        with self.assertRaisesRegex(ValueError,'identity'):check_aggregation(broken)

    def test_old_defaults_and_new_single_run_commands(self):
        self.assertEqual(plain_launcher.parse_args([]).methods,['fedavg','tailrw16','joint'])
        self.assertEqual(single.parse_args([]).arms,['frozen','ab'])
        with self.assertRaises(SystemExit),patch('sys.stderr'):
            launcher.parse_args(['--gamma','.5'])
        with self.assertRaises(SystemExit),patch('sys.stderr'):
            launcher.parse_args(['--client-concurrency','6'])
        plan=dict(contract=PLAIN_A_CONTRACT,aggregation=spec(),settings=dict(data_root=str(REPO/'DATA'),
            reference_run=str(REPO/'references/full10_clientlt'),num_workers=8,client_concurrency=8),code_sha256={})
        with tempfile.TemporaryDirectory() as temp:
            for mode in ('smoke','formal'):
                job=single.make_job(plan,'plain_a',mode)
                cmd=single.command_for(job,temp)
                for name,value in {'--sfra_variant':'s','--sfra_retention_weight':'0','--lac_a_lr_mult':'1',
                                   '--cliplora_freeze_a':'True'}.items():self.assertEqual(cmd[cmd.index(name)+1],value)
                self.assertNotIn('--b_transfer_enable',cmd)
            with self.assertRaisesRegex(ValueError,'single ordinary-A'):single.make_job(plan,'method_a')
            calls=[]
            with patch.object(launcher,'register',return_value=plan),patch.object(single,'preflight'), \
                 patch.object(single,'execute',side_effect=lambda job,*a:calls.append(job['mode'])), \
                 patch.object(launcher,'audit_smoke'),patch('builtins.print'):
                launcher.main(['--stage','run','--output-root',temp])
            self.assertEqual(calls,['smoke','formal'])
            calls.clear()
            with patch.object(launcher,'register',return_value=plan),patch.object(single,'preflight'), \
                 patch.object(single,'execute',side_effect=lambda job,*a:calls.append(job['mode'])), \
                 patch.object(launcher,'audit_smoke',side_effect=ValueError('smoke failed')),patch('builtins.print'):
                with self.assertRaisesRegex(ValueError,'smoke failed'):
                    launcher.main(['--stage','run','--output-root',temp])
            self.assertEqual(calls,['smoke'])

    def test_phase_artifact_audit_rejects_even_small_weight_identity_changes(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);job=fixtures.TestJointAggregation().audit_fixture(root,'plain_a','joint_floor16')
            result=audit_run(root,job,1)
            self.assertEqual(result['progress']['functional_correction_steps'],0)
            self.assertEqual(result['progress']['extra_optimizer_steps'],352)
            rows=read_csv(root/'aggregation_weights/r001_A.csv')
            rows[0]['weight']=float(rows[0]['weight'])+1e-13
            write_table(root/'aggregation_weights/r001_A.csv',rows)
            with self.assertRaisesRegex(ValueError,'phase weight identity'):audit_run(root,job,1)

    def test_actual_eight_way_cpu_B_then_A_uses_ID_weights_for_all_30_clients(self):
        """Real SGD through the production parallel_phase and inherited runtime path."""
        threads=torch.get_num_threads();torch.set_num_threads(1)
        try:
            with tempfile.TemporaryDirectory() as temp,patch('builtins.print'):
                s=spec();job=dict(arm='plain_a',mode='smoke',aggregation=s,settings=dict(client_concurrency=8))
                rt=object.__new__(runtime_class(job));rt.root=Path(temp);rt.job=job;rt.is_frozen=False
                rt.trainer=SimpleNamespace(model=tiny.Model());rt.args=SimpleNamespace(seed=42)
                rt.keys=sorted(n for n,_ in rt.trainer.model.named_parameters() if '_lora_' in n)
                rt.a_keys=[k for k in rt.keys if k.endswith('_A')];rt.b_keys=[k for k in rt.keys if k.endswith('_B')]
                rt.q={j:n/10847 for j,n in enumerate(s['counts']['sizes'])}
                rt.joint_weights=dict(enumerate(s['weights']));rt.schedule=[list(reversed(range(30)))]*100
                rt.events=[];rt.budget=[];rt.b_transfer=None;rt.sfra_config={};rt.stage_evaluate=lambda *a:None
                replicas=clone_models(rt.trainer.model,rt.keys,8)
                loaders={j:DataLoader(tiny.Data(1+j%8,j),batch_size=4,shuffle=True,drop_last=False) for j in range(30)}
                canonical=tiny.canonical_step();local_results={}
                def optimizer(model,factor):
                    opt=torch.optim.SGD([p for p in model.parameters() if p.requires_grad],lr=.001,momentum=.9)
                    return opt,torch.optim.lr_scheduler.StepLR(opt,3,gamma=1.) if factor=='B' else None
                def run_pool(state,clients,plans,factor):
                    def work(slot,client):
                        values,cost=train_local(replicas[slot],state,rt.keys,loaders[client].dataset,plans[client],
                            factor,'cpu',optimizer,canonical,torch.tensor([-.2,-.4]))
                        return values,cost,dict(client_id=client,slot=slot)
                    ordered=ordered_queue(clients,8,work)
                    # Deliberately invert insertion order after task completion.
                    results=dict(reversed(list(ordered.items())))
                    local_results[factor]=results
                    return results,dict(slots=8,clients=30)
                rt.client_pool=SimpleNamespace(loaders=loaders,run=run_pool)
                audit=SimpleNamespace(event_context={})
                def save(before,after,uploads,clients,weights,rnd,*args,**kwargs):
                    self.assertEqual(weights,dict(enumerate(s['weights'])))
                    self.assertEqual(set(clients),set(range(30)))
                    write_json(rt.root/'events'/audit.event_context['event_id']/'event.json',{})
                audit.normal=save;audit.save=save;rt.audit=audit
                start=snapshot(rt.trainer.model)
                after_B=rt.train_phase(start,1,'B')
                after_A=rt.train_phase(after_B,1,'A',True)
                for factor,before,after,keys,frozen in [('B',start,after_B,rt.b_keys,rt.a_keys),
                                                       ('A',after_B,after_A,rt.a_keys,rt.b_keys)]:
                    for key in keys:
                        local=local_results[factor]
                        expected=sum(s['weights'][j]*local[j][0][key].double() for j in range(30))
                        torch.testing.assert_close(after[key].double(),expected,atol=1e-6,rtol=1e-6)
                    for key in frozen:torch.testing.assert_close(before[key],after[key],rtol=0,atol=0)
                    rows=read_csv(rt.root/f'aggregation_weights/r001_{factor}.csv')
                    self.assertEqual([float(r['weight']) for r in rows],s['weights'])
                self.assertEqual(len(rt.budget),60)
                self.assertEqual(sum(int(r['optimizer_steps']) for r in rt.budget if r['phase']=='normal_B'),132)
                self.assertEqual(sum(int(r['optimizer_steps']) for r in rt.budget if r['phase']=='refresh_A'),44)
                rt.joint_weights[0]+=1e-13
                with self.assertRaisesRegex(ValueError,'phase weights changed'):
                    rt.phase_aggregation_weights(list(range(30)),2,'B')
        finally:torch.set_num_threads(threads)

    def test_integer_gate_boundaries_and_tradeoffs(self):
        for overall,tail,wanted in [(141730,28683,False),(141730,28802,False),
                (141929,28683,False),(141930,28683,True),(141730,28803,True),
                (141729,30000,False),(150000,28682,False)]:
            self.assertEqual(floor16.integer_decision(overall,tail)['passed'],wanted)
        with self.assertRaises(ValueError):floor16.integer_decision(141930.,28803)

    def test_raw_prediction_endpoint_ignores_rounded_metrics(self):
        labels=np.repeat(np.arange(100),100)
        samples={}
        for i,rnd in enumerate(range(81,101)):
            overall,tail=7086+int(i<10),1434+int(i<3)
            correct=np.zeros(10000,dtype=bool);correct[:overall-tail]=True;correct[8000:8000+tail]=True
            samples[rnd]=dict(class_id=labels,correct=correct)
        result=dict(predictions=samples,groups=dict(overall_acc=list(range(100)),head20_acc=list(range(20)),
            middle60_acc=list(range(20,80)),bottom20_tail_acc=list(range(80,100)),non_tail_acc=list(range(80))),
            curves=[dict(round=r,overall_acc=100.,bottom20_tail_acc=100.) for r in range(81,101)])
        ep=floor16.endpoint(result)
        self.assertEqual(ep['correct']['overall_acc'],141730)
        self.assertEqual(ep['correct']['bottom20_tail_acc'],28683)
        self.assertFalse(floor16.integer_decision(ep['correct']['overall_acc'],ep['correct']['bottom20_tail_acc'])['passed'])

    def test_snapshot_summary_gate_and_pack_are_self_contained_and_reject_stale_selection(self):
        with tempfile.TemporaryDirectory() as temp,patch('builtins.print'):
            folder=Path(temp);source=folder/'synthetic_baseline';root=folder/'candidate'
            job=fixtures.TestJointAggregation().audit_fixture(source,'plain_a','joint')
            baseline=audit_run(source,job,1)
            baseline['metadata'].update({k:'synthetic-same' for k in PAIR_HASHES})
            baseline['progress'].update(completed_round=100,normal_optimizer_steps=105600,
                                        extra_optimizer_steps=31680,elapsed_seconds=1.)
            baseline['curves']=[];baseline['predictions']={}
            labels=np.repeat(np.arange(100),100)
            for rnd in range(101):
                i=rnd-81
                total,tail=7086+int(0<=i<10),1434+int(0<=i<3)
                hit=np.zeros(10000,dtype=bool);hit[:total-tail]=True;hit[8000:8000+tail]=True
                sample=dict(sample_id=np.arange(10000),class_id=labels.copy(),correct=hit,
                            prediction=np.where(hit,labels,(labels+1)%100),logit_margin=np.where(hit,1.,-1.))
                baseline['predictions'][rnd]=sample
                baseline['curves'].append(dict(round=rnd,**metrics_from_prediction(sample,baseline['groups'])))
                np.savez_compressed(source/'predictions'/f'r{rnd:03d}.npz',**sample)
            with patch.object(floor16,'load_plain',return_value=baseline):
                record=floor16.snapshot_baseline(root,source)
            restored=floor16.read_baseline(root)
            self.assertEqual(floor16.endpoint(restored)['correct']['overall_acc'],141730)
            s=spec(); plan=dict(schema='joint_client_aggregation_v1',contract=PLAIN_A_CONTRACT,
                aggregation=s,settings={},code_sha256={},floor16_experiment=floor16.EXPERIMENT,
                baseline_record_sha256=digest(record))
            write_json(root/'experiment_plan.json',plan)
            write_json(root/'runs/seed42/plain_a/completion.json',dict(completed_round=100))
            candidate=copy.deepcopy(baseline);candidate['job']=single.make_job(plan,'plain_a')
            with patch.object(floor16,'audit_run',return_value=candidate):
                decision=floor16.summarize(root)
            self.assertEqual(decision['decision'],'retain_original_joint')
            # Gain exactly 120 tail-correct occurrences, exchange 120 non-tail
            # correct occurrences: overall unchanged, tail gate exactly met.
            sample=candidate['predictions'][100]
            gained=np.flatnonzero(~sample['correct'] & (sample['class_id']>=80))[:120]
            lost=np.flatnonzero(sample['correct'] & (sample['class_id']<80))[:120]
            sample['correct'][gained]=True;sample['correct'][lost]=False
            sample['prediction']=np.where(sample['correct'],labels,(labels+1)%100)
            sample['logit_margin']=np.where(sample['correct'],1.,-1.)
            candidate['curves'][100]=dict(round=100,**metrics_from_prediction(sample,candidate['groups']))
            with patch.object(floor16,'audit_run',return_value=candidate):
                decision=floor16.summarize(root)
            self.assertEqual(decision['decision'],'adopt_joint_floor16')
            self.assertEqual(load_json(root/'analysis/selected_aggregation.json')['aggregation']['rule'],'joint_floor16')
            import tarfile
            packed=pack_results(root)
            with tarfile.open(packed) as archive:
                self.assertIn('candidate/baseline/predictions/r100.npz',archive.getnames())
                self.assertIn('candidate/analysis/report.md',archive.getnames())
            # A later audit failure must not leave an earlier adoption selected.
            with patch.object(floor16,'audit_run',side_effect=ValueError('invalid actual weights')):
                with self.assertRaisesRegex(ValueError,'no adoption decision'):floor16.summarize(root)
            self.assertIsNone(load_json(root/'analysis/selected_aggregation.json')['aggregation'])
            payload=root/'baseline/predictions/r100.npz'
            payload.write_bytes(b'corrupted')
            with self.assertRaisesRegex(ValueError,'snapshot changed'):floor16.read_baseline(root)


if __name__=='__main__':unittest.main()
