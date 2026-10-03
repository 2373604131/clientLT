"""Guard math, actual two-step donor-C optimizer, worker and launch contracts."""
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch.nn import functional as F

from scripts import run_cliplora_b_class_guard as launcher
from scripts import train_cliplora_b_class_guard as worker
from tools.sfra import b_class_guard as collection
from tools.sfra.ab_validation import code_hashes
from tools.sfra.maintext import FROZEN, PAIR_HASHES
from tools.sfra.summary import METRICS, read_csv, write_csv, pack
from utils.b_class_guard_math import guard_config, guard_losses
from utils.cliplora_b_class_guard import GuardedSharedDonorBTransfer
from utils.cliplora_b_guard_runtime import runtime_class
from utils.cliplora_b_shared_transfer import SharedDonorBTransfer, shared_calibration_batches, shared_transfer_config
from utils.cliplora_sfra import SFRARuntime
from utils.sfra_resident_feedback import execution_config_v2
from test_sfra_b_shared_transfer import tiny_transfer, A_KEY, B_KEY


def transfer_config():
    return shared_transfer_config(SimpleNamespace(b_transfer_probe_step=.1,b_transfer_lr=.3,b_transfer_reg=.001,
        b_transfer_non_tail_sampling='class-cyclic',b_transfer_tail_weight=.35))


def fixture_args(root,guard='client'):
    root=Path(root);ref=root/'ref';(ref/'protocol').mkdir(parents=True,exist_ok=True)
    for name in ('partition_manifest.csv','protocol/eri_protocol.json','protocol/probe_manifest.csv'):
        (ref/name).write_text('{}',encoding='utf-8')
    launcher.write_json(ref/'bridge_metadata.json',dict(topology='client-longtail'))
    launcher.write_json(ref/'protocol/full_schedule.json',dict(schedule=[list(range(30))]*100))
    data=root/'data/cifar-100/cifar-100-python';data.mkdir(parents=True,exist_ok=True)
    for name in ('train','test','meta'):(data/name).write_bytes(b'fixture')
    return launcher.parser().parse_args(['--guard',guard,'--output-root',str(root/'output'),
        '--reference-run',str(ref),'--data-root',str(root/'data'),'--num-workers','0'])


def toy(root,variant):
    t=tiny_transfer(root)
    t.__class__=GuardedSharedDonorBTransfer
    t.model.layer.w_lora_B=torch.nn.Parameter(torch.tensor([[.6,.1],[.3,.2],[.1,-.1]]))
    t.parameters=dict(t.model.named_parameters())
    t.tail={0,1}
    t.groups=[{2:[0]}, {0:[0],1:[1,2],2:[3]}, {0:[0],1:[1,2]}]
    t.probes={1:{0:[0],1:[1,2]},2:{0:[0],1:[1,2]}}
    t.counts=torch.tensor([[0,0,1],[1,2,1],[1,2,0]])
    # Only class-1 output can move. Identical inputs with conflicting labels
    # ensure improving its majority supervision harms other class cells.
    t.original_deltas={j:{B_KEY:torch.tensor([[0.,0.],[.2,.1],[0.,0.]])*(j+1)} for j in (0,1,2)}
    images={1:torch.tensor([[1.,0.],[1.,0.],[1.,0.],[1.,0.]]),2:torch.tensor([[1.,0.],[1.,0.],[1.,0.]])}
    labels={1:torch.tensor([0,1,1,2]),2:torch.tensor([0,1,1])}
    t.batch=lambda k,pos:(images[k][pos],labels[k][pos])
    def forward(x,y,diagnostic=False):
        prediction=t.model.layer(x)
        loss=(prediction-F.one_hot(y,3)).square().mean(1)
        t.pending['summary']['diagnostic_forward_images' if diagnostic else 'algorithm_forward_images']+=len(y)
        return loss,prediction,prediction
    t.forward=forward
    t.config=guard_config(transfer_config(),variant)
    t.pending=dict(round=30,folder=Path(root),steps=[],feedback=[],summary=dict(
        algorithm_forward_images=0,algorithm_backward_images=0,diagnostic_forward_images=0,
        client_backward_batches=0,optimizer_steps=0,feedback_synchronizations=0,
        extra_downlink_bytes=0,extra_upload_bytes=0))
    manifests={k:dict(batches=shared_calibration_batches(t.groups[k],t.tail,42,30,k,'class-cyclic')) for k in t.recipients}
    return t,manifests


def completed_fixture(run,spec,offset=0.):
    run=Path(run);run.mkdir(parents=True,exist_ok=True)
    s=spec['settings'];variant=s['guard']
    cfg={**FROZEN,'variant':'full-cp','classification_weight':1.,'seed':s['seed'],'protocol_seed':s['protocol_seed'],
         'partition':'client-longtail','witness_batch_size':8,'a_parameter_order':[A_KEY],
         'base_training_config':dict(seed=s['seed'],protocol_seed=42,dirichlet_beta=.5,
                                    normal_steps_expected=300,extra_steps_expected=90),
         'b_transfer':guard_config(transfer_config(),variant),'b_class_guard_experiment':spec}
    launcher.write_json(run/'sfra_config.json',cfg)
    launcher.write_json(run/'guard_spec.json',spec)
    launcher.write_json(run/'guard_receipt.json',dict(spec_digest=collection.digest(spec),code_unchanged=True))
    done=dict(completed_round=100,official_test_passes=101,variant='full-cp',normal_optimizer_steps=300,
              extra_optimizer_steps=90,elapsed_seconds=10,b_transfer_events=8,b_transfer_optimizer_steps=16)
    launcher.write_json(run/'completion.json',done);launcher.write_json(run/'progress.json',done)
    # Include real duplicate metadata fields to catch the old dict(**row) bug.
    write_csv(run/'round_metrics.csv',[dict(round=r,seed=s['seed'],partition='client-longtail',
        **{m:50+r/100+(offset if r>=30 else 0) for m in METRICS}) for r in range(101)])
    for epoch in range(80,100):
        write_csv(run/f'per_class_accuracy_epoch_{epoch}.csv',[dict(class_id=c,per_class_acc=50+(epoch+1)/100+offset) for c in range(100)])
    launcher.write_json(run/'class_prior.json',dict(counts=[5]*100,tail_ids=list(range(80,100))))
    launcher.write_json(run/'bridge_metadata.json',{k:k+str(s['seed']) for k in PAIR_HASHES})
    launcher.write_json(run/'private_witness_manifest.json',[dict(client_id=0,class_id=80,local_positions=[0])])
    launcher.write_json(run/'execution_config.json',execution_config_v2(128,4))
    write_csv(run/'partition_manifest.csv',[dict(client_id=0,local_position=0,class_id=80,raw_sample_id=10)])
    events=[]
    for rnd in range(30,101,10):
        folder=run/f'b_transfer_rounds/r{rnd:03d}';folder.mkdir(parents=True,exist_ok=True)
        batch=dict(tail_positions=[0],non_tail_positions=[1])
        launcher.write_json(folder/'calibration_manifest.json',dict(feedback_clients=[0],shared_donor_ids=[0],
            recipients={'0':dict(donor_ids=[0],batches=[batch,batch])}))
        write_csv(folder/'optimization_steps.csv',[dict(round=rnd,step=k,guard_variant=variant,effective_transfer_norm_after=.01*k) for k in (1,2)])
        write_csv(folder/'c_loss_trace.csv',[dict(round=rnd,c_step=k,guard_variant=variant,fixed_pool_guard=0.) for k in (0,1,2)])
        for name in ('probe_metrics','client_feedback_steps','guard_class_trace','guard_client_trace'):
            write_csv(folder/(name+'.csv'),[dict(round=rnd,guard_variant=variant,client_id=0)])
        np.savez_compressed(folder/'guard_references.npz',losses=np.asarray([1.]))
        events.append(dict(round=rnd,algorithm_forward_images=4,algorithm_backward_images=4,diagnostic_forward_images=12,
            guard_diagnostic_backward_images=2,extra_upload_bytes=20,extra_downlink_bytes=20,seconds=1))
    write_csv(run/'b_transfer_rounds.csv',events)
    return cfg


class TestGuardMath(unittest.TestCase):
    def test_cancellation_and_gradient_isolate_harmed_class(self):
        ref=torch.tensor([1.,1.,1.],requires_grad=True)
        loss=torch.tensor([.9,1.08,.95],requires_grad=True)
        values=guard_losses(loss,ref,torch.tensor([80,81,0]),2)
        self.assertEqual(float(values['client'].detach()),0.)
        self.assertAlmostEqual(float(values['class'].detach()),.35*.04,places=7)
        values['class'].backward()
        torch.testing.assert_close(loss.grad,torch.tensor([0.,.175,0.]))
        self.assertIsNone(ref.grad)

    def test_sample_weights_tail_only_and_no_cross_client_cancellation(self):
        x=torch.tensor([1.2,1.2,.9],requires_grad=True)
        v=guard_losses(x,torch.ones(3),torch.tensor([80,80,81]),3)
        self.assertAlmostEqual(float(v['class'].detach()),2/3*.2,places=7)
        self.assertAlmostEqual(float(v['client'].detach()),.1,places=7)
        other=guard_losses(torch.tensor([.2]),torch.ones(1),torch.tensor([80]),1)
        self.assertGreater(float(((v['client']+other['client'])/2).detach()),0.)
        self.assertAlmostEqual(sum(c['local_weight'] for c in v['cells']),1.)

    def test_zero_reference_has_zero_gradient(self):
        x=torch.tensor([1.,2.,3.],requires_grad=True)
        v=guard_losses(x,x.detach().clone(),torch.tensor([80,81,0]),2)
        (v['client']+v['class']).backward()
        self.assertTrue(torch.equal(x.grad,torch.zeros_like(x)))

    def test_jensen_and_finite_difference_with_cross_entropy(self):
        logits=torch.tensor([[.8,.2,-.3],[.8,.2,-.3],[-.1,.2,.4]],dtype=torch.float64,requires_grad=True)
        labels=torch.tensor([0,1,2])
        current=F.cross_entropy(logits,labels,reduction='none')
        ref=current.detach()+torch.tensor([.1,-.08,.05],dtype=torch.float64)
        fn=lambda z:guard_losses(F.cross_entropy(z,labels,reduction='none'),ref,labels,2)['class']
        self.assertTrue(torch.autograd.gradcheck(fn,(logits,),eps=1e-6,atol=1e-5))
        v=guard_losses(current,ref,labels,2)
        self.assertGreaterEqual(float(v['class'].detach()),float(v['client'].detach()))

    def test_bad_references_and_config_rejected(self):
        for ref in (torch.ones(2),torch.tensor([float('nan'),1.,1.])):
            with self.assertRaises(ValueError):guard_losses(torch.ones(3),ref,torch.tensor([0,1,2]),2)
        with self.assertRaises(ValueError):guard_losses(torch.ones(2),torch.ones(2),torch.tensor([0,0]),1)
        with self.assertRaises(ValueError):guard_config(transfer_config(),'unknown')
        wrong=transfer_config();wrong['tail_weight']=.5
        with self.assertRaises(ValueError):guard_config(wrong,'class')


class TestGuardOptimizer(unittest.TestCase):
    def test_full_shared_event_commit_and_no_donor_noop(self):
        for active in (True,False):
            with self.subTest(active=active),tempfile.TemporaryDirectory() as tmp:
                t,_=toy(tmp,'class')
                t.bank=SimpleNamespace(batch_size=8)
                t.monitors={k:dict(tail_positions=[p for c,ps in t.groups[k].items() if c in t.tail for p in ps],
                                  non_tail_positions=[p for c,ps in t.groups[k].items() if c not in t.tail for p in ps])
                            for k in t.recipients}
                if not active:
                    for delta in t.original_deltas.values():delta[B_KEY].zero_()
                before=copy.deepcopy(t.model.state_dict());rng=torch.get_rng_state().clone()
                with patch('builtins.print'):
                    result=t.apply_shared(before,before,30)
                    t.observe_global(result,30,committed=True)
                self.assertTrue(torch.equal(result[A_KEY],before[A_KEY]))
                self.assertTrue(torch.equal(torch.get_rng_state(),rng))
                self.assertEqual(t.progress()['b_transfer_optimizer_steps'],2 if active else 0)
                saved=torch.load(Path(tmp)/'b_transfer_rounds/r030/commit.pt',weights_only=False)
                torch.testing.assert_close(result[B_KEY],saved['ordinary_lora'][B_KEY]+saved['transfer_residual'][B_KEY])
                self.assertEqual(t.records[0]['guard_variant'],'class')
                self.assertIsNone(t.pending)
                if active:
                    self.assertFalse(torch.equal(result[B_KEY],before[B_KEY]))
                else:
                    self.assertTrue(torch.equal(result[B_KEY],before[B_KEY]))

    def test_actual_two_steps_match_single_combined_objective(self):
        for variant in ('client','class'):
            with self.subTest(variant=variant),tempfile.TemporaryDirectory() as tmp:
                t,manifests=toy(tmp,variant)
                start=copy.deepcopy(t.model.state_dict());raw=copy.deepcopy(t.original_deltas)
                donors=[0,2];basis=torch.stack([raw[j][B_KEY] for j in donors])
                c=torch.nn.Parameter(torch.zeros(2,2,2));opt=torch.optim.Adam([c],lr=.3,betas=(.9,.999),eps=1e-8)
                expected=[]
                for step in range(2):
                    opt.zero_grad(set_to_none=True);loss=c.sum()*0.
                    for client in t.recipients:
                        batch=manifests[client]['batches'][step]
                        x,y=t.batch(client,batch['tail_positions']+batch['non_tail_positions'])
                        def per_image(b):
                            pred=.5*F.linear(F.linear(x,start[A_KEY]),b)
                            return (pred-F.one_hot(y,3)).square().mean(1)
                        current=per_image(start[B_KEY]+torch.bmm(basis,c).mean(0))
                        reference=per_image(start[B_KEY]).detach()
                        n=len(batch['tail_positions'])
                        la=current[:n].mean() if n==len(y) else .35*current[:n].mean()+.65*current[n:].mean()
                        loss=loss+(la+guard_losses(current,reference,y,n)[variant])/len(t.recipients)
                    loss=loss+.001*c.square().sum()/2
                    loss.backward();opt.step();expected.append(c.detach().clone())
                with t.session(),patch('builtins.print'):
                    result=t.calibrate_shared(donors,manifests,start)
                torch.testing.assert_close(result[B_KEY],c.detach(),rtol=2e-5,atol=2e-7)
                self.assertEqual(t.pending['summary']['optimizer_steps'],2)
                self.assertEqual(t.pending['summary']['algorithm_backward_images'],14)
                self.assertEqual(t.pending['summary']['algorithm_forward_images'],14)
                self.assertEqual(t.pending['summary']['diagnostic_forward_images'],42)
                self.assertEqual(t.pending['steps'][0]['guard_gradient_norm'],0.)
                self.assertGreater(t.pending['steps'][1]['guard_gradient_norm'],0.)
                self.assertGreater(t.pending['summary']['guard_diagnostic_backward_images'],0)
                self.assertEqual([r['c_step'] for r in t.pending['loss_trace']],[0,1,2])
                for k,v in start.items():
                    self.assertTrue(torch.equal(t.model.state_dict()[k],v));self.assertIsNone(t.parameters[k].grad)
                for j in raw:self.assertTrue(torch.equal(t.original_deltas[j][B_KEY],raw[j][B_KEY]))
                self.assertTrue((Path(tmp)/'guard_references.npz').is_file())

    def test_off_uses_exact_legacy_path(self):
        with tempfile.TemporaryDirectory() as a,tempfile.TemporaryDirectory() as b:
            old,ma=toy(a,'off');new,mb=toy(b,'off')
            old.__class__=SharedDonorBTransfer;old.config=transfer_config()
            sa=copy.deepcopy(old.model.state_dict());sb=copy.deepcopy(new.model.state_dict())
            with old.session(),patch('builtins.print'):ca=old.calibrate_shared([0,2],ma,sa)
            with new.session(),patch('builtins.print'):cb=new.calibrate_shared([0,2],mb,sb)
            self.assertTrue(torch.equal(ca[B_KEY],cb[B_KEY]))
            self.assertEqual(old.pending['steps'],new.pending['steps'])
            self.assertEqual(old.pending['feedback'],new.pending['feedback'])
            self.assertEqual(old.pending['summary'],new.pending['summary'])

    def test_wrong_reference_identity_fails_and_round_state_is_resumable(self):
        with tempfile.TemporaryDirectory() as tmp:
            t,m=toy(tmp,'class')
            t._guard_references={(1,0):dict(positions=[0],labels=torch.tensor([0]),losses=torch.ones(1))}
            with self.assertRaises(ValueError):t.checked_reference(1,0,dict(tail_positions=[1],non_tail_positions=[]),torch.tensor([0]))
            t.records=[dict(optimizer_steps=2,guard_variant='class',client_backward_batches=4,feedback_synchronizations=2)]
            state=t.state_dict();t.records=[];t.load_state_dict(state)
            self.assertEqual(t.progress()['b_transfer_optimizer_steps'],2)
            self.assertEqual(t.progress()['guard_variant'],'class')


class TestGuardLaunch(unittest.TestCase):
    def test_contract_commands_variants_preflight_and_inputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths=[]
            for mode in ('client','class','off'):
                a=fixture_args(tmp,mode);launcher.validate_args(a)
                spec=launcher.make_spec(a);launcher.preflight(a,spec)
                run=launcher.run_directory(a);paths.append(run)
                cmd=launcher.command_for(a,run,run/'protocol/full_schedule.json',run/'protocol/partition_source.csv')
                self.assertTrue(cmd[2].endswith('train_cliplora_b_class_guard.py'))
                for flag,value in [('--b_transfer_tail_weight','0.35'),('--sfra_retention_weight','10.0'),
                                   ('--sfra_classification_weight','1.0'),('--sfra_feedback_batch_size','128'),('--round','100')]:
                    self.assertEqual(cmd[cmd.index(flag)+1],value)
                self.assertNotIn('--b_problem2_variant',cmd)
            self.assertEqual(len(set(paths)),3)
            spec['code_sha256']['utils/b_class_guard_math.py']='changed'
            with self.assertRaisesRegex(ValueError,'changed'):launcher.preflight(a,spec)

    def test_launch_resume_and_config_rejection(self):
        with tempfile.TemporaryDirectory() as tmp:
            a=fixture_args(tmp,'client');run=launcher.run_directory(a)
            def fake_train(cmd,**kwargs):
                spec=collection.load_json(run/'guard_spec.json')
                completed_fixture(run,spec)
                (run/'completion.json').unlink()
                launcher.write_json(run/'progress.json',dict(completed_round=30))
                (run/'checkpoints').mkdir(exist_ok=True);(run/'checkpoints/sfra_last.pt').touch()
            a.stop_after_round=30
            with patch.object(launcher.subprocess,'run',side_effect=fake_train) as execute,patch('builtins.print'):
                launcher.execute(a)
            original=collection.load_json(run/'command.json')
            self.assertEqual(execute.call_count,1)
            a.resume=True;a.stop_after_round=0
            def finish(cmd,**kwargs):
                completed_fixture(run,collection.load_json(run/'guard_spec.json'))
            with patch.object(launcher.subprocess,'run',side_effect=finish) as execute,patch('builtins.print'):
                launcher.execute(a)
            cmd=execute.call_args.args[0]
            self.assertEqual(cmd[cmd.index('--sfra_resume')+1],str(run/'checkpoints/sfra_last.pt'))
            self.assertEqual(cmd[cmd.index('--sfra_stop_after_round')+1],'0')
            self.assertEqual(collection.load_json(run/'command.json'),original)
            a.num_workers=1
            with self.assertRaisesRegex(ValueError,'different guard contract'):launcher.execute(a)

    def test_worker_patch_is_process_local_and_hook_records_contract(self):
        with tempfile.TemporaryDirectory() as tmp:
            a=fixture_args(tmp);spec=launcher.make_spec(a);run=launcher.run_directory(a)
            launcher.write_json(run/'guard_spec.json',spec)
            cls=runtime_class(spec);r=object.__new__(cls)
            r.variant='full-cp';r.strength=10.;r.classification_strength=1.;r.method='sfra_full-cp'
            r.args=SimpleNamespace(seed=42,split_seed=42);r.sfra_config={'b_transfer':transfer_config()}
            r.configure_experiment()
            self.assertEqual(r.sfra_config['b_class_guard_experiment'],spec)
            self.assertEqual(r.sfra_config['b_transfer']['guard_variant'],'client')
            def inspect(*args,**kwargs):
                from utils import cliplora_sfra
                self.assertIsNot(cliplora_sfra.SFRARuntime,SFRARuntime)
                self.assertNotIn('--guard-spec',worker.sys.argv)
                raise RuntimeError('intentional worker failure')
            with patch.object(worker.runpy,'run_path',side_effect=inspect):
                with self.assertRaisesRegex(RuntimeError,'intentional'):
                    worker.main(['--guard-spec',str(run/'guard_spec.json'),'--output-dir',str(run)])
            from utils import cliplora_sfra
            self.assertIs(cliplora_sfra.SFRARuntime,SFRARuntime)

    def test_completed_tainted_receipt_cannot_be_relabelled_valid(self):
        with tempfile.TemporaryDirectory() as tmp:
            a=fixture_args(tmp);run=launcher.run_directory(a);spec=launcher.make_spec(a)
            completed_fixture(run,spec)
            launcher.write_json(run/'guard_receipt.json',dict(spec_digest=collection.digest(spec),code_unchanged=False))
            a.resume=True
            with self.assertRaisesRegex(ValueError,'Invalid training receipt'):launcher.execute(a)
            self.assertFalse(collection.load_json(run/'guard_receipt.json')['code_unchanged'])


class TestGuardCollection(unittest.TestCase):
    def test_old_baseline_pair_requires_same_frozen_training_sources(self):
        with tempfile.TemporaryDirectory() as tmp:
            a=fixture_args(tmp,'class');run=launcher.run_directory(a)
            spec=launcher.make_spec(a);completed_fixture(run,spec,.2)
            old=Path(tmp)/'baselines/ab/seed42/run'
            cfg=completed_fixture(old,spec)
            cfg.pop('b_class_guard_experiment');cfg['b_transfer']=transfer_config()
            launcher.write_json(old/'sfra_config.json',cfg)
            receipt=dict(code_unchanged=True,job=dict(seed=42,code_sha256=code_hashes(launcher.REPO)))
            launcher.write_json(old/'ab_validation_run.json',receipt)
            status,audits=collection.summarize(a.output_root,Path(tmp)/'baselines')
            self.assertTrue(all(x['status']=='complete' for x in status))
            self.assertTrue(audits[0]['valid'])
            receipt['job']['code_sha256']['federated_main.py']='different'
            launcher.write_json(old/'ab_validation_run.json',receipt)
            _,audits=collection.summarize(a.output_root,Path(tmp)/'baselines')
            self.assertFalse(audits[0]['valid'])
            self.assertIn('source fingerprints',audits[0]['reason'])

    def test_complete_partial_pairing_and_analysis_pack(self):
        with tempfile.TemporaryDirectory() as tmp:
            frozen=code_hashes(launcher.REPO)
            for mode,offset in [('client',0.),('class',.2)]:
                a=fixture_args(tmp,mode);run=launcher.run_directory(a)
                completed_fixture(run,launcher.make_spec(a),offset)
            a.guard='off';launcher.write_json(launcher.run_directory(a)/'guard_spec.json',launcher.make_spec(a))
            status,audits=collection.summarize(a.output_root)
            self.assertEqual([s['status'] for s in status].count('complete'),2)
            self.assertEqual([s['status'] for s in status].count('pending'),1)
            self.assertTrue(all(x['valid'] for x in audits))
            pairs=read_csv(a.output_root/'analysis/paired.csv')
            self.assertAlmostEqual(float(pairs[0]['delta_bottom20_tail_acc']),.2)
            self.assertEqual(len(read_csv(a.output_root/'analysis/curves.csv')),202)
            (launcher.run_directory(a)/'model.pt').write_bytes(b'not for archive')
            with patch('builtins.print'):pack(a.output_root)
            import tarfile
            with tarfile.open(a.output_root.parent/(a.output_root.name+'_analysis.tar.gz')) as tar:
                self.assertFalse(any(x.endswith('model.pt') for x in tar.getnames()))
                self.assertTrue(any(x.endswith('guard_references.npz') for x in tar.getnames()))
            self.assertEqual(code_hashes(launcher.REPO),frozen)

    def test_invalid_rows_and_mismatched_feedback_are_excluded(self):
        with tempfile.TemporaryDirectory() as tmp:
            for mode in ('client','class'):
                a=fixture_args(tmp,mode);completed_fixture(launcher.run_directory(a),launcher.make_spec(a))
            run=launcher.run_directory(a)
            manifest=run/'b_transfer_rounds/r030/calibration_manifest.json'
            saved=collection.load_json(manifest);saved['recipients']['0']['batches'][0]['tail_positions']=[3]
            launcher.write_json(manifest,saved)
            status,audits=collection.summarize(a.output_root)
            self.assertFalse(audits[0]['valid'])
            self.assertIn('sample identities',audits[0]['reason'])
            path=run/'per_class_accuracy_epoch_80.csv';rows=read_csv(path);rows[0]['per_class_acc']='nan';write_csv(path,rows)
            status,_=collection.summarize(a.output_root)
            self.assertEqual(sum(r['status']=='invalid' for r in status),1)


if __name__=='__main__':
    unittest.main()
