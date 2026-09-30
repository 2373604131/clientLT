"""CPU checks for frozen AB pairing and the actual direct-calibration optimizer."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from torch.nn import functional as F

from scripts import run_ab_validation as suite
from scripts import run_cliplora_sfra as launcher
from tools.sfra import ab_validation as val
from tools.sfra.calibration_reference import build_reference, control_config, protocol_signature
from tools.sfra.maintext import FROZEN, PAIR_HASHES
from tools.sfra.summary import METRICS, read_csv, write_csv, pack
from utils.cliplora_b_calibration import DirectCalibrationControl
from utils.cliplora_b_shared_transfer import shared_calibration_batches, shared_transfer_config
from utils.sfra_resident_feedback import execution_config_v2
from test_sfra_b_shared_transfer import tiny_transfer, A_KEY, B_KEY


def args(root,arms=('a','ab'),seeds=(42,43)):
    root=Path(root);dataset=root/'data/cifar-100/cifar-100-python';dataset.mkdir(parents=True,exist_ok=True)
    for name in ('train','test','meta'):(dataset/name).write_bytes(b'fixture')
    return SimpleNamespace(output_root=root,seeds=list(seeds),arms=list(arms),partition='client-longtail',
        protocol_seed=42,dirichlet_beta=.5,reference_run=None,num_workers=0,data_root=root/'data')


def completed(root,job,offset=0.):
    run=Path(root)/job['run'];run.mkdir(parents=True,exist_ok=True)
    cfg={**FROZEN,'schema_version':'sfra_cp_v1','variant':val.ARMS[job['arm']], 'seed':job['seed'],
        'protocol_seed':42,'partition':job['partition'],'witness_batch_size':8,'a_parameter_order':['layer.A'],
        'base_training_config':dict(seed=job['seed'],protocol_seed=42,dirichlet_beta=.5,
            normal_steps_expected=300,extra_steps_expected=90)}
    if cfg['variant'].endswith('-cp'):cfg['classification_weight']=1.
    if job['arm'] in ('ab','a-calibration'):
        b=shared_transfer_config(SimpleNamespace(b_transfer_probe_step=.1,b_transfer_lr=.3,b_transfer_reg=.001,
            b_transfer_non_tail_sampling='class-cyclic',b_transfer_tail_weight=.35))
        if job['arm']=='a-calibration':
            reference=build_reference(Path(root)/job['dependency']);b=control_config(b,reference)
            suite.write_json(run/'calibration_reference.json',reference)
        cfg['b_transfer']=b
    suite.write_json(run/'sfra_config.json',cfg)
    done=dict(completed_round=100,official_test_passes=101,variant=cfg['variant'],normal_optimizer_steps=300,
        extra_optimizer_steps=90,elapsed_seconds=123.,functional_correction_steps=0 if job['arm']=='s' else 270)
    if job['arm'] in ('ab','a-calibration'):
        done.update(b_transfer_events=8,b_transfer_optimizer_steps=16)
        event_rows=[]
        for rnd in range(30,101,10):
            folder=run/f'b_transfer_rounds/r{rnd:03d}';folder.mkdir(parents=True,exist_ok=True)
            batch=dict(tail_positions=[0],non_tail_positions=[1])
            suite.write_json(folder/'calibration_manifest.json',dict(feedback_clients=[0],module_order=[B_KEY],
                shared_donor_ids=[0],recipients={'0':dict(donor_ids=[0],batches=[batch,batch])}))
            write_csv(folder/'optimization_steps.csv',[dict(step=s,effective_transfer_norm_after=.01*s) for s in [1,2]])
            event_rows.append(dict(round=rnd,algorithm_forward_images=8,algorithm_backward_images=4,diagnostic_forward_images=8,
                extra_upload_bytes=12,extra_downlink_bytes=24,seconds=2))
        write_csv(run/'b_transfer_rounds.csv',event_rows)
    suite.write_json(run/'completion.json',done)
    suite.write_json(run/'ab_validation_run.json',dict(schema_version=val.SCHEMA,job=job,code_unchanged=True))
    write_csv(run/'round_metrics.csv',[dict(round=r,**{m:50+r/100+(offset if r>=30 else 0) for m in METRICS}) for r in reversed(range(101))])
    for epoch in range(80,100):
        write_csv(run/f'per_class_accuracy_epoch_{epoch}.csv',[dict(class_id=c,per_class_acc=50+(epoch+1)/100+offset) for c in range(100)])
    suite.write_json(run/'class_prior.json',dict(counts=[5]*100,prior=[.01]*100,tail_ids=list(range(80,100))))
    suite.write_json(run/'bridge_metadata.json',{k:k+str(job['seed']) for k in PAIR_HASHES})
    write_csv(run/'partition_manifest.csv',[dict(client_id=0,local_position=0,class_id=80,raw_sample_id=17)])
    suite.write_json(run/'execution_config.json',execution_config_v2(128,4))
    if job['arm']!='s':suite.write_json(run/'private_witness_manifest.json',[dict(client_id=0,class_id=80,local_positions=[0])])
    for name in ['budget','sfra_costs']:
        write_csv(run/(name+'.csv'),[dict(upload_bytes=10,modeled_downlink_bytes=20,downlink_bytes=20,forward_images=3,backward_images=2)])
    return run


class TestABSuite(unittest.TestCase):
    def test_frozen_jobs_commands_dependencies_and_source_lock(self):
        with tempfile.TemporaryDirectory() as temp:
            a=args(temp,('a','ab','a-calibration'));jobs=suite.plan_jobs(a);suite.merge_plan(temp,jobs)
            self.assertEqual(len(jobs),6)
            for j in jobs:
                cmd=suite.command_for(j,temp)
                self.assertIn('--fast-execution-v2',cmd)
                self.assertEqual(cmd[cmd.index('--feedback-batch-size')+1],'128')
                self.assertEqual(cmd[cmd.index('--protocol-seed')+1],'42')
                self.assertEqual('--b-transfer' in cmd,j['arm']!='a')
                self.assertEqual('--calibration-reference' in cmd,j['arm']=='a-calibration')
                self.assertTrue(j['run'].replace('\\','/').startswith('runs/'+j['arm']+'/'))
            changed=copy.deepcopy(jobs);changed[0]['code_sha256']['federated_main.py']='changed'
            with self.assertRaisesRegex(ValueError,'changed'):suite.merge_plan(temp,changed)
            control=next(j for j in jobs if j['arm']=='a-calibration')
            with self.assertRaisesRegex(ValueError,'paired AB first'):suite.preflight_job(control,temp,True)

    def test_reference_partition_mismatch_and_path_escape(self):
        with tempfile.TemporaryDirectory() as temp:
            a=args(temp);a.reference_run=Path(temp)/'ref';r=a.reference_run;(r/'protocol').mkdir(parents=True)
            for n in ['partition_manifest.csv','protocol/eri_protocol.json','protocol/probe_manifest.csv']:(r/n).write_text('{}')
            suite.write_json(r/'protocol/full_schedule.json',dict(schedule=[list(range(30))]*100))
            suite.write_json(r/'bridge_metadata.json',dict(topology='noniid-labeldir-fine'))
            with self.assertRaisesRegex(ValueError,'Reference partition differs'):suite.plan_jobs(a)
            with self.assertRaises(ValueError):val.safe_run(temp,'../outside')

    def test_pair_summaries_do_not_wait_for_unrelated_controls_and_separate_seed42(self):
        with tempfile.TemporaryDirectory() as temp:
            jobs=suite.plan_jobs(args(temp,('a','ab','a-calibration')));suite.merge_plan(temp,jobs)
            for j in jobs:
                if j['arm']!='a-calibration':completed(temp,j,offset=.2 if j['arm']=='ab' else 0.)
            status,audits=val.summarize(temp)
            self.assertEqual(sum(x['status']=='incomplete' for x in status),0)
            pairs=read_csv(Path(temp)/'analysis/paired_per_seed.csv')
            self.assertEqual(len(pairs),2)
            self.assertTrue(all(abs(float(x['delta_last20_bottom20_tail_acc'])-.2)<1e-9 for x in pairs))
            aggregates=read_csv(Path(temp)/'analysis/paired_summary.csv')
            self.assertEqual({x['cohort'] for x in aggregates},{'all','development','confirmation'})
            self.assertEqual(next(x['n'] for x in aggregates if x['cohort']=='all'),'2')
            self.assertEqual(next(x['delta_last20_bottom20_tail_acc_sd'] for x in aggregates if x['cohort']=='confirmation'),'')
            methods=read_csv(Path(temp)/'analysis/method_summary.csv')
            self.assertEqual(next(x['n'] for x in methods if x['arm']=='ab' and x['cohort']=='confirmation'),'1')
            classes=read_csv(Path(temp)/'analysis/paired_per_class.csv')
            self.assertEqual(len(classes),200)
            self.assertEqual(sum(x['is_tail']=='True' for x in classes),40)

    def test_control_norm_reference_samples_and_pair_mismatch_exclusion(self):
        with tempfile.TemporaryDirectory() as temp:
            jobs=suite.plan_jobs(args(temp,('a','ab','a-calibration'),(42,)));suite.merge_plan(temp,jobs)
            for j in jobs:completed(temp,j,offset=.2 if j['arm']=='ab' else .1 if j['arm']=='a-calibration' else 0.)
            status,audits=val.summarize(temp)
            self.assertTrue(all(x['eligible'] for x in audits))
            control=next(j for j in jobs if j['arm']=='a-calibration');run=Path(temp)/control['run']
            f=run/'b_transfer_rounds/r030/calibration_manifest.json';v=val.load_json(f);v['recipients']['0']['batches'][0]['tail_positions']=[9];suite.write_json(f,v)
            status,audits=val.summarize(temp)
            self.assertEqual(next(x['status'] for x in status if x['arm']=='a-calibration'),'invalid')
            self.assertFalse(next(x['eligible'] for x in audits if x['comparison']=='ab minus a-calibration'))
            self.assertFalse(next(x['eligible'] for x in audits if x['comparison']=='a-calibration minus a'))
            self.assertTrue(next(x['eligible'] for x in audits if x['comparison']=='ab minus a'))
            ab=next(j for j in jobs if j['arm']=='ab');f=Path(temp)/ab['run']/'execution_config.json';v=val.load_json(f);v['feedback_forward_batch_size']=64;suite.write_json(f,v)
            status,_=val.summarize(temp)
            self.assertEqual(next(x['status'] for x in status if x['arm']=='ab'),'invalid')

    def test_no_donor_round_remains_valid_evidence_and_has_no_reference_steps(self):
        with tempfile.TemporaryDirectory() as temp:
            jobs=suite.plan_jobs(args(temp,('ab',),(42,)));suite.merge_plan(temp,jobs);run=completed(temp,jobs[0])
            path=run/'b_transfer_rounds/r030/calibration_manifest.json';m=val.load_json(path)
            m['shared_donor_ids']=[];suite.write_json(path,m)
            write_csv(run/'b_transfer_rounds/r030/optimization_steps.csv',[])
            path=run/'completion.json';done=val.load_json(path);done['b_transfer_optimizer_steps']=14;suite.write_json(path,done)
            self.assertEqual(build_reference(run)['events']['30']['norms'],[])
            status,_=val.summarize(temp);self.assertEqual(status[0]['status'],'complete')

    def test_resume_completed_and_interrupted_registration(self):
        with tempfile.TemporaryDirectory() as temp,patch('builtins.print'):
            jobs=suite.plan_jobs(args(temp,('a',),(42,)));suite.merge_plan(temp,jobs);j=jobs[0];r=completed(temp,j)
            with patch.object(suite.subprocess,'run') as run:
                suite.execute_job(j,temp);run.assert_not_called()
            (r/'completion.json').unlink();(r/'checkpoints').mkdir();(r/'checkpoints/sfra_last.pt').touch()
            with self.assertRaisesRegex(ValueError,'--resume'):suite.execute_job(j,temp)
            (r/'ab_validation_run.json').unlink()
            suite.write_json(suite.registration(temp,j),dict(schema_version=val.SCHEMA,job=j))
            def train(command,**kwargs):
                self.assertIn('--resume',command);completed(temp,j)
            with patch.object(suite.subprocess,'run',side_effect=train):suite.execute_job(j,temp,True)
            val.check_receipt(r,j)

    def test_plan_default_never_trains_and_pack_drops_checkpoints(self):
        import tarfile
        with tempfile.TemporaryDirectory() as temp,patch('builtins.print'):
            with patch('sys.argv',['entry','--output-root',temp,'--seeds','42']),patch.object(suite.subprocess,'run') as run:
                suite.main();run.assert_not_called()
            jobs=val.load_json(Path(temp)/val.PLAN_NAME)['jobs']
            for j in jobs:completed(temp,j)
            p=Path(temp)/jobs[0]['run']/'model.pt';p.touch()
            val.summarize(temp);pack(temp)
            archive=Path(temp).parent/(Path(temp).name+'_analysis.tar.gz')
            try:
                with tarfile.open(archive) as f:
                    self.assertFalse(any(n.endswith('.pt') for n in f.getnames()))
                    self.assertTrue(any(n.endswith('paired_summary.csv') for n in f.getnames()))
            finally:archive.unlink()

    def test_backend_launcher_embeds_portable_reference_and_guards_resume(self):
        with tempfile.TemporaryDirectory() as temp,patch('builtins.print'):
            jobs=suite.plan_jobs(args(temp,('ab',),(42,)));ref=completed(temp,jobs[0]);out=Path(temp)/'direct'
            argv=['entry','--method','full-cp','--b-transfer','--transfer-mode','shared','--transfer-lr','.3',
                '--transfer-non-tail-sampling','class-cyclic','--transfer-tail-weight','.35','--calibration-reference',str(ref),
                '--output-root',str(out),'--fast-execution-v2','--feedback-batch-size','128']
            def prepare(a,r):(r/'protocol').mkdir(parents=True);return r/'protocol/full_schedule.json',''
            with patch('sys.argv',argv),patch.object(launcher,'prepare_protocol',prepare),patch.object(launcher.subprocess,'run') as execute:
                launcher.main()
            cmd=execute.call_args.args[0];self.assertIn('--b_calibration_reference',cmd)
            r=Path(cmd[cmd.index('--output-dir')+1]);self.assertIn('_directnorm_',r.name)
            (r/'checkpoints').mkdir();(r/'checkpoints/sfra_last.pt').touch()
            with patch('sys.argv',argv+['--resume']),patch.object(launcher.subprocess,'run') as execute:launcher.main()
            self.assertEqual(execute.call_args.args[0][execute.call_args.args[0].index('--b_calibration_reference')+1],str(r/'calibration_reference.json'))


class TestDirectCalibration(unittest.TestCase):
    def fixture(self,temp):
        tr=tiny_transfer(temp);tr.__class__=DirectCalibrationControl
        tr.bank=SimpleNamespace(batch_size=64)
        tr.monitors={k:dict(tail_positions=[p for c,v in tr.groups[k].items() if c in tr.tail for p in v],
                           non_tail_positions=[p for c,v in tr.groups[k].items() if c not in tr.tail for p in v]) for k in tr.recipients}
        batches={str(k):shared_calibration_batches(tr.groups[k],tr.tail,42,30,k,'class-cyclic') for k in tr.recipients}
        tr.config.update(calibration_profile='direct_norm_matched',tail_weight=.35,learning_rate=.3,regularization=0.,
            norm_reference=dict(events={'30':dict(norms=[.01,.015],feedback_clients=tr.recipients,batches=batches)}))
        return tr

    def test_actual_two_steps_match_combined_adam_projection_and_preserve_model_rng(self):
        with tempfile.TemporaryDirectory() as temp:
            tr=self.fixture(temp);start=copy.deepcopy(tr.model.state_dict());ordinary=copy.deepcopy(start);ordinary[B_KEY]+=.02
            expected=torch.nn.Parameter(torch.zeros_like(start[B_KEY]));opt=torch.optim.Adam([expected],lr=.3,betas=(.9,.999),eps=1e-8)
            for step,target in enumerate([.01,.015]):
                opt.zero_grad(set_to_none=True);losses=[]
                for k in tr.recipients:
                    b=tr.config['norm_reference']['events']['30']['batches'][str(k)][step]
                    x,y=tr.batch(k,b['tail_positions']+b['non_tail_positions']);n=len(b['tail_positions'])
                    z=.5*F.linear(F.linear(x,start[A_KEY]),ordinary[B_KEY]+expected)
                    li=(z-F.one_hot(y,2)).square().mean(1)
                    losses.append(.35*li[:n].mean()+.65*li[n:].mean() if n<len(y) else li.mean())
                torch.stack(losses).mean().backward();opt.step()
                with torch.no_grad():expected.mul_(target/float((.5*(expected.double()@start[A_KEY].double())).norm()))
            torch.manual_seed(21);rng=torch.get_rng_state().clone()
            tr.cache_updates(None,[])  # No donor deltas are needed or accessed.
            result=tr.apply_shared(start,ordinary,30)
            torch.testing.assert_close(result[B_KEY],ordinary[B_KEY]+expected.detach(),atol=1e-7,rtol=1e-6)
            self.assertTrue(torch.equal(torch.get_rng_state(),rng))
            self.assertTrue(torch.equal(result[A_KEY],ordinary[A_KEY]))
            for k,v in tr.model.state_dict().items():self.assertTrue(torch.equal(v,start[k]))
            self.assertEqual(tr.pending['summary']['algorithm_backward_images'],10)
            self.assertEqual(tr.pending['summary']['optimizer_steps'],2)
            self.assertEqual(tr.pending['summary']['learned_c_parameters'],0)
            for r,n in zip(tr.pending['steps'],[.01,.015]):self.assertAlmostEqual(r['effective_transfer_norm_after'],n,places=8)
            tr.observe_global(result,30,committed=True);tr.flush()
            state=tr.state_dict();self.assertEqual(tr.progress()['b_transfer_events'],1)
            tr.load_state_dict(state);self.assertEqual(tr.progress()['b_transfer_optimizer_steps'],2)

    def test_bad_batches_zero_direction_and_zero_radius(self):
        with tempfile.TemporaryDirectory() as temp:
            tr=self.fixture(temp);start=copy.deepcopy(tr.model.state_dict())
            residual={B_KEY:torch.zeros_like(start[B_KEY])}
            with self.assertRaisesRegex(ValueError,'zero effective'):tr.project(residual,.1,start)
            tr.project(residual,0.,start)
            tr.config['norm_reference']['events']['30']['batches']['1'][0]['tail_positions']=[99]
            with self.assertRaisesRegex(ValueError,'batches differ'):tr.apply_shared(start,start,30)
            for k,v in tr.model.state_dict().items():self.assertTrue(torch.equal(v,start[k]))

    def test_no_donor_reference_skips_feedback_optimization(self):
        with tempfile.TemporaryDirectory() as temp:
            tr=self.fixture(temp);start=copy.deepcopy(tr.model.state_dict());ordinary=copy.deepcopy(start);ordinary[B_KEY]+=.02
            tr.config['norm_reference']['events']['30']['norms']=[]
            result=tr.apply_shared(start,ordinary,30)
            self.assertTrue(torch.equal(result[B_KEY],ordinary[B_KEY]))
            self.assertEqual(tr.pending['summary']['optimizer_steps'],0)
            self.assertEqual(tr.pending['summary']['algorithm_backward_images'],0)
            for k,v in tr.model.state_dict().items():self.assertTrue(torch.equal(v,start[k]))

    def test_nonfinite_failure_restores_state(self):
        with tempfile.TemporaryDirectory() as temp:
            tr=self.fixture(temp);start=copy.deepcopy(tr.model.state_dict());original=tr.forward
            calls=[0]
            def broken(x,y,diagnostic=False):
                loss,z,c=original(x,y,diagnostic)
                if not diagnostic:
                    calls[0]+=1
                    if calls[0]>=1:loss=loss*torch.nan
                return loss,z,c
            tr.forward=broken
            with self.assertRaisesRegex(ValueError,'Nonfinite'):tr.apply_shared(start,start,30)
            for k,v in tr.model.state_dict().items():self.assertTrue(torch.equal(v,start[k]))


if __name__=='__main__':unittest.main()
