"""Small CPU checks, including one reference/cache comparison; not GPU evidence."""
import copy
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torch.nn import functional as F

from scripts import run_cliplora_method_a_v2_aggregation as launcher
from scripts import run_cliplora_joint_aggregation as single
from tests.test_sfra_execution import visual_bank
from tools.client_aggregation.protocol import load_json, write_table, PLAIN_A_CONTRACT
from tools.client_aggregation.method_a_v2 import audit_rounds
from utils.cliplora_a_refresh import train_only
from utils.cliplora_functional_feedback import snapshot
from utils.cliplora_method_a_v2 import MethodAV2Runtime
from utils.method_a_v2_feedback import SampleFeedback
from utils.method_a_v2_math import (CONFIG, SampleSourceHistory, positive_support, make_targets,
                                    history_terms, candidate_reasons, leq)
from utils.sfra_math import projected_step, flatten
from utils.sfra_resident_feedback import ResidentWitnessFeedback

REPO=Path(__file__).resolve().parents[1]
OBSERVATIONS = {}


def bank_and_feedback(cached=True):
    bank=visual_bank(fast=cached)
    if cached:bank._fast_feedback=ResidentWitnessFeedback(bank,12,0)
    for q,t in enumerate(bank.tokens):
        t['raw_sample_ids']=[100*t['client_id']+i for i in t['local_positions']]
    feedback=SampleFeedback(bank,[[50,0,2],[0,3,0]],([0],[1],[2]),torch.tensor([-.1,-.7,-.8]))
    return bank,feedback


class TestMethodAV2(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads=torch.get_num_threads();torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):torch.set_num_threads(cls.threads)

    def test_math_gradient_radius_and_candidate_joint_checks(self):
        z=torch.tensor([.1,-.2],requires_grad=True);radius=.37
        a=torch.tensor([.3,-.1])+radius*z
        m=torch.stack((a,2*a),1)
        target=torch.tensor([[.8,.9],[.2,.6]])
        sigma=torch.tensor([[.2,.4],[.3,.7]])
        omega=torch.tensor([.2,.8])[:,None]
        loss=history_terms(m,target,sigma,omega).sum()
        dm=torch.autograd.grad(loss,m,retain_graph=True)[0]
        torch.testing.assert_close(dm,-omega/(2*sigma.square())*(target-m).clamp_min(0))
        ga=torch.autograd.grad(loss,a,retain_graph=True)[0]
        objective=.5*z.square().sum()+10*loss
        gz=torch.autograd.grad(objective,z)[0]
        torch.testing.assert_close(gz,z+10*radius*ga)
        expected=z.detach()-.1*gz;expected=expected/expected.norm().clamp_min(1)
        torch.testing.assert_close(projected_step(z.detach(),radius,ga,10),expected)
        # CE alone passing is insufficient, and half uses beta squared proximity.
        j,why=candidate_reasons(torch.ones(3),torch.ones(3),.09,1.,torch.ones(2),1.)
        self.assertEqual(why,'J')
        j,why=candidate_reasons(torch.ones(3),torch.ones(3),.05,.5,torch.ones(2),1.)
        self.assertAlmostEqual(j,.75);self.assertEqual(why,'')
        with self.assertRaises(FloatingPointError):leq(float('nan'),1.)
        losses=history_terms(torch.tensor([.5,-.5]),torch.tensor([.1,.1]),torch.ones(2),torch.ones(2)/2)
        self.assertGreater(float(losses[1]),0.)

    def test_source_window_history_binding_and_restore(self):
        manifest=[dict(sample_id=i,token_id=0,class_id=0) for i in range(2)]
        h=SampleSourceHistory(torch.tensor([[.1,.2],[-.1,.3]]),manifest,1,2)
        self.assertEqual(h.valid.tolist(),[True,False]);self.assertFalse(h.known.any())
        self.assertTrue(torch.isnan(h.kappa).all())
        tiny=positive_support(torch.tensor([[[2e-6,-1.],[2e-6,-2.]]]),[1e-12,1-1e-12])
        self.assertGreater(float(tiny[0,0]),0.)
        for r in range(1,5):h.commit([[.3,.4],[.2,.4]],tiny,r)
        saved=h.state_dict()
        restored=SampleSourceHistory.restore(saved,manifest,1,2)
        for state in (h,restored):state.commit([[.3,.4],[.2,.4]],tiny,5)
        self.assertEqual(h.registered_round.tolist(),[5,5])
        torch.testing.assert_close(h.pi,torch.tensor([[1.,0.],[1.,0.]]))
        torch.testing.assert_close(h.pi,restored.pi)
        self.assertEqual(h.block_count,0)
        with self.assertRaisesRegex(ValueError,'exactly once'):h.commit([[1.,1.],[1.,1.]],tiny,5)
        # Only one sample improves: it loses its old source; the other keeps its snapshot.
        for r in range(6,11):h.commit([[.3,.4],[.4,.5]],torch.zeros(1,2),r)
        self.assertTrue(h.known[0]);self.assertFalse(h.known[1])
        self.assertEqual(h.formation_end.tolist(),[5,10])
        self.assertTrue(torch.isnan(h.pi[1]).all())
        self.assertEqual(h.reason[1],'no_positive_support')
        self.assertEqual(h.registered_round.tolist(),[5,10])
        with self.assertRaisesRegex(ValueError,'identities'):
            SampleSourceHistory.restore(saved,list(reversed(manifest)),1,2)
        targets=make_targets(h,[5,1],torch.ones(1,2),.3)
        torch.testing.assert_close(targets['omega'],torch.tensor([5/6,1/6]))

    def test_one_small_real_reference_resident_feedback_check(self):
        left,a=bank_and_feedback(False);right,b=bank_and_feedback(True)
        state=snapshot(left.model)
        rng=torch.get_rng_state().clone()
        x=a.evaluate('post_b',cached=False)
        y=b.evaluate('post_b')
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        for k in ('sample_margins','group_ce'):torch.testing.assert_close(x[k],y[k],atol=3e-6,rtol=3e-5)
        # Independent CE formula; all views/classes, original LA and logit scale.
        expected=[]
        with torch.no_grad():
            for t in left.tokens:
                ims=torch.stack([left.datasets[t['client_id']][j]['img'] for j in t['local_positions']])
                terms=[]
                for view in (ims,ims.flip(-1)):
                    f=left.core.image_encoder(view);f=f/f.norm(dim=-1,keepdim=True)
                    logits=left.core.logit_scale.exp()*f@left.text_features.T+a.adjustment
                    terms.append(F.cross_entropy(logits,torch.full((len(ims),),t['class_id'])))
                expected.append(torch.stack(terms).mean())
        torch.testing.assert_close(x['group_ce'],torch.stack([expected[0],expected[2],expected[1]]),atol=3e-6,rtol=3e-5)
        history=SampleSourceHistory(torch.ones(len(a.manifest),2)*.5,a.manifest,len(left.tokens),left.clients)
        ta=make_targets(history,a.base_mass,torch.ones(len(left.tokens),2),.37)
        tb=copy.deepcopy(ta)
        x=a.evaluate('ordinary',targets=ta,initialize_target=True,gradient=True,cached=False)
        y=b.evaluate('ordinary',targets=tb,initialize_target=True,gradient=True)
        OBSERVATIONS['small_feedback_check']=dict(
            scope='CPU tiny VisionTransformer, reference versus production resident cache',
            samples=len(a.manifest),clients=left.clients,
            margin_max_abs=float((x['sample_margins']-y['sample_margins']).abs().max()),
            gradient_max_abs=float((x['gradient']-y['gradient']).abs().max()),
            loss_abs=abs(x['loss']-y['loss']))
        for key in ('sample_margins','group_ce','loss','gradient'):
            torch.testing.assert_close(torch.as_tensor(x[key]),torch.as_tensor(y[key]),atol=4e-6,rtol=4e-5)
        torch.testing.assert_close(ta['target'],tb['target'],atol=3e-6,rtol=3e-5)
        basis,_=torch.linalg.qr(torch.arange(left.numel*2,dtype=torch.float32).reshape(left.numel,2)+1)
        x=a.evaluate('source',basis=basis,cached=False);y=b.evaluate('source',basis=basis)
        for key in ('coordinates','gradient_norms'):torch.testing.assert_close(x[key],y[key],atol=3e-6,rtol=3e-5)
        for k,v in state.items():torch.testing.assert_close(v,left.model.state_dict()[k],atol=0,rtol=0)

    def test_entry_matrix_six_round_smoke_and_old_methods(self):
        from scripts import run_cliplora_plain_a_aggregation as old
        self.assertEqual(old.parse_args([]).methods,['fedavg','tailrw16','joint'])
        with self.assertRaises(SystemExit),patch('sys.stderr'):
            launcher.parse_args(['--methods','fedavg'])
        with tempfile.TemporaryDirectory() as temp:
            args=launcher.parse_args(['--output-root',temp,'--methods',*launcher.METHODS])
            plans=launcher.register(args)
            self.assertEqual(plans,launcher.register(args))
            for rule,plan in plans.items():
                job=single.make_job(plan,'method_a_v2','smoke')
                cmd=single.command_for(job,Path(temp)/rule)
                self.assertEqual(cmd[cmd.index('--sfra_variant')+1],'source-memory-v2')
                self.assertEqual(cmd[cmd.index('--sfra_stop_after_round')+1],'6')
                self.assertNotIn('--b_transfer_enable',cmd)
                with self.assertRaisesRegex(ValueError,'six-round'):launcher.verify_smoke(temp,plan)
            plain=dict(plans['joint_floor16'],contract=PLAIN_A_CONTRACT)
            self.assertEqual(single.smoke_rounds(single.make_job(plain,'plain_a','smoke')),1)
            with patch.object(launcher,'register',return_value={'joint':plans['joint']}),patch.object(single,'preflight'), \
                 patch.object(single,'execute') as execute,patch.object(launcher,'verify_smoke',return_value={'status':'passed'}),patch('builtins.print'):
                launcher.main(['--stage','smoke','--output-root',temp])
            self.assertEqual(execute.call_count,1)
            self.assertEqual(execute.call_args.args[0]['mode'],'smoke')

    def test_full_J_rejection_then_half_or_exact_ordinary_fallback(self):
        with tempfile.TemporaryDirectory() as temp,patch('builtins.print'):
            for fail_half in (False,True):
                rt=self.make_runtime(Path(temp)/str(fail_half),1)
                rt.history=SampleSourceHistory(torch.ones(len(rt.av2.manifest),2),rt.av2.manifest,3,2)
                middle=snapshot(rt.trainer.model);ordinary=copy.deepcopy(middle)
                # Aggregated update cancels; nonzero local RMS must still permit repair.
                delta={0:{k:torch.ones_like(middle[k])*.01 for k in rt.a_keys},
                       1:{k:torch.ones_like(middle[k])*(-.01*.3/.7) for k in rt.a_keys}}
                rt.train_phase=lambda *a,**kw:(ordinary,delta)
                rt.events=[{}];phases=[]
                def observe(state,rnd,phase,**kw):
                    phases.append(phase)
                    basis=kw.get('basis')
                    ce=torch.ones(3)*(2 if phase=='half' and fail_half else 1)
                    return dict(sample_margins=torch.ones(len(rt.av2.manifest),2)*.2,
                        group_ce=ce,gradient_norms=torch.ones(3,2),
                        coordinates=torch.zeros(3,2,basis.shape[1]) if basis is not None else None,
                        gradient=torch.ones(rt.bank.numel),loss=100. if phase=='full' else 0. if phase=='half' else .2)
                rt.observe_av2=observe;rt.costs=[dict(downlink_bytes=0)]
                accepted,row=rt.refresh(middle,1)
                self.assertEqual(row['full_reject'],'J')
                self.assertEqual(row['correction_steps'],3)
                self.assertEqual(phases,['post_b_source','ordinary','repair_z1','repair_z2','full','half'])
                self.assertEqual(row['selected_beta'],0. if fail_half else .5)
                if fail_half:
                    self.assertIs(accepted,ordinary)
                    self.assertFalse(row['nonzero_correction'])
                for k in rt.b_keys:torch.testing.assert_close(accepted[k],middle[k],atol=0,rtol=0)

    def test_new_arm_uses_existing_artifact_audit_with_v2_rules(self):
        from tests import test_joint_aggregation as fixture
        from tools.client_aggregation.protocol import METHOD_A_V2_CONTRACT, digest
        from tools.client_aggregation.analysis import audit_run
        from scripts.run_ab_validation import write_json
        with tempfile.TemporaryDirectory() as temp,patch('builtins.print'):
            root=Path(temp)
            job=fixture.TestJointAggregation().audit_fixture(root,'method_a','joint')
            job.update(arm='method_a_v2',contract=METHOD_A_V2_CONTRACT)
            cfg=load_json(root/'sfra_config.json');cfg.update(CONFIG)
            cfg.pop('classification_weight',None)
            cfg['joint_aggregation']['arm']='method_a_v2'
            write_json(root/'sfra_config.json',cfg);write_json(root/'joint_job.json',job)
            receipt=load_json(root/'joint_receipt.json');receipt['job_digest']=digest(job)
            write_json(root/'joint_receipt.json',receipt)
            progress=load_json(root/'progress.json');progress['arm']='method_a_v2'
            write_json(root/'progress.json',progress)
            row=dict(round=1,variant='source-memory-v2',correction_steps=3,skip_reason='',
                history_last_committed=1,history_last_used=0,window_count=1,selected_beta=1.,
                full_reject='',half_reject='',J0=1.,J_final=.5)
            for g in ('H','M','T'):row.update({f'final_CE_{g}':1.,f'limit_CE_{g}':1.})
            write_table(root/'sfra_rounds.csv',[row])
            audit_run(root,job,1)
            row['final_CE_T']=1.01;write_table(root/'sfra_rounds.csv',[row])
            with self.assertRaisesRegex(ValueError,'group loss'):audit_run(root,job,1)

    def make_runtime(self, folder, last):
        bank,feedback=bank_and_feedback()
        rt=object.__new__(MethodAV2Runtime)
        rt.root=Path(folder);(rt.root/'checkpoints').mkdir(parents=True,exist_ok=True)
        rt.bank,rt.av2=bank,feedback
        rt.trainer=SimpleNamespace(model=bank.model,device=bank.device)
        rt.a_keys=bank.keys;rt.b_keys=sorted(n for n,_ in bank.model.named_parameters() if n.endswith('_lora_B'))
        rt.keys=sorted(rt.a_keys+rt.b_keys);rt.base_state=snapshot(bank.model)
        rt.variant='source-memory-v2';rt.strength=10.;rt.b_transfer=None
        rt.sizes=[7,3];rt.joint_weights={0:.3,1:.7};rt.job={'aggregation':{'rule':'joint'}}
        rt.sfra_config=dict(CONFIG);rt.execution_config=None;rt.config={}
        rt.events=[];rt.budget=[];rt.evaluations=[];rt.costs=[];rt.summaries=[]
        rt.elapsed_before=0.;rt.started=time.perf_counter();rt.resume_payload=None
        rt.args=SimpleNamespace(sfra_stop_after_round=last)
        labels={j:{p:t['class_id'] for t in bank.tokens if t['client_id']==j for p in t['local_positions']} for j in range(2)}
        def phase(state,rnd,factor,extra=False,return_deltas=False):
            local={};keys=rt.a_keys if factor=='A' else rt.b_keys
            for j in range(2):
                bank.model.load_state_dict(state);train_only(bank.model,factor)
                opt=torch.optim.SGD([p for p in bank.model.parameters() if p.requires_grad],lr=.001,momentum=.9)
                ims=torch.stack([x['img'] for x in bank.datasets[j]])
                y=torch.tensor([labels[j][i] for i in range(len(ims))])
                for _ in range(1 if extra else 3):
                    opt.zero_grad();f=bank.core.image_encoder(ims);f=f/f.norm(dim=-1,keepdim=True)
                    loss=F.cross_entropy(bank.core.logit_scale.exp()*f@bank.text_features.T+feedback.adjustment,y)
                    loss.backward();opt.step()
                local[j]=snapshot(bank.model)
                rt.budget.append(dict(round=rnd,phase='refresh_A' if extra else 'normal_B',optimizer_steps=1 if extra else 3))
            out=dict(state);deltas={j:{k:local[j][k]-state[k] for k in keys} for j in local}
            for k in keys:out[k]=sum(rt.joint_weights[j]*local[j][k] for j in local)
            rt.events.append(dict(round=rnd))
            return (out,deltas) if return_deltas else out
        rt.train_phase=phase
        def publish(state,rnd,_):
            rt.evaluations.append(dict(round=rnd))
            write_table(rt.root/'round_metrics.csv',rt.evaluations)
        rt.publish=publish
        return rt

    def test_tiny_six_round_runtime_and_midwindow_checkpoint_resume(self):
        # Real tiny-model B/A SGD and cached Av2 feedback; NOT the CIFAR GPU smoke.
        with tempfile.TemporaryDirectory() as temp,patch('builtins.print'):
            rt=self.make_runtime(Path(temp)/'continuous',6)
            initial=copy.deepcopy(rt.base_state);rt.run(initial)
            final=snapshot(rt.trainer.model)
            self.assertEqual(rt.history.last_committed_round,6)
            self.assertEqual(rt.summaries[4]['window_count'],0)
            self.assertEqual(rt.summaries[5]['history_last_used'],5)
            audit_rounds(rt.summaries,load_json(rt.root/'progress.json'),6)
            OBSERVATIONS['tiny_runtime_check']=dict(
                scope='CPU tiny model with two clients and three classes; NOT the CIFAR30-client GPU smoke',
                progress=load_json(rt.root/'progress.json'),rounds=copy.deepcopy(rt.summaries))
            paused=self.make_runtime(Path(temp)/'resumed',4);paused.run(copy.deepcopy(paused.base_state))
            resumed=self.make_runtime(paused.root,6)
            resumed.resume_payload=torch.load(paused.root/'checkpoints/sfra_last.pt',weights_only=False)
            resumed.run(copy.deepcopy(resumed.base_state))
            for k,v in final.items():torch.testing.assert_close(v,resumed.trainer.model.state_dict()[k],atol=0,rtol=0)
            for k in ('H','pi','kappa','sum_a','block_min'):
                torch.testing.assert_close(getattr(rt.history,k),getattr(resumed.history,k),equal_nan=True,atol=0,rtol=0)
            self.assertEqual(rt.history.reason,resumed.history.reason)


if __name__=='__main__':unittest.main()
