"""Mathematical contracts, actual two-step feedback, identity and resume guards."""
import copy
import io
import math
from contextlib import redirect_stdout
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torch.nn import functional as F

from scripts import run_cliplora_b_tail_equal as entry
from scripts import run_cliplora_b_routes as launcher
from tools.sfra import b_routes as protocol
from tools.sfra.summary import read_csv, write_csv
from tools.sfra.maintext import load_json
from scripts.run_ab_validation import write_json
from utils.b_tail_equal import ARM, OBJECTIVE, rescale_tail_coefficients, manifest_coefficients
from utils.b_route_math import EffectiveCoordinates, norm
from utils.cliplora_b_routes import RouteTransfer, route_config, group_loss
from test_sfra_b_shared_transfer import tiny_transfer, A_KEY, B_KEY


def toy(folder):
    t = tiny_transfer(folder)
    t.__class__ = RouteTransfer
    t.bank = SimpleNamespace(batch_size=2)
    t.config = route_config(t.config, dict(arm=ARM))
    t.model.layer.w_lora_B = torch.nn.Parameter(torch.tensor([[.2,-.1],[.4,.3],[-.1,.2]]))
    t.parameters = dict(t.model.named_parameters())
    t.tail = {0,1}
    t.groups = {1:{0:[0,1],1:[2],2:[3]},2:{0:[0],1:[1,2,3]}}
    t.adjustment = torch.tensor([.2, .3, .5]).log()
    images = {1:torch.tensor([[.7,.2],[-.4,1.],[.9,-.2],[.3,.4]]),
              2:torch.tensor([[.2,1.],[-.8,.4],[.1,.2],[.4,-.2]])}
    labels = {1:torch.tensor([0,0,1,2]),2:torch.tensor([0,1,1,1])}
    t.batch = lambda k,p: (images[k][p],labels[k][p])
    def forward(x,y,diagnostic=False):
        logits = t.model.layer(x)
        field = 'diagnostic_forward_images' if diagnostic else 'algorithm_forward_images'
        t.pending['summary'][field] += len(y)
        return F.cross_entropy(logits+t.adjustment,y,reduction='none'), logits, logits
    t.forward = forward
    return t


class CoefficientTest(unittest.TestCase):
    def test_actual_tail_mass_and_within_class_ratios(self):
        labels={3:[0,0,1,2],7:[0,1,1,1],9:[]}
        old,new,rows=rescale_tail_coefficients(labels,{0,1})
        self.assertAlmostEqual(sum(r['old_mass'] for r in rows if r['is_tail']),.675)
        for r in rows:
            if r['is_tail']: self.assertAlmostEqual(r['new_mass'],.3375)
        self.assertEqual(new[3][3],old[3][3])
        self.assertAlmostEqual(new[3][0]/new[7][0],old[3][0]/old[7][0])
        self.assertNotAlmostEqual(new[3][0],new[7][0])  # Not pooled same-class image means.
        self.assertAlmostEqual(sum(map(sum,new.values())),1.)

    def test_mixed_only_batches_and_absent_non_tail_classes(self):
        _,_,rows=rescale_tail_coefficients({0:[0,0,1,2],1:[0,1,3]}, {0,1})
        self.assertTrue(all(math.isclose(r['new_mass'],.175) for r in rows if r['is_tail']))
        self.assertAlmostEqual(sum(r['new_mass'] for r in rows if not r['is_tail']),.65)

    def test_non_tail_only_and_no_silent_missing_tail_renormalization(self):
        _,_,rows=rescale_tail_coefficients({1:[0,1],2:[2,2,3]}, {0,1})
        self.assertTrue(all(math.isclose(r['new_mass'],.25) for r in rows if r['is_tail']))
        with self.assertRaisesRegex(ValueError,'Missing declared tail'):
            rescale_tail_coefficients({1:[0,2]}, {0,1})
        with self.assertRaises(ValueError): rescale_tail_coefficients({1:[]},{0})

    def test_client_iteration_order_does_not_change_identity_weights(self):
        a={8:[0,0,1,2],2:[1,0]}
        first=rescale_tail_coefficients(a,{0,1})[1]
        second=rescale_tail_coefficients(dict(reversed(list(a.items()))),{0,1})[1]
        for k in a: torch.testing.assert_close(torch.tensor(first[k]),torch.tensor(second[k]),rtol=0,atol=0)


class RuntimeTest(unittest.TestCase):
    def test_actual_two_steps_match_central_objective_and_freeze_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            t=toy(tmp); start=copy.deepcopy(t.model.state_dict()); ordinary=copy.deepcopy(start)
            ordinary[B_KEY]+=.1
            manifests=t.manifests(30)
            # Reference uses a single centralized graph and independently derived coefficients.
            coords=EffectiveCoordinates({B_KEY:start[A_KEY]},{B_KEY:.5})
            x=torch.zeros_like(coords.encode({B_KEY:ordinary[B_KEY]-start[B_KEY]})[B_KEY],requires_grad=True)
            radius=norm(coords.encode({B_KEY:ordinary[B_KEY]-start[B_KEY]}).values())
            norms=[]; losses_before=[]
            for step in range(2):
                loss=x.sum()*0
                # Mixed client's class masses are (2/3*.35/2,1/3*.35/2),
                # tail-only client's masses are (1/4/2,3/4/2).
                masses={0:7/60+1/8,1:7/120+3/8}; beta=.675
                for k,v in manifests.items():
                    b=v['batches'][step]; inp,lab=t.batch(k,b['tail_positions']+b['non_tail_positions'])
                    r=coords.decode({B_KEY:x})[B_KEY]
                    logits=.5*F.linear(F.linear(inp,start[A_KEY]),ordinary[B_KEY]+r)
                    ls=F.cross_entropy(logits+t.adjustment,lab,reduction='none')
                    for i,c in enumerate(lab.tolist()):
                        alpha=(.35/3/2 if c<2 else .65/2) if k==1 else 1/4/2
                        w=alpha*beta/(2*masses[c]) if c<2 else alpha
                        loss=loss+ls[i]*w
                grad=torch.autograd.grad(loss,x)[0]; length=norm([grad])
                norms.append(length);losses_before.append(float(loss.detach()))
                update=x.detach()-radius/2*grad/length
                update=update*min(1.,radius/norm([update]))
                x=update.requires_grad_()
            expected=coords.decode({B_KEY:x})[B_KEY].detach()
            t.begin(Path(tmp),30,ARM)
            with t.session():
                t.copy_parameters(ordinary)
                actual=t.optimize(ordinary,start,manifests,ARM,1.)
            torch.testing.assert_close(actual[B_KEY],expected,atol=1e-7,rtol=1e-5)
            for row,gn,loss in zip(t.pending['steps'],norms,losses_before):
                self.assertAlmostEqual(row['gradient_norm'],gn,places=6)
                self.assertAlmostEqual(row['la_before'],loss,places=6)
            self.assertEqual(t.pending['summary']['optimizer_steps'],2)
            self.assertEqual(t.pending['summary']['algorithm_backward_images'],16)
            self.assertTrue(all(r['client_weight']==1. and r['server_gradient_scale']==1. for r in t.pending['feedback']))
            self.assertEqual(t.pending['summary']['unique_donors'],0)
            for k,v in start.items():
                torch.testing.assert_close(t.model.state_dict()[k],v,atol=0,rtol=0)
                self.assertIsNone(t.parameters[k].grad)
            self.assertLessEqual(norm(coords.encode(actual).values()),radius*(1+2e-6))
            before=t.objective_rows[0]
            self.assertAlmostEqual(before['batch1_la'],losses_before[0],places=6)
            # Reconstructed logs use actual sample identities, not completion order.
            write_csv(Path(tmp)/'partition_manifest.csv',[dict(client_id=k,class_id=c,local_position=p)
                      for k,cs in t.groups.items() for c,ps in cs.items() for p in ps])
            write_json(Path(tmp)/'calibration_manifest.json',dict(arm=ARM,shared_donor_ids=[],feedback_clients=t.recipients,recipients=manifests))
            protocol.audit_feedback_coefficients(Path(tmp),Path(tmp),t.tail)
            coefficients=read_csv(Path(tmp)/'feedback_coefficients.csv')
            coefficients[0]['new_coefficient']=float(coefficients[0]['new_coefficient'])/2
            write_csv(Path(tmp)/'feedback_coefficients.csv',coefficients)
            with self.assertRaisesRegex(ValueError,'value differs'):
                protocol.audit_feedback_coefficients(Path(tmp),Path(tmp),t.tail)

    def test_single_tail_class_reduces_to_original_f(self):
        with tempfile.TemporaryDirectory() as tmp:
            results=[]
            for arm in ('F',ARM):
                t=tiny_transfer(tmp);t.__class__=RouteTransfer;t.bank=SimpleNamespace(batch_size=2)
                start=copy.deepcopy(t.model.state_dict()); ordinary=copy.deepcopy(start);ordinary[B_KEY]+=.1
                t.begin(Path(tmp)/arm,30,arm)
                with t.session():
                    t.copy_parameters(ordinary)
                    results.append(t.optimize(ordinary,start,t.manifests(30),arm,1.))
            torch.testing.assert_close(results[0][B_KEY],results[1][B_KEY],atol=1e-7,rtol=1e-5)

    def test_identity_mismatch_and_missing_tail_fail_before_update(self):
        with tempfile.TemporaryDirectory() as tmp:
            t=toy(tmp); t.begin(Path(tmp),30,ARM)
            manifest=t.manifests(30)
            t.feedback_coefficients,_,_=manifest_coefficients(manifest,t.groups,t.tail)
            labels=t.feedback_coefficients[0][1]['labels']
            with self.assertRaisesRegex(ValueError,'labels/order'):
                t.weighted_feedback_loss(torch.ones(len(labels)),torch.tensor(labels).roll(1),1,0)
            t.tail.add(99)
            state=copy.deepcopy(t.model.state_dict())
            with self.assertRaisesRegex(ValueError,'Missing declared tail'):
                t.optimize(state,state,manifest,ARM,1.)
            self.assertEqual(t.pending['summary']['optimizer_steps'],0)
            self.assertEqual(t.pending['summary']['algorithm_forward_images'],0)


class ProtocolTest(unittest.TestCase):
    def test_independent_entry_defaults_and_legacy_defaults(self):
        with patch.object(entry,'run_routes') as run:
            entry.main(['--stage','plan'])
        args=launcher.parser().parse_args(run.call_args.args[0]);launcher.validate_args(args)
        self.assertEqual(args.arms,[ARM]);self.assertEqual(args.seeds,[42])
        self.assertEqual(str(args.output_root).replace('\\','/'),'output/cifar100_LT/b_tail_equal_v1')
        old=launcher.parser().parse_args([]);launcher.validate_args(old)
        self.assertEqual(old.arms,['P','C','F'])
        self.assertNotIn('feedback_objective',route_config({},dict(arm='F')))
        self.assertEqual(route_config({},dict(arm=ARM))['feedback_objective'],OBJECTIVE)
        with self.assertRaises(SystemExit): entry.main(['--arms','F'])

    def test_plan_preflight_no_training_and_distinct_resume_contract(self):
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()):
            with patch.object(launcher.subprocess,'run',side_effect=AssertionError('must not train')):
                entry.main(['--stage','preflight','--output-root',tmp])
            preflight=load_json(Path(tmp)/'preflight.json')
            self.assertTrue(preflight['ready']);self.assertFalse(preflight['gpu_execution_tested'])
            args=launcher.parser().parse_args(['--arms',ARM,'--output-root',tmp]);launcher.validate_args(args)
            run,spec,cmd=next(launcher.jobs(args))
            self.assertIn(ARM,str(run));self.assertEqual(spec['settings']['arm'],ARM)
            self.assertIn('utils/b_tail_equal.py',spec['code_sha256'])
            self.assertEqual(cmd[cmd.index('--sfra_variant')+1],'full-cp')
            self.assertEqual(cmd[cmd.index('--lac_la_tau')+1],'1')
            launcher.merge_plan(Path(tmp),[(run,spec,cmd)])
            bad=copy.deepcopy(spec);bad['settings']['arm']='F'
            with self.assertRaisesRegex(ValueError,'changed'):
                launcher.merge_plan(Path(tmp),[(run,bad,cmd)])
            run.mkdir(parents=True);write_json(run/'route_spec.json',bad)
            args.resume=True
            with self.assertRaisesRegex(ValueError,'different route contract'):
                launcher.execute(args,run,spec,cmd)


if __name__=='__main__':
    unittest.main()
