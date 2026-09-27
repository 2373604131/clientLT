"""Directed source constraints, actual shared-model gradients, and isolation."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch.nn import functional as F

from test_sfra_b_shared_transfer import tiny_transfer, A_KEY, B_KEY
from scripts import run_cliplora_sfra as launcher
from tools.sfra.summary import METRICS, summarize, pack, write_csv, read_csv
from utils.b_directed_math import (PURPOSE, directed_config, resolve_tail_clients, select_sources,
    build_class_bases, reconstruct_class_residual, class_macro_loss)
from utils.cliplora_b_directed import DirectedDonorBTransfer
from utils.cliplora_b_shared_transfer import shared_transfer_config
from utils.cliplora_b_transfer import differentiable_b_residual


def args(**updates):
    values = dict(num_users=3, partition='client-longtail', head_client_ratio=1/3,
        b_transfer_probe_step=.1, b_transfer_lr=.03, b_transfer_reg=.001,
        b_directed_topk=3, b_directed_min_gain=1e-6, b_directed_tail_clients='')
    return SimpleNamespace(**{**values, **updates})


def configured(root, **updates):
    t = tiny_transfer(root)
    t.__class__ = DirectedDonorBTransfer
    t.config = directed_config(t.config, args(**updates))
    t.target_clients, t.class_totals = [1, 2], {0:3}
    t.monitors = {1:dict(tail_positions=[0], non_tail_positions=[]),
                  2:dict(tail_positions=[0,1], non_tail_positions=[])}
    t.feedback_batch_size = 2
    t.bank = SimpleNamespace(batch_size=2)
    # An actually helpful donor on this toy objective; the only eligible source.
    xx = torch.cat([t.batch(1,[0])[0], t.batch(2,[0,1])[0]])
    prediction = t.model.layer(xx)
    loss = (prediction-torch.tensor([[1.,0.]]).expand_as(prediction)).square().mean()
    gradient, = torch.autograd.grad(loss, t.model.layer.w_lora_B)
    t.original_deltas[0][B_KEY] = -gradient.detach()
    return t


class DirectedMathTests(unittest.TestCase):
    def test_target_group_is_protocol_defined_not_tail_presence(self):
        self.assertEqual(resolve_tail_clients(args(num_users=30, head_client_ratio=.9)), [27,28,29])
        self.assertEqual(resolve_tail_clients(args(partition='noniid-labeldir-fine', b_directed_tail_clients='2,0')), [0,2])
        for settings in [dict(partition='noniid-labeldir-fine'), dict(b_directed_tail_clients='1,1'),
                         dict(b_directed_tail_clients='0,1,2'), dict(b_directed_tail_clients='3')]:
            with self.assertRaises(ValueError): resolve_tail_clients(args(**settings))

    def test_selection_enforces_label_absence_gain_topk_and_tie_rule(self):
        counts = torch.tensor([[0,3],[2,0],[0,0],[1,1]])
        mixtures = select_sources({0:{0:.1,2:.1},1:{1:-.1,2:.2}},counts,[3],1,1e-6)
        self.assertEqual(mixtures,{0:{0:1.},1:{2:1.}})
        for scores in [{0:{1:1.}}, {0:{3:1.}}, {0:{0:float('nan')}}]:
            with self.assertRaises(ValueError): select_sources(scores,counts,[3],3,0)
        self.assertEqual(select_sources({0:{0:1e-6}},counts,[3],3,1e-6),{})

    def test_same_union_different_edges_change_actual_forward_basis(self):
        counts=torch.zeros(3,2,dtype=torch.long)
        deltas={0:{B_KEY:torch.tensor([[1.,0.],[0.,0.]])},1:{B_KEY:torch.tensor([[0.,0.],[0.,2.]])},
                2:{B_KEY:torch.ones(2,2)*100}}
        aa={0:{0:.9,1:.1},1:{1:1.}}
        bb={0:{0:.1,1:.9},1:{1:1.}}
        _, a=build_class_bases(deltas,aa,[B_KEY],counts,[2])
        _, b=build_class_bases(deltas,bb,[B_KEY],counts,[2])
        matrices={B_KEY:torch.stack([torch.eye(2),torch.zeros(2,2)]).requires_grad_()}
        ra=reconstruct_class_residual(a,matrices,2)[B_KEY]
        rb=reconstruct_class_residual(b,matrices,2)[B_KEY]
        self.assertFalse(torch.allclose(ra,rb))
        torch.testing.assert_close(ra,(.9*deltas[0][B_KEY]+.1*deltas[1][B_KEY])/2)
        deltas[2][B_KEY] *= 10000
        _, unchanged=build_class_bases(deltas,aa,[B_KEY],counts,[2])
        torch.testing.assert_close(unchanged[B_KEY],a[B_KEY])
        self.assertFalse(a[B_KEY].requires_grad)
        ra.square().sum().backward()
        self.assertIsNotNone(matrices[B_KEY].grad)

    def test_missing_classes_do_not_renormalize_other_classes(self):
        bases={B_KEY:torch.eye(2).unsqueeze(0)}
        matrices={B_KEY:torch.eye(2).unsqueeze(0)}
        torch.testing.assert_close(reconstruct_class_residual(bases,matrices,4)[B_KEY],torch.eye(2)/4)

    def test_class_macro_is_sample_mean_within_class_not_recipient_mean(self):
        losses=torch.tensor([1.,3.,8.,4.],requires_grad=True)
        labels=torch.tensor([0,0,0,1])
        expected=(losses[:3].mean()+losses[3])/2
        actual=class_macro_loss(losses[:1],labels[:1],{0:3,1:1})+class_macro_loss(losses[1:],labels[1:],{0:3,1:1})
        torch.testing.assert_close(actual,expected)
        actual.backward()
        torch.testing.assert_close(losses.grad,torch.tensor([1/6,1/6,1/6,1/2]))
        with self.assertRaises(ValueError): class_macro_loss(torch.ones(1),torch.tensor([2]),{0:1})

    def test_profile_pins_purpose_and_rejects_other_objectives(self):
        base=shared_transfer_config(args())
        cfg=directed_config(base,args())
        self.assertEqual(cfg['purpose'],PURPOSE)
        self.assertEqual(cfg['tail_weight'],1.)
        self.assertEqual(cfg['calibration_non_tail_cap'],0)
        self.assertEqual(cfg['target_clients'],[1,2])
        for settings in [dict(b_problem2_variant='E11'),dict(b_transfer_tail_weight=.35),
                         dict(b_directed_min_gain=float('inf')),dict(b_directed_topk=0)]:
            with self.assertRaises(ValueError): directed_config(base,args(**settings))


class DirectedRuntimeTests(unittest.TestCase):
    def test_constructor_filters_leaked_tail_clients_and_uses_full_target_pool(self):
        with tempfile.TemporaryDirectory() as tmp:
            toy=tiny_transfer(tmp)
            a=args(b_directed_tail_clients='2',sfra_feedback_batch_size=2,seed=42)
            source=[[SimpleNamespace(label=c) for c, positions in g.items() for _ in positions] for g in toy.groups]
            runtime=SimpleNamespace(root=Path(tmp),args=a,b_keys=[B_KEY],tail=[0],
                sfra_config={'b_transfer':directed_config(toy.config,a)},
                bank=SimpleNamespace(model=toy.model,core=toy.model,device=torch.device('cpu'),batch_size=2,
                    tokens=[dict(client_id=1,class_id=0,local_positions=[0]),dict(client_id=2,class_id=0,local_positions=[0])]),
                audit=SimpleNamespace(counts=toy.counts),trainer=SimpleNamespace(
                    training_logit_adjustment=torch.zeros(2),dm=SimpleNamespace(dataset=SimpleNamespace(federated_train_x=source))))
            t=DirectedDonorBTransfer(runtime)
            self.assertEqual(t.recipients,[2])
            self.assertEqual(t.class_totals,{0:2})
            self.assertEqual(t.monitors[2],dict(tail_positions=[0,1],non_tail_positions=[]))
            self.assertEqual(json.loads((Path(tmp)/'b_transfer_manifest.json').read_text())['recipients'],[2])

    def test_end_to_end_screen_optimize_reconstruct_commit_and_restore(self):
        with tempfile.TemporaryDirectory() as tmp:
            t=configured(tmp)
            start=copy.deepcopy(t.model.state_dict())
            original=copy.deepcopy(t.original_deltas)
            ordinary={A_KEY:start[A_KEY],B_KEY:start[B_KEY]+.01}
            rng=torch.random.get_rng_state().clone()
            with patch('builtins.print'): result=t.apply_shared(start,ordinary,30)
            self.assertEqual(t.last_mixtures,{0:{0:1.}})
            rebuilt=reconstruct_class_residual(t.last_bases,t.last_matrices,len(t.class_totals))
            torch.testing.assert_close(result[B_KEY],ordinary[B_KEY]+rebuilt[B_KEY])
            self.assertFalse(torch.allclose(result[B_KEY],ordinary[B_KEY]))
            self.assertTrue(torch.equal(result[A_KEY],start[A_KEY]))
            for key in start:
                self.assertTrue(torch.equal(t.model.state_dict()[key],start[key]))
                self.assertIsNone(t.parameters[key].grad)
            self.assertTrue(torch.equal(rng,torch.random.get_rng_state()))
            self.assertEqual(t.pending['summary']['algorithm_backward_images'],6)
            self.assertEqual(t.pending['summary']['optimizer_steps'],2)
            self.assertTrue(all(r['non_tail_samples']==0 for r in t.pending['feedback']))
            self.assertTrue(all(r['is_tail'] for r in t.pending['metrics']))
            scores=read_csv(Path(tmp)/'b_transfer_rounds/r030/donor_scores.csv')
            self.assertTrue(all(int(r['donor_class_count'])==0 for r in scores))
            with np.load(Path(tmp)/'b_transfer_rounds/r030/matrices.npz') as archive:
                np.testing.assert_allclose(archive['source_weights'],[[1.]])
            with patch('builtins.print'): t.observe_global(result,30,committed=True)
            self.assertEqual(t.progress()['b_transfer_events'],1)
            saved=t.state_dict()
            other=configured(tmp)
            other.load_state_dict(saved)
            self.assertEqual(other.progress(),t.progress())
            torch.testing.assert_close(t.last_bases[B_KEY][0],original[0][B_KEY])

    def test_local_chunking_matches_combined_shared_adam(self):
        with tempfile.TemporaryDirectory() as tmp:
            results=[]
            for size in [1,32]:
                t=configured(tmp)
                t.feedback_batch_size=size
                start=copy.deepcopy(t.model.state_dict())
                with patch('builtins.print'): results.append(t.apply_shared(start,start,30)[B_KEY])
            torch.testing.assert_close(results[0],results[1],atol=1e-7,rtol=1e-6)

    def test_normalized_full_logit_la_forward_and_rank_four(self):
        with tempfile.TemporaryDirectory() as tmp:
            t=configured(tmp)
            layer=t.model.layer
            layer.w_lora_A=torch.nn.Parameter(torch.cat([layer.w_lora_A.detach(),torch.eye(2)],dim=0),requires_grad=False)
            layer.w_lora_B=torch.nn.Parameter(torch.cat([layer.w_lora_B.detach(),.02*torch.eye(2)],dim=1))
            t.parameters=dict(t.model.named_parameters())
            t.ranks={B_KEY:4}
            t.core=SimpleNamespace(image_encoder=layer,dtype=torch.float32,logit_scale=torch.tensor(1.))
            t.bank.text_features=torch.eye(2)
            t.adjustment=torch.tensor([-.3,.1])
            del t.forward  # Exercise inherited real normalized-logit LA, not the toy MSE override.
            xx=torch.cat([t.batch(1,[0])[0],t.batch(2,[0,1])[0]])
            feats=F.normalize(layer(xx),dim=-1)
            loss=F.cross_entropy(t.core.logit_scale.exp()*feats+t.adjustment,torch.zeros(3,dtype=torch.long))
            gradient,=torch.autograd.grad(loss,layer.w_lora_B)
            t.original_deltas={j:{B_KEY:torch.zeros_like(layer.w_lora_B)} for j in t.selected}
            t.original_deltas[0][B_KEY]=-gradient.detach()
            start=copy.deepcopy(t.model.state_dict())
            with patch('builtins.print'): result=t.apply_shared(start,start,30)
            self.assertEqual(t.last_matrices[B_KEY].shape,(1,4,4))
            self.assertTrue(torch.isfinite(result[B_KEY]).all())
            self.assertEqual(t.pending['summary']['optimizer_steps'],2)
            self.assertTrue(all(r['class_id']==0 for r in t.pending['metrics']))

    def test_joint_c_optimizer_matches_one_batched_reference_objective(self):
        with tempfile.TemporaryDirectory() as tmp:
            t=configured(tmp)
            start=copy.deepcopy(t.model.state_dict())
            donor=t.original_deltas[0][B_KEY].clone()
            with patch('builtins.print'): t.apply_shared(start,start,30)
            c=torch.nn.Parameter(torch.zeros(1,2,2))
            optimizer=torch.optim.Adam([c],lr=t.config['learning_rate'],betas=(.9,.999),eps=1e-8)
            x=torch.cat([t.batch(1,[0])[0],t.batch(2,[0,1])[0]])
            for _ in range(2):
                optimizer.zero_grad()
                b=start[B_KEY]+donor@c[0]
                logits=.5*F.linear(F.linear(x,start[A_KEY]),b)
                data=(logits-torch.tensor([[1.,0.]]).expand_as(logits)).square().mean()
                (data+t.config['regularization']*c.square().sum()).backward()
                optimizer.step()
            torch.testing.assert_close(t.last_matrices[B_KEY],c.detach(),atol=1e-7,rtol=1e-6)

    def test_no_positive_donor_is_exact_noop_with_zero_optimizer_steps(self):
        with tempfile.TemporaryDirectory() as tmp:
            t=configured(tmp,b_directed_min_gain=1e3)
            start=copy.deepcopy(t.model.state_dict())
            result=t.apply_shared(start,start,30)
            self.assertEqual(t.last_mixtures,{})
            self.assertTrue(torch.equal(result[B_KEY],start[B_KEY]))
            self.assertEqual(t.pending['summary']['optimizer_steps'],0)
            self.assertEqual(t.pending['summary']['unsupported_classes'],1)
            with np.load(Path(tmp)/'b_transfer_rounds/r030/matrices.npz') as npz:
                self.assertEqual(npz['source_weights'].shape,(0,0))

    def test_invalid_source_fails_without_changing_base_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            t=configured(tmp)
            start=copy.deepcopy(t.model.state_dict())
            t.original_deltas[0][B_KEY][0,0]=float('nan')
            with self.assertRaises(ValueError): t.apply_shared(start,start,30)
            self.assertTrue(torch.equal(t.model.state_dict()[B_KEY],start[B_KEY]))


class DirectedEntrypointTests(unittest.TestCase):
    def test_launcher_profile_resume_and_pack_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            def prepare(a,run):
                (run/'protocol').mkdir(parents=True)
                return run/'protocol/full_schedule.json',''
            argv=['entry','--output-root',tmp,'--fast-execution-v2','--feedback-batch-size','128']
            with patch('sys.argv',argv),patch.object(launcher,'prepare_protocol',prepare), \
                 patch.object(launcher.subprocess,'run') as execute,patch('builtins.print'):
                launcher.main(True,'shared',default_directed=True)
            command=execute.call_args.args[0]
            self.assertIn('--b_directed_enable',command)
            self.assertNotIn('--b_transfer_tail_weight',command)
            run=Path(command[command.index('--output-dir')+1])
            self.assertIn('_directed_k3_',run.name)
            (run/'checkpoints').mkdir()
            (run/'checkpoints/sfra_last.pt').touch()
            with patch('sys.argv',argv+['--resume']),patch.object(launcher.subprocess,'run') as execute,patch('builtins.print'):
                launcher.main(True,'shared',default_directed=True)
            expected=list(command)
            expected[expected.index('--sfra_resume')+1]=str(run/'checkpoints/sfra_last.pt')
            self.assertEqual(execute.call_args.args[0],expected)
            modified=list(command)
            modified[modified.index('--b_directed_topk')+1]='4'
            (run/'command.json').write_text(json.dumps(modified))
            with patch('sys.argv',argv+['--resume']),patch.object(launcher.subprocess,'run') as execute, \
                 patch('sys.stderr'),self.assertRaises(SystemExit):
                launcher.main(True,'shared',default_directed=True)
            execute.assert_not_called()
        with patch('sys.argv',['entry','--stage','pack']),patch('tools.sfra.summary.summarize') as summary, \
             patch('tools.sfra.summary.pack') as package:
            launcher.main(True,'shared',default_directed=True)
        self.assertEqual(summary.call_args.args[0],Path('output/cifar100_LT/sfra_b_directed'))
        self.assertEqual(package.call_args.args[0],Path('output/cifar100_LT/sfra_b_directed'))

    def test_rejects_mixed_method_arguments_before_creating_runs(self):
        for extra in [['--problem2-variant','E10'],['--transfer-tail-weight','.35'],['--donors-per-class','0'],
                      ['--transfer-mode','local'],['--tail-client-ids','1,1']]:
            with patch('sys.argv',['entry']+extra),patch('sys.stderr'), \
                 patch.object(launcher,'prepare_protocol') as prepare,self.assertRaises(SystemExit):
                launcher.main(True,'shared',default_directed=True)
            prepare.assert_not_called()

    def test_summary_collects_actual_class_source_weights_and_purpose(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'directed'
            run=root/'seed42/client-longtail/full-cp/test'
            run.mkdir(parents=True)
            t=configured(run)
            start=copy.deepcopy(t.model.state_dict())
            with patch('builtins.print'):
                state=t.apply_shared(start,start,30)
                t.observe_global(state,30,committed=True)
                t.flush()
            (run/'sfra_config.json').write_text(json.dumps(dict(variant='full-cp',seed=42,partition='client-longtail',
                retention_weight=10,classification_weight=1,b_transfer=t.config)))
            (run/'completion.json').write_text('{}')
            write_csv(run/'round_metrics.csv',[dict(round=r,**{k:float(r) for k in METRICS}) for r in range(101)])
            for name in ['sfra_rounds','sfra_costs']: write_csv(run/f'{name}.csv',[])
            with patch('builtins.print'): summarize(root);pack(root)
            result=read_csv(root/'analysis/performance.csv')[0]
            self.assertTrue(result['method'].endswith('-tail-directed'))
            self.assertEqual(result['transfer_tail_weight'],'1.0')
            self.assertEqual(result['last20_bottom20_tail_acc'],'90.5')
            self.assertEqual(len(read_csv(root/'analysis/b_directed_sources.csv')),1)
            self.assertIn(PURPOSE,(root/'analysis/report.md').read_text(encoding='utf-8'))
            self.assertTrue((root.parent/'directed_analysis.tar.gz').exists())


if __name__ == '__main__':
    unittest.main()
