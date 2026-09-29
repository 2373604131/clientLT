"""Source-to-supervision causality, joint donor C, isolation and launch/pack."""
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

from test_sfra_b_directed import args, configured
from test_sfra_b_shared_transfer import A_KEY, B_KEY
from scripts import run_cliplora_sfra as launcher
from tools.sfra.summary import METRICS, pack, read_csv, summarize, write_csv
from utils.b_directed_math import class_macro_loss
from utils.b_response_math import (build_response_targets, decision_margins, donor_residual,
    image_view, pairwise_margins, response_config, response_losses)
from utils.cliplora_b_response import ResponseDonorBTransfer
from utils.cliplora_b_shared_transfer import shared_transfer_config


def response_args(**updates):
    return args(b_directed_enable=True, **updates)


def response_transfer(root, **updates):
    t = configured(root)
    t.__class__ = ResponseDonorBTransfer
    t.config = response_config(shared_transfer_config(response_args()), response_args(**updates))
    original_batch, original_forward = t.batch, t.forward
    def batch(client, positions):
        x, y = original_batch(client, positions)
        return x.reshape(len(y), 1, 1, 2), y
    t.batch = batch
    t.forward = lambda x, y, diagnostic=False: original_forward(x.flatten(1), y, diagnostic)
    t.original_deltas[0][B_KEY] = torch.tensor([[1., 1.], [-1., -1.]])
    return t


class ResponseMathTests(unittest.TestCase):
    def test_mix_before_clipping_min_view_and_zero_control(self):
        baseline = torch.zeros(2, 2, 3, requires_grad=True)
        labels = torch.tensor([0, 1])
        d0 = torch.tensor([[[2., 0., 0.], [0., 1., 0.]], [[4., 0., 0.], [0., 1., 0.]]], requires_grad=True)
        d1 = torch.tensor([[[-1., 0., 0.], [0., 2., 0.]], [[-1., 0., 0.], [0., 2., 0.]]], requires_grad=True)
        mixtures = {0: {0:.5, 1:.5}}
        result = build_response_targets(baseline, {0:d0, 1:d1}, labels, mixtures)
        torch.testing.assert_close(result['positive_increment'], torch.tensor([[0., .5, .5], [0., 0., 0.]]))
        torch.testing.assert_close(result['competitor_weights'].sum(-1), torch.ones(2, 2))
        self.assertTrue(all(not v.requires_grad for v in result.values()))
        zero = build_response_targets(baseline, {0:d0, 1:d1}, labels, mixtures, 'zero')
        torch.testing.assert_close(zero['positive_increment'], result['positive_increment'])
        self.assertEqual(int(zero['applied_increment'].count_nonzero()), 0)
        self.assertEqual(int(zero['target_margins'].count_nonzero()), 0)
        incompatible = d0.detach().clone()
        incompatible[1, 0, 0] = -2.
        gated = build_response_targets(baseline, {0:incompatible}, labels, {0:{0:1.}})
        self.assertEqual(int(gated['positive_increment'].count_nonzero()), 0)

    def test_same_union_different_class_edges_changes_shared_C_gradient(self):
        labels = torch.tensor([0, 1])
        baseline = torch.zeros(2, 2, 3)
        donors = {0:torch.tensor([[[2., 0., 1.], [0., 1., 0.]]]).repeat(2, 1, 1),
                  1:torch.tensor([[[1., 0., 0.], [0., 3., 2.]]]).repeat(2, 1, 1)}
        mixtures = [{0:{0:1.}, 1:{1:1.}}, {0:{1:1.}, 1:{0:1.}}]
        basis = {B_KEY:torch.tensor([[[1.,0.],[0.,1.],[1.,1.]], [[0.,1.],[1.,0.],[1.,-1.]]])}
        gradients = []
        for source in mixtures:
            target = build_response_targets(baseline, donors, labels, source)
            c = torch.zeros(2, 2, 2, requires_grad=True)
            logits = donor_residual(basis, {B_KEY:c})[B_KEY].T
            loss = response_losses(logits, labels, target['target_margins'][0], target['competitor_weights'][0]).mean()
            gradients.append(torch.autograd.grad(loss, c)[0])
        self.assertFalse(torch.allclose(gradients[0], gradients[1]))

    def test_one_sided_loss_complete_competitors_and_frozen_targets(self):
        labels = torch.tensor([0])
        logits = torch.tensor([[2., 0., 1.]], requires_grad=True)
        target = torch.tensor([[0., 1., .5]], requires_grad=True)
        weights = torch.tensor([[0., .2, .8]], requires_grad=True)
        loss = response_losses(logits, labels, target, weights).sum()
        self.assertEqual(float(loss.detach()), 0.)
        loss.backward()
        self.assertIsNone(target.grad)
        self.assertIsNone(weights.grad)
        torch.testing.assert_close(decision_margins(logits, labels), torch.tensor([1.]))
        with self.assertRaises(ValueError): pairwise_margins(torch.ones(1, 1), labels)
        with self.assertRaises(ValueError): image_view(torch.ones(1, 3), 1)

    def test_profile_independent_C_and_invalid_options(self):
        config = response_config(shared_transfer_config(response_args()), response_args())
        self.assertEqual(config['calibration_profile'], 'tail_response')
        self.assertEqual(config['probe_views'], 2)
        self.assertEqual(config['tail_weight'], 1.)
        self.assertIn('per selected donor', config['c_scope'])
        for update in [dict(b_response_weight=float('nan')), dict(b_response_weight=0),
                       dict(b_response_variant='unknown'), dict(b_transfer_tail_weight=.35),
                       dict(b_problem2_variant='E10')]:
            with self.assertRaises(ValueError):
                response_config(shared_transfer_config(response_args()), response_args(**update))


class ResponseRuntimeTests(unittest.TestCase):
    def test_constructor_uses_only_protocol_tail_pool(self):
        with tempfile.TemporaryDirectory() as tmp:
            toy=response_transfer(tmp)
            a=response_args(b_directed_tail_clients='2',sfra_feedback_batch_size=2,seed=42)
            source=[[SimpleNamespace(label=c) for c,positions in g.items() for _ in positions] for g in toy.groups]
            runtime=SimpleNamespace(root=Path(tmp),args=a,b_keys=[B_KEY],tail=[0],
                sfra_config={'b_transfer':response_config(shared_transfer_config(a),a)},
                bank=SimpleNamespace(model=toy.model,core=toy.model,device=torch.device('cpu'),batch_size=2,
                    tokens=[dict(client_id=1,class_id=0,local_positions=[0]),dict(client_id=2,class_id=0,local_positions=[0])]),
                audit=SimpleNamespace(counts=toy.counts),trainer=SimpleNamespace(
                    training_logit_adjustment=torch.zeros(2),dm=SimpleNamespace(dataset=SimpleNamespace(federated_train_x=source))))
            t=ResponseDonorBTransfer(runtime)
            self.assertEqual(t.recipients,[2])
            self.assertEqual(t.class_totals,{0:2})
            self.assertEqual(t.monitors[2],dict(tail_positions=[0,1],non_tail_positions=[]))
            self.assertEqual(json.loads((Path(tmp)/'b_transfer_config.json').read_text(encoding='utf-8'))['calibration_profile'],'tail_response')

    def test_full_event_freezes_base_preserves_rng_and_reconstructs_once(self):
        with tempfile.TemporaryDirectory() as tmp:
            t = response_transfer(tmp)
            start = copy.deepcopy(t.model.state_dict())
            ordinary = {A_KEY:start[A_KEY], B_KEY:start[B_KEY]+.01}
            deltas = copy.deepcopy(t.original_deltas)
            rng = torch.random.get_rng_state().clone()
            with patch('builtins.print'): committed = t.apply_shared(start, ordinary, 30)
            self.assertEqual(t.last_mixtures, {0:{0:1.}})
            self.assertEqual(t.last_matrices[B_KEY].shape, (1, 2, 2))
            expected = deltas[0][B_KEY] @ t.last_matrices[B_KEY][0]
            torch.testing.assert_close(committed[B_KEY], ordinary[B_KEY]+expected)
            self.assertTrue(torch.equal(committed[A_KEY], start[A_KEY]))
            self.assertFalse(torch.allclose(committed[B_KEY], ordinary[B_KEY]))
            for key in start:
                self.assertTrue(torch.equal(t.model.state_dict()[key], start[key]))
                self.assertIsNone(t.parameters[key].grad)
            self.assertTrue(torch.equal(rng, torch.random.get_rng_state()))
            self.assertEqual(t.pending['summary']['algorithm_backward_images'], 12)
            self.assertEqual(t.pending['summary']['optimizer_steps'], 2)
            self.assertTrue(all(r['non_tail_samples']==0 for r in t.pending['feedback']))
            self.assertEqual([r['c_step'] for r in t.pending['loss_trace']], [0,1,2])
            with np.load(Path(tmp)/'b_transfer_rounds/r030/response_targets.npz') as archive:
                self.assertEqual(archive['client_001_baseline_logits'].shape, (2,1,2))
                self.assertIn('client_002_donor_000_logits', archive)
            with patch('builtins.print'): t.observe_global(committed,30,committed=True)
            restored = response_transfer(tmp)
            restored.load_state_dict(t.state_dict())
            self.assertEqual(restored.progress(), t.progress())

    def test_chunking_matches_single_global_Adam_reference(self):
        with tempfile.TemporaryDirectory() as tmp:
            results = []
            for size in [1, 32]:
                t = response_transfer(tmp)
                t.target_clients, t.recipients, t.class_totals = [2], [2], {0:2}
                t.probes = {2:{0:[0,1]}}
                t.counts[1,0] = 0
                t.original_deltas[1][B_KEY] = torch.tensor([[1.,.5],[-.4,-1.2]])
                t.feedback_batch_size = size
                start = copy.deepcopy(t.model.state_dict())
                delta = torch.stack([t.original_deltas[j][B_KEY].clone() for j in (0,1)])
                with patch('builtins.print'): results.append(t.apply_shared(start,start,30)[B_KEY])
                self.assertEqual(t.last_donors,[0,1])
            torch.testing.assert_close(results[0], results[1], atol=1e-7, rtol=1e-6)
            c = torch.nn.Parameter(torch.zeros(2,2,2))
            optimizer = torch.optim.Adam([c], lr=t.config['learning_rate'], betas=(.9,.999), eps=1e-8)
            for _ in range(2):
                optimizer.zero_grad()
                objective = 0.
                for client in t.recipients:
                    x, labels = t.batch(client, t.monitors[client]['tail_positions'])
                    for view in (0,1):
                        logits = .5*F.linear(F.linear(image_view(x,view).flatten(1),start[A_KEY]), start[B_KEY]+torch.bmm(delta,c).mean(0))
                        la = (logits-F.one_hot(labels,2)).square().mean(1)
                        resp = response_losses(logits,labels,t.last_targets[client]['target_margins'][view],
                                               t.last_targets[client]['competitor_weights'][view])
                        objective = objective+class_macro_loss(la+t.config['response_weight']*resp,labels,t.class_totals)/2
                objective = objective+t.config['regularization']*c.square().sum()/2
                objective.backward()
                optimizer.step()
            torch.testing.assert_close(t.last_matrices[B_KEY],c.detach(),atol=1e-7,rtol=1e-6)

    def test_zero_control_retains_teacher_screen_budget_and_baseline_reference(self):
        with tempfile.TemporaryDirectory() as tmp:
            variants = []
            for variant in ('positive','zero'):
                t = response_transfer(tmp,b_response_variant=variant)
                state = copy.deepcopy(t.model.state_dict())
                with patch('builtins.print'): t.apply_shared(state,state,30)
                variants.append(t)
            positive, zero = variants
            self.assertEqual(positive.last_mixtures,zero.last_mixtures)
            for field in ('algorithm_backward_images','algorithm_forward_images','optimizer_steps','learned_c_parameters'):
                self.assertEqual(positive.pending['summary'][field],zero.pending['summary'][field])
            for client in zero.recipients:
                torch.testing.assert_close(positive.last_targets[client]['positive_increment'],zero.last_targets[client]['positive_increment'])
                self.assertEqual(int(zero.last_targets[client]['applied_increment'].count_nonzero()),0)
                torch.testing.assert_close(zero.last_targets[client]['target_margins'],zero.last_targets[client]['baseline_margins'])

    def test_no_source_noop_and_invalid_teacher_restores_base(self):
        with tempfile.TemporaryDirectory() as tmp:
            t = response_transfer(tmp,b_directed_min_gain=1e3)
            state = copy.deepcopy(t.model.state_dict())
            result = t.apply_shared(state,state,30)
            self.assertTrue(torch.equal(result[B_KEY],state[B_KEY]))
            self.assertEqual(t.pending['summary']['optimizer_steps'],0)
            self.assertEqual(t.last_mixtures,{})
            with np.load(Path(tmp)/'b_transfer_rounds/r030/matrices.npz') as npz:
                self.assertEqual(npz['source_weights'].shape,(1,0))
            t = response_transfer(tmp)
            t.original_deltas[0][B_KEY][0,0] = float('nan')
            with self.assertRaises(ValueError): t.apply_shared(state,state,30)
            self.assertTrue(torch.equal(t.model.state_dict()[B_KEY],state[B_KEY]))

    def test_real_normalized_logit_LA_forward_rank_four(self):
        with tempfile.TemporaryDirectory() as tmp:
            t = response_transfer(tmp)
            layer = t.model.layer
            layer.w_lora_A = torch.nn.Parameter(torch.cat([layer.w_lora_A.detach(),torch.eye(2)]),requires_grad=False)
            layer.w_lora_B = torch.nn.Parameter(torch.cat([layer.w_lora_B.detach(),.02*torch.eye(2)],dim=1))
            t.parameters = dict(t.model.named_parameters())
            t.ranks = {B_KEY:4}
            t.bank.text_features = torch.eye(2)
            t.core = SimpleNamespace(image_encoder=lambda x:layer(x.flatten(1)),dtype=torch.float32,logit_scale=torch.tensor(1.))
            t.adjustment = torch.tensor([-.3,.1])
            del t.forward
            t.original_deltas = {j:{B_KEY:torch.zeros_like(layer.w_lora_B)} for j in t.selected}
            t.original_deltas[0][B_KEY] = torch.tensor([[1.,1.,1.,1.],[-1.,-1.,-1.,-1.]])
            state = copy.deepcopy(t.model.state_dict())
            with patch('builtins.print'): result = t.apply_shared(state,state,30)
            self.assertTrue(torch.isfinite(result[B_KEY]).all())
            self.assertEqual(t.pending['summary']['optimizer_steps'],2)
            self.assertEqual(t.last_matrices[B_KEY].shape,(1,4,4))


class ResponseEntryTests(unittest.TestCase):
    def run_main(self):
        launcher.main(True,'shared',default_directed=True,default_response=True)

    def test_profile_command_separate_output_resume_and_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            def prepare(a,run):
                (run/'protocol').mkdir(parents=True)
                return run/'protocol/full_schedule.json',''
            argv=['entry','--output-root',tmp,'--response-variant','zero','--fast-execution-v2','--feedback-batch-size','128']
            with patch('sys.argv',argv),patch.object(launcher,'prepare_protocol',prepare),patch.object(launcher.subprocess,'run') as execute,patch('builtins.print'):
                self.run_main()
            command=execute.call_args.args[0]
            self.assertIn('--b_response_enable',command)
            self.assertEqual(command[command.index('--b_response_variant')+1],'zero')
            run=Path(command[command.index('--output-dir')+1])
            self.assertIn('_responseC_zero_rw1',run.name)
            (run/'checkpoints').mkdir()
            (run/'checkpoints/sfra_last.pt').touch()
            with patch('sys.argv',argv+['--resume']),patch.object(launcher.subprocess,'run') as execute,patch('builtins.print'):
                self.run_main()
            expected=list(command)
            expected[expected.index('--sfra_resume')+1]=str(run/'checkpoints/sfra_last.pt')
            self.assertEqual(execute.call_args.args[0],expected)
            command[command.index('--b_response_weight')+1]='2.0'
            (run/'command.json').write_text(json.dumps(command))
            with patch('sys.argv',argv+['--resume']),patch.object(launcher.subprocess,'run') as execute,patch('sys.stderr'),self.assertRaises(SystemExit):
                self.run_main()
            execute.assert_not_called()

    def test_invalid_mix_and_pack_default_root(self):
        for extra in [['--transfer-tail-weight','.35'],['--response-weight','nan'],['--response-weight','0'],['--problem2-variant','E10']]:
            with patch('sys.argv',['entry']+extra),patch.object(launcher,'prepare_protocol') as prepare,patch('sys.stderr'),self.assertRaises(SystemExit):
                self.run_main()
            prepare.assert_not_called()
        with patch('sys.argv',['entry','--stage','pack']),patch('tools.sfra.summary.summarize') as summary,patch('tools.sfra.summary.pack') as package:
            self.run_main()
        self.assertEqual(summary.call_args.args[0],Path('output/cifar100_LT/sfra_b_response'))
        self.assertEqual(package.call_args.args[0],Path('output/cifar100_LT/sfra_b_response'))

    def test_summary_and_pack_keep_response_evidence_and_separate_method(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'response'
            run=root/'seed42/client-longtail/full-cp/test'
            run.mkdir(parents=True)
            t=response_transfer(run)
            state=copy.deepcopy(t.model.state_dict())
            with patch('builtins.print'):
                result=t.apply_shared(state,state,30)
                t.observe_global(result,30,committed=True)
                t.flush()
            (run/'sfra_config.json').write_text(json.dumps(dict(variant='full-cp',seed=42,partition='client-longtail',retention_weight=10,classification_weight=1,b_transfer=t.config)))
            (run/'completion.json').write_text('{}')
            write_csv(run/'round_metrics.csv',[dict(round=r,**{k:float(r) for k in METRICS}) for r in range(101)])
            for name in ('sfra_rounds','sfra_costs'): write_csv(run/f'{name}.csv',[])
            with patch('builtins.print'): summarize(root);pack(root)
            row=read_csv(root/'analysis/performance.csv')[0]
            self.assertTrue(row['method'].endswith('-tail-response-positive'))
            self.assertEqual(row['last20_bottom20_tail_acc'],'90.5')
            self.assertEqual(len(read_csv(root/'analysis/b_response_source_weights.csv')),1)
            report=(root/'analysis/report.md').read_text(encoding='utf-8')
            self.assertIn('Independent C per donor/module',report)
            self.assertNotIn('C indexes supported classes',report)
            import tarfile
            with tarfile.open(root.parent/'response_analysis.tar.gz') as archive:
                names=archive.getnames()
                self.assertTrue(any(n.endswith('response_targets.npz') for n in names))
                self.assertFalse(any(n.endswith('.pt') for n in names))


if __name__ == '__main__':
    unittest.main()
