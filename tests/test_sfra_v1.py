"""Small algebra/gradient checks only: no CLIP training, downloads or dataset runs."""
import copy
import json
import random
import tempfile
import unittest
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from utils.sfra_math import (FunctionalHistory, proposal_coordinates, source_statistics,
                             make_targets, functional_loss, projected_step, classification_preservation)
from utils.sfra_feedback import WitnessBank
from utils.pfrf import capture_rng_state, restore_rng_state
from utils.cliplora_sfra import SFRARuntime


class TinyImageEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.proj_lora_A = torch.nn.Parameter(torch.randn(2,4)*.1)
        self.proj_lora_B = torch.nn.Parameter(torch.randn(3,2)*.2)
        self.register_buffer('base', torch.randn(3,4))

    def forward(self, images):
        return images.flatten(1) @ (self.base+self.proj_lora_B@self.proj_lora_A).T


def tiny_bank(batch_size=2):
    torch.manual_seed(17)
    model = torch.nn.Module()
    model.image_encoder = TinyImageEncoder()
    model.dtype = torch.float32
    bank = object.__new__(WitnessBank)
    bank.model = bank.core = model
    bank.device = torch.device('cpu')
    bank.keys = ['image_encoder.proj_lora_A']
    bank.parameters = [model.image_encoder.proj_lora_A]
    bank.numel, bank.clients, bank.batch_size = 8, 2, batch_size
    bank.text_features = torch.eye(3)
    bank.datasets = [[{'img':torch.randn(1,2,2)} for _ in range(5)],
                     [{'img':torch.randn(1,2,2)} for _ in range(3)]]
    bank.tokens = [dict(client_id=0,class_id=0,local_positions=list(range(5))),
                   dict(client_id=1,class_id=1,local_positions=list(range(3)))]
    return bank


class TestSFRA(unittest.TestCase):
    def test_qr_feedback_including_rank_deficiency(self):
        torch.manual_seed(2)
        d = torch.randn(30,5)
        d[:,4] = d[:,0]+d[:,1]
        q,c,r = proposal_coordinates(d)
        g = torch.randn(7,2,30)
        torch.testing.assert_close((g@q)@c, g@d, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(q@c, d)
        torch.testing.assert_close(r, d.square().sum(0).mean().sqrt())

    def test_source_two_views_and_positive_mean(self):
        responses = torch.tensor([[[.2,.3,-.1],[.1,-.1,-.2]], [[1e-6,0.,0.],[2e-6,0.,0.]]])
        source = source_statistics(responses, torch.tensor([1/3,.6]), torch.tensor([.8,.1,.1]))
        torch.testing.assert_close(source['u'], torch.tensor([.1,0.]))
        torch.testing.assert_close(source['cache'], torch.tensor([1.,.6]))
        self.assertEqual(source['positive_count'].tolist(), [1,0])
        torch.testing.assert_close(source['predicted'], responses@torch.tensor([.8,.1,.1]))

    def test_targets_cache_history_and_single_global_normalization(self):
        f = torch.tensor([[-.1,-.2],[-.3,-.2],[-.1,-.1]])
        source = dict(supported=torch.tensor([True,False,False]), u=torch.tensor([.2,0.,0.]), cache=torch.tensor([1.,.5,.9]))
        h, valid = torch.tensor([-.05,-.15,0.]), torch.tensor([True,True,False])
        full = make_targets(f, source, h, valid, 'full')
        torch.testing.assert_close(full['target'][:2], torch.tensor([[0.,-.05],[-.15,-.15]]))
        torch.testing.assert_close(full['weights'], torch.tensor([2/3,1/3,0.]))
        self.assertEqual(full['active'].tolist(), [True,True,False])
        current = make_targets(f, source, h, valid, 'current')
        self.assertEqual(current['active'].tolist(), [True,False,False])
        flat = make_targets(f, source, h, valid, 'flat')
        torch.testing.assert_close(flat['target'], full['target'])
        torch.testing.assert_close(flat['weights'], torch.tensor([.5,.5,0.]))

    def test_empty_targets_are_inactive(self):
        source = dict(supported=torch.zeros(2,dtype=torch.bool),u=torch.zeros(2),cache=torch.ones(2)/30)
        target = make_targets(torch.ones(2,2), source, torch.zeros(2),torch.zeros(2,dtype=torch.bool),'full')
        self.assertFalse(target['active'].any())
        self.assertEqual(float(target['weights'].sum()), 0.)

    def test_history_negative_improvement_and_blocks(self):
        history = FunctionalHistory(torch.full((2,2),-.05),30)
        for rnd in range(1,5):
            history.commit(torch.full((2,2),-.01),rnd)
            self.assertFalse(history.valid.any())
        history.commit(torch.full((2,2),-.01),5)
        self.assertTrue(history.valid.all())
        for rnd in range(6,11):
            scores = torch.full((2,2),.1 if rnd != 8 else -.02)
            history.commit(scores,rnd)
        torch.testing.assert_close(history.level, torch.full((2,),-.01))
        self.assertEqual(history.block_count, 0)

    def test_no_improvement_does_not_register_initial_teacher(self):
        h = FunctionalHistory(torch.full((1,2),.1),30)
        for rnd in range(1,6):
            h.commit(torch.full((1,2),.1),rnd)
        self.assertFalse(h.valid.any())

    def test_loss_coefficient_and_proximity_gradient(self):
        f = torch.tensor([[0.,0.],[.5,.5]])
        self.assertAlmostEqual(float(functional_loss(f,torch.ones_like(f),torch.ones_like(f),torch.tensor([.2,.8]))), .2)
        z = torch.tensor([.2,.3])
        result = projected_step(z,torch.tensor(.2),torch.tensor([.3,-.4]),3.)
        torch.testing.assert_close(result,z-.1*(z+3*.2*torch.tensor([.3,-.4])))
        self.assertLessEqual(float(projected_step(z,torch.tensor(1.),torch.ones(2)*100,10).norm()),1.000001)

    def test_zero_strength_keeps_exact_ordinary_candidate(self):
        candidate = torch.randn(9)
        z = torch.zeros(9)
        for _ in range(3):
            z = projected_step(z,torch.tensor(.1),torch.randn(9),0.)
        self.assertTrue(torch.equal(candidate+.1*z,candidate))

    def test_full_gradient_scale_not_projected_scale(self):
        q,_,r = proposal_coordinates(torch.tensor([[1.],[0.]]))
        g = torch.tensor([0.,2.])
        self.assertEqual(float((g@q).norm()),0.)
        self.assertEqual(float((r*g.norm()).clamp_min(.001)),2.)

    def test_microbatch_and_real_autograd_match(self):
        bank = tiny_bank(2)
        images = torch.stack([r['img'] for r in bank.datasets[0]])
        f,g = bank.token_score(images,0,False,True)
        bank.batch_size = 8
        full,full_g = bank.token_score(images,0,False,True)
        torch.testing.assert_close(f,full)
        torch.testing.assert_close(g,full_g,atol=1e-7,rtol=1e-5)

    def test_client_gradient_sum_freezes_b_and_matches_loss(self):
        bank = tiny_bank(2)
        before = copy.deepcopy(bank.model.state_dict())
        target = dict(target=torch.full((2,2),1.5),sigma=torch.ones(2,2)*.2,
                      weights=torch.tensor([.2,.8]),active=torch.ones(2,dtype=torch.bool))
        observed = bank.evaluate(targets=target)
        values = []
        for token in bank.tokens:
            images = torch.stack([bank.datasets[token['client_id']][i]['img'] for i in token['local_positions']])
            vv = []
            for flip in (False,True):
                f = bank.core.image_encoder(images.flip(-1) if flip else images)
                scores = (f/f.norm(dim=1,keepdim=True))@bank.text_features.T
                wrong = scores.clone()
                wrong[:,token['class_id']] = -torch.inf
                vv.append((scores[:,token['class_id']]-wrong.max(1).values).mean())
            values.append(torch.stack(vv))
        loss = functional_loss(torch.stack(values),target['target'],target['sigma'],target['weights'])
        expected, = torch.autograd.grad(loss,bank.parameters)
        torch.testing.assert_close(observed['gradient'],expected.flatten(),atol=1e-6,rtol=1e-5)
        self.assertAlmostEqual(observed['loss'],float(loss.detach()),places=5)
        self.assertTrue(all(torch.equal(before[k],v) for k,v in bank.model.state_dict().items()))
        self.assertIsNone(bank.core.image_encoder.proj_lora_B.grad)

    def test_history_rng_checkpoint_roundtrip(self):
        history = FunctionalHistory(torch.full((2,2),-.05),30)
        history.commit(torch.full((2,2),-.02),1)
        history.source_cache[:] = torch.tensor([.2,.7])
        rng = capture_rng_state()
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/'checkpoint.pt'
            torch.save(dict(history=history.state_dict(),rng=rng),path)
            expected = (random.random(),np.random.rand(),torch.rand(2))
            restored = torch.load(path,weights_only=False)
        clone = FunctionalHistory(torch.zeros(2,2),30)
        clone.load_state_dict(restored['history'])
        restore_rng_state(restored['rng'])
        actual = (random.random(),np.random.rand(),torch.rand(2))
        self.assertEqual(actual[:2],expected[:2])
        torch.testing.assert_close(actual[2],expected[2])
        self.assertEqual(clone.block_count,1)
        torch.testing.assert_close(clone.source_cache,history.source_cache)
        torch.testing.assert_close(clone.block_min,history.block_min)

    def test_evaluation_does_not_consume_training_rng(self):
        bank = tiny_bank()
        rng = capture_rng_state()
        bank.evaluate(basis=torch.eye(8)[:,:2])
        actual = torch.rand(4)
        restore_rng_state(rng)
        torch.testing.assert_close(actual,torch.rand(4))

    def test_full_refresh_three_steps_and_b_unchanged(self):
        for strength in (0.,3.):
            bank = tiny_bank(2)
            runtime = object.__new__(SFRARuntime)
            runtime.trainer = SimpleNamespace(model=bank.model,device=bank.device)
            runtime.bank, runtime.a_keys = bank, bank.keys
            runtime.b_keys = ['image_encoder.proj_lora_B']
            runtime.keys = sorted(runtime.a_keys+runtime.b_keys)
            runtime.variant, runtime.strength, runtime.scaling = 'full',strength,.5
            runtime.costs, runtime.events, runtime.q = [],[{}],{0:.8,1:.2}
            middle = copy.deepcopy(bank.model.state_dict())
            runtime.history = FunctionalHistory(bank.evaluate()['scores'],2)
            runtime.history.valid[:] = True
            runtime.history.level[:] = 1.5
            da = torch.randn_like(middle[bank.keys[0]])*.01
            db = torch.randn_like(da)*.01
            ordinary = copy.deepcopy(middle)
            ordinary[bank.keys[0]] += .8*da+.2*db
            uploads = {0:{bank.keys[0]:da},1:{bank.keys[0]:db}}
            runtime.train_phase = lambda *a,**kw:(ordinary,uploads)
            with tempfile.TemporaryDirectory() as temp:
                committed,scores,arrays,summary = runtime.refresh(middle,1,Path(temp))
            self.assertEqual(summary['correction_steps'],3)
            self.assertTrue(torch.equal(committed[runtime.b_keys[0]],middle[runtime.b_keys[0]]))
            self.assertLessEqual(summary['correction_a_norm'],summary['radius']+1e-7)
            self.assertEqual(scores.shape,(2,2))
            if strength == 0:
                self.assertTrue(torch.equal(committed[bank.keys[0]],ordinary[bank.keys[0]]))
            else:
                self.assertGreater(summary['correction_a_norm'],0.)

    def test_launcher_flags_and_distinct_search_paths(self):
        from scripts import run_cliplora_sfra as launcher
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)
            def prepare(args,run):
                (run/'protocol').mkdir(parents=True)
                return run/'protocol/full_schedule.json',''
            for strength in (1,3,10):
                argv = ['run_cliplora_sfra.py','--method','full','--retention-weight',str(strength),
                        '--output-root',str(output)]
                with patch('sys.argv',argv),patch.object(launcher,'prepare_protocol',prepare),patch.object(launcher.subprocess,'run') as execute,patch('builtins.print'):
                    launcher.main()
                    command = execute.call_args.args[0]
                    self.assertEqual(command[command.index('--lac_method')+1],'s')
                    self.assertEqual(command[command.index('--sfra_variant')+1],'full')
                    self.assertEqual(float(command[command.index('--sfra_retention_weight')+1]),strength)
                    self.assertTrue(command.index('--sfra_variant') < command.index('DATALOADER.NUM_WORKERS'))
            self.assertEqual(len(list(output.glob('seed42/client-longtail/full/*/command.json'))),3)

    def test_runtime_checkpoint_restores_history_and_trims_interrupted_test_row(self):
        from utils.cliplora_bridge_audit import write_csv
        bank = tiny_bank()
        runtime = object.__new__(SFRARuntime)
        runtime.trainer = SimpleNamespace(model=bank.model,device=bank.device)
        runtime.bank, runtime.a_keys = bank, bank.keys
        runtime.b_keys = ['image_encoder.proj_lora_B']
        runtime.keys = sorted(runtime.a_keys+runtime.b_keys)
        runtime.base_state = copy.deepcopy(bank.model.state_dict())
        state = copy.deepcopy(runtime.base_state)
        state[runtime.a_keys[0]] += .01
        runtime.history = FunctionalHistory(bank.evaluate()['scores'],2)
        runtime.history.commit(torch.zeros(2,2),1)
        runtime.history.source_cache[:] = .75
        runtime.events, runtime.budget, runtime.evaluations = [],[],[]
        runtime.costs, runtime.summaries = [],[]
        runtime.started, runtime.elapsed_before = time.perf_counter(),0.
        runtime.variant, runtime.strength, runtime.sfra_config = 'full',3.,{'test':'round_boundary'}
        runtime.sizes = [5,3]
        with tempfile.TemporaryDirectory() as temp:
            runtime.root = Path(temp)
            (runtime.root/'checkpoints').mkdir()
            write_csv(runtime.root/'round_metrics.csv',[{'round':r,'overall_acc':0} for r in range(3)])
            runtime.checkpoint(state,1)
            expected = torch.rand(3)
            runtime.resume_payload = torch.load(runtime.root/'checkpoints/sfra_last.pt',weights_only=False)
            restored,completed = runtime.restore()
            self.assertEqual(completed,1)
            torch.testing.assert_close(torch.rand(3),expected)
            self.assertTrue(torch.equal(restored[runtime.a_keys[0]],state[runtime.a_keys[0]]))
            torch.testing.assert_close(runtime.history.source_cache,torch.full((2,),.75))
            self.assertEqual(runtime.history.block_count,1)
            from tools.sfra.summary import read_csv
            self.assertEqual([int(r['round']) for r in read_csv(runtime.root/'round_metrics.csv')],[0,1])

    def test_lightweight_summary_and_archive(self):
        from tools.sfra.summary import summarize, pack, write_csv, read_csv, METRICS
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)/'sfra_v1'
            run = root/'seed42/client-longtail/full/lambda10_protocol42'
            run.mkdir(parents=True)
            (run/'sfra_config.json').write_text(json.dumps(dict(variant='full',partition='client-longtail',seed=42,retention_weight=10)))
            (run/'completion.json').write_text('{}')
            write_csv(run/'round_metrics.csv',[{'round':r,**{key:float(r) for key in METRICS}} for r in range(101)])
            write_csv(run/'sfra_rounds.csv',[])
            write_csv(run/'sfra_costs.csv',[])
            (run/'weights.pt').write_bytes(b'not a real model')
            summarize(root)
            summary = read_csv(root/'analysis/performance.csv')[0]
            self.assertEqual(float(summary['last20_overall_acc']),90.5)
            pack(root)
            import tarfile
            with tarfile.open(root.parent/'sfra_v1_analysis.tar.gz') as archive:
                self.assertFalse(any(n.endswith('.pt') for n in archive.getnames()))
                self.assertTrue(any(n.endswith('report.md') for n in archive.getnames()))


class TestSFRAClassification(unittest.TestCase):
    def bank(self, batch_size=2):
        bank = tiny_bank(batch_size)
        bank.core.logit_scale = torch.nn.Parameter(torch.tensor(3.).log(), requires_grad=False)
        # Unequal original class counts, deliberately unlike the witness sample counts.
        bank.tokens = [dict(client_id=0,class_id=0,local_positions=[0,1]),
                       dict(client_id=0,class_id=2,local_positions=[2,3,4]),
                       dict(client_id=1,class_id=1,local_positions=[0,1,2])]
        bank.configure_classification([[18,0,2],[0,10,0]], torch.tensor([18.,10.,2.]).div(30).log())
        return bank

    def explicit_classification(self, bank):
        total = torch.zeros(())
        for token, weight in zip(bank.tokens, [.6, 2/30, 1/3]):
            images = torch.stack([bank.datasets[token['client_id']][i]['img'] for i in token['local_positions']])
            for flip in (False, True):
                features = bank.core.image_encoder(images.flip(-1) if flip else images)
                features = features / features.norm(dim=-1, keepdim=True)
                logits = (bank.core.logit_scale.exp()*features) @ bank.text_features.T
                logits = logits + bank.classification_logit_adjustment
                labels = torch.full((len(images),), token['class_id'], dtype=torch.long)
                total = total + .5*weight*torch.nn.functional.cross_entropy(logits, labels)
        return total

    def test_one_sided_penalty_and_exact_derivative(self):
        for value in (.8, 1., 1.3):
            actual, coefficient = classification_preservation(value, 1., .2)
            loss = torch.tensor(value, dtype=torch.float64, requires_grad=True)
            penalty = .5*((loss-1.)/.2).clamp_min(0).square()
            gradient, = torch.autograd.grad(penalty, loss)
            self.assertAlmostEqual(actual, float(penalty.detach()))
            self.assertAlmostEqual(coefficient, float(gradient))

    def test_joint_projected_step_and_first_step_identity(self):
        z, radius = torch.tensor([.1,-.2]), torch.tensor(.03)
        functional, auxiliary = torch.tensor([.2,.4]), torch.tensor([-.3,.1])
        result = projected_step(z,radius,functional,10.,auxiliary_gradient=auxiliary)
        torch.testing.assert_close(result,z-.1*(z+10*radius*functional+radius*auxiliary))
        original = projected_step(z,radius,functional,10.)
        first_cp = projected_step(z,radius,functional,10.,auxiliary_gradient=torch.zeros_like(z))
        self.assertTrue(torch.equal(original,first_cp))

    def test_classification_uses_all_tokens_original_weights_and_clip_logits(self):
        bank = self.bank()
        torch.testing.assert_close(bank.classification_client_weights,torch.tensor([2/3,1/3]))
        torch.testing.assert_close(bank.classification_token_weights,torch.tensor([.9,.1,1.]))
        target = dict(target=torch.ones(3,2), sigma=torch.ones(3,2),
                      weights=torch.tensor([1.,0.,0.]), active=torch.tensor([True,False,False]))
        observed = bank.evaluate(targets=target,all_scores=False,classification=True)
        expected = self.explicit_classification(bank)
        gradient, = torch.autograd.grad(expected,bank.parameters)
        self.assertAlmostEqual(observed['classification_loss'],float(expected.detach()),places=6)
        torch.testing.assert_close(observed['classification_gradient'],gradient.flatten(),atol=2e-7,rtol=1e-5)
        self.assertTrue((observed['classification_scores'] > 0).all())
        self.assertEqual(observed['forward_images'],16)
        self.assertEqual(observed['classification_forward_images'],16)
        self.assertEqual(observed['classification_backward_images'],16)
        self.assertEqual(observed['backward_images'],20)  # 4 functional + 16 classification.
        self.assertIsNone(bank.core.image_encoder.proj_lora_B.grad)

    def test_classification_microbatches_rng_and_no_gradient_final_evaluation(self):
        bank = self.bank(1)
        before = copy.deepcopy(bank.model.state_dict())
        rng = capture_rng_state()
        small = bank.evaluate(classification=True)
        actual = torch.rand(3)
        restore_rng_state(rng)
        torch.testing.assert_close(actual,torch.rand(3))
        bank.batch_size = 8
        large = bank.evaluate(classification=True)
        final = bank.evaluate(classification=True,classification_gradient=False)
        self.assertAlmostEqual(small['classification_loss'],large['classification_loss'],places=6)
        torch.testing.assert_close(small['classification_gradient'],large['classification_gradient'],atol=2e-7,rtol=1e-5)
        self.assertIsNone(final['classification_gradient'])
        self.assertEqual(final['backward_images'],0)
        self.assertEqual(final['classification_loss'],large['classification_loss'])
        self.assertTrue(all(torch.equal(before[k],v) for k,v in bank.model.state_dict().items()))

    def test_cp_refresh_reference_scale_logs_and_frozen_b(self):
        from tools.sfra.summary import read_csv
        bank = self.bank()
        runtime = object.__new__(SFRARuntime)
        runtime.trainer = SimpleNamespace(model=bank.model,device=bank.device)
        runtime.bank, runtime.a_keys = bank, bank.keys
        runtime.b_keys = ['image_encoder.proj_lora_B']
        runtime.keys = sorted(runtime.a_keys+runtime.b_keys)
        runtime.variant, runtime.strength, runtime.scaling = 'full-cp',10.,.5
        runtime.classification_strength = .3
        runtime.costs, runtime.events, runtime.q = [],[{}],{0:2/3,1:1/3}
        middle = copy.deepcopy(bank.model.state_dict())
        runtime.history = FunctionalHistory(bank.evaluate()['scores'],2)
        runtime.history.valid[:] = True
        runtime.history.level[:] = 1.5
        da = torch.randn_like(middle[bank.keys[0]])*.01
        db = torch.randn_like(da)*.01
        ordinary = copy.deepcopy(middle)
        ordinary[bank.keys[0]] += (2/3)*da+(1/3)*db
        runtime.train_phase = lambda *a,**kw:(ordinary,{0:{bank.keys[0]:da},1:{bank.keys[0]:db}})
        bank.model.load_state_dict(ordinary)
        reference = bank.evaluate(classification=True)
        with tempfile.TemporaryDirectory() as temp:
            committed,_,arrays,summary = runtime.refresh(middle,1,Path(temp))
            steps = read_csv(Path(temp)/'correction_steps.csv')
        self.assertEqual(len(steps),3)
        self.assertEqual(float(steps[0]['classification_penalty']),0.)
        self.assertEqual(steps[0]['functional_classification_gradient_cosine'],'')
        self.assertTrue(all(float(s['classification_reference_loss']) == reference['classification_loss'] for s in steps))
        expected_scale = max(.001,summary['radius']*float(reference['classification_gradient'].norm()))
        self.assertAlmostEqual(summary['classification_scale'],expected_scale)
        self.assertTrue(torch.equal(committed[runtime.b_keys[0]],middle[runtime.b_keys[0]]))
        torch.testing.assert_close(arrays['classification_global_token_weights'],torch.tensor([.6,2/30,1/3]))
        final = bank.evaluate(classification=True,classification_gradient=False)
        self.assertAlmostEqual(summary['committed_classification_loss'],final['classification_loss'])
        penalty,_ = classification_preservation(final['classification_loss'],reference['classification_loss'],expected_scale)
        self.assertAlmostEqual(summary['committed_classification_penalty'],penalty)

    def test_cp_launcher_paths_flags_and_resume_original_configuration(self):
        from scripts import run_cliplora_sfra as launcher
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)
            def prepare(args,run):
                (run/'protocol').mkdir(parents=True)
                return run/'protocol/full_schedule.json',''
            for mu in (.3,1.,3.):
                argv = ['run_cliplora_sfra.py','--method','full-cp','--classification-weight',str(mu),
                        '--output-root',str(output)]
                with patch('sys.argv',argv),patch.object(launcher,'prepare_protocol',prepare),patch.object(launcher.subprocess,'run') as execute,patch('builtins.print'):
                    launcher.main()
                    command = execute.call_args.args[0]
                self.assertEqual(command[command.index('--sfra_variant')+1],'full-cp')
                self.assertEqual(float(command[command.index('--sfra_classification_weight')+1]),mu)
                self.assertEqual(float(command[command.index('--sfra_retention_weight')+1]),10.)
                run = Path(command[command.index('--output-dir')+1])
                self.assertIn(f'_mu{mu:g}_',run.name)
                (run/'checkpoints').mkdir()
                (run/'checkpoints/sfra_last.pt').touch()
                with patch('sys.argv',argv+['--resume']),patch.object(launcher.subprocess,'run') as execute,patch('builtins.print'):
                    launcher.main()
                    resumed = execute.call_args.args[0]
                self.assertEqual(resumed[resumed.index('--sfra_resume')+1],str(run/'checkpoints/sfra_last.pt'))
                self.assertEqual(float(resumed[resumed.index('--sfra_classification_weight')+1]),mu)
            self.assertEqual(len(list(output.glob('seed42/client-longtail/full-cp/*/command.json'))),3)


if __name__ == '__main__':
    unittest.main()
