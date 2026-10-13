"""New-A math, real tiny-model feedback/runtime, and independent-entry regressions."""
import copy
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torch.nn import functional as F

from scripts import run_cliplora_method_a_margin as launcher
from scripts import run_cliplora_joint_aggregation as single
from tests.test_sfra_execution import visual_bank
from tools.client_aggregation.method_a_margin import audit_rounds
from tools.client_aggregation.protocol import load_json, write_table, method_a_margin_contract
from utils.cliplora_a_refresh import train_only
from utils.cliplora_functional_feedback import snapshot
from utils.cliplora_method_a_margin import MethodAMarginRuntime
from utils.method_a_margin_feedback import MarginFeedback
from utils.method_a_margin_math import (ARM, VARIANT, DEFAULTS, algorithm_config, SampleMarginHistory,
                                        make_targets, keep_terms)
from utils.sfra_math import flatten, projected_step
from utils.sfra_resident_feedback import ResidentWitnessFeedback


def feedback_fixture(cached=True, overlap=False):
    bank = visual_bank(fast=cached)
    if cached:
        bank._fast_feedback = ResidentWitnessFeedback(bank, 12, 0)
    for t in bank.tokens:
        t['raw_sample_ids'] = [100*t['client_id']+i for i in t['local_positions']]
    counts = [[50, 0, 2], [0, 3, 0]]
    if overlap:
        bank.datasets[1].append(copy.deepcopy(bank.datasets[1][0]))
        bank.tokens.append(dict(client_id=1, class_id=0, local_positions=[3], raw_sample_ids=[103]))
        counts[1][0] = 5
    feedback = MarginFeedback(bank, counts, ([0], [1], [2]), torch.tensor([-.1, -.7, -.8]))
    return bank, feedback


class TestMethodAMargin(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def test_capped_positive_history_class_balance_and_one_sided_gradient(self):
        manifest = [dict(sample_id=i, class_id=c) for i, c in enumerate([0, 0, 1, 1])]
        history = SampleMarginHistory([[.08, .1], [.004, .02], [.3, .4], [-.1, .4]], manifest)
        targets = make_targets(history, [1, 3, 5, 2], .01, .01)
        torch.testing.assert_close(targets['target'][:, 0], torch.tensor([.01, .004, .01, 0.]))
        torch.testing.assert_close(targets['omega'], torch.tensor([.125, .375, .5, 0.]))
        m = torch.tensor([[.04, .03], [.002, .003], [-.02, .02], [-.5, -.4]], requires_grad=True)
        loss = keep_terms(m, targets['target'], targets['sigma'], targets['omega'][:, None]).sum()
        gradient = torch.autograd.grad(loss, m)[0]
        expected = -targets['omega'][:, None] / (2*.01**2) * (targets['target']-m.detach()).clamp_min(0)
        torch.testing.assert_close(gradient, expected)
        self.assertTrue(torch.equal(gradient[0], torch.zeros(2)))
        self.assertTrue(torch.equal(gradient[3], torch.zeros(2)))
        self.assertLess(float(gradient[2, 0]), 0.)
        # Lowering a formerly safe sample still activates its fixed target.
        changed = m.detach().clone(); changed[0] = -.01
        self.assertGreater(float(keep_terms(changed, targets['target'], targets['sigma'], targets['omega'][:, None])[0].sum()), 0.)
        history.valid.zero_()
        empty = make_targets(history, [1, 3, 5, 2], .01, .01)
        self.assertEqual(float(empty['omega'].sum()), 0.)

    def test_joint_gradient_radius_once_and_global_ball(self):
        z = torch.tensor([.15, -.3], requires_grad=True)
        radius = .37
        a = torch.tensor([.2, -.1]) + radius*z
        logits = torch.stack((a, -a), 1)
        ce = F.cross_entropy(logits, torch.tensor([0, 1]))
        keep = keep_terms(a, torch.ones(2)*.3, .2, torch.tensor([.2, .8])).sum()
        data = 1.7*keep + .4*ce
        ga = torch.autograd.grad(data, a, retain_graph=True)[0]
        gz = torch.autograd.grad(.5*z.square().sum()+data, z)[0]
        torch.testing.assert_close(gz, z+radius*ga)
        raw = z.detach()-.1*gz
        torch.testing.assert_close(projected_step(z.detach(), radius, ga, 1.), raw/raw.norm().clamp_min(1))
        huge = projected_step(torch.zeros(4), 3., torch.ones(4)*100, 1.)
        self.assertAlmostEqual(float(huge.norm()), 1., places=6)

    def test_reliable_window_registration_next_round_and_midwindow_restore(self):
        manifest = [dict(sample_id=i, class_id=0) for i in range(3)]
        history = SampleMarginHistory([[.02, .04], [-.1, .2], [.01, -.01]], manifest)
        self.assertEqual(history.valid.tolist(), [True, False, False])
        for rnd in range(1, 5):
            history.commit([[.03, .04], [.01, .02], [-.01 if rnd == 2 else .01, .02]], rnd)
        restored = SampleMarginHistory.restore(history.state_dict(), manifest)
        before = history.H.clone()
        for h in (history, restored):
            h.commit([[.03, .04], [.01, .02], [.01, .02]], 5)
        self.assertEqual(history.valid.tolist(), [True, True, False])
        self.assertEqual(history.registered_round.tolist(), [5, 5, -1])
        self.assertFalse(torch.equal(before, history.H))
        torch.testing.assert_close(history.H, restored.H)
        frozen = make_targets(history, [1, 1, 1], .01, .01)
        history.commit([[.001, .002], [-.1, -.2], [.5, .6]], 6)
        torch.testing.assert_close(frozen['target'][:, 0], torch.tensor([.01, .01, 0.]))
        self.assertEqual(history.last_committed_round, 6)
        self.assertNotIn('pi', history.state_dict())
        with self.assertRaisesRegex(ValueError, 'exactly once'):
            history.commit([[1., 1.]]*3, 6)
        with self.assertRaisesRegex(ValueError, 'identities'):
            SampleMarginHistory.restore(history.state_dict(), list(reversed(manifest)))

    def test_joint_feedback_matches_independent_full_model_and_cache(self):
        left, ref = feedback_fixture(False, overlap=True)
        right, fast = feedback_fixture(True, overlap=True)
        history = SampleMarginHistory(torch.ones(len(ref.manifest), 2)*.2, ref.manifest)
        history.valid[::3] = False
        history.H[::3] = 0.
        targets = make_targets(history, ref.base_mass, .01, .2)
        rng = torch.get_rng_state().clone()
        flags = [(p.requires_grad, p.grad) for p in right.model.parameters()]
        x = ref.evaluate('reference', targets, True, 1.7, .4, cached=False)
        with patch('torch.autograd.grad', wraps=torch.autograd.grad) as backwards:
            y = fast.evaluate('cached', targets, True, 1.7, .4)
        self.assertEqual(backwards.call_count, y['batches'])
        self.assertEqual(y['backward_calls'], y['batches'])
        self.assertEqual(y['forward_images'], 2*len(ref.manifest))
        self.assertEqual(y['backward_images'], y['forward_images'])
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertEqual(flags, [(p.requires_grad, p.grad) for p in right.model.parameters()])
        for key in ('sample_margins', 'group_ce', 'keep_loss', 'class_loss', 'loss', 'gradient'):
            torch.testing.assert_close(torch.as_tensor(x[key]), torch.as_tensor(y[key]), atol=5e-6, rtol=5e-5)
        self.assertNotIn('coordinates', y)
        self.assertNotIn('gradient_norms', y)
        # Independent loss/gradient formula over raw images, with a class held by two clients.
        train_only(left.model, 'A'); left.model.eval()
        expected_class = expected_keep = 0.
        for q, t in enumerate(left.tokens):
            images = torch.stack([left.datasets[t['client_id']][j]['img'] for j in t['local_positions']])
            ids = torch.arange(ref.offsets[q], ref.offsets[q]+len(images))
            c = t['class_id']
            for v, ims in enumerate((images, images.flip(-1))):
                f = left.core.image_encoder(ims); f = f/f.norm(dim=-1, keepdim=True)
                cosine = f@left.text_features.T
                ce = F.cross_entropy(left.core.logit_scale.exp()*cosine+ref.adjustment,
                                     torch.full((len(ims),), c), reduction='none')
                coefficient = ref.counts[t['client_id'], c] / ref.counts[:, c].sum() / (3*2*len(ims))
                expected_class = expected_class + coefficient*ce.sum()
                rivals = cosine[:, [j for j in range(3) if j != c]].max(1).values
                margin = cosine[:, c]-rivals
                expected_keep = expected_keep + keep_terms(margin, targets['target'][ids, v], .2, targets['omega'][ids]).sum()
        expected = 1.7*expected_keep + .4*expected_class
        g = torch.cat([p.flatten() for p in torch.autograd.grad(expected, left.parameters)])
        torch.testing.assert_close(torch.tensor(y['class_loss']), expected_class.detach(), atol=4e-6, rtol=4e-5)
        torch.testing.assert_close(y['gradient'], g, atol=5e-6, rtol=5e-5)

    def test_classifier_gradient_includes_samples_without_history(self):
        bank, feedback = feedback_fixture()
        history = SampleMarginHistory(-torch.ones(len(feedback.manifest), 2), feedback.manifest)
        targets = make_targets(history, feedback.base_mass, .01, .01)
        result = feedback.evaluate('classification', targets, gradient=True)
        self.assertEqual(result['keep_loss'], 0.)
        self.assertGreater(float(result['gradient'].norm()), 0.)
        self.assertEqual(result['backward_calls'], result['batches'])
        self.assertAlmostEqual(result['class_loss'], float(result['group_ce'].mean()), places=6)

    def make_runtime(self, folder, last):
        bank, feedback = feedback_fixture()
        rt = object.__new__(MethodAMarginRuntime)
        rt.root = Path(folder); (rt.root/'checkpoints').mkdir(parents=True, exist_ok=True)
        rt.bank, rt.margin_feedback = bank, feedback
        rt.trainer = SimpleNamespace(model=bank.model, device=bank.device)
        rt.a_keys = bank.keys
        rt.b_keys = sorted(n for n, _ in bank.model.named_parameters() if n.endswith('_lora_B'))
        rt.keys = sorted(rt.a_keys+rt.b_keys); rt.base_state = snapshot(bank.model)
        rt.variant = VARIANT; rt.strength = 1.; rt.b_transfer = None
        rt.sizes = [7, 3]; rt.joint_weights = {0: .3, 1: .7}; rt.job = {'aggregation': {'rule': 'joint'}}
        rt.sfra_config = algorithm_config(); rt.execution_config = None; rt.config = {}
        rt.events = []; rt.budget = []; rt.evaluations = []; rt.costs = []; rt.summaries = []
        rt.elapsed_before = 0.; rt.started = time.perf_counter(); rt.resume_payload = None
        rt.args = SimpleNamespace(sfra_stop_after_round=last)
        labels = {j: {p: t['class_id'] for t in bank.tokens if t['client_id'] == j
                      for p in t['local_positions']} for j in range(2)}
        def phase(state, rnd, factor, extra=False, return_deltas=False):
            local = {}; keys = rt.a_keys if factor == 'A' else rt.b_keys
            for j in range(2):
                bank.model.load_state_dict(state); train_only(bank.model, factor)
                opt = torch.optim.SGD([p for p in bank.model.parameters() if p.requires_grad], lr=.001, momentum=.9)
                images = torch.stack([x['img'] for x in bank.datasets[j]])
                y = torch.tensor([labels[j][i] for i in range(len(images))])
                for _ in range(1 if extra else 3):
                    opt.zero_grad(); f = bank.core.image_encoder(images); f = f/f.norm(dim=-1, keepdim=True)
                    loss = F.cross_entropy(bank.core.logit_scale.exp()*f@bank.text_features.T+feedback.adjustment, y)
                    loss.backward(); opt.step()
                local[j] = snapshot(bank.model)
                rt.budget.append(dict(round=rnd, phase='refresh_A' if extra else 'normal_B', optimizer_steps=1 if extra else 3))
            result = dict(state)
            deltas = {j: {k: local[j][k]-state[k] for k in keys} for j in local}
            for k in keys:
                result[k] = sum(rt.joint_weights[j]*local[j][k] for j in local)
            rt.events.append(dict(round=rnd))
            return (result, deltas) if return_deltas else result
        rt.train_phase = phase
        def publish(state, rnd, _):
            rt.evaluations.append(dict(round=rnd))
            write_table(rt.root/'round_metrics.csv', rt.evaluations)
        rt.publish = publish
        return rt

    def test_third_step_committed_even_when_all_losses_increase(self):
        with tempfile.TemporaryDirectory() as temp:
            rt = self.make_runtime(temp, 1)
            rt.history = SampleMarginHistory(torch.ones(len(rt.margin_feedback.manifest), 2)*.08, rt.margin_feedback.manifest)
            ordinary = snapshot(rt.trainer.model)
            original = copy.deepcopy(ordinary)
            # A cancelled aggregated direction still has positive local RMS.
            delta = {0: {k: torch.ones_like(ordinary[k])*.01 for k in rt.a_keys},
                     1: {k: torch.ones_like(ordinary[k])*(-.01*.3/.7) for k in rt.a_keys}}
            rt.train_phase = lambda *a, **kw: (ordinary, delta)
            rt.events = [{}]
            phases = []
            def observe(state, rnd, phase, **kwargs):
                phases.append(phase)
                for k in rt.b_keys:
                    torch.testing.assert_close(state[k], ordinary[k], atol=0, rtol=0)
                final = phase == 'submitted'
                return dict(sample_margins=torch.ones(len(rt.history.H), 2)*(-.2 if final else -.01),
                    group_ce=torch.ones(3)*(100 if final else 1), keep_loss=100. if final else 1.,
                    class_loss=100. if final else 1., loss=200. if final else 2., deficit_count=len(rt.history.H),
                    gradient=torch.ones(rt.bank.numel))
            rt.observe_margin = observe
            submitted, row = rt.refresh(ordinary, 1)
            self.assertEqual(phases, ['ordinary', 'update_z0', 'update_z1', 'update_z2', 'submitted'])
            self.assertEqual(row['submitted_step'], 3)
            self.assertGreater(row['J_submitted'], row['J_ordinary'])
            self.assertGreater(row['final_CE_M'], row['ordinary_CE_M'])
            self.assertTrue(row['nonzero_correction'])
            self.assertLessEqual(row['correction_z_norm'], 1.000001)
            torch.testing.assert_close(rt.history.block_min, torch.ones(len(rt.history.H))*(-.2))
            for k in rt.a_keys:
                torch.testing.assert_close(ordinary[k], original[k], atol=0, rtol=0)
                self.assertEqual(submitted[k].device, ordinary[k].device)
            audit_rounds([row], {'functional_correction_steps': 3}, 1)

    def test_no_gap_does_not_start_a_backward_or_candidate(self):
        for reason in ('no_valid_history', 'zero_proposal_radius', 'no_history_deficit'):
            with self.subTest(reason=reason), tempfile.TemporaryDirectory() as temp:
                rt = self.make_runtime(temp, 1)
                rt.history = SampleMarginHistory(torch.ones(len(rt.margin_feedback.manifest), 2)*
                    (-1. if reason == 'no_valid_history' else .08), rt.margin_feedback.manifest)
                ordinary = snapshot(rt.trainer.model)
                delta = {j: {k: torch.ones_like(ordinary[k])*(0. if reason == 'zero_proposal_radius' else .01)
                             for k in rt.a_keys} for j in range(2)}
                rt.train_phase = lambda *a, **kw: (ordinary, delta); rt.events = [{}]
                calls = []
                def observe(state, rnd, phase, **kwargs):
                    calls.append((phase, kwargs.get('gradient', False)))
                    return dict(sample_margins=torch.ones(len(rt.history.H), 2)*.04,
                        group_ce=torch.ones(3), keep_loss=0., class_loss=1., loss=1., deficit_count=0)
                rt.observe_margin = observe
                submitted, row = rt.refresh(ordinary, 1)
                self.assertIs(submitted, ordinary)
                self.assertEqual(calls, [('ordinary', False)])
                self.assertEqual(row['skip_reason'], reason)
                self.assertEqual(row['submitted_step'], 0)
                self.assertEqual(rt.history.last_committed_round, 1)

    def test_tiny_six_round_flow_and_exact_midwindow_resume(self):
        with tempfile.TemporaryDirectory() as temp, patch('builtins.print'):
            rt = self.make_runtime(Path(temp)/'continuous', 6)
            rt.run(copy.deepcopy(rt.base_state))
            final = snapshot(rt.trainer.model)
            self.assertEqual(rt.summaries[4]['window_count'], 0)
            self.assertEqual(rt.summaries[5]['history_last_used'], 5)
            self.assertEqual(rt.history.last_committed_round, 6)
            audit_rounds(rt.summaries, load_json(rt.root/'progress.json'), 6)
            self.assertTrue(all(c['backward_calls'] == c['batches'] for c in rt.costs if c['backward_images']))
            self.assertTrue(all(c['phase'] in ('initial', 'ordinary', 'update_z0', 'update_z1', 'update_z2', 'submitted') for c in rt.costs))
            paused = self.make_runtime(Path(temp)/'resumed', 4)
            paused.run(copy.deepcopy(paused.base_state))
            resumed = self.make_runtime(paused.root, 6)
            resumed.resume_payload = torch.load(paused.root/'checkpoints/sfra_last.pt', weights_only=False)
            resumed.run(copy.deepcopy(resumed.base_state))
            for k, v in final.items():
                torch.testing.assert_close(v, resumed.trainer.model.state_dict()[k], atol=0, rtol=0)
            for k in ('H', 'valid', 'registered_round', 'block_min'):
                torch.testing.assert_close(getattr(rt.history, k), getattr(resumed.history, k), atol=0, rtol=0)
            # Round91 uses ordinary B only and does not observe/update A history.
            late = self.make_runtime(Path(temp)/'late', 91)
            late.history = copy.deepcopy(rt.history)
            late.history.last_committed_round = 90
            late.history.block_count = 0; late.history.block_min.fill_(float('inf'))
            before = late.history.state_dict()
            late.resume_payload = {'test': True}
            late.restore = lambda: (copy.deepcopy(final), 90)
            with patch.object(late, 'observe_margin', side_effect=AssertionError('No A feedback after90')):
                late.run(copy.deepcopy(final))
            for k in late.a_keys:
                torch.testing.assert_close(late.trainer.model.state_dict()[k], final[k], atol=0, rtol=0)
            torch.testing.assert_close(late.history.H, before['H'])
            self.assertEqual(late.history.last_committed_round, 90)

    def test_entry_registers_parameters_no_B_and_six_round_smoke_only(self):
        for bad in (['--epsilon', 'nan'], ['--margin-scale', '0'], ['--keep-weight', '-1'], ['--b-transfer-enable']):
            with self.assertRaises(SystemExit), patch('sys.stderr'):
                launcher.parse_args(bad)
        with tempfile.TemporaryDirectory() as temp:
            args = launcher.parse_args(['--output-root', temp])
            plan = launcher.register(args)
            self.assertEqual(plan, launcher.register(args))
            self.assertEqual(plan['settings']['margin_parameters'], DEFAULTS)
            command = single.command_for(single.make_job(plan, ARM, 'smoke'), temp)
            self.assertEqual(command[command.index('--sfra_variant')+1], VARIANT)
            self.assertEqual(command[command.index('--sfra_stop_after_round')+1], '6')
            self.assertNotIn('--b_transfer_enable', command)
            self.assertNotIn('F-tail-equal', command)
            self.assertEqual(command[command.index('--sfra_retention_weight')+1], '1.0')
            for key, value in DEFAULTS.items():
                self.assertEqual(float(command[command.index('--a_margin_'+key)+1]), value)
            with self.assertRaisesRegex(ValueError, 'six-round'):
                launcher.verify_smoke(temp, plan)
            with patch.object(launcher, 'register', return_value=plan), patch.object(single, 'preflight'), \
                    patch.object(single, 'execute') as execute, patch.object(launcher, 'verify_smoke', return_value={'status': 'passed'}), \
                    patch('builtins.print'):
                launcher.main(['--stage', 'smoke', '--output-root', temp])
            self.assertEqual(execute.call_count, 1)
            self.assertEqual(execute.call_args.args[0]['mode'], 'smoke')
            changed = launcher.parse_args(['--output-root', temp, '--epsilon', '.02'])
            with self.assertRaisesRegex(ValueError, 'changed'):
                launcher.register(changed)

    def test_real_artifact_audit_accepts_direct_commit_loss_increase(self):
        from tests import test_joint_aggregation as fixture
        from tools.client_aggregation.analysis import audit_run
        from tools.client_aggregation.protocol import digest
        from scripts.run_ab_validation import write_json
        with tempfile.TemporaryDirectory() as temp, patch('builtins.print'):
            root = Path(temp)
            job = fixture.TestJointAggregation().audit_fixture(root, 'method_a', 'joint')
            job['settings'] = dict(job.get('settings', {}), margin_parameters=DEFAULTS)
            job.update(arm=ARM, contract=method_a_margin_contract(DEFAULTS))
            cfg = load_json(root/'sfra_config.json'); cfg.update(algorithm_config())
            cfg['joint_aggregation']['arm'] = ARM
            write_json(root/'sfra_config.json', cfg); write_json(root/'joint_job.json', job)
            receipt = load_json(root/'joint_receipt.json'); receipt['job_digest'] = digest(job)
            write_json(root/'joint_receipt.json', receipt)
            progress = load_json(root/'progress.json'); progress['arm'] = ARM
            write_json(root/'progress.json', progress)
            row = dict(round=1, variant=VARIANT, correction_steps=3, submitted_step=3, correction_started=True,
                commit_rule=algorithm_config()['commit_rule'], skip_reason='', history_last_committed=1,
                history_last_used=0, window_count=1, correction_z_norm=.5, radius=.1,
                ordinary_keep=1., final_keep=2., ordinary_class=1., final_class=2.,
                J_ordinary=2., J_submitted=4.125, ordinary_deficit_count=10)
            for g in ('H', 'M', 'T'):
                row.update({f'ordinary_CE_{g}': 1., f'final_CE_{g}': 2.})
            write_table(root/'sfra_rounds.csv', [row])
            audit_run(root, job, 1)
            row['submitted_step'] = 0; write_table(root/'sfra_rounds.csv', [row])
            with self.assertRaisesRegex(ValueError, 'last computed'):
                audit_run(root, job, 1)


if __name__ == '__main__':
    unittest.main()
