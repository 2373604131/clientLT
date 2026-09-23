"""CPU numerical/unit audits only. No dataset, downloads or training runs."""
import copy
import ast
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from scripts import run_cliplora_sfra as launcher
from utils.cliplora_a_refresh import train_only
from utils.cliplora_b_transfer import differentiable_b_residual
from utils.loralib.layers import LinearLoRA, PlainMultiheadAttentionLoRA
from utils.sfra_execution import (FAST_EXECUTION_CONFIG, FrozenTextCache, FrozenVisionPrefix,
                                  LoRAStateLoader, enable_model_execution)
from utils.cliplora_sfra import SFRARuntime
from utils.sfra_fast_feedback import FastWitnessFeedback
from utils.sfra_feedback import WitnessBank
from utils.sfra_math import (FunctionalHistory, proposal_coordinates, source_statistics,
                             make_targets, projected_step, classification_preservation, flatten)


spec = importlib.util.spec_from_file_location('sfra_test_visual', Path(__file__).resolve().parents[1]/'clip/model.py')
visual_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(visual_module)


def visual_bank(fast=False, batch_size=2, forward_batch_size=12):
    torch.manual_seed(419)
    with patch('builtins.print'):
        visual = visual_module.VisionTransformer(8, 4, 8, 3, 2, 5,
                                                dict(trainer='CoOp', vision_depth=0))
    for block in list(visual.transformer.resblocks)[1:]:
        block.attn = PlainMultiheadAttentionLoRA(block.attn, enable_lora=['q', 'v'], r=4, lora_alpha=1)
    with torch.no_grad():
        for n, p in visual.named_parameters():
            if n.endswith('_lora_B'):
                p.normal_(std=.2)  # Nonzero B: A gradients must be informative.
    model = torch.nn.Module()
    model.image_encoder, model.dtype = visual, torch.float32
    model.logit_scale = torch.nn.Parameter(torch.tensor(1.4), requires_grad=False)
    train_only(model, 'B')
    bank = object.__new__(WitnessBank)
    bank.model = bank.core = model
    bank.device = torch.device('cpu')
    bank.keys = sorted(n for n, _ in model.named_parameters() if n.endswith('_lora_A'))
    parameters = dict(model.named_parameters())
    bank.parameters = [parameters[k] for k in bank.keys]
    bank.numel, bank.clients, bank.batch_size = sum(p.numel() for p in bank.parameters), 2, batch_size
    text = torch.randn(3, 5)
    bank.text_features = text/text.norm(dim=-1, keepdim=True)
    bank.datasets = [[dict(img=torch.randn(3, 8, 8)) for _ in range(7)],
                     [dict(img=torch.randn(3, 8, 8)) for _ in range(3)]]
    bank.tokens = [dict(client_id=0, class_id=0, local_positions=[0, 1, 2, 3, 4]),
                   dict(client_id=0, class_id=2, local_positions=[5, 6]),
                   dict(client_id=1, class_id=1, local_positions=[0, 1, 2])]
    bank.configure_classification(torch.tensor([[50, 0, 2], [0, 3, 0]]), torch.tensor([-.1, -.7, -.8]))
    if fast:
        for module in model.modules():
            if isinstance(module, LinearLoRA):
                module._sfra_low_rank_execution = True
        bank._fast_feedback = FastWitnessFeedback(bank, forward_batch_size)
    return bank


def targets():
    return dict(target=torch.tensor([[1.1, 1.2], [-3., -3.], [.8, .9]]),
                sigma=torch.tensor([[.2, .3], [.5, .6], [.4, .2]]),
                weights=torch.tensor([.2, .3, .5]), active=torch.tensor([True, True, True]))


class TestSFRAExecution(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def compare_feedback(self, left, right):
        fields = ('scores', 'gradient_norms', 'coordinates', 'gradient', 'loss',
                  'classification_loss', 'classification_gradient',
                  'classification_client_losses', 'classification_scores')
        for key in fields:
            if key not in left:
                continue
            with self.subTest(field=key):
                if left[key] is None:
                    self.assertIsNone(right[key])
                else:
                    torch.testing.assert_close(torch.as_tensor(left[key]), torch.as_tensor(right[key]),
                                               atol=3e-6, rtol=3e-5)
        self.assertEqual(left['forward_images'], right['forward_images'])

    def test_low_rank_output_a_b_input_gradients_and_dropout_rng(self):
        for rank in (1, 4, 8):
            for factor in ('A', 'B', 'AB'):
                for dropout in (0., .2):
                    with self.subTest(rank=rank, factor=factor, dropout=dropout):
                        torch.manual_seed(93)
                        old = LinearLoRA(torch.nn.Linear(9, 7), r=rank, lora_alpha=1, dropout_rate=dropout)
                        with torch.no_grad():
                            old.w_lora_B.normal_(std=.2)
                        train_only(old, factor)
                        new = copy.deepcopy(old)
                        new._sfra_low_rank_execution = True
                        x0 = torch.randn(3, 5, 9, requires_grad=True)
                        x1 = x0.detach().clone().requires_grad_()
                        before = torch.get_rng_state()
                        y0 = old(x0)
                        after = torch.get_rng_state()
                        torch.set_rng_state(before)
                        y1 = new(x1)
                        self.assertTrue(torch.equal(after, torch.get_rng_state()))
                        torch.testing.assert_close(y0, y1, atol=1e-6, rtol=1e-5)
                        p0 = [p for p in old.parameters() if p.requires_grad] + [x0]
                        p1 = [p for p in new.parameters() if p.requires_grad] + [x1]
                        g0 = torch.autograd.grad(y0.square().sum(), p0)
                        g1 = torch.autograd.grad(y1.square().sum(), p1)
                        for a, b in zip(g0, g1):
                            torch.testing.assert_close(a, b, atol=4e-6, rtol=2e-5)

    def test_prefix_matches_real_visual_forward_and_suffix_gradients(self):
        b = visual_bank()
        split = FrozenVisionPrefix(b.core.image_encoder)
        train_only(b.model, 'A')
        b.model.eval()
        images = torch.randn(3, 3, 8, 8)
        before = copy.deepcopy(b.model.state_dict())
        full = b.core.image_encoder(images)
        encoded = split.forward_prefix(images)
        self.assertFalse(encoded.requires_grad)
        cached = split.forward_suffix(encoded)
        torch.testing.assert_close(full, cached, atol=1e-6, rtol=1e-5)
        g0 = torch.autograd.grad(full.square().sum(), b.parameters)
        g1 = torch.autograd.grad(cached.square().sum(), b.parameters)
        for a, c in zip(g0, g1):
            torch.testing.assert_close(a, c, atol=1e-6, rtol=1e-5)
        for k, v in b.model.state_dict().items():
            self.assertTrue(torch.equal(v, before[k]))

    def test_source_retains_individual_full_gradient_norms_and_projections(self):
        old, new = visual_bank(), visual_bank(True)
        torch.manual_seed(51)
        basis, coords, _ = proposal_coordinates(torch.randn(old.numel, 4)*.01)
        a, b = old.evaluate(basis=basis), new.evaluate(basis=basis)
        self.compare_feedback(a, b)
        s0 = source_statistics(a['coordinates']@coords, torch.ones(3)/4, torch.tensor([.1, .2, .3, .4]))
        s1 = source_statistics(b['coordinates']@coords, torch.ones(3)/4, torch.tensor([.1, .2, .3, .4]))
        for key in ('supported', 'positive_count'):
            self.assertTrue(torch.equal(s0[key], s1[key]))

    def test_batched_functional_and_global_classification_gradients(self):
        for chunk in (1, 2, 8):
            for merged in (8, 12, 32):
                for cp in (False, True):
                    with self.subTest(chunk=chunk, merged=merged, cp=cp):
                        old, new = visual_bank(batch_size=chunk), visual_bank(True, chunk, merged)
                        self.compare_feedback(old.evaluate(targets=targets(), classification=cp),
                                              new.evaluate(targets=targets(), classification=cp))

    def test_no_grad_final_evaluation_and_inactive_tokens(self):
        old, new = visual_bank(), visual_bank(True)
        self.compare_feedback(old.evaluate(classification=True, classification_gradient=False),
                              new.evaluate(classification=True, classification_gradient=False))
        t = targets()
        t['active'][1] = False
        t['weights'][1] = 0
        self.compare_feedback(old.evaluate(targets=t, all_scores=False),
                              new.evaluate(targets=t, all_scores=False))
        # CP still includes the inactive functional token using original frequencies.
        self.compare_feedback(old.evaluate(targets=t, all_scores=False, classification=True),
                              new.evaluate(targets=t, all_scores=False, classification=True))

    def test_exact_zero_skips_only_functional_backward(self):
        old, new = visual_bank(), visual_bank(True)
        t = targets()
        t['target'].fill_(-3.)
        for cp in (False, True):
            a = old.evaluate(targets=t, classification=cp)
            with patch('torch.autograd.grad', wraps=torch.autograd.grad) as grad:
                b = new.evaluate(targets=t, classification=cp)
                self.assertEqual(grad.call_count, b['execution_forward_batches'] if cp else 0)
            self.compare_feedback(a, b)
            self.assertEqual(b['execution_zero_gradient_batches'], b['execution_forward_batches'])
            self.assertEqual(b['gradient'].count_nonzero().item(), 0)
            if cp:
                self.assertGreater(float(b['classification_gradient'].norm()), 0.)

    def test_feedback_preserves_rng_modes_flags_and_all_model_state(self):
        b = visual_bank(True)
        b.model.train()
        state = copy.deepcopy(b.model.state_dict())
        flags = [p.requires_grad for p in b.model.parameters()]
        modes = [m.training for m in b.model.modules()]
        rng = torch.get_rng_state().clone()
        b.evaluate(targets=targets(), classification=True)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertEqual(flags, [p.requires_grad for p in b.model.parameters()])
        self.assertEqual(modes, [m.training for m in b.model.modules()])
        for k, v in b.model.state_dict().items():
            self.assertTrue(torch.equal(state[k], v))

    def test_caches_reused_across_a_b_updates_and_invalidated_for_new_prefix(self):
        b = visual_bank(True)
        fast = b._fast_feedback
        with patch.object(fast.prefix, 'forward_prefix', wraps=fast.prefix.forward_prefix) as encode:
            initial = b.evaluate()['scores']
            calls = encode.call_count
            self.assertGreater(calls, 0)
            cached_images = dict(fast.images)
            with torch.no_grad():
                for n, p in b.model.named_parameters():
                    if n.endswith(('_lora_A', '_lora_B')):
                        p.add_(.05)
            changed = b.evaluate()['scores']
            self.assertEqual(encode.call_count, calls)
            self.assertFalse(torch.equal(initial, changed))
            with patch.object(b, '_fast_feedback', None):
                uncached = b.evaluate()['scores']
            torch.testing.assert_close(changed, uncached, atol=1e-6, rtol=1e-5)
            self.assertTrue(all(fast.images[q] is x for q, x in cached_images.items()))
            with torch.no_grad():
                b.core.image_encoder.conv1.weight.add_(.01)
            b.evaluate()
            self.assertEqual(encode.call_count, 2*calls)

    def test_groups_never_mix_clients_or_lose_views(self):
        b = visual_bank(True)
        groups = list(b._fast_feedback.groups(targets(), True, True))
        records = []
        for group in groups:
            self.assertEqual(len({b.tokens[q]['client_id'] for q, _, _, _ in group}), 1)
            self.assertLessEqual(sum(n for _, _, n, _ in group), 12)
            records.extend((q, view) for q, view, _, _ in group)
        self.assertEqual(records, [(q, v) for q in range(3) for v in range(2)])

    def test_lora_loader_restores_both_factors_once_without_resetting_frozen_weights(self):
        b = visual_bank(True)
        keys = [n for n, _ in b.model.named_parameters() if n.endswith(('_lora_A', '_lora_B'))]
        loader = LoRAStateLoader(b.model, keys)
        state = copy.deepcopy(b.model.state_dict())
        state['image_encoder.conv1.weight'].add_(.1)  # First load includes the base.
        with patch.object(b.model, 'load_state_dict', wraps=b.model.load_state_dict) as full_load:
            loader.load(state)
            frozen_version = b.model.image_encoder.conv1.weight._version
            parameters = dict(b.model.named_parameters())
            identities = {k: id(parameters[k]) for k in keys}
            for _ in range(3):
                with torch.no_grad():
                    for k in keys:
                        parameters[k].add_(1.)
                loader.load(state)
                for k in keys:
                    self.assertTrue(torch.equal(parameters[k], state[k]))
                    self.assertEqual(id(parameters[k]), identities[k])
            self.assertEqual(full_load.call_count, 1)
            self.assertEqual(b.model.image_encoder.conv1.weight._version, frozen_version)
        self.assertTrue(torch.equal(b.model.image_encoder.conv1.weight, state['image_encoder.conv1.weight']))

    def test_text_cache_invalidation_rng_and_no_checkpoint_pollution(self):
        class Text(torch.nn.Linear):
            def forward(self, prompts, tokens):
                return super().forward(prompts)
        class Prompt(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.ctx = torch.nn.Parameter(torch.randn(3, 4), requires_grad=False)
            def forward(self):
                return self.ctx
        core = torch.nn.Module()
        core.prompt_learner, core.text_encoder = Prompt(), Text(4, 5)
        core.text_encoder.requires_grad_(False)
        core.tokenized_prompts = torch.ones(3, 2, dtype=torch.long)
        keys = list(core.state_dict())
        core._sfra_text_cache = FrozenTextCache(core)
        rng = torch.get_rng_state().clone()
        with patch.object(core.text_encoder, 'forward', wraps=core.text_encoder.forward) as forward:
            a = core._sfra_text_cache.get()
            self.assertIs(a, core._sfra_text_cache.get())
            self.assertEqual(forward.call_count, 1)
            with torch.no_grad():
                core.prompt_learner.ctx.add_(.1)
            b = core._sfra_text_cache.get()
            self.assertEqual(forward.call_count, 2)
            self.assertFalse(torch.equal(a, b))
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertEqual(keys, list(core.state_dict()))

    def test_b_transfer_c_gradient_is_preserved_by_low_rank_forward(self):
        torch.manual_seed(141)
        old = LinearLoRA(torch.nn.Linear(8, 6), r=4, lora_alpha=1)
        with torch.no_grad():
            old.w_lora_B.normal_(std=.1)
        old.requires_grad_(False)
        new = copy.deepcopy(old)
        new._sfra_low_rank_execution = True
        c0 = torch.randn(4, 4, requires_grad=True)
        c1 = c0.detach().clone().requires_grad_()
        donor, x = torch.randn(6, 4), torch.randn(3, 8)
        with differentiable_b_residual({'b': old}, {'b': donor@c0}):
            y0 = old(x)
        with differentiable_b_residual({'b': new}, {'b': donor@c1}):
            y1 = new(x)
        torch.testing.assert_close(y0, y1)
        torch.testing.assert_close(torch.autograd.grad(y0.square().sum(), c0)[0],
                                   torch.autograd.grad(y1.square().sum(), c1)[0])

    def test_full_three_corrections_and_history_decisions_match(self):
        outcomes = []
        for fast in (False, True):
            b = visual_bank(fast)
            torch.manual_seed(521)
            deltas = torch.randn(b.numel, 4)*.002
            basis, coords, radius = proposal_coordinates(deltas)
            source = b.evaluate(basis=basis)
            stats = source_statistics(source['coordinates']@coords, torch.ones(3)/4, torch.tensor([.1,.2,.3,.4]))
            history = FunctionalHistory(source['scores'], 2)
            t = make_targets(source['scores'], stats, history.level, history.valid, 'full-cp')
            t['sigma'] = (radius*source['gradient_norms']).clamp_min(.001)
            ordinary = flatten(b.model.state_dict(), b.keys) + deltas.mean(1)
            z = torch.zeros_like(ordinary)
            cp_active = []
            for step in range(3):
                with torch.no_grad():
                    offset = 0
                    for p in b.parameters:
                        p.copy_((ordinary+radius*z)[offset:offset+p.numel()].reshape_as(p))
                        offset += p.numel()
                measured = b.evaluate(targets=t, classification=True)
                if step == 0:
                    reference = measured['classification_loss']
                    scale = max(.001, float(radius*measured['classification_gradient'].norm()))
                _, coefficient = classification_preservation(measured['classification_loss'], reference, scale)
                cp_active.append(coefficient > 0)
                z = projected_step(z, radius, measured['gradient'], 10.,
                                   auxiliary_gradient=coefficient*measured['classification_gradient'])
            # Include the third committed correction in the final measurement.
            with torch.no_grad():
                offset = 0
                for p in b.parameters:
                    p.copy_((ordinary+radius*z)[offset:offset+p.numel()].reshape_as(p))
                    offset += p.numel()
            scores = b.evaluate()['scores']
            for r in range(1, 6):
                registered = history.commit(scores, r)
            outcomes.append((z, scores, cp_active, registered, stats['positive_count'], ordinary+radius*z))
        # FP32 sums differ in order. Also bound the ACTUAL committed weights,
        # not only the normalized correction coordinate z (which divides by R).
        torch.testing.assert_close(outcomes[0][0], outcomes[1][0], atol=1e-5, rtol=5e-5)
        torch.testing.assert_close(outcomes[0][1], outcomes[1][1], atol=1e-6, rtol=1e-5)
        torch.testing.assert_close(outcomes[0][5], outcomes[1][5], atol=5e-7, rtol=1e-5)
        self.assertLess(float((outcomes[0][5]-outcomes[1][5]).abs().max()), 5e-7)
        self.assertEqual(outcomes[0][2], outcomes[1][2])
        for i in (3, 4):
            self.assertTrue(torch.equal(outcomes[0][i], outcomes[1][i]))

    def test_real_custom_clip_forward_uses_cache_without_changing_logits_or_image_gradient(self):
        # Exercise the actual class body without importing the unrelated trainer
        # registry/TensorBoard dependencies or downloading a pretrained CLIP.
        source = Path(__file__).resolve().parents[1]/'trainers/cliplora.py'
        tree = ast.parse(source.read_text(encoding='utf-8-sig'))
        node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'CustomCLIP')
        namespace = {'nn': torch.nn, 'torch': torch}
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), 'exec'), namespace)
        class Prompt(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.ctx = torch.nn.Parameter(torch.randn(3, 5), requires_grad=False)
            def forward(self):
                return self.ctx
        class Text(torch.nn.Linear):
            def forward(self, prompts, tokens):
                return super().forward(prompts)
        core = namespace['CustomCLIP'].__new__(namespace['CustomCLIP'])
        torch.nn.Module.__init__(core)
        core.prompt_learner, core.text_encoder = Prompt(), Text(5, 4)
        core.text_encoder.requires_grad_(False)
        core.tokenized_prompts = torch.zeros(3, 2, dtype=torch.long)
        core.image_encoder = torch.nn.Linear(6, 4)
        core.logit_scale = torch.nn.Parameter(torch.tensor(2.), requires_grad=False)
        core.dtype, core.class_residual = torch.float32, None
        inputs = torch.randn(7, 6, requires_grad=True)
        baseline = core(inputs)
        g0 = torch.autograd.grad(baseline.square().sum(), inputs)[0]
        enable_model_execution(core)
        with patch.object(core.text_encoder, 'forward', wraps=core.text_encoder.forward) as text:
            observed = core(inputs)
            core(inputs)
            self.assertEqual(text.call_count, 1)
        g1 = torch.autograd.grad(observed.square().sum(), inputs)[0]
        torch.testing.assert_close(observed, baseline, atol=0, rtol=0)
        torch.testing.assert_close(g1, g0, atol=0, rtol=0)

    def test_actual_runtime_refresh_and_execution_checkpoint_metadata(self):
        results = []
        for fast in (False, True):
            b = visual_bank(fast)
            rt = object.__new__(SFRARuntime)
            rt.trainer = SimpleNamespace(model=b.model, device=b.device)
            rt.bank, rt.a_keys = b, b.keys
            rt.b_keys = sorted(n for n, _ in b.model.named_parameters() if n.endswith('_lora_B'))
            rt.keys = sorted(rt.a_keys+rt.b_keys)
            rt._execution_loader = LoRAStateLoader(b.model, rt.keys) if fast else None
            rt.execution_config = dict(FAST_EXECUTION_CONFIG) if fast else None
            rt.variant, rt.strength, rt.scaling, rt.classification_strength = 'full-cp', 10., .5, 1.
            rt.costs, rt.events, rt.q = [], [{}], {0: .8, 1: .2}
            middle = copy.deepcopy(b.model.state_dict())
            rt.history = FunctionalHistory(b.evaluate()['scores'], 2)
            rt.history.valid[:] = True
            rt.history.level[:] = 1.3
            torch.manual_seed(318)
            deltas = {i: {k: torch.randn_like(middle[k])*.003 for k in rt.a_keys} for i in (0, 1)}
            ordinary = copy.deepcopy(middle)
            for k in rt.a_keys:
                ordinary[k] += .8*deltas[0][k]+.2*deltas[1][k]
            rt.train_phase = lambda *args, **kwargs: (ordinary, deltas)
            with tempfile.TemporaryDirectory() as temp, patch('builtins.print'):
                committed, _, arrays, summary = rt.refresh(middle, 1, Path(temp))
                steps = (Path(temp)/'correction_steps.csv').read_text().splitlines()
                self.assertEqual(len(steps), 4)
                for k in middle:
                    if k not in rt.a_keys:
                        self.assertTrue(torch.equal(middle[k], committed[k]))
                # Serialize through the real checkpoint path, without training.
                rt.root = Path(temp)
                (rt.root/'checkpoints').mkdir()
                rt.base_state, rt.sfra_config = middle, {'variant': 'full-cp'}
                rt.budget, rt.evaluations, rt.summaries = [], [], []
                rt.b_transfer, rt.elapsed_before, rt.started = None, 0., 0.
                rt.flush_records = lambda completed: None
                rt.checkpoint(committed, 1)
                saved = torch.load(rt.root/'checkpoints/sfra_last.pt', weights_only=False)
                self.assertEqual(saved.get('execution_config'), rt.execution_config)
                self.assertEqual(saved['sfra_config'], {'variant': 'full-cp'})
                # No cached images, prefix features or text tensors in checkpoints.
                self.assertNotIn('_fast_feedback', saved)
                self.assertTrue(all('_sfra_' not in k for k in saved['state_dict_overrides']))
            results.append((flatten(committed, rt.a_keys), arrays, summary))
        torch.testing.assert_close(results[0][0], results[1][0], atol=5e-7, rtol=1e-5)
        self.assertLess(float((results[0][0]-results[1][0]).abs().max()), 5e-7)
        for key in ('active', 'supported', 'positive_sources'):
            self.assertTrue(torch.equal(results[0][1][key], results[1][1][key]))
        self.assertEqual(results[0][2]['correction_steps'], results[1][2]['correction_steps'])

    def test_fast_feedback_restores_modes_flags_rng_on_exception(self):
        b = visual_bank(True)
        flags = [p.requires_grad for p in b.model.parameters()]
        modes = [m.training for m in b.model.modules()]
        rng = torch.get_rng_state().clone()
        with patch.object(b._fast_feedback.prefix, 'forward_suffix', side_effect=RuntimeError('test failure')):
            with self.assertRaisesRegex(RuntimeError, 'test failure'):
                b.evaluate(targets=targets(), classification=True)
        self.assertEqual(flags, [p.requires_grad for p in b.model.parameters()])
        self.assertEqual(modes, [m.training for m in b.model.modules()])
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))

    def test_resume_cold_caches_full_base_and_saved_rng(self):
        b = visual_bank(True)
        rt = object.__new__(SFRARuntime)
        rt.trainer, rt.bank = SimpleNamespace(model=b.model), b
        rt.sizes = [7, 3]
        keys = [n for n, _ in b.model.named_parameters() if n.endswith(('_lora_A', '_lora_B'))]
        rt._execution_loader = LoRAStateLoader(b.model, keys)
        rt.base_state = copy.deepcopy(b.model.state_dict())
        state = copy.deepcopy(rt.base_state)
        for k in keys:
            state[k].add_(.02)
        from utils.pfrf import capture_rng_state
        saved_rng = capture_rng_state()
        rt.resume_payload = dict(state_dict_overrides={k: state[k] for k in keys},
                                 events=[], budget=[], evaluations=[], functional_costs=[], round_summaries=[],
                                 elapsed_seconds=12., completed_round=1,
                                 functional_history=FunctionalHistory(torch.zeros(3, 2), 2).state_dict(),
                                 rng_state=saved_rng)
        rt.b_transfer, rt.flush_records = None, lambda completed: None
        with tempfile.TemporaryDirectory() as temp, patch('builtins.print'):
            rt.root = Path(temp)
            (rt.root/'round_metrics.csv').write_text('round,overall_acc\n0,0\n1,1\n2,2\n')
            torch.rand(20)  # Reconstruction must not replace the saved RNG boundary.
            restored, completed = rt.restore()
            self.assertEqual(completed, 1)
            self.assertNotIn('\n2,2', (rt.root/'round_metrics.csv').read_text())
            self.assertTrue(torch.equal(torch.get_rng_state(), saved_rng['torch']))
            self.assertFalse(b._fast_feedback.features)
            for k, v in b.model.state_dict().items():
                self.assertTrue(torch.equal(v, state[k]))
                self.assertTrue(torch.equal(v, restored[k]))
            # Caches are derived again from the restored base, not checkpoint payloads.
            observed = b.evaluate()['scores']
            with patch.object(b, '_fast_feedback', None):
                reference = b.evaluate()['scores']
            torch.testing.assert_close(observed, reference, atol=1e-6, rtol=1e-5)

    def test_launcher_is_opt_in_distinct_and_resumes_saved_execution(self):
        with tempfile.TemporaryDirectory() as temp:
            def prepare(args, run):
                (run/'protocol').mkdir(parents=True)
                return run/'protocol/full_schedule.json', ''
            commands = []
            for fast in (False, True):
                argv = ['run_cliplora_sfra.py', '--method', 'full-cp', '--output-root', temp,
                        '--b-transfer', '--transfer-lr', '.3', '--b-aggregation', 'uniform-transfer-rounds']
                if fast:
                    argv.append('--fast-execution')
                with patch('sys.argv', argv), patch.object(launcher, 'prepare_protocol', prepare), \
                        patch.object(launcher.subprocess, 'run') as execute, patch('builtins.print'):
                    launcher.main()
                command = execute.call_args.args[0]
                commands.append(command)
                self.assertEqual('--sfra_fast_execution' in command, fast)
                run = Path(command[command.index('--output-dir')+1])
                self.assertEqual(run.name.endswith('_fast'), fast)
                (run/'checkpoints').mkdir()
                (run/'checkpoints/sfra_last.pt').touch()
                with patch('sys.argv', argv+['--resume']), patch.object(launcher.subprocess, 'run') as execute, patch('builtins.print'):
                    launcher.main()
                resumed = execute.call_args.args[0]
                self.assertEqual('--sfra_fast_execution' in resumed, fast)
            def method_command(command):
                command = list(command)
                for flag in ('--output-dir', '--round_schedule_file', '--lac_partition_manifest'):
                    if flag in command:
                        command[command.index(flag)+1] = '<path>'
                return [v.replace('_bagg_uniform8_fast', '_bagg_uniform8') for v in command
                        if v != '--sfra_fast_execution']
            self.assertEqual(method_command(commands[0]), method_command(commands[1]))


if __name__ == '__main__':
    unittest.main()
