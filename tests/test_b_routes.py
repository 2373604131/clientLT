"""Meaningful CPU checks of route spaces, gradients and the real event optimizer."""
import copy
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import torch

from utils.b_route_math import (EffectiveCoordinates, RouteProjector, cone_projection,
                                norm, projected_step, rotated_sources)
from utils.cliplora_b_routes import RouteTransfer, group_loss
from test_sfra_b_shared_transfer import tiny_transfer, A_KEY, B_KEY


class RouteMathTest(unittest.TestCase):
    def setUp(self):
        generator = torch.Generator().manual_seed(17)
        self.a = {k: torch.randn(2, 5, generator=generator, dtype=torch.float64) for k in ('a', 'b')}
        self.d = {k: torch.randn(3, 7, 2, generator=generator, dtype=torch.float64) for k in self.a}
        self.coords = EffectiveCoordinates(self.a, dict(a=.5, b=.25))
        self.y = {k: torch.randn(7, 2, generator=generator, dtype=torch.float64) for k in self.a}

    def projector(self, arm):
        return RouteProjector(arm, self.coords, self.d, {k: v.shape for k, v in self.y.items()})

    def test_effective_coordinates_and_autograd_including_rank_deficiency(self):
        for a in (torch.tensor([[1., 2., 0.], [0., 1., 1.]], dtype=torch.float64),
                  torch.tensor([[1., 2., 0.], [2., 4., 0.]], dtype=torch.float64)):
            c = EffectiveCoordinates({'x': a}, {'x': .5})
            r = torch.arange(8, dtype=torch.float64).reshape(4, 2)/10
            x = c.encode({'x': r})['x'].requires_grad_()
            recovered = c.decode({'x': x})['x']
            torch.testing.assert_close(.5*recovered@a, .5*r@a)
            self.assertAlmostEqual(float(x.detach().norm()), float((.5*r@a).norm()))
            (.5*recovered@a).square().sum().backward()
            torch.testing.assert_close(x.grad, 2*x.detach())

    def test_nested_spaces_and_idempotence(self):
        order = ['P', 'M', 'S', 'C', 'F']
        for index, arm in enumerate(order):
            p = self.projector(arm).project(self.y, .7)
            self.assertLessEqual(norm(p.values()), .700001)
            for larger in order[index:]:
                q = self.projector(larger).project(p)
                for k in p:
                    torch.testing.assert_close(p[k], q[k], rtol=1e-5, atol=1e-8)

    def test_module_scaling_and_sign_are_distinct(self):
        coords = EffectiveCoordinates({'a': torch.ones(1, 1), 'b': torch.ones(1, 1)}, {'a': 1., 'b': 1.})
        d = {k: torch.ones(1, 1, 1) for k in ('a', 'b')}
        y = {'a': torch.tensor([[1.]]), 'b': torch.tensor([[-1.]])}
        outputs = {arm: RouteProjector(arm, coords, d, {k: v.shape for k, v in y.items()}).project(y)
                   for arm in ('P', 'M', 'S')}
        self.assertEqual(norm(outputs['P'].values()), 0.)
        torch.testing.assert_close(outputs['M']['a'], y['a'])
        self.assertEqual(float(outputs['M']['b']), 0.)
        torch.testing.assert_close(outputs['S']['b'], y['b'])

    def test_nnls_small_scales_and_redundant_zero_columns(self):
        design = torch.tensor([[1., 2., 0.], [0., 0., 0.]], dtype=torch.float64)*1e-10
        y = torch.tensor([3., -2.], dtype=torch.float64)*1e-12
        p, weights, kkt = cone_projection(design, y)
        torch.testing.assert_close(p, torch.tensor([3e-12, 0.], dtype=torch.float64), atol=1e-20, rtol=1e-7)
        self.assertTrue((weights >= 0).all())
        self.assertLess(kkt, 1e-6)

    def test_rotation_preserves_geometry_and_rng_but_changes_space(self):
        torch_state, numpy_state = torch.random.get_rng_state(), np.random.get_state()
        rotated, checks = rotated_sources(self.d, 42, 30, 0)
        self.assertTrue(torch.equal(torch_state, torch.random.get_rng_state()))
        np.testing.assert_equal(numpy_state, np.random.get_state())
        self.assertTrue(all(r['gram_relative_error'] < 1e-10 for r in checks))
        old = self.projector('C').project(self.y)
        new = RouteProjector('C-rot', self.coords, rotated, {k: v.shape for k, v in self.y.items()}).project(self.y)
        self.assertGreater(norm((old[k]-new[k] for k in old)), .1)

    def test_c_can_be_reconstructed_and_normalized_step_obeys_radius(self):
        projector = self.projector('C')
        x, _ = projected_step({k: torch.zeros_like(v) for k, v in self.y.items()}, self.y, projector, .1)
        r = self.coords.decode(x)
        matrices = projector.reconstruct_c(r)
        for k in r:
            torch.testing.assert_close(torch.bmm(self.d[k], matrices[k]).mean(0), r[k])
        self.assertLessEqual(norm(x.values()), .10000001)

    def test_real_768_by_rank4_lora_hook_matches_committed_output(self):
        from utils.loralib.layers import LinearLoRA
        from utils.cliplora_b_transfer import differentiable_b_residual
        with torch.random.fork_rng():
            torch.manual_seed(23)
            layer = LinearLoRA(torch.nn.Linear(768, 768), r=4, lora_alpha=1)
            layer.eval()
            layer._sfra_low_rank_execution = True
            for p in layer.parameters():
                p.requires_grad_(False)
            inputs = torch.randn(2, 3, 768)
            coords = EffectiveCoordinates({'x': layer.w_lora_A}, {'x': layer.scaling})
            r = torch.randn_like(layer.w_lora_B)*.01
            x = coords.encode({'x': r})['x'].requires_grad_()
            with differentiable_b_residual({'x': layer}, coords.decode({'x': x})):
                observed = layer(inputs)
                observed.square().mean().backward()
            self.assertTrue(torch.isfinite(x.grad).all())
            self.assertGreater(float(x.grad.norm()), 0.)
            with torch.no_grad():
                layer.w_lora_B.add_(r)
                committed = layer(inputs)
            torch.testing.assert_close(observed, committed, atol=2e-6, rtol=1e-5)
            self.assertAlmostEqual(norm([x]), float((layer.scaling*r.double()@layer.w_lora_A.double()).norm()), places=7)
            sources = {'x': torch.randn(30, 768, 4)*.01}
            projector = RouteProjector('C', coords, sources, {'x': x.shape})
            projected = projector.project({'x': x})
            recovered = coords.decode(projected)
            c = projector.reconstruct_c(recovered)
            rebuilt = torch.bmm(sources['x'].double(), c['x']).mean(0)
            self.assertLess(norm([rebuilt-recovered['x']])/norm(recovered.values()), 1e-6)


class RouteOptimizerTest(unittest.TestCase):
    def run_arm(self, temp, arm):
        transfer = tiny_transfer(temp)
        transfer.__class__ = RouteTransfer
        transfer.bank = SimpleNamespace(batch_size=2)
        start = copy.deepcopy(transfer.model.state_dict())
        ordinary = copy.deepcopy(start)
        ordinary[B_KEY] += .1
        transfer.begin(Path(temp), 30, arm)
        batches = transfer.manifests(30)
        with transfer.session():
            transfer.copy_parameters(ordinary)
            residual = transfer.optimize(ordinary, start, batches, arm, 1., diagnostics={1: [0], 2: [0]})
        for key, value in start.items():
            torch.testing.assert_close(transfer.model.state_dict()[key], value)
            self.assertIsNone(transfer.parameters[key].grad)
        self.assertEqual(transfer.pending['summary']['optimizer_steps'], 2 if arm != 'N' else 0)
        self.assertEqual(transfer.pending['summary']['algorithm_backward_images'], 10 if arm != 'N' else 0)
        return residual, transfer

    def test_full_rank_source_c_and_free_residual_give_identical_updates(self):
        with tempfile.TemporaryDirectory() as temp:
            c, tc = self.run_arm(Path(temp)/'C', 'C')
            f, tf = self.run_arm(Path(temp)/'F', 'F')
            torch.testing.assert_close(c[B_KEY], f[B_KEY], atol=1e-7, rtol=1e-5)
            self.assertGreater(float(f[B_KEY].norm()), 0.)
            self.assertEqual([r['step'] for r in tf.pending['steps']], [1, 2])
            self.assertTrue((Path(temp)/'C/reconstructed_c.pt').is_file())

    def test_all_arms_freeze_base_and_use_same_feedback(self):
        with tempfile.TemporaryDirectory() as temp:
            feedback = []
            for arm in ('P', 'M', 'S', 'C', 'F', 'P-rot', 'C-rot', 'N'):
                _, transfer = self.run_arm(Path(temp)/arm, arm)
                if arm != 'N':
                    feedback.append([(r['step'], r['client_id'], r['tail_samples'], r['non_tail_samples'])
                                     for r in transfer.pending['feedback']])
            self.assertTrue(all(v == feedback[0] for v in feedback))

    def test_empty_tail_group_is_finite(self):
        values = torch.tensor([1., 2.], requires_grad=True)
        loss = group_loss(values, 0)
        loss.backward()
        torch.testing.assert_close(values.grad, torch.tensor([.5, .5]))


if __name__ == '__main__':
    unittest.main()
