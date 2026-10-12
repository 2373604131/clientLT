"""Mechanism invariants, checkpoint provenance, and the complete CPU toy pipeline."""
import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from tools.a_projection_diagnosis.math import (
    row_basis, project_summary, ridge_probe, balanced_weights, split_delta,
    shrink_panel, effective_norm, with_delta, aggregate, metric_rows, response_decomposition,
)
from tools.a_projection_diagnosis.protocol import (
    fit_manifest, test_manifests, StateIndex, write_json, read_json, read_csv,
)
from tools.a_projection_diagnosis.states import recover_pre_a, StateStore
from tools.a_projection_diagnosis.backend import ClipBackend, token_summary
from tools.a_projection_diagnosis.experiment import longitudinal, event_panels
from tools.a_projection_diagnosis.summary import paired_interval, summarize
from utils.cliplora_a_refresh import state_hash


class ProjectionMathTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(6)
        self.a = torch.randn(4, 12, dtype=torch.float64)
        self.x = torch.randn(60, 2, 12, dtype=torch.float64)
        self.y = np.tile(np.arange(3), 20)

    def scores(self, a):
        z, _ = project_summary(self.x, a)
        return ridge_probe(z[:45], self.y[:45], z[45:], classes=3)[0]

    def test_gauge_invariance_and_train_scalar_normalization(self):
        change = torch.tensor([[2., .3, .2, 0], [0, 1., .2, 0], [.1, 0, 3., .4], [0, 0, .2, 1.]], dtype=torch.float64)
        np.testing.assert_allclose(self.scores(self.a), self.scores(change @ self.a), atol=1e-11)
        z, _ = project_summary(self.x, self.a)
        rotation = np.linalg.qr(np.random.default_rng(6).normal(size=(8, 8)))[0]
        base, _ = ridge_probe(z[:45], self.y[:45], z[45:], classes=3)
        rotated, _ = ridge_probe(z[:45] @ rotation, self.y[:45], z[45:] @ rotation, classes=3)
        np.testing.assert_allclose(base, rotated, atol=1e-11)

    def test_sampled_counts_not_pool_counts(self):
        labels = np.array([0] * 32 + [1] * 5)
        w = balanced_weights(labels, 2)
        self.assertAlmostEqual(w[:32].sum(), .5)
        self.assertAlmostEqual(w[32:].sum(), .5)
        with self.assertRaises(ValueError):
            balanced_weights(labels, 3)

    def test_parallel_invariance_rank_drop_and_outside_gram(self):
        k = 'm.w_lora_A'
        anchor = {k: self.a}
        delta = {k: torch.randn_like(self.a) * .02}
        inside, outside = split_delta(anchor, delta)
        np.testing.assert_allclose(self.scores(self.a), self.scores(self.a + inside[k]), atol=1e-11)
        torch.testing.assert_close(inside[k] + outside[k], delta[k])
        torch.testing.assert_close(self.a @ outside[k].T, torch.zeros(4, 4, dtype=torch.float64), atol=1e-13, rtol=0)
        d = outside[k]
        torch.testing.assert_close((self.a + d) @ (self.a + d).T, self.a @ self.a.T + d @ d.T)
        low = self.a.clone()
        low[3] = low[0]
        q, rank, _ = row_basis(low)
        self.assertEqual(rank, 3)
        self.assertTrue(torch.equal(q[3], torch.zeros_like(q[3])))
        self.assertEqual(row_basis(torch.zeros_like(self.a))[1], 0)

    def test_norm_match_shrinks_all_modules_by_one_scalar_and_excludes_zero(self):
        anchor, delta, scales = {}, {}, {}
        for i in range(6):
            k = 'm%d.w_lora_A' % i
            anchor[k], anchor[k[:-1] + 'B'] = self.a.clone(), torch.randn(9, 4, dtype=torch.float64)
            delta[k], scales[k] = torch.randn_like(self.a), .5
        panel = dict(big=delta, small={k: .2 * d for k, d in delta.items()}, zero={k: 0 * d for k, d in delta.items()})
        matched, records = shrink_panel(anchor, panel, scales)
        self.assertGreater(records[0]['target_norm'], 0)
        self.assertAlmostEqual(records[0]['coefficient'], .2)
        for r in records:
            self.assertLessEqual(r['coefficient'], 1)
        for k in delta:
            torch.testing.assert_close(matched['big'][k], delta[k] * .2)
        self.assertAlmostEqual(effective_norm(anchor, matched['big'], scales), effective_norm(anchor, matched['small'], scales))

    def test_token_axis_and_margin_lost_gained(self):
        tokens = torch.arange(5 * 3 * 8).reshape(5, 3, 8).float()
        value = token_summary(tokens)
        torch.testing.assert_close(value[:, 0], tokens[0])
        torch.testing.assert_close(value[:, 1], tokens[1:].mean(0))
        torch.testing.assert_close(value, token_summary(tokens.transpose(0, 1), True))
        anchor = dict(labels=np.array([0, 0, 1, 1]), sample_id=np.arange(4),
                      prediction=np.array([0, 1, 1, 0]), margin=np.array([.2, -.1, .3, -.2]))
        current = dict(anchor, prediction=np.array([1, 0, 1, 0]), margin=np.array([-.2, .1, .2, -.1]))
        row = metric_rows(current, dict(all=[0, 1]), anchor)[0]
        self.assertEqual((row['lost'], row['gained']), (1, 1))
        self.assertAlmostEqual(row['delta_margin'], -.05)
        with self.assertRaises(ValueError):
            metric_rows(dict(current, sample_id=np.arange(4)[::-1]), dict(all=[0, 1]), anchor)

    def test_sampling_is_order_independent_disjoint_and_uses_real_counts(self):
        rows = [dict(raw_sample_id=i, class_id=0 if i < 80 else 1, client_id=i % 3) for i in range(85)]
        self.assertEqual(fit_manifest(rows), fit_manifest(rows[::-1]))
        self.assertEqual(len(fit_manifest(rows)), 37)
        test = test_manifests(np.repeat(np.arange(3), 100))
        panel = {r['sample_id'] for r in test if r['split'] == 'panel'}
        remaining = {r['sample_id'] for r in test if r['split'] == 'remaining'}
        self.assertEqual(len(panel), 60)
        self.assertFalse(panel & remaining)
        self.assertEqual(len(panel | remaining), 300)

    def test_bootstrap_is_paired_and_conditional(self):
        result = paired_interval(np.full(100, .2), np.repeat([0, 1], 50), [0, 1])
        np.testing.assert_allclose(list(result.values()), .2)


class StateTests(unittest.TestCase):
    def test_restore_anchor_proposals_and_detect_frozen_or_weight_tampering(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'checkpoints').mkdir()
            torch.manual_seed(3)
            base = {'frozen': torch.randn(2)}
            for i in range(6):
                base['m%d.w_lora_A' % i] = torch.randn(4, 12)
                base['m%d.w_lora_B' % i] = torch.randn(12, 4)
            keys = sorted(set(base) - {'frozen'})
            a_keys = [k for k in keys if k.endswith('_A')]
            torch.save(base, root / 'checkpoints/base_model.pt')
            counts = np.ones((30, 100), dtype=int)
            selected = list(reversed(range(30)))
            source = dict(path=str(root), counts=counts, aggregation=dict(weights=[1 / 30] * 30),
                meta=dict(frozen_model_sha256=state_hash(base, ['frozen']), initial_lora_sha256=state_hash(base, keys),
                          schedule=[selected] * 100))
            store = StateStore(source)
            current = {k: v + .01 for k, v in store.initial.items()}
            pre = recover_pre_a(store.initial, current)
            for k in keys:
                torch.testing.assert_close(pre[k], store.initial[k] if k.endswith('_A') else current[k])
            # Explicit compressed history permits recovery when local uploads are absent.
            torch.save(dict(state_dict_overrides=current), root / 'checkpoints/r020.pt')
            torch.save({k: store.initial[k] for k in a_keys}, root / 'checkpoints/r019.pt')
            write_json(root / 'projection_states.json', dict(committed={'19': 'checkpoints/r019.pt', '20': 'checkpoints/r020.pt'}))
            store = StateStore(source)
            recovered, provenance = store.anchor(20)
            self.assertEqual(provenance['anchor_source'], 'historical_A_previous_B_current')
            self.assertEqual(state_hash(pre, keys), state_hash(recovered, keys))
            self.assertIsNone(store.historical_proposals(20, pre)[0])
            proposals = {j: {k: torch.randn_like(pre[k]) * .001 for k in a_keys} for j in selected}
            after = aggregate(pre, proposals, dict.fromkeys(selected, 1 / 30))
            event = root / 'events/r020_c000_main_refresh_A/state.pt'
            event.parent.mkdir(parents=True)
            payload = dict(round=20, phase='refresh_A', selected_client_ids=selected, client_class_counts=torch.as_tensor(counts[selected]),
                           anchor_lora_state=pre, actual_after_lora_state=after,
                           local_factor_deltas=list(proposals.values()), server_weights=torch.full((30,), 1 / 30, dtype=torch.float64))
            torch.save(payload, event)
            got, record = store.historical_proposals(20, pre)
            self.assertEqual(list(got), selected)
            self.assertEqual(record['reconstruction_max_abs_error'], 0)
            payload['actual_after_lora_state'][keys[1]] += .01
            torch.save(payload, event)
            with self.assertRaises(ValueError):
                store.historical_proposals(20, pre)
            with self.assertRaises(ValueError):
                store.validate(dict(base, frozen=base['frozen'] + 1))


class TinyLinear(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('weight', torch.randn(12, 12) * .1)
        self.w_lora_A = nn.Parameter(torch.randn(4, 12) * .1)
        self.w_lora_B = nn.Parameter(torch.randn(12, 4) * .1)
        self.scaling = .5

    def forward(self, x):
        return F.linear(x, self.weight) + .5 * F.linear(F.linear(x, self.w_lora_A), self.w_lora_B)


class TinyBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln_1 = nn.LayerNorm(12)
        self.attn = nn.ModuleDict(dict(q_proj=TinyLinear(), v_proj=TinyLinear()))

    def forward(self, x):
        normalized = self.ln_1(x)
        return x + self.attn['q_proj'](normalized) + self.attn['v_proj'](normalized)


class TinyClip(nn.Module):
    def __init__(self):
        super().__init__()
        self.resblocks = nn.ModuleDict({str(i): TinyBlock() for i in (9, 10, 11)})
        self.logit_scale = nn.Parameter(torch.tensor(1.))
        self.register_buffer('text', F.normalize(torch.randn(100, 12), dim=1))

    def forward(self, x):
        x = x.transpose(0, 1)
        for block in self.resblocks.values():
            x = block(x)
        return self.logit_scale.exp() * F.normalize(x.mean(0), dim=1) @ self.text.T


def tiny_backend(root):
    backend = ClipBackend.__new__(ClipBackend)
    backend.root, backend.device, backend.fingerprint = root, torch.device('cpu'), 'synthetic-test-only'
    backend.args = SimpleNamespace(eval_batch_size=64)
    backend.model = TinyClip()
    backend.keys = sorted(k for k in backend.model.state_dict() if k.endswith(('_lora_A', '_lora_B')))
    backend.a_keys = [k for k in backend.keys if k.endswith('_A')]
    modules = dict(backend.model.named_modules())
    backend.modules = {k: modules[k.rsplit('.', 1)[0]] for k in backend.a_keys}
    backend.block9 = [k for k in backend.a_keys if 'resblocks.9.' in k]
    backend.batch_first = dict.fromkeys(backend.a_keys, False)
    backend.scales = dict.fromkeys(backend.a_keys, .5)
    backend.fit_hash, backend.eval_hash, backend.costs = 'fit', 'evaluation', []
    backend.groups = dict(overall=list(range(100)), head20=list(range(20)), middle60=list(range(20, 80)), tail20=list(range(80, 100)))
    backend.loaders, backend.labels, backend.ids = {}, {}, {}
    full_labels = np.tile(np.arange(100), 3)
    full_images = torch.randn(300, 5, 12)
    for split, labels, images, ids in [
        ('fit', np.arange(100), torch.randn(100, 5, 12), np.arange(100)),
        ('test', full_labels, full_images, np.arange(300)),
        ('panel', full_labels[:100], full_images[:100], np.arange(100)),
        ('remaining', full_labels[100:], full_images[100:], np.arange(100, 300))]:
        items = [dict(img=x, label=int(y)) for x, y in zip(images, labels)]
        backend.loaders[split] = torch.utils.data.DataLoader(items, batch_size=64)
        backend.labels[split], backend.ids[split] = labels, ids
    backend.source = dict(rule='fedavg', counts=np.ones((30, 100), dtype=int), config=dict(tail_ids=list(range(80, 100))))
    return backend


class PipelineTests(unittest.TestCase):
    def test_missing_upload_replay_freezes_B_isolates_rng_and_caches_paired_proposals(self):
        torch.manual_seed(7)
        old_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
                backend = tiny_backend(Path(tmp))
                backend.args.seed = 42
                backend.source.update(partition_sha256='toy', meta=dict(schedule=[list(range(30))] * 100))
                backend.data = dict(clients=[[dict(img=torch.randn(5, 12), label=j)] for j in range(30)],
                                    loader=lambda items, training: torch.utils.data.DataLoader(items, batch_size=32, shuffle=training))
                backend.adjustment = torch.zeros(100)

                def step(model, optimizer, scaler, precision, images, labels, logit_adjustment):
                    optimizer.zero_grad()
                    F.cross_entropy(model(images) + logit_adjustment, labels).backward()
                    optimizer.step()

                backend.step = step
                anchor = {k: backend.model.state_dict()[k].clone() for k in backend.keys}
                before = torch.get_rng_state().clone()
                proposals, provenance = backend.regenerate(anchor, 20)
                self.assertEqual(len(proposals), 30)
                self.assertFalse(provenance['historical_exact_reproduction'])
                self.assertEqual(sum(r['steps'] for r in provenance['budgets']), 30)
                torch.testing.assert_close(before, torch.get_rng_state())
                self.assertTrue(any(torch.count_nonzero(v) for d in proposals.values() for v in d.values()))
                for k in backend.keys:
                    torch.testing.assert_close(backend.model.state_dict()[k], anchor[k])
                backend.step = lambda *a, **k: self.fail('Cached replay trained again')
                cached, _ = backend.regenerate(anchor, 20)
                for j in proposals:
                    for k in proposals[j]:
                        torch.testing.assert_close(proposals[j][k], cached[j][k])
        finally:
            torch.set_num_threads(old_threads)

    def test_complete_toy_pipeline_uses_real_hooks_probes_and_candidate_evaluation(self):
        torch.manual_seed(4)
        old_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
                root = Path(tmp)
                backend = tiny_backend(root / 'fedavg')
                initial = {k: backend.model.state_dict()[k].clone() for k in backend.keys}
                states = {r: {k: v + (.001 * min(r, 90) if k.endswith('_A') else .001 * r)
                              for k, v in initial.items()} for r in (0, 20, 50, 80, 90, 100)}
                proposals = {j: {k: torch.randn_like(initial[k]) * .002 for k in backend.a_keys} for j in range(30)}
                store = SimpleNamespace(committed=lambda r: states[r], anchor=lambda r: (states[r], dict(anchor_source='synthetic')),
                                        historical_proposals=lambda r, s: (proposals, dict(proposals='synthetic_test_only')))
                longitudinal(backend, store, backend.root / 'e1')
                event_panels(backend, store, 20, backend.root / 'round020')
                checks = read_json(backend.root / 'round020/numerical_checks.json')
                self.assertTrue(all(c['passed'] for c in checks))
                full = read_csv(backend.root / 'round020/full_model_candidates.csv')
                self.assertFalse(any(r['candidate'].startswith('local_') and r['split'] == 'remaining' for r in full))
                nonlocal_names = {r['candidate'] for r in full if not r['candidate'].startswith('local_')}
                self.assertEqual({r['candidate'] for r in full if r['split'] == 'remaining'}, nonlocal_names)
                self.assertEqual(len(read_csv(backend.root / 'round020/local_class_response.csv')), 3000)
                summary = summarize(root, plots=False)
                self.assertFalse(summary['complete'])
                self.assertEqual(summary['primary_rows'], 63)
                # Cache hits must preserve predictions without new model forwards.
                first = backend.measure(initial, dict(a=initial), 'fit')
                with patch.object(backend.model, 'forward', side_effect=AssertionError('cache missed')):
                    second = backend.measure(initial, dict(a=initial), 'fit')
                np.testing.assert_array_equal(first['predictions']['prediction'], second['predictions']['prediction'])
        finally:
            torch.set_num_threads(old_threads)


if __name__ == '__main__':
    unittest.main()
