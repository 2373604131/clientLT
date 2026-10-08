"""Upstream update parity, edge cases, raw-sample views and transactional state."""
import ast
import copy
import io
from pathlib import Path
import pickle
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from tools.benchmarks import common, collect, runtime, longtail_runtime
from tools.benchmarks.longtail_data import TrainingViews, seed_worker
from trainers.baselines import longtail
from trainers.baselines.common import trainable_state
from tests.test_paper_benchmarks import extracted, fake_reference, UPSTREAM
from scripts.run_paper_benchmarks import parser_for_cli


def options():
    result = common.read_json(common.CONFIG)
    result['longtail_baselines'] = common.read_json(common.LONGTAIL_CONFIG)
    return result


class SmallVisual(nn.Module):
    def __init__(self, classes=4):
        super().__init__()
        self.image_encoder = nn.Linear(3, 5)
        self.classifier = nn.Linear(5, classes, bias=False).requires_grad_(False)

    def forward(self, images):
        return self.classifier(self.image_encoder(images))


class NativeOutput(nn.Module):
    def __init__(self, core):
        super().__init__()
        self.core = core
        self.classifier = core.classifier

    def forward(self, images):
        features = self.core.image_encoder(images)
        return features, self.classifier(features)


class OfficialCalculationsTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(12)

    def test_la_optimizer_step_matches_explicit_prior_cross_entropy(self):
        model = SmallVisual()
        expected = copy.deepcopy(model)
        x, y = torch.randn(7, 3), torch.arange(7) % 4
        counts = torch.tensor([50., 12., 4., 1.])
        cfg = options()
        optimizer = torch.optim.SGD([p for p in expected.parameters() if p.requires_grad],
            lr=cfg['lr'], momentum=cfg['momentum'], weight_decay=cfg['weight_decay'])
        loss = F.cross_entropy(expected(x) + (counts / counts.sum()).log(), y)
        loss.backward(); optimizer.step()
        state, _ = runtime.local_train(model, None, [dict(img=x, label=y)], 'fedavg-lora-la', cfg, counts, 'cpu', True)
        for key, value in state.items():
            torch.testing.assert_close(value, trainable_state(expected)[key], atol=0, rtol=0)

    def test_fedlf_two_updates_match_official_local_train(self):
        # Execute the official Local.local_train body. Replace only dataset,
        # augmentation/architecture plumbing; retain upstream loss and updates.
        cfg = options(); cfg.update(local_epochs=2, momentum=0., weight_decay=0.)
        model = SmallVisual()
        original = copy.deepcopy(model)
        x, y = torch.randn(12, 3), torch.arange(12) % 3
        counts = torch.tensor([8., 4., 2., 0.])
        env = dict(torch=torch, nn=nn, F=F,
            transforms=SimpleNamespace(Compose=lambda _: lambda x: x, RandomCrop=lambda *a, **k: None,
                                       RandomHorizontalFlip=lambda: None),
            DataLoader=lambda dataset, **kw: dataset)
        extracted(UPSTREAM / 'fedlf/algorithm/fedlf.py', ['Local', 'DecorrLoss'], env)
        native = NativeOutput(original)
        obj = SimpleNamespace(local_model=native, data_client=[(x, y)], device='cpu',
            criterion=nn.CrossEntropyLoss(), feddecorr=env['DecorrLoss'](),
            optimizer=torch.optim.SGD([p for p in native.parameters() if p.requires_grad], lr=cfg['lr']))
        args = SimpleNamespace(num_classes=4, batch_size_local_training=32,
            num_epochs_local_training=2, rs_alpha=.25)
        # Upstream hardcodes feature dimension 256. Pad our raw features with
        # zeros and match DecorrLoss dimension independently in the loss test.
        # Use an actual 256-wide feature fixture for its unmodified method.
        model.image_encoder = nn.Linear(3, 256)
        model.classifier = nn.Linear(256, 4, bias=False).requires_grad_(False)
        original = copy.deepcopy(model); native = NativeOutput(original)
        obj.local_model = native
        obj.optimizer = torch.optim.SGD([p for p in native.parameters() if p.requires_grad], lr=cfg['lr'])
        env['Local'].local_train(obj, args, native.state_dict(), counts / counts.sum())
        state, costs = longtail.local_train(model, [dict(img=x, label=y)], 'fedlf', cfg, counts, 'cpu', 1)
        self.assertEqual(costs['optimizer_steps'], 2)
        self.assertEqual(costs['auxiliary_forward_images'], 12)
        for key, value in state.items():
            torch.testing.assert_close(value, trainable_state(original)[key], atol=2e-7, rtol=2e-6)

    def test_fedyoyo_two_updates_match_official_local_train(self):
        cfg = options(); cfg.update(local_epochs=2, momentum=0., weight_decay=0.)
        model = SmallVisual(); original = copy.deepcopy(model)
        weak, strong = torch.randn(12, 3), torch.randn(12, 3)
        y = torch.arange(12) % 4
        counts, prior = torch.tensor([8., 4., 2., 1.]), torch.tensor([.4, .3, .2, .1])
        env = dict(torch=torch, F=F, DataLoader=lambda dataset, **kw: dataset)
        extracted(UPSTREAM / 'fedyoyo/main_fedyoyo.py', ['Local'], env)
        native = NativeOutput(original)
        obj = SimpleNamespace(local_model=native, data_client=[(weak, weak, strong, y)],
            criterion=nn.CrossEntropyLoss(), optimizer=torch.optim.SGD([p for p in native.parameters() if p.requires_grad], lr=cfg['lr']))
        args = SimpleNamespace(num_epochs_local_training=2, batch_size_local_training=32,
            gamma=.1, tau=1.5, T=1.5, lamda=4., warmup=50)
        native_tensor = torch.tensor
        def tensor_on_cpu(*a, **kw):
            kw.pop('device', None)
            return native_tensor(*a, **kw)
        with patch.object(torch.Tensor, 'cuda', lambda self, *a, **kw: self), patch.object(torch, 'tensor', side_effect=tensor_on_cpu):
            env['Local'].local_train(obj, args, native.state_dict(), 20, None, counts, prior.clone())
        state, costs = longtail.local_train(model, [dict(img=weak, img_strong=strong, label=y)],
            'fedyoyo', cfg, counts, 'cpu', 20, prior=prior)
        self.assertEqual(costs['student_forward_images'], 48)
        for key, value in state.items():
            torch.testing.assert_close(value, trainable_state(original)[key], atol=2e-7, rtol=2e-6)

    def test_fedyoyo_effective_weights_match_official_method(self):
        core = SmallVisual(); native = NativeOutput(core)
        x, y = torch.randn(9, 3), torch.tensor([0, 0, 0, 1, 1, 1, 2, 2, 3])
        features = core.image_encoder(x).detach()
        centers = torch.randn(4, 5)
        env = dict(torch=torch, EPS=1e-6, DataLoader=lambda dataset, **kw: dataset)
        extracted(UPSTREAM / 'fedyoyo/main_fedyoyo.py', ['Local'], env)
        obj = SimpleNamespace(local_model=native, data_client=[(x, x, x, y)], args=SimpleNamespace(batch_size_local_training=32))
        with patch.object(torch.Tensor, 'cuda', lambda self, *a, **kw: self):
            expected = env['Local'].calculate_eff_weight(obj, centers, [3, 3, 2, 1])
        torch.testing.assert_close(longtail.effective_weight(features, y, centers, 4), expected, atol=1e-6, rtol=1e-6)

    def test_fedyoyo_teacher_is_detached_and_empty_mask_is_finite(self):
        spec = options()['longtail_baselines']['fedyoyo']
        labels = torch.tensor([0, 1])
        logits = torch.tensor([[4., 0.], [0., 4.], [1., 2.], [2., 1.]], requires_grad=True)
        loss, _, parts = longtail.fedyoyo_loss(logits, labels, torch.ones(2)/2, torch.ones(2), 50, spec)
        grad = torch.autograd.grad(parts['kd'], logits, retain_graph=True)[0]
        self.assertEqual(int(torch.count_nonzero(grad[:2])), 0)
        self.assertGreater(int(torch.count_nonzero(grad[2:])), 0)
        torch.autograd.grad(loss, logits)
        wrong = logits.detach().clone(); wrong[:2] = wrong[:2].flip(1); wrong.requires_grad_()
        loss, _, parts = longtail.fedyoyo_loss(wrong, labels, torch.ones(2)/2, torch.ones(2), 1, spec)
        self.assertEqual(parts['selected'], 0)
        self.assertEqual(float(parts['kd']), 0.)
        self.assertTrue(torch.isfinite(torch.autograd.grad(loss, wrong)[0]).all())

    def test_fedlf_singleton_coincident_centers_missing_classes(self):
        features = torch.ones(1, 4, requires_grad=True)
        logits = torch.randn(1, 3, requires_grad=True)
        loss, parts = longtail.fedlf_losses(features, logits, torch.tensor([0]),
            torch.ones(3, 4), torch.tensor([1., 0., 0.]), options()['longtail_baselines']['fedlf'])
        loss.backward()
        self.assertEqual(float(parts['decorr']), 0.)
        self.assertTrue(torch.isfinite(features.grad).all())
        self.assertTrue(torch.isfinite(logits.grad).all())

    def test_prior_smoothing_outlier_and_change_guards(self):
        spec = options()['longtail_baselines']['fedyoyo']
        current, prior = longtail.global_effective_prior([torch.tensor([4., 9., 20000.])], None, spec)
        torch.testing.assert_close(current, torch.tensor([5., 10., 7.5]))
        updated, _ = longtail.global_effective_prior([torch.tensor([9., 19., 29.])], current, spec)
        torch.testing.assert_close(updated, .9 * current + .1 * torch.tensor([10., 20., 30.]))
        _, prior = longtail.global_effective_prior([torch.full((3,), 2e6)], None, spec)
        torch.testing.assert_close(prior, torch.ones(3)/3)

    def test_relabel_official_functions_and_count_direction(self):
        env = dict(np=np)
        names = ['compute_normalized_priors', 'compute_classwise_statistics', 'zscore',
                 'compute_adaptive_threshold', 'get_rho_with_tanh_norm']
        extracted(UPSTREAM / 'fedrela/util/FedReLa.py', names, env)
        rng = np.random.RandomState(19)
        p = rng.dirichlet(np.ones(4), size=200).astype(np.float32)
        labels = np.array([0]*120 + [1]*60 + [2]*20)
        ids = list(range(1000, 1200)); args = SimpleNamespace(id='cliplora_adaptive')
        pred = {k: (v.copy(), int(y)) for k, v, y in zip(ids, p, labels)}
        with patch('sys.stdout', new_callable=io.StringIO):
            priors, counts = env['compute_normalized_priors'](labels, 4)
            means, stds, scores = env['compute_classwise_statistics'](p, labels, ids, 4)
            _, matrix = env['zscore'](pred, means, stds, scores, 0)
            thresholds = env['compute_adaptive_threshold'](matrix, 30, args, 0)
            np.random.seed(28)
            expected, _, _, n = env['get_rho_with_tanh_norm'](pred, scores, priors, thresholds, 4, args)
        np.random.seed(28)
        actual, audit = longtail.fedrela_mapping(p, labels, ids, 30)
        self.assertEqual(actual, expected)
        self.assertEqual(audit['changed'], n)
        self.assertGreater(n, 0)
        self.assertEqual(sum(audit['relabeled_counts']), len(labels))
        for raw_id, target in actual.items():
            self.assertLess(counts[target], counts[labels[raw_id-1000]])

    def test_relabel_balanced_or_singleton_no_spurious_changes(self):
        p = np.array([[.6, .3, .1], [.1, .7, .2], [.2, .1, .7]])
        mapping, _ = longtail.fedrela_mapping(p, [0, 1, 2], [6, 7, 8], 5)
        self.assertEqual(mapping, {})
        mapping, _ = longtail.fedrela_mapping(p[:1], [0], [6], 5)
        self.assertEqual(mapping, {})

    def test_relabel_uses_ids_and_late_lr_without_mutating_labels(self):
        model = SmallVisual(); expected = copy.deepcopy(model)
        cfg = options(); cfg['local_epochs'] = 1
        x, y, ids = torch.randn(3, 3), torch.tensor([0, 0, 1]), torch.tensor([50, 11, 80])
        target = torch.tensor([0, 2, 1])
        optim = torch.optim.SGD([p for p in expected.parameters() if p.requires_grad], lr=.0001,
            momentum=cfg['momentum'], weight_decay=cfg['weight_decay'])
        F.cross_entropy(expected(x), target).backward(); optim.step()
        state, costs = longtail.local_train(model, [dict(img=x, label=y, index=ids)], 'fedrela', cfg,
            torch.ones(4), 'cpu', 91, relabels={11: 2}, relabel_active=True)
        self.assertEqual(y.tolist(), [0, 0, 1])
        self.assertEqual(costs['relabeled_training_images'], 1)
        for key, value in state.items():
            torch.testing.assert_close(value, trainable_state(expected)[key], atol=0, rtol=0)


class DataAndProtocolTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)

    def test_defaults_do_not_change_previous_suites(self):
        self.assertEqual(tuple(parser_for_cli().parse_args([]).methods), common.METHODS)
        self.assertEqual(tuple(parser_for_cli('factors').parse_args([]).methods), common.FACTOR_METHODS)
        args = parser_for_cli('longtail').parse_args([])
        self.assertEqual(tuple(args.methods), common.LONGTAIL_METHODS)
        self.assertEqual(args.seed, 42)
        self.assertIn('longtail_benchmarks_seed42_v1', str(args.output_root))
        longtail.validate_options(options())

    def test_base_transform_exactly_matches_repository(self):
        from torchvision import transforms as T
        raw = np.random.RandomState(3).randint(0, 256, (32, 32, 3), dtype=np.uint8)
        dataset = TrainingViews([SimpleNamespace(data=raw, label=2)], [19], 'fedrela')
        expected = T.Compose([T.ToTensor(), T.Normalize((.5071, .4865, .4409), (.2673, .2564, .2762)), T.Resize([224, 224])])(raw)
        torch.testing.assert_close(dataset[0]['img'], expected, atol=0, rtol=0)
        self.assertEqual(dataset[0]['index'], 19)

    def test_three_view_modes_and_spawn_pickle(self):
        raw = np.random.RandomState(3).randint(0, 256, (32, 32, 3), dtype=np.uint8)
        items = [SimpleNamespace(data=raw, label=2)]
        for method in ('fedlf', 'fedyoyo', 'fedrela'):
            dataset = TrainingViews(items, [19], method)
            batch = dataset[0]
            self.assertEqual(batch['img'].shape, (3, 224, 224))
            self.assertTrue(torch.isfinite(batch['img']).all())
            if method == 'fedyoyo':
                self.assertFalse(torch.equal(batch['img'], batch['img_strong']))
            restored = pickle.loads(pickle.dumps(dataset))
            self.assertIsNone(restored._transforms)
            self.assertEqual(restored[0]['label'], 2)

    def test_actual_dataloader_spawn_with_official_augmentations(self):
        raw = np.random.RandomState(5).randint(0, 256, (32, 32, 3), dtype=np.uint8)
        dataset = TrainingViews([SimpleNamespace(data=raw, label=0)]*2, [9, 17], 'fedyoyo')
        dataset[0]  # cached lambda-containing upstream transforms must not be pickled
        loader = DataLoader(dataset, batch_size=2, num_workers=1, multiprocessing_context='spawn', worker_init_fn=seed_worker)
        batch = next(iter(loader))
        self.assertEqual(batch['img_strong'].shape, (2, 3, 224, 224))
        self.assertEqual(batch['index'].tolist(), [9, 17])

    def test_resume_preserves_model_prior_labels_and_smoke_collection(self):
        for method in ('fedlf', 'fedyoyo', 'fedrela'):
            with self.subTest(method=method), tempfile.TemporaryDirectory() as tmp, \
                    patch.object(common, 'source_hashes', return_value={'fixture': '1'}), \
                    patch.object(longtail_runtime, 'source_hashes', return_value={'fixture': '1'}), \
                    patch('sys.stdout', new_callable=io.StringIO):
                root = Path(tmp); fake_reference(root / 'ref')
                torch.manual_seed(14)
                x, y = torch.randn(12, 3), torch.arange(12) % 4
                batches = {c: [dict(img=x, img_strong=x.flip(0), label=y, index=torch.arange(12)+c*100)] for c in range(30)}
                counts = np.ones((30, 100), dtype=np.int64)
                def build_model(*args):
                    torch.manual_seed(42)
                    return SmallVisual(100), None
                def data(*args):
                    return batches, [], counts, 'fixed-training-images'
                full = common.make_job(method, root/'ref', root, root/'full', 0, True, 'longtail')
                split = common.make_job(method, root/'ref', root, root/'split', 0, True, 'longtail')
                common.register(full); common.register(split)
                with patch.object(runtime, 'build_config', return_value='test'), \
                     patch.object(runtime, 'build_data', side_effect=data), \
                     patch.object(runtime, 'build_model', side_effect=build_model), \
                     patch.object(longtail_runtime, 'build_views', return_value=(batches, batches)), \
                     patch.object(runtime, 'evaluate', return_value=([50]*100, [100]*100)):
                    runtime.train(full, device_override='cpu')
                    runtime.train(split, stop_after=1, device_override='cpu')
                    runtime.train(split, resume=True, device_override='cpu')
                    # Resume after the late method event is already committed.
                    runtime.train(split, resume=True, device_override='cpu')
                a = torch.load(common.run_path(full)/'checkpoint_last.pt', weights_only=False)
                b = torch.load(common.run_path(split)/'checkpoint_last.pt', weights_only=False)
                for key, value in a['state'].items():
                    torch.testing.assert_close(value, b['state'][key], atol=0, rtol=0)
                if method == 'fedyoyo':
                    torch.testing.assert_close(a['method_state']['effective_previous'], b['method_state']['effective_previous'], atol=0, rtol=0)
                    self.assertGreater(sum(x['auxiliary_forward_images'] for x in a['costs']), 0)
                if method == 'fedrela':
                    self.assertEqual(a['method_state']['relabels'], b['method_state']['relabels'])
                    self.assertEqual(a['method_state']['relabeled_round'], 2)
                result, _, _ = collect.completed_result(root/'split', split)
                self.assertTrue(result['smoke'])
                self.assertEqual(collect.collect(root/'split'), [])
                packed = collect.pack(root/'split')
                self.assertTrue(packed.is_file())

    def test_missing_auxiliary_state_rejected(self):
        with self.assertRaises(ValueError):
            longtail_runtime.validate_state('fedyoyo', dict(schema='longtail_state_v1', method='fedyoyo'), 4, 100, 90)
        with self.assertRaises(ValueError):
            longtail_runtime.validate_state('fedrela', dict(schema='longtail_state_v1', method='fedrela', relabels={}), 92, 100, 90)


if __name__ == '__main__':
    unittest.main()
