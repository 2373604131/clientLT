"""CPU protocol, real objective integration, cache parity and resume audits."""
import copy
import io
import json
from pathlib import Path
import sys
import tarfile
import tempfile
from types import SimpleNamespace, ModuleType
import unittest
from unittest.mock import patch
from contextlib import redirect_stdout

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from tools.benchmarks.common import read_json, write_json, digest
from tools.multidataset.catalog import METHODS, BATCHES, DATASETS, read_dataset
from tools.multidataset.protocol import groups, make_pool, allocate, metrics
from tools.multidataset.runner import options
from tools.multidataset.external import train as external_train
from tools.multidataset.collect import audit_job, collect


ROOT = Path(__file__).resolve().parents[1]


class ProtocolTests(unittest.TestCase):
    def test_batch_inventory_and_fixed_recipe(self):
        self.assertEqual(len(METHODS), 13)
        self.assertEqual(len(BATCHES), 7)
        self.assertEqual(sum(dataset != 'cifar100-lt' for dataset, _ in BATCHES.values()), 4)
        self.assertEqual([v for n, v in BATCHES.values() if n == 'cifar100-lt'], [.5, .75, 1.])

    def test_actual_cifar_pool_conserved_across_specialization(self):
        from tools.benchmarks.common import read_csv
        source = read_csv(ROOT / 'references/full10_clientlt/partition_manifest.csv')
        pool = [dict(raw_sample_id=int(r['raw_sample_id']), class_id=int(r['class_id'])) for r in source]
        expected = sorted(r['raw_sample_id'] for r in pool)
        for value in (.5, .75, 1.):
            rows, matrix = allocate(pool, 100, value)
            self.assertEqual(sorted(r['raw_sample_id'] for r in rows), expected)
            self.assertEqual(int(matrix[27:, 80:].sum()), round(int(matrix[:, 80:].sum())*(.1+.9*value)))
            self.assertTrue(np.all(matrix.sum(1) > 0))
            if value == 1.:
                self.assertEqual(matrix[:27, 80:].sum(), 0)
                self.assertEqual(matrix[27:, :80].sum(), 0)
    @unittest.skipUnless((ROOT/'DATA/cifar-100/cifar-100-python/train').is_file(), 'Optional real CIFAR dataset is absent')
    def test_historical_partition_reproduces_from_real_cifar(self):
        from tools.benchmarks.common import read_csv
        source = read_csv(ROOT / 'references/full10_clientlt/partition_manifest.csv')
        from utils.functional_coverage_validation import _TrainOnlyCifar100, _locate_cifar100, _exact_lt_raw_ids
        store = _TrainOnlyCifar100(_locate_cifar100(ROOT/'DATA'))
        ordered_ids = _exact_lt_raw_ids(store.labels, .01, 'exp')
        ordered_pool = [dict(raw_sample_id=int(i), class_id=int(store.labels[i])) for i in ordered_ids]
        rows, _ = allocate(ordered_pool, 100, .75, ROOT/'references/full10_clientlt', True)
        self.assertEqual([(r['raw_sample_id'], r['client_id'], r['local_position']) for r in rows],
            [(int(r['raw_sample_id']), int(r['client_id']), int(r['local_position'])) for r in source])

    def test_new_small_datasets_keep_all_classes_and_nonempty_clients(self):
        for name, size in (('food101', 750), ('cub200', 29), ('dtd', 40)):
            classes = DATASETS[name]['classes']
            source = [dict(raw_sample_id=c*size+i, source_label=c) for c in range(classes) for i in range(size)]
            pool, order = make_pool(name, [str(c) for c in range(classes)], source, None)
            again, same = make_pool(name, [str(c) for c in range(classes)], source, None)
            self.assertEqual((pool, order), (again, same))
            rows, matrix = allocate(pool, classes, .75)
            self.assertEqual(len(rows), len(pool))
            self.assertTrue(np.all(matrix.sum(0) > 0))
            self.assertTrue(np.all(matrix.sum(1) > 0))

    def test_imagenet_keeps_every_official_training_image_and_remaps_text(self):
        source = [dict(raw_sample_id=c*10+i, source_label=c) for c, n in enumerate([5, 2, 7, 3, 4]) for i in range(n)]
        pool, order = make_pool('imagenet-lt', list('abcde'), source, None)
        self.assertEqual(order, [2, 0, 4, 3, 1])
        self.assertEqual(len(pool), len(source))
        self.assertTrue(all(order[r['class_id']] == r['source_label'] for r in pool))

    def test_group_rounding_and_unbalanced_test_metrics(self):
        for count in (47, 100, 101, 200, 1000):
            group = groups(count)
            self.assertEqual(sorted(sum(group.values(), [])), list(range(count)))
            self.assertEqual(len(group['tail_ids']), count-int(.8*count))
        value = metrics([100, 0, 0, 0, 0], [100, 1, 1, 1, 1], groups(5))
        self.assertAlmostEqual(value['overall_acc'], 10000/104)
        self.assertEqual(value['macro_acc'], 20.)

    def test_dtd_official_validation_is_not_training(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/'labels').mkdir()
            for split in ('train', 'val', 'test'):
                (root/'labels'/f'{split}1.txt').write_text('\n'.join(f'c{c:02d}/{split}.jpg' for c in range(47)))
            _, train, valid, test = read_dataset('dtd', root)
            self.assertEqual((len(train), len(valid), len(test)), (47, 47, 47))
            (root/'labels/test1.txt').write_text((root/'labels/train1.txt').read_text())
            with self.assertRaisesRegex(ValueError, 'overlap'):
                read_dataset('dtd', root)

    def test_food_and_cub_official_label_order(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            food = root/'food'
            (food/'meta').mkdir(parents=True)
            names = [f'food_{i:03d}' for i in range(101)]
            (food/'meta/classes.txt').write_text('\n'.join(names))
            for split in ('train', 'test'):
                (food/'meta'/f'{split}.txt').write_text('\n'.join(f'{name}/{split}' for name in names))
            actual, train, valid, test = read_dataset('food101', food)
            self.assertEqual(actual, names)
            self.assertEqual(len(valid), 0)
            self.assertEqual(train[-1]['source_label'], 100)
            cub = root/'cub'
            cub.mkdir()
            (cub/'classes.txt').write_text('\n'.join(f'{i} {i:03d}.Bird_{i}' for i in range(1, 201)))
            records = [(i*2-1, i, 1) for i in range(1, 201)] + [(i*2, i, 0) for i in range(1, 201)]
            (cub/'images.txt').write_text('\n'.join(f'{i} bird_{c}/{i}.jpg' for i, c, _ in records))
            (cub/'image_class_labels.txt').write_text('\n'.join(f'{i} {c}' for i, c, _ in records))
            (cub/'train_test_split.txt').write_text('\n'.join(f'{i} {s}' for i, _, s in records))
            names, train, _, test = read_dataset('cub200', cub)
            self.assertEqual(names[0], 'Bird_1')
            self.assertEqual(len(train), 200)
            self.assertEqual(len(test), 200)
            self.assertEqual(train[-1]['source_label'], 199)

    def test_tar_extractor_rejects_path_escape(self):
        from scripts.prepare_multidataset_data import safe_extract
        with tempfile.TemporaryDirectory() as directory:
            archive = Path(directory)/'bad.tar'
            with tarfile.open(archive, 'w') as stream:
                entry = tarfile.TarInfo('../outside.txt')
                entry.size = 3
                stream.addfile(entry, io.BytesIO(b'bad'))
            with self.assertRaises(ValueError):
                safe_extract(archive, Path(directory)/'data')
            self.assertFalse((Path(directory)/'outside.txt').exists())


class LoRAFeature(nn.Module):
    def __init__(self):
        super().__init__()
        self.w_lora_A = nn.Parameter(torch.randn(4, 5)*.1)
        self.w_lora_B = nn.Parameter(torch.zeros(6, 4))
        self.register_buffer('base', torch.randn(6, 5)*.1)

    def forward(self, image):
        return F.linear(image, self.base) + .5*F.linear(F.linear(image, self.w_lora_A), self.w_lora_B)


class SmallModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.image_encoder = LoRAFeature()
        self.classifier = nn.Linear(6, 7, bias=False).requires_grad_(False)

    def forward(self, images):
        return self.classifier(self.image_encoder(images))


class SmallCAPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.prompt_learner = nn.Module()
        self.prompt_learner.general_ctx = nn.Parameter(torch.randn(1, 6))
        self.prompt_learner.class_aware_ctx = nn.Parameter(torch.randn(7, 2, 6))
        self.image_encoder = nn.Linear(5, 6).requires_grad_(False)

    def forward(self, images, labels=None, return_features=False):
        features = self.image_encoder(images)
        logits = features @ self.prompt_learner.class_aware_ctx.mean(1).T
        return (features, logits, None) if return_features else logits


def external_fixture(root, method):
    protocol = dict(classes=7, groups=groups(7), matrix=[[2]*7 for _ in range(30)],
        schedule=[list(range(30)) for _ in range(100)], dataset='toy', specialization_lambda=.75)
    job = dict(batch='toy', protocol=protocol, method=method, seed=42, smoke=True, config=options(),
        run_root=str(root/method), protocol_root=str(root/'protocol'))
    Path(job['run_root']).mkdir(parents=True)
    test = [dict(class_id=c) for c in range(7) for _ in range(2)]
    write_json(root/'protocol/test.json', test)
    write_json(Path(job['run_root'])/'job.json', job)
    torch.manual_seed(42)
    model = SmallCAPT() if method == 'capt' else SmallModel()
    teacher = copy.deepcopy(model).requires_grad_(False) if method in ('fedntd', 'fedpurel') else None
    batch = dict(img=torch.randn(14, 5), label=torch.arange(14)%7, index=torch.arange(14), img_strong=torch.randn(14, 5))
    loaders = {c: [batch] for c in range(30)}
    return job, model, teacher, loaders, [batch]


class ExternalIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_all_twelve_real_baseline_objectives_run_on_seven_classes(self):
        for method in METHODS[:-1]:
            with self.subTest(method=method), tempfile.TemporaryDirectory() as directory:
                job, model, teacher, loaders, test = external_fixture(Path(directory), method)
                with patch('tools.multidataset.runner.verify_source'), redirect_stdout(io.StringIO()):
                    external_train(job, model, teacher, loaders, loaders, test)
                write_json(Path(job['run_root'])/'receipt.json', dict(job_id=digest(job), source_unchanged=True, exit_code=0))
                self.assertEqual(len(audit_job(job)), 3)

    def test_resume_replays_exact_final_state_for_stateful_methods(self):
        for method in ('fedavg-lora-la', 'fedyoyo', 'fedrela', 'lora-a2'):
            with self.subTest(method=method), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                whole = external_fixture(root/'whole', method)
                split = external_fixture(root/'split', method)
                with patch('tools.multidataset.runner.verify_source'), redirect_stdout(io.StringIO()):
                    for items, limit in ((whole, 0), (split, 1)):
                        job, model, teacher, loaders, test = items
                        external_train(job, model, teacher, loaders, loaders, test, stop_after=limit)
                    job, model, teacher, loaders, test = split
                    external_train(job, model, teacher, loaders, loaders, test, resume=True)
                full = torch.load(Path(whole[0]['run_root'])/'checkpoint_last.pt', weights_only=False)
                resumed = torch.load(Path(split[0]['run_root'])/'checkpoint_last.pt', weights_only=False)
                for key in full['state']:
                    torch.testing.assert_close(full['state'][key], resumed['state'][key], atol=0, rtol=0)

    def test_wrong_count_or_missing_receipt_never_enters_table(self):
        with tempfile.TemporaryDirectory() as directory:
            job, model, teacher, loaders, test = external_fixture(Path(directory), 'fedavg-lora')
            with patch('tools.multidataset.runner.verify_source'), redirect_stdout(io.StringIO()):
                external_train(job, model, teacher, loaders, loaders, test)
            with self.assertRaises(FileNotFoundError):
                audit_job(job)
            write_json(Path(job['run_root'])/'receipt.json', dict(job_id=digest(job), source_unchanged=True, exit_code=0))
            row = read_json(Path(job['run_root'])/'rounds/r002.json')
            row['total'][0] += 1
            write_json(Path(job['run_root'])/'rounds/r002.json', row)
            with self.assertRaisesRegex(ValueError, 'population'):
                audit_job(job)

    def test_fixed_last20_and_pending_rows_in_main_tables(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            job, _, _, _, _ = external_fixture(root, 'fedavg-lora')
            job.update(smoke=False, batch='cifar-lambda050')
            job['protocol']['dataset'] = 'cifar100-lt'
            run = Path(job['run_root'])
            write_json(run/'job.json', job)
            write_json(root/'cifar-lambda050/jobs/fedavg-lora.json', job)
            for rnd in range(101):
                correct, total = [2 if rnd <= 80 else 1]*7, [2]*7
                write_json(run/'rounds'/f'r{rnd:03d}.json', dict(round=rnd, job_id=digest(job),
                    correct=correct, total=total, metrics=metrics(correct, total, job['protocol']['groups'])))
            write_json(run/'completion.json', dict(job_id=digest(job), completed_round=100, smoke=False))
            write_json(run/'receipt.json', dict(job_id=digest(job), source_unchanged=True, exit_code=0))
            write_json(run/'progress.json', dict(completed_round=100))
            pending = copy.deepcopy(job)
            pending.update(method='fedntd', run_root=str(root/'pending'))
            write_json(root/'cifar-lambda050/jobs/fedntd.json', pending)
            status = collect(root)
            self.assertEqual({r['status'] for r in status}, {'pending', 'complete_verified'})
            from tools.benchmarks.common import read_csv
            rows = read_csv(root/'analysis/main_table_cifar_lambda.csv')
            self.assertEqual(len(rows), 1)
            self.assertEqual(float(rows[0]['cifar-lambda050_tail_acc']), 50.)
            self.assertEqual(rows[0]['cifar-lambda075_tail_acc'], '')


class CacheTests(unittest.TestCase):
    def test_bounded_cache_eviction_preserves_actual_feedback(self):
        from tests.test_sfra_execution import visual_bank
        from utils.sfra_resident_feedback import ResidentWitnessFeedback
        from tools.multidataset.feedback import BoundedWitnessFeedback
        torch.set_num_threads(1)
        first, second = visual_bank(), visual_bank()
        first._fast_feedback = ResidentWitnessFeedback(first, 8, 0.)
        second._fast_feedback = BoundedWitnessFeedback(second, 8, 0., cpu_cache_gib=.000001, image_cache_gib=.000001)
        for bank in (first, second):
            bank._fast_feedback.prepare()
        a, b = first.evaluate(classification=True), second.evaluate(classification=True)
        for key in ('scores', 'classification_scores', 'classification_gradient'):
            torch.testing.assert_close(a[key], b[key], atol=0, rtol=0)
        self.assertLessEqual(second._fast_feedback.cpu_cache_bytes, second._fast_feedback.cpu_limit)
        self.assertLessEqual(second._fast_feedback.image_bytes, second._fast_feedback.image_limit)


class Prompt(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('text', F.normalize(torch.randn(7, 5), dim=-1))

    def forward(self):
        return self.text


class Text(nn.Module):
    def forward(self, prompts, tokenized):
        return prompts


class SmallCLIP(nn.Module):
    def __init__(self):
        super().__init__()
        from tests.test_sfra_execution import visual_bank
        bank = visual_bank()
        self.image_encoder = bank.core.image_encoder
        self.prompt_learner, self.text_encoder = Prompt(), Text()
        self.register_buffer('tokenized_prompts', torch.zeros(7, 1, dtype=torch.long))
        self.logit_scale = nn.Parameter(torch.tensor(1.4), requires_grad=False)
        self.dtype = torch.float32

    def forward(self, image):
        features = F.normalize(self.image_encoder(image), dim=-1)
        return self.logit_scale.exp() * features @ self.prompt_learner().T


def old_ab_fixture(root):
    from tools.multidataset.ours import FixedOldAB
    torch.manual_seed(19)
    model = SmallCLIP()
    # Six clients deliberately omit the tail; two tail holders supply feedback.
    labels = [[0, 1, 2, 3, 4], [0, 1, 2, 3, 4], [5, 6], [5, 6]]
    clients, matrix, rows, loaders = [], [], [], {}
    for client, local_labels in enumerate(labels):
        source = []
        for position, label in enumerate(local_labels):
            source.append(SimpleNamespace(tensor=torch.randn(3, 8, 8), label=label, raw_sample_id=client*10+position))
            rows.append(dict(raw_sample_id=client*10+position, class_id=label, client_id=client, local_position=position))
        clients.append(source)
        matrix.append(np.bincount(local_labels, minlength=7).tolist())
        loaders[client] = [dict(img=torch.stack([v.tensor for v in source]), label=torch.tensor(local_labels))]
    protocol = dict(classes=7, groups=groups(7), matrix=matrix,
        schedule=[list(range(4)) for _ in range(100)], classnames=[str(c) for c in range(7)],
        joint_solution=dict(weights=[.2, .2, .3, .3]))
    job = dict(method='joint-old-ab', protocol=protocol, run_root=str(root/'run'),
        protocol_root=str(root/'protocol'), seed=42, smoke=True)
    write_json(root/'protocol/train.json', rows)
    write_json(root/'run/job.json', job)
    test = [dict(img=torch.randn(7, 3, 8, 8), label=torch.arange(7))]
    write_json(root/'protocol/test.json', [dict(class_id=c) for c in range(7)])
    trainer = SimpleNamespace(model=model, device=torch.device('cpu'),
        dm=SimpleNamespace(dataset=SimpleNamespace(classnames=protocol['classnames'], federated_train_x=clients)),
        fed_train_loader_x_dict=loaders, parse_batch_train=lambda batch: (batch['img'], batch['label']))
    cfg = SimpleNamespace(OPTIM=SimpleNamespace(LR=.001, WEIGHT_DECAY=.0005))
    return job, trainer, cfg, test


class OldABIntegrationTests(unittest.TestCase):
    def test_actual_old_a_and_b_runtime_and_resume_on_new_class_count(self):
        from tools.multidataset.ours import FixedOldAB
        from tools.multidataset.data import ImageDataset
        torch.set_num_threads(1)
        manager, transforms, cliplora = [ModuleType(name) for name in ('manager', 'transforms', 'cliplora')]
        manager.DatasetCifar100 = object
        transforms.build_transform = lambda *a, **k: None
        def optimizer(model, cfg):
            return torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=.001, momentum=.9, weight_decay=.0005), None
        def step(model, opt, scaler, precision, images, labels, logit_adjustment):
            opt.zero_grad()
            loss = F.cross_entropy(model(images)+logit_adjustment, labels)
            loss.backward()
            opt.step()
        cliplora.build_cliplora_optimizer_and_scheduler, cliplora.cliplora_optimizer_step = optimizer, step
        modules = {'Dassl.dassl.data.data_manager': manager, 'Dassl.dassl.data.transforms': transforms,
                   'trainers.cliplora': cliplora}
        def getitem(dataset, index):
            item = dataset.data_source[index]
            return dict(img=item.tensor, label=item.label, index=item.raw_sample_id)
        with tempfile.TemporaryDirectory() as directory, patch.dict(sys.modules, modules), \
             patch.object(ImageDataset, '__getitem__', getitem), patch('tools.multidataset.runner.verify_source'), \
             redirect_stdout(io.StringIO()):
            root = Path(directory)
            a = old_ab_fixture(root/'whole')
            b = old_ab_fixture(root/'split')
            whole = FixedOldAB(*a, resume=False, stop_after=0)
            whole.run(whole.base_state)
            # Smoke remains two rounds, so force the first run to pause at round 1.
            split = FixedOldAB(*b, resume=False, stop_after=0)
            split.args.sfra_stop_after_round = 1
            split.run(split.base_state)
            resumed = FixedOldAB(*b, resume=True, stop_after=0)
            resumed.run(resumed.base_state)
            left = torch.load(Path(a[0]['run_root'])/'checkpoints/sfra_last.pt', weights_only=False)
            right = torch.load(Path(b[0]['run_root'])/'checkpoints/sfra_last.pt', weights_only=False)
            self.assertEqual(left['completed_round'], 2)
            self.assertEqual(right['completed_round'], 2)
            for key in left['state_dict_overrides']:
                torch.testing.assert_close(left['state_dict_overrides'][key], right['state_dict_overrides'][key], atol=0, rtol=0)
            self.assertEqual(len(whole.summaries), 2)
            self.assertEqual(len(whole.b_transfer.records), 1)
            for job in (a[0], b[0]):
                write_json(Path(job['run_root'])/'receipt.json', dict(job_id=digest(job), source_unchanged=True, exit_code=0))
                self.assertEqual(len(audit_job(job)), 3)


if __name__ == '__main__':
    unittest.main()
