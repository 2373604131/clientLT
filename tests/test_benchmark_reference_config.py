"""Regression for str(CfgNode) tuple fields and the real CIFAR loader path."""
import hashlib
from contextlib import contextmanager
import importlib.util
import io
from pathlib import Path
import pickle
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np
import torch

from tools.benchmarks import common, runtime
from tests.test_paper_benchmarks import fake_reference


@contextmanager
def repository_dataset_imports():
    package = types.ModuleType('datasets')
    package.__path__ = [str(common.REPO / 'datasets')]
    replacements = {'datasets': package, 'gdown': types.ModuleType('gdown')}
    missing = object()
    previous = {key: sys.modules.get(key, missing) for key in replacements}
    sys.modules.update(replacements)
    try:
        yield
    finally:
        # Do not restore the entire sys.modules dictionary: newly imported
        # torch/torchvision extensions must stay loaded with their C++ registry.
        for key, value in previous.items():
            if value is missing:
                sys.modules.pop(key, None)
            else:
                sys.modules[key] = value


# Shape of the historical metadata: str(CfgNode), not cfg.dump().
REFERENCE_CONFIG = """
SEED: 42
USE_CUDA: True
OUTPUT_DIR: output/reference
MODEL:
  INIT_WEIGHTS: ''
  BACKBONE:
    NAME: ViT-B/16
DATASET:
  ROOT: DATA
  NAME: Cifar100_LT
DATALOADER:
  NUM_WORKERS: 8
  K_TRANSFORMS: 1
  RETURN_IMG0: False
  TRAIN_X:
    BATCH_SIZE: 32
  TEST:
    BATCH_SIZE: 64
INPUT:
  SIZE: (224, 224)
  TRANSFORMS: ('random_resized_crop', 'random_flip', 'normalize')
  INTERPOLATION: bicubic
  PIXEL_MEAN: [0.48145466, 0.4578275, 0.40821073]
  PIXEL_STD: [0.26862954, 0.26130258, 0.27577711]
  NO_TRANSFORM: False
  RRCROP_SCALE: (0.08, 1.0)
TRAINER:
  NAME: ClipLora
  COOP:
    PREC: fp32
  PROMPTFL:
    PREC: fp32
  CLIPLORA:
    FREEZE_A: True
    COMMON_INIT_SEED: 42
    r: 4
    alpha: 1
    position: top3
    encoder: vision
    params: ['q', 'v']
    dropout_rate: 0
    SCA_ENABLED: False
"""


class ReferenceConfigTests(unittest.TestCase):
    def test_historical_tuple_strings_are_restored(self):
        import yaml
        broken = yaml.safe_load(REFERENCE_CONFIG)
        self.assertIsInstance(broken['INPUT']['TRANSFORMS'], str)
        self.assertIsInstance(broken['INPUT']['SIZE'], str)
        cfg = runtime.parse_reference_config(REFERENCE_CONFIG)
        self.assertEqual(cfg['INPUT']['SIZE'], (224, 224))
        self.assertEqual(cfg['INPUT']['TRANSFORMS'], ('random_resized_crop', 'random_flip', 'normalize'))
        self.assertEqual(cfg['INPUT']['RRCROP_SCALE'], (.08, 1.))

    def test_plain_strings_yaml_lists_and_empty_tuples(self):
        cfg = runtime.parse_reference_config("a: ()\nb: [224, 224]\nc: '(an image)'\nd: ViT-B/16\ne: \"(__import__('os').getcwd(),)\"\n")
        self.assertEqual(cfg['a'], ())
        self.assertEqual(cfg['b'], [224, 224])
        self.assertEqual(cfg['c'], '(an image)')
        self.assertEqual(cfg['d'], 'ViT-B/16')
        self.assertEqual(cfg['e'], "(__import__('os').getcwd(),)")
        with self.assertRaises(ValueError):
            runtime.parse_reference_config('[]')

    @unittest.skipUnless(all(importlib.util.find_spec(x) for x in ('torchvision', 'yacs', 'tabulate', 'ftfy')),
                         'requires real torchvision/yacs data-loader dependencies')
    def test_real_config_transform_and_loader_match_original_cifar_pixels(self):
        # Isolate this repository's namespace from the unrelated HuggingFace
        # datasets package. gdown is download-only and unused with local pickles.
        with repository_dataset_imports(), \
                tempfile.TemporaryDirectory() as tmp, patch('sys.stdout', new_callable=io.StringIO):
            from yacs.config import CfgNode
            from torchvision import transforms as T
            from Dassl.dassl.data.transforms import build_transform
            from Dassl.dassl.data.data_manager import DatasetCifar100
            root = Path(tmp); fake_reference(root / 'ref')
            # The old loader path must fail on this exact fixture.
            import yaml
            with self.assertRaises(AssertionError):
                build_transform(CfgNode(yaml.safe_load(REFERENCE_CONFIG)), True)
            images = np.random.default_rng(5).integers(0, 256, (300, 3, 32, 32), dtype=np.uint8)
            # Balanced official-test-shaped fixture, not real research results.
            test_images = np.repeat(images[:100], 100, axis=0)
            labels = np.arange(100).repeat(100).tolist()
            data = root / 'data/cifar-100/cifar-100-python'; data.mkdir(parents=True)
            for name, value in (
                ('train', {'data': images.reshape(300, -1), 'fine_labels': (np.arange(300) % 100).tolist()}),
                ('test', {'data': test_images.reshape(10000, -1), 'fine_labels': labels}),
                ('meta', {})):
                with (data / name).open('wb') as stream:
                    pickle.dump(value, stream)
            meta = common.read_json(root / 'ref/bridge_metadata.json')
            digest = hashlib.sha256()
            for image, label in zip(test_images.transpose(0, 2, 3, 1), labels):
                digest.update(str(label).encode()); digest.update(image.tobytes())
            meta.update(resolved_config=REFERENCE_CONFIG, test_sha256=digest.hexdigest(), client_sample_counts=[10] * 30)
            job = dict(method='capt', data_root=str(root / 'data'), reference_run=str(root / 'ref'),
                       output_root=str(root / 'out'), workers=0, config=common.read_json(common.CONFIG))
            original = T.Compose([T.ToTensor(), T.Normalize((.5071, .4865, .4409), (.2673, .2564, .2762)), T.Resize([224, 224])])
            for method in ('capt', 'fedavg-lora'):
                job['method'] = method
                cfg = runtime.build_config(job, meta)
                self.assertTrue(cfg.is_frozen())
                self.assertEqual(cfg.INPUT.SIZE, (224, 224))
                loaders, test, counts, _ = runtime.build_data(job, cfg, meta)
                self.assertIsInstance(loaders[0].dataset, DatasetCifar100)
                self.assertEqual(counts.sum(), 300)
                sample = loaders[0].dataset[0]
                torch.testing.assert_close(sample['img'], original(images[0].transpose(1, 2, 0)), atol=0, rtol=0)
                batch = next(iter(loaders[0]))
                self.assertEqual(tuple(batch['img'].shape), (10, 3, 224, 224))
                self.assertEqual(batch['img'].dtype, torch.float32)
                self.assertTrue(bool(torch.isfinite(batch['img']).all()))
                torch.testing.assert_close(test.dataset[0]['img'], sample['img'], atol=0, rtol=0)


if __name__ == '__main__':
    unittest.main()
