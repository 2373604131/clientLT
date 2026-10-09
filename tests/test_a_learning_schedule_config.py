"""Regression for archived input types, including the real YACS constructor."""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import yaml

from tools.a_learning_schedule.config import parse_archived_config, validate_input_config


# Historical str(CfgNode) format, including tuples that are not YAML sequences.
ARCHIVED = """
INPUT:
  SIZE: (224, 224)
  TRANSFORMS: ('random_resized_crop', 'random_flip', 'normalize')
  RRCROP_SCALE: (0.08, 1.0)
  PIXEL_MEAN: [0.48145466, 0.4578275, 0.40821073]
  PIXEL_STD: [0.26862954, 0.26130258, 0.27577711]
DATASET:
  ROOT: DATA
  SOURCE_DOMAINS: ()
OUTPUT_DIR: output/original
DATALOADER:
  NUM_WORKERS: 8
MODEL:
  INIT_WEIGHTS:
  BACKBONE:
    NAME: ViT-B/16
OPTIM:
  NAME: sgd
  LR: 0.001
  MOMENTUM: 0.9
  WEIGHT_DECAY: 0.0005
  GAMMA: 1.0
  LR_SCHEDULER: single_step
  WARMUP_EPOCH: -1
  SGD_NESTEROV: False
  SGD_DAMPNING: 0
  MAX_EPOCH: 3
TRAINER:
  COOP:
    PREC: fp32
  CLIPLORA:
    r: 4
    alpha: 1
    encoder: vision
    position: top3
    params: ['q', 'v']
    dropout_rate: 0
    SCA_ENABLED: False
    CTX_INIT: a photo of a
"""


class ArchivedConfigTests(unittest.TestCase):
    def test_tuple_restore_covers_size_transforms_scale_and_empty_sequences(self):
        broken = yaml.safe_load(ARCHIVED)
        self.assertEqual(broken['INPUT']['SIZE'][0], '(')
        with self.assertRaisesRegex(ValueError, 'INPUT.SIZE'):
            validate_input_config(broken)
        restored = parse_archived_config(ARCHIVED)
        self.assertEqual(restored['INPUT']['SIZE'], (224, 224))
        self.assertEqual(restored['INPUT']['TRANSFORMS'], ('random_resized_crop', 'random_flip', 'normalize'))
        self.assertEqual(restored['INPUT']['RRCROP_SCALE'], (.08, 1.))
        self.assertEqual(restored['DATASET']['SOURCE_DOMAINS'], ())
        self.assertEqual(restored['INPUT']['PIXEL_MEAN'], broken['INPUT']['PIXEL_MEAN'])
        self.assertEqual(restored['OPTIM'], broken['OPTIM'])
        self.assertEqual(restored['TRAINER'], broken['TRAINER'])
        self.assertEqual(validate_input_config(restored)['size'], [224, 224])

    def test_yaml_sequences_prompts_and_nonliteral_strings_are_preserved(self):
        restored = parse_archived_config("""
size: [224, 224]
prompt: '(an image)'
model: ViT-B/16
expression: "(__import__('os').getcwd(),)"
""")
        self.assertEqual(restored['size'], [224, 224])
        self.assertEqual(restored['prompt'], '(an image)')
        self.assertEqual(restored['model'], 'ViT-B/16')
        self.assertEqual(restored['expression'], "(__import__('os').getcwd(),)")
        for text in ('[]', '', 'INPUT: ['):
            with self.assertRaises(ValueError):
                parse_archived_config(text)

    @unittest.skipUnless(importlib.util.find_spec('yacs'), 'requires the real yacs package')
    def test_real_yacs_build_config_reproduces_old_failure_and_restores_types(self):
        from yacs.config import CfgNode
        from tools.a_learning_schedule.runtime import build_config
        self.assertEqual(CfgNode.load_cfg(ARCHIVED).INPUT.SIZE[0], '(')
        source = {'meta': {'resolved_config': ARCHIVED}}
        with tempfile.TemporaryDirectory() as temp:
            args = SimpleNamespace(data_root=Path(temp) / 'data', output_root=Path(temp) / 'out', num_workers=0)
            cfg = build_config(source, args)
            self.assertEqual(cfg.INPUT.SIZE[0], 224)
            self.assertEqual(cfg.INPUT.SIZE, (224, 224))
            self.assertEqual(cfg.INPUT.TRANSFORMS, ('random_resized_crop', 'random_flip', 'normalize'))
            self.assertEqual(cfg.INPUT.RRCROP_SCALE, (.08, 1.))
            self.assertEqual(cfg.TRAINER.CLIPLORA.CTX_INIT, 'a photo of a')
            self.assertEqual(cfg.OPTIM.LR, .001)
            self.assertEqual(cfg.DATASET.ROOT, str(args.data_root.resolve()))
            self.assertTrue(cfg.is_frozen())

    @unittest.skipUnless(importlib.util.find_spec('yacs'), 'requires the real yacs package')
    def test_actual_archived_e2_e3_configs_build_with_original_input_values(self):
        from tools.a_learning_schedule.runtime import build_config
        archive = Path(__file__).resolve().parents[1] / 'output/la_control_js_topology_analysis/la_control/seed42/client-longtail'
        if not archive.is_dir():
            self.skipTest('Local reference archives unavailable')
        with tempfile.TemporaryDirectory() as temp:
            args = SimpleNamespace(data_root=Path(temp), output_root=Path(temp), num_workers=0)
            for origin in ('e2', 'e3'):
                with self.subTest(origin=origin):
                    meta = json.loads((archive / origin / 'tau1_a1_protocol42/bridge_metadata.json').read_text(encoding='utf-8'))
                    cfg = build_config({'meta': meta}, args)
                    self.assertEqual(cfg.INPUT.SIZE, (224, 224))
                    self.assertEqual(cfg.INPUT.TRANSFORMS, ('random_resized_crop', 'random_flip', 'normalize'))
                    self.assertEqual(cfg.INPUT.RRCROP_SCALE, (.08, 1.))
                    self.assertEqual(cfg.MODEL.BACKBONE.NAME, 'ViT-B/16')
                    self.assertEqual(cfg.TRAINER.CLIPLORA.CTX_INIT, 'a photo of a')

    def test_preflight_rejects_bad_transform_types_before_any_gpu_launch(self):
        from scripts.run_cliplora_a_learning_schedule import parse_args, preflight
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for name in ('train', 'test', 'meta'):
                (root / name).touch()
            args = parse_args(['--shard', '1', '--output-root', temp, '--data-root', temp])
            bad = ARCHIVED.replace("('random_resized_crop', 'random_flip', 'normalize')", 'random_flip')
            source = dict(path='/source', rows=[], meta={'schedule': [], 'resolved_config': bad})
            with patch('scripts.run_cliplora_a_learning_schedule.discover_source', return_value='/source'), \
                 patch('scripts.run_cliplora_a_learning_schedule.check_source', return_value=source), \
                 contextlib.redirect_stdout(io.StringIO()), self.assertRaisesRegex(ValueError, 'INPUT.TRANSFORMS'):
                preflight(args)


if __name__ == '__main__':
    unittest.main()
