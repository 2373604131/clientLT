"""Fresh-process imports and actual CLIP adapter updates on CPU fixtures.

The checkpoints are small random CLIP models, not pretrained research models.
Trainer classes, checkpoint loading, LoRA injection, losses and backward are
real. No data download or CUDA is required; use server smoke for those paths.
"""
import importlib.util
from pathlib import Path
import subprocess
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
DEPENDENCIES = ('torchvision', 'yacs', 'tabulate', 'ftfy', 'gdown', 'timm',
                'tensorboard', 'sklearn', 'matplotlib')


def startup_worker(method):
    import gc
    import tempfile
    import types
    from unittest.mock import patch

    import torch

    sys.path.insert(0, str(ROOT))
    # Some developer environments install HuggingFace datasets. Select this
    # repository's namespace explicitly, without replacing any dataset code.
    package = types.ModuleType('datasets')
    package.__path__ = [str(ROOT / 'datasets')]
    sys.modules['datasets'] = package

    from tools.benchmarks import common, runtime
    from clip import clip

    torch.set_num_threads(2)
    assert 'Dassl.dassl.engine' not in sys.modules
    assert 'trainers.cliplora' not in sys.modules
    assert 'trainers.capt' not in sys.modules

    class DownloadReached(Exception):
        pass

    # Reach the real weight-loading boundary from an entirely fresh worker.
    # The old direct import raises ImportError before reaching this sentinel.
    simple = types.SimpleNamespace
    minimal = simple(MODEL=simple(BACKBONE=simple(NAME='ViT-B/16')),
                     TRAINER=simple(CLIPLORA=simple()))
    with patch.object(clip, '_download', side_effect=DownloadReached):
        try:
            runtime.build_model({'method': method}, minimal, {'classnames': ['cat']}, 'cpu')
        except DownloadReached:
            pass
        else:
            raise AssertionError('Model loading was not reached')

    from Dassl.dassl.engine import TRAINER_REGISTRY
    from trainers.capt import CAPT
    from trainers.cliplora import ClipLora, build_cliplora_model
    from clip.model import CLIP
    from utils.cliplora_a_refresh import state_hash
    from trainers.baselines.common import trainable_state
    from tests.test_benchmark_reference_config import REFERENCE_CONFIG
    import yaml

    assert TRAINER_REGISTRY.get('CAPT') is CAPT
    assert TRAINER_REGISTRY.get('ClipLora') is ClipLora
    print('Fresh-process trainer import passed:', method, flush=True)

    config = runtime.parse_reference_config(REFERENCE_CONFIG)
    config['INPUT']['SIZE'] = [32, 32]
    config['TRAINER']['COOP'].update(N_CTX=4, CLASS_TOKEN_POSITION='end')
    config['TRAINER']['CLIPLORA'].update(CTX_INIT='a photo of a', backbone='ViT-B/16')
    config['TRAINER']['PROMPTFL'].update(TEMPERATURE=.1, ALPHA=1., BETA=1.,
                                       GAMMA=.1, DELTA=1., MARGIN=.5)
    meta = dict(resolved_config=yaml.safe_dump(config),
                classnames=['cat', 'dog', 'bird', 'car'], seed=42,
                resolved_args={'cliplora_common_init_seed': 42})
    is_capt = method == 'capt'
    design = dict(trainer='CAPT' if is_capt else 'CoOp', vision_depth=0,
                  language_depth=0, vision_ctx=0, language_ctx=0)
    # CAPT's coupling layer uses the original 512->768 dimensions. LoRA uses
    # narrow blocks but keeps all twelve layers, including top3 q/v injection.
    torch.manual_seed(7)
    backbone = CLIP(embed_dim=512 if is_capt else 64, image_resolution=32,
                    vision_layers=1 if is_capt else 12,
                    vision_width=768 if is_capt else 64, vision_patch_size=16,
                    context_length=77, vocab_size=49408,
                    transformer_width=512 if is_capt else 64,
                    transformer_heads=8 if is_capt else 1,
                    transformer_layers=1, design_details=design)
    with tempfile.TemporaryDirectory() as tmp:
        checkpoint = Path(tmp) / 'random_clip.pt'
        torch.save(backbone.state_dict(), checkpoint)
        del backbone
        gc.collect()
        job = dict(method=method, data_root=tmp, output_root=tmp, workers=0,
                   config=common.read_json(common.CONFIG))
        cfg = runtime.build_config(job, meta)
        with patch.object(clip, '_download', return_value=str(checkpoint)):
            if not is_capt:
                reference = build_cliplora_model(cfg, meta['classnames'])
                state = reference.state_dict()
                keys = sorted(k for k in state if k.endswith(('_lora_A', '_lora_B')))
                meta['initial_lora_sha256'] = state_hash(state, keys)
                meta['frozen_model_sha256'] = state_hash(state, sorted(set(state) - set(keys)))
                del reference, state
                gc.collect()
            model, teacher = runtime.build_model(job, cfg, meta, torch.device('cpu'))

        before = trainable_state(model)
        assert before
        if is_capt:
            assert all(k.startswith(('prompt_learner.', 'coupling_function.')) for k in before)
        else:
            assert all(k.endswith(('_lora_A', '_lora_B')) for k in before)
            assert len(before) == 12  # top3 x q/v x two factors
        frozen = sorted(k for k, p in model.named_parameters() if not p.requires_grad)
        frozen_hash = state_hash(model.state_dict(), frozen)
        images = torch.randn(2, 3, 32, 32)
        labels = torch.tensor([0, 2])
        after, costs = runtime.local_train(model, teacher, [{'img': images, 'label': labels}],
                                           method, job['config'], torch.ones(4), 'cpu', smoke=True)
        assert costs['optimizer_steps'] == 1
        assert any(not torch.equal(value, before[k]) for k, value in after.items())
        assert state_hash(model.state_dict(), frozen) == frozen_hash
        if method in ('fedntd', 'fedpurel'):
            assert teacher is not None and not teacher.training
            assert all(not p.requires_grad and p.grad is None for p in teacher.parameters())
            for k, value in before.items():
                torch.testing.assert_close(teacher.state_dict()[k], value, rtol=0, atol=0)
        else:
            assert teacher is None
        model.eval()
        with torch.no_grad():
            logits = model(images)
        assert logits.shape == (2, 4) and torch.isfinite(logits).all()
        print('Actual model build, backward and evaluation passed:', method, flush=True)


@unittest.skipUnless(all(importlib.util.find_spec(x) for x in DEPENDENCIES),
                     'requires real CLIP/Dassl model dependencies')
class ModelStartupTests(unittest.TestCase):
    def run_worker(self, method):
        result = subprocess.run([sys.executable, str(Path(__file__).resolve()), '--worker', method],
                                cwd=ROOT, capture_output=True, text=True, encoding='utf-8',
                                errors='replace', timeout=180)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn('Actual model build, backward and evaluation passed: ' + method, result.stdout)

    def test_fedavg_lora_startup_and_update(self):
        self.run_worker('fedavg-lora')

    def test_capt_startup_and_update(self):
        self.run_worker('capt')

    def test_fedntd_startup_and_update(self):
        self.run_worker('fedntd')

    def test_fedpurel_startup_and_update(self):
        self.run_worker('fedpurel')


if __name__ == '__main__':
    if len(sys.argv) == 3 and sys.argv[1] == '--worker':
        startup_worker(sys.argv[2])
    else:
        unittest.main()
