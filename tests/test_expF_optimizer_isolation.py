import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

stub_tensorboard = types.ModuleType("torch.utils.tensorboard")
stub_tensorboard.SummaryWriter = lambda *args, **kwargs: None
sys.modules.setdefault("torch.utils.tensorboard", stub_tensorboard)

engine_pkg = sys.modules.get("Dassl.dassl.engine", types.ModuleType("Dassl.dassl.engine"))
engine_pkg.__path__ = [str(ROOT / "Dassl" / "dassl" / "engine")]
sys.modules["Dassl.dassl.engine"] = engine_pkg

stub_data = types.ModuleType("Dassl.dassl.data")
stub_data.DataManager = object
sys.modules.setdefault("Dassl.dassl.data", stub_data)

stub_utils = sys.modules.get("Dassl.dassl.utils", types.ModuleType("Dassl.dassl.utils"))
stub_utils.setup_logger = lambda *args, **kwargs: None
stub_utils.set_random_seed = lambda *args, **kwargs: None
stub_utils.Registry = lambda *args, **kwargs: None
stub_utils.MetricMeter = object
stub_utils.AverageMeter = object
stub_utils.tolist_if_not = lambda value: value if isinstance(value, list) else [value]
stub_utils.count_num_param = lambda *args, **kwargs: 0
stub_utils.load_checkpoint = lambda *args, **kwargs: {}
stub_utils.save_checkpoint = lambda *args, **kwargs: None
stub_utils.mkdir_if_missing = lambda *args, **kwargs: None
stub_utils.resume_from_checkpoint = lambda *args, **kwargs: 0
stub_utils.load_pretrained_weights = lambda *args, **kwargs: None
sys.modules["Dassl.dassl.utils"] = stub_utils

stub_modeling = types.ModuleType("Dassl.dassl.modeling")
stub_modeling.build_head = lambda *args, **kwargs: None
stub_modeling.build_backbone = lambda *args, **kwargs: None
sys.modules.setdefault("Dassl.dassl.modeling", stub_modeling)

stub_eval = types.ModuleType("Dassl.dassl.evaluation")
stub_eval.build_evaluator = lambda *args, **kwargs: None
sys.modules.setdefault("Dassl.dassl.evaluation", stub_eval)

stub_clip_pkg = types.ModuleType("clip")
stub_clip_pkg.__path__ = []
stub_clip_module = types.ModuleType("clip.clip")
stub_clip_module._MODELS = {}
stub_clip_module._download = lambda *args, **kwargs: None
stub_tokenizer = types.ModuleType("clip.simple_tokenizer")
stub_tokenizer.SimpleTokenizer = lambda *args, **kwargs: object()
stub_clip_pkg.clip = stub_clip_module
sys.modules.setdefault("clip", stub_clip_pkg)
sys.modules.setdefault("clip.clip", stub_clip_module)
sys.modules.setdefault("clip.simple_tokenizer", stub_tokenizer)

from Dassl.dassl.optim import build_lr_scheduler, build_optimizer
from trainers import promptfl as promptfl_module
from trainers.promptfl import PromptFL


def _optim_cfg(lr=0.05):
    return SimpleNamespace(
        NAME="sgd",
        LR=lr,
        WEIGHT_DECAY=0.0,
        MOMENTUM=0.9,
        SGD_DAMPNING=0.0,
        SGD_NESTEROV=False,
        RMSPROP_ALPHA=0.99,
        ADAM_BETA1=0.9,
        ADAM_BETA2=0.999,
        STAGED_LR=False,
        NEW_LAYERS=(),
        BASE_LR_MULT=0.1,
        LR_SCHEDULER="single_step",
        STEPSIZE=(-1,),
        GAMMA=0.1,
        MAX_EPOCH=3,
        WARMUP_EPOCH=0,
        WARMUP_RECOUNT=True,
        WARMUP_TYPE="linear",
        WARMUP_CONS_LR=1e-5,
        WARMUP_MIN_LR=1e-5,
    )


def _make_fake_promptfl(prec="fp32", lr=0.05):
    model = torch.nn.Linear(4, 2)
    cfg = SimpleNamespace(
        OPTIM=_optim_cfg(lr=lr),
        TRAINER=SimpleNamespace(PROMPTFL=SimpleNamespace(PREC=prec)),
    )
    optim = build_optimizer(model.parameters(), cfg.OPTIM)
    sched = build_lr_scheduler(optim, cfg.OPTIM)

    trainer = PromptFL.__new__(PromptFL)
    trainer.cfg = cfg
    trainer._models = {"prompt_learner": model}
    trainer._optims = {"prompt_learner": optim}
    trainer._scheds = {"prompt_learner": sched}
    trainer.optim = optim
    trainer.sched = sched
    trainer.scaler = object() if prec == "amp" else None
    trainer.register_model = lambda *args, **kwargs: pytest.fail("register_model must not be called")
    return trainer


def _populate_optimizer_state(trainer):
    trainer.optim.zero_grad()
    output = trainer._models["prompt_learner"](torch.ones(2, 4)).sum()
    output.backward()
    trainer.optim.step()


def test_promptfl_reset_rebuilds_optimizer_scheduler_and_clears_state():
    trainer = _make_fake_promptfl(lr=0.05)
    params_before = list(trainer._models["prompt_learner"].parameters())
    old_optim = trainer.optim
    old_sched = trainer.sched

    _populate_optimizer_state(trainer)
    assert len(trainer.optim.state) > 0
    weights_before = [p.detach().clone() for p in params_before]

    PromptFL.reset_optimizer_and_scheduler(trainer)

    assert trainer.optim is not old_optim
    assert trainer.sched is not old_sched
    assert trainer.optim is trainer._optims["prompt_learner"]
    assert trainer.sched is trainer._scheds["prompt_learner"]
    assert trainer.sched.optimizer is trainer.optim
    assert len(trainer.optim.state) == 0
    assert trainer.optim.param_groups[0]["lr"] == pytest.approx(0.05)
    assert all(param is before for param, before in zip(trainer._models["prompt_learner"].parameters(), params_before))
    for param, weight in zip(params_before, weights_before):
        assert torch.equal(param.detach(), weight)


def test_promptfl_reset_rebuilds_amp_scaler(monkeypatch):
    class FakeScaler:
        pass

    monkeypatch.setattr(promptfl_module, "GradScaler", FakeScaler)
    trainer = _make_fake_promptfl(prec="amp")
    old_scaler = trainer.scaler

    PromptFL.reset_optimizer_and_scheduler(trainer)

    assert isinstance(trainer.scaler, FakeScaler)
    assert trainer.scaler is not old_scaler


def test_promptfl_reset_keeps_non_amp_scaler_empty():
    trainer = _make_fake_promptfl(prec="fp32")

    PromptFL.reset_optimizer_and_scheduler(trainer)

    assert trainer.scaler is None


def test_two_fake_clients_each_start_from_empty_optimizer_state():
    trainer = _make_fake_promptfl()

    for _ in range(2):
        PromptFL.reset_optimizer_and_scheduler(trainer)
        assert len(trainer.optim.state) == 0
        _populate_optimizer_state(trainer)
        assert len(trainer.optim.state) > 0
