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
engine_pkg.build_trainer = lambda *args, **kwargs: None
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

stub_dassl_config = types.ModuleType("Dassl.dassl.config")
stub_dassl_config.get_cfg_default = lambda: None
sys.modules.setdefault("Dassl.dassl.config", stub_dassl_config)

stub_fed_utils = types.ModuleType("utils.fed_utils")
stub_fed_utils.average_weights = lambda *args, **kwargs: None
sys.modules.setdefault("utils.fed_utils", stub_fed_utils)

stub_prompt_loss = types.ModuleType("loss.prompt_loss")
stub_prompt_loss.PromptLoss = object
stub_prompt_loss.update_class_priors = lambda *args, **kwargs: None
sys.modules.setdefault("loss.prompt_loss", stub_prompt_loss)

stub_capt = types.ModuleType("trainers.capt")
stub_capt.MABScheduler = object
sys.modules.setdefault("trainers.capt", stub_capt)

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
from Dassl.dassl.engine.trainer import TrainerBase
from federated_main import (
    run_promptfl_local_train_with_scheduler_policy,
    validate_scheduler_step_delta,
)
from trainers.promptfl import PromptFL


class CountingScheduler:
    def __init__(self, owner=None):
        self.owner = owner
        self.last_epoch = 0
        self.steps = []

    def step(self):
        self.last_epoch += 1
        self.steps.append(getattr(self.owner, "phase", "epoch_start"))


class DummyTrainer(TrainerBase):
    def __init__(self, max_epoch=1, fail=False):
        super().__init__()
        self.start_epoch = 0
        self.max_epoch = max_epoch
        self.fail = fail
        self.phase = "epoch_start"
        self.model = torch.nn.Linear(1, 1)
        self.sched = CountingScheduler(self)
        self.register_model("model", self.model, None, self.sched)

    def before_train(self, is_fed=False):
        pass

    def before_epoch(self):
        self.phase = "epoch_body"

    def run_epoch(self, idx=-1, global_epoch=-1, global_weight=None, fedprox=False, mu=0.5):
        if self.fail:
            raise RuntimeError("synthetic failure")
        self.phase = "epoch_end"
        self.update_lr()
        self.phase = "epoch_start"

    def after_epoch(self):
        pass

    def after_train(self, idx=-1, global_epoch=-1, is_fed=False):
        pass


class FederatedDummyTrainer(DummyTrainer):
    def train(self, idx=-1, global_epoch=0, is_fed=False, global_weight=None, fedprox=False, mu=0.5):
        super().train(
            self.start_epoch,
            self.max_epoch,
            idx=idx,
            global_epoch=global_epoch,
            is_fed=is_fed,
            global_weight=global_weight,
            fedprox=fedprox,
            mu=mu,
        )


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


def _make_fake_promptfl():
    model = torch.nn.Linear(4, 2)
    cfg = SimpleNamespace(
        OPTIM=_optim_cfg(),
        TRAINER=SimpleNamespace(PROMPTFL=SimpleNamespace(PREC="fp32")),
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
    trainer.scaler = None
    return trainer


def test_default_scheduler_policy_keeps_two_steps_for_one_epoch():
    trainer = DummyTrainer(max_epoch=1)

    trainer.train(0, 1, is_fed=True)

    assert trainer.sched.steps == ["epoch_start", "epoch_end"]
    assert trainer.sched.last_epoch == 2


def test_single_step_policy_uses_epoch_end_only_for_one_and_five_epochs():
    one_epoch = DummyTrainer(max_epoch=1)
    one_epoch._skip_scheduler_step_at_epoch_start = True
    one_epoch.train(0, 1, is_fed=True)

    assert one_epoch.sched.steps == ["epoch_end"]
    assert one_epoch.sched.last_epoch == 1

    five_epochs = DummyTrainer(max_epoch=5)
    five_epochs._skip_scheduler_step_at_epoch_start = True
    five_epochs.train(0, 5, is_fed=True)

    assert five_epochs.sched.steps == ["epoch_end"] * 5
    assert five_epochs.sched.last_epoch == 5


def test_scheduler_skip_attribute_restored_after_success_and_failure():
    args = SimpleNamespace(federated_single_scheduler_step=True)

    trainer = FederatedDummyTrainer(max_epoch=1)
    trainer._skip_scheduler_step_at_epoch_start = "previous"
    run_promptfl_local_train_with_scheduler_policy(trainer, idx=0, epoch=0, args=args, local_epochs=1)
    assert trainer._skip_scheduler_step_at_epoch_start == "previous"

    failing = FederatedDummyTrainer(max_epoch=1, fail=True)
    failing._skip_scheduler_step_at_epoch_start = "previous"
    with pytest.raises(RuntimeError, match="synthetic failure"):
        run_promptfl_local_train_with_scheduler_policy(failing, idx=0, epoch=0, args=args, local_epochs=1)
    assert failing._skip_scheduler_step_at_epoch_start == "previous"


def test_scheduler_delta_assertion_and_single_train_call():
    assert validate_scheduler_step_delta(3, 5, 10, 15) == 5

    with pytest.raises(RuntimeError, match="client 3.*local_epochs=5.*observed_delta=4"):
        validate_scheduler_step_delta(3, 5, 10, 14)

    class OneCallTrainer:
        def __init__(self):
            self.calls = 0
            self.sched = SimpleNamespace(last_epoch=0)

        def train(self, idx=-1, global_epoch=0, is_fed=False):
            self.calls += 1
            self.sched.last_epoch += 5

    trainer = OneCallTrainer()
    args = SimpleNamespace(federated_single_scheduler_step=True)
    run_promptfl_local_train_with_scheduler_policy(trainer, idx=2, epoch=0, args=args, local_epochs=5)

    assert trainer.calls == 1
    assert trainer._skip_scheduler_step_at_epoch_start is False


def test_other_trainers_are_unaffected_by_default_guard():
    trainer = DummyTrainer(max_epoch=1)
    assert not hasattr(trainer, "_skip_scheduler_step_at_epoch_start")

    trainer.train(0, 1, is_fed=True)

    assert trainer.sched.steps == ["epoch_start", "epoch_end"]


def test_optimizer_reset_binds_scheduler_to_new_optimizer():
    trainer = _make_fake_promptfl()
    old_optimizer = trainer.optim

    PromptFL.reset_optimizer_and_scheduler(trainer)

    assert trainer.optim is not old_optimizer
    assert trainer.sched.optimizer is trainer.optim
