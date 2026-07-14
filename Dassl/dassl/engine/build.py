from Dassl.dassl.utils import Registry, check_availability
from trainers.clip import CLIP
from trainers.promptfl import PromptFL, Baseline
from trainers.coop import CoOp
from trainers.cocoop import CoCoOp
from trainers.maple import MaPLe
from trainers.capt import CAPT
from trainers.fedclip import FedClip
from trainers.fedtef import FedTEF
from trainers.fedclip_tail_module import FedClipTailModule
from trainers.cliplora import ClipLora
from trainers.kgcoop import KgCoOp
from trainers.promptfl_general_only import PromptFLGeneralOnly

TRAINER_REGISTRY = Registry("TRAINER")
TRAINER_REGISTRY.register(CLIP)
TRAINER_REGISTRY.register(PromptFL)
TRAINER_REGISTRY.register(Baseline)
TRAINER_REGISTRY.register(CoOp)
TRAINER_REGISTRY.register(CoCoOp)
TRAINER_REGISTRY.register(MaPLe)
TRAINER_REGISTRY.register(CAPT)
TRAINER_REGISTRY.register(FedClip)
TRAINER_REGISTRY.register(FedTEF)
TRAINER_REGISTRY.register(FedClipTailModule)
TRAINER_REGISTRY.register(ClipLora)
TRAINER_REGISTRY.register(KgCoOp)
TRAINER_REGISTRY.register(PromptFLGeneralOnly)

def build_trainer(cfg):
    avai_trainers = TRAINER_REGISTRY.registered_names()
    # print("avai_trainers",avai_trainers)
    check_availability(cfg.TRAINER.NAME, avai_trainers)
    if cfg.VERBOSE:
        print("Loading trainer: {}".format(cfg.TRAINER.NAME))
    return TRAINER_REGISTRY.get(cfg.TRAINER.NAME)(cfg)
