from trainers.promptfl import PromptFL


class PromptFLGeneralOnly(PromptFL):
    """PromptFL ablation that removes the class-aware prompt branch."""

    def __init__(self, cfg):
        cfg.defrost()
        cfg.TRAINER.PROMPTFL.CSC = False
        cfg.freeze()
        super().__init__(cfg)
