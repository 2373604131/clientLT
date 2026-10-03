"""Runtime extension using the existing pre-resume configuration hook."""
from utils.b_class_guard_math import guard_config
from utils.cliplora_b_class_guard import GuardedSharedDonorBTransfer
from utils.cliplora_sfra import SFRARuntime


def runtime_class(spec):
    class GuardRuntime(SFRARuntime):
        def configure_experiment(self):
            if self.variant != 'full-cp' or self.strength != 10 or self.classification_strength != 1:
                raise ValueError('B guard freezes Method A at Full-CP lambda10 mu1')
            if self.args.seed != spec['settings']['seed'] or self.args.split_seed != spec['settings']['protocol_seed']:
                raise ValueError('Worker seed differs from launcher contract')
            self.sfra_config['b_transfer'] = guard_config(self.sfra_config['b_transfer'],spec['settings']['guard'])
            self.sfra_config['b_class_guard_experiment'] = spec
            self.method += '_guard_'+spec['settings']['guard']

        def __init__(self,*args,**kwargs):
            super().__init__(*args,**kwargs)
            # Base construction initializes witnesses and validates the full
            # checkpoint contract first. Neither transfer constructor trains.
            self.b_transfer = GuardedSharedDonorBTransfer(self)
            print(f'B class guard: {spec["settings"]["guard"]}; same shared C; fixed two-step commit.',flush=True)

    return GuardRuntime
