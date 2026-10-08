"""Per-worker runtime installation; importing this module changes no legacy class."""
from utils.cliplora_sfra import SFRARuntime
from utils.cliplora_b_routes import RouteTransfer, route_config


def runtime_class(spec):
    if spec['mode'] == 'replay':
        from utils.b_route_replay import replay_runtime_class
        return replay_runtime_class(spec)

    class RouteRuntime(SFRARuntime):
        def configure_experiment(self):
            settings = spec['settings']
            if (self.variant != 'full-cp' or self.strength != 10 or self.classification_strength != 1 or
                self.args.seed != settings['seed'] or self.args.split_seed != 42):
                raise ValueError('Route worker differs from frozen A/seed contract')
            self.sfra_config['b_route_experiment'] = spec
            if settings['arm'] != 'N':
                self.sfra_config['b_transfer'] = route_config(self.sfra_config['b_transfer'], settings)
                self.method += '_route_'+settings['arm']
            elif 'b_transfer' in self.sfra_config:
                raise ValueError('N must not enable any transfer')

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            from tools.sfra.b_routes import validate_profile
            validate_profile(self.sfra_config, spec)
            if spec['settings']['arm'] != 'N':
                self.b_transfer = RouteTransfer(self)
            print(f'B routes: {spec["settings"]["arm"]}; normalized effective gradient, two steps, rho=1.', flush=True)

    return RouteRuntime
