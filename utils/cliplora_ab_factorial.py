"""Opt-in joint-aggregation integration of the existing F-tail-equal optimizer."""
from types import SimpleNamespace

from utils.cliplora_b_routes import route_config
from utils.cliplora_b_shared_transfer import shared_transfer_config


def transfer_config(mode):
    args = SimpleNamespace(b_transfer_lr=.3, b_transfer_probe_step=.1, b_transfer_reg=.001,
        b_transfer_non_tail_sampling='class-cyclic', b_transfer_tail_weight=.35)
    value = route_config(shared_transfer_config(args), dict(arm='F-tail-equal', rho=1.))
    value.update(aggregation='ordinary_joint_class_weighted_B_then_one_shared_residual',
        calibration_anchor='actual ordinary joint-class-weighted B with round-start A')
    if mode == 'smoke':
        value['rounds'] = [1, 6]  # Disposable: exercises transfer before/after history registration.
    return value


def with_tail_equal(base):
    class TailEqualRuntime(base):
        def initialize_feedback(self):
            super().initialize_feedback()
            if self.bank is None:
                # Ordinary A needs B's private batches, but no A functional history.
                from utils.sfra_feedback import WitnessBank
                from tools.client_aggregation.protocol import read_csv
                from scripts.run_ab_validation import write_json
                tokens = self.resume_payload['witness_tokens'] if self.resume_payload else None
                self.bank = WitnessBank(self.trainer, self.cfg, self.a_keys, self.args.seed,
                    self.args.sfra_witness_batch_size, tokens=tokens)
                identity = {(int(r['client_id']), int(r['local_position'])): int(r['raw_sample_id'])
                    for r in read_csv(self.root/'partition_manifest.csv')}
                for token in self.bank.tokens:
                    token['raw_sample_ids'] = [identity[token['client_id'], p] for p in token['local_positions']]
                write_json(self.root/'private_witness_manifest.json', self.bank.tokens)

        def history_from_state(self, state):
            if self.job['arm'] == 'plain_a':
                if state is not None:
                    raise ValueError('Ordinary A must not restore an A protection history')
                return None
            return super().history_from_state(state)

        def run(self, initial_state):
            if self.job['arm'] != 'plain_a':
                return super().run(initial_state)
            from utils.cliplora_functional_feedback import snapshot
            from utils.cliplora_a_refresh import train_only
            from utils.sfra_math import require_finite
            last = self.args.sfra_stop_after_round or 100
            if self.resume_payload:
                state, completed = self.restore()
            else:
                self.load_training_state(initial_state)
                state, completed = snapshot(self.trainer.model), 0
                self.publish(state, 0, 0)
                self.checkpoint(state, 0)
            for rnd in range(completed+1, last+1):
                state = self.train_phase(state, rnd, 'B')
                if rnd <= 90:
                    state = self.train_phase(state, rnd, 'A', True)
                require_finite(**{k: state[k] for k in self.keys})
                self.load_training_state(state)
                train_only(self.trainer.model, 'B')
                self.publish(state, rnd, rnd)
                self.checkpoint(state, rnd)
    return TailEqualRuntime
