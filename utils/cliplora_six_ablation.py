"""Feedback-only witnesses for B-only cells, without activating A retention."""


def with_six_ablation(base):
    class SixAblationRuntime(base):
        def initialize_feedback(self):
            super().initialize_feedback()
            from tools.client_aggregation.six_ablation import has_b
            if has_b(self.job) and self.bank is None:
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
            if self.job['arm'] in ('frozen', 'plain_a'):
                if state is not None:
                    raise ValueError('Frozen/plain A must not restore retention history')
                return None
            return super().history_from_state(state)

        def run(self, initial_state):
            if self.job['arm'] == 'method_a':
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
                if self.job['arm'] == 'plain_a' and rnd <= 90:
                    state = self.train_phase(state, rnd, 'A', True)
                require_finite(**{k: state[k] for k in self.keys})
                self.load_training_state(state)
                train_only(self.trainer.model, 'B')
                if self.b_transfer is not None:
                    self.b_transfer.observe_global(state, rnd, committed=True)
                self.publish(state, rnd, rnd)
                self.checkpoint(state, rnd)
    return SixAblationRuntime
