"""Change only when ordinary A and its unchanged margin correction are executed."""
import time

import torch

from tools.client_aggregation.a_schedule import history_stop, rounds_for
from utils.cliplora_a_refresh import append_rows, state_hash, train_only
from utils.cliplora_functional_feedback import snapshot
from utils.cliplora_method_a_margin import MethodAMarginRuntime
from utils.method_a_margin_math import SampleMarginHistory, VARIANT
from utils.sfra_math import require_finite


class ScheduledMarginRuntime(MethodAMarginRuntime):
    def history_from_state(self, state):
        end = history_stop(self.job)
        if not end:
            if state is not None:
                raise ValueError('Fully frozen A must not restore protection history')
            return None
        result = SampleMarginHistory.restore(state, self.margin_feedback.manifest)
        if result.last_committed_round != min(self.resume_payload['completed_round'], end):
            raise ValueError('Calendar history does not match checkpoint')
        return result

    def run(self, initial_state):
        active = set(rounds_for(self.job))
        end = history_stop(self.job)
        last = self.args.sfra_stop_after_round or 100
        if self.resume_payload:
            state, completed = self.restore()
            from tools.client_aggregation.protocol import read_csv, write_table
            audit = self.root/'a_schedule_audit.csv'
            if audit.exists():
                write_table(audit, [r for r in read_csv(audit) if int(r['round']) <= completed])
        else:
            self.load_training_state(initial_state)
            state, completed = snapshot(self.trainer.model), 0
            self.history = None
            if end:
                initial = self.observe_margin(state, 0, 'initial')
                self.history = SampleMarginHistory(initial['sample_margins'], self.margin_feedback.manifest)
            self.publish(state, 0, 0)
            self.checkpoint(state, 0)
        for rnd in range(completed+1, last+1):
            before_a = {k: state[k].clone() for k in self.a_keys}
            before_hash = state_hash(state, self.a_keys)
            state = self.train_phase(state, rnd, 'B')
            if any(not torch.equal(state[k], before_a[k]) for k in self.a_keys):
                raise ValueError('Ordinary or extra B changed frozen A')
            if rnd in active:
                state, row = self.refresh(state, rnd)
            else:
                row = dict(round=rnd, variant=VARIANT, correction_steps=0, submitted_step=0,
                    correction_started=False, nonzero_correction=False, skip_reason='A_frozen_schedule',
                    commit_rule=self.sfra_config['commit_rule'], history_observation_seconds=0.)
                if rnd <= end:
                    # Preserve the original five-calendar-round history; no A gradients or optimizer here.
                    started = time.perf_counter()
                    observed = self.observe_margin(state, rnd, 'history_only')
                    changed = self.history.commit(observed['sample_margins'], rnd)
                    row.update(history_observation_seconds=time.perf_counter()-started,
                               history_registered=int(changed.sum()), window_count=self.history.block_count)
                row['history_last_committed'] = self.history.last_committed_round if self.history is not None else 0
                if any(not torch.equal(state[k], before_a[k]) for k in self.a_keys):
                    raise ValueError('A changed on a frozen round')
            require_finite(**{k: state[k] for k in self.keys})
            self.summaries.append(row)
            self.load_training_state(state)
            train_only(self.trainer.model, 'B')
            self.publish(state, rnd, rnd)
            append_rows(self.root/'a_schedule_audit.csv', [dict(round=rnd, A_open=rnd in active,
                A_before_sha256=before_hash, A_after_sha256=state_hash(state, self.a_keys),
                history_last_committed=row['history_last_committed'])])
            self.checkpoint(state, rnd)
            print(f'A-SCHEDULE COMMITTED {rnd}/100: A_open={rnd in active}, correction_steps={row["correction_steps"]}', flush=True)
