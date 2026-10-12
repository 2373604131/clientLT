"""Restore frozen bases, historical anchors, and paired local A proposals."""
from pathlib import Path

import numpy as np
import torch

from tools.a_projection_diagnosis.math import aggregate
from tools.a_projection_diagnosis.protocol import StateIndex, read_json
from utils.cliplora_a_refresh import state_hash


def load(path):
    # Inputs are this project's trusted experiment artifacts, never downloaded pickles.
    return torch.load(path, map_location='cpu', weights_only=False)


def recover_pre_a(previous, current):
    if set(previous) != set(current):
        raise ValueError('Committed LoRA keys differ')
    return {k: (previous[k] if k.endswith('_lora_A') else current[k]).clone() for k in current}


class StateStore:
    def __init__(self, source):
        self.source, self.index = source, StateIndex(source)
        self.base = load(self.index.root / 'checkpoints/base_model.pt')
        self.keys = sorted(k for k in self.base if k.endswith(('_lora_A', '_lora_B')))
        self.a_keys = [k for k in self.keys if k.endswith('_lora_A')]
        if len(self.a_keys) != 6 or len(self.keys) != 12:
            raise ValueError('Expected six A/B module pairs')
        self.frozen = sorted(set(self.base) - set(self.keys))
        if state_hash(self.base, self.frozen) != source['meta']['frozen_model_sha256']:
            raise ValueError('Frozen base differs from source metadata')
        if state_hash(self.base, self.keys) != source['meta']['initial_lora_sha256']:
            raise ValueError('Base is not the common initialization')
        self.initial = self.validate(self.base)

    def validate(self, state):
        if not all(k in state for k in self.keys):
            raise ValueError('Checkpoint lacks complete A+B; merge overrides with frozen base first')
        for k, value in state.items():
            if k not in self.base:
                raise ValueError('Unknown model key: ' + k)
            reference = self.base[k]
            if value.shape != reference.shape or value.dtype != reference.dtype or not torch.isfinite(value).all():
                raise ValueError('Checkpoint shape/dtype/finiteness differs: ' + k)
            if k not in self.keys and not torch.equal(value, reference):
                raise ValueError('Frozen tensor changed; pre-A recovery would be invalid: ' + k)
        return {k: state[k].clone() for k in self.keys}

    def read_spec(self, spec, allow_a_only=False):
        payload = load(spec['path'])
        if 'key' in spec:
            state = payload[spec['key']]
        elif 'state_dict_overrides' in payload:
            state = dict(self.base, **payload['state_dict_overrides'])
            if payload.get('full_state_sha256') and state_hash(state, sorted(state)) != payload['full_state_sha256']:
                raise ValueError('Compressed checkpoint hash mismatch')
        else:
            state = payload.get('model_state_dict', payload.get('state_dict', payload))
        if allow_a_only and set(state) == set(self.a_keys):
            # Only previous A is needed in (A_{t-1}, B_t). Its placeholder B is
            # overwritten by the current committed B before any evaluation.
            state = dict(self.initial, **state)
        state = self.validate(state)
        sidecar = Path(spec['path']).with_name('event.json')
        if sidecar.is_file() and spec.get('key') in ('anchor_lora_state', 'actual_after_lora_state'):
            expected = read_json(sidecar)['before_lora_sha256' if spec['key'] == 'anchor_lora_state' else 'after_lora_sha256']
            if state_hash(state, self.keys) != expected:
                raise ValueError('Event state hash mismatch')
        return state

    def committed(self, rnd):
        spec = self.index.spec('committed', rnd)
        if not spec:
            raise FileNotFoundError('Missing committed round%d. Provide native events or projection_states.json.' % rnd)
        return self.read_spec(spec)

    def anchor(self, rnd):
        spec = self.index.spec('pre_a', rnd)
        if spec:
            return self.read_spec(spec), dict(anchor_source='historical_pre_A', path=spec['path'])
        previous = self.index.spec('committed', rnd - 1)
        if not previous:
            raise FileNotFoundError('Missing previous A for pre-A recovery: round%d' % (rnd - 1))
        return recover_pre_a(self.read_spec(previous, allow_a_only=True), self.committed(rnd)), dict(
            anchor_source='historical_A_previous_B_current', previous_round=rnd - 1, current_round=rnd)

    def historical_proposals(self, rnd, anchor):
        path = self.index.event(rnd)
        if path is None:
            return None, dict(proposals='missing')
        payload = load(path)
        if 'local_factor_deltas' not in payload:
            return None, dict(proposals='missing_in_event')
        if payload.get('round') != rnd or payload.get('phase') != 'refresh_A':
            raise ValueError('Wrong event round/phase')
        selected = list(map(int, payload['selected_client_ids']))
        if selected != list(map(int, self.source['meta']['schedule'][rnd - 1])):
            raise ValueError('Event client order differs from recorded schedule')
        if not np.array_equal(np.asarray(payload['client_class_counts']), self.source['counts'][selected]):
            raise ValueError('Event client/class counts differ')
        if len(payload['local_factor_deltas']) != len(selected):
            raise ValueError('Missing local proposals')
        before, after = self.validate(payload['anchor_lora_state']), self.validate(payload['actual_after_lora_state'])
        if state_hash(before, self.keys) != state_hash(anchor, self.keys):
            raise ValueError('Proposals come from a different anchor')
        for k in self.keys:
            if k not in self.a_keys and not torch.equal(before[k], after[k]):
                raise ValueError('B changed during A phase')
        proposals = dict(zip(selected, payload['local_factor_deltas']))
        for d in proposals.values():
            if set(d) != set(self.a_keys) or any(d[k].shape != anchor[k].shape or d[k].dtype != anchor[k].dtype
                                                or not torch.isfinite(d[k]).all() for k in d):
                raise ValueError('Malformed proposal')
        weights = dict(zip(selected, map(float, payload['server_weights'])))
        if not np.allclose([weights[j] for j in range(30)], self.source['aggregation']['weights'], rtol=0, atol=1e-10):
            raise ValueError('Historical weights differ from declared aggregation')
        rebuilt = aggregate(anchor, proposals, weights)
        error = max(float((rebuilt[k] - after[k]).abs().max()) for k in self.keys)
        if error > 1e-5:
            raise ValueError('Historical aggregate reconstruction failed: %g' % error)
        return proposals, dict(proposals='original_historical_uploads', path=str(path),
                               reconstruction_max_abs_error=error, selected=selected)
