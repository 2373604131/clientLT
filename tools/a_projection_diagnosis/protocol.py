"""Frozen sampling, source validation, and explicit historical-state discovery."""
import hashlib
from pathlib import Path

import numpy as np

from tools.a_learning_schedule.protocol import read_csv, read_json, write_csv, write_json, digest, file_digest

VERSION = 'a_projection_diagnosis_v1'
SNAPSHOTS = (0, 20, 50, 80, 90, 100)
ANCHORS = (20, 50, 80)
PROTOCOL = dict(version=VERSION, seed=42, topology='client-longtail', snapshots=list(SNAPSHOTS),
                anchors=list(ANCHORS), fit_per_class=32, panel_per_class=20, sampling_seed=20261012,
                pooling='CLS + mean(all patch tokens)', module_dimensions=8, modules=6,
                svd_relative_tolerance=1e-6, probe_loss='class-balanced ridge squared loss',
                ridge_lambda=.01, normalization='fit center + one scalar RMS; no channel z-score',
                primary_change='mean(round80,round90) minus round20',
                rules=['fedavg', 'tailrw16', 'joint'], joint_lambda=.1, tailrw_gamma=16,
                norm_matching='minimum positive effective norm; one scalar; shrink only',
                remaining_scope='remaining official test samples, not independent confirmation',
                primary_local_response='unscaled cosine margin change',
                claim='linear readability of the prescribed input summaries under a fixed sample budget',
                independent_training_seeds=1)


def sample_key(raw_id, domain='train'):
    prefix = 'A-projection-v1|20261012|' if domain == 'train' else 'A-projection-v1|20261012|test|'
    return hashlib.sha256((prefix + str(raw_id)).encode()).hexdigest()


def fit_manifest(rows, per_class=32):
    result = []
    for c in sorted({int(r['class_id']) for r in rows}):
        selected = sorted((r for r in rows if int(r['class_id']) == c),
                          key=lambda r: sample_key(int(r['raw_sample_id'])))[:per_class]
        result.extend(dict(raw_sample_id=int(r['raw_sample_id']), class_id=c,
                           client_id=int(r['client_id']), sampled_class_count=len(selected)) for r in selected)
    return result


def test_manifests(labels, per_class=20):
    labels = np.asarray(labels)
    panel = set()
    for c in sorted(set(labels.tolist())):
        ids = np.flatnonzero(labels == c).tolist()
        if len(ids) <= per_class:
            raise ValueError('Both panel and remaining samples must cover every class')
        panel.update(sorted(ids, key=lambda i: sample_key(i, 'test'))[:per_class])
    return [dict(sample_id=i, class_id=int(c), split='panel' if i in panel else 'remaining')
            for i, c in enumerate(labels)]


def check_source(path, rule):
    path = Path(path).resolve()
    meta, config = read_json(path / 'bridge_metadata.json'), read_json(path / 'control_config.json')
    expected = dict(method='s', seed=42, topology='client-longtail', la_tau=1., a_lr_mult=1.,
                    normal_trainable_factor='B', extra_trainable_factor='A', control_enabled=False,
                    extra_a_lr=.001, extra_weight_decay=0.)
    for key, value in expected.items():
        if config.get(key) != value:
            raise ValueError('Not the ordinary A B3/A1 protocol: %s=%r' % (key, config.get(key)))
    if meta.get('method') != 's':
        raise ValueError('Source metadata is not ordinary A')
    sfra = read_json(path / 'sfra_config.json')
    if (sfra.get('variant') != 's' or sfra.get('retention_weight') != 0
            or sfra.get('correction_steps') != 0 or sfra.get('b_transfer')):
        raise ValueError('Repair/retention/transfer is enabled; pre-A reconstruction is not valid')
    for key, ids in dict(head_ids=list(range(20)), middle_ids=list(range(20, 80)), tail_ids=list(range(80, 100))).items():
        if config.get(key) != ids:
            raise ValueError('Unexpected registered class grouping: ' + key)
    if config.get('candidate_rounds') != list(range(1, 91)):
        raise ValueError('Expected A in rounds1..90 and B-only rounds91..100')
    rows = sorted(read_csv(path / 'partition_manifest.csv'), key=lambda r: (int(r['client_id']), int(r['local_position'])))
    counts = np.zeros((30, 100), dtype=np.int64)
    positions = [[] for _ in range(30)]
    for r in rows:
        j, c = int(r['client_id']), int(r['class_id'])
        if not 0 <= j < 30 or not 0 <= c < 100:
            raise ValueError('Invalid client/class')
        counts[j, c] += 1
        positions[j].append(int(r['local_position']))
    if len(rows) != 10847 or len({int(r['raw_sample_id']) for r in rows}) != len(rows):
        raise ValueError('Expected 10847 unique training samples')
    if any(p != list(range(len(p))) for p in positions) or counts.sum(1).tolist() != meta['client_sample_counts']:
        raise ValueError('Partition order or counts differ from training metadata')
    if len(meta['schedule']) != 100 or any(sorted(map(int, s)) != list(range(30)) for s in meta['schedule']):
        raise ValueError('Expected 100 rounds with full client participation')
    solution = path / 'aggregation_solution.json'
    if not solution.is_file():
        solution = path.parents[2] / 'experiment_plan.json'
    spec = read_json(solution)
    spec = spec.get('aggregation', spec)
    if spec.get('rule', 'joint') != rule or not np.array_equal(spec['matrix'], counts):
        raise ValueError('Aggregation rule/counts differ from the requested trajectory')
    return dict(path=str(path), rule=rule, config=config, meta=meta, rows=rows, counts=counts,
                sizes=counts.sum(1).tolist(), aggregation=spec,
                partition_sha256=file_digest(path / 'partition_manifest.csv'))


def check_pair(sources):
    if len(sources) < 2:
        return dict(paired=False, reason='FedAvg only; no TailRW longitudinal claim')
    a, b = sources
    if a['rows'] != b['rows']:
        raise ValueError('Trajectory training samples/assignment/order differ')
    for key in ('initial_lora_sha256', 'frozen_model_sha256', 'pool_sha256', 'test_sha256', 'schedule_sha256'):
        if not a['meta'].get(key) or a['meta'][key] != b['meta'].get(key):
            raise ValueError('Trajectory mismatch: ' + key)
    from tools.a_learning_schedule.config import parse_archived_config
    ca, cb = (parse_archived_config(s['meta']['resolved_config']) for s in sources)
    for key in ('INPUT', 'OPTIM', 'TRAINER'):
        if ca[key] != cb[key]:
            raise ValueError('Trajectory config mismatch: ' + key)
    return dict(paired=True, same_recorded_environment=a['meta'].get('environment') == b['meta'].get('environment'))


class StateIndex:
    """Native bridge events, or an explicit map for separately saved checkpoints.

    Optional projection_states.json:
      {"committed":{"19":{"path":"checkpoints/r019.pt"}},
       "pre_a":{"20":{"path":"...", "key":"anchor_lora_state"}}}
    No filename guessing for ambiguous last.pt/decision checkpoints.
    """
    def __init__(self, source):
        self.root = Path(source['path'])
        path = self.root / 'projection_states.json'
        self.mapping = read_json(path) if path.is_file() else {}

    def event(self, rnd, phase='refresh_A'):
        path = self.root / 'events' / ('r%03d_c000_main_%s' % (rnd, phase)) / 'state.pt'
        return path if path.is_file() else None

    def spec(self, category, rnd):
        value = self.mapping.get(category, {}).get(str(rnd))
        if value:
            value = dict(path=value) if isinstance(value, str) else dict(value)
            value['path'] = str(self.root / value['path'])
            return value if Path(value['path']).is_file() else None
        if category == 'committed':
            if rnd == 0:
                p = self.root / 'checkpoints/base_model.pt'
                return dict(path=str(p)) if p.is_file() else None
            p = self.event(rnd, 'refresh_A' if rnd <= 90 else 'normal_B')
            return dict(path=str(p), key='actual_after_lora_state') if p else None
        if category == 'pre_a':
            p = self.event(rnd)
            if p:
                return dict(path=str(p), key='anchor_lora_state')
            p = self.event(rnd, 'normal_B')
            return dict(path=str(p), key='actual_after_lora_state') if p else None
        raise ValueError(category)

    def inventory(self, need_events=True):
        rows = []
        base = self.root / 'checkpoints/base_model.pt'
        rows.append(dict(artifact='frozen_base', available=base.is_file(), path=str(base)))
        for rnd in SNAPSHOTS:
            spec = self.spec('committed', rnd)
            rows.append(dict(artifact='committed_%03d' % rnd, available=bool(spec), path=spec['path'] if spec else ''))
        if need_events:
            for rnd in ANCHORS:
                direct = self.spec('pre_a', rnd)
                previous, current = self.spec('committed', rnd - 1), self.spec('committed', rnd)
                recover = bool(previous and current)
                rows.append(dict(artifact='pre_a_%03d' % rnd, available=bool(direct) or recover,
                                 path=direct['path'] if direct else '', recovery='direct' if direct else
                                 'A_previous+B_current' if recover else 'missing',
                                 previous_A_path=previous['path'] if previous else ''))
                event = self.event(rnd)
                rows.append(dict(artifact='proposals_%03d' % rnd, available=bool(event), path=str(event or ''),
                                 recovery='inspect_event_payload' if event else 'regenerate_from_historical_anchor'))
        return rows


def aggregation_weights(source):
    from tools.client_aggregation.weights import solve_weights
    counts = source['counts']
    sizes, tails = counts.sum(1), counts[:, source['config']['tail_ids']].sum(1)
    joint = solve_weights(counts, .1)['weights']
    return {name: dict(enumerate(map(float, w))) for name, w in dict(
        fedavg=sizes / sizes.sum(), tailrw16=(sizes + 16 * tails) / (sizes.sum() + 16 * tails.sum()), joint=joint).items()}
