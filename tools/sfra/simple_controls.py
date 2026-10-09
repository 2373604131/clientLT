"""Seed42 simple-control protocol and portable, standard-library-only bookkeeping."""
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

from tools.sfra.maintext import FROZEN, PAIR_HASHES, TRAINING_FILES, digest, load_json
from tools.sfra.summary import METRICS, read_csv, write_csv

SCHEMA = 'method_a_simple_controls_v1'
PLAN_NAME = 'control_protocol.json'
NEW_METHODS = ('tailrw-g1', 'tailrw-g4', 'tailrw-g16', 'cover-cp')
METHODS = ('s', 'full-cp', *NEW_METHODS)
GAMMA = {'s': 0., 'full-cp': 0., 'cover-cp': 0.,
         'tailrw-g1': 1., 'tailrw-g4': 4., 'tailrw-g16': 16.}
CONTRACT = dict(schema_version=SCHEMA, seed=42, protocol_seed=42, rounds=100,
    partition='client-longtail', methods=list(METHODS), gamma=[1, 4, 16],
    tailrw='(n_j + gamma*n_j_tail)/(N + gamma*N_tail); B1..100 and A1..90',
    cover='inverse full-partition class holder count; normalized over original active tokens',
    targets='unchanged current/history/activation rules; each arm evolves its own history',
    A=FROZEN, classification_weight=1., execution='fast-v2 f128 c4 fp32',
    primary_endpoint='committed rounds81..100 Tail20 mean',
    selection='all three prespecified gamma values reported; no test-based selection')
EXTRA_FILES = ('utils/sfra_execution.py', 'utils/sfra_fast_feedback.py',
    'utils/sfra_resident_feedback.py', 'utils/cliplora_bridge_audit.py',
    'utils/cliplora_functional_feedback.py', 'utils/pfrf.py', 'utils/b_aggregation.py',
    'Dassl/dassl/data/data_manager.py', 'Dassl/dassl/engine/trainer.py')
IMPLEMENTATION_FILES = ('utils/method_a_simple_controls.py', 'tools/sfra/simple_controls.py',
    'tools/sfra/simple_controls_summary.py', 'scripts/run_method_a_simple_controls.py')
# federated_main has an opt-in dispatch/argument addition only. Its full digest is still
# frozen for new jobs; legacy pairing compares the unchanged training implementation.
PAIR_CODE_FILES = tuple(n for n in dict.fromkeys(TRAINING_FILES + EXTRA_FILES)
                        if n != 'federated_main.py')


def implementation_variant(method):
    if method not in METHODS:
        raise ValueError('Unknown simple control: ' + str(method))
    return 'full-cp' if method in ('full-cp', 'cover-cp') else 's'


def code_hashes(repo):
    names = tuple(dict.fromkeys(TRAINING_FILES + EXTRA_FILES + IMPLEMENTATION_FILES))
    return {n: hashlib.sha256((Path(repo)/n).read_text(encoding='utf-8').encode()).hexdigest()
            for n in names}


def write_table(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_csv(path, rows)


def partition_counts(rows, tail_ids):
    tail = set(map(int, tail_ids))
    if not tail or len(tail) != len(tail_ids):
        raise ValueError('Tail IDs must be nonempty and unique')
    clients, classes, pairs, tail_counts = Counter(), Counter(), Counter(), Counter()
    raw_ids, positions = set(), set()
    for row in rows:
        j, c, pos, raw = (int(row[k]) for k in ('client_id', 'class_id', 'local_position', 'raw_sample_id'))
        if min(j, c, pos, raw) < 0 or raw in raw_ids or (j, pos) in positions:
            raise ValueError('Invalid/duplicate partition sample identity')
        raw_ids.add(raw); positions.add((j, pos))
        clients[j] += 1; classes[c] += 1; pairs[j, c] += 1
        tail_counts[j] += int(c in tail)
    if not clients or not tail.issubset(classes) or not sum(tail_counts.values()):
        raise ValueError('Missing training clients or tail classes')
    if set(clients) != set(range(len(clients))) or set(classes) != set(range(len(classes))):
        raise ValueError('Client and class IDs must be contiguous')
    if positions != {(j, p) for j, n in clients.items() for p in range(n)}:
        raise ValueError('Local sample positions must be contiguous')
    return dict(sizes=[clients[j] for j in range(len(clients))],
        tail_counts=[tail_counts[j] for j in range(len(clients))],
        class_counts=[classes[c] for c in range(len(classes))],
        coverage=[sum((j, c) in pairs for j in clients) for c in range(len(classes))],
        tail_ids=sorted(tail))


def tail_weights(sizes, tail_counts, gamma):
    if (len(sizes) != len(tail_counts) or not sizes or not math.isfinite(gamma) or gamma < 0
            or any(not math.isfinite(n) or not math.isfinite(t) or n <= 0 or t < 0 or t > n
                   for n, t in zip(sizes, tail_counts)) or sum(tail_counts) <= 0):
        raise ValueError('Invalid counts or tail weighting strength')
    denominator = sum(sizes) + gamma * sum(tail_counts)
    return [(n + gamma*t)/denominator for n, t in zip(sizes, tail_counts)]


def static_tables(counts):
    sizes, tail = counts['sizes'], counts['tail_counts']
    weights = {g: tail_weights(sizes, tail, g) for g in (0, 1, 4, 16)}
    client_rows = [dict(client_id=j, n=n, n_tail=tail[j], tail_fraction=tail[j]/n,
        **{f'weight_g{g}': weights[g][j] for g in weights}) for j, n in enumerate(sizes)]
    class_rows = [dict(class_id=c, n=n, holders=counts['coverage'][c],
        inverse_coverage=1/counts['coverage'][c], is_tail=c in counts['tail_ids'])
        for c, n in enumerate(counts['class_counts'])]
    return client_rows, class_rows


def probe_manifest_replay(path, expected_sha256):
    """Recover recorded CSV bytes only when LF/CRLF conversion proves an exact match."""
    path = Path(path)
    original = path.read_bytes()
    lf = original.replace(b'\r\n', b'\n')
    candidates = [('unchanged', original), ('LF', lf), ('CRLF', lf.replace(b'\n', b'\r\n'))]
    hashes = {name: hashlib.sha256(value).hexdigest() for name, value in candidates}
    for name, value in candidates:
        if hashes[name] == expected_sha256:
            return value, dict(source_sha256=hashes['unchanged'], expected_sha256=expected_sha256,
                replayed_sha256=hashes[name], newline_conversion=name)
    raise ValueError(f'{path}: probe_manifest_sha256 mismatch cannot be repaired by LF/CRLF conversion; '
                     f'expected={expected_sha256}, observed={hashes}. '
                     'Restore the original reference manifest; do not replace the recorded hash.')


def protocol_info(reference):
    root = Path(reference)
    names = ('partition_manifest.csv', 'bridge_metadata.json', 'protocol/full_schedule.json',
             'protocol/eri_protocol.json', 'protocol/probe_manifest.csv')
    fingerprint = {n: hashlib.sha256((root/n).read_bytes()).hexdigest() for n in names}
    meta = load_json(root/'bridge_metadata.json')
    # Detect a corrupt/mismatched reference before any costly model initialization.
    # The reference remains read-only; exact byte restoration happens in each new run.
    probe_manifest_replay(root/'protocol/probe_manifest.csv', meta['probe_manifest_sha256'])
    if meta['topology'] != 'client-longtail':
        raise ValueError('This suite requires the frozen Client-LT protocol')
    schedule = load_json(root/'protocol/full_schedule.json')
    schedule = schedule['schedule'] if isinstance(schedule, dict) else schedule
    if len(schedule) != 100 or any(sorted(x) != list(range(30)) for x in schedule):
        raise ValueError('Expected 100 full-participation rounds')
    counts = partition_counts(read_csv(root/'partition_manifest.csv'),
        load_json(root/'protocol/eri_protocol.json')['tail_class_ids'])
    if len(counts['sizes']) != 30 or len(counts['class_counts']) != 100 or len(counts['tail_ids']) != 20:
        raise ValueError('Expected 30 clients, 100 classes and Tail20')
    return fingerprint, counts


def reference_code(run):
    for filename in ('simple_control_job.json', 'ab_validation_run.json'):
        path = Path(run)/filename
        if path.is_file():
            data = load_json(path)
            if filename == 'ab_validation_run.json':
                if not data.get('code_unchanged'):
                    raise ValueError('Legacy run changed code during training')
                data = data['job']
            return data['code_sha256']
    raise ValueError('Missing recorded training-source provenance')


def validate_config(run, method):
    run = Path(run)
    cfg = load_json(run/'sfra_config.json')
    expected = dict(FROZEN, variant=implementation_variant(method), seed=42, protocol_seed=42,
                    partition='client-longtail', witness_batch_size=8)
    if expected['variant'] == 'full-cp':
        expected['classification_weight'] = 1.
    for k, v in expected.items():
        if cfg.get(k) != v:
            raise ValueError(f'Frozen configuration mismatch: {k}')
    if cfg.get('b_transfer') or cfg.get('b_aggregation'):
        raise ValueError('Unexpected B transfer or alternative B aggregation')
    ex = load_json(run/'execution_config.json')
    if (ex.get('version'), ex.get('feedback_forward_batch_size'), ex.get('device_cache_gib')) != (2, 128, 4.):
        raise ValueError('Expected fast-v2 f128 c4 execution')
    base = cfg['base_training_config']
    for k, v in dict(seed=42, protocol_seed=42, topology='client-longtail', la_tau=1.,
                    normal_steps_expected=105600, extra_steps_expected=31680).items():
        if base.get(k) != v:
            raise ValueError('Frozen base training mismatch: ' + k)
    control = cfg.get('simple_control')
    if method in NEW_METHODS and control != control_config(method):
        raise ValueError('Wrong/missing simple-control semantics')
    if method in ('s', 'full-cp') and control not in (None, control_config(method)):
        raise ValueError('Reference contains an unexpected control')
    return cfg


def control_config(method):
    return dict(schema_version=SCHEMA, method=method, gamma=GAMMA[method], seed=42,
                aggregation_scope='B1..100,A1..90' if method.startswith('tailrw-') else 'original',
                priority='inverse_class_holders' if method == 'cover-cp' else 'original')


def pair_signature(run, cfg):
    run = Path(run)
    meta, code = load_json(run/'bridge_metadata.json'), reference_code(run)
    missing = set(PAIR_CODE_FILES) - set(code)
    if missing:
        raise ValueError('Missing training source records: ' + ', '.join(sorted(missing)))
    return dict(**{k: meta[k] for k in PAIR_HASHES},
        partition=digest(read_csv(run/'partition_manifest.csv')),
        execution=digest(load_json(run/'execution_config.json')),
        base_training=digest(cfg['base_training_config']),
        class_prior=digest(load_json(run/'class_prior.json')),
        training_code=digest({n: code[n] for n in PAIR_CODE_FILES}))
