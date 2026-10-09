"""Dependency-free protocol, checkpoint discovery and shared-prefix plans."""
import csv
import hashlib
import json
import os
from pathlib import Path

VERSION = 'a_learning_schedule_v1'
ARMS = ('BB', 'AB', 'AA_short', 'AA_long')
ROUNDS = (20, 50, 80)
HORIZON = 10
PRIMARY = (8, 9, 10)
CONTRASTS = (('AB', 'BB'), ('AA_long', 'AB'), ('AA_long', 'AA_short'))
PROTOCOL = dict(version=VERSION, horizon=HORIZON, primary_offsets=list(PRIMARY),
                rounds=list(ROUNDS), normal_epochs=3, extra_epochs=1,
                lr=.001, momentum=.9, normal_weight_decay=.0005,
                extra_weight_decay=0., batch_size=32, la_tau=1.,
                short_gap=1, long_gap=5, anchor_phase='post_normal_B_pre_extra',
                feedback_per_client_class=8, norm_matching='pairwise_minimum_positive',
                text_cache=True, low_rank_forward_rewrite=False)


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    os.replace(temporary, path)


def read_csv(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(k for row in rows for k in row))
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def file_digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def actions(arm):
    if arm not in ARMS:
        raise ValueError('Unknown arm: ' + arm)
    first = 'B' if arm == 'BB' else 'A'
    second = 'B' if arm in ('BB', 'AB') else 'A'
    second_h = 1 if arm == 'AA_short' else 5
    result = [(0, 'extra', first)]
    for h in range(1, HORIZON + 1):
        result.append((h, 'normal', 'B'))
        if h == second_h:
            result.append((h, 'extra', second))
    return result


def plan(arm):
    prefix = []
    parent = 'anchor'
    result = []
    for h, kind, factor in actions(arm):
        prefix.append((h, kind, factor))
        identity = 'h%02d_%s_%s_%s' % (h, kind, factor, digest(prefix)[:12])
        result.append(dict(id=identity, parent=parent, h=h, kind=kind, factor=factor))
        parent = identity
    return result


def final_nodes(arm):
    """For each offset, select AFTER its last intervention, never its pre-extra state."""
    return {node['h']: node for node in plan(arm)}


def paired_probes():
    a0, b0 = plan('AB')[0], plan('BB')[0]
    a5 = next(n for n in plan('AA_long') if n['kind'] == 'extra' and n['h'] == 5)
    b5 = next(n for n in plan('AB') if n['kind'] == 'extra' and n['h'] == 5)
    return ((a0, b0), (a5, b5))


def phase_seed(seed, anchor_round, node, client):
    # No arm, active factor, source history, device, or process order in this seed.
    # Extra updates at different offsets also see the same second-dose stream.
    if node['kind'] == 'normal':
        return seed + 1000003 * (anchor_round + node['h']) + 1009 * client
    dose = 0 if node['h'] == 0 else 1
    return seed + 1000003 * anchor_round + 1009 * client + 700000001 + dose * 1000033


def source_event(source, origin, anchor_round):
    phase = 'extra_B' if origin == 'e2' else 'refresh_A'
    return Path(source) / 'events' / ('r%03d_c000_main_%s' % (anchor_round, phase)) / 'state.pt'


def discover_source(source_root, seed, origin, override=None):
    if override:
        return Path(override).resolve()
    parent = Path(source_root).resolve() / ('seed%d' % seed) / 'client-longtail' / origin
    matches = []
    for path in parent.glob('*/control_config.json'):
        cfg = read_json(path)
        if cfg.get('la_tau') == 1 and cfg.get('a_lr_mult') == 1:
            matches.append(path.parent)
    if len(matches) != 1:
        raise ValueError('Expected one tau=1, A-lr=1 source under %s; found %d. Specify --%s-run explicitly.'
                         % (parent, len(matches), origin))
    return matches[0]


def check_source(source, origin, seed, rounds, require_weights=True):
    source = Path(source)
    for name in ('bridge_metadata.json', 'control_config.json', 'partition_manifest.csv'):
        if not (source / name).is_file():
            raise FileNotFoundError('Missing source metadata: ' + str(source / name))
    meta, cfg = read_json(source / 'bridge_metadata.json'), read_json(source / 'control_config.json')
    if (cfg.get('method'), cfg.get('seed'), cfg.get('topology'), cfg.get('la_tau'), cfg.get('a_lr_mult')) != (
            origin, seed, 'client-longtail', 1, 1):
        raise ValueError('Source must be matching-seed Client-LT E2/E3, LA=1, A lr multiplier=1: ' + str(source))
    rows = sorted(read_csv(source / 'partition_manifest.csv'), key=lambda r: (int(r['client_id']), int(r['local_position'])))
    counts = [[0] * 100 for _ in range(30)]
    positions = [[] for _ in range(30)]
    raw_ids = []
    for row in rows:
        k, c = int(row['client_id']), int(row['class_id'])
        if not 0 <= k < 30 or not 0 <= c < 100:
            raise ValueError('Invalid client/class in partition')
        counts[k][c] += 1
        positions[k].append(int(row['local_position']))
        raw_ids.append(int(row['raw_sample_id']))
    if len(rows) != 10847 or len(set(raw_ids)) != len(rows):
        raise ValueError('Expected the original 10847 distinct training images')
    if any(not p or p != list(range(len(p))) for p in positions):
        raise ValueError('Client positions are missing or duplicated')
    sizes = [sum(c) for c in counts]
    if sizes != meta['client_sample_counts']:
        raise ValueError('Partition sizes disagree with source metadata')
    schedule = meta['schedule']
    if len(schedule) < max(rounds) + HORIZON or any(sorted(map(int, s)) != list(range(30)) for s in schedule):
        raise ValueError('Expected a full-participation source schedule through all continuations')
    missing = []
    for path in [source / 'checkpoints/base_model.pt'] + [source_event(source, origin, r) for r in rounds]:
        if not path.is_file() or path.stat().st_size == 0:
            missing.append(str(path))
    if require_weights and missing:
        raise FileNotFoundError('Full SERVER checkpoints are required; a lightweight analysis archive is insufficient:\n' + '\n'.join(missing))
    return dict(path=str(source), meta=meta, config=cfg, rows=rows, counts=counts, sizes=sizes,
                partition_sha256=file_digest(source / 'partition_manifest.csv'), missing_weights=missing)


def job_root(output_root, seed, origin, anchor_round):
    return Path(output_root) / 'runs' / ('seed%d' % seed) / origin / ('round%03d' % anchor_round)
