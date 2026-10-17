"""Frozen shared pools, Client-LT allocations, content identities and paper metrics."""
from collections import defaultdict
from pathlib import Path
import hashlib
import math
import numpy as np

from tools.benchmarks.common import read_json, read_csv, write_json, write_csv, digest
from tools.multidataset.catalog import DATASETS, read_dataset

SCHEMA = 'clientlt_multidataset_v1'


def groups(classes):
    # Match partition_client_longtail: tail is C - floor(0.8*C), not hard-coded 20.
    h, t = int(classes * .2), int(classes * .8)
    return dict(head_ids=list(range(h)), middle_ids=list(range(h, t)), tail_ids=list(range(t, classes)))


def make_pool(name, names, train, source_reference, seed=42):
    """Rank classes using training data ONLY; new synthetic pools use fixed seed42."""
    by_class = defaultdict(list)
    for row in train:
        by_class[row['source_label']].append(row)
    classes = len(names)
    if name == 'cifar100-lt':
        source = read_csv(Path(source_reference) / 'partition_manifest.csv')
        lookup = {r['raw_sample_id']: r for r in train}
        from utils.functional_coverage_validation import _exact_lt_raw_ids
        labels = np.array([r['source_label'] for r in sorted(train, key=lambda r: r['raw_sample_id'])])
        ordered_ids = list(map(int, _exact_lt_raw_ids(labels, .01, 'exp')))
        if sorted(ordered_ids) != sorted(int(r['raw_sample_id']) for r in source):
            raise ValueError('Frozen CIFAR pool differs from its exact original LT construction')
        source_labels = {int(r['raw_sample_id']): int(r['class_id']) for r in source}
        pool = []
        for raw_id in ordered_ids:
            item = lookup[raw_id]
            if item['source_label'] != source_labels[raw_id]:
                raise ValueError('CIFAR reference labels differ from official training data')
            pool.append(dict(item, class_id=item['source_label']))
        if len({r['raw_sample_id'] for r in pool}) != len(pool):
            raise ValueError('Duplicate CIFAR reference samples')
        order = list(range(classes))
    else:
        rng = np.random.RandomState(seed)
        order = (sorted(range(classes), key=lambda c: (-len(by_class[c]), c)) if name == 'imagenet-lt'
                 else list(map(int, rng.permutation(classes))))
        pool = []
        maximum = min(map(len, by_class.values()))
        for rank, original in enumerate(order):
            rows = by_class[original]
            if name != 'imagenet-lt':
                target = max(1, int(maximum * DATASETS[name]['imbalance'] ** (-rank / (classes - 1))))
                rows = [rows[i] for i in rng.permutation(len(rows))[:target]]
            pool.extend(dict(r, class_id=rank) for r in rows)
    if {r['class_id'] for r in pool} != set(range(classes)):
        raise ValueError('The long-tail pool must retain every class')
    return pool, order


def allocate(pool, classes, specialization, source_reference=None, preserve_reference=False):
    if not math.isfinite(specialization) or not 0 <= specialization <= 1:
        raise ValueError('specialization_lambda must lie in [0,1]')
    if preserve_reference:
        rows = [{k: int(v) for k, v in r.items()} for r in read_csv(Path(source_reference) / 'partition_manifest.csv')]
        lookup = {r['raw_sample_id']: r for r in pool}
        rows = [dict(lookup[r['raw_sample_id']], client_id=r['client_id'], local_position=r['local_position']) for r in rows]
        regenerated, _ = allocate(pool, classes, specialization)
        signature = lambda values: sorted((r['raw_sample_id'], r['class_id'], r['client_id'], r['local_position']) for r in values)
        if signature(rows) != signature(regenerated):
            raise ValueError('lambda=.75 no longer reproduces the historical reference; do not mix sweep protocols')
    else:
        from utils.datasplit import partition_client_longtail
        labels = np.array([r['class_id'] for r in pool])
        mapping = partition_client_longtail(labels, 30, classes, specialization_lambda=specialization,
            intra_group_alpha=.5, head_leakage_scale=3., rng=np.random.RandomState(42))
        rows = [dict(pool[int(i)], client_id=c, local_position=p)
                for c in range(30) for p, i in enumerate(mapping[c])]
    rows.sort(key=lambda r: (r['client_id'], r['local_position']))
    if sorted(r['raw_sample_id'] for r in rows) != sorted(r['raw_sample_id'] for r in pool):
        raise ValueError('Partition must conserve the exact global sample pool')
    matrix = np.zeros((30, classes), dtype=np.int64)
    for r in rows:
        matrix[r['client_id'], r['class_id']] += 1
    if (matrix.sum(1) == 0).any():
        raise ValueError('Empty client: revise the registered dataset protocol; no hidden resampling')
    return rows, matrix


def content_hashes(rows):
    """Hash each selected file once; CIFAR archive files are also hashed once."""
    values = {}
    for path in sorted({r['path'] for r in rows}):
        h = hashlib.sha256()
        with Path(path).open('rb') as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                h.update(chunk)
        values[path] = h.hexdigest()
    return values


def prepare(name, root, destination, specialization, source_reference, dtd_split=1):
    from scripts.run_ab_validation import file_lock
    from tools.client_aggregation.weights import solve_weights
    destination = Path(destination)
    with file_lock(destination.parent / 'locks' / (destination.name + '.lock')):
        if (destination / 'protocol.json').exists():
            protocol = read_json(destination / 'protocol.json')
            if (protocol['dataset'] != name or protocol['data_root'] != str(Path(root).resolve())
                or protocol['specialization_lambda'] != specialization or protocol['dtd_split'] != dtd_split):
                raise ValueError('Existing protocol settings changed; use a new output root')
            verify(destination, images=True)
            return protocol
        names, train, valid, test = read_dataset(name, root, dtd_split)
        pool, order = make_pool(name, names, train, source_reference)
        inverse = {original: rank for rank, original in enumerate(order)}
        valid = [dict(r, class_id=inverse[r['source_label']]) for r in valid]
        test = [dict(r, class_id=inverse[r['source_label']]) for r in test]
        preserve = name == 'cifar100-lt' and specialization == .75
        if name == 'cifar100-lt':
            source = read_json(Path(source_reference) / 'bridge_metadata.json')
            actual = source['resolved_args']
            if source['topology'] != 'client-longtail' or actual.get('specialization_lambda') != .75:
                raise ValueError('CIFAR reference must be the existing lambda=.75 Client-LT run')
        rows, matrix = allocate(pool, len(names), specialization, source_reference, preserve)
        schedule = read_json(Path(source_reference) / 'protocol/full_schedule.json')
        schedule = schedule['schedule'] if isinstance(schedule, dict) else schedule
        if len(schedule) != 100 or any(sorted(x) != list(range(30)) for x in schedule):
            raise ValueError('Need the frozen 100-round, 30-client full-participation schedule')
        solution = solve_weights(matrix, .1)
        solution.pop('seconds', None)
        protocol = dict(schema=SCHEMA, dataset=name, data_root=str(Path(root).resolve()),
            classes=len(names), classnames=[names[i] for i in order], source_class_order=order,
            specialization_lambda=specialization, joint_aggregation_strength=.1,
            split_seed=42, pool_seed=42, dtd_split=dtd_split, clients=30, rounds=100,
            global_imbalance=DATASETS[name]['imbalance'], official_split=DATASETS[name]['split'],
            preserve_cifar_reference_allocation=preserve, schedule=schedule,
            groups=groups(len(names)), matrix=matrix.tolist(), joint_solution=solution,
            pool_sha256=digest(sorted(pool, key=lambda r: r['raw_sample_id'])),
            partition_sha256=digest(rows), test_sha256=digest(test), validation_sha256=digest(valid),
            input_transform='legacy_cifar' if name == 'cifar100-lt' else 'clip_224',
            reference_config=read_json(Path(source_reference) / 'bridge_metadata.json')['resolved_config'])
        if name == 'imagenet-lt':
            counts = matrix.sum(0)
            protocol['groups'].update(many_shot_ids=np.flatnonzero(counts > 100).tolist(),
                medium_shot_ids=np.flatnonzero((counts >= 20) & (counts <= 100)).tolist(),
                few_shot_ids=np.flatnonzero(counts < 20).tolist())
        write_json(destination / 'train.json', rows)
        write_json(destination / 'validation.json', valid)
        write_json(destination / 'test.json', test)
        write_json(destination / 'image_hashes.json', content_hashes(rows + valid + test))
        protocol['image_hashes_sha256'] = digest(read_json(destination / 'image_hashes.json'))
        write_json(destination / 'protocol.json', protocol)
        write_csv(destination / 'partition_manifest.csv', [dict(raw_sample_id=r['raw_sample_id'],
            class_id=r['class_id'], client_id=r['client_id'], local_position=r['local_position']) for r in rows])
        verify(destination, images=False)
        return protocol


def verify(root, images=False):
    root = Path(root)
    protocol = read_json(root / 'protocol.json')
    train, valid, test = [read_json(root / (s + '.json')) for s in ('train', 'validation', 'test')]
    for rows, key in ((train, 'partition_sha256'), (valid, 'validation_sha256'), (test, 'test_sha256')):
        if digest(rows) != protocol[key]:
            raise ValueError('Changed protocol file: ' + key)
    hashes = read_json(root / 'image_hashes.json')
    if digest(hashes) != protocol['image_hashes_sha256']:
        raise ValueError('Changed image content manifest')
    if images and content_hashes(train + valid + test) != hashes:
        raise ValueError('Dataset image contents changed after preparation')
    from tools.client_aggregation.weights import audit_weights
    audit_weights(protocol['joint_solution']['weights'], protocol['matrix'], .1)
    return protocol


def metrics(correct, total, group):
    correct, total = np.asarray(correct), np.asarray(total)
    if (correct.ndim != 1 or correct.shape != total.shape or not np.isfinite(correct).all()
        or not np.isfinite(total).all() or (total <= 0).any() or (correct < 0).any() or (correct > total).any()
        or (correct != np.floor(correct)).any() or (total != np.floor(total)).any()):
        raise ValueError('Invalid per-class counts or missing test classes')
    acc = 100. * correct / total
    result = dict(overall_acc=float(100. * correct.sum() / total.sum()), macro_acc=float(acc.mean()),
        head_acc=float(acc[group['head_ids']].mean()), middle_acc=float(acc[group['middle_ids']].mean()),
        tail_acc=float(acc[group['tail_ids']].mean()))
    for name in ('many_shot', 'medium_shot', 'few_shot'):
        if name + '_ids' in group:
            indices = group[name + '_ids']
            if not indices:
                raise ValueError('Empty ImageNet-LT frequency group: ' + name)
            result[name + '_acc'] = float(acc[indices].mean())
    return result
