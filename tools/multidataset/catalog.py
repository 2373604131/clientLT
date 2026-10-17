"""Metadata readers. No image downloads, model imports, or training at import time."""
from pathlib import Path
import json
import pickle

DATASETS = {
    'imagenet-lt': dict(classes=1000, imbalance=None, split='official ImageNet-LT manifests'),
    'food101': dict(classes=101, imbalance=100, split='official train/test'),
    'cub200': dict(classes=200, imbalance=10, split='official train_test_split.txt'),
    'dtd': dict(classes=47, imbalance=10, split='official train1/val1/test1; validation excluded from training'),
    'cifar100-lt': dict(classes=100, imbalance=100, split='existing frozen CIFAR100-LT training pool'),
}
GROUPS = {
    'longtail': ('fedavg-lora', 'fedavg-lora-la', 'capt', 'fedpurel', 'fedntd', 'fedlf', 'fedyoyo', 'fedrela'),
    'factors': ('ffa-lora', 'rolora', 'fedsvd', 'lora-a2'),
    'ours': ('joint-old-ab',),
}
METHODS = sum(GROUPS.values(), ())
BATCHES = {
    'imagenet': ('imagenet-lt', .75), 'food': ('food101', .75),
    'cub': ('cub200', .75), 'dtd': ('dtd', .75),
    'cifar-lambda050': ('cifar100-lt', .5),
    'cifar-lambda075': ('cifar100-lt', .75),
    'cifar-lambda100': ('cifar100-lt', 1.),
}


def lines(path):
    return [s.strip() for s in Path(path).read_text(encoding='utf-8-sig').splitlines() if s.strip()]


def safe_path(root, relative):
    root = Path(root).resolve()
    relative = Path(relative)
    if relative.is_absolute() or '..' in relative.parts:
        raise ValueError('Image path must stay inside dataset root: ' + str(relative))
    # A user-created train/val symlink may point at the server's shared ImageNet.
    return root / relative


def record(path, label, raw_id):
    return dict(path=str(path), source_label=int(label), raw_sample_id=int(raw_id))


def read_dataset(name, root, split_id=1):
    """Return names, official train/validation/test. Labels are source labels here."""
    root = Path(root).resolve()
    if name == 'food101':
        names = lines(root / 'meta/classes.txt')
        labels = {v: k for k, v in enumerate(names)}
        def part(split):
            return [record(safe_path(root / 'images', p + '.jpg'), labels[p.split('/')[0]], i)
                    for i, p in enumerate(lines(root / 'meta' / (split + '.txt')))]
        result = names, part('train'), [], part('test')
    elif name == 'cub200':
        def pairs(filename):
            return {int(a): b for a, b in (s.split(maxsplit=1) for s in lines(root / filename))}
        classes, images = pairs('classes.txt'), pairs('images.txt')
        labels, splits = pairs('image_class_labels.txt'), pairs('train_test_split.txt')
        if set(images) != set(labels) or set(images) != set(splits):
            raise ValueError('CUB image IDs, labels and split IDs do not match')
        names = [classes[i].split('.', 1)[-1] for i in range(1, len(classes) + 1)]
        parts = {0: [], 1: []}
        for i, path in sorted(images.items()):
            parts[int(splits[i])].append(record(safe_path(root / 'images', path), int(labels[i])-1, i))
        result = names, parts[1], [], parts[0]
    elif name == 'dtd':
        if not 1 <= split_id <= 10:
            raise ValueError('DTD split must be in 1..10')
        paths = {s: lines(root / 'labels' / f'{s}{split_id}.txt') for s in ('train', 'val', 'test')}
        names = sorted({p.split('/')[0] for values in paths.values() for p in values})
        labels = {v: k for k, v in enumerate(names)}
        def part(split):
            return [record(safe_path(root / 'images', p), labels[p.split('/')[0]], i)
                    for i, p in enumerate(paths[split])]
        result = names, part('train'), part('val'), part('test')
    elif name == 'imagenet-lt':
        # A JSON list indexed by ORIGINAL manifest label, never alphabetically sorted.
        names = json.loads((root / 'classnames.json').read_text(encoding='utf-8-sig'))
        if not isinstance(names, list) or any(not isinstance(n, str) or not n.strip() for n in names):
            raise ValueError('ImageNet classnames.json must be a list in original manifest label order')
        def part(split):
            rows = []
            for i, line in enumerate(lines(root / f'ImageNet_LT_{split}.txt')):
                path, label = line.rsplit(maxsplit=1)
                rows.append(record(safe_path(root, path), int(label), i))
            return rows
        result = names, part('train'), part('val'), part('test')
    elif name == 'cifar100-lt':
        from utils.functional_coverage_validation import _locate_cifar100
        folder = _locate_cifar100(root)
        def load(filename):
            with (folder / filename).open('rb') as stream:
                return pickle.load(stream, encoding='latin1')
        names = load('meta')['fine_label_names']
        def part(split):
            return [dict(path=str(folder / split), source_label=int(label), raw_sample_id=i, array_index=i)
                    for i, label in enumerate(load(split)['fine_labels'])]
        result = names, part('train'), [], part('test')
    else:
        raise ValueError('Unknown dataset: ' + name)
    names, train, valid, test = result
    classes = DATASETS[name]['classes']
    if len(names) != classes:
        raise ValueError(f'{name}: expected {classes} class names, found {len(names)}')
    for split_name, rows in zip(('train', 'validation', 'test'), (train, valid, test)):
        if rows and {r['source_label'] for r in rows} != set(range(classes)):
            raise ValueError(f'{name}: {split_name} must cover every class')
        keys = [(r['path'], r.get('array_index')) for r in rows]
        if len(keys) != len(set(keys)):
            raise ValueError(f'{name}: duplicate images in {split_name}')
    sets = [{(r['path'], r.get('array_index')) for r in rows} for rows in (train, valid, test)]
    if sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2]:
        raise ValueError('Official train/validation/test splits overlap')
    return result
