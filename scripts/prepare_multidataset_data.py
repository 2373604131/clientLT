"""Explicit server-side downloads and ImageNet-LT metadata preparation.

Nothing is downloaded until this script is invoked. ImageNet images are NEVER
downloaded here; use an existing authorized ImageNet image directory.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tarfile
import urllib.request

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

SOURCES = {
    'food101': ('https://data.vision.ee.ethz.ch/cvl/food-101.tar.gz', '85eeb15f3717b99a5da872d97d918f87', 'food-101'),
    'cub200': ('https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz?download=1', '97eceeb196236b17998738112f37df78', 'CUB_200_2011'),
    'dtd': ('https://www.robots.ox.ac.uk/~vgg/data/dtd/download/dtd-r1.0.1.tar.gz', 'fff73e5086ae6bdbea199a49dfb8a4c1', 'dtd'),
}
MANIFEST_BASE = 'https://raw.githubusercontent.com/facebookresearch/classifier-balancing/master/data/ImageNet_LT/'


def file_digest(path, algorithm='sha256'):
    result = hashlib.new(algorithm)
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            result.update(block)
    return result.hexdigest()


def download(url, target, md5=None):
    target = Path(target)
    if target.exists():
        if md5 and file_digest(target, 'md5') != md5:
            raise ValueError('Existing archive checksum mismatch: ' + str(target))
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + '.part')
    offset = partial.stat().st_size if partial.exists() else 0
    headers = {'User-Agent': 'clientLT-dataset-preparation/1.0'}
    if offset:
        headers['Range'] = f'bytes={offset}-'
    request = urllib.request.Request(url, headers=headers)
    print('Downloading:', url, 'resume offset=', offset, flush=True)
    with urllib.request.urlopen(request, timeout=120) as response:
        append = offset > 0 and response.status == 206
        if append and not response.headers.get('Content-Range', '').startswith(f'bytes {offset}-'):
            raise ValueError('Server returned the wrong resume range')
        with partial.open('ab' if append else 'wb') as stream:
            shutil.copyfileobj(response, stream, length=1024*1024)
    if md5 and file_digest(partial, 'md5') != md5:
        raise ValueError('Downloaded archive checksum mismatch: ' + str(partial))
    partial.replace(target)


def safe_extract(archive, destination):
    """Reject traversal/links; retain matching existing files and resume partial extraction."""
    destination = Path(destination).resolve()
    with tarfile.open(archive, 'r:*') as stream:
        for entry in stream:
            relative = Path(entry.name)
            target = (destination / relative).resolve()
            if (relative.is_absolute() or '..' in relative.parts or target != destination and destination not in target.parents
                or not (entry.isdir() or entry.isfile())):
                raise ValueError('Unsafe archive entry: ' + entry.name)
            if entry.isdir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            source = stream.extractfile(entry)
            if target.exists():
                if target.stat().st_size != entry.size or hashlib.sha256(source.read()).hexdigest() != file_digest(target):
                    raise ValueError('Existing dataset file differs; refusing overwrite: ' + str(target))
                continue
            temporary = target.with_name(target.name + '.extracting')
            with temporary.open('wb') as output:
                shutil.copyfileobj(source, output)
            temporary.replace(target)


def prepare_imagenet(args):
    from tools.multidataset.catalog import lines, read_dataset
    root = args.data_root / 'imagenet-lt'
    root.mkdir(parents=True, exist_ok=True)
    if not args.imagenet_source or not (args.synset_words or args.devkit_meta):
        raise ValueError('ImageNet-LT requires --imagenet-source and either --synset-words or --devkit-meta')
    for split in ('train', 'val'):
        source, target = (args.imagenet_source / split).resolve(), root / split
        if not source.is_dir():
            raise ValueError('Missing shared ImageNet image directory: ' + str(source))
        if target.exists():
            if target.resolve() != source:
                raise ValueError('Existing image link points to another source: ' + str(target))
        else:
            target.symlink_to(source, target_is_directory=True)
    hashes = {}
    for split in ('train', 'val', 'test'):
        name = f'ImageNet_LT_{split}.txt'
        download(MANIFEST_BASE + name, root / name)
        hashes[name] = file_digest(root / name)
    if args.synset_words:
        words = dict(line.split(maxsplit=1) for line in lines(args.synset_words))
    else:
        from scipy.io import loadmat
        meta = loadmat(args.devkit_meta, squeeze_me=True, struct_as_record=False)
        words = {str(s.WNID): str(s.words) for s in meta['synsets']}
    by_label = {}
    for line in lines(root / 'ImageNet_LT_train.txt'):
        path, label = line.rsplit(maxsplit=1)
        wnid = next((p for p in Path(path).parts if len(p) == 9 and p.startswith('n') and p[1:].isdigit()), None)
        if wnid is None or wnid not in words:
            raise ValueError('Missing original ImageNet synset name: ' + str(wnid))
        label = int(label)
        if label in by_label and by_label[label][0] != wnid:
            raise ValueError('A manifest label maps to multiple synsets')
        by_label[label] = (wnid, words[wnid].split(',')[0].strip())
    if sorted(by_label) != list(range(1000)):
        raise ValueError('Expected all 1000 original ImageNet-LT labels')
    from tools.benchmarks.common import write_json, read_json
    names = [by_label[i][1] for i in range(1000)]
    if (root / 'classnames.json').exists() and read_json(root / 'classnames.json') != names:
        raise ValueError('Existing ImageNet class names differ')
    write_json(root / 'classnames.json', names)
    write_json(root / 'metadata_sources.json', dict(manifest_base=MANIFEST_BASE, manifest_sha256=hashes,
        classnames_sha256=file_digest(root / 'classnames.json'), original_label_synsets=by_label))
    _, train, valid, test = read_dataset('imagenet-lt', root)
    missing = [row['path'] for row in train + valid + test if not Path(row['path']).is_file()]
    if missing:
        raise ValueError(f'{len(missing)} manifest images are missing. Match the official paths; examples: {missing[:5]}')
    print('ImageNet-LT metadata ready:', root)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--datasets', nargs='+', choices=(*SOURCES, 'imagenet-lt'), required=True)
    parser.add_argument('--data-root', type=Path, default=Path('DATA'))
    parser.add_argument('--imagenet-source', type=Path)
    parser.add_argument('--synset-words', type=Path, help='Official wnid followed by class description, one per line')
    parser.add_argument('--devkit-meta', type=Path, help='ILSVRC2012_devkit_t12/data/meta.mat')
    args = parser.parse_args()
    try:
        from scripts.run_ab_validation import file_lock
        for dataset in args.datasets:
            with file_lock(args.data_root / '.locks' / (dataset + '.lock')):
                if dataset == 'imagenet-lt':
                    prepare_imagenet(args)
                    continue
                url, md5, directory = SOURCES[dataset]
                archive = args.data_root / '.archives' / (dataset + '.tar.gz')
                download(url, archive, md5)
                safe_extract(archive, args.data_root)
                from tools.multidataset.catalog import read_dataset
                read_dataset(dataset, args.data_root / directory)
                print('READY:', args.data_root / directory, flush=True)
    except (OSError, ValueError, KeyError) as exc:
        parser.error(str(exc))


if __name__ == '__main__':
    main()
