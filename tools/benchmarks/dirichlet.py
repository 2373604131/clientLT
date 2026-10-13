"""One immutable ordinary-Dirichlet allocation shared by all paper baselines."""
import copy
import math
from pathlib import Path
import re

from tools.benchmarks.common import (REFERENCE_FILES, digest, manifest_rows, read_json,
                                     reference_contract, write_csv, write_json)

PARTITION = 'noniid-labeldir-fine'
GROUPS = {
    'longtail': ('fedavg-lora', 'fedavg-lora-la', 'capt', 'fedpurel', 'fedntd',
                 'fedlf', 'fedyoyo', 'fedrela'),
    'factors': ('ffa-lora', 'rolora', 'fedsvd', 'lora-a2'),
    'ours': ('a', 'ab'),
}
ALL = sum(GROUPS.values(), ())


def select_methods(group='all', methods=None, shard=0, shards=1):
    selected = list(methods if methods is not None else (ALL if group == 'all' else GROUPS[group]))
    if not selected or len(selected) != len(set(selected)) or any(m not in ALL for m in selected):
        raise ValueError('Select distinct registered methods')
    if not 1 <= shards <= len(selected) or not 0 <= shard < shards:
        raise ValueError('Require 1 <= num-shards <= method count and 0 <= shard-index < num-shards')
    return selected[shard::shards]


def partition_rows(source_rows, beta, split_seed=42):
    """Canonical within-class order is recorded; no Client-LT capacities used."""
    import numpy as np
    from utils.datasplit import partition_fine_class_dirichlet
    if not math.isfinite(beta) or beta <= 0 or split_seed != 42:
        raise ValueError('Require finite positive beta and protocol seed42')
    pool = sorted(source_rows, key=lambda r: (r['class_id'], r['raw_sample_id']))
    labels = np.array([r['class_id'] for r in pool], dtype=np.int64)
    allocation, _ = partition_fine_class_dirichlet(
        labels, np.repeat(np.arange(100), 100), 30, 100, beta, split_seed)
    rows = []
    for client in range(30):
        for position, index in enumerate(allocation[client]):
            sample = pool[int(index)]
            rows.append(dict(raw_sample_id=sample['raw_sample_id'], class_id=sample['class_id'],
                             client_id=client, local_position=position))
    return rows


def partition_statistics(rows):
    sizes, counts = [0] * 30, [0] * 100
    for row in rows:
        sizes[row['client_id']] += 1
        counts[row['class_id']] += 1
    return sizes, counts


def validate_partition(root, rows, meta):
    recipe = read_json(Path(root) / 'protocol/partition_protocol.json')
    beta = recipe.get('dirichlet_beta', float('nan'))
    if (recipe.get('schema') != 'paper_dirichlet_reference_v1'
            or recipe.get('topology') != PARTITION or recipe.get('split_seed') != 42
            or not isinstance(beta, (int, float)) or not math.isfinite(beta) or beta <= 0
            or recipe.get('reuse_clientlt_capacities') is not False
            or recipe.get('pool_order') != 'class_id then raw_sample_id'
            or recipe.get('partition_sha256') != digest(rows)):
        raise ValueError('Invalid or changed ordinary-Dirichlet partition recipe')
    sizes, counts = partition_statistics(rows)
    if (min(sizes) < 10 or sizes != meta['client_sample_counts']
            or counts != recipe['global_class_counts']
            or recipe['source_protocol']['pool'] != meta['pool_sha256']
            or meta['resolved_args']['partition'] != PARTITION
            or meta['resolved_args']['split_seed'] != 42
            or meta['resolved_args']['beta'] != beta):
        raise ValueError('Dirichlet metadata/counts disagree with the recorded allocation')
    return recipe


def prepare_reference(source, destination, beta=.5):
    """Generate on CPU; reuse exact images, probes, schedule and init fingerprints.

    This is a derived reference, never a training result. Existing references
    are checked against regeneration, and are never silently replaced.
    """
    from scripts.run_ab_validation import file_lock
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if destination == source or source in destination.parents:
        raise ValueError('The new reference must be outside the existing Client-LT reference')
    original = read_json(source / 'bridge_metadata.json')
    if original['topology'] != 'client-longtail':
        raise ValueError('--reference-run must identify the original Client-LT sample pool')
    source_contract = reference_contract(source)
    source_rows = manifest_rows(source / 'partition_manifest.csv')
    rows = partition_rows(source_rows, beta)
    sizes, counts = partition_statistics(rows)
    if counts != partition_statistics(source_rows)[1]:
        raise ValueError('Dirichlet must preserve every global class count')
    recipe = dict(schema='paper_dirichlet_reference_v1', topology=PARTITION,
                  split_seed=42, dirichlet_beta=beta, clients=30,
                  min_client_train_samples=10, reuse_clientlt_capacities=False,
                  pool_order='class_id then raw_sample_id',
                  implementation='utils.datasplit.partition_fine_class_dirichlet',
                  source_protocol=source_contract, partition_sha256=digest(rows),
                  global_class_counts=counts, client_sample_counts=sizes,
                  steps_per_epoch=sum(math.ceil(n / 32) for n in sizes),
                  initialization='Reuse source fingerprints; every worker verifies actual tensors')
    # Copy only common input metadata. Historical training results, costs and
    # environment must not masquerade as a completed Dirichlet experiment.
    keys = ('seed', 'resolved_args', 'resolved_config', 'classnames', 'initial_lora_sha256',
            'frozen_model_sha256', 'schedule_sha256', 'schedule', 'pool_sha256',
            'test_sha256', 'probe_images_sha256', 'probe_manifest_sha256', 'protocol')
    meta = {key: copy.deepcopy(original[key]) for key in keys}
    meta.update(schema_version='paper_dirichlet_reference_v1', topology=PARTITION,
                method='reference_only_no_training', client_sample_counts=sizes,
                source_reference_files=source_contract['reference_files'])
    meta['resolved_args'].update(partition=PARTITION, split_seed=42, beta=beta,
                                 lac_partition_manifest='')
    for key, value in (('PARTITION', PARTITION), ('BETA', str(beta)),
                       ('SPLIT_SEED', '42'), ('PARTITION_MANIFEST', "''")):
        meta['resolved_config'] = re.sub(r'^  ' + key + r':.*$', '  ' + key + ': ' + value,
                                         meta['resolved_config'], flags=re.MULTILINE)
    with file_lock(destination.parent / 'locks/dirichlet_reference.lock'):
        if destination.exists():
            if (manifest_rows(destination / 'partition_manifest.csv') != rows
                    or read_json(destination / 'bridge_metadata.json') != meta
                    or read_json(destination / 'protocol/partition_protocol.json') != recipe):
                raise ValueError('Existing Dirichlet reference changed; preserve it and use a new output root')
            for name in REFERENCE_FILES[2:]:
                if (destination / name).read_bytes() != (source / name).read_bytes():
                    raise ValueError('Shared schedule/probes changed: ' + name)
        else:
            write_csv(destination / 'partition_manifest.csv', rows)
            write_json(destination / 'bridge_metadata.json', meta)
            write_json(destination / 'protocol/partition_protocol.json', recipe)
            for name in REFERENCE_FILES[2:]:
                target = destination / name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes((source / name).read_bytes())
        reference_contract(destination)
    return destination
