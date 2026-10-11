"""Tail30 partition intervention; reuse the existing joint trainer unchanged.

The historical Tail20 diagnostic/probe protocol is retained. Final summaries add
Head20/Middle50/Tail30 from the same official predictions, without another pass.
"""
import copy
import pickle
from pathlib import Path
import statistics

import numpy as np

from scripts.run_ab_validation import file_lock, write_json
from tools.client_aggregation.protocol import (CONTRACT, PLAIN_A_CONTRACT, METHOD_A_CONTRACT,
    SCHEMA, aggregation_spec, check_aggregation, code_hashes, digest, load_json,
    partition_signature, read_csv, write_table, write_weight_tables)

EXPERIMENT = 'joint_tail30_v1'
ARMS = ('frozen', 'plain_a', 'method_a')
ROOT = Path(__file__).resolve().parents[2]
PARTITION = dict(head_client_ratio=.9, tail_client_ratio=.1,
                 head_class_ratio=.7, tail_class_ratio=.3,
                 specialization_lambda=.75, intra_group_alpha=.5, head_leakage_scale=3.)


def partition_rows(raw_ids, labels, tail_ratio):
    from utils.datasplit import partition_client_longtail
    parameters = dict(PARTITION, head_class_ratio=1-tail_ratio, tail_class_ratio=tail_ratio)
    allocation = partition_client_longtail(labels, 30, 100, **parameters, rng=np.random.RandomState(42))
    return [dict(raw_sample_id=int(raw_ids[i]), class_id=int(labels[i]), client_id=j, local_position=p)
            for j in range(30) for p, i in enumerate(allocation[j])]


def prepare_reference(data_root, destination, source=None):
    """Reconstruct the original pool and verify Tail20 replay before changing its boundary."""
    from utils.functional_coverage_validation import _exact_lt_raw_ids
    from tools.sfra.simple_controls import probe_manifest_replay
    source = Path(source or ROOT/'references/full10_clientlt')
    destination = Path(destination)
    with (Path(data_root)/'cifar-100/cifar-100-python/train').open('rb') as f:
        train = pickle.load(f, encoding='latin1')
    labels = np.asarray(train['fine_labels'], dtype=np.int64)
    raw_ids = _exact_lt_raw_ids(labels, .01, 'exp')
    labels = labels[raw_ids]
    old = read_csv(source/'partition_manifest.csv')
    if partition_signature(partition_rows(raw_ids, labels, .2)) != partition_signature(old):
        raise ValueError('Seed42 Tail20 reconstruction differs from the bundled protocol')
    rows = partition_rows(raw_ids, labels, .3)
    if sorted((int(r['raw_sample_id']), int(r['class_id'])) for r in old) != sorted(
            (r['raw_sample_id'], r['class_id']) for r in rows):
        raise ValueError('Tail30 changed the raw training pool')
    matrix = np.zeros((30,100), dtype=np.int64)
    for r in rows:
        matrix[r['client_id'], r['class_id']] += 1
    if matrix.sum() != 10847 or matrix[:,70:].sum() != 305 or (matrix.sum(1) <= 0).any():
        raise ValueError('Unexpected Tail30 pool or empty client')
    source_meta = load_json(source/'bridge_metadata.json')
    meta = copy.deepcopy(source_meta)
    meta['client_sample_counts'] = matrix.sum(1).tolist()
    meta['resolved_args'].update(PARTITION)
    # Reference hashes for frozen model/initialization/test/probes remain inherited
    # and are checked against the actual model/data by the existing runtime.
    meta.pop('resolved_config', None)
    meta.pop('control_config', None)
    meta.pop('environment', None)
    receipt = dict(experiment=EXPERIMENT, seed=42, partition_parameters=PARTITION,
        partition_tail_ids=list(range(70,100)), diagnostic_tail_ids=list(range(80,100)),
        report_groups=dict(Head20=list(range(20)), Middle50=list(range(20,70)), Tail30=list(range(70,100))),
        source='bundled references/full10_clientlt; no completed control run required',
        source_partition_sha256=partition_signature(old), partition_sha256=partition_signature(rows),
        pool_sha256=source_meta['pool_sha256'], same_training_pool=True,
        tail20_reconstruction_matches=True, matrix_sha256=digest(matrix.tolist()),
        class_counts=matrix.sum(0).tolist(), client_sizes=matrix.sum(1).tolist(),
        steps_per_epoch=sum((int(n)+31)//32 for n in matrix.sum(1)),
        training_tail_samples=305, diagnostic_tail20_samples=153)
    meta['partition_intervention'] = receipt
    files = {name:(source/name).read_bytes() for name in ('protocol/full_schedule.json','protocol/eri_protocol.json')}
    files['protocol/probe_manifest.csv'], _ = probe_manifest_replay(
        source/'protocol/probe_manifest.csv', meta['probe_manifest_sha256'])
    with file_lock(destination.parent/'locks/reference.lock'):
        if (destination/'tail30_partition.json').exists():
            if load_json(destination/'tail30_partition.json') != receipt:
                raise ValueError('Registered Tail30 partition changed; use a new output root')
            if read_csv(destination/'partition_manifest.csv') != [dict((k,str(v)) for k,v in r.items()) for r in rows]:
                raise ValueError('Registered Tail30 manifest changed')
            if load_json(destination/'bridge_metadata.json') != meta:
                raise ValueError('Registered Tail30 reference metadata changed')
            if any((destination/n).read_bytes() != b for n,b in files.items()):
                raise ValueError('Registered Tail30 schedule/probe changed')
            return receipt
        if destination.exists() and any(destination.iterdir()):
            raise ValueError('Incomplete/unregistered reference directory; preserve it and choose a new output root')
        for name, payload in files.items():
            target = destination/name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(payload)
        write_table(destination/'partition_manifest.csv', rows)
        write_json(destination/'bridge_metadata.json', meta)
        write_json(destination/'tail30_partition.json', receipt)
    return receipt


def validate_spec(spec):
    if spec.get('rule','joint') != 'joint' or spec['lambda_value'] != .1:
        raise ValueError('Tail30 permits only original joint lambda=0.1')
    info = spec.get('partition_intervention', {})
    if (info.get('experiment') != EXPERIMENT or info.get('partition_parameters') != PARTITION
            or info.get('partition_sha256') != spec['partition_sha256']
            or info.get('matrix_sha256') != spec['matrix_sha256']
            or info.get('client_sizes') != spec['counts']['sizes']
            or info.get('class_counts') != spec['counts']['class_counts']):
        raise ValueError('Tail30 intervention identity mismatch')
    check_aggregation(spec)


def contract_for(arm, counts):
    if arm not in ARMS:
        raise ValueError('Tail30 supports frozen, plain_a and method_a only; no AB/Av2')
    base = {'frozen':CONTRACT, 'plain_a':PLAIN_A_CONTRACT, 'method_a':METHOD_A_CONTRACT}[arm]
    steps = sum((n+31)//32 for n in counts['sizes'])
    result = copy.deepcopy(base)
    if arm == 'frozen':
        for key in ('ab','B','A','B_ROUNDS'):
            result.pop(key, None)
    result.update(experiment=EXPERIMENT, arms=[arm], partition_parameters=PARTITION,
        normal_B_steps=300*steps, extra_A_steps=0 if arm=='frozen' else 90*steps,
        total_local_steps=(300 if arm=='frozen' else 390)*steps,
        primary='rounds81..100 Overall/Head20/Middle50/Tail30; retain Tail20/Middle60 for comparison',
        diagnostics='unchanged Tail20 probes/online metrics; Tail30 from saved predictions only',
        interpretation='Tail30 data partition; same raw pool; original joint re-solved; no B/C transfer')
    return result


def register(args):
    root = args.output_root.resolve()
    receipt = prepare_reference(args.data_root, root/'reference')
    with file_lock(root/'locks/tail30_plan.lock'):
        # Reuse a persisted solution exactly, including across cross-platform resume.
        existing = next((root/a/'experiment_plan.json' for a in ARMS
                         if (root/a/'experiment_plan.json').exists()), None)
        if existing is None:
            spec = aggregation_spec(root/'reference', .1)
            spec['partition_intervention'] = receipt
            spec['solver'].pop('seconds', None)
        else:
            from tools.client_aggregation.protocol import count_matrix
            spec = load_json(existing)['aggregation']
            fingerprint, counts, matrix = count_matrix(root/'reference')
            if (spec['input_fingerprint'] != fingerprint or spec['counts'] != counts
                    or spec['matrix_sha256'] != digest(matrix.tolist())
                    or spec['partition_intervention'] != receipt):
                raise ValueError('Tail30 aggregation inputs changed')
            validate_spec(spec)
        hashes = code_hashes(ROOT)
        plans = {}
        for arm in ARMS:
            settings = dict(seed=42, lambda_value=.1, reference_run=str(root/'reference'),
                data_root=str(args.data_root.resolve()), num_workers=args.num_workers,
                client_concurrency=args.client_concurrency, parallel_validation='off',
                partition_experiment=EXPERIMENT)
            if args.cuda_policy != 'legacy':
                settings['cuda_policy'] = args.cuda_policy
            plan = dict(schema=SCHEMA, contract=contract_for(arm, spec['counts']), settings=settings,
                        code_sha256=hashes, aggregation=spec)
            path = root/arm/'experiment_plan.json'
            if path.exists():
                saved = load_json(path)
                # Resume uses the exact persisted fp64 weights, while rechecking inputs.
                if any(saved[k] != plan[k] for k in ('schema','contract','settings','code_sha256')):
                    raise ValueError('Tail30 plan/source/settings changed; use a new output root')
                for key in ('matrix_sha256','partition_sha256','input_fingerprint','partition_intervention'):
                    if saved['aggregation'][key] != spec[key]:
                        raise ValueError('Tail30 aggregation inputs changed')
                validate_spec(saved['aggregation'])
                plan = saved
            else:
                validate_spec(spec)
                write_json(path, plan)
                write_weight_tables(path.parent, spec)
            plans[arm] = plan
        if len({p['aggregation']['weights_sha256'] for p in plans.values()}) != 1:
            raise ValueError('Three arms must use identical joint weights')
    return plans


def status(root):
    rows = []
    for arm in ARMS:
        run = Path(root)/arm/'runs/seed42'/arm
        path = run/('completion.json' if (run/'completion.json').exists() else 'progress.json')
        done = load_json(path) if path.exists() else {}
        rows.append(dict(arm=arm, completed_round=done.get('completed_round',0),
                         status='finished_pending_audit' if path.name=='completion.json' else 'incomplete' if done else 'missing'))
    return rows


def summarize(root):
    from scripts.run_cliplora_joint_aggregation import make_job
    from tools.client_aggregation.analysis import audit_run
    root = Path(root)
    rows, results = [], {}
    for row in status(root):
        arm = row['arm']
        if row['status'] != 'finished_pending_audit':
            rows.append(row)
            continue
        job = make_job(load_json(root/arm/'experiment_plan.json'), arm)
        result = audit_run(root/arm/job['run'], job)
        results[arm] = result
        groups = dict(Overall=range(100), Head20=range(20), Middle50=range(20,70),
                      Tail30=range(70,100), Middle60=range(20,80), Tail20=range(80,100))
        perf = {}
        for name, ids in groups.items():
            values = []
            for rnd in range(81,101):
                p = result['predictions'][rnd]
                values.append(100*float(p['correct'][np.isin(p['class_id'], list(ids))].mean()))
            perf[name] = statistics.mean(values)
        done = result['progress']
        rows.append(dict(row, status='complete', **perf, seed=42,
            local_steps=done['total_local_optimizer_steps'], correction_steps=done['functional_correction_steps'],
            elapsed_hours=done['elapsed_seconds']/3600,
            environment=str(result['metadata'].get('environment',{}))))
    if len(results)>1:
        first = next(iter(results.values()))
        for result in results.values():
            for key in ('pool_sha256','test_sha256','initial_lora_sha256','frozen_model_sha256','schedule_sha256'):
                if result['metadata'][key] != first['metadata'][key]:
                    raise ValueError('Tail30 cross-arm identity mismatch: '+key)
            if not np.array_equal(first['predictions'][0]['prediction'],result['predictions'][0]['prediction']):
                raise ValueError('Tail30 arms have different initial predictions')
    write_table(root/'analysis/performance.csv', rows)
    write_json(root/'analysis/report_protocol.json', dict(experiment=EXPERIMENT,
        partition_tail_ids=list(range(70,100)), window='rounds81-100',
        missing_controls='No automatic old-control reruns; compare separately with Tail20-partition results',
        independent_seeds=1, budget_matched=False))
    return rows
