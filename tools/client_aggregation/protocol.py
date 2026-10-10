"""Immutable protocol: two trajectories, one aggregation rule, no test selection."""
from pathlib import Path

import numpy as np

from tools.sfra.maintext import FROZEN, PAIR_HASHES, digest, load_json
from tools.sfra.ab_validation import code_hashes as ab_code_hashes
from tools.sfra.simple_controls import protocol_info, read_csv, write_table, tail_weights
from tools.client_aggregation.weights import audit_weights, solve_weights

SCHEMA = 'joint_client_aggregation_v1'
ARMS = ('frozen', 'ab')
DIAGNOSTIC_ROUNDS = (1, 20, 30, 50, 70, 80, 90, 100)
B_ROUNDS = (30, 40, 50, 60, 70, 80, 90, 100)
FILES = ('tools/client_aggregation/__init__.py', 'tools/client_aggregation/weights.py',
    'tools/client_aggregation/protocol.py', 'tools/client_aggregation/runtime.py',
    'tools/client_aggregation/analysis.py', 'tools/client_aggregation/plots.py',
    'tools/client_aggregation/parallel.py', 'scripts/run_cliplora_joint_aggregation.py',
    'scripts/train_cliplora_joint_aggregation.py', 'tools/sfra/simple_controls.py',
    'scripts/run_method_a_simple_controls.py', 'Dassl/dassl/data/data_manager.py',
    'Dassl/dassl/data/samplers.py', 'Dassl/dassl/optim/optimizer.py',
    'Dassl/dassl/optim/lr_scheduler.py', 'utils/cliplora_loss.py',
    'tools/client_aggregation/frozen_tailrw.py',
    'scripts/run_cliplora_frozen_aggregation.py', 'tools/client_aggregation/frozen_controls.py',
    'tools/client_aggregation/numerics.py',
    'scripts/run_cliplora_plain_a_aggregation.py', 'tools/client_aggregation/plain_a.py')
CONTRACT = dict(schema=SCHEMA, seed=42, rounds=100, clients=30, partition='client-longtail',
    arms=list(ARMS), aggregation='KL(uniform || R.T @ w) + lambda/2 * chi_square(w || sample_weights)',
    scope='ordinary B aggregation every round; ordinary A aggregation when A is opened',
    frozen='initial LoRA A fixed for all 100 rounds; ordinary B only; no functional repair or transfer',
    ab='Full-CP lambda10 mu1, A rounds1..90; original shared donor C, class-cyclic, tail_weight0.35',
    A=FROZEN, B=dict(rounds=list(B_ROUNDS), learning_rate=.3, probe_step=.1,
        regularization=.001, steps=2, non_tail_sampling='class-cyclic', tail_weight=.35),
    primary='committed rounds81..100 Tail20 mean; Overall/Head20/Middle60 reported jointly',
    interpretation='bundled A-learning + retention + shared-B effect; not budget-matched or separate A/B attribution',
    diagnostics='read-only official test; never fed to optimizer, weights, history, gates, or checkpoint selection',
    diagnostics_rounds=list(DIAGNOSTIC_ROUNDS), lambda_policy='default0.1 is a fixed exploratory setting, not test-selected',
    unchanged='LA training prior, local loss, optimizer, source priority, CP sample weighting and B recipient objective')

# Separate opt-in experiment; ARMS and the legacy two-arm default stay unchanged.
PLAIN_A_CONTRACT = dict(schema=SCHEMA, experiment='plain_a_aggregation_v1', seed=42,
    rounds=100, clients=30, arms=['plain_a'], partition='client-longtail',
    phase_order='local B, aggregate B, local A, aggregate A, official evaluation',
    B_rounds=list(range(1, 101)), B_epochs=3,
    A_rounds=list(range(1, 91)), A_epochs=1, A_lr=.001, A_momentum=.9, A_weight_decay=0.,
    A_optimizer='fresh SGD per client per phase; fixed lr; B frozen during A training',
    final_rounds='91..100: B only, retain round90 A',
    aggregation_scope='same static client weights for both B and A phases',
    local_loss='unchanged global-count LA, tau=1',
    normal_B_steps=105600, extra_A_steps=31680, total_local_steps=137280,
    functional_correction=False, source_C_transfer=False, adaptive_A_schedule=False,
    primary='rounds81..100 mean Overall; Head20/Middle60/Tail20 jointly',
    diagnostics_rounds=list(DIAGNOSTIC_ROUNDS),
    diagnostics='read-only test; no feedback, scheduling, weight fitting or checkpoint selection',
    interpretation='aggregation-only contrast among three ordinary-A runs; not a budget-matched frozen-A contrast')


def validate_arm(job):
    if job['arm'] == 'plain_a':
        if job.get('contract') != PLAIN_A_CONTRACT:
            raise ValueError('Plain A requires its separate fixed-schedule contract')
    elif job['arm'] not in ARMS:
        raise ValueError('Unknown experiment arm: ' + str(job['arm']))
    elif job['aggregation'].get('rule') in ('tailrw16', 'fedavg') and job['arm'] != 'frozen':
        raise ValueError('Legacy aggregation controls only support permanently frozen A')


def code_hashes(repo):
    import hashlib
    values = ab_code_hashes(repo)
    for name in FILES:
        values[name] = hashlib.sha256((Path(repo) / name).read_text(encoding='utf-8').encode()).hexdigest()
    return values


def count_matrix(reference):
    fingerprint, counts = protocol_info(reference)
    matrix = np.zeros((30, 100), dtype=np.int64)
    for row in read_csv(Path(reference) / 'partition_manifest.csv'):
        matrix[int(row['client_id']), int(row['class_id'])] += 1
    return fingerprint, counts, matrix


def partition_signature(rows):
    return digest(sorted(tuple(int(row[k]) for k in ('client_id', 'local_position', 'raw_sample_id', 'class_id')) for row in rows))


def aggregation_name(spec):
    rule = spec.get('rule', 'joint')
    if rule == 'joint':
        return 'joint_class_distribution_client_weights'
    if rule == 'tailrw16':
        return 'tail_count_reweighted_clients_gamma16'
    if rule == 'fedavg':
        return 'sample_count_weighted_fedavg'
    raise ValueError('Unknown aggregation rule: ' + str(rule))


def tailrw_statistics(matrix, tail_ids):
    """Exactly the historical gamma=16 rule, derived from training counts only."""
    from tools.client_aggregation.weights import problem
    q, local, target = problem(matrix, 1.)  # Validate counts; no convex optimization.
    matrix = np.asarray(matrix)
    if (len(tail_ids) != 20 or len(set(tail_ids)) != 20
            or any(not isinstance(c, int) or c < 0 or c >= matrix.shape[1] for c in tail_ids)):
        raise ValueError('TailRW16 requires 20 distinct valid training-defined tail classes')
    weights = tail_weights(matrix.sum(1).tolist(), matrix[:, tail_ids].sum(1).tolist(), 16.)
    return dict(weights=weights, sample_weights=q.tolist(), mixture=(local.T @ weights).tolist(),
        target=target.tolist(), gamma=16., solver='closed-form historical TailRW gamma16',
        formula='(n_j + 16*n_j_tail)/(N + 16*N_tail)')


def aggregation_spec(reference, strength, rule='joint'):
    fingerprint, counts, matrix = count_matrix(reference)
    if rule not in ('joint', 'tailrw16', 'fedavg'):
        raise ValueError('Unknown aggregation rule: ' + str(rule))
    solution = (solve_weights(matrix, strength) if rule == 'joint' else
                fedavg_statistics(matrix) if rule == 'fedavg' else
                tailrw_statistics(matrix, counts['tail_ids']))
    weights = solution['weights']
    spec = dict(input_fingerprint=fingerprint, counts=counts, matrix=matrix.tolist(),
        partition_sha256=partition_signature(read_csv(Path(reference) / 'partition_manifest.csv')),
        lambda_value=strength if rule == 'joint' else None, weights=weights, weights_sha256=digest(weights),
        matrix_sha256=digest(matrix.tolist()), solver=solution)
    if rule == 'tailrw16':
        spec.update(rule=rule, gamma=16.)
    elif rule == 'fedavg':
        spec.update(rule=rule)
    return spec


def fedavg_statistics(matrix):
    from tools.client_aggregation.weights import problem
    q, local, target = problem(matrix, 1.)
    return dict(weights=q.tolist(), sample_weights=q.tolist(), mixture=(local.T @ q).tolist(),
        target=target.tolist(), solver='closed-form sample counts', formula='n_j/N')


def check_aggregation(spec):
    if digest(spec['weights']) != spec['weights_sha256'] or digest(spec['matrix']) != spec['matrix_sha256']:
        raise ValueError('Frozen weights/counts digest mismatch')
    aggregation_name(spec)  # Reject unrecognized policies, including in archived jobs.
    if spec.get('rule') == 'fedavg':
        expected = fedavg_statistics(spec['matrix'])
        if (spec['lambda_value'] is not None or spec.get('gamma') is not None
                or np.asarray(spec['weights']).shape != np.asarray(expected['weights']).shape
                or not np.allclose(spec['weights'], expected['weights'], rtol=0, atol=1e-12)):
            raise ValueError('FedAvg weights differ from n_j/N')
        return expected
    if spec.get('rule') == 'tailrw16':
        expected = tailrw_statistics(spec['matrix'], spec['counts']['tail_ids'])
        if (spec.get('gamma') != 16. or spec['lambda_value'] is not None
                or np.asarray(spec['weights']).shape != np.asarray(expected['weights']).shape
                or not np.allclose(spec['weights'], expected['weights'], rtol=0, atol=1e-12)):
            raise ValueError('TailRW16 weights differ from the historical count formula')
        return expected
    return audit_weights(spec['weights'], spec['matrix'], spec['lambda_value'])


def client_execution_config(job):
    from tools.client_aggregation.numerics import policy_for, DETERMINISTIC
    slots = job.get('settings', {}).get('client_concurrency', 1)
    if slots in (4, 6):
        from tools.client_aggregation.parallel import EXECUTION
        result = dict(EXECUTION, max_concurrent_clients=slots)
    elif slots == 1:
        result = dict(version=1, backend='original_serial', max_concurrent_clients=1)
    else:
        raise ValueError('Only serial, four-client or six-client execution is supported')
    if policy_for(job) == 'deterministic':
        result['cuda_numerics'] = dict(DETERMINISTIC)
    if job.get('settings', {}).get('parallel_validation') == 'off':
        result['parallel_validation'] = 'off'
    return result


def audit_runtime_config(run, job):
    validate_arm(job)
    cfg = load_json(Path(run) / 'sfra_config.json')
    expected_execution = client_execution_config(job)
    if (cfg.get('client_execution') != expected_execution
            or load_json(Path(run) / 'client_execution.json') != expected_execution):
        raise ValueError('Client execution mode differs from the registered experiment')
    if 'cuda_numerics' in expected_execution:
        from tools.client_aggregation.numerics import verify_record
        verify_record(load_json(Path(run) / 'cuda_numerics.json'))
    frozen = job['arm'] == 'frozen'
    plain = job['arm'] == 'plain_a'
    if cfg.get('joint_aggregation') != dict(schema=SCHEMA, arm=job['arm'], mode=job['mode'],
            lambda_value=job['aggregation']['lambda_value'], weights_sha256=job['aggregation']['weights_sha256']):
        raise ValueError('Runtime aggregation differs from registered job')
    expected = dict(FROZEN, variant='s' if frozen or plain else 'full-cp',
        refresh_rounds=[] if frozen else list(range(1, 91)))
    if plain:
        expected.update(retention_weight=0., correction_steps=0, commit_rule='ordinary_A_without_correction')
    elif not frozen:
        expected['classification_weight'] = 1.
    for key, value in expected.items():
        if cfg.get(key) != value:
            raise ValueError('A runtime configuration mismatch: ' + key)
    base = cfg['base_training_config']
    expected_base = dict(seed=42, protocol_seed=42, topology='client-longtail', la_tau=1.,
        aggregation=aggregation_name(job['aggregation']), normal_steps_expected=105600,
        extra_steps_expected=0 if frozen else 31680)
    if plain:
        expected_base.update(normal_trainable_factor='B', extra_trainable_factor='A',
            candidate_rounds=list(range(1, 91)), control_enabled=False,
            a_lr_mult=1., extra_a_lr=.001, extra_weight_decay=0., normal_b_lr=.001)
    for key, value in expected_base.items():
        if base.get(key) != value:
            raise ValueError('Base training mismatch: ' + key)
    transfer = cfg.get('b_transfer')
    if (frozen or plain) and transfer:
        raise ValueError('Frozen/plain-A arm must not execute B transfer')
    if job['arm'] == 'ab':
        wanted = dict(CONTRACT['B'], rounds=[1] if job['mode'] == 'smoke' else list(B_ROUNDS), mode='shared')
        if not transfer or any(transfer.get(k) != v for k, v in wanted.items()):
            raise ValueError('Shared B does not match the frozen AB method')
    execution = load_json(Path(run) / 'execution_config.json')
    if any(execution.get(k) != v for k, v in dict(version=2, feedback_forward_batch_size=128, device_cache_gib=4.).items()):
        raise ValueError('Execution must be fast-v2 f128 c4')
    return cfg


def write_weight_tables(root, spec):
    check = check_aggregation(spec)
    matrix = np.asarray(spec['matrix'])
    write_table(Path(root) / 'client_weights.csv', [dict(client_id=j, n=int(matrix[j].sum()),
        n_tail=int(matrix[j, spec['counts']['tail_ids']].sum()), sample_weight=check['sample_weights'][j],
        weight=spec['weights'][j]) for j in range(30)])
    write_table(Path(root) / 'class_support_proxy.csv', [dict(class_id=c, n=int(matrix[:, c].sum()),
        holders=int((matrix[:, c] > 0).sum()), sample_mixture=float(matrix[:, c].sum() / matrix.sum()),
        joint_mixture=check['mixture'][c], target=check['target'][c]) for c in range(100)])
