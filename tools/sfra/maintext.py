"""Standard-library-only, paired summaries for the frozen Method A experiments."""
import hashlib
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

from tools.sfra.summary import METRICS, read_csv, write_csv


METHODS = ('s', 'full-cp', 'flat-cp', 'current-cp', 'full')
LABELS = {'s': 'Same-frequency S', 'full-cp': 'Method A (Full-CP)',
          'flat-cp': 'w/o source priority', 'current-cp': 'w/o history', 'full': 'w/o CP'}
PLAN_NAME = 'maintext_plan.json'
SCHEMA = 'method_a_maintext_v1'
FROZEN = dict(retention_weight=10., correction_steps=3, correction_step_size=.1,
              current_gain_fraction=.5, history_window=5, history_improvement=.001,
              witness_per_local_class=8, witness_views=2, positive_response_threshold=1e-6,
              sigma_floor=.001, radius='unweighted_client_proposal_RMS',
              source_coordinates='untruncated_reduced_QR', precision='fp32',
              commit_rule='fixed_third_step', refresh_rounds=list(range(1, 91)))
PAIR_HASHES = ('pool_sha256', 'test_sha256', 'schedule_sha256', 'frozen_model_sha256',
               'initial_lora_sha256', 'probe_images_sha256', 'probe_manifest_sha256')
TRAINING_FILES = ('federated_main.py', 'trainers/cliplora.py', 'utils/sfra_math.py',
                  'utils/sfra_feedback.py', 'utils/cliplora_sfra.py', 'utils/cliplora_la_control.py',
                  'utils/cliplora_a_refresh.py', 'utils/datasplit.py', 'utils/loralib/layers.py',
                  'utils/lora_aggregation.py', 'scripts/cliplora_fresh_protocol.py',
                  'scripts/run_cliplora_sfra.py', 'scripts/run_cliplora_a_refresh.py',
                  'configs/trainers/PromptFL/vit_b16.yaml', 'configs/datasets/cifar100_LT.yaml')


def load_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


def code_hashes(repo):
    return {name: hashlib.sha256((Path(repo)/name).read_text(encoding='utf-8').encode()).hexdigest()
            for name in TRAINING_FILES}


def block_key(job):
    return job['partition'], job['protocol_seed'], job['seed'], job['dirichlet_beta']


def validate_config(run, job):
    cfg = load_json(Path(run)/'sfra_config.json')
    expected = {**FROZEN, 'variant': job['method'], 'seed': job['seed'],
                'protocol_seed': job['protocol_seed'], 'partition': job['partition'],
                'witness_batch_size': job['witness_batch_size']}
    if job['method'].endswith('-cp'):
        expected['classification_weight'] = 1.
    for name, value in expected.items():
        if cfg.get(name) != value:
            raise ValueError(f'{run}: {name} differs: {cfg.get(name)!r} != {value!r}')
    if cfg.get('b_transfer'):
        raise ValueError(f'{run}: B transfer is outside the Method A suite')
    if not cfg.get('a_parameter_order') or not cfg.get('base_training_config'):
        raise ValueError(f'{run}: missing model/training configuration')
    if cfg['base_training_config'].get('dirichlet_beta') != job['dirichlet_beta']:
        raise ValueError(f'{run}: Dirichlet beta differs from plan')
    return cfg


def check_receipt(run, job):
    receipt = load_json(Path(run)/'maintext_run.json')
    if receipt.get('schema_version') != SCHEMA or receipt.get('job') != job:
        raise ValueError(f'{run}: run provenance differs from the planned job')
    if not receipt.get('code_unchanged'):
        raise ValueError(f'{run}: training source changed during execution')


def _finite(value):
    value = float(value)
    if not math.isfinite(value):
        raise ValueError('nonfinite metric')
    return value


def read_result(run, job):
    """Only complete, registered runs with exactly 101 official evaluations qualify."""
    cfg = validate_config(run, job)
    check_receipt(run, job)
    complete = load_json(run/'completion.json')
    if complete.get('completed_round') != 100 or complete.get('variant') != job['method']:
        raise ValueError(f'{run}: invalid completion marker')
    if complete.get('official_test_passes') != 101:
        raise ValueError(f'{run}: expected 101 official evaluations')
    for name, expected in (('normal_optimizer_steps', 'normal_steps_expected'),
                           ('extra_optimizer_steps', 'extra_steps_expected')):
        if complete.get(name) != cfg['base_training_config'].get(expected):
            raise ValueError(f'{run}: unexpected training budget: {name}')
    metrics = sorted(read_csv(run/'round_metrics.csv'), key=lambda r: int(r['round']))
    if [int(r['round']) for r in metrics] != list(range(101)):
        raise ValueError(f'{run}: missing or duplicate official rounds')
    for row in metrics:
        for name in METRICS:
            value = _finite(row[name])
            if not 0 <= value <= 100:
                raise ValueError(f'{run}: accuracy outside [0,100]')
    result = dict(run=job['run'], method=job['method'], partition=job['partition'],
                  seed=job['seed'], protocol_seed=job['protocol_seed'], dirichlet_beta=job['dirichlet_beta'])
    for name in METRICS:
        result['last20_'+name] = statistics.mean(float(r[name]) for r in metrics[81:101])
    tail = [float(r['bottom20_tail_acc']) for r in metrics]
    result.update(tail_round90=tail[90], tail_round100=tail[100],
                  tail_change_90_to_100=tail[100]-tail[90],
                  tail_peak_to_final=max(tail[1:])-tail[100],
                  simulation_wall_seconds=_finite(complete['elapsed_seconds']))
    budget = read_csv(run/'budget.csv')
    costs = read_csv(run/'sfra_costs.csv')
    evaluations = read_csv(run/'evaluation_budget.csv')
    total = lambda rows, key: sum(_finite(r.get(key) or 0) for r in rows)
    result['modeled_upload_mib'] = (total(budget, 'upload_bytes') + total(costs, 'upload_bytes')
                                    + total(evaluations, 'upload_bytes')) / 2**20
    result['modeled_downlink_mib'] = (total(budget, 'modeled_downlink_bytes') + total(costs, 'downlink_bytes')
                                      + total(evaluations, 'modeled_downlink_bytes')) / 2**20
    result['functional_seconds'] = total(costs, 'seconds')
    result['local_optimizer_steps'] = complete['normal_optimizer_steps'] + complete['extra_optimizer_steps']
    result['functional_correction_steps'] = complete['functional_correction_steps']
    rounds = read_csv(run/'sfra_rounds.csv')
    if job['method'] != 's' and sorted(int(r['round']) for r in rounds) != list(range(1, 101)):
        raise ValueError(f'{run}: missing or duplicate functional summary rounds')
    increases = [_finite(r['committed_classification_loss_increase']) for r in rounds
                 if int(r['round']) <= 90 and r.get('committed_classification_loss_increase') not in ('', None)]
    if job['method'].endswith('-cp') and len(increases) != 90:
        raise ValueError(f'{run}: missing final CP diagnostics for active rounds')
    result['cp_loss_increase_rate'] = statistics.mean(x > 1e-8 for x in increases) if increases else ''
    result['cp_positive_loss_increase_mean'] = statistics.mean(max(0., x) for x in increases) if increases else ''
    metadata = load_json(run/'bridge_metadata.json')
    signature = {}
    for name in PAIR_HASHES:
        if not metadata.get(name):
            raise ValueError(f'{run}: missing protocol fingerprint {name}')
        signature[name] = metadata[name]
    partition = read_csv(run/'partition_manifest.csv')
    if not partition:
        raise ValueError(f'{run}: empty partition manifest')
    signature['partition_manifest'] = digest(sorted(partition, key=lambda r: (int(r['client_id']), int(r['local_position']))))
    signature['training_config'] = digest(cfg['base_training_config'])
    signature['a_parameter_order'] = digest(cfg['a_parameter_order'])
    signature['code_sha256'] = digest(job['code_sha256'])
    witness = None if job['method'] == 's' else digest(load_json(run/'private_witness_manifest.json'))
    return result, signature, witness


def summarize_maintext(root):
    root = Path(root).resolve()
    plan = load_json(root/PLAN_NAME)
    if plan.get('schema_version') != SCHEMA:
        raise ValueError('Unrecognized Method A plan')
    out = root/'maintext_analysis'
    out.mkdir(parents=True, exist_ok=True)
    statuses, records, signatures, witnesses, blocks = {}, {}, {}, {}, defaultdict(list)
    for job in plan['jobs']:
        key = job['run']
        run = (root/key).resolve()
        if not run.is_relative_to(root):
            raise ValueError('Planned run escapes the suite directory')
        blocks[block_key(job)].append(key)
        statuses[key] = dict(run=key, method=job['method'], seed=job['seed'], partition=job['partition'],
                             status='missing', reason='')
        if not run.exists():
            continue
        if not (run/'completion.json').is_file():
            statuses[key]['status'] = 'incomplete'
            continue
        try:
            records[key], signatures[key], witnesses[key] = read_result(run, job)
            statuses[key]['status'] = 'complete'
        except (OSError, ValueError, KeyError, TypeError) as error:
            statuses[key].update(status='invalid', reason=str(error))
    audits, included = [], set()
    for block, keys in sorted(blocks.items()):
        ready = [k for k in keys if statuses[k]['status'] == 'complete']
        mismatch = []
        if ready:
            reference = signatures[ready[0]]
            mismatch = [name for name in reference if any(signatures[k][name] != reference[name] for k in ready)]
            witness_values = {witnesses[k] for k in ready if witnesses[k] is not None}
            if len(witness_values) > 1:
                mismatch.append('witness_manifest')
        if mismatch:
            for key in ready:
                statuses[key].update(status='invalid_pair', reason=', '.join(mismatch))
        methods = {records[k]['method'] for k in ready}
        eligible = not mismatch and len(ready) == len(METHODS) and methods == set(METHODS)
        if eligible:
            included.update(ready)
        audits.append(dict(partition=block[0], protocol_seed=block[1], seed=block[2], dirichlet_beta=block[3],
                           completed_methods=len(ready), eligible=eligible, mismatches=', '.join(mismatch)))
    available = [records[k] for k in sorted(records) if statuses[k]['status'] == 'complete']
    groups = defaultdict(list)
    for key in sorted(included):
        r = records[key]
        groups[(r['partition'], r['protocol_seed'], r['dirichlet_beta'], r['method'])].append(r)
    numeric = ['last20_'+name for name in METRICS] + [
        'tail_peak_to_final', 'tail_change_90_to_100', 'simulation_wall_seconds',
        'modeled_upload_mib', 'modeled_downlink_mib', 'functional_seconds']
    aggregates = []
    for (partition, protocol_seed, beta, method), rows in sorted(groups.items()):
        item = dict(partition=partition, protocol_seed=protocol_seed, dirichlet_beta=beta, method=method,
                    seeds=','.join(str(r['seed']) for r in rows), n=len(rows))
        for name in numeric:
            values = [r[name] for r in rows]
            item[name+'_mean'] = statistics.mean(values)
            item[name+'_sd'] = statistics.stdev(values) if len(values) > 1 else ''
        aggregates.append(item)
    differences = defaultdict(list)
    for block, keys in sorted(blocks.items()):
        if not set(keys).issubset(included):
            continue
        rows = {records[k]['method']: records[k] for k in keys}
        for method in METHODS:
            if method == 'full-cp':
                continue
            differences[(block[0], block[1], block[3], method)].append({
                'seed': block[2], **{name: rows['full-cp'][name]-rows[method][name]
                                   for name in numeric[:len(METRICS)+2]}})
    paired = []
    for (partition, protocol_seed, beta, method), rows in sorted(differences.items()):
        item = dict(partition=partition, protocol_seed=protocol_seed, dirichlet_beta=beta,
                    comparison='full-cp minus '+method, n=len(rows), seeds=','.join(str(r['seed']) for r in rows))
        for name in numeric[:len(METRICS)+2]:
            values = [r[name] for r in rows]
            item[name+'_delta_mean'] = statistics.mean(values)
            item[name+'_delta_sd'] = statistics.stdev(values) if len(values) > 1 else ''
        paired.append(item)
    for name, rows in [('status', list(statuses.values())), ('protocol_audit', audits),
                       ('per_run', available), ('main_table', [r for r in aggregates if r['method'] in ('s', 'full-cp')]),
                       ('ablation_table', [r for r in aggregates if r['method'] != 's']),
                       ('all_methods', aggregates), ('paired_differences', paired)]:
        write_csv(out/f'{name}.csv', rows)
    lines = ['# Method A main-text experiments', '',
             'Official endpoint: mean accuracy of committed rounds 81–100. No best-checkpoint selection.',
             'Tables use only complete five-method blocks with matching protocols and witnesses.',
             'Individual complete runs remain in per_run.csv while other methods are pending.', '',
             f'Eligible seed/partition blocks: {sum(a["eligible"] for a in audits)} / {len(audits)}.',
             'Values are mean +/- sample SD across training seeds; a single seed has no SD.', '']
    for group in sorted({(r['partition'], r['protocol_seed'], r['dirichlet_beta']) for r in aggregates}):
        lines += [f'## {group[0]} / protocol seed {group[1]} / beta {group[2]:g}', '',
                  '| Method | n | Overall | Head20 | Middle60 | Tail20 | Tail peak-to-final | Down/up MiB |',
                  '|---|---:|---:|---:|---:|---:|---:|---:|']
        def cell(row, name):
            mean, sd = row[name+'_mean'], row[name+'_sd']
            return f'{mean:.3f}' if sd == '' else f'{mean:.3f} +/- {sd:.3f}'
        subset = [r for r in aggregates if (r['partition'], r['protocol_seed'], r['dirichlet_beta']) == group]
        for row in sorted(subset, key=lambda r: METHODS.index(r['method'])):
            cells = [cell(row, 'last20_'+name) for name in METRICS[:4]]
            lines.append(f'| {LABELS[row["method"]]} | {row["n"]} | '+ ' | '.join(cells)
                         + f' | {cell(row, "tail_peak_to_final")} | {row["modeled_downlink_mib_mean"]:.1f}'
                         + f' / {row["modeled_upload_mib_mean"]:.1f} |')
        lines.append('')
    lines += ['Communication is modeled tensor payload, not measured network traffic.',
              'Wall time includes serial simulation, evaluation and serialization; it is not distributed FL latency.',
              'Protocol seed is fixed within a block; training seeds vary initialization, training RNG and witnesses.',
              'Standard Dirichlet is an optional robustness condition, not a fixed-marginal topology intervention.',
              'Ablations support component effects; they do not prove concentration causes forgetting.', '',
              'See status.csv for incomplete/invalid runs and protocol_audit.csv for pairing checks.']
    (out/'report.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print(f'Method A summary: {out / "report.md"}', flush=True)
    return statuses
