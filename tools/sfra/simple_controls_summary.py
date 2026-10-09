"""Independent seed42 comparisons; incomplete or mismatched runs never become paired evidence."""
import io
import json
import math
from pathlib import Path
import statistics
import tarfile

from tools.sfra.simple_controls import (CONTRACT, GAMMA, IMPLEMENTATION_FILES, METHODS, NEW_METHODS,
    PLAN_NAME, SCHEMA, METRICS, digest, implementation_variant, load_json, pair_signature,
    partition_counts, read_csv, tail_weights, validate_config, write_table)


def finite(x):
    value = float(x)
    if not math.isfinite(value):
        raise ValueError('Nonfinite recorded metric')
    return value


def boolean(x):
    if x in (True, 'True', 'true', '1', 1):
        return True
    if x in (False, 'False', 'false', '0', 0):
        return False
    raise ValueError('Invalid boolean record')


def audit_control_logs(run, method, counts):
    phases = [(r, 'B') for r in range(1, 101)] + [(r, 'A') for r in range(1, 91)]
    expected_files = {f'r{rnd:03d}_{factor}.csv' for rnd, factor in phases}
    if {p.name for p in (run/'aggregation_weights').glob('*.csv')} != expected_files:
        raise ValueError('Unexpected/missing aggregation phases')
    expected = tail_weights(counts['sizes'], counts['tail_counts'], GAMMA[method])
    aggregated = []
    for rnd, factor in phases:
        rows = read_csv(run/'aggregation_weights'/f'r{rnd:03d}_{factor}.csv')
        if len(rows) != 30 or {int(x['client_id']) for x in rows} != set(range(30)):
            raise ValueError('Incomplete/duplicate aggregation weight records')
        for row in rows:
            if int(row['round']) != rnd or row['factor'] != factor:
                raise ValueError('Aggregation phase label mismatch')
            j = int(row['client_id'])
            if not math.isclose(finite(row['weight']), expected[j], abs_tol=1e-10, rel_tol=1e-8):
                raise ValueError('Actual aggregation weight differs from the frozen rule')
            if not math.isclose(finite(row['original_weight']), counts['sizes'][j]/sum(counts['sizes']), abs_tol=1e-10):
                raise ValueError('Original sample weight was overwritten')
        aggregated.extend(dict(method=method, **x) for x in rows)
    priorities = []
    if implementation_variant(method) == 'full-cp':
        tokens = load_json(run/'private_witness_manifest.json')
        if {p.name for p in (run/'functional_priority').glob('*.csv')} != {f'r{r:03d}.csv' for r in range(1, 91)}:
            raise ValueError('Unexpected/missing functional priority rounds')
        for rnd in range(1, 91):
            rows = read_csv(run/'functional_priority'/f'r{rnd:03d}.csv')
            if len(rows) != len(tokens) or [int(x['token_id']) for x in rows] != list(range(len(tokens))):
                raise ValueError('Incomplete/duplicate functional priority records')
            raw = []
            for row, token in zip(rows, tokens):
                c = int(row['class_id']); m = counts['coverage'][c]
                if (int(row['client_id']), c) != (token['client_id'], token['class_id']) or int(row['round']) != rnd:
                    raise ValueError('Functional token identity mismatch')
                rho = 1/m if method == 'cover-cp' else finite(row['source_rho'])
                if int(row['holders']) != m or not math.isclose(finite(row['actual_rho']), rho, rel_tol=1e-6):
                    raise ValueError('Actual functional priority differs from the frozen rule')
                raw.append(rho if boolean(row['active']) else 0.)
            denominator = sum(raw)
            for row, value in zip(rows, raw):
                target = value/denominator if denominator else 0.
                if not math.isclose(finite(row['weight']), target, rel_tol=2e-5, abs_tol=1e-8):
                    raise ValueError('Functional weight normalization differs')
            priorities.extend(dict(method=method, **x) for x in rows)
    elif any((run/'functional_priority').glob('*.csv')) or any((run/'sfra_rounds').glob('r*/tokens.npz')):
        raise ValueError('Nonfunctional control unexpectedly ran functional feedback')
    return aggregated, priorities


def read_result(run, method, job=None):
    run = Path(run)
    cfg = validate_config(run, method)
    own_job = run/'simple_control_job.json'
    is_new = own_job.is_file()
    if job is not None and (not is_new or load_json(own_job) != job):
        raise ValueError('Registered job differs from run metadata')
    if is_new:
        registered = load_json(own_job)
        if registered['contract'] != CONTRACT or registered['method'] != method or registered['seed'] != 42:
            raise ValueError('Wrong registered control contract')
        receipt = load_json(run/'simple_control_receipt.json')
        if receipt.get('schema_version') != SCHEMA or not receipt.get('code_unchanged'):
            raise ValueError('Missing/invalid stable-source completion receipt')
    elif method not in ('s', 'full-cp'):
        raise ValueError('New controls must have a registered job')
    done = load_json(run/'completion.json')
    if (done.get('completed_round'), done.get('official_test_passes'), done.get('variant')) != (100, 101, implementation_variant(method)):
        raise ValueError('Expected 100 completed rounds and 101 evaluations')
    if done.get('normal_optimizer_steps') != 105600 or done.get('extra_optimizer_steps') != 31680:
        raise ValueError('Local optimization budget differs')
    correction_steps = int(done['functional_correction_steps'])
    if not 0 <= correction_steps <= 270 or implementation_variant(method) == 's' and correction_steps != 0:
        raise ValueError('Unexpected functional correction budget')
    summaries = read_csv(run/'sfra_rounds.csv') if (run/'sfra_rounds.csv').is_file() else []
    if implementation_variant(method) == 'full-cp':
        if [int(r['round']) for r in summaries] != list(range(1, 101)):
            raise ValueError('Missing/duplicate functional round summaries')
        if any(int(r['correction_steps']) not in ((0, 3) if int(r['round']) <= 90 else (0,)) for r in summaries):
            raise ValueError('Functional correction must use three steps or explicitly skip')
    elif summaries:
        raise ValueError('Ordinary training unexpectedly contains functional round summaries')
    if sum(int(r['correction_steps']) for r in summaries) != correction_steps:
        raise ValueError('Round and completion correction budgets disagree')
    if is_new and done.get('experiment') != method:
        raise ValueError('Completion marker has wrong experiment')
    prior = load_json(run/'class_prior.json')
    counts = partition_counts(read_csv(run/'partition_manifest.csv'), prior['tail_ids'])
    if prior['counts'] != counts['class_counts'] or len(counts['tail_ids']) != 20:
        raise ValueError('Class prior differs from the actual partition')
    if is_new and counts != registered['counts']:
        raise ValueError('Actual partition counts differ from the registered counts')
    curves = sorted(read_csv(run/'round_metrics.csv'), key=lambda r: int(r['round']))
    if [int(r['round']) for r in curves] != list(range(101)):
        raise ValueError('Missing/duplicate committed rounds')
    for row in curves:
        if any(not 0 <= finite(row[m]) <= 100 for m in METRICS):
            raise ValueError('Accuracy outside [0,100]')
    performance = dict(method=method, seed=42, gamma=GAMMA[method], run=str(run),
        origin='new' if is_new else 'reused', repetitions=1,
        **{'last20_'+m: statistics.mean(finite(x[m]) for x in curves[81:101]) for m in METRICS},
        final_tail=finite(curves[100]['bottom20_tail_acc']),
        tail_change_90_to_100=finite(curves[100]['bottom20_tail_acc'])-finite(curves[90]['bottom20_tail_acc']),
        tail_peak_to_final=max(finite(x['bottom20_tail_acc']) for x in curves[1:])-finite(curves[100]['bottom20_tail_acc']))
    base = cfg['base_training_config']
    groups = dict(overall_acc=list(range(100)), head20_acc=base['head_ids'], middle60_acc=base['middle_ids'],
                  bottom20_tail_acc=counts['tail_ids'], non_tail_acc=[c for c in range(100) if c not in counts['tail_ids']])
    for rnd in range(81, 101):
        rows = read_csv(run/f'per_class_accuracy_epoch_{rnd-1}.csv')
        if len(rows) != 100 or {int(x['class_id']) for x in rows} != set(range(100)):
            raise ValueError('Incomplete/duplicate per-class results')
        acc = {int(x['class_id']): finite(x['per_class_acc']) for x in rows}
        if any(not 0 <= x <= 100 for x in acc.values()):
            raise ValueError('Invalid per-class accuracy')
        for metric, ids in groups.items():
            if not math.isclose(statistics.mean(acc[c] for c in ids), finite(curves[rnd][metric]), abs_tol=1e-7):
                raise ValueError('Per-class and official metrics disagree')
    weights, priorities = audit_control_logs(run, method, counts) if is_new else ([], [])
    witness = digest(load_json(run/'private_witness_manifest.json')) if implementation_variant(method) == 'full-cp' else None
    return dict(performance=performance, curves=curves, signature=pair_signature(run, cfg),
                witness=witness, counts=counts, groups=groups, run=run, is_new=is_new,
                weights=weights, priorities=priorities, completion=done)


def sample_metrics(result):
    import numpy as np
    run, method = result['run'], result['performance']['method']
    paths = [run/'predictions'/f'r{rnd:03d}.npz' for rnd in range(101)]
    if not all(p.is_file() for p in paths):
        if result['is_new']:
            raise ValueError('New runs require all 101 per-sample prediction files')
        return [], 'unavailable in reused archive', None
    rows, initial, labels, previous = [], None, None, None
    initial_signature = None
    for rnd, path in enumerate(paths):
        with np.load(path, allow_pickle=False) as z:
            ids, current_labels, pred, correct = (z[k] for k in ('sample_id', 'class_id', 'prediction', 'correct'))
            if (len(ids) != 10000 or not np.array_equal(ids, np.arange(10000))
                    or current_labels.shape != ids.shape or pred.shape != ids.shape or correct.shape != ids.shape
                    or not np.array_equal(correct, pred == current_labels) or not np.isin(pred, np.arange(100)).all()
                    or not np.isin(current_labels, np.arange(100)).all()
                    or not np.all(np.bincount(current_labels, minlength=100) == 100)):
                raise ValueError('Invalid prediction identities/labels/correctness')
            correct = correct.astype(bool)
            if rnd == 0:
                initial, labels = correct.copy(), current_labels.copy()
                initial_signature = digest(dict(labels=labels.tolist(), correct=initial.tolist()))
            elif not np.array_equal(labels, current_labels):
                raise ValueError('Test identities changed across rounds')
            per_class = np.bincount(labels, weights=correct, minlength=100)
            # There are 100 examples per class, so correct counts equal accuracy in percent.
            for metric, group_ids in result['groups'].items():
                if not math.isclose(float(per_class[group_ids].mean()),
                                    finite(result['curves'][rnd][metric]), abs_tol=1e-7):
                    raise ValueError('Prediction and official metrics disagree: ' + metric)
            tail_mask = np.isin(labels, result['counts']['tail_ids'])
            for group, mask in [('Tail20', tail_mask), ('Non-tail', ~tail_mask)]:
                old, new = mask & initial, mask & ~initial
                rows.append(dict(method=method, round=rnd, group=group, initial_correct=int(old.sum()),
                    initial_wrong=int(new.sum()), retained=int(correct[old].sum()), learned=int(correct[new].sum()),
                    retention_percent=float(correct[old].mean()*100) if old.any() else '',
                    acquisition_percent=float(correct[new].mean()*100) if new.any() else '',
                    correct_to_wrong=int((mask & previous & ~correct).sum()) if previous is not None else '',
                    wrong_to_correct=int((mask & ~previous & correct).sum()) if previous is not None else ''))
            previous = correct.copy()
    return rows, 'complete; descriptive per-trajectory transitions', initial_signature


def cost_metrics(result):
    run = result['run']
    def records(name):
        return read_csv(run/name) if (run/name).is_file() else []
    budget, feedback, evaluation = records('budget.csv'), records('sfra_costs.csv'), records('evaluation_budget.csv')
    def total(rows, key):
        return sum(finite(x.get(key) or 0) for x in rows)
    if feedback and result['performance']['method'].startswith('tailrw-'):
        raise ValueError('TailRW unexpectedly performed functional feedback')
    elapsed = finite(result['completion']['elapsed_seconds'])
    official = total([x for x in evaluation if x.get('kind') == 'official_test'], 'seconds')
    other_eval = total([x for x in evaluation if x.get('kind') != 'official_test'], 'seconds')
    resource = load_json(run/'resource_usage.json') if (run/'resource_usage.json').is_file() else {}
    return dict(method=result['performance']['method'], elapsed_seconds=elapsed,
        official_evaluation_seconds=official, other_evaluation_seconds=other_eval,
        elapsed_excluding_recorded_evaluations=elapsed-official-other_eval,
        timing_scope='sequential simulation; remaining time includes logging/checkpoints',
        local_optimizer_steps=result['completion']['normal_optimizer_steps']+result['completion']['extra_optimizer_steps'],
        local_training_images=total(budget, 'sample_presentations'),
        functional_correction_steps=result['completion']['functional_correction_steps'],
        A_forward_images=total(feedback, 'forward_images'), A_backward_images=total(feedback, 'backward_images'),
        local_upload_bytes=total(budget, 'upload_bytes'), local_downlink_bytes=total(budget, 'modeled_downlink_bytes'),
        A_upload_bytes=total(feedback, 'upload_bytes'), A_downlink_bytes=total(feedback, 'downlink_bytes'),
        feedback_records=len(feedback), peak_allocated_bytes=resource.get('peak_allocated_bytes', ''),
        hardware=json.dumps(load_json(run/'bridge_metadata.json').get('environment', {}), ensure_ascii=False),
        static_count_metadata='reuses registered training partition counts; no extra image passes')


def resolve_reference(root, path):
    path = Path(path)
    return path if path.is_absolute() else Path(root)/path


def summarize(root):
    root = Path(root)
    plan = load_json(root/PLAN_NAME)
    if plan['contract'] != CONTRACT:
        raise ValueError('Unexpected simple-control contract')
    out = root/'analysis'; out.mkdir(parents=True, exist_ok=True)
    jobs = {j['method']: j for j in plan['jobs']}
    status, results, transitions, costs, retention = [], {}, [], [], []
    for method in METHODS:
        job = jobs.get(method)
        run = root/job['run'] if job else resolve_reference(root, plan['baselines'][method]) if method in plan['baselines'] else None
        row = dict(method=method, status='missing', reason='', run=str(run) if run else '')
        if run is not None and run.is_dir():
            row['status'] = 'incomplete'
            if (run/'completion.json').is_file():
                try:
                    result = read_result(run, method, job)
                    samples, note, initial_sig = sample_metrics(result)
                    result['initial_predictions'] = initial_sig
                    cost = cost_metrics(result)
                    results[method] = result
                    costs.append(cost); transitions.extend(samples)
                    for group in ('Tail20', 'Non-tail'):
                        endpoint = [s for s in samples if s['group'] == group and 81 <= s['round'] <= 100]
                        if endpoint:
                            retention.append(dict(method=method, seed=42, group=group,
                                initial_correct=endpoint[0]['initial_correct'], initial_wrong=endpoint[0]['initial_wrong'],
                                **{'last20_mean_'+k: statistics.mean(s[k] for s in endpoint) if endpoint[0][k] != '' else ''
                                   for k in ('retained', 'learned', 'retention_percent', 'acquisition_percent')}))
                    row.update(status='complete', sample_metrics=note)
                except (OSError, ValueError, KeyError, TypeError) as error:
                    row.update(status='invalid', reason=str(error))
        status.append(row)
    contrasts = [('full-cp', m) for m in NEW_METHODS] + [(m, 's') for m in ('full-cp', *NEW_METHODS)]
    paired, audits = [], []
    for left, right in contrasts:
        audit = dict(method=left+' minus '+right, status='pending', reason='missing or incomplete paired run')
        if left in results and right in results:
            a, b = results[left], results[right]
            mismatches = [k for k in a['signature'] if a['signature'][k] != b['signature'][k]]
            if a['witness'] is not None and b['witness'] is not None and a['witness'] != b['witness']:
                mismatches.append('witness')
            if (a['initial_predictions'] is not None and b['initial_predictions'] is not None
                    and a['initial_predictions'] != b['initial_predictions']):
                mismatches.append('initial_predictions')
            audit.update(status='mismatch' if mismatches else 'eligible', reason=', '.join(mismatches))
            if not mismatches:
                paired.append(dict(comparison=audit['method'], seed=42, repetitions=1,
                    **{'delta_'+m: a['performance']['last20_'+m]-b['performance']['last20_'+m] for m in METRICS},
                    delta_tail_peak_to_final=a['performance']['tail_peak_to_final']-b['performance']['tail_peak_to_final']))
        audits.append(audit)
    tables = dict(status=status, pair_audit=audits, performance=[r['performance'] for r in results.values()],
        paired_per_seed=paired, costs=costs, sample_transitions=transitions, sample_retention=retention,
        curves=[dict(x, method=m) for m, r in results.items() for x in r['curves']],
        aggregation_weights=[x for r in results.values() for x in r['weights']],
        functional_priority=[x for r in results.values() for x in r['priorities']])
    for name, rows in tables.items():
        write_table(out/(name+'.csv'), rows)
    lines = ['# Method A simple controls — seed42', '',
        'Fixed endpoint: mean committed-round Tail20 accuracy over rounds 81–100. All three gamma values are reported.',
        'One development seed only: no seed standard deviation, significance, or generalization claim.', '',
        '| Method | Status | Tail20 | Overall | Reason |', '|---|---|---:|---:|---|']
    for row in status:
        p = results.get(row['method'], {}).get('performance', {})
        tail = f'{p["last20_bottom20_tail_acc"]:.4f}' if p else ''
        overall = f'{p["last20_overall_acc"]:.4f}' if p else ''
        lines.append(f'| {row["method"]} | {row["status"]} | {tail} | {overall} | {row["reason"]} |')
    lines += ['', 'Paired differences (percentage points; only protocol-compatible pairs):', '',
              '| Comparison | Tail20 | Overall |', '|---|---:|---:|']
    lines += [f'| {r["comparison"]} | {r["delta_bottom20_tail_acc"]:+.4f} | {r["delta_overall_acc"]:+.4f} |' for r in paired]
    lines += ['', 'Cover-CP retains source-based current targets and their cost. It isolates the priority rule only.',
        'TailRW changes both normal factor aggregation stages and has no functional correction.',
        'Higher tail accuracy with lower overall accuracy is a tradeoff, not an automatic win.',
        'Missing predictions in legacy references stay unavailable. Costs are simulated messages and sequential runtime.',
        'See pair_audit.csv for incompatible pairs. Reuse compares unchanged core training code; federated_main routing differs.', '']
    (out/'report.md').write_text('\n'.join(lines), encoding='utf-8')
    return status + audits


def pack(root, repo):
    root, repo = Path(root), Path(repo)
    plan = load_json(root/PLAN_NAME)
    portable = dict(plan, baselines={m: 'references/'+m for m in plan['baselines']})
    destination = root.with_name(root.name+'_results.tar.gz')
    allowed, excluded = {'.json', '.csv', '.npz', '.yaml', '.md'}, {'checkpoints', 'events', 'formal_lora', 'locks'}
    with tarfile.open(destination, 'w:gz') as archive:
        def add_tree(source, target):
            for path in sorted(source.rglob('*')):
                relative = path.relative_to(source)
                if path.is_file() and path.suffix in allowed and not (set(relative.parts) & excluded):
                    if source == root and relative.as_posix() == PLAN_NAME:
                        continue
                    archive.add(path, arcname=target+'/'+relative.as_posix())
        add_tree(root, root.name)
        for method, value in plan['baselines'].items():
            source = resolve_reference(root, value)
            if source.is_dir():
                add_tree(source, root.name+'/references/'+method)
        data = (json.dumps(portable, indent=2)+'\n').encode()
        entry = tarfile.TarInfo(root.name+'/'+PLAN_NAME); entry.size = len(data)
        archive.addfile(entry, io.BytesIO(data))
        for name in ('federated_main.py', *IMPLEMENTATION_FILES):
            archive.add(repo/name, arcname=root.name+'/implementation/'+name)
    return destination
