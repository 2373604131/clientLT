"""Audit complete runs, recompute fixed groups, preserve incomplete/error status."""
import io
import math
from pathlib import Path
import statistics
import tarfile

from tools.benchmarks.common import (LABELS, METRICS, digest, job_id, job_path, manifest_rows,
    read_csv, read_json, run_path, write_csv, write_json)


def imported_run(root, method):
    index = Path(root) / 'imports.json'
    entry = read_json(index).get(method) if index.exists() else None
    if entry is None:
        return None
    packaged = Path(root) / 'imported' / method
    return packaged if packaged.exists() else Path(entry['path'])


def locate_run(root, job):
    imported = imported_run(root, job['method'])
    if imported is not None:
        return imported
    local = run_path({**job, 'output_root': str(Path(root).resolve())})
    return local if local.exists() else run_path(job)


def validate_ab(run, job):
    from tools.sfra.ab_validation import read_result, code_hashes
    from tools.benchmarks.common import REPO
    receipt = read_json(run / 'ab_validation_run.json')
    old_job = receipt['job']
    if old_job['seed'] != 42 or old_job['arm'] != job['method'] or old_job['protocol_seed'] != 42:
        raise ValueError('Imported A/AB has wrong seed/arm/protocol seed')
    if old_job['code_sha256'] != code_hashes(REPO):
        raise ValueError('Imported A/AB training source differs from frozen local source')
    result, _, _, classes, curve = read_result(run, old_job)
    meta = read_json(run / 'bridge_metadata.json')
    actual = dict(partition=digest(manifest_rows(run / 'partition_manifest.csv')),
        schedule=meta['schedule_sha256'], pool=meta['pool_sha256'], test=meta['test_sha256'],
        classnames=digest(meta['classnames']))
    if any(actual[k] != job['protocol'][k] for k in actual):
        raise ValueError('Imported A/AB does not match the benchmark data/schedule')
    prior = read_json(run / 'class_prior.json')
    base = read_json(run / 'sfra_config.json')['base_training_config']
    if (prior['tail_ids'] != list(range(80, 100)) or base['head_ids'] != list(range(20))
        or base['middle_ids'] != list(range(20, 80))):
        raise ValueError('Imported class groups differ from fixed 20/60/20 definition')
    return result, classes, curve


def import_ab(root, existing):
    from scripts.run_ab_validation import file_lock
    root, existing = Path(root), Path(existing).resolve()
    plan = read_json(existing / 'ab_validation_plan.json')
    candidates = {}
    for method in ('a', 'ab'):
        entries = [x for x in plan['jobs'] if x['arm'] == method and x['seed'] == 42]
        if len(entries) != 1:
            raise ValueError(f'Expected one seed42 {method} run in {existing}')
        run = (existing / entries[0]['run']).resolve()
        if not run.is_relative_to(existing):
            raise ValueError('Imported run leaves its source suite')
        job = read_json(job_path(root, method))
        validate_ab(run, job)
        candidates[method] = dict(path=str(run), protocol=job['protocol'], seed=42)
    with file_lock(root / 'locks/imports.lock'):
        current = read_json(root / 'imports.json') if (root / 'imports.json').exists() else {}
        for method, entry in candidates.items():
            if method in current and current[method] != entry:
                raise ValueError('Another A/AB source is already registered')
            if run_path(read_json(job_path(root, method))).exists():
                raise ValueError('A new A/AB run already exists; do not replace it with an imported run')
        write_json(root / 'imports.json', {**current, **candidates})
    print('Verified and registered seed42 A and AB; source results were not modified.')


def metrics(correct, total):
    if len(correct) != 100 or len(total) != 100 or any(n != 100 for n in total):
        raise ValueError('Expected 100 classes and 100 test images per class')
    if any(not isinstance(x, int) or x < 0 or x > n for x, n in zip(correct, total)):
        raise ValueError('Invalid class correct counts')
    values = [100 * x / n for x, n in zip(correct, total)]
    return dict(overall_acc=100 * sum(correct) / sum(total), head20_acc=statistics.mean(values[:20]),
        middle60_acc=statistics.mean(values[20:80]), bottom20_tail_acc=statistics.mean(values[80:]),
        non_tail_acc=statistics.mean(values[:80])), values


def completed_result(root, job):
    run = locate_run(root, job)
    if job['method'] in ('a', 'ab'):
        result, classes, curve = validate_ab(run, job)
        done = read_json(run / 'completion.json')
        return dict(method=job['method'], label=job['label'], seed=42,
            local_optimizer_steps=done['normal_optimizer_steps'] + done['extra_optimizer_steps'],
            server_correction_steps=done.get('functional_correction_steps', 0),
            **{k: v for k, v in result.items() if k.startswith(('last20_', 'final_', 'tail_', 'A_', 'B_', 'local_'))
               or k == 'elapsed_seconds'}), classes, curve
    done = read_json(run / 'completion.json')
    info = read_json(run / 'run.json')
    expected_rounds = 2 if job['smoke'] else 100
    if (done.get('job_id') != job_id(job) or info.get('job_id') != job_id(job)
        or info.get('job') != job or done.get('completed_round') != expected_rounds
        or done.get('official_test_passes') != expected_rounds + 1
        or done.get('smoke') != job['smoke'] or not done.get('source_unchanged')):
        raise ValueError('Completion/receipt does not match this job')
    curve, class_values = [], []
    for rnd in range(expected_rounds + 1):
        record = read_json(run / 'rounds' / f'r{rnd:03d}.json')
        if record['round'] != rnd or record['job_id'] != job_id(job):
            raise ValueError('Round identity differs')
        measured, per_class = metrics(record['correct'], record['total'])
        if any(not math.isclose(float(record['metrics'][m]), measured[m], abs_tol=1e-8, rel_tol=0) for m in METRICS):
            raise ValueError('Saved aggregate differs from per-class recomputation')
        curve.append(dict(round=rnd, **measured))
        class_values.append(per_class)
    costs = read_csv(run / 'costs.csv')
    if [int(x['round']) for x in costs] != list(range(1, expected_rounds + 1)):
        raise ValueError('Missing/duplicate cost rows')
    steps = sum(int(x['optimizer_steps']) for x in costs)
    if steps != done['optimizer_steps']:
        raise ValueError('Optimizer-step count differs')
    # Validate against raw partition capacities, not only a self-reported total.
    capacities = [0] * 30
    reference = Path(job['reference_run']) / 'partition_manifest.csv'
    if not reference.exists():
        reference = Path(root) / 'protocol/partition_manifest.csv'
    for row in manifest_rows(reference):
        capacities[row['client_id']] += 1
    selection_epochs = job['config']['factor_baselines']['lora-a2']['selection_epochs'] if job['method'] == 'lora-a2' else 0
    expected_steps = 4 * (1 + bool(selection_epochs)) if job['smoke'] else 100 * (3 + selection_epochs) * sum(math.ceil(n / 32) for n in capacities)
    if steps != expected_steps:
        raise ValueError('Local optimization budget differs from fixed protocol')
    if selection_epochs:
        expected_selection = 4 if job['smoke'] else 100 * selection_epochs * sum(math.ceil(n / 32) for n in capacities)
        if sum(int(x['selection_optimizer_steps']) for x in costs) != expected_selection:
            raise ValueError('LoRA-A2 rank-selection compute differs from declared budget')
    window = list(range(81, 101)) if not job['smoke'] else [expected_rounds]
    result = dict(method=job['method'], label=job['label'], seed=42, smoke=job['smoke'],
                  trainable_parameters=info['trainable_parameters'], optimizer_steps=steps)
    for name in METRICS:
        result['last20_' + name] = statistics.mean(curve[r][name] for r in window)
        result['final_' + name] = curve[-1][name]
    result['tail_peak_to_final'] = max(r['bottom20_tail_acc'] for r in curve[1:]) - curve[-1]['bottom20_tail_acc']
    for key in ('train_seconds', 'upload_bytes', 'downlink_bytes', 'student_forward_images', 'teacher_forward_images', 'backward_images'):
        result[key] = sum(float(x[key]) for x in costs)
    if selection_epochs:
        result['selection_optimizer_steps'] = sum(int(x['selection_optimizer_steps']) for x in costs)
        result['selection_forward_images'] = sum(int(x['selection_forward_images']) for x in costs)
    if job['method'] in ('fedlf', 'fedyoyo', 'fedrela'):
        for key in ('auxiliary_forward_images', 'statistics_upload_bytes', 'statistics_download_bytes',
                    'selected_distillation_images', 'relabeled_training_images'):
            result[key] = sum(int(x[key]) for x in costs)
        audits = [read_json(run / 'method_audits' / f'r{rnd:03d}.json') for rnd in range(1, expected_rounds + 1)]
        if any(x.get('round') != rnd or x.get('method') != job['method'] for rnd, x in enumerate(audits, 1)):
            raise ValueError('Missing/mismatched long-tail method audits')
        if job['method'] == 'fedyoyo':
            for audit in audits:
                prior = audit['estimated_prior']
                if len(prior) != 100 or any(not math.isfinite(v) or v <= 0 for v in prior) or not math.isclose(sum(prior), 1., abs_tol=1e-5):
                    raise ValueError('Invalid FedYoYo estimated global prior')
        if job['method'] == 'fedrela':
            relabel_round = 2 if job['smoke'] else job['config']['longtail_baselines']['fedrela']['relabel_after_round'] + 1
            events = [x for x in audits if 'relabel_clients' in x]
            if len(events) != 1 or events[0]['round'] != relabel_round:
                raise ValueError('Expected exactly one FedReLa relabel event at its declared round')
            result['relabeled_samples'] = sum(x['changed'] for x in events[0]['relabel_clients'])
    classes = {c: statistics.mean(class_values[r][c] for r in window) for c in range(100)}
    return result, classes, curve


def status(root):
    root = Path(root)
    if not (root / 'jobs').is_dir():
        raise FileNotFoundError('No registered benchmark jobs under ' + str(root))
    result = []
    for path in sorted((root / 'jobs').glob('*.json')):
        job = read_json(path)
        run = locate_run(root, job)
        record = dict(method=job['method'], seed=42, status='not_started', completed_round=0, detail='')
        if (run / 'completion.json').exists():
            record.update(status='complete_pending_audit', completed_round=read_json(run / 'completion.json').get('completed_round', 0))
        elif (run / 'progress.json').exists():
            progress = read_json(run / 'progress.json')
            record.update(status='partial', completed_round=progress.get('completed_round', progress.get('completed_rounds', 0)),
                          detail=str({k: v for k, v in progress.items() if k in ('phase', 'active_round', 'client_position', 'clients_this_round', 'client_id')}))
        elif run.exists():
            record['status'] = 'initializing_or_interrupted'
        result.append(record)
    return result


def collect(root):
    from scripts.run_ab_validation import file_lock
    root = Path(root)
    with file_lock(root / 'locks/collection.lock'):
        records = status(root)
        results, curves, classes = [], [], []
        for row in records:
            if row['status'] != 'complete_pending_audit':
                continue
            job = read_json(job_path(root, row['method']))
            try:
                result, per_class, curve = completed_result(root, job)
                row['status'] = 'smoke_only' if job['smoke'] else 'complete_verified'
                if job['smoke']:
                    continue
                run_curves = [{**x, 'method': job['method']} for x in curve]
                run_classes = [dict(method=job['method'], class_id=c, last20_acc=v) for c, v in per_class.items()]
                results.append(result)
                curves.extend(run_curves)
                classes.extend(run_classes)
            except (OSError, ValueError, KeyError, TypeError) as error:
                row.update(status='invalid', detail=str(error))
        out = root / 'analysis'
        write_csv(out / 'status.csv', records)
        write_csv(out / 'performance.csv', results)
        write_csv(out / 'costs.csv', [{k: v for k, v in r.items() if not k.startswith(('last20_', 'final_', 'tail_'))}
                                    for r in results])
        write_csv(out / 'curves.csv', curves)
        write_csv(out / 'per_class.csv', classes)
        paired = []
        by_method = {r['method']: r for r in results}
        for method in ('a', 'ab'):
            for baseline in by_method:
                if method not in by_method or baseline == method:
                    continue
                paired.append(dict(method=method, reference=baseline, seed=42,
                    **{m + '_delta_pp': by_method[method]['last20_' + m] - by_method[baseline]['last20_' + m] for m in METRICS}))
        write_csv(out / 'paired.csv', paired)
        lines = ['# Seed42 protocol-adapted method comparison', '',
            'Endpoint: committed rounds81..100 mean. One development seed; no SD/significance claim.', '',
            '| Method | Overall | Head20 | Middle60 | Tail20 |', '|---|---:|---:|---:|---:|']
        for r in results:
            lines.append('| ' + r['label'] + ' | ' + ' | '.join(f"{r['last20_' + m]:.4f}" for m in METRICS[:4]) + ' |')
        lines += ['', 'CAPT uses fixed aggregation, without test-controlled MAB. FedPuReL is its global stage on matched LoRA, not the personalized full method.',
            'FedNTD uses a frozen round-start teacher. FedPuReL uses frozen zero-shot CLIP. FedAvg jointly trains both LoRA factors.',
            'All methods share the reference CIFAR100 base normalization/resize. FedLF retains crop/flip; FedYoYo adds its official weak/strong training views.',
            'Three local epochs for external methods do not equal A/AB total computation; report extra costs separately.',
            'LoRA-A2 adds one TRAINING-data selection epoch before three masked training epochs; selection costs are included.',
            'FFA-LoRA/RoLoRA/FedSVD/LoRA-A2 use shared factors and the matched initialization/scaling/SGD protocol. These are CLIP adaptations, not native-paper reproductions.',
            'FedLF/FedYoYo are CLIP-LoRA adaptations. FedYoYo includes feature-based prior estimation; extra statistic passes/communication are counted.',
            'FedReLa is the official relabel module on a FedAvg-LoRA CE host, with a declared late learning-rate reduction; labels and model state resume together.',
            'Incomplete/invalid/smoke runs are excluded, never filled with zero. See status.csv.',
            'B donor screening and its added value over direct calibration remain separate claims; this table does not prove them.']
        (out / 'report.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
        if any(r['status'] == 'invalid' for r in records):
            raise ValueError('Invalid completed runs excluded; see analysis/status.csv')
        return results


def pack(root):
    root = Path(root).resolve()
    collect(root)
    destination = root.parent / (root.name + '_analysis.tar.gz')
    temp = destination.with_name(destination.name + '.tmp')
    # Only immutable completed-run artifacts and current analysis; no weights,
    # image files, arbitrary logs, or lock/checkpoint files enter the archive.
    with tarfile.open(temp, 'w:gz') as tar:
        for path in [root / 'suite.json', root / 'imports.json', *sorted((root / 'jobs').glob('*.json')),
                     *sorted((root / 'analysis').glob('*'))]:
            if path.is_file():
                tar.add(path, arcname=root.name + '/' + path.relative_to(root).as_posix())
        jobs = [read_json(p) for p in sorted((root / 'jobs').glob('*.json'))]
        if jobs:
            reference = Path(jobs[0]['reference_run'])
            for name in ('partition_manifest.csv', 'bridge_metadata.json'):
                path = reference / name
                if path.exists():
                    tar.add(path, arcname=root.name + '/protocol/' + name)
        for job in jobs:
            run = locate_run(root, job)
            if not (run / 'completion.json').exists():
                continue
            if imported_run(root, job['method']) is not None:
                relative = Path('imported') / job['method']
            else:
                relative = run_path({**job, 'output_root': str(root)}).relative_to(root)
            for path in sorted(run.rglob('*')):
                if path.is_file() and path.suffix.lower() in ('.json', '.csv', '.yaml', '.md'):
                    tar.add(path, arcname=root.name + '/' + (relative / path.relative_to(run)).as_posix())
    temp.replace(destination)
    return destination
