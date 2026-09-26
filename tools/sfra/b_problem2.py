"""CPU-only preflight, provenance and summaries for saved-event B replay."""
import csv
import hashlib
import json
from pathlib import Path

SCHEMA = 'b_problem2_replay_v1'
ROUNDS = (30, 60, 90)
VARIANTS = ('E00', 'E10', 'E01', 'E11')


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def read_csv(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(dict.fromkeys(k for row in rows for k in row)))
        writer.writeheader()
        writer.writerows(rows)


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def required_files(rounds):
    names = ['sfra_config.json', 'command.json', 'bridge_metadata.json', 'partition_manifest.csv',
             'private_witness_manifest.json', 'b_transfer_manifest.json', 'checkpoints/base_model.pt',
             'protocol/full_schedule.json', 'protocol/eri_protocol.json', 'protocol/probe_manifest.csv']
    for rnd in rounds:
        names += [f'events/r{rnd:03d}_c000_main_normal_B/{name}' for name in ('state.pt', 'event.json')]
        names += [f'b_transfer_rounds/r{rnd:03d}/{name}' for name in
                  ('commit.pt', 'matrices.npz', 'calibration_manifest.json', 'donor_scores.csv', 'probe_metrics.csv')]
    return names


def preflight(source, rounds=ROUNDS):
    source = Path(source).resolve()
    if not rounds or len(set(rounds)) != len(rounds) or any(r not in range(30, 101, 10) for r in rounds):
        raise ValueError('Replay rounds must be unique B events in 30,40,...,100')
    missing = [name for name in required_files(rounds) if not (source/name).is_file()]
    errors, details = [], []
    if (source/'sfra_config.json').is_file():
        cfg = read_json(source/'sfra_config.json')
        transfer = cfg.get('b_transfer', {})
        if cfg.get('variant') != 'full-cp' or transfer.get('mode') != 'shared':
            errors.append('Source must be full-cp with shared B transfer')
        if transfer.get('calibration_profile') == 'problem2':
            errors.append('Use the original shared/tradeoff reference, not a problem2 descendant')
        if cfg.get('b_aggregation', {}).get('mode', 'sample') != 'sample':
            errors.append('Source must use ordinary sample-weighted B')
        if transfer.get('steps') != 2:
            errors.append('This protocol requires two C steps')
        for rnd in rounds:
            path = source/f'b_transfer_rounds/r{rnd:03d}/calibration_manifest.json'
            if path.is_file():
                manifest = read_json(path)
                details.append(dict(round=rnd, source_donors=len(manifest['shared_donor_ids']),
                                    feedback_clients=len(manifest['feedback_clients'])))
    return dict(schema_version=SCHEMA, source=str(source), rounds=list(rounds),
                ready=not missing and not errors, missing=missing, errors=errors, events=details,
                note='File/config preflight only; tensor and identity checks run on replay. No training started.')


def receipt(source, rounds):
    source = Path(source)
    names = required_files(rounds)
    if (source/'execution_config.json').is_file():
        names.append('execution_config.json')
    return {name: file_hash(source/name) for name in names}


def replace_argument(command, flag, value):
    if flag in command:
        command[command.index(flag)+1] = str(value)
    else:
        index = command.index('DATALOADER.NUM_WORKERS')
        command[index:index] = [flag, str(value)]


def frequency_metrics(run):
    """Committed rounds 81..100 are per-class epochs 80..99."""
    run = Path(run)
    files = [run/f'per_class_accuracy_epoch_{r-1}.csv' for r in range(81,101)]
    if not (run/'class_prior.json').is_file() or any(not p.is_file() for p in files):
        return dict(frequency_metrics_available=False)
    counts = read_json(run/'class_prior.json')['counts']
    groups = {'many': [c for c,n in enumerate(counts) if n>100],
              'medium': [c for c,n in enumerate(counts) if 20<=n<=100],
              'few': [c for c,n in enumerate(counts) if n<20]}
    results = {name:[] for name in groups}
    for path in files:
        rows = read_csv(path)
        values = {int(r['class_id']):float(r['per_class_acc']) for r in rows}
        if len(rows) != len(values) or set(values) != set(range(len(counts))):
            raise ValueError(f'Incomplete or duplicate class accuracy: {path}')
        for name, ids in groups.items():
            if ids:
                results[name].append(sum(values[c] for c in ids)/len(ids))
    return dict(frequency_metrics_available=True,
                **{f'{name}_class_count':len(ids) for name,ids in groups.items()},
                **{f'last20_{name}_acc':sum(values)/len(values) if values else None
                   for name,values in results.items()})


def summarize(root):
    root = Path(root)
    plan = read_json(root/'replay_plan.json')
    tables = {name: [] for name in ('class_change_summary', 'class_changes', 'class_weights',
                                   'class_loss_trace', 'c_loss_trace', 'optimization_steps')}
    status, costs, diagnostics = [], [], {name: [] for name in ('donor_contributions', 'norm_matched', 'equivalence')}
    for rnd in plan['rounds']:
        event = root/f'r{rnd:03d}'
        done = (event/'replay_completion.json').is_file()
        status.append(dict(round=rnd, complete=done))
        if not done:
            continue
        completion = read_json(event/'replay_completion.json')
        costs.append(dict(round=rnd, **{name:completion.get(name) for name in (
            'seconds','extra_diagnostic_seconds','total_optimizer_steps','total_algorithm_forward_images',
            'total_algorithm_backward_images','total_diagnostic_forward_images')}))
        for variant in plan['variants']:
            folder = event/variant/'b_transfer_rounds'/f'r{rnd:03d}'
            for name in tables:
                tables[name].extend(dict(row, variant=variant, round=rnd) for row in read_csv(folder/f'{name}.csv'))
        for name in diagnostics:
            path = event/f'{name}.csv'
            if path.is_file():
                diagnostics[name].extend(read_csv(path))
    for name, rows in {**tables, **diagnostics, 'status': status, 'costs': costs}.items():
        write_csv(root/'analysis'/f'{name}.csv', rows)
    differences = []
    values = {(int(r['round']), r['variant'], r['group']): r for r in tables['class_change_summary']}
    for rnd in plan['rounds']:
        for group in ('tail', 'non_tail'):
            for a, b in (('E10','E00'), ('E01','E00'), ('E11','E10'), ('E11','E01')):
                if (rnd,a,group) in values and (rnd,b,group) in values:
                    ra, rb = values[rnd,a,group], values[rnd,b,group]
                    differences.append(dict(round=rnd, group=group, comparison=f'{a}-{b}',
                        **{k: float(ra[k])-float(rb[k]) for k in ('macro_gain','positive_gain','harm','max_harm')}))
    write_csv(root/'analysis/paired_differences.csv', differences)
    result = dict(complete=all(r['complete'] for r in status), rounds=status,
                  scope='Same-state training-side diagnostics; not 100-round results or independent seeds')
    write_json(root/'analysis/status.json', result)
    lines = ['# B problem 2: saved-event replay', '',
             'Same-state training-side monitors; calibration overlap is recorded. These are not independent seeds or full-training test results.', '',
             '| Round | Variant | Group | Net LA gain | Positive gain | Harm | Harmed units / total |',
             '|---|---|---|---:|---:|---:|---:|']
    for row in tables['class_change_summary']:
        lines.append(f"| {row['round']} | {row['variant']} | {row['group']} | "
                     f"{float(row['macro_gain']):.6g} | {float(row['positive_gain']):.6g} | "
                     f"{float(row['harm']):.6g} | {row['harmed_units']} / {row['units']} |")
    lines += ['', 'Check equivalence.csv before interpreting differences; a changed donor pool is not an equivalence test.',
              'Norm-matched results control update size; donor removal holds C and the original denominator fixed.',
              'costs.csv separates the four calibrations and extra forward diagnostics. No test metric selected an update.',
              f"Completed events: {sum(r['complete'] for r in status)}/{len(status)}."]
    (root/'analysis/report.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    return result
