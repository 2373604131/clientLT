"""Collect seed-42 CAPT global-start runs and a descriptive summary (stdlib only)."""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import re
import statistics
import zipfile


FILES = (
    'command.json', 'finished.json', 'run.log', 'log.txt', 'round_metrics.csv',
    'client_class_counts.csv', 'partition_summary.json',
    'client_split_fingerprint.json', 'selected_clients.csv',
)
METRICS = ('overall_acc', 'head_acc', 'tail_acc', 'macro_f1')


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def read_csv(path):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def options(command):
    if not isinstance(command, list):
        raise ValueError('command.json must contain a tokenized command list')
    return {str(token): str(command[i + 1]) for i, token in enumerate(command[:-1])
            if str(token).startswith('--')}


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def hmean(head, tail):
    return 2 * head * tail / (head + tail) if head + tail else 0.0


def inspect_run(path, manifest, opts, index):
    warnings = [f'Missing {name}' for name in FILES if not (path / name).is_file()]
    row = dict(run_id=f'run_{index:02d}', original_path=str(path),
               partition=opts.get('--partition'), seed=opts.get('--seed'),
               specialization_lambda=opts.get('--specialization_lambda'),
               intra_group_alpha=opts.get('--intra_group_alpha'), beta=opts.get('--beta'),
               global_parameter_reset=opts.get('--capt_reset_global_before_client'),
               optimizer_reset=opts.get('--capt_matched_v2', 'False'),
               fixed_aggregation_frequency=opts.get('--capt_fixed_global_agg_freq', '0'),
               requested_rounds=int(opts['--round']), num_users=opts.get('--num_users'),
               frac=opts.get('--frac'), local_epochs=opts.get('--local_epochs'),
               lr=opts.get('--lr'), batch_size=opts.get('--train_batch_size'),
               schedule_sha256=manifest.get('client_schedule_sha256'))
    finish = read_json(path / 'finished.json') if (path / 'finished.json').exists() else {}
    row['normal_exit'] = finish.get('exit_code') == 0
    if not row['normal_exit']:
        warnings.append('No successful finished.json')
    if finish.get('requested_rounds') != row['requested_rounds']:
        warnings.append('finished.json requested_rounds mismatch or missing')
    rounds = sorted(read_csv(path / 'round_metrics.csv'), key=lambda r: int(r['epoch']))
    if not rounds:
        raise ValueError(f'Empty metrics: {path}')
    epochs = [int(r['epoch']) for r in rounds]
    if len(set(epochs)) != len(epochs):
        raise ValueError(f'Duplicate evaluated epochs: {path}')
    final = rounds[-1]
    row['evaluated_server_states'] = len(rounds)
    row['last_evaluated_outer_round'] = int(final['epoch']) + 1
    for metric in METRICS:
        row['final_' + metric] = float(final[metric])
    row['final_hmean'] = hmean(float(final['head_acc']), float(final['tail_acc']))
    late = [r for r in rounds if row['requested_rounds'] - 20 <= int(r['epoch']) < row['requested_rounds']]
    row['last20_evaluation_count'] = len(late)
    for metric in METRICS:
        row['last20_evaluated_mean_' + metric] = statistics.mean(float(r[metric]) for r in late) if late else None
    row['last20_evaluated_mean_hmean'] = statistics.mean(hmean(float(r['head_acc']), float(r['tail_acc'])) for r in late) if late else None
    for metric in ('overall_acc', 'tail_acc'):
        best = max(rounds, key=lambda r: float(r[metric]))
        row['best_' + metric] = float(best[metric])
        row['best_' + metric + '_outer_round'] = int(best['epoch']) + 1
    matrix_hash = None
    if (path / 'client_class_counts.csv').exists():
        counts = sorted(read_csv(path / 'client_class_counts.csv'), key=lambda r: int(r['client_id']))
        classes = sorted((k for k in counts[0] if re.fullmatch(r'class_\d+', k)), key=lambda k: int(k[6:]))
        matrix = [[int(float(r[k])) for k in classes] for r in counts]
        matrix_hash = canonical_hash(matrix)
        row['client_class_matrix_sha256'] = matrix_hash
    if (path / 'selected_clients.csv').exists():
        selected = read_csv(path / 'selected_clients.csv')
        actual = {}
        for r in selected:
            actual.setdefault(int(r['epoch_index']), []).append((int(r['selection_order']), int(r['client_id'])))
        actual_schedule = [[c for _, c in sorted(actual[e])] for e in sorted(actual)]
        row['audited_training_rounds'] = len(actual)
        row['actual_schedule_sha256'] = canonical_hash(actual_schedule)
        if sorted(actual) != list(range(row['requested_rounds'])):
            warnings.append('Selected-client audit does not cover every requested round')
        if any(len(set(clients)) != len(clients) for clients in actual_schedule):
            warnings.append('Duplicate client within a training round')
        if row['actual_schedule_sha256'] != row['schedule_sha256']:
            warnings.append('Actual selected-client schedule differs from recorded schedule hash')
    if (path / 'run.log').exists():
        log = (path / 'run.log').read_text(encoding='utf-8', errors='replace')
        row['logged_training_rounds'] = len(re.findall(r'Epoch \d+: CAPT cluster training', log))
        row['logged_global_aggregations'] = log.count('Global aggregation completed')
        if row['logged_training_rounds'] != row['requested_rounds']:
            warnings.append('Training-log round count differs from requested rounds')
        if row['logged_global_aggregations'] != len(rounds):
            warnings.append('Aggregation count differs from evaluation count; inspect evaluation interval/log')
    for epoch in epochs:
        if not (path / f'per_class_accuracy_epoch_{epoch}.csv').exists():
            warnings.append(f'Missing per-class accuracy for epoch {epoch}')
    row['warnings'] = ' | '.join(warnings)
    return row, dict(run_id=row['run_id'], original_path=str(path), options=opts,
                     command_manifest=manifest, warnings=warnings, matrix_sha256=matrix_hash)


def csv_text(rows):
    fields = list(dict.fromkeys(k for row in rows for k in row))
    stream = io.StringIO(newline='')
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--roots', nargs='+', type=Path, default=[Path('output/cifar100_LT')],
                        help='Experiment parent directories or individual run directories')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--expected-runs', type=int, default=6)
    parser.add_argument('--out', type=Path, default=Path('output/capt_six_seed42_results.zip'))
    parser.add_argument('--list-only', action='store_true', help='Inspect run identities without creating a ZIP')
    args = parser.parse_args(argv)
    found = {}
    for root in args.roots:
        if not root.is_dir():
            parser.error(f'Root does not exist: {root}')
        for command_file in sorted(root.rglob('command.json')):
            manifest = read_json(command_file)
            opts = options(manifest.get('command', []))
            if (opts.get('--trainer') == 'CAPT' and opts.get('--model') == 'cluster'
                    and opts.get('--seed') == str(args.seed)
                    and opts.get('--capt_reset_global_before_client', '').lower() == 'true'):
                found[command_file.parent.resolve()] = (manifest, opts)
    print(f'Found {len(found)} CAPT global-start runs for seed {args.seed}.')
    rows, details, files = [], [], []
    for index, (path, (manifest, opts)) in enumerate(sorted(found.items()), 1):
        print(f"[{index:02d}] partition={opts.get('--partition')} lambda={opts.get('--specialization_lambda')} "
              f"fixed_freq={opts.get('--capt_fixed_global_agg_freq', '0')} "
              f"reset_optimizer={opts.get('--capt_matched_v2', 'False')}\n     {path}")
        row, detail = inspect_run(path, manifest, opts, index)
        rows.append(row)
        details.append(detail)
        if row['warnings']:
            print('     AUDIT: ' + row['warnings'])
        prefix = f'runs/{row["run_id"]}'
        chosen = {path / name for name in FILES if (path / name).is_file()}
        chosen.update(path.glob('per_class_accuracy_epoch_*.csv'))
        chosen.update((path / 'prompt_params').glob('*.pth'))
        for source in sorted(chosen):
            files.append((source, f'{prefix}/{source.relative_to(path).as_posix()}'))
        # Copied runs retain their original Linux paths, so also search nearby schedules.
        schedules = set()
        recorded = opts.get('--client_schedule_file')
        if recorded and Path(recorded).is_file():
            schedules.add(Path(recorded).resolve())
        for ancestor in [path, *list(path.parents)[:3]]:
            schedules.update(p.resolve() for p in (ancestor / 'schedules').glob('*.json'))
        for i, source in enumerate(sorted(schedules), 1):
            files.append((source, f'{prefix}/schedules/{i:02d}_{source.name}'))
    if args.list_only:
        return
    if len(found) != args.expected_runs:
        parser.error(f'Expected {args.expected_runs} runs, found {len(found)}. '
                     'Use --roots with the three exact experiment roots (or six run directories). No ZIP written.')
    if not rows:
        parser.error('No runs to collect')
    args.out.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation protects existing collections from accidental replacement.
    with zipfile.ZipFile(args.out, mode='x', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        hashes = []
        for source, target in files:
            content = source.read_bytes()
            archive.writestr(target, content)
            hashes.append(dict(archive_path=target, source_path=str(source), bytes=len(content),
                               sha256=hashlib.sha256(content).hexdigest()))
        archive.writestr('summary.csv', '\ufeff' + csv_text(rows))
        archive.writestr('manifest.json', json.dumps(dict(seed=args.seed, runs=details,
                                                         files=hashes), ensure_ascii=False, indent=2))
        archive.writestr('README.txt',
            'CAPT global-start collection. Raw artifacts are under runs/run_XX/.\n'
            'See manifest.json for original paths, full commands, source hashes, and collection checksums.\n'
            'summary.csv: final = last evaluated server state; epochs are reported as one-based outer rounds.\n'
            'Last-20 means include ONLY evaluations in the final 20 requested outer rounds; not 20 evaluations.\n'
            'Best overall and best tail may be DIFFERENT checkpoints. All accuracy/F1 values use percent units.\n'
            'This is one seed, not multi-seed statistics. Warnings must be reviewed before comparing runs.\n'
            'Matching matrices and schedules is necessary but not sufficient for a controlled comparison.\n'
            'Use command options to identify protocols; lambda applies to Client-LT, not fine-Dirichlet.\n')
        archive.write(Path(__file__), 'collector_source.py')
    print(f'Written: {args.out.resolve()} ({args.out.stat().st_size / 1024 / 1024:.1f} MiB)')
    print('Includes summary.csv, manifest.json, raw logs/metrics, per-class results, schedules and prompt snapshots.')


if __name__ == '__main__':
    main()
