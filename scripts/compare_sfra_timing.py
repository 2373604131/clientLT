"""Read-only comparison of matched SFRA runs; no training dependencies."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from statistics import mean


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def load_timings(root):
    progress = read_json(root / 'progress.json')
    completed = int(progress['completed_round'])
    rows = {}
    for filename, kind in (('event_manifest.csv', 'train'), ('sfra_costs.csv', 'feedback'),
                           ('evaluation_budget.csv', 'test')):
        with (root / filename).open(encoding='utf-8-sig', newline='') as stream:
            for record in csv.DictReader(stream):
                if kind == 'test':
                    if not record['label'].startswith('official_round_'):
                        continue
                    rnd = int(record['label'].rsplit('_', 1)[1])
                    phase = 'official_test'
                else:
                    if kind == 'train' and record.get('branch', 'main') != 'main':
                        continue
                    rnd, phase = int(record['round']), record['phase']
                if not 1 <= rnd <= completed:
                    continue
                seconds = float(record['seconds'])
                if not math.isfinite(seconds) or seconds < 0:
                    raise ValueError(f'Invalid timing in {root / filename}: {seconds}')
                stage = f'{kind}/{phase}'
                bucket = rows.setdefault(rnd, {})
                bucket[stage] = bucket.get(stage, 0.) + seconds
    return progress, rows


def verify_comparable(left, right):
    if read_json(left / 'sfra_config.json') != read_json(right / 'sfra_config.json'):
        raise ValueError('Algorithm/training configurations differ; speedup would not be a matched comparison')
    for filename in ('partition_manifest.csv', 'protocol/full_schedule.json'):
        digest = lambda root: hashlib.sha256((root / filename).read_bytes()).hexdigest()
        if digest(left) != digest(right):
            raise ValueError(f'Experiment protocol differs: {filename}')
    a, b = (read_json(root / 'bridge_metadata.json') for root in (left, right))
    for key in ('initial_lora_sha256', 'frozen_model_sha256', 'test_sha256'):
        if not a.get(key) or a[key] != b.get(key):
            raise ValueError(f'Missing or different initial model/test fingerprint: {key}')


def compare(left, right, first_round=2, last_round=90):
    verify_comparable(left, right)
    lp, a = load_timings(left)
    rp, b = load_timings(right)
    common = sorted(r for r in a.keys() & b.keys() if first_round <= r <= last_round)
    if not common:
        raise ValueError('No common completed rounds in the requested range')
    print(f'Baseline: {left}\nCandidate: {right}')
    print('Matched completed rounds: ' + ', '.join(map(str, common)))
    print(f'{"Stage (seconds/round)":42s} {"Baseline":>10s} {"Candidate":>10s} {"Speedup":>9s}')
    phases = sorted(set().union(*(a[r].keys() | b[r].keys() for r in common)))
    for phase in phases:
        old = mean(a[r].get(phase, 0.) for r in common)
        new = mean(b[r].get(phase, 0.) for r in common)
        if old == new == 0:
            continue
        ratio = f'{old / new:.2f}x' if new else 'n/a'
        print(f'{phase:42s} {old:10.3f} {new:10.3f} {ratio:>9s}')
    totals = []
    for title, prefix in (('Feedback subtotal', 'feedback/'), ('All recorded stages', '')):
        old = mean(sum(v for k, v in a[r].items() if k.startswith(prefix)) for r in common)
        new = mean(sum(v for k, v in b[r].items() if k.startswith(prefix)) for r in common)
        ratio = f'{old / new:.2f}x' if new else 'n/a'
        print(f'{title:42s} {old:10.3f} {new:10.3f} {ratio:>9s}')
        totals.append((old, new))
    print('Stage sums exclude untimed checkpoint/I/O overhead; they are not whole-job wall time.')
    if lp['completed_round'] == rp['completed_round']:
        old, new = float(lp['elapsed_seconds']), float(rp['elapsed_seconds'])
        if new > 0:
            print(f'Runtime elapsed through round {lp["completed_round"]} (including cold cache/checkpoints): '
                  f'{old:.1f}s -> {new:.1f}s, {old/new:.2f}x')
    else:
        print('Total elapsed times have different completed-round counts; no total-time speedup reported.')
    print('Use the same GPU and CPU allocation, run sequentially, and check accuracy separately. '
          'Rounds 2-3 do not measure the B-transfer events starting at round 30.')
    return totals


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('baseline', type=Path)
    parser.add_argument('candidate', type=Path)
    parser.add_argument('--first-round', type=int, default=2)
    parser.add_argument('--last-round', type=int, default=90)
    args = parser.parse_args()
    if not 1 <= args.first_round <= args.last_round <= 100:
        parser.error('Require 1 <= first-round <= last-round <= 100')
    try:
        compare(args.baseline, args.candidate, args.first_round, args.last_round)
    except (OSError, KeyError, ValueError) as error:
        parser.exit(1, f'{error}\n')


if __name__ == '__main__':
    main()
