"""Read existing shared-B class metrics; no model loading or training."""
import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean


WORKSPACE = Path(__file__).resolve().parents[2]
DEFAULT_RUN = WORKSPACE / (
    'output/sfra_b_shared_transfer_analysis/sfra_b_shared_transfer/seed42/'
    'client-longtail/full-cp/'
    'lambda10_mu1_protocol42_bshared_b_lr0.3_probe0.1_reg0.001_fast_v2_f64_c4'
)


def write_csv(path, rows):
    with path.open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def analyze(run, output):
    units, summaries = [], []
    folders = sorted((run / 'b_transfer_rounds').glob('r*'))
    if not folders:
        raise ValueError('No shared-transfer events found')
    for folder in folders:
        phases = defaultdict(dict)
        with (folder / 'probe_metrics.csv').open(encoding='utf-8-sig', newline='') as handle:
            for row in csv.DictReader(handle):
                key = (int(row['client_id']), int(row['class_id']))
                if key in phases[row['phase']]:
                    raise ValueError(f'Duplicate metric: {folder.name}, {row["phase"]}, {key}')
                phases[row['phase']][key] = row
        before, after = phases['ordinary_global_B'], phases['transferred_global_B']
        if not before or before.keys() != after.keys():
            raise ValueError(f'Missing or unmatched shared-B phases in {folder}')
        event_units = []
        for key, b in sorted(before.items()):
            a = after[key]
            if b['samples'] != a['samples'] or b['is_tail'] != a['is_tail']:
                raise ValueError(f'Unmatched measurement population in {folder}, {key}')
            gain = float(b['la']) - float(a['la'])
            item = dict(round=int(b['round']), client_id=key[0], class_id=key[1],
                        group='tail' if b['is_tail'].lower() == 'true' else 'non_tail',
                        samples=int(b['samples']), before_la=float(b['la']),
                        after_la=float(a['la']), la_gain=gain,
                        positive_gain=max(gain, 0.), harm=max(-gain, 0.),
                        accuracy_change_pp=float(a['accuracy']) - float(b['accuracy']))
            event_units.append(item)
        units.extend(event_units)
        for group in ('tail', 'non_tail'):
            selected = [r for r in event_units if r['group'] == group]
            by_class = defaultdict(list)
            for row in selected:
                by_class[row['class_id']].append(row)
            if not selected:
                continue
            macro = {field: mean(mean(r[field] for r in values) for values in by_class.values())
                     for field in ('la_gain', 'positive_gain', 'harm', 'accuracy_change_pp')}
            assert math.isclose(macro['la_gain'], macro['positive_gain'] - macro['harm'], abs_tol=1e-12)
            summaries.append(dict(
                round=selected[0]['round'], group=group, units=len(selected), classes=len(by_class),
                improved_units=sum(r['la_gain'] > 0 for r in selected),
                harmed_units=sum(r['la_gain'] < 0 for r in selected),
                unchanged_units=sum(r['la_gain'] == 0 for r in selected),
                harmed_class_means=sum(mean(r['la_gain'] for r in values) < 0 for values in by_class.values()),
                class_macro_la_gain=macro['la_gain'], class_macro_positive_gain=macro['positive_gain'],
                class_macro_harm=macro['harm'], maximum_unit_harm=max(r['harm'] for r in selected),
                class_macro_accuracy_change_pp=macro['accuracy_change_pp']))
    report = dict(
        source_run=str(run), events=len(folders),
        comparison='ordinary_global_B to transferred_global_B, before the same-round A phase',
        sign_rule='gain = before_LA - after_LA; strict zero, no threshold tuning',
        weighting='mean within each class across receivers, then mean over observed classes',
        limitations=[
            'Training-side probes may overlap calibration; not held-out generalization evidence.',
            'Round-receiver-class rows are repeated measurements, not independent seeds.',
            'Non-tail probes cover only observed classes and feedback clients.',
            'These counts cannot be compared as like-for-like percentages with raw donor-pair conflict rates.'
        ], totals={})
    for group in ('tail', 'non_tail'):
        selected = [r for r in units if r['group'] == group]
        per_round = [r for r in summaries if r['group'] == group]
        report['totals'][group] = dict(
            units=len(selected), improved_units=sum(r['la_gain'] > 0 for r in selected),
            harmed_units=sum(r['la_gain'] < 0 for r in selected),
            unchanged_units=sum(r['la_gain'] == 0 for r in selected),
            mean_round_class_macro_gain=mean(r['class_macro_la_gain'] for r in per_round),
            mean_round_class_macro_harm=mean(r['class_macro_harm'] for r in per_round))
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / '问题2_旧共享B_逐类变化.csv', units)
    write_csv(output / '问题2_旧共享B_逐轮汇总.csv', summaries)
    (output / '问题2_旧共享B_诊断摘要.json').write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(dict(events=report['events'], totals=report['totals']), ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, default=DEFAULT_RUN)
    parser.add_argument('--output', type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    analyze(args.run, args.output)
