"""New-A run checks and independent summaries. Loss values never select a model."""
import math
from pathlib import Path
from statistics import mean

import numpy as np

from scripts.run_ab_validation import write_json
from tools.client_aggregation.protocol import load_json, read_csv, write_table
from utils.method_a_margin_math import ARM, VARIANT, algorithm_config


def audit_rounds(rows, progress, completed):
    if [int(r['round']) for r in rows] != list(range(1, completed+1)):
        raise ValueError('Missing new-A submitted round records')
    total = 0
    for row in rows:
        rnd, steps = int(row['round']), int(row['correction_steps'])
        if (row['variant'] != VARIANT or row['commit_rule'] != algorithm_config()['commit_rule']
                or steps not in ((0, 3) if rnd <= 90 else (0,))):
            raise ValueError('New A must commit its fixed third step without candidate selection')
        if any(k in row for k in ('selected_beta', 'full_reject', 'half_reject', 'limit_CE_H')):
            raise ValueError('Candidate acceptance/rollback fields do not belong to new A')
        if int(row['history_last_committed']) != min(rnd, 90):
            raise ValueError('New-A history advanced outside the submitted schedule')
        if int(row['submitted_step']) != steps:
            raise ValueError('New A did not submit the last computed update')
        started = str(row['correction_started']).lower() == 'true'
        if started != (steps == 3):
            raise ValueError('New-A trigger/update mismatch')
        if steps == 0:
            permitted = (('no_valid_history', 'zero_proposal_radius', 'no_history_deficit')
                         if rnd <= 90 else ('B_only_schedule',))
            if row['skip_reason'] not in permitted:
                raise ValueError('New-A updates were skipped for an unregistered reason')
        elif row['skip_reason']:
            raise ValueError('New A cannot skip after computing updates')
        total += steps
        if rnd > 90:
            continue
        if int(row['history_last_used']) != rnd-1 or int(row['window_count']) != rnd % 5:
            raise ValueError('New-A history must be used in the next round')
        if not 0 <= float(row['correction_z_norm']) <= 1+1e-6:
            raise ValueError('New-A global correction radius was exceeded')
        for key in ('radius', 'ordinary_keep', 'final_keep', 'ordinary_class', 'final_class',
                    'J_ordinary', 'J_submitted', *(f'{phase}_CE_{g}' for phase in ('ordinary', 'final') for g in ('H', 'M', 'T'))):
            if not math.isfinite(float(row[key])):
                raise ValueError('Nonfinite new-A record: '+key)
        if steps and (int(row['ordinary_deficit_count']) <= 0 or float(row['radius']) <= 0):
            raise ValueError('New-A correction has no gap or no radius')
        # Do not require lower CE or J: neither is an acceptance criterion in this algorithm.
    if total != progress['functional_correction_steps']:
        raise ValueError('New-A update budget differs')


def status(root):
    run = Path(root)/'runs/seed42'/ARM
    complete = (run/'completion.json').is_file()
    path = run/('completion.json' if complete else 'progress.json')
    progress = load_json(path) if path.is_file() else {}
    return dict(arm=ARM, status='completed_unverified' if complete else 'incomplete' if progress else 'not_started',
                completed_round=progress.get('completed_round', 0))


def compare_plain(candidate, baseline):
    from tools.client_aggregation.frozen_tailrw import paired_audit
    from tools.client_aggregation.method_a import INTEGRATION_FILES
    if candidate['job']['arm'] != ARM or baseline['job']['arm'] != 'plain_a':
        raise ValueError('New A compares to same-aggregation ordinary A only')
    pair = paired_audit(candidate, baseline)
    allowed = INTEGRATION_FILES | {'federated_main.py', 'utils/cliplora_sfra.py',
        'utils/method_a_margin_math.py', 'utils/method_a_margin_feedback.py',
        'utils/cliplora_method_a_margin.py', 'tools/client_aggregation/method_a_margin.py',
        'scripts/run_cliplora_method_a_margin.py'}
    intentional = {'variant', 'budget.functional_correction_steps', *('code.'+p for p in allowed)}
    pair['mismatches'] = [m for m in pair['mismatches'] if m not in intentional]
    for name in pair['source_differences']:
        if name not in allowed and 'code.'+name not in pair['mismatches']:
            pair['mismatches'].append('code.'+name)
    if (baseline['job']['aggregation'].get('rule', 'joint') != 'joint'
            or not np.allclose(candidate['job']['aggregation']['weights'], baseline['job']['aggregation']['weights'], rtol=0, atol=1e-8)):
        pair['mismatches'].append('aggregation.weights')
    pair.update(status='mismatch' if pair['mismatches'] else 'execution_differs' if not pair['same_recorded_environment'] else 'eligible',
        contrast='necessary-margin A minus same-joint ordinary A; no supplementary B',
        local_SGD_budget_matched=not any(m.startswith('budget.') for m in pair['mismatches']),
        total_compute_budget_matched=False)
    return pair


def summarize(root, baseline_root):
    from scripts.run_cliplora_joint_aggregation import make_job
    from tools.client_aggregation.analysis import METRICS, audit_run, sample_dynamics, stage_comparisons
    from tools.client_aggregation.plain_a import load_plain
    root = Path(root)
    out = root/'analysis'
    out.mkdir(parents=True, exist_ok=True)
    row = status(root)
    performance, costs = [], []
    pair = dict(status='unavailable', baseline_root=str(baseline_root))
    if row['status'] == 'completed_unverified':
        try:
            job = make_job(load_json(root/'experiment_plan.json'), ARM)
            result = audit_run(root/job['run'], job)
            def endpoint(r):
                return {m: mean(float(x[m]) for x in r['curves'] if 81 <= int(x['round']) <= 100) for m in METRICS}
            scores = endpoint(result)
            performance.append(dict(arm=ARM, **scores))
            costs.append(result['progress'])
            write_table(out/'sample_dynamics.csv', sample_dynamics(result))
            write_table(out/'stage_effects.csv', stage_comparisons(result))
            write_table(out/'margin_rounds.csv', read_csv(root/job['run']/'sfra_rounds.csv'))
            row['status'] = 'complete'
            if (Path(baseline_root)/'runs/seed42/plain_a/completion.json').is_file():
                try:
                    baseline = load_plain(baseline_root, 'joint')
                    pair.update(compare_plain(result, baseline))
                    if pair['status'] == 'eligible':
                        base = endpoint(baseline)
                        performance.append(dict(arm='plain_a', **base))
                        pair['delta_pp'] = {m: scores[m]-base[m] for m in METRICS}
                except (ValueError, OSError, KeyError) as error:
                    pair.update(status='unavailable', reason=str(error))
        except (ValueError, OSError, KeyError) as error:
            row.update(status='invalid', reason=str(error))
    write_table(out/'status.csv', [row])
    write_table(out/'performance.csv', performance)
    write_table(out/'costs.csv', costs)
    write_json(out/'comparison.json', dict(pair=pair, independent_seeds=1,
        endpoint='submitted rounds81..100 mean; not independent repetitions',
        note='No test-driven updates, acceptance, rollback, automatic baselines or B combination'))
    print(row)
    return row
