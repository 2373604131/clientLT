"""Three fixed A schedules with the unchanged archived donor-C B."""
import copy
import math

EXPERIMENT = 'selected_a_schedule_old_b_v1'
SCHEDULES = {'a_until60': list(range(1, 61)), 'a_every10': list(range(10, 91, 10)), 'a_frozen': []}
FILES = {'tools/client_aggregation/a_schedule.py', 'utils/cliplora_margin_a_schedule.py',
         'scripts/run_cliplora_a_schedule.py'}


def enabled(job):
    return job.get('settings', {}).get('a_schedule_experiment') == EXPERIMENT


def rounds_for(job, completed=100):
    rounds = SCHEDULES[job['settings']['a_schedule_cell']] if enabled(job) else ([] if job['arm']=='frozen' else range(1, 91))
    return [r for r in rounds if r <= completed]


def history_stop(job):
    return max(rounds_for(job), default=0)


def contract_for(cell, parameters):
    from tools.client_aggregation.new_a_old_b import contract_for as original
    rounds = SCHEDULES[cell]
    value = original(parameters)
    value.update(experiment=EXPERIMENT, A_rounds=rounds, extra_A_steps=352*len(rounds),
        total_local_steps=105600+352*len(rounds), max_correction_steps=3*len(rounds),
        final_rounds='A changes only on the registered rounds; ordinary and archived B keep their original schedule',
        interpretation='A schedule intervention; A local SGD and feedback budgets intentionally differ',
        history_policy='original consecutive five-calendar-round windows through last A update; forward-only observations on intervening B-only rounds; no history for fully frozen A')
    value['method_a_margin'] = dict(value['method_a_margin'], refresh_rounds=rounds,
        history_stop_round=max(rounds, default=0))
    return value


def candidate_plan(reference, cell, hashes):
    from scripts.run_cliplora_joint_aggregation import make_job
    plan = copy.deepcopy(reference)
    for k in ('factorial_experiment', 'factorial_cell', 'b_comparison_experiment'):
        plan['settings'].pop(k, None)
    plan['settings'].update(a_schedule_experiment=EXPERIMENT, a_schedule_cell=cell)
    plan['contract'] = contract_for(cell, plan['settings']['margin_parameters'])
    plan['code_sha256'] = hashes
    make_job(plan, 'method_a_margin')
    return plan


def validate(job):
    s = job['settings']
    if (s.get('a_schedule_cell') not in SCHEDULES or job['arm'] != 'method_a_margin'
            or job['mode'] != 'formal' or s.get('seed') != 42 or s.get('client_concurrency') != 8
            or any(s.get(k) for k in ('factorial_experiment', 'b_comparison_experiment'))
            or job['aggregation'].get('rule', 'joint') != 'joint' or job['aggregation']['lambda_value'] != .1
            or job['contract'] != contract_for(s['a_schedule_cell'], s['margin_parameters'])):
        raise ValueError('A schedule differs from its fixed old-B experiment contract')
    p = s['margin_parameters']
    if (p['keep_weight'], p['learn_weight'], p.get('correction_step_size')) != (1., 3., 1.):
        raise ValueError('A schedules must keep the selected S4 coefficients and step size')


def configure(runtime, job):
    rounds = rounds_for(job)
    runtime.rounds, runtime.periodic = rounds, bool(rounds)
    steps = sum((n+31)//32 for n in runtime.sizes)
    runtime.config.update(candidate_rounds=rounds, extra_steps_expected=steps*len(rounds),
                          extra_trainable_factor='A' if rounds else None)
    runtime.sfra_config.update(refresh_rounds=rounds, history_stop_round=history_stop(job))


def audit_rounds(rows, progress, completed, job):
    from utils.method_a_margin_math import VARIANT, algorithm_config
    active = set(rounds_for(job, completed))
    end = history_stop(job)
    if [int(r['round']) for r in rows] != list(range(1, completed+1)):
        raise ValueError('Missing scheduled-A round records')
    total = 0
    for row in rows:
        rnd, steps = int(row['round']), int(row['correction_steps'])
        if (row['variant'] != VARIANT or row['commit_rule'] != algorithm_config()['commit_rule']
                or steps not in ((0,3) if rnd in active else (0,))
                or int(row['submitted_step']) != steps
                or (str(row['correction_started']).lower()=='true') != (steps==3)
                or int(row['history_last_committed']) != min(rnd, end)):
            raise ValueError('Scheduled A correction/history mismatch')
        if rnd not in active:
            if row['skip_reason'] != 'A_frozen_schedule':
                raise ValueError('Off-schedule A must be explicitly frozen')
        else:
            if int(row['history_last_used']) != rnd-1 or int(row['window_count']) != rnd%5:
                raise ValueError('Sparse A must use preceding calendar-round history')
            if steps == 0 and row['skip_reason'] not in ('no_valid_history','zero_proposal_radius','no_history_deficit'):
                raise ValueError('Unexpected scheduled correction skip')
            if steps and row['skip_reason']:
                raise ValueError('Cannot skip after correction')
            if not 0 <= float(row['correction_z_norm']) <= 1+1e-6:
                raise ValueError('Correction radius exceeded')
            for key in ('radius','ordinary_keep','final_keep','ordinary_class','final_class','J_ordinary','J_submitted',
                        *(f'{p}_CE_{g}' for p in ('ordinary','final') for g in ('H','M','T'))):
                if not math.isfinite(float(row[key])):
                    raise ValueError('Nonfinite correction: '+key)
            if steps and (int(row['ordinary_deficit_count']) <= 0 or float(row['radius']) <= 0):
                raise ValueError('Correction has no gap/radius')
        if any(k in row for k in ('selected_beta','full_reject','half_reject','limit_CE_H')):
            raise ValueError('Unexpected fallback/acceptance fields')
        total += steps
    if total != progress['functional_correction_steps']:
        raise ValueError('Functional update budget differs')
