"""Opt-in pairing of unchanged margin A and the archived shared donor-C B."""
import copy
import math
from pathlib import Path
from types import SimpleNamespace

EXPERIMENT = 'new_a_old_b_joint_v1'
RUN = 'runs/seed42/method_a_margin'
EVALUATION = ('CIFAR-100 official test, previously inspected for development; '
              'these are development comparisons, not independent test conclusions.')
INTEGRATION_FILES = {
    'tools/client_aggregation/protocol.py', 'tools/client_aggregation/runtime.py',
    'tools/client_aggregation/analysis.py', 'tools/client_aggregation/new_a_old_b.py',
    'scripts/run_cliplora_new_a_old_b.py',
    'scripts/run_cliplora_joint_aggregation.py',
    # Previously added optional A step-size argument, with unchanged default 0.1.
    'federated_main.py', 'utils/method_a_margin_math.py', 'utils/cliplora_method_a_margin.py',
}


def enabled(job):
    from tools.client_aggregation.a_schedule import enabled as scheduled
    return job.get('settings', {}).get('b_comparison_experiment') == EXPERIMENT or scheduled(job)


def transfer_config(mode='formal'):
    from utils.cliplora_b_shared_transfer import shared_transfer_config
    if mode not in ('formal', 'smoke'):
        raise ValueError('Unknown old-B mode')
    args = SimpleNamespace(b_transfer_lr=.3, b_transfer_probe_step=.1, b_transfer_reg=.001,
                          b_transfer_non_tail_sampling='class-cyclic', b_transfer_tail_weight=.35)
    cfg = shared_transfer_config(args)
    cfg.update(aggregation='ordinary_joint_class_weighted_B_then_one_shared_residual',
               calibration_anchor='actual ordinary joint-class-weighted B with round-start A')
    if mode == 'smoke':
        cfg['rounds'] = [1]  # One disposable CUDA round exercises the actual donor/C integration.
    return cfg


def contract_for(parameters):
    from tools.client_aggregation.protocol import method_a_margin_contract
    return dict(method_a_margin_contract(parameters), experiment=EXPERIMENT,
                method_B=True, source_C_transfer=True, source_feedback=True,
                old_B=transfer_config(),
                phase_order='ordinary joint B, scheduled shared donor C, ordinary joint A through90, margin A, commit history',
                evaluation=EVALUATION,
                interpretation='same-A comparison of absent/archived/current supplemental B; local SGD matched, extra costs reported')


def validate(job):
    s = job['settings']
    if (job['arm'] != 'method_a_margin' or job['mode'] not in ('formal', 'smoke')
            or s.get('seed') != 42 or s.get('client_concurrency') != 8
            or s.get('factorial_experiment') or s.get('factorial_cell')
            or job['aggregation'].get('rule', 'joint') != 'joint'
            or job['aggregation']['lambda_value'] != .1
            or job['contract'] != contract_for(s.get('margin_parameters', {}))):
        raise ValueError('New A + old B requires its registered seed42 joint0.1 contract')


def candidate_plan(reference, hashes):
    from scripts.run_cliplora_joint_aggregation import make_job
    from tools.client_aggregation.protocol import validate_arm
    old = make_job(reference, 'method_a_margin')
    validate_arm(old)
    if (reference['settings'].get('factorial_cell') not in ('new_a_joint', 'new_a_new_b_joint')
            or reference['settings'].get('client_concurrency') != 8):
        raise ValueError('A reference must be an existing eight-client new-A joint run')
    plan = copy.deepcopy(reference)
    plan['settings'].pop('factorial_experiment')
    plan['settings'].pop('factorial_cell')
    plan['settings']['b_comparison_experiment'] = EXPERIMENT
    plan['contract'] = contract_for(plan['settings']['margin_parameters'])
    if plan['contract']['method_a_margin'] != reference['contract']['method_a_margin']:
        raise ValueError('A configuration changed while attaching old B')
    plan['code_sha256'] = hashes
    make_job(plan, 'method_a_margin')
    return plan


def audit_b(run, job, progress, completed):
    from tools.client_aggregation.protocol import load_json, read_csv
    run = Path(run)
    rows = read_csv(run/'b_transfer_rounds.csv')
    expected = [r for r in transfer_config(job['mode'])['rounds'] if r <= completed]
    if [int(r['round']) for r in rows] != expected or progress.get('b_transfer_events') != len(expected):
        raise ValueError('Old-B event schedule differs')
    total = 0
    for row in rows:
        folder = run/'b_transfer_rounds'/f'r{int(row["round"]):03d}'
        manifest = load_json(folder/'calibration_manifest.json')
        donors = manifest['shared_donor_ids']
        wanted = 2 if donors else 0
        steps = read_csv(folder/'optimization_steps.csv')
        if (row['mode'] != 'shared' or row.get('calibration_profile') != 'coverage_tradeoff'
                or row.get('route_arm') or int(row['optimizer_steps']) != wanted
                or [int(s['step']) for s in steps] != list(range(1, wanted+1))
                or int(row['feedback_synchronizations']) != wanted
                or float(row['learning_rate']) != .3 or float(row['tail_weight']) != .35
                or row['non_tail_sampling'] != 'class-cyclic'):
            raise ValueError('Old-B C optimizer, loss or sampling changed')
        if not math.isfinite(float(row['effective_global_transfer_norm'])):
            raise ValueError('Nonfinite old-B effective update')
        total += wanted
    if progress.get('b_transfer_optimizer_steps') != total:
        raise ValueError('Old-B total optimizer budget differs')


def pair_check(candidate, reference):
    from tools.client_aggregation.frozen_tailrw import paired_audit
    pair = paired_audit(candidate, reference)
    intentional = {'budget.functional_correction_steps', *('code.'+p for p in INTEGRATION_FILES)}
    pair['mismatches'] = [m for m in pair['mismatches'] if m not in intentional]
    for field, a, b in (
            ('A.configuration', candidate['job']['contract']['method_a_margin'], reference['job']['contract']['method_a_margin']),
            ('aggregation', candidate['job']['aggregation'], reference['job']['aggregation'])):
        if a != b:
            pair['mismatches'].append(field)
    for name in pair['source_differences']:
        if name not in INTEGRATION_FILES and 'code.'+name not in pair['mismatches']:
            pair['mismatches'].append('code.'+name)
    pair.update(status='mismatch' if pair['mismatches'] else
                'eligible' if pair['same_recorded_environment'] else 'environment_differs',
                contrast='same-A old donor-C B versus registered reference',
                budget_matched=False,
                local_SGD_budget_matched=not any(m.startswith('budget.') for m in pair['mismatches']),
                total_compute_budget_matched=False,
                note='History and gap-triggered A step counts may diverge after B; no history is shared.')
    return pair
