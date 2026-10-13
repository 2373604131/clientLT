"""Fixed two-by-two ablation: ordinary/new A crossed with absent/new B."""
from pathlib import Path
from statistics import mean

from scripts.run_ab_validation import write_json
from tools.client_aggregation.protocol import load_json, read_csv, write_table

EXPERIMENT = 'ab_factorial_joint_v1'
CELLS = {
    'plain_joint': ('plain_a', False),
    'new_a_joint': ('method_a_margin', False),
    'new_a_new_b_joint': ('method_a_margin', True),
    'plain_new_b_joint': ('plain_a', True),
}


def enabled(job):
    return job.get('settings', {}).get('factorial_experiment') == EXPERIMENT


def has_new_b(job):
    return enabled(job) and CELLS[job['settings']['factorial_cell']][1]


def contract_for(cell, parameters):
    from tools.client_aggregation.protocol import PLAIN_A_CONTRACT, method_a_margin_contract
    from utils.cliplora_ab_factorial import transfer_config
    arm, new_b = CELLS[cell]
    value = dict(method_a_margin_contract(parameters) if arm == 'method_a_margin' else PLAIN_A_CONTRACT)
    value.update(experiment=EXPERIMENT, factorial_cell=cell, method_B=new_b,
        new_B=transfer_config('formal') if new_b else None,
        phase_order='ordinary B, optional F-tail-equal B, ordinary A through90, optional new A, commit',
        primary='committed rounds81..100 mean Overall; jointly report Head20/Middle60/Tail20',
        interpretation='two-by-two component contrasts; equal local SGD, additional feedback cost reported')
    return value


def validate(job):
    settings = job['settings']
    cell = settings.get('factorial_cell')
    if (cell not in CELLS or job['arm'] != CELLS[cell][0]
            or job['contract'] != contract_for(cell, settings.get('margin_parameters', {}))
            or settings.get('client_concurrency') != 8
            or settings.get('seed') != 42 or job['mode'] not in ('formal', 'smoke')
            or job['aggregation'].get('rule', 'joint') != 'joint'
            or job['aggregation']['lambda_value'] != .1):
        raise ValueError('Factorial job must use its fixed cell, joint0.1, seed42 and eight clients per GPU')


def audit_b(run, job, progress, completed):
    from tools.sfra.b_routes import audit_feedback_coefficients, finite
    from utils.cliplora_ab_factorial import transfer_config
    expected = [r for r in transfer_config(job['mode'])['rounds'] if r <= completed]
    rows = read_csv(run/'b_transfer_rounds.csv')
    if ([int(r['round']) for r in rows] != expected
            or progress.get('b_transfer_events') != len(expected)
            or progress.get('b_transfer_optimizer_steps') != 2*len(expected)):
        raise ValueError('Wrong F-tail-equal schedule or optimizer budget')
    for row in rows:
        folder = run/'b_transfer_rounds'/f'r{int(row["round"]):03d}'
        steps = read_csv(folder/'optimization_steps.csv')
        if (row.get('route_arm') != 'F-tail-equal' or int(row['optimizer_steps']) != 2
                or [int(s['step']) for s in steps] != [1, 2]):
            raise ValueError('New B must commit its fixed two free-update steps')
        for step in steps:
            if finite(step['effective_transfer_norm_after']) > finite(step['radius'])*(1+2e-6)+1e-12:
                raise ValueError('New B exceeded its effective-update radius')
        audit_feedback_coefficients(run, folder, job['aggregation']['counts']['tail_ids'])


def feedback_preflight(reference):
    """Verify every declared tail is covered in both fixed batches before GPU work."""
    from collections import defaultdict
    from utils.cliplora_b_shared_transfer import shared_calibration_batches
    from utils.b_tail_equal import manifest_coefficients
    from tools.client_aggregation.protocol import count_matrix, B_ROUNDS
    _, counts, _ = count_matrix(reference)
    groups = defaultdict(lambda: defaultdict(list))
    for row in read_csv(Path(reference)/'partition_manifest.csv'):
        groups[int(row['client_id'])][int(row['class_id'])].append(int(row['local_position']))
    tail = set(counts['tail_ids'])
    recipients = [k for k in sorted(groups) if tail.intersection(groups[k])]
    for rnd in (1, 6, *B_ROUNDS):
        manifests = {k: dict(batches=shared_calibration_batches(groups[k], tail, 42, rnd, k, 'class-cyclic'))
            for k in recipients}
        manifest_coefficients(manifests, groups, tail)
    return dict(checked_rounds=[1, 6, *B_ROUNDS], feedback_clients=recipients, tail_classes=len(tail))


def pair_check(left, right, historical=False):
    """Audit pairing separately from performance; changing A/B is intentional."""
    from tools.client_aggregation.frozen_tailrw import paired_audit
    from tools.client_aggregation.method_a import INTEGRATION_FILES
    pair = paired_audit(left, right)
    allowed = set(INTEGRATION_FILES) | {'federated_main.py', 'utils/cliplora_sfra.py',
        'utils/method_a_margin_math.py', 'utils/method_a_margin_feedback.py', 'utils/cliplora_method_a_margin.py',
        'tools/client_aggregation/method_a_margin.py', 'scripts/run_cliplora_method_a_margin.py',
        'utils/cliplora_ab_factorial.py', 'tools/client_aggregation/ab_factorial.py',
        'scripts/run_cliplora_ab_factorial.py', 'utils/b_route_math.py', 'utils/b_tail_equal.py',
        'utils/cliplora_b_routes.py', 'tools/sfra/b_routes.py'}
    intentional = {'variant', 'budget.functional_correction_steps'}
    pair['mismatches'] = [m for m in pair['mismatches'] if m not in intentional
        and not (historical and m.startswith('code.') and m[5:] in allowed)]
    if not historical and pair['source_differences']:
        pair['mismatches'].append('source_differences')
    if historical:
        for name in pair['source_differences']:
            if name not in allowed and 'code.'+name not in pair['mismatches']:
                pair['mismatches'].append('code.'+name)
    if left['job']['aggregation']['weights'] != right['job']['aggregation']['weights']:
        pair['mismatches'].append('aggregation.weights')
    pair.update(status='mismatch' if pair['mismatches'] else
        'historical' if historical else 'execution_differs' if not pair['same_recorded_environment'] else 'eligible',
        total_compute_budget_matched=False, historical_baseline=historical)
    return pair


def summarize(root):
    from scripts.run_cliplora_joint_aggregation import make_job
    from tools.client_aggregation.analysis import METRICS, audit_run, sample_dynamics, stage_comparisons
    root = Path(root)
    suite = load_json(root/'factorial_plan.json')
    statuses, results, performance, costs, dynamics, stages = [], {}, [], [], [], []
    for cell, item in suite['cells'].items():
        cell_root = Path(item['root'])
        plan = load_json(cell_root/'experiment_plan.json')
        job = make_job(plan, CELLS[cell][0])
        run = cell_root/job['run']
        p = run/'progress.json'
        progress = load_json(p) if p.is_file() else {}
        row = dict(cell=cell, status='incomplete' if progress else 'not_started',
            completed_round=progress.get('completed_round', 0), reused=item['reused'], reason='')
        if (run/'completion.json').is_file():
            try:
                result = audit_run(run, job)
                results[cell] = result
                row.update(status='complete', completed_round=100)
                performance.append(dict(cell=cell, **{m: mean(float(r[m]) for r in result['curves'][81:101]) for m in METRICS}))
                costs.append(dict(result['progress'], cell=cell))
                dynamics.extend(dict(r, cell=cell) for r in sample_dynamics(result))
                stages.extend(dict(r, cell=cell) for r in stage_comparisons(result))
            except (ValueError, OSError, KeyError, TypeError, AssertionError) as error:
                row.update(status='invalid', reason=str(error))
        statuses.append(row)
        print(row)
    perf = {r['cell']: r for r in performance}
    pairs = []
    for label, left, right in (
            ('new_A_without_B', 'new_a_joint', 'plain_joint'),
            ('new_B_given_new_A', 'new_a_new_b_joint', 'new_a_joint'),
            ('new_A_given_new_B', 'new_a_new_b_joint', 'plain_new_b_joint'),
            ('new_B_given_plain_A', 'plain_new_b_joint', 'plain_joint')):
        pair = dict(contrast=label, left=left, right=right, status='unavailable')
        if left in results and right in results:
            pair.update(pair_check(results[left], results[right], suite['cells'][right]['reused']))
            if pair['status'] != 'mismatch':
                pair['delta_pp'] = {m: perf[left][m]-perf[right][m] for m in METRICS}
        pairs.append(pair)
    interaction = None
    if len(results) == 4 and all(p['status'] not in ('mismatch', 'unavailable') for p in pairs):
        interaction = {m: perf['new_a_new_b_joint'][m]-perf['new_a_joint'][m]
            -perf['plain_new_b_joint'][m]+perf['plain_joint'][m] for m in METRICS}
    out = root/'analysis'
    out.mkdir(exist_ok=True)
    for name, rows in dict(status=statuses, performance=performance, costs=costs,
            sample_dynamics=dynamics, stage_effects=stages).items():
        write_table(out/(name+'.csv'), rows)
    write_json(out/'contrasts.json', dict(pairs=pairs, interaction_pp=interaction,
        independent_seeds=1, endpoint='committed rounds81..100 mean; rounds are not independent seeds'))
    return statuses, pairs
