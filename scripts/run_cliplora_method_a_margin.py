"""Independent new A + original joint. Plan by default; never auto-run a matrix or B."""
import argparse
import os
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from scripts import run_cliplora_joint_aggregation as single
from scripts.run_ab_validation import file_lock, write_json
from tools.client_aggregation.protocol import (SCHEMA, aggregation_spec, check_aggregation, code_hashes,
    count_matrix, digest, load_json, method_a_margin_contract, read_csv, write_weight_tables)
from tools.client_aggregation.method_a_margin import status, summarize
from utils.method_a_margin_math import ARM, DEFAULTS, algorithm_config


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    p.add_argument('--stage', choices=('plan', 'preflight', 'smoke', 'run', 'status', 'summary', 'pack'), default='plan')
    p.add_argument('--seed', type=int, choices=(42,), default=42)
    p.add_argument('--client-concurrency', type=int, choices=(1, 4, 6, 8), default=8)
    p.add_argument('--num-workers', type=int, default=8)
    p.add_argument('--reference-run', type=Path, default=Path('references/full10_clientlt'))
    p.add_argument('--data-root', type=Path, default=Path('DATA'))
    p.add_argument('--output-root', type=Path, default=Path('output/cifar100_LT/method_a_margin_v1_joint'))
    p.add_argument('--baseline-root', type=Path, default=Path('output/cifar100_LT/plain_a_aggregation_v1_parallel8/joint'))
    p.add_argument('--stop-after-round', type=int, default=0)
    for key, value in DEFAULTS.items():
        p.add_argument('--'+key.replace('_', '-'), type=float, default=value)
    args = p.parse_args(argv)
    if args.num_workers < 0 or not 0 <= args.stop_after_round <= 100:
        p.error('Workers must be nonnegative and stopping round in 0..100')
    try:
        algorithm_config(**parameters(args))
    except ValueError as error:
        p.error(str(error))
    return args


def parameters(args):
    return {k: getattr(args, k) for k in DEFAULTS}


def register(args):
    root = args.output_root.resolve()
    settings = dict(seed=42, reference_run=str(args.reference_run.resolve()), data_root=str(args.data_root.resolve()),
        num_workers=args.num_workers, client_concurrency=args.client_concurrency, parallel_validation='off',
        aggregation_rule='joint', lambda_value=.1, margin_parameters=parameters(args))
    contract = method_a_margin_contract(settings['margin_parameters'])
    hashes = code_hashes(REPO)
    fingerprint, counts, matrix = count_matrix(args.reference_run)
    with file_lock(root/'locks/registration.lock'):
        path = root/'experiment_plan.json'
        if path.is_file():
            plan = load_json(path)
            if (plan['contract'] != contract or plan['settings'] != settings or plan['code_sha256'] != hashes
                    or plan['aggregation']['input_fingerprint'] != fingerprint
                    or plan['aggregation']['counts'] != counts
                    or plan['aggregation']['matrix_sha256'] != digest(matrix.tolist())):
                raise ValueError('New-A source/parameters/inputs changed; preserve the old run and use a new output root')
            check_aggregation(plan['aggregation'])
        else:
            spec = aggregation_spec(args.reference_run, .1, 'joint')
            plan = dict(schema=SCHEMA, contract=contract, settings=settings, code_sha256=hashes, aggregation=spec)
            write_json(path, plan)
            write_weight_tables(root, spec)
    return plan


def verify_smoke(root, plan):
    from tools.client_aggregation.analysis import audit_run
    job = single.make_job(plan, ARM, 'smoke')
    run = Path(root)/job['run']
    if not (run/'smoke_complete.json').is_file():
        raise ValueError('Run the independent new-A six-round check first: --stage smoke')
    result = audit_run(run, job, 6)
    rows = read_csv(run/'sfra_rounds.csv')
    report = dict(status='passed', completed_round=6, round5_registration=rows[4], round6_use=rows[5],
        progress=result['progress'], feedback_costs=read_csv(run/'sfra_costs.csv'),
        note='Execution/history check only; no accuracy gate or B events')
    write_json(Path(root)/'six_round_smoke.json', report)
    return report


def main(argv=None):
    args = parse_args(argv)
    os.chdir(REPO)
    root = args.output_root.resolve()
    if args.stage == 'status':
        print(status(root))
        smoke = root/'smoke/seed42'/ARM/'progress.json'
        if smoke.is_file():
            print('smoke:', load_json(smoke)['completed_round'], '/6')
        return
    if args.stage in ('summary', 'pack'):
        row = summarize(root, args.baseline_root)
        if row['status'] == 'invalid':
            raise ValueError('Invalid new-A result; see analysis/status.csv')
        if args.stage == 'pack':
            from tools.client_aggregation.analysis import pack_results
            print('Archive:', pack_results(root))
        return
    plan = register(args)
    print('Registered independent new A + joint:', parameters(args), root, flush=True)
    if args.stage == 'plan':
        return
    single.preflight(plan, check_cuda=args.stage in ('smoke', 'run'))
    if args.stage == 'preflight':
        print('Inputs/configuration passed. A GPU smoke has not been executed.')
        return
    if args.stage == 'smoke':
        single.execute(single.make_job(plan, ARM, 'smoke'), root)
        print('SIX-ROUND SMOKE:', verify_smoke(root, plan)['status'])
        return
    verify_smoke(root, plan)
    single.execute(single.make_job(plan, ARM), root, args.stop_after_round)


if __name__ == '__main__':
    try:
        main()
    except (ValueError, OSError, KeyError) as error:
        raise SystemExit(str(error))
