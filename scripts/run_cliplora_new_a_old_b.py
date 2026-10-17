"""One fresh new-A + archived-B trajectory, paired to completed same-A references."""
import argparse
import json
import os
from pathlib import Path
from statistics import mean
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from scripts import run_cliplora_joint_aggregation as single
from scripts.run_ab_validation import file_lock, write_json
from tools.client_aggregation.protocol import code_hashes, load_json, read_csv, write_table, write_weight_tables
from tools.client_aggregation.new_a_old_b import (
    RUN, EVALUATION, INTEGRATION_FILES, candidate_plan, pair_check)

IDENTITIES = ('initial_lora_sha256', 'frozen_model_sha256', 'schedule_sha256',
              'pool_sha256', 'test_sha256', 'probe_images_sha256', 'probe_manifest_sha256',
              'client_sample_counts', 'classnames')


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    p.add_argument('--stage', choices=('plan', 'preflight', 'smoke', 'run', 'status', 'startup-audit', 'summary'), default='plan')
    p.add_argument('--reference-a', type=Path, required=True,
                   help='Completed new-A joint experiment; copy settings, never weights/history')
    p.add_argument('--compare-root', type=Path, action='append', default=[])
    p.add_argument('--output-root', type=Path, required=True)
    p.add_argument('--gpu', type=int, default=5)
    args = p.parse_args(argv)
    if args.gpu < 0:
        p.error('GPU must be nonnegative')
    return args


def load_reference(root):
    from tools.client_aggregation.analysis import audit_run
    root = Path(root)
    plan = load_json(root/'experiment_plan.json')
    job = single.make_job(plan, 'method_a_margin')
    if plan['settings'].get('factorial_cell') not in ('new_a_joint', 'new_a_new_b_joint'):
        raise ValueError('Reference must be a registered new-A factorial/search trajectory')
    return plan, audit_run(root/RUN, job)


def endpoint(result):
    from scripts.run_cliplora_ab_parameter_search import endpoint as fixed_endpoint
    return fixed_endpoint(result['curves'])


def register(args):
    root = args.output_root.resolve()
    paths = list(dict.fromkeys([args.reference_a.resolve(), *(p.resolve() for p in args.compare_root)]))
    if root in paths:
        raise ValueError('Output must be isolated from every reference')
    source, anchor = load_reference(paths[0])
    hashes = code_hashes(REPO)
    plan = candidate_plan(source, hashes)
    references = []
    for path in paths:
        ref_plan, result = (source, anchor) if path == paths[0] else load_reference(path)
        if (ref_plan['contract']['method_a_margin'] != plan['contract']['method_a_margin']
                or ref_plan['aggregation'] != plan['aggregation']):
            raise ValueError('Reference A parameters or aggregation differ: '+str(path))
        if result['config']['client_execution'] != anchor['config']['client_execution']:
            raise ValueError('Reference execution policy/concurrency differs')
        changed = sorted(k for k in set(hashes) | set(ref_plan['code_sha256'])
                         if hashes.get(k) != ref_plan['code_sha256'].get(k))
        if set(changed) - INTEGRATION_FILES:
            raise ValueError('Unexpected source changes: '+str(set(changed)-INTEGRATION_FILES))
        identity = {k: result['metadata'][k] for k in IDENTITIES}
        if any(identity[k] != anchor['metadata'][k] for k in IDENTITIES):
            raise ValueError('Reference initialization/data/schedule differs')
        references.append(dict(root=str(path), cell=ref_plan['settings']['factorial_cell'],
            scores=endpoint(result), identity=identity, source_differences=changed,
            extra_B=bool(result['config'].get('b_transfer')), completed_round=100))
    audit = dict(evaluation=EVALUATION, references=references,
        A=plan['contract']['method_a_margin'], B=plan['contract']['old_B'],
        protocol='same seed42 initialization and partition; independent fresh history; B3/A1; A frozen after90',
        endpoint='mean of committed rounds81..100; Overall, Head20, Middle60, Tail20',
        execution='one fresh 100-round run after disposable integration smoke; no parameter search or rollback')
    with file_lock(root/'locks/registration.lock'):
        for name, data in (('experiment_plan.json', plan), ('reference_audit.json', audit)):
            path = root/name
            if path.is_file() and load_json(path) != data:
                raise ValueError('Registered experiment changed; preserve output and use another root')
            write_json(path, data)
        write_weight_tables(root, plan['aggregation'])
    return plan


def startup_audit(root):
    import numpy as np
    from tools.client_aggregation.protocol import audit_runtime_config
    root = Path(root)
    plan = load_json(root/'experiment_plan.json')
    run = root/RUN
    cfg = audit_runtime_config(run, single.make_job(plan, 'method_a_margin'))
    meta = load_json(run/'bridge_metadata.json')
    command = load_json(run/'command.json')
    if command[command.index('--sfra_resume')+1]:
        raise ValueError('New comparison must start from initialization')
    checks = []
    for ref in load_json(root/'reference_audit.json')['references']:
        reference_run = Path(ref['root'])/RUN
        if any(meta[k] != ref['identity'][k] for k in IDENTITIES):
            raise ValueError('Startup data/initialization differs from reference')
        for name in ('private_witness_manifest.json', 'margin_witness_manifest.json'):
            if load_json(run/name) != load_json(reference_run/name):
                raise ValueError('Startup witness identity differs: '+name)
        with np.load(run/'predictions/r000.npz') as a, np.load(reference_run/'predictions/r000.npz') as b:
            for name in ('sample_id', 'class_id', 'prediction', 'correct'):
                if not np.array_equal(a[name], b[name]):
                    raise ValueError('Initial evaluation differs: '+name)
        checks.append(dict(reference=ref['root'], passed=True))
    value = dict(passed=True, references=checks, fresh_initialization=True, independent_history=True,
                 A=plan['contract']['method_a_margin'], B=cfg['b_transfer'])
    write_json(root/'startup_audit.json', value)
    print(json.dumps(value, indent=2), flush=True)
    return value


def summarize(root):
    from tools.client_aggregation.analysis import audit_run
    from scripts.run_cliplora_ab_parameter_search import actual_corrections
    root = Path(root)
    plan = load_json(root/'experiment_plan.json')
    result = audit_run(root/RUN, single.make_job(plan, 'method_a_margin'))
    startup_audit(root)
    out = root/'analysis'
    out.mkdir(parents=True, exist_ok=True)
    scores = endpoint(result)
    rows = [dict(method='new_a_old_b_joint', **scores)]
    pairs = []
    for ref in load_json(root/'reference_audit.json')['references']:
        _, other = load_reference(Path(ref['root']))
        pair = pair_check(result, other)
        pair.update(reference=ref['root'], cell=ref['cell'])
        pairs.append(pair)
        other_scores = endpoint(other)
        rows.append(dict(method=ref['cell'], **other_scores))
        if pair['status'] == 'mismatch':
            write_json(out/'pair_audit.json', pairs)
            raise ValueError('Completed result is not paired to reference: '+str(pair['mismatches']))
        pair['delta_pp'] = {k: scores[k]-other_scores[k] for k in scores}
    write_table(out/'performance.csv', rows)
    write_json(out/'pair_audit.json', pairs)
    corrections = actual_corrections(root/RUN, out/'actual_A_corrections.csv')
    a_rows = read_csv(root/RUN/'sfra_rounds.csv')
    b_rows = read_csv(root/RUN/'b_transfer_rounds.csv')
    write_table(out/'A_rounds.csv', a_rows)
    write_table(out/'B_rounds.csv', b_rows)
    report = dict(completed_and_audited=True, evaluation=EVALUATION, performance=rows, comparisons=pairs,
        A_corrections=corrections,
        B_effective_LoRA_transfer_norm_mean=mean(float(r['effective_global_transfer_norm']) for r in b_rows),
        A_repair_seconds_total=sum(float(r.get('repair_seconds') or 0) for r in a_rows),
        B_transfer_seconds_total=sum(float(r['seconds']) for r in b_rows), progress=result['progress'])
    write_json(out/'report.json', report)
    write_json(root/'finalization_receipt.json', dict(completed_and_audited=True, reporting_only=True))
    print(json.dumps(report, indent=2), flush=True)


def main(argv=None):
    args = parse_args(argv)
    root = args.output_root.resolve()
    os.chdir(REPO)
    if args.stage == 'startup-audit':
        startup_audit(root)
        return
    if args.stage == 'summary':
        summarize(root)
        return
    if args.stage == 'status':
        for mode in ('smoke', 'runs'):
            path = root/mode/'seed42/method_a_margin/progress.json'
            print(mode, load_json(path) if path.is_file() else 'not_started')
        return
    plan = register(args)
    if args.stage == 'plan':
        return
    os.environ.update(CUDA_VISIBLE_DEVICES=str(args.gpu), CUDA_DEVICE_ORDER='PCI_BUS_ID')
    single.preflight(plan, check_cuda=args.stage in ('smoke', 'run'))
    if args.stage == 'preflight':
        print('Reference/configuration/source audit passed; no training started.', flush=True)
        return
    single.execute(single.make_job(plan, 'method_a_margin', 'smoke'), root)
    if args.stage == 'run':
        single.execute(single.make_job(plan, 'method_a_margin'), root)
        summarize(root)


if __name__ == '__main__':
    main()
