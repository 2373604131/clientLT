"""Exactly one joint_floor16 + ordinary-A experiment; seed42, 8 concurrent clients.

--stage run audits a separate one-round smoke, then resumes/executes the sole
100-round trajectory. Baseline results are needed only for the final comparison.
CUDA_VISIBLE_DEVICES selects the GPU. No gamma/lambda/method search is exposed.
"""
import argparse
import os
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO))
from scripts import run_cliplora_joint_aggregation as single
from scripts.run_ab_validation import file_lock, write_json
from tools.client_aggregation.floor16 import (RULE, REFERENCE, EXPERIMENT,
    status, audit_smoke, summarize)
from tools.client_aggregation.protocol import (SCHEMA, PLAIN_A_CONTRACT, code_hashes,
    load_json, digest, count_matrix, check_aggregation, write_weight_tables, partition_signature, read_csv)


def parse_args(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage',choices=('plan','preflight','smoke','run','status','summary','pack'),default='plan')
    p.add_argument('--seed',type=int,choices=(42,),default=42)
    p.add_argument('--client-concurrency',type=int,choices=(8,),default=8)
    p.add_argument('--reference-run',type=Path,default=Path('references/full10_clientlt'))
    p.add_argument('--data-root',type=Path,default=Path('DATA'))
    p.add_argument('--baseline-root',type=Path,default=Path('output/cifar100_LT/plain_a_aggregation_v1_parallel8/joint'),
                   help='Optional original joint + ordinary-A results, read only during summary/pack; not required for training')
    p.add_argument('--output-root',type=Path,default=Path('output/cifar100_LT/joint_floor16_plain_a_v1_parallel8'))
    p.add_argument('--stop-after-round',type=int,default=0,help='Pause the single formal run at a committed round; 0 resumes to 100')
    args=p.parse_args(argv)
    if not 0 <= args.stop_after_round <= 100:
        p.error('Invalid stopping round')
    return args


def register(args):
    root=args.output_root.resolve()
    baseline=args.baseline_root.resolve()
    if root==baseline or root in baseline.parents or baseline in root.parents:
        raise ValueError('Use an independent output root outside the baseline experiment')
    spec=load_json(REPO/REFERENCE)
    check_aggregation(spec)
    fingerprint,counts,matrix=count_matrix(args.reference_run)
    csv_files={'partition_manifest.csv','protocol/probe_manifest.csv'}
    if (any(spec['input_fingerprint'].get(k)!=v for k,v in fingerprint.items() if k not in csv_files)
            or spec['counts']!=counts or spec['matrix_sha256']!=digest(matrix.tolist())
            or spec['partition_sha256']!=partition_signature(read_csv(args.reference_run/'partition_manifest.csv'))):
        raise ValueError('Offline floor16 weights belong to a different training partition')
    # Windows/Linux newline conversion must not masquerade as a data change.
    # Validate canonical sample identity and the existing probe replay protocol,
    # then register this deployment's raw file fingerprints. Weights stay exact.
    from tools.sfra.simple_controls import probe_manifest_replay
    probe_manifest_replay(args.reference_run/'protocol/probe_manifest.csv',
        load_json(args.reference_run/'bridge_metadata.json')['probe_manifest_sha256'])
    spec=dict(spec,input_fingerprint=fingerprint)
    hashes=code_hashes(REPO)
    settings=dict(seed=42,reference_run=str(args.reference_run.resolve()),data_root=str(args.data_root.resolve()),
        num_workers=8,client_concurrency=8,parallel_validation='off',aggregation_rule=RULE,lambda_value=.1)
    with file_lock(root/'locks/plan.lock'):
        path=root/'experiment_plan.json'
        wanted=dict(schema=SCHEMA,contract=PLAIN_A_CONTRACT,settings=settings,aggregation=spec,
                    code_sha256=hashes,floor16_experiment=EXPERIMENT)
        if path.is_file():
            plan=load_json(path)
            if any(plan.get(k)!=v for k,v in wanted.items()):
                raise ValueError('Registered source/settings/weights/gate changed; do not overwrite or retune this run')
        else:
            plan=wanted
            write_json(path,plan)
            write_weight_tables(root,spec)
        write_json(root/'preflight_audit.json',dict(weights_sha256=spec['weights_sha256'],
            weight_source=REFERENCE,lower_bounds_sha256=spec['lower_bounds_sha256'],
            baseline_comparison='deferred_to_summary',
            source_postprocessing='none',solver_audit=check_aggregation(spec),
            note='CPU input/weight audit, not a GPU training result'))
    return plan


def main(argv=None):
    args=parse_args(argv); os.chdir(REPO);root=args.output_root.resolve()
    if args.stage=='status':
        s=status(root)
        print(f'{RULE}: {s["status"]}, round={s["completed_round"]}/100, B_steps={s["B_steps"]}, A_steps={s["A_steps"]}')
        return
    if args.stage in ('summary','pack'):
        summarize(root,args.baseline_root)
        if args.stage=='pack':
            from tools.client_aggregation.analysis import pack_results
            print('Archive:',pack_results(root))
        return
    plan=register(args)
    print('Registered ONE joint_floor16 + ordinary-A run; no Full-CP or C; weights='+plan['aggregation']['weights_sha256'],flush=True)
    if args.stage=='plan':return
    single.preflight(plan,check_cuda=args.stage in ('smoke','run'))
    print('Input and lower-bound audit passed. GPU execution is verified by the separate one-round smoke.',flush=True)
    if args.stage=='preflight':return
    with file_lock(root/'locks/sole_run.lock',timeout=.1):
        single.execute(single.make_job(plan,'plain_a','smoke'),root)
        audit_smoke(root,plan)
        print('SMOKE PASSED: exact same offline weights in B/A; ordinary-A path; baseline comparison deferred to summary.',flush=True)
        if args.stage=='run':
            single.execute(single.make_job(plan,'plain_a'),root,args.stop_after_round)
            if (root/'runs/seed42/plain_a/completion.json').is_file():
                summarize(root,args.baseline_root)
                from tools.client_aggregation.analysis import pack_results
                print('Archive:',pack_results(root))


if __name__=='__main__':
    try:main()
    except (ValueError,OSError,KeyError) as error:raise SystemExit(str(error))
