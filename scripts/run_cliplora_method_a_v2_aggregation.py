"""Independent source-memory-v2 runs. One six-round joint smoke; no automatic matrix."""
import argparse
import os
from pathlib import Path
import sys

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO))
from scripts import run_cliplora_joint_aggregation as single
from scripts.run_ab_validation import file_lock, write_json
from tools.client_aggregation.method_a_v2 import METHODS, ARM, status_rows, summarize
from tools.client_aggregation.protocol import (SCHEMA, METHOD_A_V2_CONTRACT, aggregation_spec,
    code_hashes, count_matrix, digest, load_json, check_aggregation, write_weight_tables,
    partition_signature, read_csv)


def parse_args(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage',choices=('plan','preflight','smoke','run','status','summary','pack'),default='plan')
    p.add_argument('--methods',nargs='+',choices=METHODS,default=['joint'])
    p.add_argument('--seed',type=int,choices=(42,),default=42)
    p.add_argument('--client-concurrency',type=int,choices=(8,),default=8)
    p.add_argument('--reference-run',type=Path,default=Path('references/full10_clientlt'))
    p.add_argument('--data-root',type=Path,default=Path('DATA'))
    p.add_argument('--output-root',type=Path,default=Path('output/cifar100_LT/method_a_v2_aggregation_v1_parallel8'))
    p.add_argument('--plain-root',type=Path,default=Path('output/cifar100_LT/plain_a_aggregation_v1_parallel8'))
    p.add_argument('--floor-plain-root',type=Path,default=Path('output/cifar100_LT/joint_floor16_plain_a_v1_parallel8'))
    p.add_argument('--stop-after-round',type=int,default=0)
    args=p.parse_args(argv)
    if len(set(args.methods))!=len(args.methods) or not 0<=args.stop_after_round<=100:
        p.error('Use distinct methods and stopping round0..100')
    if args.stage=='smoke' and args.methods!=['joint']:
        p.error('The only smoke is joint+Av2, six rounds; use --methods joint')
    return args


def floor_spec(reference):
    from tools.sfra.simple_controls import probe_manifest_replay
    spec=load_json(REPO/'references/joint_floor16_seed42/aggregation_spec.json')
    fingerprint,counts,matrix=count_matrix(reference)
    csvs={'partition_manifest.csv','protocol/probe_manifest.csv'}
    if (counts!=spec['counts'] or digest(matrix.tolist())!=spec['matrix_sha256']
            or any(spec['input_fingerprint'].get(k)!=v for k,v in fingerprint.items() if k not in csvs)
            or partition_signature(read_csv(reference/'partition_manifest.csv'))!=spec['partition_sha256']):
        raise ValueError('Frozen floor16 solution belongs to different training inputs')
    probe_manifest_replay(reference/'protocol/probe_manifest.csv',
        load_json(reference/'bridge_metadata.json')['probe_manifest_sha256'])
    spec=dict(spec,input_fingerprint=fingerprint)
    check_aggregation(spec)  # Recomputes gamma and lower bounds from actual counts.
    return spec


def register(args):
    root=args.output_root.resolve()
    hashes=code_hashes(REPO)
    fingerprint,counts,matrix=count_matrix(args.reference_run)
    plans={}
    with file_lock(root/'locks/registration.lock'):
        for rule in args.methods:
            folder=root/rule;path=folder/'experiment_plan.json'
            settings=dict(seed=42,reference_run=str(args.reference_run.resolve()),data_root=str(args.data_root.resolve()),
                num_workers=8,client_concurrency=8,parallel_validation='off',aggregation_rule=rule,
                lambda_value=None if rule=='tailrw16' else .1)
            if path.is_file():
                plan=load_json(path)
                if (plan['contract']!=METHOD_A_V2_CONTRACT or plan['settings']!=settings or plan['code_sha256']!=hashes
                        or plan['aggregation']['input_fingerprint']!=fingerprint
                        or plan['aggregation']['counts']!=counts or plan['aggregation']['matrix_sha256']!=digest(matrix.tolist())):
                    raise ValueError('Registered Av2 source, configuration or inputs changed; preserve this run')
                check_aggregation(plan['aggregation'])
            else:
                spec=floor_spec(args.reference_run) if rule=='joint_floor16' else aggregation_spec(args.reference_run,.1,rule)
                plan=dict(schema=SCHEMA,contract=METHOD_A_V2_CONTRACT,settings=settings,code_sha256=hashes,aggregation=spec)
                write_json(path,plan);write_weight_tables(folder,spec)
            plans[rule]=plan
    return plans


def verify_smoke(root, plan=None):
    from tools.client_aggregation.analysis import audit_run
    folder=Path(root)/'joint'
    path=folder/'smoke/seed42/method_a_v2/smoke_complete.json'
    if not path.is_file():
        raise ValueError('Run the single joint+Av2 six-round check first: --stage smoke --methods joint')
    smoke_plan=load_json(folder/'experiment_plan.json')
    if smoke_plan['code_sha256']!=code_hashes(REPO) or smoke_plan['contract']!=METHOD_A_V2_CONTRACT:
        raise ValueError('Joint smoke does not match the current Av2 source/contract')
    if plan is not None:
        for key in ('seed','reference_run','data_root','num_workers','client_concurrency','parallel_validation'):
            if smoke_plan['settings'][key]!=plan['settings'][key]:
                raise ValueError('Joint smoke used different '+key)
        if smoke_plan['aggregation']['input_fingerprint']!=plan['aggregation']['input_fingerprint']:
            raise ValueError('Joint smoke used different training inputs')
    job=single.make_job(smoke_plan,ARM,'smoke')
    result=audit_run(folder/job['run'],job,6)
    rows=read_csv(folder/job['run']/'sfra_rounds.csv')
    report=dict(status='passed',completed_round=6,aggregation='joint',variant='source-memory-v2',
        round5_registration=rows[4],round6_use=rows[5],cost=result['progress'],
        round_records=rows,feedback_costs=read_csv(folder/job['run']/'sfra_costs.csv'),
        candidates=[dict(round=r['round'],beta=r['selected_beta'],nonzero=r['nonzero_correction'],
            full_reject=r['full_reject'],half_reject=r['half_reject']) for r in rows],
        note='Execution/state check only; test accuracy is not a selection criterion')
    write_json(Path(root)/'six_round_smoke.json',report)
    return report


def main(argv=None):
    args=parse_args(argv);os.chdir(REPO);root=args.output_root.resolve()
    if args.stage=='status':
        for row in status_rows(root):print(row)
        smoke=root/'joint/smoke/seed42'/ARM/'progress.json'
        if smoke.is_file():print('joint smoke:',load_json(smoke)['completed_round'],'/6')
        return
    if args.stage in ('summary','pack'):
        rows=summarize(root,args.plain_root,args.floor_plain_root)
        if any(r['status']=='invalid' for r in rows):raise ValueError('Invalid results; see analysis/status.csv')
        if args.stage=='pack':
            from tools.client_aggregation.analysis import pack_results
            print('Archive:',pack_results(root))
        return
    plans=register(args)
    if args.stage=='plan':return
    for rule,plan in plans.items():
        single.preflight(plan,check_cuda=args.stage in ('smoke','run'))
    if args.stage=='preflight':
        print('Av2 inputs/configuration passed; GPU smoke has not been executed.')
        return
    if args.stage=='smoke':
        single.execute(single.make_job(plans['joint'],ARM,'smoke'),root/'joint')
        print('SIX-ROUND SMOKE:',verify_smoke(root)['status'])
        return  # Never fall through into a 100-round experiment.
    for rule,plan in plans.items():
        verify_smoke(root,plan)
        single.execute(single.make_job(plan,ARM),root/rule,args.stop_after_round)


if __name__=='__main__':
    try:main()
    except (ValueError,OSError,KeyError) as error:raise SystemExit(str(error))
