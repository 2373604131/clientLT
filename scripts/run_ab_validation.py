"""Plan/preflight/run the frozen A+shared-B validation; default never trains."""
import argparse
from contextlib import contextmanager
import json
import math
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time
import uuid

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO))
from scripts.run_cliplora_sfra import run_directory
from tools.sfra.ab_validation import (ARMS,CONTRACT,PLAN_NAME,SCHEMA,check_receipt,code_hashes,
    digest,load_json,safe_run,summarize,validate_config)
from tools.sfra.calibration_reference import build_reference


def write_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp')
    temp.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')
    temp.replace(path)


@contextmanager
def file_lock(path,timeout=10):
    """OS locks release automatically after a killed launcher; no stale PID cleanup."""
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a+b') as stream:
        if path.stat().st_size==0:
            stream.write(b'0');stream.flush()
        deadline=time.monotonic()+timeout
        while True:
            stream.seek(0)
            try:
                if os.name=='nt':
                    import msvcrt
                    msvcrt.locking(stream.fileno(),msvcrt.LK_NBLCK,1)
                else:
                    import fcntl
                    fcntl.flock(stream.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
                break
            except OSError:
                if time.monotonic()>=deadline:
                    raise ValueError(f'Another launcher is using {path}; do not launch the same run twice')
                time.sleep(.05)
        try:
            yield
        finally:
            stream.seek(0)
            if os.name=='nt':
                msvcrt.locking(stream.fileno(),msvcrt.LK_UNLCK,1)
            else:
                fcntl.flock(stream.fileno(),fcntl.LOCK_UN)


def launcher_args(args,arm,seed):
    transfer=arm in ('ab','a-calibration')
    return argparse.Namespace(output_root=args.output_root/'runs'/arm,seed=seed,partition=args.partition,
        protocol_seed=args.protocol_seed,dirichlet_beta=args.dirichlet_beta,method=ARMS[arm],
        retention_weight=10.,classification_weight=1.,b_transfer=transfer,transfer_mode='shared' if transfer else 'local',
        transfer_lr=.3,probe_step=.1,transfer_reg=.001,transfer_non_tail_sampling='class-cyclic' if transfer else 'sample',
        transfer_tail_weight=.35 if transfer else .5,calibration_reference='paired-ab' if arm=='a-calibration' else None,
        fast_execution_v2=True,feedback_batch_size=128,feedback_cache_gib=4.,b_aggregation='sample')


def input_fingerprint(args):
    if args.reference_run:
        root=args.reference_run.resolve()
        names=['partition_manifest.csv','bridge_metadata.json','protocol/full_schedule.json','protocol/eri_protocol.json','protocol/probe_manifest.csv']
        from tools.sfra.calibration_reference import file_hash
        for n in names:
            if not (root/n).is_file():
                raise ValueError(f'Missing reference protocol file: {root/n}')
        meta=load_json(root/'bridge_metadata.json')
        if meta['topology']!=args.partition:
            raise ValueError('Reference partition differs; never reuse Client-LT capacities for ordinary Dirichlet')
        schedule=load_json(root/'protocol/full_schedule.json')
        schedule=schedule['schedule'] if isinstance(schedule,dict) else schedule
        if len(schedule)!=100 or any(sorted(x)!=list(range(30)) for x in schedule):
            raise ValueError('Reference requires 100 rounds of full participation')
        return {n:file_hash(root/n) for n in names}
    return {'protocol':'fresh partition; fixed protocol seed; generated deterministic full-participation schedule'}


def plan_jobs(args):
    hashes=code_hashes(REPO);fingerprint=input_fingerprint(args);jobs=[]
    arms=list(args.arms)
    if 'a-calibration' in arms and 'ab' not in arms:
        arms.append('ab')  # Register dependency; not an implicit request to launch it.
    arms=sorted(arms,key=lambda x:list(ARMS).index(x))
    for seed in args.seeds:
        for arm in arms:
            run=run_directory(launcher_args(args,arm,seed)).relative_to(args.output_root.resolve()).as_posix()
            dependency=run_directory(launcher_args(args,'ab',seed)).relative_to(args.output_root.resolve()).as_posix() if arm=='a-calibration' else None
            jobs.append(dict(run=run,arm=arm,seed=seed,partition=args.partition,protocol_seed=args.protocol_seed,
                dirichlet_beta=args.dirichlet_beta,data_root=str(args.data_root.resolve()),num_workers=args.num_workers,
                reference_run=str(args.reference_run.resolve()) if args.reference_run else None,
                input_fingerprint=fingerprint,code_sha256=hashes,dependency=dependency))
    return jobs


def ensure_fresh_schedule(root,job):
    """Pin fresh runs to a declared schedule instead of mutable bridge fallbacks."""
    import numpy as np
    path=Path(root)/'protocols'/f'full_schedule_p{job["protocol_seed"]}.json'
    rng=np.random.default_rng(job['protocol_seed'])
    payload=dict(schedule=[rng.choice(30,30,replace=False).tolist() for _ in range(100)])
    with file_lock(Path(root)/'locks/protocol.lock'):
        if path.exists():
            if load_json(path)!=payload:
                raise ValueError('Frozen fresh-protocol schedule changed')
        else:
            write_json(path,payload)
    return path


def merge_plan(root,jobs):
    root=Path(root);path=root/PLAN_NAME
    with file_lock(root/'locks/plan.lock'):
        plan=load_json(path) if path.exists() else dict(schema_version=SCHEMA,contract=CONTRACT,jobs=[])
        if plan.get('schema_version')!=SCHEMA or plan.get('contract')!=CONTRACT:
            raise ValueError('Different validation contract; use a new output root')
        saved={x['run']:x for x in plan['jobs']}
        for job in jobs:
            safe_run(root,job['run'])
            if job['run'] in saved and saved[job['run']]!=job:
                raise ValueError('Frozen job configuration/source changed: '+job['run'])
            saved[job['run']]=job
        plan['jobs']=list(saved.values());write_json(path,plan)


def command_for(job,root):
    command=[sys.executable,'-u',str(REPO/'scripts/run_cliplora_sfra.py'),'--method',ARMS[job['arm']],
        '--retention-weight','10','--classification-weight','1','--seed',str(job['seed']),
        '--protocol-seed',str(job['protocol_seed']),'--partition',job['partition'],
        '--dirichlet-beta',str(job['dirichlet_beta']),'--data-root',job['data_root'],
        '--output-root',str(Path(root).resolve()/'runs'/job['arm']),'--num-workers',str(job['num_workers']),
        '--witness-batch-size','8','--fast-execution-v2','--feedback-batch-size','128','--feedback-cache-gib','4']
    if job['reference_run']:
        command+=['--reference-run',job['reference_run']]
    else:
        command+=['--fresh-protocol','--schedule-file',str(Path(root).resolve()/'protocols'/f'full_schedule_p{job["protocol_seed"]}.json')]
    if job['arm'] in ('ab','a-calibration'):
        command+=['--b-transfer','--transfer-mode','shared','--transfer-lr','0.3','--probe-step','0.1',
            '--transfer-reg','0.001','--transfer-non-tail-sampling','class-cyclic','--transfer-tail-weight','0.35']
    if job['dependency']:
        command+=['--calibration-reference',str(safe_run(root,job['dependency']))]
    return command


def registration(root,job):
    return Path(root)/'launches'/(digest(job['run'])+'.json')


def verify_receipt(root,run,job,resume):
    try:
        check_receipt(run,job)
    except FileNotFoundError:
        if not resume:
            raise ValueError('Missing run receipt; use --resume after an interrupted launcher')
        if load_json(registration(root,job))!=dict(schema_version=SCHEMA,job=job):
            raise ValueError('Cannot recover an unregistered run')
        write_json(run/'ab_validation_run.json',dict(schema_version=SCHEMA,job=job,code_unchanged=True))


def preflight_job(job,root,require_dependency=False):
    if code_hashes(REPO)!=job['code_sha256']:
        raise ValueError('Training code changed since plan; preserve old output and use a new root')
    if job['reference_run']:
        args=argparse.Namespace(reference_run=Path(job['reference_run']),partition=job['partition'])
        if input_fingerprint(args)!=job['input_fingerprint']:
            raise ValueError('Reference protocol files changed since plan')
    if not Path(job['data_root']).is_dir():
        raise ValueError('Missing data root: '+job['data_root'])
    dataset=Path(job['data_root'])/'cifar-100/cifar-100-python'
    if any(not (dataset/name).is_file() or (dataset/name).stat().st_size==0 for name in ('train','test','meta')):
        raise ValueError('Missing/empty CIFAR-100 train/test/meta under '+str(dataset))
    if not job['reference_run']:
        ensure_fresh_schedule(root,job)
    if job['dependency']:
        dependency=safe_run(root,job['dependency'])
        if (dependency/'completion.json').is_file():
            plan=load_json(Path(root)/PLAN_NAME)
            paired=next(x for x in plan['jobs'] if x['run']==job['dependency'])
            validate_config(dependency,paired);check_receipt(dependency,paired)
            build_reference(dependency)
        elif require_dependency:
            raise ValueError('Run paired AB first: '+job['dependency'])
    return 'waiting for paired AB' if job['dependency'] and not (safe_run(root,job['dependency'])/'completion.json').is_file() else 'ready'


def execute_job(job,root,resume=False):
    root=Path(root)
    with file_lock(root/'locks'/(digest(job['run'])+'.lock'),timeout=.1):
        preflight_job(job,root,require_dependency=True)
        run=safe_run(root,job['run']);command=command_for(job,root)
        if (run/'completion.json').is_file():
            validate_config(run,job);verify_receipt(root,run,job,resume)
            if load_json(run/'completion.json').get('completed_round')!=100:
                raise ValueError('Invalid completion marker')
            print('SKIP completed:',run,flush=True);return
        if run.exists() and any(run.iterdir()):
            if not resume:
                raise ValueError('Partial run exists; use --resume: '+str(run))
            if not (run/'checkpoints/sfra_last.pt').is_file():
                raise ValueError('No round-boundary checkpoint; preserve partial output and use a new root')
            validate_config(run,job);verify_receipt(root,run,job,resume);command+=['--resume']
        print(shlex.join(command),flush=True)
        write_json(registration(root,job),dict(schema_version=SCHEMA,job=job))
        try:
            subprocess.run(command,cwd=REPO,check=True)
        finally:
            unchanged=code_hashes(REPO)==job['code_sha256']
            if run.is_dir():
                write_json(run/'ab_validation_run.json',dict(schema_version=SCHEMA,job=job,code_unchanged=unchanged))
        if not unchanged:
            raise ValueError('Code changed during training; results cannot enter frozen paired tables')
        if not (run/'completion.json').is_file():
            raise ValueError('Training returned before completion; resume explicitly')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',choices=['plan','preflight','train','summary','pack'],default='plan')
    parser.add_argument('--arms',nargs='+',choices=list(ARMS),default=['a','ab'])
    parser.add_argument('--seeds',nargs='+',type=int,default=[42,43,44])
    parser.add_argument('--partition',choices=['client-longtail','noniid-labeldir-fine'],default='client-longtail')
    parser.add_argument('--protocol-seed',type=int,default=42)
    parser.add_argument('--dirichlet-beta',type=float,default=.5)
    parser.add_argument('--reference-run',type=Path)
    parser.add_argument('--data-root',type=Path,default=Path('DATA'))
    parser.add_argument('--output-root',type=Path,default=Path('output/cifar100_LT/ab_validation'))
    parser.add_argument('--num-workers',type=int,default=8)
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args();os.chdir(REPO);args.output_root=args.output_root.resolve()
    try:
        if args.stage in ('summary','pack'):
            status,audits=summarize(args.output_root)
            if args.stage=='pack':
                from tools.sfra.summary import pack
                pack(args.output_root)
            print('Report:',args.output_root/'analysis/report.md')
            if any(x['status']=='invalid' for x in status) or any(x['reason'] not in ('','pending') for x in audits):
                raise ValueError('Invalid/mismatched results excluded; see analysis/status.csv and pair_audit.csv')
            return
        if len(set(args.seeds))!=len(args.seeds) or any(x<0 for x in args.seeds) or args.protocol_seed<0:
            raise ValueError('Seeds must be distinct nonnegative integers')
        if len(set(args.arms))!=len(args.arms) or args.num_workers<0 or not math.isfinite(args.dirichlet_beta) or args.dirichlet_beta<=0:
            raise ValueError('Invalid arms, workers or beta')
        jobs=plan_jobs(args);merge_plan(args.output_root,jobs)
        for job in jobs:
            if not job['reference_run']:
                ensure_fresh_schedule(args.output_root,job)
        selected=[x for x in jobs if x['arm'] in args.arms]
        print('Frozen AB; selected runs:',len(selected),'; plan:',args.output_root/PLAN_NAME,flush=True)
        for job in selected:
            if args.stage=='plan':
                print(shlex.join(command_for(job,args.output_root)))
            elif args.stage=='preflight':
                print(job['run'],preflight_job(job,args.output_root))
            else:
                execute_job(job,args.output_root,args.resume)
        # Summarize explicitly after parallel GPU jobs finish, avoiding racing snapshots.
    except (OSError,ValueError,KeyError,StopIteration) as error:
        parser.error(str(error))


if __name__=='__main__':
    main()
