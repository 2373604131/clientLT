"""Run one frozen shared-C guard pilot, resume it, or collect/pack results."""
import argparse
import os
from pathlib import Path
import shlex
import subprocess
import sys
from types import SimpleNamespace

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO))

from scripts.run_ab_validation import file_lock, input_fingerprint, write_json
from scripts.run_cliplora_a_refresh import build_command
from scripts.run_cliplora_sfra import prepare_protocol
from tools.sfra.b_class_guard import SCHEMA, source_hashes, check_sources, summarize, validate_profile
from tools.sfra.maintext import digest, load_json


def parser():
    p=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    p.add_argument('--stage',choices=['plan','preflight','train','status','summary','pack'],default='train')
    p.add_argument('--guard',choices=['client','class','off'],default='client')
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--protocol-seed',type=int,default=42)
    p.add_argument('--reference-run',type=Path,default=Path('references/full10_clientlt'))
    p.add_argument('--data-root',type=Path,default=Path('DATA'))
    p.add_argument('--output-root',type=Path,default=Path('output/cifar100_LT/sfra_b_class_guard'))
    p.add_argument('--compare-root',type=Path,help='Optional completed frozen A/AB result root; never used in training')
    p.add_argument('--num-workers',type=int,default=8)
    p.add_argument('--fast-execution-v2',action='store_true',default=True,help='Frozen on for this pilot')
    p.add_argument('--feedback-batch-size',type=int,default=128)
    p.add_argument('--feedback-cache-gib',type=float,default=4.)
    p.add_argument('--stop-after-round',type=int,default=None)
    p.add_argument('--resume',action='store_true')
    return p


def validate_args(args):
    if args.seed<0 or args.protocol_seed<0 or args.num_workers<0:
        raise ValueError('Seed/protocol/workers must be nonnegative')
    if args.feedback_batch_size!=128 or args.feedback_cache_gib!=4.:
        raise ValueError('This paired pilot freezes fast-v2 f128 c4')
    if args.stop_after_round is not None and not 0<=args.stop_after_round<=100:
        raise ValueError('stop-after-round must be in 0..100')


def run_directory(args):
    return args.output_root.resolve()/f'seed{args.seed}'/'client-longtail'/'full-cp'/(
        f'lambda10_mu1_protocol{args.protocol_seed}_bshared_ntclass-cyclic_tw0.35_guard_{args.guard}_fast_v2_f128_c4')


def make_spec(args):
    settings=dict(guard=args.guard,seed=args.seed,protocol_seed=args.protocol_seed,partition='client-longtail',
                  data_root=str(args.data_root.resolve()),reference_run=str(args.reference_run.resolve()),
                  num_workers=args.num_workers,feedback_batch_size=128,feedback_cache_gib=4.)
    inputs=input_fingerprint(SimpleNamespace(reference_run=args.reference_run,partition='client-longtail'))
    return dict(schema_version=SCHEMA,settings=settings,input_fingerprint=inputs,code_sha256=source_hashes(REPO))


def preflight(args,spec):
    check_sources(spec,REPO)
    if make_spec(args)!=spec:
        raise ValueError('Saved guard contract/reference differs; do not change an experiment on resume')
    dataset=args.data_root/'cifar-100/cifar-100-python'
    if any(not (dataset/n).is_file() or (dataset/n).stat().st_size==0 for n in ('train','test','meta')):
        raise ValueError('Missing CIFAR-100 train/test/meta under '+str(dataset))


def command_for(args,run,schedule,manifest):
    base=SimpleNamespace(output_root=args.output_root,data_root=args.data_root.resolve(),seed=args.seed,
        schedule_file=schedule,partition='client-longtail',matched_beta=.5,rank=4,refresh_interval=10,
        refresh_epochs=1,refresh_lr=.001,resume=None,num_workers=args.num_workers)
    cmd,_=build_command(base,'off')
    cmd[2]=str(REPO/'scripts/train_cliplora_b_class_guard.py')
    cmd[cmd.index('--output-dir')+1]=str(run)
    cmd[cmd.index('--split_seed')+1]=str(args.protocol_seed)
    index=cmd.index('DATALOADER.NUM_WORKERS')
    cmd[index:index]=['--guard-spec',str(run/'guard_spec.json'),
        '--lac_method','s','--lac_partition_manifest',str(manifest),'--lac_la_tau','1','--lac_a_lr_mult','1',
        '--sfra_variant','full-cp','--sfra_retention_weight','10.0','--sfra_classification_weight','1.0',
        '--sfra_witness_batch_size','8','--sfra_resume','',
        '--sfra_fast_execution_v2','--sfra_feedback_batch_size','128','--sfra_feedback_cache_gib','4.0',
        '--b_transfer_enable','--b_transfer_mode','shared','--b_transfer_lr','0.3',
        '--b_transfer_probe_step','0.1','--b_transfer_reg','0.001',
        '--b_transfer_non_tail_sampling','class-cyclic','--b_transfer_tail_weight','0.35']
    return cmd


def set_stop(cmd,value):
    cmd=list(cmd)
    if value is not None:
        flag='--sfra_stop_after_round'
        if flag in cmd:
            cmd[cmd.index(flag)+1]=str(value)
        else:
            i=cmd.index('DATALOADER.NUM_WORKERS');cmd[i:i]=[flag,str(value)]
    return cmd


def execute(args):
    run=run_directory(args)
    spec=make_spec(args);preflight(args,spec)
    with file_lock(args.output_root.resolve()/'locks'/(digest(str(run))+'.lock'),timeout=.1):
        saved=run/'guard_spec.json'
        if run.exists() and any(run.iterdir()):
            if not saved.is_file() or load_json(saved)!=spec:
                raise ValueError('Nonempty output has a different guard contract; use a new output root')
            if not args.resume:
                raise ValueError('Existing run: add --resume or use a new output root')
            receipt_path=run/'guard_receipt.json'
            if receipt_path.is_file():
                receipt=load_json(receipt_path)
                if receipt.get('spec_digest')!=digest(spec) or not receipt.get('code_unchanged'):
                    raise ValueError('Invalid training receipt; do not resume a run whose sources changed during training')
            if (run/'completion.json').is_file():
                validate_profile(load_json(run/'sfra_config.json'),spec)
                if load_json(run/'completion.json').get('completed_round')!=100:
                    raise ValueError('Invalid completion marker')
                write_json(run/'guard_receipt.json',dict(spec_digest=digest(spec),code_unchanged=True))
                print('SKIP completed:',run,flush=True);return
            checkpoint=run/'checkpoints/sfra_last.pt'
            if not checkpoint.is_file():
                raise ValueError('No round-boundary checkpoint; preserve partial output and choose a new output root')
            validate_profile(load_json(run/'sfra_config.json'),spec)
            cmd=load_json(run/'command.json')
            expected=command_for(args,run,run/'protocol/full_schedule.json',run/'protocol/partition_source.csv')
            # Saved command may only differ in an explicit stop boundary.
            comparable=list(cmd)
            comparable[0]=sys.executable
            if '--sfra_stop_after_round' in comparable:
                i=comparable.index('--sfra_stop_after_round');del comparable[i:i+2]
            if comparable!=expected:
                raise ValueError('Saved worker command changed')
            cmd[0]=sys.executable
            cmd[cmd.index('--sfra_resume')+1]=str(checkpoint)
        else:
            if args.resume:
                raise ValueError('Cannot resume a run that does not exist')
            # All required input checks happened before creating this directory.
            protocol_args=SimpleNamespace(reference_run=args.reference_run,partition='client-longtail')
            schedule,manifest=prepare_protocol(protocol_args,run)
            write_json(saved,spec)
            cmd=command_for(args,run,schedule,manifest)
            write_json(run/'command.json',set_stop(cmd,args.stop_after_round))
        cmd=set_stop(cmd,args.stop_after_round)
        print(shlex.join(cmd),flush=True)
        try:
            subprocess.run(cmd,cwd=REPO,check=True)
        finally:
            unchanged=source_hashes(REPO)==spec['code_sha256']
            write_json(run/'guard_receipt.json',dict(spec_digest=digest(spec),code_unchanged=unchanged))
        if not unchanged:
            raise ValueError('Sources changed during training; do not use this result as a matched run')
        if not (run/'progress.json').is_file():
            raise ValueError('Worker returned without a saved round-boundary progress record')
        completed=load_json(run/'progress.json').get('completed_round',-1)
        requested=args.stop_after_round
        if requested is None:
            requested=int(cmd[cmd.index('--sfra_stop_after_round')+1]) if '--sfra_stop_after_round' in cmd else 0
        if requested==0 and not (run/'completion.json').is_file():
            raise ValueError(f'Worker stopped at round {completed}; no completion marker, resume explicitly')
        print('Run finished; progress:',run/'progress.json',flush=True)


def main(argv=None):
    p=parser();args=p.parse_args(argv);os.chdir(REPO)
    args.output_root=args.output_root.resolve()
    try:
        if args.stage in ('summary','pack'):
            status,audits=summarize(args.output_root,args.compare_root)
            if args.stage=='pack':
                from tools.sfra.summary import pack
                pack(args.output_root)
            print('Report:',args.output_root/'analysis/report.md')
            if any(x['status']=='invalid' for x in status) or any(not x['valid'] for x in audits):
                raise ValueError('Invalid runs/pairs excluded; inspect status.csv and pair_audit.csv')
            return
        if args.stage=='status':
            for path in sorted(args.output_root.glob('seed*/*/*/*/guard_spec.json')):
                s=load_json(path)['settings'];progress=path.parent/'progress.json'
                state=load_json(progress) if progress.is_file() else {}
                print(f'seed{s["seed"]} {s["guard"]}: round {state.get("completed_round",0)}/100; '
                      f'B events {state.get("b_transfer_events",0)}/8; {path.parent}')
            return
        validate_args(args)
        if args.stage in ('plan','preflight'):
            spec=make_spec(args)
            if args.stage=='preflight':preflight(args,spec)
            run=run_directory(args)
            print('Run:',run)
            print(shlex.join(set_stop(command_for(args,run,run/'protocol/full_schedule.json',run/'protocol/partition_source.csv'),args.stop_after_round)))
            if args.stage=='preflight':print('File/configuration preflight passed; GPU execution is not tested.')
        else:
            execute(args)
    except (OSError,ValueError,KeyError) as error:
        p.error(str(error))


if __name__=='__main__':
    main()
