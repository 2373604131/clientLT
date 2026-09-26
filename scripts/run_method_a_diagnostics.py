"""Analyze saved logs, check state files, and run the fixed 2 x 3 x 5 diagnostic."""
import argparse
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from tools.sfra.diagnostics import (SCHEMA, ANCHORS, HORIZON, BRANCHES, analyze, preflight,
                                    read_json, read_csv, write_json, write_csv, file_hash, class_groups)
from tools.sfra.maintext import TRAINING_FILES

DEFAULT_SOURCE = REPO/'output/sfra_cp_analysis/sfra_cp/seed42/client-longtail/full-cp/lambda10_mu1_protocol42'
CODE_FILES = tuple(dict.fromkeys((*TRAINING_FILES, 'utils/sfra_diagnostics.py',
    'utils/cliplora_bridge_audit.py', 'utils/cliplora_functional_feedback.py',
    'utils/sfra_fast_feedback.py', 'utils/sfra_resident_feedback.py', 'utils/sfra_execution.py',
    'scripts/run_method_a_diagnostics.py', 'tools/sfra/diagnostics.py')))


def source_receipt(source):
    names = ['sfra_config.json', 'command.json', 'bridge_metadata.json', 'private_witness_manifest.json',
             'partition_manifest.csv', 'checkpoints/base_model.pt',
             'protocol/full_schedule.json', 'protocol/eri_protocol.json', 'protocol/probe_manifest.csv']
    names += [f'sfra_rounds/r{r:03d}/tokens.npz' for r in range(1, max(ANCHORS)+1)]
    names += [f'sfra_rounds/r{a:03d}/commit.pt' for a in ANCHORS]
    names += [f'events/r{a+1:03d}_c000_main_normal_B/event.json' for a in ANCHORS]
    if (source/'execution_config.json').exists():
        names += ['execution_config.json']
    return {n: file_hash(source/n) for n in names}


def replace_argument(command, flag, value):
    if flag in command:
        command[command.index(flag)+1] = str(value)
    else:
        index = command.index('DATALOADER.NUM_WORKERS')
        command[index:index] = [flag, str(value)]


def build_command(source, run, data_root, workers):
    # Keep the reference execution mode during anchor/history reconstruction.
    # The runtime applies the explicit job override only after those audits.
    command = read_json(source/'command.json')
    command[0] = sys.executable
    if 'federated_main.py' not in command:
        raise ValueError('Unknown training entry in the reference command')
    command[command.index('federated_main.py')] = str(REPO/'federated_main.py')
    for flag, value in {'--root': data_root, '--output-dir': run,
                        '--client_schedule_file': run/'protocol/full_schedule.json',
                        '--lac_partition_manifest': run/'protocol/partition_source.csv',
                        '--sfra_resume': '', '--sfra_stop_after_round': 0,
                        '--method_a_diagnostic_manifest': run/'diagnostic_job.json'}.items():
        replace_argument(command, flag, value)
    command[command.index('DATALOADER.NUM_WORKERS')+1] = str(workers)
    return command


def summarize(root):
    rows, differences, witness_differences, retention, missing = [], [], [], [], []
    for anchor in ANCHORS:
        for branch in BRANCHES:
            run = root/f'anchor{anchor}'/branch
            if not (run/'diagnostic_completion.json').is_file():
                missing.append(str(run))
                continue
            metrics = read_csv(run/'candidate_metrics.csv')
            for group in ('Overall', 'Many35', 'Medium35', 'Few30', 'Tail20'):
                start = [r for r in metrics if r['kind']=='anchor' and r['group']==group]
                selected = [r for r in metrics if r['kind']=='committed' and r['group']==group]
                if len(start)!=1 or sorted(int(r['round']) for r in selected)!=list(range(anchor+1, anchor+HORIZON+1)):
                    raise ValueError(f'Wrong diagnostic endpoints: {run}/{group}')
                endpoint = next(r for r in selected if int(r['round'])==anchor+HORIZON)
                rows.append(dict(anchor=anchor, branch=branch, group=group,
                    anchor_accuracy=float(start[0]['accuracy']),
                    mean_five_accuracy=sum(float(r['accuracy']) for r in selected)/HORIZON,
                    final_accuracy=float(endpoint['accuracy']),
                    final_minus_anchor=float(endpoint['accuracy'])-float(start[0]['accuracy'])))
            baseline = read_csv(root/f'anchor{anchor}/full-cp/predictions/r{anchor:03d}_full-cp_anchor.csv')
            final = read_csv(run/f'predictions/r{anchor+HORIZON:03d}_{branch}_committed.csv')
            if [(r['sample_id'],r['class_id']) for r in baseline] != [(r['sample_id'],r['class_id']) for r in final]:
                raise ValueError(f'Prediction identities differ: {run}')
            groups = class_groups(read_json(run/'class_prior.json')['counts'])
            for group, ids in groups.items():
                pairs = [(a,b) for a,b in zip(baseline,final) if int(a['class_id']) in ids]
                old_correct = sum(int(a['correct']) for a,b in pairs)
                retained = sum(int(a['correct']) and int(b['correct']) for a,b in pairs)
                gained = sum(not int(a['correct']) and int(b['correct']) for a,b in pairs)
                retention.append(dict(anchor=anchor,branch=branch,group=group,anchor_correct=old_correct,
                    retained_correct=retained,correct_to_wrong=old_correct-retained,wrong_to_correct=gained,
                    retention_fraction=retained/old_correct if old_correct else None,
                    cohort='common_anchor_correct; origin_not_inferred'))
        full = root/f'anchor{anchor}/full-cp/candidate_metrics.csv'
        if full.is_file():
            metrics = read_csv(full)
            for group in ('Overall', 'Many35', 'Medium35', 'Few30', 'Tail20'):
                for candidate in ('ordinary', 'no-cp', 'norm-matched'):
                    a = [r for r in metrics if r['kind']=='same-state' and r['candidate']=='full-cp' and r['group']==group]
                    b = [r for r in metrics if r['kind']=='same-state' and r['candidate']==candidate and r['group']==group]
                    if len(a)==len(b)==1:
                        differences.append(dict(anchor=anchor, group=group, comparison='full-cp minus '+candidate,
                            **{k:float(a[0][k])-float(b[0][k]) for k in ('accuracy','la_loss','ce_loss','margin')}))
            witness = read_csv(root/f'anchor{anchor}/full-cp/witness_metrics.csv')
            for group in ('Overall', 'Many35', 'Medium35', 'Few30', 'Tail20'):
                for candidate in ('ordinary','no-cp','norm-matched'):
                    a = [r for r in witness if r['kind']=='same-state' and r['candidate']=='full-cp' and r['group']==group]
                    b = [r for r in witness if r['kind']=='same-state' and r['candidate']==candidate and r['group']==group]
                    if len(a)==len(b)==1:
                        witness_differences.append(dict(anchor=anchor,group=group,comparison='full-cp minus '+candidate,
                            **{k:float(a[0][k])-float(b[0][k]) for k in ('la_loss','margin')}))
    write_csv(root/'short_run_summary.csv', rows)
    write_csv(root/'same_state_differences.csv', differences)
    write_csv(root/'same_state_witness_differences.csv', witness_differences)
    write_csv(root/'sample_retention.csv', retention)
    write_json(root/'suite_status.json', dict(complete=not missing, missing=missing,
        note='One training seed; anchors/rounds/classes are not independent seeds'))
    report = ['# 方法A短程对照', '', '固定两个起点、三条分支、每条五轮。以下均为独立评估集准确率。', '',
              '| 起点 | 分支 | 类别组 | 起点准确率 | 五轮平均 | 第五轮 | 第五轮减起点 |',
              '|---|---|---|---:|---:|---:|---:|']
    for r in rows:
        report.append(f"| {r['anchor']} | {r['branch']} | {r['group']} | {r['anchor_accuracy']:.3f} | "
                      f"{r['mean_five_accuracy']:.3f} | {r['final_accuracy']:.3f} | {r['final_minus_anchor']:+.3f} |")
    report += ['', f'未完成分支：{len(missing)}。',
               '等步幅分支借用Full-CP训练更新幅度序列；它是诊断控制，不是独立低成本算法。',
               '本表不证明完整100轮优越性、来源机制独立贡献或跨seed稳定性。']
    (root/'short_run_report.md').write_text('\n'.join(report)+'\n', encoding='utf-8')
    return {'complete':not missing, 'completed_branches':6-len(missing)}


def train(args):
    source, root = args.source_run.resolve(), args.output_root.resolve()
    check = preflight(source, root)
    if not check['ready_files']:
        raise ValueError('Missing reference files (see preflight.json): '+', '.join(check['missing']))
    if args.data_root is None or not args.data_root.is_dir():
        raise ValueError('Provide the existing training data directory with --data-root')
    import torch
    if not torch.cuda.is_available():
        raise ValueError('No CUDA GPU available; analysis/preflight work on CPU, training requires the server GPU')
    receipt = source_receipt(source)
    hashes = {n: file_hash(REPO/n) for n in CODE_FILES}
    plan = dict(schema_version=SCHEMA, source=str(source), anchors=list(ANCHORS), horizon=HORIZON,
                branches=list(BRANCHES), source_sha256=receipt, code_sha256=hashes,
                data_root=str(args.data_root.resolve()), num_workers=args.num_workers)
    if getattr(args, 'fast_execution_v2', False):
        from utils.sfra_resident_feedback import execution_config_v2
        plan['execution_override'] = execution_config_v2(args.feedback_batch_size, args.feedback_cache_gib)
    plan_file = root/'diagnostic_plan.json'
    if plan_file.exists() and read_json(plan_file) != plan:
        raise ValueError('Plan/source/code changed; use a new output root')
    write_json(plan_file, plan)
    selected_anchor = getattr(args, 'anchor', None)
    selected_branch = getattr(args, 'branch', None)
    anchors = ANCHORS if selected_anchor is None else (selected_anchor,)
    branches = BRANCHES if selected_branch is None else (selected_branch,)
    for anchor in anchors:
        for branch in branches:
            run = root/f'anchor{anchor}'/branch
            full_run = root/f'anchor{anchor}'/'full-cp'
            if branch != 'full-cp' and not (full_run/'diagnostic_completion.json').is_file():
                raise ValueError(f'Run --anchor {anchor} --branch full-cp to completion first; '
                                 f'{branch} needs its paired reference: {full_run}')
            job = {**plan, 'anchor':anchor, 'branch':branch,
                   'full_branch':str(root/f'anchor{anchor}'/'full-cp'), 'resume':args.resume}
            job_file = run/'diagnostic_job.json'
            if job_file.exists():
                previous = read_json(job_file)
                if {k:v for k,v in previous.items() if k!='resume'} != {k:v for k,v in job.items() if k!='resume'}:
                    raise ValueError(f'Job mismatch: {run}')
                if (run/'diagnostic_completion.json').exists():
                    print(f'Skipping completed branch: {run}', flush=True)
                    continue
                if not args.resume or not (run/'diagnostic_last.pt').is_file():
                    raise ValueError(f'Incomplete output: {run}; use --resume with its checkpoint, or a new output root')
            elif run.exists() and any(run.iterdir()):
                raise ValueError(f'Unregistered nonempty output: {run}')
            else:
                (run/'protocol').mkdir(parents=True)
                for file in (source/'protocol').iterdir():
                    if file.is_file():
                        shutil.copy2(file, run/'protocol'/file.name)
                shutil.copy2(source/'partition_manifest.csv', run/'protocol/partition_source.csv')
                shutil.copy2(source/'bridge_metadata.json', run/'protocol/source_metadata.json')
            write_json(job_file, job)
            command = build_command(source, run, args.data_root.resolve(), args.num_workers)
            write_json(run/'command.json', command)
            print(f'Running anchor {anchor}, branch {branch}, {HORIZON} rounds', flush=True)
            subprocess.run(command, cwd=REPO, check=True)
            if {n:file_hash(REPO/n) for n in CODE_FILES} != hashes:
                raise ValueError('Training code changed during execution; results require audit')
            if not (run/'diagnostic_completion.json').exists():
                raise ValueError(f'Child exited without diagnostic completion: {run}')
    if source_receipt(source) != receipt:
        raise ValueError('Reference files changed during execution; results require audit')
    return summarize(root)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('analyze','preflight','train','summary'), default='analyze')
    parser.add_argument('--source-run', type=Path, default=DEFAULT_SOURCE)
    parser.add_argument('--output-root', type=Path, default=REPO/'output/method_a_diagnostics_20260927')
    parser.add_argument('--data-root', type=Path)
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--fast-execution-v2', action='store_true',
                        help='After reference anchor/history audits, use v2 for all paired branches')
    parser.add_argument('--feedback-batch-size', type=int, default=128,
                        help='V2 feedback forward batch; training batch is unchanged')
    parser.add_argument('--feedback-cache-gib', type=float, default=4.,
                        help='V2 GPU fixed-feature cache budget, between 0 and 4 GiB')
    parser.add_argument('--anchor', type=int, choices=ANCHORS,
                        help='Train only this anchor (default: both anchors)')
    parser.add_argument('--branch', choices=BRANCHES,
                        help='Train only this branch; full-cp must finish first at the same anchor')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--no-plot', action='store_true')
    args = parser.parse_args()
    if args.stage != 'train' and (args.anchor is not None or args.branch is not None):
        parser.error('--anchor and --branch apply only to --stage train')
    if args.fast_execution_v2:
        if args.stage != 'train':
            parser.error('--fast-execution-v2 applies only to --stage train')
        if args.feedback_batch_size < 8:
            parser.error('--feedback-batch-size must be at least 8')
        if not math.isfinite(args.feedback_cache_gib) or not 0 <= args.feedback_cache_gib <= 4:
            parser.error('--feedback-cache-gib must be finite and between 0 and 4')
    elif args.feedback_batch_size != 128 or args.feedback_cache_gib != 4.:
        parser.error('Feedback tuning requires --fast-execution-v2')
    source, output = args.source_run.resolve(), args.output_root.resolve()
    if output==source or output in source.parents or source in output.parents:
        parser.error('Output must be separate from the source run and its parent directories')
    try:
        if args.stage=='analyze':
            result = analyze(source, output, make_plot=not args.no_plot)
        elif args.stage=='preflight':
            result = preflight(source, output)
        elif args.stage=='summary':
            result = summarize(output)
        else:
            result = train(args)
    except (ValueError, FileNotFoundError) as error:
        parser.error(str(error))
    print(json.dumps(result, ensure_ascii=True, indent=2))


if __name__=='__main__':
    main()
