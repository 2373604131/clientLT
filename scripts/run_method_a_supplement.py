"""Run the four frozen seed42 100-round A controls; no smoke runs or hash gates."""
import argparse
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from tools.sfra.supplement import (SCHEMA, METHODS, RULES, read_json, write_json, validate_source, summarize)

DEFAULT_SOURCE = REPO/'output/cifar100_LT/sfra_cp/seed42/client-longtail/full-cp/lambda10_mu1_protocol42'
DEFAULT_OUTPUT = REPO/'output/method_a_supplement_seed42'


def set_argument(command, flag, value):
    if flag in command:
        command[command.index(flag)+1] = str(value)
    else:
        at = command.index('DATALOADER.NUM_WORKERS')
        command[at:at] = [flag, str(value)]


def build_command(source, run, data_root, method, resume):
    command = list(read_json(source/'command.json'))
    command[0] = sys.executable
    index = next((i for i, x in enumerate(command) if Path(x).name == 'federated_main.py'), None)
    if index is None or 'DATALOADER.NUM_WORKERS' not in command:
        raise ValueError('Unrecognized Full-CP reference command')
    command[index] = str(REPO/'federated_main.py')
    variant = 's' if method=='norm-matched' else ('current-cp' if method=='current-cp' else 'full-cp')
    overrides = {'--root':data_root, '--output-dir':run,
        '--client_schedule_file':run/'protocol/full_schedule.json',
        '--lac_partition_manifest':run/'protocol/partition_source.csv',
        '--sfra_variant':variant, '--sfra_resume':run/'checkpoints/sfra_last.pt' if resume else '',
        '--sfra_stop_after_round':0, '--method_a_supplement_manifest':run/'supplement_job.json'}
    for flag, value in overrides.items():
        set_argument(command, flag, value)
    # Execution version, worker count and feedback batch size are inherited, not retuned.
    if '--b_transfer_enable' in command:
        raise ValueError('B transfer must be disabled in the reference command')
    for flag, value in {'--seed':'42', '--split_seed':'42', '--round':'100',
                        '--sfra_retention_weight':'10', '--sfra_classification_weight':'1'}.items():
        if flag not in command or float(command[command.index(flag)+1]) != float(value):
            raise ValueError('Unexpected reference command setting: '+flag)
    for flag in ('--method_a_diagnostic_manifest', '--b_problem2_replay_manifest'):
        if flag in command and command[command.index(flag)+1]:
            raise ValueError('Reference command contains another experiment runtime: '+flag)
    execution = read_json(source/'execution_config.json') if (source/'execution_config.json').is_file() else None
    if bool(execution) != any(f in command for f in ('--sfra_fast_execution', '--sfra_fast_execution_v2')):
        raise ValueError('Reference command and execution_config disagree')
    return command


def train(args):
    source, root, data = args.source_run.resolve(), args.output_root.resolve(), args.data_root.resolve()
    if root == source or root in source.parents or source in root.parents:
        raise ValueError('Output root must be separate from the original Full-CP run')
    if not data.is_dir():
        raise FileNotFoundError('Training data directory: '+str(data))
    _, norms = validate_source(source)
    methods = METHODS if args.method=='all' else (args.method,)
    results = {}
    for method in methods:
        run = root/method
        job = dict(schema_version=SCHEMA, experiment=method, seed=42, protocol_seed=42,
            rounds=100, a_refresh_rounds=list(range(1,91)), source_run=str(source), data_root=str(data),
            shuffle_seed=420928, target_rule=RULES[method], full_effective_norms=norms,
            reference_command=read_json(source/'command.json'),
            reference_execution=(read_json(source/'execution_config.json')
                                 if (source/'execution_config.json').is_file() else None),
            execution='inherit reference command exactly', hyperparameters='lambda10_mu1',
            test_selection='none; fixed final third correction and fixed 100-round horizon')
        job_path = run/'supplement_job.json'
        if job_path.is_file():
            if read_json(job_path) != job:
                raise ValueError('Registered job settings differ; choose a separate output root: '+str(run))
            if (run/'completion.json').is_file():
                results[method] = 'already complete'
                continue
        elif run.exists() and any(run.iterdir()):
            raise ValueError('Refusing to mix with unregistered output: '+str(run))
        else:
            (run/'protocol').mkdir(parents=True)
            for name in ('full_schedule.json', 'eri_protocol.json', 'probe_manifest.csv'):
                shutil.copy2(source/'protocol'/name, run/'protocol'/name)
            shutil.copy2(source/'partition_manifest.csv', run/'protocol/partition_source.csv')
            shutil.copy2(source/'bridge_metadata.json', run/'protocol/source_metadata.json')
            write_json(job_path, job)
        resume = (run/'checkpoints/sfra_last.pt').is_file()
        if not resume and (run/'round_metrics.csv').is_file():
            raise ValueError('Output has evaluations but no committed checkpoint; preserve it and use a new output root')
        command = build_command(source, run, data, method, resume)
        write_json(run/'command.json', command)
        print(f'{method}: {"resume" if resume else "start"} seed42, 100 rounds, reference execution retained', flush=True)
        subprocess.run(command, cwd=REPO, check=True)
        if not (run/'completion.json').is_file() or int(read_json(run/'completion.json')['completed_round']) != 100:
            raise RuntimeError('Training exited without a 100-round completion: '+str(run))
        results[method] = 'complete'
    return results


def baseline_arguments(values):
    result = {}
    for value in values:
        name, separator, path = value.partition('=')
        if not separator or name not in ('s', 'flat-cp') or name in result:
            raise ValueError('Use --baseline s=PATH and/or --baseline flat-cp=PATH once each')
        result[name] = Path(path).resolve()
    return result


def pack(root, baselines, short_roots):
    status = summarize(root, baselines, short_roots)
    baselines = {k:Path(v) for k,v in status['resolved_baselines'].items()}
    short_roots = [Path(v) for v in status['resolved_short_roots']]
    destination = root.with_name(root.name+'_results.zip')
    # Include predictions/witness arrays and all small diagnostics, never model weights or datasets.
    suffixes = {'.json', '.csv', '.npz', '.md', '.yaml'}
    with zipfile.ZipFile(destination, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(root.rglob('*')):
            if path.is_file() and path.suffix in suffixes and not {'events', 'checkpoints', 'formal_lora', 'references', 'short_inputs'} & set(path.relative_to(root).parts):
                archive.write(path, root.name+'/'+path.relative_to(root).as_posix())
        source = Path(next(read_json(root/m/'supplement_job.json')['source_run']
                           for m in METHODS if (root/m/'supplement_job.json').is_file()))
        if not source.is_dir():
            source = root/'references/full-cp'
        for label, run in {'full-cp':source, **baselines}.items():
            for path in sorted(run.rglob('*')):
                if path.is_file() and path.suffix in {'.json', '.csv', '.npz'} and not {'events', 'checkpoints'} & set(path.relative_to(run).parts):
                    archive.write(path, root.name+'/references/'+label+'/'+path.relative_to(run).as_posix())
        for short in short_roots:
            for path in sorted(short.rglob('*')):
                if path.is_file() and (path.name in ('diagnostic_completion.json', 'diagnostic_job.json')
                                     or path.suffix == '.csv' and path.parent.name == 'predictions'):
                    archive.write(path, root.name+'/short_inputs/'+short.name+'/'+path.relative_to(short).as_posix())
        for name in ('federated_main.py', 'utils/cliplora_sfra.py', 'utils/cliplora_la_control.py',
                     'utils/method_a_supplement.py', 'tools/sfra/supplement.py', 'scripts/run_method_a_supplement.py'):
            archive.write(REPO/name, root.name+'/implementation/'+name)
    return dict(archive=str(destination), **status)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('train', 'summary', 'pack'), default='train')
    parser.add_argument('--method', choices=('all', *METHODS), default='all')
    parser.add_argument('--seed', type=int, choices=(42,), default=42)
    parser.add_argument('--source-run', type=Path, default=DEFAULT_SOURCE)
    parser.add_argument('--output-root', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--data-root', type=Path, default=REPO/'DATA')
    parser.add_argument('--baseline', action='append', default=[], metavar='NAME=PATH')
    parser.add_argument('--short-root', type=Path, action='append', default=[])
    args = parser.parse_args()
    baselines = baseline_arguments(args.baseline)
    if args.stage=='train':
        result = train(args)
    elif args.stage=='summary':
        result = summarize(args.output_root.resolve(), baselines, args.short_root)
    else:
        result = pack(args.output_root.resolve(), baselines, args.short_root)
    import json
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
