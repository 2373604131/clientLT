"""Run the four seed42 controls concurrently, one experiment per GPU."""
import argparse
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from scripts import run_method_a_simple_controls as launcher
from scripts.run_ab_validation import file_lock


def parse_args(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument('--gpus', nargs='+', type=int)
    parser.add_argument('--parallel', action='store_true', help=argparse.SUPPRESS)  # Old command alias.
    options, remaining = parser.parse_known_args(argv)
    if '--help' in remaining or '-h' in remaining:
        print(__doc__)
        print('Additional option: --gpus 0 1 2 3 (distinct CUDA device IDs, paired with --methods in order; overrides inherited CUDA_VISIBLE_DEVICES)\n')
    args = launcher.parse_args(remaining)
    if any(method not in launcher.NEW_METHODS for method in args.methods):
        parser.error('This launcher runs the four new controls only; train s/full-cp using the base launcher')
    # A subset retains its default device: Cover-CP alone still selects GPU 3.
    gpus = options.gpus if options.gpus is not None else [launcher.NEW_METHODS.index(m) for m in args.methods]
    if len(gpus) != len(args.methods) or len(set(gpus)) != len(gpus) or any(gpu < 0 for gpu in gpus):
        parser.error('Supply one distinct, nonnegative GPU ID per selected method')
    return args, [str(gpu) for gpu in gpus]


def command_for(args, stage, methods=None):
    command = [sys.executable, '-u', str(REPO/'scripts/run_method_a_simple_controls.py'),
        '--stage', stage, '--seed', '42', '--methods', *(args.methods if methods is None else methods),
        '--reference-run', str(args.reference_run), '--data-root', str(args.data_root),
        '--output-root', str(args.output_root), '--num-workers', str(args.num_workers)]
    for value in args.baseline:
        command += ['--baseline', value]
    if args.resume:
        command.append('--resume')
    return command


def stop_owned(process):
    """Stop this wrapper's process tree, including its federated_main child."""
    if process.poll() is not None:
        return
    if os.name == 'nt':
        subprocess.run(['taskkill', '/PID', str(process.pid), '/T', '/F'],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
    else:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            return
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
    process.wait()


def parallel_train(args, env, gpus):
    root = (REPO/args.output_root).resolve()
    with file_lock(root/'locks/parallel_launcher.lock', timeout=.1):
        # Freeze/check the whole suite before starting any GPU process.
        subprocess.run(command_for(args, 'preflight'), cwd=REPO, env=env, check=True)
        log_root = root/'launcher_logs'
        log_root.mkdir(parents=True, exist_ok=True)
        processes, streams, results = {}, [], {}
        try:
            for method, gpu in zip(args.methods, gpus):
                command = command_for(args, 'train', [method])
                child_env = dict(env, CUDA_VISIBLE_DEVICES=gpu)
                log = log_root/f'gpu{gpu}_{method}.log'
                stream = log.open('a', encoding='utf-8')
                streams.append(stream)
                stream.write(f'\n[{time.strftime("%Y-%m-%d %H:%M:%S")}] CUDA_VISIBLE_DEVICES={gpu} {shlex.join(command)}\n')
                stream.flush()
                group = (dict(creationflags=subprocess.CREATE_NEW_PROCESS_GROUP) if os.name == 'nt'
                         else dict(start_new_session=True))
                process = subprocess.Popen(command, cwd=REPO, env=child_env, stdout=stream,
                                           stderr=subprocess.STDOUT, **group)
                processes[method] = process
                print(f'START {method}: GPU={gpu} seed=42 PID={process.pid} log={log}', flush=True)
            while len(results) < len(processes):
                for method, process in processes.items():
                    if method in results:
                        continue
                    code = process.poll()
                    if code is not None:
                        results[method] = code
                        print(f'EXIT {method}: {code}', flush=True)
                if len(results) < len(processes):
                    time.sleep(1)
        finally:
            for process in processes.values():
                stop_owned(process)
            for stream in streams:
                stream.close()
        failed = [method for method, code in results.items() if code]
        if failed:
            raise ValueError('Failed runs: '+', '.join(failed)+'. See '+str(log_root))


def main(argv=None):
    args, gpus = parse_args(argv)
    env = os.environ.copy()
    print('Seed42; one experiment per GPU: '+', '.join(f'{m} -> GPU {gpu}' for m, gpu in zip(args.methods, gpus)), flush=True)
    if args.stage == 'train':
        parallel_train(args, env, gpus)
    else:
        subprocess.run(command_for(args, args.stage), cwd=REPO, env=env, check=True)


def cli():
    try:
        main()
    except KeyboardInterrupt:
        raise SystemExit(130)
    except subprocess.CalledProcessError as error:
        raise SystemExit(error.returncode)
    except (OSError, ValueError) as error:
        raise SystemExit(str(error))


if __name__ == '__main__':
    cli()
