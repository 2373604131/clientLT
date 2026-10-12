"""GPU-free verification of four-worker dispatch, isolation, and failure recovery."""
from concurrent.futures import ThreadPoolExecutor
import contextlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch

from scripts.run_a_projection_diagnosis import parse_args, execute
from tools.a_projection_diagnosis.parallel import (
    selected_jobs, task_root, gpu_tokens, child_command, drain_queue, launch,
)
from tools.a_projection_diagnosis.protocol import read_json


class ParallelTests(unittest.TestCase):
    def args(self, root):
        return parse_args(['--stage', 'server', '--gpus', '0', '1', '2', '3', '--output-root', str(root)])

    def test_exact_five_jobs_disjoint_paths_and_no_recursive_dispatch(self):
        args = self.args(Path('output/example'))
        jobs = selected_jobs(args)
        self.assertEqual([j['id'] for j in jobs], ['fedavg_round020', 'fedavg_round050', 'fedavg_round080', 'fedavg_e1', 'tailrw16_e1'])
        self.assertEqual(len({task_root(args.output_root, j) for j in jobs}), 5)
        for job in jobs:
            command = child_command(args, job, Path.cwd())
            self.assertNotIn('--gpus', command)
            self.assertEqual(command[command.index('--device') + 1], 'cuda:0')
            parsed = parse_args(command[3:])  # python -u script, then arguments
            self.assertEqual(selected_jobs(parsed), [job])
        args.part, args.trajectories, args.rounds = 'events', ['fedavg'], [50]
        self.assertEqual([j['id'] for j in selected_jobs(args)], ['fedavg_round050'])

    def test_respects_allocation_and_rejects_invalid_devices(self):
        self.assertEqual(gpu_tokens([0, 2], {'CUDA_VISIBLE_DEVICES': '3,5,GPU-abcd,7'}), ['3', 'GPU-abcd'])
        self.assertEqual(gpu_tokens([0, 1, 2, 3], {}), ['0', '1', '2', '3'])
        for ids, env in [([0, 0], {}), ([-1], {}), ([0, 1, 2, 3, 4], {}), ([2], {'CUDA_VISIBLE_DEVICES': '4,5'}),
                         ([0], {'CUDA_VISIBLE_DEVICES': ''}), ([0], {'CUDA_VISIBLE_DEVICES': '-1'})]:
            with self.assertRaises(ValueError):
                gpu_tokens(ids, env)
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            parse_args(['--stage', 'run', '--gpus', '0', '1', '--device', 'cuda:3'])

    def test_four_slots_are_concurrent_each_job_once_and_slot_never_overlaps(self):
        jobs = [dict(id=str(i)) for i in range(5)]
        barrier, lock = threading.Barrier(4), threading.Lock()
        active, seen, peak = set(), [], [0]

        def work(slot, job):
            with lock:
                self.assertNotIn(slot, active)
                active.add(slot)
                peak[0] = max(peak[0], len(active))
                seen.append(job['id'])
            if int(job['id']) < 4:
                barrier.wait(timeout=5)
            with lock:
                active.remove(slot)
            return dict(job=job['id'], status='complete')

        results = drain_queue(jobs, ['0', '1', '2', '3'], work)
        self.assertEqual(peak[0], 4)
        self.assertEqual(sorted(seen), [str(i) for i in range(5)])
        self.assertEqual(len(results), 5)

    def test_launch_failure_does_not_drop_other_independent_jobs(self):
        def work(slot, job):
            if job['id'] == 'bad':
                raise OSError('simulated child start failure')
            return dict(job=job['id'], status='complete')
        results = drain_queue([dict(id=x) for x in ('bad', 'ok1', 'ok2')], ['0', '1'], work)
        self.assertEqual({r['job'] for r in results}, {'bad', 'ok1', 'ok2'})
        self.assertEqual(sum(r['status'] == 'failed' for r in results), 1)

    def test_real_child_processes_receive_one_gpu_and_failures_are_visible(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            root = Path(tmp)
            args = self.args(root)

            def command(args, job, repo):
                # Real CPU subprocesses exercise Popen, environments, logs and status.
                program = ('import os,sys,time; print("GPU="+os.environ["CUDA_VISIBLE_DEVICES"]); '
                           'print("THREADS="+os.environ["OMP_NUM_THREADS"]); time.sleep(.1); '
                           'sys.exit(3 if sys.argv[1]=="fedavg_round050" else 0)')
                return [sys.executable, '-c', program, job['id']]

            with patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': '2,4,7,9'}), patch(
                    'tools.a_projection_diagnosis.parallel.child_command', side_effect=command):
                with self.assertRaisesRegex(RuntimeError, 'fedavg_round050'):
                    launch(args, Path.cwd())
                self.assertEqual(os.environ['CUDA_VISIBLE_DEVICES'], '2,4,7,9')
            report = read_json(root / 'parallel_status.json')
            self.assertEqual(len(report['jobs']), 5)
            self.assertEqual(sum(j['status'] == 'complete' for j in report['jobs']), 4)
            for job in report['jobs']:
                log = Path(job['log']).read_text(encoding='utf-8')
                self.assertIn('GPU=' + job['device_token'], log)
                self.assertIn('THREADS=2', log)
                self.assertIn(job['device_token'], ['2', '4', '7', '9'])

    def test_same_trajectory_concurrent_events_use_separate_backend_directories(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            root = Path(tmp)
            source = dict(rule='fedavg', path=str(root / 'source'), meta={}, config={}, partition_sha256='partition')
            calls, lock = [], threading.Lock()
            barrier = threading.Barrier(3)

            class Backend:
                def __init__(self, args, source, store, root, fingerprint):
                    self.root = root

            def panels(backend, store, rnd, target):
                self.assertEqual(backend.root, target)
                barrier.wait(timeout=5)  # all three must run; trajectory-wide locks would fail
                with lock:
                    calls.append((rnd, target))

            args = [parse_args(['--stage', 'run', '--part', 'events', '--rounds', str(r),
                                '--trajectories', 'fedavg', '--output-root', str(root)]) for r in (20, 50, 80)]
            with patch('tools.a_projection_diagnosis.states.StateStore', return_value=object()), patch(
                    'tools.a_projection_diagnosis.backend.ClipBackend', Backend), patch(
                    'tools.a_projection_diagnosis.experiment.event_panels', side_effect=panels):
                with ThreadPoolExecutor(max_workers=3) as pool:
                    futures = [pool.submit(execute, a, [source], 'test') for a in args]
                    for future in futures:
                        future.result()
            self.assertEqual(sorted(r for r, _ in calls), [20, 50, 80])
            self.assertEqual(len({p for _, p in calls}), 3)

    def test_resume_skips_complete_formal_task_before_loading_model(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            root = Path(tmp)
            source = dict(rule='fedavg', path=str(root / 'source'), meta={}, config={}, partition_sha256='partition')
            args = parse_args(['--stage', 'run', '--part', 'e1', '--trajectories', 'fedavg', '--output-root', str(root)])
            folder = root / 'fedavg/e1'
            folder.mkdir(parents=True)
            for name in ('completion.json', 'numerical_checks.json', 'projection_geometry.csv',
                         'representation_curves.csv', 'full_model_curves.csv'):
                (folder / name).write_text('{}', encoding='utf-8')
            with patch('tools.a_projection_diagnosis.states.StateStore', side_effect=AssertionError('loaded completed model')):
                execute(args, [source], 'test')
                args.stage = 'smoke'
                execute(args, [source], 'test', skip_completed_smoke=True)


if __name__ == '__main__':
    unittest.main()
