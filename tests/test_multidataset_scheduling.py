"""GPU exclusivity, smoke gates and checkpoint-aware scheduling without CUDA."""
import json
from pathlib import Path
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from tools.benchmarks.common import write_json, read_json
from tools.multidataset.runner import run_jobs
from tools.multidataset.scheduling import estimate_seconds, project_loads


class SchedulingTests(unittest.TestCase):
    def test_resumed_work_and_projected_balance(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            job = dict(method='capt', smoke=False, run_root=str(root), protocol={'matrix': [[10847]]})
            estimates = {'methods': {'capt': {'seconds': 1000}}}
            self.assertEqual(estimate_seconds(job, estimates), 1000)
            write_json(root/'progress.json', {'completed_round': 60})
            self.assertEqual(estimate_seconds(job, estimates), 400)
            self.assertEqual(estimate_seconds(job, estimates, 80), 200)
            with self.assertRaises(ValueError):
                estimate_seconds(job, {'methods': {'capt': {'seconds': float('nan')}}})
        entries = [dict(key=str(i), path=str(i), estimated_seconds=v)
                   for i, v in enumerate([9, 8, 7, 6, 5, 4])]
        projected = project_loads(entries, [3, 4, 5])
        self.assertEqual([x['estimated_seconds'] for x in projected.values()], [13, 13, 13])

    def test_dynamic_gpu_exclusivity_and_failed_smoke_blocks_only_its_job(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths, gates = [], []
            for i in range(6):
                for smoke, targets in [(False, paths), (True, gates)]:
                    job = dict(batch='b'+str(i), method='fedavg-lora', seed=42, smoke=smoke,
                               run_root=str(root/f'r{i}_{smoke}'), protocol={'matrix': [[10847]]})
                    path = root/job['batch']/'jobs'/('method_smoke.json' if smoke else 'method.json')
                    write_json(path, job)
                    targets.append(path)
            active, history, guard = set(), [], threading.Lock()
            first_wave = threading.Barrier(3)
            def fake_run(command, **kwargs):
                job = read_json(command[command.index('--job')+1])
                gpu = kwargs['env']['CUDA_VISIBLE_DEVICES']
                with guard:
                    self.assertNotIn(gpu, active)
                    active.add(gpu)
                    history.append((job['batch'], job['smoke'], gpu))
                    number = len(history)
                if number <= 3:
                    first_wave.wait(timeout=5)
                time.sleep(.01)
                with guard:
                    active.remove(gpu)
                return SimpleNamespace(returncode=int(job['batch']=='b0' and job['smoke']))
            report = root/'queue.json'
            with patch('tools.multidataset.runner.subprocess.run', side_effect=fake_run), \
                 patch('tools.multidataset.collect.audit_job', return_value=[]):
                with self.assertRaisesRegex(RuntimeError, 'independent jobs failed'):
                    run_jobs(paths, [3,4,5], smoke_paths=gates, schedule_path=report)
            state = read_json(report)
            self.assertEqual((len(state['completed']), len(state['failed'])), (5,1))
            self.assertEqual(state['pending'], [])
            self.assertEqual(state['active'], {})
            self.assertEqual({gpu for _,_,gpu in history}, {'3','4','5'})
            self.assertFalse(any(batch=='b0' and not smoke for batch,smoke,_ in history))
            for i in range(1,6):
                events = [(smoke,gpu) for batch,smoke,gpu in history if batch=='b'+str(i)]
                self.assertEqual([v[0] for v in events], [True,False])
                self.assertEqual(events[0][1], events[1][1])


if __name__ == '__main__':
    unittest.main()
