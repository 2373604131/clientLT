"""Process orchestration checks only; never start training or touch CUDA."""
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from scripts import run_method_a_simple_controls_parallel as parallel
from tools.sfra.simple_controls import NEW_METHODS


class TestParallelLauncher(unittest.TestCase):
    def test_default_mapping_and_custom_gpu_assignment(self):
        args, gpus = parallel.parse_args(['--stage', 'train'])
        self.assertEqual(args.methods, list(NEW_METHODS))
        self.assertEqual(gpus, ['0', '1', '2', '3'])
        self.assertEqual(args.seed, 42)
        _, subset = parallel.parse_args(['--methods', 'cover-cp'])
        self.assertEqual(subset, ['3'])
        _, custom = parallel.parse_args(['--gpus', '4', '5', '6', '7'])
        self.assertEqual(custom, ['4', '5', '6', '7'])
        for invalid in (['--gpus', '0', '0', '0', '0'], ['--gpus', '0', '1'],
                        ['--gpus', '-1', '1', '2', '3'], ['--seed', '3407']):
            with patch('sys.stderr'), self.assertRaises(SystemExit):
                parallel.parse_args(invalid)

    def test_parallel_starts_all_four_before_waiting_and_isolates_gpu_environments(self):
        with tempfile.TemporaryDirectory() as temp:
            events, launched = [], []
            def start(command, **kwargs):
                method = command[command.index('--methods')+1]
                events.append(('start', method))
                launched.append((command, kwargs))
                process = Mock(pid=100+len(launched))
                def poll():
                    events.append(('poll', method))
                    return 0
                process.poll.side_effect = poll
                return process
            with patch.object(parallel.subprocess, 'run') as preflight, \
                    patch.object(parallel.subprocess, 'Popen', side_effect=start), patch('builtins.print'), \
                    patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': '7'}):
                parallel.main(['--stage', 'train', '--output-root', temp, '--resume', '--gpus', '4', '5', '6', '7'])
                self.assertEqual(os.environ['CUDA_VISIBLE_DEVICES'], '7')
            command = preflight.call_args.args[0]
            self.assertEqual(command[command.index('--stage')+1], 'preflight')
            self.assertEqual(events[:4], [('start', method) for method in NEW_METHODS])
            self.assertEqual(len(launched), 4)
            for gpu, (command, options) in zip(('4', '5', '6', '7'), launched):
                self.assertEqual(options['env']['CUDA_VISIBLE_DEVICES'], gpu)
                self.assertEqual(command[command.index('--seed')+1], '42')
                self.assertIn('--resume', command)
                self.assertTrue(options['stdout'].closed)
                method = command[command.index('--methods')+1]
                self.assertTrue((Path(temp)/'launcher_logs'/f'gpu{gpu}_{method}.log').is_file())
            self.assertEqual(len({id(options['env']) for _, options in launched}), 4)
            self.assertEqual(len(list((Path(temp)/'launcher_logs').glob('*.log'))), 4)

    def test_nontraining_stage_does_not_start_gpu_workers(self):
        with patch.object(parallel.subprocess, 'run') as run, \
                patch.object(parallel.subprocess, 'Popen') as start, patch('builtins.print'):
            parallel.main(['--stage', 'preflight'])
            command = run.call_args.args[0]
            self.assertEqual(command[command.index('--stage')+1], 'preflight')
            start.assert_not_called()

    def test_preflight_failure_starts_no_training_and_child_failure_is_reported(self):
        with tempfile.TemporaryDirectory() as temp, patch('builtins.print'):
            argv = ['--stage', 'train', '--parallel', '--output-root', temp]
            with patch.object(parallel.subprocess, 'run', side_effect=ValueError('preflight failed')), \
                    patch.object(parallel.subprocess, 'Popen') as start:
                with self.assertRaisesRegex(ValueError, 'preflight failed'):
                    parallel.main(argv)
                start.assert_not_called()
            children = [Mock(pid=100+i, poll=Mock(return_value=(1 if i == 2 else 0))) for i in range(4)]
            with patch.object(parallel.subprocess, 'run'), \
                    patch.object(parallel.subprocess, 'Popen', side_effect=children):
                with self.assertRaisesRegex(ValueError, 'Failed runs: tailrw-g16'):
                    parallel.main(argv)

    def test_partial_launch_failure_stops_only_owned_children(self):
        with tempfile.TemporaryDirectory() as temp, patch('builtins.print'):
            child = Mock(pid=101)
            with patch.object(parallel.subprocess, 'run'), \
                    patch.object(parallel.subprocess, 'Popen', side_effect=[child, OSError('spawn failed')]), \
                    patch.object(parallel, 'stop_owned') as stop:
                with self.assertRaisesRegex(OSError, 'spawn failed'):
                    parallel.main(['--stage', 'train', '--parallel', '--output-root', temp])
                stop.assert_called_once_with(child)


if __name__ == '__main__':
    unittest.main()
