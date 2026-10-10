"""CPU checks for the optional CUDA policy; production does no parity pilot."""
import os
import tempfile
import unittest
from unittest.mock import patch

import torch

from scripts import run_cliplora_joint_aggregation as single
from scripts import run_cliplora_frozen_aggregation as suite
from tools.client_aggregation.numerics import (DETERMINISTIC, prepare_environment,
    apply_policy, verify_active, verify_record, policy_for)
from tools.client_aggregation.protocol import client_execution_config


class TestCudaNumerics(unittest.TestCase):
    def test_registered_policy_defaults_and_propagation(self):
        self.assertEqual(policy_for({}), 'legacy')  # archived plans stay auditable
        self.assertNotIn('cuda_numerics', client_execution_config({'settings': {'client_concurrency': 6}}))
        self.assertEqual(single.parse_args([]).cuda_policy, 'legacy')
        for rule in ('fedavg', 'tailrw16'):
            args = single.parse_args(['--aggregation-rule', rule])
            self.assertEqual(args.cuda_policy, 'legacy')
            for cmd in single.server_commands(args, 'frozen'):
                self.assertEqual(cmd[cmd.index('--cuda-policy')+1], 'legacy')
        args = suite.parse_args([])
        for rule in suite.METHODS:
            for cmd in suite.commands(args, rule):
                self.assertEqual(cmd[cmd.index('--cuda-policy')+1], 'legacy')

    def test_environment_and_backend_policy_restore_benchmark_override(self):
        before = (torch.are_deterministic_algorithms_enabled(), torch.is_deterministic_algorithms_warn_only_enabled(),
                  torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic)
        precision = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
        try:
            with patch.dict(os.environ, {'CUBLAS_WORKSPACE_CONFIG': ':16:8'}):
                prepare_environment('deterministic')
                self.assertEqual(os.environ['CUBLAS_WORKSPACE_CONFIG'], ':4096:8')
                apply_policy('deterministic')
                verify_active('deterministic')
                torch.backends.cudnn.benchmark = True  # federated_main's old setup
                with self.assertRaisesRegex(ValueError, 'cudnn_benchmark'):
                    verify_active('deterministic')
                apply_policy('deterministic')  # runtime reapplies before training
                verify_active('deterministic')
                self.assertEqual(precision, (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32))
            with patch.dict(os.environ, {'CUBLAS_WORKSPACE_CONFIG': ':bad:'}):
                with self.assertRaisesRegex(ValueError, 'before CUDA'):
                    apply_policy('deterministic')
        finally:
            torch.use_deterministic_algorithms(before[0], warn_only=before[1])
            torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic = before[2:]

    def test_policy_is_frozen_in_plan_and_runtime_record(self):
        with tempfile.TemporaryDirectory() as temp:
            args = single.parse_args(['--aggregation-rule', 'tailrw16', '--output-root', temp, '--cuda-policy', 'deterministic'])
            plan = single.register(args)
            self.assertEqual(plan['settings']['cuda_policy'], 'deterministic')
            self.assertEqual(client_execution_config(single.make_job(plan, 'frozen'))['cuda_numerics'], DETERMINISTIC)
            args.cuda_policy = 'legacy'
            with self.assertRaisesRegex(ValueError, 'changed'): single.register(args)
        record = {k:v for k,v in DETERMINISTIC.items() if k not in ('policy', 'tf32')}
        verify_record(record)
        record['warn_only'] = True
        with self.assertRaisesRegex(ValueError, 'warn_only'): verify_record(record)

if __name__ == '__main__':
    unittest.main()
