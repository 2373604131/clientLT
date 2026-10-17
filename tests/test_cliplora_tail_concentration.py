"""Partition and launch-safety checks; no model construction or training."""
import contextlib
import io
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from scripts.cliplora_tail_concentration_protocol import (
    LAMBDAS, audit_partition, controlled_partition, reference_partition,
)
from scripts.run_cliplora_tail_concentration import command_for, main, preparation_lock, run_path


class TailConcentrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        counts = [int(500 * .01 ** (c / 99)) for c in range(100)]
        cls.labels = np.repeat(np.arange(100), counts)
        cls.reference = reference_partition(cls.labels, 42)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.tmp = Path(self.temp.name)

    def test_intervention_preserves_pool_and_non_tail_assignments(self):
        observed = []
        for value in LAMBDAS:
            mapping = controlled_partition(self.labels, value, 42)
            stats, counts = audit_partition(self.labels, mapping, self.reference, value)
            self.assertEqual(counts.shape, (30, 100))
            np.testing.assert_array_equal(np.sort(np.concatenate(list(mapping.values()))), np.arange(10847))
            for client in range(30):
                actual = mapping[client][self.labels[mapping[client]] < 80]
                expected = self.reference[client][self.labels[self.reference[client]] < 80]
                np.testing.assert_array_equal(actual, expected)
            self.assertEqual(stats['non_tail_samples_on_designated_clients'], 11)
            observed.append(stats['tail_samples_on_designated_clients'])
        self.assertEqual(observed, [15, 50, 84, 119, 153])

    def test_reference_matches_production_partition(self):
        actual = controlled_partition(self.labels, .75, 42)
        for k in range(30):
            np.testing.assert_array_equal(actual[k], self.reference[k])

    def test_invalid_concentration_rejected(self):
        for value in [-.1, 1.1, float('nan')]:
            with self.subTest(value=value), self.assertRaises(ValueError):
                controlled_partition(self.labels, value, 42)

    def test_default_plan_never_starts_a_process_or_creates_outputs(self):
        output = self.tmp / 'no_training'
        with patch('sys.argv', ['runner', '--output-root', str(output)]), \
                patch('subprocess.run', side_effect=AssertionError('Training was launched')), \
                contextlib.redirect_stdout(io.StringIO()) as buffer:
            main()
        self.assertFalse(output.exists())
        self.assertEqual(buffer.getvalue().count('--lac_method j'), 5)

    def test_commands_keep_j_baseline_and_use_distinct_manifests(self):
        args = SimpleNamespace(output_root=self.tmp, data_root=Path('DATA'), seed=42,
                               protocol_seed=42, num_workers=8)
        destinations = set()
        for value in LAMBDAS:
            run = run_path(args, value)
            cmd = command_for(args, value, run)
            get = lambda key: cmd[cmd.index(key) + 1]
            for key, expected in {'--lac_method': 'j', '--imb_factor': '0.01',
                                  '--intra_group_alpha': '0.5', '--num_users': '30',
                                  '--round': '100', '--local_epochs': '3', '--frac': '1.0',
                                  '--lac_la_tau': '1.0', '--a_refresh_epochs': '1',
                                  '--a_refresh_interval': '10', '--cliplora_rank': '4'}.items():
                self.assertEqual(get(key), expected)
            self.assertEqual(float(get('--specialization_lambda')), value)
            self.assertEqual(Path(get('--lac_partition_manifest')).parent.parent, run)
            destinations.add(get('--output-dir'))
        self.assertEqual(len(destinations), 5)

    def test_lock_cannot_remove_another_process_lock(self):
        lock = self.tmp / '.prepare.lock'
        with preparation_lock(lock):
            with self.assertRaises(FileExistsError):
                with preparation_lock(lock):
                    self.fail('Concurrent ownership')
            self.assertTrue(lock.exists())
        self.assertFalse(lock.exists())

    def test_lock_released_on_error(self):
        lock = self.tmp / '.prepare.lock'
        with self.assertRaises(RuntimeError):
            with preparation_lock(lock):
                raise RuntimeError('Preparation failed')
        self.assertFalse(lock.exists())


if __name__ == '__main__':
    unittest.main()
