"""Fixture and boundary tests for the training-free P0/V1 analysis."""

from __future__ import annotations

import unittest

import numpy as np

from tools.analysis.context_colocation_metrics import (
    class_set_coverage,
    generate_frequency_matched_null_sets,
    generic_context_metrics,
    support_weights,
    topology_metrics,
)
from utils.datasplit import partition_client_longtail_controlled


class ContextColocationMetricTests(unittest.TestCase):
    def setUp(self):
        # clients x classes; class 4 is the fixture tail class.
        self.counts = np.asarray(
            [
                [2, 0, 1, 0, 2],
                [0, 3, 1, 0, 1],
                [0, 0, 0, 4, 0],
            ],
            dtype=np.int64,
        )

    def test_support_and_three_weights(self):
        support, weights = support_weights(self.counts, 4)
        np.testing.assert_array_equal(support, [0, 1])
        np.testing.assert_allclose(weights["client_unweighted"], [0.5, 0.5])
        np.testing.assert_allclose(weights["tail_mass_weighted"], [2 / 3, 1 / 3])
        np.testing.assert_allclose(weights["fedavg_weighted"], [0.5, 0.5])

    def test_topology_by_hand(self):
        metrics = topology_metrics(self.counts, 4)
        self.assertEqual(metrics["support_client_count"], 2)
        self.assertAlmostEqual(metrics["top2_tail_client_mass"], 1.0)
        self.assertAlmostEqual(metrics["effective_support_clients"], 9 / 5)
        self.assertAlmostEqual(metrics["tail_mass_accounted_for"], 1.0)

    def test_generic_breadth_and_dose_by_hand(self):
        metrics = generic_context_metrics(self.counts, 4, [0, 1, 2, 3])
        # client 0 has classes {0,2}; client 1 has {1,2}; q_tail=(2/3,1/3).
        self.assertAlmostEqual(
            metrics["generic_companion_class_count_tail_mass_weighted"], 2.0
        )
        self.assertAlmostEqual(
            metrics["generic_companion_class_fraction_tail_mass_weighted"], 0.5
        )
        self.assertAlmostEqual(
            metrics["generic_companion_sample_count_tail_mass_weighted"], 10 / 3
        )
        self.assertAlmostEqual(
            metrics["generic_companion_sample_fraction_tail_mass_weighted"], 2 / 3
        )

    def test_uniform_coverage_by_hand(self):
        # R={0,1}. Coverage is 1/2 on both supporting clients.
        self.assertAlmostEqual(
            class_set_coverage(self.counts, 4, [0, 1], "tail_mass_weighted"), 0.5
        )

    def test_empty_support_raises(self):
        counts_with_empty_class = np.pad(self.counts, ((0, 0), (0, 1)))
        with self.assertRaisesRegex(ValueError, "empty support"):
            support_weights(counts_with_empty_class, 5)

    def test_duplicate_class_set_raises(self):
        with self.assertRaisesRegex(ValueError, "distinct"):
            class_set_coverage(self.counts, 4, [0, 0], "client_unweighted")

    def test_null_is_deterministic_distinct_and_frequency_matched(self):
        quintiles = {class_id: class_id // 2 for class_id in range(10)}
        first = generate_frequency_matched_null_sets(9, [0, 2, 4, 6, 8], quintiles, 20)
        second = generate_frequency_matched_null_sets(9, [0, 2, 4, 6, 8], quintiles, 20)
        self.assertEqual(first, second)
        for sampled in first:
            self.assertEqual(len(sampled), len(set(sampled)))
            self.assertNotIn(9, sampled)
            self.assertEqual(sorted(quintiles[x] for x in sampled), [0, 1, 2, 3, 4])

    def test_null_candidate_shortage_raises(self):
        with self.assertRaisesRegex(ValueError, "candidates"):
            generate_frequency_matched_null_sets(
                0, [0, 1, 2], {0: 0, 1: 0, 2: 0}, draws=1
            )


class ControlledPartitionTests(unittest.TestCase):
    def test_zero_tail_leakage_and_per_client_purity(self):
        labels = np.asarray([0] * 20 + [1] * 15 + [2] * 10 + [3] * 8 + [4] * 12)
        partition = partition_client_longtail_controlled(
            labels,
            n_parties=3,
            num_classes=5,
            head_client_ratio=2 / 3,
            tail_client_ratio=1 / 3,
            tail_class_ratio=0.2,
            intra_group_alpha=0.5,
            tail_client_min_purity=0.8,
            tail_class_ids=[3],
            rng=np.random.RandomState(42),
        )
        merged = np.concatenate([partition[k] for k in range(3)])
        np.testing.assert_array_equal(np.sort(merged), np.arange(labels.size))
        self.assertFalse(np.any(labels[partition[0]] == 3))
        self.assertFalse(np.any(labels[partition[1]] == 3))
        specialist_labels = labels[partition[2]]
        self.assertGreaterEqual(np.mean(specialist_labels == 3), 0.8)
        self.assertLessEqual(np.sum(specialist_labels != 3), 2)


if __name__ == "__main__":
    unittest.main()
