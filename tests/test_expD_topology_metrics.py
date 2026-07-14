import math
import unittest

import numpy as np

from experiments.expD_utils import (
    build_client_class_counts,
    compute_topology_metrics,
    create_client_schedule,
    import_datasplit_functions,
)


class ExpDTopologyMetricsTest(unittest.TestCase):
    def test_synthetic_splits_preserve_samples_and_counts(self):
        labels = np.repeat(np.arange(10), 20)
        num_clients = 8
        num_classes = 10
        partition_client_longtail, partition_dirichlet_fine_labels = import_datasplit_functions()

        dirichlet_map = partition_dirichlet_fine_labels(
            labels,
            n_parties=num_clients,
            beta=0.5,
            seed=7,
            min_client_samples=1,
        )
        clientlt_map = partition_client_longtail(
            labels,
            n_parties=num_clients,
            num_classes=num_classes,
            head_client_ratio=0.75,
            tail_client_ratio=0.25,
            head_class_ratio=0.8,
            tail_class_ratio=0.2,
            specialization_lambda=1.0,
            intra_group_alpha=0.5,
            head_leakage_scale=3.0,
        )

        dir_counts = build_client_class_counts(labels, dirichlet_map, num_clients, num_classes)
        clt_counts = build_client_class_counts(labels, clientlt_map, num_clients, num_classes)

        self.assertEqual(int(dir_counts.sum()), len(labels))
        self.assertEqual(int(clt_counts.sum()), len(labels))
        self.assertTrue(np.array_equal(dir_counts.sum(axis=0), clt_counts.sum(axis=0)))
        self.assertTrue(np.array_equal(dir_counts.sum(axis=0), np.bincount(labels, minlength=num_classes)))

    def test_topology_metrics_ranges_and_zero_variance_coexposure(self):
        labels = np.repeat(np.arange(10), 20)
        num_clients = 8
        num_classes = 10
        partition_client_longtail, _ = import_datasplit_functions()
        net_map = partition_client_longtail(
            labels,
            n_parties=num_clients,
            num_classes=num_classes,
            head_client_ratio=0.75,
            tail_client_ratio=0.25,
            head_class_ratio=0.8,
            tail_class_ratio=0.2,
            specialization_lambda=1.0,
            intra_group_alpha=0.5,
            head_leakage_scale=3.0,
        )
        counts = build_client_class_counts(labels, net_map, num_clients, num_classes)
        schedule = [list(range(num_clients)) for _ in range(5)]

        metrics, per_client, per_tail = compute_topology_metrics(
            counts,
            schedule,
            protocol="clientlt",
            tail_class_ratio=0.2,
            tail_client_ratio=0.25,
        )

        self.assertGreaterEqual(metrics["tail_support_jaccard_mean"], 0.0)
        self.assertLessEqual(metrics["tail_support_jaccard_mean"], 1.0)
        self.assertGreaterEqual(metrics["tail_effective_client_number_mean"], 1.0)
        self.assertEqual(metrics["tail_coexposure_corr_valid_pair_count"], 0)
        self.assertTrue(math.isnan(metrics["tail_coexposure_corr_mean"]))
        self.assertEqual(len(per_client), num_clients)
        self.assertEqual(len(per_tail), 2)

    def test_client_schedule_shape_and_hash_stable(self):
        schedule_a = create_client_schedule(10, 0.2, 6, 3)
        schedule_b = create_client_schedule(10, 0.2, 6, 3)
        self.assertEqual(schedule_a, schedule_b)
        self.assertEqual(len(schedule_a), 6)
        self.assertTrue(all(len(round_clients) == 2 for round_clients in schedule_a))


if __name__ == "__main__":
    unittest.main()
