import numpy as np
import torch
import unittest
import tempfile
from pathlib import Path
from types import SimpleNamespace

from tools.boundary_evidence.core import (
    choose_matched_control,
    class_cluster_summary,
    coexposure_rate,
    hard_negative_ranking,
    metric_deltas,
    pairwise_boundary_metrics,
)
from tools.boundary_evidence.run import ROOT, _build_local_manifests, summarize
from tools.semantic_acquisition.common import file_sha256, write_csv, write_json


class _Inputs:
    class_names = [str(index) for index in range(5)]
    raw_train_ids = np.arange(12)
    labels = np.asarray([0, 0, 0, 1, 1, 1, 2, 2, 3, 3, 4, 4])


class BoundaryEvidenceTests(unittest.TestCase):
    def test_hard_negative_ranking_uses_real_pair_margin_and_excludes_target(self):
        logits = torch.tensor([1.0, 4.0, 3.0, -2.0])
        self.assertEqual(hard_negative_ranking(logits, 0), [1, 2, 3])

    def test_control_is_frequency_nearest_outside_top20(self):
        counts = [10, 20, 31, 29, 100]
        self.assertEqual(choose_matched_control(counts, 0, 1, [1, 2]), 3)

    def test_coexposure_conditions_on_target_carriers(self):
        counts = np.asarray([[2, 1], [1, 0], [0, 4], [3, 2]])
        result = coexposure_rate(counts, 0, 1)
        self.assertEqual(result["carrier_count"], 3)
        self.assertEqual(result["joint_carrier_count"], 2)
        self.assertAlmostEqual(result["q"], 2 / 3)

    def test_pairwise_metrics_and_changes_are_two_sided(self):
        before_c = torch.tensor([[1.0, 2.0], [3.0, 2.0]])
        before_h = torch.tensor([[2.0, 3.0], [4.0, 3.0]])
        after_c = torch.tensor([[3.0, 2.0], [4.0, 2.0]])
        after_h = torch.tensor([[1.0, 4.0], [2.0, 3.0]])
        before = pairwise_boundary_metrics(before_c, before_h, 0, 1)
        after = pairwise_boundary_metrics(after_c, after_h, 0, 1)
        delta = metric_deltas(before, after)
        self.assertGreater(delta["delta_m_c"], 0)
        self.assertGreater(delta["delta_m_h"], 0)
        self.assertGreater(delta["delta_pair_accuracy"], 0)

    def test_class_cluster_summary_averages_pairs_inside_class(self):
        summary = class_cluster_summary({80: [0.0, 2.0], 81: [4.0, 4.0]}, draws=100, seed=7)
        self.assertAlmostEqual(summary["mean"], 2.5)
        self.assertEqual(summary["tail_class_count"], 2)

    def test_local_manifest_has_only_two_paired_conditions(self):
        pairs = [{"tail_class": 0, "hard_class": 1, "hard_rank": 1, "control_class": 2}]
        base, execution = _build_local_manifests(_Inputs(), pairs, [42])
        frame = __import__("pandas").DataFrame(base)
        self.assertEqual(set(frame.condition), {"hard_competitor", "matched_control"})
        left = frame[frame.condition == "hard_competitor"]
        right = frame[frame.condition == "matched_control"]
        self.assertEqual(
            left[left.slot_role == "tail"].base_sample_id.tolist(),
            right[right.slot_role == "tail"].base_sample_id.tolist(),
        )
        self.assertEqual(len(left), len(right))
        self.assertEqual(len(execution), 3 * len(base))

    def test_summary_uses_three_delta_metrics_and_class_clusters(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            a_rows = []
            b_rows = []
            for seed in (42, 2026):
                for c in (80, 81):
                    for rank in range(1, 6):
                        h = rank
                        for topology, q in (("clientlt", 0.2), ("matched_dirichlet", 0.6)):
                            a_rows.append({
                                "data_seed": seed, "topology": topology, "tail_class": c,
                                "hard_class": h, "hard_rank": rank, "carrier_count": 2,
                                "joint_carrier_count": int(q > 0.5), "q": q,
                            })
                        for condition, offset in (("hard_competitor", 1.0), ("matched_control", 0.0)):
                            b_rows.append({
                                "data_seed": seed, "tail_class": c, "hard_class": h,
                                "hard_rank": rank, "control_class": 10 + rank,
                                "condition": condition,
                                "before_m_c": 0.0, "before_m_h": 0.0,
                                "before_pair_accuracy": 0.5,
                                "after_m_c": 1.0 + offset, "after_m_h": offset,
                                "after_pair_accuracy": 0.5 + 0.1 * offset,
                                "delta_m_c": 1.0 + offset, "delta_m_h": offset,
                                "delta_pair_accuracy": 0.1 * offset,
                                "update_norm_diagnostic": 1.0, "optimizer_steps": 3,
                                "scheduler_steps": 3, "precision": "fp32",
                            })
            write_csv(output / "experiment_a_coexposure.csv", a_rows)
            write_csv(output / "experiment_b_metrics.csv", b_rows)
            implementation = {
                name: file_sha256(ROOT / name)
                for name in (
                    "tools/boundary_evidence/core.py",
                    "tools/boundary_evidence/run.py",
                )
            }
            write_json(output / "experiment_contract.json", {
                "hard_k": 5, "conditions": ["hard_competitor", "matched_control"],
                "data_seeds": [42, 2026], "tail_classes": [80, 81],
                "manifest_hashes": {
                    "experiment_a_coexposure.csv": file_sha256(output / "experiment_a_coexposure.csv")
                },
                "implementation_hashes": implementation,
            })
            result = summarize(SimpleNamespace(output_dir=output))
            self.assertEqual(result["verdict"], "BOUNDARY_EVIDENCE_ASYMMETRY_SUPPORTED")
            self.assertEqual(
                result["experiment_b"]["hard_competitor_minus_matched_control"]["delta_m_h"]["tail_class_count"],
                2,
            )
            main = __import__("pandas").read_csv(output / "experiment_b_main_results.csv")
            self.assertEqual(len(main), 3)
            self.assertEqual(
                set(main.columns) & {"delta_m_c", "delta_m_h", "delta_pair_accuracy"},
                {"delta_m_c", "delta_m_h", "delta_pair_accuracy"},
            )


if __name__ == "__main__":
    unittest.main()
