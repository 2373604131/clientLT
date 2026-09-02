from __future__ import annotations

import math
import unittest

import torch

from tools.compatibility_retention.core import (
    additive_post_state,
    sample_weights,
    tail_retention_rows,
)


class CompatibilityRetentionBridgeTests(unittest.TestCase):
    def test_additive_post_state_applies_both_deltas_once(self):
        theta0 = {"lora_x": torch.tensor([1.0, 2.0])}
        local = {"lora_x": torch.tensor([3.0, 1.0])}
        background = {"lora_x": torch.tensor([0.5, 4.0])}
        post = additive_post_state(theta0, local, background)
        self.assertTrue(torch.equal(post["lora_x"], torch.tensor([2.5, 3.0])))

    def test_background_weights_are_ordinary_sample_weighted_fedavg(self):
        weights = sample_weights({0: 10, 1: 30, 2: 20}, [0, 2])
        self.assertEqual(weights, {0: 1 / 3, 2: 2 / 3})
        self.assertTrue(math.isclose(sum(weights.values()), 1.0))

    def test_retention_ratio_is_formed_after_within_class_averaging(self):
        rows = []
        # Pair-level ratios differ from ratio of means, so this catches the order.
        for condition, local, post in (
            ("hard_competitor", [1.0, 3.0], [0.8, 2.4]),
            ("matched_control", [2.0, 4.0], [0.4, 0.8]),
        ):
            for index in range(2):
                rows.append({
                    "tail_class": 80,
                    "condition": condition,
                    "g_local": local[index],
                    "g_post": post[index],
                })
        result = tail_retention_rows(rows)[0]
        self.assertAlmostEqual(result["hard_retention_ratio"], 0.8)
        self.assertAlmostEqual(result["control_retention_ratio"], 0.2)
        self.assertAlmostEqual(result["hard_minus_control_retention_ratio"], 0.6)

    def test_nonpositive_class_level_local_gain_invalidates_ratio(self):
        rows = [
            {"tail_class": 80, "condition": "hard_competitor", "g_local": 0.0, "g_post": 1.0},
            {"tail_class": 80, "condition": "matched_control", "g_local": 1.0, "g_post": 1.0},
        ]
        with self.assertRaisesRegex(ValueError, "non-positive class-level local gain"):
            tail_retention_rows(rows)

    def test_runner_freezes_single_endpoint_and_identical_background_invariant(self):
        from pathlib import Path

        root = Path(__file__).resolve().parents[1]
        source = (root / "tools" / "compatibility_retention" / "run.py").read_text(encoding="utf-8")
        shell = (root / "scripts" / "run_compatibility_retention_bridge.sh").read_text(encoding="utf-8")
        self.assertIn('"primary_endpoint": "R_c=G_post_c/G_local_c"', source)
        self.assertIn("group.background_state_hash.nunique() != 1", source)
        self.assertIn('contract["tail_update_scale"] != 1.0', source)
        self.assertIn("class_cluster_summary", source)
        self.assertIn('choices=("prepare", "background", "bridge", "summarize", "all")', source)
        self.assertIn('"$@"', shell)


if __name__ == "__main__":
    unittest.main()
