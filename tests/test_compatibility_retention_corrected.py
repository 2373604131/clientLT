from __future__ import annotations

import unittest
from pathlib import Path

from tools.compatibility_retention.corrected_core import (
    background_adjusted_components,
    corrected_tail_retention_rows,
)


class CorrectedCompatibilityRetentionTests(unittest.TestCase):
    def test_components_subtract_background_only_direct_effect(self):
        result = background_adjusted_components(
            theta0_m_c=1.0,
            local_m_c=3.0,
            background_only_m_c=11.0,
            post_m_c=12.0,
        )
        self.assertEqual(result["g_local"], 2.0)
        self.assertEqual(result["g_post_marginal"], 1.0)

    def test_corrected_ratio_uses_marginal_post_gain_after_class_averaging(self):
        rows = []
        for condition, local, marginal in (
            ("hard_competitor", [1.0, 3.0], [0.5, 1.5]),
            ("matched_control", [2.0, 4.0], [0.4, 0.8]),
        ):
            for index in range(2):
                rows.append({
                    "tail_class": 80,
                    "condition": condition,
                    "g_local": local[index],
                    "g_post_marginal": marginal[index],
                })
        result = corrected_tail_retention_rows(rows)[0]
        self.assertAlmostEqual(result["hard_corrected_retention_ratio"], 0.5)
        self.assertAlmostEqual(result["control_corrected_retention_ratio"], 0.2)
        self.assertAlmostEqual(
            result["hard_minus_control_corrected_retention_ratio"], 0.3
        )

    def test_nonpositive_class_local_gain_is_not_silently_dropped(self):
        rows = [
            {
                "tail_class": 80,
                "condition": "hard_competitor",
                "g_local": 0.0,
                "g_post_marginal": 1.0,
            },
            {
                "tail_class": 80,
                "condition": "matched_control",
                "g_local": 1.0,
                "g_post_marginal": 1.0,
            },
        ]
        with self.assertRaisesRegex(ValueError, "non-positive class-level local gain"):
            corrected_tail_retention_rows(rows)

    def test_v2_runner_freezes_background_adjusted_endpoint(self):
        root = Path(__file__).resolve().parents[1]
        source = (root / "tools" / "compatibility_retention" / "corrected.py").read_text(
            encoding="utf-8"
        )
        shell = (root / "scripts" / "run_compatibility_retention_bridge_v2.sh").read_text(
            encoding="utf-8"
        )
        self.assertIn("post_m_c=float(row[\"post_m_c\"])", source)
        self.assertIn("background_only_m_c=float(bg.background_only_m_c)", source)
        self.assertIn('"v1_verdict_status": "SUPERSEDED_DENOMINATOR_ARTIFACT_NOT_EVIDENCE"', source)
        self.assertIn('choices=("prepare", "background-only", "summarize", "all")', source)
        self.assertIn('"$@"', shell)


if __name__ == "__main__":
    unittest.main()

