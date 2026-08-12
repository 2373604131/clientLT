import tempfile
import unittest
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

from tools.semantic_acquisition.common import stable_hash, write_json
from tools.semantic_acquisition.manifests import (
    DEFAULT_DATA,
    DEFAULT_V1,
    EXPECTED_BUDGETS,
    build_manifests,
    quota_vector,
    write_bundle,
)
from tools.semantic_acquisition.metrics import (
    classification_metrics,
    summarize_v2_rows,
    vector_comparison,
)
from utils.cliplora_loss import fixed_denominator_cross_entropy
from utils.lora_aggregation import aggregate_lora_state
from tools.semantic_acquisition.summarize import (
    _v2_invalid_reasons,
    average_v3_draws,
    summarize_v2_joint,
    summarize_v2_topology,
)


class ManifestTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bundle = build_manifests(DEFAULT_V1, DEFAULT_DATA, [42, 2026], 3)

    def test_frozen_budgets_and_quotas(self):
        self.assertEqual(self.bundle.inputs.budgets, EXPECTED_BUDGETS)
        for budget in EXPECTED_BUDGETS.values():
            quota = quota_vector(budget)
            self.assertEqual(sum(quota), budget)
            self.assertLessEqual(max(quota) - min(quota), 1)

    def test_matching_constraints(self):
        rows = self.bundle.matching_rows
        self.assertEqual(len(rows), 2 * 20 * 3 * 10)
        for row in rows:
            self.assertEqual(
                self.bundle.inputs.quintiles[row["related_class"]],
                self.bundle.inputs.quintiles[row["unrelated_class"]],
            )
            self.assertNotIn(row["unrelated_class"], self.bundle.inputs.top30[row["tail_class"]])
            self.assertNotIn(row["unrelated_class"], self.bundle.inputs.tail_classes)

    def test_v3_conservation_and_equal_clients(self):
        import pandas as pd
        base = pd.DataFrame(self.bundle.base_rows)
        v3 = base[base.stage == "v3"]
        for _, group in v3.groupby(["data_seed", "tail_class", "draw"]):
            self.assertEqual(group.groupby("condition").base_multiset_hash.first().nunique(), 1)
            sizes = group.groupby(["condition", "client_role"]).size()
            for _, placement_sizes in sizes.groupby(level=0):
                self.assertEqual(len(set(placement_sizes.tolist())), 1)

    def test_filler_is_fixed_and_disjoint(self):
        import pandas as pd
        base = pd.DataFrame(self.bundle.base_rows)
        v3 = base[base.stage == "v3"]
        for (seed, class_id), group in v3.groupby(["data_seed", "tail_class"]):
            filler = group[group.is_filler.astype(str).str.lower().isin(["true", "1"])]
            expected = None
            for _, placement in filler.groupby(["draw", "condition"]):
                ids = tuple(sorted(placement.base_sample_id.unique()))
                expected = ids if expected is None else expected
                self.assertEqual(ids, expected)
            non_filler_ids = set(group[~group.is_filler.astype(str).str.lower().isin(["true", "1"])].base_sample_id)
            self.assertTrue(set(expected).isdisjoint(non_filler_ids))

    def test_baseline_and_runtime_share_optimizer_step(self):
        trainer_source = (Path(__file__).resolve().parents[1] / "trainers" / "cliplora.py").read_text(encoding="utf-8")
        runtime_source = (Path(__file__).resolve().parents[1] / "tools" / "semantic_acquisition" / "runtime.py").read_text(encoding="utf-8")
        self.assertIn("def cliplora_optimizer_step", trainer_source)
        self.assertGreaterEqual(trainer_source.count("cliplora_optimizer_step("), 2)
        self.assertIn("cliplora_optimizer_step(", runtime_source)

    def test_runtime_bootstraps_dassl_before_direct_cliplora_import(self):
        runtime_source = (Path(__file__).resolve().parents[1] / "tools" / "semantic_acquisition" / "runtime.py").read_text(encoding="utf-8")
        helper = runtime_source.split("def _load_cliplora_api():", 1)[1].split("def build_experiment_cfg", 1)[0]
        self.assertLess(
            helper.index("import Dassl.dassl.engine"),
            helper.index("from trainers.cliplora import"),
        )

    def test_runtime_requires_successful_not_attempted_steps(self):
        runtime_source = (Path(__file__).resolve().parents[1] / "tools" / "semantic_acquisition" / "runtime.py").read_text(encoding="utf-8")
        self.assertIn("if successful != expected_steps or overflow_count != 0", runtime_source)
        self.assertIn('runtime["optimizer_steps_successful"] == 3', runtime_source)
        self.assertIn("def _validate_smoke_provenance", runtime_source)
        self.assertIn("def _implementation_hashes", runtime_source)
        self.assertIn("tracked_diff_sha256", runtime_source)
        self.assertIn("v2_topology_base_manifest.csv", runtime_source)

    def test_topology_replay_uses_unweighted_baseline_loss_path(self):
        runtime_source = (Path(__file__).resolve().parents[1] / "tools" / "semantic_acquisition" / "runtime.py").read_text(encoding="utf-8")
        self.assertIn('use_loss_weights = "loss_weight" in execution.columns', runtime_source)
        self.assertIn('weights if use_loss_weights else None', runtime_source)
        self.assertIn('"baseline_unweighted_ce"', runtime_source)

    def test_mechanism_precision_is_fp32_and_mainline_remains_amp(self):
        runtime_source = (Path(__file__).resolve().parents[1] / "tools" / "semantic_acquisition" / "runtime.py").read_text(encoding="utf-8")
        manifest_source = (Path(__file__).resolve().parents[1] / "tools" / "semantic_acquisition" / "manifests.py").read_text(encoding="utf-8")
        self.assertIn('cfg.TRAINER.COOP.PREC = "fp32"', runtime_source)
        self.assertIn('"mainline_precision": "amp"', runtime_source)
        self.assertIn('"precision": "fp32", "mainline_precision": "amp"', manifest_source)

    def test_manifest_rebuild_is_identical(self):
        second = build_manifests(DEFAULT_V1, DEFAULT_DATA, [42, 2026], 3)
        self.assertEqual(stable_hash(self.bundle.base_rows), stable_hash(second.base_rows))
        self.assertEqual(stable_hash(self.bundle.execution_rows), stable_hash(second.execution_rows))
        with tempfile.TemporaryDirectory() as first_dir, tempfile.TemporaryDirectory() as second_dir:
            write_bundle(self.bundle, Path(first_dir))
            write_bundle(second, Path(second_dir))
            for name in (
                "base_sample_manifest.csv", "execution_slot_manifest.csv", "matching_manifest.csv",
                "v2_topology_base_manifest.csv", "v2_topology_execution_manifest.csv",
            ):
                self.assertEqual((Path(first_dir) / name).read_bytes(), (Path(second_dir) / name).read_bytes())

    def test_topology_replay_uses_same_complete_universe(self):
        base = pd.DataFrame(self.bundle.topology_base_rows)
        execution = pd.DataFrame(self.bundle.topology_execution_rows)
        fairness = pd.DataFrame(self.bundle.topology_fairness_rows)
        self.assertEqual(len(base), 4 * 10847)
        self.assertEqual(len(execution), 4 * 3 * 10847)
        for _, group in base.groupby(["data_seed", "topology"]):
            self.assertEqual(len(group), 10847)
            self.assertEqual(group.base_sample_id.nunique(), 10847)
            self.assertAlmostEqual(group.groupby("client_id").fedavg_weight.first().sum(), 1.0)
        for _, group in execution.groupby("data_seed"):
            for _, epoch_rows in group.groupby("epoch"):
                signatures = {
                    topology: sorted(zip(rows.base_sample_id, rows.augmentation_seed))
                    for topology, rows in epoch_rows.groupby("topology")
                }
                self.assertEqual(signatures["Dirichlet"], signatures["ClientLT-controlled"])
        controlled = fairness[fairness.topology == "ClientLT-controlled"]
        self.assertTrue(controlled.controlled_tail_constraints.all())
        self.assertTrue((controlled.controlled_tail_count == 153).all())
        self.assertTrue((controlled.controlled_companion_count <= 38).all())
        self.assertTrue((controlled.controlled_min_specialist_purity >= 0.8).all())


class MathTests(unittest.TestCase):
    def test_default_loss_is_baseline_exact(self):
        torch.manual_seed(1)
        logits = torch.randn(7, 100)
        labels = torch.randint(0, 100, (7,))
        self.assertTrue(torch.equal(fixed_denominator_cross_entropy(logits, labels), F.cross_entropy(logits, labels)))

    def test_masked_loss_keeps_actual_batch_denominator(self):
        logits = torch.tensor([[2.0, 0.0], [0.0, 2.0], [1.0, -1.0]])
        labels = torch.tensor([0, 1, 1])
        weights = torch.tensor([1.0, 1.0, 0.0])
        expected = (F.cross_entropy(logits, labels, reduction="none") * weights).sum() / 3
        self.assertTrue(torch.allclose(fixed_denominator_cross_entropy(logits, labels, weights), expected))

    def test_metrics_fixture(self):
        logits = torch.full((2, 100), -5.0)
        logits[:, 3] = torch.tensor([2.0, 1.0])
        logits[:, 4] = torch.tensor([1.0, 2.0])
        labels = torch.tensor([3, 3])
        metrics = classification_metrics(logits, labels, 3)
        self.assertAlmostEqual(metrics["margin"], 0.0)
        self.assertAlmostEqual(metrics["accuracy"], 0.5)
        self.assertEqual(metrics["hardest_negative_class"], 4)

    def test_actual_lora_aggregation_matches_hand_calculation(self):
        global_state = {"x.lora_A": torch.tensor([0.0]), "frozen": torch.tensor([9.0])}
        local = {0: {"x.lora_A": torch.tensor([2.0])}, 1: {"x.lora_A": torch.tensor([4.0])}}
        result = aggregate_lora_state(global_state, local, [0, 1], ["x.lora_A"], {0: 0.5, 1: 0.5})
        self.assertEqual(result["x.lora_A"].item(), 3.0)
        self.assertEqual(result["frozen"].item(), 9.0)

    def test_raw_gradient_and_plain_sgd_invariance(self):
        torch.manual_seed(4)
        model = torch.nn.Linear(3, 2, bias=False)
        x = torch.randn(8, 3)
        y = torch.randint(0, 2, (8,))

        def grad(indices):
            model.zero_grad(set_to_none=True)
            F.cross_entropy(model(x[indices]), y[indices]).backward()
            return model.weight.grad.detach().clone()

        left = 0.5 * grad(slice(0, 4)) + 0.5 * grad(slice(4, 8))
        right = 0.5 * grad(torch.tensor([0, 1, 4, 5])) + 0.5 * grad(torch.tensor([2, 3, 6, 7]))
        comparison = vector_comparison(left, right)
        self.assertLess(comparison["max_abs"], 1e-6)
        theta = model.weight.detach().clone()
        self.assertTrue(torch.allclose(theta - 0.002 * left, theta - 0.002 * right, atol=1e-7, rtol=0))

    def test_summarizer_averages_draws_before_clusters(self):
        rows = []
        for seed in (42, 2026):
            for class_id in (90, 92):
                rows.append({"data_seed": seed, "tail_class": class_id, "condition": "related", "g_margin": 3.0})
                rows.append({"data_seed": seed, "tail_class": class_id, "condition": "tail_only_masked", "g_margin": 1.0})
                for draw, value in enumerate((0.0, 1.0, 2.0)):
                    rows.append({"data_seed": seed, "tail_class": class_id, "condition": f"matched_unrelated_r{draw}", "g_margin": value})
        summary = summarize_v2_rows(rows, bootstrap_draws=100)
        self.assertTrue(all(row["delta_sem"] == 2.0 for row in summary["paired_rows"]))
        self.assertTrue(all(row["delta_pos"] == 2.0 for row in summary["paired_rows"]))

    def test_v3_summarizer_collapses_draws_within_seed_class(self):
        rows = [
            {"data_seed": 42, "tail_class": 90, "draw": draw, "delta_location_e3": value}
            for draw, value in enumerate((1.0, 2.0, 6.0))
        ]
        averaged = average_v3_draws(rows)
        self.assertEqual(len(averaged), 1)
        self.assertEqual(averaged[0]["delta_location_e3"], 3.0)

    def test_json_rejects_non_finite_values(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                write_json(Path(directory) / "bad.json", {"value": float("nan")})

    def test_v2_invalid_reason_reports_amp_mismatch(self):
        import pandas as pd
        frame = pd.DataFrame([{
            "data_seed": 42, "tail_class": 90, "draw": -1, "condition": "related",
            "theta0_hash_equal": True, "pretrain_logits_equal": True,
            "optimizer_steps_equal": True, "scheduler_steps_equal": True,
            "masked_gradient_invariant": True, "tail_augmented_tensors_equal": True,
            "amp_overflow_equal": False, "pass": False, "amp_overflow_count": 1,
            "amp_scale_signature": "abc", "masked_gradient_relative_l2": 0.0,
            "masked_gradient_max_abs": 0.0,
        }])
        reasons = _v2_invalid_reasons(frame)
        self.assertEqual(len(reasons), 1)
        self.assertIn("amp_overflow_equal", reasons[0]["failed_checks"])

    def test_topology_and_joint_verdict_direction(self):
        with tempfile.TemporaryDirectory() as topology_name, tempfile.TemporaryDirectory() as intervention_name, tempfile.TemporaryDirectory() as v1_name:
            topology_dir = Path(topology_name)
            intervention_dir = Path(intervention_name)
            v1_dir = Path(v1_name)
            metrics = []
            for seed in (42, 2026):
                for class_id in range(80, 100):
                    for topology, margin, nll, accuracy in (
                        ("Dirichlet", 1.0, 0.5, 0.8),
                        ("ClientLT-controlled", 0.0, 0.0, 0.4),
                    ):
                        metrics.append({
                            "data_seed": seed, "tail_class": class_id, "epoch": 3,
                            "topology": topology, "g_margin": margin, "g_nll": nll,
                            "after_accuracy": accuracy,
                        })
            pd.DataFrame(metrics).to_csv(topology_dir / "v2_topology_metrics.csv", index=False)
            pd.DataFrame([{"pass": True}]).to_csv(topology_dir / "fairness_invariants.csv", index=False)
            topology_summary = summarize_v2_topology(topology_dir)
            self.assertEqual(topology_summary["verdict"], "DIRICHLET_FORMATION_ADVANTAGE")

            (intervention_dir / "v2_summary.json").write_text(json.dumps({
                "verdict": "POSITIVE_SEMANTIC_TRANSFER", "valid_comparison": True,
            }), encoding="utf-8")
            intervention = []
            v1 = []
            for seed in (42, 2026):
                for class_id in range(80, 100):
                    intervention.append({
                        "data_seed": seed, "tail_class": class_id,
                        "delta_sem": 1.0, "delta_pos": 0.5,
                    })
                    v1.append({
                        "seed": seed, "tail_class_id": class_id,
                        "delta_generic_context_primary": 0.25,
                        "delta_semantic_specific_tail_mass_weighted": 0.75,
                    })
            pd.DataFrame(intervention).to_csv(intervention_dir / "v2_paired_effects.csv", index=False)
            pd.DataFrame(v1).to_csv(v1_dir / "v1_paired_deltas.csv", index=False)
            joint = summarize_v2_joint(topology_dir, intervention_dir, v1_dir)
            self.assertEqual(joint["verdict"], "FORMATION_CHAIN_SUPPORTED")


if __name__ == "__main__":
    unittest.main()
