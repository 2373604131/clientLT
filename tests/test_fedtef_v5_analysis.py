import csv
import importlib.util
import json
import os
import tempfile


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODULE_PATH = os.path.join(PROJECT_ROOT, "scripts", "analyze_fedtef_v5_components.py")
SPEC = importlib.util.spec_from_file_location("analyze_fedtef_v5_components", MODULE_PATH)
ANALYZER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ANALYZER)


def _write_csv(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_component_analyzer_summarizes_branch_and_retention_metrics():
    with tempfile.TemporaryDirectory() as root:
        run_dir = os.path.join(root, "full_v5", "seed1", "frac0.4", "FedTEF-v5_noniid-labeldir")
        os.makedirs(run_dir, exist_ok=True)
        branch_common = {
            "epoch": 5,
            "num_classes": 20,
            "num_samples": 2000,
            "base_acc": 50.0,
            "tail_only_acc": 5.0,
            "fused_acc": 52.0,
            "fused_minus_base_acc": 2.0,
            "base_margin": 1.0,
            "base_cosine_margin": 0.1,
            "fused_margin": 1.2,
            "changed_rate": 0.1,
            "right_flip_rate": 0.05,
            "wrong_flip_rate": 0.01,
            "residual_abs": 0.2,
            "fused_delta_abs": 0.1,
            "semantic_rescue_gate": 0.6,
        }
        _write_csv(
            os.path.join(run_dir, "fedtef_branch_diagnostics.csv"),
            [
                {**branch_common, "scope": "all"},
                {**branch_common, "scope": "head", "base_acc": 60.0, "fused_acc": 60.0},
                {**branch_common, "scope": "tail", "base_acc": 10.0, "fused_acc": 20.0},
            ],
        )
        _write_csv(
            os.path.join(run_dir, "shared_stream_update_diagnostics.csv"),
            [{
                "epoch": 5,
                "stream": "img_adapter",
                "local_update_norm_weighted": 2.0,
                "fedavg_update_norm": 1.0,
                "cancellation_rate": 0.5,
            }],
        )
        _write_csv(
            os.path.join(run_dir, "tailagg_diagnostics_per_class.csv"),
            [{
                "epoch": 5,
                "class_group": "tail",
                "local_energy_sum": 2.0,
                "local_energy_mean_observed": 1.0,
                "fedavg_retention_ratio": 0.2,
                "tailagg_retention_ratio": 0.6,
                "memory_row_norm": 0.8,
            }],
        )
        _write_csv(
            os.path.join(run_dir, "fedtef_v2_gate_history.csv"),
            [{"tail_overlap_count": 10, "protected_count": 30}],
        )
        with open(os.path.join(run_dir, "client_split_fingerprint.json"), "w", encoding="utf-8") as f:
            json.dump({"global_membership_sha256": "same-split"}, f)

        summary = ANALYZER.summarize_run("full_v5", run_dir)
        assert summary["base_tail"] == 10.0
        assert summary["fused_tail"] == 20.0
        assert summary["tail_tailagg_retention"] == 0.6
        assert summary["img_adapter_fedavg_update_norm"] == 1.0
        report = ANALYZER.build_report([summary], [])
        assert "Final fusion" in report
        assert "favorable rescue pattern" in report


if __name__ == "__main__":
    test_component_analyzer_summarizes_branch_and_retention_metrics()
    print("FedTEF-v5 analysis sanity checks passed")
