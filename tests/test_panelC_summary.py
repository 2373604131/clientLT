import csv
import json
import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SUMMARY_SCRIPT = REPO_ROOT / "scripts" / "summarize_panelC.py"
PLOT_SCRIPT = REPO_ROOT / "scripts" / "plot_panelC.py"


def _write_run(root, partition, alpha, seed, tail_acc, topology_offset=0.0):
    if partition == "noniid-labeldir-fine":
        run_name = f"partition={partition}_alpha={alpha}_IF=0.01_localE=5_seed={seed}"
    else:
        run_name = f"partition={partition}_lambda=0.75_alpha={alpha}_rho=3.0_IF=0.01_localE=5_seed={seed}"
    run_dir = root / run_name
    run_dir.mkdir(parents=True)

    metrics_path = run_dir / "round_metrics.csv"
    with metrics_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "epoch",
                "overall_acc",
                "non_tail_acc",
                "bottom20_tail_acc",
                "macro_per_class_acc",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "epoch": 98,
                "overall_acc": 1.0,
                "non_tail_acc": 1.0,
                "bottom20_tail_acc": 1.0,
                "macro_per_class_acc": 1.0,
            }
        )
        writer.writerow(
            {
                "epoch": 99,
                "overall_acc": 60.0 + tail_acc,
                "non_tail_acc": 70.0 + tail_acc,
                "bottom20_tail_acc": tail_acc,
                "macro_per_class_acc": 50.0 + tail_acc,
            }
        )

    summary = {
        "tail_effective_client_number_mean": 4.0 + topology_offset,
        "tail_top1_client_mass_mean": 0.5 + topology_offset,
        "tail_top2_client_mass_mean": 0.75 + topology_offset,
        "client_sample_cv": 0.2 + topology_offset,
        "tail_to_tail_budget": 20,
        "non_tail_to_tail_budget": 2,
        "actual_tail_client_purity": 20 / 22,
    }
    (run_dir / "partition_summary.json").write_text(
        json.dumps(summary),
        encoding="utf-8",
    )
    return run_dir


def _read_csv(path):
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def test_panelC_summary_outputs_all_required_tables_and_uses_epoch_99(tmp_path):
    root = tmp_path / "PanelC"
    out = tmp_path / "summary"
    for seed in (1, 42):
        _write_run(root, "noniid-labeldir-fine", "0.1", seed, 10.0 + seed / 100)
        _write_run(root, "client-longtail", "0.1", seed, 15.0 + seed / 100, 0.1)

    subprocess.run(
        [
            sys.executable,
            str(SUMMARY_SCRIPT),
            "--root",
            str(root),
            "--output-dir",
            str(out),
            "--epoch",
            "99",
            "--strict",
        ],
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )

    all_runs = _read_csv(out / "panel_c_all_runs.csv")
    summary = _read_csv(out / "panel_c_summary.csv")
    paired = _read_csv(out / "panel_c_paired_delta.csv")

    assert len(all_runs) == 4
    assert len(summary) == 2
    assert len(paired) == 2
    assert all(float(row["bottom20_tail_acc"]) > 10 for row in all_runs)
    assert {row["partition"] for row in all_runs} == {
        "noniid-labeldir-fine",
        "client-longtail",
    }
    dirichlet_rows = [row for row in all_runs if row["partition"] == "noniid-labeldir-fine"]
    clientlt_rows = [row for row in all_runs if row["partition"] == "client-longtail"]
    assert all(row["actual_tail_client_purity"] == "" for row in dirichlet_rows)
    assert all(float(row["actual_tail_client_purity"]) > 0 for row in clientlt_rows)
    assert all(float(row["clientlt_minus_dirichlet"]) == pytest.approx(5.0) for row in paired)
    assert "tail_effective_client_number_mean" in summary[0]
    assert "tail_top2_client_mass_std" in summary[0]
    dirichlet_summary = [row for row in summary if row["partition"] == "noniid-labeldir-fine"]
    assert dirichlet_summary[0]["actual_tail_client_purity_mean"] == ""


def test_panelC_plot_writes_pdf_from_summary(tmp_path):
    pytest.importorskip("matplotlib")
    assert 'ax.set_xlabel("Concentration parameter")' in PLOT_SCRIPT.read_text(encoding="utf-8")
    root = tmp_path / "PanelC"
    out = tmp_path / "summary"
    for seed in (1, 42):
        for alpha in ("0.1", "0.5"):
            _write_run(root, "noniid-labeldir-fine", alpha, seed, 10.0 + float(alpha))
            _write_run(root, "client-longtail", alpha, seed, 15.0 + float(alpha), 0.1)

    subprocess.run(
        [
            sys.executable,
            str(SUMMARY_SCRIPT),
            "--root",
            str(root),
            "--output-dir",
            str(out),
            "--strict",
        ],
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )
    pdf_path = out / "panel_c.pdf"
    subprocess.run(
        [
            sys.executable,
            str(PLOT_SCRIPT),
            "--summary-csv",
            str(out / "panel_c_summary.csv"),
            "--output",
            str(pdf_path),
        ],
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )
    assert pdf_path.exists()
    assert pdf_path.stat().st_size > 0
