import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.eri_closure.attribution import (
    integrated_client_effects,
    rows_from_effects,
    signed_budgets,
)
from tools.eri_closure.analysis import load_round_dump, payload_vectors
from tools.eri_closure.dump import load_eri_round_dump, save_eri_round_dump
from tools.eri_closure.protocol import parse_eri_rounds
from tools.eri_closure.summary import (
    _budgets_with_access,
    _method_location,
    _product_shapley,
    summarize,
)


def test_path_integral_closes_for_quadratic_functional():
    theta = torch.tensor([1.0, -2.0], dtype=torch.float64)
    client_deltas = torch.tensor([[1.0, 0.5], [-0.5, 2.0]], dtype=torch.float64)
    weights = torch.tensor([0.25, 0.75], dtype=torch.float64)
    matrix = torch.tensor([[2.0, 0.0], [0.0, 3.0]], dtype=torch.float64)

    def gradient(vector, _class_id):
        return matrix @ vector

    effects, aggregate = integrated_client_effects(
        theta, client_deltas, weights, [80], gradient, quadrature_points=4
    )
    direct_change = 0.5 * (theta + aggregate) @ matrix @ (theta + aggregate) - 0.5 * theta @ matrix @ theta
    assert effects.sum().item() == pytest.approx(direct_change.item(), abs=1e-10)


def test_four_signed_budgets_keep_supporter_harm_separate():
    effects = torch.tensor([2.0, -3.0, 5.0, -7.0])
    supports = torch.tensor([True, True, False, False])
    budget = signed_budgets(effects, supports)
    assert budget["W"] == 2.0
    assert budget["H"] == 3.0
    assert budget["D"] == 5.0
    assert budget["R"] == 7.0
    assert budget["ERI"] == pytest.approx(7 / 7)
    clients, rows = rows_from_effects(
        effects[None, :], [80], [2, 3, 5, 7], torch.tensor([[0] * 80 + [1], [0] * 80 + [1], [0] * 81, [0] * 81]),
        communication_round=1, method="test",
    )
    assert rows[0]["H"] == 3.0 and rows[0]["D"] == 5.0
    assert [row["signed_role"] for row in clients] == [
        "support_write", "support_harm", "donor", "rewriter"
    ]


def test_access_efficiency_decomposes_positive_support_write():
    effects = torch.tensor([2.0, -3.0, 5.0, -7.0])
    clients, budgets = rows_from_effects(
        effects[None, :],
        [80],
        [2, 3, 5, 7],
        torch.tensor([[0] * 80 + [1], [0] * 80 + [1], [0] * 81, [0] * 81]),
        communication_round=1,
        method="test",
        aggregation_weights=torch.tensor([0.1, 0.2, 0.3, 0.4]),
    )
    row = budgets[0]
    assert row["support_access"] == pytest.approx(0.3)
    assert row["positive_support_access"] == pytest.approx(0.1)
    assert row["support_write_success_rate"] == pytest.approx(1.0 / 3.0)
    assert row["positive_write_strength"] == pytest.approx(20.0)
    assert row["write_factorization_absolute_error"] == pytest.approx(0.0)
    assert (
        row["support_access"]
        * row["support_write_success_rate"]
        * row["positive_write_strength"]
    ) == pytest.approx(row["W"])
    assert row["positive_write_efficiency"] == pytest.approx(row["W"] / row["support_access"])
    assert row["support_effective_clients"] == pytest.approx(0.3**2 / (0.1**2 + 0.2**2))
    assert clients[0]["functional_effect_per_unit_weight"] == pytest.approx(20.0)


def test_old_analysis_csv_backfills_three_factor_write_decomposition(tmp_path):
    import csv

    run = tmp_path / "runs" / "clientlt_fedavg" / "seed42"
    analysis = run / "eri_closure" / "analysis"
    analysis.mkdir(parents=True)
    with (run / "lora_aggregation_weights.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["communication_round", "client_id", "aggregation_weight"]
        )
        writer.writeheader()
        writer.writerows([
            {"communication_round": 10, "client_id": 0, "aggregation_weight": 0.1},
            {"communication_round": 10, "client_id": 1, "aggregation_weight": 0.2},
            {"communication_round": 10, "client_id": 2, "aggregation_weight": 0.7},
        ])
    with (analysis / "client_effects.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "communication_round", "class_id", "method", "client_id",
            "supports_class", "functional_effect", "signed_role",
        ])
        writer.writeheader()
        writer.writerows([
            {"communication_round": 10, "class_id": 80, "method": "trained_server", "client_id": 0, "supports_class": 1, "functional_effect": 2.0, "signed_role": "support_write"},
            {"communication_round": 10, "class_id": 80, "method": "trained_server", "client_id": 1, "supports_class": 1, "functional_effect": -3.0, "signed_role": "support_harm"},
            {"communication_round": 10, "class_id": 80, "method": "trained_server", "client_id": 2, "supports_class": 0, "functional_effect": 5.0, "signed_role": "donor"},
        ])
    enriched = _budgets_with_access([{
        "communication_round": 10, "class_id": 80, "method": "trained_server", "W": 2.0,
    }], run)[0]
    assert enriched["support_access"] == pytest.approx(0.3)
    assert enriched["positive_support_access"] == pytest.approx(0.1)
    assert enriched["support_write_success_rate"] == pytest.approx(1.0 / 3.0)
    assert enriched["positive_write_strength"] == pytest.approx(20.0)
    assert enriched["write_factorization_absolute_error"] == pytest.approx(0.0)


def test_three_factor_shapley_closes_and_routes_access_failure_to_server():
    reference = {"A": 2.0, "rho": 0.5, "mu": 3.0}
    target = {"A": 0.5, "rho": 0.6, "mu": 4.0}
    contributions = _product_shapley(reference, target)
    assert sum(contributions.values()) == pytest.approx(
        target["A"] * target["rho"] * target["mu"]
        - reference["A"] * reference["rho"] * reference["mu"]
    )
    assert _method_location(0.25, 1.2, 4.0 / 3.0) == "server_aggregation"


def test_round_dump_reconstructs_ordered_server_update(tmp_path):
    before = {"image_encoder.lora_A": torch.tensor([0.0, 1.0])}
    local = [
        {"image_encoder.lora_A": torch.tensor([2.0, 3.0])},
        {"image_encoder.lora_A": torch.tensor([6.0, 9.0])},
    ]
    after = {"image_encoder.lora_A": torch.tensor([5.0, 7.5])}
    path = save_eri_round_dump(
        output_dir=tmp_path, args=SimpleNamespace(cliplora_aggregation="fedavg", partition="client-longtail"), cfg="cfg",
        epoch=0, global_before=before, global_after=after, local_weights=local,
        selected_clients=[0, 1], client_sample_counts=[2, 6],
        client_class_counts={0: torch.tensor([1, 1]), 1: torch.tensor([0, 6])},
        global_class_counts=torch.tensor([1, 7]), trainable_keys=list(before),
        server_weights={0: 0.25, 1: 0.75}, aggregation_details={},
    )
    payload, metadata = load_eri_round_dump(path)
    assert payload["selected_client_ids"] == [0, 1]
    assert metadata["reconstruction"]["passed"]
    assert payload["server_weights"].tolist() == pytest.approx([0.25, 0.75])

    # Guard the on-disk writer/attribution-reader contract.
    analysis_payload, _ = load_round_dump(path)
    _, before_vector, after_vector, _, _ = payload_vectors(analysis_payload)
    assert before_vector.tolist() == pytest.approx([0.0, 1.0])
    assert after_vector.tolist() == pytest.approx([5.0, 7.5])


def test_eri_round_parser_rejects_out_of_range():
    assert parse_eri_rounds("1,3,3,10", 10) == [1, 3, 10]
    with pytest.raises(ValueError):
        parse_eri_rounds("0,1", 10)
    with pytest.raises(ValueError):
        parse_eri_rounds("11", 10)


def test_summary_reports_paired_intervention_and_retention(tmp_path):
    import csv
    import json

    for partition, prefix, fed_r, control_r in (
        ("client-longtail", "clientlt", 6.0, 2.0),
        ("matched-dirichlet", "dirichlet", 4.0, 3.0),
    ):
        for aggregation, rewrite, final in (("fedavg", fed_r, 50.0), ("support_normalized", control_r, 60.0)):
            run = tmp_path / "runs" / f"{prefix}_{aggregation}" / "seed1"
            analysis = run / "eri_closure" / "analysis"; analysis.mkdir(parents=True)
            dump = run / "eri_closure" / "dumps" / "round_001"; dump.mkdir(parents=True)
            (dump / "metadata.json").write_text(json.dumps({"resolved_args": {"seed": 1, "partition": partition, "cliplora_aggregation": aggregation}}), encoding="utf-8")
            with (analysis / "round_signed_budgets.csv").open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=["communication_round", "class_id", "method", "W", "H", "D", "R"])
                writer.writeheader(); writer.writerow({"communication_round": 1, "class_id": 80, "method": "trained_server", "W": 2, "H": 1, "D": 1, "R": rewrite})
            with (run / "eri_closure" / "test_per_class_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=["communication_round", "class_id", "accuracy_percent", "mean_true_log_odds"])
                writer.writeheader(); writer.writerow({"communication_round": 1, "class_id": 80, "accuracy_percent": 70, "mean_true_log_odds": 0}); writer.writerow({"communication_round": 2, "class_id": 80, "accuracy_percent": final, "mean_true_log_odds": 0})
    result = summarize(tmp_path, maintenance_start_round=1)
    with (result / "paired_intervention_effects.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    clientlt = next(row for row in rows if row["partition"] == "client-longtail")
    assert float(clientlt["CERI_delta_control_minus_fedavg"]) < 0
    assert float(clientlt["retention_delta_pp"]) > 0
