import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from scripts.analyze_d2_conflict import (
    binary_auc,
    compute_geometry,
    reconstruct_training_fedavg,
    safe_cosine,
    sign_disagreement,
    spearman,
)
from scripts.analyze_d2b_scalar_ceiling import (
    compose_class_logits,
    compose_scalar_logits,
    optimize_weight_distributions,
    utility_interaction_report,
    vector_from_weights,
)
from scripts.analyze_d3_boundary import (
    apply_logit_adjustment,
    centroid_logits,
    fit_balanced_ridge,
    nearest_centroid_weights,
    ridge_logits,
    select_tau,
)
from scripts.analyze_p0_head_pareto import (
    _lookup,
    envelope_auc,
    match_class_to_scalar,
    pareto_frontier,
    recover_alternative_weights,
    select_budget_choices,
    weights_at_gamma,
)
from scripts.run_d23 import analyzer_command, build_dump_command, dump_complete, load_frozen
from scripts.run_g0_d1 import CONFIGS
from utils.cusp_minimal import make_flat_spec
from utils.d23_common import class_split, stratified_fit_calibration_split


def _dump_payload():
    before = {"block.q_lora_A": torch.zeros(2)}
    local = [
        {"block.q_lora_A": torch.tensor([1.0, 0.0])},
        {"block.q_lora_A": torch.tensor([0.0, 1.0])},
        {"block.q_lora_A": torch.tensor([-1.0, 0.0])},
    ]
    weights = torch.tensor([0.2, 0.3, 0.5], dtype=torch.float64)
    after = {"block.q_lora_A": sum(float(w) * state["block.q_lora_A"] for w, state in zip(weights, local))}
    counts = torch.tensor([
        [10, 3, 3, 3, 2],
        [10, 3, 3, 3, 3],
        [10, 3, 3, 3, 0],
    ])
    return {
        "flatten_spec": make_flat_spec(before).as_dict(),
        "global_before_trainable": before,
        "global_after_fedavg_trainable": after,
        "local_trainable_states": local,
        "selected_client_ids": [0, 1, 2],
        "fedavg_weights": weights,
        "client_class_counts": counts,
        "global_class_counts": counts.sum(0),
    }


def test_d2_geometry_uses_leave_one_client_out_support_reference():
    rows, audit = compute_geometry(_dump_payload(), {"communication_round": 20})
    by_client = {row["client_id"]: row for row in rows}

    assert audit["fedavg_reconstruction_relative_error"] < 1e-6
    # Client 0 sees client 1's [0, 1] direction, not a direction containing itself.
    assert by_client[0]["peer_support_count"] == 1
    assert by_client[0]["cosine_to_support_direction"] == pytest.approx(0.0)
    # A non-supporter sees both supporters as its independent reference.
    assert by_client[2]["peer_support_count"] == 2


def test_d2_reconstruction_matches_training_fp32_accumulation_not_delta_algebra():
    payload = _dump_payload()
    spec = make_flat_spec(payload["global_before_trainable"])
    reconstructed = reconstruct_training_fedavg(payload, spec)
    expected = payload["global_after_fedavg_trainable"]["block.q_lora_A"].double()

    assert torch.equal(reconstructed, expected)


def test_d2_statistics_have_expected_direction():
    assert safe_cosine(torch.tensor([1.0, 0.0]), torch.tensor([-1.0, 0.0])) == -1.0
    assert sign_disagreement(torch.tensor([1.0, -2.0]), torch.tensor([-1.0, -3.0])) == 0.5
    assert spearman([1, 2, 3], [10, 20, 30]) == pytest.approx(1.0)
    assert binary_auc([False, True], [0.0, 1.0]) == pytest.approx(1.0)


def test_d3_train_split_preserves_every_class_on_both_sides():
    labels = torch.tensor([0] * 5 + [1] * 7 + [2] * 9)
    fit, calibration = stratified_fit_calibration_split(labels, seed=42)

    assert set(labels[fit].tolist()) == {0, 1, 2}
    assert set(labels[calibration].tolist()) == {0, 1, 2}
    assert not set(fit.tolist()).intersection(calibration.tolist())


def test_d3_centroid_and_balanced_ridge_recover_separable_features():
    features = torch.tensor([[2.0, 0.0], [1.0, 0.0], [0.0, 2.0], [0.0, 1.0]])
    labels = torch.tensor([0, 0, 1, 1])
    centroids = nearest_centroid_weights(features, labels, 2)
    ridge = fit_balanced_ridge(features, labels, 2, ridge=1e-2)

    assert torch.equal(centroid_logits(features, centroids).argmax(1), labels)
    assert torch.equal(ridge_logits(features, ridge).argmax(1), labels)


def test_d3_logit_adjustment_and_tau_selection_use_calibration_only():
    logits = torch.tensor([[2.0, 1.0], [2.0, 1.5], [1.0, 2.0], [1.0, 2.0]])
    labels = torch.tensor([0, 1, 1, 1])
    priors = torch.tensor([0.9, 0.1])
    adjusted = apply_logit_adjustment(logits, priors, 1.0)
    selected, rows = select_tau(logits, labels, priors, [0.0, 1.0], [0], [1])

    assert adjusted[:, 1].sub(logits[:, 1]).mean() > adjusted[:, 0].sub(logits[:, 0]).mean()
    assert selected in {0.0, 1.0}
    assert len(rows) == 2


def test_d23_dump_command_is_one_seed42_candidate_r4_trajectory(tmp_path):
    freeze_path = tmp_path / "freeze.json"
    frozen = {
        "verdict": "PASS",
        "selected_config_id": "candidate_r4",
        "selected_config": CONFIGS["candidate_r4"],
    }
    freeze_path.write_text(json.dumps(frozen), encoding="utf-8")
    loaded = load_frozen(freeze_path)
    args = SimpleNamespace(
        output_root=tmp_path / "d23",
        data_root=Path("DATA"),
        python_bin="python",
        lr=0.001,
        test_batch_size=128,
        num_workers=4,
    )
    root, command = build_dump_command(args, loaded)

    assert command[command.index("--round") + 1] == "80"
    assert command[command.index("--v0_dump_rounds") + 1] == "20,50,80"
    assert command[command.index("--seed") + 1] == "42"
    assert command[command.index("--partition") + 1] == "client-longtail"
    assert command[command.index("--cliplora_rank") + 1] == "4"
    assert root.name == "dump_seed42"
    assert not dump_complete(root)


def test_class_split_has_only_eighty_head_and_twenty_tail_classes():
    head, tail = class_split(torch.arange(100, 0, -1))
    assert len(head) == 80
    assert len(tail) == 20
    assert set(head).isdisjoint(tail)


def test_d2b_utility_report_detects_non_additive_client_class_interaction():
    rows = []
    for class_id in range(80, 100):
        for client_id in range(30):
            rows.append({
                "communication_round": 20,
                "class_id": class_id,
                "client_id": client_id,
                "tail_margin_contribution": (
                    (1.0 if (class_id + client_id) % 2 else -1.0)
                    * (1.0 + 0.01 * class_id + 0.001 * client_id)
                ),
            })
    report = utility_interaction_report(rows, 20)

    assert report["interaction_energy_ratio"] > 0.9
    assert report["clients_with_both_beneficial_and_harmful_classes_rate"] == 1.0
    assert report["classes_with_both_beneficial_and_harmful_clients_rate"] == 1.0


def test_d2b_scalar_and_class_composition_use_actual_fedavg_mass_response():
    baseline = torch.zeros(1, 3)
    responses = torch.tensor([
        [[1.0, 2.0, 3.0]],
        [[0.5, 1.0, 1.5]],
    ])
    fedavg = torch.tensor([0.5, 0.5])
    scalar = compose_scalar_logits(
        baseline, responses, fedavg, torch.tensor([1.0, 0.0])
    )
    classwise = compose_class_logits(
        baseline,
        responses,
        fedavg,
        torch.tensor([[1.0], [0.0]]),
        tail=[2],
    )

    assert torch.allclose(scalar, torch.tensor([[0.5, 1.0, 1.5]]))
    assert torch.allclose(classwise, torch.tensor([[0.0, 0.0, 1.5]]))


def test_d2b_vector_reconstruction_supports_scalar_and_per_class_weights():
    before = torch.tensor([10.0, 10.0], dtype=torch.float64)
    deltas = torch.tensor([[1.0, 0.0], [0.0, 2.0]], dtype=torch.float64)

    assert torch.equal(
        vector_from_weights(before, deltas, torch.tensor([0.25, 0.75])),
        torch.tensor([10.25, 11.5], dtype=torch.float64),
    )


def test_d2b_weight_optimization_returns_normalized_scalar_and_class_distributions():
    labels = torch.tensor([0, 1, 2] * 4)
    baseline = torch.zeros(len(labels), 3)
    responses = torch.randn(2, len(labels), 3, generator=torch.Generator().manual_seed(7))
    scalar, classwise, trace = optimize_weight_distributions(
        baseline,
        responses,
        labels,
        torch.tensor([0.4, 0.6]),
        tail=[2],
        steps=2,
        samples_per_class=2,
        learning_rate=0.01,
        kl_weight=1e-3,
        seed=42,
        device=torch.device("cpu"),
    )

    assert scalar.shape == (2,)
    assert classwise.shape == (2, 1)
    assert scalar.sum() == pytest.approx(1.0)
    assert classwise[:, 0].sum() == pytest.approx(1.0)
    assert trace[-1]["step"] == 2


def test_d23_launcher_exposes_d2b_without_new_training(tmp_path):
    args = SimpleNamespace(
        python_bin="python",
        output_root=tmp_path,
        eval_batch_size=128,
    )
    command = analyzer_command(args, "d2b", tmp_path / "dump_seed42")

    assert command[2] == "scripts/analyze_d2b_scalar_ceiling.py"
    assert "--d2-utility" in command
    assert str(tmp_path / "d2" / "d2_client_class_utility.csv") in command


def test_p0_recovers_full_endpoint_from_frozen_gamma_mixture():
    base = torch.tensor([0.25, 0.75])
    alternative = torch.tensor([0.75, 0.25])
    selected = weights_at_gamma(base, alternative, 0.4)

    recovered = recover_alternative_weights(base, selected, 0.4)

    assert torch.allclose(recovered, alternative.double())


def test_p0_budget_selection_maximizes_harmonic_under_head_constraint():
    rows = [
        {
            "communication_round": 20,
            "method": "scalar",
            "gamma": 0.0,
            "tau": 1.0,
            "head_accuracy": 70.0,
            "tail_accuracy": 60.0,
            "balanced_accuracy": 68.0,
            "head_tail_harmonic": 64.0,
        },
        {
            "communication_round": 20,
            "method": "scalar",
            "gamma": 0.5,
            "tau": 1.0,
            "head_accuracy": 69.4,
            "tail_accuracy": 66.0,
            "balanced_accuracy": 68.5,
            "head_tail_harmonic": 67.0,
        },
    ]

    choices = select_budget_choices(rows, reference_head=70.0, budgets=[0.5, 1.0])

    assert choices[0]["gamma"] == 0.0
    assert choices[1]["gamma"] == 0.5


def test_p0_direct_matching_and_frontier_do_not_use_unmatched_head_points():
    scalar = [
        {
            "communication_round": 20,
            "method": "scalar",
            "gamma": 0.0,
            "tau": 0.0,
            "head_accuracy": 70.0,
            "tail_accuracy": 50.0,
            "balanced_accuracy": 66.0,
            "head_tail_harmonic": 58.0,
        },
        {
            "communication_round": 20,
            "method": "scalar",
            "gamma": 1.0,
            "tau": 0.0,
            "head_accuracy": 69.8,
            "tail_accuracy": 60.0,
            "balanced_accuracy": 68.0,
            "head_tail_harmonic": 64.0,
        },
    ]
    classwise = [
        {
            "communication_round": 20,
            "method": "class_conditional",
            "gamma": 0.5,
            "tau": 0.0,
            "head_accuracy": 69.9,
            "tail_accuracy": 62.0,
            "balanced_accuracy": 68.5,
            "head_tail_harmonic": 65.0,
        },
        {
            "communication_round": 20,
            "method": "class_conditional",
            "gamma": 1.0,
            "tau": 0.0,
            "head_accuracy": 68.0,
            "tail_accuracy": 70.0,
            "balanced_accuracy": 68.4,
            "head_tail_harmonic": 68.9,
        },
    ]

    matches = match_class_to_scalar(classwise, scalar, tolerance=0.25)
    frontier = pareto_frontier(scalar, "head_tail_harmonic")

    assert matches[0]["matched"] is True
    assert matches[0]["scalar_gamma"] == 1.0
    assert matches[1]["matched"] is False
    assert {row["gamma"] for row in frontier} == {0.0, 1.0}


def test_d23_launcher_exposes_p0_without_new_training(tmp_path):
    args = SimpleNamespace(
        python_bin="python",
        output_root=tmp_path,
        eval_batch_size=128,
    )
    command = analyzer_command(args, "p0", tmp_path / "dump_seed42")

    assert command[2] == "scripts/analyze_p0_head_pareto.py"
    assert "--d2b-dir" in command
    assert str(tmp_path / "d2b") in command


def test_p0_candidate_lookup_is_scoped_by_communication_round():
    rows = [
        {
            "communication_round": round_id,
            "method": "scalar",
            "gamma": 0.6,
            "tau": 1.0,
            "head_accuracy": float(round_id),
        }
        for round_id in (20, 50, 80)
    ]

    selected = _lookup(rows, 50, "scalar", 0.6, 1.0)

    assert selected["communication_round"] == 50
    assert selected["head_accuracy"] == 50.0


def test_p0_envelope_auc_keeps_exact_upper_endpoint_feasible():
    rows = [
        {"head_accuracy": 45.03750011557713, "tail_accuracy": 80.0},
        {"head_accuracy": 73.8000002130866, "tail_accuracy": 60.0},
    ]

    area = envelope_auc(
        rows,
        "tail_accuracy",
        x_low=45.03750011557713,
        x_high=73.8000002130866,
    )

    assert math.isfinite(area)
    assert area > 0.0
