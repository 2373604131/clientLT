import argparse

import torch

from scripts.run_stage2c_temporal import build_stage2c_command
from utils.stage2c_temporal import (
    Stage2CTemporalDiagnostic,
    combine_model_state,
    diagnose_route,
    parse_stage2c_rounds,
    split_model_state,
)


def test_parse_stage2c_rounds_is_one_based_unique_and_bounded():
    assert parse_stage2c_rounds("20,3,20,80", 80) == [3, 20, 80]
    try:
        parse_stage2c_rounds("0,3", 80)
    except ValueError:
        pass
    else:
        raise AssertionError("round zero must be rejected")


def test_split_recombine_and_zero_residual_are_exact():
    state = {
        "fixed": torch.tensor([7.0]),
        "image_encoder.block.lora_A": torch.tensor([2.0]),
        "class_residual.weight": torch.tensor([[3.0, 4.0]]),
    }
    fixed, shared, residual = split_model_state(
        state,
        ["image_encoder.block.lora_A"],
        ["class_residual.weight"],
    )
    rebuilt = combine_model_state(fixed, shared, residual)
    assert set(rebuilt) == set(state)
    assert all(torch.equal(rebuilt[key], state[key]) for key in state)
    zero = combine_model_state(fixed, shared, residual, zero_residual=True)
    assert torch.equal(zero["fixed"], state["fixed"])
    assert torch.equal(zero["image_encoder.block.lora_A"], state["image_encoder.block.lora_A"])
    assert torch.count_nonzero(zero["class_residual.weight"]).item() == 0


def _route_rows(values):
    rows = []
    for shared_round in (3, 80):
        for residual_round in (3, 80, "zero"):
            rows.append({
                "shared_round": shared_round,
                "residual_round": str(residual_round),
                "head_tail_h_mean": values[(shared_round, residual_round)],
            })
    return rows


def test_route_distinguishes_alignment_residual_and_shared_failures():
    alignment = diagnose_route(_route_rows({
        (3, 3): 70, (3, 80): 69, (3, "zero"): 60,
        (80, 3): 60, (80, 80): 68, (80, "zero"): 60,
    }), 3, 80, substantive_drop=2)
    assert alignment["recommended_route"] == "temporal_residual_transport_alignment"

    residual = diagnose_route(_route_rows({
        (3, 3): 70, (3, 80): 60, (3, "zero"): 60,
        (80, 3): 68, (80, 80): 69, (80, "zero"): 60,
    }), 3, 80, substantive_drop=2)
    assert residual["recommended_route"] == "residual_aggregation_or_stability"

    shared = diagnose_route(_route_rows({
        (3, 3): 70, (3, 80): 69, (3, "zero"): 65,
        (80, 3): 66, (80, 80): 67, (80, "zero"): 55,
    }), 3, 80, substantive_drop=2)
    assert shared["recommended_route"] == "shared_substrate_degradation"


def test_aggregation_stage_export_has_exact_additive_decomposition(tmp_path):
    runtime = Stage2CTemporalDiagnostic(
        output_dir=tmp_path,
        checkpoint_rounds=[3],
        shared_keys=["x.lora_A"],
        residual_keys=["class_residual.weight"],
        class_counts=[10, 1],
        tail_ratio=0.5,
        protocol={},
    )
    margins = {0: 1.0, 1: -1.0}
    runtime.record_stage(3, "pre_aggregation", [50, 50, 40, {0: 80, 1: 20}], margins)
    runtime.record_stage(3, "after_shared", [55, 45, 45, {0: 82, 1: 28}], margins)
    runtime.record_stage(3, "after_full", [60, 40, 50, {0: 81, 1: 39}], margins)
    summary, per_class = runtime._write_stage_outputs()
    assert abs(summary[0]["decomposition_error_tail_acc"]) < 1e-12
    assert all(abs(row["decomposition_error_accuracy"]) < 1e-12 for row in per_class)


def test_launcher_freezes_exact_sca_diagnostic_flags(tmp_path):
    args = argparse.Namespace(
        python_bin="python",
        data_root=tmp_path / "DATA",
        output_root=tmp_path / "output",
        frac=0.4,
        rounds=80,
        lr=0.001,
        test_batch_size=256,
        eval_interval=1,
        matched_beta=0.5,
        num_workers=8,
        sca_scale=10.0,
        sca_clamp=3.0,
        sca_lr_mult=5.0,
        support_min_fraction=0.0,
        support_weighting="class_count",
        checkpoint_rounds="3,20,50,80",
        substantive_drop=2.0,
    )
    output_dir, command = build_stage2c_command(
        args, {"position": "top3", "rank": 2, "alpha": 1, "params": ["q", "v"]}
    )
    assert output_dir.name == "stage2c_temporal_clientlt"
    pairs = dict(zip(command, command[1:]))
    assert pairs["--partition"] == "client-longtail"
    assert pairs["--cliplora_residual_aggregation"] == "class_separable"
    assert pairs["--cliplora_stage2c_enable"] == "True"
    assert pairs["--cliplora_stage2c_rounds"] == "3,20,50,80"


def test_full_cross_swap_runtime_writes_all_combinations(tmp_path):
    class TinyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("fixed", torch.tensor([1.0]))
            self.register_parameter("x_lora_A", torch.nn.Parameter(torch.tensor([0.0])))
            self.register_parameter(
                "class_residual_weight", torch.nn.Parameter(torch.tensor([0.0]))
            )

    class TinyTrainer(object):
        def __init__(self):
            self.model = TinyModel()
            self.last_global_test_class_margins = {}

        def global_test(self, is_global=True, current_epoch=0):
            shared = float(self.model.x_lora_A.detach().item())
            residual = float(self.model.class_residual_weight.detach().item())
            per_class = {0: 50.0 + shared, 1: 20.0 + residual}
            self.last_global_test_class_margins = {0: shared, 1: residual}
            overall = sum(per_class.values()) / 2.0
            return [overall, 100.0 - overall, overall, per_class]

    trainer = TinyTrainer()
    runtime = Stage2CTemporalDiagnostic(
        output_dir=tmp_path,
        checkpoint_rounds=[3, 80],
        shared_keys=["x_lora_A"],
        residual_keys=["class_residual_weight"],
        class_counts=[10, 1],
        tail_ratio=0.5,
        protocol={},
    )
    for communication_round, shared, residual in ((3, 3.0, 10.0), (80, 8.0, 2.0)):
        state = trainer.model.state_dict()
        state["x_lora_A"] = torch.tensor([shared])
        state["class_residual_weight"] = torch.tensor([residual])
        runtime.save_checkpoint(communication_round, state)
        result = [40.0, 60.0, 40.0, {0: 60.0, 1: 20.0}]
        for stage in runtime.STAGES:
            runtime.record_stage(
                communication_round, stage, result, {0: 0.5, 1: -0.5}
            )
    payload = runtime.run_cross_swap(trainer, trainer.model.state_dict())
    assert payload["num_cross_swap_evaluations"] == 4
    assert payload["num_zero_residual_evaluations"] == 2
    assert (tmp_path / "stage2c" / "cross_swap_summary.csv").exists()
    assert (tmp_path / "stage2c" / "stage2c_summary.json").exists()
