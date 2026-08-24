import copy

import pytest
import torch
from torch import nn

from utils.stage3_vectors import (
    EPS_NORM,
    NORM_RELATIVE_TOLERANCE,
    LoRAFlatSpec,
    build_model_lora_flat_spec,
    compose_fixed_norm_upload,
    extract_lora_state,
    flatten_model,
    flatten_state,
    load_lora_vector,
    make_flat_spec,
    roundtrip_vector,
    unflatten_vector,
)


class ToyVisionLoRA(nn.Module):
    def __init__(self):
        super().__init__()
        self.image_encoder = nn.Module()
        # Intentionally register v before q; the spec must sort full names.
        self.image_encoder.v_lora_B = nn.Parameter(torch.arange(6.0).reshape(2, 3))
        self.image_encoder.q_lora_A = nn.Parameter(torch.tensor([[1.0, 2.0]]))
        self.image_encoder.frozen_weight = nn.Parameter(
            torch.tensor([9.0]), requires_grad=False
        )


def test_model_flat_spec_is_sorted_visual_lora_only_and_hash_roundtrips():
    model = ToyVisionLoRA()
    spec = build_model_lora_flat_spec(model)

    assert spec.names == (
        "image_encoder.q_lora_A",
        "image_encoder.v_lora_B",
    )
    assert [entry.offset for entry in spec.entries] == [0, 2]
    assert [entry.numel for entry in spec.entries] == [2, 6]
    assert spec.numel == 8
    assert LoRAFlatSpec.from_dict(spec.as_dict()) == spec

    tampered = copy.deepcopy(spec.as_dict())
    tampered["entries"][0]["shape"] = [2, 1]
    with pytest.raises(ValueError, match="hash mismatch"):
        LoRAFlatSpec.from_dict(tampered)


def test_model_flat_spec_rejects_any_non_visual_trainable_parameter():
    model = ToyVisionLoRA()
    model.text_adapter = nn.Parameter(torch.ones(1))

    with pytest.raises(ValueError, match="vision-LoRA-only"):
        build_model_lora_flat_spec(model)


def test_flatten_unflatten_and_model_load_use_one_spec():
    model = ToyVisionLoRA()
    spec = build_model_lora_flat_spec(model)
    original = flatten_model(model, spec)
    state = extract_lora_state(model, spec)

    restored = unflatten_vector(original, spec, like=state)
    assert torch.equal(flatten_state(restored, spec), original)

    replacement = original + torch.arange(spec.numel, dtype=torch.float32) / 10
    load_lora_vector(model, replacement, spec)
    assert torch.equal(flatten_model(model, spec), replacement)
    assert model.image_encoder.frozen_weight.item() == 9.0


def test_flatten_boundary_restores_original_mixed_dtypes():
    state = {
        "image_encoder.q_lora_A": torch.tensor([1.0, 2.0], dtype=torch.float16),
        "image_encoder.v_lora_B": torch.tensor([3.0, 4.0], dtype=torch.float32),
    }
    spec = make_flat_spec(state)
    vector = torch.tensor([1.25, 2.25, 3.25, 4.25], dtype=torch.float32)
    restored = unflatten_vector(vector, spec, like=state)

    assert restored["image_encoder.q_lora_A"].dtype == torch.float16
    assert restored["image_encoder.v_lora_B"].dtype == torch.float32
    assert torch.equal(roundtrip_vector(vector, spec, like=state), flatten_state(restored, spec))


def test_fixed_norm_composition_rotates_without_changing_budget():
    ce = torch.tensor([3.0, 4.0, 0.0, 0.0])
    fcc = torch.tensor([0.0, 0.0, 1.0, 0.0])
    rtc = torch.tensor([0.0, 0.0, 0.0, 1.0])
    state = {
        "image_encoder.q_lora_A": torch.zeros(2),
        "image_encoder.v_lora_B": torch.zeros(2),
    }
    spec = make_flat_spec(state)

    upload, report = compose_fixed_norm_upload(
        ce,
        fcc_direction=fcc,
        rtc_direction=rtc,
        lambda_fcc=0.5,
        lambda_rtc=0.5,
        degradation=0.4,
        spec=spec,
        like=state,
    )

    assert not torch.equal(upload, ce)
    assert upload.norm().item() == pytest.approx(ce.norm().item(), rel=1e-6)
    assert report["relative_norm_error"] < NORM_RELATIVE_TOLERANCE
    assert report["fcc_active"] is True
    assert report["rtc_active"] is True


def test_zero_ce_and_near_zero_composition_have_frozen_fallbacks():
    zero, zero_report = compose_fixed_norm_upload(
        torch.zeros(3),
        fcc_direction=torch.ones(3),
        lambda_fcc=0.5,
    )
    assert torch.equal(zero, torch.zeros(3))
    assert zero_report["fallback"] == "zero_ce_delta"
    assert zero_report["relative_norm_error"] == 0.0

    ce = torch.tensor([1.0, 0.0])
    upload, report = compose_fixed_norm_upload(
        ce,
        fcc_direction=-ce,
        lambda_fcc=1.0,
    )
    assert torch.equal(upload, ce)
    assert report["fallback"] == "near_zero_composition_to_ce"


def test_opposite_fcc_and_rtc_corrections_do_not_change_ce_direction():
    ce = torch.tensor([2.0, 0.0, 0.0])
    fcc = torch.tensor([0.0, 1.0, 0.0])
    rtc = -fcc
    upload, report = compose_fixed_norm_upload(
        ce,
        fcc_direction=fcc,
        rtc_direction=rtc,
        lambda_fcc=0.5,
        lambda_rtc=0.5,
        degradation=1.0,
    )

    assert torch.allclose(upload, ce)
    assert report["norm_gate_pass"] is True


def test_mixed_precision_reports_actual_post_roundtrip_norm():
    generator = torch.Generator().manual_seed(42)
    ce = torch.randn(8192, generator=generator)
    direction = torch.randn(8192, generator=generator)
    state = {
        "image_encoder.q_lora_A": torch.zeros(4096, dtype=torch.float16),
        "image_encoder.v_lora_B": torch.zeros(4096, dtype=torch.float32),
    }
    spec = make_flat_spec(state)

    upload, report = compose_fixed_norm_upload(
        ce,
        fcc_direction=direction,
        lambda_fcc=0.5,
        spec=spec,
        like=state,
        enforce_tolerance=False,
    )
    actual = float(upload.float().norm().item())
    expected_error = abs(actual - float(ce.norm().item())) / max(
        float(ce.norm().item()), EPS_NORM
    )
    assert report["actual_upload_norm"] == pytest.approx(actual)
    assert report["relative_norm_error"] == pytest.approx(expected_error)
    assert report["dtype_roundtrips"] >= 1


def test_norm_gate_measures_the_actual_anchor_plus_upload_state():
    state = {
        "image_encoder.q_lora_A": torch.full(
            (4096,), 2.0, dtype=torch.float16
        ),
        "image_encoder.v_lora_B": torch.full(
            (4096,), -3.0, dtype=torch.float32
        ),
    }
    spec = make_flat_spec(state)
    anchor = flatten_state(state, spec)
    generator = torch.Generator().manual_seed(7)
    ce = torch.randn(spec.numel, generator=generator) * 0.01
    fcc = torch.randn(spec.numel, generator=generator)
    upload, report = compose_fixed_norm_upload(
        ce,
        fcc_direction=fcc,
        lambda_fcc=0.5,
        spec=spec,
        like=state,
        anchor_vector=anchor,
        enforce_tolerance=False,
    )
    final_state = unflatten_vector(anchor + upload, spec, like=state)
    actual = flatten_state(final_state, spec) - flatten_state(state, spec)
    assert torch.equal(upload, actual)
    assert report["actual_upload_norm"] == pytest.approx(actual.norm().item())


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"fcc_direction": torch.tensor([float("nan"), 0.0])}, "NaN or Inf"),
        ({"degradation": 1.1}, r"\[0, 1\]"),
        ({"lambda_fcc": -0.5}, "non-negative"),
    ],
)
def test_composition_rejects_invalid_numerical_inputs(kwargs, message):
    with pytest.raises(ValueError, match=message):
        compose_fixed_norm_upload(torch.tensor([1.0, 0.0]), **kwargs)
