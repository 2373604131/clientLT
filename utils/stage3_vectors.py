"""Stage-3 vector primitives for P-FCC and D-RTC.

This module implements only the frozen v1.0.1 substrate shared by all five
conditions.  It does not select proposals, compute restore gradients, or
change the existing FedAvg path.  Those operations are intentionally kept out
of batch 1 so a zero-coefficient run can later be checked against the original
ClipLora implementation.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Mapping, Sequence

import torch


STAGE3_VECTOR_SCHEMA = "p_fcc_d_rtc_flat_spec_v1"
EPS_NORM = 1e-12
NORM_RELATIVE_TOLERANCE = 1e-6


def _canonical_json(value) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _sha256_json(value) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _dtype_from_name(name: str) -> torch.dtype:
    short_name = str(name).rsplit(".", 1)[-1]
    dtype = getattr(torch, short_name, None)
    if not isinstance(dtype, torch.dtype):
        raise ValueError(f"Unsupported torch dtype in flatten spec: {name!r}")
    return dtype


def _is_visual_lora_name(name: str) -> bool:
    parts = str(name).split(".")
    return "image_encoder" in parts and "lora_" in str(name)


@dataclass(frozen=True)
class FlatEntry:
    """One tensor segment in the unique Stage-3 LoRA vector."""

    name: str
    shape: tuple[int, ...]
    numel: int
    dtype: str
    offset: int

    @property
    def end(self) -> int:
        return int(self.offset + self.numel)

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "shape": list(self.shape),
            "numel": int(self.numel),
            "dtype": self.dtype,
            "offset": int(self.offset),
        }

    @classmethod
    def from_dict(cls, value: Mapping) -> "FlatEntry":
        return cls(
            name=str(value["name"]),
            shape=tuple(int(item) for item in value["shape"]),
            numel=int(value["numel"]),
            dtype=str(value["dtype"]),
            offset=int(value["offset"]),
        )


@dataclass(frozen=True)
class LoRAFlatSpec:
    """Frozen sorted specification for shared trainable vision-LoRA tensors."""

    entries: tuple[FlatEntry, ...]
    spec_hash: str
    schema_version: str = STAGE3_VECTOR_SCHEMA

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(entry.name for entry in self.entries)

    @property
    def numel(self) -> int:
        return self.entries[-1].end if self.entries else 0

    def as_dict(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "entries": [entry.as_dict() for entry in self.entries],
            "numel": int(self.numel),
            "spec_hash": self.spec_hash,
        }

    @classmethod
    def from_dict(cls, value: Mapping) -> "LoRAFlatSpec":
        schema = str(value.get("schema_version", ""))
        if schema != STAGE3_VECTOR_SCHEMA:
            raise ValueError(
                f"Unsupported Stage-3 flatten schema {schema!r}; "
                f"expected {STAGE3_VECTOR_SCHEMA!r}"
            )
        entries = tuple(FlatEntry.from_dict(item) for item in value["entries"])
        spec = _make_spec(entries)
        supplied_hash = str(value.get("spec_hash", ""))
        if supplied_hash != spec.spec_hash:
            raise ValueError(
                "Flatten spec hash mismatch: "
                f"checkpoint={supplied_hash!r}, computed={spec.spec_hash!r}"
            )
        if int(value.get("numel", spec.numel)) != spec.numel:
            raise ValueError("Flatten spec numel does not match its entries")
        return spec


def _make_spec(entries: Sequence[FlatEntry]) -> LoRAFlatSpec:
    entries = tuple(entries)
    names = [entry.name for entry in entries]
    if names != sorted(names):
        raise ValueError("Flatten spec entries must use sorted full parameter names")
    if len(set(names)) != len(names):
        raise ValueError("Flatten spec contains duplicate parameter names")

    cursor = 0
    for entry in entries:
        if entry.offset != cursor:
            raise ValueError(
                f"Non-contiguous flatten offset for {entry.name!r}: "
                f"got {entry.offset}, expected {cursor}"
            )
        expected_numel = math.prod(entry.shape)
        if entry.numel != expected_numel:
            raise ValueError(
                f"Incorrect numel for {entry.name!r}: "
                f"got {entry.numel}, expected {expected_numel}"
            )
        if not torch.empty((), dtype=_dtype_from_name(entry.dtype)).is_floating_point():
            raise ValueError(f"Non-floating tensor in LoRA flatten spec: {entry.name!r}")
        cursor = entry.end

    hash_payload = {
        "schema_version": STAGE3_VECTOR_SCHEMA,
        "entries": [entry.as_dict() for entry in entries],
    }
    return LoRAFlatSpec(entries=entries, spec_hash=_sha256_json(hash_payload))


def make_flat_spec(
    named_tensors: Mapping[str, torch.Tensor] | Sequence[tuple[str, torch.Tensor]],
    *,
    require_visual_lora: bool = True,
) -> LoRAFlatSpec:
    """Create a sorted spec from a mapping or named tensor sequence.

    Production callers should keep ``require_visual_lora=True``.  The relaxed
    form exists for small synthetic correctness tests only.
    """
    items = list(named_tensors.items()) if isinstance(named_tensors, Mapping) else list(named_tensors)
    items.sort(key=lambda item: item[0])
    if not items:
        raise ValueError("Cannot create an empty Stage-3 flatten spec")

    entries = []
    cursor = 0
    for name, tensor in items:
        name = str(name)
        if require_visual_lora and not _is_visual_lora_name(name):
            raise ValueError(
                "Stage-3 flatten spec may contain only shared trainable "
                f"vision-LoRA tensors; got {name!r}"
            )
        if not torch.is_tensor(tensor) or not torch.is_floating_point(tensor):
            raise ValueError(f"Flatten tensor {name!r} must be floating point")
        numel = int(tensor.numel())
        entries.append(
            FlatEntry(
                name=name,
                shape=tuple(int(item) for item in tensor.shape),
                numel=numel,
                dtype=str(tensor.dtype),
                offset=cursor,
            )
        )
        cursor += numel
    return _make_spec(entries)


def build_model_lora_flat_spec(model) -> LoRAFlatSpec:
    """Freeze the unique trainable vision-LoRA spec exposed by ``model``."""
    trainable = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]
    if not trainable:
        raise ValueError("Model exposes no trainable parameters")
    invalid = [name for name, _ in trainable if not _is_visual_lora_name(name)]
    if invalid:
        raise ValueError(
            "Stage-3 requires vision-LoRA-only trainable parameters; "
            f"unexpected trainable names: {sorted(invalid)}"
        )
    return make_flat_spec(trainable, require_visual_lora=True)


def validate_mapping(
    state: Mapping[str, torch.Tensor],
    spec: LoRAFlatSpec,
    *,
    require_dtype: bool = True,
) -> None:
    """Validate all required tensors while allowing unrelated frozen entries."""
    for entry in spec.entries:
        if entry.name not in state:
            raise KeyError(f"State is missing Stage-3 tensor {entry.name!r}")
        tensor = state[entry.name]
        if tuple(tensor.shape) != entry.shape:
            raise ValueError(
                f"Shape mismatch for {entry.name!r}: "
                f"got {tuple(tensor.shape)}, expected {entry.shape}"
            )
        if require_dtype and str(tensor.dtype) != entry.dtype:
            raise ValueError(
                f"Dtype mismatch for {entry.name!r}: "
                f"got {tensor.dtype}, expected {entry.dtype}"
            )


def flatten_state(
    state: Mapping[str, torch.Tensor],
    spec: LoRAFlatSpec,
    *,
    device: torch.device | str | None = None,
    require_dtype: bool = True,
) -> torch.Tensor:
    """Flatten a state in spec order using an FP32 numerical substrate."""
    validate_mapping(state, spec, require_dtype=require_dtype)
    chunks = [
        state[entry.name].detach().to(device=device, dtype=torch.float32).reshape(-1)
        for entry in spec.entries
    ]
    vector = torch.cat(chunks)
    if vector.numel() != spec.numel:
        raise RuntimeError("Flattened vector length does not match frozen spec")
    return vector


def flatten_model(model, spec: LoRAFlatSpec, *, device=None) -> torch.Tensor:
    return flatten_state(dict(model.named_parameters()), spec, device=device)


def unflatten_vector(
    vector: torch.Tensor,
    spec: LoRAFlatSpec,
    *,
    like: Mapping[str, torch.Tensor] | None = None,
    device: torch.device | str | None = None,
) -> dict[str, torch.Tensor]:
    """Restore a vector to the original tensor dtypes and shapes."""
    flat = torch.as_tensor(vector, dtype=torch.float32).reshape(-1)
    if flat.numel() != spec.numel:
        raise ValueError(
            f"Vector has {flat.numel()} elements, expected {spec.numel}"
        )
    if like is not None:
        validate_mapping(like, spec, require_dtype=True)
    result = {}
    for entry in spec.entries:
        target_dtype = _dtype_from_name(entry.dtype)
        target_device = device
        if like is not None:
            if entry.name not in like:
                raise KeyError(f"Reference mapping is missing {entry.name!r}")
            target_dtype = like[entry.name].dtype
            target_device = like[entry.name].device
        result[entry.name] = (
            flat[entry.offset : entry.end]
            .reshape(entry.shape)
            .to(device=target_device, dtype=target_dtype)
            .clone()
        )
    return result


def roundtrip_vector(
    vector: torch.Tensor,
    spec: LoRAFlatSpec,
    *,
    like: Mapping[str, torch.Tensor] | None = None,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Apply the model-dtype unflatten/flatten boundary required by v1.0.1."""
    restored = unflatten_vector(vector, spec, like=like, device=device)
    output_device = torch.as_tensor(vector).device
    return flatten_state(restored, spec, device=output_device)


def extract_lora_state(model, spec: LoRAFlatSpec) -> dict[str, torch.Tensor]:
    parameters = dict(model.named_parameters())
    validate_mapping(parameters, spec)
    return {
        entry.name: parameters[entry.name].detach().cpu().clone()
        for entry in spec.entries
    }


def load_lora_vector(model, vector: torch.Tensor, spec: LoRAFlatSpec) -> None:
    """Load exactly the shared trainable vision-LoRA tensors into a model."""
    parameters = dict(model.named_parameters())
    state = unflatten_vector(vector, spec, like=parameters)
    with torch.no_grad():
        for entry in spec.entries:
            parameters[entry.name].copy_(state[entry.name])


def lora_delta_vector(
    local_state: Mapping[str, torch.Tensor],
    incoming_state: Mapping[str, torch.Tensor],
    spec: LoRAFlatSpec,
    *,
    device=None,
) -> torch.Tensor:
    return flatten_state(local_state, spec, device=device) - flatten_state(
        incoming_state, spec, device=device
    )


def _finite_vector(value, *, name: str, numel: int, device) -> torch.Tensor:
    vector = torch.as_tensor(value, dtype=torch.float32, device=device).reshape(-1)
    if vector.numel() != numel:
        raise ValueError(f"{name} has {vector.numel()} values, expected {numel}")
    if not bool(torch.isfinite(vector).all()):
        raise ValueError(f"{name} contains NaN or Inf")
    return vector


def _scaled_direction(direction: torch.Tensor | None, target_norm: torch.Tensor) -> tuple[torch.Tensor, bool]:
    if direction is None:
        return torch.zeros((), device=target_norm.device), False
    norm = torch.linalg.vector_norm(direction)
    if float(norm.item()) <= EPS_NORM:
        return torch.zeros_like(direction), False
    return direction * (target_norm / norm), True


def relative_norm_error(actual: float, target: float) -> float:
    if actual == 0.0 and target == 0.0:
        return 0.0
    return abs(float(actual) - float(target)) / max(abs(float(target)), EPS_NORM)


class NormBudgetError(RuntimeError):
    """Raised when the model-dtype upload violates the frozen norm Gate."""


def compose_fixed_norm_upload(
    ce_delta,
    *,
    fcc_direction=None,
    rtc_direction=None,
    lambda_fcc: float = 0.0,
    lambda_rtc: float = 0.0,
    degradation: float = 0.0,
    spec: LoRAFlatSpec | None = None,
    like: Mapping[str, torch.Tensor] | None = None,
    anchor_vector=None,
    enforce_tolerance: bool = True,
    max_dtype_roundtrips: int = 4,
) -> tuple[torch.Tensor, dict]:
    """Compose P-FCC/D-RTC directions without increasing the CE norm.

    All arithmetic is FP32.  When ``spec`` is supplied, the candidate is cast
    through the actual model tensor dtypes and flattened again before the norm
    Gate is evaluated.
    """
    ce = torch.as_tensor(ce_delta, dtype=torch.float32).reshape(-1)
    device = ce.device
    if ce.numel() == 0:
        raise ValueError("ce_delta must be non-empty")
    ce = _finite_vector(ce, name="ce_delta", numel=ce.numel(), device=device)
    if spec is not None and ce.numel() != spec.numel:
        raise ValueError(
            f"ce_delta has {ce.numel()} values but spec requires {spec.numel}"
        )
    if spec is None and like is not None:
        raise ValueError("like requires a flatten spec")
    anchor = None
    if anchor_vector is not None:
        if spec is None:
            raise ValueError("anchor_vector requires a flatten spec")
        anchor = _finite_vector(
            anchor_vector,
            name="anchor_vector",
            numel=ce.numel(),
            device=device,
        )

    for name, value in (("lambda_fcc", lambda_fcc), ("lambda_rtc", lambda_rtc)):
        if not math.isfinite(float(value)) or float(value) < 0:
            raise ValueError(f"{name} must be finite and non-negative")
    if not math.isfinite(float(degradation)) or not 0.0 <= float(degradation) <= 1.0:
        raise ValueError("degradation must lie in [0, 1]")

    fcc = None if fcc_direction is None else _finite_vector(
        fcc_direction, name="fcc_direction", numel=ce.numel(), device=device
    )
    rtc = None if rtc_direction is None else _finite_vector(
        rtc_direction, name="rtc_direction", numel=ce.numel(), device=device
    )

    target_norm_tensor = torch.linalg.vector_norm(ce)
    target_norm = float(target_norm_tensor.item())
    fallback = "none"
    fcc_active = False
    rtc_active = False

    if target_norm <= EPS_NORM:
        candidate = torch.zeros_like(ce)
        fallback = "zero_ce_delta"
        z_norm = 0.0
    else:
        fcc_scaled, fcc_active = _scaled_direction(fcc, target_norm_tensor)
        rtc_scaled, rtc_active = _scaled_direction(rtc, target_norm_tensor)
        if fcc is None:
            fcc_scaled = torch.zeros_like(ce)
        if rtc is None:
            rtc_scaled = torch.zeros_like(ce)
        z = (
            ce
            + float(lambda_fcc) * fcc_scaled
            + float(lambda_rtc) * float(degradation) * rtc_scaled
        )
        z_norm_tensor = torch.linalg.vector_norm(z)
        z_norm = float(z_norm_tensor.item())
        if z_norm <= EPS_NORM:
            candidate = ce.clone()
            fallback = "near_zero_composition_to_ce"
        else:
            candidate = z * (target_norm_tensor / z_norm_tensor)

    dtype_roundtrips = 0
    if spec is not None:
        # A second scaling pass compensates ordinary FP32 casting roundoff.  A
        # genuinely coarse dtype may still be unable to represent the exact
        # budget; that is surfaced by the formal Gate instead of hidden.
        roundtrip_budget = max(1, int(max_dtype_roundtrips))
        for attempt in range(roundtrip_budget):
            if anchor is None:
                candidate = roundtrip_vector(candidate, spec, like=like)
            else:
                rounded_anchor = roundtrip_vector(anchor, spec, like=like)
                rounded_final = roundtrip_vector(anchor + candidate, spec, like=like)
                candidate = rounded_final - rounded_anchor
            dtype_roundtrips += 1
            actual_tensor = torch.linalg.vector_norm(candidate)
            actual = float(actual_tensor.item())
            error = relative_norm_error(actual, target_norm)
            if error < NORM_RELATIVE_TOLERANCE or target_norm <= EPS_NORM:
                break
            if actual <= EPS_NORM:
                break
            # Never leave the loop with an unrepresentable, freshly rescaled
            # vector.  On the last attempt the model-dtype value is the
            # authoritative upload and the formal norm Gate decides whether
            # that representation is sufficiently accurate.
            if attempt + 1 < roundtrip_budget:
                candidate = candidate * (target_norm / actual)

    actual_norm = float(torch.linalg.vector_norm(candidate).item())
    error = relative_norm_error(actual_norm, target_norm)
    report = {
        "target_ce_norm": target_norm,
        "composition_norm_before_final_scaling": z_norm,
        "actual_upload_norm": actual_norm,
        "relative_norm_error": error,
        "norm_gate_pass": bool(error < NORM_RELATIVE_TOLERANCE),
        "fallback": fallback,
        "fcc_active": bool(fcc_active and float(lambda_fcc) > 0),
        "rtc_active": bool(
            rtc_active and float(lambda_rtc) > 0 and float(degradation) > 0
        ),
        "degradation": float(degradation),
        "dtype_roundtrips": int(dtype_roundtrips),
    }
    if enforce_tolerance and not report["norm_gate_pass"]:
        raise NormBudgetError(
            "Stage-3 upload norm mismatch after model-dtype roundtrip: "
            f"relative_error={error:.9g}, tolerance={NORM_RELATIVE_TOLERANCE}"
        )
    return candidate, report
