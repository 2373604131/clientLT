"""V1 mode-representation stability primitives.

V1 is an offline experiment over V0 round dumps.  It never constructs CLIP,
uses labels, or touches the official test set.  Its only input is the matrix of
same-round client LoRA uploads already frozen by V0.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Mapping, Sequence

import torch

from utils.cusp_minimal import FlatSpec, flatten_state


V1_SCHEMA_VERSION = "v1_mode_stability_v1"
EPS = 1e-12


def stable_seed(*parts: object) -> int:
    text = json.dumps(parts, ensure_ascii=True, separators=(",", ":"), default=str)
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % (2**63 - 1)


def layer_name(parameter_name: str) -> str:
    """Map LoRA A/B tensors from one transformer block to a shared layer."""
    match = re.search(r"^(.*?\.resblocks\.\d+)(?:\.|$)", str(parameter_name))
    if match:
        return match.group(1)
    name = str(parameter_name)
    for suffix in (".w_lora_A", ".w_lora_B", ".lora_A", ".lora_B"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name.rsplit(".", 1)[0] if "." in name else name


def layer_segments(spec: FlatSpec) -> dict[str, tuple[tuple[int, int], ...]]:
    grouped: dict[str, list[tuple[int, int]]] = {}
    for key, offset in zip(spec.keys, spec.offsets):
        grouped.setdefault(layer_name(key), []).append((int(offset[0]), int(offset[1])))
    return {name: tuple(segments) for name, segments in sorted(grouped.items())}


def select_segments(matrix: torch.Tensor, segments: Sequence[tuple[int, int]]) -> torch.Tensor:
    chunks = [matrix[:, start:end] for start, end in segments]
    return torch.cat(chunks, dim=1) if chunks else matrix.new_empty((matrix.shape[0], 0))


@dataclass(frozen=True)
class UploadSet:
    source_id: str
    client_ids: tuple[int, ...]
    raw_deltas: torch.Tensor
    fedavg_weights: torch.Tensor
    spec: FlatSpec
    layers: Mapping[str, tuple[tuple[int, int], ...]]
    seed: int
    round_id: int
    partition: str


@dataclass(frozen=True)
class DisagreementSet:
    source_id: str
    client_ids: tuple[int, ...]
    matrix: torch.Tensor
    layers: Mapping[str, tuple[tuple[int, int], ...]]
    fedavg_norm: float
    orthogonality_error: float


@dataclass(frozen=True)
class ModeSet:
    directions: torch.Tensor
    singular_values: torch.Tensor
    labels: tuple[str, ...]

    @property
    def rank(self) -> int:
        return int(self.directions.shape[0])


def upload_set_from_payload(payload: Mapping, metadata: Mapping, source_id: str) -> UploadSet:
    spec = FlatSpec.from_dict(payload["flatten_spec"])
    before = flatten_state(payload["global_before_trainable"], spec)
    local = torch.stack(
        [flatten_state(state, spec) for state in payload["local_trainable_states"]], dim=0
    )
    deltas = local - before.unsqueeze(0)
    weights = torch.as_tensor(payload["fedavg_weights"], dtype=torch.float64).reshape(-1)
    client_ids = tuple(int(value) for value in payload["selected_client_ids"])
    if deltas.shape[0] != len(client_ids) or weights.numel() != len(client_ids):
        raise ValueError("V1 dump has inconsistent selected-client dimensions")
    if deltas.shape[1] != spec.numel:
        raise ValueError("V1 dump flattened dimension does not match its spec")
    if not bool(torch.isfinite(deltas).all()) or not bool(torch.isfinite(weights).all()):
        raise ValueError("V1 dump contains NaN or Inf")
    if bool((weights < 0).any()) or float(weights.sum().item()) <= EPS:
        raise ValueError("V1 dump has invalid FedAvg weights")
    weights = weights / weights.sum()
    resolved = metadata.get("resolved_args", {})
    return UploadSet(
        source_id=str(source_id),
        client_ids=client_ids,
        raw_deltas=deltas,
        fedavg_weights=weights,
        spec=spec,
        layers=layer_segments(spec),
        seed=int(resolved.get("seed", -1)),
        round_id=int(metadata.get("communication_round", -1)),
        partition=str(resolved.get("partition", "")),
    )


def build_disagreement_set(
    uploads: UploadSet,
    *,
    row_indices: Sequence[int] | None = None,
    weight_multipliers: torch.Tensor | None = None,
) -> DisagreementSet:
    """Reproduce the CMSA-compatible FedAvg-orthogonal disagreement matrix."""
    if row_indices is None:
        indices = torch.arange(len(uploads.client_ids), dtype=torch.long)
    else:
        indices = torch.as_tensor(list(row_indices), dtype=torch.long)
    if indices.numel() < 2:
        raise ValueError("V1 requires at least two retained clients")
    raw = uploads.raw_deltas[indices]
    weights = uploads.fedavg_weights[indices].clone()
    if weight_multipliers is not None:
        multipliers = torch.as_tensor(weight_multipliers, dtype=torch.float64).reshape(-1)
        if multipliers.numel() != indices.numel() or bool((multipliers <= 0).any()):
            raise ValueError("weight multipliers must be positive and match retained clients")
        weights *= multipliers
    weights /= weights.sum()
    fedavg = (weights[:, None] * raw).sum(dim=0)
    fedavg_norm = float(torch.linalg.vector_norm(fedavg).item())
    residual = raw - fedavg.unsqueeze(0)
    if fedavg_norm > EPS:
        unit = fedavg / fedavg_norm
        residual = residual - (residual @ unit).unsqueeze(1) * unit.unsqueeze(0)
        orth_error = float(torch.max(torch.abs(residual @ unit)).item())
    else:
        orth_error = math.nan
    matrix = torch.sqrt(weights).unsqueeze(1) * residual
    return DisagreementSet(
        source_id=uploads.source_id,
        client_ids=tuple(uploads.client_ids[int(index)] for index in indices.tolist()),
        matrix=matrix,
        layers=uploads.layers,
        fedavg_norm=fedavg_norm,
        orthogonality_error=orth_error,
    )


def _modes_from_features(
    feature_matrix: torch.Tensor,
    target_matrix: torch.Tensor,
    rank: int,
    *,
    label_prefix: str,
) -> ModeSet:
    features = torch.as_tensor(feature_matrix, dtype=torch.float64)
    target = torch.as_tensor(target_matrix, dtype=torch.float64)
    if features.ndim != 2 or target.ndim != 2 or features.shape[0] != target.shape[0]:
        raise ValueError("feature and target matrices must be 2D with matching rows")
    gram = features @ features.T
    eigenvalues, eigenvectors = torch.linalg.eigh(gram)
    order = torch.argsort(eigenvalues, descending=True)
    eigenvalues = torch.clamp(eigenvalues[order], min=0.0)
    eigenvectors = eigenvectors[:, order]
    if eigenvalues.numel() == 0:
        return ModeSet(target.new_empty((0, target.shape[1])), target.new_empty(0), ())
    tolerance = max(EPS, float(eigenvalues[0].item()) * 1e-10)
    effective = int((eigenvalues > tolerance).sum().item())
    kept = min(max(int(rank), 0), effective)
    if kept == 0:
        return ModeSet(target.new_empty((0, target.shape[1])), target.new_empty(0), ())
    coefficients = eigenvectors[:, :kept].T
    directions = coefficients @ target
    norms = torch.linalg.vector_norm(directions, dim=1)
    valid = norms > EPS
    directions = directions[valid] / norms[valid].unsqueeze(1)
    singular = torch.sqrt(eigenvalues[:kept])[valid]
    labels = tuple(f"{label_prefix}_{index}" for index in range(directions.shape[0]))
    return ModeSet(directions=directions, singular_values=singular, labels=labels)


def whole_client_modes(data: DisagreementSet) -> ModeSet:
    norms = torch.linalg.vector_norm(data.matrix, dim=1)
    valid = norms > EPS
    directions = data.matrix[valid] / norms[valid].unsqueeze(1)
    ids = [data.client_ids[index] for index in torch.nonzero(valid, as_tuple=False).reshape(-1).tolist()]
    return ModeSet(
        directions=directions,
        singular_values=norms[valid],
        labels=tuple(f"client_{client_id}" for client_id in ids),
    )


def svd_atom_modes(data: DisagreementSet, rank: int) -> ModeSet:
    return _modes_from_features(data.matrix, data.matrix, rank, label_prefix="atom")


def layerwise_modes(data: DisagreementSet, rank: int) -> dict[str, ModeSet]:
    result = {}
    for name, segments in data.layers.items():
        matrix = select_segments(data.matrix, segments)
        result[name] = _modes_from_features(matrix, matrix, rank, label_prefix="layer_atom")
    return result


def countsketch_features(
    data: DisagreementSet,
    *,
    sketch_dim: int,
    seed: int,
) -> torch.Tensor:
    if int(sketch_dim) < 1:
        raise ValueError("sketch_dim must be positive")
    sketches = []
    for layer_index, (name, segments) in enumerate(sorted(data.layers.items())):
        matrix = select_segments(data.matrix, segments)
        generator = torch.Generator(device="cpu")
        generator.manual_seed(stable_seed("v1-countsketch", seed, layer_index, name))
        buckets = torch.randint(int(sketch_dim), (matrix.shape[1],), generator=generator)
        signs = torch.randint(0, 2, (matrix.shape[1],), generator=generator, dtype=torch.int64)
        signs = signs.to(torch.float64).mul_(2.0).sub_(1.0)
        sketch = torch.zeros((matrix.shape[0], int(sketch_dim)), dtype=torch.float64)
        sketch.scatter_add_(1, buckets.unsqueeze(0).expand(matrix.shape[0], -1), matrix * signs)
        source_norm = torch.linalg.vector_norm(matrix)
        sketch_norm = torch.linalg.vector_norm(sketch)
        if float(source_norm.item()) > EPS and float(sketch_norm.item()) > EPS:
            sketch *= source_norm / sketch_norm
        sketches.append(sketch)
    return torch.cat(sketches, dim=1)


def joint_sketch_modes(
    data: DisagreementSet,
    rank: int,
    *,
    sketch_dim: int,
    seed: int,
) -> ModeSet:
    features = countsketch_features(data, sketch_dim=sketch_dim, seed=seed)
    return _modes_from_features(features, data.matrix, rank, label_prefix="joint_sketch")


def degenerate_groups(singular_values: torch.Tensor, relative_gap: float) -> list[list[int]]:
    values = torch.as_tensor(singular_values, dtype=torch.float64).reshape(-1)
    if values.numel() == 0:
        return []
    groups = [[0]]
    for index in range(values.numel() - 1):
        gap = float((values[index] - values[index + 1]).abs().item()) / max(
            float(values[index].abs().item()), EPS
        )
        if gap <= float(relative_gap):
            groups[-1].append(index + 1)
        else:
            groups.append([index + 1])
    return groups


@lru_cache(maxsize=None)
def _assignment_dp(scores: tuple[tuple[float, ...], ...]) -> tuple[float, tuple[tuple[int, int], ...]]:
    rows = len(scores)
    cols = len(scores[0]) if rows else 0
    if rows == 0 or cols == 0:
        return 0.0, ()
    if rows > cols:
        transposed = tuple(tuple(scores[row][col] for row in range(rows)) for col in range(cols))
        value, pairs = _assignment_dp(transposed)
        return value, tuple((col, row) for row, col in pairs)

    @lru_cache(maxsize=None)
    def solve(row: int, used: int) -> tuple[float, tuple[tuple[int, int], ...]]:
        if row == rows:
            return 0.0, ()
        best_value = -math.inf
        best_pairs: tuple[tuple[int, int], ...] = ()
        for col in range(cols):
            if used & (1 << col):
                continue
            suffix_value, suffix_pairs = solve(row + 1, used | (1 << col))
            value = float(scores[row][col]) + suffix_value
            pairs = ((row, col),) + suffix_pairs
            if value > best_value + 1e-15 or (
                abs(value - best_value) <= 1e-15 and pairs < best_pairs
            ):
                best_value, best_pairs = value, pairs
        return best_value, best_pairs

    return solve(0, 0)


def optimal_assignment(score_matrix: torch.Tensor) -> list[tuple[int, int]]:
    scores = torch.as_tensor(score_matrix, dtype=torch.float64)
    if scores.ndim != 2:
        raise ValueError("assignment scores must be a matrix")
    rows, cols = scores.shape
    if rows == 0 or cols == 0:
        return []
    if max(rows, cols) <= 12:
        frozen = tuple(tuple(float(value) for value in row) for row in scores.tolist())
        return list(_assignment_dp(frozen)[1])
    # Rank-16 modes and the 30-client diagnostic exceed the exact bitmask-DP
    # budget. scikit-learn is already a repository dependency and brings
    # SciPy, so use its exact O(n^3) Hungarian implementation rather than a
    # greedy approximation that can change the representation verdict.
    try:
        from scipy.optimize import linear_sum_assignment
    except ImportError as exc:  # pragma: no cover - guarded by requirements
        raise RuntimeError(
            "V1 exact matching for more than 12 modes requires scipy"
        ) from exc
    row_ids, col_ids = linear_sum_assignment((-scores).numpy())
    return sorted((int(row), int(col)) for row, col in zip(row_ids, col_ids))


def compare_atom_modes(reference: ModeSet, candidate: ModeSet) -> tuple[dict, list[dict]]:
    if reference.rank == 0 or candidate.rank == 0:
        return {"stability_score": math.nan, "worst_match": math.nan, "match_count": 0}, []
    cosine = reference.directions @ candidate.directions.T
    pairs = optimal_assignment(torch.abs(cosine))
    rows = []
    for left, right in pairs:
        rows.append({
            "reference_mode": reference.labels[left],
            "candidate_mode": candidate.labels[right],
            "signed_cosine": float(cosine[left, right].item()),
            "absolute_cosine": float(abs(cosine[left, right].item())),
        })
    values = [row["absolute_cosine"] for row in rows]
    return {
        "stability_score": float(sum(values) / len(values)),
        "worst_match": float(min(values)),
        "match_count": len(values),
    }, rows

def compare_client_modes(reference: ModeSet, candidate: ModeSet) -> tuple[dict, list[dict]]:
    ref = {label: index for index, label in enumerate(reference.labels)}
    cand = {label: index for index, label in enumerate(candidate.labels)}
    labels = sorted(set(ref).intersection(cand))
    rows = []
    for label in labels:
        value = float(torch.dot(reference.directions[ref[label]], candidate.directions[cand[label]]).item())
        rows.append({
            "reference_mode": label,
            "candidate_mode": label,
            "signed_cosine": value,
            "absolute_cosine": abs(value),
        })
    values = [row["absolute_cosine"] for row in rows]
    return {
        "stability_score": float(sum(values) / len(values)) if values else math.nan,
        "worst_match": float(min(values)) if values else math.nan,
        "match_count": len(values),
    }, rows


def compare_degenerate_subspaces(
    reference: ModeSet,
    candidate: ModeSet,
    *,
    relative_gap: float,
) -> tuple[dict, list[dict]]:
    ref_groups = degenerate_groups(reference.singular_values, relative_gap)
    cand_groups = degenerate_groups(candidate.singular_values, relative_gap)
    if not ref_groups or not cand_groups:
        return {"stability_score": math.nan, "worst_match": math.nan, "match_count": 0}, []
    scores = torch.zeros((len(ref_groups), len(cand_groups)), dtype=torch.float64)
    principal: dict[tuple[int, int], torch.Tensor] = {}
    for left, ref_indices in enumerate(ref_groups):
        ref_basis = reference.directions[ref_indices]
        for right, cand_indices in enumerate(cand_groups):
            cand_basis = candidate.directions[cand_indices]
            cosines = torch.linalg.svdvals(ref_basis @ cand_basis.T)
            overlap = float(cosines.square().sum().item()) / max(len(ref_indices), len(cand_indices))
            scores[left, right] = overlap
            principal[(left, right)] = cosines
    pairs = optimal_assignment(scores)
    rows = []
    for left, right in pairs:
        cosines = principal[(left, right)]
        rows.append({
            "reference_mode": "+".join(reference.labels[index] for index in ref_groups[left]),
            "candidate_mode": "+".join(candidate.labels[index] for index in cand_groups[right]),
            "absolute_cosine": float(scores[left, right].item()),
            "signed_cosine": math.nan,
            "minimum_principal_cosine": float(cosines.min().item()),
            "reference_group_dim": len(ref_groups[left]),
            "candidate_group_dim": len(cand_groups[right]),
        })
    values = [row["absolute_cosine"] for row in rows]
    return {
        "stability_score": float(sum(values) / len(values)),
        "worst_match": float(min(values)),
        "match_count": len(values),
    }, rows
