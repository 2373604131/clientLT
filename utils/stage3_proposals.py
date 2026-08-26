"""Deterministic previous-round proposal bank for Stage-3 P-FCC.

The server builds this bank from ordinary LoRA uploads belonging to one
experiment condition and one completed communication round.  Raw updates and
cluster membership remain server-internal.  Client payloads contain only
multi-source mixed prototypes and protocol metadata required to interpret
their shared LoRA vector space.
"""

from __future__ import annotations

import math
import os
import statistics
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import torch

from utils.stage3_private_state import stable_seed
from utils.stage3_vectors import EPS_NORM, LoRAFlatSpec


PROPOSAL_BANK_SCHEMA = "p_fcc_d_rtc_proposal_bank_v1"
MAX_PROTOTYPES = 6
MIN_CLUSTER_SOURCES = 4
MIN_LOO_SOURCES = 3
KMEANS_MAX_ITER = 100
KMEANS_TOLERANCE = 1e-6


@dataclass(frozen=True)
class ClientUpload:
    """Server-side view of one ordinary client LoRA upload."""

    client_id: int
    vector: torch.Tensor
    spec_hash: str
    condition: str
    round_id: int


@dataclass(frozen=True)
class ValidatedUpdate:
    client_id: int
    vector: torch.Tensor
    unit_direction: torch.Tensor
    clipped_vector: torch.Tensor
    original_norm: float
    clipped_norm: float


@dataclass(frozen=True)
class MixedProposal:
    """A client-visible multi-source mixed prototype without member IDs."""

    proposal_id: int
    vector: torch.Tensor
    source_count: int
    source_round: int
    spec_hash: str

    def as_payload_dict(self) -> dict:
        return {
            "proposal_id": int(self.proposal_id),
            "vector": self.vector.detach().cpu().to(torch.float32).clone(),
            "source_count": int(self.source_count),
            "source_round": int(self.source_round),
            "spec_hash": self.spec_hash,
        }


@dataclass(frozen=True)
class ClientProposalPayload:
    """Condition-local downlink for one client in the next round."""

    schema_version: str
    condition: str
    target_round: int
    spec_hash: str
    proposals: tuple[MixedProposal, ...]

    def as_dict(self) -> dict:
        payload = {
            "schema_version": self.schema_version,
            "condition": self.condition,
            "target_round": int(self.target_round),
            "spec_hash": self.spec_hash,
            "proposals": [proposal.as_payload_dict() for proposal in self.proposals],
        }
        assert_client_payload_is_private(payload)
        return payload


@dataclass(frozen=True)
class ProposalCluster:
    """Server-private cluster state needed for exact leave-one-out."""

    cluster_id: int
    clipped_updates: tuple[tuple[int, torch.Tensor], ...]

    @property
    def member_client_ids(self) -> tuple[int, ...]:
        return tuple(client_id for client_id, _ in self.clipped_updates)

    @property
    def source_count(self) -> int:
        return len(self.clipped_updates)

    @property
    def clipped_sum(self) -> torch.Tensor:
        return torch.stack(
            [vector for _, vector in self.clipped_updates], dim=0
        ).sum(dim=0, dtype=torch.float32)

    def contribution(self, client_id: int) -> torch.Tensor | None:
        for member_id, vector in self.clipped_updates:
            if member_id == int(client_id):
                return vector
        return None

    def as_state_dict(self) -> dict:
        return {
            "cluster_id": int(self.cluster_id),
            "clipped_updates": [
                {
                    "client_id": int(client_id),
                    "vector": vector.detach().cpu().to(torch.float32).clone(),
                }
                for client_id, vector in self.clipped_updates
            ],
        }

    @classmethod
    def from_state_dict(cls, value: Mapping, *, spec_numel: int) -> "ProposalCluster":
        updates = []
        seen = set()
        for item in value["clipped_updates"]:
            client_id = int(item["client_id"])
            if client_id in seen:
                raise ValueError("Proposal cluster checkpoint contains duplicate clients")
            seen.add(client_id)
            vector = torch.as_tensor(item["vector"], dtype=torch.float32).reshape(-1).cpu().clone()
            if vector.numel() != int(spec_numel):
                raise ValueError("Proposal cluster vector has the wrong flattened size")
            if not bool(torch.isfinite(vector).all()):
                raise ValueError("Proposal cluster vector contains NaN or Inf")
            updates.append((client_id, vector))
        updates.sort(key=lambda item: item[0])
        if len(updates) < MIN_CLUSTER_SOURCES:
            raise ValueError("Saved proposal cluster violates the minimum source count")
        return cls(cluster_id=int(value["cluster_id"]), clipped_updates=tuple(updates))


_BANNED_PAYLOAD_FRAGMENTS = (
    "member",
    "client_id",
    "raw_update",
    "sample",
    "label",
    "logit",
    "loss",
    "utility",
    "accepted",
    "degradation",
    "trigger",
    "tail_carrier",
)


def assert_client_payload_is_private(value, path: str = "payload") -> None:
    """Fail if a client downlink accidentally exposes server/private fields."""
    if isinstance(value, Mapping):
        for key, child in value.items():
            key_lower = str(key).lower()
            if any(fragment in key_lower for fragment in _BANNED_PAYLOAD_FRAGMENTS):
                raise ValueError(f"Forbidden proposal payload field at {path}: {key!r}")
            assert_client_payload_is_private(child, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            assert_client_payload_is_private(child, f"{path}[{index}]")


def _unit(vector: torch.Tensor) -> torch.Tensor:
    norm = torch.linalg.vector_norm(vector)
    if float(norm.item()) <= EPS_NORM:
        raise ValueError("Cannot normalize a near-zero proposal vector")
    return vector / norm


def _cluster_center(unit_rows: torch.Tensor, row_ids: Sequence[int]) -> torch.Tensor:
    """Normalize a cluster mean, with a deterministic antipodal fallback."""
    ordered = sorted(int(row_id) for row_id in row_ids)
    mean = unit_rows[ordered].mean(dim=0, dtype=torch.float32)
    norm = torch.linalg.vector_norm(mean)
    if float(norm.item()) <= EPS_NORM:
        # A spherical center is undefined for an exactly antipodal mean.  Use
        # the lowest sorted input row only for subsequent assignment/merging;
        # the clipped prototype itself is still discarded if its mean is zero.
        return unit_rows[ordered[0]].clone()
    return mean / norm


def _kmeans_plus_plus(
    unit_rows: torch.Tensor,
    cluster_count: int,
    *,
    seed: int,
) -> dict[int, torch.Tensor]:
    row_count = int(unit_rows.shape[0])
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    first = int(torch.randint(row_count, (1,), generator=generator).item())
    chosen = [first]

    while len(chosen) < int(cluster_count):
        center_rows = unit_rows[chosen]
        cosine = unit_rows @ center_rows.T
        maximum = cosine.max(dim=1).values
        # For unit vectors this is squared Euclidean (chord) distance, i.e.
        # the D^2 weight used by k-means++.
        weights = torch.clamp(2.0 - 2.0 * maximum, min=0.0).to(torch.float64)
        weights[torch.tensor(chosen, dtype=torch.long)] = 0.0
        total = float(weights.sum().item())
        if total <= EPS_NORM:
            next_row = min(index for index in range(row_count) if index not in chosen)
        else:
            draw = float(torch.rand((), generator=generator).item()) * total
            cumulative = torch.cumsum(weights, dim=0)
            candidates = torch.nonzero(cumulative > draw, as_tuple=False).reshape(-1)
            if candidates.numel() == 0:
                positive = torch.nonzero(weights > 0, as_tuple=False).reshape(-1)
                next_row = int(positive[-1].item())
            else:
                next_row = int(candidates[0].item())
            if next_row in chosen:
                positive = [
                    index
                    for index in range(row_count)
                    if index not in chosen and float(weights[index].item()) > 0
                ]
                next_row = min(positive) if positive else min(
                    index for index in range(row_count) if index not in chosen
                )
        chosen.append(next_row)

    return {
        cluster_id: unit_rows[row_id].clone()
        for cluster_id, row_id in enumerate(chosen)
    }


def _assign(unit_rows: torch.Tensor, centers: Mapping[int, torch.Tensor]) -> dict[int, list[int]]:
    cluster_ids = sorted(centers)
    center_matrix = torch.stack([centers[cluster_id] for cluster_id in cluster_ids], dim=0)
    similarities = unit_rows @ center_matrix.T
    # torch.argmax returns the first maximum, so sorted IDs implement the
    # frozen lower-cluster-index cosine tie-break.
    positions = similarities.argmax(dim=1).tolist()
    clusters = {cluster_id: [] for cluster_id in cluster_ids}
    for row_id, position in enumerate(positions):
        clusters[cluster_ids[int(position)]].append(row_id)
    return {cluster_id: rows for cluster_id, rows in clusters.items() if rows}


def deterministic_spherical_kmeans(
    unit_rows: torch.Tensor,
    cluster_count: int,
    *,
    seed: int,
    max_iter: int = KMEANS_MAX_ITER,
    tolerance: float = KMEANS_TOLERANCE,
) -> dict[int, list[int]]:
    """Run one deterministic spherical k-means initialization and fit."""
    rows = torch.as_tensor(unit_rows, dtype=torch.float32).detach().cpu()
    if rows.ndim != 2 or rows.shape[0] == 0 or rows.shape[1] == 0:
        raise ValueError("unit_rows must be a non-empty 2D tensor")
    if not bool(torch.isfinite(rows).all()):
        raise ValueError("unit_rows contains NaN or Inf")
    norms = torch.linalg.vector_norm(rows, dim=1)
    if not bool(torch.all(torch.abs(norms - 1.0) < 1e-5)):
        raise ValueError("deterministic_spherical_kmeans requires unit rows")
    if not 1 <= int(cluster_count) <= rows.shape[0]:
        raise ValueError("cluster_count must lie between 1 and the row count")

    centers = _kmeans_plus_plus(rows, int(cluster_count), seed=int(seed))
    previous_assignment = None
    clusters = None
    for _ in range(int(max_iter)):
        clusters = _assign(rows, centers)
        new_centers = {
            cluster_id: _cluster_center(rows, row_ids)
            for cluster_id, row_ids in sorted(clusters.items())
        }
        assignment = tuple(
            next(cluster_id for cluster_id, ids in clusters.items() if row_id in ids)
            for row_id in range(rows.shape[0])
        )
        common_ids = sorted(set(centers) & set(new_centers))
        maximum_shift = max(
            (
                float(torch.linalg.vector_norm(new_centers[item] - centers[item]).item())
                for item in common_ids
            ),
            default=float("inf"),
        )
        removed_empty = set(centers) != set(new_centers)
        centers = new_centers
        if (
            not removed_empty
            and previous_assignment == assignment
            and maximum_shift <= float(tolerance)
        ):
            break
        previous_assignment = assignment
    if clusters is None:
        raise RuntimeError("Spherical k-means did not execute")
    # Reassign once from the final recomputed centers, then recompute centers as
    # required by the contract.  Only memberships are returned.
    clusters = _assign(rows, centers)
    _ = {
        cluster_id: _cluster_center(rows, row_ids)
        for cluster_id, row_ids in sorted(clusters.items())
    }
    return {cluster_id: sorted(row_ids) for cluster_id, row_ids in sorted(clusters.items())}


def merge_small_clusters(
    clusters: Mapping[int, Sequence[int]],
    unit_rows: torch.Tensor,
    *,
    minimum_sources: int = MIN_CLUSTER_SOURCES,
) -> dict[int, list[int]]:
    """Merge every undersized cluster into its nearest surviving center."""
    rows = torch.as_tensor(unit_rows, dtype=torch.float32).detach().cpu()
    merged = {
        int(cluster_id): sorted(int(row_id) for row_id in row_ids)
        for cluster_id, row_ids in clusters.items()
        if row_ids
    }
    if not merged:
        return {}

    while True:
        small = [
            cluster_id
            for cluster_id, row_ids in merged.items()
            if len(row_ids) < int(minimum_sources)
        ]
        if not small:
            break
        if len(merged) == 1:
            # The caller only invokes this with at least four total valid
            # updates.  Reaching this branch with a smaller sole cluster means
            # the supplied memberships were incomplete.
            if len(next(iter(merged.values()))) < int(minimum_sources):
                raise ValueError("Cannot satisfy minimum cluster source count")
            break
        source_id = min(small, key=lambda item: (len(merged[item]), item))
        source_center = _cluster_center(rows, merged[source_id])
        destination_ids = sorted(item for item in merged if item != source_id)
        best_id = destination_ids[0]
        best_similarity = float(
            torch.dot(source_center, _cluster_center(rows, merged[best_id])).item()
        )
        for destination_id in destination_ids[1:]:
            similarity = float(
                torch.dot(
                    source_center, _cluster_center(rows, merged[destination_id])
                ).item()
            )
            if similarity > best_similarity:
                best_id = destination_id
                best_similarity = similarity
        merged[best_id] = sorted(merged[best_id] + merged[source_id])
        del merged[source_id]
        # Centers are intentionally recomputed on the next loop iteration.

    return {cluster_id: row_ids for cluster_id, row_ids in sorted(merged.items())}


def _prototype_from_sum(
    vector_sum: torch.Tensor,
    source_count: int,
    median_norm: float,
) -> torch.Tensor | None:
    if int(source_count) <= 0:
        return None
    mean = vector_sum.to(torch.float32) / float(source_count)
    norm = torch.linalg.vector_norm(mean)
    if float(norm.item()) <= EPS_NORM:
        return None
    return mean * (float(median_norm) / norm)


@dataclass(frozen=True)
class ProposalBank:
    """One-round, one-condition server proposal bank."""

    condition: str
    source_round: int
    global_seed: int
    spec_hash: str
    spec_numel: int
    median_update_norm: float
    valid_update_count: int
    initial_cluster_count: int
    clusters: tuple[ProposalCluster, ...]
    invalid_update_reasons: tuple[tuple[int, str], ...]
    schema_version: str = PROPOSAL_BANK_SCHEMA

    @property
    def target_round(self) -> int:
        return int(self.source_round + 1)

    @property
    def is_empty(self) -> bool:
        return len(self.clusters) == 0

    def payload_for(
        self,
        client_id: int,
        *,
        expected_condition: str | None = None,
        expected_target_round: int | None = None,
        expected_spec_hash: str | None = None,
    ) -> ClientProposalPayload:
        if expected_condition is not None and self.condition != str(expected_condition):
            raise ValueError("Proposal bank condition mismatch")
        if expected_target_round is not None and self.target_round != int(expected_target_round):
            raise ValueError("Proposal bank target round mismatch")
        if expected_spec_hash is not None and self.spec_hash != str(expected_spec_hash):
            raise ValueError("Proposal bank flatten spec mismatch")

        proposals = []
        for cluster in sorted(self.clusters, key=lambda item: item.cluster_id):
            own = cluster.contribution(int(client_id))
            if own is None:
                source_count = cluster.source_count
                vector_sum = cluster.clipped_sum
            else:
                source_count = cluster.source_count - 1
                if source_count < MIN_LOO_SOURCES:
                    continue
                vector_sum = cluster.clipped_sum - own
            prototype = _prototype_from_sum(
                vector_sum, source_count, self.median_update_norm
            )
            if prototype is None:
                continue
            proposals.append(
                MixedProposal(
                    proposal_id=cluster.cluster_id,
                    vector=prototype.detach().cpu().to(torch.float32).clone(),
                    source_count=source_count,
                    source_round=self.source_round,
                    spec_hash=self.spec_hash,
                )
            )
        return ClientProposalPayload(
            schema_version=PROPOSAL_BANK_SCHEMA,
            condition=self.condition,
            target_round=self.target_round,
            spec_hash=self.spec_hash,
            proposals=tuple(proposals),
        )

    def diagnostics(self) -> dict:
        return {
            "condition": self.condition,
            "source_round": self.source_round,
            "target_round": self.target_round,
            "spec_hash": self.spec_hash,
            "valid_update_count": self.valid_update_count,
            "invalid_update_count": len(self.invalid_update_reasons),
            "invalid_update_reasons": dict(self.invalid_update_reasons),
            "median_update_norm": self.median_update_norm,
            "initial_cluster_count": self.initial_cluster_count,
            "final_cluster_count": len(self.clusters),
            "cluster_source_counts": {
                cluster.cluster_id: cluster.source_count for cluster in self.clusters
            },
        }

    def state_dict(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "condition": self.condition,
            "source_round": int(self.source_round),
            "global_seed": int(self.global_seed),
            "spec_hash": self.spec_hash,
            "spec_numel": int(self.spec_numel),
            "median_update_norm": float(self.median_update_norm),
            "valid_update_count": int(self.valid_update_count),
            "initial_cluster_count": int(self.initial_cluster_count),
            "clusters": [cluster.as_state_dict() for cluster in self.clusters],
            "invalid_update_reasons": [
                [int(client_id), str(reason)]
                for client_id, reason in self.invalid_update_reasons
            ],
        }

    @classmethod
    def from_state_dict(
        cls,
        value: Mapping,
        *,
        expected_global_seed: int | None = None,
        expected_condition: str | None = None,
        expected_source_round: int | None = None,
        expected_spec_hash: str | None = None,
    ) -> "ProposalBank":
        schema = str(value.get("schema_version", ""))
        if schema != PROPOSAL_BANK_SCHEMA:
            raise ValueError("Unsupported proposal bank checkpoint schema")
        bank = cls(
            condition=str(value["condition"]),
            source_round=int(value["source_round"]),
            global_seed=int(value["global_seed"]),
            spec_hash=str(value["spec_hash"]),
            spec_numel=int(value["spec_numel"]),
            median_update_norm=float(value["median_update_norm"]),
            valid_update_count=int(value["valid_update_count"]),
            initial_cluster_count=int(value["initial_cluster_count"]),
            clusters=tuple(
                ProposalCluster.from_state_dict(
                    item, spec_numel=int(value["spec_numel"])
                )
                for item in value.get("clusters", [])
            ),
            invalid_update_reasons=tuple(
                (int(item[0]), str(item[1]))
                for item in value.get("invalid_update_reasons", [])
            ),
        )
        bank._validate()
        if expected_global_seed is not None and bank.global_seed != int(expected_global_seed):
            raise ValueError("Proposal bank checkpoint global seed mismatch")
        if expected_condition is not None and bank.condition != str(expected_condition):
            raise ValueError("Proposal bank checkpoint condition mismatch")
        if expected_source_round is not None and bank.source_round != int(expected_source_round):
            raise ValueError("Proposal bank checkpoint source round mismatch")
        if expected_spec_hash is not None and bank.spec_hash != str(expected_spec_hash):
            raise ValueError("Proposal bank checkpoint flatten spec mismatch")
        return bank

    def _validate(self) -> None:
        if not self.condition:
            raise ValueError("Proposal bank condition must be non-empty")
        if not self.spec_hash or self.spec_numel <= 0:
            raise ValueError("Proposal bank requires a valid flatten spec")
        if not math.isfinite(self.median_update_norm) or self.median_update_norm < 0:
            raise ValueError("Proposal bank median norm is invalid")
        if self.valid_update_count < 0:
            raise ValueError("Proposal bank valid update count is invalid")
        expected_initial = min(
            MAX_PROTOTYPES, self.valid_update_count // MIN_CLUSTER_SOURCES
        )
        if self.initial_cluster_count != expected_initial:
            raise ValueError("Proposal bank initial cluster count is inconsistent")
        if len(self.clusters) > self.initial_cluster_count:
            raise ValueError("Proposal bank replenished clusters after merging")
        if self.valid_update_count >= MIN_CLUSTER_SOURCES and self.median_update_norm <= EPS_NORM:
            raise ValueError("Non-empty valid update set requires a positive median norm")
        cluster_ids = [cluster.cluster_id for cluster in self.clusters]
        if cluster_ids != sorted(cluster_ids) or len(set(cluster_ids)) != len(cluster_ids):
            raise ValueError("Proposal cluster IDs must be unique and sorted")
        member_ids = []
        for cluster in self.clusters:
            if cluster.cluster_id < 0:
                raise ValueError("Proposal cluster ID must be non-negative")
            if cluster.source_count < MIN_CLUSTER_SOURCES:
                raise ValueError("Proposal cluster violates minimum sources")
            member_ids.extend(cluster.member_client_ids)
            for _, clipped in cluster.clipped_updates:
                clipped_norm = float(torch.linalg.vector_norm(clipped).item())
                if not math.isfinite(clipped_norm) or clipped_norm <= EPS_NORM:
                    raise ValueError("Saved bank contains an invalid clipped update")
                if clipped_norm > self.median_update_norm * (1.0 + 1e-6):
                    raise ValueError("Saved update exceeds the median clipping budget")
            prototype = _prototype_from_sum(
                cluster.clipped_sum,
                cluster.source_count,
                self.median_update_norm,
            )
            if prototype is None:
                raise ValueError("Saved bank contains a near-zero full prototype")
        if len(member_ids) != len(set(member_ids)):
            raise ValueError("A valid client appears in multiple proposal clusters")
        invalid_ids = [client_id for client_id, _ in self.invalid_update_reasons]
        if len(invalid_ids) != len(set(invalid_ids)):
            raise ValueError("Proposal bank repeats an invalid client ID")
        if set(member_ids) & set(invalid_ids):
            raise ValueError("A client cannot be both valid and invalid")

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(
            prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
        )
        os.close(descriptor)
        temporary_path = Path(temporary)
        try:
            torch.save(self.state_dict(), temporary_path)
            os.replace(temporary_path, path)
        finally:
            if temporary_path.exists():
                temporary_path.unlink()
        return path

    @classmethod
    def load(cls, path: str | Path, **expected) -> "ProposalBank":
        try:
            value = torch.load(Path(path), map_location="cpu", weights_only=False)
        except TypeError:
            value = torch.load(Path(path), map_location="cpu")
        if not isinstance(value, Mapping):
            raise ValueError("Proposal bank checkpoint is not a mapping")
        return cls.from_state_dict(value, **expected)


def _validate_uploads(
    uploads: Sequence[ClientUpload],
    *,
    spec: LoRAFlatSpec,
    condition: str,
    source_round: int,
) -> tuple[list[tuple[int, torch.Tensor, float]], list[tuple[int, str]]]:
    seen = set()
    valid = []
    invalid = []
    for upload in sorted(uploads, key=lambda item: int(item.client_id)):
        client_id = int(upload.client_id)
        if client_id in seen:
            raise ValueError(f"Duplicate upload for client {client_id}")
        seen.add(client_id)
        if str(upload.condition) != str(condition):
            raise ValueError(
                f"Cross-condition upload contamination for client {client_id}: "
                f"{upload.condition!r} != {condition!r}"
            )
        if int(upload.round_id) != int(source_round):
            raise ValueError(
                f"Stale/future upload round for client {client_id}: "
                f"{upload.round_id} != {source_round}"
            )
        if str(upload.spec_hash) != spec.spec_hash:
            invalid.append((client_id, "spec_hash_mismatch"))
            continue
        try:
            raw = torch.as_tensor(upload.vector)
        except (TypeError, ValueError, RuntimeError):
            invalid.append((client_id, "not_tensor_convertible"))
            continue
        if not torch.is_floating_point(raw):
            invalid.append((client_id, "nonfloating_dtype"))
            continue
        if raw.ndim != 1 or raw.numel() != spec.numel:
            invalid.append((client_id, "shape_or_numel_mismatch"))
            continue
        vector = raw.detach().cpu().to(torch.float32).clone()
        if not bool(torch.isfinite(vector).all()):
            invalid.append((client_id, "nonfinite"))
            continue
        norm = float(torch.linalg.vector_norm(vector).item())
        if norm <= EPS_NORM:
            invalid.append((client_id, "near_zero_norm"))
            continue
        valid.append((client_id, vector, norm))
    return valid, invalid


def build_proposal_bank(
    uploads: Sequence[ClientUpload],
    *,
    spec: LoRAFlatSpec,
    global_seed: int,
    source_round: int,
    condition: str,
) -> ProposalBank:
    """Build the condition's next-round bank from this round's uploads."""
    if not str(condition):
        raise ValueError("condition must be non-empty")
    valid_rows, invalid = _validate_uploads(
        uploads,
        spec=spec,
        condition=str(condition),
        source_round=int(source_round),
    )
    valid_count = len(valid_rows)
    initial_cluster_count = min(MAX_PROTOTYPES, valid_count // MIN_CLUSTER_SOURCES)
    if valid_count < MIN_CLUSTER_SOURCES:
        bank = ProposalBank(
            condition=str(condition),
            source_round=int(source_round),
            global_seed=int(global_seed),
            spec_hash=spec.spec_hash,
            spec_numel=spec.numel,
            median_update_norm=0.0,
            valid_update_count=valid_count,
            initial_cluster_count=0,
            clusters=(),
            invalid_update_reasons=tuple(invalid),
        )
        bank._validate()
        return bank

    median_norm = float(statistics.median(row[2] for row in valid_rows))
    processed = []
    for client_id, vector, original_norm in valid_rows:
        unit = vector / original_norm
        scale = min(1.0, median_norm / original_norm)
        clipped = vector * scale
        processed.append(
            ValidatedUpdate(
                client_id=client_id,
                vector=vector,
                unit_direction=unit,
                clipped_vector=clipped,
                original_norm=original_norm,
                clipped_norm=float(torch.linalg.vector_norm(clipped).item()),
            )
        )

    unit_rows = torch.stack([row.unit_direction for row in processed], dim=0)
    clusters = deterministic_spherical_kmeans(
        unit_rows,
        initial_cluster_count,
        seed=stable_seed(
            "proposal-cluster",
            int(global_seed),
            int(source_round),
        ),
    )
    clusters = merge_small_clusters(clusters, unit_rows)

    final_clusters = []
    for cluster_id, row_ids in sorted(clusters.items()):
        clipped_updates = tuple(
            sorted(
                (
                    (
                        processed[row_id].client_id,
                        processed[row_id].clipped_vector.detach().cpu().to(torch.float32).clone(),
                    )
                    for row_id in row_ids
                ),
                key=lambda item: item[0],
            )
        )
        cluster = ProposalCluster(
            cluster_id=int(cluster_id), clipped_updates=clipped_updates
        )
        if cluster.source_count < MIN_CLUSTER_SOURCES:
            raise RuntimeError("Small-cluster merge failed its source-count contract")
        if _prototype_from_sum(
            cluster.clipped_sum, cluster.source_count, median_norm
        ) is None:
            continue
        final_clusters.append(cluster)

    bank = ProposalBank(
        condition=str(condition),
        source_round=int(source_round),
        global_seed=int(global_seed),
        spec_hash=spec.spec_hash,
        spec_numel=spec.numel,
        median_update_norm=median_norm,
        valid_update_count=valid_count,
        initial_cluster_count=initial_cluster_count,
        clusters=tuple(final_clusters),
        invalid_update_reasons=tuple(invalid),
    )
    bank._validate()
    return bank
