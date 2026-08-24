"""Client-private Stage-3 evidence and incoming-global reference state.

The objects in this module are local protocol state.  They are deliberately
separate from server aggregation payloads and never expose class lists,
labels, logits, losses, proposal utilities, or trigger values upstream.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Hashable, Mapping, Sequence

import torch
import torch.nn.functional as F

from utils.stage3_vectors import EPS_NORM


PRIVATE_STATE_SCHEMA = "p_fcc_d_rtc_client_private_state_v1"
FUNCTIONAL_MEMORY_SIZE = 32
AUDIT_VIEW_SIZE = 28


def _canonical_json(value) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def stable_seed(*parts, bits: int = 63) -> int:
    digest = hashlib.sha256(_canonical_json(list(parts)).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") & ((1 << bits) - 1)


def _stable_id(value: Hashable):
    if isinstance(value, bool) or not isinstance(value, (int, str)):
        raise TypeError(
            "Stage-3 sample IDs must be globally stable int or str values; "
            f"got {type(value).__name__}"
        )
    return value


def _id_sort_key(value: Hashable) -> str:
    return _canonical_json([type(value).__name__, value])


def _permuted(values: Sequence, *seed_parts) -> list:
    seed = stable_seed(*seed_parts)

    def rank(value):
        digest = hashlib.sha256(
            _canonical_json([seed, type(value).__name__, value]).encode("utf-8")
        ).hexdigest()
        return digest, _id_sort_key(value)

    return sorted(values, key=rank)


def _balanced_select(
    samples: Sequence[tuple[Hashable, int]],
    maximum: int,
    *,
    namespace: str,
    global_seed: int,
    client_id: int,
) -> list[tuple[Hashable, int]]:
    if maximum < 0:
        raise ValueError("maximum must be non-negative")
    if not samples or maximum == 0:
        return []

    by_class: dict[int, list[tuple[Hashable, int]]] = {}
    for sample_id, label in samples:
        by_class.setdefault(int(label), []).append((sample_id, int(label)))

    class_ids = _permuted(
        list(by_class), f"{namespace}-classes", int(global_seed), int(client_id)
    )
    for class_id in class_ids:
        by_class[class_id] = _permuted(
            by_class[class_id],
            f"{namespace}-samples",
            int(global_seed),
            int(client_id),
            int(class_id),
        )

    cursors = {class_id: 0 for class_id in class_ids}
    selected = []
    limit = min(int(maximum), len(samples))
    while len(selected) < limit:
        made_progress = False
        for class_id in class_ids:
            cursor = cursors[class_id]
            bucket = by_class[class_id]
            if cursor >= len(bucket):
                continue
            selected.append(bucket[cursor])
            cursors[class_id] = cursor + 1
            made_progress = True
            if len(selected) == limit:
                break
        if not made_progress:
            break
    return selected


def _evidence_fingerprint(
    *,
    global_seed: int,
    client_id: int,
    memory: Sequence[tuple[Hashable, int]],
    audit: Sequence[tuple[Hashable, int]],
) -> str:
    payload = {
        "schema_version": PRIVATE_STATE_SCHEMA,
        "global_seed": int(global_seed),
        "client_id": int(client_id),
        "memory": [[sample_id, int(label)] for sample_id, label in memory],
        "audit": [[sample_id, int(label)] for sample_id, label in audit],
    }
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ClientEvidenceView:
    """Stable, disjoint IDs for functional memory and offline audit."""

    client_id: int
    global_seed: int
    memory_sample_ids: tuple[Hashable, ...]
    audit_sample_ids: tuple[Hashable, ...]
    fingerprint: str

    def __post_init__(self):
        memory = set(self.memory_sample_ids)
        audit = set(self.audit_sample_ids)
        if len(memory) != len(self.memory_sample_ids):
            raise ValueError("Functional memory contains duplicate sample IDs")
        if len(audit) != len(self.audit_sample_ids):
            raise ValueError("Audit view contains duplicate sample IDs")
        if memory & audit:
            raise ValueError("Functional memory and audit view must be disjoint")
        if len(memory) > FUNCTIONAL_MEMORY_SIZE:
            raise ValueError("Functional memory exceeds the frozen size 32")
        if len(audit) > AUDIT_VIEW_SIZE:
            raise ValueError("Audit view exceeds the frozen size 28")

    def as_dict(self) -> dict:
        return {
            "client_id": int(self.client_id),
            "global_seed": int(self.global_seed),
            "memory_sample_ids": list(self.memory_sample_ids),
            "audit_sample_ids": list(self.audit_sample_ids),
            "fingerprint": self.fingerprint,
        }

    @classmethod
    def from_dict(cls, value: Mapping) -> "ClientEvidenceView":
        return cls(
            client_id=int(value["client_id"]),
            global_seed=int(value["global_seed"]),
            memory_sample_ids=tuple(_stable_id(item) for item in value["memory_sample_ids"]),
            audit_sample_ids=tuple(_stable_id(item) for item in value["audit_sample_ids"]),
            fingerprint=str(value["fingerprint"]),
        )


def build_client_evidence_view(
    sample_ids: Sequence[Hashable],
    labels: Sequence[int] | torch.Tensor,
    *,
    global_seed: int,
    client_id: int,
) -> ClientEvidenceView:
    """Build deterministic class-balanced ``E_k`` and disjoint ``A_k`` IDs."""
    label_values = torch.as_tensor(labels, dtype=torch.long).reshape(-1).tolist()
    if len(sample_ids) != len(label_values):
        raise ValueError("sample_ids and labels must have the same length")
    ids = [_stable_id(value) for value in sample_ids]
    if len(set(ids)) != len(ids):
        raise ValueError("Client data contains duplicate global sample IDs")
    if any(int(label) < 0 for label in label_values):
        raise ValueError("Class labels must be non-negative")

    all_samples = [(sample_id, int(label)) for sample_id, label in zip(ids, label_values)]
    memory = _balanced_select(
        all_samples,
        FUNCTIONAL_MEMORY_SIZE,
        namespace="functional-memory",
        global_seed=global_seed,
        client_id=client_id,
    )
    memory_ids = {sample_id for sample_id, _ in memory}
    remaining = [item for item in all_samples if item[0] not in memory_ids]
    audit = _balanced_select(
        remaining,
        AUDIT_VIEW_SIZE,
        namespace="audit-view",
        global_seed=global_seed,
        client_id=client_id,
    )
    fingerprint = _evidence_fingerprint(
        global_seed=global_seed,
        client_id=client_id,
        memory=memory,
        audit=audit,
    )
    return ClientEvidenceView(
        client_id=int(client_id),
        global_seed=int(global_seed),
        memory_sample_ids=tuple(sample_id for sample_id, _ in memory),
        audit_sample_ids=tuple(sample_id for sample_id, _ in audit),
        fingerprint=fingerprint,
    )


@dataclass
class IncomingGlobalReference:
    """Historical best incoming-global functional state for one client."""

    reference_logits: torch.Tensor
    reference_ce: float
    reference_round: int
    reference_update_count: int
    memory_fingerprint: str

    def __post_init__(self):
        logits = self.reference_logits.detach().cpu().to(torch.float32).contiguous().clone()
        if logits.ndim != 2 or logits.shape[0] == 0:
            raise ValueError("Reference logits must be a non-empty [samples, classes] tensor")
        if not bool(torch.isfinite(logits).all()):
            raise ValueError("Reference logits contain NaN or Inf")
        if not math.isfinite(float(self.reference_ce)) or float(self.reference_ce) < 0:
            raise ValueError("reference_ce must be finite and non-negative")
        if int(self.reference_update_count) < 1:
            raise ValueError("reference_update_count must include initialization")
        if not self.memory_fingerprint:
            raise ValueError("Reference requires a functional-memory fingerprint")
        self.reference_logits = logits
        self.reference_ce = float(self.reference_ce)
        self.reference_round = int(self.reference_round)
        self.reference_update_count = int(self.reference_update_count)
        self.memory_fingerprint = str(self.memory_fingerprint)

    def as_dict(self) -> dict:
        return {
            "reference_logits": self.reference_logits.detach().cpu().to(torch.float32).clone(),
            "reference_ce": float(self.reference_ce),
            "reference_round": int(self.reference_round),
            "reference_update_count": int(self.reference_update_count),
            "memory_fingerprint": self.memory_fingerprint,
        }

    @classmethod
    def from_dict(cls, value: Mapping) -> "IncomingGlobalReference":
        return cls(
            reference_logits=torch.as_tensor(value["reference_logits"]),
            reference_ce=float(value["reference_ce"]),
            reference_round=int(value["reference_round"]),
            reference_update_count=int(value["reference_update_count"]),
            memory_fingerprint=str(value["memory_fingerprint"]),
        )


@dataclass(frozen=True)
class ReferenceObservation:
    current_ce: float
    degradation: float
    initialized: bool
    updated: bool
    reference_ce_before: float | None
    reference_ce_after: float
    reference_round: int
    reference_update_count: int

    def as_dict(self) -> dict:
        return {
            "current_ce": self.current_ce,
            "degradation": self.degradation,
            "initialized": self.initialized,
            "updated": self.updated,
            "reference_ce_before": self.reference_ce_before,
            "reference_ce_after": self.reference_ce_after,
            "reference_round": self.reference_round,
            "reference_update_count": self.reference_update_count,
        }


def _incoming_ce(logits: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, float]:
    detached = torch.as_tensor(logits).detach().to(torch.float32)
    targets = torch.as_tensor(labels, dtype=torch.long, device=detached.device).reshape(-1)
    if detached.ndim != 2 or detached.shape[0] == 0:
        raise ValueError("Incoming logits must be non-empty [samples, classes]")
    if detached.shape[0] != targets.numel():
        raise ValueError("Incoming logits and labels have different sample counts")
    if not bool(torch.isfinite(detached).all()):
        raise ValueError("Incoming logits contain NaN or Inf")
    if bool((targets < 0).any()) or bool((targets >= detached.shape[1]).any()):
        raise ValueError("Incoming labels are outside the logits class range")
    ce = float(F.cross_entropy(detached, targets, reduction="mean").item())
    if not math.isfinite(ce):
        raise ValueError("Incoming-global cross entropy is not finite")
    return detached.detach().cpu().contiguous().clone(), ce


def observe_incoming_global(
    reference: IncomingGlobalReference | None,
    *,
    logits: torch.Tensor,
    labels: torch.Tensor,
    round_id: int,
    memory_fingerprint: str,
) -> tuple[IncomingGlobalReference, ReferenceObservation]:
    """Update/read a reference from the unmodified incoming global only.

    The function name is intentionally narrow: callers must not pass local-CE,
    proposal-probed, restored, or final-upload logits.
    """
    current_logits, current_ce = _incoming_ce(logits, labels)
    fingerprint = str(memory_fingerprint)
    if not fingerprint:
        raise ValueError("memory_fingerprint is required")

    if reference is None:
        new_reference = IncomingGlobalReference(
            reference_logits=current_logits,
            reference_ce=current_ce,
            reference_round=int(round_id),
            reference_update_count=1,
            memory_fingerprint=fingerprint,
        )
        return new_reference, ReferenceObservation(
            current_ce=current_ce,
            degradation=0.0,
            initialized=True,
            updated=True,
            reference_ce_before=None,
            reference_ce_after=current_ce,
            reference_round=int(round_id),
            reference_update_count=1,
        )

    if reference.memory_fingerprint != fingerprint:
        raise ValueError(
            "Functional-memory fingerprint changed for an existing reference; "
            "refusing to silently reset private history"
        )
    if tuple(reference.reference_logits.shape) != tuple(current_logits.shape):
        raise ValueError("Incoming logits shape differs from the saved reference")

    reference_before = float(reference.reference_ce)
    if current_ce < reference_before:
        new_reference = IncomingGlobalReference(
            reference_logits=current_logits,
            reference_ce=current_ce,
            reference_round=int(round_id),
            reference_update_count=reference.reference_update_count + 1,
            memory_fingerprint=fingerprint,
        )
        degradation = 0.0
        updated = True
    else:
        new_reference = reference
        degradation = min(
            1.0,
            max(0.0, (current_ce - reference_before) / (reference_before + EPS_NORM)),
        )
        updated = False

    return new_reference, ReferenceObservation(
        current_ce=current_ce,
        degradation=float(degradation),
        initialized=False,
        updated=updated,
        reference_ce_before=reference_before,
        reference_ce_after=float(new_reference.reference_ce),
        reference_round=int(new_reference.reference_round),
        reference_update_count=int(new_reference.reference_update_count),
    )


@dataclass
class ClientPrivateState:
    evidence: ClientEvidenceView
    reference: IncomingGlobalReference | None = None

    def as_dict(self) -> dict:
        return {
            "evidence": self.evidence.as_dict(),
            "reference": None if self.reference is None else self.reference.as_dict(),
        }

    @classmethod
    def from_dict(cls, value: Mapping) -> "ClientPrivateState":
        reference = value.get("reference")
        return cls(
            evidence=ClientEvidenceView.from_dict(value["evidence"]),
            reference=None if reference is None else IncomingGlobalReference.from_dict(reference),
        )


class ClientPrivateStateStore:
    """Condition-local resumable collection of client-private state."""

    def __init__(self, *, global_seed: int, condition: str, flatten_spec_hash: str):
        if not str(condition):
            raise ValueError("condition must be non-empty")
        if not str(flatten_spec_hash):
            raise ValueError("flatten_spec_hash must be non-empty")
        self.global_seed = int(global_seed)
        self.condition = str(condition)
        self.flatten_spec_hash = str(flatten_spec_hash)
        self._clients: dict[int, ClientPrivateState] = {}

    @property
    def client_ids(self) -> tuple[int, ...]:
        return tuple(sorted(self._clients))

    def get(self, client_id: int) -> ClientPrivateState | None:
        return self._clients.get(int(client_id))

    def get_or_create_evidence(
        self,
        client_id: int,
        sample_ids: Sequence[Hashable],
        labels: Sequence[int] | torch.Tensor,
    ) -> ClientEvidenceView:
        candidate = build_client_evidence_view(
            sample_ids,
            labels,
            global_seed=self.global_seed,
            client_id=int(client_id),
        )
        existing = self._clients.get(int(client_id))
        if existing is None:
            self._clients[int(client_id)] = ClientPrivateState(evidence=candidate)
            return candidate
        if existing.evidence != candidate:
            raise ValueError(
                f"Client {client_id} local data/evidence fingerprint changed after initialization"
            )
        return existing.evidence

    def observe_incoming_global(
        self,
        client_id: int,
        *,
        logits: torch.Tensor,
        labels: torch.Tensor,
        round_id: int,
    ) -> ReferenceObservation:
        client_id = int(client_id)
        state = self._clients.get(client_id)
        if state is None:
            raise KeyError(
                f"Client {client_id} has no functional memory; initialize evidence first"
            )
        if int(torch.as_tensor(labels).numel()) != len(state.evidence.memory_sample_ids):
            raise ValueError("Reference labels do not match the functional-memory size")
        reference, observation = observe_incoming_global(
            state.reference,
            logits=logits,
            labels=labels,
            round_id=int(round_id),
            memory_fingerprint=state.evidence.fingerprint,
        )
        state.reference = reference
        return observation

    def state_dict(self) -> dict:
        return {
            "schema_version": PRIVATE_STATE_SCHEMA,
            "global_seed": self.global_seed,
            "condition": self.condition,
            "flatten_spec_hash": self.flatten_spec_hash,
            "clients": {
                str(client_id): self._clients[client_id].as_dict()
                for client_id in sorted(self._clients)
            },
        }

    @classmethod
    def from_state_dict(
        cls,
        value: Mapping,
        *,
        expected_global_seed: int | None = None,
        expected_condition: str | None = None,
        expected_flatten_spec_hash: str | None = None,
    ) -> "ClientPrivateStateStore":
        schema = str(value.get("schema_version", ""))
        if schema != PRIVATE_STATE_SCHEMA:
            raise ValueError(
                f"Unsupported private-state schema {schema!r}; expected {PRIVATE_STATE_SCHEMA!r}"
            )
        store = cls(
            global_seed=int(value["global_seed"]),
            condition=str(value["condition"]),
            flatten_spec_hash=str(value["flatten_spec_hash"]),
        )
        if expected_global_seed is not None and store.global_seed != int(expected_global_seed):
            raise ValueError("Private-state checkpoint global seed mismatch")
        if expected_condition is not None and store.condition != str(expected_condition):
            raise ValueError("Private-state checkpoint condition mismatch")
        if (
            expected_flatten_spec_hash is not None
            and store.flatten_spec_hash != str(expected_flatten_spec_hash)
        ):
            raise ValueError("Private-state checkpoint flatten spec mismatch")

        clients = value.get("clients", {})
        for key, payload in clients.items():
            client_id = int(key)
            state = ClientPrivateState.from_dict(payload)
            if state.evidence.client_id != client_id:
                raise ValueError("Private-state checkpoint client ID mismatch")
            if state.evidence.global_seed != store.global_seed:
                raise ValueError("Evidence-view global seed differs from checkpoint")
            store._clients[client_id] = state
        return store

    def save(self, path: str | Path) -> Path:
        """Atomically save local simulation state without changing its payload."""
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
    def load(
        cls,
        path: str | Path,
        *,
        expected_global_seed: int | None = None,
        expected_condition: str | None = None,
        expected_flatten_spec_hash: str | None = None,
    ) -> "ClientPrivateStateStore":
        try:
            value = torch.load(Path(path), map_location="cpu", weights_only=False)
        except TypeError:  # PyTorch versions before weights_only was added.
            value = torch.load(Path(path), map_location="cpu")
        if not isinstance(value, Mapping):
            raise ValueError("Private-state checkpoint is not a mapping")
        return cls.from_state_dict(
            value,
            expected_global_seed=expected_global_seed,
            expected_condition=expected_condition,
            expected_flatten_spec_hash=expected_flatten_spec_hash,
        )
