"""Client-local P-FCC, Random-Proposal, and D-RTC method primitives.

All functional decisions in this module consume only a client's private
``EvidenceBatch``.  The module knows nothing about global tail-class IDs,
client class counts, or server test data.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from utils.stage3_private_state import IncomingGlobalReference, stable_seed
from utils.stage3_proposals import ClientProposalPayload, MixedProposal
from utils.stage3_vectors import (
    EPS_NORM,
    LoRAFlatSpec,
    flatten_model,
    load_lora_vector,
)


PROPOSAL_PROBE_DOSE = 0.5
MAX_ACCEPTED_PROPOSALS = 2
RESTORE_TEMPERATURE = 2.0
RESTORE_CE_WEIGHT = 0.5
RESTORE_KL_WEIGHT = 0.5
POSTLOCAL_FCC_MULTIPLIERS = (0.0, 0.25, 0.5, 1.0)


@dataclass(frozen=True)
class EvidenceBatch:
    images: torch.Tensor
    labels: torch.Tensor
    sample_ids: tuple[int | str, ...]

    def __post_init__(self):
        images = torch.as_tensor(self.images).detach().cpu().clone()
        labels = torch.as_tensor(self.labels, dtype=torch.long).detach().cpu().reshape(-1).clone()
        if images.ndim < 2 or images.shape[0] == 0:
            raise ValueError("EvidenceBatch images must contain at least one sample")
        if images.shape[0] != labels.numel() or labels.numel() != len(self.sample_ids):
            raise ValueError("EvidenceBatch images, labels, and sample IDs must align")
        if not bool(torch.isfinite(images).all()):
            raise ValueError("EvidenceBatch images contain NaN or Inf")
        if bool((labels < 0).any()):
            raise ValueError("EvidenceBatch labels must be non-negative")
        if len(set(self.sample_ids)) != len(self.sample_ids):
            raise ValueError("EvidenceBatch contains duplicate sample IDs")
        object.__setattr__(self, "images", images)
        object.__setattr__(self, "labels", labels)

    @property
    def size(self) -> int:
        return int(self.labels.numel())

    def on(self, device) -> tuple[torch.Tensor, torch.Tensor]:
        return self.images.to(device), self.labels.to(device)


@dataclass(frozen=True)
class ProposalProbe:
    proposal_id: int
    utility: float
    proposal_ce: float
    source_count: int
    per_class_utility: tuple[tuple[int, float], ...]


@dataclass(frozen=True)
class ProposalSelection:
    mode: str
    base_ce: float
    probes: tuple[ProposalProbe, ...]
    selected_proposal_ids: tuple[int, ...]
    selected_weights: tuple[float, ...]
    direction: torch.Tensor
    forward_count: int


@dataclass(frozen=True)
class RestoreResult:
    direction: torch.Tensor
    loss: float
    ce: float
    kl: float
    backward_count: int
    gradient_norm: float


@dataclass(frozen=True)
class PostLocalCandidateProbe:
    multiplier: float
    ce: float


@dataclass(frozen=True)
class PostLocalFCCDecision:
    mode: str
    probes: tuple[PostLocalCandidateProbe, ...]
    selected_multiplier: float | None
    selected_ce: float | None
    zero_multiplier_ce: float
    forward_count: int


def _model_device(model) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def _module_training_state(model) -> tuple[tuple[torch.nn.Module, bool], ...]:
    return tuple((module, bool(module.training)) for module in model.modules())


def _restore_module_training_state(states) -> None:
    for module, training in states:
        module.training = bool(training)


def _rng_state() -> dict:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.random.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def _restore_rng_state(state: Mapping) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.random.set_rng_state(state["torch"])
    if state["cuda"] is not None:
        torch.cuda.set_rng_state_all(state["cuda"])


@contextmanager
def isolated_model_probe(model, spec: LoRAFlatSpec, anchor_vector: torch.Tensor):
    """Restore LoRA, buffers, modes, gradients, and RNG after one probe."""
    parameters = dict(model.named_parameters())
    original_vector = flatten_model(model, spec).detach().cpu().clone()
    original_buffers = {
        name: buffer.detach().clone() for name, buffer in model.named_buffers()
    }
    original_grads = {
        entry.name: (
            None
            if parameters[entry.name].grad is None
            else parameters[entry.name].grad.detach().clone()
        )
        for entry in spec.entries
    }
    modes = _module_training_state(model)
    rng = _rng_state()
    try:
        load_lora_vector(model, anchor_vector, spec)
        model.eval()
        yield
    finally:
        load_lora_vector(model, original_vector, spec)
        buffers = dict(model.named_buffers())
        with torch.no_grad():
            for name, value in original_buffers.items():
                buffers[name].copy_(value)
        for entry in spec.entries:
            gradient = original_grads[entry.name]
            parameters[entry.name].grad = None if gradient is None else gradient.clone()
        _restore_module_training_state(modes)
        _restore_rng_state(rng)


def _forward_logits(model, batch: EvidenceBatch, *, with_grad: bool) -> torch.Tensor:
    device = _model_device(model)
    images, _ = batch.on(device)
    if with_grad:
        logits = model(images)
    else:
        with torch.no_grad():
            logits = model(images)
    if not torch.is_tensor(logits) or logits.ndim != 2:
        raise ValueError("Stage-3 model must return a [samples, classes] logits tensor")
    if logits.shape[0] != batch.size:
        raise ValueError("Model logits do not align with the private evidence batch")
    if not bool(torch.isfinite(logits).all()):
        raise ValueError("Model produced NaN or Inf logits on private evidence")
    return logits.float()


def evaluate_private_logits(model, batch: EvidenceBatch) -> torch.Tensor:
    modes = _module_training_state(model)
    try:
        model.eval()
        return _forward_logits(model, batch, with_grad=False).detach().cpu().float().clone()
    finally:
        _restore_module_training_state(modes)


def evaluate_postlocal_fcc_candidates(
    model,
    spec: LoRAFlatSpec,
    incoming_vector: torch.Tensor,
    local_ce_vector: torch.Tensor,
    evidence: EvidenceBatch,
    candidate_uploads: Mapping[float, torch.Tensor],
    *,
    mode: str,
) -> PostLocalFCCDecision:
    """Probe actual fixed-norm uploads after local CE without state leakage.

    ``private`` deterministically chooses the lowest-CE multiplier, including
    the zero-FCC fallback. ``random`` performs the same four forwards but
    retains the full random correction, and ``audit`` records all candidates
    without making an algorithmic decision.
    """
    mode = str(mode)
    if mode not in ("private", "random", "audit"):
        raise ValueError(f"Unknown post-local FCC mode: {mode!r}")
    normalized = {float(key): value for key, value in candidate_uploads.items()}
    expected = set(POSTLOCAL_FCC_MULTIPLIERS)
    if set(normalized) != expected:
        raise ValueError(
            "Post-local FCC candidates must use the frozen multipliers "
            f"{POSTLOCAL_FCC_MULTIPLIERS}"
        )
    incoming = torch.as_tensor(incoming_vector, dtype=torch.float32).reshape(-1)
    local = torch.as_tensor(local_ce_vector, dtype=torch.float32).reshape(-1)
    if incoming.numel() != spec.numel or local.numel() != spec.numel:
        raise ValueError("Post-local FCC anchors do not match the flatten spec")

    probes = []
    for multiplier in POSTLOCAL_FCC_MULTIPLIERS:
        upload = torch.as_tensor(
            normalized[multiplier], dtype=torch.float32
        ).reshape(-1)
        if upload.numel() != spec.numel or not bool(torch.isfinite(upload).all()):
            raise ValueError("Post-local FCC candidate upload is invalid")
        candidate = incoming + upload
        with isolated_model_probe(model, spec, local):
            load_lora_vector(model, candidate, spec)
            logits = _forward_logits(model, evidence, with_grad=False)
        ce, _ = _ce_and_per_class(logits, evidence.labels)
        probes.append(PostLocalCandidateProbe(multiplier=multiplier, ce=ce))

    selected_multiplier = None
    selected_ce = None
    if mode == "private":
        selected = min(probes, key=lambda item: (item.ce, item.multiplier))
        selected_multiplier = float(selected.multiplier)
        selected_ce = float(selected.ce)
    elif mode == "random":
        selected_multiplier = 1.0
        selected_ce = next(item.ce for item in probes if item.multiplier == 1.0)

    zero_ce = next(item.ce for item in probes if item.multiplier == 0.0)
    return PostLocalFCCDecision(
        mode=mode,
        probes=tuple(probes),
        selected_multiplier=selected_multiplier,
        selected_ce=selected_ce,
        zero_multiplier_ce=float(zero_ce),
        forward_count=len(probes),
    )


def _ce_and_per_class(logits: torch.Tensor, labels: torch.Tensor) -> tuple[float, dict[int, float]]:
    targets = labels.to(logits.device)
    losses = F.cross_entropy(logits.float(), targets, reduction="none")
    mean = float(losses.mean().item())
    by_class = {}
    for class_id in sorted(set(int(item) for item in targets.detach().cpu().tolist())):
        mask = targets == int(class_id)
        by_class[class_id] = float(losses[mask].mean().item())
    return mean, by_class


def _proposal_by_id(payload: ClientProposalPayload) -> dict[int, MixedProposal]:
    proposals = {}
    for proposal in payload.proposals:
        proposal_id = int(proposal.proposal_id)
        if proposal_id in proposals:
            raise ValueError("Client proposal payload repeats a proposal ID")
        if proposal.spec_hash != payload.spec_hash:
            raise ValueError("Proposal and payload flatten specs differ")
        proposals[proposal_id] = proposal
    return proposals


def _hash_rank(seed: int, proposal_id: int) -> str:
    text = json.dumps([int(seed), int(proposal_id)], separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def evaluate_and_select_proposals(
    model,
    spec: LoRAFlatSpec,
    incoming_vector: torch.Tensor,
    evidence: EvidenceBatch,
    payload: ClientProposalPayload,
    *,
    mode: str,
    global_seed: int,
    round_id: int,
    client_id: int,
    probe_dose: float = PROPOSAL_PROBE_DOSE,
) -> ProposalSelection:
    """Independently probe every proposal, then privately or randomly select."""
    mode = str(mode)
    if mode not in ("none", "private", "random"):
        raise ValueError(f"Unknown Stage-3 proposal selection mode: {mode!r}")
    if payload.spec_hash != spec.spec_hash:
        raise ValueError("Client proposal payload uses the wrong flatten spec")
    if payload.target_round != int(round_id):
        raise ValueError("Client proposal payload uses the wrong target round")
    if not math.isfinite(float(probe_dose)) or float(probe_dose) < 0:
        raise ValueError("probe_dose must be finite and non-negative")
    anchor = torch.as_tensor(incoming_vector, dtype=torch.float32).reshape(-1)
    if anchor.numel() != spec.numel:
        raise ValueError("Incoming vector does not match the flatten spec")
    proposals = _proposal_by_id(payload)

    with isolated_model_probe(model, spec, anchor):
        load_lora_vector(model, anchor, spec)
        base_logits = _forward_logits(model, evidence, with_grad=False)
    base_ce, base_by_class = _ce_and_per_class(base_logits, evidence.labels)

    probes = []
    for proposal_id in sorted(proposals):
        proposal = proposals[proposal_id]
        vector = torch.as_tensor(proposal.vector, dtype=torch.float32).reshape(-1)
        if vector.numel() != spec.numel or not bool(torch.isfinite(vector).all()):
            raise ValueError("Proposal vector is invalid for the shared LoRA spec")
        candidate = anchor + float(probe_dose) * vector
        with isolated_model_probe(model, spec, anchor):
            load_lora_vector(model, candidate, spec)
            proposal_logits = _forward_logits(model, evidence, with_grad=False)
        proposal_ce, proposal_by_class = _ce_and_per_class(
            proposal_logits, evidence.labels
        )
        per_class = tuple(
            (
                class_id,
                float(base_by_class[class_id] - proposal_by_class[class_id]),
            )
            for class_id in sorted(base_by_class)
        )
        probes.append(
            ProposalProbe(
                proposal_id=proposal_id,
                utility=float(base_ce - proposal_ce),
                proposal_ce=proposal_ce,
                source_count=int(proposal.source_count),
                per_class_utility=per_class,
            )
        )

    selected_ids = []
    weights = []
    if mode == "private":
        positive = [probe for probe in probes if probe.utility > 0.0]
        positive.sort(key=lambda item: (-item.utility, item.proposal_id))
        selected = positive[:MAX_ACCEPTED_PROPOSALS]
        selected_ids = [item.proposal_id for item in selected]
        utility_sum = sum(item.utility for item in selected)
        if selected and utility_sum > 0.0:
            weights = [item.utility / utility_sum for item in selected]
    elif mode == "random" and proposals:
        count = min(MAX_ACCEPTED_PROPOSALS, len(proposals))
        seed = stable_seed(
            "random-proposal", int(global_seed), int(round_id), int(client_id)
        )
        selected_ids = sorted(
            proposals,
            key=lambda proposal_id: (_hash_rank(seed, proposal_id), proposal_id),
        )[:count]
        weights = [1.0 / count] * count

    direction = torch.zeros(spec.numel, dtype=torch.float32)
    for proposal_id, weight in zip(selected_ids, weights):
        direction.add_(
            torch.as_tensor(proposals[proposal_id].vector, dtype=torch.float32).reshape(-1),
            alpha=float(weight),
        )
    return ProposalSelection(
        mode=mode,
        base_ce=base_ce,
        probes=tuple(probes),
        selected_proposal_ids=tuple(int(item) for item in selected_ids),
        selected_weights=tuple(float(item) for item in weights),
        direction=direction,
        forward_count=1 + len(probes),
    )


def evaluate_proposals_for_audit(
    model,
    spec: LoRAFlatSpec,
    incoming_vector: torch.Tensor,
    audit: EvidenceBatch | None,
    payload: ClientProposalPayload,
    *,
    probe_dose: float = PROPOSAL_PROBE_DOSE,
) -> tuple[ProposalProbe, ...]:
    """Selection-independent A_k audit; never returns a correction direction."""
    if audit is None:
        return ()
    result = evaluate_and_select_proposals(
        model,
        spec,
        incoming_vector,
        audit,
        payload,
        mode="none",
        global_seed=0,
        round_id=payload.target_round,
        client_id=-1,
        probe_dose=probe_dose,
    )
    return result.probes


def _flatten_parameter_gradients(model, spec: LoRAFlatSpec) -> torch.Tensor:
    parameters = dict(model.named_parameters())
    chunks = []
    for entry in spec.entries:
        gradient = parameters[entry.name].grad
        if gradient is None:
            chunks.append(torch.zeros(entry.numel, dtype=torch.float32))
        else:
            chunks.append(gradient.detach().cpu().to(torch.float32).reshape(-1).clone())
    return torch.cat(chunks)


def compute_restore_direction(
    model,
    spec: LoRAFlatSpec,
    evidence: EvidenceBatch,
    reference: IncomingGlobalReference,
    *,
    degradation: float,
    temperature: float = RESTORE_TEMPERATURE,
) -> RestoreResult:
    """Compute one D-RTC restore backward without an optimizer step."""
    if not math.isfinite(float(degradation)) or not 0.0 <= float(degradation) <= 1.0:
        raise ValueError("degradation must lie in [0, 1]")
    if float(degradation) == 0.0:
        return RestoreResult(
            direction=torch.zeros(spec.numel, dtype=torch.float32),
            loss=0.0,
            ce=0.0,
            kl=0.0,
            backward_count=0,
            gradient_norm=0.0,
        )
    if float(temperature) != RESTORE_TEMPERATURE:
        raise ValueError("Stage-3 v1 freezes D-RTC temperature at 2")
    if reference.reference_logits.shape[0] != evidence.size:
        raise ValueError("Reference logits and functional memory have different sizes")

    parameters = dict(model.named_parameters())
    spec_names = set(spec.names)
    modes = _module_training_state(model)
    model.zero_grad(set_to_none=True)
    try:
        model.eval()
        logits = _forward_logits(model, evidence, with_grad=True)
        labels = evidence.labels.to(logits.device)
        reference_logits = reference.reference_logits.to(logits.device).float()
        if tuple(reference_logits.shape) != tuple(logits.shape):
            raise ValueError("Reference and current logits shapes differ")
        ce = F.cross_entropy(logits, labels, reduction="mean")
        log_probabilities = F.log_softmax(logits / float(temperature), dim=1)
        reference_probabilities = F.softmax(
            reference_logits / float(temperature), dim=1
        )
        kl = F.kl_div(
            log_probabilities,
            reference_probabilities,
            reduction="batchmean",
        )
        loss = (
            RESTORE_CE_WEIGHT * ce
            + RESTORE_KL_WEIGHT * float(temperature) ** 2 * kl
        )
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("D-RTC restore loss is NaN or Inf")
        loss.backward()
        for name, parameter in parameters.items():
            if name not in spec_names and parameter.grad is not None:
                if bool(torch.any(parameter.grad.detach() != 0)):
                    raise RuntimeError(
                        f"D-RTC produced a gradient outside shared vision-LoRA: {name}"
                    )
        gradient = _flatten_parameter_gradients(model, spec)
        direction = -gradient
        gradient_norm = float(torch.linalg.vector_norm(gradient).item())
        result = RestoreResult(
            direction=direction,
            loss=float(loss.detach().item()),
            ce=float(ce.detach().item()),
            kl=float(kl.detach().item()),
            backward_count=1,
            gradient_norm=gradient_norm,
        )
    finally:
        model.zero_grad(set_to_none=True)
        _restore_module_training_state(modes)
    return result
