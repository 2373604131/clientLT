import copy
import math

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F


class TailResidualExpert(nn.Module):
    """Class-wise residual delta-logit expert with zero residual init."""

    def __init__(self, feature_dim, num_classes, dtype=torch.float32, hidden_dim=None):
        super().__init__()
        hidden_dim = int(hidden_dim or feature_dim)
        self.fc1 = nn.Linear(feature_dim, hidden_dim).to(dtype)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(hidden_dim, num_classes).to(dtype)
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, image_features, logits_base=None, text_features=None):
        hidden = self.relu(self.fc1(image_features))
        return self.fc2(hidden)


class CosineResidualTailExpert(nn.Module):
    """V1-style class-wise cosine residual expert for FedTEF-v2 diagnostics.

    This stream intentionally keeps a stronger class-wise tail classifier than
    the zero-init MLP. It still returns delta logits and is fused only through
    FedTEF-v2 residual_add.
    """

    def __init__(
        self,
        feature_dim,
        num_classes,
        dtype=torch.float32,
        init_mode="normal_residual",
        init_logit_scale=10.0,
        learnable_scale=True,
        use_bias=True,
        logit_scale_max=100.0,
    ):
        super().__init__()
        self.init_mode = str(init_mode).lower()
        self.init_logit_scale = float(init_logit_scale)
        self.learnable_scale = bool(learnable_scale)
        self.use_bias = bool(use_bias)
        self.logit_scale_max = float(logit_scale_max)
        self.weight = nn.Parameter(torch.empty(num_classes, feature_dim, dtype=dtype))
        if self.use_bias:
            self.bias = nn.Parameter(torch.empty(num_classes, dtype=dtype))
        else:
            self.register_parameter("bias", None)

        nn.init.normal_(self.weight, std=0.02)
        if self.use_bias:
            nn.init.zeros_(self.bias)

        init_scale = max(self.init_logit_scale, 1e-12)
        logit_scale = torch.tensor(math.log(init_scale), dtype=dtype)
        if self.learnable_scale:
            self.logit_scale = nn.Parameter(logit_scale)
        else:
            self.register_buffer("logit_scale", logit_scale)

        if self.init_mode == "zero_residual":
            self.output_gain = nn.Parameter(torch.zeros((), dtype=dtype))
            print(
                "CosineResidualTailExpert zero_residual uses a learnable zero "
                "output_gain; normal_residual is recommended for capacity debugging."
            )
        elif self.init_mode == "normal_residual":
            self.register_buffer("output_gain", torch.ones((), dtype=dtype))
        else:
            raise ValueError(f"Unknown FedTEF-v2 cosine tail init_mode: {init_mode}")

    def forward(self, image_features, logits_base=None, text_features=None):
        image = F.normalize(image_features.float(), dim=-1)
        weight = F.normalize(self.weight.float(), dim=-1)
        logits = image @ weight.t()
        if self.learnable_scale:
            scale = self.logit_scale.float().exp().clamp(max=self.logit_scale_max)
        else:
            scale = torch.tensor(
                self.init_logit_scale,
                device=logits.device,
                dtype=logits.dtype,
            ).clamp(max=self.logit_scale_max)
        logits = scale * logits
        if self.use_bias:
            logits = logits + self.bias.float()
        logits = self.output_gain.float() * logits
        return logits.to(dtype=image_features.dtype)


class SemanticResidualMemoryExpert(nn.Module):
    """Text-aligned class memory that produces a semantic delta logit.

    Each class stores a residual prototype in the CLIP text space. The memory
    starts at zero, so the stream is initially equivalent to the base branch.
    """

    def __init__(
        self,
        feature_dim,
        num_classes,
        dtype=torch.float32,
        init_logit_scale=10.0,
        learnable_scale=True,
        use_bias=True,
        logit_scale_max=100.0,
    ):
        super().__init__()
        self.init_logit_scale = float(init_logit_scale)
        self.learnable_scale = bool(learnable_scale)
        self.use_bias = bool(use_bias)
        self.logit_scale_max = float(logit_scale_max)
        self.memory = nn.Parameter(torch.zeros(num_classes, feature_dim, dtype=dtype))
        if self.use_bias:
            self.bias = nn.Parameter(torch.zeros(num_classes, dtype=dtype))
        else:
            self.register_parameter("bias", None)

        init_scale = max(self.init_logit_scale, 1e-12)
        logit_scale = torch.tensor(math.log(init_scale), dtype=dtype)
        if self.learnable_scale:
            self.logit_scale = nn.Parameter(logit_scale)
        else:
            self.register_buffer("logit_scale", logit_scale)

    def forward(self, image_features, logits_base=None, text_features=None):
        if text_features is None:
            raise ValueError("SemanticResidualMemoryExpert requires text_features")

        image = F.normalize(image_features.float(), dim=-1)
        base_input = text_features.detach().float()
        base_text = F.normalize(base_input, dim=-1)
        memory_text = F.normalize(base_input + self.memory.float(), dim=-1)
        if self.learnable_scale:
            scale = self.logit_scale.float().exp().clamp(max=self.logit_scale_max)
        else:
            scale = torch.tensor(
                self.init_logit_scale,
                device=image.device,
                dtype=image.dtype,
            ).clamp(max=self.logit_scale_max)
        residual = scale * (image @ (memory_text - base_text).t())
        if self.use_bias:
            residual = residual + self.bias.float()
        return residual.to(dtype=image_features.dtype)


def compute_controlled_hard_negative_loss(
    logits_base,
    residual_tail,
    labels,
    protected_label,
    topm=5,
    residual_lambda=1.0,
):
    """Residual CE on {positive label} union base Top-M hard negatives."""

    active_ids = torch.nonzero(protected_label, as_tuple=False).view(-1)
    if active_ids.numel() == 0:
        return residual_tail.sum() * 0.0
    topm = max(1, min(int(topm), logits_base.shape[1] - 1))
    logits_base_safe = torch.nan_to_num(
        logits_base.float(),
        nan=0.0,
        posinf=80.0,
        neginf=-80.0,
    ).clamp(min=-80.0, max=80.0)
    residual_tail_safe = torch.nan_to_num(
        residual_tail.float(),
        nan=0.0,
        posinf=80.0,
        neginf=-80.0,
    ).clamp(min=-80.0, max=80.0)
    logits_objective = logits_base_safe.detach() + float(residual_lambda) * residual_tail_safe
    losses = []
    for sample_idx in active_ids.tolist():
        y = int(labels[sample_idx].item())
        base_row = logits_base_safe[sample_idx].detach().clone()
        base_row[y] = torch.finfo(base_row.dtype).min
        hard_ids = torch.topk(base_row, topm).indices
        class_ids = torch.cat([
            torch.tensor([y], device=hard_ids.device, dtype=hard_ids.dtype),
            hard_ids,
        ])
        local_logits = logits_objective[sample_idx, class_ids].view(1, -1)
        local_target = torch.zeros(1, device=local_logits.device, dtype=torch.long)
        losses.append(F.cross_entropy(local_logits, local_target, reduction="mean"))
    return torch.stack(losses).mean()


def compute_tie_break_mask(num_classes, k, mode="random", seed=0, current_round=0, dataset_name=""):
    mode = str(mode).lower()
    protected = torch.zeros(num_classes, dtype=torch.bool)
    if mode == "none":
        return protected
    dataset_name = str(dataset_name).lower()
    if mode == "oracle_bottom20" and "cifar100" in dataset_name and num_classes == 100:
        ids = list(range(max(0, num_classes - k), num_classes))
        print("Oracle bottom20 warmup is enabled. This is not a privacy-preserving main result.")
    else:
        rng = np.random.default_rng(int(seed) + int(current_round) * 1009)
        ids = rng.choice(num_classes, size=k, replace=False).tolist()
        if ids == list(range(k)):
            ids = rng.choice(num_classes, size=k, replace=False).tolist()
        print(f"FedTEF tie-break selected classes: {ids}")
    protected[torch.as_tensor(ids, dtype=torch.long)] = True
    return protected


def compute_warmup_mask(num_classes, k, mode, seed=0, current_round=0, dataset_name=""):
    mode = str(mode).lower()
    if mode == "round_robin":
        protected = torch.zeros(num_classes, dtype=torch.bool)
        start = (int(current_round) * int(k)) % int(num_classes)
        ids = [int((start + offset) % num_classes) for offset in range(k)]
        protected[torch.as_tensor(ids, dtype=torch.long)] = True
        print(f"FedTEF-v2 round-robin warmup classes: {ids}")
        return protected
    if mode == "all_low":
        return torch.ones(num_classes, dtype=torch.bool)
    if mode == "oracle_bottom20":
        return compute_tie_break_mask(
            num_classes,
            k,
            mode="oracle_bottom20",
            seed=seed,
            current_round=current_round,
            dataset_name=dataset_name,
        )
    if mode in ("random", "none"):
        return compute_tie_break_mask(
            num_classes,
            k,
            mode=mode,
            seed=seed,
            current_round=current_round,
            dataset_name=dataset_name,
        )
    raise ValueError(f"Unknown FedTEF-v2 warmup_mode: {mode}")


class ExposureTracker:
    """Server-side low-exposure gate from class-wise tail update energy."""

    def __init__(
        self,
        num_classes,
        rho=0.9,
        eps=1e-6,
        gate_mode="soft",
        temperature=1.0,
        threshold=None,
        tail_topk=20,
        round0_tie_break="random",
        warmup_mode="round_robin",
        warmup_rounds=5,
        seed=0,
        dataset_name="",
    ):
        self.num_classes = int(num_classes)
        self.rho = float(rho)
        self.eps = float(eps)
        self.gate_mode = str(gate_mode).lower()
        self.temperature = max(float(temperature), self.eps)
        self.threshold = threshold
        self.tail_topk = max(1, min(int(tail_topk), self.num_classes))
        self.round0_tie_break = round0_tie_break
        self.warmup_mode = str(warmup_mode).lower()
        self.warmup_rounds = max(0, int(warmup_rounds))
        self.seed = int(seed)
        self.dataset_name = dataset_name
        self.exposure = torch.zeros(self.num_classes, dtype=torch.float32)
        self.opportunity_count = torch.zeros(self.num_classes, dtype=torch.float32)
        self.last_energy = torch.zeros(self.num_classes, dtype=torch.float32)
        self.round = 0

    def update_from_energy(self, energy, gate=None):
        energy = torch.as_tensor(energy, dtype=torch.float32).cpu()
        self.last_energy = energy.clone()
        if gate is not None:
            gate = torch.as_tensor(gate, dtype=torch.float32).cpu()
            self.opportunity_count += (gate > self.eps).to(dtype=torch.float32)
        mean_energy = energy.mean()
        if mean_energy <= self.eps:
            energy_norm = torch.zeros_like(energy)
        else:
            energy_norm = energy / (mean_energy + self.eps)
        self.exposure = self.rho * self.exposure + (1.0 - self.rho) * energy_norm
        self.round += 1
        return self.exposure

    def effective_exposure(self):
        if self.opportunity_count.max().item() <= self.eps:
            opportunity_norm = torch.zeros_like(self.opportunity_count)
        else:
            opportunity_norm = self.opportunity_count / (self.opportunity_count.mean() + self.eps)
        return self.exposure + opportunity_norm

    def compute_scores(self):
        return 1.0 / (self.effective_exposure() + self.eps)

    def compute_gate(self, current_round=0):
        if int(current_round) < self.warmup_rounds and self.warmup_mode != "none":
            mask = compute_warmup_mask(
                self.num_classes,
                self.tail_topk,
                self.warmup_mode,
                seed=self.seed,
                current_round=current_round,
                dataset_name=self.dataset_name,
            )
            scores = mask.float()
            return mask.float(), scores, mask

        scores = self.compute_scores()
        tied = torch.isclose(scores.max(), scores.min())
        if tied:
            mask = compute_tie_break_mask(
                self.num_classes,
                self.tail_topk,
                mode=self.round0_tie_break,
                seed=self.seed,
                current_round=current_round,
                dataset_name=self.dataset_name,
            )
            return mask.float(), scores, mask

        if self.gate_mode == "hard_topk":
            mask = torch.zeros(self.num_classes, dtype=torch.bool)
            mask[torch.topk(scores, self.tail_topk).indices.cpu()] = True
            return mask.float(), scores, mask

        sorted_scores = torch.sort(scores, descending=True).values
        threshold = self.threshold
        if threshold is None:
            threshold = sorted_scores[self.tail_topk - 1].item()
        score_std = scores.std(unbiased=False)
        normalized = (scores - scores.mean()) / (score_std + self.eps)
        threshold_norm = (float(threshold) - scores.mean()) / (score_std + self.eps)
        soft_gate = torch.sigmoid((normalized - threshold_norm) / self.temperature)
        mask = torch.zeros(self.num_classes, dtype=torch.bool)
        mask[torch.topk(scores, self.tail_topk).indices.cpu()] = True
        gate = soft_gate * mask.float()
        return gate.float(), scores, mask


class TailNeedTracker(ExposureTracker):
    """Persistent FedTEF-v2 gate from scarcity and residual-update need.

    This tracker is intentionally privacy-friendly: it uses only server-side
    tail-stream update energy and the previous protected set. It does not use
    client class counts or any oracle class ordering.
    """

    def __init__(
        self,
        num_classes,
        rho=0.9,
        eps=1e-6,
        gate_mode="soft",
        temperature=1.0,
        threshold=None,
        tail_topk=20,
        round0_tie_break="random",
        warmup_mode="round_robin",
        warmup_rounds=5,
        seed=0,
        dataset_name="",
        w_scarcity=0.3,
        w_residual=1.0,
        w_forgetting=0.0,
        w_uncertainty=0.0,
        beta=0.9,
        min_hold=8,
        exit_ratio=0.7,
        budget=0,
    ):
        super().__init__(
            num_classes=num_classes,
            rho=rho,
            eps=eps,
            gate_mode=gate_mode,
            temperature=temperature,
            threshold=threshold,
            tail_topk=tail_topk,
            round0_tie_break=round0_tie_break,
            warmup_mode=warmup_mode,
            warmup_rounds=warmup_rounds,
            seed=seed,
            dataset_name=dataset_name,
        )
        self.score_mode = "tail_need"
        self.w_scarcity = float(w_scarcity)
        self.w_residual = float(w_residual)
        self.w_forgetting = float(w_forgetting)
        self.w_uncertainty = float(w_uncertainty)
        self.beta = float(beta)
        self.gate_min_hold = max(0, int(min_hold))
        self.gate_exit_ratio = float(exit_ratio)
        self.gate_budget = int(budget) if int(budget) > 0 else self.tail_topk
        self.gate_budget = max(1, min(self.gate_budget, self.num_classes))

        # Initialize update frequency optimistically. Otherwise never-updated
        # rows dominate scarcity with 1/sqrt(eps) and the gate degenerates into
        # pure round-robin exploration.
        self.update_freq_ema = torch.ones(self.num_classes, dtype=torch.float32)
        self.residual_need_ema = torch.zeros(self.num_classes, dtype=torch.float32)
        self.margin_ema = torch.zeros(self.num_classes, dtype=torch.float32)
        self.best_margin_ema = torch.zeros(self.num_classes, dtype=torch.float32)
        self.confidence_ema = torch.zeros(self.num_classes, dtype=torch.float32)
        self.best_confidence_ema = torch.zeros(self.num_classes, dtype=torch.float32)
        self.tail_need_ema = torch.zeros(self.num_classes, dtype=torch.float32)
        self.scarcity_score = torch.ones(self.num_classes, dtype=torch.float32)
        self.residual_need_score = torch.zeros(self.num_classes, dtype=torch.float32)
        self.forgetting_score = torch.zeros(self.num_classes, dtype=torch.float32)
        self.uncertainty_score = torch.zeros(self.num_classes, dtype=torch.float32)

        self.protected_lifetime = torch.zeros(self.num_classes, dtype=torch.float32)
        self.hold_counter = torch.zeros(self.num_classes, dtype=torch.float32)
        self.protected_mask = torch.zeros(self.num_classes, dtype=torch.bool)
        self.prev_protected_mask = torch.zeros(self.num_classes, dtype=torch.bool)
        self.last_jaccard = 1.0
        self.last_churn_rate = 0.0
        self.last_avg_protected_lifetime = 0.0

    def _normalize_by_mean(self, values):
        values = torch.as_tensor(values, dtype=torch.float32).cpu()
        mean_value = values.mean()
        if mean_value <= self.eps:
            return torch.zeros_like(values)
        return values / (mean_value + self.eps)

    def update_from_energy(self, energy, gate=None):
        energy = torch.as_tensor(energy, dtype=torch.float32).cpu()
        self.last_energy = energy.clone()

        if gate is not None:
            gate = torch.as_tensor(gate, dtype=torch.float32).cpu()
            self.opportunity_count += (gate > self.eps).to(dtype=torch.float32)

        positive = (energy > self.eps).to(dtype=torch.float32)
        positive_energy = energy[positive.bool()]
        if positive_energy.numel() > 0 and positive_energy.mean().item() > self.eps:
            energy_norm = energy / (positive_energy.mean() + self.eps)
        else:
            energy_norm = torch.zeros_like(energy)

        self.update_freq_ema = self.beta * self.update_freq_ema + (1.0 - self.beta) * positive
        self.residual_need_ema = (
            self.beta * self.residual_need_ema
            + (1.0 - self.beta) * energy_norm
        )
        self.scarcity_score = 1.0 / torch.sqrt(self.update_freq_ema + self.eps)
        self.residual_need_score = self._normalize_by_mean(self.residual_need_ema)

        raw_tail_need = (
            self.w_scarcity * self._normalize_by_mean(self.scarcity_score)
            + self.w_residual * self.residual_need_score
            + self.w_forgetting * self.forgetting_score
            + self.w_uncertainty * self.uncertainty_score
        )
        self.tail_need_ema = self.beta * self.tail_need_ema + (1.0 - self.beta) * raw_tail_need
        self.exposure = self.tail_need_ema.clone()
        self.round += 1
        return self.exposure

    def compute_scores(self):
        return self.tail_need_ema.clone()

    def _compute_soft_gate(self, scores, mask):
        if self.gate_mode == "hard_topk":
            return mask.float()
        sorted_scores = torch.sort(scores, descending=True).values
        threshold = self.threshold
        if threshold is None:
            threshold = sorted_scores[min(self.gate_budget, scores.numel()) - 1].item()
        score_std = scores.std(unbiased=False)
        normalized = (scores - scores.mean()) / (score_std + self.eps)
        threshold_norm = (float(threshold) - scores.mean()) / (score_std + self.eps)
        soft_gate = torch.sigmoid((normalized - threshold_norm) / self.temperature)
        return (soft_gate * mask.float()).float()

    def _update_persistence_stats(self, previous, selected):
        union = torch.logical_or(previous, selected).sum().item()
        inter = torch.logical_and(previous, selected).sum().item()
        self.last_jaccard = float(inter / union) if union > 0 else 1.0
        changed = torch.logical_xor(previous, selected).sum().item()
        self.last_churn_rate = float(changed / max(selected.sum().item(), 1.0))

        newly_selected = torch.logical_and(selected, torch.logical_not(previous))
        removed = torch.logical_and(previous, torch.logical_not(selected))
        stayed = torch.logical_and(selected, previous)
        self.protected_lifetime[removed] = 0.0
        self.protected_lifetime[newly_selected] = 1.0
        self.protected_lifetime[stayed] += 1.0
        self.hold_counter[removed] = 0.0
        self.hold_counter[newly_selected] = float(self.gate_min_hold)
        self.hold_counter[stayed] = torch.clamp(self.hold_counter[stayed] - 1.0, min=0.0)
        if selected.sum().item() > 0:
            self.last_avg_protected_lifetime = self.protected_lifetime[selected].mean().item()
        else:
            self.last_avg_protected_lifetime = 0.0

    def compute_gate(self, current_round=0):
        current_round = int(current_round)
        if current_round < self.warmup_rounds and self.warmup_mode != "none":
            mask = compute_warmup_mask(
                self.num_classes,
                self.tail_topk,
                self.warmup_mode,
                seed=self.seed,
                current_round=current_round,
                dataset_name=self.dataset_name,
            )
            scores = mask.float()
            # Warmup remains stateless for the persistent selector. The dynamic
            # protected set starts after warmup from tail_need_ema.
            return mask.float(), scores, mask

        scores = self.compute_scores()
        tied = torch.isclose(scores.max(), scores.min())
        if tied:
            selected = compute_tie_break_mask(
                self.num_classes,
                self.gate_budget,
                mode=self.round0_tie_break,
                seed=self.seed,
                current_round=current_round,
                dataset_name=self.dataset_name,
            )
            previous = self.protected_mask.clone()
            self.prev_protected_mask = previous
            self.protected_mask = selected.clone()
            self._update_persistence_stats(previous, selected)
            return selected.float(), scores, selected

        previous = self.protected_mask.clone()
        sorted_scores, sorted_ids = torch.sort(scores, descending=True)
        rank = torch.empty(self.num_classes, dtype=torch.long)
        rank[sorted_ids] = torch.arange(self.num_classes, dtype=torch.long)
        top_budget_mean = sorted_scores[:self.gate_budget].mean().item()
        exit_threshold = float(self.gate_exit_ratio) * top_budget_mean

        forced_keep = torch.logical_and(previous, self.hold_counter > 0)
        eligible_keep = torch.logical_and(
            previous,
            torch.logical_or(scores >= exit_threshold, rank < 2 * self.gate_budget),
        )
        selected = torch.logical_or(forced_keep, eligible_keep)

        selected_ids = torch.nonzero(selected, as_tuple=False).view(-1)
        if selected_ids.numel() > self.gate_budget:
            forced_ids = torch.nonzero(forced_keep, as_tuple=False).view(-1)
            if forced_ids.numel() >= self.gate_budget:
                keep_order = torch.argsort(scores[forced_ids], descending=True)[:self.gate_budget]
                selected = torch.zeros(self.num_classes, dtype=torch.bool)
                selected[forced_ids[keep_order]] = True
            else:
                remaining_slots = self.gate_budget - forced_ids.numel()
                optional = torch.logical_and(selected, torch.logical_not(forced_keep))
                optional_ids = torch.nonzero(optional, as_tuple=False).view(-1)
                keep_order = torch.argsort(scores[optional_ids], descending=True)[:remaining_slots]
                selected = torch.zeros(self.num_classes, dtype=torch.bool)
                selected[forced_ids] = True
                selected[optional_ids[keep_order]] = True

        if selected.sum().item() < self.gate_budget:
            selected = selected.clone()
            for class_idx in sorted_ids.tolist():
                if selected[class_idx]:
                    continue
                selected[class_idx] = True
                if selected.sum().item() >= self.gate_budget:
                    break

        self.prev_protected_mask = previous
        self.protected_mask = selected.clone()
        self._update_persistence_stats(previous, selected)
        gate = self._compute_soft_gate(scores, selected)
        return gate, scores, selected


class GradientPriorTracker(ExposureTracker):
    """Privacy-friendly gate from class-wise tail-stream gradient proxies.

    The server only observes class-wise tail-stream row updates that are
    already part of model upload. A larger positive row update is interpreted
    as a larger class-prior/exposure proxy, so the gate protects classes with a
    lower prior proxy instead of classes with larger raw gradients.
    """

    def __init__(
        self,
        num_classes,
        rho=0.9,
        eps=1e-6,
        gate_mode="soft",
        temperature=1.0,
        threshold=None,
        tail_topk=20,
        round0_tie_break="random",
        warmup_mode="round_robin",
        warmup_rounds=5,
        seed=0,
        dataset_name="",
        prior_floor=1e-3,
        score_power=0.5,
        lock_rounds=0,
        lock_mode="full_refresh",
        refine_ratio=0.2,
        refine_max_swap=4,
        refine_margin=1.5,
        lock_gate_floor=0.0,
        update_all_rows=False,
    ):
        super().__init__(
            num_classes=num_classes,
            rho=rho,
            eps=eps,
            gate_mode=gate_mode,
            temperature=temperature,
            threshold=threshold,
            tail_topk=tail_topk,
            round0_tie_break=round0_tie_break,
            warmup_mode=warmup_mode,
            warmup_rounds=warmup_rounds,
            seed=seed,
            dataset_name=dataset_name,
        )
        self.score_mode = "gradient_prior"
        self.prior_floor = max(float(prior_floor), float(eps))
        self.score_power = max(float(score_power), float(eps))
        self.lock_rounds = max(0, int(lock_rounds))
        self.lock_mode = str(lock_mode).lower()
        if self.lock_mode not in ("full_refresh", "anchor_refine", "anchor_until_end"):
            raise ValueError(f"Unknown gradient_prior lock_mode: {lock_mode}")
        self.refine_ratio = max(0.0, float(refine_ratio))
        self.refine_max_swap = max(0, int(refine_max_swap))
        self.refine_margin = max(1.0, float(refine_margin))
        self.lock_gate_floor = max(0.0, float(lock_gate_floor))
        self.update_all_rows = bool(update_all_rows)
        self.class_prior_ema = torch.ones(self.num_classes, dtype=torch.float32)
        self.class_prior_proxy = torch.ones(self.num_classes, dtype=torch.float32)
        self.positive_proxy_ema = torch.zeros(self.num_classes, dtype=torch.float32)
        self.last_positive_proxy = torch.zeros(self.num_classes, dtype=torch.float32)
        self.classifier_row_evidence = torch.ones(self.num_classes, dtype=torch.float32)
        self.observed_count = torch.zeros(self.num_classes, dtype=torch.float32)
        self.exposure = self.class_prior_ema.clone()
        self.locked_mask = torch.zeros(self.num_classes, dtype=torch.bool)
        self.locked_until_round = -1
        self.lock_source_round = -1
        self.last_lock_active = False
        self.last_refine_swaps = 0
        self.last_refine_added_ids = []
        self.last_refine_removed_ids = []

    def update_from_gradient_proxy(self, positive_proxy, gate=None):
        positive_proxy = torch.as_tensor(positive_proxy, dtype=torch.float32).cpu()
        self.last_positive_proxy = positive_proxy.clone()
        self.last_energy = positive_proxy.clone()

        if gate is None or self.update_all_rows:
            observed = torch.ones(self.num_classes, dtype=torch.bool)
            self.opportunity_count += observed.to(dtype=torch.float32)
        else:
            gate = torch.as_tensor(gate, dtype=torch.float32).cpu()
            observed = gate > self.eps
            self.opportunity_count += observed.to(dtype=torch.float32)

        if observed.any():
            observed_energy = positive_proxy[observed]
            positive_energy = observed_energy[observed_energy > self.eps]
            evidence = positive_energy.mean() if positive_energy.numel() > 0 else observed_energy.mean()
            if evidence.item() <= self.eps:
                prior_proxy = torch.zeros_like(positive_proxy)
                evidence_value = self.eps
            else:
                prior_proxy = positive_proxy / (evidence + self.eps)
                evidence_value = evidence.item()
            self.class_prior_proxy[observed] = prior_proxy[observed]
            self.positive_proxy_ema[observed] = (
                self.rho * self.positive_proxy_ema[observed]
                + (1.0 - self.rho) * positive_proxy[observed]
            )
            self.class_prior_ema[observed] = (
                self.rho * self.class_prior_ema[observed]
                + (1.0 - self.rho) * prior_proxy[observed]
            )
            self.classifier_row_evidence[observed] = evidence_value
            self.observed_count[observed] += 1.0

        self.class_prior_ema = torch.clamp(self.class_prior_ema, min=0.0)
        self.exposure = self.class_prior_ema.clone()
        self.round += 1
        return self.exposure

    def update_from_energy(self, energy, gate=None):
        # Keep the old server call shape usable. For this tracker, class-wise
        # row-update energy is a prior proxy, not a tail-need score.
        return self.update_from_gradient_proxy(energy, gate=gate)

    def compute_scores(self):
        prior = torch.clamp(self.class_prior_ema, min=self.prior_floor)
        return 1.0 / torch.pow(prior, self.score_power)

    def _compute_gate_from_mask(self, scores, mask):
        if self.gate_mode == "hard_topk":
            return mask.float()
        sorted_scores = torch.sort(scores, descending=True).values
        threshold = self.threshold
        if threshold is None:
            threshold = sorted_scores[self.tail_topk - 1].item()
        score_std = scores.std(unbiased=False)
        normalized = (scores - scores.mean()) / (score_std + self.eps)
        threshold_norm = (float(threshold) - scores.mean()) / (score_std + self.eps)
        soft_gate = torch.sigmoid((normalized - threshold_norm) / self.temperature)
        gate = soft_gate * mask.float()
        if self.lock_gate_floor > 0 and mask.any():
            floor = torch.full_like(gate, min(self.lock_gate_floor, 1.0))
            gate = torch.where(mask, torch.maximum(gate, floor), gate)
        return gate.float()

    def _select_dynamic_mask(self, scores, current_round):
        tied = torch.isclose(scores.max(), scores.min())
        if tied:
            return compute_tie_break_mask(
                self.num_classes,
                self.tail_topk,
                mode=self.round0_tie_break,
                seed=self.seed,
                current_round=current_round,
                dataset_name=self.dataset_name,
            )
        mask = torch.zeros(self.num_classes, dtype=torch.bool)
        mask[torch.topk(scores, self.tail_topk).indices.cpu()] = True
        return mask

    def _refresh_locked_mask(self, scores, dynamic_mask, current_round):
        self.last_refine_swaps = 0
        self.last_refine_added_ids = []
        self.last_refine_removed_ids = []

        lock_missing = self.locked_mask.sum().item() == 0
        if lock_missing or self.lock_mode == "full_refresh":
            self.locked_mask = dynamic_mask.clone()
            refresh_kind = "initialized" if lock_missing else "full_refresh"
        elif self.lock_mode == "anchor_until_end":
            refresh_kind = "kept_anchor"
        else:
            selected = self.locked_mask.clone()
            if self.refine_max_swap > 0:
                max_swap = self.refine_max_swap
            else:
                max_swap = int(math.ceil(self.tail_topk * self.refine_ratio))
            max_swap = max(0, min(max_swap, self.tail_topk))

            protected_ids = torch.nonzero(selected, as_tuple=False).view(-1)
            candidate_ids = torch.nonzero(torch.logical_not(selected), as_tuple=False).view(-1)
            if protected_ids.numel() > 0 and candidate_ids.numel() > 0 and max_swap > 0:
                protected_order = protected_ids[torch.argsort(scores[protected_ids], descending=False)]
                candidate_order = candidate_ids[torch.argsort(scores[candidate_ids], descending=True)]
                for remove_id, add_id in zip(protected_order.tolist(), candidate_order.tolist()):
                    if self.last_refine_swaps >= max_swap:
                        break
                    remove_score = scores[int(remove_id)].item()
                    add_score = scores[int(add_id)].item()
                    if add_score <= remove_score * self.refine_margin + self.eps:
                        break
                    selected[int(remove_id)] = False
                    selected[int(add_id)] = True
                    self.last_refine_removed_ids.append(int(remove_id))
                    self.last_refine_added_ids.append(int(add_id))
                    self.last_refine_swaps += 1
            self.locked_mask = selected
            refresh_kind = "anchor_refine"

        self.lock_source_round = current_round
        if self.lock_mode == "anchor_until_end":
            self.locked_until_round = 10 ** 9
        else:
            self.locked_until_round = current_round + self.lock_rounds
        ids = torch.nonzero(self.locked_mask, as_tuple=False).view(-1).tolist()
        print(
            "FedTEF-v2 gradient_prior lock refreshed "
            f"mode={self.lock_mode}/{refresh_kind}; round={current_round}; "
            f"locked_until={self.locked_until_round}; swaps={self.last_refine_swaps}; "
            f"added={self.last_refine_added_ids}; removed={self.last_refine_removed_ids}; "
            f"classes={ids}"
        )

    def compute_gate(self, current_round=0):
        current_round = int(current_round)
        self.last_lock_active = False
        if current_round < self.warmup_rounds and self.warmup_mode != "none":
            mask = compute_warmup_mask(
                self.num_classes,
                self.tail_topk,
                self.warmup_mode,
                seed=self.seed,
                current_round=current_round,
                dataset_name=self.dataset_name,
            )
            scores = mask.float()
            return mask.float(), scores, mask

        scores = self.compute_scores()
        dynamic_mask = self._select_dynamic_mask(scores, current_round)
        selected = dynamic_mask

        if self.lock_rounds > 0:
            lock_missing = self.locked_mask.sum().item() == 0
            lock_expired = current_round >= self.locked_until_round
            if lock_missing or lock_expired:
                self._refresh_locked_mask(scores, dynamic_mask, current_round)
            selected = self.locked_mask.clone()
            self.last_lock_active = current_round < self.locked_until_round

        gate = self._compute_gate_from_mask(scores, selected)
        return gate, scores, selected


class LowExposureRouterTracker(GradientPriorTracker):
    """V6 low-exposure evidence router.

    This is the FedTEF-v6 gate used for protected evidence routing. It keeps the
    V2/V3 gradient-prior idea, but treats every class-wise positive row update
    as a privacy-friendly prior observation. Large positive row evidence means
    a class was already exposed in local training; protected classes are chosen
    from the inverse prior. Fusion and TailAgg still receive a sparse top-K
    gate, so head classes can continue improving through the shared stream
    without being overwritten by the residual branch.
    """

    def __init__(self, *args, update_all_rows=True, **kwargs):
        super().__init__(*args, update_all_rows=update_all_rows, **kwargs)
        self.score_mode = "low_exposure_router"


class TopologyExposureSurvivalObserver(ExposureTracker):
    """Topology-aware observer for low-exposure tail evidence.

    The observer tracks the four signals from the FedTEF-v10 design:
    exposure mass E, difficulty D, survival ratio S, and age/intermittency A.
    Scores are split into an exposure/age budget and a survival/difficulty
    budget, then stabilized with minimum-hold hysteresis.
    """

    def __init__(
        self,
        num_classes,
        rho=0.9,
        eps=1e-6,
        gate_mode="hard_topk",
        temperature=1.0,
        threshold=None,
        tail_topk=30,
        round0_tie_break="random",
        warmup_mode="all_low",
        warmup_rounds=5,
        seed=0,
        dataset_name="",
        exposure_budget=20,
        survival_budget=10,
        min_hold=5,
        replace_margin=1.2,
        difficulty_power=1.0,
        w_exposure=1.0,
        w_age=0.5,
        w_survival=1.0,
        evidence_threshold=1e-6,
        reliability_floor=0.3,
    ):
        super().__init__(
            num_classes=num_classes,
            rho=rho,
            eps=eps,
            gate_mode=gate_mode,
            temperature=temperature,
            threshold=threshold,
            tail_topk=tail_topk,
            round0_tie_break=round0_tie_break,
            warmup_mode=warmup_mode,
            warmup_rounds=warmup_rounds,
            seed=seed,
            dataset_name=dataset_name,
        )
        self.score_mode = "topology_observer"
        self.exposure_budget = max(0, min(int(exposure_budget), self.num_classes))
        self.survival_budget = max(0, min(int(survival_budget), self.num_classes))
        if self.exposure_budget + self.survival_budget <= 0:
            self.exposure_budget = max(1, min(int(tail_topk), self.num_classes))
        if self.exposure_budget + self.survival_budget > self.num_classes:
            self.survival_budget = self.num_classes - self.exposure_budget
        self.tail_topk = max(1, min(self.exposure_budget + self.survival_budget, self.num_classes))
        self.min_hold = max(0, int(min_hold))
        self.replace_margin = max(1.0, float(replace_margin))
        self.difficulty_power = max(float(difficulty_power), 0.0)
        self.w_exposure = max(float(w_exposure), 0.0)
        self.w_age = max(float(w_age), 0.0)
        self.w_survival = max(float(w_survival), 0.0)
        self.evidence_threshold = max(float(evidence_threshold), float(eps))
        self.reliability_floor = min(max(float(reliability_floor), 0.0), 1.0)

        self.exposure_mass = torch.zeros(self.num_classes, dtype=torch.float32)
        self.difficulty = torch.zeros(self.num_classes, dtype=torch.float32)
        self.survival = torch.ones(self.num_classes, dtype=torch.float32)
        self.age = torch.zeros(self.num_classes, dtype=torch.float32)
        self.low_exposure_score = torch.ones(self.num_classes, dtype=torch.float32)
        self.age_score = torch.zeros(self.num_classes, dtype=torch.float32)
        self.low_survival_score = torch.zeros(self.num_classes, dtype=torch.float32)
        self.exposure_component = torch.zeros(self.num_classes, dtype=torch.float32)
        self.survival_component = torch.zeros(self.num_classes, dtype=torch.float32)
        self.topology_score = torch.zeros(self.num_classes, dtype=torch.float32)
        self.reliability = self.survival.clone()

        self.protected_mask = torch.zeros(self.num_classes, dtype=torch.bool)
        self.hold_counter = torch.zeros(self.num_classes, dtype=torch.float32)
        self.protected_lifetime = torch.zeros(self.num_classes, dtype=torch.float32)
        self.last_jaccard = 1.0
        self.last_churn_rate = 0.0
        self.last_avg_protected_lifetime = 0.0
        self.last_exposure_ids = []
        self.last_survival_ids = []

    def _normalize_positive(self, values):
        values = torch.as_tensor(values, dtype=torch.float32).cpu()
        positive = values[values > self.eps]
        if positive.numel() == 0:
            return torch.zeros_like(values)
        return values / (positive.mean() + self.eps)

    def _inverse_normalized(self, values):
        norm = self._normalize_positive(values)
        if norm.max().item() <= self.eps:
            return torch.ones_like(norm)
        return 1.0 / (norm + self.eps)

    def update(
        self,
        exposure_proxy,
        difficulty_proxy=None,
        survival_ratio=None,
        gate=None,
    ):
        exposure_proxy = torch.as_tensor(exposure_proxy, dtype=torch.float32).cpu()
        if difficulty_proxy is None:
            difficulty_proxy = torch.zeros_like(exposure_proxy)
        difficulty_proxy = torch.as_tensor(difficulty_proxy, dtype=torch.float32).cpu()
        if survival_ratio is None:
            survival_ratio = torch.ones_like(exposure_proxy)
        survival_ratio = torch.as_tensor(survival_ratio, dtype=torch.float32).cpu()
        survival_ratio = torch.clamp(survival_ratio, min=0.0, max=1.0)

        observed = exposure_proxy > self.evidence_threshold
        self.age += 1.0
        self.age[observed] = 0.0
        self.opportunity_count += observed.to(dtype=torch.float32)
        self.observed_count = getattr(
            self,
            "observed_count",
            torch.zeros(self.num_classes, dtype=torch.float32),
        )
        self.observed_count += observed.to(dtype=torch.float32)

        self.exposure_mass = self.rho * self.exposure_mass + (1.0 - self.rho) * exposure_proxy
        self.difficulty = self.rho * self.difficulty + (1.0 - self.rho) * difficulty_proxy
        self.survival = self.rho * self.survival + (1.0 - self.rho) * survival_ratio

        self.low_exposure_score = self._inverse_normalized(self.exposure_mass)
        max_age = torch.clamp(self.age.max(), min=1.0)
        self.age_score = self.age / max_age
        self.low_survival_score = torch.clamp(1.0 - self.survival, min=0.0, max=1.0)

        difficulty_gate = self._normalize_positive(self.difficulty)
        if self.difficulty_power != 1.0:
            difficulty_gate = torch.pow(
                torch.clamp(difficulty_gate, min=0.0),
                self.difficulty_power,
            )
        self.exposure_component = difficulty_gate * (
            self.w_exposure * self.low_exposure_score
            + self.w_age * self.age_score
        )
        self.survival_component = difficulty_gate * (
            self.w_survival * self.low_survival_score
        )
        self.topology_score = self.exposure_component + self.survival_component
        self.exposure = self.topology_score.clone()
        self.last_energy = exposure_proxy.clone()
        # The model applies the release floor at prediction time. Keep the
        # observer output as raw survival reliability to avoid double flooring.
        self.reliability = torch.clamp(self.survival.clone(), min=0.0, max=1.0)
        self.round += 1
        return self.exposure

    def update_from_energy(self, energy, gate=None):
        return self.update(energy, gate=gate)

    def compute_scores(self):
        return self.topology_score.clone()

    def _select_budget(self, scores, budget, excluded=None):
        selected = torch.zeros(self.num_classes, dtype=torch.bool)
        if budget <= 0:
            return selected
        candidate_scores = scores.clone()
        if excluded is not None:
            candidate_scores[excluded] = -float("inf")
        if torch.isinf(candidate_scores).all():
            return selected
        k = min(int(budget), int((~torch.isinf(candidate_scores)).sum().item()))
        if k <= 0:
            return selected
        selected[torch.topk(candidate_scores, k).indices.cpu()] = True
        return selected

    def _apply_hysteresis(self, proposed):
        previous = self.protected_mask.clone()
        forced = torch.logical_and(previous, self.hold_counter > 0)
        selected = previous.clone()

        new_ids = torch.nonzero(torch.logical_and(proposed, ~previous), as_tuple=False).view(-1)
        if new_ids.numel() > 0:
            new_ids = new_ids[torch.argsort(self.topology_score[new_ids], descending=True)]
            for new_id in new_ids.tolist():
                if selected.sum().item() < self.tail_topk:
                    selected[new_id] = True
                    continue
                replaceable = torch.nonzero(torch.logical_and(selected, ~forced), as_tuple=False).view(-1)
                if replaceable.numel() == 0:
                    break
                weakest = replaceable[torch.argmin(self.topology_score[replaceable])]
                weakest_score = self.topology_score[weakest]
                new_score = self.topology_score[int(new_id)]
                if weakest_score <= self.eps or new_score > self.replace_margin * weakest_score:
                    selected[weakest] = False
                    selected[int(new_id)] = True

        selected = torch.logical_or(selected, forced)

        if selected.sum().item() > self.tail_topk:
            forced_ids = torch.nonzero(forced, as_tuple=False).view(-1)
            optional_ids = torch.nonzero(torch.logical_and(selected, ~forced), as_tuple=False).view(-1)
            new_selected = torch.zeros(self.num_classes, dtype=torch.bool)
            if forced_ids.numel() >= self.tail_topk:
                keep = forced_ids[torch.argsort(self.topology_score[forced_ids], descending=True)[:self.tail_topk]]
                new_selected[keep] = True
            else:
                new_selected[forced_ids] = True
                slots = self.tail_topk - forced_ids.numel()
                if optional_ids.numel() > 0 and slots > 0:
                    keep = optional_ids[torch.argsort(self.topology_score[optional_ids], descending=True)[:slots]]
                    new_selected[keep] = True
            selected = new_selected

        if selected.sum().item() < self.tail_topk:
            sorted_ids = torch.argsort(self.topology_score, descending=True)
            for class_idx in sorted_ids.tolist():
                if selected[class_idx]:
                    continue
                selected[class_idx] = True
                if selected.sum().item() >= self.tail_topk:
                    break

        removed = torch.logical_and(previous, ~selected)
        new = torch.logical_and(selected, ~previous)
        stayed = torch.logical_and(selected, previous)
        self.protected_lifetime[removed] = 0.0
        self.protected_lifetime[new] = 1.0
        self.protected_lifetime[stayed] += 1.0
        self.hold_counter[removed] = 0.0
        self.hold_counter[new] = float(self.min_hold)
        self.hold_counter[stayed] = torch.clamp(self.hold_counter[stayed] - 1.0, min=0.0)
        union = torch.logical_or(previous, selected).sum().item()
        inter = torch.logical_and(previous, selected).sum().item()
        self.last_jaccard = float(inter / union) if union > 0 else 1.0
        self.last_churn_rate = float(torch.logical_xor(previous, selected).sum().item() / max(selected.sum().item(), 1.0))
        self.last_avg_protected_lifetime = (
            self.protected_lifetime[selected].mean().item()
            if selected.any()
            else 0.0
        )
        self.protected_mask = selected.clone()
        return selected

    def _compute_gate_from_mask(self, scores, mask):
        if self.gate_mode == "hard_topk":
            return mask.float()
        if not mask.any():
            return mask.float()
        selected_scores = scores[mask]
        threshold = self.threshold
        if threshold is None:
            threshold = selected_scores.min().item()
        score_std = scores.std(unbiased=False)
        normalized = (scores - scores.mean()) / (score_std + self.eps)
        threshold_norm = (float(threshold) - scores.mean()) / (score_std + self.eps)
        soft_gate = torch.sigmoid((normalized - threshold_norm) / self.temperature)
        return (soft_gate * mask.float()).float()

    def compute_gate(self, current_round=0):
        current_round = int(current_round)
        if current_round < self.warmup_rounds and self.warmup_mode != "none":
            mask = compute_warmup_mask(
                self.num_classes,
                self.tail_topk,
                self.warmup_mode,
                seed=self.seed,
                current_round=current_round,
                dataset_name=self.dataset_name,
            )
            self.protected_mask = mask.clone()
            return mask.float(), mask.float(), mask

        exposure_mask = self._select_budget(self.exposure_component, self.exposure_budget)
        survival_mask = self._select_budget(
            self.survival_component,
            self.survival_budget,
            excluded=exposure_mask,
        )
        proposed = torch.logical_or(exposure_mask, survival_mask)
        if proposed.sum().item() == 0:
            proposed = self._select_budget(self.topology_score, self.tail_topk)
        selected = self._apply_hysteresis(proposed)
        self.last_exposure_ids = torch.nonzero(exposure_mask, as_tuple=False).view(-1).tolist()
        self.last_survival_ids = torch.nonzero(survival_mask, as_tuple=False).view(-1).tolist()
        gate = self._compute_gate_from_mask(self.topology_score, selected)
        return gate, self.topology_score.clone(), selected


class EvidenceMemoryTracker(ExposureTracker):
    """Continuous gate for persistent semantic memory under sparse evidence.

    The tracker consumes positive-only tail-stream row updates. It does not
    estimate or upload class counts. Exploration comes from observed positive
    row evidence, while residual fusion remains sparse over protected classes.
    """

    def __init__(
        self,
        num_classes,
        rho=0.9,
        eps=1e-6,
        gate_mode="soft",
        temperature=1.0,
        threshold=None,
        tail_topk=20,
        round0_tie_break="random",
        warmup_mode="all_low",
        warmup_rounds=5,
        seed=0,
        dataset_name="",
        reliability_tau=2.0,
        gate_floor=0.05,
        residual_weight=0.25,
    ):
        super().__init__(
            num_classes=num_classes,
            rho=rho,
            eps=eps,
            gate_mode=gate_mode,
            temperature=temperature,
            threshold=threshold,
            tail_topk=tail_topk,
            round0_tie_break=round0_tie_break,
            warmup_mode=warmup_mode,
            warmup_rounds=warmup_rounds,
            seed=seed,
            dataset_name=dataset_name,
        )
        self.score_mode = "evidence_memory"
        self.reliability_tau = max(float(reliability_tau), self.eps)
        self.gate_floor = min(max(float(gate_floor), 0.0), 1.0)
        self.residual_weight = max(float(residual_weight), 0.0)
        self.evidence_ema = torch.zeros(self.num_classes, dtype=torch.float32)
        self.observation_ema = torch.zeros(self.num_classes, dtype=torch.float32)
        self.residual_energy_ema = torch.zeros(self.num_classes, dtype=torch.float32)
        self.reliability = torch.zeros(self.num_classes, dtype=torch.float32)
        self.scarcity_score = torch.zeros(self.num_classes, dtype=torch.float32)
        self.tail_need_ema = torch.zeros(self.num_classes, dtype=torch.float32)
        self.last_positive_proxy = torch.zeros(self.num_classes, dtype=torch.float32)
        self.observed_count = torch.zeros(self.num_classes, dtype=torch.float32)
        self.protected_mask = torch.zeros(self.num_classes, dtype=torch.bool)

    def _normalize_positive(self, values):
        values = torch.as_tensor(values, dtype=torch.float32).cpu()
        positive = values[values > self.eps]
        if positive.numel() == 0:
            return torch.zeros_like(values)
        return values / (positive.mean() + self.eps)

    def update_from_evidence(self, positive_proxy, aggregated_energy=None, gate=None):
        positive_proxy = torch.as_tensor(positive_proxy, dtype=torch.float32).cpu()
        if aggregated_energy is None:
            aggregated_energy = positive_proxy
        aggregated_energy = torch.as_tensor(aggregated_energy, dtype=torch.float32).cpu()
        observed = positive_proxy > self.eps

        self.last_positive_proxy = positive_proxy.clone()
        self.last_energy = aggregated_energy.clone()
        self.observed_count += observed.to(dtype=torch.float32)
        self.opportunity_count += observed.to(dtype=torch.float32)

        evidence_norm = self._normalize_positive(positive_proxy)
        residual_norm = self._normalize_positive(aggregated_energy)
        observed_float = observed.to(dtype=torch.float32)
        self.evidence_ema = self.rho * self.evidence_ema + (1.0 - self.rho) * evidence_norm
        self.observation_ema = self.rho * self.observation_ema + (1.0 - self.rho) * observed_float
        self.residual_energy_ema = (
            self.rho * self.residual_energy_ema
            + (1.0 - self.rho) * residual_norm
        )
        self.reliability = 1.0 - torch.exp(-self.observed_count / self.reliability_tau)
        self.scarcity_score = 1.0 / torch.sqrt(self.observation_ema + self.eps)
        scarcity_norm = self.scarcity_score / (self.scarcity_score.mean() + self.eps)
        residual_need = self.residual_energy_ema / (self.residual_energy_ema.mean() + self.eps)
        self.tail_need_ema = self.reliability * (
            scarcity_norm + self.residual_weight * residual_need
        )
        self.exposure = self.evidence_ema.clone()
        self.round += 1
        return self.exposure

    def update_from_energy(self, energy, gate=None):
        return self.update_from_evidence(energy, aggregated_energy=energy, gate=gate)

    def compute_scores(self):
        return self.tail_need_ema.clone()

    def compute_gate(self, current_round=0):
        current_round = int(current_round)
        if current_round < self.warmup_rounds and self.warmup_mode != "none":
            mask = compute_warmup_mask(
                self.num_classes,
                self.tail_topk,
                self.warmup_mode,
                seed=self.seed,
                current_round=current_round,
                dataset_name=self.dataset_name,
            )
            gate = mask.float()
            return gate, gate.clone(), mask

        scores = self.compute_scores()
        topk = max(1, min(int(self.tail_topk), self.num_classes))
        if torch.isclose(scores.max(), scores.min()):
            mask = compute_tie_break_mask(
                self.num_classes,
                topk,
                mode=self.round0_tie_break,
                seed=self.seed,
                current_round=current_round,
                dataset_name=self.dataset_name,
            )
        else:
            mask = torch.zeros(self.num_classes, dtype=torch.bool)
            mask[torch.topk(scores, topk).indices.cpu()] = True

        if self.gate_mode == "hard_topk":
            gate = mask.float()
        else:
            if torch.isclose(scores.max(), scores.min()):
                soft_gate = mask.float()
            else:
                sorted_scores = torch.sort(scores, descending=True).values
                threshold = self.threshold
                if threshold is None:
                    threshold = sorted_scores[topk - 1].item()
                score_std = scores.std(unbiased=False)
                normalized = (scores - scores.mean()) / (score_std + self.eps)
                threshold_norm = (float(threshold) - scores.mean()) / (score_std + self.eps)
                soft_gate = torch.sigmoid((normalized - threshold_norm) / self.temperature)

            # EvidenceMemory may still learn from every observed positive row
            # through EVIDENCE_MEMORY_UPDATE_ALL_ROWS, but residual fusion and
            # TailAgg protection must stay sparse.
            gate = soft_gate * mask.float()
        self.protected_mask = mask.clone()
        return gate.float(), scores, mask


def is_tail_stream_key(key):
    return (
        key.startswith("tail_stream.")
        or ".tail_stream." in key
        or key == "routed_prompt_delta"
        or key.endswith(".routed_prompt_delta")
    )


def is_shared_stream_key(key, train_img_adap=False, train_lora=False):
    if key.startswith("prompt_learner.") or ".prompt_learner." in key:
        return True
    if train_img_adap and (key.startswith("img_adap.") or ".img_adap." in key):
        return True
    if train_lora and "lora_" in key:
        return True
    return False


def fedavg_keys(global_weights, local_weights, idxs_users, datanumber_client, keys):
    total_weight = sum([datanumber_client[int(idx)] for idx in idxs_users])
    for key in keys:
        temp = torch.zeros_like(global_weights[key])
        for client_idx in idxs_users:
            temp += (datanumber_client[int(client_idx)] / total_weight) * local_weights[int(client_idx)][key]
        global_weights[key] = temp
    return global_weights


def compute_tail_stream_gradient_prior_proxy(
    global_weights,
    local_weights,
    idxs_users,
    num_classes,
    eps=1e-6,
):
    """Estimate class prior from class-wise tail-stream row updates.

    This uses only model-update rows that clients already upload. Because the
    FedTEF-v2 tail-stream gradient mask keeps only positive protected rows, the
    resulting row energy is a positive-label classifier evidence proxy rather
    than an explicit class-count vector.
    """

    # The low-exposure prior should come from the positive classifier/evidence
    # stream, not from already-routed protected prompt rows. Routed prompt
    # deltas are TailAgg-protected below, but excluding them here avoids a
    # feedback loop where the protected set defines its own prior.
    tail_keys = [
        key for key in global_weights.keys()
        if key.startswith("tail_stream.") or ".tail_stream." in key
    ]
    classwise_keys = [
        key for key in tail_keys
        if global_weights[key].ndim >= 1 and global_weights[key].shape[0] == num_classes
    ]
    positive_proxy = torch.zeros(num_classes, dtype=torch.float32)
    client_observed = torch.zeros(num_classes, dtype=torch.float32)
    if not classwise_keys:
        print("FedTEF-v2 gradient-prior warning: no class-wise tail-stream keys found.")
        return positive_proxy, client_observed

    old_tail = {key: global_weights[key].detach().cpu().float() for key in classwise_keys}
    for client_idx in idxs_users:
        client_idx = int(client_idx)
        client_energy = torch.zeros(num_classes, dtype=torch.float32)
        for key in classwise_keys:
            delta = (local_weights[client_idx][key].detach().cpu().float() - old_tail[key])
            flat = delta.reshape(num_classes, -1)
            client_energy += torch.linalg.vector_norm(flat, dim=1)
        positive_proxy += client_energy
        client_observed += (client_energy > float(eps)).to(dtype=torch.float32)

    return positive_proxy, client_observed


def compute_tail_stream_positive_update_stats(
    global_weights,
    local_weights,
    idxs_users,
    num_classes,
    eps=1e-6,
):
    """Compute positive row-update mass and survival ratio for FedTEF-v10.

    For each class c, exposure mass is sum_k ||u^+_{k,c}|| and survival is
    ||sum_k u^+_{k,c}|| / (sum_k ||u^+_{k,c}|| + eps). Positivity is enforced
    before upload by the client-side tail-row mask; the server only reads the
    class-wise tail-stream rows that clients already upload.
    """

    tail_keys = [
        key for key in global_weights.keys()
        if key.startswith("tail_stream.") or ".tail_stream." in key
    ]
    classwise_keys = [
        key for key in tail_keys
        if global_weights[key].ndim >= 1 and global_weights[key].shape[0] == num_classes
    ]
    positive_proxy = torch.zeros(num_classes, dtype=torch.float32)
    client_observed = torch.zeros(num_classes, dtype=torch.float32)
    survival_numerator = torch.zeros(num_classes, dtype=torch.float32)
    if not classwise_keys:
        print("FedTEF-v10 observer warning: no class-wise tail-stream keys found.")
        return positive_proxy, client_observed, torch.ones(num_classes, dtype=torch.float32)

    old_tail = {key: global_weights[key].detach().cpu().float() for key in classwise_keys}
    class_delta_sum = None
    for client_idx in idxs_users:
        client_idx = int(client_idx)
        client_energy = torch.zeros(num_classes, dtype=torch.float32)
        client_flat_parts = []
        for key in classwise_keys:
            delta = local_weights[client_idx][key].detach().cpu().float() - old_tail[key]
            flat = delta.reshape(num_classes, -1)
            client_energy += torch.linalg.vector_norm(flat, dim=1)
            client_flat_parts.append(flat)
        client_flat = torch.cat(client_flat_parts, dim=1)
        if class_delta_sum is None:
            class_delta_sum = torch.zeros_like(client_flat)
        class_delta_sum += client_flat
        positive_proxy += client_energy
        client_observed += (client_energy > float(eps)).to(dtype=torch.float32)

    if class_delta_sum is not None:
        survival_numerator = torch.linalg.vector_norm(class_delta_sum, dim=1)
    survival_ratio = survival_numerator / (positive_proxy + float(eps))
    observed = positive_proxy > float(eps)
    survival_ratio = torch.where(observed, survival_ratio, torch.ones_like(survival_ratio))
    survival_ratio = torch.clamp(survival_ratio, min=0.0, max=1.0)
    return positive_proxy, client_observed, survival_ratio


def fedtef_v10_evidence_preserving_tailagg(
    global_weights,
    local_weights,
    idxs_users,
    gate,
    num_classes,
    survival_ratio=None,
    evidence_threshold=1e-6,
    update_clip=10.0,
    base_momentum=0.6,
    low_survival_momentum=0.25,
    eps=1e-6,
    return_diagnostics=False,
):
    """Evidence-preserving row aggregation for FedTEF-v10.

    Rows with no positive evidence are kept exactly. Rows with evidence are
    updated only from clients whose positive row update is non-zero, weighted by
    clipped update mass. Low-survival rows use a smaller server step.
    """

    old_global = copy.deepcopy(global_weights)
    tail_keys = [key for key in global_weights.keys() if is_tail_stream_key(key)]
    classwise_keys = [
        key for key in tail_keys
        if global_weights[key].ndim >= 1 and global_weights[key].shape[0] == num_classes
    ]
    non_classwise_keys = [key for key in tail_keys if key not in classwise_keys]
    gate = torch.as_tensor(gate, dtype=torch.float32).cpu()
    if survival_ratio is None:
        survival_ratio = torch.ones(num_classes, dtype=torch.float32)
    survival_ratio = torch.as_tensor(survival_ratio, dtype=torch.float32).cpu().clamp(0.0, 1.0)
    evidence_threshold = max(float(evidence_threshold), float(eps))
    update_clip = max(float(update_clip), float(eps))
    base_momentum = min(max(float(base_momentum), 0.0), 1.0)
    low_survival_momentum = min(max(float(low_survival_momentum), 0.0), 1.0)

    client_energy = {
        int(client_idx): torch.zeros(num_classes, dtype=torch.float32)
        for client_idx in idxs_users
    }
    for client_idx in idxs_users:
        client_idx = int(client_idx)
        for key in classwise_keys:
            delta = (
                local_weights[client_idx][key].detach().cpu()
                - old_global[key].detach().cpu()
            ).float()
            flat = delta.reshape(num_classes, -1)
            client_energy[client_idx] += torch.linalg.vector_norm(flat, dim=1)

    aggregated_energy = torch.zeros(num_classes, dtype=torch.float32)
    kept_row_tensors = 0
    updated_row_tensors = 0
    kept_class_mask = torch.zeros(num_classes, dtype=torch.bool)
    updated_class_mask = torch.zeros(num_classes, dtype=torch.bool)
    row_momentum = torch.zeros(num_classes, dtype=torch.float32)
    for key in classwise_keys:
        old_value = old_global[key]
        new_value = old_value.clone()
        for class_idx in range(num_classes):
            norms = torch.tensor(
                [client_energy[int(client_idx)][class_idx].item() for client_idx in idxs_users],
                dtype=torch.float32,
            )
            valid = norms > evidence_threshold
            if not valid.any() or gate[class_idx].item() <= eps:
                kept_row_tensors += 1
                kept_class_mask[class_idx] = True
                continue

            weights = torch.clamp(norms[valid], max=update_clip) + eps
            weights = weights / weights.sum()
            agg_value = torch.zeros_like(old_value[class_idx])
            valid_clients = [int(idx) for idx, keep in zip(idxs_users, valid.tolist()) if keep]
            for pos, client_idx in enumerate(valid_clients):
                agg_value += weights[pos].to(
                    device=agg_value.device,
                    dtype=agg_value.dtype,
                ) * local_weights[client_idx][key][class_idx]
            survival = survival_ratio[class_idx].item()
            momentum = low_survival_momentum + (base_momentum - low_survival_momentum) * survival
            momentum = min(max(momentum, low_survival_momentum), base_momentum)
            new_value[class_idx] = (1.0 - momentum) * old_value[class_idx] + momentum * agg_value
            row_momentum[class_idx] = float(momentum)
            aggregated_energy[class_idx] += (
                new_value[class_idx].detach().float().cpu()
                - old_value[class_idx].detach().float().cpu()
            ).norm()
            updated_row_tensors += 1
            updated_class_mask[class_idx] = True
        global_weights[key] = new_value

    if non_classwise_keys:
        global_weights = fedavg_keys(
            global_weights,
            local_weights,
            idxs_users,
            [1 for _ in range(max(idxs_users) + 1)],
            non_classwise_keys,
        )

    client_energy_matrix = (
        torch.stack([client_energy[int(client_idx)] for client_idx in idxs_users], dim=0)
        if len(idxs_users) > 0
        else torch.zeros(0, num_classes)
    )
    observed_client_count = (
        (client_energy_matrix > evidence_threshold).sum(dim=0).float()
        if client_energy_matrix.numel()
        else torch.zeros(num_classes, dtype=torch.float32)
    )
    local_energy_sum = (
        client_energy_matrix.sum(dim=0)
        if client_energy_matrix.numel()
        else torch.zeros(num_classes, dtype=torch.float32)
    )
    diagnostics = {
        "mode": "evidence_preserving",
        "fallback_count": int(kept_row_tensors),
        "kept_row_tensors": int(kept_row_tensors),
        "updated_rows": int(updated_row_tensors),
        "updated_row_tensors": int(updated_row_tensors),
        "kept_classes": int(kept_class_mask.sum().item()),
        "updated_classes": int(updated_class_mask.sum().item()),
        "local_energy_sum": local_energy_sum,
        "observed_client_count": observed_client_count,
        "tailagg_row_energy": aggregated_energy,
        "fedavg_row_energy": torch.zeros(num_classes, dtype=torch.float32),
        "memory_row_norm": torch.zeros(num_classes, dtype=torch.float32),
        "survival_ratio": survival_ratio,
        "row_momentum": row_momentum,
    }
    for key in classwise_keys:
        rows = global_weights[key].detach().float().cpu().reshape(num_classes, -1)
        diagnostics["memory_row_norm"] += torch.linalg.vector_norm(rows, dim=1)
    diagnostics["local_energy_mean_observed"] = local_energy_sum / torch.clamp(
        observed_client_count,
        min=1.0,
    )

    print(
        "FedTEF-v10 evidence-preserving TailAgg "
        f"updated/kept row_tensors: {updated_row_tensors}/{kept_row_tensors}; "
        f"updated/kept classes: {int(updated_class_mask.sum().item())}/{int(kept_class_mask.sum().item())}; "
        f"energy min/max/mean: {aggregated_energy.min().item():.6f}/"
        f"{aggregated_energy.max().item():.6f}/"
        f"{aggregated_energy.mean().item():.6f}"
    )
    if return_diagnostics:
        return global_weights, aggregated_energy, diagnostics
    return global_weights, aggregated_energy


def fedtef_v2_tailagg(
    global_weights,
    local_weights,
    idxs_users,
    datanumber_client,
    gate,
    num_classes,
    mode="row_update_norm",
    fallback="fedavg_or_keep",
    conflict_gamma=1.0,
    min_agreement=-1.0,
    memory_momentum=1.0,
    eps=1e-6,
    return_diagnostics=False,
):
    old_global = copy.deepcopy(global_weights)
    tail_keys = [key for key in global_weights.keys() if is_tail_stream_key(key)]
    classwise_keys = [
        key for key in tail_keys
        if global_weights[key].ndim >= 1 and global_weights[key].shape[0] == num_classes
    ]
    non_classwise_keys = [key for key in tail_keys if key not in classwise_keys]
    gate = torch.as_tensor(gate, dtype=torch.float32).cpu()
    mode = str(mode).lower()
    evidence_memory_mode = mode == "evidence_memory"
    conflict_gamma = max(float(conflict_gamma), 0.0)
    min_agreement = float(min_agreement)
    memory_momentum = min(max(float(memory_momentum), 0.0), 1.0)
    client_energy = {
        int(client_idx): torch.zeros(num_classes, dtype=torch.float32)
        for client_idx in idxs_users
    }
    for client_idx in idxs_users:
        client_idx = int(client_idx)
        for key in classwise_keys:
            delta = (local_weights[client_idx][key].detach().cpu() - old_global[key].detach().cpu()).float()
            flat = delta.reshape(num_classes, -1)
            client_energy[client_idx] += torch.linalg.vector_norm(flat, dim=1)

    aggregated_energy = torch.zeros(num_classes, dtype=torch.float32)
    fallback_count = 0
    for key in classwise_keys:
        old_value = old_global[key]
        new_value = old_value.clone()
        for class_idx in range(num_classes):
            norms = torch.tensor(
                [client_energy[int(client_idx)][class_idx].item() for client_idx in idxs_users],
                dtype=torch.float32,
            )
            if not evidence_memory_mode and gate[class_idx].item() <= eps:
                fallback_count += 1
                continue
            if norms.sum().item() <= eps:
                fallback_count += 1
                if not evidence_memory_mode and str(fallback).lower() == "fedavg_or_keep":
                    temp = torch.zeros_like(old_value[class_idx])
                    total_weight = sum([datanumber_client[int(idx)] for idx in idxs_users])
                    for client_idx in idxs_users:
                        temp += (datanumber_client[int(client_idx)] / total_weight) * local_weights[int(client_idx)][key][class_idx]
                    new_value[class_idx] = temp
                continue
            if mode in ("conflict_aware", "evidence_memory"):
                deltas = []
                for client_idx in idxs_users:
                    delta = local_weights[int(client_idx)][key][class_idx] - old_value[class_idx]
                    deltas.append(delta.detach().float().reshape(-1).cpu())
                stacked = torch.stack(deltas, dim=0)
                mean_delta = stacked.mean(dim=0)
                mean_norm = torch.linalg.vector_norm(mean_delta)
                if mean_norm.item() <= eps:
                    agreements = torch.zeros_like(norms)
                else:
                    agreements = F.cosine_similarity(
                        stacked,
                        mean_delta.view(1, -1).expand_as(stacked),
                        dim=1,
                        eps=eps,
                    )
                mean_agreement = agreements.mean().item()
                if mean_agreement < min_agreement:
                    fallback_count += 1
                    if str(fallback).lower() == "fedavg_or_keep":
                        temp = torch.zeros_like(old_value[class_idx])
                        total_weight = sum([datanumber_client[int(idx)] for idx in idxs_users])
                        for client_idx in idxs_users:
                            temp += (datanumber_client[int(client_idx)] / total_weight) * local_weights[int(client_idx)][key][class_idx]
                        new_value[class_idx] = temp
                    continue
                agreement_weight = torch.clamp(agreements, min=0.0)
                if conflict_gamma > 0:
                    agreement_weight = torch.pow(agreement_weight + eps, conflict_gamma)
                weights = norms * agreement_weight + eps
            else:
                weights = gate[class_idx].item() * norms + eps
            weights = weights / weights.sum()
            row_delta = torch.zeros_like(old_value[class_idx])
            for pos, client_idx in enumerate(idxs_users):
                delta = local_weights[int(client_idx)][key][class_idx] - old_value[class_idx]
                row_delta += weights[pos].to(delta.device, dtype=delta.dtype) * delta
            if evidence_memory_mode:
                row_delta = memory_momentum * row_delta
            new_value[class_idx] = old_value[class_idx] + row_delta
            aggregated_energy[class_idx] += row_delta.detach().float().norm().cpu()
        global_weights[key] = new_value

    if non_classwise_keys:
        print(f"FedTEF-v2 TailAgg: FedAvg for non class-wise tail params: {non_classwise_keys}")
        global_weights = fedavg_keys(global_weights, local_weights, idxs_users, datanumber_client, non_classwise_keys)

    fedavg_norm = 0.0
    tailagg_norm = 0.0
    fedavg_row_energy = torch.zeros(num_classes, dtype=torch.float32)
    tailagg_row_energy = torch.zeros(num_classes, dtype=torch.float32)
    fedavg_reference = copy.deepcopy(old_global)
    fedavg_reference = fedavg_keys(fedavg_reference, local_weights, idxs_users, datanumber_client, tail_keys)
    for key in tail_keys:
        fedavg_norm += (fedavg_reference[key].detach().float().cpu() - old_global[key].detach().float().cpu()).norm().item()
        tailagg_norm += (global_weights[key].detach().float().cpu() - old_global[key].detach().float().cpu()).norm().item()
        if key in classwise_keys:
            fedavg_delta = (
                fedavg_reference[key].detach().float().cpu()
                - old_global[key].detach().float().cpu()
            ).reshape(num_classes, -1)
            tailagg_delta = (
                global_weights[key].detach().float().cpu()
                - old_global[key].detach().float().cpu()
            ).reshape(num_classes, -1)
            fedavg_row_energy += torch.linalg.vector_norm(fedavg_delta, dim=1)
            tailagg_row_energy += torch.linalg.vector_norm(tailagg_delta, dim=1)

    print_fedtef_v2_tailagg_diagnostics(aggregated_energy, fallback_count, tailagg_norm, fedavg_norm, mode=mode)
    if not return_diagnostics:
        return global_weights, aggregated_energy

    client_energy_matrix = torch.stack(
        [client_energy[int(client_idx)] for client_idx in idxs_users],
        dim=0,
    )
    observed_client_count = (client_energy_matrix > float(eps)).sum(dim=0).float()
    local_energy_sum = client_energy_matrix.sum(dim=0)
    local_energy_mean_observed = local_energy_sum / torch.clamp(observed_client_count, min=1.0)
    memory_row_norm = torch.zeros(num_classes, dtype=torch.float32)
    for key in classwise_keys:
        rows = global_weights[key].detach().float().cpu().reshape(num_classes, -1)
        memory_row_norm += torch.linalg.vector_norm(rows, dim=1)

    diagnostics = {
        "mode": mode,
        "fallback_count": int(fallback_count),
        "tailagg_norm": float(tailagg_norm),
        "fedavg_norm": float(fedavg_norm),
        "local_energy_sum": local_energy_sum,
        "local_energy_mean_observed": local_energy_mean_observed,
        "observed_client_count": observed_client_count,
        "fedavg_row_energy": fedavg_row_energy,
        "tailagg_row_energy": tailagg_row_energy,
        "memory_row_norm": memory_row_norm,
    }
    return global_weights, aggregated_energy, diagnostics


def print_fedtef_v2_tailagg_diagnostics(energy, fallback_count, tailagg_norm, fedavg_norm, mode="row_update_norm"):
    energy = energy.float()
    topk = min(10, energy.numel())
    top_ids = torch.topk(energy, topk).indices.tolist()
    near_zero = int((energy <= 1e-8).sum().item())
    print(f"FedTEF-v2 TailAgg mode: {mode}")
    print(
        "FedTEF-v2 TailAgg row update norm "
        f"min/max/mean: {energy.min().item():.6f}/"
        f"{energy.max().item():.6f}/"
        f"{energy.mean().item():.6f}"
    )
    print(f"FedTEF-v2 TailAgg top updated classes: {top_ids}")
    print(f"FedTEF-v2 TailAgg near-zero update classes: {near_zero}")
    print(f"FedTEF-v2 TailAgg fallback rows: {fallback_count}")
    print(f"FedTEF-v2 TailAgg vs FedAvg tail update norm: {tailagg_norm:.6f}/{fedavg_norm:.6f}")
