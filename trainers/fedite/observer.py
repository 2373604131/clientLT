import math
import random
from typing import Dict, Iterable, List, Optional

import torch

from .utils import robust_normalize, tensor_stats


class EvidenceTopologyObserver:
    """Class-client evidence topology observer.

    Topology in FedITE refers to class-client evidence topology, not physical
    communication topology. The observer consumes server-aggregated clipped
    summaries only; it does not log per-client class distributions.
    """

    def __init__(
        self,
        num_classes,
        protected_ratio=0.2,
        exploration_ratio=0.05,
        beta=0.9,
        reliability_min=0.05,
        warmup_rounds=0,
        warmup_mode="round_robin",
        tau_enter=None,
        tau_exit=None,
        exit_patience=2,
        w_e=0.25,
        w_k=0.25,
        w_g=0.25,
        w_u=0.25,
        r_m=1 / 3,
        r_k=1 / 3,
        r_s=1 / 3,
        stability_temperature=1.0,
        reliability_support_m0=2.0,
        reliability_client_q0=1.0,
        selection_rho=0.5,
        seed=1,
        eps=1e-12,
    ):
        self.num_classes = int(num_classes)
        self.protected_ratio = float(protected_ratio)
        self.exploration_ratio = float(exploration_ratio)
        self.beta = float(beta)
        self.reliability_min = float(reliability_min)
        self.warmup_rounds = int(warmup_rounds)
        self.warmup_mode = str(warmup_mode)
        self.tau_enter = None if tau_enter is None else float(tau_enter)
        self.tau_exit = None if tau_exit is None else float(tau_exit)
        self.exit_patience = int(exit_patience)
        self.stability_temperature = max(float(stability_temperature), 1e-6)
        self.reliability_support_m0 = max(float(reliability_support_m0), 1e-6)
        self.reliability_client_q0 = max(float(reliability_client_q0), 1e-6)
        self.selection_rho = min(max(float(selection_rho), 0.0), 1.0)
        self.rng = random.Random(seed)
        self.eps = float(eps)

        self.risk_weights = self._normalize_weights([w_e, w_k, w_g, w_u])
        self.reliability_weights = self._normalize_weights([r_m, r_k, r_s])

        zeros = torch.zeros(self.num_classes, dtype=torch.float32)
        self.EMA_M = zeros.clone()
        self.EMA_Q = zeros.clone()
        self.EMA_H = zeros.clone()
        self.EMA_N_eff = zeros.clone()
        self.EMA_E = zeros.clone()
        self.EMA_U = zeros.clone()
        self.EMA_U2 = zeros.clone()
        self.EMA_UpdateVar = zeros.clone()
        self.Gap = zeros.clone()
        self.SeenCount = zeros.clone()
        self.UpdateObservationCount = zeros.clone()
        self.D = zeros.clone()
        self.R = zeros.clone()
        self.S = zeros.clone()
        self.Rarity = zeros.clone()
        self.protected_classes: List[int] = []
        self.exploration_classes: List[int] = []
        self.exit_counter = zeros.clone()
        self.last_selection_info = {}

    @staticmethod
    def _normalize_weights(weights):
        weights = torch.as_tensor(weights, dtype=torch.float32).clamp_min(0.0)
        if weights.sum() <= 0:
            return torch.ones_like(weights) / max(int(weights.numel()), 1)
        return weights / weights.sum()

    def _ema(self, old, new):
        return self.beta * old + (1.0 - self.beta) * new

    def update(self, round_stats: Dict):
        M = torch.as_tensor(round_stats.get("M", torch.zeros(self.num_classes)), dtype=torch.float32)
        Q = torch.as_tensor(round_stats.get("Q", torch.zeros(self.num_classes)), dtype=torch.float32)
        H = torch.as_tensor(round_stats.get("H", torch.zeros(self.num_classes)), dtype=torch.float32)
        U = torch.as_tensor(round_stats.get("U", torch.zeros(self.num_classes)), dtype=torch.float32)
        write_count = torch.as_tensor(round_stats.get("write_count", torch.zeros(self.num_classes)), dtype=torch.float32)

        M = torch.nan_to_num(M, nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)
        Q = torch.nan_to_num(Q, nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)
        H = torch.nan_to_num(H, nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)
        U = torch.nan_to_num(U, nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)
        N_eff = torch.where(M > 0, M.pow(2) / (H + self.eps), torch.zeros_like(M))

        self.EMA_M = self._ema(self.EMA_M, M)
        self.EMA_Q = self._ema(self.EMA_Q, Q)
        self.EMA_H = self._ema(self.EMA_H, H)
        self.EMA_N_eff = self._ema(self.EMA_N_eff, N_eff)
        self.EMA_E = self.EMA_M.clone()
        observed_class = (M > 0) | (Q > 0)
        self.SeenCount += observed_class.float()

        observed_u = write_count > 0
        if observed_u.any():
            self.EMA_U[observed_u] = self._ema(self.EMA_U[observed_u], U[observed_u])
            self.EMA_U2[observed_u] = self._ema(self.EMA_U2[observed_u], U[observed_u].pow(2))
            self.UpdateObservationCount[observed_u] += 1.0

        self.EMA_UpdateVar = (self.EMA_U2 - self.EMA_U.pow(2)).clamp_min(0.0)
        self.Gap = torch.where(Q > 0, torch.zeros_like(self.Gap), self.Gap + 1.0)

        self.compute_survival_risk()
        self.compute_reliability()
        self.compute_protection_score()
        return self.get_round_summary(M=M, Q=Q, H=H, N_eff=N_eff, U=U)

    def compute_survival_risk(self):
        low_exposure = 1.0 - robust_normalize(self.EMA_E, eps=self.eps)
        low_coverage = 1.0 - robust_normalize(self.EMA_N_eff, eps=self.eps)
        long_gap = robust_normalize(self.Gap, eps=self.eps)
        weak_write = 1.0 - robust_normalize(self.EMA_U, eps=self.eps)
        unseen = self.UpdateObservationCount == 0
        weak_write = torch.where(unseen, torch.full_like(weak_write, 0.5), weak_write)
        w = self.risk_weights
        self.D = (w[0] * low_exposure + w[1] * low_coverage + w[2] * long_gap + w[3] * weak_write).clamp(0.0, 1.0)
        return self.D

    def compute_reliability(self):
        support_confidence = (1.0 - torch.exp(-self.EMA_M / self.reliability_support_m0)).clamp(0.0, 1.0)
        coverage_confidence = (1.0 - torch.exp(-self.EMA_Q / self.reliability_client_q0)).clamp(0.0, 1.0)
        update_stability = torch.exp(-self.EMA_UpdateVar / self.stability_temperature).clamp(0.0, 1.0)
        update_stability = torch.where(
            self.UpdateObservationCount == 0,
            torch.full_like(update_stability, 0.5),
            update_stability,
        )
        r = self.reliability_weights
        reliability_now = (r[0] * support_confidence + r[1] * coverage_confidence + r[2] * update_stability).clamp(0.0, 1.0)
        reliability_now = torch.where(self.SeenCount > 0, reliability_now, torch.zeros_like(reliability_now))
        self.R = self._ema(self.R, reliability_now).clamp(0.0, 1.0)
        return self.R

    def compute_protection_score(self):
        exposure_mass = torch.log1p(self.EMA_E.clamp_min(0.0))
        rarity = 1.0 - robust_normalize(exposure_mass, eps=self.eps)
        self.Rarity = torch.where(self.SeenCount > 0, rarity, torch.zeros_like(rarity)).clamp(0.0, 1.0)
        soft_reliability = self.selection_rho + (1.0 - self.selection_rho) * self.R
        soft_rarity = 0.5 + 0.5 * self.Rarity
        self.S = (self.D * soft_reliability * soft_rarity).clamp(0.0, 1.0)
        return self.S

    def _rarity_quota(self, protected_k):
        # CIFAR-LT and the FedITE experiments use the bottom 20% as the tail
        # metric. The quota is capped by the protected budget and only affects
        # selection, not the trainable model.
        tail_like = max(1, int(round(self.num_classes * 0.2)))
        return min(int(protected_k), tail_like)

    def _rank_by_score(self, candidates):
        return sorted(
            candidates,
            key=lambda c: (
                -float(self.S[c].item()),
                -float(self.Rarity[c].item()),
                float(self.EMA_E[c].item()),
                int(c),
            ),
        )

    def _rank_by_rarity(self, candidates):
        return sorted(
            candidates,
            key=lambda c: (
                float(self.EMA_E[c].item()),
                float(self.EMA_N_eff[c].item()),
                -float(self.D[c].item()),
                int(c),
            ),
        )

    def _warmup_classes(self, round_idx, k):
        if self.warmup_mode == "none" or k <= 0:
            return []
        if self.warmup_mode == "random":
            ids = list(range(self.num_classes))
            self.rng.shuffle(ids)
            return sorted(ids[:k])
        if self.warmup_mode == "round_robin":
            start = (int(round_idx) * k) % self.num_classes
            return sorted([(start + j) % self.num_classes for j in range(k)])
        raise ValueError(f"Unknown FedITE warmup mode: {self.warmup_mode}")

    def select_protected_classes(self, round_idx):
        k = max(1, int(round(self.num_classes * self.protected_ratio)))
        previous = set() if int(round_idx) == self.warmup_rounds else set(self.protected_classes)

        if round_idx < self.warmup_rounds:
            previous_for_log = set(self.protected_classes)
            selected = self._warmup_classes(round_idx, k)
            self.protected_classes = selected
            self.last_selection_info = {
                "mode": "warmup",
                "new": sorted(set(selected) - previous_for_log),
                "removed": sorted(previous_for_log - set(selected)),
                "overlap": len(previous_for_log.intersection(selected)),
            }
            return selected

        if int(round_idx) == self.warmup_rounds:
            self.protected_classes = []
            self.exploration_classes = []
            self.exit_counter.zero_()

        candidates = [
            c for c in range(self.num_classes)
            if float(self.SeenCount[c].item()) > 0
        ]
        ranked = self._rank_by_score(candidates)
        rare_quota = min(len(candidates), self._rarity_quota(k))
        rare_selected = set(self._rank_by_rarity(candidates)[:rare_quota])

        if self.tau_enter is not None and self.tau_exit is not None:
            selected = set(rare_selected)
            for c in previous:
                if len(selected) >= k:
                    break
                if c not in candidates:
                    continue
                if float(self.S[c].item()) < self.tau_exit:
                    self.exit_counter[c] += 1
                else:
                    self.exit_counter[c] = 0
                if self.exit_counter[c] < self.exit_patience:
                    selected.add(c)
            for c in ranked:
                if len(selected) >= k:
                    break
                if float(self.S[c].item()) >= self.tau_enter or c in previous:
                    selected.add(c)
            for c in ranked:
                if len(selected) >= k:
                    break
                    selected.add(c)
            selected = sorted(selected)[:k]
        else:
            keep_n = min(len(previous), max(0, int(round((k - len(rare_selected)) * 0.5))))
            keep = [c for c in ranked if c in previous][:keep_n]
            selected = list(sorted(rare_selected))
            for c in keep:
                if len(selected) >= k:
                    break
                if c not in selected:
                    selected.append(c)
            for c in ranked:
                if len(selected) >= k:
                    break
                if c not in selected:
                    selected.append(c)
            selected = sorted(selected)

        self.protected_classes = selected
        self._select_exploration_classes(k)
        self.last_selection_info = {
            "mode": "score",
            "new": sorted(set(selected) - previous),
            "removed": sorted(previous - set(selected)),
            "overlap": len(previous.intersection(selected)),
            "rare_quota": int(rare_quota),
            "rare_selected": sorted(rare_selected.intersection(selected)),
        }
        return selected

    def _select_exploration_classes(self, protected_k):
        k = max(0, int(round(self.num_classes * self.exploration_ratio)))
        if k == 0:
            self.exploration_classes = []
            return
        protected = set(self.protected_classes)
        low_reliability = [
            c for c in range(self.num_classes)
            if c not in protected and float(self.SeenCount[c].item()) > 0 and float(self.R[c].item()) < self.reliability_min
        ]
        ranked = sorted(low_reliability, key=lambda c: float(self.D[c].item()), reverse=True)
        self.exploration_classes = sorted(ranked[:k])

    def get_round_evidence_strength(self):
        if not self.protected_classes:
            return 0.0
        return float(self.S[self.protected_classes].mean().item())

    def get_class_state(self):
        return torch.stack(
            [
                self.D,
                self.R,
                self.S,
                robust_normalize(self.EMA_E, eps=self.eps),
                robust_normalize(self.EMA_N_eff, eps=self.eps),
                robust_normalize(self.Gap, eps=self.eps),
            ],
            dim=1,
        ).float()

    def get_round_summary(self, **current):
        summary = {
            "M": tensor_stats(current.get("M", self.EMA_M)),
            "N_eff": tensor_stats(current.get("N_eff", self.EMA_N_eff)),
            "SeenCount": tensor_stats(self.SeenCount),
            "Gap": tensor_stats(self.Gap),
            "D": tensor_stats(self.D),
            "R": tensor_stats(self.R),
            "S": tensor_stats(self.S),
            "Rarity": tensor_stats(self.Rarity),
            "protected_classes": list(self.protected_classes),
            "exploration_classes": list(self.exploration_classes),
        }
        summary.update(self.last_selection_info)
        return summary

    def state_dict(self):
        return {
            "EMA_M": self.EMA_M.clone(),
            "EMA_Q": self.EMA_Q.clone(),
            "EMA_H": self.EMA_H.clone(),
            "EMA_N_eff": self.EMA_N_eff.clone(),
            "EMA_E": self.EMA_E.clone(),
            "EMA_U": self.EMA_U.clone(),
            "EMA_U2": self.EMA_U2.clone(),
            "EMA_UpdateVar": self.EMA_UpdateVar.clone(),
            "Gap": self.Gap.clone(),
            "SeenCount": self.SeenCount.clone(),
            "UpdateObservationCount": self.UpdateObservationCount.clone(),
            "D": self.D.clone(),
            "R": self.R.clone(),
            "S": self.S.clone(),
            "Rarity": self.Rarity.clone(),
            "protected_classes": list(self.protected_classes),
            "exploration_classes": list(self.exploration_classes),
            "exit_counter": self.exit_counter.clone(),
        }

    def load_state_dict(self, state):
        for key, value in state.items():
            if hasattr(self, key):
                if isinstance(getattr(self, key), torch.Tensor):
                    setattr(self, key, torch.as_tensor(value, dtype=torch.float32).clone())
                else:
                    setattr(self, key, list(value) if isinstance(value, (list, tuple)) else value)

    def class_state_rows(self, round_idx):
        rows = []
        for c in range(self.num_classes):
            rows.append({
                "round": int(round_idx),
                "class_id": int(c),
                "EMA_M": float(self.EMA_M[c].item()),
                "EMA_Q": float(self.EMA_Q[c].item()),
                "EMA_H": float(self.EMA_H[c].item()),
                "EMA_N_eff": float(self.EMA_N_eff[c].item()),
                "EMA_E": float(self.EMA_E[c].item()),
                "EMA_U": float(self.EMA_U[c].item()),
                "EMA_U2": float(self.EMA_U2[c].item()),
                "Var_U": float(self.EMA_UpdateVar[c].item()),
                "Gap": float(self.Gap[c].item()),
                "SeenCount": float(self.SeenCount[c].item()),
                "D": float(self.D[c].item()),
                "R": float(self.R[c].item()),
                "S": float(self.S[c].item()),
                "Rarity": float(self.Rarity[c].item()),
                "is_protected": int(c in set(self.protected_classes)),
                "is_exploration": int(c in set(self.exploration_classes)),
            })
        return rows
