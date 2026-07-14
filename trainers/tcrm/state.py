from dataclasses import dataclass
from typing import Dict, List

import torch
import torch.nn.functional as F


def project_perp(rho, z):
    z = F.normalize(z.float(), dim=-1)
    rho = rho.float()
    return rho - z * (rho * z).sum(dim=-1, keepdim=True)


def project_l2_ball(rho, radius):
    radius = float(radius)
    norm = rho.float().norm(dim=-1, keepdim=True)
    scale = torch.clamp(radius / norm.clamp_min(1e-12), max=1.0)
    return rho * scale


def sanitize_residual(rho, z, radius):
    return project_l2_ball(project_perp(rho, z), radius)


@dataclass
class TCRMCoreState:
    prompt_state: Dict[str, torch.Tensor]
    rho: torch.Tensor
    zero_shot_text: torch.Tensor
    M: torch.Tensor
    C: torch.Tensor
    D: torch.Tensor
    age: torch.Tensor
    r_pre: torch.Tensor
    width_gate: torch.Tensor
    last_write: torch.Tensor
    last_direction_consistency: torch.Tensor
    last_local_gain: torch.Tensor
    last_corroboration: torch.Tensor
    last_decay: torch.Tensor
    last_num_valid_contributors: torch.Tensor
    last_candidate_skip_count: torch.Tensor
    tail_class_ids: List[int]
    tail_index_of_class: Dict[int, int]
    non_tail_class_ids: List[int]
    class_prior: torch.Tensor
    m0: float
    d0: float
    stale_horizon: float
    rho_norm_bound: float = 0.2

    def sanitize_(self):
        tail_z = self.zero_shot_text[self.tail_class_ids].to(self.rho.device, dtype=torch.float32)
        self.rho = sanitize_residual(self.rho.float(), tail_z, self.rho_norm_bound)
        return self


def positive_median_or_one(values):
    values = torch.as_tensor(values, dtype=torch.float32)
    positive = values[values > 0]
    if positive.numel() == 0:
        return 1.0
    return float(torch.median(positive).clamp_min(1.0).item())


def init_core_state(
    prompt_state,
    zero_shot_text,
    topology_tensors,
    tail_class_ids,
    non_tail_class_ids,
    class_prior,
    total_rounds,
    rho_norm_bound=0.2,
    stale_horizon_ratio=0.25,
):
    zero_shot_text = F.normalize(zero_shot_text.float(), dim=-1)
    tail_class_ids = [int(c) for c in tail_class_ids]
    dim = int(zero_shot_text.shape[1])
    num_tail = len(tail_class_ids)
    M = topology_tensors["M"].float().clone()
    C = topology_tensors["C"].float().clone()
    D = topology_tensors["D"].float().clone()
    zeros = torch.zeros(num_tail, dtype=torch.float32)
    stale_horizon = max(3.0, float(int(0.25 * int(total_rounds)))) if stale_horizon_ratio is None else max(3.0, float(int(float(stale_horizon_ratio) * int(total_rounds))))
    state = TCRMCoreState(
        prompt_state={k: v.detach().cpu().clone() for k, v in prompt_state.items()},
        rho=torch.zeros(num_tail, dim, dtype=torch.float32),
        zero_shot_text=zero_shot_text.detach().cpu().clone(),
        M=M,
        C=C,
        D=D,
        age=zeros.clone(),
        r_pre=zeros.clone(),
        width_gate=zeros.clone(),
        last_write=zeros.clone(),
        last_direction_consistency=zeros.clone(),
        last_local_gain=zeros.clone(),
        last_corroboration=zeros.clone(),
        last_decay=zeros.clone(),
        last_num_valid_contributors=zeros.clone(),
        last_candidate_skip_count=zeros.clone(),
        tail_class_ids=tail_class_ids,
        tail_index_of_class={int(c): i for i, c in enumerate(tail_class_ids)},
        non_tail_class_ids=[int(c) for c in non_tail_class_ids],
        class_prior=torch.as_tensor(class_prior, dtype=torch.float32).clone(),
        m0=positive_median_or_one(M),
        d0=positive_median_or_one(D),
        stale_horizon=stale_horizon,
        rho_norm_bound=float(rho_norm_bound),
    )
    state.sanitize_()
    return state
