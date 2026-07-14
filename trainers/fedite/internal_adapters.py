from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBottleneckAdapter(nn.Module):
    """Residual bottleneck adapter for internal CLIP tokens."""

    def __init__(self, dim, bottleneck=64, dropout=0.0, zero_init=True, up_init_std=None):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.down = nn.Linear(dim, bottleneck)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(float(dropout))
        self.up = nn.Linear(bottleneck, dim)
        if zero_init:
            nn.init.zeros_(self.up.weight)
            nn.init.zeros_(self.up.bias)
        elif up_init_std is not None:
            nn.init.normal_(self.up.weight, std=float(up_init_std))
            nn.init.zeros_(self.up.bias)

    def forward(self, tokens):
        orig_dtype = tokens.dtype
        x = self.norm(tokens.float())
        x = self.down(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.up(x)
        return x.to(orig_dtype)


class SharedSemanticAdapter(ResidualBottleneckAdapter):
    """Shared internal semantic adapter used for all samples."""


class TailEvidenceBasisAdapter(nn.Module):
    """Evidence-conditioned tail adapter basis.

    Each basis is an independent near-zero bottleneck adapter. Shared adapters
    stay exactly zero-initialized, but tail bases use a tiny up-projection init
    so the VisualBasisHead and EvidenceStateEncoder receive gradients from the
    first protected tail step while preserving the CLIP prior.
    """

    def __init__(self, dim, bottleneck=64, num_basis=4, dropout=0.0, basis_dropout=0.0, up_init_std=1e-4):
        super().__init__()
        self.num_basis = int(num_basis)
        self.basis_dropout = float(basis_dropout)
        self.bases = nn.ModuleList([
            ResidualBottleneckAdapter(
                dim,
                bottleneck,
                dropout,
                zero_init=False,
                up_init_std=up_init_std,
            )
            for _ in range(self.num_basis)
        ])

    def _drop_basis_weights(self, basis_weights):
        if not self.training or self.basis_dropout <= 0:
            return basis_weights
        keep = torch.rand_like(basis_weights) > self.basis_dropout
        if keep.sum(dim=1).eq(0).any():
            keep[keep.sum(dim=1).eq(0), 0] = True
        weights = basis_weights * keep.to(basis_weights.dtype)
        return weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-12)

    def forward(self, tokens, basis_weights, return_diagnostics=False):
        basis_weights = self._drop_basis_weights(basis_weights)
        outputs = torch.stack([basis(tokens) for basis in self.bases], dim=1)
        tail_delta = torch.einsum("bk,bkld->bld", basis_weights.to(outputs.dtype), outputs)
        if not return_diagnostics:
            return tail_delta

        with torch.no_grad():
            weight_mean = basis_weights.detach().float().mean(dim=0)
            entropy = -(basis_weights.detach().float().clamp_min(1e-12).log() * basis_weights.detach().float()).sum(dim=1)
            diagnostics = {
                "basis_weight_mean": weight_mean.cpu(),
                "basis_entropy": float(entropy.mean().item()) if entropy.numel() else 0.0,
                "basis_max_share": float(weight_mean.max().item()) if weight_mean.numel() else 0.0,
                "effective_basis_num": float(torch.exp(-(weight_mean.clamp_min(1e-12) * weight_mean.clamp_min(1e-12).log()).sum()).item()) if weight_mean.numel() else 0.0,
                "basis_output_norm": outputs.detach().float().norm(dim=-1).mean(dim=(0, 2)).cpu(),
                "tail_delta_norm": float(tail_delta.detach().float().norm(dim=-1).mean().item()) if tail_delta.numel() else 0.0,
            }
        return tail_delta, diagnostics


class TokenConditionedTailnessRouter(nn.Module):
    """Token-conditioned tailness router with optional token mask.

    The gate head is trained by router utility loss. The visual basis head is
    trained by the positive-only tail branch. Evidence logits can be added to
    visual basis logits before softmax.
    """

    def __init__(self, dim, hidden=None, num_basis=4, token_selective=False):
        super().__init__()
        hidden = int(hidden or max(32, dim // 4))
        self.num_basis = int(num_basis)
        self.token_selective = bool(token_selective)
        self.gate_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )
        self.basis_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, self.num_basis),
        )
        self.token_proj = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 1)) if self.token_selective else None

    def gate_only(self, cls_token):
        gate_logits = self.gate_head(cls_token.float()).squeeze(-1)
        return torch.sigmoid(gate_logits)

    def basis_logits(self, cls_token):
        return self.basis_head(cls_token.float())

    def forward(
        self,
        cls_token,
        token_features=None,
        evidence_basis_logits: Optional[torch.Tensor] = None,
        detach_gate=False,
        return_logits=False,
    ):
        gate = self.gate_only(cls_token)
        if detach_gate:
            gate = gate.detach()
        visual_logits = self.basis_logits(cls_token)
        logits = visual_logits
        if evidence_basis_logits is not None:
            logits = logits + evidence_basis_logits.to(logits.device, logits.dtype)
        basis_weights = F.softmax(logits, dim=-1)
        token_mask = None
        if self.token_proj is not None and token_features is not None:
            token_mask = torch.sigmoid(self.token_proj(token_features.float())).to(token_features.dtype)
        if return_logits:
            return gate, basis_weights, token_mask, visual_logits, logits
        return gate, basis_weights, token_mask


# Backward-compatible alias for older imports. The main method name is
# TokenConditionedTailnessRouter because patch-level token-selective writing is
# optional and disabled by default.
TokenSelectiveTailnessRouter = TokenConditionedTailnessRouter


class EvidenceStateEncoder(nn.Module):
    """Shared MLP mapping class evidence states to tail basis logits."""

    def __init__(self, state_dim=6, hidden=32, num_basis=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(state_dim),
            nn.Linear(state_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, num_basis),
        )

    def forward(self, class_state):
        return self.net(class_state.float())
