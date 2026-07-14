import importlib.util
import os
from typing import Dict, Iterable, List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .internal_adapters import (
    EvidenceStateEncoder,
    SharedSemanticAdapter,
    TailEvidenceBasisAdapter,
    TokenConditionedTailnessRouter,
)
from .utils import is_in_classes, parse_int_list


def _load_simple_tokenizer_cls():
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    path = os.path.join(repo_root, "clip", "simple_tokenizer.py")
    spec = importlib.util.spec_from_file_location("fedite_clip_simple_tokenizer", path)
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
        return module.SimpleTokenizer
    except ModuleNotFoundError:
        return None


_Tokenizer = _load_simple_tokenizer_cls()
_tokenizer = _Tokenizer() if _Tokenizer is not None else None


def _clip_tokenize(texts, context_length=77, truncate=False):
    if isinstance(texts, str):
        texts = [texts]
    if _tokenizer is not None:
        sot_token = _tokenizer.encoder["<|startoftext|>"]
        eot_token = _tokenizer.encoder["<|endoftext|>"]
        all_tokens = [[sot_token] + _tokenizer.encode(text) + [eot_token] for text in texts]
    else:
        # Fallback for lightweight tests when CLIP tokenizer dependencies are
        # absent. Real experiments should install requirements.txt and will use
        # the official CLIP BPE tokenizer above.
        sot_token, eot_token = 49406, 49407
        all_tokens = []
        for text in texts:
            byte_tokens = [1000 + int(b) for b in text.encode("utf-8")[: context_length - 2]]
            all_tokens.append([sot_token] + byte_tokens + [eot_token])
    result = torch.zeros(len(all_tokens), context_length, dtype=torch.long)
    for i, tokens in enumerate(all_tokens):
        if len(tokens) > context_length:
            if truncate:
                tokens = tokens[:context_length]
                tokens[-1] = eot_token
            else:
                raise RuntimeError(f"Input {texts[i]} is too long for context length {context_length}")
        result[i, : len(tokens)] = torch.tensor(tokens, dtype=torch.long)
    return result


class FixedOrGeneralPromptLearner(nn.Module):
    """Fixed template prompts with optional general trainable context delta."""

    def __init__(self, clip_model, classnames, prompt_ctx="a photo of a", train_prompt=False, n_ctx=4):
        super().__init__()
        self.classnames = [name.replace("_", " ") for name in classnames]
        self.prompt_ctx = str(prompt_ctx or "a photo of a")
        self.train_prompt = bool(train_prompt)
        self.n_ctx = int(n_ctx)
        prompts = [f"{self.prompt_ctx} {name}." for name in self.classnames]
        tokenized = _clip_tokenize(prompts)
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized).type(clip_model.dtype).detach().clone()
        self.register_buffer("fixed_embedding", embedding)
        self.register_buffer("tokenized_prompts", tokenized)
        if self.train_prompt:
            ctx_dim = embedding.shape[-1]
            self.ctx_delta = nn.Parameter(torch.zeros(self.n_ctx, ctx_dim, dtype=embedding.dtype))
        else:
            self.register_parameter("ctx_delta", None)

    def forward(self):
        prompts = self.fixed_embedding
        if self.ctx_delta is None:
            return prompts
        prompts = prompts.clone()
        usable = min(self.n_ctx, prompts.shape[1] - 2)
        prompts[:, 1:1 + usable, :] = prompts[:, 1:1 + usable, :] + self.ctx_delta[:usable].to(prompts.dtype)
        return prompts


class FrozenTextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(prompts.dtype)
        x = x.permute(1, 0, 2)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)
        x = self.ln_final(x).type(prompts.dtype)
        x = x[torch.arange(x.shape[0], device=x.device), tokenized_prompts.argmax(dim=-1)] @ self.text_projection
        return x


class FedITEInternalVisualWrapper(nn.Module):
    """Manual CLIP ViT visual forward with internal FedITE adapters."""

    def __init__(
        self,
        visual,
        adapter_layers,
        bottleneck=64,
        num_tail_basis=4,
        alpha_shared=1.0,
        alpha_tail=1.0,
        adapter_dropout=0.0,
        basis_dropout=0.0,
        token_selective=False,
    ):
        super().__init__()
        required = ["conv1", "class_embedding", "positional_embedding", "ln_pre", "transformer", "ln_post"]
        missing = [name for name in required if not hasattr(visual, name)]
        if missing or not hasattr(visual.transformer, "resblocks"):
            raise ValueError("FedITE currently supports CLIP ViT visual encoders only.")
        self.visual = visual
        self.adapter_layers = sorted(set(int(i) for i in adapter_layers))
        self.alpha_shared = float(alpha_shared)
        self.alpha_tail = float(alpha_tail)
        self.num_tail_basis = int(num_tail_basis)
        self.dim = int(visual.ln_pre.normalized_shape[0])
        self.output_dim = int(getattr(visual, "output_dim", visual.proj.shape[1] if getattr(visual, "proj", None) is not None else self.dim))
        self.num_blocks = len(visual.transformer.resblocks)
        bad_layers = [i for i in self.adapter_layers if i < 0 or i >= self.num_blocks]
        if bad_layers:
            raise ValueError(f"FedITE adapter layer index out of range: {bad_layers}; CLIP ViT has {self.num_blocks} blocks")
        self.suffix_start = min(self.adapter_layers) if self.adapter_layers else self.num_blocks

        self.shared_adapters = nn.ModuleDict({
            str(i): SharedSemanticAdapter(self.dim, bottleneck, adapter_dropout, zero_init=True)
            for i in self.adapter_layers
        })
        self.tail_adapters = nn.ModuleDict({
            str(i): TailEvidenceBasisAdapter(
                self.dim,
                bottleneck=bottleneck,
                num_basis=num_tail_basis,
                dropout=adapter_dropout,
                basis_dropout=basis_dropout,
            )
            for i in self.adapter_layers
        })
        self.routers = nn.ModuleDict({
            str(i): TokenConditionedTailnessRouter(
                self.dim,
                hidden=max(32, self.dim // 4),
                num_basis=num_tail_basis,
                token_selective=token_selective,
            )
            for i in self.adapter_layers
        })

    def _stem(self, image):
        dtype = self.visual.conv1.weight.dtype
        x = self.visual.conv1(image.type(dtype))
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1)
        cls = self.visual.class_embedding.to(x.dtype) + torch.zeros(
            x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device
        )
        x = torch.cat([cls, x], dim=1)
        if x.shape[1] != self.visual.positional_embedding.shape[0]:
            raise ValueError(
                "FedITE visual positional length mismatch. "
                f"Got {x.shape[1]} tokens, expected {self.visual.positional_embedding.shape[0]}. "
                "Resize images to the CLIP input resolution before calling FedITE."
            )
        x = x + self.visual.positional_embedding.to(x.dtype)
        return self.visual.ln_pre(x)

    @torch.no_grad()
    def encode_prefix(self, image):
        """Run the frozen ViT prefix once, before the first FedITE write layer."""
        x = self._stem(image)
        x = x.permute(1, 0, 2)
        for i in range(self.suffix_start):
            x = self.visual.transformer.resblocks[i](x)
        return x.detach()

    def select_prefix(self, prefix_tokens, indices):
        if prefix_tokens is None:
            return None
        return prefix_tokens.index_select(1, indices.to(prefix_tokens.device))

    def forward(
        self,
        image,
        mode="shared",
        labels=None,
        protected_classes: Optional[Sequence[int]] = None,
        sample_evidence_strength: Optional[torch.Tensor] = None,
        sample_evidence_logits: Optional[torch.Tensor] = None,
        detach_gate=False,
        return_diagnostics=False,
        return_write_evidence=False,
        prefix_tokens=None,
    ):
        mode = str(mode)
        if mode not in {"base", "shared", "router", "final"}:
            raise ValueError(f"Unknown FedITE visual mode: {mode}")

        if prefix_tokens is None:
            x = self._stem(image)
            x = x.permute(1, 0, 2)
            start_block = 0
        else:
            x = prefix_tokens
            start_block = self.suffix_start
        diagnostics = {
            "shared_delta_norms": [],
            "shared_input_norms": [],
            "tail_write_norms": [],
            "tail_gates": [],
            "basis_weights": [],
            "cls_tokens": [],
            "basis_diagnostics": [],
        }

        for i in range(start_block, self.num_blocks):
            block = self.visual.transformer.resblocks[i]
            x = block(x)
            key = str(i)
            if key not in self.shared_adapters:
                continue

            x_nld = x.permute(1, 0, 2)
            if mode in {"shared", "router", "final"}:
                shared_input_norm = x_nld.detach().float().norm(dim=-1).mean()
                shared_delta = self.shared_adapters[key](x_nld)
                x_nld = x_nld + self.alpha_shared * shared_delta
                if return_diagnostics:
                    diagnostics["shared_input_norms"].append(float(shared_input_norm.item()))
                    diagnostics["shared_delta_norms"].append(float(shared_delta.detach().float().norm(dim=-1).mean().item()))

            if mode == "router":
                cls_token = x_nld[:, 0, :].detach()
                gate = self.routers[key].gate_only(cls_token)
                diagnostics["tail_gates"].append(gate)
                diagnostics["cls_tokens"].append(cls_token)

            if mode == "final":
                cls_token = x_nld[:, 0, :]
                evidence_logits = sample_evidence_logits
                gate, basis_weights, token_mask = self.routers[key](
                    cls_token,
                    token_features=x_nld,
                    evidence_basis_logits=evidence_logits,
                    detach_gate=detach_gate,
                )
                tail_out = self.tail_adapters[key](x_nld, basis_weights, return_diagnostics=return_diagnostics)
                if return_diagnostics:
                    tail_delta, basis_diag = tail_out
                    diagnostics["basis_diagnostics"].append(basis_diag)
                else:
                    tail_delta = tail_out

                effective_gate = gate
                if labels is not None:
                    mask = is_in_classes(labels, protected_classes).to(effective_gate.device, effective_gate.dtype)
                    effective_gate = effective_gate * mask
                if sample_evidence_strength is not None:
                    effective_gate = effective_gate * sample_evidence_strength.to(effective_gate.device, effective_gate.dtype)
                effective_gate = effective_gate.clamp(0.0, 1.0)
                write_delta = self.alpha_tail * effective_gate[:, None, None].to(tail_delta.dtype) * tail_delta
                if token_mask is not None:
                    write_delta = write_delta * token_mask
                x_nld = x_nld + write_delta
                if return_diagnostics:
                    diagnostics["tail_gates"].append(gate)
                    diagnostics["basis_weights"].append(basis_weights)
                if return_diagnostics or return_write_evidence:
                    diagnostics["tail_write_norms"].append(write_delta.detach().float()[:, 0, :].norm(dim=-1))

            x = x_nld.permute(1, 0, 2)

        x = x.permute(1, 0, 2)
        cls = self.visual.ln_post(x[:, 0, :])
        if getattr(self.visual, "proj", None) is not None:
            cls = cls @ self.visual.proj
        if not (return_diagnostics or return_write_evidence):
            return cls
        return cls, diagnostics


class FedITEModel(nn.Module):
    def __init__(self, clip_model, classnames, args):
        super().__init__()
        self.clip_model = clip_model
        self.classnames = list(classnames)
        self.num_classes = len(classnames)
        self.dtype = getattr(clip_model, "dtype", torch.float32)
        for param in self.clip_model.parameters():
            param.requires_grad_(False)

        adapter_layers = parse_int_list(getattr(args, "fedite_adapter_layers", "10,11"))
        self.visual_wrapper = FedITEInternalVisualWrapper(
            clip_model.visual,
            adapter_layers=adapter_layers,
            bottleneck=int(getattr(args, "fedite_adapter_bottleneck", 64)),
            num_tail_basis=int(getattr(args, "fedite_num_tail_basis", 2)),
            alpha_shared=float(getattr(args, "fedite_alpha_shared", 1.0)),
            alpha_tail=float(getattr(args, "fedite_alpha_tail", 1.0)),
            adapter_dropout=float(getattr(args, "fedite_adapter_dropout", 0.0)),
            basis_dropout=float(getattr(args, "fedite_basis_dropout", 0.0)),
            token_selective=bool(getattr(args, "fedite_token_selective", False)),
        )
        self.text_encoder = FrozenTextEncoder(clip_model)
        self.prompt_learner = FixedOrGeneralPromptLearner(
            clip_model,
            classnames,
            prompt_ctx=getattr(args, "fedite_prompt_ctx", "a photo of a"),
            train_prompt=bool(getattr(args, "fedite_train_prompt", False)),
            n_ctx=int(getattr(args, "fedite_prompt_n_ctx", 4)),
        )
        self.evidence_encoder = EvidenceStateEncoder(
            state_dim=6,
            hidden=max(16, int(getattr(args, "fedite_num_tail_basis", 2)) * 8),
            num_basis=int(getattr(args, "fedite_num_tail_basis", 2)),
        )
        self.class_basis_logits = nn.Parameter(
            torch.zeros(self.num_classes, int(getattr(args, "fedite_num_tail_basis", 2)), dtype=torch.float32)
        )
        self.class_basis_scale = float(getattr(args, "fedite_class_basis_scale", 1.0))
        self.logit_scale = clip_model.logit_scale
        self.register_buffer("class_evidence_state", torch.zeros(self.num_classes, 6, dtype=torch.float32))
        self.register_buffer("protected_class_mask", torch.zeros(self.num_classes, dtype=torch.bool))
        self.round_evidence_strength = 1.0
        self.cache_text_features = bool(getattr(args, "fedite_cache_text_features", True))
        self._cached_text_features = None
        self._ensure_trainable_flags()

    def _ensure_trainable_flags(self):
        for name, param in self.named_parameters():
            if name.startswith("clip_model.") or name.startswith("text_encoder.") or name == "logit_scale":
                param.requires_grad_(False)
            elif name.startswith("prompt_learner.") and self.prompt_learner.ctx_delta is None:
                param.requires_grad_(False)
            else:
                param.requires_grad_(True)

    def encode_text(self):
        device = next(self.parameters()).device
        if (
            self.cache_text_features
            and self.prompt_learner.ctx_delta is None
            and self._cached_text_features is not None
            and self._cached_text_features.device == device
        ):
            return self._cached_text_features
        prompts = self.prompt_learner().to(next(self.parameters()).device)
        tokenized = self.prompt_learner.tokenized_prompts.to(prompts.device)
        text_features = self.text_encoder(prompts, tokenized)
        text_features = F.normalize(text_features.float(), dim=-1)
        if self.cache_text_features and self.prompt_learner.ctx_delta is None:
            self._cached_text_features = text_features.detach()
        return text_features

    def _logits_from_image_features(self, image_features):
        text_features = self.encode_text()
        image_features = F.normalize(image_features.float(), dim=-1)
        scale = self.logit_scale.float().exp().clamp(max=100.0)
        return scale * image_features @ text_features.t()

    def encode_visual_prefix(self, images):
        return self.visual_wrapper.encode_prefix(images)

    def select_visual_prefix(self, prefix_tokens, indices):
        return self.visual_wrapper.select_prefix(prefix_tokens, indices)

    def forward_shared(self, images, return_diagnostics=False, prefix_tokens=None, compute_base=True):
        shared = self.visual_wrapper(
            images,
            mode="shared",
            return_diagnostics=return_diagnostics,
            prefix_tokens=prefix_tokens,
        )
        if return_diagnostics:
            shared_features, diagnostics = shared
        else:
            shared_features, diagnostics = shared, {}
        base_features = None
        logits_base = None
        if compute_base:
            with torch.no_grad():
                base_features = self.visual_wrapper(
                    images,
                    mode="base",
                    return_diagnostics=False,
                    prefix_tokens=prefix_tokens,
                )
            logits_base = self._logits_from_image_features(base_features.detach())
        logits_shared = self._logits_from_image_features(shared_features)
        return {
            "logits_shared": logits_shared,
            "logits_base": logits_base,
            "shared_image_features": shared_features,
            "base_image_features": base_features.detach() if base_features is not None else None,
            "diagnostics": diagnostics,
        }

    def forward_router_train(self, images, labels, class_state, return_diagnostics=False, prefix_tokens=None):
        router = self.visual_wrapper(
            images,
            mode="router",
            return_diagnostics=True,
            prefix_tokens=prefix_tokens,
        )
        image_features, diagnostics = router
        logits_shared = self._logits_from_image_features(image_features.detach())
        return {
            "logits_shared": logits_shared.detach(),
            "tail_gates": diagnostics["tail_gates"],
            "diagnostics": diagnostics,
        }

    def forward_tail_train(
        self,
        images,
        labels,
        class_state=None,
        return_diagnostics=False,
        return_write_evidence=True,
        prefix_tokens=None,
    ):
        if class_state is None:
            class_state = self.class_evidence_state
        class_state = class_state.to(images.device, dtype=torch.float32)
        labels = labels.long()
        sample_state = class_state[labels]
        sample_evidence_logits = self.evidence_encoder(sample_state)
        sample_evidence_logits = sample_evidence_logits + float(self.class_basis_scale) * self.class_basis_logits[labels].to(
            sample_evidence_logits.device,
            sample_evidence_logits.dtype,
        )
        sample_evidence_strength = torch.sqrt(sample_state[:, 2].clamp(0.0, 1.0)) * float(self.round_evidence_strength)
        features = self.visual_wrapper(
            images,
            mode="final",
            labels=labels,
            protected_classes=self.get_protected_classes(),
            sample_evidence_strength=sample_evidence_strength,
            sample_evidence_logits=sample_evidence_logits,
            detach_gate=True,
            return_diagnostics=return_diagnostics,
            return_write_evidence=return_write_evidence,
            prefix_tokens=prefix_tokens,
        )
        if return_diagnostics or return_write_evidence:
            image_features, diagnostics = features
        else:
            image_features, diagnostics = features, {}
        logits = self._logits_from_image_features(image_features)
        return {
            "logits": logits,
            "image_features": image_features,
            "sample_evidence_strength": sample_evidence_strength,
            "diagnostics": diagnostics,
        }

    @torch.no_grad()
    def _soft_evidence_from_shared(self, logits_shared, class_state, topk=5, temperature=2.0):
        probs = F.softmax(logits_shared.float() / max(float(temperature), 1e-6), dim=-1)
        k = min(int(topk), probs.shape[1])
        top_prob, top_idx = torch.topk(probs, k=k, dim=-1)
        top_prob = top_prob / top_prob.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        state = class_state.to(logits_shared.device, dtype=torch.float32)
        sample_state = (top_prob[..., None] * state[top_idx]).sum(dim=1)
        strength = (top_prob * torch.sqrt(state[top_idx, 2].clamp(0.0, 1.0))).sum(dim=1).clamp(0.0, 1.0)
        return sample_state, strength, top_idx, top_prob

    def forward_inference(
        self,
        images,
        class_state=None,
        inference_topk=5,
        inference_temperature=2.0,
        return_diagnostics=False,
        prefix_tokens=None,
        return_dict=False,
    ):
        if class_state is None:
            class_state = self.class_evidence_state
        shared = self.forward_shared(
            images,
            return_diagnostics=False,
            prefix_tokens=prefix_tokens,
            compute_base=False,
        )
        logits_shared = shared["logits_shared"]
        sample_state, strength, top_idx, top_prob = self._soft_evidence_from_shared(
            logits_shared,
            class_state,
            topk=inference_topk,
            temperature=inference_temperature,
        )
        sample_evidence_logits = self.evidence_encoder(sample_state)
        class_basis = self.class_basis_logits.to(logits_shared.device, dtype=sample_evidence_logits.dtype)
        sample_class_basis = (top_prob[..., None].to(class_basis.dtype) * class_basis[top_idx]).sum(dim=1)
        sample_evidence_logits = sample_evidence_logits + float(self.class_basis_scale) * sample_class_basis
        features = self.visual_wrapper(
            images,
            mode="final",
            labels=None,
            protected_classes=None,
            sample_evidence_strength=strength * float(self.round_evidence_strength),
            sample_evidence_logits=sample_evidence_logits,
            detach_gate=False,
            return_diagnostics=return_diagnostics,
            prefix_tokens=prefix_tokens,
        )
        if return_diagnostics:
            image_features, diagnostics = features
        else:
            image_features, diagnostics = features, {}
        logits = self._logits_from_image_features(image_features)
        output = {
            "logits": logits,
            "logits_final": logits,
            "logits_shared": logits_shared,
            "image_features": image_features,
            "shared_image_features": shared["shared_image_features"],
            "base_image_features": shared["base_image_features"],
            "sample_evidence_strength": strength,
            "topk_idx": top_idx,
            "topk_prob": top_prob,
            "diagnostics": diagnostics,
        }
        if return_diagnostics or return_dict:
            return output
        return logits

    def forward(self, images, labels=None, return_diagnostics=False, **kwargs):
        # Public forward is always label-free inference. Training must call
        # forward_shared, forward_router_train, or forward_tail_train explicitly.
        # The labels argument is accepted only to avoid accidental API crashes;
        # it is deliberately ignored to prevent evaluation-time label leakage.
        return self.forward_inference(images, return_diagnostics=return_diagnostics, **kwargs)

    def set_protected_classes(self, protected_classes):
        mask = torch.zeros(self.num_classes, dtype=torch.bool, device=self.protected_class_mask.device)
        if protected_classes is not None:
            for class_id in protected_classes:
                if 0 <= int(class_id) < self.num_classes:
                    mask[int(class_id)] = True
        self.protected_class_mask.copy_(mask)

    def get_protected_classes(self):
        return torch.where(self.protected_class_mask.detach().cpu())[0].tolist()

    def set_class_evidence_state(self, class_state):
        if isinstance(class_state, dict):
            class_state = class_state.get("class_state", class_state)
        class_state = torch.as_tensor(class_state, dtype=torch.float32, device=self.class_evidence_state.device)
        if class_state.shape != self.class_evidence_state.shape:
            raise ValueError(f"Expected class evidence state shape {tuple(self.class_evidence_state.shape)}, got {tuple(class_state.shape)}")
        self.class_evidence_state.copy_(class_state)

    def set_round_evidence_strength(self, value):
        self.round_evidence_strength = float(value)

    def _key_groups(self):
        shared, gate, tail, frozen = [], [], [], []
        for name, _ in self.named_parameters():
            if name.startswith("visual_wrapper.shared_adapters"):
                shared.append(name)
            elif name.startswith("prompt_learner.") and self.prompt_learner.ctx_delta is not None:
                shared.append(name)
            elif ".gate_head." in name:
                gate.append(name)
            elif (
                name == "class_basis_logits"
                or
                name.startswith("visual_wrapper.tail_adapters")
                or ".basis_head." in name
                or ".token_proj." in name
                or name.startswith("evidence_encoder")
            ):
                tail.append(name)
            else:
                frozen.append(name)
        return shared, gate, tail, frozen

    def get_shared_parameter_keys(self):
        return self._key_groups()[0]

    def get_gate_parameter_keys(self):
        return self._key_groups()[1]

    def get_router_parameter_keys(self):
        return self.get_gate_parameter_keys()

    def get_tail_parameter_keys(self):
        return self._key_groups()[2]

    def get_frozen_parameter_keys(self):
        return self._key_groups()[3]

    def get_trainable_parameter_groups(self):
        keys = {
            "shared": set(self.get_shared_parameter_keys()),
            "gate": set(self.get_gate_parameter_keys()),
            "tail": set(self.get_tail_parameter_keys()),
        }
        groups = {name: [] for name in keys}
        for name, param in self.named_parameters():
            for group, group_keys in keys.items():
                if name in group_keys:
                    groups[group].append(param)
        return groups

    def set_trainable_groups(self, shared=False, gate=False, tail=False):
        shared_keys = set(self.get_shared_parameter_keys()) if shared else set()
        gate_keys = set(self.get_gate_parameter_keys()) if gate else set()
        tail_keys = set(self.get_tail_parameter_keys()) if tail else set()
        trainable = shared_keys | gate_keys | tail_keys
        for name, param in self.named_parameters():
            param.requires_grad_(name in trainable)

    def parameter_counts(self):
        shared = set(self.get_shared_parameter_keys())
        gate = set(self.get_gate_parameter_keys())
        tail = set(self.get_tail_parameter_keys())
        frozen = set(self.get_frozen_parameter_keys())
        out = {"shared": 0, "gate": 0, "tail": 0, "frozen": 0, "trainable": 0, "total": 0}
        for name, param in self.named_parameters():
            n = int(param.numel())
            out["total"] += n
            if name in shared:
                out["shared"] += n
            elif name in gate:
                out["gate"] += n
            elif name in tail:
                out["tail"] += n
            else:
                out["frozen"] += n
        out["trainable"] = out["shared"] + out["gate"] + out["tail"]
        out["trainable_ratio"] = out["trainable"] / max(out["total"], 1)
        return out
