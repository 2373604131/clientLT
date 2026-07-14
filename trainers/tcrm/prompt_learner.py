from typing import Iterable, List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from clip import clip
except Exception:  # pragma: no cover - lightweight unit tests may not import CLIP.
    clip = None


def clip_tokenize(texts):
    if clip is None:
        raise RuntimeError("CLIP tokenizer is unavailable; install project CLIP dependencies.")
    return clip.tokenize(texts)


def clip_tokenize_to_model_device(texts, clip_model):
    device = clip_model.token_embedding.weight.device
    return clip_tokenize(texts).to(device)


class FrozenTextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype
        for param in self.parameters():
            param.requires_grad_(False)

    def forward(self, prompts, tokenized_prompts):
        tokenized_prompts = tokenized_prompts.to(prompts.device)
        x = prompts + self.positional_embedding.type(prompts.dtype)
        x = x.permute(1, 0, 2)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)
        x = self.ln_final(x).type(prompts.dtype)
        x = x[torch.arange(x.shape[0], device=x.device), tokenized_prompts.argmax(dim=-1)] @ self.text_projection
        return x


class GeneralPromptLearner(nn.Module):
    """CAPT/CoOp-style general context, extracted without Dassl or vision prompts."""

    def __init__(self, clip_model, classnames, prompt_ctx="a photo of a", n_ctx=4):
        super().__init__()
        self.classnames = [name.replace("_", " ") for name in classnames]
        self.prompt_ctx = str(prompt_ctx or "a photo of a")
        self.n_ctx = int(n_ctx)
        prompts = [f"{self.prompt_ctx} {name}." for name in self.classnames]
        tokenized = clip_tokenize_to_model_device(prompts, clip_model)
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized).type(clip_model.dtype).detach().clone()
        self.register_buffer("fixed_embedding", embedding)
        self.register_buffer("tokenized_prompts", tokenized)
        self.ctx_delta = nn.Parameter(torch.zeros(self.n_ctx, embedding.shape[-1], dtype=torch.float32))
        self.text_encoder = FrozenTextEncoder(clip_model)
        self.dtype = clip_model.dtype

    def prompt_embeddings(self, class_ids=None):
        if class_ids is not None:
            class_ids = torch.as_tensor(class_ids, dtype=torch.long, device=self.fixed_embedding.device)
        prompts = self.fixed_embedding if class_ids is None else self.fixed_embedding.index_select(0, class_ids)
        prompts = prompts.clone()
        usable = min(self.n_ctx, prompts.shape[1] - 2)
        prompts[:, 1:1 + usable, :] = prompts[:, 1:1 + usable, :] + self.ctx_delta[:usable].to(prompts.dtype)
        return prompts

    def forward(self, class_ids=None):
        if class_ids is None:
            tokenized = self.tokenized_prompts
        else:
            class_ids = torch.as_tensor(class_ids, dtype=torch.long, device=self.tokenized_prompts.device)
            tokenized = self.tokenized_prompts.index_select(0, class_ids)
        prompts = self.prompt_embeddings(class_ids)
        features = self.text_encoder(prompts, tokenized)
        return F.normalize(features.float(), dim=-1)

    def trainable_state(self):
        return {"ctx_delta": self.ctx_delta.detach().cpu().float().clone()}

    def load_trainable_state(self, state):
        if "ctx_delta" not in state:
            raise KeyError("TCRM prompt state is missing ctx_delta")
        self.ctx_delta.data.copy_(state["ctx_delta"].to(self.ctx_delta.device, dtype=self.ctx_delta.dtype))


@torch.no_grad()
def zero_shot_text_features(clip_model, classnames, prompt_ctx="a photo of a", device="cpu"):
    learner = GeneralPromptLearner(clip_model, classnames, prompt_ctx=prompt_ctx, n_ctx=1).to(device)
    learner.ctx_delta.zero_()
    text = learner.forward().detach().cpu().float()
    return F.normalize(text, dim=-1)
