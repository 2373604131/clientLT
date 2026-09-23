"""Opt-in execution helpers. No losses, aggregation rules or training budgets."""
import torch


FAST_EXECUTION_CONFIG = dict(
    version=1, text_cache=True, witness_image_cache=True,
    frozen_visual_prefix_cache='cpu_original_dtype', lora_only_restore=True,
    low_rank_forward=True, feedback_forward_batch_size=32,
    feedback_batching='within_client; individual source gradients retained',
    zero_gradient_skip='exactly zero functional coefficient only',
)


def tensor_stamp(tensors):
    # Loading new weights or changing device/dtype invalidates derived caches.
    return tuple((id(t), t._version, t.device, t.dtype, t.data_ptr()) for t in tensors)


class FrozenTextCache:
    def __init__(self, core):
        self.core = core
        self.tensors = (list(core.prompt_learner.parameters()) + list(core.prompt_learner.buffers())
                        + list(core.text_encoder.parameters()) + list(core.text_encoder.buffers())
                        + [core.tokenized_prompts])
        self.stamp, self.value = None, None

    def get(self):
        stamp = tensor_stamp(self.tensors)
        if stamp != self.stamp:
            with torch.no_grad():
                text = self.core.text_encoder(self.core.prompt_learner(), self.core.tokenized_prompts)
                self.value = (text / text.norm(dim=-1, keepdim=True)).detach()
            self.stamp = stamp
        return self.value


def enable_model_execution(model):
    """Only installed by the SFRA runtime, after its vision-only checks."""
    from utils.loralib.layers import LinearLoRA
    core = model.module if hasattr(model, 'module') else model
    text_parameters = list(core.prompt_learner.parameters()) + list(core.text_encoder.parameters())
    if any(p.requires_grad for p in text_parameters):
        raise ValueError('SFRA text caching requires frozen text and prompt parameters')
    core._sfra_text_cache = FrozenTextCache(core)
    for module in core.modules():
        if isinstance(module, LinearLoRA):
            module._sfra_low_rank_execution = True


class LoRAStateLoader:
    """Load the full base once, then restore BOTH factors between clients.

    Scoped to SFRA: its runtime asserts that all other parameters are frozen,
    and its CLIP has no mutable running-statistic buffers. A new runtime/loader
    always loads the full state again, including when resuming a checkpoint.
    """
    def __init__(self, model, keys):
        self.model, self.keys = model, tuple(keys)
        self.parameters = dict(model.named_parameters())
        self.initialized = False

    def load(self, state):
        if not self.initialized:
            self.model.load_state_dict(state, strict=True)
            self.initialized = True
            return
        with torch.no_grad():
            for key in self.keys:
                self.parameters[key].copy_(state[key])


class FrozenVisionPrefix:
    """Exact split of the repository's unprompted VisionTransformer forward.

    Cache ALL patch/CLS tokens before the first LoRA block, not the final CLS
    embedding. The suffix remains differentiable through every adapted block.
    """
    def __init__(self, encoder):
        self.encoder = encoder
        if encoder.VPT_shallow:
            raise ValueError('SFRA prefix caching expects the unprompted vision encoder')
        blocks = list(encoder.transformer.resblocks)
        self.first = next(i for i, block in enumerate(blocks)
                          if any(n.endswith('_lora_A') for n, _ in block.named_parameters()))
        self.prefix, self.suffix = blocks[:self.first], blocks[self.first:]
        self.tensors = [encoder.class_embedding, encoder.positional_embedding]
        for module in [encoder.conv1, encoder.ln_pre, *self.prefix]:
            self.tensors.extend(module.parameters())
            self.tensors.extend(module.buffers())
        if any(t.requires_grad for t in self.tensors):
            raise ValueError('Cannot cache a trainable visual prefix')

    @torch.no_grad()
    def forward_prefix(self, images):
        e = self.encoder
        x = e.conv1(images)
        x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)
        cls = e.class_embedding.to(x.dtype) + torch.zeros(
            x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device)
        x = torch.cat([cls, x], dim=1)
        x = e.ln_pre(x + e.positional_embedding.to(x.dtype)).permute(1, 0, 2)
        for block in self.prefix:
            x = block(x)
        return x.permute(1, 0, 2).detach()

    def forward_suffix(self, tokens):
        x = tokens.permute(1, 0, 2)
        for block in self.suffix:
            x = block(x)
        x = self.encoder.ln_post(x.permute(1, 0, 2)[:, 0, :])
        if self.encoder.proj is not None:
            x = x @ self.encoder.proj
        return x
