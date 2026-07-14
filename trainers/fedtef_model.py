import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _cfg_value(cfg, name, default):
    return getattr(cfg, name, default) if cfg is not None else default


class ImageAdapter(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Tanh(),
            nn.Linear(dim, dim),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x):
        return self.net(x)


class TailResidualStream(nn.Module):
    def __init__(
        self,
        feature_dim,
        num_classes,
        dtype,
        init_mode="normal_residual",
        init_logit_scale=10.0,
        learnable_scale=True,
        use_bias=True,
        logit_scale_max=100.0,
        residual_clamp=3.0,
    ):
        super().__init__()
        if str(init_mode).lower() == "zero_residual":
            raise ValueError(
                "cosine residual tail stream cannot use zero_residual init; "
                "use normal_residual or implement a separate linear residual stream."
            )
        self.weight = nn.Parameter(torch.empty(num_classes, feature_dim, dtype=dtype))
        nn.init.normal_(self.weight, std=0.02)
        if use_bias:
            self.bias = nn.Parameter(torch.zeros(num_classes, dtype=dtype))
        else:
            self.register_parameter("bias", None)
        log_scale = torch.tensor(math.log(max(float(init_logit_scale), 1e-6)), dtype=dtype)
        if learnable_scale:
            self.logit_scale = nn.Parameter(log_scale)
        else:
            self.register_buffer("logit_scale", log_scale)
        self.logit_scale_max = float(logit_scale_max)
        self.residual_clamp = float(residual_clamp)

    def forward(self, image_features):
        features = F.normalize(image_features.float(), dim=-1)
        weight = F.normalize(self.weight.float(), dim=-1)
        scale = self.logit_scale.float().exp().clamp(max=self.logit_scale_max)
        residual = scale * (features @ weight.t())
        if self.bias is not None:
            residual = residual + self.bias.float()
        if self.residual_clamp > 0:
            residual = residual.clamp(-self.residual_clamp, self.residual_clamp)
        return residual.to(dtype=image_features.dtype)


class FedTEFCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        from trainers.promptfl import PromptLearner, TextEncoder

        fedtef_cfg = cfg.TRAINER.FEDTEF
        self.prompt_learner = PromptLearner(cfg, classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype

        self.num_classes = len(classnames)
        feature_dim = clip_model.visual.output_dim
        text_dim = clip_model.ln_final.weight.shape[0]

        self.use_tail_expert = bool(_cfg_value(fedtef_cfg, "USE_TAIL_EXPERT", True))
        self.train_img_adapter = bool(_cfg_value(fedtef_cfg, "TRAIN_IMG_ADAP", True))
        self.train_routed_prompt = bool(_cfg_value(fedtef_cfg, "TRAIN_ROUTED_PROMPT", True))
        self.adapter_eta = float(
            _cfg_value(fedtef_cfg, "IMG_ADAP_ETA", _cfg_value(fedtef_cfg, "ADAPTER_ETA", 0.3))
        )
        self.fusion_lambda = float(_cfg_value(fedtef_cfg, "FUSION_LAMBDA", 0.5))
        self.release_floor = float(_cfg_value(fedtef_cfg, "V10_RELEASE_FLOOR", 0.3))
        self.sample_lambda_min = float(_cfg_value(fedtef_cfg, "V10_SAMPLE_LAMBDA_MIN", 0.2))
        self.sample_lambda_max = float(_cfg_value(fedtef_cfg, "V10_SAMPLE_LAMBDA_MAX", 1.0))
        self.sample_margin = float(_cfg_value(fedtef_cfg, "V10_SAMPLE_MARGIN", 1.0))
        self.sample_temperature = max(float(_cfg_value(fedtef_cfg, "V10_SAMPLE_TEMPERATURE", 1.0)), 1e-6)
        self.scale_calibration = bool(_cfg_value(fedtef_cfg, "SCALE_CALIBRATION", True))
        self.scale_clamp_max = float(_cfg_value(fedtef_cfg, "SCALE_CLAMP_MAX", 3.0))
        self.tail_stream_detach_base = bool(_cfg_value(fedtef_cfg, "TAIL_STREAM_DETACH_BASE", True))
        self.routed_prompt_scale = float(_cfg_value(fedtef_cfg, "ROUTED_PROMPT_SCALE", 0.5))
        self.n_ctx = int(getattr(self.prompt_learner, "n_ctx", cfg.TRAINER.PROMPTFL.N_CTX))

        self.img_adap = ImageAdapter(feature_dim).to(self.dtype)
        self.tail_stream = TailResidualStream(
            feature_dim=feature_dim,
            num_classes=self.num_classes,
            dtype=self.dtype,
            init_mode=str(_cfg_value(fedtef_cfg, "INIT_TAIL_MODE", "normal_residual")),
            init_logit_scale=float(_cfg_value(fedtef_cfg, "TAIL_INIT_LOGIT_SCALE", 10.0)),
            learnable_scale=bool(_cfg_value(fedtef_cfg, "TAIL_LEARNABLE_SCALE", True)),
            use_bias=bool(_cfg_value(fedtef_cfg, "TAIL_USE_BIAS", True)),
            logit_scale_max=float(_cfg_value(fedtef_cfg, "TAIL_LOGIT_SCALE_MAX", 100.0)),
            residual_clamp=float(_cfg_value(fedtef_cfg, "RESIDUAL_CLAMP", 3.0)),
        )
        self.routed_prompt_delta = nn.Parameter(
            torch.zeros(self.num_classes, self.n_ctx, text_dim, dtype=self.dtype)
        )
        self.register_buffer("tail_gate", torch.zeros(self.num_classes, dtype=torch.float32))
        self.register_buffer("tail_score", torch.ones(self.num_classes, dtype=torch.float32))
        self.register_buffer("protected_tail_mask", torch.zeros(self.num_classes, dtype=torch.bool))
        self.register_buffer("tail_reliability", torch.ones(self.num_classes, dtype=torch.float32))

    @property
    def image_adapter(self):
        return self.img_adap

    def set_tail_context(
        self,
        tail_score=None,
        protected_mask=None,
        protected_tail_mask=None,
        gate=None,
        release_reliability=None,
    ):
        if protected_tail_mask is not None and protected_mask is None:
            protected_mask = protected_tail_mask
        if gate is None and protected_mask is not None:
            gate = torch.as_tensor(protected_mask).float()
        if tail_score is not None:
            self.tail_score.copy_(torch.as_tensor(tail_score, dtype=torch.float32, device=self.tail_score.device))
        if protected_mask is not None:
            self.protected_tail_mask.copy_(
                torch.as_tensor(protected_mask, dtype=torch.bool, device=self.protected_tail_mask.device)
            )
        if gate is not None:
            self.tail_gate.copy_(torch.as_tensor(gate, dtype=torch.float32, device=self.tail_gate.device))
        if release_reliability is not None:
            self.tail_reliability.copy_(
                torch.as_tensor(release_reliability, dtype=torch.float32, device=self.tail_reliability.device).clamp(0.0, 1.0)
            )

    def apply_routed_prompt(self, prompts):
        if not self.train_routed_prompt:
            return prompts
        gate = self.tail_gate.detach().to(prompts.device, prompts.dtype)
        delta = self.routed_prompt_scale * gate.view(-1, 1, 1) * self.routed_prompt_delta
        routed = prompts.clone()
        routed[:, 1:1 + self.n_ctx, :] = routed[:, 1:1 + self.n_ctx, :] + delta
        return routed

    def encode_base(self, image):
        image_features = self.image_encoder(image.type(self.dtype))
        if self.train_img_adapter:
            image_features = image_features + self.adapter_eta * self.img_adap(image_features)
        prompts = self.apply_routed_prompt(self.prompt_learner())
        text_features = self.text_encoder(prompts, self.tokenized_prompts)
        image_features = F.normalize(image_features.float(), dim=-1).to(self.dtype)
        text_features = F.normalize(text_features.float(), dim=-1).to(self.dtype)
        logit_scale = self.logit_scale.float().exp().clamp(max=100.0).to(image_features.dtype)
        logits_base = logit_scale * image_features @ text_features.t()
        return image_features, text_features, logits_base

    def _sample_release(self, logits_base):
        values = torch.topk(logits_base.detach().float(), k=min(2, logits_base.shape[1]), dim=1).values
        if values.shape[1] == 1:
            margin = torch.zeros_like(values[:, 0])
        else:
            margin = values[:, 0] - values[:, 1]
        uncertain = torch.sigmoid((self.sample_margin - margin) / self.sample_temperature)
        return self.sample_lambda_min + (self.sample_lambda_max - self.sample_lambda_min) * uncertain

    def _calibrate_residual(self, logits_base, residual_tail):
        if not self.scale_calibration:
            return residual_tail
        base_std = logits_base.detach().float().std(dim=1, keepdim=True).clamp_min(1e-6)
        residual_std = residual_tail.detach().float().std(dim=1, keepdim=True).clamp_min(1e-6)
        scale = (base_std / residual_std).clamp(max=self.scale_clamp_max)
        return residual_tail * scale.to(residual_tail.dtype)

    def forward(self, image, return_features=False, return_dict=False):
        image_features, text_features, logits_base = self.encode_base(image)
        tail_features = image_features.detach() if self.tail_stream_detach_base else image_features
        residual_tail = self.tail_stream(tail_features) if self.use_tail_expert else torch.zeros_like(logits_base)
        residual_tail = self._calibrate_residual(logits_base, residual_tail)

        gate = self.tail_gate.to(device=logits_base.device, dtype=logits_base.dtype)
        reliability = self.tail_reliability.to(device=logits_base.device, dtype=logits_base.dtype).clamp(0.0, 1.0)
        class_release = gate * (self.release_floor + (1.0 - self.release_floor) * reliability)
        sample_release = self._sample_release(logits_base).to(logits_base.dtype).view(-1, 1)
        gated_residual = self.fusion_lambda * sample_release * class_release.view(1, -1) * residual_tail
        logits = logits_base + gated_residual

        if return_dict:
            return {
                "image_features": image_features,
                "text_features": text_features,
                "logits_base": logits_base,
                "base_logits": logits_base,
                "residual_tail": residual_tail,
                "tail_logits": residual_tail,
                "gated_residual": gated_residual,
                "logits": logits,
                "logits_fused": logits,
                "gate": gate,
                "tail_score": self.tail_score.to(device=logits.device, dtype=logits.dtype),
                "release_reliability": reliability,
                "class_release_gate": class_release,
            }
        if return_features:
            return image_features, logits, text_features
        return logits
