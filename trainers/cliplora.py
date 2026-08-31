import os.path as osp

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast

# from Dassl.dassl.engine import TRAINER_REGISTRY, TrainerX
from Dassl.dassl.metrics import compute_accuracy
from Dassl.dassl.engine.trainer import TrainerX
from Dassl.dassl.utils import load_pretrained_weights, load_checkpoint
from Dassl.dassl.optim import build_optimizer, build_lr_scheduler

from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
from utils.loralib.utils import mark_only_lora_as_trainable, apply_lora, get_lora_parameters, lora_state_dict, save_lora, load_lora
from utils.cliplora_loss import fixed_denominator_cross_entropy
from utils.class_residual import (
    ClassResidualHead,
    mask_class_residual_gradients,
    unwrap_model,
)


_tokenizer = _Tokenizer()


def load_clip_to_cpu(cfg):
    backbone_name = cfg.MODEL.BACKBONE.NAME
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")
    design_details = {"trainer": 'CoOp',
                      "vision_depth": 0,
                      "language_depth": 0, "vision_ctx": 0,
                      "language_ctx": 0}
    model = clip.build_model(state_dict or model.state_dict(), design_details)

    return model


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x


class PromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = cfg.TRAINER.COOP.N_CTX
        ctx_init = cfg.TRAINER.CLIPLORA.CTX_INIT
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = cfg.INPUT.SIZE[0]
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        ctx_init = ctx_init.replace("_", " ")
        n_ctx = len(ctx_init.split(" "))
        prompt = clip.tokenize(ctx_init)
        with torch.no_grad():
            embedding = clip_model.token_embedding(prompt).type(dtype)
        ctx_vectors = embedding[0, 1 : 1 + n_ctx, :]
        prompt_prefix = ctx_init

        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")

        self.ctx = nn.Parameter(ctx_vectors)  # to be optimized

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx :, :])  # CLS, EOS

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens
        self.class_token_position = cfg.TRAINER.COOP.CLASS_TOKEN_POSITION

    def forward(self):
        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        prefix = self.token_prefix
        suffix = self.token_suffix
        prompts = torch.cat(
            [
                prefix,  # (n_cls, 1, dim)
                ctx,     # (n_cls, n_ctx, dim)
                suffix,  # (n_cls, *, dim)
            ],
            dim=1,
        )

        return prompts


class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.prompt_learner = PromptLearner(cfg, classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        self.sca_enabled = bool(getattr(cfg.TRAINER.CLIPLORA, "SCA_ENABLED", False))
        self.class_residual = None
        if self.sca_enabled:
            self.class_residual = ClassResidualHead(
                num_classes=len(classnames),
                feature_dim=int(clip_model.visual.output_dim),
                scale=float(getattr(cfg.TRAINER.CLIPLORA, "SCA_SCALE", 10.0)),
                clamp=float(getattr(cfg.TRAINER.CLIPLORA, "SCA_CLAMP", 3.0)),
                use_bias=bool(getattr(cfg.TRAINER.CLIPLORA, "SCA_USE_BIAS", True)),
            )

        # 应用LoRA
        self.list_lora_layers = apply_lora(cfg, clip_model)
        mark_only_lora_as_trainable(clip_model)

    def forward(self, image):
        image_features = self.image_encoder(image.type(self.dtype))

        prompts = self.prompt_learner()
        tokenized_prompts = self.tokenized_prompts
        text_features = self.text_encoder(prompts, tokenized_prompts)

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.t()

        if self.class_residual is not None:
            logits = logits + self.class_residual(image_features).to(logits.dtype)

        return logits


def build_cliplora_model(cfg, classnames):
    """Build the exact model used by ``ClipLora`` without a DataManager.

    Mechanism experiments use this entry point so their model construction and
    trainable-parameter semantics cannot drift from the federated trainer.
    """
    print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
    clip_model = load_clip_to_cpu(cfg)
    if cfg.TRAINER.COOP.PREC in ("fp32", "amp"):
        clip_model.float()

    print("Building custom CLIP")
    model = CustomCLIP(cfg, classnames, clip_model)
    for name, param in model.named_parameters():
        param.requires_grad = "lora_" in name or (
            bool(getattr(cfg.TRAINER.CLIPLORA, "SCA_ENABLED", False))
            and name.startswith("class_residual.")
        )

    trainable = [(name, param) for name, param in model.named_parameters() if param.requires_grad]
    if not trainable:
        raise RuntimeError("ClipLora has no trainable LoRA parameters")
    if cfg.TRAINER.CLIPLORA.encoder == "vision":
        text_lora = [name for name, _ in trainable if name.startswith("text_encoder.")]
        if text_lora:
            raise RuntimeError(f"Vision-only ClipLora unexpectedly exposed text LoRA parameters: {text_lora}")

    print(
        "ClipLora effective config: "
        f"encoder={cfg.TRAINER.CLIPLORA.encoder} "
        f"position={cfg.TRAINER.CLIPLORA.position} "
        f"rank={cfg.TRAINER.CLIPLORA.r} "
        f"alpha={cfg.TRAINER.CLIPLORA.alpha} "
        f"params={list(cfg.TRAINER.CLIPLORA.params)} "
        f"dropout={cfg.TRAINER.CLIPLORA.dropout_rate} "
        f"sca_enabled={bool(getattr(cfg.TRAINER.CLIPLORA, 'SCA_ENABLED', False))} "
        f"precision={cfg.TRAINER.COOP.PREC} "
        f"trainable_params={sum(param.numel() for _, param in trainable)}"
    )
    if cfg.MODEL.INIT_WEIGHTS:
        load_pretrained_weights(model.prompt_learner, cfg.MODEL.INIT_WEIGHTS)
    return model


def build_cliplora_optimizer_and_scheduler(model, cfg):
    """Use the same optimizer/scheduler factories as the federated trainer."""
    lora_params = list(get_lora_parameters(model))
    if bool(getattr(cfg.TRAINER.CLIPLORA, "SCA_ENABLED", False)):
        core = unwrap_model(model)
        residual_params = [
            param for param in core.class_residual.parameters() if param.requires_grad
        ]
        param_groups = [
            {"params": lora_params, "lr": float(cfg.OPTIM.LR)},
            {
                "params": residual_params,
                "lr": float(cfg.OPTIM.LR)
                * float(getattr(cfg.TRAINER.CLIPLORA, "SCA_LR_MULT", 5.0)),
            },
        ]
        optim = build_optimizer(None, cfg.OPTIM, param_groups=param_groups)
    else:
        optim = build_optimizer(lora_params, cfg.OPTIM)
    sched = build_lr_scheduler(optim, cfg.OPTIM)
    return optim, sched


def cliplora_optimizer_step(
    model, optimizer, scaler, precision, images, labels, loss_weight=None,
    reject_nonfinite_amp=False, post_backward=None,
):
    """One canonical ClipLora optimizer step, shared by trainer and audits."""
    if precision == "amp":
        old_scale = float(scaler.get_scale())
        with autocast():
            output = model(images)
            loss = fixed_denominator_cross_entropy(output, labels, loss_weight)
        if reject_nonfinite_amp and not torch.isfinite(loss).all():
            raise FloatingPointError("Loss is infinite or NaN")
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        if post_backward is not None:
            scaler.unscale_(optimizer)
            post_backward()
        scaler.step(optimizer)
        scaler.update()
        new_scale = float(scaler.get_scale())
    else:
        output = model(images)
        loss = fixed_denominator_cross_entropy(output, labels, loss_weight)
        if not torch.isfinite(loss).all():
            raise FloatingPointError("Loss is infinite or NaN")
        optimizer.zero_grad()
        loss.backward()
        if post_backward is not None:
            post_backward()
        optimizer.step()
        old_scale = new_scale = None
    return output, loss, {
        "amp_scale_before": old_scale,
        "amp_scale_after": new_scale,
        "amp_overflow": bool(new_scale < old_scale) if old_scale is not None else False,
    }


# @TRAINER_REGISTRY.register()
class ClipLora(TrainerX):
    """Context Optimization (CoOp).

    Learning to Prompt for Vision-Language Models
    https://arxiv.org/abs/2109.01134
    """

    def check_cfg(self, cfg):
        assert cfg.TRAINER.COOP.PREC in ["fp16", "fp32", "amp"]

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames
        self.model = build_cliplora_model(cfg, classnames)

        self.model.to(self.device)
        self.cls_num_list = self.get_cls_num_list()
        # 只优化LoRA参数
        self.optim, self.sched = build_cliplora_optimizer_and_scheduler(self.model, cfg)
        self.register_model("lora", self.model, self.optim, self.sched)

        self.scaler = GradScaler() if cfg.TRAINER.COOP.PREC == "amp" else None

        # Note that multi-gpu training could be slow because CLIP's size is
        # big, which slows down the copy operation in DataParallel
        device_count = torch.cuda.device_count()
        if device_count > 1:
            print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
            self.model = nn.DataParallel(self.model)

    def reset_optimizer_and_scheduler(self):
        """Start every FedAvg client from an independent local optimizer."""
        new_optim, new_sched = build_cliplora_optimizer_and_scheduler(self.model, self.cfg)
        self.optim = new_optim
        self.sched = new_sched
        self._optims["lora"] = new_optim
        self._scheds["lora"] = new_sched
        self.scaler = GradScaler() if self.cfg.TRAINER.COOP.PREC == "amp" else None

    def forward_backward(self, batch):
        image, label = self.parse_batch_train(batch)
        prec = self.cfg.TRAINER.COOP.PREC
        output, loss, _ = cliplora_optimizer_step(
            self.model,
            self.optim,
            self.scaler,
            prec,
            image,
            label,
            batch.get("loss_weight"),
            post_backward=(
                (lambda: mask_class_residual_gradients(self.model, label))
                if bool(getattr(self.cfg.TRAINER.CLIPLORA, "SCA_ENABLED", False))
                else None
            ),
        )

        loss_summary = {
            "loss": loss.item(),
            "acc": compute_accuracy(output, label)[0].item(),
        }

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary

    def parse_batch_train(self, batch):
        input = batch["img"]
        label = batch["label"]
        input = input.to(self.device)
        label = label.to(self.device)
        return input, label


    def get_cls_num_list(self):
        y_train = self.dm.dataset.y_train
        cls_num_list = [0] * self.num_classes
        for label in y_train:
            cls_num_list[label] += 1
        # print("cls_num_list:", cls_num_list)
        return cls_num_list


    def load_model(self, directory, epoch=None):
        if not directory:
            print("Note that load_model() is skipped as no pretrained model is given")
            return

        names = self.get_model_names()

        # By default, the best model is loaded
        model_file = "model-best.pth.tar"

        if epoch is not None:
            model_file = "model.pth.tar-" + str(epoch)

        for name in names:
            model_path = osp.join(directory, name, model_file)

            if not osp.exists(model_path):
                raise FileNotFoundError('Model not found at "{}"'.format(model_path))

            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint["state_dict"]
            epoch = checkpoint["epoch"]

            # Ignore fixed token vectors
            if "token_prefix" in state_dict:
                del state_dict["token_prefix"]

            if "token_suffix" in state_dict:
                del state_dict["token_suffix"]

            print("Loading weights to {} " 'from "{}" (epoch = {})'.format(name, model_path, epoch))
            # set strict=False
            self._models[name].load_state_dict(state_dict, strict=False)
