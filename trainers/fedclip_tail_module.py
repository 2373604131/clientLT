import os

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast

from Dassl.dassl.metrics import compute_accuracy
from Dassl.dassl.optim import build_optimizer, build_lr_scheduler
from Dassl.dassl.utils import count_num_param, load_pretrained_weights

from trainers.fedclip import CustomCLIP as FedClipCustomCLIP
from trainers.fedclip import FedClip, load_clip_to_cpu


def build_tail_class_mask(cls_num_list, cutoff):
    sorted_classes = sorted(
        range(len(cls_num_list)),
        key=lambda class_id: cls_num_list[class_id],
        reverse=True,
    )
    total_samples = sum(cls_num_list)
    cumulative_samples = 0
    mask = torch.zeros(len(cls_num_list), dtype=torch.bool)

    for class_id in sorted_classes:
        cumulative_samples += cls_num_list[class_id]
        if cumulative_samples > cutoff * total_samples:
            mask[class_id] = True

    return mask


class TailPromptResidualCLIP(FedClipCustomCLIP):
    """FedClip with a tiny class-specific residual on tail-class prompts."""

    def __init__(self, cfg, classnames, clip_model, tail_class_mask):
        super().__init__(cfg, classnames, clip_model)

        n_cls = len(classnames)
        n_ctx = self.prompt_learner.n_ctx
        ctx_dim = self.prompt_learner.token_prefix.shape[-1]
        self.tail_prompt_residual = nn.Parameter(
            torch.zeros(n_cls, n_ctx, ctx_dim, dtype=self.dtype)
        )
        self.register_buffer(
            "tail_class_mask",
            tail_class_mask.to(dtype=self.dtype).view(n_cls, 1, 1),
        )

    def forward(self, image, return_features=False):
        image_features = self.image_encoder(image.type(self.dtype))
        image_features_att = self.img_adap(image_features)
        image_features = torch.mul(image_features_att, image_features)

        prompts = self.prompt_learner()
        prompts = prompts.clone()
        n_ctx = self.prompt_learner.n_ctx
        prompts[:, 1:1 + n_ctx, :] = (
            prompts[:, 1:1 + n_ctx, :]
            + self.tail_prompt_residual * self.tail_class_mask
        )

        tokenized_prompts = self.tokenized_prompts
        text_features = self.text_encoder(prompts, tokenized_prompts)

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.t()

        if return_features:
            return image_features, logits, text_features
        return logits


class FedClipTailModule(FedClip):
    """FedClip plus a class-aware tail prompt residual module."""

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames
        print(self.dm.dataset)

        self.cls_num_list = self.get_cls_num_list()
        tail_cutoff = cfg.TRAINER.FEDCLIP_TAIL.CUTOFF
        tail_class_mask = build_tail_class_mask(self.cls_num_list, tail_cutoff)
        tail_classes = torch.nonzero(tail_class_mask, as_tuple=False).view(-1).tolist()
        print(f"FedClipTailModule tail cutoff: {tail_cutoff}")
        print(f"FedClipTailModule tail classes: {tail_classes}")

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)

        if cfg.TRAINER.PROMPTFL.PREC == "fp32" or cfg.TRAINER.PROMPTFL.PREC == "amp":
            clip_model.float()

        print("Building custom CLIP with tail prompt residual")
        self.model = TailPromptResidualCLIP(cfg, classnames, clip_model, tail_class_mask)

        print("Turning off gradients except img_adap and tail_prompt_residual")
        for name, param in self.model.named_parameters():
            if "img_adap" not in name and "tail_prompt_residual" not in name:
                param.requires_grad_(False)
        for param in self.model.img_adap.parameters():
            param.requires_grad = True
        self.model.tail_prompt_residual.requires_grad_(True)

        print(f"# params: {count_num_param(self.model):,}")
        print(f"# img adapter params: {count_num_param(self.model.img_adap):,}")
        print(f"# tail prompt residual params: {self.model.tail_prompt_residual.numel():,}")

        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model.prompt_learner, cfg.MODEL.INIT_WEIGHTS)

        self.model.to(self.device)
        trainable_params = list(self.model.img_adap.parameters()) + [self.model.tail_prompt_residual]
        self.optim = build_optimizer(trainable_params, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("fedclip_tail_module", self.model, self.optim, self.sched)

        self.scaler = GradScaler() if cfg.TRAINER.PROMPTFL.PREC == "amp" else None

        os.environ["CUDA_VISIBLE_DEVICES"] = "0,3,2,1"
        device_count = torch.cuda.device_count()
        if device_count > 1:
            print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")

    def forward_backward(self, batch, global_weight=None, fedprox=False, mu=0.5):
        image, label = self.parse_batch_train(batch)
        prec = self.cfg.TRAINER.PROMPTFL.PREC

        if prec == "amp":
            with autocast():
                output = self.model(image)
                loss = F.cross_entropy(output, label)
            self.optim.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optim)
            self.mask_grad(label)
            self.scaler.step(self.optim)
            self.scaler.update()
        else:
            _, output, _ = self.model(image, return_features=True)
            loss = F.cross_entropy(output, label)
            if fedprox:
                model_weight = self.model.state_dict()
                fed_prox_reg = (
                    (mu / 2)
                    * torch.norm(
                        model_weight["tail_prompt_residual"]
                        - global_weight["tail_prompt_residual"]
                    ) ** 2
                )
                loss += fed_prox_reg
            self.model_zero_grad()
            self.model_backward(loss)
            self.mask_grad(label)
            self.model_update()

        loss_summary = {
            "loss": loss.item(),
            "acc": compute_accuracy(output, label)[0].item(),
        }

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary

    def mask_grad(self, labels):
        param = self.model.tail_prompt_residual
        if param.grad is None:
            return

        tail_mask = self.model.tail_class_mask.view(-1).bool()
        unique_labels = torch.unique(labels)
        selected_labels = unique_labels[tail_mask[unique_labels]]
        grad_mask = torch.zeros_like(param.data)
        if selected_labels.numel() > 0:
            grad_mask[selected_labels, :, :] = 1
        param.grad.data.mul_(grad_mask)
