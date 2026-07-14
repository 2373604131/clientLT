import os

import torch
from torch.cuda.amp import GradScaler, autocast

from Dassl.dassl.metrics import compute_accuracy
from Dassl.dassl.optim import build_lr_scheduler, build_optimizer
from Dassl.dassl.utils import count_num_param, load_pretrained_weights

from trainers.fedclip import FedClip, load_clip_to_cpu
from trainers.fedtef_diagnostics import assert_finite_outputs
from trainers.fedtef_loss import (
    accumulate_difficulty,
    build_positive_row_mask,
    compute_fedtef_loss,
    mask_classwise_grad,
)
from trainers.fedtef_model import FedTEFCLIP


class FedTEF(FedClip):
    """Topo-FedTEF / FedTEF-ESR clean trainer.

    The trainer intentionally owns only local training behavior. Model forward,
    losses, observer routing, aggregation, and diagnostics live in separate
    fedtef_* modules.
    """

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames
        print(self.dm.dataset)
        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)
        if cfg.TRAINER.PROMPTFL.PREC in ("fp32", "amp"):
            clip_model.float()

        print("Building Topo-FedTEF clean CLIP")
        self.model = FedTEFCLIP(cfg, classnames, clip_model)
        self.num_classes = len(classnames)
        self.cls_num_list = self.get_cls_num_list()
        self._configure_trainable_params()

        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model.prompt_learner, cfg.MODEL.INIT_WEIGHTS)

        self.model.to(self.device)
        param_groups = self._build_param_groups()
        self.optim = build_optimizer(None, cfg.OPTIM, param_groups=param_groups)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("fedtef", self.model, self.optim, self.sched)
        self.scaler = GradScaler() if cfg.TRAINER.PROMPTFL.PREC == "amp" else None
        self.reset_round_stats()

        print(f"# params: {count_num_param(self.model):,}")
        print(f"# prompt learner params: {count_num_param(self.model.prompt_learner):,}")
        print(f"# image adapter params: {count_num_param(self.model.img_adap):,}")
        print(f"# tail residual stream params: {count_num_param(self.model.tail_stream):,}")
        print(
            "Topo-FedTEF objective: "
            "shared acquisition + topology observer + positive residual preservation + "
            "evidence-preserving TailAgg + margin-conditioned release"
        )

        os.environ["CUDA_VISIBLE_DEVICES"] = "0,3,2,1"
        device_count = torch.cuda.device_count()
        if device_count > 1:
            print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")

    def _configure_trainable_params(self):
        cfg = self.cfg.TRAINER.FEDTEF
        for _, param in self.model.named_parameters():
            param.requires_grad_(False)
        if bool(getattr(cfg, "TRAIN_PROMPT", True)):
            for param in self.model.prompt_learner.parameters():
                param.requires_grad_(True)
        if bool(getattr(cfg, "TRAIN_IMG_ADAP", True)):
            for param in self.model.img_adap.parameters():
                param.requires_grad_(True)
        if bool(getattr(cfg, "TRAIN_TAIL_STREAM", True)) and bool(getattr(cfg, "USE_TAIL_EXPERT", True)):
            for param in self.model.tail_stream.parameters():
                param.requires_grad_(True)
        if bool(getattr(cfg, "TRAIN_ROUTED_PROMPT", True)):
            self.model.routed_prompt_delta.requires_grad_(True)

        total = 0
        print("Topo-FedTEF trainable parameters:")
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                total += param.numel()
                print(f"  {name}: {param.numel()}")
        print(f"Topo-FedTEF total trainable params: {total}")

    def _build_param_groups(self):
        cfg = self.cfg
        fedtef_cfg = cfg.TRAINER.FEDTEF
        base_lr = cfg.OPTIM.LR
        groups = []

        prompt_params = [p for p in self.model.prompt_learner.parameters() if p.requires_grad]
        if prompt_params:
            groups.append({"params": prompt_params, "lr": base_lr})

        img_params = [p for p in self.model.img_adap.parameters() if p.requires_grad]
        if img_params:
            groups.append({"params": img_params, "lr": base_lr})

        tail_params = [p for p in self.model.tail_stream.parameters() if p.requires_grad]
        if tail_params:
            groups.append({
                "params": tail_params,
                "lr": base_lr * float(getattr(fedtef_cfg, "TAIL_EXPERT_LR_MULT", 8.0)),
            })

        if self.model.routed_prompt_delta.requires_grad:
            groups.append({
                "params": [self.model.routed_prompt_delta],
                "lr": base_lr * float(getattr(fedtef_cfg, "ROUTED_PROMPT_LR_MULT", 2.0)),
            })

        if not groups:
            raise ValueError("Topo-FedTEF has no trainable parameters; check TRAIN_* config flags.")
        return groups

    def reset_optimizer_and_scheduler(self):
        self.optim = build_optimizer(None, self.cfg.OPTIM, param_groups=self._build_param_groups())
        self.sched = build_lr_scheduler(self.optim, self.cfg.OPTIM)
        self._optims["fedtef"] = self.optim
        self._scheds["fedtef"] = self.sched
        self.scaler = GradScaler() if self.cfg.TRAINER.PROMPTFL.PREC == "amp" else None

    def current_lr(self):
        if self.optim is None or not self.optim.param_groups:
            return 0.0
        return float(self.optim.param_groups[0]["lr"])

    def reset_round_stats(self):
        num_classes = getattr(self.model, "num_classes", getattr(self, "num_classes", 0))
        self.fedtef_v10_difficulty = torch.zeros(num_classes, dtype=torch.float32)
        self.fedtef_v10_difficulty_count = torch.zeros(num_classes, dtype=torch.float32)

    def _accumulate_difficulty(self, outputs, labels):
        margin = float(getattr(self.cfg.TRAINER.FEDTEF, "V10_DIFFICULTY_MARGIN", 1.0))
        diff_sum, diff_count = accumulate_difficulty(
            outputs["logits_base"].detach(),
            labels,
            margin_target=margin,
        )
        self.fedtef_v10_difficulty += diff_sum.detach().cpu()
        self.fedtef_v10_difficulty_count += diff_count.detach().cpu()

    def _mask_tail_and_routed_grad(self, labels):
        cfg = self.cfg.TRAINER.FEDTEF
        row_mask = build_positive_row_mask(
            labels,
            self.model.tail_gate.detach(),
            self.model.num_classes,
            eps=float(getattr(cfg, "EXPOSURE_EPS", 1e-6)),
        )
        mask_classwise_grad(self.model.tail_stream.weight, row_mask)
        mask_classwise_grad(self.model.tail_stream.bias, row_mask)

        if bool(getattr(cfg, "TRAIN_ROUTED_PROMPT", True)):
            routed_mask = row_mask
            if bool(getattr(cfg, "ROUTED_PROMPT_UPDATE_ALL_ROWS", False)):
                routed_mask = torch.zeros_like(row_mask)
                routed_mask[labels.unique()] = True
            mask_classwise_grad(self.model.routed_prompt_delta, routed_mask)

    def _check_grad_finite(self):
        for name, param in self.model.named_parameters():
            if param.grad is not None and not torch.isfinite(param.grad).all():
                raise FloatingPointError(f"Gradient for {name} is NaN/Inf")

    def forward_backward(self, batch, global_weight=None, fedprox=False, mu=0.5):
        image, labels = self.parse_batch_train(batch)
        if getattr(self, "batch_idx", 0) == 0:
            self.reset_round_stats()
        prec = self.cfg.TRAINER.PROMPTFL.PREC
        gate = self.model.tail_gate.detach()
        tail_score = self.model.tail_score.detach()

        if prec == "amp":
            with autocast():
                outputs = self.model(image, return_dict=True)
                assert_finite_outputs(outputs)
                loss, loss_items = compute_fedtef_loss(
                    outputs,
                    labels,
                    gate=gate,
                    tail_score=tail_score,
                    cfg=self.cfg.TRAINER.FEDTEF,
                )
            self._accumulate_difficulty(outputs, labels)
            self.optim.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optim)
            self._mask_tail_and_routed_grad(labels)
            self._check_grad_finite()
            self.scaler.step(self.optim)
            self.scaler.update()
        else:
            outputs = self.model(image, return_dict=True)
            assert_finite_outputs(outputs)
            loss, loss_items = compute_fedtef_loss(
                outputs,
                labels,
                gate=gate,
                tail_score=tail_score,
                cfg=self.cfg.TRAINER.FEDTEF,
            )
            self._accumulate_difficulty(outputs, labels)
            self.model_zero_grad()
            self.model_backward(loss)
            self._mask_tail_and_routed_grad(labels)
            self._check_grad_finite()
            self.model_update()

        logits = outputs["logits"].detach()
        loss_items["acc"] = compute_accuracy(logits, labels)[0].item()
        loss_items["base_acc"] = compute_accuracy(outputs["logits_base"].detach(), labels)[0].item()

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()
        return loss_items
