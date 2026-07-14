import os.path as osp
import os
import time
import torch
import torch.nn as nn

from torch.cuda.amp import GradScaler, autocast

from Dassl.dassl.engine.trainer import TrainerX

from Dassl.dassl.metrics import compute_accuracy


from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

from Dassl.dassl.data import DataManager
from Dassl.dassl.optim import build_optimizer, build_lr_scheduler
from Dassl.dassl.utils import (
    MetricMeter, AverageMeter, tolist_if_not, count_num_param, load_checkpoint,
    save_checkpoint, mkdir_if_missing, resume_from_checkpoint,
    load_pretrained_weights
)
from loss.prompt_loss import PromptLoss


import numpy as np
import random



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
    design_details = {"trainer": 'CAPT',
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
        n_ctx = cfg.TRAINER.PROMPTFL.N_CTX
        ctx_init = cfg.TRAINER.PROMPTFL.CTX_INIT
        self.dtype = clip_model.dtype  # Store the dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = cfg.INPUT.SIZE[0]
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        if ctx_init:
            # use given words to initialize context vectors
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = len(ctx_init.split(" "))
            prompt = clip.tokenize(ctx_init)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(self.dtype)
            ctx_vectors = embedding[0, 1: 1 + n_ctx, :]
            prompt_prefix = ctx_init
        else:
            if cfg.TRAINER.PROMPTFL.CSC:
                # random initialization
                n_general = cfg.TRAINER.PROMPTFL.n_general
                n_specific = n_ctx - n_general
                print("Initializing a generic context and class-aware contexts")
                self.general_ctx = nn.Parameter(torch.empty(n_general, ctx_dim, dtype=self.dtype))
                self.class_aware_ctx = nn.Parameter(torch.empty(n_cls, n_specific, ctx_dim, dtype=self.dtype))
                nn.init.normal_(self.general_ctx, std=0.02)
                nn.init.normal_(self.class_aware_ctx, std=0.02)
                prompt_prefix = " ".join(["X"] * n_ctx)
                print(f"general_ctx shape: {self.general_ctx.shape}")
                print(f"specific_ctx shape: {self.class_aware_ctx.shape}")
                self.ctx = None
            else:
                print("Initializing a generic context")
                ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=self.dtype)
                nn.init.normal_(ctx_vectors, std=0.02)
                prompt_prefix = " ".join(["X"] * n_ctx)
                self.ctx = nn.Parameter(ctx_vectors)  # to be optimized
                print(f"only general_ctx shape: {self.ctx}")

        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(self.dtype)

        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx:, :])  # CLS, EOS

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens
        self.class_token_position = cfg.TRAINER.PROMPTFL.CLASS_TOKEN_POSITION

    def forward(self, labels=None):
        if self.ctx != None:
            ctx = self.ctx
            if ctx.dim() == 2:
                ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)
        else:
            general_ctx_expanded = self.general_ctx.unsqueeze(0).expand(self.n_cls, -1, -1)
            ctx = torch.cat([general_ctx_expanded, self.class_aware_ctx], dim=1)
            # print(f"Combined ctx shape: {ctx.shape}")
            ctx = ctx.to(self.dtype)

        prefix = self.token_prefix
        suffix = self.token_suffix

        if self.class_token_position == "end":
            prompts = torch.cat(
                [
                    prefix,  # (n_cls, 1, dim)
                    ctx,  # (n_cls, n_ctx, dim)
                    suffix,  # (n_cls, *, dim)
                ],
                dim=1,
            )

        elif self.class_token_position == "middle":
            half_n_ctx = self.n_ctx // 2
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i: i + 1, :, :]
                class_i = suffix[i: i + 1, :name_len, :]
                suffix_i = suffix[i: i + 1, name_len:, :]
                ctx_i_half1 = ctx[i: i + 1, :half_n_ctx, :]
                ctx_i_half2 = ctx[i: i + 1, half_n_ctx:, :]
                prompt = torch.cat(
                    [
                        prefix_i,  # (1, 1, dim)
                        ctx_i_half1,  # (1, n_ctx//2, dim)
                        class_i,  # (1, name_len, dim)
                        ctx_i_half2,  # (1, n_ctx//2, dim)
                        suffix_i,  # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        elif self.class_token_position == "front":
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i: i + 1, :, :]
                class_i = suffix[i: i + 1, :name_len, :]
                suffix_i = suffix[i: i + 1, name_len:, :]
                ctx_i = ctx[i: i + 1, :, :]
                prompt = torch.cat(
                    [
                        prefix_i,  # (1, 1, dim)
                        class_i,  # (1, name_len, dim)
                        ctx_i,  # (1, n_ctx, dim)
                        suffix_i,  # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        else:
            raise ValueError

        return prompts, self.general_ctx



class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.prompt_learner = PromptLearner(cfg, classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype


        self.prompt_loss = PromptLoss(
            num_classes=len(classnames),
            temperature=cfg.TRAINER.PROMPTFL.TEMPERATURE,
            alpha=cfg.TRAINER.PROMPTFL.ALPHA,
            beta=cfg.TRAINER.PROMPTFL.BETA,
            gamma=cfg.TRAINER.PROMPTFL.GAMMA,
            delta=cfg.TRAINER.PROMPTFL.DELTA,
            margin=cfg.TRAINER.PROMPTFL.MARGIN
        )

        # 添加耦合函数 F
        if cfg.MODEL.BACKBONE.NAME == 'ViT-B/16':
            self.coupling_function = nn.Linear(512, 768)  # vitb16
        else:
            self.coupling_function = nn.Linear(512, 64)  # Rn50

        self.coupling_function.half()
        self.coupling_function.requires_grad_(True)


    def forward(self, image, label=None, return_features=False):

        prompts, shared_ctx = self.prompt_learner(label)
        # 使用耦合函数 F 将语言提示映射到视觉提示
        vision_prompts = self.coupling_function(shared_ctx)
        # 使用视觉提示进行图像编码
        image_features = self.image_encoder(image.type(self.dtype), vision_prompts)
        # image_features = self.image_encoder(image.type(self.dtype))
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        # print(f"Prompts shape: {prompts.shape}")

        tokenized_prompts = self.tokenized_prompts
        text_features = self.text_encoder(prompts, tokenized_prompts)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # print(f"Text features shape: {text_features.shape}")

        logit_scale = self.logit_scale.exp()
        ce_logits = logit_scale * image_features @ text_features.t()

        # print(f"CE logits shape: {ce_logits.shape}")

        if return_features:
            return image_features, ce_logits, text_features
        else:
            return ce_logits



class CAPT(TrainerX):

    def check_cfg(self, cfg):
        assert cfg.TRAINER.PROMPTFL.PREC in ["fp16", "fp32", "amp"]

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames
        print(self.dm.dataset)

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)

        if cfg.TRAINER.PROMPTFL.PREC == "fp32" or cfg.TRAINER.PROMPTFL.PREC == "amp":
            # CLIP's default precision is fp16
            clip_model.float()

        print("Building custom CLIP")
        self.model = CustomCLIP(cfg, classnames, clip_model)
        self.cls_num_list = self.get_cls_num_list()

        print("Turning off gradients in both the image and the text encoder")
        for name, param in self.model.named_parameters():
            # print(name,":",param.size())
            # if "prompt_learner" not in name:
            if "prompt_learner" not in name and "coupling_function" not in name:
                param.requires_grad_(False)
        print(f"# params: {count_num_param(self.model):,}")
        print(f"# prompt learner params: {count_num_param(self.model.prompt_learner):,}")


        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model.prompt_learner, cfg.MODEL.INIT_WEIGHTS)

        self.model.to(self.device)


        trainable_params = list(self.model.prompt_learner.parameters()) + list(
            self.model.coupling_function.parameters())
        self.optim = build_optimizer(trainable_params, cfg.OPTIM)

        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("prompt_learner", self.model.prompt_learner, self.optim, self.sched)

        self.scaler = GradScaler() if cfg.TRAINER.PROMPTFL.PREC == "amp" else None

        # Note that multi-gpu training could be slow because CLIP's size is
        # big, which slows down the copy operation in DataParallel
        os.environ["CUDA_VISIBLE_DEVICES"] = "0,3,2,1"
        device_count = torch.cuda.device_count()
        if device_count > 1:
            print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
            # self.model = nn.DataParallel(self.model, device_ids=[1])

    def forward_backward(self, batch):
        image, label = self.parse_batch_train(batch)

        image_features, ce_logits, _ = self.model(image, label, return_features=True)


        total_loss, general_loss, class_aware_loss = self.model.prompt_loss(general_prompt=self.model.prompt_learner.general_ctx,
            class_aware_prompts=self.model.prompt_learner.class_aware_ctx,image_features=image_features,labels=label,
            class_priors=self.get_class_priors(), x=ce_logits, cls_num_list=self.cls_num_list)

        # ce_loss = self.logit_adjust_loss(ce_logits, label)

        # loss = total_loss + ce_loss

        self.model_backward_and_update(total_loss)
        self.mask_grad(label)

        loss_summary = {
            "loss": total_loss.item(),
            # "la_loss": ce_loss.item(),
            "ge_loss": general_loss.item(),
            "ca_loss": class_aware_loss.item(),
            "acc": compute_accuracy(ce_logits, label)[0].item(),
        }


        return loss_summary




    def get_class_priors(self):
        cls_num_list = self.get_cls_num_list()
        total_samples = sum(cls_num_list)
        class_priors = torch.tensor([count / total_samples for count in cls_num_list], device=self.device)
        return class_priors

    def mask_grad(self, labels):
        unique_labels = torch.unique(labels)
        for name, param in self.model.named_parameters():
            if name == "prompt_learner.general_ctx":
                # Do nothing, allow all gradients to pass
                pass
            elif name == "prompt_learner.class_aware_ctx":
                grad_mask = torch.zeros_like(param.data)
                grad_mask[unique_labels, :, :] = 1
                param.grad.data.mul_(grad_mask)

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
                raise FileNotFoundError('Model bash main.sh caltech101 rn50_ep50 end 16 1 Falsenot found at "{}"'.format(model_path))

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



# class MABScheduler:
#     def __init__(self, num_arms, epsilon=0.1, min_value=1, max_value=10):
#         self.num_arms = num_arms
#         self.epsilon = epsilon
#         self.counts = np.zeros(num_arms)
#         self.values = np.zeros(num_arms)
#         self.min_value = min_value
#         self.max_value = max_value
#
#     def select_arm(self):
#         if np.random.random() < self.epsilon:
#             return np.random.randint(self.num_arms)
#         else:
#             return np.argmax(self.values)
#
#     def update(self, chosen_arm, reward):
#         self.counts[chosen_arm] += 1
#         n = self.counts[chosen_arm]
#         value = self.values[chosen_arm]
#         new_value = ((n - 1) / n) * value + (1 / n) * reward
#         self.values[chosen_arm] = new_value
#
#     def get_value(self, arm):
#         return self.min_value + int(arm * (self.max_value - self.min_value) / (self.num_arms - 1))
#
#     def get_arm_from_value(self, value):
#         arm = int((value - self.min_value) * (self.num_arms - 1) / (self.max_value - self.min_value))
#         return max(0, min(arm, self.num_arms - 1))

class MABScheduler:
    def __init__(self, arms, initial_learning_rate=0.1, decay_factor=0.95):
        self.arms = arms
        self.values = {arm: 0 for arm in arms}
        self.counts = {arm: 0 for arm in arms}
        self.initial_learning_rate = initial_learning_rate
        self.decay_factor = decay_factor

    def select_arm(self, epsilon=0.1):
        if random.random() < epsilon:
            return random.choice(self.arms)
        else:
            return max(self.arms, key=lambda arm: self.values[arm] / (self.counts[arm] + 1e-5) + np.sqrt(
                2 * np.log(sum(self.counts.values()) + 1) / (self.counts[arm] + 1e-5)))

    def update(self, arm, reward, epoch):
        learning_rate = self.decay_learning_rate(epoch)
        self.counts[arm] += 1
        self.values[arm] += learning_rate * (reward - self.values[arm])

    def get_value(self, arm):
        return arm

    def decay_learning_rate(self, epoch):
        return self.initial_learning_rate * (self.decay_factor ** epoch)

    @staticmethod
    def calculate_reward(accuracy, f1_score, convergence_rate):
        return 0.6 * accuracy + 0.3 * f1_score + 0.1 * convergence_rate

    @staticmethod
    def calculate_convergence_rate(acc_list, window_size=5):
        if len(acc_list) < window_size:
            return 0
        recent_acc = acc_list[-window_size:]
        return (recent_acc[-1] - recent_acc[0]) / window_size

    def get_arm_from_value(self, value):
        return value