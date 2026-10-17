"""The existing CLIP/LoRA and CAPT implementations with dataset-specific text."""
from pathlib import Path
import copy
import random
import numpy as np
import torch

from tools.benchmarks.common import read_json, write_json, digest


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def build_config(job):
    from yacs.config import CfgNode as CN
    from tools.benchmarks.runtime import parse_reference_config
    cfg = CN(parse_reference_config(job['protocol']['reference_config']))
    cfg.defrost()
    cfg.SEED = job['seed']
    cfg.OUTPUT_DIR = job['run_root']
    cfg.MODEL.INIT_WEIGHTS = ''
    cfg.DATASET.ROOT = job['protocol']['data_root']
    cfg.DATASET.NAME = 'Cifar100_LT' if job['protocol']['dataset'] == 'cifar100-lt' else job['protocol']['dataset']
    cfg.DATASET.NUM_CLASSES = job['protocol']['classes']
    cfg.DATASET.PARTITION = 'client-longtail'
    cfg.DATASET.SPECIALIZATION_LAMBDA = job['protocol']['specialization_lambda']
    cfg.DATASET.PARTITION_MANIFEST = str(Path(job['protocol_root']) / 'partition_manifest.csv')
    cfg.DATASET.SPLIT_SEED = 42
    if job['protocol']['global_imbalance'] is not None:
        cfg.DATASET.IMB_FACTOR = 1. / job['protocol']['global_imbalance']
    cfg.DATALOADER.NUM_WORKERS = job['workers']
    cfg.DATALOADER.TRAIN_X.BATCH_SIZE = 32
    cfg.DATALOADER.TEST.BATCH_SIZE = 64
    cfg.TRAINER.COOP.PREC = cfg.TRAINER.PROMPTFL.PREC = 'fp32'
    cfg.TRAINER.NAME = 'CAPT' if job['method'] == 'capt' else 'ClipLora'
    cfg.TRAINER.CLIPLORA.FREEZE_A = False
    cfg.TRAINER.CLIPLORA.COMMON_INIT_SEED = job['seed']
    if (cfg.MODEL.BACKBONE.NAME != 'ViT-B/16' or cfg.TRAINER.CLIPLORA.r != 4
        or cfg.TRAINER.CLIPLORA.alpha != 1 or cfg.TRAINER.CLIPLORA.position != 'top3'
        or cfg.TRAINER.CLIPLORA.encoder != 'vision' or list(cfg.TRAINER.CLIPLORA.params) != ['q', 'v']
        or cfg.TRAINER.CLIPLORA.dropout_rate != 0 or cfg.TRAINER.CLIPLORA.SCA_ENABLED):
        raise ValueError('Require the frozen ViT-B/16 top3 q/v rank4 alpha1 visual-LoRA model')
    if job['method'] == 'capt':
        cfg.TRAINER.PROMPTFL.N_CTX = 4
        cfg.TRAINER.PROMPTFL.n_general = 1
        cfg.TRAINER.PROMPTFL.CSC = True
        cfg.TRAINER.PROMPTFL.CTX_INIT = False
        cfg.TRAINER.PROMPTFL.CLASS_TOKEN_POSITION = 'end'
    cfg.freeze()
    return cfg


def build_model(job, cfg, device):
    import Dassl.dassl.engine  # initialize the existing trainer registry first
    from utils.cliplora_a_refresh import state_hash
    seed_all(job['seed'])
    if job['method'] == 'capt':
        from trainers.capt import CustomCLIP, load_clip_to_cpu
        model = CustomCLIP(cfg, job['protocol']['classnames'], load_clip_to_cpu(cfg).float())
        for name, parameter in model.named_parameters():
            parameter.requires_grad_(name.startswith(('prompt_learner.', 'coupling_function.')))
    else:
        from trainers.cliplora import build_cliplora_model
        model = build_cliplora_model(cfg, job['protocol']['classnames'])
        state = model.state_dict()
        keys = sorted(k for k in state if k.endswith(('_lora_A', '_lora_B')))
        identity = dict(lora=state_hash(state, keys), frozen=state_hash(state, sorted(set(state)-set(keys))))
        # Every LoRA method in the same cell/seed must start from identical tensors.
        from scripts.run_ab_validation import file_lock
        target = Path(job['protocol_root']) / 'initializations' / f"seed{job['seed']}.json"
        with file_lock(target.with_suffix('.lock')):
            if target.exists() and read_json(target) != identity:
                raise ValueError('Initial model tensors differ across methods; check environment/checkpoint')
            write_json(target, identity)
        if job['protocol']['dataset'] == 'cifar100-lt' and job['seed'] == 42:
            source = read_json(Path(job['reference_run']) / 'bridge_metadata.json')
            if identity != dict(lora=source['initial_lora_sha256'], frozen=source['frozen_model_sha256']):
                raise ValueError('CIFAR initial tensors differ from the historical frozen reference')
    model.to(device)
    teacher = copy.deepcopy(model).requires_grad_(False).eval() if job['method'] in ('fedntd', 'fedpurel') else None
    if job['method'] != 'capt':
        from utils.sfra_execution import enable_model_execution
        enable_model_execution(model)
        if teacher is not None:
            enable_model_execution(teacher)
    return model, teacher


def evaluate(model, loader, device, classes):
    from utils.cliplora_a_refresh import isolated_rng
    correct, total = torch.zeros(classes, dtype=torch.long), torch.zeros(classes, dtype=torch.long)
    modes = [(m, m.training) for m in model.modules()]
    try:
        with isolated_rng(), torch.no_grad():
            model.eval()
            for batch in loader:
                labels = batch['label'].to(device)
                logits = model(batch['img'].to(device))
                if logits.shape[1] != classes or not torch.isfinite(logits).all():
                    raise ValueError('Nonfinite logits or wrong classifier size')
                total += torch.bincount(labels.cpu(), minlength=classes)
                correct += torch.bincount(labels[logits.argmax(1) == labels].cpu(), minlength=classes)
    finally:
        for module, mode in modes:
            module.training = mode
    return correct.tolist(), total.tolist()
