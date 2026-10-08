# python version 3.7.1
# -*- coding: utf-8 -*-
"""Post-training global realignment eval (no local norm_init)."""

import os
import copy
import numpy as np
import random
import torch

from options import args_parser_cifar10
from util.dataset import myDataset
from model.build_model import build_model
from util.realign_eval import evaluate_global_realignment

np.set_printoptions(threshold=np.inf)
dataset_switch = "cifar10"


def _output_dir(dataset: str) -> str:
    return "./output/cifar10/" if dataset == "cifar10" else "./output/cifar100/"


def _load_checkpoints(args, load_dir: str):
    tag = getattr(args, "ckpt_tag", None) or "epoch_u100"
    model = torch.load(load_dir + args.id + "model_" + tag + ".pth", map_location="cpu").to(args.device)
    g_head = torch.load(load_dir + args.id + "g_head_" + tag + ".pth", map_location="cpu").to(args.device)
    g_aux = torch.load(load_dir + args.id + "g_aux_" + tag + ".pth", map_location="cpu").to(args.device)
    print(f"Loaded {load_dir}{args.id}model_{tag}.pth")
    return model, g_head, g_aux


if __name__ == "__main__":
    args = args_parser_cifar10()
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

    datasetObj = myDataset(args)
    if args.balanced_global:
        _, dataset_test, _, _ = datasetObj.get_balanced_dataset(datasetObj.get_args())
    else:
        _, dataset_test, _, _ = datasetObj.get_imbalanced_dataset(datasetObj.get_args())

    model = build_model(args)
    load_dir = _output_dir(args.dataset)
    model, g_head, g_aux = _load_checkpoints(args, load_dir)

    acc, shot = evaluate_global_realignment(model, g_aux, dataset_test, datasetObj, args)
    print(
        f"{args.id} realign "
        f"overall={acc:.4f} head/mid/tail={shot['head']:.4f}/{shot['middle']:.4f}/{shot['tail']:.4f}"
    )
