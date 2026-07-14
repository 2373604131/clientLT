from Dassl.dassl.utils import Registry, check_availability
from datasets.cifar100 import Cifar100
from datasets.cifar10 import Cifar10
from datasets.cifar10_LT import Cifar10_LT
from datasets.cifar100_LT import Cifar100_LT
from datasets.fmnist import FashionMNIST
from datasets.fmnist_LT import FashionMNIST_LT
from datasets.imagenet_LT import ImageNet_LT

DATASET_REGISTRY = Registry("DATASET")
DATASET_REGISTRY.register(Cifar100)
DATASET_REGISTRY.register(Cifar10)
DATASET_REGISTRY.register(Cifar10_LT)
DATASET_REGISTRY.register(Cifar100_LT)
DATASET_REGISTRY.register(FashionMNIST)
DATASET_REGISTRY.register(FashionMNIST_LT)
DATASET_REGISTRY.register(ImageNet_LT)

def build_dataset(cfg):
    avai_datasets = DATASET_REGISTRY.registered_names()
    check_availability(cfg.DATASET.NAME, avai_datasets)
    if cfg.VERBOSE:
        print("Loading dataset: {}".format(cfg.DATASET.NAME))
    return DATASET_REGISTRY.get(cfg.DATASET.NAME)(cfg)
