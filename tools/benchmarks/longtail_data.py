"""Indexed training-only views on the existing, frozen CIFAR100 partition."""
import random

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from tools.benchmarks.common import manifest_rows
from trainers.baselines.upstream import definitions


def seed_worker(worker_id):
    seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(seed)
    random.seed(seed)


class TrainingViews(Dataset):
    def __init__(self, items, raw_ids, method, size=(224, 224), statistics=False):
        self.items, self.raw_ids = items, list(raw_ids)
        self.method, self.size, self.statistics = method, tuple(size), statistics
        self._transforms = None
        if len(items) != len(raw_ids) or len(set(raw_ids)) != len(raw_ids):
            raise ValueError('Training sample IDs must be unique and aligned')

    def __len__(self):
        return len(self.items)

    def __getstate__(self):
        # Upstream AutoAugment contains lambdas; rebuild it inside each worker
        # instead of pickling it (Windows spawn and Linux spawn are supported).
        state = dict(self.__dict__)
        state['_transforms'] = None
        return state

    def transforms(self):
        if self._transforms is None:
            from torchvision import transforms as T
            norm = T.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762))
            resize = T.Resize(self.size)
            base = T.Compose([T.ToTensor(), norm, resize])
            if self.statistics or self.method == 'fedrela':
                self._transforms = (base,)
            elif self.method == 'fedlf':
                self._transforms = (T.Compose([T.ToTensor(), norm, T.RandomCrop(32, padding=4),
                                               T.RandomHorizontalFlip(), resize]),)
            elif self.method == 'fedyoyo':
                upstream = definitions('fedyoyo', 'data_loader/autoaug.py', ('Cutout', 'CIFAR10Policy', 'SubPolicy'))
                weak = T.Compose([T.RandomCrop(32, padding=4), T.RandomHorizontalFlip(),
                                  T.RandomRotation(15), T.ToTensor(), norm, resize])
                strong = T.Compose([T.RandomCrop(32, padding=4), T.RandomHorizontalFlip(),
                    upstream.CIFAR10Policy(), T.ToTensor(), upstream.Cutout(n_holes=1, length=16), norm, resize])
                self._transforms = (weak, strong)
            else:
                raise ValueError('Unknown training view: ' + self.method)
        return self._transforms

    def __getitem__(self, position):
        from PIL import Image
        item = self.items[position]
        image = Image.fromarray(item.data) if self.method == 'fedyoyo' and not self.statistics else item.data
        views = self.transforms()
        out = dict(img=views[0](image), label=int(item.label), index=self.raw_ids[position])
        if len(views) == 2:
            out['img_strong'] = views[1](image)
        return out


def build_views(job, cfg, base_loaders):
    from pathlib import Path
    ids = {client: [] for client in base_loaders}
    for row in manifest_rows(Path(job['reference_run']) / 'partition_manifest.csv'):
        ids[row['client_id']].append(row['raw_sample_id'])
    train, statistics = {}, {}
    for client, loader in base_loaders.items():
        items = loader.dataset.data_source
        for target, is_statistics in ((train, False), (statistics, True)):
            dataset = TrainingViews(items, ids[client], job['method'], cfg.INPUT.SIZE, is_statistics)
            target[client] = DataLoader(dataset, batch_size=job['config']['batch_size'], shuffle=True,
                drop_last=False, num_workers=job['workers'], pin_memory=torch.cuda.is_available(),
                worker_init_fn=seed_worker, persistent_workers=False)
    return train, statistics
