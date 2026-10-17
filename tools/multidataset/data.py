"""Lazy image loading; explicit transforms shared by methods and feedback."""
from pathlib import Path
from types import SimpleNamespace
import pickle
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from tools.benchmarks.common import read_json
from tools.benchmarks.longtail_data import seed_worker


class ImageDataset(Dataset):
    # Also implements the constructor expected by the unchanged WitnessBank.
    def __init__(self, cfg, data_source, transform=None, is_train=False, strong=False):
        self.cfg, self.data_source = cfg, list(data_source)
        self.is_train, self.strong = is_train, strong
        self._transforms = None

    def __len__(self):
        return len(self.data_source)

    def __getstate__(self):
        state = dict(self.__dict__)
        state['_transforms'] = None
        return state

    def transforms(self):
        if self._transforms is None:
            from torchvision import transforms as T
            if self.cfg.DATASET.NAME == 'Cifar100_LT':
                self._transforms = (T.Compose([T.ToTensor(),
                    T.Normalize((.5071, .4865, .4409), (.2673, .2564, .2762)), T.Resize((224, 224))]),)
            else:
                norm = T.Normalize((.48145466, .4578275, .40821073), (.26862954, .26130258, .27577711))
                if self.is_train:
                    base = [T.RandomResizedCrop(224, interpolation=T.InterpolationMode.BICUBIC), T.RandomHorizontalFlip()]
                else:
                    base = [T.Resize(224, interpolation=T.InterpolationMode.BICUBIC), T.CenterCrop(224)]
                self._transforms = (T.Compose(base + [T.ToTensor(), norm]),)
                if self.strong:
                    self._transforms += (T.Compose(base + [T.AutoAugment(T.AutoAugmentPolicy.IMAGENET), T.ToTensor(), norm]),)
        return self._transforms

    def __getitem__(self, index):
        from PIL import Image
        item = self.data_source[index]
        if hasattr(item, 'data'):
            image = item.data
        else:
            with Image.open(item.impath) as handle:
                image = handle.convert('RGB')
        transforms = self.transforms()
        out = dict(img=transforms[0](image), label=item.label, index=item.raw_sample_id)
        if len(transforms) == 2:
            out['img_strong'] = transforms[1](image)
        return out


def build_data(job, cfg):
    protocol = job['protocol']
    root = Path(job['protocol_root'])
    train, test = read_json(root / 'train.json'), read_json(root / 'test.json')
    arrays = {}
    def item(row):
        values = dict(label=row['class_id'], raw_sample_id=row['raw_sample_id'], domain=0)
        if 'array_index' in row:
            if row['path'] not in arrays:
                with Path(row['path']).open('rb') as stream:
                    raw = pickle.load(stream, encoding='latin1')
                arrays[row['path']] = np.asarray(raw['data'], dtype=np.uint8).reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
            values['data'] = arrays[row['path']][row['array_index']]
        else:
            values['impath'] = row['path']
        return SimpleNamespace(**values)
    clients = [[] for _ in range(protocol['clients'])]
    for row in train:
        clients[row['client_id']].append(item(row))
    test_items = [item(row) for row in test]
    def loader(dataset, training):
        return DataLoader(dataset, batch_size=32 if training else 64, shuffle=training,
            num_workers=job['workers'], drop_last=False, pin_memory=torch.cuda.is_available(), worker_init_fn=seed_worker)
    train_loaders, statistics = {}, {}
    for c, source in enumerate(clients):
        if protocol['dataset'] == 'cifar100-lt' and job['method'] in ('fedlf', 'fedyoyo', 'fedrela'):
            from tools.benchmarks.longtail_data import TrainingViews
            ids = [x.raw_sample_id for x in source]
            train_set = TrainingViews(source, ids, job['method'])
            stat_set = TrainingViews(source, ids, job['method'], statistics=True)
        else:
            train_set = ImageDataset(cfg, source, is_train=True, strong=job['method'] == 'fedyoyo')
            stat_set = ImageDataset(cfg, source, is_train=False)
        train_loaders[c], statistics[c] = loader(train_set, True), loader(stat_set, False)
    return train_loaders, statistics, loader(ImageDataset(cfg, test_items), False), clients
