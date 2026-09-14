"""Training-side feedback. This module never reads official test or offline probes."""
import csv
from collections import Counter
from contextlib import contextmanager
from pathlib import Path
import time

import numpy as np
import torch

from utils.cliplora_a_refresh import isolated_rng


def snapshot(model):
    """Complete recoverable global state, including every parameter and buffer."""
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


@contextmanager
def observational_model(model):
    state = snapshot(model)
    modes = [(module, module.training) for module in model.modules()]
    with isolated_rng():
        try:
            model.eval()
            with torch.no_grad():
                yield
        finally:
            model.load_state_dict(state, strict=True)
            for module, mode in modes:
                module.training = mode


def restore_partition_manifest(path, pool, cfg):
    from utils.functional_coverage_validation import _TrainOnlyCifar100, _exact_lt_raw_ids, _locate_cifar100
    store = _TrainOnlyCifar100(_locate_cifar100(Path(cfg.DATASET.ROOT)))
    ids = _exact_lt_raw_ids(store.labels, cfg.DATASET.IMB_FACTOR, cfg.DATASET.IMB_TYPE).tolist()
    assert len(pool) == len(ids)
    for raw_id, item in zip(ids, pool):
        assert int(store.labels[raw_id]) == int(item.label)
        assert np.array_equal(store.images[raw_id], np.asarray(item.data))
    positions = {raw_id: i for i, raw_id in enumerate(ids)}
    with open(path, encoding='utf-8-sig', newline='') as stream:
        rows = list(csv.DictReader(stream))
    assert sorted(int(r['raw_sample_id']) for r in rows) == sorted(ids)
    mapping, counts = {}, {}
    for client in range(cfg.DATASET.USERS):
        local = sorted((r for r in rows if int(r['client_id']) == client), key=lambda r: int(r['local_position']))
        assert [int(r['local_position']) for r in local] == list(range(len(local)))
        mapping[client] = [positions[int(r['raw_sample_id'])] for r in local]
        assert all(int(r['class_id']) == int(pool[i].label) for r, i in zip(local, mapping[client]))
        counts[client] = dict(Counter(int(pool[i].label) for i in mapping[client]))
    return mapping, counts


class TrainingSideFeedback:
    def __init__(self, trainer, cfg):
        from Dassl.dassl.data.data_manager import build_data_loader
        from Dassl.dassl.data.transforms import build_transform
        self.trainer = trainer
        self.classes = len(trainer.dm.dataset.classnames)
        self.loaders = {}
        with isolated_rng():
            transform = build_transform(cfg, is_train=False)
            for client, source in enumerate(trainer.dm.dataset.federated_train_x):
                self.loaders[client] = build_data_loader(
                    cfg, sampler_type='SequentialSampler', data_source=list(source),
                    batch_size=cfg.DATALOADER.TEST.BATCH_SIZE, tfm=transform,
                    is_train=False, class_names=trainer.dm.dataset.classnames, drop_last=False,
                )

    def observe(self, state):
        started = time.perf_counter()
        sums = torch.zeros(self.classes, dtype=torch.float64)
        counts = torch.zeros(self.classes, dtype=torch.long)
        model = self.trainer.model
        uploads = []
        with observational_model(model):
            model.load_state_dict(state, strict=True)
            core = model.module if hasattr(model, 'module') else model
            scale = core.logit_scale.exp()
            for client, loader in self.loaders.items():
                local_sum = torch.zeros_like(sums)
                local_count = torch.zeros_like(counts)
                for batch in loader:
                    images, labels = self.trainer.parse_batch_test(batch)
                    scores = model(images) / scale
                    correct = scores.gather(1, labels[:, None]).squeeze(1)
                    competitors = scores.clone()
                    competitors.scatter_(1, labels[:, None], -torch.inf)
                    margins = (correct - competitors.max(1).values).double().cpu()
                    y = labels.cpu()
                    local_sum.scatter_add_(0, y, margins)
                    local_count.scatter_add_(0, y, torch.ones_like(y))
                # Simulated class-statistics upload, NOT secure aggregation or q weighting.
                sums += local_sum
                counts += local_count
                uploads.append({'client_id': client, 'margin_sum': local_sum.tolist(), 'count': local_count.tolist()})
        return (sums / counts).numpy(), uploads, time.perf_counter() - started
