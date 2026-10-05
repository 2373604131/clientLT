"""Actual external-baseline training on the frozen raw-sample protocol.

Only training batches enter optimization. Evaluation is observational. Each
checkpoint commits one complete round; interrupted clients are replayed from
the last committed server state with fresh client optimizers.
"""
import copy
import hashlib
import math
import os
from pathlib import Path
import pickle
import platform
import random
import time

import numpy as np
import torch
from torch.nn import functional as F

from tools.benchmarks.common import (FACTOR_METHODS, METRICS, REPO, digest, job_id, manifest_rows,
    read_json, run_path, source_hashes, write_csv, write_json)
from trainers.baselines.common import load_trainable, trainable_state, weighted_average


def build_config(job, meta):
    import yaml
    from yacs.config import CfgNode as CN
    cfg = CN(yaml.safe_load(meta['resolved_config']))
    cfg.defrost()
    cfg.SEED = 42
    cfg.DATASET.ROOT = job['data_root']
    cfg.MODEL.INIT_WEIGHTS = ''
    cfg.OUTPUT_DIR = str(run_path(job))
    cfg.DATALOADER.NUM_WORKERS = job['workers']
    cfg.DATALOADER.TRAIN_X.BATCH_SIZE = 32
    cfg.DATALOADER.TEST.BATCH_SIZE = 64
    cfg.TRAINER.COOP.PREC = 'fp32'
    cfg.TRAINER.PROMPTFL.PREC = 'fp32'
    cfg.TRAINER.NAME = 'CAPT' if job['method'] == 'capt' else 'ClipLora'
    cfg.TRAINER.CLIPLORA.FREEZE_A = False  # genuine joint local LoRA A/B updates
    cfg.TRAINER.CLIPLORA.COMMON_INIT_SEED = 42
    if (cfg.MODEL.BACKBONE.NAME != 'ViT-B/16' or cfg.DATASET.NAME != 'Cifar100_LT'
        or cfg.TRAINER.CLIPLORA.r != 4 or cfg.TRAINER.CLIPLORA.alpha != 1
        or cfg.TRAINER.CLIPLORA.position != 'top3' or cfg.TRAINER.CLIPLORA.encoder != 'vision'
        or list(cfg.TRAINER.CLIPLORA.params) != ['q', 'v'] or cfg.TRAINER.CLIPLORA.dropout_rate != 0
        or cfg.TRAINER.CLIPLORA.SCA_ENABLED):
        raise ValueError('Reference is not the frozen ViT-B/16 rank4 top3 q/v configuration')
    if job['method'] == 'capt':
        cfg.TRAINER.PROMPTFL.N_CTX = 4
        cfg.TRAINER.PROMPTFL.n_general = 1
        cfg.TRAINER.PROMPTFL.CSC = True
        cfg.TRAINER.PROMPTFL.CTX_INIT = False
        cfg.TRAINER.PROMPTFL.CLASS_TOKEN_POSITION = 'end'
    if job['method'] in FACTOR_METHODS:
        from trainers.baselines.factor_freezing import validate_options
        validate_options(job['config'])
    cfg.freeze()
    return cfg


def build_data(job, cfg, meta):
    from types import SimpleNamespace
    from Dassl.dassl.data.data_manager import build_data_loader
    from Dassl.dassl.data.transforms import build_transform
    from utils.functional_coverage_validation import _TrainOnlyCifar100, _locate_cifar100
    folder = _locate_cifar100(Path(job['data_root']))
    store = _TrainOnlyCifar100(folder)
    rows = manifest_rows(Path(job['reference_run']) / 'partition_manifest.csv')
    clients = [[] for _ in range(30)]
    counts = np.zeros((30, 100), dtype=np.int64)
    train_digest = hashlib.sha256()
    for row in rows:
        raw, label, client = row['raw_sample_id'], row['class_id'], row['client_id']
        if raw >= len(store.labels) or int(store.labels[raw]) != label:
            raise ValueError('Manifest labels differ from actual CIFAR100 training data')
        image = store.images[raw]
        train_digest.update(image.tobytes())
        counts[client, label] += 1
        clients[client].append(SimpleNamespace(data=image, label=label, domain=0))
    with (folder / 'test').open('rb') as stream:
        test = pickle.load(stream, encoding='latin1')
    images = np.asarray(test['data'], dtype=np.uint8).reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    labels = list(map(int, test['fine_labels']))
    if len(labels) != 10000 or not np.array_equal(np.bincount(labels, minlength=100), np.full(100, 100)):
        raise ValueError('Expected the balanced official CIFAR100 test set')
    test_items, test_digest = [], hashlib.sha256()
    for image, label in zip(images, labels):
        test_digest.update(str(label).encode())
        test_digest.update(image.tobytes())
        test_items.append(SimpleNamespace(data=image, label=label, domain=0))
    if test_digest.hexdigest() != meta['test_sha256']:
        raise ValueError('Official test content differs from A/AB reference')
    if counts.sum(1).tolist() != meta['client_sample_counts']:
        raise ValueError('Client capacities differ from reference')
    # Uses the repository's ACTUAL DatasetCifar100 transform, including its
    # fixed CIFAR normalization and 224 resize, rather than assuming YAML wins.
    train_tfm, test_tfm = build_transform(cfg, True), build_transform(cfg, False)
    def loader(items, training):
        return build_data_loader(cfg, sampler_type='RandomSampler' if training else 'SequentialSampler',
            data_source=items, batch_size=32 if training else 64, tfm=train_tfm if training else test_tfm,
            is_train=training, drop_last=False, class_names=meta['classnames'])
    loaders = {i: loader(items, True) for i, items in enumerate(clients)}
    return loaders, loader(test_items, False), counts, train_digest.hexdigest()


def build_model(job, cfg, meta, device):
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    if job['method'] == 'capt':
        from trainers.capt import CustomCLIP, load_clip_to_cpu
        model = CustomCLIP(cfg, meta['classnames'], load_clip_to_cpu(cfg).float())
        for name, parameter in model.named_parameters():
            parameter.requires_grad_(name.startswith(('prompt_learner.', 'coupling_function.')))
    else:
        from trainers.cliplora import build_cliplora_model
        from utils.cliplora_a_refresh import state_hash
        model = build_cliplora_model(cfg, meta['classnames'])
        state = model.state_dict()
        keys = sorted(k for k in state if k.endswith(('_lora_A', '_lora_B')))
        frozen = sorted(set(state) - set(keys))
        if state_hash(state, frozen) != meta['frozen_model_sha256']:
            raise ValueError('Frozen model does not match A/AB reference')
        # A source metadata may come from another seed; only claim tensor
        # identity when the reference initialization seed itself is 42.
        if meta['resolved_args'].get('cliplora_common_init_seed', meta['seed']) == 42:
            if state_hash(state, keys) != meta['initial_lora_sha256']:
                raise ValueError('Initial LoRA factors differ from reference seed42')
    model = model.to(device)
    teacher = None
    if job['method'] in ('fedntd', 'fedpurel'):
        teacher = copy.deepcopy(model)
        teacher.requires_grad_(False).eval()
    if job['method'] != 'capt':
        from utils.sfra_execution import enable_model_execution
        enable_model_execution(model)
        if teacher is not None:
            enable_model_execution(teacher)
    return model, teacher


def local_train(model, teacher, loader, method, options, global_counts, device, smoke=False):
    from trainers.baselines import capt, fedntd, fedpurel
    params = [p for p in model.parameters() if p.requires_grad]
    # No optimizer state is inherited from a previously simulated client.
    optimizer = torch.optim.SGD(params, lr=options['lr'], momentum=options['momentum'],
                                weight_decay=options['weight_decay'])
    model.train()
    costs = dict(optimizer_steps=0, student_forward_images=0, teacher_forward_images=0,
                 backward_images=0, projection_conflicts=0, loss_sum=0.)
    adjustment = (global_counts.float() / global_counts.sum()).log().to(device)
    for _ in range(1 if smoke else options['local_epochs']):
        for batch_index, batch in enumerate(loader):
            images, labels = batch['img'].to(device), batch['label'].to(device)
            optimizer.zero_grad(set_to_none=True)
            if method == 'capt':
                features, logits, _ = model(images, labels, return_features=True)
                objective = capt.loss(model.prompt_learner.general_ctx, model.prompt_learner.class_aware_ctx,
                    features, logits, labels, global_counts, options['capt']['temperature'])
            else:
                logits = model(images)
                if teacher is not None:
                    with torch.no_grad():
                        teacher_logits = teacher(images)
                    costs['teacher_forward_images'] += len(labels)
                if method == 'fedntd':
                    objective = fedntd.loss(logits, labels, teacher_logits, **options['fedntd'])
                elif method == 'fedpurel':
                    objective, kl = fedpurel.losses(logits, teacher_logits, labels)
                    if not torch.isfinite(kl):
                        raise FloatingPointError('Nonfinite FedPuReL KL')
                else:
                    objective = F.cross_entropy(logits + adjustment if method == 'fedavg-lora-la' else logits, labels)
            if not torch.isfinite(objective):
                raise FloatingPointError('Nonfinite training loss')
            if method == 'fedpurel':
                costs['projection_conflicts'] += fedpurel.backward(objective, kl, params,
                    options['fedpurel']['projection_weight'])
            else:
                objective.backward()
            if any(p.grad is not None and not bool(torch.isfinite(p.grad).all()) for p in params):
                raise FloatingPointError('Nonfinite gradient')
            optimizer.step()
            costs['optimizer_steps'] += 1
            costs['student_forward_images'] += len(labels)
            costs['backward_images'] += len(labels) * (2 if method == 'fedpurel' else 1)
            costs['loss_sum'] += float(objective.detach())
            if smoke:
                break
    return trainable_state(model), costs


def metrics_from_counts(correct, total):
    correct, total = np.asarray(correct), np.asarray(total)
    if correct.shape != (100,) or total.shape != (100,) or np.any(total <= 0) or np.any(correct < 0) or np.any(correct > total):
        raise ValueError('Invalid per-class evaluation counts')
    acc = 100. * correct / total
    return dict(overall_acc=100. * correct.sum() / total.sum(), head20_acc=float(acc[:20].mean()),
                middle60_acc=float(acc[20:80].mean()), bottom20_tail_acc=float(acc[80:].mean()),
                non_tail_acc=float(acc[:80].mean()))


def evaluate(model, loader, device):
    from utils.cliplora_a_refresh import isolated_rng
    correct = torch.zeros(100, dtype=torch.long)
    total = torch.zeros_like(correct)
    training = model.training
    try:
        with isolated_rng(), torch.no_grad():
            model.eval()
            for batch in loader:
                images, labels = batch['img'].to(device), batch['label'].to(device)
                logits = model(images)
                if not bool(torch.isfinite(logits).all()):
                    raise FloatingPointError('Nonfinite test logits')
                predicted = logits.argmax(1)
                total += torch.bincount(labels.cpu(), minlength=100)
                correct += torch.bincount(labels[predicted == labels].cpu(), minlength=100)
    finally:
        model.train(training)
    return correct.tolist(), total.tolist()


def save_checkpoint(path, payload):
    path = Path(path)
    tmp = path.with_name(path.name + '.tmp')
    torch.save(payload, tmp)
    tmp.replace(path)


def train(job, resume=False, stop_after=None, device_override=None):
    from utils.cliplora_a_refresh import isolated_rng
    from utils.pfrf import capture_rng_state, restore_rng_state
    from trainers.baselines import capt
    root, options = run_path(job), job['config']
    last = root / 'checkpoint_last.pt'
    if root.exists() and any(root.iterdir()) and not resume:
        raise ValueError('Run directory already has files; use --resume')
    root.mkdir(parents=True, exist_ok=True)
    identity = job_id(job)
    if (root / 'run.json').exists() and read_json(root / 'run.json')['job_id'] != identity:
        raise ValueError('Run identity changed')
    meta = read_json(Path(job['reference_run']) / 'bridge_metadata.json')
    cfg = build_config(job, meta)
    random.seed(42); np.random.seed(42); torch.manual_seed(42); torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    loaders, test_loader, counts, train_sha = build_data(job, cfg, meta)
    device = torch.device(device_override or 'cuda:0')
    model, teacher = build_model(job, cfg, meta, device)
    state = trainable_state(model)
    write_json(root / 'run.json', dict(job_id=identity, job=job, resolved_config=str(cfg),
        train_images_sha256=train_sha, trainable_parameters=sum(v.numel() for v in state.values()),
        trainable_keys=list(state), evaluation='one shared global model on the balanced 10000-image test',
        rng_policy='external baselines: isolated per-round/client seed; A/AB retains original RNG policy',
        input_transform='repository DatasetCifar100: CIFAR normalization then 224 resize',
        communication_accounting='estimated parameter payload, not measured network traffic; excludes initial common model distribution and transport headers',
        parameter_count_scope='all stored adapter parameters; factor baselines report active/update counts per client separately',
        environment=dict(python=platform.python_version(), torch=str(torch.__version__),
            cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU test fixture',
            visible_devices=os.getenv('CUDA_VISIBLE_DEVICES'))))
    (root / 'resolved_config.yaml').write_text(str(cfg), encoding='utf-8')
    total_rounds = 2 if job['smoke'] else options['rounds']
    completed, elapsed, costs = -1, 0., []
    if last.exists():
        if not resume:
            raise ValueError('Explicit --resume required')
        saved = torch.load(last, map_location='cpu', weights_only=False)
        if saved['job_id'] != identity or saved['train_images_sha256'] != train_sha:
            raise ValueError('Checkpoint protocol/source/data differs')
        completed, state, elapsed, costs = saved['round'], saved['state'], saved['elapsed'], saved['costs']
        load_trainable(model, state)
        restore_rng_state(saved['rng'])
    elif resume:
        print('No committed round yet; rebuilding deterministic initialization.', flush=True)
    schedule = read_json(Path(job['reference_run']) / 'protocol/full_schedule.json')
    schedule = schedule['schedule'] if isinstance(schedule, dict) else schedule
    global_counts = torch.tensor(counts.sum(0), dtype=torch.float32, device=device)

    def commit(round_id, elapsed_so_far):
        correct, total = evaluate(model, test_loader, device)
        if source_hashes() != job['source_hashes']:
            raise ValueError('Source changed during training; refusing to commit')
        write_json(root / 'rounds' / f'r{round_id:03d}.json', dict(round=round_id, job_id=identity,
            correct=correct, total=total, metrics=metrics_from_counts(correct, total)))
        save_checkpoint(last, dict(job_id=identity, round=round_id, state=trainable_state(model),
            train_images_sha256=train_sha, rng=capture_rng_state(), elapsed=elapsed_so_far, costs=costs))
        write_json(root / 'progress.json', dict(method=job['method'], completed_round=round_id,
            total_rounds=total_rounds, smoke=job['smoke'], checkpoint='checkpoint_last.pt'))
        print(f"COMMITTED {round_id}/{total_rounds}: {job['label']} {metrics_from_counts(correct, total)}", flush=True)
    if completed < 0:
        commit(0, elapsed)
        completed = 0
    limit = min(total_rounds, stop_after) if stop_after is not None else total_rounds
    for rnd in range(completed + 1, limit + 1):
        started = time.perf_counter()
        clients = schedule[rnd - 1][:2] if job['smoke'] else schedule[rnd - 1]
        if job['method'] == 'fedntd':
            teacher.load_state_dict(state, strict=False)  # refreshed once per ROUND
            teacher.eval()
        local, round_costs = {}, []
        factor_audits = []
        for position, client in enumerate(clients, 1):
            load_trainable(model, state)
            write_json(root / 'progress.json', dict(method=job['method'], completed_round=rnd - 1,
                active_round=rnd, client_position=position, clients_this_round=len(clients),
                client_id=client, total_rounds=total_rounds, smoke=job['smoke']))
            with isolated_rng(42 * 100000 + rnd * 30 + client):
                if job['method'] in FACTOR_METHODS:
                    from trainers.baselines import factor_freezing
                    local[client], cost, audit = factor_freezing.local_train(
                        model, loaders[client], job['method'], options, device, rnd, smoke=job['smoke'])
                    factor_audits.append(dict(client=client, **audit))
                else:
                    local[client], cost = local_train(model, teacher, loaders[client], job['method'], options,
                                                     global_counts, device, smoke=job['smoke'])
            round_costs.append(dict(round=rnd, client=client, **cost))
            print(f"Round {rnd}/{total_rounds} client {position}/{len(clients)} id={client} steps={cost['optimizer_steps']}", flush=True)
        if job['method'] in FACTOR_METHODS:
            state, aggregation_audit = factor_freezing.aggregate(job['method'], state, local, clients,
                [int(counts[c].sum()) for c in clients], rnd)
            write_json(root / 'factor_audits' / f'r{rnd:03d}.json',
                dict(round=rnd, clients=factor_audits, **aggregation_audit))
        elif job['method'] == 'capt':
            with isolated_rng(4200000 + rnd):
                state = capt.aggregate(local, clients, counts, state, options['capt']['clusters'])
        else:
            state = weighted_average(local, clients, [int(counts[c].sum()) for c in clients])
        load_trainable(model, state)
        seconds = time.perf_counter() - started
        size = sum(x.numel() * x.element_size() for x in state.values())
        upload = sum(x['payload_upload_bytes'] for x in round_costs) if job['method'] in FACTOR_METHODS else len(clients) * size
        download = sum(x['payload_download_bytes'] for x in round_costs) if job['method'] in FACTOR_METHODS else len(clients) * size
        costs.append(dict(round=rnd, train_seconds=seconds, upload_bytes=upload,
            downlink_bytes=download, **{k: sum(x[k] for x in round_costs) for k in round_costs[0] if k not in ('round', 'client')}))
        write_json(root / 'client_costs' / f'r{rnd:03d}.json', round_costs)
        elapsed += seconds
        commit(rnd, elapsed)
        completed = rnd
    write_csv(root / 'costs.csv', costs)
    if completed == total_rounds:
        write_json(root / 'completion.json', dict(job_id=identity, completed_round=completed,
            smoke=job['smoke'], official_test_passes=total_rounds + 1, train_seconds=elapsed,
            optimizer_steps=sum(c['optimizer_steps'] for c in costs), source_unchanged=source_hashes() == job['source_hashes']))
    else:
        print('Stopped at requested round boundary; resume uses the same frozen job.', flush=True)
