"""Round-transactional execution of the three global Fed-LT adapters.

The ordinary LA arm keeps using runtime.train unchanged. All data/model/test
construction is shared with that worker; only method-specific training and
auxiliary state differ here. Test output never selects or updates any state.
"""
import os
from pathlib import Path
import platform
import random
import time

import numpy as np
import torch

from tools.benchmarks import runtime
from tools.benchmarks.common import job_id, read_json, run_path, source_hashes, write_csv, write_json
from tools.benchmarks.longtail_data import build_views
from trainers.baselines import longtail
from trainers.baselines.common import trainable_state, load_trainable, weighted_average


def validate_state(method, state, completed, class_count, active_after, smoke=False):
    if state.get('schema') != 'longtail_state_v1' or state.get('method') != method:
        raise ValueError('Missing or mismatched long-tail checkpoint state')
    if method == 'fedyoyo' and completed > 0:
        value = state.get('effective_previous')
        if not torch.is_tensor(value) or value.shape != (class_count,) or not torch.isfinite(value).all() or (value <= 0).any():
            raise ValueError('Missing/corrupt FedYoYo previous-round distribution')
    if method == 'fedrela' and completed > active_after:
        if state.get('relabeled_round') != active_after + 1 or not isinstance(state.get('relabels'), dict):
            raise ValueError('Missing FedReLa relabel checkpoint state')
        if not smoke and set(state['relabels']) != set(range(30)):
            raise ValueError('FedReLa checkpoint must include all 30 client label maps')
        for client, mapping in state['relabels'].items():
            if not isinstance(client, int) or not 0 <= client < 30 or not isinstance(mapping, dict):
                raise ValueError('Invalid FedReLa client map')
            if any(not isinstance(k, int) or k < 0 or not isinstance(v, int) or not 0 <= v < class_count
                   for k, v in mapping.items()):
                raise ValueError('Invalid FedReLa sample/label map')


def train(job, resume=False, stop_after=None, device_override=None):
    from utils.cliplora_a_refresh import isolated_rng
    from utils.pfrf import capture_rng_state, restore_rng_state
    options, method = job['config'], job['method']
    longtail.validate_options(options)
    root = run_path(job)
    checkpoint = root / 'checkpoint_last.pt'
    if root.exists() and any(root.iterdir()) and not resume:
        raise ValueError('Run directory already has files; use --resume')
    root.mkdir(parents=True, exist_ok=True)
    identity = job_id(job)
    if (root / 'run.json').exists() and read_json(root / 'run.json')['job_id'] != identity:
        raise ValueError('Run identity changed')
    meta = read_json(Path(job['reference_run']) / 'bridge_metadata.json')
    cfg = runtime.build_config(job, meta)
    random.seed(42); np.random.seed(42); torch.manual_seed(42); torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    base_loaders, test_loader, counts, train_sha = runtime.build_data(job, cfg, meta)
    train_loaders, statistics_loaders = build_views(job, cfg, base_loaders)
    device = torch.device(device_override or 'cuda:0')
    model, _ = runtime.build_model(job, cfg, meta, device)
    state = trainable_state(model)
    total_rounds = 2 if job['smoke'] else options['rounds']
    # Smoke explicitly exercises the late relabel stage within its second round.
    active_after = 1 if job['smoke'] else options['longtail_baselines']['fedrela']['relabel_after_round']
    auxiliary = dict(schema='longtail_state_v1', method=method,
        effective_previous=None, relabeled_round=None, relabels={})
    completed, elapsed, costs = -1, 0., []
    if checkpoint.exists():
        if not resume:
            raise ValueError('Explicit --resume required')
        saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
        if saved['job_id'] != identity or saved['train_images_sha256'] != train_sha:
            raise ValueError('Checkpoint source/protocol/training images differ')
        completed, state, elapsed, costs = saved['round'], saved['state'], saved['elapsed'], saved['costs']
        auxiliary = saved.get('method_state', {})
        validate_state(method, auxiliary, completed, counts.shape[1], active_after, job['smoke'])
        load_trainable(model, state)
        restore_rng_state(saved['rng'])
    elif resume:
        print('No committed round yet; rebuilding deterministic initialization.', flush=True)
    write_json(root / 'run.json', dict(job_id=identity, job=job, resolved_config=str(cfg),
        train_images_sha256=train_sha, trainable_parameters=sum(x.numel() for x in state.values()),
        trainable_keys=list(state), evaluation='one shared global model; fixed balanced CIFAR100 test',
        input_transform='CIFAR100 normalize then 224 resize; FedLF crop/flip; FedYoYo official weak/strong views',
        method_implementation='pinned official calculations adapted to shared CLIP-LoRA; see docs/longtail_benchmarks_seed42.md',
        relabel_after_round=active_after if method == 'fedrela' else None,
        rng_policy='isolated round/client seeds; auxiliary state and RNG committed together',
        communication_accounting='parameter payload plus FedYoYo class-statistic payload; initial common backbone excluded',
        environment=dict(python=platform.python_version(), torch=str(torch.__version__), cuda=torch.version.cuda,
            gpu=torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU test fixture',
            visible_devices=os.getenv('CUDA_VISIBLE_DEVICES'))))
    (root / 'resolved_config.yaml').write_text(str(cfg), encoding='utf-8')
    schedule = read_json(Path(job['reference_run']) / 'protocol/full_schedule.json')
    schedule = schedule['schedule'] if isinstance(schedule, dict) else schedule

    def commit(round_id):
        correct, total = runtime.evaluate(model, test_loader, device)
        if source_hashes() != job['source_hashes']:
            raise ValueError('Source changed during training; refusing to commit')
        metrics = runtime.metrics_from_counts(correct, total)
        write_json(root / 'rounds' / f'r{round_id:03d}.json', dict(round=round_id, job_id=identity,
            correct=correct, total=total, metrics=metrics))
        runtime.save_checkpoint(checkpoint, dict(job_id=identity, round=round_id, state=trainable_state(model),
            train_images_sha256=train_sha, rng=capture_rng_state(), elapsed=elapsed, costs=costs, method_state=auxiliary))
        write_csv(root / 'costs.csv', costs)
        write_json(root / 'progress.json', dict(method=method, completed_round=round_id,
            total_rounds=total_rounds, smoke=job['smoke'], checkpoint='checkpoint_last.pt'))
        print(f"COMMITTED {round_id}/{total_rounds}: {job['label']} {metrics}", flush=True)

    if completed < 0:
        commit(0)
        completed = 0
    limit = min(total_rounds, stop_after) if stop_after is not None else total_rounds
    for rnd in range(completed + 1, limit + 1):
        started = time.perf_counter()
        clients = schedule[rnd - 1][:2] if job['smoke'] else schedule[rnd - 1]
        prior = None
        statistics_cost = {c: longtail.new_costs() for c in clients}
        audit = dict(round=rnd, method=method)
        if method == 'fedyoyo':
            values = []
            for position, client in enumerate(clients, 1):
                write_json(root / 'progress.json', dict(method=method, completed_round=rnd - 1,
                    active_round=rnd, phase='effective_class_distribution', client_position=position,
                    clients_this_round=len(clients), client_id=client, total_rounds=total_rounds, smoke=job['smoke']))
                load_trainable(model, state)
                with isolated_rng(14200000 + rnd * 30 + client):
                    value, seen = longtail.estimate_effective(model, statistics_loaders[client], counts.shape[1],
                        device, options['longtail_baselines']['fedyoyo'], job['smoke'])
                values.append(value)
                statistics_cost[client]['auxiliary_forward_images'] = seen
                statistics_cost[client]['statistics_upload_bytes'] = value.numel() * value.element_size()
                statistics_cost[client]['statistics_download_bytes'] = value.numel() * value.element_size()
                print(f'Round {rnd} prior statistics {position}/{len(clients)} client={client}', flush=True)
            effective, prior = longtail.global_effective_prior(values, auxiliary['effective_previous'], options['longtail_baselines']['fedyoyo'])
            auxiliary['effective_previous'] = effective.cpu()
            audit.update(effective_class_counts=effective.cpu().tolist(), estimated_prior=prior.cpu().tolist(),
                         statistics_images=sum(x['auxiliary_forward_images'] for x in statistics_cost.values()))
        if method == 'fedrela' and rnd == active_after + 1:
            spec = options['longtail_baselines']['fedrela']
            maps, label_audits = {}, []
            for client in clients:
                load_trainable(model, state)  # every client uses the SAME round-start model
                with isolated_rng(24200000 + rnd * 30 + client):
                    probabilities, labels, ids, seen = longtail.collect_posteriors(
                        model, statistics_loaders[client], device, spec['posterior_passes'])
                    maps[client], detail = longtail.fedrela_mapping(probabilities, labels, ids, spec['threshold_percent'])
                statistics_cost[client]['auxiliary_forward_images'] = seen
                label_audits.append(dict(client=client, **detail,
                    changes=[dict(raw_sample_id=k, new_label=v) for k, v in sorted(maps[client].items())]))
            auxiliary.update(relabels=maps, relabeled_round=rnd)
            audit['relabel_clients'] = label_audits
        local, client_costs = {}, []
        for position, client in enumerate(clients, 1):
            load_trainable(model, state)
            write_json(root / 'progress.json', dict(method=method, completed_round=rnd - 1, active_round=rnd,
                phase='local_train', client_position=position, clients_this_round=len(clients), client_id=client,
                total_rounds=total_rounds, smoke=job['smoke']))
            relabel_active = method == 'fedrela' and rnd > active_after
            if relabel_active and client not in auxiliary['relabels']:
                raise ValueError('Missing relabel map for participating client')
            with isolated_rng(42 * 100000 + rnd * 30 + client):
                local[client], cost = longtail.local_train(model, train_loaders[client], method, options,
                    torch.tensor(counts[client], dtype=torch.float32), device, rnd, prior=prior,
                    relabels=auxiliary['relabels'].get(client), relabel_active=relabel_active, smoke=job['smoke'])
            for key in cost:
                cost[key] += statistics_cost[client][key]
            client_costs.append(dict(round=rnd, client=client, **cost))
            print(f"Round {rnd}/{total_rounds} client {position}/{len(clients)} id={client} steps={cost['optimizer_steps']}", flush=True)
        state = weighted_average(local, clients, [int(counts[c].sum()) for c in clients])
        load_trainable(model, state)
        seconds = time.perf_counter() - started
        payload = sum(x.numel() * x.element_size() for x in state.values()) * len(clients)
        costs.append(dict(round=rnd, train_seconds=seconds,
            upload_bytes=payload + sum(x['statistics_upload_bytes'] for x in client_costs),
            downlink_bytes=payload + sum(x['statistics_download_bytes'] for x in client_costs),
            **{k: sum(x[k] for x in client_costs) for k in longtail.new_costs()}))
        write_json(root / 'client_costs' / f'r{rnd:03d}.json', client_costs)
        write_json(root / 'method_audits' / f'r{rnd:03d}.json', audit)
        elapsed += seconds
        commit(rnd)
        completed = rnd
    if completed == total_rounds:
        write_json(root / 'completion.json', dict(job_id=identity, completed_round=completed, smoke=job['smoke'],
            official_test_passes=total_rounds + 1, train_seconds=elapsed,
            optimizer_steps=sum(c['optimizer_steps'] for c in costs), source_unchanged=source_hashes() == job['source_hashes']))
    else:
        print('Stopped at requested round boundary; model and method state can both be resumed.', flush=True)
