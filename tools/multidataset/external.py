"""All twelve existing baseline objectives on the shared multi-dataset protocol."""
from pathlib import Path
import time
import torch

from tools.benchmarks.common import read_json, write_json, write_csv, digest, FACTOR_METHODS
from tools.multidataset.model import evaluate
from tools.multidataset.protocol import metrics
from trainers.baselines.common import trainable_state, load_trainable, weighted_average


def train(job, model, teacher, loaders, statistics, test_loader, resume=False, stop_after=0):
    from tools.benchmarks.runtime import local_train, save_checkpoint
    from trainers.baselines import capt, factor_freezing, longtail
    from tools.benchmarks.longtail_runtime import validate_state
    from utils.cliplora_a_refresh import isolated_rng
    from utils.pfrf import capture_rng_state, restore_rng_state
    from tools.multidataset.runner import verify_source
    options, method, seed = job['config'], job['method'], job['seed']
    root = Path(job['run_root'])
    checkpoint, identity = root / 'checkpoint_last.pt', digest(job)
    device = next(model.parameters()).device
    counts = torch.tensor(job['protocol']['matrix'], dtype=torch.float32)
    classes = counts.shape[1]
    state, completed, elapsed, costs = trainable_state(model), -1, 0., []
    total_rounds = 2 if job['smoke'] else 100
    active_after = 1 if job['smoke'] else options['longtail_baselines']['fedrela']['relabel_after_round']
    auxiliary = dict(schema='longtail_state_v1', method=method,
        effective_previous=None, relabeled_round=None, relabels={})
    if checkpoint.exists():
        if not resume:
            raise ValueError('Existing checkpoint requires resume')
        saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
        if saved['job_id'] != identity:
            raise ValueError('Checkpoint source/settings differ')
        state, completed, elapsed, costs = saved['state'], saved['round'], saved['elapsed'], saved['costs']
        auxiliary = saved['method_state']
        if method in ('fedlf', 'fedyoyo', 'fedrela'):
            validate_state(method, auxiliary, completed, classes, active_after, job['smoke'])
        load_trainable(model, state)
        restore_rng_state(saved['rng'])

    def commit(rnd):
        correct, total = evaluate(model, test_loader, device, classes)
        verify_source(job)
        values = metrics(correct, total, job['protocol']['groups'])
        write_json(root / 'rounds' / f'r{rnd:03d}.json', dict(round=rnd, job_id=identity,
            correct=correct, total=total, metrics=values))
        save_checkpoint(checkpoint, dict(job_id=identity, round=rnd, state=trainable_state(model),
            rng=capture_rng_state(), elapsed=elapsed, costs=costs, method_state=auxiliary))
        write_csv(root / 'costs.csv', costs)
        write_json(root / 'progress.json', dict(job_id=identity, completed_round=rnd,
            total_rounds=total_rounds, smoke=job['smoke']))
        print(f"COMMITTED {rnd}/{total_rounds}: {method} {values}", flush=True)

    if completed < 0:
        commit(0)
        completed = 0
    limit = min(total_rounds, stop_after) if stop_after else total_rounds
    for rnd in range(completed + 1, limit + 1):
        started = time.perf_counter()
        clients = job['protocol']['schedule'][rnd-1]
        if job['smoke']:
            clients = clients[:2]
        prior = None
        auxiliary_cost = {c: longtail.new_costs() for c in clients}
        method_audit = dict(round=rnd, method=method)
        if method == 'fedntd':
            teacher.load_state_dict(state, strict=False)
            teacher.eval()
        if method == 'fedyoyo':
            values = []
            for client in clients:
                load_trainable(model, state)
                with isolated_rng(seed * 100000 + 10000000 + rnd * 30 + client):
                    value, seen = longtail.estimate_effective(model, statistics[client], classes,
                        device, options['longtail_baselines']['fedyoyo'], job['smoke'])
                values.append(value)
                auxiliary_cost[client].update(auxiliary_forward_images=seen,
                    statistics_upload_bytes=value.numel()*value.element_size(),
                    statistics_download_bytes=value.numel()*value.element_size())
            effective, prior = longtail.global_effective_prior(values, auxiliary['effective_previous'], options['longtail_baselines']['fedyoyo'])
            auxiliary['effective_previous'] = effective.cpu()
            method_audit.update(effective_class_counts=effective.cpu().tolist(), estimated_prior=prior.cpu().tolist())
        if method == 'fedrela' and rnd == active_after + 1:
            spec = options['longtail_baselines']['fedrela']
            maps, audits = {}, []
            for client in clients:
                load_trainable(model, state)
                with isolated_rng(seed * 100000 + 20000000 + rnd * 30 + client):
                    probabilities, labels, ids, seen = longtail.collect_posteriors(model, statistics[client], device, spec['posterior_passes'])
                    maps[client], detail = longtail.fedrela_mapping(probabilities, labels, ids, spec['threshold_percent'])
                auxiliary_cost[client]['auxiliary_forward_images'] = seen
                audits.append(dict(client=client, **detail))
            auxiliary.update(relabels=maps, relabeled_round=rnd)
            method_audit['relabel_clients'] = audits
        local, client_costs, factor_audits = {}, [], []
        for position, client in enumerate(clients, 1):
            write_json(root / 'progress.json', dict(job_id=identity, completed_round=rnd-1, active_round=rnd,
                client_position=position, clients_this_round=len(clients), client_id=client, total_rounds=total_rounds))
            load_trainable(model, state)
            with isolated_rng(seed * 100000 + rnd * 30 + client):
                if method in FACTOR_METHODS:
                    local[client], cost, audit = factor_freezing.local_train(model, loaders[client], method,
                        options, device, rnd, smoke=job['smoke'])
                    factor_audits.append(dict(client=client, **audit))
                elif method in ('fedlf', 'fedyoyo', 'fedrela'):
                    active = method == 'fedrela' and rnd > active_after
                    if active and client not in auxiliary['relabels']:
                        raise ValueError('Missing committed client relabel map')
                    local[client], cost = longtail.local_train(model, loaders[client], method, options, counts[client],
                        device, rnd, prior=prior, relabels=auxiliary['relabels'].get(client), relabel_active=active, smoke=job['smoke'])
                    for key in cost:
                        cost[key] += auxiliary_cost[client][key]
                else:
                    local[client], cost = local_train(model, teacher, loaders[client], method, options,
                        counts.sum(0).to(device), device, smoke=job['smoke'])
            client_costs.append(dict(client=client, **cost))
            print(f'Round {rnd}/{total_rounds} client {position}/{len(clients)} id={client} steps={cost["optimizer_steps"]}', flush=True)
        if method in FACTOR_METHODS:
            state, audit = factor_freezing.aggregate(method, state, local, clients, [int(counts[c].sum()) for c in clients], rnd)
            method_audit.update(clients=factor_audits, **audit)
        elif method == 'capt':
            with isolated_rng(seed * 100000 + rnd):
                state = capt.aggregate(local, clients, counts.numpy(), state, options['capt']['clusters'])
        else:
            state = weighted_average(local, clients, [int(counts[c].sum()) for c in clients])
        load_trainable(model, state)
        seconds = time.perf_counter() - started
        payload = sum(v.numel()*v.element_size() for v in state.values()) * len(clients)
        upload = sum(c['payload_upload_bytes'] for c in client_costs) if method in FACTOR_METHODS else payload
        download = sum(c['payload_download_bytes'] for c in client_costs) if method in FACTOR_METHODS else payload
        costs.append(dict(round=rnd, train_seconds=seconds,
            upload_bytes=upload + sum(c.get('statistics_upload_bytes', 0) for c in client_costs),
            downlink_bytes=download + sum(c.get('statistics_download_bytes', 0) for c in client_costs),
            **{k: sum(c[k] for c in client_costs) for k in client_costs[0] if k != 'client'}))
        write_json(root / 'client_costs' / f'r{rnd:03d}.json', client_costs)
        write_json(root / 'method_audits' / f'r{rnd:03d}.json', method_audit)
        elapsed += seconds
        commit(rnd)
        completed = rnd
    if completed == total_rounds:
        write_json(root / 'completion.json', dict(job_id=identity, completed_round=completed, smoke=job['smoke']))
