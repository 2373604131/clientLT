"""Independent LoRA clients on one GPU, with an unpadded four-slot work queue.

Plan loader RNG on the main thread, then execute deterministic local batches.
Only immutable non-LoRA parameters share storage. Both A and B are private,
including A when it happens to be frozen in the current phase.
"""
from concurrent.futures import ThreadPoolExecutor
import copy
import hashlib
import json
from queue import Empty, Queue
import threading
import time

import torch
from torch import nn
from torch.utils.data import DataLoader, RandomSampler, default_collate

from utils.cliplora_a_refresh import isolated_rng, state_hash


EXECUTION = dict(version=1, backend='threads_with_private_cuda_streams',
    max_concurrent_clients=4, cpu_intraop_threads=1, dispatch='work_queue; no padding or duplicated clients',
    synchronization='all selected clients finish before one round aggregation',
    model_storage='shared immutable backbone; private A/B, buffers, gradients, optimizer',
    batch_rng='serial-order DataLoader index replay; A uses existing isolated client seed',
    local_B_epochs=3, local_A_epochs=1, precision='fp32')


def ordered_queue(clients, slots, work):
    """No barrier assumes four arrivals: 1/2/3 remaining clients simply drain."""
    clients = list(clients)
    if slots < 1 or len(set(clients)) != len(clients):
        raise ValueError('Positive slots and distinct client identities are required')
    if not clients:
        return {}
    queue, results, lock = Queue(), {}, threading.Lock()
    for client in clients:
        queue.put(client)

    def worker(slot):
        while True:
            try:
                client = queue.get_nowait()
            except Empty:
                return
            result = work(slot, client)
            with lock:
                results[client] = result

    # Waiting on every future also finishes outstanding GPU work on failure.
    errors = []
    with ThreadPoolExecutor(max_workers=min(slots, len(clients)), thread_name_prefix='lora-client') as executor:
        futures = [executor.submit(worker, slot) for slot in range(min(slots, len(clients)))]
        for future in futures:
            try:
                future.result()
            except BaseException as error:
                errors.append(error)
    if errors:
        raise errors[0]
    if set(results) != set(clients):
        raise RuntimeError('Missing local client results; aggregation is forbidden')
    # Completion order must never affect floating-point aggregation order.
    return {client: results[client] for client in clients}


def plan_batches(loader, epochs):
    if (not isinstance(loader.sampler, RandomSampler) or loader.sampler.replacement
            or loader.generator is not None or loader.sampler.generator is not None
            or loader.sampler.num_samples != len(loader.dataset) or loader.batch_size is None
            or loader.drop_last or epochs < 1):
        raise ValueError('Parallel replay requires the original full-coverage RandomSampler loader')
    shadow = DataLoader(range(len(loader.dataset)), batch_size=loader.batch_size,
                        shuffle=True, drop_last=False, num_workers=0)
    return [[batch.tolist() for batch in shadow] for _ in range(epochs)]


def plan_phase(loaders, clients, factor, seed, rnd):
    plans = {}
    for client in clients:
        if factor == 'A':
            # No global CUDA reseeding while workers run. Planning is completed
            # before workers start; these are the same CPU draws as isolated_rng.
            with torch.random.fork_rng(devices=[]):
                torch.default_generator.manual_seed(seed + 1000003*rnd + 1009*client)
                plans[client] = plan_batches(loaders[client], 1)
        elif factor == 'B':
            plans[client] = plan_batches(loaders[client], 3)
        else:
            raise ValueError('Only separate A/B phases are supported')
    return plans


def plan_digest(plans):
    return hashlib.sha256(json.dumps(plans, separators=(',', ':')).encode()).hexdigest()


def validate_model(model, keys):
    parameters = dict(model.named_parameters())
    if (isinstance(model, nn.DataParallel) or not keys or not set(keys) <= parameters.keys()
            or any(p.requires_grad for n, p in parameters.items() if n not in keys)):
        raise ValueError('Expected one-device vision LoRA with an immutable backbone')
    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            raise ValueError('Mutable BatchNorm is unsupported in this executor')
        if isinstance(module, nn.modules.dropout._DropoutNd) and module.p:
            raise ValueError('Concurrent local training requires dropout=0')
        if isinstance(module, nn.MultiheadAttention) and module.dropout:
            raise ValueError('Concurrent attention requires dropout=0')
        if getattr(module, 'dropout_rate', 0):
            raise ValueError('Concurrent LoRA requires dropout_rate=0')


def clone_models(model, keys, slots):
    validate_model(model, keys)
    shared = {id(p): p for n, p in model.named_parameters() if n not in keys}
    replicas = [copy.deepcopy(model, memo=dict(shared)) for _ in range(slots)]
    original = dict(model.named_parameters())
    for replica in replicas:
        for name, value in replica.named_parameters():
            same_storage = value.data_ptr() == original[name].data_ptr()
            if same_storage == (name in keys):
                raise RuntimeError('Invalid shared/private parameter storage: ' + name)
    return replicas


def compare_states(left, right, atol=1e-6, rtol=1e-5):
    if left.keys() != right.keys():
        raise ValueError('Compared state keys differ')
    return dict(close=all(torch.allclose(left[k], right[k], atol=atol, rtol=rtol) for k in left),
        bitwise_equal=all(torch.equal(left[k], right[k]) for k in left),
        max_abs=max(float((left[k].double()-right[k].double()).abs().max()) for k in left),
        atol=atol, rtol=rtol)


def train_local(model, state, keys, dataset, plan, factor, device, optimizer_factory, step, adjustment):
    """Canonical optimizer step and scheduler, independent of a mutable Trainer."""
    parameters = dict(model.named_parameters())
    active = [k for k in keys if k.endswith('_lora_' + factor)]
    with torch.no_grad():
        for key in keys:
            parameters[key].copy_(state[key])
            parameters[key].requires_grad_(key in active)
            parameters[key].grad = None
    model.train()
    optimizer, scheduler = optimizer_factory(model, factor)
    steps = samples = 0
    for epoch in plan:
        for indices in epoch:
            batch = default_collate([dataset[i] for i in indices])
            images, labels = batch['img'].to(device), batch['label'].to(device)
            step(model, optimizer, None, 'fp32', images, labels,
                 loss_weight=batch.get('loss_weight'), logit_adjustment=adjustment)
            steps += 1
            samples += len(labels)
        if scheduler is not None:
            scheduler.step()
    frozen = [k for k in keys if k not in active]
    values = {k: parameters[k].detach().cpu().clone() for k in keys}
    if any(not torch.equal(values[k], state[k].cpu()) for k in frozen):
        raise RuntimeError('Local training changed the frozen factor')
    if any(not torch.isfinite(v).all() for v in values.values()):
        raise FloatingPointError('Nonfinite local LoRA result')
    # Release per-client autograd storage before a slot takes its next client.
    for key in keys:
        parameters[key].grad = None
    return {k: values[k] for k in active}, dict(optimizer_steps=steps,
        sample_presentations=samples, scheduler_steps=len(plan) if scheduler is not None else 0)


class LoRAClientPool:
    def __init__(self, runtime):
        self.runtime = runtime
        self.device = torch.device(runtime.trainer.device)
        if self.device.type != 'cuda' or torch.cuda.device_count() != 1:
            raise ValueError('Four-client execution needs exactly one visible CUDA GPU per experiment')
        self.keys = runtime.keys
        self.loaders = runtime.trainer.fed_train_loader_x_dict
        for loader in self.loaders.values():
            dataset = loader.dataset
            if (type(dataset).__name__ != 'DatasetCifar100' or dataset.cfg.DATASET.NAME != 'Cifar100_LT'
                    or dataset.transform is None or isinstance(dataset.transform, (tuple, list))
                    or dataset.k_tfm != 1):
                raise ValueError('Parallel replay is restricted to the deterministic CIFAR-100 wrapper')
        # Imports occur after federated_main registered the training API.
        from trainers.cliplora import build_cliplora_optimizer_and_scheduler, cliplora_optimizer_step
        self.step = cliplora_optimizer_step

        def optimizer_factory(model, factor):
            if factor == 'B':
                return build_cliplora_optimizer_and_scheduler(model, runtime.cfg)
            return torch.optim.SGD([p for p in model.parameters() if p.requires_grad],
                lr=runtime.config['extra_a_lr'], momentum=.9, weight_decay=0.), None

        self.optimizer_factory = optimizer_factory
        self.models = clone_models(runtime.trainer.model, self.keys, 4)
        from utils.sfra_execution import enable_model_execution
        # Build each private cache on the main stream before threaded use.
        for model in self.models:
            enable_model_execution(model)
            model._sfra_text_cache.get()
        self.streams = [torch.cuda.Stream(device=self.device) for _ in range(4)]
        torch.cuda.synchronize(self.device)

    def run(self, state, clients, plans, factor, slots=4):
        if slots not in (1, 4):
            raise ValueError('This execution attempt compares one and four slots')
        torch.cuda.synchronize(self.device)
        allocated_before = torch.cuda.memory_allocated(self.device)
        started = time.perf_counter()
        cpu_rng = torch.get_rng_state().clone()

        def work(slot, client):
            with torch.cuda.device(self.device), torch.cuda.stream(self.streams[slot]):
                try:
                    values, costs = train_local(self.models[slot], state, self.keys,
                        self.loaders[client].dataset, plans[client], factor, self.device,
                        self.optimizer_factory, self.step, self.runtime.trainer.training_logit_adjustment)
                    return values, costs, dict(client_id=client, slot=slot,
                        batch_plan_sha256=plan_digest(plans[client]))
                finally:
                    self.streams[slot].synchronize()

        results = ordered_queue(clients, slots, work)
        torch.cuda.synchronize(self.device)
        if not torch.equal(cpu_rng, torch.get_rng_state()):
            raise RuntimeError('A concurrent worker consumed process-global RNG')
        return results, dict(seconds=time.perf_counter()-started, slots=slots, clients=len(clients),
            allocated_before_bytes=allocated_before,
            process_peak_allocated_bytes=torch.cuda.max_memory_allocated(self.device),
            process_peak_reserved_bytes=torch.cuda.max_memory_reserved(self.device))

    def benchmark(self, state, clients, plans, factor):
        """Uncommitted six-client pilot, including a final queue of <4 clients."""
        chosen = list(clients[:6])
        with isolated_rng():
            # A short warm-up for all four streams, excluded from both timings.
            warm = {c: [[plans[c][0][0]]] for c in chosen[:4]}
            self.run(state, chosen[:4], warm, factor)
            before = state_hash(self.runtime.trainer.model.state_dict(), self.keys)
            serial, serial_cost = self.run(state, chosen, plans, factor, slots=1)
            parallel, parallel_cost = self.run(state, chosen, plans, factor, slots=4)
            comparisons = [dict(client_id=c, **compare_states(serial[c][0], parallel[c][0])) for c in chosen]
            # A two-client submission explicitly exercises the short final group.
            short, short_cost = self.run(state, chosen[-2:], plans, factor, slots=4)
            short_comparisons = [dict(client_id=c, **compare_states(serial[c][0], short[c][0])) for c in chosen[-2:]]
            if state_hash(self.runtime.trainer.model.state_dict(), self.keys) != before:
                raise RuntimeError('Benchmark modified the server LoRA')
            record = dict(factor=factor, clients=chosen, serial=serial_cost, parallel=parallel_cost,
                short_group=short_cost, short_comparisons=short_comparisons,
                speedup=serial_cost['seconds']/max(parallel_cost['seconds'], 1e-12), comparisons=comparisons,
                passed=(all(r['close'] and serial[r['client_id']][1] == parallel[r['client_id']][1] for r in comparisons)
                        and all(r['close'] and serial[r['client_id']][1] == short[r['client_id']][1] for r in short_comparisons)),
                scope='one uncommitted pilot, same local batches/updates; excludes planning and server feedback; not fourfold speed claim')
            from scripts.run_ab_validation import write_json
            write_json(self.runtime.root/'parallel_benchmark'/('factor_'+factor+'.json'), record)
            if not record['passed']:
                raise ValueError('Serial/four-client numerical comparison failed; see parallel_benchmark')
            print(f'PARALLEL PILOT factor={factor} clients={len(chosen)} '
                  f'speedup={record["speedup"]:.3f}x max_abs={max(r["max_abs"] for r in comparisons):.3g}', flush=True)
            return record


def parallel_phase(runtime, state, rnd, factor, extra, branch, candidate):
    """The same ordinary aggregation/bridge contract, after all client tasks."""
    from scripts.run_ab_validation import write_json
    from utils.cliplora_a_refresh import aggregate_refresh_deltas, train_only
    from utils.lora_aggregation import aggregate_lora_state
    phase = 'refresh_A' if extra else 'normal_B'
    if branch != 'main' or (factor, extra) not in (('B', False), ('A', True)):
        raise ValueError('Unsupported parallel phase')
    selected = list(map(int, runtime.schedule[rnd-1]))
    if sorted(selected) != list(range(30)):
        raise ValueError('This experiment still requires all 30 distinct clients per round')
    event_id = f'r{rnd:03d}_c{candidate:03d}_{branch}_{phase}'
    started = time.perf_counter()
    pool = runtime.client_pool
    plans = plan_phase(pool.loaders, selected, factor, runtime.args.seed, rnd)
    if runtime.job['mode'] == 'smoke' and rnd == 1:
        pool.benchmark(state, selected, plans, factor)
    results, execution = pool.run(state, selected, plans, factor)
    keys = runtime.a_keys if factor == 'A' else runtime.b_keys
    local = {c: results[c][0] for c in selected}
    deltas = {c: {k: local[c][k]-state[k] for k in keys} for c in selected}
    execution.update(round=rnd, factor=factor, selected_client_ids=selected,
                     client_audits=[results[c][2] for c in selected])
    write_json(runtime.root/'parallel_execution'/f'r{rnd:03d}_{factor}.json', execution)
    for c in selected:
        cost = results[c][1]
        runtime.budget.append(dict(event_id=event_id, round=rnd, candidate_round=candidate,
            branch=branch, phase=phase, client_id=c, **cost, committed=True, trainable_factor=factor,
            a_optimizer_steps=cost['optimizer_steps'] if extra else 0,
            b_optimizer_steps=0 if extra else cost['optimizer_steps'],
            upload_bytes=sum(v.numel()*v.element_size() for v in local[c].values()),
            modeled_downlink_bytes=sum(state[k].numel()*state[k].element_size() for k in runtime.keys)))
    weights = runtime.phase_aggregation_weights(selected, rnd, factor, extra, branch)
    if not extra:
        local, deltas = runtime.prepare_b_aggregation(state, local, deltas, selected, rnd)
    after = (aggregate_refresh_deltas(state, deltas, weights, keys) if extra else
             aggregate_lora_state(state, local, selected, keys, weights))
    runtime.audit.event_context = dict(event_id=event_id, branch=branch, candidate_round=candidate)
    if extra:
        runtime.audit.save(state, after, deltas, selected, weights, rnd, phase)
    else:
        runtime.audit.normal(state, after, local, selected, weights, rnd, factor=factor)
    from tools.client_aggregation.protocol import load_json
    info = load_json(runtime.root/'events'/event_id/'event.json')
    event = dict(info, committed=True, decision_round=rnd, seconds=time.perf_counter()-started,
                 state_path=f'events/{event_id}/state.pt')
    runtime.events.append(event)
    runtime.load_training_state(after)
    train_only(runtime.trainer.model, 'B')
    # Preserve the original shared-C wrapper and its single residual injection.
    if not extra and runtime.b_transfer is not None and runtime.b_transfer.scheduled(rnd):
        after = runtime.b_transfer.apply_shared(state, after, rnd)
        event.update(state_role='ordinary_B_before_shared_transfer',
                     shared_transfer_state_path=f'b_transfer_rounds/r{rnd:03d}/commit.pt')
        write_json(runtime.root/'events'/event_id/'event.json', event)
    runtime.load_training_state(after)
    train_only(runtime.trainer.model, 'B')
    print(f'PARALLEL phase complete: {event_id}; clients={len(selected)}; slots=4', flush=True)
    return after, deltas
