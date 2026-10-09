"""Independent CAPT clients, four threads/streams, one CUDA device.

Only frozen parameters share storage. Local prompts, coupling parameters,
gradients and freshly-created optimizers belong to one client at a time.
Batch orders are planned before launching threads; workers never reseed a
process-global RNG. The supported CIFAR wrapper has deterministic transforms.
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

from trainers.baselines.common import load_trainable, trainable_state


def plan_batches(loader, seed, epochs):
    """Reproduce DataLoader base-seed + RandomSampler draws, without images.

    CAPT forward has no stochastic layers, and DatasetCifar100 ignores random
    YAML augmentations in favor of its deterministic normalization/resize.
    Thus these are the same batch indices as the serial benchmark worker.
    """
    if not isinstance(loader.sampler, RandomSampler) or loader.sampler.replacement:
        raise ValueError('Only the existing nonreplacement RandomSampler is supported')
    if loader.generator is not None or loader.sampler.generator is not None:
        raise ValueError('Unexpected private loader/sampler generator')
    if epochs < 1:
        raise ValueError('epochs must be positive')
    # CPU only: torch.manual_seed would also mutate active CUDA RNGs.
    with torch.random.fork_rng(devices=[]):
        torch.default_generator.manual_seed(seed)
        shadow = DataLoader(range(len(loader.dataset)), batch_size=loader.batch_size,
                            shuffle=True, drop_last=loader.drop_last, num_workers=0)
        return [[batch.tolist() for batch in shadow] for _ in range(epochs)]


class ReplayLoader:
    def __init__(self, dataset, plans):
        self.dataset, self.plans, self.epoch = dataset, plans, 0

    def __iter__(self):
        if self.epoch >= len(self.plans):
            raise ValueError('Training requested an unplanned local epoch')
        batches = self.plans[self.epoch]
        self.epoch += 1
        for indices in batches:
            yield default_collate([self.dataset[i] for i in indices])


def batch_digest(plans):
    return hashlib.sha256(json.dumps(plans, separators=(',', ':')).encode()).hexdigest()


def state_digest(state):
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def compare_states(left, right, atol=1e-6, rtol=1e-5):
    if left.keys() != right.keys():
        raise ValueError('State keys differ')
    max_abs, sum_diff, sum_ref = 0., 0., 0.
    close, exact = True, True
    for key in left:
        a, b = left[key].double(), right[key].double()
        d = a-b
        max_abs = max(max_abs, float(d.abs().max()))
        sum_diff += float(d.square().sum())
        sum_ref += float(a.square().sum())
        close &= torch.allclose(a,b,atol=atol,rtol=rtol)
        exact &= torch.equal(a,b)
    return dict(close=bool(close), bitwise_equal=bool(exact), max_abs=max_abs,
                relative_l2=(sum_diff/max(sum_ref,1e-30))**.5, atol=atol, rtol=rtol)


def validate_model(model):
    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            raise ValueError('Mutable batchnorm is outside this frozen CAPT executor')
        if isinstance(module, nn.modules.dropout._DropoutNd) and module.p:
            raise ValueError('Stochastic layers need explicit per-client RNG support')
        if isinstance(module, nn.MultiheadAttention) and module.dropout:
            raise ValueError('Attention dropout needs explicit per-client RNG support')
    names = {n for n,p in model.named_parameters() if p.requires_grad}
    if not names or any(not n.startswith(('prompt_learner.', 'coupling_function.')) for n in names):
        raise ValueError('This executor is restricted to CAPT prompts/coupling parameters')


class ClientPool:
    def __init__(self, model, slots=4, device='cuda:0'):
        if slots != 4:
            raise ValueError('This pilot explicitly tests four concurrent clients')
        self.device = torch.device(device)
        if self.device.type != 'cuda':
            raise ValueError('Four-stream execution requires one CUDA GPU')
        validate_model(model)
        self.server = model
        frozen = {id(p):p for p in model.parameters() if not p.requires_grad}
        self.models = [copy.deepcopy(model, memo=dict(frozen)) for _ in range(slots)]
        self.streams = [torch.cuda.Stream(device=self.device) for _ in range(slots)]
        self.executor = ThreadPoolExecutor(max_workers=slots, thread_name_prefix='capt-client')
        self.source_frozen = {n:p for n,p in model.named_parameters() if not p.requires_grad}
        for replica in self.models:
            for name, p in replica.named_parameters():
                if p.requires_grad:
                    assert p.data_ptr() != dict(model.named_parameters())[name].data_ptr()
                else:
                    assert p.data_ptr() == self.source_frozen[name].data_ptr()
        torch.cuda.synchronize(self.device)

    def close(self):
        self.executor.shutdown(wait=True)

    def run(self, start, clients, loaders, plans, options, global_counts, concurrency=4, smoke=False):
        from tools.benchmarks.runtime import local_train
        if concurrency not in (1,4) or len(set(clients)) != len(clients) or not clients:
            raise ValueError('Use one or four slots and distinct clients')
        queue = Queue()
        for c in clients:
            queue.put(c)
        lock = threading.Lock()
        local, costs, audits = {}, {}, {}
        expected_start = state_digest(start)
        torch.cuda.synchronize(self.device)
        torch.cuda.reset_peak_memory_stats(self.device)
        started = time.perf_counter()

        def worker(slot):
            with torch.cuda.device(self.device), torch.cuda.stream(self.streams[slot]):
                while True:
                    try:
                        client = queue.get_nowait()
                    except Empty:
                        return
                    model = self.models[slot]
                    load_trainable(model, start)
                    initial = state_digest(trainable_state(model))
                    if initial != expected_start:
                        raise ValueError('Client did not start from this round global state')
                    loader = ReplayLoader(loaders[client].dataset, plans[client])
                    client_started = time.perf_counter()
                    state, cost = local_train(model, None, loader, 'capt', options,
                                             global_counts, self.device, smoke=smoke)
                    self.streams[slot].synchronize()
                    audit = dict(client=client, slot=slot, global_start_sha256=initial,
                                 batch_plan_sha256=batch_digest(plans[client]),
                                 seconds=time.perf_counter()-client_started,
                                 optimizer='fresh SGD; no cross-client optimizer state')
                    with lock:
                        local[client], costs[client], audits[client] = state, cost, audit
                        print(f'CAPT slot={slot} client={client} done={len(local)}/{len(clients)} '
                              f'steps={cost["optimizer_steps"]}',flush=True)

        futures = [self.executor.submit(worker,slot) for slot in range(concurrency)]
        failures = []
        for f in futures:
            try:
                f.result()
            except BaseException as error:
                failures.append(error)
        torch.cuda.synchronize(self.device)
        if failures:
            raise failures[0]
        assert set(local) == set(clients)
        return local, costs, dict(seconds=time.perf_counter()-started,
            peak_allocated_gib=torch.cuda.max_memory_allocated(self.device)/2**30,
            peak_reserved_gib=torch.cuda.max_memory_reserved(self.device)/2**30,
            concurrency=concurrency, client_audits=[audits[c] for c in clients])
