"""CAPT seed42/Client-LT: one GPU, four independent client CUDA streams.

This uses the fixed-aggregation paper baseline, NOT the native MAB launcher.
Default benchmark compares original serial updates against four concurrent
clients. Full train still aggregates all 30 clients, once per round.
"""
import argparse
import hashlib
import os
from pathlib import Path
import platform
import random
import sys
import time

# Configure cuBLAS before any CUDA work. Its default multi-stream workspace
# selection is not reproducible; tiny discrepancies compound over local steps.
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage', choices=('preflight','benchmark','train','status','summary','pack'), default='benchmark')
    p.add_argument('--seed', type=int, choices=(42,), default=42)
    p.add_argument('--parallel-clients', type=int, choices=(4,), default=4)
    p.add_argument('--reference-run', type=Path, default=Path('references/full10_clientlt'))
    p.add_argument('--data-root', type=Path, default=Path('DATA'))
    p.add_argument('--output-root', type=Path, default=Path('output/cifar100_LT/capt_single_gpu_parallel_seed42_v1'))
    p.add_argument('--num-workers', type=int, default=0, help='Evaluation loader workers; client images use the four threads')
    p.add_argument('--cpu-threads', type=int, default=2)
    p.add_argument('--benchmark-clients', type=int, default=4, choices=(4,30))
    p.add_argument('--benchmark-batches', type=int, default=0, help='0: all batches/three epochs; positive: short validation only')
    p.add_argument('--benchmark-repeats', type=int, default=2)
    p.add_argument('--benchmark-evaluate', action='store_true', help='Also evaluate the two pilot models on all 10,000 test images')
    p.add_argument('--stop-after', type=int, default=None)
    p.add_argument('--resume', action='store_true')
    return p


def extra_hashes():
    files = [Path(__file__), *sorted((REPO/'tools/capt_parallel').glob('*.py'))]
    return {str(p.relative_to(REPO)).replace('\\','/'):hashlib.sha256(p.read_text(encoding='utf-8').encode()).hexdigest() for p in files}


def make_job(args):
    from tools.benchmarks import common
    job = common.make_job('capt', args.reference_run, args.data_root, args.output_root, args.num_workers)
    job['execution'] = dict(backend='single-process-four-CUDA-streams', parallel_clients=4,
        cpu_threads=args.cpu_threads, common_global_start=True, fresh_optimizer_per_client=True,
        frozen_parameters='shared read-only storage', local_data='deterministic CIFAR transforms; planned original sample order',
        deterministic_algorithms=True, cublas_workspace_config=os.environ['CUBLAS_WORKSPACE_CONFIG'],
        source_hashes=extra_hashes())
    return job


def build(job):
    import numpy as np
    import torch
    from tools.benchmarks import runtime, common
    if os.environ.get('CUBLAS_WORKSPACE_CONFIG') not in (':4096:8', ':16:8'):
        raise ValueError('Use CUBLAS_WORKSPACE_CONFIG=:4096:8 or :16:8 for reproducible CUDA streams')
    torch.use_deterministic_algorithms(True)
    random.seed(42); np.random.seed(42); torch.manual_seed(42); torch.cuda.manual_seed_all(42)
    torch.set_num_threads(job['execution']['cpu_threads'])
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    meta=common.read_json(Path(job['reference_run'])/'bridge_metadata.json')
    cfg=runtime.build_config(job,meta)
    loaders,test,counts,train_sha=runtime.build_data(job,cfg,meta)
    # Guard the deterministic wrapper assumed by the concurrent RNG protocol.
    from Dassl.dassl.data.data_manager import DatasetCifar100
    for loader in loaders.values():
        if not isinstance(loader.dataset,DatasetCifar100) or isinstance(loader.dataset.transform,(tuple,list)):
            raise ValueError('This parallel executor requires the deterministic CIFAR100 wrapper')
    model,_=runtime.build_model(job,cfg,meta,torch.device('cuda:0'))
    schedule=common.read_json(Path(job['reference_run'])/'protocol/full_schedule.json')
    if isinstance(schedule,dict): schedule=schedule['schedule']
    return model,loaders,test,counts,train_sha,cfg,schedule


def benchmark(args,job):
    import numpy as np
    import torch
    from tools.benchmarks import common,runtime
    from tools.capt_parallel.engine import ClientPool,plan_batches,compare_states,state_digest
    from trainers.baselines.common import trainable_state,load_trainable
    from trainers.baselines import capt
    from utils.cliplora_a_refresh import isolated_rng
    root=Path(job['output_root'])/'parallel_benchmark'
    if root.exists() and any(root.iterdir()):
        raise ValueError('Benchmark output exists; use another --output-root')
    root.mkdir(parents=True)
    common.write_json(root/'job.json',job)
    model,loaders,test,counts,train_sha,cfg,schedule=build(job)
    clients=schedule[0][:args.benchmark_clients]
    state=trainable_state(model)
    global_counts=torch.tensor(counts.sum(0),dtype=torch.float32,device='cuda:0')
    options=dict(job['config'])
    plans={c:plan_batches(loaders[c],42*100000+30+c,options['local_epochs']) for c in clients}
    if args.benchmark_batches:
        options['local_epochs']=1
        plans={c:[v[0][:args.benchmark_batches]] for c,v in plans.items()}
    from tools.capt_parallel.engine import ReplayLoader
    pool=ClientPool(model)
    # Warm all worker streams/model shapes; reset weights/optimizer before measurement.
    warm_plans={c:[plans[c][0][:1]] for c in clients[:4]}
    try:
        pool.run(state,clients[:4],loaders,warm_plans,options,global_counts,smoke=True)
        timings=[]; comparisons=[]; aggregates={}
        for repeat in range(args.benchmark_repeats):
            outputs={}
            for mode in (('serial','parallel4') if repeat%2==0 else ('parallel4','serial')):
                torch.cuda.synchronize()
                # Release idle per-stream allocator caches before BOTH modes;
                # otherwise Windows WDDM can page idle caches during evaluation.
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                begin=time.perf_counter()
                if mode=='serial':
                    local={}; costs={}
                    for c in clients:
                        load_trainable(model,state)
                        assert state_digest(trainable_state(model))==state_digest(state)
                        with isolated_rng(42*100000+30+c):
                            # Full benchmark uses the UNCHANGED original DataLoader/worker.
                            loader=ReplayLoader(loaders[c].dataset,plans[c]) if args.benchmark_batches else loaders[c]
                            local[c],costs[c]=runtime.local_train(model,None,loader,'capt',options,global_counts,'cuda:0')
                        print(f'SERIAL repeat={repeat+1} client={c} steps={costs[c]["optimizer_steps"]}',flush=True)
                    torch.cuda.synchronize()
                    audit=dict(seconds=time.perf_counter()-begin,peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                               peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30)
                else:
                    # Include sample planning in timed parallel execution, not model construction.
                    planned={c:plan_batches(loaders[c],42*100000+30+c,job['config']['local_epochs']) for c in clients}
                    if args.benchmark_batches: planned={c:[v[0][:args.benchmark_batches]] for c,v in planned.items()}
                    local,costs,audit=pool.run(state,clients,loaders,planned,options,global_counts)
                    audit['seconds']=time.perf_counter()-begin
                outputs[mode]=(local,costs)
                with isolated_rng(4200001):
                    aggregates[mode]=capt.aggregate(local,clients,counts,state,options['capt']['clusters'])
                timings.append(dict(repeat=repeat+1,mode=mode,clients=len(clients),
                    optimizer_steps=sum(c['optimizer_steps'] for c in costs.values()),**audit))
                common.write_csv(root/'timings.csv',[{k:v for k,v in x.items() if k!='client_audits'} for x in timings])
                print(f'TIMING {mode} seconds={audit["seconds"]:.3f} '
                      f'peak_allocated_gib={audit["peak_allocated_gib"]:.3f}',flush=True)
            for c in clients:
                checks=compare_states(outputs['serial'][0][c],outputs['parallel4'][0][c])
                assert outputs['serial'][1][c]['optimizer_steps']==outputs['parallel4'][1][c]['optimizer_steps']
                comparisons.append(dict(repeat=repeat+1,client=c,**checks))
            comparisons.append(dict(repeat=repeat+1,client='aggregate',**compare_states(aggregates['serial'],aggregates['parallel4'])))
        evaluations={}
        if args.benchmark_evaluate:
            torch.cuda.empty_cache()
            for mode, weights in aggregates.items():
                load_trainable(model,weights)
                correct,total=runtime.evaluate(model,test,'cuda:0')
                evaluations[mode]=dict(metrics=runtime.metrics_from_counts(correct,total),correct=correct,total=total)
        passed=all(x['close'] for x in comparisons)
        serial=np.median([x['seconds'] for x in timings if x['mode']=='serial'])
        parallel=np.median([x['seconds'] for x in timings if x['mode']=='parallel4'])
        report=dict(purpose='execution validation only; subset/short runs must not enter paper tables',
            passed=passed,clients=clients,full_local_training=not bool(args.benchmark_batches),
            serial_median_seconds=float(serial),parallel4_median_seconds=float(parallel),speedup=float(serial/parallel),
            timings=timings,comparisons=comparisons,evaluations=evaluations,train_images_sha256=train_sha,
            environment=dict(torch=str(torch.__version__),cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(0),
                             platform=platform.platform(),deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
                             cublas_workspace_config=os.environ['CUBLAS_WORKSPACE_CONFIG']),
            memory_scope='PyTorch process peaks; four persistent replicas exist in BOTH timing modes; not total nvidia-smi usage')
        common.write_json(root/'report.json',report)
        common.write_csv(root/'timings.csv',[{k:v for k,v in x.items() if k!='client_audits'} for x in timings])
        common.write_csv(root/'equivalence.csv',comparisons)
        print(f'PASSED={passed} serial={serial:.2f}s parallel4={parallel:.2f}s speedup={serial/parallel:.3f}x',flush=True)
        print('Report:',root/'report.json',flush=True)
        if not passed: raise ValueError('Parallel parameter parity failed; inspect report before formal training')
    finally:
        pool.close()


def train(args,job):
    import numpy as np
    import torch
    from tools.benchmarks import common,runtime
    from tools.capt_parallel.engine import ClientPool,plan_batches
    from trainers.baselines.common import trainable_state,load_trainable
    from trainers.baselines import capt
    from utils.cliplora_a_refresh import isolated_rng
    from utils.pfrf import capture_rng_state,restore_rng_state
    common.register(job)
    root=common.run_path(job)
    checkpoint=root/'checkpoint_last.pt'
    if root.exists() and any(root.iterdir()) and not args.resume:
        raise ValueError('Training output exists; use --resume or a new output root')
    model,loaders,test,counts,train_sha,cfg,schedule=build(job)
    state=trainable_state(model);completed=0;costs=[];elapsed=0.
    identity=common.job_id(job)
    if checkpoint.exists():
        saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
        if saved['job_id']!=identity or saved['train_images_sha256']!=train_sha:
            raise ValueError('Checkpoint protocol/source/data differs')
        state,completed,costs,elapsed=saved['state'],saved['round'],saved['costs'],saved['elapsed']
        load_trainable(model,state);restore_rng_state(saved['rng'])
    root.mkdir(parents=True,exist_ok=True)
    common.write_json(root/'run.json',dict(job_id=identity,job=job,resolved_config=str(cfg),train_images_sha256=train_sha,
        trainable_parameters=sum(v.numel() for v in state.values()),trainable_keys=list(state),
        evaluation='one shared global model, all 30 clients per round, fixed balanced test',
        environment=dict(python=platform.python_version(),torch=str(torch.__version__),cuda=torch.version.cuda,
                         gpu=torch.cuda.get_device_name(0),visible_devices=os.getenv('CUDA_VISIBLE_DEVICES'))))
    (root/'resolved_config.yaml').write_text(str(cfg),encoding='utf-8')
    def commit(rnd):
        correct,total=runtime.evaluate(model,test,'cuda:0')
        if common.source_hashes()!=job['source_hashes'] or extra_hashes()!=job['execution']['source_hashes']:
            raise ValueError('Source changed during training')
        common.write_json(root/'rounds'/f'r{rnd:03d}.json',dict(round=rnd,job_id=identity,correct=correct,total=total,
                                                          metrics=runtime.metrics_from_counts(correct,total)))
        runtime.save_checkpoint(checkpoint,dict(job_id=identity,round=rnd,state=state,rng=capture_rng_state(),
                                               costs=costs,elapsed=elapsed,train_images_sha256=train_sha))
        common.write_csv(root/'costs.csv',costs)
        common.write_json(root/'progress.json',dict(method='capt',completed_round=rnd,total_rounds=100,smoke=False,parallel_clients=4))
        print(f'COMMITTED {rnd}/100: {runtime.metrics_from_counts(correct,total)}',flush=True)
    if not checkpoint.exists(): commit(0)
    pool=ClientPool(model)
    global_counts=torch.tensor(counts.sum(0),dtype=torch.float32,device='cuda:0')
    try:
        for rnd in range(completed+1,(args.stop_after or 100)+1):
            begin=time.perf_counter();clients=schedule[rnd-1]
            plans={c:plan_batches(loaders[c],42*100000+rnd*30+c,job['config']['local_epochs']) for c in clients}
            local,client_costs,audit=pool.run(state,clients,loaders,plans,job['config'],global_counts)
            with isolated_rng(4200000+rnd):
                state=capt.aggregate(local,clients,counts,state,job['config']['capt']['clusters'])
            load_trainable(model,state);torch.cuda.synchronize()
            seconds=time.perf_counter()-begin;elapsed+=seconds
            payload=sum(v.numel()*v.element_size() for v in state.values())*len(clients)
            costs.append(dict(round=rnd,train_seconds=seconds,upload_bytes=payload,downlink_bytes=payload,
                              **{k:sum(x[k] for x in client_costs.values()) for k in next(iter(client_costs.values()))}))
            common.write_json(root/'client_costs'/f'r{rnd:03d}.json',[dict(round=rnd,client=c,**client_costs[c]) for c in clients])
            common.write_json(root/'parallel_audits'/f'r{rnd:03d}.json',audit)
            commit(rnd);completed=rnd
        if completed==100:
            common.write_json(root/'completion.json',dict(job_id=identity,completed_round=100,smoke=False,official_test_passes=101,
                train_seconds=elapsed,optimizer_steps=sum(c['optimizer_steps'] for c in costs),source_unchanged=True))
    finally:
        pool.close()


def main(argv=None):
    p=parser();args=p.parse_args(argv)
    if args.cpu_threads<1 or args.num_workers<0 or args.benchmark_batches<0 or args.benchmark_repeats<1:
        p.error('Invalid worker/thread/benchmark counts')
    if args.stop_after is not None and not 1<=args.stop_after<=100: p.error('--stop-after must be in 1..100')
    args.output_root=args.output_root.resolve()
    from tools.benchmarks import common,collect
    if args.stage in ('status','summary','pack'):
        if args.stage=='status':
            for row in collect.status(args.output_root):print(row)
        else:
            collect.collect(args.output_root)
            if args.stage=='pack':print(collect.pack(args.output_root))
        return
    job=make_job(args)
    common.preflight(job,require_cuda=True)
    if args.stage=='preflight':
        print('READY: CAPT fixed aggregation, shared global start, fresh optimizers, four streams on cuda:0')
        return
    from scripts.run_ab_validation import file_lock
    with file_lock(args.output_root/'locks/capt_parallel.lock',timeout=.1):
        try:
            (benchmark if args.stage=='benchmark' else train)(args,job)
        except Exception as error:
            common.write_json(args.output_root/'parallel_failure.json',dict(type=type(error).__name__,message=str(error)))
            raise


if __name__=='__main__':
    main()
