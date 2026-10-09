"""Execution invariants for the isolated single-GPU CAPT pilot."""
import copy
import os
import unittest
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from tools.capt_parallel.engine import plan_batches,ReplayLoader,ClientPool,compare_states,state_digest
from trainers.baselines.common import trainable_state,load_trainable
from tools.benchmarks.runtime import local_train


class Samples(Dataset):
    def __init__(self,n=13,offset=0):
        self.n,self.offset=n,offset
    def __len__(self):return self.n
    def __getitem__(self,i):
        return dict(img=torch.tensor([i/13,self.offset/7,1.,.25]),label=(i+self.offset)%3,index=i)


class TinyCapt(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone=nn.Linear(4,4)
        self.backbone.requires_grad_(False)
        self.prompt_learner=nn.Module()
        self.prompt_learner.general_ctx=nn.Parameter(torch.randn(1,4))
        self.prompt_learner.class_aware_ctx=nn.Parameter(torch.randn(3,2,4))
        self.coupling_function=nn.Linear(4,4)
    def forward(self,x,labels=None,return_features=False):
        features=self.backbone(x)+self.coupling_function(self.prompt_learner.general_ctx)
        logits=features@self.prompt_learner.class_aware_ctx.mean(1).t()
        return (features,logits,None) if return_features else logits


class CaptParallelTests(unittest.TestCase):
    def test_batch_plan_matches_original_loader_across_epochs_and_preserves_rng(self):
        loader=DataLoader(Samples(),batch_size=4,shuffle=True,drop_last=False)
        for seed in (42,4200047):
            before=torch.get_rng_state().clone()
            plan=plan_batches(loader,seed,3)
            self.assertTrue(torch.equal(before,torch.get_rng_state()))
            with torch.random.fork_rng(devices=[]):
                torch.default_generator.manual_seed(seed)
                actual=[[b['index'].tolist() for b in loader] for _ in range(3)]
            self.assertEqual(plan,actual)
            replay=ReplayLoader(loader.dataset,plan)
            self.assertEqual([[b['index'].tolist() for b in replay] for _ in range(3)],actual)
            with self.assertRaises(ValueError):list(replay)

    def test_replay_reproduces_original_client_update(self):
        torch.manual_seed(3)
        model=TinyCapt();state=trainable_state(model)
        loader=DataLoader(Samples(),batch_size=4,shuffle=True)
        options=dict(lr=.01,momentum=.9,weight_decay=.0005,local_epochs=3,capt=dict(temperature=.1))
        plan=plan_batches(loader,84,3)
        with torch.random.fork_rng(devices=[]):
            torch.default_generator.manual_seed(84)
            a,ca=local_train(model,None,loader,'capt',options,torch.ones(3),'cpu')
        load_trainable(model,state)
        b,cb=local_train(model,None,ReplayLoader(loader.dataset,plan),'capt',options,torch.ones(3),'cpu')
        self.assertTrue(compare_states(a,b)['bitwise_equal'])
        self.assertEqual(ca,cb)

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA required for four concurrent streams')
    def test_four_streams_start_independently_match_serial_and_leave_server_frozen(self):
        previous=torch.are_deterministic_algorithms_enabled()
        torch.use_deterministic_algorithms(True)
        self.addCleanup(torch.use_deterministic_algorithms,previous)
        torch.manual_seed(9)
        model=TinyCapt().cuda();state=trainable_state(model)
        backbone=copy.deepcopy(model.backbone.state_dict())
        # Six clients exercise slot reuse and a final partial wave on four slots.
        clients=list(range(6))
        loaders={i:DataLoader(Samples(13+i,i),batch_size=4,shuffle=True) for i in clients}
        plans={i:plan_batches(loaders[i],42+i,2) for i in clients}
        options=dict(lr=.01,momentum=.9,weight_decay=.0005,local_epochs=2,capt=dict(temperature=.1))
        pool=ClientPool(model)
        try:
            a,ca,_=pool.run(state,clients,loaders,plans,options,torch.ones(3).cuda(),concurrency=1)
            b,cb,audit=pool.run(state,clients,loaders,plans,options,torch.ones(3).cuda(),concurrency=4)
            self.assertEqual(set(b),set(clients))
            for i in clients:
                self.assertTrue(compare_states(a[i],b[i])['close'])
                self.assertEqual(ca[i]['optimizer_steps'],cb[i]['optimizer_steps'])
            self.assertEqual({x['global_start_sha256'] for x in audit['client_audits']},{state_digest(state)})
            self.assertTrue(compare_states(trainable_state(model),state)['bitwise_equal'])
            for k,v in model.backbone.state_dict().items():self.assertTrue(torch.equal(v,backbone[k]))
        finally:pool.close()

    def test_committed_round_resume_matches_uninterrupted_training(self):
        from scripts import run_capt_single_gpu_parallel as runner
        from contextlib import ExitStack
        loaders={i:DataLoader(Samples(4,i),batch_size=4,shuffle=True) for i in range(4)}
        counts=np.ones((4,3),dtype=np.int64)
        options=dict(lr=.01,momentum=.9,weight_decay=.0005,local_epochs=1,capt=dict(temperature=.1,clusters=2))
        def build(job):
            torch.manual_seed(17)
            return TinyCapt(),loaders,None,counts,'same-images',{},[[0,1,2,3],[3,2,1,0]]
        class CpuPool:
            def __init__(self,model):self.model=model
            def close(self):pass
            def run(self,start,clients,loaders,plans,options,global_counts):
                states,costs={},{}
                for c in clients:
                    load_trainable(self.model,start)
                    states[c],costs[c]=local_train(self.model,None,ReplayLoader(loaders[c].dataset,plans[c]),
                                                  'capt',options,global_counts,'cpu')
                return states,costs,dict(concurrency=4)
        original_tensor=torch.tensor
        def cpu_tensor(*a,**kw):
            if kw.get('device')=='cuda:0':kw['device']='cpu'
            return original_tensor(*a,**kw)
        with tempfile.TemporaryDirectory() as tmp, ExitStack() as stack:
            for target,value in [('scripts.run_capt_single_gpu_parallel.build',build),
                ('tools.capt_parallel.engine.ClientPool',CpuPool),
                ('tools.benchmarks.runtime.evaluate',lambda *a:([0]*100,[100]*100)),
                ('tools.benchmarks.common.source_hashes',lambda:{}),
                ('scripts.run_capt_single_gpu_parallel.extra_hashes',lambda:{}),
                ('torch.cuda.is_available',lambda:False),('torch.cuda.synchronize',lambda *a:None),
                ('torch.cuda.get_device_name',lambda *a:'CPU resume fixture'),('torch.tensor',cpu_tensor)]:
                stack.enter_context(patch(target,value))
            def job(name):return dict(method='capt',config=options,output_root=str(Path(tmp)/name),
                                      protocol={},smoke=False,source_hashes={},execution={'source_hashes':{}})
            split,whole=job('split'),job('whole')
            runner.train(SimpleNamespace(stop_after=1,resume=False),split)
            runner.train(SimpleNamespace(stop_after=2,resume=True),split)
            runner.train(SimpleNamespace(stop_after=2,resume=False),whole)
            def saved(j):return torch.load(Path(j['output_root'])/'runs/capt/checkpoint_last.pt',weights_only=False)
            a,b=saved(split),saved(whole)
            self.assertEqual(a['round'],2)
            self.assertEqual(len(a['costs']),2)
            self.assertTrue(compare_states(a['state'],b['state'])['bitwise_equal'])


if __name__=='__main__':unittest.main()
