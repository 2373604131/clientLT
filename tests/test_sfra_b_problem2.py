"""CPU correctness, compatibility and replay checks; no CIFAR/CLIP or GPU training."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from test_sfra_b_shared_transfer import tiny_transfer, A_KEY, B_KEY
from utils.b_problem2_math import calibration_weights, feedback_loss, problem2_config, summarize_changes
from utils.cliplora_b_problem2 import Problem2DonorBTransfer
from utils.cliplora_b_transfer import calibration_batches, reconstruct_residual
from utils.cliplora_a_refresh import state_hash
from utils.b_problem2_replay import Problem2ReplayRuntime, without_donor
from scripts import run_cliplora_sfra as launcher
from scripts.run_cliplora_b_problem2_replay import build_command
from tools.sfra.b_problem2 import preflight, summarize, write_csv, write_json, read_csv, frequency_metrics
from tools.sfra.summary import METRICS, summarize as summarize_training


def configured(root, variant='E00', beta=1.):
    transfer = tiny_transfer(root, regularization=.2)
    transfer.__class__ = Problem2DonorBTransfer
    transfer.config = problem2_config(transfer.config, variant, beta)
    transfer.bank = SimpleNamespace(batch_size=64)
    transfer.monitors = {1:dict(tail_positions=[0], non_tail_positions=[1,2]),
                         2:dict(tail_positions=[0,1], non_tail_positions=[])}
    transfer.pending = dict(round=30,folder=Path(root),steps=[],feedback=[],metrics=[],receivers=[],
        loss_trace=[],class_trace=[],summary=dict(algorithm_forward_images=0,algorithm_backward_images=0,
        diagnostic_forward_images=0,client_backward_batches=0,optimizer_steps=0,feedback_synchronizations=0,
        extra_downlink_bytes=0,extra_upload_bytes=0))
    return transfer


def manifests(transfer):
    return {k:dict(batches=calibration_batches(transfer.groups[k],{0},42,30,k)) for k in transfer.recipients}


class TestProblem2Math(unittest.TestCase):
    def test_class_weights_preserve_actual_mass_with_tail_only_clients(self):
        labels = {0:[0,0,1,2,2],1:[0,1],2:[0,2,2]}
        old,mass = calibration_weights(labels,{0,1},.35,False)
        new,new_mass = calibration_weights(labels,{0,1},.35,True)
        self.assertEqual(mass,new_mass)
        self.assertAlmostEqual(mass[True],(1+2*.35)/3)
        self.assertAlmostEqual(sum(w for (k,c),w in new.items() if c==0),mass[True]/2)
        self.assertAlmostEqual(sum(w for (k,c),w in new.items() if c==1),mass[True]/2)
        self.assertNotEqual(old,new)
        losses = {k:torch.arange(1,len(y)+1,dtype=torch.float64) for k,y in labels.items()}
        weighted = sum(old[k,c]*losses[k][torch.tensor(y)==c].mean()
                       for k,y in labels.items() for c in set(y))
        direct=[]
        for k,y in labels.items():
            mask=torch.tensor([c in {0,1} for c in y])
            direct.append(.35*losses[k][mask].mean()+.65*losses[k][~mask].mean()
                          if (~mask).any() else losses[k].mean())
        torch.testing.assert_close(weighted,torch.stack(direct).mean())

    def test_positive_harm_before_averaging_and_no_baseline_gradient(self):
        values=torch.tensor([1.2,.2],requires_grad=True)
        objective,classification,harm=feedback_loss(values,torch.tensor([0,1]),7,
            {(7,0):.5,(7,1):.5},{0:1.,1:1.},1.)
        self.assertLess(float(classification.detach()),1.)
        self.assertAlmostEqual(float(harm.detach()),.1,places=6)
        objective.backward()
        torch.testing.assert_close(values.grad,torch.tensor([1.,.5]))
        baseline=torch.ones(2,requires_grad=True)
        obj,_,_=feedback_loss(baseline,torch.tensor([0,1]),7,
            {(7,0):.5,(7,1):.5},{0:1.,1:1.},1.)
        obj.backward()
        torch.testing.assert_close(baseline.grad,torch.tensor([.5,.5]))

    def test_loss_validation_and_common_reporting(self):
        for beta in (-1.,float('nan'),float('inf')):
            with self.assertRaises(ValueError): problem2_config({'mode':'shared'},'E11',beta)
        with self.assertRaises(ValueError): calibration_weights({0:[2]}, {0}, .5, True)
        rows=[dict(class_id=0,group='tail',gain=.4),dict(class_id=0,group='tail',gain=-.2),
              dict(class_id=1,group='tail',gain=.3)]
        r=summarize_changes(rows)[0]
        self.assertAlmostEqual(r['macro_gain'],.2)
        self.assertAlmostEqual(r['positive_gain']-r['harm'],r['macro_gain'])

    def test_donor_removal_keeps_original_denominator(self):
        deltas={0:{B_KEY:torch.eye(2)},1:{B_KEY:2*torch.eye(2)}}
        matrices={B_KEY:torch.stack([torch.eye(2),torch.eye(2)])}
        residual=reconstruct_residual(deltas,[0,1],matrices,[B_KEY])
        removed=without_donor(residual,deltas,matrices,[0,1],0,[B_KEY])
        torch.testing.assert_close(removed[B_KEY],torch.eye(2))
        torch.testing.assert_close(without_donor(residual,deltas,matrices,[0,1],5,[B_KEY])[B_KEY],residual[B_KEY])


class TestProblem2Optimization(unittest.TestCase):
    def test_e00_matches_legacy_joint_adam_and_keeps_base_frozen(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            old=tiny_transfer(root,regularization=.2)
            new=configured(root)
            old.pending=copy.deepcopy(new.pending)
            start=copy.deepcopy(new.model.state_dict())
            original=copy.deepcopy(new.original_deltas)
            batches=manifests(new)
            with old.session(),patch('builtins.print'):
                expected=old.calibrate_shared([0,1,2],batches,start)
            with new.session(),patch('builtins.print'):
                learned=new.calibrate_shared([0,1,2],batches,start)
            torch.testing.assert_close(learned[B_KEY],expected[B_KEY],atol=1e-7,rtol=1e-6)
            for key,value in start.items():
                self.assertTrue(torch.equal(new.model.state_dict()[key],value))
                self.assertIsNone(new.parameters[key].grad)
            for donor in original:
                self.assertTrue(torch.equal(original[donor][B_KEY],new.original_deltas[donor][B_KEY]))
            self.assertEqual(new.pending['summary']['optimizer_steps'],2)
            self.assertEqual(new.pending['summary']['algorithm_backward_images'],10)
            self.assertEqual(new.pending['summary']['algorithm_forward_images'],20)
            self.assertEqual(new.pending['summary']['diagnostic_forward_images'],20)

    def test_all_four_arms_share_first_step_when_only_harm_changes(self):
        with tempfile.TemporaryDirectory() as tmp:
            learned={}
            for variant in ('E00','E01','E10','E11'):
                folder=Path(tmp)/variant
                folder.mkdir()
                transfer=configured(folder,variant)
                batches={1:dict(batches=[dict(tail_positions=[0],non_tail_positions=[1]),
                                         dict(tail_positions=[0],non_tail_positions=[2])]),
                         2:dict(batches=[dict(tail_positions=[0],non_tail_positions=[]),
                                         dict(tail_positions=[1],non_tail_positions=[])])}
                with transfer.session(),patch('builtins.print'):
                    transfer.calibrate_shared([0,1,2],batches,copy.deepcopy(transfer.model.state_dict()))
                learned[variant]=transfer.last_matrix_steps
                self.assertNotEqual(transfer.p2_baselines[0][2][0],transfer.p2_baselines[1][2][0])
                for step in (1,2):
                    row=transfer.pending['steps'][step-1]
                    before=transfer.pending['loss_trace'][step-1]
                    self.assertAlmostEqual(row['la_before'],before[f'batch{step}_la'],places=6)
                    self.assertAlmostEqual(row['harm_before'],before[f'batch{step}_harm'],places=6)
            for a,b in [('E00','E01'),('E10','E11')]:
                torch.testing.assert_close(learned[a][1][B_KEY],learned[b][1][B_KEY],atol=1e-7,rtol=1e-6)

    def test_shared_apply_commits_once_and_records_overlap(self):
        with tempfile.TemporaryDirectory() as tmp:
            transfer=configured(tmp,'E11')
            start=copy.deepcopy(transfer.model.state_dict())
            ordinary={A_KEY:start[A_KEY],B_KEY:start[B_KEY]+.1}
            originals=copy.deepcopy(transfer.original_deltas)
            with patch('builtins.print'):
                committed=transfer.apply_shared(start,ordinary,30)
            expected=reconstruct_residual(originals,[0,1,2],transfer.last_matrices,[B_KEY])
            torch.testing.assert_close(committed[B_KEY],ordinary[B_KEY]+expected[B_KEY])
            self.assertTrue(torch.equal(committed[A_KEY],start[A_KEY]))
            self.assertTrue(torch.equal(transfer.model.state_dict()[B_KEY],start[B_KEY]))
            self.assertEqual(transfer.pending['summary']['unique_donors'],3)
            changes=read_csv(Path(tmp)/'b_transfer_rounds/r030/class_changes.csv')
            self.assertTrue(all(int(r['calibration_overlap'])>0 for r in changes))
            with patch('builtins.print'):
                transfer.observe_global(committed,30,committed=True)
            self.assertEqual(len(transfer.records),1)
            self.assertEqual(transfer.progress()['b_transfer_optimizer_steps'],2)


class TestProblem2Entrypoints(unittest.TestCase):
    def test_frequency_metrics_use_round_minus_one_and_report_missing_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            self.assertFalse(frequency_metrics(root)['frequency_metrics_available'])
            write_json(root/'class_prior.json',dict(counts=[200,50,10]))
            for epoch in range(80,100):
                write_csv(root/f'per_class_accuracy_epoch_{epoch}.csv',
                          [dict(class_id=c,per_class_acc=epoch-c) for c in range(3)])
            result=frequency_metrics(root)
            self.assertEqual(result['last20_many_acc'],89.5)
            self.assertEqual(result['last20_medium_acc'],88.5)
            self.assertEqual(result['last20_few_acc'],87.5)

    def test_event_state_loader_checks_hashes_deltas_and_committed_residual(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            transfer=configured(root)
            start=copy.deepcopy(transfer.model.state_dict())
            original=copy.deepcopy(transfer.original_deltas)
            selected=[0,1,2]
            weights=torch.tensor([.8,.15,.05])
            ordinary={A_KEY:start[A_KEY],B_KEY:start[B_KEY]+sum(original[j][B_KEY]*w for j,w in zip(selected,weights))}
            (root/'checkpoints').mkdir()
            torch.save(start,root/'checkpoints/base_model.pt')
            event=root/'events/r030_c000_main_normal_B'
            event.mkdir(parents=True)
            torch.save(dict(round=30,phase='normal_B',anchor_lora_state=start,actual_after_lora_state=ordinary,
                selected_client_ids=selected,local_factor_deltas=[original[j] for j in selected],
                client_class_counts=transfer.counts,server_weights=weights),event/'state.pt')
            write_json(event/'event.json',dict(before_lora_sha256=state_hash(start,[A_KEY,B_KEY]),
                                              after_lora_sha256=state_hash(ordinary,[A_KEY,B_KEY])))
            with patch('builtins.print'): transfer.apply_shared(start,ordinary,30)
            runtime=object.__new__(Problem2ReplayRuntime)
            runtime.job={'round':30}
            runtime.source=root
            runtime.keys=[A_KEY,B_KEY]
            runtime.a_keys=[A_KEY]
            runtime.b_keys=[B_KEY]
            runtime.source_meta=dict(initial_lora_sha256=state_hash(start,runtime.keys),
                                     frozen_model_sha256=state_hash(start,[]))
            runtime.trainer=runtime.global_trainer=SimpleNamespace(model=transfer.model)
            runtime.audit=SimpleNamespace(counts=transfer.counts)
            runtime.b_transfer=transfer
            loaded=runtime.load_event()
            torch.testing.assert_close(loaded[1][B_KEY],ordinary[B_KEY])
            runtime.source_meta['initial_lora_sha256']='tampered'
            with self.assertRaisesRegex(ValueError,'identity'): runtime.load_event()

    def test_launcher_separate_directories_and_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            def prepare(args,run):
                (run/'protocol').mkdir(parents=True)
                return run/'protocol/full_schedule.json',''
            directories=[]
            for variant in ('E00','E10','E01','E11'):
                argv=['entry.py','--output-root',tmp,'--problem2-variant',variant,'--transfer-tail-weight','.35']
                with patch('sys.argv',argv),patch.object(launcher,'prepare_protocol',prepare),\
                        patch.object(launcher.subprocess,'run') as call,patch('builtins.print'):
                    launcher.main(True,'shared','class-cyclic',.5,'E00')
                command=call.call_args.args[0]
                self.assertEqual(command[command.index('--b_problem2_variant')+1],variant)
                run=Path(command[command.index('--output-dir')+1])
                directories.append(run)
                (run/'checkpoints').mkdir()
                (run/'checkpoints/sfra_last.pt').touch()
                with patch('sys.argv',argv+['--resume']),patch.object(launcher.subprocess,'run') as call,patch('builtins.print'):
                    launcher.main(True,'shared','class-cyclic',.5,'E00')
                resumed=call.call_args.args[0]
                self.assertEqual(resumed[resumed.index('--sfra_resume')+1],str(run/'checkpoints/sfra_last.pt'))
            self.assertEqual(len(set(directories)),4)

    def test_preflight_missing_source_and_replay_command(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            report=preflight(root,[30])
            self.assertFalse(report['ready'])
            self.assertIn('events/r030_c000_main_normal_B/state.pt',report['missing'])
            write_json(root/'command.json',['python','federated_main.py','--output-dir','old',
                '--sfra_resume','old-checkpoint','DATALOADER.NUM_WORKERS','8'])
            command=build_command(root,root/'replay',root/'data',0)
            self.assertEqual(command[command.index('--sfra_resume')+1],'')
            self.assertEqual(command[command.index('--b_problem2_variant')+1],'off')
            self.assertEqual(command[command.index('--b_problem2_replay_manifest')+1],str(root/'replay/replay_job.json'))
            with self.assertRaises(ValueError): preflight(root,[31])

    def test_full_training_summary_identifies_four_methods(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            for variant in ('E00','E10','E01','E11'):
                run=root/f'seed42/client-longtail/full-cp/{variant}'
                run.mkdir(parents=True)
                cfg=configured(run,variant).config
                write_json(run/'sfra_config.json',dict(variant='full-cp',partition='client-longtail',seed=42,
                    retention_weight=10,classification_weight=1,b_transfer=cfg))
                write_json(run/'completion.json',{})
                write_csv(run/'round_metrics.csv',[dict(round=i,**{m:float(i) for m in METRICS}) for i in range(101)])
                for name in ('sfra_rounds','sfra_costs','b_transfer_rounds'):
                    write_csv(run/f'{name}.csv',[])
            with patch('builtins.print'): summarize_training(root)
            rows=read_csv(root/'analysis/performance.csv')
            self.assertEqual(len({r['method'] for r in rows}),4)
            self.assertEqual({r['problem2_variant'] for r in rows},{'E00','E10','E01','E11'})

    def test_toy_event_replay_all_arms_and_conditional_diagnostics(self):
        with tempfile.TemporaryDirectory() as tmp:
            source=Path(tmp)/'source'
            source.mkdir()
            original=configured(source)
            start=copy.deepcopy(original.model.state_dict())
            ordinary={A_KEY:start[A_KEY],B_KEY:start[B_KEY]+.1}
            deltas=copy.deepcopy(original.original_deltas)
            with patch('builtins.print'): original.apply_shared(start,ordinary,30)
            folder=source/'b_transfer_rounds/r030'
            write_csv(folder/'donor_scores.csv',[dict(receiver=1,class_id=0,donor=0,gain=.1)])
            manifest=json.loads((folder/'calibration_manifest.json').read_text())
            runtime=object.__new__(Problem2ReplayRuntime)
            runtime.source=source
            runtime.root=Path(tmp)/'suite/r030'
            runtime.root.mkdir(parents=True)
            runtime.keys=[A_KEY,B_KEY]
            runtime.b_keys=[B_KEY]
            runtime.args=SimpleNamespace(seed=42)
            runtime.sfra_config={'b_transfer':tiny_transfer(source).config}
            runtime.job=dict(round=30,variants=['E00','E10','E01','E11'],harm_beta=1.,donor_diagnostics=True)
            runtime.load_event=lambda:(start,ordinary,deltas,[0,1,2],manifest,original.last_matrices,original.last_residual)
            def factory(owner):
                transfer=configured(owner.root,owner.sfra_config['b_transfer']['problem2_variant'])
                transfer.runtime=owner
                return transfer
            with patch('utils.b_problem2_replay.Problem2DonorBTransfer',side_effect=factory),patch('builtins.print'):
                runtime.run(None)
            result=json.loads((runtime.root/'replay_completion.json').read_text())
            self.assertEqual(result['total_optimizer_steps'],8)
            self.assertTrue(result['equivalence']['pool_equal'])
            self.assertTrue(result['equivalence']['c_close'])
            self.assertEqual(len(read_csv(runtime.root/'donor_contributions.csv')),1)
            write_json(runtime.root.parent/'replay_plan.json',dict(rounds=[30],variants=runtime.job['variants']))
            status=summarize(runtime.root.parent)
            self.assertTrue(status['complete'])
            self.assertEqual(len(read_csv(runtime.root.parent/'analysis/paired_differences.csv')),8)


if __name__ == '__main__':
    unittest.main()
