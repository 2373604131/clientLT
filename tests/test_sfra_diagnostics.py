"""Meaningful CPU checks for isolated interventions, norm matching and replay."""
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

import test_sfra_v1 as fixtures
from tools.sfra.diagnostics import preflight, read_json, read_csv, write_json, write_csv
from utils.sfra_diagnostics import (MethodADiagnosticRuntime, norm_matched_candidate,
                                    effective_norm, recover_history)
from utils.sfra_math import FunctionalHistory
from utils.sfra_resident_feedback import execution_config_v2, ResidentWitnessFeedback
from scripts.run_method_a_diagnostics import build_command, train, summarize


class TestNormMatching(unittest.TestCase):
    def test_effective_norm_not_parameter_norm_and_enlargement(self):
        before = {'x_lora_A':torch.zeros(2, 2), 'x_lora_B':torch.diag(torch.tensor([1., 10.])),
                  'unchanged':torch.ones(3)}
        ordinary = {**before, 'x_lora_A':torch.tensor([[1., 0.], [0., 1.]])}
        target = 2*effective_norm(before, ordinary, ['x_lora_A'], .5)
        result, info = norm_matched_candidate(before, ordinary, target, ['x_lora_A'], .5)
        self.assertAlmostEqual(info['alpha'], 2.)
        self.assertEqual(info['update_kind'], 'enlarged')
        torch.testing.assert_close(result['x_lora_A'], ordinary['x_lora_A']*2)
        self.assertTrue(torch.equal(result['x_lora_B'], before['x_lora_B']))
        self.assertTrue(torch.equal(result['unchanged'], before['unchanged']))
        self.assertAlmostEqual(effective_norm(before, result, ['x_lora_A'], .5), target)

    def test_zero_and_nonfinite_fail_closed(self):
        before = {'x_lora_A':torch.zeros(2, 2), 'x_lora_B':torch.eye(2)}
        with self.assertRaisesRegex(ValueError, 'cannot match'):
            norm_matched_candidate(before, before, 1., ['x_lora_A'], .5)
        with self.assertRaisesRegex(ValueError, 'Invalid'):
            norm_matched_candidate(before, before, float('nan'), ['x_lora_A'], .5)

    def test_all_layers_use_one_scalar(self):
        base = {f'{i}_lora_{k}':torch.randn(2,2) for i in range(2) for k in ('A','B')}
        keys = ['0_lora_A','1_lora_A']
        ordinary = {**base, **{k:base[k]+torch.randn(2,2)*.1 for k in keys}}
        target = .37*effective_norm(base, ordinary, keys, .5)
        result, info = norm_matched_candidate(base, ordinary, target, keys, .5)
        for k in keys:
            torch.testing.assert_close(result[k]-base[k], .37*(ordinary[k]-base[k]), atol=1e-7, rtol=1e-5)
        self.assertAlmostEqual(info['alpha'], .37)


class TestHistoryReplay(unittest.TestCase):
    def test_preserves_history_and_source_cache_and_rejects_corruption(self):
        initial = torch.full((2,2), -.1)
        history = FunctionalHistory(initial, 2)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for rnd in range(1, 11):
                scores = torch.full((2,2), -.05 if rnd<=5 else .01)
                history.commit(scores, rnd)
                history.source_cache[:] = torch.tensor([.25, .75])
                path = root/f'sfra_rounds/r{rnd:03d}/tokens.npz'
                path.parent.mkdir(parents=True)
                np.savez(path, F_committed=scores.numpy(), history_after=history.level.numpy(),
                         history_valid_after=history.valid.numpy(), source_cache=history.source_cache.numpy())
            restored = recover_history(initial, root, 10, 2)
            self.assertEqual(restored.block_count, 0)
            for key in ('reference','level','valid','source_cache','block_min'):
                torch.testing.assert_close(getattr(restored,key), getattr(history,key))
            with self.assertRaises(ValueError):
                recover_history(torch.ones_like(initial), root, 10, 2)


def tiny_runtime(root, branch='full-cp', full_root=None, resume=False):
    bank = fixtures.TestSFRAClassification().bank()
    runtime = object.__new__(MethodADiagnosticRuntime)
    runtime.trainer = SimpleNamespace(model=bank.model, device=bank.device)
    runtime.bank, runtime.a_keys = bank, bank.keys
    runtime.b_keys = ['image_encoder.proj_lora_B']
    runtime.keys = sorted(runtime.a_keys+runtime.b_keys)
    runtime.variant, runtime.strength, runtime.classification_strength = 'full-cp', 10., 1.
    runtime.scaling = .5
    runtime.root = Path(root)
    runtime.root.mkdir(parents=True, exist_ok=True)
    runtime.base_state = copy.deepcopy(bank.model.state_dict())
    runtime.anchor, runtime.branch_name = 60, branch
    runtime.sizes, runtime.q, runtime.args = [5,3], {0:2/3,1:1/3}, SimpleNamespace(seed=42)
    runtime.job = dict(anchor=60, branch=branch, full_branch=str(full_root or root), resume=resume)
    for key in ('metrics','witness_metrics','norms','evaluation_rows','costs','budget','events'):
        setattr(runtime,key,[])
    initial_scores = bank.evaluate()['scores']
    def load_anchor():
        runtime.history = FunctionalHistory(initial_scores, 2)
        runtime.history.valid[:] = True
        runtime.history.level[:] = .4
        return copy.deepcopy(runtime.base_state)
    runtime.load_anchor = load_anchor
    def train_phase(state, rnd, factor, extra=False, return_deltas=False):
        after = dict(state)
        if factor=='B':
            for k in runtime.b_keys:
                after[k] = state[k] + torch.randn_like(state[k])*.001
            return after
        deltas = {i:{k:torch.randn_like(state[k])*.005 for k in runtime.a_keys} for i in range(2)}
        for k in runtime.a_keys:
            after[k] = state[k] + sum(runtime.q[i]*deltas[i][k] for i in range(2))
        runtime.events.append(dict(round=rnd))
        return (after,deltas) if return_deltas else after
    runtime.train_phase = train_phase
    def evaluate(state,rnd,candidate,kind):
        runtime.load_training_state(state)
        observed = bank.evaluate(classification=True, classification_gradient=False)
        runtime.metrics.append(dict(round=rnd,candidate=candidate,kind=kind,group='Overall',
                                    accuracy=float(observed['scores'].mean())))
        return observed['scores']
    runtime.evaluate_candidate = evaluate
    return runtime


class TestPairedBranches(unittest.TestCase):
    def test_actual_gradient_branches_share_start_match_norm_and_keep_mu0_isolated(self):
        with tempfile.TemporaryDirectory() as temp, patch('builtins.print'):
            root = Path(temp)
            full = tiny_runtime(root/'full-cp')
            full.run(None)
            original_history = copy.deepcopy(full.history.state_dict())
            ordinary = tiny_runtime(root/'ordinary', 'ordinary', root/'full-cp')
            ordinary.run(None)
            matched = tiny_runtime(root/'norm-matched','norm-matched',root/'full-cp')
            matched.run(None)
            self.assertEqual(full.classification_strength,1.)
            self.assertEqual(full.history.block_count,0)
            self.assertEqual(len(full.norms),5)
            self.assertEqual(len(list((root/'full-cp/no_cp_probe').glob('r*'))),1)
            self.assertEqual(sum(r['candidate']=='no-cp' for r in full.metrics),1)
            for a,b in zip(full.norms,matched.norms):
                self.assertAlmostEqual(a['actual_effective_norm'],b['actual_effective_norm'],delta=1e-8)
            self.assertEqual(full.norms[0]['ordinary_lora_sha256'],ordinary.norms[0]['ordinary_lora_sha256'])
            for key,value in original_history.items():
                if torch.is_tensor(value):
                    torch.testing.assert_close(full.history.state_dict()[key],value)
            self.assertTrue(read_json(root/'full-cp/diagnostic_completion.json')['completed_round']==65)

    def test_interrupted_branch_resumes_with_same_result(self):
        with tempfile.TemporaryDirectory() as temp, patch('builtins.print'):
            root = Path(temp)
            expected = tiny_runtime(root/'complete')
            expected.run(None)
            interrupted = tiny_runtime(root/'interrupted')
            actual_train = interrupted.train_phase
            def fail(state,rnd,*args,**kwargs):
                if rnd==64:
                    raise RuntimeError('simulated interruption')
                return actual_train(state,rnd,*args,**kwargs)
            interrupted.train_phase = fail
            with self.assertRaisesRegex(RuntimeError,'interruption'):
                interrupted.run(None)
            resumed = tiny_runtime(root/'interrupted',resume=True)
            resumed.run(None)
            a = torch.load(root/'complete/diagnostic_last.pt',weights_only=False)
            b = torch.load(root/'interrupted/diagnostic_last.pt',weights_only=False)
            for key in a['state_dict_overrides']:
                torch.testing.assert_close(a['state_dict_overrides'][key],b['state_dict_overrides'][key],atol=0,rtol=0)
            self.assertEqual(len(resumed.norms),5)
            self.assertEqual(sum(r['candidate']=='no-cp' for r in resumed.metrics),1)


class TestLauncher(unittest.TestCase):
    def test_accelerated_anchor_runs_three_branches_in_order_and_freezes_execution_plan(self):
        with tempfile.TemporaryDirectory() as temp, patch('builtins.print'):
            root = Path(temp)
            source = root/'source'
            (source/'protocol').mkdir(parents=True)
            (source/'partition_manifest.csv').write_text('fixture', encoding='utf-8')
            write_json(source/'bridge_metadata.json', {})
            args = SimpleNamespace(source_run=source, output_root=root/'gpu1', data_root=root,
                                   num_workers=0, resume=False, anchor=60, branch=None,
                                   fast_execution_v2=True, feedback_batch_size=128, feedback_cache_gib=4.)
            jobs = []
            def child(command, **kwargs):
                run = Path(command[1])
                job = read_json(run/'diagnostic_job.json')
                jobs.append(job)
                write_json(run/'diagnostic_completion.json', {'completed_round':job['anchor']+5})
            with patch.multiple('scripts.run_method_a_diagnostics',
                                preflight=lambda *a: {'ready_files':True}, source_receipt=lambda *a: {},
                                file_hash=lambda *a: 'fixture',
                                build_command=lambda source, run, *a: ['fixture', str(run)]), \
                    patch('torch.cuda.is_available', return_value=True), \
                    patch('scripts.run_method_a_diagnostics.summarize', return_value={}), \
                    patch('scripts.run_method_a_diagnostics.subprocess.run', side_effect=child):
                train(args)
                self.assertEqual([j['branch'] for j in jobs], ['full-cp','ordinary','norm-matched'])
                self.assertTrue(all(j['anchor']==60 for j in jobs))
                self.assertTrue(all(j['execution_override']==execution_config_v2(128,4.) for j in jobs))
                args.feedback_batch_size = 64
                with self.assertRaisesRegex(ValueError, 'Plan/source/code changed'):
                    train(args)
                args.feedback_batch_size, args.anchor, args.output_root = 128, 80, root/'gpu2'
                train(args)
                self.assertEqual([j['anchor'] for j in jobs], [60]*3+[80]*3)
                self.assertTrue(all(Path(j['full_branch']) == (root/'gpu2/anchor80/full-cp').resolve()
                                    for j in jobs[3:]))

    def test_single_branch_launch_keeps_plan_and_requires_completed_reference(self):
        with tempfile.TemporaryDirectory() as temp, patch('builtins.print'):
            root = Path(temp)
            source, output = root/'source', root/'output'
            (source/'protocol').mkdir(parents=True)
            (source/'partition_manifest.csv').write_text('fixture', encoding='utf-8')
            write_json(source/'bridge_metadata.json', {})
            args = SimpleNamespace(source_run=source, output_root=output, data_root=root,
                                   num_workers=0, resume=False, anchor=80, branch='norm-matched')
            def child(command, **kwargs):
                write_json(Path(command[1])/'diagnostic_completion.json', {'completed_round':85})
            with patch.multiple('scripts.run_method_a_diagnostics',
                                preflight=lambda *a: {'ready_files':True},
                                source_receipt=lambda *a: {}, file_hash=lambda *a: 'fixture',
                                build_command=lambda source, run, *a: ['fixture', str(run)]), \
                    patch('torch.cuda.is_available', return_value=True), \
                    patch('scripts.run_method_a_diagnostics.summarize', return_value={}), \
                    patch('scripts.run_method_a_diagnostics.subprocess.run', side_effect=child) as launch:
                with self.assertRaisesRegex(ValueError, '--anchor 80 --branch full-cp'):
                    train(args)
                launch.assert_not_called()
                self.assertFalse((output/'anchor80/norm-matched').exists())
                for index, branch in enumerate(('full-cp', 'ordinary', 'norm-matched'), 1):
                    args.branch = branch
                    train(args)
                    self.assertEqual(launch.call_count, index)
                    self.assertEqual(Path(launch.call_args.args[0][1]), (output/'anchor80'/branch).resolve())
                self.assertFalse((output/'anchor60').exists())
                plan = read_json(output/'diagnostic_plan.json')
                self.assertEqual(plan['anchors'], [60, 80])
                self.assertEqual(plan['branches'], ['full-cp', 'ordinary', 'norm-matched'])
                train(args)
                self.assertEqual(launch.call_count, 3)

    def test_missing_states_cannot_launch_training(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root/'source'
            source.mkdir()
            write_json(source/'sfra_config.json',dict(variant='full-cp',retention_weight=10.,classification_weight=1.,
                seed=42,protocol_seed=42,partition='client-longtail',correction_steps=3,correction_step_size=.1))
            result = preflight(source,root/'output')
            self.assertFalse(result['ready_files'])
            args = SimpleNamespace(source_run=source,output_root=root/'output',data_root=root,num_workers=0,resume=False)
            with patch('scripts.run_method_a_diagnostics.subprocess.run') as call:
                with self.assertRaisesRegex(ValueError,'Missing reference'):
                    train(args)
                call.assert_not_called()

    def test_command_relocates_only_io_and_adds_isolated_entry(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            write_json(root/'command.json',['python','-u','federated_main.py','--root','OLD',
                '--output-dir','OLD_OUT','--client_schedule_file','OLD_SCHEDULE','--lac_partition_manifest','OLD_PARTITION',
                '--sfra_resume','','--sfra_classification_weight','1','DATALOADER.NUM_WORKERS','8'])
            command = build_command(root,root/'run',root/'data',0)
            self.assertEqual(command[command.index('--sfra_classification_weight')+1],'1')
            self.assertEqual(command[command.index('--lac_partition_manifest')+1],str(root/'run/protocol/partition_source.csv'))
            self.assertLess(command.index('--method_a_diagnostic_manifest'),command.index('DATALOADER.NUM_WORKERS'))
            self.assertNotIn('--sfra_fast_execution_v2', command)  # anchor reconstruction uses original mode

    def test_summary_uses_fixed_window_and_common_sample_cohort(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            groups = ['Overall','Many35','Medium35','Few30','Tail20']
            baseline = [dict(sample_id=0,class_id=45,correct=1),dict(sample_id=1,class_id=80,correct=0)]
            for anchor in (60,80):
                for branch in ('full-cp','ordinary','norm-matched'):
                    run = root/f'anchor{anchor}'/branch
                    metrics = [dict(round=anchor,candidate=branch,kind='anchor',group=g,accuracy=50.) for g in groups]
                    metrics += [dict(round=r,candidate=branch,kind='committed',group=g,accuracy=float(r-anchor)*10.)
                                for g in groups for r in range(anchor+1,anchor+6)]
                    write_csv(run/'candidate_metrics.csv',metrics)
                    write_csv(run/'witness_metrics.csv',[])
                    write_json(run/'class_prior.json',{'counts':[200]*35+[50]*35+[5]*30})
                    write_json(run/'diagnostic_completion.json',{'completed_round':anchor+5})
                    final = [dict(sample_id=0,class_id=45,correct=int(branch!='ordinary')),
                             dict(sample_id=1,class_id=80,correct=1)]
                    write_csv(run/f'predictions/r{anchor+5:03d}_{branch}_committed.csv',final)
                    if branch=='full-cp':
                        write_csv(run/f'predictions/r{anchor:03d}_full-cp_anchor.csv',baseline)
            result = summarize(root)
            self.assertTrue(result['complete'])
            summary = read_csv(root/'short_run_summary.csv')
            self.assertTrue(all(float(r['mean_five_accuracy'])==30. for r in summary))
            retention = read_csv(root/'sample_retention.csv')
            ordinary = next(r for r in retention if r['branch']=='ordinary' and r['group']=='Medium35')
            self.assertEqual(int(ordinary['correct_to_wrong']),1)
            self.assertEqual(float(ordinary['retention_fraction']),0.)


class TestPredictionEvaluation(unittest.TestCase):
    def test_sample_ids_and_predictions_are_observational(self):
        with tempfile.TemporaryDirectory() as temp, patch('builtins.print'):
            runtime = tiny_runtime(Path(temp))
            del runtime.evaluate_candidate
            bank = runtime.bank
            text = torch.nn.functional.normalize(torch.randn(100,3),dim=1)
            def inference(images):
                features = torch.nn.functional.normalize(bank.core.image_encoder(images),dim=1)
                return bank.core.logit_scale.exp() * features @ text.T
            data = [dict(img=torch.randn(1,2,2),label=c) for c in range(100)]
            runtime.global_trainer = SimpleNamespace(model=bank.model,device=torch.device('cpu'),
                test_loader=torch.utils.data.DataLoader(data,batch_size=20,shuffle=False),
                parse_batch_test=lambda batch:(batch['img'],batch['label']),model_inference=inference)
            runtime.trainer.training_logit_adjustment = torch.zeros(100)
            runtime.groups = {'Overall':np.arange(3)}
            before = copy.deepcopy(bank.model.state_dict())
            rng = torch.random.get_rng_state().clone()
            runtime.evaluate_candidate(before,61,'full-cp','same-state')
            self.assertTrue(torch.equal(rng,torch.random.get_rng_state()))
            for key,value in before.items():
                torch.testing.assert_close(value,bank.model.state_dict()[key],rtol=0,atol=0)
            rows = read_csv(Path(temp)/'predictions/r061_full-cp_same-state.csv')
            self.assertEqual([int(r['sample_id']) for r in rows],list(range(100)))
            self.assertEqual([int(r['class_id']) for r in rows],list(range(100)))


class TestDiagnosticExecutionOverride(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def test_real_visual_switch_preserves_anchor_history_rng_and_checks_reproduction(self):
        from test_sfra_execution import visual_bank
        with tempfile.TemporaryDirectory() as temp, patch('builtins.print'):
            root = Path(temp)
            bank = visual_bank(False)
            bank.core.prompt_learner = torch.nn.Linear(1, 1).requires_grad_(False)
            bank.core.text_encoder = torch.nn.Linear(1, 1).requires_grad_(False)
            bank.core.tokenized_prompts = torch.zeros(1, dtype=torch.long)
            state = copy.deepcopy(bank.model.state_dict())
            reference = bank.evaluate()['scores']
            rt = object.__new__(MethodADiagnosticRuntime)
            rt.trainer = SimpleNamespace(model=bank.model)
            rt.global_trainer = SimpleNamespace(model=copy.deepcopy(bank.model))
            rt.bank, rt.source, rt.anchor, rt.root = bank, root/'source', 60, root/'run'
            rt.keys = [k for k in state if k.endswith(('_lora_A','_lora_B'))]
            rt.execution_config = None
            rt.job = dict(execution_override=execution_config_v2(128, 4.))
            rt.history = FunctionalHistory(reference, bank.clients)
            history = copy.deepcopy(rt.history.state_dict())
            path = rt.source/'sfra_rounds/r060/tokens.npz'
            path.parent.mkdir(parents=True)
            np.savez(path, F_committed=reference.numpy())
            rng = torch.get_rng_state().clone()
            rt.activate_execution_override(state)
            self.assertIsInstance(bank._fast_feedback, ResidentWitnessFeedback)
            self.assertEqual(bank._fast_feedback.forward_batch_size, 128)
            self.assertEqual(bank._fast_feedback.cache_limit, 4*1024**3)
            self.assertTrue(torch.equal(rng, torch.get_rng_state()))
            for key, value in state.items():
                torch.testing.assert_close(value, bank.model.state_dict()[key], rtol=0, atol=0)
            for key, value in history.items():
                if torch.is_tensor(value):
                    torch.testing.assert_close(value, rt.history.state_dict()[key], rtol=0, atol=0)
            audit = read_json(rt.root/'diagnostic_execution_audit.json')
            self.assertTrue(audit['switch_after_reference_anchor_and_history_audits'])
            self.assertIsNone(audit['reference_execution'])
            self.assertLessEqual(audit['anchor_margin_max_error'], 5e-6)
            np.savez(path, F_committed=reference.numpy()+.1)
            with self.assertRaisesRegex(ValueError, 'Accelerated anchor witness reproduction error'):
                rt.activate_execution_override(state)

    def test_runtime_switches_after_anchor_load_and_before_training_also_on_resume(self):
        with tempfile.TemporaryDirectory() as temp, patch('builtins.print'):
            rt = tiny_runtime(Path(temp)/'run')
            rt.job['execution_override'] = execution_config_v2(128,4.)
            reference_load = rt.load_anchor
            calls = []
            def load():
                state = reference_load()
                calls.append('anchor')
                return state
            def activate(state):
                self.assertEqual(calls[-1], 'anchor')
                self.assertTrue(rt.history.valid.all())
                calls.append('execution')
            original_train = rt.train_phase
            def train_phase(*args, **kwargs):
                self.assertEqual(calls[-1], 'execution')
                return original_train(*args, **kwargs)
            rt.load_anchor, rt.activate_execution_override, rt.train_phase = load, activate, train_phase
            rt.run(None)
            rt.job['resume'] = True
            rt.run(None)
            self.assertEqual(calls, ['anchor','execution','anchor','execution'])


if __name__=='__main__':
    unittest.main()
