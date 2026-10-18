"""CPU integration checks; synthetic fixtures are not experimental results."""
import copy
from contextlib import nullcontext
from pathlib import Path
import shutil
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torch.utils.data import DataLoader

from scripts import run_six_ablation as launcher
from scripts import run_cliplora_joint_aggregation as single
from scripts.run_ab_validation import write_json
from tools.client_aggregation import six_ablation as six
from tools.client_aggregation.analysis import expected_stages, audit_run
from tools.client_aggregation.parallel import (LoRAClientPool, clone_models, plan_phase,
    train_local, phase_logit_adjustment, compare_states)
from tools.client_aggregation.protocol import (SCHEMA, aggregation_spec, audit_runtime_config,
    digest, load_json, read_csv, validate_arm, write_table, client_execution_config)
from tools.client_aggregation.runtime import runtime_class
from tools.sfra.maintext import FROZEN
from utils.cliplora_functional_feedback import snapshot
from test_joint_client_parallel import Data, Model, canonical_step
import test_joint_aggregation as joint_tests
from test_joint_aggregation import TinyModel

REPO = Path(__file__).resolve().parents[1]


class TestSixAblation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)
        cls.spec = aggregation_spec(REPO/'references/full10_clientlt', .1, 'fedavg')

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def job(self, cell, mode='smoke'):
        return dict(schema=SCHEMA, arm=six.CELLS[cell][0], mode=mode,
            aggregation=copy.deepcopy(self.spec), contract=six.contract_for(cell),
            settings=dict(seed=42, client_concurrency=10, parallel_validation='off',
                six_ablation_experiment=six.EXPERIMENT, six_ablation_cell=cell))

    def test_six_cells_parameter_and_gpu_assignment(self):
        args = launcher.parse_args(['--stage', 'server'])
        self.assertEqual(args.client_concurrency, 10)
        self.assertEqual(args.output_root.name, 'six_ablation_methodA_LA_fedavg_parallel10')
        mapping = launcher.assignment(args)
        self.assertEqual(mapping, dict(frozen_a=0, plain_a=1, method_a=2,
                                      frozen_a_b=0, plain_a_b=1, method_a_b=2))
        args = launcher.parse_args(['--gpus', '2', '4', '7', '--client-concurrency', '13'])
        self.assertEqual(launcher.assignment(args)['method_a_b'], 7)
        for arguments in (['--client-concurrency', '1'], ['--client-concurrency', '31'],
                          ['--gpus', '0', '0', '1'], ['--stage', 'run']):
            with self.assertRaises(SystemExit), patch('sys.stderr'):
                launcher.parse_args(arguments)
        for cell in six.CELLS:
            job = self.job(cell)
            validate_arm(job)
            self.assertEqual(client_execution_config(job)['max_concurrent_clients'], 10)
            self.assertEqual(six.has_b(job), cell.endswith('_b'))
            for change in ('loss', 'aggregation', 'slots'):
                bad = copy.deepcopy(job)
                if change == 'loss':
                    bad['contract']['ordinary_phase_losses']['A']['la_tau'] = 1. - bad['contract']['ordinary_phase_losses']['A']['la_tau']
                elif change == 'aggregation':
                    bad['aggregation']['rule'] = 'joint'
                else:
                    bad['settings']['client_concurrency'] = 1
                with self.assertRaises(ValueError):
                    validate_arm(bad)

    def test_registration_fedavg_and_resume_identity(self):
        with tempfile.TemporaryDirectory() as temp:
            args = launcher.parse_args(['--output-root', temp])
            suite = launcher.register(args)
            self.assertEqual(suite, launcher.register(args))
            self.assertEqual(len(suite['cells']), 6)
            for cell, (arm, _) in six.CELLS.items():
                plan = load_json(Path(temp)/cell/'experiment_plan.json')
                self.assertEqual(plan['aggregation']['weights'], self.spec['solver']['sample_weights'])
                job = single.make_job(plan, arm)
                command = single.command_for(job, Path(temp)/cell)
                self.assertEqual(command[command.index('--lac_la_tau')+1], '1')
                self.assertEqual(command[command.index('--sfra_variant')+1], 'full-cp' if arm == 'method_a' else 's')
            args.client_concurrency = 12
            with self.assertRaisesRegex(ValueError, 'Registered settings'):
                launcher.register(args)

    def test_dispatch_starts_six_isolated_jobs_two_per_gpu(self):
        calls = []
        def popen(command, **kwargs):
            calls.append((command, kwargs))
            return SimpleNamespace(pid=100+len(calls), poll=lambda: 0, returncode=0)
        with tempfile.TemporaryDirectory() as temp, patch('builtins.print'):
            args = launcher.parse_args(['--stage', 'server', '--output-root', temp])
            with patch.object(launcher.subprocess, 'Popen', side_effect=popen):
                launcher.dispatch(args)
            self.assertEqual(len(calls), 6)
            for gpu in ('0', '1', '2'):
                self.assertEqual(sum(kw['env']['CUDA_VISIBLE_DEVICES'] == gpu for _, kw in calls), 2)
            for command, kwargs in calls:
                self.assertEqual(command[command.index('--client-concurrency')+1], '10')
                self.assertEqual(command[command.index('--stage')+1], 'run')
                self.assertIn('CUDA_DEVICE_ORDER', kwargs['env'])

    def test_single_worker_smokes_then_formal_and_propagates_failure(self):
        with tempfile.TemporaryDirectory() as temp, patch('builtins.print'):
            args = ['--stage', 'run', '--cells', 'plain_a_b', '--output-root', temp]
            with patch.object(single, 'preflight'), patch.object(single, 'execute') as execute:
                launcher.main(args)
                self.assertEqual([c.args[0]['mode'] for c in execute.call_args_list], ['smoke', 'formal'])
            with patch.object(single, 'preflight'), patch.object(single, 'execute', side_effect=ValueError('smoke failure')) as execute:
                with self.assertRaisesRegex(ValueError, 'smoke failure'):
                    launcher.main(args)
                self.assertEqual(execute.call_count, 1)

    def test_real_pool_uses_LA_for_method_A_and_CE_for_plain_A(self):
        torch.manual_seed(29)
        model = Model()
        with torch.no_grad():
            model.linear.w_lora_B.copy_(torch.tensor([[.1, .2], [-.3, .4]]))
        keys = sorted(n for n, _ in model.named_parameters() if '_lora_' in n)
        state = snapshot(model)
        adjustment = torch.tensor([-.2, -2.1])
        runtime = SimpleNamespace(job=self.job('method_a_b'),
                                  trainer=SimpleNamespace(training_logit_adjustment=adjustment))
        pool = object.__new__(LoRAClientPool)
        pool.runtime, pool.device, pool.slots, pool.keys = runtime, torch.device('cpu'), 10, keys
        pool.models = clone_models(model, keys, 10)
        pool.loaders = {c: DataLoader(Data(3+c%3, c), batch_size=4, shuffle=True) for c in range(30)}
        pool.streams = [SimpleNamespace(synchronize=lambda: None) for _ in range(10)]
        pool.optimizer_factory = lambda m, factor: (torch.optim.SGD(
            [p for p in m.parameters() if p.requires_grad], lr=.02, momentum=.9), None)
        actual_step = canonical_step()
        observed, lock = [], threading.Lock()
        def step(*args, **kwargs):
            with lock:
                observed.append(kwargs['logit_adjustment'])
            return actual_step(*args, **kwargs)
        pool.step = step
        cases = [(cell, factor) for cell, (arm, _) in six.CELLS.items()
                 for factor in ('A', 'B') if arm != 'frozen' or factor == 'B']
        for cell, factor in cases:
            runtime.job = self.job(cell)
            expected_adjustment = adjustment if factor == 'B' or cell in ('method_a', 'method_a_b') else None
            plans = plan_phase(pool.loaders, list(pool.loaders), factor, 42, 1)
            observed.clear()
            # Exercise the actual pool/work queue/optimizer path on CPU, mocking only CUDA plumbing.
            with patch('torch.cuda.synchronize'), patch('torch.cuda.memory_allocated', return_value=0), \
                 patch('torch.cuda.max_memory_allocated', return_value=0), \
                 patch('torch.cuda.max_memory_reserved', return_value=0), \
                 patch('torch.cuda.device', side_effect=lambda _: nullcontext()), \
                 patch('torch.cuda.stream', side_effect=lambda _: nullcontext()):
                result, execution = pool.run(state, list(pool.loaders), plans, factor)
            self.assertEqual(execution['slots'], 10)
            self.assertEqual(list(result), list(range(30)))
            self.assertTrue(observed)
            self.assertTrue(all(x is expected_adjustment for x in observed), (cell, factor))
            expected, _ = train_local(copy.deepcopy(model), state, keys, pool.loaders[0].dataset,
                plans[0], factor, 'cpu', pool.optimizer_factory, actual_step, expected_adjustment)
            self.assertTrue(compare_states(expected, result[0][0])['bitwise_equal'])
            opposite, _ = train_local(copy.deepcopy(model), state, keys, pool.loaders[0].dataset,
                plans[0], factor, 'cpu', pool.optimizer_factory, actual_step, adjustment if expected_adjustment is None else None)
            self.assertFalse(compare_states(opposite, result[0][0])['bitwise_equal'])
        self.assertIs(runtime.trainer.training_logit_adjustment, adjustment)
        runtime.job = {}
        self.assertIs(phase_logit_adjustment(runtime, 'A'), adjustment)  # Legacy behavior retained.
        runtime.job = self.job('method_a_b')
        runtime.trainer.training_logit_adjustment = None
        with self.assertRaisesRegex(ValueError, 'missing its logit adjustment'):
            phase_logit_adjustment(runtime, 'A')
        runtime.job = self.job('plain_a')
        self.assertIsNone(phase_logit_adjustment(runtime, 'A'))

    def configure(self, root, cell):
        job = self.job(cell)
        rt = object.__new__(runtime_class(job))
        rt.root, rt.variant = root, 'full-cp' if job['arm'] == 'method_a' else 's'
        rt.q = dict(enumerate(self.spec['weights']))
        rt.config = dict(seed=42, protocol_seed=42, topology='client-longtail', la_tau=1.,
            normal_steps_expected=105600, extra_steps_expected=31680,
            normal_trainable_factor='B', control_enabled=False, a_lr_mult=1., extra_a_lr=.001,
            extra_weight_decay=0., normal_b_lr=.001, extra_trainable_factor='A',
            candidate_rounds=list(range(1, 91)))
        rt.sfra_config = dict(FROZEN, variant=rt.variant, base_training_config=rt.config)
        rt.rounds = list(range(1, 91))
        if job['arm'] == 'method_a':
            rt.sfra_config['classification_weight'] = 1.
        rt.audit = SimpleNamespace(counts=torch.tensor(self.spec['matrix']), meta={})
        root.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(REPO/'references/full10_clientlt/partition_manifest.csv', root/'partition_manifest.csv')
        rt.configure_experiment()
        write_json(root/'sfra_config.json', rt.sfra_config)
        write_json(root/'execution_config.json', dict(version=2, feedback_forward_batch_size=128, device_cache_gib=4.))
        audit_runtime_config(root, job)
        return rt, job

    def test_runtime_six_configurations_and_no_history_in_B_only_cells(self):
        with tempfile.TemporaryDirectory() as temp:
            for cell in six.CELLS:
                rt, job = self.configure(Path(temp)/cell, cell)
                self.assertEqual(rt.config['ordinary_phase_losses']['A']['la_tau'], 1. if job['arm'] == 'method_a' else 0.)
                self.assertEqual(rt.sfra_config.get('b_transfer'), six.transfer_config('smoke') if six.has_b(job) else None)
                self.assertEqual(rt.rounds, [] if job['arm'] == 'frozen' else list(range(1, 91)))
                self.assertEqual((1, 'after_B_transfer') in set(expected_stages(job, 1)), six.has_b(job))
                if job['arm'] != 'method_a':
                    self.assertIsNone(rt.history_from_state(None))
                    with self.assertRaises(ValueError):
                        rt.history_from_state({'unexpected': 'retention'})

    def test_frozen_and_plain_B_finalize_once_and_resume_next_round(self):
        for cell in ('frozen_a_b', 'plain_a_b'):
            rt = object.__new__(runtime_class(self.job(cell)))
            rt.job = self.job(cell)
            rt.args = SimpleNamespace(sfra_stop_after_round=2)
            rt.trainer = SimpleNamespace(model=TinyModel())
            rt.keys = ['proj_lora_A', 'proj_lora_B']
            rt.resume_payload = None
            rt.history = None
            calls, observed, published, saved = [], [], [], []
            rt.load_training_state = lambda state: rt.trainer.model.load_state_dict(state)
            def train(state, rnd, factor, extra=False):
                calls.append((rnd, factor))
                out = copy.deepcopy(state)
                out['proj_lora_'+factor] += 1
                return out
            rt.train_phase = train
            rt.b_transfer = SimpleNamespace(observe_global=lambda state, rnd, committed: observed.append(rnd))
            rt.publish = lambda state, rnd, decision: published.append(rnd)
            rt.checkpoint = lambda state, rnd: saved.append((rnd, copy.deepcopy(state)))
            rt.run(snapshot(rt.trainer.model))
            self.assertEqual(observed, [1, 2])
            self.assertEqual(published, [0, 1, 2])
            self.assertEqual([f for _, f in calls], ['B', 'B'] if cell == 'frozen_a_b' else ['B', 'A', 'B', 'A'])
            rt.resume_payload = {'completed_round': 1}
            rt.restore = lambda: (saved[1][1], 1)
            calls.clear(); observed.clear()
            rt.run(saved[0][1])
            self.assertEqual(observed, [2])
            self.assertTrue(all(rnd == 2 for rnd, _ in calls))

    def test_complete_artifact_audit_all_six_cells_and_wrong_A_loss_rejected(self):
        fixture = joint_tests.TestJointAggregation()
        with tempfile.TemporaryDirectory() as temp, patch('builtins.print'):
            for cell, (arm, use_b) in six.CELLS.items():
                root = Path(temp)/cell
                fixture.audit_fixture(root, arm)
                rt, job = self.configure(root, cell)
                write_json(root/'joint_job.json', job)
                write_json(root/'joint_receipt.json', dict(schema=SCHEMA, job_digest=digest(job), code_unchanged=True, exit_code=0))
                p = load_json(root/'progress.json')
                p.update(weights_sha256=self.spec['weights_sha256'], b_transfer_events=int(use_b), b_transfer_optimizer_steps=0)
                write_json(root/'progress.json', p)
                for factor, phase in [('B', 'normal_B')]+([] if arm == 'frozen' else [('A', 'refresh_A')]):
                    rt.phase_aggregation_weights(list(range(30)), 1, factor, factor == 'A')
                    event_path = root/'events'/('r001_c000_main_'+phase)/'event.json'
                    event = load_json(event_path)
                    event['server_weights'] = self.spec['weights']
                    write_json(event_path, event)
                    write_json(root/'parallel_execution'/f'r001_{factor}.json', dict(round=1, factor=factor,
                        slots=10, clients=30, selected_client_ids=list(range(30)),
                        client_audits=[dict(client_id=c, slot=c%10) for c in range(30)],
                        ordinary_loss=six.losses_for(job)[factor], logit_adjustment_applied=factor == 'B' or arm == 'method_a'))
                if use_b:
                    folder = root/'b_transfer_rounds/r001'
                    write_json(folder/'calibration_manifest.json', dict(shared_donor_ids=[]))
                    write_table(folder/'optimization_steps.csv', [])
                    write_table(root/'b_transfer_rounds.csv', [dict(round=1, mode='shared', calibration_profile='coverage_tradeoff',
                        optimizer_steps=0, feedback_synchronizations=0, learning_rate=.3, tail_weight=.35,
                        non_tail_sampling='class-cyclic', effective_global_transfer_norm=0., A_updated_this_round=arm != 'frozen')])
                    dst = root/'stage_predictions/after_B_transfer'
                    dst.mkdir(parents=True, exist_ok=True)
                    for ext in ('json', 'npz'):
                        shutil.copyfile(root/'stage_predictions/ordinary_B'/('r001.'+ext), dst/('r001.'+ext))
                audit_run(root, job, 1)
                if arm != 'frozen':
                    path = root/'parallel_execution/r001_A.json'
                    bad = load_json(path); bad['logit_adjustment_applied'] = not bad['logit_adjustment_applied']; write_json(path, bad)
                    with self.assertRaisesRegex(ValueError, 'phase loss audit'):
                        audit_run(root, job, 1)

    def test_pairing_allows_method_A_loss_change_and_rejects_misrecorded_loss(self):
        import numpy as np
        from tools.client_aggregation.protocol import PAIR_HASHES
        def result(cell):
            job = self.job(cell)
            job['code_sha256'] = {'source': 'paired'}
            return dict(job=job, metadata={k: 'same' for k in PAIR_HASHES},
                config=dict(client_execution=client_execution_config(job), ordinary_phase_losses=six.losses_for(job)),
                predictions={0: {k: np.array([0, 1]) for k in ('sample_id', 'class_id', 'prediction', 'correct')}})
        left, right = result('method_a_b'), result('plain_a_b')
        pair = launcher.pair_check(left, right)
        self.assertEqual(pair['status'], 'eligible')
        self.assertEqual(pair['A_local_loss_left']['la_tau'], 1.)
        self.assertEqual(pair['A_local_loss_right']['la_tau'], 0.)
        left['config']['ordinary_phase_losses']['A']['la_tau'] = 0.
        self.assertIn('left.ordinary_phase_losses', launcher.pair_check(left, right)['mismatches'])

    def test_old_version_cannot_be_relabelled_or_resumed_as_new(self):
        with tempfile.TemporaryDirectory() as temp:
            args = launcher.parse_args(['--output-root', temp])
            suite = launcher.register(args)
            suite['experiment'] = 'six_ablation_ce_a_la_b_fedavg_v1'
            write_json(Path(temp)/'six_ablation_plan.json', suite)
            with self.assertRaisesRegex(ValueError, 'Registered settings'):
                launcher.register(args)
            with self.assertRaisesRegex(ValueError, 'different ablation version'):
                launcher.summarize(Path(temp))

    def test_actual_shared_B_checkpoint_restore_without_A_history(self):
        import test_new_a_old_b as transfer_fixture
        from utils.cliplora_b_shared_transfer import SharedDonorBTransfer
        def make(root, cell, last):
            rt = transfer_fixture.TestNewAOldB().make_tiny(root, last)
            rt.job = self.job(cell, 'formal')
            rt.__class__ = runtime_class(rt.job)
            rt.is_frozen = cell == 'frozen_a_b'
            rt.initial = copy.deepcopy(rt.base_state)
            rt.variant = 's'
            rt.history = None
            rt.sfra_config.update(variant='s', refresh_rounds=rt.job['contract']['A_rounds'],
                                  b_transfer=six.transfer_config('formal'))
            rt.b_transfer = SharedDonorBTransfer(rt)
            return rt
        for cell in ('frozen_a_b', 'plain_a_b'):
            with tempfile.TemporaryDirectory() as temp, patch('builtins.print'):
                full = make(Path(temp)/'full', cell, 31)
                full.run(copy.deepcopy(full.base_state))
                final = snapshot(full.trainer.model)
                self.assertIsNone(full.history)
                self.assertIsNone(full.b_transfer.pending)
                self.assertEqual([r['round'] for r in full.b_transfer.records], [30])
                progress = load_json(full.root/'progress.json')
                six.audit_b(full.root, full.job, progress, 31)
                self.assertEqual(progress['functional_correction_steps'], 0)
                paused = make(Path(temp)/'resume', cell, 29)
                paused.run(copy.deepcopy(paused.base_state))
                resumed = make(paused.root, cell, 31)
                resumed.resume_payload = torch.load(paused.root/'checkpoints/sfra_last.pt', weights_only=False)
                resumed.run(copy.deepcopy(resumed.base_state))
                self.assertIsNone(resumed.history)
                self.assertEqual([r['round'] for r in resumed.b_transfer.records], [30])
                for key, value in final.items():
                    torch.testing.assert_close(value, resumed.trainer.model.state_dict()[key], atol=0, rtol=0)


if __name__ == '__main__':
    unittest.main()
