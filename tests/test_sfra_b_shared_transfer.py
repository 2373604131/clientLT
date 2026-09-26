"""CPU tensor/launcher checks; no dataset loading, CLIP training or GPU smoke run."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
import numpy as np
from torch.nn import functional as F

from scripts import run_cliplora_sfra as launcher
from tools.sfra.summary import METRICS, pack, read_csv, summarize, write_csv
from utils.cliplora_b_transfer import calibration_batches, transfer_config
from utils.cliplora_b_shared_transfer import (SharedDonorBTransfer, shared_transfer_config,
                                            shared_calibration_batches)
from utils.cliplora_sfra import SFRARuntime


A_KEY, B_KEY = 'layer.w_lora_A', 'layer.w_lora_B'


class ToyLayer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.w_lora_A = torch.nn.Parameter(torch.tensor([[1., .3], [-.2, .8]]), requires_grad=False)
        self.w_lora_B = torch.nn.Parameter(torch.tensor([[.2, -.1], [.4, .3]]))
        self.scaling = .5

    def forward(self, images):
        return self.scaling*F.linear(F.linear(images, self.w_lora_A), self.w_lora_B)


def tiny_transfer(root, regularization=.001):
    args = SimpleNamespace(b_transfer_probe_step=.1, b_transfer_lr=.03,
                           b_transfer_reg=regularization, sfra_b_aggregation='sample')
    transfer = object.__new__(SharedDonorBTransfer)
    model = torch.nn.Module()
    model.layer = ToyLayer()
    transfer.model, transfer.device = model, torch.device('cpu')
    transfer.runtime = SimpleNamespace(root=Path(root), args=SimpleNamespace(seed=42),
                                       q={0: .8, 1: .15, 2: .05}, keys=[A_KEY, B_KEY])
    transfer.config, transfer.keys = shared_transfer_config(args), [B_KEY]
    transfer.parameters = dict(model.named_parameters())
    transfer.modules, transfer.ranks = {B_KEY: model.layer}, {B_KEY: 2}
    transfer.recipients, transfer.tail = [1, 2], {0}
    transfer.groups = [{1: [0, 1]}, {0: [0], 1: [1, 2]}, {0: [0, 1]}]
    transfer.probes, transfer.counts = {1: {0: [0]}, 2: {0: [0, 1]}}, torch.tensor([[0, 2], [1, 2], [2, 0]])
    transfer.records = []
    images = {1: torch.tensor([[.7, .2], [-.4, 1.], [.9, -.2]]),
              2: torch.tensor([[.2, 1.], [-.8, .4]])}
    labels = {1: torch.tensor([0, 1, 1]), 2: torch.tensor([0, 0])}
    transfer.batch = lambda k, positions: (images[k][positions], labels[k][positions])

    def forward(x, y, diagnostic=False):
        prediction = model.layer(x)
        losses = (prediction-F.one_hot(y, 2)).square().mean(1)
        field = 'diagnostic_forward_images' if diagnostic else 'algorithm_forward_images'
        transfer.pending['summary'][field] += len(y)
        return losses, prediction, prediction

    transfer.forward = forward
    transfer.original_deltas = {
        0: {B_KEY: torch.tensor([[.2, .1], [.3, .4]])},
        1: {B_KEY: torch.tensor([[.1, -.2], [.1, -.3]])},
        2: {B_KEY: torch.tensor([[.3, -.2], [.2, .1]])}}
    transfer.selected = [0, 1, 2]
    return transfer


class TestSharedC(unittest.TestCase):
    def test_class_cyclic_preserves_tail_batches_budget_and_rng(self):
        groups = {80: list(range(7)), 81: list(range(7, 12))}
        groups.update({c: list(range(100+100*c, 100+100*c+(60 if c == 0 else 3))) for c in range(40)})
        original = copy.deepcopy(groups)
        legacy = calibration_batches(groups, {80, 81}, 42, 30, 5)
        np.random.seed(73)
        state = np.random.get_state()
        new = shared_calibration_batches(groups, {80, 81}, 42, 30, 5, 'class-cyclic')
        after = np.random.get_state()
        self.assertEqual(state[0], after[0])
        np.testing.assert_array_equal(state[1], after[1])
        self.assertEqual(state[2:], after[2:])
        self.assertEqual(new, shared_calibration_batches(groups, {80, 81}, 42, 30, 5, 'class-cyclic'))
        self.assertEqual(legacy, shared_calibration_batches(groups, {80, 81}, 42, 30, 5))
        self.assertEqual(groups, original)
        labels = {p: c for c, positions in groups.items() for p in positions}
        seen = set()
        for old, updated in zip(legacy, new):
            self.assertEqual(updated['tail_positions'], old['tail_positions'])
            self.assertEqual(len(updated['non_tail_positions']), len(old['non_tail_positions']))
            self.assertEqual(len(set(updated['non_tail_positions'])), 16)
            self.assertTrue(all(labels[p] not in {80, 81} for p in updated['non_tail_positions']))
            seen.update(labels[p] for p in updated['non_tail_positions'])
        self.assertEqual(len(seen), 32)  # Cursor continues across the two steps.

    def test_class_cyclic_sparse_and_tail_only_clients(self):
        for groups in ({80: [0, 1]}, {80: [0], 0: [1], 1: [2]},
                       {80: [0], 0: list(range(1, 45)), 1: [45]}):
            with self.subTest(groups=groups):
                old = calibration_batches(groups, {80}, 42, 30, 1)
                new = shared_calibration_batches(groups, {80}, 42, 30, 1, 'class-cyclic')
                valid = {p for c, positions in groups.items() if c != 80 for p in positions}
                for a, b in zip(old, new):
                    self.assertEqual(a['tail_positions'], b['tail_positions'])
                    self.assertEqual(len(a['non_tail_positions']), len(b['non_tail_positions']))
                    self.assertEqual(len(set(b['non_tail_positions'])), len(b['non_tail_positions']))
                    self.assertTrue(set(b['non_tail_positions']) <= valid)

    def test_weighted_shared_c_matches_combined_loss_with_rank_four(self):
        for weight in (.5, .35, .2):
            with self.subTest(weight=weight), tempfile.TemporaryDirectory() as temp:
                transfer = tiny_transfer(temp, regularization=.2)
                layer = transfer.model.layer
                layer.w_lora_A = torch.nn.Parameter(torch.tensor([[1., .3], [-.2, .8], [.5, -.4], [.1, .6]]), requires_grad=False)
                layer.w_lora_B = torch.nn.Parameter(torch.tensor([[.2, -.1, .1, .3], [.4, .3, -.2, .1]]))
                transfer.parameters = dict(transfer.model.named_parameters())
                transfer.ranks = {B_KEY: 4}
                transfer.original_deltas = {
                    k: {B_KEY: torch.cat([v[B_KEY], v[B_KEY]*.5], dim=1)}
                    for k, v in transfer.original_deltas.items()}
                transfer.config = shared_transfer_config(SimpleNamespace(
                    b_transfer_probe_step=.1, b_transfer_lr=.03, b_transfer_reg=.2,
                    b_transfer_non_tail_sampling='class-cyclic', b_transfer_tail_weight=weight))
                start = copy.deepcopy(transfer.model.state_dict())
                originals = copy.deepcopy(transfer.original_deltas)
                donors = [0, 2]
                manifests = {k: dict(batches=shared_calibration_batches(
                    transfer.groups[k], {0}, 42, 30, k, 'class-cyclic')) for k in transfer.recipients}
                transfer.pending = dict(round=30, folder=Path(temp), steps=[], feedback=[], summary=dict(
                    algorithm_forward_images=0, algorithm_backward_images=0, client_backward_batches=0,
                    diagnostic_forward_images=0,
                    optimizer_steps=0, feedback_synchronizations=0, extra_downlink_bytes=0, extra_upload_bytes=0))
                donor_tensors = torch.stack([originals[k][B_KEY] for k in donors])
                expected = torch.nn.Parameter(torch.zeros(2, 4, 4))
                optimizer = torch.optim.Adam([expected], lr=.03, betas=(.9, .999), eps=1e-8)
                for step in range(2):
                    optimizer.zero_grad(set_to_none=True)
                    b = start[B_KEY]+torch.bmm(donor_tensors, expected).mean(0)
                    objectives = []
                    for k in transfer.recipients:
                        batch = manifests[k]['batches'][step]
                        x, y = transfer.batch(k, batch['tail_positions']+batch['non_tail_positions'])
                        losses = (.5*F.linear(F.linear(x, start[A_KEY]), b)-F.one_hot(y, 2)).square().mean(1)
                        n = len(batch['tail_positions'])
                        objective = losses[:n].mean()
                        if n < len(y):
                            objective = weight*objective+(1-weight)*losses[n:].mean()
                        objectives.append(objective)
                    loss = torch.stack(objectives).mean()+.2*expected.square().sum()/2
                    loss.backward()
                    optimizer.step()
                with transfer.session(), patch('builtins.print'):
                    learned = transfer.calibrate_shared(donors, manifests, start)[B_KEY]
                self.assertEqual(tuple(learned.shape), (2, 4, 4))
                torch.testing.assert_close(learned, expected.detach(), atol=1e-7, rtol=1e-6)
                self.assertGreater(float(learned.abs().sum()), 0.)
                for key, value in transfer.model.state_dict().items():
                    self.assertTrue(torch.equal(value, start[key]))
                    self.assertIsNone(transfer.parameters[key].grad)
                for k in originals:
                    self.assertTrue(torch.equal(originals[k][B_KEY], transfer.original_deltas[k][B_KEY]))
                self.assertEqual(transfer.pending['summary']['optimizer_steps'], 2)
                self.assertEqual(transfer.pending['summary']['client_backward_batches'], 4)
                self.assertEqual(transfer.pending['summary']['algorithm_backward_images'], 10)
                self.assertEqual(transfer.pending['summary']['algorithm_forward_images'], 10)
                self.assertEqual(transfer.pending['summary']['diagnostic_forward_images'], 30)
                trace = transfer.pending['loss_trace']
                self.assertEqual([r['c_step'] for r in trace], [0, 1, 2])
                self.assertEqual(len(read_csv(Path(temp)/'c_loss_trace.csv')), 3)
                for row in trace:
                    self.assertAlmostEqual(row['fixed_pool_la'], (row['batch1_la']+row['batch2_la'])/2)
                    self.assertAlmostEqual(row['fixed_pool_objective'], row['fixed_pool_la']+row['regularization_penalty'])
                    self.assertIsNone(row['transfer_to_ordinary_ratio'])  # Toy ordinary B == start B.
                for step, row in enumerate(transfer.pending['steps'], 1):
                    self.assertAlmostEqual(row['la_before'], trace[step-1][f'batch{step}_la'], places=6)
                    self.assertEqual(row['same_batch_la_after'], trace[step][f'batch{step}_la'])
                    self.assertEqual(row['fixed_pool_objective_after'], trace[step]['fixed_pool_objective'])
                for row in transfer.pending['feedback']:
                    self.assertEqual(row['tail_loss_weight'], weight if row['client_id'] == 1 else 1.)

    def test_tradeoff_launcher_three_weights_resume_and_default_collection_root(self):
        with tempfile.TemporaryDirectory() as temp:
            def prepare(args, run):
                (run/'protocol').mkdir(parents=True)
                return run/'protocol/full_schedule.json', ''

            directories = []
            for weight in (.5, .35, .2):
                argv = ['entry.py', '--output-root', temp, '--transfer-tail-weight', str(weight), '--fast-execution-v2']
                with patch('sys.argv', argv), patch.object(launcher, 'prepare_protocol', prepare), \
                        patch.object(launcher.subprocess, 'run') as execute, patch('builtins.print'):
                    launcher.main(True, 'shared', 'class-cyclic', .35)
                command = execute.call_args.args[0]
                self.assertEqual(command[command.index('--b_transfer_tail_weight')+1], str(weight))
                self.assertEqual(command[command.index('--b_transfer_non_tail_sampling')+1], 'class-cyclic')
                self.assertEqual(command[command.index('--cliplora_rank')+1], '4')
                self.assertEqual(command[command.index('--sfra_retention_weight')+1], '10.0')
                self.assertEqual(command[command.index('--sfra_classification_weight')+1], '1.0')
                run = Path(command[command.index('--output-dir')+1])
                self.assertIn(f'_ntclass-cyclic_tw{weight:g}_fast_v2', run.name)
                directories.append(run)
                (run/'checkpoints').mkdir()
                (run/'checkpoints/sfra_last.pt').touch()
                with patch('sys.argv', argv+['--resume']), patch.object(launcher.subprocess, 'run') as execute, patch('builtins.print'):
                    launcher.main(True, 'shared', 'class-cyclic', .35)
                resumed = execute.call_args.args[0]
                expected = list(command)
                expected[expected.index('--sfra_resume')+1] = str(run/'checkpoints/sfra_last.pt')
                self.assertEqual(resumed, expected)
            self.assertEqual(len(set(directories)), 3)
            with patch('sys.argv', ['entry.py', '--stage', 'pack']), \
                    patch('tools.sfra.summary.summarize') as summarize_mock, \
                    patch('tools.sfra.summary.pack') as pack_mock:
                launcher.main(True, 'shared', 'class-cyclic', .35)
            expected_root = Path('output/cifar100_LT/sfra_b_shared_tradeoff')
            self.assertEqual(summarize_mock.call_args.args[0], expected_root)
            self.assertEqual(pack_mock.call_args.args[0], expected_root)

    def test_tradeoff_config_and_summary_do_not_mix_with_legacy(self):
        import tarfile
        base = dict(b_transfer_probe_step=.1, b_transfer_lr=.3, b_transfer_reg=.001)
        legacy = shared_transfer_config(SimpleNamespace(**base))
        explicit = shared_transfer_config(SimpleNamespace(**base, b_transfer_non_tail_sampling='sample', b_transfer_tail_weight=.5))
        self.assertEqual(legacy, explicit)
        self.assertNotIn('tail_weight', legacy)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)/'tradeoff'
            for weight in (.5, .35, .2):
                run = root/f'seed42/client-longtail/full-cp/tw{weight:g}'
                run.mkdir(parents=True)
                cfg = shared_transfer_config(SimpleNamespace(**base,
                    b_transfer_non_tail_sampling='class-cyclic', b_transfer_tail_weight=weight))
                self.assertEqual(cfg['steps'], 2)
                self.assertEqual(cfg['rounds'], legacy['rounds'])
                for field in ('donor_rule', 'donor_pool', 'c_scope', 'aggregation', 'regularization'):
                    self.assertEqual(cfg[field], legacy[field])
                (run/'sfra_config.json').write_text(json.dumps(dict(variant='full-cp', seed=42,
                    partition='client-longtail', retention_weight=10, classification_weight=1, b_transfer=cfg)))
                (run/'completion.json').write_text('{}')
                write_csv(run/'round_metrics.csv', [dict(round=r, **{k:float(r) for k in METRICS}) for r in range(101)])
                for name in ('sfra_rounds', 'sfra_costs'):
                    write_csv(run/f'{name}.csv', [])
                write_csv(run/'b_transfer_rounds.csv', [dict(round=30, optimizer_steps=2, client_backward_batches=44)])
                folder = run/'b_transfer_rounds/r030'
                folder.mkdir(parents=True)
                for name in ('receiver_summary', 'optimization_steps', 'probe_metrics'):
                    write_csv(folder/f'{name}.csv', [])
                write_csv(folder/'c_loss_trace.csv', [dict(round=30, c_step=s, fixed_pool_la=1.-.1*s)
                                                       for s in range(3)])
            with patch('builtins.print'):
                summarize(root)
                pack(root)
            rows = read_csv(root/'analysis/performance.csv')
            self.assertEqual(len(rows), 3)
            self.assertEqual({float(r['transfer_tail_weight']) for r in rows}, {.5, .35, .2})
            self.assertTrue(all(r['method'] == 'full-cp+B-shared-tradeoff' for r in rows))
            self.assertTrue(all(r['transfer_non_tail_sampling'] == 'class-cyclic' for r in rows))
            self.assertEqual(len(read_csv(root/'analysis/b_transfer_loss_trace.csv')), 9)
            with tarfile.open(Path(temp)/'tradeoff_analysis.tar.gz') as archive:
                self.assertIn('tradeoff/analysis/performance.csv', archive.getnames())

    def test_joint_adam_matches_single_combined_objective_and_freezes_base(self):
        with tempfile.TemporaryDirectory() as temp:
            transfer = tiny_transfer(temp, regularization=.2)
            start = copy.deepcopy(transfer.model.state_dict())
            donors = [0, 2]
            manifests = {k: dict(batches=calibration_batches(transfer.groups[k], {0}, 42, 30, k))
                         for k in transfer.recipients}
            transfer.pending = dict(round=30, folder=Path(temp), steps=[], feedback=[], summary=dict(
                algorithm_forward_images=0, algorithm_backward_images=0, client_backward_batches=0,
                diagnostic_forward_images=0,
                optimizer_steps=0, feedback_synchronizations=0, extra_downlink_bytes=0, extra_upload_bytes=0))
            tensors = torch.stack([transfer.original_deltas[j][B_KEY] for j in donors])
            expected = torch.nn.Parameter(torch.zeros(2, 2, 2))
            optimizer = torch.optim.Adam([expected], lr=.03, betas=(.9, .999), eps=1e-8)
            for step in range(2):
                optimizer.zero_grad(set_to_none=True)
                effective_b = start[B_KEY]+torch.bmm(tensors, expected).mean(0)
                objectives = []
                for k in transfer.recipients:
                    batch = manifests[k]['batches'][step]
                    x, y = transfer.batch(k, batch['tail_positions']+batch['non_tail_positions'])
                    pred = .5*F.linear(F.linear(x, start[A_KEY]), effective_b)
                    losses = (pred-F.one_hot(y, 2)).square().mean(1)
                    nt = len(batch['tail_positions'])
                    groups = [losses[:nt].mean()]
                    if nt < len(y):
                        groups.append(losses[nt:].mean())
                    objectives.append(torch.stack(groups).mean())
                objective = torch.stack(objectives).mean()+.2*expected.square().sum()/2
                objective.backward()
                optimizer.step()
            with transfer.session(), patch('builtins.print'):
                learned = transfer.calibrate_shared(donors, manifests, start)[B_KEY]
            torch.testing.assert_close(learned, expected.detach(), atol=1e-7, rtol=1e-6)
            for key, value in transfer.model.state_dict().items():
                self.assertTrue(torch.equal(value, start[key]))
            self.assertIsNone(transfer.model.layer.w_lora_A.grad)
            self.assertIsNone(transfer.model.layer.w_lora_B.grad)
            self.assertTrue(transfer.model.layer.w_lora_B.requires_grad)
            self.assertEqual(transfer.pending['summary']['optimizer_steps'], 2)
            self.assertEqual(transfer.pending['summary']['client_backward_batches'], 4)
            self.assertEqual(transfer.pending['summary']['algorithm_backward_images'], 10)
            self.assertEqual([r['shared_c_version'] for r in transfer.pending['feedback']], [0, 0, 1, 1])
            self.assertEqual(len(transfer.model.layer._forward_hooks), 0)

    def test_shared_screen_anchor_and_single_unweighted_post_fedavg_commit(self):
        for positive in (True, False):
            with self.subTest(positive=positive), tempfile.TemporaryDirectory() as temp:
                transfer = tiny_transfer(temp)
                start = copy.deepcopy(transfer.model.state_dict())
                ordinary = {A_KEY: start[A_KEY], B_KEY: start[B_KEY]+1.}
                originals = copy.deepcopy(transfer.original_deltas)
                measured_anchors = []

                def score(client, positions, diagnostic):
                    b = transfer.model.layer.w_lora_B.detach().clone()
                    measured_anchors.append(b)
                    return {0: {'la': 100-float(b.sum()) if positive else 100.}}

                def measure(*unused):
                    loss = 100-float(transfer.model.layer.w_lora_B.detach().sum())
                    return dict(tail_la=loss, non_tail_la=loss, tail_accuracy=0., non_tail_accuracy=0.)

                def calibrate(donors, manifests, anchor):
                    self.assertEqual(donors, [0])
                    torch.testing.assert_close(transfer.model.layer.w_lora_B, ordinary[B_KEY])
                    for k in transfer.recipients:
                        self.assertEqual(manifests[k]['batches'], calibration_batches(transfer.groups[k], {0}, 42, 30, k))
                    return {B_KEY: .5*torch.eye(2).unsqueeze(0)}

                transfer.score_positions, transfer.measure_client = score, measure
                transfer.calibrate_shared = calibrate
                with patch('builtins.print'):
                    result = transfer.apply_shared(start, ordinary, 30)
                residual = .5*originals[0][B_KEY] if positive else torch.zeros_like(start[B_KEY])
                torch.testing.assert_close(result[B_KEY], ordinary[B_KEY]+residual)
                self.assertTrue(torch.equal(result[A_KEY], start[A_KEY]))
                self.assertTrue(torch.equal(ordinary[B_KEY], start[B_KEY]+1.))
                torch.testing.assert_close(measured_anchors[0], ordinary[B_KEY])
                self.assertEqual(transfer.pending['summary']['unique_donors'], int(positive))
                self.assertTrue((Path(temp)/'b_transfer_rounds/r030/commit.pt').is_file())
                transfer.observe_global(result, 30, committed=True)
                self.assertEqual(len(transfer.records), 1)
                self.assertNotIn('local_tail_la_gain_mean', transfer.records[0])
                self.assertIn('shared_receiver_tail_la_gain_mean', transfer.records[0])
                transfer.flush()

    def test_actual_runtime_commits_after_raw_fedavg_and_leaves_a_phase_untouched(self):
        with tempfile.TemporaryDirectory() as temp:
            runtime = object.__new__(SFRARuntime)
            runtime.sfra_config = {'b_transfer': {'mode': 'shared'}}
            runtime.q = {k: (k+1)/465 for k in range(30)}
            model = torch.nn.Module()
            model.register_parameter('w_lora_A', torch.nn.Parameter(torch.ones(1, 1)))
            model.register_parameter('w_lora_B', torch.nn.Parameter(torch.zeros(1, 1)))
            runtime.trainer = SimpleNamespace(model=model, _writer=None, reset_optimizer_and_scheduler=lambda: None)
            runtime.root, runtime.args = Path(temp), SimpleNamespace(seed=42)
            runtime.a_keys, runtime.b_keys = ['w_lora_A'], ['w_lora_B']
            runtime.keys = runtime.a_keys+runtime.b_keys
            runtime.sizes, runtime.budget, runtime.events = list(range(1, 31)), [], []
            runtime.schedule = [list(range(30)) for _ in range(100)]
            captured = {}

            def normal_train(trainer, client, *unused):
                with torch.no_grad():
                    trainer.model.w_lora_B.fill_(client+1)
                return None, None, 3, 3

            def cache(deltas, selected):
                captured['deltas'] = deltas

            def apply(start, ordinary, rnd):
                expected = sum(runtime.q[k]*(k+1) for k in range(30))
                self.assertAlmostEqual(float(ordinary['w_lora_B']), expected, places=5)
                self.assertEqual(float(captured['deltas'][0]['w_lora_B']), 1.)
                return {**ordinary, 'w_lora_B': ordinary['w_lora_B']+2.}

            def audit(before, after, local, selected, weights, rnd, factor):
                expected = sum(weights[k]*(k+1) for k in selected)
                self.assertAlmostEqual(float(after['w_lora_B']), expected, places=5)
                event_id = runtime.audit.event_context['event_id']
                folder = runtime.root/'events'/event_id
                folder.mkdir(parents=True)
                (folder/'event.json').write_text(json.dumps(dict(event_id=event_id)))

            runtime.normal_train, runtime.audit = normal_train, SimpleNamespace(normal=audit)
            runtime.b_transfer = SimpleNamespace(scheduled=lambda r: r == 30, cache_updates=cache, apply_shared=apply)
            state = copy.deepcopy(model.state_dict())
            with patch('builtins.print'):
                ordinary = runtime.train_phase(state, 29)
                shared, deltas = runtime.train_phase(state, 30, return_deltas=True)
            torch.testing.assert_close(shared['w_lora_B'], ordinary['w_lora_B']+2.)
            torch.testing.assert_close(model.w_lora_B, shared['w_lora_B'])
            self.assertEqual(float(deltas[0]['w_lora_B']), 1.)
            self.assertTrue(torch.equal(shared['w_lora_A'], state['w_lora_A']))
            self.assertEqual(runtime.events[-1]['state_role'], 'ordinary_B_before_shared_transfer')
            self.assertIs(runtime.phase_aggregation_weights(list(range(30)), 30, 'A', True), runtime.q)

    def test_launcher_isolation_saved_resume_and_legacy_config(self):
        with tempfile.TemporaryDirectory() as temp:
            def prepare(args, run):
                (run/'protocol').mkdir(parents=True)
                return run/'protocol/full_schedule.json', ''

            commands = []
            for mode in ('local', 'shared'):
                argv = ['entry.py', '--output-root', temp, '--fast-execution']
                with patch('sys.argv', argv), patch.object(launcher, 'prepare_protocol', prepare), \
                        patch.object(launcher.subprocess, 'run') as execute, patch('builtins.print'):
                    launcher.main(default_b_transfer=True, default_transfer_mode=mode)
                    command = execute.call_args.args[0]
                commands.append(command)
                self.assertEqual('--b_transfer_mode' in command, mode == 'shared')
                self.assertEqual(command[command.index('--b_transfer_lr')+1], '0.3' if mode == 'shared' else '0.1')
                run = Path(command[command.index('--output-dir')+1])
                self.assertEqual('_bshared_' in run.name, mode == 'shared')
                (run/'checkpoints').mkdir()
                (run/'checkpoints/sfra_last.pt').touch()
                with patch('sys.argv', argv+['--resume']), patch.object(launcher.subprocess, 'run') as execute, patch('builtins.print'):
                    launcher.main(default_b_transfer=True, default_transfer_mode=mode)
                    resumed = execute.call_args.args[0]
                self.assertEqual('--b_transfer_mode' in resumed, mode == 'shared')
            self.assertNotEqual(commands[0][commands[0].index('--output-dir')+1],
                                commands[1][commands[1].index('--output-dir')+1])
            args = SimpleNamespace(b_transfer_probe_step=.1, b_transfer_lr=.3, b_transfer_reg=.001)
            self.assertEqual(transfer_config(args)['schema_version'], 'donor_b_v1')
            self.assertNotIn('mode', transfer_config(args))

    def test_shared_summary_and_pack_label_mode_correctly(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)/'shared'
            run = root/'seed42/client-longtail/full-cp/shared_setting'
            run.mkdir(parents=True)
            args = SimpleNamespace(b_transfer_probe_step=.1, b_transfer_lr=.3, b_transfer_reg=.001)
            (run/'sfra_config.json').write_text(json.dumps(dict(variant='full-cp', seed=42,
                partition='client-longtail', retention_weight=10, classification_weight=1,
                b_transfer=shared_transfer_config(args))))
            (run/'completion.json').write_text('{}')
            write_csv(run/'round_metrics.csv', [dict(round=r, **{k:float(r) for k in METRICS}) for r in range(101)])
            for name in ('sfra_rounds', 'sfra_costs'):
                write_csv(run/f'{name}.csv', [])
            write_csv(run/'b_transfer_rounds.csv', [dict(round=30, optimizer_steps=2, client_backward_batches=44)])
            with patch('builtins.print'):
                summarize(root)
                pack(root)
            row = read_csv(root/'analysis/performance.csv')[0]
            self.assertEqual(row['method'], 'full-cp+B-shared')
            self.assertEqual(row['transfer_mode'], 'shared')
            self.assertTrue((Path(temp)/'shared_analysis.tar.gz').is_file())


if __name__ == '__main__':
    unittest.main()
