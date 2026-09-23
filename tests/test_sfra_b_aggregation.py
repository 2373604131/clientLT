"""CPU unit checks for the eight-round intervention; no dataset or CLIP runs."""
import copy
import json
import tempfile
import unittest
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from scripts import run_cliplora_sfra as launcher
from tools.sfra.summary import METRICS, read_csv, summarize, write_csv
from utils.b_aggregation import B_TRANSFER_ROUNDS, phase_weights
from utils.cliplora_b_transfer import DonorBTransfer, transfer_config
from utils.cliplora_sfra import SFRARuntime


class TestBAggregation(unittest.TestCase):
    def test_only_eight_normal_b_rounds_change_without_mutating_sample_weights(self):
        sample = {k: (k+1)/465 for k in range(30)}
        original = dict(sample)
        selected = list(reversed(range(30)))
        changed = []
        for rnd in range(1, 101):
            weights = phase_weights(sample, selected, rnd, 'uniform-transfer-rounds')
            if weights != sample:
                changed.append(rnd)
                self.assertEqual(weights, {k: 1/30 for k in selected})
            for factor, extra, branch in [('A', True, 'main'), ('AB', False, 'main'),
                                         ('B', True, 'main'), ('B', False, 'lookahead')]:
                self.assertIs(phase_weights(sample, selected, rnd, 'uniform-transfer-rounds',
                                           factor, extra, branch), sample)
            self.assertIs(phase_weights(sample, selected, rnd), sample)
        self.assertEqual(changed, list(B_TRANSFER_ROUNDS))
        self.assertEqual(sample, original)

    def test_actual_phase_aggregates_whole_b_and_audit_uses_the_same_weights(self):
        with tempfile.TemporaryDirectory() as temp:
            runtime = object.__new__(SFRARuntime)
            model = torch.nn.Module()
            model.register_parameter('w_lora_A', torch.nn.Parameter(torch.ones(1, 1)))
            model.register_parameter('w_lora_B', torch.nn.Parameter(torch.zeros(1, 1)))
            runtime.trainer = SimpleNamespace(model=model, _writer=None, reset_optimizer_and_scheduler=lambda: None)
            runtime.a_keys, runtime.b_keys = ['w_lora_A'], ['w_lora_B']
            runtime.keys = runtime.a_keys + runtime.b_keys
            runtime.root, runtime.args = Path(temp), SimpleNamespace(seed=42)
            runtime.q = {k: (k+1)/465 for k in range(30)}
            runtime.sizes = [k+1 for k in range(30)]
            runtime.schedule = [list(range(30)) for _ in range(100)]
            runtime.budget, runtime.events = [], []
            runtime.sfra_config = {'b_aggregation': {'mode': 'uniform-transfer-rounds'}}
            runtime.b_transfer = None

            def local_train(trainer, client, *unused):
                with torch.no_grad():
                    trainer.model.w_lora_B.fill_(client+1)
                return None, None, 3, 3

            seen = []

            def audit(before, after, local_states, selected, weights, rnd, factor):
                seen.append(dict(weights))
                path = runtime.root/'events'/runtime.audit.event_context['event_id']
                path.mkdir(parents=True)
                (path/'event.json').write_text('{}')

            runtime.normal_train = local_train
            runtime.audit = SimpleNamespace(normal=audit)
            state = copy.deepcopy(model.state_dict())
            with patch('builtins.print'):
                for rnd in (29, 30, 31, 90, 100):
                    result = runtime.train_phase(state, rnd)
                    weights = runtime.phase_aggregation_weights(list(range(30)), rnd, 'B')
                    expected = sum(weights[k]*(k+1) for k in range(30))
                    self.assertAlmostEqual(float(result['w_lora_B']), expected, places=5)
                    self.assertEqual(seen[-1], weights)
                    self.assertTrue(torch.equal(result['w_lora_A'], state['w_lora_A']))
                    self.assertEqual(runtime.phase_aggregation_weights(list(range(30)), rnd, 'A', True), runtime.q)

    def test_transfer_still_calibrates_local_b_but_diagnostics_use_actual_weights(self):
        for mode, ordinary, residual_norm in [('sample', 2.6, .1), ('uniform-transfer-rounds', 5., .5)]:
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as temp:
                cfg_args = SimpleNamespace(b_transfer_probe_step=.1, b_transfer_lr=.3,
                                           b_transfer_reg=.001, sfra_b_aggregation=mode)
                runtime = object.__new__(SFRARuntime)
                runtime.q = {0: .9, 1: .1}
                runtime.root, runtime.args = Path(temp), SimpleNamespace(seed=42)
                runtime.sfra_config = {'b_aggregation': {'mode': mode}}
                transfer = object.__new__(DonorBTransfer)
                transfer.runtime, transfer.config = runtime, transfer_config(cfg_args)
                transfer.keys, transfer.recipients = ['w_lora_B'], [1]
                transfer.tail, transfer.probes = {0}, {1: {0: [0]}}
                transfer.groups, transfer.counts = [{}, {0: [0]}], torch.tensor([[0], [1]])
                current, observed = {}, {}
                transfer.copy_parameters = lambda values: current.update(values)
                transfer.session = nullcontext
                transfer.measure_all = lambda phase: observed.update({phase: float(current['w_lora_B'])})
                transfer.measure_client = lambda *unused: {'tail_la': 10-float(current['w_lora_B'])}
                transfer.score_positions = lambda *unused: {0: {'la': 10-float(current['w_lora_B'])}}
                transfer.effective_norm = lambda values, start: float(values['w_lora_B'].norm())
                transfer.write_pending = lambda: None

                def calibrate(client, donors, deltas, batches, start):
                    self.assertEqual(float(current['w_lora_B']), 8.)
                    self.assertEqual(donors, [0])
                    self.assertEqual(len(batches), 2)
                    return {'w_lora_B': torch.full((1, 1, 1), .5)}

                transfer.calibrate = calibrate
                start = {'w_lora_B': torch.zeros(1, 1)}
                local = {0: {'w_lora_B': torch.tensor([[2.]])}, 1: {'w_lora_B': torch.tensor([[8.]])}}
                deltas = copy.deepcopy(local)
                with patch('builtins.print'):
                    new, _ = transfer.apply(start, local, deltas, [0, 1], 30)
                self.assertAlmostEqual(observed['ordinary_global_B'], ordinary, places=6)
                self.assertAlmostEqual(transfer.pending['summary']['effective_global_transfer_norm'], residual_norm, places=6)
                self.assertEqual(float(new[1]['w_lora_B']), 9.)
                self.assertEqual(float(local[1]['w_lora_B']), 8.)
                row = transfer.pending['receivers'][0]
                self.assertEqual(row['sample_weight'], .1)
                self.assertEqual(row['aggregation_weight'], .5 if mode != 'sample' else .1)

    def test_launch_two_separate_runs_and_resume_saved_flags(self):
        with tempfile.TemporaryDirectory() as temp:
            def prepare(args, run):
                (run/'protocol').mkdir(parents=True)
                return run/'protocol/full_schedule.json', ''

            runs = []
            for transfer in (False, True):
                argv = ['run_cliplora_sfra.py', '--method', 'full-cp', '--output-root', temp,
                        '--b-aggregation', 'uniform-transfer-rounds']
                if transfer:
                    argv += ['--b-transfer', '--transfer-lr', '0.3']
                with patch('sys.argv', argv), patch.object(launcher, 'prepare_protocol', prepare), \
                        patch.object(launcher.subprocess, 'run') as execute, patch('builtins.print'):
                    launcher.main()
                    command = execute.call_args.args[0]
                self.assertEqual(command[command.index('--sfra_b_aggregation')+1], 'uniform-transfer-rounds')
                self.assertEqual('--b_transfer_enable' in command, transfer)
                self.assertEqual(command[command.index('--sfra_classification_weight')+1], '1.0')
                self.assertLess(command.index('--sfra_b_aggregation'), command.index('DATALOADER.NUM_WORKERS'))
                run = Path(command[command.index('--output-dir')+1])
                self.assertTrue(run.name.endswith('_bagg_uniform8'))
                runs.append(run)
                (run/'checkpoints').mkdir()
                (run/'checkpoints/sfra_last.pt').touch()
                with patch('sys.argv', argv+['--resume']), patch.object(launcher.subprocess, 'run') as execute, patch('builtins.print'):
                    launcher.main()
                    resumed = execute.call_args.args[0]
                self.assertEqual(resumed[resumed.index('--sfra_b_aggregation')+1], 'uniform-transfer-rounds')
                self.assertEqual('--b_transfer_enable' in resumed, transfer)
            self.assertNotEqual(*runs)

    def test_summary_identifies_new_rule_and_exports_actual_a_and_b_weights(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            run = root/'seed42/client-longtail/full-cp/lambda10_mu1_protocol42_bagg_uniform8'
            run.mkdir(parents=True)
            (run/'sfra_config.json').write_text(json.dumps(dict(variant='full-cp', seed=42,
                partition='client-longtail', retention_weight=10, classification_weight=1,
                b_aggregation=dict(mode='uniform-transfer-rounds'))))
            (run/'completion.json').write_text('{}')
            (run/'bridge_metadata.json').write_text(json.dumps({'client_sample_counts': [9, 1]}))
            write_csv(run/'round_metrics.csv', [dict(round=r, **{k:float(r) for k in METRICS}) for r in range(101)])
            write_csv(run/'sfra_rounds.csv', [])
            write_csv(run/'sfra_costs.csv', [])
            write_csv(run/'event_manifest.csv', [dict(round=30, phase='normal_B', selected_client_ids=[0, 1], server_weights=[.5, .5]),
                                               dict(round=30, phase='refresh_A', selected_client_ids=[0, 1], server_weights=[.9, .1])])
            with patch('builtins.print'):
                summarize(root)
            perf = read_csv(root/'analysis/performance.csv')[0]
            self.assertEqual(perf['method'], 'full-cp+uniform8')
            self.assertEqual(perf['b_transfer_enabled'], 'False')
            self.assertEqual(perf['b_aggregation'], 'uniform-transfer-rounds')
            rows = read_csv(root/'analysis/aggregation_weights.csv')
            self.assertEqual([float(r['aggregation_weight']) for r in rows], [.5, .5, .9, .1])


if __name__ == '__main__':
    unittest.main()
