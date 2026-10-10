"""Frozen-A controls, six-slot audits, GPU dispatch and read-only reporting."""
import copy
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

import numpy as np

from scripts import run_cliplora_frozen_aggregation as suite
from scripts import run_cliplora_joint_aggregation as single
from scripts.run_ab_validation import write_json
from tools.client_aggregation.analysis import audit_run
from tools.client_aggregation.frozen_controls import summarize
from tools.client_aggregation.frozen_tailrw import paired_audit
from tools.client_aggregation.protocol import (aggregation_spec, check_aggregation, client_execution_config,
    digest, load_json, PAIR_HASHES)
import test_joint_aggregation as fixtures

REPO = Path(__file__).resolve().parents[1]


class TestFrozenControls(unittest.TestCase):
    def test_fedavg_is_exact_sample_counts_and_rejects_tampering(self):
        with patch('tools.client_aggregation.protocol.solve_weights', side_effect=AssertionError('no solver')):
            spec = aggregation_spec(REPO/'references/full10_clientlt', .1, 'fedavg')
            np.testing.assert_allclose(spec['weights'], np.array(spec['counts']['sizes'])/10847, atol=0, rtol=0)
            check_aggregation(spec)
        broken = copy.deepcopy(spec)
        broken['weights'][0] += .001; broken['weights'][1] -= .001
        broken['weights_sha256'] = digest(broken['weights'])
        with self.assertRaisesRegex(ValueError, 'n_j/N'):
            check_aggregation(broken)

    def test_two_plans_share_protocol_and_budget_but_not_weights(self):
        with tempfile.TemporaryDirectory() as temp:
            args = suite.parse_args(['--output-root', temp])
            self.assertEqual(args.gpus, [2, 3]); self.assertEqual(args.client_concurrency, 6)
            plans = suite.register(args)
            self.assertEqual(plans, suite.register(args))
            self.assertEqual(set(plans), {'fedavg', 'tailrw16'})
            for rule, plan in plans.items():
                self.assertEqual(plan['contract']['optimizer_steps'], 105600)
                self.assertEqual(plan['contract']['extra_A_epochs'], 0)
                self.assertEqual(plan['contract']['extra_B_epochs'], 0)
                self.assertFalse(plan['contract']['functional_correction'])
                self.assertFalse(plan['contract']['source_C_transfer'])
                job = single.make_job(plan, 'frozen')
                cmd = single.command_for(job, Path(temp)/rule)
                self.assertNotIn('--b_transfer_enable', cmd)
                self.assertEqual(cmd[cmd.index('--sfra_variant')+1], 's')
                self.assertEqual(cmd[cmd.index('--cliplora_freeze_a')+1], 'True')
                self.assertEqual(client_execution_config(job)['max_concurrent_clients'], 6)
                with self.assertRaises(ValueError): single.make_job(plan, 'ab')
            a, b = plans['fedavg'], plans['tailrw16']
            for key in ('input_fingerprint', 'partition_sha256', 'matrix_sha256', 'counts'):
                self.assertEqual(a['aggregation'][key], b['aggregation'][key])
            self.assertNotEqual(a['aggregation']['weights'], b['aggregation']['weights'])
            self.assertEqual(a['code_sha256'], b['code_sha256'])
            self.assertEqual(a['settings']['parallel_validation'], 'off')
            self.assertNotIn('cuda_policy', a['settings'])
            args.client_concurrency = 4
            with self.assertRaisesRegex(ValueError, 'changed'): suite.register(args)

    def test_gpu_dispatch_and_smoke_failure_stops_only_its_formal_run(self):
        with tempfile.TemporaryDirectory() as temp, patch('builtins.print'):
            args = suite.parse_args(['--stage', 'server', '--output-root', temp])
            calls, lock = [], threading.Lock()
            def run(cmd, **kwargs):
                rule = cmd[cmd.index('--aggregation-rule')+1]
                stage = cmd[cmd.index('--stage')+1]
                with lock: calls.append((rule, stage, kwargs['env'], cmd))
                return type('Result', (), dict(returncode=1 if rule == 'tailrw16' else 0))()
            with patch.object(suite.subprocess, 'run', side_effect=run), self.assertRaisesRegex(ValueError, 'failed'):
                suite.launch(args)
            for rule, gpu in (('tailrw16', '2'), ('fedavg', '3')):
                selected = [r for r in calls if r[0] == rule]
                self.assertEqual([r[1] for r in selected], ['smoke'] if rule == 'tailrw16' else ['smoke', 'run'])
                self.assertTrue(all(r[2]['CUDA_VISIBLE_DEVICES'] == gpu for r in selected))
                self.assertTrue(all(r[3][r[3].index('--client-concurrency')+1] == '6' for r in selected))
            args.methods = ['fedavg']; args.stage = 'smoke'
            self.assertEqual([cmd[cmd.index('--stage')+1] for cmd in suite.commands(args, 'fedavg')], ['smoke'])

    def test_six_slot_artifact_audit_and_bad_slot_rejection(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            job = fixtures.TestJointAggregation().audit_fixture(root, 'frozen', 'fedavg')
            job['settings'] = dict(client_concurrency=6)
            write_json(root/'joint_job.json', job)
            receipt = load_json(root/'joint_receipt.json'); receipt['job_digest'] = digest(job)
            write_json(root/'joint_receipt.json', receipt)
            cfg = load_json(root/'sfra_config.json'); cfg['client_execution'] = client_execution_config(job)
            write_json(root/'sfra_config.json', cfg); write_json(root/'client_execution.json', cfg['client_execution'])
            execution = dict(round=1, factor='B', slots=6, clients=30, selected_client_ids=list(range(30)),
                client_audits=[dict(client_id=c, slot=c%6) for c in range(30)])
            path = root/'parallel_execution/r001_B.json'
            write_json(path, execution)
            # No parity-pilot file is needed for a successful one-round smoke.
            self.assertEqual(audit_run(root, job, 1)['progress']['extra_optimizer_steps'], 0)
            execution['client_audits'][0]['slot'] = 6
            write_json(path, execution)
            with self.assertRaisesRegex(ValueError, 'execution record'): audit_run(root, job, 1)
            execution['client_audits'][0]['slot'] = 0; write_json(path, execution)
            # Even a retained failure report cannot become a numerical acceptance gate.
            write_json(root/'parallel_benchmark/factor_B.json', dict(passed=False,
                comparisons=[dict(close=False, max_abs=7.840804755687714e-6)]))
            self.assertEqual(audit_run(root, job, 1)['progress']['completed_round'], 1)

    def test_summary_matches_new_controls_and_labels_historical_execution_difference(self):
        with tempfile.TemporaryDirectory() as temp, patch('builtins.print'):
            root = Path(temp)/'controls'; old = Path(temp)/'old_joint'
            results = {}
            for rule in ('fedavg', 'tailrw16', 'joint'):
                folder = old if rule == 'joint' else root/rule
                run = folder/'runs/seed42/frozen'
                job = fixtures.TestJointAggregation().audit_fixture(run, 'frozen', rule)
                r = audit_run(run, job, 1)
                r['metadata'].update({k:'synthetic-identity' for k in PAIR_HASHES})
                r['config']['client_execution'] = client_execution_config({'settings': {'client_concurrency': 4 if rule == 'joint' else 6}})
                r['curves'] = [dict(r['curves'][0], round=i) for i in range(101)]
                r['predictions'] = {i:r['predictions'][0] for i in range(101)}
                r['progress'].update(completed_round=100, elapsed_seconds=1.)
                write_json(run/'completion.json', r['progress'])
                results[rule] = r
            self.assertEqual(paired_audit(results['fedavg'], results['joint'])['status'], 'mismatch')
            with patch('tools.client_aggregation.frozen_controls.load_frozen', side_effect=lambda folder, rule:results[rule]):
                status, pairs = summarize(root, old)
            self.assertEqual([s['status'] for s in status], ['complete']*3)
            self.assertEqual([p['status'] for p in pairs], ['eligible', 'execution_differs', 'execution_differs'])
            self.assertEqual(pairs[0]['delta_pp']['overall_acc'], 0.)
            self.assertFalse((old/'analysis').exists())
            self.assertIn('4并行', (root/'analysis/report.md').read_text(encoding='utf-8'))
            results['fedavg']['job']['code_sha256'] = {'tools/client_aggregation/parallel.py': 'changed'}
            with patch('tools.client_aggregation.frozen_controls.load_frozen', side_effect=lambda folder, rule:results[rule]):
                _, pairs = summarize(root, old)
            self.assertEqual(pairs[0]['status'], 'mismatch')
            self.assertNotIn('delta_pp', pairs[0])


if __name__ == '__main__':
    unittest.main()
