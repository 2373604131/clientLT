"""No data/GPU/downloads: CP ablations, resumable launcher and paired summaries."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
import test_sfra_v1 as fixtures
from utils.sfra_math import make_targets, FunctionalHistory
from utils.cliplora_sfra import SFRARuntime
from scripts import run_method_a_maintext as suite
from tools.sfra import maintext
from tools.sfra.summary import METRICS, write_csv, read_csv


def args_for(root, seeds=(42,)):
    return SimpleNamespace(output_root=Path(root), seeds=list(seeds), protocol_seed=42,
                           partition='client-longtail', dirichlet_beta=.5, witness_batch_size=8,
                           num_workers=0, data_root=Path(root)/'data', reference_run=None, schedule_file=None)


def completed_fixture(root, job, offset=0.):
    run = Path(root)/job['run']
    run.mkdir(parents=True, exist_ok=True)
    cfg = {**maintext.FROZEN, 'variant': job['method'], 'seed': job['seed'],
           'protocol_seed': job['protocol_seed'], 'partition': job['partition'],
           'witness_batch_size': job['witness_batch_size'], 'a_parameter_order': ['layer.A'],
           'base_training_config': dict(seed=job['seed'], method='s', dirichlet_beta=.5,
                                        normal_steps_expected=300, extra_steps_expected=90)}
    if job['method'].endswith('-cp'):
        cfg['classification_weight'] = 1.
    suite.write_json(run/'sfra_config.json', cfg)
    suite.write_json(run/'completion.json', dict(completed_round=100, variant=job['method'], official_test_passes=101,
                     elapsed_seconds=123., normal_optimizer_steps=300, extra_optimizer_steps=90,
                     functional_correction_steps=0 if job['method'] == 's' else 270))
    suite.write_json(run/'maintext_run.json', dict(schema_version=maintext.SCHEMA, job=job, code_unchanged=True))
    # Deliberately reverse the file: the final metric is round 100, not the last CSV row.
    write_csv(run/'round_metrics.csv', [{'round': r, **{m: 50+offset+r/100 for m in METRICS}}
                                      for r in reversed(range(101))])
    write_csv(run/'budget.csv', [dict(upload_bytes=2**20, modeled_downlink_bytes=2*2**20)])
    write_csv(run/'sfra_costs.csv', [dict(seconds=10, upload_bytes=2**20, downlink_bytes=3*2**20)])
    write_csv(run/'evaluation_budget.csv', [dict(upload_bytes=0, modeled_downlink_bytes=0)])
    round_rows = []
    if job['method'] != 's':
        for r in range(1, 101):
            row = dict(round=r)
            if job['method'].endswith('-cp') and r <= 90:
                row['committed_classification_loss_increase'] = .001 if r%2 else -.001
            round_rows.append(row)
    write_csv(run/'sfra_rounds.csv', round_rows)
    suite.write_json(run/'bridge_metadata.json', {name: name+str(job['seed']) for name in maintext.PAIR_HASHES})
    write_csv(run/'partition_manifest.csv', [dict(client_id=0, local_position=0, class_id=0, raw_sample_id=17)])
    if job['method'] != 's':
        suite.write_json(run/'private_witness_manifest.json', [dict(token_id=0, client_id=0, raw_sample_ids=[17])])
    return run


class TestCPMaintextAblations(unittest.TestCase):
    def test_each_target_ablation_changes_only_its_intended_factor(self):
        scores = torch.tensor([[-.3,-.2], [-.5,-.4], [-.1,-.1]])
        source = dict(supported=torch.tensor([True, False, False]), u=torch.tensor([.2, 0., 0.]),
                      cache=torch.tensor([.9, .3, .8]))
        history, valid = torch.tensor([.1, .2, 0.]), torch.tensor([True, True, False])
        original = copy.deepcopy(source)
        full = make_targets(scores, source, history, valid, 'full-cp')
        flat = make_targets(scores, source, history, valid, 'flat-cp')
        current = make_targets(scores, source, history, valid, 'current-cp')
        for name in ('current', 'target', 'active'):
            torch.testing.assert_close(full[name], flat[name])
        torch.testing.assert_close(flat['weights'], torch.tensor([.5, .5, 0.]))
        torch.testing.assert_close(full['weights'], torch.tensor([.75, .25, 0.]))
        torch.testing.assert_close(current['target'][0], torch.tensor([-.2, -.1]))
        self.assertEqual(current['active'].tolist(), [True, False, False])
        for variant in ('full', 'flat', 'current'):
            plain = make_targets(scores, source, history, valid, variant)
            cp = make_targets(scores, source, history, valid, variant+'-cp')
            for key in plain:
                torch.testing.assert_close(plain[key], cp[key])
        for key in source:
            torch.testing.assert_close(source[key], original[key])

    def test_new_variants_keep_cp_and_freeze_b_in_three_actual_gradient_steps(self):
        for variant in ('flat-cp', 'current-cp'):
            with self.subTest(variant=variant), tempfile.TemporaryDirectory() as temp:
                bank = fixtures.TestSFRAClassification().bank()
                runtime = object.__new__(SFRARuntime)
                runtime.trainer = SimpleNamespace(model=bank.model, device=bank.device)
                runtime.bank, runtime.a_keys = bank, bank.keys
                runtime.b_keys = ['image_encoder.proj_lora_B']
                runtime.keys = sorted(runtime.a_keys+runtime.b_keys)
                runtime.variant, runtime.strength, runtime.scaling = variant, 10., .5
                runtime.classification_strength = 1.
                runtime.costs, runtime.events, runtime.q = [], [{}], {0:2/3, 1:1/3}
                middle = copy.deepcopy(bank.model.state_dict())
                runtime.history = FunctionalHistory(bank.evaluate()['scores'], 2)
                runtime.history.valid[:] = True
                runtime.history.level[:] = 1.5
                delta = torch.randn_like(middle[bank.keys[0]])*.01
                ordinary = copy.deepcopy(middle)
                ordinary[bank.keys[0]] += delta/3
                runtime.train_phase = lambda *a, **kw: (ordinary, {0:{bank.keys[0]:delta}, 1:{bank.keys[0]:-delta}})
                with patch('builtins.print'):
                    committed, _, arrays, summary = runtime.refresh(middle, 1, Path(temp))
                steps = read_csv(Path(temp)/'correction_steps.csv')
                self.assertEqual(len(steps), 3)
                self.assertEqual(float(steps[0]['classification_penalty']), 0.)
                self.assertEqual(summary['classification_weight'], 1.)
                self.assertIn('committed_classification_loss_increase', summary)
                self.assertTrue(torch.equal(committed[runtime.b_keys[0]], middle[runtime.b_keys[0]]))
                if variant == 'flat-cp':
                    torch.testing.assert_close(arrays['weights'], torch.ones(3)/3)
                else:
                    torch.testing.assert_close(arrays['active'], arrays['supported'])

    def test_cp_launcher_keeps_mu_in_paths_for_both_new_variants(self):
        from scripts import run_cliplora_sfra as launcher
        with tempfile.TemporaryDirectory() as temp:
            def prepare(args, run):
                (run/'protocol').mkdir(parents=True)
                return run/'protocol/full_schedule.json', ''
            for method in ('flat-cp', 'current-cp'):
                argv = ['run_cliplora_sfra.py', '--method', method, '--output-root', temp]
                with patch('sys.argv', argv), patch.object(launcher, 'prepare_protocol', prepare), \
                        patch.object(launcher.subprocess, 'run') as execute, patch('builtins.print'):
                    launcher.main()
                command = execute.call_args.args[0]
                self.assertEqual(command[command.index('--sfra_variant')+1], method)
                self.assertIn('_mu1_', command[command.index('--output-dir')+1])
                self.assertEqual(command[command.index('--sfra_classification_weight')+1], '1.0')


class TestMaintextSuite(unittest.TestCase):
    def test_train_entry_dispatches_all_five_and_writes_paired_report(self):
        with tempfile.TemporaryDirectory() as temp:
            jobs = suite.plan_jobs(args_for(temp))
            def fake_train(command, **kwargs):
                method = command[command.index('--method')+1]
                job = next(j for j in jobs if j['method'] == method)
                completed_fixture(temp, job)
            argv = ['run_method_a_maintext.py', '--stage', 'train', '--data-root', str(Path(temp)/'data'),
                    '--output-root', temp, '--num-workers', '0']
            with patch('sys.argv', argv), patch.object(suite.subprocess, 'run', side_effect=fake_train) as execute, \
                    patch('builtins.print'):
                suite.main()
            self.assertEqual(execute.call_count, 5)
            report = Path(temp)/'maintext_analysis'
            self.assertEqual(len(read_csv(report/'main_table.csv')), 2)
            self.assertEqual(len(read_csv(report/'ablation_table.csv')), 4)

    def test_plan_is_paired_and_locks_changed_config(self):
        with tempfile.TemporaryDirectory() as temp:
            jobs = suite.plan_jobs(args_for(temp, (42, 43)))
            self.assertEqual(len(jobs), 10)
            self.assertEqual(len({j['run'] for j in jobs}), 10)
            for job in jobs:
                command = suite.command_for(job, temp)
                self.assertNotIn('--b-transfer', command)
                self.assertIn('--fresh-protocol', command)
                self.assertEqual(command[command.index('--protocol-seed')+1], '42')
            suite.merge_plan(temp, jobs)
            suite.merge_plan(temp, jobs)
            changed = copy.deepcopy(jobs)
            changed[0]['code_sha256']['utils/sfra_math.py'] = 'changed'
            with self.assertRaisesRegex(ValueError, 'changed'):
                suite.merge_plan(temp, changed)
            self.assertEqual(maintext.load_json(Path(temp)/maintext.PLAN_NAME)['jobs'], jobs)

    def test_resume_and_skip_are_explicit_and_do_not_restart_partial_runs(self):
        with tempfile.TemporaryDirectory() as temp:
            job = suite.plan_jobs(args_for(temp))[1]
            run = completed_fixture(temp, job)
            with patch.object(suite.subprocess, 'run') as execute, patch('builtins.print'):
                suite.execute_job(job, temp)
                execute.assert_not_called()
            (run/'completion.json').unlink()
            (run/'checkpoints').mkdir()
            (run/'checkpoints/sfra_last.pt').touch()
            with self.assertRaisesRegex(ValueError, '--resume'):
                suite.execute_job(job, temp)
            def fake_train(command, **kwargs):
                self.assertIn('--resume', command)
                completed_fixture(temp, job)
            with patch.object(suite.subprocess, 'run', side_effect=fake_train), patch('builtins.print'):
                suite.execute_job(job, temp, resume=True)
            self.assertTrue(maintext.load_json(run/'maintext_run.json')['code_unchanged'])

    def test_hard_killed_launcher_recovers_only_a_registered_run(self):
        with tempfile.TemporaryDirectory() as temp, patch('builtins.print'):
            job = suite.plan_jobs(args_for(temp))[1]
            run = completed_fixture(temp, job)
            (run/'maintext_run.json').unlink()
            with self.assertRaises(ValueError):
                suite.execute_job(job, temp)
            with self.assertRaises(FileNotFoundError):
                suite.execute_job(job, temp, resume=True)
            suite.write_json(suite.start_record_path(temp, job), dict(schema_version=maintext.SCHEMA, job=job))
            with patch.object(suite.subprocess, 'run') as execute:
                suite.execute_job(job, temp, resume=True)
                execute.assert_not_called()
            maintext.check_receipt(run, job)

    def test_paired_results_costs_sd_and_no_best_checkpoint_selection(self):
        with tempfile.TemporaryDirectory() as temp, patch('builtins.print'):
            jobs = suite.plan_jobs(args_for(temp, (42, 43)))
            suite.merge_plan(temp, jobs)
            for job in jobs:
                completed_fixture(temp, job, offset=(job['seed']-42)*2+(1 if job['method']=='full-cp' else 0))
            states = maintext.summarize_maintext(temp)
            self.assertTrue(all(r['status']=='complete' for r in states.values()))
            output = Path(temp)/'maintext_analysis'
            table = read_csv(output/'all_methods.csv')
            full = next(r for r in table if r['method']=='full-cp')
            self.assertEqual(int(full['n']), 2)
            self.assertAlmostEqual(float(full['last20_overall_acc_mean']), 52.905)
            self.assertAlmostEqual(float(full['last20_overall_acc_sd']), 2**.5)
            self.assertEqual(float(full['modeled_downlink_mib_mean']), 5.)
            self.assertEqual(float(full['modeled_upload_mib_mean']), 2.)
            self.assertEqual(float(full['tail_peak_to_final_mean']), 0.)
            paired = read_csv(output/'paired_differences.csv')
            self.assertEqual(len(paired), 4)
            self.assertTrue(all(float(r['last20_overall_acc_delta_mean']) == 1. for r in paired))
            cp = next(r for r in read_csv(output/'per_run.csv') if r['method']=='full-cp')
            self.assertEqual(float(cp['cp_loss_increase_rate']), .5)

    def test_incomplete_seed_is_excluded_from_all_aggregate_methods(self):
        with tempfile.TemporaryDirectory() as temp, patch('builtins.print'):
            jobs = suite.plan_jobs(args_for(temp, (42, 43)))
            suite.merge_plan(temp, jobs)
            for job in jobs[:-1]:
                completed_fixture(temp, job)
            maintext.summarize_maintext(temp)
            table = read_csv(Path(temp)/'maintext_analysis/all_methods.csv')
            self.assertTrue(all(r['n']=='1' and r['seeds']=='42' for r in table))
            self.assertTrue(all(r['last20_overall_acc_sd']=='' for r in table))

    def test_witness_mismatch_and_duplicate_rounds_fail_closed(self):
        for corruption in ('witness', 'rounds'):
            with self.subTest(corruption=corruption), tempfile.TemporaryDirectory() as temp, patch('builtins.print'):
                jobs = suite.plan_jobs(args_for(temp))
                suite.merge_plan(temp, jobs)
                runs = [completed_fixture(temp, job) for job in jobs]
                if corruption == 'witness':
                    suite.write_json(runs[2]/'private_witness_manifest.json', [{'raw_sample_ids':[99]}])
                else:
                    rows = read_csv(runs[2]/'round_metrics.csv')
                    rows[-1]['round'] = 1
                    write_csv(runs[2]/'round_metrics.csv', rows)
                states = maintext.summarize_maintext(temp)
                self.assertTrue(any(r['status'].startswith('invalid') for r in states.values()))
                self.assertEqual(read_csv(Path(temp)/'maintext_analysis/all_methods.csv'), [])


if __name__ == '__main__':
    unittest.main()
