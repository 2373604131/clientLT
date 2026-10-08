"""Exercise replay, paired collection, source guards and real sparse optimization."""
import copy
import io
from contextlib import redirect_stdout
from pathlib import Path
import tempfile
import subprocess
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from scripts import run_cliplora_b_routes as launcher
from scripts.run_ab_validation import write_json
from tools.sfra import b_routes as report
from tools.sfra.maintext import digest, load_json
from tools.sfra.summary import read_csv, write_csv
from utils.b_route_replay import completed_branch, legacy_optimize, replay_runtime_class
from utils.b_route_sampling import image_identity, make_folds
from utils.cliplora_b_routes import RouteTransfer, route_config
from utils.cliplora_b_shared_transfer import SharedDonorBTransfer
from utils.cliplora_a_refresh import state_hash
from test_sfra_b_shared_transfer import tiny_transfer, A_KEY, B_KEY
from test_sfra_ab_validation import completed, args as ab_args
from scripts.run_ab_validation import plan_jobs


def tiny_route(folder):
    transfer = tiny_transfer(folder)
    transfer.__class__ = RouteTransfer
    transfer.bank = SimpleNamespace(batch_size=2)
    transfer.monitors = {1: dict(tail_positions=[0], non_tail_positions=[1, 2]),
                         2: dict(tail_positions=[0, 1], non_tail_positions=[])}
    transfer.config = route_config(transfer.config, dict(arm='C'))
    return transfer


class WorkflowTest(unittest.TestCase):
    def test_parallel_processes_keep_every_registered_job(self):
        # Exercise OS locks across processes, including Windows spawn behavior.
        script = '''
from pathlib import Path
import sys, time
from scripts import run_cliplora_b_routes as launcher
root, index = Path(sys.argv[1]), sys.argv[2]
original = launcher.load_json
def slow_read(path):
    value = original(path)
    time.sleep(.05)
    return value
launcher.load_json = slow_read
launcher.merge_plan(root, [(root/('job'+index), {'index': index}, [])])
'''
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            launcher.merge_plan(root, [(root/'initial', dict(index='initial'), [])])
            processes = []
            try:
                for index in range(4):
                    processes.append(subprocess.Popen([sys.executable, '-c', script, temp, str(index)],
                        cwd=launcher.REPO, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True))
                for process in processes:
                    stdout, stderr = process.communicate(timeout=30)
                    self.assertEqual(process.returncode, 0, stdout+stderr)
            finally:
                for process in processes:
                    if process.poll() is None:
                        process.kill()
                        process.communicate()
            registered = load_json(root/'route_plan.json')['jobs']
            self.assertEqual({j['spec']['index'] for j in registered}, {'initial', '0', '1', '2', '3'})

    def test_preflight_keeps_separate_records_for_gpu_shards(self):
        with tempfile.TemporaryDirectory() as temp:
            args = launcher.parser().parse_args(['--output-root', temp, '--arms', 'P'])
            launcher.write_preflight(args, dict(ready=True, mode='full'))
            args.arms = ['C']
            launcher.write_preflight(args, dict(ready=True, mode='full'))
            records = [load_json(p) for p in (Path(temp)/'preflights').glob('*.json')]
            self.assertEqual({r['request']['arms'][0] for r in records}, {'P', 'C'})
            self.assertEqual(load_json(Path(temp)/'preflight.json')['request']['arms'], ['C'])

    def test_saved_event_loader_rejects_wrong_weights_and_changed_a(self):
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp)
            spec = dict(mode='replay', round=30, source_run=str(source))
            runtime = object.__new__(replay_runtime_class(spec))
            t = tiny_route(temp)
            base = copy.deepcopy(t.model.state_dict())
            runtime.source, runtime.keys = source, [A_KEY, B_KEY]
            runtime.a_keys, runtime.b_keys = [A_KEY], [B_KEY]
            runtime.trainer = runtime.global_trainer = SimpleNamespace(model=t.model)
            counts = torch.ones(30, 2, dtype=torch.long)
            runtime.audit = SimpleNamespace(counts=counts)
            runtime.source_meta = dict(initial_lora_sha256=state_hash(base, runtime.keys),
                                       frozen_model_sha256=state_hash(base, []))
            (source/'checkpoints').mkdir()
            torch.save(base, source/'checkpoints/base_model.pt')
            folder = source/'events/r030_c000_main_normal_B'
            folder.mkdir(parents=True)
            deltas = [{B_KEY: torch.full_like(base[B_KEY], .03)} for _ in range(30)]
            ordinary = copy.deepcopy(base)
            ordinary[B_KEY] += .03
            payload = dict(round=30, phase='normal_B', anchor_lora_state=base, actual_after_lora_state=ordinary,
                selected_client_ids=list(range(30)), local_factor_deltas=deltas, server_weights=[1/30]*30, client_class_counts=counts)
            def save():
                torch.save(payload, folder/'state.pt')
                write_json(folder/'event.json', dict(before_lora_sha256=state_hash(base, runtime.keys),
                                                    after_lora_sha256=state_hash(ordinary, runtime.keys)))
            save()
            _, loaded, _, donors = runtime.load_event()
            self.assertEqual(donors, list(range(30)))
            torch.testing.assert_close(loaded[B_KEY], ordinary[B_KEY])
            payload['server_weights'][0] = .5
            save()
            with self.assertRaises(AssertionError):
                runtime.load_event()
            payload['server_weights'][0] = 1/30
            ordinary[A_KEY] += .01
            save()
            with self.assertRaisesRegex(ValueError, 'changed A'):
                runtime.load_event()

    def test_image_folds_disjoint_across_clients_and_stable_across_events(self):
        groups = [{0: [0, 1], 1: [2]}, {0: [0, 1], 1: [2, 3]}]
        rows = [dict(client_id=k, local_position=p, class_id=c, raw_sample_id=raw)
                for k, p, c, raw in [(0, 0, 0, 10), (0, 1, 0, 11), (0, 2, 1, 20),
                                      (1, 0, 0, 10), (1, 1, 0, 12), (1, 2, 1, 21), (1, 3, 1, 22)]]
        identity = image_identity(rows)
        rng = np.random.get_state()
        manifests = []
        for rnd in (30, 60):
            for fold in (0, 1):
                fit, diagnostic, _, manifest = make_folds(groups, identity, [0, 1], {0}, 42, rnd, fold)
                training = {identity[k, p] for k, g in enumerate(fit) for ps in g.values() for p in ps}
                held = {identity[k, p] for k, ps in diagnostic.items() for p in ps}
                self.assertFalse(training & held)
                self.assertEqual(training | held, {10, 11, 12, 20, 21, 22})
                manifests.append(manifest['images'])
        self.assertTrue(all(x == manifests[0] for x in manifests))
        np.testing.assert_equal(rng, np.random.get_state())

    def test_one_shared_commit_and_after_a_accounting(self):
        with tempfile.TemporaryDirectory() as temp:
            t = tiny_route(temp)
            start = copy.deepcopy(t.model.state_dict())
            ordinary = copy.deepcopy(start)
            ordinary[B_KEY] += .1
            ordinary['frozen_backbone_fixture'] = torch.ones(20, 20)
            donors = copy.deepcopy(t.original_deltas)
            committed = t.apply_shared(start, ordinary, 30)
            torch.testing.assert_close(committed[A_KEY], ordinary[A_KEY])
            saved = torch.load(Path(temp)/'b_transfer_rounds/r030/commit.pt', weights_only=False)
            self.assertNotIn('frozen_backbone_fixture', saved['ordinary_lora'])
            self.assertNotIn('frozen_backbone_fixture', saved['committed_lora'])
            torch.testing.assert_close(committed[B_KEY], ordinary[B_KEY]+saved['transfer_residual'][B_KEY])
            torch.testing.assert_close(t.model.state_dict()[B_KEY], start[B_KEY])
            t.observe_global(committed, 30, committed=True)
            self.assertEqual(t.progress()['b_transfer_optimizer_steps'], 2)
            self.assertEqual(t.progress()['b_transfer_events'], 1)
            self.assertTrue(t.records[0]['A_updated_this_round'])
            self.assertIsNone(t.pending)
            for client, value in donors.items():
                torch.testing.assert_close(value[B_KEY], tiny_transfer(temp).original_deltas[client][B_KEY])

    def test_legacy_sparse_implementation_equals_original_on_original_batches(self):
        with tempfile.TemporaryDirectory() as temp, redirect_stdout(io.StringIO()):
            a, b = tiny_route(Path(temp)/'old'), tiny_route(Path(temp)/'new')
            for t in (a, b):
                t.config.update(learning_rate=.03, regularization=.001, tail_weight=.35)
                t.begin(t.runtime.root, 30, 'O')
            start = copy.deepcopy(a.model.state_dict())
            ordinary = copy.deepcopy(start)
            ordinary[B_KEY] += .1
            manifests = a.manifests(30)
            with a.session():
                a.copy_parameters(ordinary)
                old = SharedDonorBTransfer.calibrate_shared(a, a.selected, manifests, start)
            with b.session():
                b.copy_parameters(ordinary)
                new = legacy_optimize(b, start, ordinary, manifests, {1: [0], 2: [0]})
            torch.testing.assert_close(old[B_KEY], new[B_KEY], rtol=1e-6, atol=1e-8)
            manifests[1]['batches'][0]['tail_positions'] = []
            b.begin(b.runtime.root, 30, 'O')
            with b.session():
                b.copy_parameters(ordinary)
                sparse = legacy_optimize(b, start, ordinary, manifests, {1: [0]})
            self.assertTrue(torch.isfinite(sparse[B_KEY]).all())

    def test_cpu_replay_all_routes_resume_and_portable_summary(self):
        with tempfile.TemporaryDirectory() as temp, redirect_stdout(io.StringIO()):
            root = Path(temp)
            source = root/'source'
            source.mkdir()
            event = root/'replay/seed42/r030'
            event.mkdir(parents=True)
            spec = dict(mode='replay', seed=42, round=30, source_run=str(source), arms=['N', 'P', 'M', 'S', 'C', 'F'],
                        rhos=[1.], rotations=True, include_legacy=False, source_probe=True)
            runtime = object.__new__(replay_runtime_class(spec))
            runtime.source, runtime.event_root, runtime.root = source, event, event
            runtime.sfra_config, runtime.args = {}, SimpleNamespace(seed=42)
            runtime.keys, runtime.b_keys, runtime.a_keys = [A_KEY, B_KEY], [B_KEY], [A_KEY]
            sample = tiny_route(event/'fixture')
            runtime.audit = SimpleNamespace(counts=sample.counts)
            start = copy.deepcopy(sample.model.state_dict())
            ordinary = copy.deepcopy(start)
            ordinary[B_KEY] += .1
            runtime.load_event = lambda: (start, ordinary, sample.original_deltas, sample.selected)
            def make_transfer(folder, settings, legacy=False):
                t = tiny_route(folder)
                t.runtime = runtime
                runtime.root = folder
                t.config = route_config(t.config, settings)
                return t
            runtime.new_transfer = make_transfer
            rows = [dict(client_id=k, local_position=p, class_id=c, raw_sample_id=100*k+p)
                    for k, g in enumerate(sample.groups) for c, ps in g.items() for p in ps]
            write_csv(source/'partition_manifest.csv', rows)
            write_json(event/'route_spec.json', spec)
            runtime.run(None)
            write_json(event/'route_receipt.json', dict(spec_digest=digest(spec), code_unchanged=True))
            status = report.summarize_replay(root)
            self.assertEqual(status[0]['status'], 'valid', status)
            paired = read_csv(root/'analysis/replay_paired.csv')
            self.assertTrue(any(r['comparison'] == 'C minus mean(3 rotations)' for r in paired))
            path = event/'fold0/rho1/F'
            summary = load_json(path/'summary.json')
            self.assertEqual(summary['algorithm_backward_images'], 4*summary['primary_two_steps_algorithm_backward_images'])
            self.assertEqual(summary['diagnostic_extra_steps_algorithm_backward_images'], 3*summary['primary_two_steps_algorithm_backward_images'])
            with patch.object(RouteTransfer, 'optimize', side_effect=AssertionError('completed branch was recomputed')):
                runtime.run(None)
            saved = load_json(path/'branch_completion.json')
            self.assertTrue(completed_branch(path, saved['identity']))
            (path/'residual_step2.pt').write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'artifact changed'):
                completed_branch(path, saved['identity'])
            # Portable scalar collection is deliberately independent of tensor files.
            self.assertEqual(report.summarize_replay(root)[0]['status'], 'valid')
            (path/'group_metrics.csv').write_text('changed')
            self.assertEqual(report.summarize_replay(root)[0]['status'], 'invalid')

    def test_launcher_default_plan_no_process_and_replay_path_protection(self):
        with tempfile.TemporaryDirectory() as temp, redirect_stdout(io.StringIO()):
            out = Path(temp)/'out'
            with patch.object(launcher.subprocess, 'run', side_effect=AssertionError('must not train')):
                launcher.main(['--output-root', str(out), '--arms', 'N', 'P', 'C', 'F'])
            self.assertFalse(out.exists())
            with self.assertRaises(ValueError):
                launcher.separate_paths(out/'nested', out)
            args = launcher.parser().parse_args(['--mode', 'replay', '--source-run', str(out), '--output-root', str(out/'child')])
            with self.assertRaisesRegex(ValueError, 'non-nested'):
                launcher.validate_args(args)
            missing = report.replay_preflight(out, [30])
            self.assertFalse(missing['ready'])
            self.assertIn('checkpoints/base_model.pt', missing['missing'])

    def test_full_resume_uses_round_checkpoint_and_preserves_registered_plan(self):
        with tempfile.TemporaryDirectory() as temp, redirect_stdout(io.StringIO()):
            root = Path(temp)
            args = launcher.parser().parse_args(['--output-root', str(root/'out'), '--arms', 'C', '--stop-after-round', '30'])
            launcher.validate_args(args)
            run, spec, command = next(launcher.jobs(args))
            def fake_worker(cmd, **kwargs):
                checkpoint = run/'checkpoints/sfra_last.pt'
                checkpoint.parent.mkdir(parents=True, exist_ok=True)
                checkpoint.write_bytes(b'fixture')
                write_json(run/'sfra_config.json', {})
                write_json(run/'progress.json', dict(completed_round=int(cmd[cmd.index('--sfra_stop_after_round')+1])))
            with patch.object(launcher.subprocess, 'run', side_effect=fake_worker):
                launcher.execute(args, run, spec, command)
            args.resume, args.stop_after_round = True, 60
            with patch.object(launcher, 'validate_profile'), patch.object(launcher.subprocess, 'run', side_effect=fake_worker) as call:
                launcher.execute(args, run, spec, command)
            resumed = call.call_args.args[0]
            self.assertEqual(resumed[resumed.index('--sfra_resume')+1], str(run/'checkpoints/sfra_last.pt'))
            self.assertEqual(resumed[resumed.index('--sfra_stop_after_round')+1], '60')
            launcher.merge_plan(args.output_root, [(run, spec, command)])
            second = copy.deepcopy(spec)
            second['settings']['seed'] = 0
            launcher.merge_plan(args.output_root, [(run.parent/'seed0', second, command)])
            self.assertEqual(len(load_json(args.output_root/'route_plan.json')['jobs']), 2)
            with self.assertRaisesRegex(ValueError, 'changed'):
                launcher.merge_plan(args.output_root, [(run, second, command)])

    def test_full_run_pair_audits_exclude_changed_feedback(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            old_jobs = plan_jobs(ab_args(root, ('a', 'ab'), (42,)))
            baseline = completed(root, old_jobs[0])
            for arm, offset in [('P', .2), ('C', .3), ('F', .1)]:
                job = dict(old_jobs[1], run=f'runs/{arm}/seed42')
                run = completed(root, job, offset=offset)
                spec = dict(schema_version=report.SCHEMA, mode='full', code_sha256=report.source_hashes(launcher.REPO),
                            settings=dict(arm=arm, seed=42, protocol_seed=42, partition='client-longtail'))
                write_json(run/'route_spec.json', spec)
                cfg = load_json(run/'sfra_config.json')
                cfg.update(b_route_experiment=spec, b_transfer=route_config(cfg['b_transfer'], spec['settings']))
                write_json(run/'sfra_config.json', cfg)
                write_json(run/'route_receipt.json', dict(spec_digest=digest(spec), code_unchanged=True))
                events = read_csv(run/'b_transfer_rounds.csv')
                for row in events:
                    row['route_arm'] = arm
                    path = run/f'b_transfer_rounds/r{int(row["round"]):03d}/optimization_steps.csv'
                    steps = read_csv(path)
                    for step in steps:
                        step['radius'] = 1.
                    write_csv(path, steps)
                write_csv(run/'b_transfer_rounds.csv', events)
            status, audit = report.summarize(root, [baseline])
            self.assertTrue(all(x['status'].startswith('valid') for x in status), status)
            self.assertEqual(len(audit), 6)
            self.assertTrue(all(x['valid'] for x in audit))
            path = root/'runs/C/seed42/b_transfer_rounds/r030/calibration_manifest.json'
            manifest = load_json(path)
            manifest['recipients']['0']['batches'][0]['tail_positions'] = [99]
            write_json(path, manifest)
            _, audit = report.summarize(root, [baseline])
            self.assertFalse(next(x['valid'] for x in audit if x['comparison'] == 'C minus P'))
            self.assertTrue(next(x['valid'] for x in audit if x['comparison'] == 'P minus F'))


if __name__ == '__main__':
    unittest.main()
