"""Relocation discovery and single-device execution without loading CLIP."""
import contextlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import weakref

from scripts.run_a_projection_diagnosis import parse_args, main, execute
from tests.test_a_projection_artifact_scan import make_run, write_json
from tools.a_projection_diagnosis.discovery import discover_source
from tools.a_projection_diagnosis.protocol import check_source


def make_valid_source(root, tensors=True):
    """Synthetic partition satisfies structural checks; no real training data."""
    run = make_run(root, tensors=tensors)
    counts, rows = [[0] * 100 for _ in range(30)], []
    for raw in range(10847):
        client, cls = raw % 30, raw % 100
        position = sum(counts[client])
        rows.append('%d,%d,%d,%d' % (raw, client, cls, position))
        counts[client][cls] += 1
    (run / 'partition_manifest.csv').write_text('raw_sample_id,client_id,class_id,local_position\n' + '\n'.join(rows), encoding='utf-8')
    write_json(run / 'bridge_metadata.json', dict(method='s', client_sample_counts=list(map(sum, counts)),
        schedule=[list(range(30))] * 100, resolved_args={}))
    write_json(run / 'aggregation_solution.json', dict(rule='fedavg', matrix=counts))
    return run


class DiscoveryTests(unittest.TestCase):
    def discover(self, root, **kwargs):
        return discover_source('fedavg', root / 'missing', auto=True, repo=root,
                               historical_repos=(), **kwargs)

    def test_relocated_complete_suite_and_explicit_missing_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run = make_valid_source(root / 'output/plain_a_aggregation_relocated')
            source, report = self.discover(root)
            self.assertEqual(source['path'], str(run.resolve()))
            self.assertTrue(report['candidates'][0]['required_models_ready'])
            source, report = self.discover(root, override=root / 'explicit_missing')
            self.assertIsNone(source)
            self.assertIn('Explicit source unavailable', report['error'])
            self.assertEqual(len(report['checked']), 1)

    def test_archive_provenance_locates_original_tensors(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            original = make_valid_source(root / 'storage/renamed_suite')
            archive = make_valid_source(root / 'output/plain_a_aggregation_results/plain_a_aggregation_v1', tensors=False)
            source = check_source(archive, 'fedavg')
            source['meta']['resolved_args']['output_dir'] = str(original)
            write_json(archive / 'bridge_metadata.json', source['meta'])
            selected, report = self.discover(root)
            self.assertEqual(selected['path'], str(original.resolve()))
            self.assertEqual(len(report['candidates']), 2)

    def test_metadata_only_is_reported_without_fabricating_readiness(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run = make_valid_source(root / 'output/plain_a_aggregation_results/plain_a_aggregation_v1', tensors=False)
            selected, report = self.discover(root)
            self.assertEqual(selected['path'], str(run.resolve()))
            self.assertFalse(report['candidates'][0]['required_models_ready'])
            self.assertIn('frozen_base', report['candidates'][0]['missing_models'])

    def test_multiple_complete_runs_require_explicit_selection_and_reject_other_protocol(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run = make_valid_source(root / 'output/plain_a_aggregation_a')
            make_valid_source(root / 'output/plain_a_aggregation_b')
            source, report = self.discover(root)
            self.assertIsNone(source)
            self.assertIn('Multiple complete', report['error'])
            config = check_source(run, 'fedavg')['config']
            config['control_enabled'] = True
            write_json(run / 'control_config.json', config)
            source, report = self.discover(root)
            self.assertEqual(len(report['rejected']), 1)
            self.assertTrue(source['path'].endswith(str(Path('plain_a_aggregation_b/fedavg/runs/seed42/plain_a'))))


class SingleDeviceTests(unittest.TestCase):
    def test_server_default_runs_smoke_then_formal_in_process_and_never_launches_gpu_queue(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            args = parse_args(['--stage', 'server'])
            self.assertIsNone(args.gpus)
            calls = []
            with patch('scripts.run_a_projection_diagnosis.inventory', return_value=([], [], [])), patch(
                    'scripts.run_a_projection_diagnosis.require_inputs', return_value='test'), patch(
                    'scripts.run_a_projection_diagnosis.execute', side_effect=lambda args, *a, **k: calls.append((args.stage, args.device))), patch(
                    'tools.a_projection_diagnosis.summary.summarize') as summary, patch(
                    'tools.a_projection_diagnosis.parallel.launch', side_effect=AssertionError('parallel queue invoked')):
                main(['--stage', 'server', '--output-root', tmp])
            self.assertEqual(calls, [('smoke', 'cuda:0'), ('run', 'cuda:0')])
            summary.assert_called_once()

    def test_five_full_tasks_execute_in_protocol_order_with_one_live_model(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            root, live, calls, devices = Path(tmp), weakref.WeakSet(), [], []
            sources = [dict(rule=rule, path=str(Path(tmp) / rule), meta={}, config={}, partition_sha256='test')
                       for rule in ('fedavg', 'tailrw16')]
            test = self
            class Backend:
                def __init__(self, args, source, store, root, fingerprint):
                    test.assertEqual(len(live), 0, 'previous model was not released')
                    live.add(self)
                    devices.append(args.device)
            def e1(backend, store, root):
                calls.append((root.parent.name, 'e1'))
            def event(backend, store, rnd, root):
                calls.append((root.parent.name, rnd))
            args = parse_args(['--stage', 'run', '--output-root', str(root)])
            with patch('tools.a_projection_diagnosis.states.StateStore', return_value=object()), patch(
                    'tools.a_projection_diagnosis.backend.ClipBackend', Backend), patch(
                    'tools.a_projection_diagnosis.experiment.longitudinal', new=e1), patch(
                    'tools.a_projection_diagnosis.experiment.event_panels', new=event):
                execute(args, sources, 'test')
            self.assertEqual(calls, [('fedavg', 'e1'), ('tailrw16', 'e1'), ('fedavg', 20), ('fedavg', 50), ('fedavg', 80)])
            self.assertEqual(devices, ['cuda:0'] * 5)
            self.assertEqual(len(live), 0)


if __name__ == '__main__':
    unittest.main()
