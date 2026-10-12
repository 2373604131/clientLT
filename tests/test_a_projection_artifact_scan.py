"""Portable file search tests. Tiny sentinel .pt files are never unpickled."""
import contextlib
import io
import json
from pathlib import Path
import tarfile
import tempfile
import unittest
from unittest.mock import patch
import zipfile

from scripts.check_a_projection_artifacts import scan, format_report, main


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding='utf-8')


def make_run(root, rule='fedavg', tensors=True):
    run = root / rule / 'runs/seed42/plain_a'
    write_json(run / 'bridge_metadata.json', dict(method='s', resolved_args=dict(output_dir='/old/server/clientLT/run')))
    write_json(run / 'control_config.json', dict(method='s', seed=42, topology='client-longtail',
        la_tau=1., a_lr_mult=1., normal_trainable_factor='B', extra_trainable_factor='A',
        control_enabled=False, extra_a_lr=.001, extra_weight_decay=0., candidate_rounds=list(range(1, 91)),
        head_ids=list(range(20)), middle_ids=list(range(20, 80)), tail_ids=list(range(80, 100))))
    write_json(run / 'sfra_config.json', dict(variant='s', retention_weight=0., correction_steps=0, b_transfer=False))
    write_json(run / 'aggregation_solution.json', dict(rule=rule))
    (run / 'partition_manifest.csv').write_text('raw_sample_id,client_id,class_id,local_position\n', encoding='utf-8')
    if tensors:
        paths = ['checkpoints/base_model.pt'] + [
            'events/r%03d_c000_main_%s/state.pt' % (r, 'normal_B' if r == 100 else 'refresh_A')
            for r in (20, 50, 80, 90, 100)]
        for relative in paths:
            file = run / relative
            file.parent.mkdir(parents=True, exist_ok=True)
            file.write_bytes(b'presence-only fixture; not a torch model')
    return run


class ArtifactSearchTests(unittest.TestCase):
    def test_complete_files_and_metadata_only_are_distinct_and_never_claim_tensor_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            make_run(root, 'fedavg')
            make_run(root, 'tailrw16', tensors=False)
            report = scan([root, root / 'fedavg', root / 'absent'], progress=lambda _: None)
            self.assertEqual(len(report['runs']), 2)  # overlapping roots deduplicated
            by_rule = {r['rule']: r for r in report['runs']}
            self.assertEqual(by_rule['fedavg']['status'], 'FILES_PRESENT')
            self.assertEqual(by_rule['fedavg']['missing_models'], [])
            self.assertFalse(by_rule['fedavg']['tensors_validated'])
            self.assertEqual(by_rule['tailrw16']['status'], 'METADATA_ONLY')
            self.assertIn('committed_100', by_rule['tailrw16']['missing_models'])
            self.assertIn('missing_or_not_readable', report['roots'][2]['status'])
            self.assertIn('/old/server/clientLT/run', format_report(report))

    def test_zip_and_tar_archives_find_both_full_and_lightweight_runs_without_extraction(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, storage = root / 'source', root / 'storage'
            storage.mkdir()
            make_run(source)
            with zipfile.ZipFile(storage / 'full.zip', 'w') as archive:
                for file in source.rglob('*'):
                    if file.is_file():
                        archive.write(file, file.relative_to(source).as_posix())
            with tarfile.open(storage / 'results.tar.gz', 'w:gz') as archive:
                for file in source.rglob('*'):
                    if file.is_file() and file.suffix != '.pt':
                        archive.add(file, arcname=file.relative_to(source).as_posix())
            report = scan([storage], archives=True, progress=lambda _: None)
            self.assertEqual(report['counts']['archives_checked'], 2)
            self.assertEqual(sorted(r['status'] for r in report['runs']), ['ARCHIVE_FILES_PRESENT', 'METADATA_ONLY'])
            self.assertEqual(report['error_count'], 0)
            self.assertEqual(len(list(storage.iterdir())), 2)
            self.assertTrue(all(r['archive'] for r in report['runs']))

    def test_mapped_previous_a_recovers_anchor_but_cannot_claim_historical_proposals(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run = make_run(root)
            original = run / 'events/r020_c000_main_refresh_A/state.pt'
            original.rename(run / 'checkpoints/after20.pt')
            (run / 'checkpoints/after19.pt').write_bytes(b'A-only fixture')
            write_json(run / 'projection_states.json', dict(committed={'19': 'checkpoints/after19.pt', '20': 'checkpoints/after20.pt'}))
            row = scan([root], progress=lambda _: None)['runs'][0]
            self.assertEqual(row['status'], 'FILES_PRESENT')
            self.assertEqual(row['pre_a'][0]['recovery'], 'A_previous+B_current')
            self.assertFalse(row['proposal_event_files'][0]['present'])

    def test_other_protocol_and_empty_model_are_not_reported_complete(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run = make_run(root)
            config_path = run / 'control_config.json'
            config = json.loads(config_path.read_text())
            config['control_enabled'] = True
            write_json(config_path, config)
            (run / 'checkpoints/base_model.pt').write_bytes(b'')
            row = scan([root], progress=lambda _: None)['runs'][0]
            self.assertEqual(row['status'], 'PARTIAL_OR_OTHER_PROTOCOL')
            self.assertFalse(row['target_protocol_screen'])
            self.assertIn('frozen_base_and_initial', row['missing_models'])

    def test_orphan_model_and_legacy_bridge_dump_are_preserved_as_clues(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            file = root / 'renamed/bridge_dumps/round_020/refresh_A/state.pt'
            file.parent.mkdir(parents=True)
            file.write_bytes(b'legacy')
            row = scan([root], progress=lambda _: None)['runs'][0]
            self.assertEqual(Path(row['location']), root / 'renamed')
            self.assertEqual(row['tensor_file_count'], 1)
            self.assertIn('bridge_metadata.json', row['missing_metadata'])
            self.assertIn('committed_020', row['missing_models'])

    def test_read_errors_and_bad_archives_do_not_abort_the_search(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            make_run(root)
            (root / 'broken.zip').write_bytes(b'not a zip')
            report = scan([root], archives=True, progress=lambda _: None)
            self.assertEqual(report['runs'][0]['status'], 'FILES_PRESENT')
            self.assertEqual(report['error_count'], 1)
            self.assertIn('broken.zip', report['errors'][0]['path'])
            def walk(path, followlinks, onerror):
                onerror(PermissionError(13, 'Permission denied', str(root / 'denied')))
                return iter([])
            with patch('scripts.check_a_projection_artifacts.os.walk', side_effect=walk):
                report = scan([root], progress=lambda _: None)
            self.assertEqual(report['roots'][0]['status'], 'scanned_with_errors')
            self.assertEqual(report['error_count'], 1)

    def test_fallback_experiment_plan_inside_tar_uses_normalized_member_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run = make_run(root / 'source')
            (run / 'aggregation_solution.json').unlink()
            write_json(run.parents[2] / 'experiment_plan.json', dict(aggregation=dict(rule='fedavg')))
            storage = root / 'storage'
            storage.mkdir()
            with tarfile.open(storage / 'full.tar.gz', 'w:gz') as archive:
                archive.add(root / 'source', arcname='relocated_suite')
            report = scan([storage], archives=True, progress=lambda _: None)
            self.assertEqual(report['runs'][0]['status'], 'ARCHIVE_FILES_PRESENT')

    def test_cli_writes_portable_json_and_text_reports(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            root = Path(tmp)
            make_run(root / 'source', tensors=False)
            prefix = root / 'report/server_a'
            main(['--roots', str(root / 'source'), '--output-prefix', str(prefix)])
            report = json.loads(Path(str(prefix) + '.json').read_text(encoding='utf-8'))
            self.assertEqual(report['runs'][0]['status'], 'METADATA_ONLY')
            self.assertIn('METADATA_ONLY', Path(str(prefix) + '.txt').read_text(encoding='utf-8'))


if __name__ == '__main__':
    unittest.main()
