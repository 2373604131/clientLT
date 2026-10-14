"""Reproduce Linux LF checkout against historical CRLF metadata, without GPU."""
import contextlib
import io
from pathlib import Path
import shutil
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from scripts import recover_dirichlet_ab as recovery
from scripts.run_cliplora_sfra import prepare_protocol
from tools.benchmarks import common, dirichlet

SOURCE = common.REPO / 'references/full10_clientlt'


class ProbeRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()

    def linux_source(self):
        root = self.root / 'linux_source'
        shutil.copytree(SOURCE, root)
        probe = root / 'protocol/probe_manifest.csv'
        probe.write_bytes(probe.read_bytes().replace(b'\r\n', b'\n'))
        return root

    def legacy_suite(self):
        root = self.root / 'legacy_suite'
        reference = dirichlet.prepare_reference(self.linux_source(), root / 'reference')
        probe = reference / 'protocol/probe_manifest.csv'
        probe.write_bytes(probe.read_bytes().replace(b'\r\n', b'\n'))
        recipe_path = reference / 'protocol/partition_protocol.json'
        recipe = common.read_json(recipe_path)
        recipe.pop('probe_manifest_copy')
        common.write_json(recipe_path, recipe)
        self.register_legacy(root)
        return root

    def register_legacy(self, root):
        common.write_json(root / 'jobs/a.json', dict(seed=42, protocol=dict(reference_files={
            name: common.file_hash(root / 'reference' / name) for name in recovery.REFERENCE_NAMES})))

    def test_exact_and_both_newline_directions_require_original_hash(self):
        crlf, lf = b'a,b\r\n1,2\r\n', b'a,b\n1,2\n'
        self.assertEqual(recovery.restore_probe_bytes(crlf, recovery.sha256(crlf)), (crlf, 'unchanged'))
        self.assertEqual(recovery.restore_probe_bytes(lf, recovery.sha256(crlf)), (crlf, 'restored_crlf'))
        self.assertEqual(recovery.restore_probe_bytes(crlf, recovery.sha256(lf)), (lf, 'restored_lf'))
        with self.assertRaisesRegex(ValueError, 'content differs'):
            recovery.restore_probe_bytes(b'a,b\n1,3\n', recovery.sha256(crlf))

    def test_linux_reference_generation_restores_historical_hash_before_sfra(self):
        source = self.linux_source()
        before = {name: common.file_hash(source / name) for name in common.REFERENCE_FILES}
        ref = dirichlet.prepare_reference(source, self.root / 'new_reference')
        meta = common.read_json(ref / 'bridge_metadata.json')
        self.assertEqual(common.file_hash(ref / 'protocol/probe_manifest.csv'), meta['probe_manifest_sha256'])
        self.assertEqual(common.read_json(ref / 'protocol/partition_protocol.json')['probe_manifest_copy']['conversion'],
                         'restored_crlf')
        self.assertEqual(dirichlet.prepare_reference(source, ref), ref)
        self.assertEqual(before, {name: common.file_hash(source / name) for name in common.REFERENCE_FILES})
        self.assert_sfra_probe_matches(ref)

    def assert_sfra_probe_matches(self, ref):
        run = self.root / 'sfra_run'
        prepare_protocol(SimpleNamespace(reference_run=ref, partition=dirichlet.PARTITION), run)
        # Exact values consumed by BridgeAudit and LAControlRuntime's failed assertion.
        actual = common.file_hash(run / 'protocol/probe_manifest.csv')
        expected = common.read_json(run / 'protocol/source_metadata.json')['probe_manifest_sha256']
        self.assertEqual(actual, expected)

    def test_recovery_copies_only_probe_bytes_and_leaves_registered_source_untouched(self):
        source = self.legacy_suite()
        before = {name: (source / 'reference' / name).read_bytes() for name in recovery.REFERENCE_NAMES}
        with self.assertRaisesRegex(ValueError, 'Probe manifest bytes differ'):
            common.reference_contract(source / 'reference')
        ref, receipt = recovery.recover_reference(source, self.root / 'recovered')
        self.assertEqual(receipt['changed_files'], ['protocol/probe_manifest.csv'])
        self.assertEqual(receipt['probe_conversion'], 'restored_crlf')
        common.reference_contract(ref)
        for name, data in before.items():
            self.assertEqual((source / 'reference' / name).read_bytes(), data)
            if name != 'protocol/probe_manifest.csv':
                self.assertEqual((ref / name).read_bytes(), data)
        self.assertEqual(recovery.recover_reference(source, self.root / 'recovered'), (ref, receipt))
        self.assert_sfra_probe_matches(ref)

    def test_new_reference_rejects_changed_sample_id_instead_of_changing_hash(self):
        source = self.linux_source()
        path = source / 'protocol/probe_manifest.csv'
        data = path.read_bytes()
        path.write_bytes(data.replace(b'80,0,', b'81,0,', 1))
        with self.assertRaisesRegex(ValueError, 'content differs'):
            dirichlet.prepare_reference(source, self.root / 'rejected')
        self.assertFalse((self.root / 'rejected').exists())

    def test_recovery_rejects_source_tampering_even_if_newlines_are_repairable(self):
        source = self.legacy_suite()
        path = source / 'reference/protocol/probe_manifest.csv'
        path.write_bytes(path.read_bytes().replace(b'80,0,', b'81,0,', 1))
        with self.assertRaisesRegex(ValueError, 'changed since job registration'):
            recovery.recover_reference(source, self.root / 'rejected')
        self.assertFalse((self.root / 'rejected').exists())

    def test_recovery_rejects_unrecoverable_content_with_matching_registration(self):
        source = self.legacy_suite()
        path = source / 'reference/protocol/probe_manifest.csv'
        path.write_bytes(path.read_bytes().replace(b'80,0,', b'81,0,', 1))
        self.register_legacy(source)
        with self.assertRaisesRegex(ValueError, 'content differs'):
            recovery.recover_reference(source, self.root / 'rejected')

    def test_recovery_never_overwrites_original_or_existing_unrelated_output(self):
        source = self.legacy_suite()
        with self.assertRaisesRegex(ValueError, 'separate directory'):
            recovery.recover_reference(source, source)
        with self.assertRaisesRegex(ValueError, 'separate directory'):
            recovery.recover_reference(source, source / 'recovery')
        out = self.root / 'unrelated'
        out.mkdir()
        (out / 'keep.txt').write_text('preserve')
        with self.assertRaisesRegex(ValueError, 'must be empty'):
            recovery.recover_reference(source, out)
        self.assertEqual((out / 'keep.txt').read_text(), 'preserve')

    def test_train_delegates_only_to_existing_a_ab_launcher_and_new_root(self):
        source = self.legacy_suite()
        out = self.root / 'recovered'
        with patch.object(recovery.subprocess, 'run') as child, contextlib.redirect_stdout(io.StringIO()):
            recovery.main(['--stage', 'train', '--source-root', str(source), '--output-root', str(out)])
        command = child.call_args.args[0]
        self.assertTrue(command[2].endswith('run_paper_benchmarks.py'))
        self.assertEqual(command[command.index('--methods') + 1:command.index('--seed')], ['a', 'ab'])
        self.assertEqual(Path(command[command.index('--reference-run') + 1]), out / 'reference')
        self.assertEqual(Path(command[command.index('--output-root') + 1]), out)
        self.assertNotIn('--resume', command)


if __name__ == '__main__':
    unittest.main()
