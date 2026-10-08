"""Regression checks for Git LF checkout of a CRLF-hashed frozen probe CSV."""
from contextlib import redirect_stderr, redirect_stdout
import hashlib
import io
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

from scripts import run_cliplora_b_routes as launcher
from scripts.run_ab_validation import write_json
from scripts.run_cliplora_sfra import prepare_protocol
from tools.sfra.b_routes import reference_probe_manifest, replay_preflight, verify_prepared_probe
from tools.sfra.maintext import load_json
from tools.sfra.summary import read_csv


def reference_fixture(root):
    source = Path(root)/'reference'
    original = launcher.REPO/'references/full10_clientlt'
    for name in ('partition_manifest.csv', 'bridge_metadata.json', 'protocol/full_schedule.json',
                 'protocol/eri_protocol.json', 'protocol/probe_manifest.csv'):
        target = source/name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(original/name, target)
    # This is how the existing file in Git is checked out on the Linux server.
    path = source/'protocol/probe_manifest.csv'
    path.write_bytes(path.read_bytes().replace(b'\r\n', b'\n'))
    return source


class ProbeProtocolTest(unittest.TestCase):
    def test_linux_checkout_reproduces_assertion_and_copy_restores_exact_original(self):
        with tempfile.TemporaryDirectory() as temp, redirect_stdout(io.StringIO()):
            source = reference_fixture(temp)
            original = {p: p.read_bytes() for p in source.rglob('*') if p.is_file()}
            expected = load_json(source/'bridge_metadata.json')['probe_manifest_sha256']
            legacy = Path(temp)/'legacy'
            prepare_protocol(type('Args', (), dict(reference_run=source, partition='client-longtail'))(), legacy)
            copied = legacy/'protocol/probe_manifest.csv'
            self.assertNotEqual(hashlib.sha256(copied.read_bytes()).hexdigest(), expected)
            with self.assertRaisesRegex(ValueError, 'Prepared probe manifest differs'):
                verify_prepared_probe(legacy)

            args = launcher.parser().parse_args(['--reference-run', str(source), '--output-root', str(Path(temp)/'fixed'),
                                                 '--arms', 'P', 'C', 'F', '--seeds', '42'])
            launcher.validate_args(args)
            jobs = list(launcher.jobs(args))
            self.assertEqual(len(jobs), 3)
            for run, spec, _ in jobs:
                self.assertEqual(spec['probe_manifest_identity']['newline_conversion'], 'CRLF')
                launcher.prepare_route_protocol(source, run, spec)
                verify_prepared_probe(run)
                self.assertEqual(hashlib.sha256((run/'protocol/probe_manifest.csv').read_bytes()).hexdigest(), expected)
                self.assertEqual(read_csv(source/'protocol/probe_manifest.csv'), read_csv(run/'protocol/probe_manifest.csv'))
                self.assertEqual((run/'protocol/source_metadata.json').read_bytes(), original[source/'bridge_metadata.json'])
                self.assertEqual(load_json(run/'protocol/probe_manifest_identity.json'), spec['probe_manifest_identity'])
            for path, content in original.items():
                self.assertEqual(path.read_bytes(), content)

    def test_preflight_checks_linux_checkout_without_loading_model_or_modifying_reference(self):
        with tempfile.TemporaryDirectory() as temp, redirect_stdout(io.StringIO()):
            source = reference_fixture(temp)
            before = (source/'protocol/probe_manifest.csv').read_bytes()
            output = Path(temp)/'result'
            with patch.object(launcher.subprocess, 'run', side_effect=AssertionError('must not start GPU worker')):
                launcher.main(['--stage', 'preflight', '--reference-run', str(source), '--output-root', str(output),
                               '--arms', 'P', 'C', 'F', '--seeds', '42'])
            result = load_json(output/'preflight.json')
            self.assertTrue(result['ready'])
            self.assertTrue(all(j['probe_manifest_identity']['newline_conversion'] == 'CRLF' for j in result['jobs']))
            self.assertFalse((output/'runs').exists())
            self.assertEqual((source/'protocol/probe_manifest.csv').read_bytes(), before)

    def test_real_content_or_encoding_change_is_rejected_during_preflight(self):
        with tempfile.TemporaryDirectory() as temp, redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            source = reference_fixture(temp)
            path = source/'protocol/probe_manifest.csv'
            raw = path.read_bytes()
            fields = raw.splitlines()[1].split(b',')
            changed = fields.copy()
            changed[2] = str(int(fields[2])+1).encode()
            mutations = [raw.replace(b','.join(fields), b','.join(changed), 1), b'\xef\xbb\xbf'+raw, raw+b' ']
            for index, value in enumerate(mutations):
                path.write_bytes(value)
                output = Path(temp)/f'out{index}'
                with patch.object(launcher.subprocess, 'run', side_effect=AssertionError('must not start worker')):
                    with self.assertRaises(SystemExit) as caught:
                        launcher.main(['--stage', 'preflight', '--reference-run', str(source), '--output-root', str(output)])
                self.assertEqual(caught.exception.code, 2)
                result = load_json(output/'preflight.json')
                self.assertFalse(result['ready'])
                self.assertIn('Neither LF nor CRLF matches', result['error'])
                self.assertFalse((output/'runs').exists())
                self.assertEqual(path.read_bytes(), value)
                replay = replay_preflight(source, [30])
                self.assertTrue(any('Probe manifest identity mismatch' in error for error in replay['errors']))

    def test_exact_bytes_are_preserved_and_lf_metadata_is_supported(self):
        with tempfile.TemporaryDirectory() as temp:
            source = reference_fixture(temp)
            path = source/'protocol/probe_manifest.csv'
            lf = path.read_bytes()
            meta = load_json(source/'bridge_metadata.json')
            meta['probe_manifest_sha256'] = hashlib.sha256(lf).hexdigest()
            write_json(source/'bridge_metadata.json', meta)
            recovered, audit = reference_probe_manifest(source)
            self.assertEqual(recovered, lf)
            self.assertEqual(audit['newline_conversion'], 'none')
            path.write_bytes(lf.replace(b'\n', b'\r\n'))
            recovered, audit = reference_probe_manifest(source)
            self.assertEqual(recovered, lf)
            self.assertEqual(audit['newline_conversion'], 'LF')


if __name__ == '__main__':
    unittest.main()
