"""Official snapshot hashes must survive Git add and Linux/Windows checkout."""
import hashlib
import io
import json
from pathlib import Path
import shutil
import subprocess
import tarfile
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SNAPSHOTS = Path('third_party/paper_baselines')


class BaselineGitCheckoutTests(unittest.TestCase):
    def test_official_bytes_survive_autocrlf_and_checkouts(self):
        # Use real Git staging/checkout, not a replacement-string approximation.
        # This catches a missing/misspelled attribute as well as CRLF snapshots.
        originals = {}
        for receipt in sorted((ROOT / SNAPSHOTS).glob('*/UPSTREAM.json')):
            metadata = json.loads(receipt.read_text(encoding='utf-8'))
            for name, expected in metadata['files'].items():
                path = receipt.parent / name
                content = path.read_bytes()
                self.assertEqual(hashlib.sha256(content).hexdigest(), expected, str(path))
                originals[path.relative_to(ROOT).as_posix()] = content
        self.assertTrue(originals)
        self.assertTrue(any(b'\r\n' in content for content in originals.values()))
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / 'source'
            source.mkdir()
            shutil.copyfile(ROOT / '.gitattributes', source / '.gitattributes')
            for name, content in originals.items():
                target = source / name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(content)

            def git(*args):
                return subprocess.run(['git', '-C', str(source), *args], check=True,
                                      stdout=subprocess.PIPE, stderr=subprocess.PIPE).stdout

            git('init', '--quiet')
            git('-c', 'core.autocrlf=true', '-c', 'core.safecrlf=false', 'add', '--', '.')
            tree = git('write-tree').decode('ascii').strip()
            archive = git('archive', '--format=tar', tree)
            with tarfile.open(fileobj=io.BytesIO(archive), mode='r:') as saved:
                for name, content in originals.items():
                    self.assertEqual(saved.extractfile(name).read(), content, name)
            for autocrlf in ('false', 'true'):
                destination = Path(directory) / ('checkout_' + autocrlf)
                destination.mkdir()
                git('-c', 'core.autocrlf=' + autocrlf, 'checkout-index', '--all',
                    '--prefix=' + destination.as_posix() + '/')
                for name, content in originals.items():
                    self.assertEqual((destination / name).read_bytes(), content,
                                     autocrlf + ': ' + name)


if __name__ == '__main__':
    unittest.main()
