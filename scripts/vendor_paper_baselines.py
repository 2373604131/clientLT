"""Fetch pinned, unmodified official reference files (never executed as a package)."""
import hashlib
import argparse
import io
import json
from pathlib import Path
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]
SOURCES = {
    'fedpurel': ('shihaohou/FedPuReL', '20ce20a15f449631c9aa655be585a0a39a18a4cd'),
    'fedntd': ('Lee-Gihun/FedNTD', 'be00ee598139654abe650c446a7ba3bc4e340233'),
    'capt': ('shihaohou/CAPT', '840b7cab26c04fc9e65540a980b6ce020d62d618'),
    'fedsvd': ('seanie12/fed-svd', 'e73398ba34b6b269c3bf76ed9d74906761fc1a4d'),
    'fedlf': ('18sym/FedLF', '127cbaf363ec3a1f40c116f6b1de3c1f7a705764'),
    'fedyoyo': ('shanss132/FedYoYo', 'fc6d728febb461ca46eedd142ef835b6c6572f88'),
    'fedrela': ('guangzhengh/FedReLa', '20a58bd75f48b3243f80adb35537b5b4ba252773'),
}


def selected(method, path):
    if path.lower().startswith(('license', 'copying', 'notice', 'readme')):
        return True
    if method in ('fedlf', 'fedyoyo', 'fedrela'):
        return path.endswith(('.py', '.json', '.yaml', '.yml', '.sh', '.txt', '.md'))
    if method == 'fedntd':
        return path.endswith(('.py', '.json', '.yaml')) and path.startswith(('algorithms/', 'train_tools/', 'config/'))
    if method == 'fedpurel':
        return path.endswith(('.py', '.yaml', '.sh')) and (path.startswith(('trainers/', 'utils/', 'configs/', 'scripts/')) or path in ('federated_main.py', 'Dassl/dassl/engine/trainer.py'))
    if method == 'fedsvd':
        return path.endswith(('.py', '.sh', '.txt')) and (path.startswith(('fedmm/', 'modules/', 'misc/', 'scripts/')) or path in ('parser.py', 'requirements.txt'))
    return path in ('trainers/capt.py', 'loss/prompt_loss.py', 'federated_main.py', 'scripts/capt.sh', 'clip/model.py', 'configs/trainers/CAPT/vit_b16.yaml')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--methods', nargs='+', choices=list(SOURCES), default=list(SOURCES))
    args = parser.parse_args()
    for name in args.methods:
        repo, commit = SOURCES[name]
        dest = ROOT / 'third_party/paper_baselines' / name
        receipt = dest / 'UPSTREAM.json'
        if receipt.exists():
            old = json.loads(receipt.read_text(encoding='utf-8'))
            if old['commit'] != commit or any(hashlib.sha256((dest / p).read_bytes()).hexdigest() != h for p, h in old['files'].items()):
                raise ValueError('Reference snapshot changed: ' + str(dest))
            if name != 'fedpurel' or 'Dassl/dassl/engine/trainer.py' in old['files']:
                print('Verified existing snapshot:', name, flush=True)
                continue
        url = f'https://codeload.github.com/{repo}/zip/{commit}'
        print('Fetching', repo, commit, flush=True)
        request = urllib.request.Request(url, headers={'User-Agent': 'CAPT-paper-benchmarks'})
        with urllib.request.urlopen(request, timeout=90) as response:
            archive = response.read()
        files = {}
        with zipfile.ZipFile(io.BytesIO(archive)) as source:
            for info in source.infolist():
                if info.is_dir():
                    continue
                path = info.filename.split('/', 1)[1]
                if not selected(name, path):
                    continue
                target = (dest / path).resolve()
                if not target.is_relative_to(dest.resolve()):
                    raise ValueError('Unsafe upstream archive member')
                content = source.read(info)
                if target.exists() and target.read_bytes() != content:
                    raise ValueError('Refusing to overwrite changed reference: ' + str(target))
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(content)
                files[path] = hashlib.sha256(content).hexdigest()
        if not files:
            raise ValueError('Empty source snapshot')
        receipt.write_text(json.dumps(dict(repository='https://github.com/' + repo, commit=commit,
            archive_sha256=hashlib.sha256(archive).hexdigest(), files=files), indent=2) + '\n', encoding='utf-8')
        print('Saved', len(files), 'reference files:', dest, flush=True)


if __name__ == '__main__':
    main()
