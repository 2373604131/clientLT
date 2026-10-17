"""Create an isolated source snapshot, excluding images, models and old results."""
from pathlib import Path
import hashlib
import json
import zipfile

REPO = Path(__file__).resolve().parents[1]
NAME = 'clientLT_multidataset_20261015'


def main():
    directories = ('scripts', 'tools', 'trainers', 'utils', 'clip', 'Dassl', 'datasets',
                   'loss', 'configs', 'third_party/paper_baselines', 'tests', 'references/full10_clientlt')
    extensions = {'.py', '.json', '.yaml', '.yml', '.txt', '.md', '.sh', '.csv', '.gz'}
    files = [REPO/'federated_main.py', REPO/'requirements.txt', REPO/'README.md',
             REPO/'docs/multidataset_clientlt_old_ab_20261015.txt']
    for directory in directories:
        files.extend(p for p in (REPO/directory).rglob('*') if p.is_file() and p.suffix in extensions
                     and not any(part in ('.git', '__pycache__') for part in p.parts))
    files = sorted(set(files))
    output = REPO/'output/releases'/f'{NAME}.zip'
    output.parent.mkdir(parents=True, exist_ok=True)
    hashes = {p.relative_to(REPO).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    with zipfile.ZipFile(output, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            archive.write(path, str(Path(NAME)/path.relative_to(REPO)))
        archive.writestr(NAME+'/RELEASE_SHA256.json', json.dumps(hashes, indent=2)+'\n')
    print(output)
    print('Files:', len(files), 'Bytes:', output.stat().st_size)


if __name__ == '__main__':
    main()
