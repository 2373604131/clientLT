"""Recover pre-training A/AB probe-newline failures without editing active code.

Copy ONLY this file to an active server checkout. The script creates a new
reference/output directory and delegates to the existing benchmark launcher.
It never changes source references, registered jobs, or training code.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

REPO = Path(__file__).resolve().parents[1]
REFERENCE_NAMES = ('partition_manifest.csv', 'bridge_metadata.json',
                   'protocol/full_schedule.json', 'protocol/eri_protocol.json',
                   'protocol/probe_manifest.csv', 'protocol/partition_protocol.json')


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def restore_probe_bytes(data, expected):
    """Accept only exact original bytes or a provable LF/CRLF conversion.

    Never replace the expected hash with a hash of unverified input. A changed
    sample ID, row order, BOM or other content must remain a hard error.
    """
    if sha256(data) == expected:
        return data, 'unchanged'
    lf = data.replace(b'\r\n', b'\n')
    for name, candidate in (('restored_lf', lf), ('restored_crlf', lf.replace(b'\n', b'\r\n'))):
        if sha256(candidate) == expected:
            return candidate, name
    raise ValueError('Probe manifest content differs from its recorded SHA256; '
                     'LF/CRLF conversion cannot recover the original file. Do not bypass this check.')


def recover_reference(source_root, output_root):
    source_root, output_root = Path(source_root).resolve(), Path(output_root).resolve()
    if output_root == source_root or source_root in output_root.parents or output_root in source_root.parents:
        raise ValueError('Recovery output must be a separate directory outside the original suite')
    source = source_root / 'reference'
    files = {name: (source / name).read_bytes() for name in REFERENCE_NAMES}
    meta = json.loads(files['bridge_metadata.json'].decode('utf-8-sig'))
    recipe = json.loads(files['protocol/partition_protocol.json'].decode('utf-8-sig'))
    if (meta['topology'] != 'noniid-labeldir-fine' or recipe['split_seed'] != 42
            or recipe['schema'] != 'paper_dirichlet_reference_v1'):
        raise ValueError('Expected the registered seed42 ordinary-Dirichlet benchmark')
    # Bind to the already registered inputs, including the pre-recovery LF file.
    registered = sorted((source_root / 'jobs').glob('*.json'))
    if not registered:
        raise ValueError('No registered jobs in the source suite')
    source_hashes = {name: sha256(data) for name, data in files.items()}
    for path in registered:
        job = json.loads(path.read_text(encoding='utf-8-sig'))
        if job.get('seed') != 42 or job['protocol']['reference_files'] != source_hashes:
            raise ValueError('Original reference changed since job registration: ' + str(path))
    probe = 'protocol/probe_manifest.csv'
    restored, conversion = restore_probe_bytes(files[probe], meta['probe_manifest_sha256'])
    files[probe] = restored
    receipt = dict(schema='dirichlet_probe_recovery_v1', source_root=str(source_root),
                   source_reference_files=source_hashes,
                   recovered_reference_files={name: sha256(data) for name, data in files.items()},
                   probe_conversion=conversion, original_probe_sha256=meta['probe_manifest_sha256'],
                   changed_files=[probe] if conversion != 'unchanged' else [],
                   training_changes='none; only original probe file bytes restored',
                   recovery_script_sha256=sha256(Path(__file__).read_bytes()))
    reference = output_root / 'reference'
    receipt_path = output_root / 'probe_recovery.json'
    if reference.exists() or receipt_path.exists():
        if not receipt_path.is_file() or json.loads(receipt_path.read_text(encoding='utf-8')) != receipt:
            raise ValueError('Recovery output has another configuration; use a new output root')
        if any((reference / name).read_bytes() != data for name, data in files.items()):
            raise ValueError('Recovered reference changed; refusing to overwrite it')
    else:
        if output_root.exists() and any(output_root.iterdir()):
            raise ValueError('Recovery output must be empty for its first preparation')
        for name, data in files.items():
            target = reference / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
        receipt_path.write_text(json.dumps(receipt, indent=2) + '\n', encoding='utf-8')
    return reference, receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('prepare', 'plan', 'preflight', 'train', 'status', 'summary', 'pack'),
                        default='prepare')
    parser.add_argument('--source-root', type=Path,
                        default=Path('output/cifar100_LT/dirichlet_benchmarks_seed42_beta0.5_v1'))
    parser.add_argument('--output-root', type=Path,
                        default=Path('output/cifar100_LT/dirichlet_benchmarks_seed42_beta0.5_ours_probe_fix1'))
    parser.add_argument('--data-root', type=Path, default=Path('DATA'))
    parser.add_argument('--methods', nargs='+', choices=('a', 'ab'), default=['a', 'ab'])
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args(argv)
    try:
        if args.num_workers < 0 or len(set(args.methods)) != len(args.methods):
            raise ValueError('Require nonnegative workers and distinct methods')
        root = args.output_root.resolve()
        if args.stage in ('status', 'summary', 'pack'):
            command = [sys.executable, '-u', str(REPO / 'scripts/run_paper_benchmarks.py'),
                       '--stage', args.stage, '--output-root', str(root)]
        else:
            reference, receipt = recover_reference(args.source_root, root)
            print('Probe:', receipt['probe_conversion'], '; verified original SHA256:',
                  receipt['original_probe_sha256'], flush=True)
            print('New reference:', reference, flush=True)
            if args.stage == 'prepare':
                print('Preparation only; original suite unchanged; no training launched.')
                return
            command = [sys.executable, '-u', str(REPO / 'scripts/run_paper_benchmarks.py'),
                       '--stage', args.stage, '--methods', *args.methods, '--seed', '42',
                       '--reference-run', str(reference), '--data-root', str(args.data_root.resolve()),
                       '--output-root', str(root), '--num-workers', str(args.num_workers)]
            if args.resume:
                command.append('--resume')
        subprocess.run(command, cwd=REPO, check=True)
    except (OSError, ValueError, KeyError, subprocess.CalledProcessError) as error:
        parser.error(str(error))


if __name__ == '__main__':
    main()
