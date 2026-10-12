"""Standalone, standard-library-only search for A-projection source artifacts.

Reads directory entries, small JSON files, and optionally archive headers.
Does not load torch checkpoints, extract archives, or start training.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import os
from pathlib import Path, PurePosixPath
import socket
import tarfile
import time
import zipfile


SNAPSHOTS = (20, 50, 80, 90, 100)
ANCHORS = (20, 50, 80)
JSON_NAMES = {'bridge_metadata.json', 'control_config.json', 'sfra_config.json',
              'aggregation_solution.json', 'experiment_plan.json', 'projection_states.json'}
MARKERS = JSON_NAMES - {'experiment_plan.json'} | {'partition_manifest.csv'}
SKIP_NAMES = {'.git', '__pycache__', '.cache', '.venv', 'venv', 'node_modules'}
SPECIAL_DIRS = {'/proc', '/sys', '/dev', '/run'}
MAX_JSON = 8 * 1024 * 1024


def run_for_file(path):
    """Works for both native Paths and archive PurePosixPaths."""
    if path.name in MARKERS:
        return path.parent
    if path.name == 'base_model.pt' and path.parent.name == 'checkpoints':
        return path.parent.parent
    if path.name == 'state.pt':
        for parent in path.parents:
            if parent.name in ('events', 'bridge_dumps'):
                return parent.parent
    if path.suffix in ('.pt', '.pth', '.ckpt') and path.parent.name == 'checkpoints':
        return path.parent.parent
    return None


def inspect_run(run, file_info, read_json, *, location, archive=False, found_tensors=()):
    """Presence screen only; the diagnostic runner validates tensors/protocol."""
    issues, checks = [], []

    def read(relative):
        path = run / relative
        if not file_info(path)['present']:
            return {}
        try:
            value = read_json(path)
            if not isinstance(value, dict):
                raise ValueError('expected a JSON object')
            return value
        except (OSError, ValueError, TypeError, KeyError, RuntimeError) as error:
            issues.append('%s: %s' % (path, error))
            return {}

    def check(label, relative, group):
        info = file_info(run / relative)
        checks.append(dict(artifact=label, group=group, **info))
        return info['present']

    for name in ('bridge_metadata.json', 'control_config.json', 'sfra_config.json', 'partition_manifest.csv'):
        check(name, name, 'metadata')
    solution = 'aggregation_solution.json'
    if not file_info(run / solution)['present']:
        solution = '../../../experiment_plan.json'
    check('aggregation_spec', solution, 'metadata')
    meta, config, sfra = read('bridge_metadata.json'), read('control_config.json'), read('sfra_config.json')
    spec = read(solution)
    spec = spec.get('aggregation', spec)
    if not isinstance(spec, dict):
        issues.append('Invalid aggregation specification')
        spec = {}
    rule = spec.get('rule', 'joint') if spec else None
    rule_source = 'aggregation_spec' if rule else 'directory_hint'
    if not rule:
        rule = next((p for p in reversed(run.parts) if p in ('fedavg', 'tailrw16', 'joint')), 'unknown')
    expected = dict(method='s', seed=42, topology='client-longtail', la_tau=1., a_lr_mult=1.,
                    normal_trainable_factor='B', extra_trainable_factor='A', control_enabled=False,
                    extra_a_lr=.001, extra_weight_decay=0., candidate_rounds=list(range(1, 91)),
                    head_ids=list(range(20)), middle_ids=list(range(20, 80)), tail_ids=list(range(80, 100)))
    mismatches = ['control_config.' + k for k, v in expected.items() if config.get(k) != v]
    if meta.get('method') != 's':
        mismatches.append('bridge_metadata.method')
    if sfra.get('variant') != 's' or sfra.get('retention_weight') != 0 or sfra.get('correction_steps') != 0 or sfra.get('b_transfer'):
        mismatches.append('sfra ordinary-A settings')
    target = not mismatches and rule in ('fedavg', 'tailrw16') and rule_source == 'aggregation_spec'
    mapping = read('projection_states.json')

    def event(rnd, phase='refresh_A'):
        return 'events/r%03d_c000_main_%s/state.pt' % (rnd, phase)

    def mapped(category, rnd):
        entries = mapping.get(category, {})
        if not isinstance(entries, dict):
            issues.append('Invalid projection_states category: ' + category)
            return None
        value = entries.get(str(rnd))
        if isinstance(value, str):
            return value
        if isinstance(value, dict) and isinstance(value.get('path'), str):
            return value['path']
        if value is not None:
            issues.append('Invalid projection_states entry: %s/%s' % (category, rnd))
        return None

    def committed(rnd):
        return mapped('committed', rnd) or ('checkpoints/base_model.pt' if rnd == 0 else event(rnd, 'refresh_A' if rnd <= 90 else 'normal_B'))

    check('frozen_base_and_initial', 'checkpoints/base_model.pt', 'model')
    # An explicit committed-0 override is honored by StateIndex as well.
    if mapped('committed', 0):
        check('committed_000', committed(0), 'model')
    for rnd in SNAPSHOTS:
        check('committed_%03d' % rnd, committed(rnd), 'model')
    anchors, proposal_files = [], []
    for rnd in ANCHORS:
        explicit = mapped('pre_a', rnd)
        paths = [explicit] if explicit else [event(rnd), event(rnd, 'normal_B')]
        direct = next((p for p in paths if file_info(run / p)['present']), None)
        recovered = (not direct and file_info(run / committed(rnd - 1))['present'] and
                     file_info(run / committed(rnd))['present'])
        anchor = dict(round=rnd, present=bool(direct or recovered),
                      recovery='direct' if direct else 'A_previous+B_current' if recovered else 'missing',
                      paths=[str(run / direct)] if direct else [str(run / committed(rnd - 1)), str(run / committed(rnd))])
        anchors.append(anchor)
        proposal_files.append(dict(round=rnd, **file_info(run / event(rnd)),
                                   payload_checked=False, note='local_factor_deltas not inspected'))
    missing_meta = [c['artifact'] for c in checks if c['group'] == 'metadata' and not c['present']]
    missing_models = [c['artifact'] for c in checks if c['group'] == 'model' and not c['present']]
    if rule != 'tailrw16':
        missing_models += ['pre_a_%03d' % a['round'] for a in anchors if not a['present']]
    files_present = not missing_meta and not missing_models
    if files_present and target and not issues:
        status = 'ARCHIVE_FILES_PRESENT' if archive else 'FILES_PRESENT'
    elif not found_tensors and not any(c['present'] for c in checks if c['group'] == 'model'):
        status = 'METADATA_ONLY'
    else:
        status = 'PARTIAL_OR_OTHER_PROTOCOL'
    resolved_args = meta.get('resolved_args', {})
    recorded = resolved_args.get('output_dir') if isinstance(resolved_args, dict) else None
    return dict(location=location, archive=archive, rule=rule, rule_source=rule_source, status=status,
                target_protocol_screen=target, protocol_mismatches=mismatches,
                required_files_present=files_present, missing_metadata=missing_meta,
                missing_models=missing_models, checks=checks, pre_a=anchors, proposal_event_files=proposal_files,
                tensor_file_count=len(found_tensors), tensor_examples=sorted(map(str, found_tensors))[:12],
                recorded_server_run=recorded, recorded_server_run_exists=bool(recorded and Path(recorded).is_absolute() and Path(recorded).is_dir()),
                issues=issues, tensors_validated=False)


def archive_runs(path):
    """Inspect headers/small JSON only; never extract or unpickle members."""
    entries, jsons, candidates = {}, {}, {}

    def add(name, size, reader):
        member = PurePosixPath(name)
        if member.is_absolute() or '..' in member.parts:
            return
        parent = run_for_file(member)
        if parent is not None:
            candidates.setdefault(parent, [])
            if member.suffix in ('.pt', '.pth', '.ckpt'):
                candidates[parent].append(member)
        # Keep just relevant headers, not every dataset image in an archive.
        if parent is not None or member.name == 'experiment_plan.json' or member.suffix in ('.pt', '.pth', '.ckpt'):
            entries[str(member)] = size
        if member.name in JSON_NAMES:
            try:
                if size > MAX_JSON:
                    raise ValueError('JSON exceeds 8 MiB limit')
                jsons[str(member)] = json.loads(reader().decode('utf-8-sig'))
            except (ValueError, UnicodeError, OSError, RuntimeError) as error:
                jsons[str(member)] = ValueError(str(error))

    if path.name.lower().endswith('.zip'):
        with zipfile.ZipFile(path) as archive:
            for entry in archive.infolist():
                if not entry.is_dir():
                    add(entry.filename, entry.file_size, lambda e=entry: archive.read(e))
    else:
        # Streaming also supports compressed tar files without repeated decompression.
        with tarfile.open(path, 'r|*') as archive:
            for entry in archive:
                if entry.isfile():
                    add(entry.name, entry.size, lambda e=entry: archive.extractfile(e).read(MAX_JSON + 1))

    def key(path):
        return os.path.normpath(str(path)).replace('\\', '/')

    def info(path):
        size = entries.get(key(path))
        return dict(path=str(path), present=size is not None and size > 0, bytes=size)

    def read(path):
        value = jsons[key(path)]
        if isinstance(value, Exception):
            raise value
        return value

    return [inspect_run(run, info, read, location=str(path) + '::' + str(run), archive=True, found_tensors=tensors)
            for run, tensors in sorted(candidates.items())]


def scan(roots, archives=False, progress=print, skip_names=SKIP_NAMES):
    errors, error_count, candidates, archive_paths, visited = [], [0], {}, set(), set()
    counts = dict(directories=0, files=0, archives_checked=0)
    started, last_progress = time.monotonic(), [time.monotonic()]
    root_reports = []

    def error(path, exc):
        error_count[0] += 1
        if len(errors) < 100:
            errors.append(dict(path=str(path), error=str(exc)))

    def info(path):
        try:
            stat = path.stat()
            return dict(path=str(path), present=path.is_file() and stat.st_size > 0, bytes=stat.st_size)
        except FileNotFoundError:
            return dict(path=str(path), present=False, bytes=None)
        except OSError as exc:
            error(path, exc)
            return dict(path=str(path), present=False, bytes=None, error=str(exc))

    def read(path):
        with path.open('rb') as stream:
            data = stream.read(MAX_JSON + 1)
        if len(data) > MAX_JSON:
            raise ValueError('JSON exceeds 8 MiB limit')
        return json.loads(data.decode('utf-8-sig'))

    for requested in roots:
        root = Path(requested).expanduser().absolute()
        root_report = dict(path=str(root), status='scanned')
        root_reports.append(root_report)
        if not root.is_dir():
            root_report['status'] = 'missing_or_not_readable_directory'
            progress('SKIP root: ' + str(root))
            continue
        progress('SCAN root: ' + str(root))
        before_errors = error_count[0]
        for directory, dirs, files in os.walk(root, followlinks=True, onerror=lambda exc: error(exc.filename, exc)):
            path = Path(directory)
            try:
                stat = path.stat()
                # Follow mounted/symlinked storage, but break cycles and duplicate roots.
                identity = (stat.st_dev, stat.st_ino) if stat.st_ino else str(path.resolve())
            except OSError as exc:
                error(path, exc)
                dirs[:] = []
                continue
            if identity in visited or str(path.resolve()) in SPECIAL_DIRS:
                dirs[:] = []
                continue
            visited.add(identity)
            dirs[:] = [d for d in dirs if d not in skip_names]
            counts['directories'] += 1
            counts['files'] += len(files)
            for name in files:
                file = path / name
                run = run_for_file(file)
                if run is not None:
                    candidates.setdefault(run, [])
                    if file.suffix in ('.pt', '.pth', '.ckpt'):
                        candidates[run].append(file)
                lower = name.lower()
                if archives and lower.endswith(('.zip', '.tar', '.tar.gz', '.tgz', '.tar.bz2', '.tar.xz')) and not lower.endswith('.pth.tar'):
                    archive_paths.add(file)
            if time.monotonic() - last_progress[0] >= 10:
                progress('... %d directories, %d files, %d run candidates' % (counts['directories'], counts['files'], len(candidates)))
                last_progress[0] = time.monotonic()
        if error_count[0] > before_errors:
            root_report['status'] = 'scanned_with_errors'
    runs = []
    for run, tensors in sorted(candidates.items()):
        try:
            runs.append(inspect_run(run, info, read, location=str(run), found_tensors=tensors))
        except (OSError, ValueError, TypeError, KeyError, AttributeError) as exc:
            error(run, exc)
    for path in sorted(archive_paths):
        progress('SCAN archive headers: ' + str(path))
        counts['archives_checked'] += 1
        try:
            runs.extend(archive_runs(path))
        except (OSError, ValueError, TypeError, KeyError, AttributeError, RuntimeError, EOFError, tarfile.TarError, zipfile.BadZipFile) as exc:
            error(path, exc)
    return dict(schema='a_projection_artifact_scan_v1', host=socket.gethostname(),
                time_utc=datetime.now(timezone.utc).isoformat(), roots=root_reports,
                skipped_directory_names=sorted(skip_names), skipped_special_directories=sorted(SPECIAL_DIRS),
                archives_scanned=archives, counts=counts, seconds=time.monotonic() - started,
                runs=runs, errors=errors, error_count=error_count[0],
                limitations=['Presence and small configuration screen only; no tensor loading or hash/partition validation.',
                             'FILES_PRESENT is a candidate for diagnostic preflight/smoke, not proof of valid weights.',
                             'Missing files are missing only in scanned readable locations; this does not prove deletion.',
                             'Archive members are not extracted; historical local proposal payloads remain unverified.'])


def format_report(report):
    lines = ['A PROJECTION ARTIFACT SEARCH', 'Host: ' + report['host'],
             'Time UTC: ' + report['time_utc'],
             'Counts: ' + json.dumps(report['counts']),
             'Status counts: ' + json.dumps(dict(Counter(r['status'] for r in report['runs']))),
             'Read errors: %d (first 100 retained in JSON)' % report['error_count'], '', 'Search roots:']
    lines += ['  [%s] %s' % (r['status'], r['path']) for r in report['roots']]
    lines += ['', 'FILES_PRESENT = file presence only; tensor/protocol validation still required.',
              'METADATA_ONLY = no recognized model files found in this run.',
              'ARCHIVE_FILES_PRESENT = candidate files inside archive; not yet extracted.',
              'PARTIAL_OR_OTHER_PROTOCOL = missing artifacts or a different/unverified experiment.', '']
    for run in sorted(report['runs'], key=lambda r: (not r['target_protocol_screen'], r['status'] != 'FILES_PRESENT', r['location'])):
        lines += ['[%s] %s' % (run['status'], run['rule']), '  Location: ' + run['location'],
                  '  Target ordinary-A config screen: ' + str(run['target_protocol_screen']),
                  '  Tensor files: %d (payloads NOT checked)' % run['tensor_file_count'],
                  '  Missing metadata: ' + (', '.join(run['missing_metadata']) or 'none'),
                  '  Missing models: ' + (', '.join(run['missing_models']) or 'none')]
        if run['recorded_server_run']:
            lines.append('  Recorded original run: %s (exists here: %s)' % (run['recorded_server_run'], run['recorded_server_run_exists']))
        if run['protocol_mismatches']:
            lines.append('  Config mismatch/missing: ' + ', '.join(run['protocol_mismatches']))
        lines += ['  Read issue: ' + issue for issue in run['issues']]
        lines.append('')
    if not report['runs']:
        lines.append('No recognized run artifacts found in scanned locations.')
    lines += ['Limitations:'] + ['  ' + note for note in report['limitations']]
    if report['errors']:
        lines += ['', 'First read errors:'] + ['  %s: %s' % (e['path'], e['error']) for e in report['errors'][:10]]
    return '\n'.join(lines) + '\n'


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--roots', nargs='+', type=Path, default=[Path.cwd()], help='Recursively search these storage/repository roots')
    parser.add_argument('--archives', action='store_true', help='Also inspect ZIP/TAR headers; compressed TAR scanning can be slow')
    parser.add_argument('--output-prefix', type=Path, default=Path('output') / ('a_projection_files_' + socket.gethostname()))
    args = parser.parse_args(argv)
    report = scan(args.roots, args.archives, progress=lambda s: print(s, flush=True))
    prefix = args.output_prefix.expanduser().absolute()
    prefix.parent.mkdir(parents=True, exist_ok=True)
    json_path, text_path = Path(str(prefix) + '.json'), Path(str(prefix) + '.txt')
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    summary = format_report(report)
    text_path.write_text(summary, encoding='utf-8')
    print(summary)
    print('REPORT JSON:', json_path)
    print('REPORT TXT: ', text_path)


if __name__ == '__main__':
    main()
