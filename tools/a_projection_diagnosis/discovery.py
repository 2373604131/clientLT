"""Find relocated ordinary-A runs without substituting a different experiment."""
from pathlib import Path

from tools.a_projection_diagnosis.protocol import ANCHORS, StateIndex, check_source

DEFAULT_SOURCE_ROOT = Path('output/cifar100_LT/plain_a_aggregation_v1_parallel8')
# Recorded in the seed42 archives, not assumed to be mounted on every server.
HISTORICAL_REPOS = (Path('/data/run01/scze847/yzh/code/clientLT'),)


def artifact_required(row, part='all', rounds=ANCHORS):
    name = row['artifact']
    return (name == 'frozen_base'
            or part in ('all', 'e1') and name.startswith('committed_')
            or part in ('all', 'events') and name.startswith('pre_a_') and int(name[-3:]) in rounds)


def candidate_runs(root, rule):
    """Bounded known layouts; never recursively crawl datasets or tensor caches."""
    root = Path(root)
    suffix = Path(rule) / 'runs/seed42/plain_a'
    yield root / suffix
    if root.name == 'plain_a':
        yield root
    for prefix in (root, root / 'output', root / 'cifar100_LT', root / 'output/cifar100_LT'):
        for suite in sorted(prefix.glob('plain_a_aggregation*')):
            if suite.is_dir():
                yield suite / suffix
                # Result archive: suite_results/suite/{rule}/runs/seed42/plain_a.
                for nested in sorted(suite.glob('plain_a_aggregation*')):
                    if nested.is_dir():
                        yield nested / suffix


def discover_source(rule, source_root, override=None, *, auto=False, search_roots=(),
                    repo=None, historical_repos=None, part='all', rounds=ANCHORS):
    """Explicit input wins. Auto mode follows archive provenance and requires
    an unambiguous, protocol-valid run; it never selects by accuracy or recency.
    """
    requested = Path(override) if override is not None else Path(source_root) / rule / 'runs/seed42/plain_a'
    report = dict(requested=str(requested.resolve()), automatic=bool(auto and override is None),
                  checked=[], candidates=[], rejected=[])
    source_by_path = {}

    def inspect(path):
        path = Path(path).resolve()
        label = str(path)
        if label in report['checked']:
            return None
        report['checked'].append(label)
        if not (path / 'bridge_metadata.json').is_file():
            return None
        try:
            source = check_source(path, rule)
            inventory = StateIndex(source).inventory(need_events=rule == 'fedavg')
            needed = [r for r in inventory if artifact_required(r, part, rounds)]
            missing = [r['artifact'] for r in needed if not r['available']]
            status = dict(path=label, required_models_ready=not missing, missing_models=missing,
                          recorded_server_run=source['meta'].get('resolved_args', {}).get('output_dir'))
            source_by_path[label] = source
            report['candidates'].append(status)
            return status
        except (OSError, ValueError, KeyError) as error:
            report['rejected'].append(dict(path=label, reason=str(error)))
            return None

    first = inspect(requested)
    if override is not None or not auto:
        if first:
            return source_by_path[first['path']], dict(report, selected=first['path'])
        reason = report['rejected'][0]['reason'] if report['rejected'] else 'bridge_metadata.json is absent'
        return None, dict(report, error='Explicit source unavailable: %s (%s). Correct --source-root/--%s-run; explicit paths are not silently replaced.'
                         % (requested.resolve(), reason, rule))
    if first and first['required_models_ready']:
        return source_by_path[first['path']], dict(report, selected=first['path'])

    roots = [Path(repo or Path.cwd()) / 'output', Path(source_root)] + list(map(Path, search_roots))
    for old in HISTORICAL_REPOS if historical_repos is None else historical_repos:
        # A POSIX provenance path should not become G:\data\... on Windows.
        if Path(old).is_absolute():
            roots.append(Path(old))
    for root in roots:
        for path in candidate_runs(root, rule):
            inspect(path)

    # Read only protocol-validated candidates; a relocated lightweight archive
    # can tell us where its original tensors were saved.
    for candidate in list(report['candidates']):
        old = candidate['recorded_server_run']
        if old and Path(old).is_absolute():
            inspect(Path(old))

    ready = [c for c in report['candidates'] if c['required_models_ready']]
    if len(ready) == 1:
        chosen = ready[0]
    elif len(ready) > 1:
        paths = '\n'.join(c['path'] for c in ready)
        return None, dict(report, error='Multiple complete %s runs found; specify --%s-run explicitly:\n%s' % (rule, rule, paths))
    elif first:
        chosen = first
    elif len(report['candidates']) == 1:
        chosen = report['candidates'][0]
    elif report['candidates']:
        paths = '\n'.join(c['path'] for c in report['candidates'])
        return None, dict(report, error='Found %s metadata, but no run has all required model tensors. These may be lightweight result archives. '
                         'Check source_discovery.json and copy original checkpoints/events before running:\n%s' % (rule, paths))
    else:
        return None, dict(report, error='No ordinary-A %s metadata found at the requested or known locations. '
                         'The current repository may differ from the historical training repository. '
                         'Use --search-root PATH for another repository/output directory, or --%s-run PATH for the exact run. '
                         'See source_discovery.json for checked paths; this does not establish that weights were deleted.' % (rule, rule))
    return source_by_path[chosen['path']], dict(report, selected=chosen['path'])
