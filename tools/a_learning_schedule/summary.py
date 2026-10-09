"""Read-only result aggregation; primary endpoint is fixed at offsets 8--10."""
import statistics
from pathlib import Path

from tools.a_learning_schedule.protocol import (
    ARMS, CONTRASTS, PRIMARY, job_root, read_csv, read_json, write_csv, write_json,
)

GROUPS = ('overall', 'head20', 'middle60', 'tail20')


def status_rows(output_root, seeds, origins, rounds):
    rows = []
    for seed in seeds:
        for origin in origins:
            for rnd in rounds:
                root = job_root(output_root, seed, origin, rnd)
                complete = (root / 'complete.json').is_file()
                required = ('curves.csv', 'events.csv', 'budget.csv', 'norm_probes.csv', 'anchor_info.json')
                missing = [name for name in required if not (root / name).is_file()]
                state = 'complete' if complete and not missing else ('incomplete' if root.exists() else 'missing')
                nodes = len(list((root / 'nodes').glob('*.pt')))
                rows.append(dict(seed=seed, origin=origin, anchor_round=rnd, status=state,
                                 saved_training_nodes=nodes, reason=', '.join(missing) if complete else '', run=str(root)))
    return rows


def primary_means(curves):
    lookup = {(r['arm'], int(r['h'])): r for r in curves}
    if len(lookup) != len(ARMS) * 11 or len(curves) != len(lookup):
        raise ValueError('Expected four arms with exactly offsets 0..10')
    result = {}
    fields = [k for k in curves[0] if k.startswith(('test_', 'feedback_'))]
    for arm in ARMS:
        if not all((arm, h) in lookup for h in range(11)):
            raise ValueError('Missing observation for ' + arm)
        result[arm] = {key: statistics.mean(float(lookup[arm, h][key]) for h in PRIMARY) for key in fields}
    return result


def summarize(output_root, seeds, origins, rounds, plots=True):
    output_root = Path(output_root)
    destination = output_root / 'analysis'
    statuses = status_rows(output_root, seeds, origins, rounds)
    write_csv(destination / 'status.csv', statuses)
    curves, per_run, pairs, costs = [], [], [], []
    for status in statuses:
        if status['status'] != 'complete':
            continue
        identity = {k: status[k] for k in ('seed', 'origin', 'anchor_round')}
        root = Path(status['run'])
        current = read_csv(root / 'curves.csv')
        means = primary_means(current)
        curves.extend(dict(**identity, **r) for r in current)
        for arm in ARMS:
            per_run.append(dict(**identity, arm=arm, **means[arm]))
        for a, b in CONTRASTS:
            pairs.append(dict(**identity, comparison=a + '-' + b,
                              **{key: means[a][key] - means[b][key] for key in means[a]}))
        info = read_json(root / 'complete.json')
        budget = read_csv(root / 'budget.csv')
        costs.append(dict(**identity, actual_optimizer_steps=info['actual_optimizer_steps'],
                          logical_steps_per_arm=info['logical_optimizer_steps_per_arm'],
                          unique_training_nodes=info['unique_training_nodes'],
                          training_client_seconds=sum(float(r['seconds']) for r in budget)))
    write_csv(destination / 'curves.csv', curves)
    write_csv(destination / 'per_run.csv', per_run)
    write_csv(destination / 'paired.csv', pairs)
    write_csv(destination / 'costs.csv', costs)
    # Never use anchors or repeated rounds as independent training seeds.
    across = []
    for origin in origins:
        for rnd in rounds:
            for a, b in CONTRASTS:
                selected = [r for r in pairs if r['origin'] == origin and r['anchor_round'] == rnd and r['comparison'] == a + '-' + b]
                if not selected:
                    continue
                row = dict(origin=origin, anchor_round=rnd, comparison=a + '-' + b,
                           seed_count=len(selected), seeds=' '.join(str(r['seed']) for r in selected))
                for group in GROUPS:
                    values = [r['test_' + group] for r in selected]
                    row[group + '_mean_pp'] = statistics.mean(values)
                    row[group + '_std_pp'] = statistics.stdev(values) if len(values) > 1 else None
                across.append(row)
    write_csv(destination / 'paired_across_seeds.csv', across)
    plot_error = None
    if plots and curves:
        try:
            make_plots(destination, curves, pairs, origins, rounds)
        except ImportError as error:
            plot_error = str(error)
    lines = ['# A learning opportunities and spacing', '',
             'Completed anchors: %d/%d. Four arms per anchor. Primary endpoint: mean of offsets 8, 9, 10.'
             % (sum(r['status'] == 'complete' for r in statuses), len(statuses)), '',
             'Positive paired accuracy differences are percentage points. New-correct/forgotten values are sample counts relative to the shared anchor.',
             'E2 and E3 are distinct training histories. Their anchors are not independent seeds. Single-seed results have no between-seed error bar.',
             'Before offset 5 the two AA arms have unequal completed intervention counts; the primary comparison is after both doses.',
             'Immediate norm-matched probes are diagnostic only; they are not norm-matched ten-round trajectories.', '',
             '| Seed | Source | Anchor | Comparison | Overall | Head20 | Middle60 | Tail20 |',
             '|---|---|---:|---|---:|---:|---:|---:|']
    for row in pairs:
        lines.append('| %d | %s | %d | %s | %+.4f | %+.4f | %+.4f | %+.4f |' % (
            row['seed'], row['origin'], row['anchor_round'], row['comparison'],
            *(row['test_' + group] for group in GROUPS)))
    if plot_error:
        lines += ['', 'Plot generation unavailable: ' + plot_error]
    destination.mkdir(parents=True, exist_ok=True)
    (destination / 'report.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    write_json(destination / 'summary.json', dict(completed=sum(r['status'] == 'complete' for r in statuses),
               expected=len(statuses), seeds=list(seeds), origins=list(origins), rounds=list(rounds), plot_error=plot_error))
    return statuses


def make_plots(destination, curves, pairs, origins, rounds):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors = {'BB': '#7a7a7a', 'AB': '#247ba0', 'AA_short': '#d95d39', 'AA_long': '#29975d'}
    seeds = sorted(set(int(r['seed']) for r in curves))
    for origin in origins:
        selected = [r for r in curves if r['origin'] == origin]
        if not selected:
            continue
        fig, axes = plt.subplots(2, len(rounds), figsize=(4.2 * len(rounds), 6.4), squeeze=False)
        for col, rnd in enumerate(rounds):
            for row_index, group in enumerate(('overall', 'tail20')):
                ax = axes[row_index, col]
                for arm in ARMS:
                    for seed in seeds:
                        points = sorted((r for r in selected if int(r['anchor_round']) == rnd and int(r['seed']) == seed and r['arm'] == arm), key=lambda r: int(r['h']))
                        if not points:
                            continue
                        ax.plot([int(r['h']) for r in points], [float(r['test_' + group]) for r in points],
                                color=colors[arm], label=arm if seed == seeds[0] else None,
                                alpha=1 if len(seeds) == 1 else .6, linewidth=1.6)
                ax.axvline(1, color='gray', linestyle=':', alpha=.5)
                ax.axvline(5, color='gray', linestyle=':', alpha=.5)
                ax.axvspan(8, 10, color='#dddddd', alpha=.3)
                ax.set(title='%s, round %d' % (origin.upper(), rnd), xlabel='B continuation rounds', ylabel=group + ' accuracy (%)')
                ax.grid(alpha=.15)
        axes[0, 0].legend(fontsize=8)
        fig.tight_layout()
        for extension in ('png', 'pdf'):
            fig.savefig(destination / (origin + '_trajectories.' + extension), dpi=180)
        plt.close(fig)
    # Paired contrasts and learning/forgetting counts, showing every seed as a point.
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    labels = []
    for i, (a, b) in enumerate(CONTRASTS):
        matching = [r for r in pairs if r['comparison'] == a + '-' + b]
        for j, row in enumerate(matching):
            label = '%s/%s/r%s' % (row['seed'], row['origin'], row['anchor_round'])
            labels.append(label)
            axes[i].scatter(row['test_overall'], row['test_tail20'], label=label, s=26)
        axes[i].axhline(0, color='gray', linewidth=.7)
        axes[i].axvline(0, color='gray', linewidth=.7)
        axes[i].set(title=a + ' - ' + b, xlabel='Overall change (pp)', ylabel='Tail20 change (pp)')
    if pairs:
        axes[-1].legend(fontsize=6, loc='best')
    fig.tight_layout()
    for extension in ('png', 'pdf'):
        fig.savefig(destination / ('paired_tradeoffs.' + extension), dpi=180)
    plt.close(fig)
    for origin in origins:
        fig, axes = plt.subplots(1, len(rounds), figsize=(4.2 * len(rounds), 3.8), squeeze=False)
        for col, rnd in enumerate(rounds):
            ax = axes[0, col]
            for index, arm in enumerate(ARMS):
                points = [r for r in curves if r['origin'] == origin and int(r['anchor_round']) == rnd and r['arm'] == arm and int(r['h']) in PRIMARY]
                if not points:
                    continue
                learned = statistics.mean(float(r['test_tail20_new_correct']) for r in points)
                forgotten = statistics.mean(float(r['test_tail20_forgotten']) for r in points)
                ax.bar(index - .18, learned, .36, color='#29975d', label='New correct' if index == 0 else None)
                ax.bar(index + .18, forgotten, .36, color='#d95d39', label='Forgotten' if index == 0 else None)
            ax.set_xticks(range(4))
            ax.set_xticklabels(ARMS, rotation=20)
            ax.set(title='%s r%d' % (origin, rnd), ylabel='Tail sample counts; offsets 8-10 mean')
        axes[0, 0].legend(fontsize=8)
        fig.tight_layout()
        for extension in ('png', 'pdf'):
            fig.savefig(destination / (origin + '_learning_forgetting.' + extension), dpi=180)
        plt.close(fig)
