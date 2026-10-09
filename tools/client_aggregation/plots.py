"""Standalone figures from audited records; no model access or selection."""
from pathlib import Path


def render_plots(out, tables):
    if not tables['curves']:
        return [], 'No completed audited trajectories yet.'
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        return [], 'Matplotlib is unavailable; CSV and Markdown reports are complete.'
    out = Path(out)
    colors = dict(frozen='#3977A8', ab='#D26B30')
    labels = dict(frozen='Frozen LoRA A', ab='Full A+B')
    groups = [('overall_acc', 'Overall'), ('head20_acc', 'Head20'),
              ('middle60_acc', 'Middle60'), ('bottom20_tail_acc', 'Tail20')]
    arms = [arm for arm in colors if any(r['arm'] == arm for r in tables['curves'])]
    paths = []

    def save(fig, name):
        fig.tight_layout()
        for extension in ('png', 'svg'):
            path = out / (name + '.' + extension)
            fig.savefig(path, dpi=160, bbox_inches='tight')
            paths.append(path.name)
        plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    for ax, (metric, title) in zip(axes.flat, groups):
        for arm in arms:
            rows = [r for r in tables['curves'] if r['arm'] == arm]
            ax.plot([int(r['round']) for r in rows], [float(r[metric]) for r in rows],
                    color=colors[arm], label=labels[arm], linewidth=1.7)
        ax.axvspan(81, 100, color='gray', alpha=.1)
        ax.set(title=title, xlabel='Committed round', ylabel='Accuracy (%)', xlim=(0, 100))
        ax.grid(alpha=.2); ax.legend(fontsize=8)
    fig.suptitle('Same aggregation, seed 42; shaded: prespecified reporting window')
    save(fig, 'learning_curves')

    fig, axes = plt.subplots(2, 3, figsize=(13, 7), squeeze=False)
    for col, (metric, title) in enumerate(groups[1:]):
        for row, (field, kind) in enumerate((('never_correct_so_far', 'Never correct so far'),
                                            ('previously_correct_now_wrong', 'Previously correct, now wrong'))):
            ax = axes[row, col]
            for arm in arms:
                rows = [r for r in tables['sample_dynamics'] if r['arm'] == arm and r['group'] == metric]
                ax.plot([r['round'] for r in rows], [r[field] for r in rows],
                        color=colors[arm], label=labels[arm], linewidth=1.7)
            ax.set(title=title + ': ' + kind, xlabel='Committed round', ylabel='Test examples', xlim=(0, 100))
            ax.grid(alpha=.2); ax.legend(fontsize=8)
    fig.suptitle('Error trajectories within each class group (different group sizes)')
    save(fig, 'remaining_errors')

    if tables['stage_summary']:
        operations = [('ordinary_B_learning', 'Ordinary B'),
            ('B_shared_transfer', 'Shared B supplement'), ('ordinary_A_learning', 'Ordinary A'),
            ('A_functional_correction', 'A correction'),
            ('B_aggregation_same_local_updates', 'New vs sample weights: B'),
            ('A_aggregation_same_local_updates', 'New vs sample weights: A')]
        fig, axes = plt.subplots(1, 3, figsize=(14, 5), sharey=True)
        for ax, (metric, title) in zip(axes, groups[1:]):
            for arm in arms:
                records = {r['operation']: r for r in tables['stage_summary']
                           if r['arm'] == arm and r['group'] == metric}
                positions = [i + (-.18 if arm == 'frozen' else .18)
                             for i, (operation, _) in enumerate(operations) if operation in records]
                values = [records[operation]['mean_delta_accuracy']
                          for operation, _ in operations if operation in records]
                ax.barh(positions, values, height=.34, color=colors[arm], label=labels[arm])
            ax.set_yticks(range(len(operations)), [label for _, label in operations])
            ax.axvline(0, color='gray', linewidth=.7)
            ax.set(title=title, xlabel='Immediate accuracy change (pp)')
            ax.grid(axis='x', alpha=.2); ax.legend(fontsize=8)
        axes[0].invert_yaxis()
        fig.suptitle('Descriptive means at selected events; not long-term component effects')
        save(fig, 'stage_changes')
    return paths, ''
