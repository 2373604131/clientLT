"""Adapted from Vivid basic.line; actual single-run curves, no invented bands."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from figure_common import HERE, setup, save, pu

setup()
curves = pd.read_csv(HERE / 'data/training_curves.csv')
summary = pd.read_csv(HERE / 'data/retention_summary.csv').set_index(['partition', 'method'])
fig, axes = plt.subplots(1, 3, figsize=(7.2, 3.8), sharex=True, sharey=True)
fig.subplots_adjust(left=.085, right=.985, bottom=.31, top=.71, wspace=.16)
methods = [('j', '(a) J: joint A/B updates'), ('s', '(b) S: frequent A updates'),
           ('e3', '(c) E3: periodic A updates')]
partitions = [('client-longtail', 'Client-LT'), ('noniid-labeldir-fine', 'Class-wise Dirichlet')]
markers, styles = ['o', 's'], ['-', '--']
colors = [pu.PALETTE[2], pu.PALETTE[0]]
for ax, (method, title) in zip(axes, methods):
    for i, (partition, name) in enumerate(partitions):
        run = curves[(curves.partition == partition) & (curves.method == method)]
        x, y = run['round'].to_numpy(), run.bottom20_tail_acc.to_numpy()
        is_focus = partition == 'client-longtail'
        # One run per condition: template confidence bands intentionally omitted.
        ax.plot(x, y, linestyle=styles[i], marker=markers[i], markevery=10,
                color=colors[i], linewidth=1.6 if is_focus else 1.3,
                markersize=3.7, markeredgecolor='white', markeredgewidth=.55,
                label=name, zorder=3)
        max_idx = np.argmax(y)
        ax.scatter(x[max_idx], y[max_idx], s=48, color=colors[i],
                   edgecolor='white', linewidth=.6, zorder=4, marker='*')
        r = summary.loc[(partition, method)]
        ax.text(.03, -.47 - i * .14,
                f'{name}: {r.peak_to_final_pp:.2f} pp', transform=ax.transAxes,
                color=colors[i], fontsize=8)
    ax.set_title(title, loc='left', pad=8, fontsize=8)
    ax.set_xlabel('Communication round')
    ax.set_xlim(0, 100)
    ax.set_xticks([0, 50, 100])
    ax.set_ylim(58, 75)
    ax.set_yticks([60, 65, 70, 75])
    ax.grid(axis='y', alpha=.22, linestyle='--', color=pu.COLORS['grid'])
    ax.tick_params(length=3)
axes[0].set_ylabel('Tail20 accuracy (%)')
fig.legend(*axes[0].get_legend_handles_labels(), loc='upper center',
           bbox_to_anchor=(.53, .855), ncol=2, frameon=False)
fig.text(.085, .97, 'Tail accuracy decline depends on the update strategy', fontsize=11,
         fontweight='bold', va='top')
fig.text(.085, .91, 'Same sample pool | rank-4 LoRA + LA | split / training seed 42 | no smoothing', fontsize=8)
fig.text(.085, .172, 'Peak-to-final decrease (percentage points):', fontsize=8, fontweight='bold')
save(fig, 'retention_comparison')
