"""Vivid basic.line adapted to the paired J runs, with no simulated intervals.

Retains the recipe's sequence loop, white-edged markers, main-line emphasis,
true maximum and arrow annotation. No smoothing or inferred uncertainty.
"""
import numpy as np
from figure_common import pu


def draw_curves(ax, runs, *, lang='en'):
    markers = ['s', 'o']
    styles = ['--', '-']
    colors = [pu.PALETTE[0], pu.PALETTE[2]]
    labels = ['Dirichlet', 'Client-LT']
    for i, (name, run) in enumerate(runs.items()):
        x = run['round'].to_numpy()
        y = run['bottom20_tail_acc'].to_numpy()
        assert np.array_equal(x, np.arange(101))
        is_focus = name == 'client-longtail'
        # A single paired seed: the template's simulated bands are removed.
        ax.plot(x, y, linestyle=styles[i], marker=markers[i], markevery=20,
                color=colors[i], linewidth=1.9 if is_focus else 1.5,
                markersize=3.5, markeredgecolor='white', markeredgewidth=.65,
                label=labels[i], zorder=3)
        max_idx = np.argmax(y)
        if is_focus:
            ax.scatter(x[max_idx], y[max_idx], s=52, color=colors[i],
                       edgecolor='white', linewidth=.65, zorder=4, marker='*')
            drop = y[max_idx] - y[-1]
            label = ('峰值→末轮' if lang == 'zh' else 'Peak → final') + f'\n−{drop:.2f} pp'
            ax.annotate(label, xy=(100, y[-1]), xytext=(75, 65),
                        fontsize=8, ha='center', color=colors[i],
                        arrowprops=dict(arrowstyle='->', color=colors[i], lw=.9))
    ax.set_xlabel('通信轮次' if lang == 'zh' else 'Communication round', labelpad=3)
    ax.set_ylabel('尾部类别准确率（%）' if lang == 'zh' else 'Tail accuracy (%)', labelpad=4)
    ax.legend(frameon=False, labelspacing=.22, handlelength=2.0,
              fontsize=8, loc='lower left', borderaxespad=.25)
    ax.set_xlim(0, 100)
    ax.set_xticks([0, 50, 100])
    ax.set_ylim(55, 76)
    ax.set_yticks([55, 60, 65, 70, 75])
    ax.tick_params(length=3, pad=3, labelsize=8)
    ax.grid(axis='y', alpha=.2, linestyle='--', color=pu.COLORS['grid'])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
