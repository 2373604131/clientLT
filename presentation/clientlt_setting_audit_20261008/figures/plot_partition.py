"""Adapted in-place from Vivid screenshot.marginal_heatmap.
Keeps the heatmap + column totals + row totals. Dense cells are not labelled;
selected numerical margins and structural annotations replace tiny cell text.
The original complete recipe is retained under .vivid/template-sources.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.patches import Rectangle
from figure_common import HERE, setup, save, pu, palette_cmap


def marginal_panel(fig, gs, data, title, right_max=700, label_y=True, normalized=False):
    inner = gs.subgridspec(2, 2, width_ratios=[8, 1.35], height_ratios=[1, 3.1],
                          wspace=.08, hspace=.08)
    ax_heatmap = fig.add_subplot(inner[1, 0])
    cmap = palette_cmap('sequential')
    cmap.set_bad('white')
    norm = Normalize(0, 100 if normalized else np.log10(501))
    values = 100 * data / data.sum(axis=0) if normalized else np.log10(data + 1)
    shown = np.ma.masked_where(data == 0, values)
    ax_heatmap.imshow(shown, cmap=cmap, norm=norm, aspect='auto', interpolation='none')
    # Preserve white cell boundaries from the recipe, at a scale for 30 x 100 cells.
    for i in range(data.shape[0] + 1):
        ax_heatmap.axhline(i - .5, color='white', linewidth=.10)
    for j in range(data.shape[1] + 1):
        ax_heatmap.axvline(j - .5, color='white', linewidth=.08)
    ax_heatmap.axvline(79.5, color=pu.PALETTE[2], linewidth=1)
    ax_heatmap.axhline(26.5, color='#333333', linestyle='--', linewidth=.8)
    ax_heatmap.add_patch(Rectangle((79.5, 26.5), 20, 3, fill=False,
                                  edgecolor='#222222', linewidth=1.1))
    ax_heatmap.set_xticks([0, 40, 79, 99], ['0', '40', '79', '99'])
    ax_heatmap.set_yticks([0, 14, 26, 29], ['0', '14', '26', '29'])
    ax_heatmap.tick_params(length=2, pad=2)
    ax_heatmap.set_xlabel('Class ID (tail: 80–99)')
    if label_y:
        ax_heatmap.set_ylabel('Client ID')
    ax_heatmap.set_xlim(-.5, 99.5)
    ax_heatmap.set_ylim(29.5, -.5)
    bar_data_year = data.sum(axis=0)
    bar_data_month = data.sum(axis=1)
    ax_top = fig.add_subplot(inner[0, 0], sharex=ax_heatmap)
    ax_top.bar(np.arange(100), bar_data_year, width=1,
               color=[pu.PALETTE[0]] * 80 + [pu.PALETTE[2]] * 20)
    ax_top.set_ylim(0, 580)
    ax_top.set_yticks([0, 500])
    ax_top.tick_params(bottom=False, labelbottom=False, length=2, pad=2)
    ax_top.set_title(title, loc='left', pad=13, fontsize=9, fontweight='bold')
    ax_top.text(.02, .84, 'Samples / class', transform=ax_top.transAxes, fontsize=8)
    ax_top.text(99, 160, str(int(bar_data_year[-1])), ha='right', fontsize=8,
                color=pu.PALETTE[2])
    ax_right = fig.add_subplot(inner[1, 1], sharey=ax_heatmap)
    ax_right.barh(np.arange(30), bar_data_month, height=.83,
                 color=[pu.PALETTE[0]] * 27 + [pu.PALETTE[2]] * 3)
    ax_right.set_xlim(0, right_max)
    ax_right.set_xticks([0, right_max])
    ax_right.tick_params(left=False, labelleft=False, length=2, pad=2)
    ax_right.set_title('Samples\n/ client', fontsize=8, pad=5)
    ax_right.set_ylim(29.5, -.5)
    for ax in (ax_top, ax_right):
        ax.grid(False)
    return ScalarMappable(norm=norm, cmap=cmap)


def colorbar(fig, mappable, y=.12, normalized=False):
    cax = fig.add_axes([.33, y, .34, .026])
    ticks = [0, 20, 40, 60, 80, 100] if normalized else np.log10(np.array([0, 1, 5, 20, 100, 500]) + 1)
    cb = fig.colorbar(mappable, cax=cax, orientation='horizontal', ticks=ticks)
    cb.ax.set_xticklabels(['0', '20', '40', '60', '80', '100'] if normalized else ['0', '1', '5', '20', '100', '500'])
    cb.ax.tick_params(length=2, pad=2)
    cb.set_label('Share of each class at a client (%; white = 0)' if normalized else
                 'Samples per cell (log scale; white = 0)', labelpad=3)


def main():
    setup()
    fig = plt.figure(figsize=(7.2, 4.5))
    gs = fig.add_gridspec(1, 2, left=.08, right=.96, bottom=.31, top=.82, wspace=.29)
    stats = pd.read_csv(HERE / 'data/partition_summary.csv').set_index('partition')
    for i, (partition, title) in enumerate([
        ('noniid-labeldir-fine', r'(a) Class-wise Dirichlet, $\beta=0.5$'),
        ('client-longtail', r'(b) Client-LT, $\lambda_T=0.75$')]):
        matrix = pd.read_csv(HERE / f'data/counts_{partition}.csv').iloc[:, 1:].to_numpy()
        sm = marginal_panel(fig, gs[i], matrix, title, label_y=(i == 0), normalized=True)
        share = 100 * stats.loc[partition, 'largest3_tail_share']
        x = .08 if i == 0 else .572
        fig.text(x, .20, f'Top 3 by tail count: {share:.1f}% of tail samples', fontsize=8,
                 fontweight='bold')
    colorbar(fig, sm, y=.10, normalized=True)
    fig.text(.08, .95, 'Same global sample pool; different client allocation', fontsize=11,
             fontweight='bold', va='top')
    fig.text(.08, .892, 'CIFAR-100-LT | IF = 100 | 30 clients | split seed 42', fontsize=8)
    save(fig, 'partition_comparison')

if __name__ == '__main__':
    main()
