"""Vivid competition.spatiotemporal_heatmap, adapted to actual client/class counts.

The recipe's rectangular sns.heatmap, continuous scale and white cell edges
are retained. A shared colorbar replaces duplicate bars; 600 cell annotations
would be unreadable at the requested compact paper size and are omitted.
"""
import numpy as np
import seaborn as sns
from matplotlib.patches import Rectangle
from matplotlib.colors import LinearSegmentedColormap
from figure_common import pu
from _utils.palette_maps import palette_stops


def draw_heatmap(ax, data, *, lang='en', show_ylabel=False, selected_group=False):
    assert data.shape == (30, 20)
    assert int(data.sum()) == 153
    assert np.all(data >= 0)
    # Bright blue scale requested by the author. Keep the count normalization
    # linear and shared; use a visible cyan for one sample and navy for nine.
    # Zero is white both in the masked cells and at the colorbar origin.
    theme_blue = palette_stops('sequential', count=3)[-1]
    cmap = LinearSegmentedColormap.from_list('intro_bright_blue', [
        (0., '#FFFFFF'), (1 / 9, pu.PALETTE[1]),
        (.45, theme_blue), (1., pu.PALETTE[3]),
    ], N=256)
    sns.heatmap(data, annot=False, fmt='d', cmap=cmap,
                vmin=0, vmax=9, mask=data == 0,
                xticklabels=False, yticklabels=False,
                linewidths=0.22, linecolor='white', cbar=False, ax=ax)
    ax.set_xticks([.5, 9.5, 19.5], ['80', '89', '99'])
    ax.set_yticks([.5, 14.5, 29.5], ['0', '14', '29'])
    ax.set_xlabel('尾部类别' if lang == 'zh' else 'Tail class', labelpad=2)
    ax.set_ylabel(('客户端' if lang == 'zh' else 'Client') if show_ylabel else '', labelpad=3)
    ax.tick_params(length=0, pad=3, labelsize=8, labelrotation=0)
    if selected_group:
        ax.add_patch(Rectangle((0, 27), 20, 3, fill=False, edgecolor=pu.PALETTE[2],
                               linewidth=.8, clip_on=False))
    return ax.collections[0]
