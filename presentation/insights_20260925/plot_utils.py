"""Shared publication style for the three empirical findings."""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.text import Text

INK = '#203448'
MUTED = '#627181'
GRID = '#E2E7EC'
CLT = '#D55E00'
DIR = '#0072B2'
METHOD_COLORS = {'E2': '#697787', 'E3': '#009E73', 'S': CLT, 'J': '#8A659E'}
METHOD_MARKERS = {'E2': 's', 'E3': 'D', 'S': 'o', 'J': '^'}


def style(lang):
    plt.rcParams.update({
        'font.family': 'Microsoft YaHei' if lang == 'zh' else 'Arial',
        'font.size': 9, 'axes.labelsize': 9, 'axes.titlesize': 10,
        'xtick.labelsize': 8, 'ytick.labelsize': 8, 'legend.fontsize': 8,
        'text.color': INK, 'axes.labelcolor': INK,
        'xtick.color': MUTED, 'ytick.color': MUTED,
        'axes.edgecolor': '#AAB6C0', 'axes.linewidth': .65,
        'axes.spines.top': False, 'axes.spines.right': False,
        'axes.unicode_minus': False, 'figure.facecolor': 'white',
        'savefig.facecolor': 'white', 'pdf.fonttype': 42,
        'svg.fonttype': 'none', 'lines.linewidth': 1.6,
    })


def clean(ax, both=False):
    ax.grid(axis='both' if both else 'y', color=GRID, linewidth=.55)
    ax.set_axisbelow(True)
    ax.tick_params(length=3, width=.6)


def panel(ax, title):
    ax.set_title(title, loc='left', pad=10, fontweight='bold')


def export(fig, path: Path, audit):
    """Preserve physical size and flag text outside the actual canvas."""
    # 8.5 pt at 7.2 inches stays above 8 pt when inserted at 17.5 cm.
    for obj in fig.findobj(Text):
        if obj.get_fontsize() < 8.5:
            obj.set_fontsize(8.5)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    canvas = fig.bbox
    outside = []
    sizes = []
    for obj in fig.findobj(Text):
        if not obj.get_visible() or not obj.get_text().strip():
            continue
        bb = obj.get_window_extent(renderer)
        if bb.width == 0 or bb.height == 0:
            continue
        sizes.append(obj.get_fontsize())
        if bb.x0 < -1 or bb.y0 < -1 or bb.x1 > canvas.width+1 or bb.y1 > canvas.height+1:
            outside.append(obj.get_text())
    path.parent.mkdir(parents=True, exist_ok=True)
    for ext in ('png', 'pdf', 'svg'):
        fig.savefig(path.with_suffix('.'+ext), dpi=300)
    audit.append({'figure': path.stem, 'language': path.parent.name,
                  'size_inches': fig.get_size_inches().tolist(),
                  'minimum_font_pt': min(sizes),
                  'minimum_font_pt_at_17_5_cm': min(sizes)*17.5/(fig.get_figwidth()*2.54),
                  'out_of_canvas_text': outside})
    plt.close(fig)
