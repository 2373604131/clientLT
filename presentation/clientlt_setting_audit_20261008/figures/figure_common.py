"""Shared sizing and Vivid project palette for the audited figures."""
from pathlib import Path
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
from _utils import plot_utils as pu
from _utils.palette_maps import palette_cmap

def setup():
    pu.setup_style()
    plt.rcParams.update({
        'font.family': 'sans-serif', 'font.sans-serif': ['DejaVu Sans', 'Microsoft YaHei'],
        'font.size': 8, 'axes.labelsize': 8, 'axes.titlesize': 9,
        'xtick.labelsize': 8, 'ytick.labelsize': 8, 'legend.fontsize': 8,
        'axes.edgecolor': '#555555', 'text.color': '#222222',
        'axes.labelcolor': '#222222', 'xtick.color': '#333333', 'ytick.color': '#333333',
        'axes.spines.top': False, 'axes.spines.right': False,
        'figure.facecolor': 'white', 'axes.facecolor': 'white',
        'axes.grid': False, 'pdf.fonttype': 42, 'ps.fonttype': 42,
        'svg.fonttype': 'none',
    })

def save(fig, name):
    for ext in ('png', 'pdf', 'svg'):
        fig.savefig(HERE / 'figures' / f'{name}.{ext}', dpi=220,
                    facecolor='white', bbox_inches=None)
    plt.close(fig)
