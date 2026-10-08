"""Compact paper Figure 1. Reuses two source-tracked Vivid plotting recipes.

Native and target width: 7.2 inches (183 mm); scale=1, minimum font=8pt.
Run from the task workspace: python figures/plot_intro_compact.py
"""
import hashlib
import json
import sys
import matplotlib.pyplot as plt
import pandas as pd
from figure_common import HERE, ROOT, setup
sys.path.insert(0, str(HERE / '.vivid/python-deps'))
from intro_heatmaps import draw_heatmap
from intro_curves import draw_curves


def source_data():
    audit = json.loads((HERE / 'data/audit.json').read_text(encoding='utf-8'))
    matrices, runs, sources = {}, {}, {}
    for partition in ('noniid-labeldir-fine', 'client-longtail'):
        source = ROOT / audit['runs'][partition + '/j']['source']
        matrix = pd.read_csv(source / 'client_class_counts.csv')
        matrices[partition] = matrix[[f'class_{i}' for i in range(80, 100)]].to_numpy()
        runs[partition] = pd.read_csv(source / 'round_metrics.csv')
        completion = json.loads((source / 'completion.json').read_text(encoding='utf-8'))
        assert completion['completed_round'] == 100
        sources[partition] = {
            'source': source.relative_to(ROOT).as_posix(),
            'hashes': {name: hashlib.sha256((source / name).read_bytes()).hexdigest()
                       for name in ('client_class_counts.csv', 'round_metrics.csv', 'partition_manifest.csv')},
            'peak': float(runs[partition].bottom20_tail_acc.max()),
            'final': float(runs[partition].bottom20_tail_acc.iloc[-1]),
        }
    assert audit['same_global_samples_and_labels']
    assert audit['config_checks']['j']['mismatches'] == {}
    return matrices, runs, sources


def plot(lang, matrices, runs):
    setup()
    plt.rcParams.update({'font.size': 8.5, 'axes.labelsize': 8.5, 'axes.titlesize': 9,
                         'axes.linewidth': .7})
    if lang == 'zh':
        plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'DejaVu Sans']
    fig = plt.figure(figsize=(7.2, 2.65))
    axes = [fig.add_axes([.063, .31, .183, .555]),
            fig.add_axes([.316, .31, .183, .555]),
            fig.add_axes([.626, .31, .359, .555])]
    for ax, (partition, data) in zip(axes, matrices.items()):
        selected = partition == 'client-longtail'
        sm = draw_heatmap(ax, data, lang=lang, show_ylabel=not selected, selected_group=selected)
        ax.set_title('(b) Client-LT' if selected else '(a) Dirichlet', loc='left', pad=8,
                     fontsize=9)
    cax = fig.add_axes([.137, .13, .285, .024])
    cb = fig.colorbar(sm, cax=cax, orientation='horizontal', ticks=[0, 3, 6, 9])
    cb.ax.tick_params(length=2, pad=2, labelsize=8)
    cb.set_label('样本数' if lang == 'zh' else 'Sample count', labelpad=1, fontsize=8)
    cb.outline.set_linewidth(.5)
    cb.solids.set_rasterized(False)
    draw_curves(axes[2], runs, lang=lang)
    axes[2].set_title('(c) 尾部类别遗忘' if lang == 'zh' else '(c) Tail-class forgetting',
                      loc='left', pad=8, fontsize=9)
    for ext in ('pdf', 'svg', 'png'):
        fig.savefig(HERE / 'figures' / f'intro_compact_bright_{lang}.{ext}', dpi=320,
                    facecolor='white', bbox_inches=None)
    plt.close(fig)


if __name__ == '__main__':
    matrices, runs, sources = source_data()
    (HERE / 'data/intro_compact_sources.json').write_text(json.dumps(sources, indent=2), encoding='utf-8')
    for lang in ('en', 'zh'):
        plot(lang, matrices, runs)
    print('Rendered bright blue/orange English and Chinese figures from original J run files.')
