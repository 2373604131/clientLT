"""Rebuild insight figures from raw CIFAR100-LT seed42 experiment files.

Run: python presentation/insights_20260925/make_insights.py
No training, inference, smoothing, or generated numerical data.
"""
from pathlib import Path
import argparse
import hashlib
import json
from decimal import Decimal, ROUND_HALF_UP

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.lines import Line2D

from plot_utils import (style, clean, panel, export, INK, MUTED, GRID,
                        CLT, DIR, METHOD_COLORS, METHOD_MARKERS)

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DATA = HERE / 'data'
DATA.mkdir(exist_ok=True)
PARTS = ['client-longtail', 'noniid-labeldir-fine']
METHODS = ['E2', 'E3', 'S', 'J']
TABLE = ROOT / 'presentation/results_table_20260920/完整结果_长表.csv'
AUDIT = {'dataset': 'CIFAR100-LT', 'seed': 42, 'rounds': [0, 100],
         'tail_classes': list(range(80, 100)), 'runs': [], 'figures': []}


def txt(lang, en, zh):
    return zh if lang == 'zh' else en


def fmt(value, digits=2):
    return format(Decimal(str(round(float(value), 8))).quantize(
        Decimal(10) ** -digits, rounding=ROUND_HALF_UP), f'.{digits}f')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_data():
    index = pd.read_csv(TABLE)
    chosen = index[index.method.isin(METHODS)]
    matrices, metrics, stats, perclass, partitions = {}, {}, [], [], {}
    manifests, schedules = {}, {}
    for row in chosen.itertuples():
        run = ROOT / row.source
        key = (row.partition, row.method)
        raw = pd.read_csv(run / 'round_metrics.csv').sort_values('round')
        assert raw['round'].tolist() == list(range(101)), key
        acc = []
        source_hashes = {}
        for rnd in range(101):
            path = run / f'per_class_accuracy_epoch_{rnd-1}.csv'
            df = pd.read_csv(path).sort_values('class_id')
            assert df.class_id.tolist() == list(range(100)), (key, rnd)
            # Remove only the <=1e-8 CSV floating point dust, as in the existing table.
            acc.append(df.per_class_acc.to_numpy().round(8))
            source_hashes[path.name] = sha(path)
        acc = np.array(acc)
        assert np.all((acc >= 0) & (acc <= 100))
        calculated = pd.DataFrame({
            'round': np.arange(101), 'overall': acc.mean(1),
            'head': acc[:, :20].mean(1), 'middle': acc[:, 20:80].mean(1),
            'tail': acc[:, 80:].mean(1), 'nontail': acc[:, :80].mean(1),
        })
        columns = {'overall': 'overall_acc', 'head': 'head20_acc',
                   'middle': 'middle60_acc', 'tail': 'bottom20_tail_acc',
                   'nontail': 'non_tail_acc'}
        errors = {}
        for metric, raw_col in columns.items():
            if raw_col in raw:
                errors[metric] = float(np.max(np.abs(calculated[metric]-raw[raw_col].to_numpy())))
                assert errors[metric] < 1e-7, (key, metric, errors[metric])
        rec = dict(partition=row.partition, method=row.method, seed=42, source=row.source)
        for metric in columns:
            rec[metric] = float(calculated.loc[81:100, metric].mean())
            assert abs(rec[metric]-getattr(row, metric)) < 1e-7, (key, metric)
        peak_round = int(calculated.loc[1:, 'tail'].idxmax())
        rec.update(peak_round=peak_round, tail_peak=float(calculated.loc[peak_round, 'tail']),
                   tail_final=float(calculated.loc[100, 'tail']))
        rec['drop'] = rec['tail_peak']-rec['tail_final']
        completion = json.loads((run/'completion.json').read_text(encoding='utf-8-sig'))
        assert completion['completed_round'] == 100, key
        rec['optimizer_steps'] = completion.get('total_local_optimizer_steps', completion.get('total_optimizer_steps'))
        assert abs(rec['drop']-row.drop) < 1e-7
        stats.append(rec)
        matrices[key] = acc
        metrics[key] = calculated
        for c in range(80, 100):
            peak = float(acc[1:, c].max())
            perclass.append(dict(partition=row.partition, method=row.method, class_id=c,
                                 peak_accuracy=peak, final_accuracy=float(acc[-1, c]),
                                 drop=peak-float(acc[-1, c])))
        manifest_path = run/'partition_manifest.csv'
        manifests[key] = pd.read_csv(manifest_path)
        schedules[key] = json.loads((run/'protocol/full_schedule.json').read_text(encoding='utf-8-sig'))['schedule']
        AUDIT['runs'].append(dict(partition=row.partition, method=row.method, source=row.source,
            raw_metrics_sha256=sha(run/'round_metrics.csv'), partition_sha256=sha(manifest_path),
            per_class_sha256=source_hashes, maximum_metric_error=errors,
            complete_101_evaluations=True, matches_existing_last20_table=True,
            optimizer_steps=rec['optimizer_steps']))
    for part in PARTS:
        ref = manifests[part, 'S']
        for method in METHODS:
            assert ref.equals(manifests[part, method]), (part, method, 'partition')
            assert schedules[part, method] == schedules[part, 'S'], (part, method, 'schedule')
        assert ref.raw_sample_id.is_unique
        mat = pd.crosstab(ref.class_id, ref.client_id).reindex(
            index=range(100), columns=range(30), fill_value=0).to_numpy()
        assert mat.shape == (100, 30) and mat.sum() == 10847 and mat[80:].sum() == 153
        assert np.all(np.diff(mat.sum(1)) <= 0)
        partitions[part] = mat
        pd.DataFrame(mat, index=pd.Index(range(100), name='class_id'),
                     columns=[f'client_{c}' for c in range(30)]).to_csv(DATA/f'counts_{part}.csv')
    pool = lambda p: manifests[p, 'S'][['raw_sample_id', 'class_id']].sort_values('raw_sample_id').reset_index(drop=True)
    assert pool(PARTS[0]).equals(pool(PARTS[1]))
    assert np.array_equal(partitions[PARTS[0]].sum(1), partitions[PARTS[1]].sum(1))
    AUDIT['partition_checks'] = dict(identical_raw_sample_label_pool=True,
        identical_global_class_counts=True, total_samples=10847, tail_samples=153,
        same_partition_and_schedule_within_each_topology=True,
        ordinary_dirichlet_beta=0.5, matched_dirichlet_excluded=True)
    summary = pd.DataFrame(stats)
    summary.to_csv(DATA/'performance_retention.csv', index=False)
    pd.DataFrame(perclass).to_csv(DATA/'per_class_peak_to_final.csv', index=False)
    curves = pd.concat([df.assign(partition=p, method=m) for (p,m), df in metrics.items()], ignore_index=True)
    curves.to_csv(DATA/'training_curves.csv', index=False)
    return partitions, metrics, summary, pd.DataFrame(perclass)


def names(lang):
    return ['Client-LT', txt(lang, 'Dirichlet', '标准 Dirichlet')]


def footer(fig, text):
    fig.text(.075, .035, text, fontsize=8, color=MUTED, va='bottom', linespacing=1.5)


def heatmap(ax, data, lang, title, short=False):
    # Integer counts: zero has the same gray in both heatmap and colorbar.
    cmap = ListedColormap(['#F0F3F6'] + [plt.get_cmap('viridis')(v) for v in np.linspace(.10,1,9)])
    im = ax.pcolormesh(np.arange(31)-.5, np.arange(80,101)-.5, data[80:],
                       cmap=cmap, norm=BoundaryNorm(np.arange(-.5,10.5),10), rasterized=False,
                       edgecolors='white', linewidth=.15)
    ax.set_ylim(99.5, 79.5)
    ax.set_xlim(-.5, 29.5)
    ax.set_xticks([0, 10, 20, 29] if short else [0, 5, 10, 15, 20, 25, 29])
    ax.set_yticks([80, 90, 99] if short else [80,85,90,95,99])
    ax.set_xlabel(txt(lang, 'Client ID', '客户端编号'), labelpad=3)
    ax.set_ylabel(txt(lang, 'Tail class ID', '尾类编号'), labelpad=3)
    panel(ax, title)
    ax.tick_params(length=0)
    return im


def histogram(ax, data, lang, compact=False):
    counts = data.sum(1)
    ax.bar(np.arange(100), counts, width=1, color='#A7B4C1', linewidth=0)
    ax.bar(np.arange(80,100), counts[80:], width=1, color=CLT, linewidth=0)
    ax.set(xlim=(-1,100), ylim=(0,550), xticks=[0,50,99], yticks=[0,250,500])
    ax.set_xlabel(txt(lang, 'Class ID (decreasing frequency)', '类别编号（按样本量降序）'), labelpad=3)
    ax.set_ylabel(txt(lang, 'Samples', '样本数'), labelpad=3)
    clean(ax)
    if not compact:
        ax.text(.43,.78,txt(lang,'Identical counts in both partitions\n10,847 images · 30 clients',
            '两种划分的每类数量完全相同\n10,847 张图片 · 30 个客户端'), transform=ax.transAxes,
            fontsize=8.5, linespacing=1.6)
    ax.annotate('Tail20', xy=(90,7), xytext=(70,220 if not compact else 360), fontsize=8,
                color=CLT, arrowprops=dict(arrowstyle='-',color=CLT,lw=.8))


def insight1(parts, metrics, summary, lang):
    fig = plt.figure(figsize=(7.2,4.8))
    gs = fig.add_gridspec(2,2,left=.08,right=.91,bottom=.21,top=.92,
                          height_ratios=[.8,1.25],hspace=.95,wspace=.4)
    ax = fig.add_subplot(gs[0,0])
    histogram(ax,parts[PARTS[0]],lang)
    panel(ax,txt(lang,'(a) One shared class distribution','(a) 相同的全局类别数量'))
    ax = fig.add_subplot(gs[0,1])
    vals = [summary[(summary.partition==p)&(summary.method=='S')].iloc[0]['tail'] for p in PARTS]
    ax.plot(vals,[0,1],color='#A7B4C1',lw=1)
    for i,(v,c,m) in enumerate(zip(vals,[CLT,DIR],['o','^'])):
        ax.scatter(v,i,c=c,marker=m,s=45,zorder=3)
        ax.annotate(fmt(v,3)+'%',(v,i),xytext=(6,0),textcoords='offset points',va='center',fontsize=8.5,color=c)
    ax.set(xlim=(60,78),ylim=(-.55,1.55),yticks=[0,1],yticklabels=names(lang),xticks=[60,65,70,75])
    ax.set_xlabel(txt(lang,'Tail20 accuracy (%) · rounds 81–100','尾类准确率（%）· 第 81–100 轮均值'),fontsize=8)
    panel(ax,txt(lang,'(b) Different outcomes under S','(b) 同一策略 S，尾类结果不同'))
    clean(ax,both=True)
    for j,p in enumerate(PARTS):
        ax = fig.add_subplot(gs[1,j])
        im=heatmap(ax,parts[p],lang,f'({chr(99+j)}) {names(lang)[j]}')
    cax=fig.add_axes([.932,.21,.015,.36])
    cb=fig.colorbar(im,cax=cax,ticks=[0,3,6,9])
    cb.set_label(txt(lang,'Training samples / cell','每格训练样本数'),fontsize=8,labelpad=4)
    footer(fig,txt(lang,
        'CIFAR100-LT · seed 42 · LA in both runs. Tail20: class IDs 80–99 (153 training images).\n'
        'Heatmaps use the same scale; light gray = zero. Client IDs are local to each partition.',
        'CIFAR100-LT · seed 42 · 两组均使用 LA。Tail20 为类别 80–99，共 153 张训练图片。\n'
        '热图共用色标，浅灰表示 0；客户端编号仅在各自划分内有意义。'))
    export(fig,HERE/lang/'01_client_distribution',AUDIT['figures'])


def curve(ax, metrics, method, lang, annotate=True, compact=False):
    for part,col,ls,m,name in zip(PARTS,[CLT,DIR],['-','--'],['o','^'],names(lang)):
        df=metrics[part,method]
        ax.plot(df['round'],df['tail'],color=col,ls=ls,marker=m,markevery=10,
                markersize=3.1 if compact else 3.8,markeredgewidth=.5,
                markerfacecolor='white',label=name)
        pr=int(df.loc[1:,'tail'].idxmax()); peak=float(df.loc[pr,'tail']); end=float(df.iloc[-1]['tail'])
        ax.scatter([pr],[peak],s=22,marker='D',color=col,zorder=4)
        if annotate:
            ax.plot([pr,105],[peak,peak],color=col,ls=':',lw=.7,alpha=.75)
            ax.vlines(105,end,peak,color=col,lw=1.25)
            ax.hlines([end,peak],103.5,106.5,color=col,lw=1.25)
            ax.text(109,(peak+end)/2,fmt(peak-end)+'\npp',color=col,fontsize=8,va='center')
        else:
            ax.text(.05,.13 if part==PARTS[0] else .035,
                    f'{name}: −{fmt(peak-end)} pp',transform=ax.transAxes,color=col,fontsize=8)
    if method=='S':
        ax.axvspan(90,100,color='#CDD5DC',alpha=.24,zorder=0)
        ax.axvline(90,color='#929FAB',lw=.7,ls=':')
    ax.set(xlim=(0,128 if annotate else 103),ylim=(55,77),xticks=[0,25,50,75,100],yticks=[55,60,65,70,75])
    ax.set_xlabel(txt(lang,'Federated round','联邦训练轮次'))
    ax.set_ylabel(txt(lang,'Tail20 accuracy (%)','尾类准确率（%）'))
    clean(ax)


def insight2(parts, metrics, summary, lang):
    fig,axes=plt.subplots(1,2,figsize=(7.2,3.8))
    fig.subplots_adjust(left=.085,right=.97,bottom=.25,top=.79,wspace=.28)
    for ax,m in zip(axes,['S','J']):
        curve(ax,metrics,m,lang)
    panel(axes[0],txt(lang,'(a) S: dense alternating updates','(a) S：密集交替更新'))
    panel(axes[1],txt(lang,'(b) J: joint A/B updates','(b) J：A/B 联合更新'))
    axes[1].set_ylabel('')
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.5,.97),ncol=2,frameon=False)
    footer(fig,txt(lang,
        'Raw round-by-round curves · fixed task and training pool · one run (seed 42).\n'
        'Brackets: peak in rounds 1–100 minus round 100. Gray region in S: rounds 91–100 train B only.',
        '同一任务、固定训练数据 · 原始逐轮曲线 · 单次运行（seed 42）。\n'
        '右侧标线表示第 1–100 轮峰值减第 100 轮；S 的灰色区域表示第 91–100 轮仅训练 B。'))
    export(fig,HERE/lang/'02_tail_forgetting',AUDIT['figures'])


def scatter(ax, summary, lang, compact=False):
    part=summary[summary.partition==PARTS[0]].set_index('method')
    offsets={'E2':(-23,-16),'E3':(-9,9),'S':(7,-4),'J':(-13,9)}
    for m in METHODS:
        r=part.loc[m]
        ax.scatter(r.nontail,r['tail'],c=METHOD_COLORS[m],marker=METHOD_MARKERS[m],
                   s=42 if compact else 56,zorder=4,edgecolors='white',linewidths=.6)
        ax.annotate(m,(r.nontail,r['tail']),xytext=offsets[m],textcoords='offset points',
                    fontsize=8 if compact else 9,color=METHOD_COLORS[m],fontweight='bold')
    e2,e3=part.loc['E2'],part.loc['E3']
    ax.annotate('',xy=(e3.nontail,e3['tail']),xytext=(e2.nontail,e2['tail']),
                arrowprops=dict(arrowstyle='->',lw=1.05,color=METHOD_COLORS['E3'],shrinkA=6,shrinkB=6))
    if not compact:
        nd,td=fmt(e3.nontail-e2.nontail),fmt(e3['tail']-e2['tail'])
        ax.text(.035,.18,txt(lang,f'E2 → E3\nNon-tail +{nd} pp\nTail +{td} pp',
                f'E2 → E3\n非尾类 +{nd} pp\n尾类 +{td} pp'),
                transform=ax.transAxes,fontsize=8.5,color=METHOD_COLORS['E3'],linespacing=1.5)
    ax.set(xlim=(65,76),ylim=(55,77),xticks=[65,70,75],yticks=[55,60,65,70,75])
    ax.set_xlabel(txt(lang,'Non-tail80 accuracy (%)','非尾类准确率（%）'))
    ax.set_ylabel(txt(lang,'Tail20 accuracy (%)','尾类准确率（%）'))
    clean(ax,True)


def insight3(parts,metrics,summary,lang):
    fig,axes=plt.subplots(1,2,figsize=(7.2,4.25),gridspec_kw={'width_ratios':[1,1]})
    fig.subplots_adjust(left=.08,right=.98,bottom=.32,top=.87,wspace=.4)
    scatter(axes[0],summary,lang)
    panel(axes[0],txt(lang,'(a) Continued adaptation can help both','(a) 适当更新，两方面都能改善'))
    axes[0].text(.5,-.27,txt(lang,'Mean accuracy over rounds 81–100','第 81–100 轮平均准确率'),
                ha='center',transform=axes[0].transAxes,fontsize=8,color=MUTED)
    ax=axes[1]
    part=summary[summary.partition==PARTS[0]].set_index('method')
    for y,m in enumerate(METHODS):
        r=part.loc[m];col=METHOD_COLORS[m]
        ax.plot([r.tail_final,r.tail_peak],[y,y],color=col,lw=2)
        ax.scatter(r.tail_peak,y,facecolors='white',edgecolors=col,s=48,marker='o',lw=1.2,zorder=3)
        ax.scatter(r.tail_final,y,color=col,s=29,marker='o',zorder=4)
        ax.text(77,y,fmt(r['drop'])+' pp',fontsize=8.5,va='center',color=col)
    ax.set(xlim=(55,84),ylim=(3.7,-.65),yticks=range(4),yticklabels=METHODS,xticks=[55,60,65,70,75])
    ax.spines['bottom'].set_bounds(55,75)
    ax.set_xlabel(txt(lang,'Tail20 accuracy (%)','尾类准确率（%）'))
    panel(ax,txt(lang,'(b) Retention depends on the strategy','(b) 不同更新策略，保持效果不同'))
    ax.text(77,-.64,txt(lang,'Drop','回落'),fontsize=8.5,color=MUTED)
    ax.grid(axis='x',color=GRID,lw=.55);ax.set_axisbelow(True)
    handles=[Line2D([0],[0],marker='o',ls='none',markerfacecolor='white',markeredgecolor=MUTED,label=txt(lang,'Peak','峰值')),
             Line2D([0],[0],marker='o',ls='none',color=MUTED,label=txt(lang,'Round 100','第 100 轮'))]
    ax.legend(handles=handles,loc='lower center',bbox_to_anchor=(.45,-.32),ncol=2,frameon=False)
    footer(fig,txt(lang,
        'Client-LT · LA · seed 42. E2: fixed A; E3: 9 extra A updates; S: 90 extra A updates.\n'
        'J: daily joint A/B + 9 extra A updates. S has more optimizer steps; this is a strategy comparison.\n'
        'Peak: maximum over rounds 1–100. Drop: peak minus round 100; it is not the last-20 mean.',
        'Client-LT · LA · seed 42。E2：固定 A；E3：9 次额外 A 更新；S：90 次额外 A 更新。\n'
        'J：日常 A/B 联训 + 9 次额外 A 更新。S 的优化步数更多，本图比较完整训练策略。\n'
        '峰值取第 1–100 轮最大值；回落为峰值减第 100 轮，与末 20 轮均值分开统计。'))
    export(fig,HERE/lang/'03_adaptation_and_retention',AUDIT['figures'])


def overview(parts,metrics,summary,lang):
    fig=plt.figure(figsize=(7.2,4.25))
    outer=fig.add_gridspec(1,3,left=.075,right=.98,bottom=.31,top=.88,wspace=.48,width_ratios=[1.0,1.12,1.08])
    left=outer[0].subgridspec(3,1,height_ratios=[.72,1,1],hspace=1.05)
    ax=fig.add_subplot(left[0]);histogram(ax,parts[PARTS[0]],lang,compact=True)
    ax.set_xlabel(txt(lang,'Class ID','类别编号'),fontsize=8,labelpad=1)
    ax.set_yticks([0,500]);ax.set_xticks([0,99]);ax.set_ylabel(txt(lang,'Count','数量'),fontsize=8)
    for j,p in enumerate(PARTS):
        ax=fig.add_subplot(left[j+1]);im=heatmap(ax,parts[p],lang,names(lang)[j],short=True)
        ax.set_title(names(lang)[j],loc='left',fontsize=8,pad=3,fontweight='normal')
        ax.set_ylabel(txt(lang,'Class','类别'),fontsize=8)
        if j==0:ax.set_xlabel('');ax.set_xticklabels([])
    cbax=fig.add_axes([.075,.208,.213,.016])
    fig.colorbar(im,cax=cbax,orientation='horizontal',ticks=[0,3,6,9])
    ax=fig.add_subplot(outer[1]);curve(ax,metrics,'S',lang,annotate=False,compact=True)
    ax.set_xticks([0,50,100]);ax.set_ylabel(txt(lang,'Tail20 accuracy (%)','尾类准确率（%）'),labelpad=2,fontsize=8)
    ax.set_xlabel(txt(lang,'Round','轮次'),fontsize=8)
    ax.legend(loc='upper left',bbox_to_anchor=(-.03,1.005),frameon=False,handlelength=1.5,fontsize=8)
    ax=fig.add_subplot(outer[2]);scatter(ax,summary,lang,compact=True)
    ax.set_ylabel(txt(lang,'Tail20 accuracy (%)','尾类准确率（%）'),labelpad=2,fontsize=8)
    ax.set_xlabel(txt(lang,'Non-tail80 accuracy (%)','非尾类准确率（%）'),fontsize=8)
    titles=txt(lang,['(a) Same counts,\n different placement','(b) Existing ability\n can deteriorate','(c) Updating A\n can improve both'],
               ['(a) 数量相同\n分布不同','(b) 已有能力\n仍会回落','(c) 适当更新\n两方面均可改善'])
    for spec,title in zip(outer,titles):
        bb=spec.get_position(fig)
        fig.text(bb.x0,.965,title,ha='left',va='top',fontsize=9,fontweight='bold',linespacing=1.4)
    footer(fig,txt(lang,
        'CIFAR100-LT · seed 42 · one run. (a) Same 10,847 images; identical per-class counts.\n'
        'Heatmaps: samples per class/client, common scale 0–9; gray = 0. (b) S, raw curves; peak-to-final drop.\n'
        'S trains B only in rounds 91–100 (gray). (c) Client-LT, rounds 81–100; S uses a larger budget.',
        'CIFAR100-LT · seed 42 · 单次运行。(a) 同一批 10,847 张图片，每类数量相同。\n'
        '热图共用 0–9 样本色标，浅灰为 0。(b) S 的原始曲线及峰后回落；灰区仅训练 B。\n'
        '(c) Client-LT，第 81–100 轮均值；E2 固定 A，E3 定期更新，S 密集更新，J 联训；S 预算更高。'))
    export(fig,HERE/lang/'00_three_insights',AUDIT['figures'])


def perclass_supplement(drops,lang):
    fig,axes=plt.subplots(1,2,figsize=(7.2,3.6),sharey=True)
    fig.subplots_adjust(left=.09,right=.97,bottom=.25,top=.86,wspace=.2)
    upper = int(np.ceil(drops['drop'].max()/10)*10)
    for ax,m in zip(axes,['S','J']):
        arrays=[drops[(drops.partition==p)&(drops.method==m)].sort_values('class_id')['drop'].to_numpy() for p in PARTS]
        bp=ax.boxplot(arrays,positions=[0,1],widths=.35,patch_artist=True,showfliers=False,
                       medianprops={'color':INK,'lw':1.5},whiskerprops={'color':MUTED},capprops={'color':MUTED})
        for j,(a,col) in enumerate(zip(arrays,[CLT,DIR])):
            bp['boxes'][j].set(facecolor=col+'20',edgecolor=col)
            # Fixed offsets display every class without random sampling or obscured values.
            order=np.argsort(a,kind='stable');offset=np.empty(len(a))
            offset[order]=np.tile([-.11,-.055,0,.055,.11],4)
            ax.scatter(j+offset,a,s=16,color=col,marker='o' if j==0 else '^',alpha=.8,zorder=3)
        ax.set(xticks=[0,1],xticklabels=names(lang),ylim=(-1,upper+2),yticks=np.arange(0,upper+1,10))
        panel(ax,f'{m}: '+txt(lang,'20 individual tail classes','20 个尾类分别统计'))
        clean(ax)
    axes[0].set_ylabel(txt(lang,'Per-class peak-to-final drop (pp)','逐类峰值到最终值的回落（pp）'))
    footer(fig,txt(lang,
        'Each point is one class, not an independent run. Peaks are computed separately over rounds 1–100.\n'
        'Boxes: class quartiles; line: median; whiskers: 1.5 IQR. Class-average drop differs from group-curve drop.',
        '每个点代表一个类别，并非独立实验种子；各类分别取第 1–100 轮峰值，再减第 100 轮。\n'
        '箱体为类别四分位区间，横线为中位数，须为 1.5 IQR；逐类回落均值不等于组平均曲线回落。'))
    export(fig,HERE/lang/'04_per_class_drop_supplement',AUDIT['figures'])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--from-snapshot',action='store_true',help='Render the bundled CSVs without the original experiment tree.')
    args=parser.parse_args()
    if args.from_snapshot:
        parts={p:pd.read_csv(DATA/f'counts_{p}.csv',index_col=0).to_numpy() for p in PARTS}
        curves=pd.read_csv(DATA/'training_curves.csv')
        metrics={(p,m):d.sort_values('round').reset_index(drop=True) for (p,m),d in curves.groupby(['partition','method'])}
        summary=pd.read_csv(DATA/'performance_retention.csv')
        drops=pd.read_csv(DATA/'per_class_peak_to_final.csv')
        AUDIT['source_mode']='Bundled derived CSV snapshots; raw-run audit is in validation.json.'
    else:
        parts,metrics,summary,drops=load_data()
        AUDIT['source_mode']='Recomputed and checked against original per-class and round CSVs.'
    for lang in ['en','zh']:
        style(lang)
        insight1(parts,metrics,summary,lang)
        insight2(parts,metrics,summary,lang)
        insight3(parts,metrics,summary,lang)
        overview(parts,metrics,summary,lang)
        perclass_supplement(drops,lang)
    AUDIT['design_checks'] = dict(no_smoothing=True,no_synthetic_error_bars=True,
        source_hypothesis_not_claimed_as_result=True,ours_excluded_from_motivation=True,
        e3_included=True,all_curves_include_initialization=True,
        statistical_significance_not_claimed=True,vector_pdf_and_svg=True)
    audit_name='validation_snapshot.json' if args.from_snapshot else 'validation.json'
    (HERE/audit_name).write_text(json.dumps(AUDIT,ensure_ascii=False,indent=2),encoding='utf-8')
    print(summary[['partition','method','nontail','tail','tail_peak','tail_final','drop','optimizer_steps']].to_string(index=False))
    print(json.dumps(AUDIT['figures'],ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
