"""Validate archived numbers, draw sourced figures, and build the Chinese CVPR working copy."""
from pathlib import Path
import csv
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
WORK = Path(__file__).resolve().parent
MAIN = ROOT/'docs/论文中文初稿_CVPR_精简重写版_20261016.md'
SUPP = ROOT/'docs/论文中文初稿_CVPR_精简重写版_补充材料_20261016.md'
FIG = WORK/'figures'
BUILD = WORK/'typeset'
for p in [FIG, BUILD]:
    p.mkdir(exist_ok=True)

def rows(relative):
    with (ROOT/relative).open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))

def tables(text):
    result, buf = [], []
    for line in text.splitlines()+['']:
        if line.startswith('|'):
            buf.append([c.strip() for c in line.strip('|').split('|')])
        elif buf:
            result.append(buf[2:])
            buf = []
    return result

def validate(text):
    body, refs = text.split('## 参考文献')
    ids = set()
    for item in re.findall(r'\[([0-9, –]+)\]', body):
        for part in item.split(','):
            nums = [int(v) for v in part.strip().split('–')]
            ids.update(range(nums[0],nums[-1]+1))
    assert ids == set(range(1,23))
    assert len(re.findall(r'^\[\d+\]', refs, re.M)) == 22
    assert sum(l=='$$' for l in text.splitlines())==16
    assert not any(t in text for t in ['TODO','待补','审稿人','占位'])
    tabs = tables(text)
    assert list(map(len,tabs)) == [12,6,6]
    baselines={r['method']:r for r in rows('output/benchmark_tables_20261013/main_table_clientlt.csv')}
    keys=['fedavg-lora','fedavg-lora-la','capt','fedpurel','fedntd','fedlf','fedyoyo','fedrela','rolora','fedsvd','lora-a2']
    fields=['overall_acc','head20_acc','middle60_acc','bottom20_tail_acc']
    verified=0
    def check(actual, expected):
        nonlocal verified
        assert len(actual)==len(expected)
        assert all(abs(float(a)-float(e))<=0.005001 for a,e in zip(actual,expected)),(actual,expected)
        verified+=len(actual)
    for tab,key in zip(tabs[0],keys):
        check(tab[1:],[baselines[key][f] for f in fields])
    final=next(r for r in rows('output/old_ab_decision_20261015/joint_and_strong_control.csv') if r['method']=='oldA_oldB_joint')
    check(tabs[0][-1][1:],[final[f] for f in ['Overall','Head20','Middle60','Tail20']])
    paired={(r['arm'],int(r['seed'])):r for r in rows('output/ab_decision_review_20261003/per_run.csv')}
    for tab,arm in zip(tabs[1],['s','a-flat','a-current','a','ab','a-calibration']):
        check(tab[1:],[paired[arm,seed][f] for seed in [42,3407] for f in ['last20_overall_acc','last20_bottom20_tail_acc']])
    retention={(int(r['anchor']),r['branch']):r for r in rows('output/method_a_analysis_20260927/sample_retention.csv') if r['group']=='Tail20'}
    branches={'普通更新':'ordinary','幅度匹配':'norm-matched','方法 A':'full-cp'}
    for tab in tabs[2]:
        r=retention[int(tab[0]),branches[tab[1]]]
        check(tab[2:],[r['correct_to_wrong'],r['wrong_to_correct'],100*(int(r['retained_correct'])+int(r['wrong_to_correct']))/int(r['samples'])])
    prior=(ROOT/'docs/论文中文初稿_CVPR_joint旧A旧B_20261016.md').read_text(encoding='utf-8').split('## 参考文献')[0]
    count=lambda t:len(re.findall(r'[\u4e00-\u9fff]',t))
    report=dict(main_chinese_characters=count(body),old_main_chinese_characters=count(prior),
                reduction_percent=round(100*(1-count(body)/count(prior)),1),display_equations=8,
                numeric_table_values_verified=verified,reference_count=22,
                aggregation_protocols_separated=True)
    return report

def figures():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,
                         'axes.spines.right':False,'pdf.fonttype':42})
    split={r['partition']:r for r in rows('presentation/clientlt_setting_audit_20261008/data/partition_summary.csv')}
    data=rows('presentation/clientlt_setting_audit_20261008/data/training_curves.csv')
    fig,axs=plt.subplots(1,2,figsize=(7,2.12),gridspec_kw={'width_ratios':[0.85,1.5]})
    colors=['#C8794D','#4E7BA6','#71966E']
    parts=['client-longtail','noniid-labeldir-fine','matched-dirichlet']
    labels=['Client-LT','Dirichlet','Matched']
    shares=[100*float(split[p]['largest3_tail_share']) for p in parts]
    axs[0].bar(range(3),shares,color=colors,width=.58)
    axs[0].set_xticks(range(3),labels,fontsize=8)
    axs[0].set_ylim(0,100)
    axs[0].set_ylabel('Tail samples in top 3 clients (%)',fontsize=8)
    for i,val in enumerate(shares):
        axs[0].text(i,val+3,f'{val:.1f}',ha='center',fontsize=8)
    for p,label,color in zip(parts,labels,colors):
        points=sorted([r for r in data if r['partition']==p and r['method']=='s'],key=lambda r:int(r['round']))
        assert len(points)==101
        axs[1].plot([int(r['round']) for r in points],[float(r['bottom20_tail_acc']) for r in points],label=label,color=color,lw=1.4)
    axs[1].set(xlabel='Communication round',ylabel='Tail20 accuracy (%)',xlim=(0,100),ylim=(65.5,74))
    axs[1].legend(loc='lower center',ncol=3,fontsize=7,frameon=False,columnspacing=.7)
    axs[1].grid(axis='y',alpha=.15)
    fig.tight_layout(pad=.6,w_pad=1.2)
    fig.savefig(FIG/'motivation.pdf',bbox_inches='tight')
    fig.savefig(FIG/'motivation.png',dpi=220,bbox_inches='tight')
    plt.close(fig)
    source=ROOT/'presentation/framework_old_ab_20261015/v16_consistent_components'
    code=(source/'figures/framework_consistent_en.tex').read_text(encoding='utf-8')
    code=code.replace('joint aggregation','Aggregation').replace('Local B + joint','Local B + agg.').replace('Local A + joint','Local A + agg.')
    (FIG/'framework.tex').write_text(code,encoding='utf-8')
    shutil.copytree(source/'assets',FIG/'assets',dirs_exist_ok=True)
    run_latex(FIG,'framework',2)
    result=subprocess.run(['rungs','-dSAFER','-dBATCH','-dNOPAUSE','-sDEVICE=png16m','-r180',
                           '-sOutputFile=framework.png','framework.pdf'],cwd=FIG,capture_output=True)
    assert result.returncode==0,result.stderr[-1000:]

def escape(t):
    replacements={'\\':r'\textbackslash{}','&':r'\&','%':r'\%','#':r'\#','_':r'\_','{':r'\{','}':r'\}',
                  '~':r'\textasciitilde{}','^':r'\textasciicircum{}','→':r'$\rightarrow$','²':r'$^2$',
                  '①':'(1)','②':'(2)','③':'(3)'}
    return ''.join(replacements.get(c,c) for c in t)

def inline(t):
    result=[]
    pattern=r'(\$[^$]+\$|\[[^\]]+\]\([^)]+\)|\*\*.+?\*\*)'
    for chunk in re.split(pattern,t):
        if chunk.startswith('$') and chunk.endswith('$'):
            result.append(chunk)
        elif chunk.startswith('**'):
            result.append(r'\textbf{'+escape(chunk[2:-2])+'}')
        elif re.fullmatch(r'\[[^\]]+\]\([^)]+\)',chunk):
            m=re.fullmatch(r'\[([^\]]+)\]\(([^)]+)\)',chunk)
            result.append(r'\href{'+m[2]+'}{'+escape(m[1])+'}')
        else:
            result.append(escape(chunk))
    return ''.join(result)

PREAMBLE=r'''\documentclass[10pt,twocolumn,letterpaper]{article}
\usepackage[pagenumbers]{cvpr}
\usepackage{xeCJK}
\setmainfont{Times New Roman}
\setCJKmainfont[BoldFont=FandolSong-Bold,ItalicFont=FandolKai-Regular]{FandolSong-Regular}
\setCJKsansfont{FandolHei-Regular}
\usepackage{tabularx,array}
\usepackage{balance}
\definecolor{cvprblue}{rgb}{0.21,0.49,0.74}
\usepackage[breaklinks,colorlinks,allcolors=cvprblue]{hyperref}
\setlength{\emergencystretch}{1em}
\renewcommand{\abstractname}{摘要}
\renewcommand{\figurename}{图}
\renewcommand{\tablename}{表}
\renewcommand{\refname}{参考文献}
\hfuzz=0.2pt
\author{中文工作稿}
\date{}
'''

def render(path, stem):
    text=path.read_text(encoding='utf-8')
    lines=text.splitlines()
    title=lines[0][2:]
    prefix=PREAMBLE+(r'\renewcommand{\thetable}{S\arabic{table}}' if path==SUPP else '')
    out=[prefix,r'\title{'+inline(title)+r'}',r'\begin{document}',r'\maketitle']
    i=1; abstract=False; refs=False; tabcount=0
    while i<len(lines):
        line=lines[i]
        if not line.strip():
            i+=1; continue
        if line=='$$':
            eq=[];i+=1
            while lines[i]!='$$':
                eq.append(lines[i]);i+=1
            equation='\n'.join(eq)
            if equation.startswith('F_{qv}'):
                equation=equation.replace('F_{qv}(A)=',r'\begin{aligned}F_{qv}(A)&=')
                equation=equation.replace(r'\left[s_c',r'\!\Bigl[s_c')
                equation=equation.replace(r'-\max',r'\\ &\quad-\max')
                equation=equation.replace(r'\right].',r'\Bigr].\end{aligned}')
            out.append(r'\begin{equation}'+equation+r'\end{equation}')
        elif line.startswith('!['):
            match=re.fullmatch(r'!\[[^\]]*\]\(([^)]+)\)',line)
            source=(path.parent/match[1]).resolve().with_suffix('.pdf')
            shutil.copyfile(source,BUILD/source.name)
            while not lines[i+1].strip():i+=1
            caption=re.sub(r'^图 \d+\.\s*','',lines[i+1]);i+=1
            out.append(r'\begin{figure*}[t]\centering\includegraphics[width=\textwidth]{'+source.name+r'}\caption{'+inline(caption)+r'}\end{figure*}')
        elif re.match(r'^表 (?:S)?\d+\.',line):
            caption=re.sub(r'^表 (?:S)?\d+\.\s*','',line)
            while not lines[i+1].strip():i+=1
            tab=[];i+=1
            while i<len(lines) and lines[i].startswith('|'):
                tab.append([c.strip() for c in lines[i].strip('|').split('|')]);i+=1
            i-=1;tabcount+=1
            wide=path==MAIN and tabcount==1
            env='table*' if wide else 'minipage'
            width=r'\textwidth' if wide else r'\columnwidth'
            if path==MAIN and tabcount==2:
                header=r'& \multicolumn{2}{c}{种子 42} & \multicolumn{2}{c}{种子 3407} \\ \cmidrule(lr){2-3}\cmidrule(lr){4-5} 配置 & Overall & Tail20 & Overall & Tail20 \\'
            else:
                header=' & '.join(inline(v) for v in tab[0])+r' \\'
            if wide:
                out.append(r'\begin{table*}[t]\centering\caption{'+inline(caption)+r'}\small\setlength{\tabcolsep}{3pt}')
                out.append(r'\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}lrrrr@{}}\toprule '+header+r'\midrule')
            else:
                out.append(r'\par\medskip\noindent\begin{minipage}{\columnwidth}\centering\captionof{table}{'+inline(caption)+r'}\small\setlength{\tabcolsep}{3pt}')
                out.append(r'\begin{tabularx}{'+width+'}{X'+'r'*(len(tab[0])-1)+r'}\toprule '+header+r'\midrule')
            for row in tab[2:]:
                out.append(' & '.join(inline(v) for v in row)+r' \\')
            out.append(r'\bottomrule\end{'+('tabular*' if wide else 'tabularx')+r'}\end{'+env+'}'+('' if wide else r'\par\medskip'))
        elif line.startswith('|'):
            # Supplementary S2 diagnostic table has a preceding prose caption.
            tab=[]
            while i<len(lines) and lines[i].startswith('|'):
                tab.append([c.strip() for c in lines[i].strip('|').split('|')]);i+=1
            i-=1
            out.append(r'\begin{center}\small\begin{tabularx}{\columnwidth}{Xrr}\toprule')
            out.append(' & '.join(inline(v) for v in tab[0])+r' \\ \midrule')
            out.extend(' & '.join(inline(v) for v in row)+r' \\' for row in tab[2:])
            out.append(r'\bottomrule\end{tabularx}\end{center}')
        elif line=='## 摘要':
            out.append(r'\begin{abstract}');abstract=True
        elif line=='## 参考文献':
            out.extend([r'\clearpage',r'\typeout{MAIN-PAGES-END; REFERENCES-START-PAGE=\thepage}',r'\begin{thebibliography}{22}']);refs=True
        elif line.startswith('## '):
            if abstract:out.append(r'\end{abstract}');abstract=False
            out.append(r'\section*{'+inline(line[3:])+'}')
        elif line.startswith('### '):
            out.append(r'\subsection*{'+inline(line[4:])+'}')
        elif re.match(r'^\[\d+\]',line):
            m=re.match(r'^\[(\d+)\]\s*(.*)',line)
            out.append(r'\bibitem{r'+m[1]+'}'+inline(m[2]))
        else:
            out.append(inline(line)+'\n')
        i+=1
    if refs:out.append(r'\end{thebibliography}')
    if path==MAIN:
        # Declare wide floats early enough to appear alongside their discussion.
        for fig_name,section in [('motivation.pdf',r'\section*{2.'),('framework.pdf',r'\subsection*{4.1')]:
            index=next(i for i,s in enumerate(out) if s.startswith(r'\begin{figure*}') and fig_name in s)
            floating=out.pop(index)
            insert=next(i for i,s in enumerate(out) if s.startswith(section))
            out.insert(insert,floating)
        first=next(i for i,s in enumerate(out) if s.startswith(r'\begin{table*}'))
        last=next(i for i in range(first,len(out)) if r'\end{table*}' in out[i])
        floating=out[first:last+1]
        del out[first:last+1]
        insert=next(i for i,s in enumerate(out) if s.startswith(r'\subsection*{4.3'))
        out[insert:insert]=floating
        index=next(i for i,s in enumerate(out) if s.startswith(r'\subsection*{5.4'))
        out.insert(index,r'\balance')
    out.append(r'\end{document}')
    shutil.copyfile(WORK/'template/cvpr.sty',BUILD/'cvpr.sty')
    (BUILD/f'{stem}.tex').write_text('\n\n'.join(out),encoding='utf-8')
    run_latex(BUILD,stem,2)
    shutil.copyfile(BUILD/f'{stem}.pdf',path.with_suffix('.pdf'))
    log=(BUILD/f'{stem}.log').read_text(encoding='utf-8',errors='replace')
    return dict(output=re.findall(r'Output written on[^\n]+',log),
                overfull_boxes=re.findall(r'Overfull[^\n]+',log),
                missing_glyphs=re.findall(r'Missing character[^\n]+',log),
                references_start_page=re.findall(r'REFERENCES-START-PAGE=(\d+)',log))

def run_latex(cwd,stem,n):
    env=os.environ.copy();env['TEXMFVAR']=str(BUILD/'texmf-var')
    for i in range(n):
        p=subprocess.run(['xelatex','-no-shell-escape','-interaction=nonstopmode','-halt-on-error',f'{stem}.tex'],
                         cwd=cwd,env=env,capture_output=True,text=True,encoding='utf-8',errors='replace')
        (cwd/f'{stem}-compile-{i+1}.txt').write_text(p.stdout+'\n'+p.stderr,encoding='utf-8')
        if p.returncode:
            print(p.stdout[-5000:]);raise RuntimeError(stem)

if __name__=='__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    report=validate(MAIN.read_text(encoding='utf-8'))
    if '--figures' in sys.argv:figures()
    if '--render' in sys.argv:
        report['main_layout']=render(MAIN,'main')
        report['supplement_layout']=render(SUPP,'supplement')
        main_pages=int(report['main_layout']['references_start_page'][0])-1
        assert main_pages<=8,main_pages
        report['main_pages']=main_pages
        report['template_commit']='12909ae'
        report['layout_configuration']='Official CVPR 2026 style; 10pt, two columns, US letter; added Chinese fonts; original margins'
        report['english_translation_page_count_verified']=False
    (WORK/'validation.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False,indent=2))
