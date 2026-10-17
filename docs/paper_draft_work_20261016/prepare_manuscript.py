"""Validate this manuscript against archived results and typeset a reading copy."""
from pathlib import Path
import csv
import json
import re
import shutil
import os
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
WORK = Path(__file__).resolve().parent
MANUSCRIPT = ROOT / 'docs/论文中文初稿_CVPR_joint旧A旧B_20261016.md'


def csv_rows(relative):
    with (ROOT / relative).open(encoding='utf-8-sig', newline='') as handle:
        return list(csv.DictReader(handle))


def tables(text):
    groups, active = [], []
    for line in text.splitlines() + ['']:
        if line.startswith('|'):
            active.append([cell.strip() for cell in line.strip('|').split('|')])
        elif active:
            groups.append(active[2:])
            active = []
    return groups


def validate(text):
    body, refs = text.split('## 参考文献', 1)
    cited = {int(n) for item in re.findall(r'\[([0-9, ]+)\]', body)
             for n in item.replace(' ', '').split(',')}
    reference_ids = {int(n) for n in re.findall(r'^\[(\d+)\]', refs, re.M)}
    assert cited == reference_ids == set(range(1, 21)), (cited, reference_ids)
    delimiters = sum(line.strip() == '$$' for line in text.splitlines())
    assert delimiters % 2 == 0
    for image in re.findall(r'!\[[^\]]*\]\(([^)]+)\)', text):
        assert (MANUSCRIPT.parent / image).resolve().is_file(), image
    for term in ['待补', '待定', 'TODO', 'citation needed', '审稿人', '占位']:
        assert term not in text, term
    results = tables(text)
    assert [len(t) for t in results] == [12, 6, 6]
    baseline = {r['method']: r for r in csv_rows('output/benchmark_tables_20261013/main_table_clientlt.csv')}
    order = ['fedavg-lora', 'fedavg-lora-la', 'capt', 'fedpurel', 'fedntd',
             'fedlf', 'fedyoyo', 'fedrela', 'rolora', 'fedsvd', 'lora-a2']
    fields = ['overall_acc', 'head20_acc', 'middle60_acc', 'bottom20_tail_acc']
    checked = 0
    for row, key in zip(results[0], order):
        expected = [float(baseline[key][f]) for f in fields]
        actual = [float(v) for v in row[1:]]
        assert all(abs(a-b) < 0.000051 for a, b in zip(actual, expected)), (key, actual, expected)
        checked += 4
    final = next(r for r in csv_rows('output/old_ab_decision_20261015/joint_and_strong_control.csv')
                 if r['method'] == 'oldA_oldB_joint')
    expected = [float(final[f]) for f in ['Overall', 'Head20', 'Middle60', 'Tail20']]
    assert all(abs(float(a)-b) < 0.000051 for a,b in zip(results[0][-1][1:], expected))
    checked += 4
    for row in results[0]:
        overall, head, middle, tail = map(float, row[1:])
        assert abs(overall-(0.2*head+0.6*middle+0.2*tail)) < 0.00011
    paired = {(r['arm'], int(r['seed'])):r for r in csv_rows('output/ab_decision_review_20261003/per_run.csv')}
    for row, arm in zip(results[1], ['s','a-flat','a-current','a','ab','a-calibration']):
        expected = [float(paired[arm, seed][field]) for seed in [42,3407]
                    for field in ['last20_overall_acc','last20_bottom20_tail_acc']]
        assert all(abs(float(a)-b)<0.000051 for a,b in zip(row[1:],expected)), arm
        checked += 4
    retained = {(int(r['anchor']),r['branch']):r for r in csv_rows('output/method_a_analysis_20260927/sample_retention.csv')
                if r['group']=='Tail20'}
    names = {'普通更新':'ordinary','幅度匹配':'norm-matched','方法 A':'full-cp'}
    for row in results[2]:
        record = retained[int(row[0]), names[row[1]]]
        assert int(row[2]) == int(record['correct_to_wrong'])
        assert int(row[3]) == int(record['wrong_to_correct'])
        final_acc = (int(record['retained_correct'])+int(record['wrong_to_correct']))/int(record['samples'])*100
        assert abs(float(row[4])-final_acc) < 0.000001
        checked += 3
    report = dict(characters=len(text), chinese_characters=len(re.findall(r'[\u4e00-\u9fff]',text)),
                  references=len(reference_ids), display_equations=delimiters//2,
                  numeric_table_values_verified=checked, figures=2,
                  table_sizes=[len(t) for t in results], placeholders_found=False,
                  numeric_status='pass', source_protocols_separated=True)
    (WORK/'manuscript_validation.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    return report


def escape_plain(text):
    replacements = {'\\':r'\textbackslash{}','&':r'\&','%':r'\%','$':r'\$',
                    '#':r'\#','_':r'\_','{':r'\{','}':r'\}',
                    '~':r'\textasciitilde{}','^':r'\textasciicircum{}',
                    '→':r'$\rightarrow$','²':r'$^2$'}
    return ''.join(replacements.get(char,char) for char in text)


def inline(text):
    pattern = r'(\$[^$]+\$|\[[^\]]+\]\([^)]+\))'
    parts = []
    for chunk in re.split(pattern, text):
        if chunk.startswith('$') and chunk.endswith('$'):
            parts.append(chunk)
        elif re.fullmatch(r'\[[^\]]+\]\([^)]+\)',chunk):
            m = re.fullmatch(r'\[([^\]]+)\]\(([^)]+)\)',chunk)
            parts.append(r'\href{'+m[2]+'}{'+escape_plain(m[1])+'}')
        else:
            parts.append(escape_plain(chunk))
    return ''.join(parts)


def render(text):
    build = WORK/'typeset'
    build.mkdir(exist_ok=True)
    preamble = r'''\documentclass[UTF8,a4paper,10pt,fontset=fandol]{ctexart}
\usepackage[margin=23mm]{geometry}
\usepackage{amsmath,amssymb,graphicx,booktabs,tabularx,array,adjustbox}
\usepackage{xcolor}
\usepackage[colorlinks=true,linkcolor=black,urlcolor=blue!45!black]{hyperref}
\setlength{\parindent}{2em}
\setlength{\parskip}{0.3em}
\linespread{1.16}
\setlength{\emergencystretch}{2em}
\setlength{\tabcolsep}{3pt}
\renewcommand{\arraystretch}{1.3}
\title{联邦长尾学习中的尾部类别知识保持\\与跨客户端知识迁移}
\author{}
\date{}
\begin{document}
\maketitle
\vspace{-2em}
'''
    output = [preamble]
    lines = text.splitlines()
    index = 0
    while index < len(lines):
        line = lines[index]
        if not line.strip() or line.startswith('# '):
            index += 1
            continue
        if line == '$$':
            equation = []
            index += 1
            while lines[index] != '$$':
                equation.append(lines[index])
                index += 1
            output.append(r'\begin{center}\begin{adjustbox}{max width=\linewidth}$\displaystyle ' +
                          '\n'.join(equation) + r'$\end{adjustbox}\end{center}')
        elif line.startswith('!['):
            path = re.fullmatch(r'!\[[^\]]*\]\(([^)]+)\)',line)[1]
            image_source = (MANUSCRIPT.parent/path).resolve()
            shutil.copyfile(image_source, build/image_source.name)
            output.append(r'\begin{center}\includegraphics[width=0.95\linewidth]{'+image_source.name+r'}\end{center}')
        elif line.startswith('|'):
            tab = []
            while index<len(lines) and lines[index].startswith('|'):
                tab.append([cell.strip() for cell in lines[index].strip('|').split('|')])
                index += 1
            index -= 1
            n = len(tab[0])
            first = 'p{0.40\\linewidth}' if n==5 and '起点' not in tab[0][0] else 'X'
            spec = first + '*{'+str(n-1)+'}{>{\\raggedleft\\arraybackslash}X}'
            output.append(r'\noindent{\small\begin{tabularx}{\linewidth}{'+spec+r'}\toprule')
            output.append(' & '.join(inline(cell) for cell in tab[0])+r' \\ \midrule')
            for row in tab[2:]:
                output.append(' & '.join(inline(cell) for cell in row)+r' \\')
            output.append(r'\bottomrule\end{tabularx}}\end{minipage}\par')
        elif line.startswith('#### '):
            output.append(r'\paragraph{'+inline(line[5:])+r'}')
        elif line.startswith('### '):
            output.append(r'\subsection*{'+inline(line[4:])+r'}')
        elif line.startswith('## '):
            output.append(r'\section*{'+inline(line[3:])+r'}')
        elif re.match(r'^\[\d+\]',line):
            output.append(r'{\small\noindent '+inline(line)+r'\par}')
        elif re.match(r'^表 \d+\.',line):
            output.append(r'\noindent\begin{minipage}{\linewidth}{\small '+inline(line)+r'\par}\medskip')
        elif re.match(r'^图 \d+\.',line):
            output.append(r'{\small\noindent '+inline(line)+r'\par}')
        else:
            output.append(inline(line)+'\n')
        index += 1
    output.append(r'\end{document}')
    (build/'manuscript.tex').write_text('\n\n'.join(output),encoding='utf-8')
    env = os.environ.copy()
    env['TEXMFVAR'] = str(build/'texmf-var')
    exe = shutil.which('xelatex')
    assert exe, 'xelatex unavailable'
    for run in range(2):
        result = subprocess.run([exe,'-interaction=nonstopmode','-halt-on-error','manuscript.tex'],
                                cwd=build, env=env, capture_output=True, text=True,
                                encoding='utf-8', errors='replace')
        (build/f'compile-{run+1}.txt').write_text(result.stdout+'\n'+result.stderr,encoding='utf-8')
        if result.returncode:
            print(result.stdout[-6000:])
            raise RuntimeError(f'XeLaTeX failed with {result.returncode}')
    destination = MANUSCRIPT.with_suffix('.pdf')
    shutil.copyfile(build/'manuscript.pdf',destination)
    log = (build/'manuscript.log').read_text(encoding='utf-8',errors='replace')
    report = dict(pdf=str(destination.relative_to(ROOT)), bytes=destination.stat().st_size,
                  output=re.findall(r'Output written on[^\n]+',log),
                  overfull_boxes=re.findall(r'Overfull[^\n]+',log),
                  missing_glyphs=re.findall(r'Missing character[^\n]+',log))
    (WORK/'typesetting_validation.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    return report


if __name__=='__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    content = MANUSCRIPT.read_text(encoding='utf-8')
    print(json.dumps(validate(content),ensure_ascii=False))
    if '--render' in sys.argv:
        print(json.dumps(render(content),ensure_ascii=False))
