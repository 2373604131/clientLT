"""Original bilingual A+B paper figure, using one scene for TikZ and Draw.io.

References guide visual grammar only; every algorithmic element follows the
2026-10-13 margin-keep-v1 + F-tail-equal joint contract. No experimental curves.
"""
from pathlib import Path
import html
import json
import math
import re
import shutil
import subprocess
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parent
FIG = ROOT / 'figures'
FIG.mkdir(exist_ok=True)
W, H = 18.0, 10.25
COLORS = dict(Ink='27313A', Muted='647078', Line='AFB8BE',
              Blue='2376A6', BlueLight='DAEAF4', BlueBg='F4F8FB',
              Orange='C77B30', OrangeLight='F7DFC4', OrangeBg='FFF8F0',
              Green='54816B', GreenLight='DCE9E1', GreenBg='F5F8F5',
              White='FFFFFF', Gray='DDE2E5', Soft='F7F8F9')


class Scene:
    def __init__(self, lang):
        self.lang, self.items = lang, []

    def rect(self, x, y, w, h, fill='White', stroke='Line', radius=0, dashed=False, lw=.65):
        self.items.append(dict(kind='rect', x=x, y=y, w=w, h=h, fill=fill,
                               stroke=stroke, radius=radius, dashed=dashed, lw=lw))

    def ellipse(self, x, y, w, h, fill='White', stroke='Line', lw=.65):
        self.items.append(dict(kind='ellipse', x=x-w/2, y=y-h/2, w=w, h=h,
                               fill=fill, stroke=stroke, lw=lw))

    def line(self, pts, color='Ink', arrow=False, dashed=False, lw=.7, both=False):
        self.items.append(dict(kind='line', pts=pts, color=color, arrow=arrow,
                               dashed=dashed, lw=lw, both=both))

    def text(self, x, y, en, zh=None, size=8.5, color='Ink', bold=False, width=None):
        text = zh if self.lang == 'zh' and zh is not None else en
        self.items.append(dict(kind='text', x=x, y=y, text=text, size=size,
                               color=color, bold=bold, max_width=width))

    def snow(self, x, y, r=.10):
        for a in (0, math.pi/3, 2*math.pi/3):
            dx, dy = r*math.cos(a), r*math.sin(a)
            self.line([(x-dx, y-dy), (x+dx, y+dy)], 'Blue', lw=.65)

    def learn(self, x, y, r=.10):
        self.line([(x-r, y+r), (x+r, y-r)], 'Orange', arrow=True, lw=1.05)

    def matrix(self, x, y, w=.66, h=.66, color='Blue', label=None):
        self.rect(x, y, w, h, color+'Light', color, lw=.7)
        for j in (1, 2, 3):
            self.line([(x+j*w/4, y), (x+j*w/4, y+h)], color, lw=.24)
            self.line([(x, y+j*h/4), (x+w, y+j*h/4)], color, lw=.24)
        if label:
            self.text(x+w/2, y+h+.24, label, size=9)

    def monitor(self, x, y, label=None, w=.94):
        self.rect(x-w/2, y-.31, w, .62, 'White', 'Muted', .03, lw=.65)
        self.line([(x, y+.31), (x, y+.43)], 'Muted')
        self.line([(x-.24, y+.43), (x+.24, y+.43)], 'Muted')
        self.rect(x-.34, y-.16, .27, .31, 'BlueLight', 'Blue', lw=.45)
        self.rect(x+.07, y-.16, .27, .31, 'OrangeLight', 'Orange', lw=.45)
        self.text(x-.205, y, '$A$', size=8.3)
        self.text(x+.205, y, '$B$', size=8.3)
        if label:
            self.text(x, y+.76, label, size=8.3)

    def data_tiles(self, x, y):
        for i, c in enumerate(('Orange', 'Orange', 'Muted', 'Muted')):
            fill = 'OrangeLight' if c == 'Orange' else 'Gray'
            self.rect(x+i*.21, y, .15, .18, fill, c, lw=.35)

    def histbars(self, x, y, heights, color='Orange', pitch=.37, bw=.22):
        self.line([(x-.1, y), (x+(len(heights)-1)*pitch+bw+.1, y)], 'Muted', lw=.5)
        for j, h in enumerate(heights):
            self.rect(x+j*pitch, y-h, bw, h, color+'Light', color, lw=.55)


def build(lang):
    s = Scene(lang)
    # Three unequal academic panels: shared architecture, B mechanism, A mechanism.
    for x, w, fill, stroke in ((.04, 4.51, 'GreenBg', 'Green'),
                                (4.72, 6.23, 'OrangeBg', 'Orange'),
                                (11.12, 6.84, 'BlueBg', 'Blue')):
        s.rect(x, .04, w, 8.75, fill, stroke, .09, lw=.6)
        s.line([(x+.14, .65), (x+w-.14, .65)], stroke, lw=.5)
    s.text(2.30, .35, '(a) Federated LoRA training', '(a) 联邦 LoRA 训练', 9.15, 'Green', True, 4.23)
    s.text(7.835, .35, '(b) Tail-class-equal feedback', '(b) 尾类等权反馈', 9.3, 'Orange', True, 5.94)
    s.text(14.54, .35, '(c) Necessary-margin retention', '(c) 必要判别余量保持', 9.3, 'Blue', True, 6.55)

    # (a) Server / clients, with data remaining beside each client.
    s.text(2.30, .91, 'Shared global model', '共享全局模型', 8.6)
    s.rect(1.80, 1.18, .66, .78, 'White', 'Muted', .02, lw=.8)
    for y in (1.27, 1.49, 1.71):
        s.rect(1.87, y, .52, .15, 'Gray', 'Muted', lw=.4)
        s.ellipse(2.28, y+.075, .035, .035, 'Green', 'Green', .2)
    s.text(3.22, 1.56, '$(A_t,B_t)$', size=9.5)
    for x in (.81, 2.28, 3.77):
        s.line([(2.13, 2.02), (x, 2.63)], 'Green', arrow=True, both=True, lw=.65)
        s.monitor(x, 3.01)
        s.text(x, 3.69, {'0.81':'Client 1', '2.28':'Client k', '3.77':'Client K'}[str(x)],
               {'0.81':'客户端 1', '2.28':'客户端 k', '3.77':'客户端 K'}[str(x)], 8.3)
    s.text(3.27, 1.94, 'joint', 'joint 聚合', size=8.3, color='Green')
    s.line([(.25, 4.02), (4.34, 4.02)], 'Line', dashed=True, lw=.55)
    s.text(2.30, 4.30, 'Visual encoder: LoRA layer', '视觉编码器中的 LoRA 层', 8.7, bold=True, width=4.05)

    # Actual LoRA branch: W0 and sBA in parallel, followed by cosine scoring.
    s.text(.42, 5.82, '$x$', size=10)
    s.line([(.65, 5.82), (1.73, 5.82)], arrow=True)
    s.ellipse(.86, 5.82, .055, .055, 'Ink', 'Ink', .3)
    s.line([(.86, 5.82), (.86, 4.93), (1.29, 4.93)], arrow=True)
    s.rect(1.31, 4.69, .63, .48, 'BlueLight', 'Blue', lw=.7)
    s.text(1.625, 4.93, '$A$', size=10)
    s.line([(1.97, 4.93), (2.48, 4.93)], arrow=True)
    s.rect(2.50, 4.69, .63, .48, 'OrangeLight', 'Orange', lw=.7)
    s.text(2.815, 4.93, '$B$', size=10)
    s.line([(3.16, 4.93), (3.72, 4.93), (3.72, 5.64)], arrow=True)
    s.rect(1.76, 5.50, 1.22, .64, 'White', 'Muted', lw=.8)
    s.text(2.36, 5.82, '$W^0$', size=10)
    s.snow(2.79, 5.61)
    s.line([(3.01, 5.82), (3.54, 5.82)], arrow=True)
    s.ellipse(3.72, 5.82, .34, .34, 'White', 'Ink')
    s.text(3.72, 5.81, '$+$', size=11)
    s.line([(3.72, 6.02), (3.72, 6.76)], arrow=True)
    s.text(1.70, 6.41, '$W=W^0+sBA$', size=8.6)
    s.rect(1.68, 6.79, 2.12, .55, 'White', 'Muted', .03, lw=.6)
    s.text(2.74, 7.065, 'Cosine scores', '余弦分类分数', 8.5, width=2.0)
    s.text(.81, 6.98, r'Text\\features', r'文本\\特征', 8.3)
    s.snow(.80, 7.40)
    s.line([(1.24, 7.06), (1.65, 7.06)], arrow=True)
    s.text(2.32, 7.77, 'Alternate local B / A learning', '本地交替学习 B / A', 8.3, width=4.15)
    s.snow(.48, 8.34)
    s.text(1.12, 8.34, 'Frozen', '冻结', 8.3)
    s.learn(2.15, 8.34)
    s.text(3.23, 8.34, 'Updated factor', '更新因子', 8.3)

    # (b) Local feedback and equal class-wise mass.
    s.text(7.835, .94, 'Refine B; hold A fixed', '修正 B，固定 A', 8.8)
    for x, label in ((5.91, '$k$'), (9.71, '$l$')):
        s.monitor(x, 1.79)
        s.data_tiles(x-.38, 2.35)
        s.text(x, 2.82, 'Local LA loss', '本地 LA 损失', 8.4)
        s.text(x-.80, 1.77, label, size=8.7)
    s.text(7.82, 1.83, r'$\cdots$', size=12)
    s.text(7.82, 2.37, 'Tail-present clients', '含尾类样本的客户端', 8.4, width=3.20)
    s.line([(5.91, 3.01), (5.91, 3.19)], 'Orange', arrow=True)
    s.line([(9.71, 3.01), (9.71, 3.19), (6.26, 3.19), (6.26, 3.28)], 'Orange', arrow=True)
    # Bars represent weights, not measured accuracies; same total mass before/after.
    s.histbars(5.43, 4.18, (.62, .30, .78), pitch=.48, bw=.29)
    s.histbars(8.90, 4.18, (.5667, .5667, .5667), pitch=.48, bw=.29)
    s.line([(7.04, 3.84), (8.40, 3.84)], 'Orange', arrow=True, lw=.95)
    s.text(7.74, 3.48, 'Rescale', '重标定', 8.5)
    s.text(6.04, 4.47, 'Original tail mass', '原始尾类权重', 8.3)
    s.text(9.54, 4.47, 'Equal per class', '尾类等权', 8.3)
    s.text(7.835, 4.99, r'$\widetilde a_i=a_i\,\beta_T/(|\mathcal T|s_{y_i})$', size=9.0, width=5.70)
    s.text(7.835, 5.42, 'Non-tail weights unchanged', '非尾类系数保持不变', 8.3)
    s.rect(4.98, 5.87, 5.68, 2.58, 'White', 'Orange', .05, dashed=True, lw=.65)
    s.text(7.82, 6.12, 'Server: shared B update', '服务器：共享 B 补充更新', 8.7, 'Orange', True)
    s.matrix(5.26, 6.65, .69, .60, 'Orange', r'$B_{\rm ord}$')
    s.text(6.32, 6.96, '$+$', size=12)
    s.matrix(6.78, 6.65, .69, .60, 'Orange', r'$\Delta B$')
    s.learn(7.60, 6.49)
    s.line([(7.74, 6.96), (8.79, 6.96)], 'Orange', arrow=True, lw=.9)
    s.matrix(9.05, 6.65, .69, .60, 'Orange', r'$B_{\rm new}$')
    s.line([(7.12, 5.65), (7.12, 5.84)], 'Orange', arrow=True, dashed=True)
    s.text(7.82, 7.86, '2 projected steps in effective coordinates', '有效更新坐标中的 2 步投影', 8.3, width=5.35)
    s.text(7.82, 8.23, r'$\|X\|_F\leq R_B$', size=8.7)

    # (c) History and classification are explicitly different sample populations.
    s.text(14.54, .94, 'Refine A; hold B fixed', '修正 A，固定 B', 8.8)
    s.text(12.91, 1.40, 'Reliable history', '可靠历史', 8.6, 'Blue', True)
    for j in range(5):
        for v in range(2):
            s.rect(11.72+j*.46, 1.75+v*.28, .38, .23,
                   'BlueLight', 'Blue', lw=.5)
    s.text(12.84, 2.52, '5 rounds, two views', '五轮、双视图', 8.3)
    s.line([(12.84, 2.71), (12.84, 2.99)], 'Blue', arrow=True)
    s.text(12.84, 3.26, r'$P_i=\min(H_i,\epsilon)$', size=9.2, width=3.05)
    s.text(16.07, 1.40, 'Fixed witnesses', '固定训练见证', 8.6, 'Blue', True)
    # Native image symbols: two views, not unverified experimental images.
    for x in (15.35,16.10):
        s.rect(x+.06, 1.78, .61, .54, 'White', 'Line', lw=.45)
        s.rect(x, 1.73, .61, .54, 'White', 'Muted', lw=.6)
        s.line([(x+.07,2.15),(x+.24,1.93),(x+.39,2.08),(x+.46,1.99),(x+.56,2.15)], 'Blue', lw=.6)
        s.ellipse(x+.46, 1.86, .07, .07, 'OrangeLight', 'Orange', .4)
    s.text(16.04, 2.52, 'All samples', '全部见证样本', 8.3)
    s.line([(16.04, 2.75), (16.04, 3.77)], 'Blue', arrow=True)
    # Return path is behind foreground nodes; it represents observations of
    # committed models, never a loss-selected candidate.
    s.line([(16.12,7.79),(17.72,7.79),(17.72,1.08),(14.36,1.08),(14.36,2.00),(14.03,2.00)],
           'Blue', arrow=True, dashed=True, lw=.60)
    s.line([(12.84, 3.51), (12.84, 3.77)], 'Blue', arrow=True)
    s.rect(11.49, 3.80, 2.75, 1.47, 'White', 'Blue', .04, lw=.65)
    s.rect(14.71, 3.80, 2.68, 1.47, 'White', 'Blue', .04, lw=.65)
    s.text(12.865, 4.07, 'Reliable samples', '可靠历史样本', 8.3, width=2.50)
    s.text(12.865, 4.52, r'$\mathcal L_{\rm keep}$', size=11, color='Blue')
    s.text(12.865, 4.98, r'$[(P_i-m_{iv})_+/\sigma]^2$', size=8.4, width=2.60)
    s.text(16.05, 4.07, 'All classes', '全部类别', 8.3)
    s.text(16.05, 4.52, r'$\mathcal L_{\rm cls}$', size=11, color='Blue')
    s.text(16.05, 4.98, 'Class-balanced LA', '类别等权 LA', 8.3, width=2.50)
    s.line([(12.865, 5.31), (12.865, 5.53), (14.46, 5.53), (14.46, 5.70)], 'Blue', arrow=True)
    s.line([(16.05, 5.31), (16.05, 5.53), (14.64, 5.53), (14.64, 5.70)], 'Blue', arrow=True)
    s.text(14.54, 5.98, r'$J=\frac12\|z\|^2+\lambda_k\mathcal L_{\rm keep}+\lambda_c\mathcal L_{\rm cls}$',
           size=9.0, width=5.83)
    s.line([(14.54, 6.19), (14.54, 6.43)], 'Blue', arrow=True)
    # Three steps are encoded as three arrows, so z0 is not mistaken for an update.
    for j,x in enumerate((11.96,13.55,15.14,16.73)):
        s.ellipse(x, 6.86, .59, .59, 'White' if j<3 else 'BlueLight', 'Blue', .75)
        s.text(x, 6.86, '$z_'+str(j)+'$', size=9.3, color='Blue')
        if j<3:
            s.line([(x+.33,6.86),(x+1.25,6.86)],'Blue',arrow=True,lw=.8)
    s.text(14.54, 7.33, r'3 projected steps; $\|z\|\leq1$', r'3 步投影更新；$\|z\|\leq1$', 8.4)
    s.text(14.54, 7.80, r'$A_{\rm new}=A_{\rm ord}+R_Az_3$', size=9.3, width=5.95)
    s.text(14.54, 8.36, 'Triggered by a reliable-margin deficit', '可靠余量出现缺口时启动', 8.3, width=5.95)

    # Separate time axis: not a second model architecture or an inference path.
    s.text(.92, 9.20, 'Round t', '第 t 轮', 8.8, bold=True)
    stages = [
        (2.20, 3.10, 'Local B + joint', '本地 B 学习与聚合', 'Soft', 'Muted'),
        (6.00, 3.10, 'B feedback (b)', 'B 反馈修正 (b)', 'OrangeLight', 'Orange'),
        (9.80, 3.10, 'Local A + joint', '本地 A 学习与聚合', 'Soft', 'Muted'),
        (13.60, 3.10, 'A retention (c)', 'A 保持与学习 (c)', 'BlueLight', 'Blue')]
    for i,(x,w,en,zh,fill,stroke) in enumerate(stages):
        s.rect(x, 8.99, w, .57, fill, stroke, .03, lw=.65)
        s.text(x+w/2, 9.275, en, zh, 8.4, width=w-.16)
        if i<3:
            s.line([(x+w+.05,9.275),(x+w+.64,9.275)],arrow=True,lw=.75)
    s.line([(16.74,9.275),(17.10,9.275)],arrow=True)
    s.text(17.52,9.275,'$(A,B)$',size=8.6)
    s.text(3.75, 9.86, 'Every round', '每轮执行', 8.3, color='Muted')
    s.text(7.55, 9.86, r'$t=30,40,\ldots,100$', size=8.3, color='Orange')
    s.text(11.35, 9.86, r'$t\leq90$', size=8.3, color='Muted')
    s.text(15.15, 9.86, r'$t\leq90$', size=8.3, color='Blue')
    return s


def tikz(scene, stem):
    out = [r'\documentclass[tikz,border=1pt]{standalone}',
           r'\usepackage{fontspec,xeCJK,amsmath,amssymb,lmodern}',
           r'\setmainfont{Times New Roman}', r'\setCJKmainfont{Microsoft YaHei}',
           r'\usetikzlibrary{arrows.meta,calc}',
           r'\tikzset{every node/.style={inner sep=0pt,outer sep=0pt,align=center}}']
    for name, value in COLORS.items():
        out.append(r'\definecolor{'+name+'}{HTML}{'+value+'}')
    out += [r'\begin{document}',r'\begin{tikzpicture}[x=1cm,y=-1cm]',
            rf'\path[use as bounding box] (0,0) rectangle ({W},{H});']
    for i,item in enumerate(scene.items):
        kind = item['kind']
        if kind in ('rect','ellipse'):
            st=[f"fill={item['fill']}",f"draw={item['stroke']}",f"line width={item['lw']}pt"]
            if item.get('dashed'):
                st.append('dash pattern=on 2.4pt off 1.8pt')
            if kind=='rect':
                if item['radius']:
                    st.append(f"rounded corners={item['radius']}cm")
                shape=f"({item['x']},{item['y']}) rectangle ({item['x']+item['w']},{item['y']+item['h']})"
            else:
                shape=f"({item['x']+item['w']/2},{item['y']+item['h']/2}) ellipse ({item['w']/2}cm and {item['h']/2}cm)"
            out.append(r'\draw['+','.join(st)+'] '+shape+';')
        elif kind=='line':
            st=[f"draw={item['color']}",f"line width={item['lw']}pt",'line join=round']
            if item['arrow']:
                tip='Stealth[length=3.7pt,width=3pt]'
                st.append(('{'+tip+'}-{'+tip+'}') if item['both'] else ('-{'+tip+'}'))
            if item['dashed']:
                st.append('dash pattern=on 2.5pt off 1.8pt')
            out.append(r'\draw['+','.join(st)+'] '+' -- '.join(f'({x},{y})' for x,y in item['pts'])+';')
        elif kind=='text':
            st=[f"text={item['color']}",rf"font=\fontsize{{{item['size']}}}{{{item['size']+1.7}}}\selectfont"+(r'\bfseries' if item['bold'] else '')]
            out.append(r'\node['+','.join(st)+f"] (n{i}) at ({item['x']},{item['y']}) {{"+item['text']+'};')
            out.append(rf'\path let \p1=($(n{i}.east)-(n{i}.west)$), \p2=($(n{i}.north)-(n{i}.south)$) in \pgfextra{{\typeout{{SIZE|{i}|\x1|\y2}}}};')
    out += [r'\end{tikzpicture}', r'\end{document}']
    path=FIG/(stem+'.tex')
    path.write_text('\n'.join(out)+'\n',encoding='utf-8')
    return path


def measured_sizes(log):
    return {int(i):(float(w)/28.45274,abs(float(h))/28.45274)
            for i,w,h in re.findall(r'SIZE\|(\d+)\|([\d.-]+)pt\|([\d.-]+)pt', log)}


def drawio(scene, stem, sizes):
    # 100 editor units per cm; measured TeX text bounds give matching box sizes.
    scale=100
    mx=ET.Element('mxfile',host='app.diagrams.net',version='26.0.0')
    diagram=ET.SubElement(mx,'diagram',id='method-ab',name='A+B overview')
    model=ET.SubElement(diagram,'mxGraphModel',dx='1800',dy='1025',grid='1',gridSize='10',
        guides='1',tooltips='1',connect='1',arrows='1',fold='1',page='1',pageScale='1',
        pageWidth=str(W*scale),pageHeight=str(H*scale),math='1',shadow='0',background='#FFFFFF')
    root=ET.SubElement(model,'root')
    ET.SubElement(root,'mxCell',id='0')
    ET.SubElement(root,'mxCell',id='1',parent='0')
    for i,d in enumerate(scene.items):
        attrs=dict(id=f'n{i}',parent='1')
        if d['kind']=='line':
            attrs.update(edge='1',value='')
            attrs['style']=f"edgeStyle=none;rounded=0;html=1;strokeColor=#{COLORS[d['color']]};strokeWidth={d['lw']*scale/28.45274:.3f};endArrow={'block' if d['arrow'] else 'none'};endFill=1;startArrow={'block' if d['both'] else 'none'};startFill=1;endSize=9;startSize=9;"+('dashed=1;dashPattern=5 4;' if d['dashed'] else '')
            cell=ET.SubElement(root,'mxCell',**attrs)
            geo=ET.SubElement(cell,'mxGeometry',relative='1',attrib={'as':'geometry'})
            for role,p in (('sourcePoint',d['pts'][0]),('targetPoint',d['pts'][-1])):
                ET.SubElement(geo,'mxPoint',x=str(p[0]*scale),y=str(p[1]*scale),attrib={'as':role})
            if len(d['pts'])>2:
                arr=ET.SubElement(geo,'Array',attrib={'as':'points'})
                for x,y in d['pts'][1:-1]:
                    ET.SubElement(arr,'mxPoint',x=str(x*scale),y=str(y*scale))
            continue
        attrs['vertex']='1'
        if d['kind']=='text':
            parts=re.split(r'(\$[^$]*\$)',d['text'])
            content=''.join((r'\('+p[1:-1]+r'\)') if p.startswith('$') else html.escape(p).replace('\\\\','<br>') for p in parts)
            attrs['value']=content
            fs=d['size']*scale/28.45274
            attrs['style']=f"text;html=1;align=center;verticalAlign=middle;whiteSpace=wrap;overflow=visible;rounded=0;strokeColor=none;fillColor=none;fontColor=#{COLORS[d['color']]};fontFamily={'Microsoft YaHei' if scene.lang=='zh' else 'Times New Roman'};fontSize={fs:.2f};fontStyle={1 if d['bold'] else 0};spacing=0;"
            w,h=sizes[i]; w+=.12; h+=.10
            x,y=d['x']-w/2,d['y']-h/2
        else:
            attrs['value']=''
            x,y,w,h=(d[k] for k in ('x','y','w','h'))
            attrs['style']=f"shape={'ellipse' if d['kind']=='ellipse' else 'rectangle'};html=1;fillColor=#{COLORS[d['fill']]};strokeColor=#{COLORS[d['stroke']]};strokeWidth={d['lw']*scale/28.45274:.3f};"
            if d.get('radius'):
                attrs['style']+='rounded=1;absoluteArcSize=1;arcSize='+str(d['radius']*scale)+';'
            if d.get('dashed'):
                attrs['style']+='dashed=1;dashPattern=5 4;'
        cell=ET.SubElement(root,'mxCell',**attrs)
        ET.SubElement(cell,'mxGeometry',x=str(x*scale),y=str(y*scale),width=str(w*scale),height=str(h*scale),attrib={'as':'geometry'})
    ET.indent(mx)
    ET.ElementTree(mx).write(FIG/(stem+'.drawio'),encoding='utf-8',xml_declaration=True)


def run():
    tex=shutil.which('xelatex')
    if not tex:
        raise RuntimeError('XeLaTeX is required')
    audits={}
    for lang in ('en','zh'):
        stem='method_ab_'+lang
        scene=build(lang)
        source=tikz(scene,stem)
        result=subprocess.run([tex,'-interaction=nonstopmode','-halt-on-error',
            '-output-directory='+str(FIG),str(source)],cwd=ROOT,capture_output=True)
        output=(result.stdout+result.stderr).decode('utf-8',errors='replace')
        (FIG/(stem+'.compile.txt')).write_text(output,encoding='utf-8')
        if result.returncode:
            raise RuntimeError(output[-5000:])
        sizes=measured_sizes(output)
        texts={i:d for i,d in enumerate(scene.items) if d['kind']=='text'}
        if set(sizes)!=set(texts):
            raise RuntimeError('Not all text bounds were captured')
        violations=[]
        for i,d in texts.items():
            w,h=sizes[i]
            if d['max_width'] and w>d['max_width']+.02:
                violations.append(dict(id=i,text=d['text'],width_cm=w,allowed_cm=d['max_width']))
            if min(d['x']-w/2,d['y']-h/2)<0 or d['x']+w/2>W or d['y']+h/2>H:
                violations.append(dict(id=i,text=d['text'],issue='outside canvas'))
        drawio(scene,stem,sizes)
        subprocess.run([shutil.which('pdftoppm'),'-png','-singlefile','-r','220',stem+'.pdf',stem],cwd=FIG,check=True,capture_output=True)
        subprocess.run([shutil.which('pdftocairo'),'-svg',stem+'.pdf',stem+'.svg'],cwd=FIG,check=True,capture_output=True)
        audits[lang]=dict(objects=len(scene.items),text_count=len(texts),min_font_pt=min(d['size'] for d in texts.values()),
            paper_width_mm=177.8,source_width_mm=180.71,min_paper_font_pt=8.3*177.8/180.71,
            text_width_violations=violations,drawio_render='not run: native Draw.io executable unavailable',
            preview_source='XeLaTeX, same scene as native Draw.io XML',raster_assets=0)
        print(lang, 'objects=',len(scene.items),'width violations=',len(violations))
        for v in violations:
            print(v)
    (ROOT/'geometry_audit.json').write_text(json.dumps(audits,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    return audits


if __name__=='__main__':
    run()
