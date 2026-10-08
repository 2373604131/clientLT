"""Bilingual, deterministic TikZ overview of Method A's frozen full-cp code.
Icons, score bars and response signs are schematic, not experimental results.
"""
from pathlib import Path
import json

HERE = Path(__file__).resolve().parent
FIG = HERE / "figures"
FIG.mkdir(exist_ok=True)
TEXT = {
    "zh": {
        "local": "客户端训练与反馈", "effect": "更新影响与目标计算",
        "correct": "历史保持与全局修正", "samples": "本地训练图片",
        "client": "客户端", "clip": "CLIP", "train": "训练", "fixed": "固定",
        "updates": "本轮客户端更新", "feedback": "本地类别反馈",
        "fixed_samples": r"每个 $q=(k,c)$ 固定至多 8 张",
        "views": "原图与水平翻转两个视图",
        "score": "正确类", "wrong": "最强错误类", "margin": "预测间隔",
        "base": r"共同起点 $(A_t,\bar B_t)$",
        "agg": "普通聚合：按样本量加权",
        "response": "各客户端更新对本地类别的影响",
        "positive": "两视图均为正，才保留", "signs": "示意",
        "current": "当前改善目标", "u": r"$u_q$：正向变化的均值",
        "weight": "预测间隔保持损失的权重",
        "concentration": r"$N_q^{\mathrm{eff}}$：有效正向客户端数",
        "target": "合并当前目标与有效历史", "history": "历史目标",
        "window": "五轮一组，取两视图最小值",
        "register": "持续改善后登记，下一轮生效",
        "ordinary": "修正起点与范围", "objective": "聚合后修正",
        "terms": "幅度 / 间隔保持 / 分类保持",
        "step": "3 步投影梯度下降", "output": "全局模型",
        "next": "来自过去提交的全局模型",
        "legend": r"训练：LoRA $A$ \qquad 固定：$B$ 及 CLIP 主干 \qquad 虚线：跨轮历史反馈",
    },
    "en": {
        "local": "Local Training & Feedback", "effect": "Update Effects & Targets",
        "correct": "History & Global Correction", "samples": "Local training images",
        "client": "Client", "clip": "CLIP", "train": "Train", "fixed": "Fix",
        "updates": "Client updates in this round", "feedback": "Local class feedback",
        "fixed_samples": r"Fixed images per $q=(k,c)$: $\leq8$",
        "views": "Original + horizontal flip",
        "score": "True class", "wrong": "Strongest rival", "margin": "Margin",
        "base": r"Common model $(A_t,\bar B_t)$",
        "agg": "Sample-weighted aggregation",
        "response": "Class-wise update effects",
        "positive": "Positive in both views", "signs": "Example",
        "current": "Current target", "u": r"$u_q$: mean positive effect",
        "weight": "Retention weights",
        "concentration": r"$N_q^{\mathrm{eff}}$: effective positive client count",
        "target": "Combined target",
        "history": "Historical target",
        "window": "Block minimum, both views",
        "register": "New history applies next round",
        "ordinary": "Correction region", "objective": "Global correction",
        "terms": "Proximity / retention / LA loss",
        "step": "3 projected gradient steps", "output": "Global model",
        "next": "Past committed global models",
        "legend": r"Train: LoRA $A$ \qquad Freeze: $B$ and CLIP \qquad Dashed: history feedback",
    },
}

PREAMBLE = r"""
\documentclass[tikz,border=1.5pt]{standalone}
\usepackage{fontspec}
\usepackage{xeCJK}
\usepackage{amsmath,amssymb}
\usepackage{lmodern}
\setmainfont{Arial}
\setsansfont{Arial}
\setCJKmainfont{Microsoft YaHei}
\renewcommand{\familydefault}{\sfdefault}
\usetikzlibrary{arrows.meta,calc}
\definecolor{Blue}{HTML}{1479E0}
\definecolor{Orange}{HTML}{F28E2B}
\definecolor{Green}{HTML}{159B75}
\definecolor{Ink}{HTML}{202D3D}
\definecolor{Neutral}{HTML}{65758A}
\definecolor{Line}{HTML}{C6D0DB}
\tikzset{
  every node/.style={font=\fontsize{8.6}{10.2}\selectfont,text=black,inner sep=0pt,outer sep=0pt},
  arr/.style={-{Stealth[length=4.5pt,width=3.5pt]},line width=.8pt,draw=Ink},
  bluearr/.style={arr,draw=Blue},
  orangearr/.style={arr,draw=Orange},
  greenarr/.style={arr,draw=Green},
  histarr/.style={arr,draw=Neutral,dash pattern=on 3pt off 2pt},
}
\begin{document}
\begin{tikzpicture}[x=1cm,y=-1cm]
\path[use as bounding box] (-.02,-.02) rectangle (18.02,9.35);
"""

def build(lang):
    t = TEXT[lang]
    out = [PREAMBLE]
    metadata = []
    def emit(s):
        out.append(s)
    def node(x,y,content,size=8.6,bold=False,width=None):
        size=max(size,8.3)
        content=content.replace("&",r"\&")
        style=[rf"font=\fontsize{{{size}}}{{{size+1.7}}}\selectfont"+(r"\bfseries" if bold else "")]
        if width is not None:
            style += [f"text width={width}cm","align=center"]
        name=f"label{len(metadata)}"
        emit(rf"\node[{','.join(style)}] ({name}) at ({x},{y}) {{{content}}};")
        emit(rf"\path let \p1=({name}.south west), \p2=({name}.north east) in \pgfextra{{\typeout{{NODEBOUNDS|{name}|\x1|\y1|\x2|\y2}}}};")
        metadata.append({"id":name,"text":content,"font_pt":size,"center_cm":[x,y]})
    def box(x1,y1,x2,y2,color="Line",tint=None,width=.7):
        fill="white" if tint is None else f"{color}!{tint}"
        emit(rf"\draw[draw={color},fill={fill},rounded corners=2pt,line width={width}pt] ({x1},{y1}) rectangle ({x2},{y2});")
    def arrow(points,style="arr"):
        emit(rf"\draw[{style}] "+" -- ".join(f"({x},{y})" for x,y in points)+";")
    def image_tile(x,y,color):
        box(x,y,x+.42,y+.34,color,None,.55)
        emit(rf"\fill[{color}!45] ({x+.02},{y+.31}) -- ({x+.15},{y+.11}) -- ({x+.25},{y+.25}) -- ({x+.33},{y+.17}) -- ({x+.40},{y+.31}) -- cycle;")
        emit(rf"\fill[{color}] ({x+.32},{y+.08}) circle (.025);")
    def lock(x,y):
        emit(rf"\draw[draw=Neutral,line width=.55pt] ({x-.035},{y}) -- ({x-.035},{y-.05}) arc[start angle=180,end angle=0,x radius=.035cm,y radius=.045cm] -- ({x+.035},{y});")
        emit(rf"\draw[draw=Neutral,fill=white,line width=.55pt] ({x-.06},{y}) rectangle ({x+.06},{y+.09});")

    for x1,x2,color,letter,title in [
        (0,5.1,"Blue","A",t["local"]),
        (5.45,11.55,"Orange","B",t["effect"]),
        (11.9,18,"Green","C",t["correct"]),
    ]:
        box(x1,.02,x2,8.96,color,3,.8)
        emit(rf"\draw[draw={color},line width=2pt] ({x1+.02},.03) -- ({x2-.02},.03);")
        node(x1+.25,.39,letter,10.5,True)
        node((x1+x2)/2+.10,.39,title,10.8 if lang=="zh" else 9.3,True)

    node(2.55,.92,t["samples"])
    for x,j in [(1.0,"1"),(2.55,"2"),(4.10,"K")]:
        image_tile(x-.30,1.14,"Blue")
        image_tile(x-.11,1.22,"Orange")
        arrow([(x,1.65),(x,1.94)],"bluearr")
        box(x-.64,1.98,x+.64,3.07,"Blue",None,.8)
        node(x,2.18,t["client"]+r" $"+j+"$",8.8,True)
        box(x-.54,2.38,x+.54,2.66,"Line",12,.45)
        node(x-.05,2.51,t["clip"],8.3)
        lock(x+.41,2.48)
        box(x-.54,2.77,x-.04,2.98,"Blue",12,.6)
        box(x+.06,2.77,x+.55,2.98,"Line",12,.6)
        node(x-.29,2.875,r"$A$",8.5,True)
        node(x+.30,2.875,r"$B$",8.5)
        arrow([(x,3.08),(x,3.48)],"bluearr")
        for jbar in range(3):
            box(x-.31+jbar*.205,3.51,x-.17+jbar*.205,3.82,"Blue",12+18*jbar,.45)
        node(x,4.10,rf"$\Delta A_{{{j}}}$",9.2)
    emit(r"\draw[draw=Blue,line width=.7pt] (1,4.25) -- (1,4.66) -- (4.8,4.66);")
    emit(r"\draw[draw=Blue,line width=.7pt] (2.55,4.25) -- (2.55,4.66);")
    emit(r"\draw[draw=Blue,line width=.7pt] (4.1,4.25) -- (4.1,4.66);")
    arrow([(4.8,4.66),(5.25,4.66),(5.25,1.41),(5.90,1.41)],"bluearr")
    arrow([(5.25,3.02),(5.90,3.02)],"bluearr")
    emit(r"\fill[Blue] (5.25,3.02) circle (.035);")

    box(.32,4.99,4.79,8.55,"Line",None,.65)
    node(2.55,5.23,t["feedback"],9.4,True)
    node(2.55,5.61,t["fixed_samples"])
    image_tile(.63,5.95,"Blue")
    image_tile(.87,6.03,"Blue")
    arrow([(1.42,6.21),(1.85,6.21)])
    node(3.22,5.98,t["score"],8.3)
    node(3.22,6.51,t["wrong"],8.3)
    emit(r"\draw[draw=Orange,fill=Orange!25,line width=.6pt] (2.36,6.13) rectangle (4.26,6.30);")
    emit(r"\draw[draw=Line,fill=Line!35,line width=.6pt] (2.36,6.66) rectangle (3.62,6.83);")
    emit(r"\draw[draw=Ink,line width=.6pt] (3.62,6.96) -- (4.26,6.96);")
    emit(r"\draw[draw=Ink,line width=.6pt] (3.62,6.91) -- (3.62,7.01) (4.26,6.91) -- (4.26,7.01);")
    node(3.94,7.20,t["margin"],8.3)
    node(2.55,7.55,r"$F_{qv}=\operatorname{mean}(s_c-\max_{c'\ne c}s_{c'})$",8.5)
    node(2.55,7.93,r"$g_{qv}=\nabla_A F_{qv}(A_t,\bar B_t)$",9.0)
    node(2.55,8.32,t["views"],8.3)
    arrow([(4.79,7.93),(5.36,7.93),(5.36,4.22),(5.90,4.22)],"orangearr")

    box(5.90,.89,11.13,1.93,"Blue",None,.8)
    node(8.515,1.13,t["agg"],9.3,True)
    node(8.515,1.58,r"$A^{\mathrm{ord}}=A_t+\sum_j p_j\Delta A_j$",10)
    arrow([(11.13,1.41),(12.72,1.41)],"bluearr")
    node(8.50,2.31,t["response"],9.2,True)
    box(5.90,2.67,11.13,4.59,"Orange",None,.8)
    node(8.515,2.99,r"$r_{qvj}=\langle g_{qv},\Delta A_j\rangle$",10)
    for col,label in enumerate([r"$j=1$",r"$j=2$",r"$\cdots$",r"$j=K$"]):
        node(7.04+col*.79,3.46,label,8.3)
    for row,label in enumerate([r"$q_1$",r"$q_2$"]):
        node(6.35,3.87+row*.36,label,8.5)
        for col in range(4):
            sign=[["+", "+", r"$\cdots$", "$-$"],["$-$", "+", r"$\cdots$", "$-$"]][row][col]
            positive=sign=="+"
            box(6.76+col*.79,3.71+row*.36,7.32+col*.79,4.00+row*.36,
                "Orange" if positive else "Line",15 if positive else None,.5)
            node(7.04+col*.79,3.855+row*.36,sign,9)
    node(10.49,3.99,t["signs"],8.3)
    node(8.515,4.81,t["positive"])

    arrow([(9.0,4.96),(9.0,5.19)],"orangearr")
    box(6.62,5.22,11.13,6.27,"Orange",None,.8)
    node(8.875,5.45,t["current"],9.0,True)
    node(8.875,5.85,r"$T^{\mathrm{cur}}_{qv}=\min(2,F^0_{qv}+0.5u_q)$",9.1)
    node(8.875,6.48,t["u"],8.4)
    arrow([(9.0,5.05),(6.16,5.05),(6.16,6.76),(6.62,6.76)],"orangearr")
    emit(r"\fill[Orange] (9.0,5.05) circle (.028);")
    box(6.62,6.75,11.13,8.28,"Orange",None,.8)
    node(8.875,6.99,t["weight"],9.0,True)
    node(8.875,7.49,r"$\omega_q\propto 1/N_q^{\mathrm{eff}}$",10.5)
    node(8.875,8.03,t["concentration"],8.3)
    node(8.50,8.61,r"$q=(k,c),\quad v\in\{1,2\},\quad p_j=n_j/\sum_i n_i$",8.7)

    box(12.72,.89,17.20,1.93,"Blue",None,.8)
    node(14.96,1.12,t["ordinary"],9.3,True)
    node(14.96,1.58,r"$A(z)=A^{\mathrm{ord}}+Rz,\quad \|z\|_2\leq1$",9.4)
    emit(r"\draw[draw=Blue,line width=.8pt] (12.72,1.64) -- (12.27,1.64) -- (12.27,5.63);")
    arrow([(12.27,5.88),(12.27,6.87),(12.72,6.87)],"bluearr")

    box(12.72,2.63,17.20,4.58,"Green",None,.8)
    node(14.96,2.90,t["history"]+r" $H_q$",9.6,True)
    for i,lab in enumerate(["1","2","3","4","5"]):
        xx=13.08+i*.76
        box(xx,3.19,xx+.68,3.63,"Green",9,.7)
        node(xx+.34,3.41,lab,8.5)
    node(14.96,3.92,t["window"],8.4)
    node(14.96,4.30,t["register"],8.3)
    arrow([(14.96,4.58),(14.96,5.19)],"greenarr")
    box(12.72,5.22,17.20,6.27,"Green",None,.8)
    node(14.96,5.45,t["target"],8.7,True)
    node(14.96,5.86,r"$T_{qv}=\max(T^{\mathrm{cur}}_{qv},H_q)$",10)
    arrow([(11.13,5.75),(12.72,5.75)],"orangearr")
    arrow([(14.96,6.27),(14.96,6.68)],"greenarr")

    box(12.72,6.71,17.20,7.84,"Green",None,1.05)
    node(14.96,6.94,t["objective"],9.3,True)
    node(14.96,7.30,r"$\frac12\|z\|^2+10\mathcal L_{\rm ret}+\mathcal L_{\rm cls}$",10.0)
    node(14.96,7.65,t["terms"],8.3)
    arrow([(11.13,7.37),(12.72,7.37)],"orangearr")
    node(11.93,7.10,r"$\omega_q$",9)
    arrow([(14.96,7.84),(14.96,8.03)],"greenarr")
    node(14.96,8.17,t["step"],8.3)
    arrow([(14.96,8.31),(14.96,8.40)],"greenarr")
    box(12.72,8.43,17.20,8.85,"Green",9,1)
    node(14.96,8.64,t["output"]+r" $(A_{t+1},\bar B_t)$",8.8,True)
    arrow([(17.20,8.64),(17.65,8.64),(17.65,3.93),(17.20,3.93)],"histarr")
    node(14.96,2.25,t["next"],8.3)
    node(2.55,8.78,t["base"],8.3)
    node(8.9,9.23,t["legend"],8.5)
    emit(r"\end{tikzpicture}\end{document}")
    path=FIG/f"tikz_method_a_overview_{lang}.tex"
    path.write_text("\n".join(out)+"\n",encoding="utf-8")
    (FIG/f"tikz_method_a_overview_{lang}.labels.json").write_text(
        json.dumps(metadata,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    return path

def main():
    for lang in ("zh","en"):
        print(build(lang).name)
    cfg=HERE/".vivid"/"config.json"
    config=json.loads(cfg.read_text(encoding="utf-8"))
    config.update({"palette":"custom",
                   "palette_custom":["#1479E0","#F28E2B","#159B75","#202D3D"],
                   "language":"zh","title":"Method A overview",
                   "palette_reason":"User's bright-color preference and reference"})
    cfg.write_text(json.dumps(config,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")

if __name__=="__main__":
    main()
