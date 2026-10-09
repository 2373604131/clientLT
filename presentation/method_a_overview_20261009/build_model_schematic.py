"""Model-centric Method A schematic; predictions and vectors are illustrative."""
from pathlib import Path
import json
import math
import pickle
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
FIG = ROOT / "figures"
ASSETS = ROOT / "assets"
FIG.mkdir(exist_ok=True)
ASSETS.mkdir(exist_ok=True)


def prepare_images():
    data_root = REPO / "data/cifar-100/cifar-100-python"
    with (data_root / "meta").open("rb") as f:
        meta = pickle.load(f, encoding="bytes")
    with (data_root / "train").open("rb") as f:
        train = pickle.load(f, encoding="bytes")
    names = [s.decode() for s in meta[b"fine_label_names"]]
    labels = np.asarray(train[b"fine_labels"])
    manifest = []
    for name in ("apple", "bicycle", "butterfly"):
        idx = int(np.flatnonzero(labels == names.index(name))[0])
        pixels = train[b"data"][idx].reshape(3, 32, 32).transpose(1, 2, 0)
        Image.fromarray(pixels).save(ASSETS / (name + ".png"))
        manifest.append({"class": name, "train_index": idx,
                         "source": "data/cifar-100/cifar-100-python/train",
                         "usage": "Illustrative input only; not an identified feedback subset."})
    (ASSETS / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


TEXT = {
    "zh": {
        "left": "本地 CLIP 训练", "middle": "服务器聚合", "right": "聚合后的模型修正",
        "images": "本地图片", "prompt": r"a photo of [class]",
        "visual": r"图像\\编码器", "text": r"文本\\编码器",
        "features": "图像特征", "txt_features": "类别文本特征",
        "cos": "余弦相似度", "la": r"分类损失 $\mathcal L_{\mathrm{LA}}$",
        "client": "客户端", "weighted": r"按样本量\\加权聚合", "server": "服务器",
        "upload": r"上传 LoRA $A$", "local_eval": "本地固定图片",
        "correct": r"修正 $A$：3 步", "fixed_text": "固定类别文本特征",
        "effect": "客户端更新的影响", "history": "历史预测间隔",
        "rounds": "5 轮一组", "target": r"目标 $T_q$",
        "frozen": "冻结", "trainable": "训练", "stage": r"$A$ 阶段：$B$ 已完成训练与聚合",
        "grad": "损失梯度", "local_note": "图片与损失计算均在客户端",
    },
    "en": {
        "left": "Local CLIP Training", "middle": "Aggregation", "right": "Post-aggregation Correction",
        "images": "Local images", "prompt": r"a photo of [class]",
        "visual": r"Image\\encoder", "text": r"Text\\encoder",
        "features": "Image features", "txt_features": "Class text features",
        "cos": "Cosine similarity", "la": r"Classification loss $\mathcal L_{\mathrm{LA}}$",
        "client": "Client", "weighted": r"Sample-weighted\\aggregation", "server": "Server",
        "upload": r"Upload LoRA $A$", "local_eval": "Fixed local images",
        "correct": r"Update $A$: 3 steps", "fixed_text": "Fixed class text features",
        "effect": "Effects of client updates", "history": "Historical margins",
        "rounds": "5-round block", "target": r"Target $T_q$",
        "frozen": "Frozen", "trainable": "Trainable", "stage": r"$A$ stage: $B$ already trained and aggregated",
        "grad": "Loss gradient", "local_note": "Images and loss evaluation stay on clients",
    },
}

PREAMBLE = r"""
\documentclass[tikz,border=1.5pt]{standalone}
\usepackage{fontspec,xeCJK,amsmath,amssymb,lmodern,graphicx}
\setmainfont{Arial}
\setsansfont{Arial}
\setCJKmainfont{Microsoft YaHei}
\renewcommand{\familydefault}{\sfdefault}
\usetikzlibrary{arrows.meta,calc}
\definecolor{Blue}{HTML}{1479E0}
\definecolor{Orange}{HTML}{F28E2B}
\definecolor{Green}{HTML}{159B75}
\definecolor{Ink}{HTML}{24344A}
\definecolor{Gray}{HTML}{8392A5}
\definecolor{Pale}{HTML}{E4EBF3}
\tikzset{
 every node/.style={font=\fontsize{8.8}{10.2}\selectfont,text=Ink,inner sep=0pt,outer sep=0pt},
 arr/.style={-{Stealth[length=4.6pt,width=3.6pt]},line width=.85pt,draw=Ink},
 grad/.style={arr,draw=Orange,line width=1.1pt,dash pattern=on 3pt off 1.8pt},
 info/.style={arr,draw=Gray,line width=.7pt},
}
\begin{document}
\begin{tikzpicture}[x=1cm,y=-1cm]
\path[use as bounding box] (0,0) rectangle (18,8.65);
"""


def build(lang):
    t = TEXT[lang]
    out = [PREAMBLE]
    labels = []
    def emit(s):
        out.append(s)
    def node(x, y, text, size=8.8, color="Ink", bold=False, opts=""):
        ident = "label" + str(len(labels))
        font = r"\fontsize{" + str(size) + "}{" + str(size * 1.17) + r"}\selectfont"
        if bold:
            font += r"\bfseries"
        emit(rf"\node[align=center,font={{{font}}},text={color},{opts}] ({ident}) at ({x},{y}) {{{text}}};")
        emit(rf"\path let \p1=({ident}.south west),\p2=({ident}.north east) in \pgfextra{{\typeout{{NODEBOUNDS|{ident}|\x1|\y1|\x2|\y2}}}};")
        labels.append({"id": ident, "text": text, "font_pt": size, "center_cm": [x, y]})
    def path(points, style="arr"):
        emit(r"\draw[" + style + "] " + " -- ".join(f"({x},{y})" for x, y in points) + ";")
    def rect(x, y, w, h, fill, draw, radius=".8pt", lw=".8pt"):
        emit(rf"\draw[fill={fill},draw={draw},line width={lw},rounded corners={radius}] ({x-w/2},{y-h/2}) rectangle ({x+w/2},{y+h/2});")
    def snow(x, y, scale=1):
        for a in (0, 60, 120):
            dx, dy = .13 * scale * math.cos(math.radians(a)), .13 * scale * math.sin(math.radians(a))
            path([(x-dx, y-dy), (x+dx, y+dy)], "draw=Blue,line width=.95pt")
            for sign in (-1, 1):
                ex, ey = x + sign * dx * .64, y + sign * dy * .64
                for side in (-1, 1):
                    ang = math.radians(a + (180 if sign == -1 else 0) + 180 + side * 50)
                    path([(ex, ey), (ex+.052*scale*math.cos(ang), ey+.052*scale*math.sin(ang))],
                         "draw=Blue,line width=.8pt")
    def flame(x, y, scale=1):
        emit(rf"\begin{{scope}}[shift={{({x},{y})}},scale={scale}]")
        emit(r"\path[fill=Orange,draw=Orange!80!Ink,line width=.3pt] (0,-.18) .. controls (.04,-.02) and (.17,-.08) .. (.14,.065) .. controls (.12,.2) and (-.13,.2) .. (-.14,.06) .. controls (-.17,-.04) and (-.04,-.065) .. (0,-.18) -- cycle;")
        emit(r"\path[fill=yellow!65!white] (.014,-.02) .. controls (.07,.06) and (.085,.09) .. (.038,.14) .. controls (-.02,.17) and (-.075,.12) .. (-.043,.057) -- cycle;")
        emit(r"\end{scope}")
    def photo(x, y, size=.7, which="butterfly", angle=0):
        emit(rf"\begin{{scope}}[shift={{({x},{y})}},rotate={angle}]")
        rect(0, 0, size+.06, size+.06, "white", "Gray", "0pt", ".5pt")
        emit(rf"\node[inner sep=0pt] at (0,0) {{\includegraphics[width={size}cm,height={size}cm]{{assets/{which}.png}}}};")
        emit(r"\end{scope}")
    def photos(x, y, small=False):
        s = .61 if small else .7
        photo(x-.35, y+.01, s, "apple", 9)
        photo(x+.35, y-.015, s, "bicycle", -7)
        photo(x, y+.13, s, "butterfly")
    def encoder(x, y, w, h, label):
        emit(rf"\draw[fill=Blue!9,draw=Blue,line width=1.05pt] ({x-w/2},{y-h/2}) -- ({x+w/2},{y-h/2}) -- ({x+w*.33},{y+h/2}) -- ({x-w*.33},{y+h/2}) -- cycle;")
        node(x, y-.06, label, 10.3 if lang == "zh" else 9.3, "Blue", True)
        snow(x+w/2-.08, y-h/2+.06, 1.05)
    def lora(x, y):
        node(x, y-.47, "LoRA", 8.8, "Ink", True)
        rect(x, y, .65, .4, "Orange!20", "Orange")
        node(x, y, r"$A$", 11, "Ink", True)
        flame(x+.32, y-.2, .93)
        path([(x, y+.22), (x, y+.43)])
        rect(x, y+.67, .65, .4, "Blue!9", "Blue")
        node(x, y+.67, r"$B$", 11, "Ink", True)
        snow(x+.32, y+.48, .95)
    def feature(x, y, color="Blue", width=1.2):
        n = 7
        for k in range(n):
            rect(x-width/2 + (k+.5)*width/n, y, width/n-.02, .22,
                 color+"!"+str([20,70,35,90,45,65,30][k]), color+"!65", "0pt", ".35pt")
    def cosine(x, y):
        emit(rf"\draw[fill=white,draw=Ink,line width=.8pt] ({x},{y}) circle (.22);")
        node(x, y-.015, r"$\odot$", 13)
    def scores(x, y, wide=False):
        # y is the baseline, not a measured quantity or a claimed result.
        step = .38 if wide else .27
        hs = [.27,.72,.53,.21]
        left = x-1.5*step
        for j, h in enumerate(hs):
            c = "Green" if j == 1 else "Gray"
            rect(left+j*step, y-h/2, .18, h, c+"!"+("85" if j == 1 else "65"), c, "0pt", ".4pt")
        path([(left-.17, y-.86), (left-.17,y+.035), (left+3*step+.16,y+.035)], "draw=Ink,line width=.6pt")
        if wide:
            node(left+step-.025,y+.22,r"$s_c$",8.8)
            node(left+2*step+.025,y+.22,r"$s_{c'}$",8.8)
            bx=left+3*step+.35
            path([(bx-.06,y-.72),(bx+.06,y-.72),(bx,y-.72),(bx,y-.53),(bx-.06,y-.53),(bx+.06,y-.53)],"draw=Green,line width=1pt")
            node(bx+.22,y-.64,r"$F_q$",9.2,"Green")
    def server(x, y):
        for k in range(3):
            yy = y+(k-1)*.27
            rect(x, yy, 1.08, .22, "Blue!12", "Blue", ".7pt")
            for j in range(2):
                emit(rf"\fill[Green] ({x-.38+j*.12},{yy}) circle (.028);")
            path([(x+.05,yy),(x+.36,yy)],"draw=Blue,line width=.7pt")
    def laptop(x, y):
        rect(x,y,.67,.44,"white","Blue","1pt")
        rect(x,y-.015,.54,.3,"Blue!12","Blue!12","0pt")
        path([(x-.4,y+.27),(x+.4,y+.27),(x+.34,y+.33),(x-.34,y+.33),(x-.4,y+.27)],"draw=Blue,fill=Blue!15,line width=.6pt")
        # Tiny data stacks make the object a client with local data.
        for k in range(3):
            rect(x+.3,y+.37+k*.055,.24,.075,"Green!20","Green",".8pt",".4pt")
    def panel(x0, x1, title, color, size=11.8):
        rect((x0+x1)/2,4.015,x1-x0,7.97,color+"!2",color+"!50","2pt",".7pt")
        emit(rf"\path[fill={color}!10] ({x0+.01},.045) rectangle ({x1-.01},.75);")
        node((x0+x1)/2,.395,title,size,color,True)

    panel(.025,6.3,t["left"],"Blue")
    panel(6.45,9.45,t["middle"],"Blue",10.5)
    panel(9.6,17.975,t["right"],"Green",11.6)

    # Local CLIP: real pictures, two frozen encoders and explicit LoRA factors.
    node(1.6,1.05,t["images"])
    photos(1.6,1.62)
    rect(4.82,1.61,2.35,.58,"Green!9","Green")
    node(4.82,1.61,t["prompt"],8.8,"Ink")
    snow(5.84,1.33)
    path([(1.6,2.09),(1.6,2.33)])
    path([(4.82,1.92),(4.82,2.33)])
    encoder(1.6,2.97,2.13,1.25,t["visual"])
    encoder(4.82,2.97,1.98,1.25,t["text"])
    lora(3.1,2.6)
    path([(2.76,3.27),(2.38,3.27)],"arr,draw=Orange")
    path([(1.6,3.61),(1.6,3.96)])
    path([(4.82,3.61),(4.82,3.96)])
    feature(1.6,4.11)
    feature(4.82,4.11,"Green")
    node(1.6,4.45,t["features"])
    node(4.82,4.45,t["txt_features"])
    path([(1.6,4.63),(1.6,4.96),(2.94,4.96)])
    path([(4.82,4.63),(4.82,4.96),(3.42,4.96)])
    cosine(3.18,4.96)
    node(3.18,5.35,t["cos"])
    path([(3.18,5.54),(3.18,5.82)])
    scores(3.18,6.73)
    path([(3.18,6.87),(3.18,7.09)])
    node(3.18,7.39,t["la"],9.3,"Ink",True)

    # Ordinary federated aggregation: client sample weights are unchanged.
    node(7.95,1.26,t["weighted"],9.3,"Ink",True)
    server(7.95,2.94)
    node(7.95,2.24,t["server"],9.3,"Blue",True)
    for k,x in enumerate((6.95,7.95,8.95)):
        laptop(x,5.55)
        rect(x,4.67,.43,.33,"Orange!17","Orange",".4pt")
        node(x,4.67,r"$A_" + ("1","j","K")[k] + r"$",9.2)
        path([(x,5.26),(x,4.88)],"arr,draw=Blue")
        path([(x,4.46),(x,4.0),(7.56+k*.39,3.38)],"arr,draw=Blue")
    node(7.95,6.32,t["client"]+r" $1,\ldots,K$",9.2)
    node(7.95,6.93,t["upload"],9.3,"Orange",True)
    path([(8.54,2.94),(10.08,2.94)],"arr,draw=Orange,line width=1.2pt")
    node(9.3,2.62,r"$A^{\mathrm{ord}}$",10.2,"Orange",opts="fill=white,inner sep=1pt")

    # Candidate model: only A is corrected. All image evaluation is client-local.
    node(11.18,1.05,t["local_eval"])
    photos(11.18,1.62,True)
    path([(11.18,2.055),(11.18,2.33)])
    encoder(11.18,2.97,2.13,1.25,t["visual"])
    lora(12.78,2.6)
    path([(12.43,3.27),(11.94,3.27)],"arr,draw=Orange")
    path([(11.18,3.61),(11.18,3.91)])
    feature(11.18,4.08)
    node(15.46,1.92,t["fixed_text"],9.2)
    feature(15.48,2.39,"Green",1.6)
    snow(16.49,2.38)
    path([(15.48,2.56),(15.48,3.12),(13.72,3.12),(13.72,3.84)])
    path([(11.86,4.08),(13.47,4.08)])
    cosine(13.72,4.08)
    path([(13.96,4.08),(14.85,4.08)])
    scores(15.63,4.38,True)
    path([(15.63,4.78),(15.63,4.99)])
    node(15.63,5.26,r"$\mathcal L_{\mathrm{ret}}+\mathcal L_{\mathrm{cls}}$",11.5,"Ink",True)
    path([(16.55,5.26),(17.58,5.26),(17.58,1.09),(13.39,1.09),(13.39,2.6),(13.12,2.6)],"grad")
    node(15.13,1.09,t["correct"],9.4,"Orange",True,opts="fill=white,inner sep=2pt")

    # Compact graphical mechanisms: vector effects and committed-margin history.
    path([(9.84,5.66),(17.73,5.66)],"draw=Green!24,line width=.65pt")
    node(11.8,5.96,t["effect"],9.4 if lang == "zh" else 8.9,"Green",True)
    node(15.88,5.96,t["history"],9.4 if lang == "zh" else 8.9,"Green",True)
    ox,oy=10.39,7.12
    emit(rf"\fill[Ink] ({ox},{oy}) circle (.038);")
    path([(ox,oy),(11.33,6.33)],"arr,draw=Green,line width=1.4pt")
    node(11.45,6.4,r"$g_q$",9.2,"Green",opts="anchor=west")
    path([(ox,oy),(11.78,6.94)],"arr,draw=Blue,line width=1.1pt")
    path([(ox,oy),(10.88,6.39)],"arr,draw=Blue,line width=1.1pt")
    path([(ox,oy),(10.01,6.49)],"arr,draw=Gray,line width=1.1pt")
    node(11.42,7.25,r"$\Delta A_j$",9.3,"Blue")
    node(12.64,6.48,r"$\omega_q$",11,"Orange")
    node(12.62,7.25,r"$T_q^{\mathrm{cur}}$",10.2,"Green")
    path([(11.81,6.7),(12.23,6.48)],"info")
    path([(11.87,7.07),(12.19,7.23)],"info")
    path([(12.99,6.48),(13.79,6.48),(13.79,5.26),(14.66,5.26)],"arr,draw=Orange")

    base=7.26
    for k,h in enumerate((.64,.88,.77,.69,.94)):
        x=14.12+k*.38
        rect(x,base-h/2,.21,h,"Green!"+str((40,70,55,45,85)[k]),"Green","0pt",".4pt")
    path([(13.94,base+.03),(15.87,base+.03)],"draw=Gray,line width=.6pt")
    path([(13.95,base-.64),(16.06,base-.64)],"draw=Green,dash pattern=on 2pt off 1.5pt,line width=.9pt")
    node(16.25,6.5,r"$H_q$",9.8,"Green")
    node(14.94,7.57,t["rounds"],8.8)
    emit(r"\draw[fill=Green!8,draw=Green,line width=.9pt] (17.09,6.86) circle (.34);")
    node(17.09,6.86,r"$T_q$",10.5,"Green",True)
    path([(16.03,6.62),(16.5,6.62),(16.76,6.77)],"arr,draw=Green")
    path([(13.01,7.25),(13.26,7.25),(13.26,7.81),(17.09,7.81),(17.09,7.22)],"arr,draw=Green")
    path([(17.09,6.49),(17.09,5.62),(16.03,5.62),(16.03,5.48)],"arr,draw=Green")

    # One shared legend; parameter-status symbols repeat directly on the model.
    flame(.42,8.34)
    node(1.3,8.34,t["trainable"],9)
    snow(2.4,8.34)
    node(3.08,8.34,t["frozen"],9)
    path([(3.98,8.34),(4.55,8.34)],"grad")
    node(5.66,8.34,t["grad"],9)
    node(12.3,8.34,t["stage"],9.2)
    # Output models and numeric improvements are deliberately not invented.
    emit(r"\end{tikzpicture}")
    emit(r"\end{document}")
    stem="tikz_method_a_model_"+lang
    (FIG/(stem+".tex")).write_text("\n".join(out)+"\n",encoding="utf-8")
    (FIG/(stem+".labels.json")).write_text(json.dumps(labels,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")


if __name__ == "__main__":
    prepare_images()
    for language in ("zh","en"):
        build(language)
    print("Model schematic sources and illustrative input assets generated.")
