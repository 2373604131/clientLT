"""Expand only Method A's correction panel into a legible mechanism schematic."""
from pathlib import Path
import json
import math
import subprocess
import sys
import re
import itertools
from build_model_schematic import PREAMBLE, prepare_images

ROOT=Path(__file__).resolve().parent
FIG=ROOT/"figures"
W,H=18,10.5
TXT={
    "zh": {
        "one":"估计客户端更新对类别的影响",
        "two":"结合当前与历史表现确定目标",
        "three":r"根据目标缺口修正共享 $A$",
        "same":r"本地同类图片\\$q=(k,c)$",
        "common":"共同模型",
        "filter":"两视图均为正才保留",
        "current":r"当前目标 $T_q^{\mathrm{cur}}$",
        "updates":r"客户端更新 $\Delta A_j$",
        "mean":r"正向影响均值 $u_q$",
        "conc":r"正向影响越集中",
        "weight":r"保持权重越大",
        "past":"已提交模型：5 轮一组",
        "register":"取最低值，改善后登记",
        "max":"取较大值",
        "target":"保持目标",
        "fixed_images":"固定训练图片",
        "visual":r"CLIP\\图像编码器",
        "fixed_text":"固定文本特征",
        "ordinary":"普通聚合",
        "margin":r"预测间隔 $F_q$",
        "gap":"未达目标",
        "ret":"保持损失",
        "classification":"分类损失",
        "cls_excess":"仅惩罚上升部分",
        "steps":"限制修正幅度 · 3 步更新",
        "train":r"仅更新 $A$",
        "frozen":r"$B$ 和 CLIP 固定",
        "schematic":"机制示意，柱高不代表实验结果",
    },
    "en": {
        "one":"Estimate class-wise update effects",
        "two":"Combine current and historical targets",
        "three":r"Correct shared $A$ using target shortfalls",
        "same":r"Class images\\$q=(k,c)$",
        "common":r"Common\\model",
        "filter":"Positive in both views",
        "current":r"Current target $T_q^{\mathrm{cur}}$",
        "updates":r"Client updates $\Delta A_j$",
        "mean":r"Mean positive effect $u_q$",
        "conc":"More concentrated",
        "weight":r"Larger retention\\weight",
        "past":"Committed models: 5-round block",
        "register":r"Register improved\\block minimum",
        "max":"Take the larger",
        "target":"Retention target",
        "fixed_images":r"Fixed local\\images",
        "visual":r"CLIP\\encoder",
        "fixed_text":"Fixed text features",
        "ordinary":"Aggregation",
        "margin":r"Prediction margin $F_q$",
        "gap":r"Target\\shortfall",
        "ret":"Retention loss",
        "classification":"Classification loss",
        "cls_excess":"Penalize only the increase",
        "steps":"Bounded correction: 3 updates",
        "train":r"Update $A$ only",
        "frozen":r"$B$ and CLIP fixed",
        "schematic":"Schematic bars; no empirical results",
    }
}


class Draw:
    def __init__(self,lang):
        self.lang=lang
        self.parts=[PREAMBLE.replace("(18,8.65)","(18,10.5)")]
        self.labels=[]
    def e(self,s): self.parts.append(s)
    def n(self,x,y,s,size=9,color="Ink",bold=False,opts=""):
        ident="label"+str(len(self.labels))
        font=rf"\fontsize{{{size}}}{{{size*1.18}}}\selectfont"+(r"\bfseries" if bold else "")
        self.e(rf"\node[align=center,text={color},font={{{font}}},{opts}] ({ident}) at ({x},{y}) {{{s}}};")
        self.e(rf"\path let \p1=({ident}.south west),\p2=({ident}.north east) in \pgfextra{{\typeout{{NODEBOUNDS|{ident}|\x1|\y1|\x2|\y2}}}};")
        self.labels.append({"id":ident,"text":s,"font_pt":size,"center_cm":[x,y]})
    def line(self,points,style="arr"):
        self.e(r"\draw["+style+"] "+" -- ".join(f"({x},{y})" for x,y in points)+";")
    def box(self,x,y,w,h,fill="white",color="Gray",radius="1pt",lw=".7pt"):
        self.e(rf"\draw[rounded corners={radius},fill={fill},draw={color},line width={lw}] ({x-w/2},{y-h/2}) rectangle ({x+w/2},{y+h/2});")
    def dot(self,x,y,r=.04,color="Ink"):
        self.e(rf"\fill[{color}] ({x},{y}) circle ({r});")
    def snow(self,x,y):
        for a in (0,60,120):
            dx,dy=.13*math.cos(math.radians(a)),.13*math.sin(math.radians(a))
            self.line([(x-dx,y-dy),(x+dx,y+dy)],"draw=Blue,line width=1pt")
            for sign in (-1,1):
                px,py=x+sign*.65*dx,y+sign*.65*dy
                for side in (-1,1):
                    ang=math.radians(a+(180 if sign<0 else 0)+180+side*50)
                    self.line([(px,py),(px+.055*math.cos(ang),py+.055*math.sin(ang))],"draw=Blue,line width=.7pt")
    def flame(self,x,y):
        self.e(rf"\begin{{scope}}[shift={{({x},{y})}}]")
        self.e(r"\path[fill=Orange,draw=Orange!80!Ink,line width=.3pt] (0,-.18) .. controls (.04,-.02) and (.17,-.08) .. (.14,.065) .. controls (.12,.2) and (-.13,.2) .. (-.14,.06) .. controls (-.17,-.04) and (-.04,-.065) .. (0,-.18) -- cycle;")
        self.e(r"\path[fill=yellow!65!white] (.014,-.02) .. controls (.07,.06) and (.085,.09) .. (.038,.14) .. controls (-.02,.17) and (-.075,.12) .. (-.043,.057) -- cycle;")
        self.e(r"\end{scope}")
    def photo(self,x,y,size=.65,flip=False):
        self.box(x,y,size+.055,size+.055,"white","Gray","0pt",".4pt")
        scale="-1" if flip else "1"
        self.e(rf"\node[xscale={scale}] at ({x},{y}) {{\includegraphics[width={size}cm]{{assets/butterfly.png}}}};")
    def feature(self,x,y,width=.8,color="Blue"):
        for j in range(5):
            self.box(x-width/2+(j+.5)*width/5,y,width/5-.015,.2,color+"!"+str((25,75,40,90,55)[j]),color+"!65","0pt",".3pt")
    def badge(self,x,y,s,color="Orange"):
        self.box(x,y,.7,.42,color+"!12",color)
        self.n(x,y,s,10.5,color)
    def heading(self,x,y,num,s,color="Green",size=10.5):
        self.e(rf"\draw[fill={color},draw={color}] ({x},{y}) circle (.19);")
        self.n(x,y-.003,str(num),9.5,"white",True)
        self.n(x+.34,y,s,size,color,True,opts="anchor=west")
    def save(self):
        self.e(r"\end{tikzpicture}")
        self.e(r"\end{document}")
        stem="tikz_method_a_detail_"+self.lang
        (FIG/(stem+".tex")).write_text("\n".join(self.parts)+"\n",encoding="utf-8")
        (FIG/(stem+".labels.json")).write_text(json.dumps(self.labels,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
        return stem


def build(lang):
    t=TXT[lang]
    d=Draw(lang)
    # Responsibility regions, not a sequence of paragraph boxes.
    d.box(9,5.04,17.94,9.98,"white","Green!50","3pt")
    d.e(r"\path[fill=Green!3] (.04,.05) rectangle (17.96,4.8);")
    d.line([(9.5,.23),(9.5,4.55)],"draw=Green!25,line width=.6pt")
    d.line([(.25,4.83),(17.75,4.83)],"draw=Green!35,line width=.7pt")
    d.heading(.47,.47,1,t["one"])
    d.heading(9.9,.47,2,t["two"],size=10 if lang=="zh" else 9.7)

    # One class on one client, two views, common-model gradient.
    d.n(1.08,1.29,t["same"],9.1)
    d.photo(.83,2.2,.65)
    d.photo(1.12,2.42,.65,True)
    d.n(1.06,3.15 if lang=="en" else 3.04,t["common"],8.8)
    d.n(1.06,3.78 if lang=="en" else 3.62,r"$F_q^0$",10,"Blue")
    d.line([(1.52,2.23),(2.64,2.23)])
    d.n(1.99,1.96,r"$g_{qv}$",10.1,"Blue")
    # Columns identify updates, rows identify views; signs encode selection.
    xs=(3.12,3.83,4.54,5.25)
    d.n(4.19,.97,t["updates"],8.8,"Blue")
    for x,index in zip(xs,("1","2","3","K")):
        d.box(x,1.35,.55,.39,"Blue!10","Blue")
        d.n(x,1.35,r"$"+index+r"$",9.2,"Blue")
    d.n(2.63,1.93,r"$v_1$",9.1)
    d.n(2.63,2.51,r"$v_2$",9.1)
    signs=((1,1,-1,1),(1,1,-1,-1))
    for v,y in enumerate((1.93,2.51)):
        for j,x in enumerate(xs):
            sign=signs[v][j]
            c="Green" if sign>0 else "Gray"
            d.box(x,y,.49,.42,c+"!13",c,".6pt",".5pt")
            d.n(x,y,r"$+$" if sign>0 else r"$-$",12,c,True)
    d.line([(4.19,2.81),(4.19,3.08)],"arr,draw=Green")
    # Positive effects after the robust two-view filter; arbitrary schematic heights.
    base=3.86
    for x,h in zip(xs,(.64,.23,0,0)):
        if h:
            d.box(x,base-h/2,.24,h,"Green!65","Green","0pt",".5pt")
        else:
            d.line([(x-.11,base),(x+.11,base)],"draw=Gray,line width=1.3pt")
    d.line([(2.87,base+.03),(5.55,base+.03)],"draw=Gray,line width=.55pt")
    d.n(4.18,4.17,t["filter"],8.8)
    d.n(3.9,4.53,r"$r_{qvj}=\langle g_{qv},\Delta A_j\rangle$",9.5,"Ink")

    # The mean determines a current improvement target; concentration determines weight.
    d.n(7.53,1.25,t["current"],9.3,"Green",True)
    d.n(7.52,1.76,r"$\min\{2,F_q^0+0.5u_q\}$",9.5,"Green")
    d.line([(5.56,3.56),(5.89,3.56),(5.89,1.76),(6.05,1.76)],"arr,draw=Green")
    d.n(7.52,2.23,t["mean"],8.8)
    d.line([(5.57,3.73),(6.15,3.73),(6.15,3.22),(6.95,3.22)],"arr,draw=Orange")
    d.n(7.63,2.73,t["conc"],9.0)
    d.badge(7.63,3.23,r"$\omega_q\uparrow$")
    d.n(7.63,3.85 if lang=="en" else 3.75,t["weight"],9.0,"Orange",True)
    # Same named orange weight is used explicitly as a multiplier in the lower loss.
    # Avoid a long crossing arrow that would obscure the actual forward model.

    # Five committed rounds; the minimum is registered only when it improves the reference.
    d.n(12.15,1.25,t["past"],9.3)
    histx=(10.58,11.21,11.84,12.47,13.10)
    base=3.2
    for x,h in zip(histx,(.66,.94,.80,.73,1.05)):
        d.box(x,base-h/2,.31,h,"Green!55","Green","0pt",".55pt")
    d.line([(10.32,base+.03),(13.44,base+.03)],"draw=Gray,line width=.6pt")
    d.line([(10.3,base-.66),(13.68,base-.66)],"draw=Green,dash pattern=on 2.8pt off 1.8pt,line width=1pt")
    d.n(13.98,2.36,r"$H_q$",10.5,"Green")
    d.n(11.88,3.78 if lang=="en" else 3.72,t["register"],9)
    d.line([(9.15,1.76),(16.45,1.76),(16.45,2.57)],"arr,draw=Green")
    d.line([(13.73,2.54),(14.54,2.54),(14.54,3.0),(16.01,3.0)],"arr,draw=Green")
    d.e(r"\draw[fill=Green!9,draw=Green,line width=1pt] (16.45,3) circle (.4);")
    d.n(16.45,3,r"$\max$",10.3,"Green")
    d.n(16.45,3.62,t["max"],9.0)
    d.line([(16.45,3.83),(16.45,4.01)],"arr,draw=Green")
    d.badge(16.45,4.25,r"$T_q$","Green")
    d.n(14.73,4.25,t["target"],9.3,"Green",True)

    # A is initialized at the ordinary aggregation; the backbone and B stay fixed.
    d.heading(.47,5.15,3,t["three"])
    d.n(1.13,6.07,t["fixed_images"],8.8)
    d.photo(.95,6.88,.67)
    d.photo(1.2,7.07,.67,True)
    d.line([(1.59,6.95),(2.16,6.95)])
    d.e(r"\draw[fill=Blue!9,draw=Blue,line width=1.1pt] (2.2,6.22) -- (4,6.55) -- (4,7.35) -- (2.2,7.68) -- cycle;")
    d.n(3.1,6.95,t["visual"],9.2 if lang=="zh" else 8.8,"Blue",True)
    d.snow(3.78,6.47)
    d.badge(2.65,8.27,r"$A$")
    d.flame(2.98,8.05)
    d.badge(3.75,8.27,r"$B$","Blue")
    d.snow(4.08,8.06)
    d.line([(3.02,8.27),(3.37,8.27)])
    d.line([(3.75,8.03),(3.75,7.43)],"arr,draw=Orange")
    d.n(3.22,8.73,"LoRA",9.1,"Ink",True)
    d.n(1.08,7.85,t["ordinary"],8.8)
    d.n(1.04,8.27,r"$A^{\mathrm{ord}}$",10.0,"Orange")
    d.line([(1.57,8.27),(2.25,8.27)],"arr,draw=Orange")

    # Frozen text features feed cosine classification.
    d.line([(4.06,6.95),(4.26,6.95)])
    d.feature(4.64,6.95,.69)
    d.line([(5.02,6.95),(5.26,6.95)])
    d.n(5.54,5.78,t["fixed_text"],8.8)
    d.feature(5.53,6.19,.92,"Green")
    d.snow(6.16,6.18)
    d.line([(5.53,6.35),(5.53,6.68)])
    d.e(r"\draw[fill=white,draw=Ink,line width=.8pt] (5.53,6.95) circle (.23);")
    d.n(5.53,6.95,r"$\odot$",13)
    d.line([(5.79,6.95),(6.34,6.95)])
    for x,h,c in ((6.72,.84,"Green"),(7.18,.54,"Gray")):
        d.box(x,7.37-h/2,.26,h,c+"!65",c,"0pt",".5pt")
    d.line([(6.45,6.4),(6.45,7.4),(7.57,7.4)],"draw=Ink,line width=.55pt")
    d.n(6.72,7.63,r"$s_c$",8.8)
    d.n(7.2,7.63,r"$s_{c'}$",8.8)
    d.n(7.02,8.1,t["margin"],8.8)
    d.line([(7.52,6.53),(7.73,6.53),(7.73,6.83),(7.52,6.83)],"draw=Green,line width=.9pt")
    d.line([(7.89,6.95),(9.31,6.95)],"arr,draw=Blue")

    # Explicit target shortfall. Only this gap is penalized.
    d.line([(16.45,4.49),(16.45,5.1),(9.93,5.1),(9.93,5.9)],"arr,draw=Green")
    d.box(9.93,6.47,.57,.95,"Orange!20","Orange","0pt",".55pt")
    d.box(9.93,7.14,.57,.39,"Blue!60","Blue","0pt",".55pt")
    d.line([(9.42,5.995),(10.5,5.995)],"draw=Green,dash pattern=on 2.5pt off 1.5pt,line width=1.2pt")
    d.line([(9.42,6.945),(10.5,6.945)],"draw=Blue,line width=1.1pt")
    d.n(10.82,5.99,r"$T_q$",10.5,"Green")
    d.n(11.37,7.05,r"$F_q(A)$",10.0,"Blue")
    d.line([(10.66,6.05),(10.56,6.05),(10.56,6.89),(10.66,6.89)],"draw=Orange,line width=1pt")
    d.n(11.58 if lang=="zh" else 12.2,6.44,t["gap"],9,"Orange",True)
    d.line([(9.93,7.4),(9.93,7.76)],"arr,draw=Orange")
    d.badge(8.98,8.12,r"$\omega_q$")
    d.n(9.52,8.12,r"$\times$",11,"Ink")
    d.n(10.88,8.12,r"$[T_q-F_q(A)]_+^2$",10.4,"Orange")
    d.n(10.24,8.66,t["ret"],9.3,"Orange",True)

    # A separate excess-risk illustration distinguishes the classification regularizer.
    d.n(14.85,5.91,t["classification"],9.3,"Ink",True)
    for x,h,c in ((14.35,.64,"Gray"),(15.21,1.17,"Blue")):
        d.box(x,7.38-h/2,.38,h,c+"!55",c,"0pt",".5pt")
    d.line([(13.98,7.4),(15.61,7.4)],"draw=Gray,line width=.6pt")
    d.line([(14.11,6.74),(15.79,6.74)],"draw=Gray,dash pattern=on 2pt off 1.5pt,line width=.7pt")
    d.box(15.21,6.475,.38,.53,"Orange!27","Orange","0pt",".65pt")
    d.line([(15.74,6.23),(15.88,6.23),(15.88,6.72),(15.74,6.72)],"draw=Orange,line width=1pt")
    d.n(14.3,7.65,r"$C^{\mathrm{ord}}$",9.2)
    d.n(15.24,7.65,r"$C(A)$",9.2)
    d.n(14.85,8.12,r"$\mathcal R_{\mathrm{cls}}$",11,"Ink")
    d.n(14.85,8.66,t["cls_excess"],8.8)

    # Gradient feedback into A (never B); weights and targets stay fixed in these updates.
    d.line([(10.24,8.91),(10.24,9.05)],"arr,draw=Orange")
    d.line([(14.85,8.92),(14.85,9.31),(10.51,9.31)])
    d.e(r"\draw[fill=white,draw=Ink,line width=.8pt] (10.24,9.31) circle (.23);")
    d.n(10.24,9.31,r"$+$",12)
    d.line([(9.97,9.31),(2.65,9.31),(2.65,8.54)],"grad")
    d.n(6.39,9.31,t["steps"],9.5,"Orange",True,opts="fill=white,inner sep=2pt")

    d.flame(.42,10.27)
    d.n(1.67,10.27,t["train"],9)
    d.snow(3.6,10.27)
    d.n(5.16,10.27,t["frozen"],9)
    d.n(13.69,10.27,t["schematic"],8.8,"Gray")
    return d.save()


def audit(stem):
    log=(FIG/(stem+".log")).read_text(encoding="utf-8",errors="replace")
    labels=json.loads((FIG/(stem+".labels.json")).read_text(encoding="utf-8"))
    bounds={}
    for m in re.finditer(r"NODEBOUNDS\|(label\d+)\|([-\d.]+)pt\|([-\d.]+)pt\|([-\d.]+)pt\|([-\d.]+)pt",log):
        x0,y0,x1,y1=map(float,m.groups()[1:])
        bounds[m.group(1)]=[min(x0,x1),min(y0,y1),max(x0,x1),max(y0,y1)]
    assert len(bounds)==len(labels)
    overlaps=[]
    for a,b in itertools.combinations(labels,2):
        aa,bb=bounds[a["id"]],bounds[b["id"]]
        dx=min(aa[2],bb[2])-max(aa[0],bb[0])
        dy=min(aa[3],bb[3])-max(aa[1],bb[1])
        if dx>.25 and dy>.25:
            overlaps.append({"a":a["text"],"b":b["text"],"overlap_pt":[dx,dy]})
    outside=[]
    for label in labels:
        x0,y0,x1,y1=bounds[label["id"]]
        if x0<-.2 or x1>W*28.45274+.2 or y0< -H*28.45274-.2 or y1>.2:
            outside.append(label["text"])
    report={"labels_measured":len(labels),"text_overlaps":overlaps,"outside_canvas":outside,
            "min_font_pt":min(l["font_pt"] for l in labels),
            "min_font_at_177_8mm_pt":round(min(l["font_pt"] for l in labels)*177.8/181.06,2),
            "overfull":bool(re.search(r"Overfull \\[hv]box",log)),
            "missing_glyph":bool(re.search(r"Missing character:",log))}
    return report


if __name__=="__main__":
    if not (ROOT/"assets/butterfly.png").exists():
        prepare_images()
    paths=json.loads((ROOT/".vivid/runtime.json").read_text(encoding="utf-8"))["paths"]
    texbin=Path(paths["xelatex"]).parent
    reports={}
    for lang in ("zh","en"):
        stem=build(lang)
        compiled=subprocess.run([paths["xelatex"],"-interaction=nonstopmode","-halt-on-error",
                                 "-output-directory=figures","figures/"+stem+".tex"],cwd=ROOT,capture_output=True)
        (FIG/(stem+".compile.txt")).write_bytes(compiled.stdout+compiled.stderr)
        if compiled.returncode:
            print(compiled.stdout.decode("utf-8",errors="replace")[-5000:])
            raise SystemExit(compiled.returncode)
        for exe,args in [
            ("pdftoppm.exe",["-png","-singlefile","-r","210","figures/"+stem+".pdf","figures/"+stem]),
            ("pdftocairo.exe",["-svg","figures/"+stem+".pdf","figures/"+stem+".svg"])]:
            subprocess.run([str(texbin/exe),*args],cwd=ROOT,check=True)
        reports[lang]=audit(stem)
        print(lang,json.dumps(reports[lang],ensure_ascii=True))
    (ROOT/"method_a_detail_geometry_review.json").write_text(json.dumps(reports,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
