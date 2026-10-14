"""Original bilingual A+B paper figure, using one scene for TikZ and Draw.io.

References guide visual grammar only; every algorithmic element follows the
2026-10-13 margin-keep-v1 + F-tail-equal joint contract. No experimental curves.
"""
from pathlib import Path
import html
import base64
import json
import math
import re
import shutil
import subprocess
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parent
FIG = ROOT / 'figures'
FIG.mkdir(exist_ok=True)
W, H = 18.0, 12.1
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

    def polygon(self, points, fill='White', stroke='Muted', lw=.7):
        self.items.append(dict(kind='polygon', points=points, fill=fill, stroke=stroke, lw=lw))

    def photo(self, x, y, name, w=.65, h=.65):
        self.items.append(dict(kind='photo', x=x, y=y, w=w, h=h, name=name))

    def snow(self, x, y, r=.11):
        for a in (0, math.pi/3, 2*math.pi/3):
            dx,dy=r*math.cos(a),r*math.sin(a)
            self.line([(x-dx,y-dy),(x+dx,y+dy)],'Blue',lw=.65)
            for sign in (-1,1):
                ex,ey=x+sign*dx*.67,y+sign*dy*.67
                for bend in (-math.pi/4,math.pi/4):
                    vx=r*.35*math.cos(a+bend)*sign
                    vy=r*.35*math.sin(a+bend)*sign
                    self.line([(ex,ey),(ex-vx,ey-vy)],'Blue',lw=.45)

    def flame(self, x, y, r=.13):
        pts=[(-.1,-1),(.2,-.3),(.5,-.65),(.85,.1),(.75,.65),(.35,1),
             (-.3,1),(-.75,.65),(-.8,.1),(-.45,-.4),(-.4,.2),(0,-.3)]
        self.polygon([(x+u*r,y+v*r) for u,v in pts],'Orange','Orange',.3)
        self.polygon([(x-.25*r,y+.8*r),(x+.25*r,y+.8*r),(x+.16*r,y+.3*r),
                      (x,y+.45*r),(x-.04*r,y+.02*r)],'OrangeLight','OrangeLight',.15)

    def text(self, x, y, en, zh=None, size=8.5, color='Ink', bold=False, width=None):
        text = zh if self.lang == 'zh' and zh is not None else en
        self.items.append(dict(kind='text', x=x, y=y, text=text, size=size,
                               color=color, bold=bold, max_width=width))

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


def tikz(scene, stem):
    out = [r'\documentclass[tikz,border=1pt]{standalone}',
           r'\usepackage{fontspec,xeCJK,amsmath,amssymb,lmodern,graphicx}',
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
        elif kind=='polygon':
            st=f"fill={item['fill']},draw={item['stroke']},line width={item['lw']}pt,line join=round"
            out.append(r'\draw['+st+'] '+' -- '.join(f'({x},{y})' for x,y in item['points'])+' -- cycle;')
        elif kind=='photo':
            out.append(r'\node at ('+str(item['x']+item['w']/2)+','+str(item['y']+item['h']/2)+r') {\includegraphics[width='+str(item['w'])+'cm,height='+str(item['h'])+'cm]{assets/'+item['name']+'}};')
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
            content=''.join((r'\('+p[1:-1]+r'\)') if p.startswith('$') else html.escape(p).replace('\\\\','<br>').replace('{[CLASS]}','[CLASS]') for p in parts)
            attrs['value']=content
            fs=d['size']*scale/28.45274
            attrs['style']=f"text;html=1;align=center;verticalAlign=middle;whiteSpace=wrap;overflow=visible;rounded=0;strokeColor=none;fillColor=none;fontColor=#{COLORS[d['color']]};fontFamily={'Microsoft YaHei' if scene.lang=='zh' else 'Times New Roman'};fontSize={fs:.2f};fontStyle={1 if d['bold'] else 0};spacing=0;"
            w,h=sizes[i]; w+=.12; h+=.10
            x,y=d['x']-w/2,d['y']-h/2
        elif d['kind']=='polygon':
            x=min(p[0] for p in d['points']); y=min(p[1] for p in d['points'])
            w=max(p[0] for p in d['points'])-x; h=max(p[1] for p in d['points'])-y
            pad=d['lw']/28.45274
            x-=pad; y-=pad; w+=2*pad; h+=2*pad
            points=' '.join(f'{(px-x)*scale:.3f},{(py-y)*scale:.3f}' for px,py in d['points'])
            svg=f'<svg xmlns="http://www.w3.org/2000/svg" width="{w*scale}" height="{h*scale}" viewBox="0 0 {w*scale} {h*scale}"><polygon points="{points}" fill="#{COLORS[d["fill"]]}" stroke="#{COLORS[d["stroke"]]}" stroke-width="{d["lw"]*scale/28.45274}" stroke-linejoin="round"/></svg>'
            payload=base64.b64encode(svg.encode()).decode()
            attrs.update(value='',style=f'shape=image;imageAspect=0;aspect=fixed;image=data:image/svg+xml,{payload};')
        elif d['kind']=='photo':
            x,y,w,h=(d[k] for k in ('x','y','w','h'))
            payload=base64.b64encode((ROOT/'assets'/d['name']).read_bytes()).decode()
            attrs.update(value='',style=f'shape=image;imageAspect=0;aspect=fixed;image=data:image/png,{payload};')
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
        stem='framework_v3_'+lang
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
        overlaps=[]
        for i,d in texts.items():
            wi,hi=sizes[i]
            for j,e in texts.items():
                if j<=i: continue
                wj,hj=sizes[j]
                dx=min(d['x']+wi/2,e['x']+wj/2)-max(d['x']-wi/2,e['x']-wj/2)
                dy=min(d['y']+hi/2,e['y']+hj/2)-max(d['y']-hi/2,e['y']-hj/2)
                if dx>.015 and dy>.015:
                    overlaps.append(dict(first=d['text'],second=e['text'],intersection_cm=[dx,dy]))
        audits[lang]=dict(objects=len(scene.items),text_count=len(texts),min_font_pt=min(d['size'] for d in texts.values()),
            paper_width_mm=177.8,source_width_mm=180.71,min_paper_font_pt=8.3*177.8/180.71,
            text_width_violations=violations,drawio_render='not run: native Draw.io executable unavailable',
            preview_source='XeLaTeX, same scene as Draw.io XML',text_overlaps=overlaps,
            raster_assets='CIFAR-100 input thumbnails only; structure and typography are vector')
        print(lang, 'objects=',len(scene.items),'width violations=',len(violations))
        for v in violations:
            print(v)
        for v in overlaps:
            print('OVERLAP',v)
    (ROOT/'geometry_audit.json').write_text(json.dumps(audits,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    return audits

