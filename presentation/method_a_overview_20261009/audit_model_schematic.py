"""Check actual TeX text bounds and final-scale typography for the model figure."""
from pathlib import Path
import itertools
import json
import re

ROOT=Path(__file__).resolve().parent
reports={}
for lang in ("zh","en"):
    stem="tikz_method_a_model_"+lang
    log=(ROOT/"figures"/(stem+".log")).read_text(encoding="utf-8",errors="replace")
    labels=json.loads((ROOT/"figures"/(stem+".labels.json")).read_text(encoding="utf-8"))
    bounds={}
    for m in re.finditer(r"NODEBOUNDS\|(label\d+)\|([-\d.]+)pt\|([-\d.]+)pt\|([-\d.]+)pt\|([-\d.]+)pt",log):
        x0,y0,x1,y1=map(float,m.groups()[1:])
        bounds[m.group(1)]=[min(x0,x1),min(y0,y1),max(x0,x1),max(y0,y1)]
    assert len(bounds)==len(labels),(lang,len(bounds),len(labels))
    overlaps=[]
    for a,b in itertools.combinations(labels,2):
        aa,bb=bounds[a["id"]],bounds[b["id"]]
        dx=min(aa[2],bb[2])-max(aa[0],bb[0])
        dy=min(aa[3],bb[3])-max(aa[1],bb[1])
        if dx>.25 and dy>.25:
            overlaps.append({"a":a["text"],"b":b["text"],"overlap_pt":[dx,dy]})
    outside=[]
    for l in labels:
        x0,y0,x1,y1=bounds[l["id"]]
        if x0<-.2 or x1>18*28.45274+.2 or y0< -8.65*28.45274-.2 or y1>.2:
            outside.append(l["text"])
    reports[lang]={
        "labels_measured":len(labels),"text_overlaps":overlaps,
        "outside_canvas":outside,
        "min_body_font_pt":min(l["font_pt"] for l in labels),
        "min_font_at_177_8mm_pt":round(min(l["font_pt"] for l in labels)*177.8/181.06,2),
        "overfull":bool(re.search(r"Overfull \\[hv]box",log)),
        "missing_glyph":bool(re.search(r"Missing character:",log)),
        "scope":"Text geometry only; graphic-arrow contacts require actual image review.",
    }
(ROOT/"model_geometry_review.json").write_text(json.dumps(reports,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
for lang,report in reports.items():
    print(lang,json.dumps(report,ensure_ascii=True))
if any(v["text_overlaps"] or v["outside_canvas"] or v["overfull"] or v["missing_glyph"] for v in reports.values()):
    raise SystemExit(1)
