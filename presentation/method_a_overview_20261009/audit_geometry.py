"""Audit measured TikZ text bounds, not generic character-width estimates."""
from pathlib import Path
import json
import re
import itertools

HERE=Path(__file__).resolve().parent
reports={}
for lang in ("zh","en"):
    stem=f"tikz_method_a_overview_{lang}"
    log=(HERE/"figures"/f"{stem}.log").read_text(encoding="utf-8",errors="replace")
    labels=json.loads((HERE/"figures"/f"{stem}.labels.json").read_text(encoding="utf-8"))
    measured={}
    for m in re.finditer(r"NODEBOUNDS\|(label\d+)\|([-\d.]+)pt\|([-\d.]+)pt\|([-\d.]+)pt\|([-\d.]+)pt",log):
        vals=list(map(float,m.groups()[1:]))
        x0,y0,x1,y1=vals
        measured[m.group(1)]=[min(x0,x1),min(y0,y1),max(x0,x1),max(y0,y1)]
    assert len(measured)==len(labels),(len(measured),len(labels))
    overlap=[]
    for a,b in itertools.combinations(labels,2):
        aa,bb=measured[a["id"]],measured[b["id"]]
        dx=min(aa[2],bb[2])-max(aa[0],bb[0])
        dy=min(aa[3],bb[3])-max(aa[1],bb[1])
        if dx>.25 and dy>.25:
            overlap.append({"a":a["text"],"b":b["text"],"overlap_pt":[dx,dy]})
    outside=[]
    for label in labels:
        x0,y0,x1,y1=measured[label["id"]]
        if x0 < -.02*28.45274-.2 or x1>18.02*28.45274+.2:
            outside.append(label["text"])
    reports[lang]={
        "measurement":"TikZ label south-west and north-east anchors in TeX points",
        "labels_measured":len(measured),"text_overlaps":overlap,
        "labels_outside_canvas":outside,
        "smallest_body_font_pt":min(x["font_pt"] for x in labels),
        "embedded_math_subscripts":"Smaller script sizes are intentional; inspect actual output.",
        "overfull":bool(re.search(r"Overfull \\[hv]box",log)),
    }
(HERE/"geometry_review.json").write_text(json.dumps(reports,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
for lang,data in reports.items():
    print(lang,json.dumps(data,ensure_ascii=True))
if any(v["text_overlaps"] or v["labels_outside_canvas"] or v["overfull"] for v in reports.values()):
    raise SystemExit(1)
