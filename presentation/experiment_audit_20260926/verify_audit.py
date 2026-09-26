"""Validate coverage, source preservation, workbook integrity and key reported claims."""
import csv, hashlib, json, re, zipfile
from collections import Counter
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote
from xml.etree import ElementTree as ET

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[1]
raw=json.loads((OUT/'inventory.json').read_text(encoding='utf-8'))
classified=json.loads((OUT/'classified_inventory.json').read_text(encoding='utf-8'))
rr=classified['runs'];byid={r['run_id']:r for r in rr}
assert len(rr)==263 and len(byid)==263
assert {r['run_id'] for r in raw['runs']}==set(byid)
assert len(classified['families'])==91
assert all(r['category'] and r['use'] and r['next_action'] for r in rr)
assert Counter(r['category'] for r in rr)==Counter({'必须保留':28,'待定用途':119,'建议归档':116})
assert not raw['errors']
preserved=0
for r in rr:
    assert (ROOT/r['run_path']).is_dir()
    assert (ROOT/r['preferred_result_path']).is_dir()
    if r['metrics_sha256']:
        assert hashlib.sha256((ROOT/r['curve_source']).read_bytes()).hexdigest()==r['metrics_sha256']
        preserved+=1
    if r['category']=='必须保留':
        assert r['window_start']==81 and r['window_end']==100
        assert r['incomplete_per_class_files']==0
        assert not r.get('cross_seed_identical_curve')
    if all(r.get(k) is not None for k in ['last20_head20','last20_middle60','last20_tail','last20_overall']):
        assert abs(.2*r['last20_head20']+.6*r['last20_middle60']+.2*r['last20_tail']-r['last20_overall'])<1e-7,r['run_id']

base,matched=byid['Rcc29da3dc'],byid['Rb6a37e628']
assert base['global_count_hash']==matched['global_count_hash']
assert base['client_count_hash']==matched['client_count_hash'] and base['client_count_hash']
assert abs(base['tail_peak_to_last']-31.65)<1e-8
assert abs(matched['tail_peak_to_last']-19.35)<1e-8
a,flat,s,b=map(byid.get,['R29b405139','R2026b9a85','R3b1af80e9','R852d06edf'])
assert abs(a['last20_tail']-s['last20_tail']-2.2725)<1e-8
assert abs(a['last20_overall']-s['last20_overall']+.277)<1e-8
assert abs(a['last20_tail']-flat['last20_tail']-.88)<1e-8
assert abs(b['last20_tail']-a['last20_tail']-.1525)<1e-8
assert abs(b['last20_overall']-a['last20_overall']-.0695)<1e-8
assert b['tail_peak_to_last']==a['tail_peak_to_last']==1.0
assert not any('current-cp' in r['method_recorded'] for r in rr)

with zipfile.ZipFile(OUT/'实验总账.xlsx') as z:
    assert z.testzip() is None
    for name in z.namelist():ET.fromstring(z.read(name))
    ns={'m':'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
    sheets=ET.fromstring(z.read('xl/workbook.xml')).findall('m:sheets/m:sheet',ns)
    assert len(sheets)==10
    rows=ET.fromstring(z.read('xl/worksheets/sheet3.xml')).findall('m:sheetData/m:row',ns)
    assert len(rows)==264

class Links(HTMLParser):
    def __init__(self):super().__init__();self.links=[];self.rows=0
    def handle_starttag(self,tag,attrs):
        a=dict(attrs)
        if tag=='a':self.links.append(a.get('href',''))
        if tag=='tr' and 'data-cat' in a:self.rows+=1
parser=Links();parser.feed((OUT/'实验总览.html').read_text(encoding='utf-8'))
assert parser.rows==263
checked=0
for target in parser.links:
    if not target or target.startswith('#'):continue
    assert (OUT/unquote(target)).exists(),target
    checked+=1
report=(OUT/'实验审计与取舍建议.md').read_text(encoding='utf-8')
for target in re.findall(r'\]\(([^)]+)\)',report):
    assert (OUT/target).exists(),target
    checked+=1

# Validate representative values quoted in the narrative tables against source-derived metrics.
for identifier in ['R3b1af80e9','R23269e0bb','R29b405139','R2026b9a85']:
    r=byid[identifier]
    fragment=' | '.join(f"{r[k]:.4f}" for k in ['last20_overall','last20_head20','last20_middle60','last20_tail'])
    assert fragment in report,fragment
for identifier in ['R29b405139','R5e75471ac','R852d06edf','Rc84498c4f','R42b48f11a']:
    r=byid[identifier]
    fragment=' | '.join(f"{r[k]:.4f}" for k in ['last20_overall','last20_tail','last20_few_lt20'])
    assert fragment in report,fragment

result={'coverage':'263/263运行目录和91/91族级资产已分类','source_metric_files_sha256_unchanged':preserved,
        'xlsx':'10工作表XML完整，所有运行表263行','html':'263行及筛选属性齐全；未进行浏览器渲染验收',
        'checked_existing_links':checked,'key_numeric_claims':'原始LoRA、A/S、来源消融、B及报告表格核对通过',
        'training_performed':False,'original_results_modified':False}
(OUT/'validation.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(result,ensure_ascii=False))
