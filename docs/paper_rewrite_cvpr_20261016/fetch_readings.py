"""Fetch official reading copies and a pinned CVPR layout; no downloaded code is run."""
from pathlib import Path
import concurrent.futures
import json
import subprocess
import urllib.request

WORK = Path(__file__).resolve().parent
SOURCES = {
    'moon': 'https://openaccess.thecvf.com/content/CVPR2021/papers/Li_Model-Contrastive_Federated_Learning_CVPR_2021_paper.pdf',
    'fedftg': 'https://openaccess.thecvf.com/content/CVPR2022/papers/Zhang_Fine-Tuning_Global_Model_via_Data-Free_Knowledge_Distillation_for_Non-IID_Federated_CVPR_2022_paper.pdf',
    'fedpurel': 'https://openaccess.thecvf.com/content/CVPR2026/papers/Hou_Fine-Tuning_Impairs_the_Balancedness_of_Foundation_Models_in_Long-tailed_Personalized_CVPR_2026_paper.pdf',
    'neglected_tails': 'https://openaccess.thecvf.com/content/CVPR2024/papers/Parashar_The_Neglected_Tails_in_Vision-Language_Models_CVPR_2024_paper.pdf',
    'deit_lt': 'https://openaccess.thecvf.com/content/CVPR2024/papers/Rangwani_DeiT-LT_Distillation_Strikes_Back_for_Vision_Transformer_Training_on_Long-Tailed_CVPR_2024_paper.pdf',
    'category_extrapolation': 'https://openaccess.thecvf.com/content/CVPR2025/papers/Zhao_Learning_from_Neighbors_Category_Extrapolation_for_Long-Tail_Learning_CVPR_2025_paper.pdf',
}

def fetch(item):
    key, url = item
    pdf = WORK/'readings'/f'{key}.pdf'
    pdf.parent.mkdir(parents=True, exist_ok=True)
    try:
        if not pdf.exists():
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req, timeout=45) as response:
                data = response.read()
            assert data.startswith(b'%PDF'), 'not PDF'
            pdf.write_bytes(data)
        subprocess.run(['pdftotext', '-layout', str(pdf), str(pdf.with_suffix('.txt'))], check=True, capture_output=True)
        pages = pdf.with_suffix('.txt').read_text(encoding='utf-8').count('\f')
        return dict(key=key, url=url, status='downloaded_and_extracted', pages=pages)
    except Exception as error:
        return dict(key=key, url=url, status='failed', error=str(error))

with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
    records = list(pool.map(fetch, SOURCES.items()))
for name in ['cvpr.sty', 'main.tex', 'README.md']:
    url = f'https://raw.githubusercontent.com/cvpr-org/author-kit/12909ae/{name}'
    dst = WORK/'template'/name
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        with urllib.request.urlopen(url, timeout=30) as response:
            dst.write_bytes(response.read())
        records.append(dict(key=name, url=url, status='downloaded'))
    except Exception as error:
        records.append(dict(key=name, url=url, status='failed', error=str(error)))
(WORK/'source_manifest.json').write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps(records, ensure_ascii=False, indent=2))
