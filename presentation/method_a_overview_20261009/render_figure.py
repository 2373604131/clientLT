"""Compile reproducible TikZ, export vector SVG and raster previews."""
from pathlib import Path
import json
import subprocess
import sys

ROOT=Path(__file__).resolve().parent
paths=json.loads((ROOT/".vivid/runtime.json").read_text(encoding="utf-8"))["paths"]
texbin=Path(paths["xelatex"]).parent
subprocess.run([sys.executable,str(ROOT/"build_figure.py")],cwd=ROOT,check=True)
for lang in ("zh","en"):
    stem=f"tikz_method_a_overview_{lang}"
    result=subprocess.run([paths["xelatex"],"-interaction=nonstopmode","-halt-on-error",
                           "-output-directory=figures",f"figures/{stem}.tex"],
                          cwd=ROOT,capture_output=True)
    (ROOT/"figures"/f"{stem}.compile.txt").write_bytes(result.stdout+result.stderr)
    if result.returncode:
        print(result.stdout.decode("utf-8",errors="replace")[-5000:])
        raise SystemExit(result.returncode)
    for exe,args in [
        ("pdftoppm.exe",["-png","-singlefile","-r","240",f"figures/{stem}.pdf",f"figures/{stem}"]),
        ("pdftocairo.exe",["-svg",f"figures/{stem}.pdf",f"figures/{stem}.svg"]),
    ]:
        subprocess.run([str(texbin/exe),*args],cwd=ROOT,check=True)
    print(f"{lang}: TEX / PDF / SVG / PNG generated")
