"""Compile the model schematic independently of the superseded text overview."""
from pathlib import Path
import json
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
paths = json.loads((ROOT/".vivid/runtime.json").read_text(encoding="utf-8"))["paths"]
texbin = Path(paths["xelatex"]).parent
subprocess.run([sys.executable,str(ROOT/"build_model_schematic.py")],cwd=ROOT,check=True)
for lang in ("zh","en"):
    stem="tikz_method_a_model_"+lang
    result=subprocess.run([paths["xelatex"],"-interaction=nonstopmode","-halt-on-error",
                           "-output-directory=figures","figures/"+stem+".tex"],
                          cwd=ROOT,capture_output=True)
    (ROOT/"figures"/(stem+".compile.txt")).write_bytes(result.stdout+result.stderr)
    if result.returncode:
        print(result.stdout.decode("utf-8",errors="replace")[-6000:])
        raise SystemExit(result.returncode)
    for exe,args in [
        ("pdftoppm.exe",["-png","-singlefile","-r","240","figures/"+stem+".pdf","figures/"+stem]),
        ("pdftocairo.exe",["-svg","figures/"+stem+".pdf","figures/"+stem+".svg"]),
    ]:
        subprocess.run([str(texbin/exe),*args],cwd=ROOT,check=True)
    print(lang+": model schematic PDF / SVG / PNG generated")
