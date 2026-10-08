"""Load explicitly selected, pinned upstream definitions without CLI side effects.

No upstream repository is added to sys.path. Definition selection avoids their
top-level argument parsing, dataset downloads and absolute `utils` imports.
The receipt is verified before execution; original source files stay unchanged.
"""
import ast
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import types

ROOT = Path(__file__).resolve().parents[2] / 'third_party/paper_baselines'


@lru_cache(None)
def definitions(method, filename, names):
    path = ROOT / method / filename
    receipt = json.loads((ROOT / method / 'UPSTREAM.json').read_text(encoding='utf-8'))
    content = path.read_bytes()
    if hashlib.sha256(content).hexdigest() != receipt['files'][filename]:
        raise ValueError('Modified upstream source: ' + str(path))
    tree = ast.parse(content.decode('utf-8-sig'))
    nodes = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names]
    if {n.name for n in nodes} != set(names):
        raise ValueError('Missing upstream definitions: ' + str(names))
    import numpy as np
    import torch
    from torch import nn
    from torch.nn import functional as F
    env = dict(torch=torch, nn=nn, F=F, np=np)
    if method == 'fedyoyo' and filename.endswith('autoaug.py'):
        import random
        from PIL import Image, ImageEnhance, ImageOps
        env.update(random=random, Image=Image, ImageEnhance=ImageEnhance, ImageOps=ImageOps)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), env)
    return types.SimpleNamespace(**{name: env[name] for name in names})
