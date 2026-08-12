from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import random
from contextlib import contextmanager
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import torch


def canonical_json(value) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def stable_hash(value) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def stable_seed(*parts, bits: int = 63) -> int:
    digest = hashlib.sha256(canonical_json(list(parts)).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") & ((1 << bits) - 1)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_mapping_hash(state: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        tensor = state[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(np.asarray(tensor.shape, dtype=np.int64).tobytes())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def assert_finite(value, path: str = "root") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            assert_finite(child, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            assert_finite(child, f"{path}[{index}]")
    elif isinstance(value, (float, np.floating)) and not math.isfinite(float(value)):
        raise ValueError(f"Non-finite value at {path}: {value}")


def write_json(path: Path, value) -> None:
    assert_finite(value)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: Sequence[Mapping], fieldnames: Sequence[str] | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    if fieldnames is None:
        fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="raise")
        writer.writeheader()
        for row in rows:
            assert_finite(row)
            writer.writerow(row)


def deterministic_choice(values: Iterable[int], count: int, *seed_parts) -> list[int]:
    values = sorted(int(value) for value in values)
    if count < 0 or count > len(values):
        raise ValueError(f"Cannot choose {count} values from a pool of {len(values)}")
    generator = np.random.default_rng(stable_seed(*seed_parts))
    if count == 0:
        return []
    positions = generator.choice(len(values), size=count, replace=False)
    return [values[int(position)] for position in positions]


@contextmanager
def isolated_rng(seed: int):
    """Bind stochastic torchvision transforms to one manifested seed."""
    py_state = random.getstate()
    np_state = np.random.get_state()
    torch_state = torch.random.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    seed32 = int(seed) % (2**32)
    try:
        random.seed(seed32)
        np.random.seed(seed32)
        torch.manual_seed(int(seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(seed))
        yield
    finally:
        random.setstate(py_state)
        np.random.set_state(np_state)
        torch.random.set_rng_state(torch_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)


def flatten_spec(named_parameters: Sequence[tuple[str, torch.Tensor]]) -> tuple[list[dict], str]:
    offset = 0
    rows = []
    for name, parameter in sorted(named_parameters, key=lambda item: item[0]):
        size = int(parameter.numel())
        rows.append({
            "name": name,
            "shape": list(parameter.shape),
            "dtype": str(parameter.dtype),
            "offset_start": offset,
            "offset_end": offset + size,
        })
        offset += size
    return rows, stable_hash(rows)
