from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Mapping, Sequence

from tools.semantic_acquisition.common import assert_finite


FAMILY_FILES = {
    "visual_subgroup_coverage": "visual_subgroup_coverage.csv",
    "multi_view_robustness": "multi_view_robustness.csv",
    "neighbor_discrimination_breadth": "neighbor_discrimination_breadth.csv",
}


def _csv_value(value):
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return value


def append_breadth_artifacts(
    output_dir: Path,
    results: Mapping[str, Sequence[Mapping]],
    *,
    run_metadata: Mapping,
) -> list[Path]:
    """Append all three families under one enforced evaluation schema.

    The function rejects missing or extra families so a run cannot silently
    report only whichever metric family looks favorable.
    """
    if set(results) != set(FAMILY_FILES):
        raise ValueError(
            f"breadth result families differ from contract: {sorted(results)}"
        )
    required_metadata = {"seed", "topology", "round"}
    if not required_metadata <= set(run_metadata):
        raise ValueError(
            f"run metadata lacks: {sorted(required_metadata - set(run_metadata))}"
        )
    assert_finite(run_metadata)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_keys = sorted(run_metadata)
    written = []
    for family, filename in FAMILY_FILES.items():
        rows = []
        for metric_row in results[family]:
            combined = {
                **{key: run_metadata[key] for key in metadata_keys},
                "metric_family": family,
                **metric_row,
            }
            assert_finite(combined)
            rows.append({key: _csv_value(value) for key, value in combined.items()})
        if not rows:
            raise ValueError(f"metric family {family} has no rows")
        path = output_dir / filename
        fieldnames = list(rows[0])
        if any(list(row) != fieldnames for row in rows):
            raise ValueError(f"metric family {family} produced inconsistent row schemas")
        exists = path.exists()
        if exists:
            with path.open("r", encoding="utf-8", newline="") as handle:
                existing_fields = next(csv.reader(handle), None)
            if existing_fields != fieldnames:
                raise RuntimeError(f"existing breadth artifact schema differs: {path}")
        with path.open("a", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            if not exists:
                writer.writeheader()
            writer.writerows(rows)
        written.append(path)
    return written
