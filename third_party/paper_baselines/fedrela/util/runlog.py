import csv
import os
import json
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any


@dataclass
class RoundLogRow:
    rnd: int
    acc_overall: float
    acc_head: float
    acc_middle: float
    acc_tail: float
    avg_local_train_loss: float
    num_users_trained: int
    relabel_start: int
    relabel_period: int
    alpha_dirichlet: float
    IF: float
    seed: int
    overall_peak: int
    snapshot_tag: str


@dataclass
class RealignRoundLogRow:
    rnd: int
    acc_overall: float
    acc_head: float
    acc_middle: float
    acc_tail: float
    relabel_start: int
    seed: int


class RunLogger:
    def __init__(self, log_dir: str, run_id: str, filename_suffix: str = ""):
        os.makedirs(log_dir, exist_ok=True)
        safe_suffix = f"_{filename_suffix}" if filename_suffix else ""
        self.path = os.path.join(log_dir, f"{run_id}{safe_suffix}.csv")
        self._writer = None
        self._fp = None

    def _ensure_open(self, fieldnames):
        if self._writer is not None:
            return
        new_file = not os.path.exists(self.path)
        self._fp = open(self.path, "a", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._fp, fieldnames=fieldnames)
        if new_file:
            self._writer.writeheader()

    def log_round(self, row: RoundLogRow):
        d = asdict(row)
        self._ensure_open(fieldnames=list(d.keys()))
        self._writer.writerow(d)
        self._fp.flush()

    def close(self):
        if self._fp is not None:
            self._fp.close()
            self._fp = None
            self._writer = None


def parse_epoch_milestones(milestones: str, rounds: int):
    """
    Parse comma-separated fractions into sorted unique integer round endpoints (1-based).
    Each fraction f produces endpoint t = int(f * rounds), clamped to [1, rounds].
    We snapshot model state at the end of round rnd when (rnd + 1) == t.
    """
    if not milestones:
        return []
    fracs = []
    for part in milestones.split(","):
        part = part.strip()
        if not part:
            continue
        fracs.append(float(part))
    endpoints = []
    for f in fracs:
        t = int(f * rounds)
        t = max(1, min(t, rounds))
        endpoints.append(t)
    return sorted(set(endpoints))


def save_confusion_json(log_dir: str, run_id: str, snapshot_tag: str, rnd: int, confusion, labels):
    """Save confusion matrix (nested list) as JSON for markdown export."""
    os.makedirs(log_dir, exist_ok=True)
    conf_dir = os.path.join(log_dir, "confusion")
    os.makedirs(conf_dir, exist_ok=True)
    path = os.path.join(conf_dir, f"{run_id}_{snapshot_tag}.json")
    payload = {
        "run_id": run_id,
        "snapshot_tag": snapshot_tag,
        "rnd": int(rnd),
        "labels": list(labels),
        "confusion": confusion,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)
    return path
