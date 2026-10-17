"""Longest estimated jobs first; a shared queue keeps free GPUs working."""
import math
from pathlib import Path

from tools.benchmarks.common import read_json


RELATIVE_COST = {'fedavg-lora': 1., 'fedavg-lora-la': 1., 'capt': 2.3,
    'fedpurel': 1.6, 'fedntd': 1.5, 'fedlf': 1.2, 'fedyoyo': 2.2,
    'fedrela': 1.05, 'ffa-lora': .85, 'rolora': .95, 'fedsvd': 1.08,
    'lora-a2': 1.35, 'joint-old-ab': 2.3}


def estimate_seconds(job, estimates=None, stop_after=0):
    entry = (estimates or {}).get('methods', {}).get(job['method'])
    seconds = float(entry['seconds']) if entry else 20000 * RELATIVE_COST[job['method']]
    if not math.isfinite(seconds) or seconds <= 0:
        raise ValueError('Duration estimates must be positive finite seconds')
    # The measured template is CIFAR's frozen 10847-image pool.
    size = sum(map(sum, job['protocol']['matrix']))
    seconds *= size / 10847
    final = 2 if job['smoke'] else min(100, stop_after or 100)
    progress = Path(job['run_root']) / 'progress.json'
    completed = int(read_json(progress).get('completed_round', 0)) if progress.exists() else 0
    # Smoke has much less local training, but still performs all test passes.
    if job['smoke']:
        return max(1., seconds * .02 * max(0, final-completed) / 2)
    return max(1., seconds * max(0, final-completed) / 100)


def prioritize(entries):
    return sorted(entries, key=lambda e: (-e['estimated_seconds'], str(e['path'])))


def project_loads(entries, gpus):
    """LPT projection only. Actual assignment uses whichever GPU finishes first."""
    result = {gpu: dict(estimated_seconds=0., jobs=[]) for gpu in gpus}
    for entry in prioritize(entries):
        gpu = min(gpus, key=lambda g: (result[g]['estimated_seconds'], gpus.index(g)))
        result[gpu]['estimated_seconds'] += entry['estimated_seconds']
        result[gpu]['jobs'].append(entry['key'])
    return result
