"""Refresh final reporting after the already-running dispatcher exits; no training actions."""
import json
from pathlib import Path
import subprocess
import sys
import time

root = Path('/data/yzh/clientLT/output/cifar100_LT/ab_parameter_search_20261014_01')
repo = Path('/data/yzh/clientLT/tmp/ab_parameter_search_20261014_01')
pid = json.loads((root/'server_launch.json').read_text())['pid']
proc = Path('/proc')/str(pid)/'stat'
def identity():
    try:
        fields = proc.read_text().rsplit(')', 1)[1].split()
        return None if fields[0] == 'Z' else fields[19]
    except FileNotFoundError:
        return None
start = identity()
print('Waiting for existing dispatcher', pid, 'without modifying any training process', flush=True)
while start is not None and identity() == start:
    time.sleep(30)
command = [sys.executable, '-u', str(repo/'scripts/run_cliplora_ab_parameter_search.py'),
           '--stage', 'summary', '--output-root', str(root)]
result = subprocess.run(command, cwd=repo)
report = json.loads((root/'analysis/report.json').read_text()) if (root/'analysis/report.json').exists() else {}
record = dict(finished_at=time.time(), exit_code=result.returncode,
              all_six_completed_and_audited=report.get('all_six_completed_and_audited', False),
              errors=report.get('errors', {}), reporting_only=True)
(root/'finalization_receipt.json').write_text(json.dumps(record, indent=2)+'\n')
print(json.dumps(record, indent=2), flush=True)
sys.exit(0 if result.returncode == 0 and record['all_six_completed_and_audited'] else 1)
