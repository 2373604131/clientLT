"""Independent probe/schedule setup; never copy Client-LT client capacities."""

import json
import shutil
from pathlib import Path


def prepare_fresh_protocol(args, run):
    import numpy as np
    from tools.eri_closure.protocol import build_protocol

    seed = getattr(args, 'protocol_seed', args.seed)
    root = Path(run) / 'protocol'
    build_protocol(root, data_root=args.data_root,
                   audit_rounds=[10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
    repo = Path(__file__).resolve().parents[1]
    bridge = Path(getattr(args, 'bridge_root', 'output/cifar100_LT/a_refresh_topology_bridge'))
    reference = bridge / f'seed{seed}' / 'client-longtail' / 'c1'
    candidates = [reference / 'protocol/full_schedule.json',
                  repo / f'output/cifar100_LT/v2_matched/full_schedule_seed{seed}.json']
    source = getattr(args, 'schedule_file', None)
    if source is None:
        source = next((p for p in candidates if p.is_file()), None)
    if source is not None:
        payload = json.loads(Path(source).read_text(encoding='utf-8'))
        schedule = payload['schedule'] if isinstance(payload, dict) else payload
    else:
        rng = np.random.default_rng(seed)
        schedule = [rng.choice(30, 30, replace=False).tolist() for _ in range(100)]
    assert len(schedule) == 100 and all(sorted(row) == list(range(30)) for row in schedule)
    schedule_path = root / 'full_schedule.json'
    schedule_path.write_text(json.dumps({'schedule': schedule}, indent=2), encoding='utf-8')
    # This reference checks common images/probes/initialization only. It must
    # never become a partition_source.csv or supply the new client sizes.
    if (reference / 'bridge_metadata.json').is_file():
        shutil.copy2(reference / 'bridge_metadata.json', root / 'reference_metadata.json')
    protocol = {
        'schema_version': 'cliplora_independent_partition_v1',
        'topology': args.partition, 'split_seed': seed,
        'dirichlet_beta': getattr(args, 'dirichlet_beta', 0.5),
        'partition_source': 'utils.datasplit.partition_data_LT',
        'reuse_clientlt_capacities': False,
        'client_weights': 'actual n_k / N from this partition',
        'budget': 'same local epochs; optimizer steps computed from actual client sizes',
        'schedule_source': str(source) if source is not None else 'default_rng full participation',
    }
    (root / 'partition_protocol.json').write_text(json.dumps(protocol, indent=2), encoding='utf-8')
    return schedule_path
