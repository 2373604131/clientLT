"""Run one independent topology-bridge cell on the GPU allocated to this node."""

import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
from types import SimpleNamespace

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from scripts.run_cliplora_a_refresh import build_command


def prepare(args, run):
    if args.partition == 'noniid-labeldir-fine':
        from scripts.cliplora_fresh_protocol import prepare_fresh_protocol
        return prepare_fresh_protocol(args, run)
    import numpy as np
    from tools.eri_closure.protocol import build_protocol

    protocol_dir = run / "protocol"
    build_protocol(protocol_dir, data_root=args.data_root,
                   audit_rounds=[10,20,30,40,50,60,70,80,90,100])
    legacy = REPO / f"output/cifar100_LT/v2_matched/full_schedule_seed{args.seed}.json"
    source = args.schedule_file or (legacy if legacy.exists() else None)
    if source:
        payload = json.loads(Path(source).read_text(encoding="utf-8"))
        schedule = payload["schedule"] if isinstance(payload, dict) else payload
    else:
        rng = np.random.default_rng(args.seed)
        schedule = [rng.choice(30, 30, replace=False).tolist() for _ in range(100)]
    assert len(schedule) == 100 and all(sorted(row)==list(range(30)) for row in schedule)
    path = protocol_dir / "full_schedule.json"
    path.write_text(json.dumps({"schedule": schedule}, indent=2), encoding="utf-8")
    (protocol_dir/"bridge_protocol.json").write_text(json.dumps({
        "schema_version":"a_refresh_bridge_v1", "seed":args.seed,
        "topologies":["client-longtail","noniid-labeldir-fine"], "methods":["c1","c2"],
        "rank":4,"alpha":1,"scaling":0.5,"precision":"fp32","dirichlet_beta":args.dirichlet_beta,
        "normal_rounds":100,"normal_local_epochs":3,"normal_lr":0.001,
        "refresh_rounds":list(range(10,100,10)),"refresh_epochs":1,"refresh_lr":0.001,
        "dump_all_normal_and_refresh_events":True,"primary_endpoint":"last20",
        "gap_sign":"Dir-minus-CLT","probe_is_offline_only":True,
        "probe_seed":20260904,"probe_samples_per_tail_class":10,
        "protocol_copies":"independent deterministic copy per cell; no shared-file writes"
    },indent=2),encoding="utf-8")
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["run", "train", "analyze", "summary"], default="run")
    parser.add_argument("--partition", choices=["client-longtail", "noniid-labeldir-fine", "matched-dirichlet"])
    parser.add_argument("--dirichlet-beta", type=float, default=0.5)
    parser.add_argument("--dirichlet-partition", choices=["noniid-labeldir-fine", "matched-dirichlet"],
                        default="noniid-labeldir-fine", help="Dirichlet condition to summarize; matched is historical only")
    parser.add_argument("--method", choices=["c1", "c2"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-root", type=Path, default=Path("DATA"))
    parser.add_argument("--output-root", type=Path, default=Path("output/cifar100_LT/a_refresh_topology_bridge"))
    parser.add_argument("--schedule-file", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--normal-rounds", default="10,20,40,60,80,90,100")
    parser.add_argument("--quadrature-segments", type=int, default=1)
    args = parser.parse_args()
    os.chdir(REPO)
    args.data_root = args.data_root.resolve()
    args.output_root = args.output_root.resolve()
    if args.stage == "summary":
        subprocess.run([sys.executable, "-u", "scripts/analyze_cliplora_topology_bridge.py",
                        "--stage", "summary", "--output-root", str(args.output_root),
                        "--seed", str(args.seed), "--dirichlet-partition", args.dirichlet_partition], check=True)
        return
    if not args.partition or not args.method:
        parser.error("--partition and --method are required for this stage")
    run = args.output_root / f"seed{args.seed}" / args.partition / args.method
    if args.stage in ("run", "train"):
        if args.partition == 'matched-dirichlet':
            parser.error('matched-dirichlet is historical-analysis only. New training uses noniid-labeldir-fine.')
        # Never mix an earlier trajectory into this experiment's CSV/dump files.
        if run.exists() and any(run.iterdir()):
            parser.error(f"Output is not empty: {run}. Use --stage analyze for completed training, or a new --output-root.")
        schedule = prepare(args, run)
        baseline = SimpleNamespace(**vars(args), rank=4, matched_beta=args.dirichlet_beta,
                                   num_workers=8, refresh_interval=10,
                                   refresh_epochs=1, refresh_lr=0.001, resume=None)
        baseline.schedule_file = schedule
        command, _ = build_command(baseline, args.method)
        command[command.index("--output-dir")+1] = str(run)
        index = command.index("DATALOADER.NUM_WORKERS")
        command[index:index] = ["--cliplora_bridge_audit", "True"]
        (run / "command.json").write_text(json.dumps(command, indent=2), encoding="utf-8")
        print(shlex.join(command), flush=True)
        subprocess.run(command, check=True)
    if args.stage in ("run", "analyze"):
        subprocess.run([sys.executable, "-u", "scripts/analyze_cliplora_topology_bridge.py",
                        "--stage", "attribute", "--run-dir", str(run),
                        "--data-root", str(args.data_root), "--device", args.device,
                        "--normal-rounds", args.normal_rounds,
                        "--quadrature-segments", str(args.quadrature_segments)], check=True)
    print(f"Completed {args.stage}: {run}", flush=True)


if __name__ == "__main__":
    main()
