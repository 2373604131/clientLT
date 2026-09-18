"""Attribute one completed cell, or summarize the matched four-cell experiment."""
import argparse
import os
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["attribute","summary"], required=True)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--output-root", type=Path, default=Path("output/cifar100_LT/a_refresh_topology_bridge"))
    parser.add_argument("--data-root", type=Path, default=Path("DATA"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dirichlet-partition", choices=["noniid-labeldir-fine", "matched-dirichlet"],
                        default="noniid-labeldir-fine")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--normal-rounds", default="10,20,40,60,80,90,100")
    parser.add_argument("--quadrature-segments", type=int, default=1)
    args = parser.parse_args()
    os.chdir(REPO)
    if args.stage == "attribute":
        from tools.a_refresh_bridge.analysis import attribute_run
        rounds = list(range(1,101)) if args.normal_rounds=="all" else sorted(set(map(int,args.normal_rounds.split(","))))
        attribute_run(args.run_dir,args.data_root,args.device,rounds,args.quadrature_segments)
    else:
        from tools.a_refresh_bridge.summary import summarize
        summarize(args.output_root,args.seed,args.dirichlet_partition)


if __name__ == "__main__":
    main()
