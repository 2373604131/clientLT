# Phase 2: client topology to Functional Breadth

This audit uses **real client-local updates**. It does not substitute the
Carrier-B class candidates for clients.

The manifest stage rebuilds the exact seed-42 Client-LT and fixed-marginal
matched-Dirichlet partitions used by the completed SCA factorial. It verifies
the generated 30×100 count matrices against all four completed cells and
verifies their actual common 80-round, `frac=0.4` client schedule.

The two local-update stages train all 30 clients once from the same frozen
theta0 with the Carrier-B-compatible VisualLoRA substrate. This is 60 local
updates, not federated long-horizon training. Both topologies have identical
client sizes and hence identical per-client optimizer-step budgets.

Functional boundary gains are evaluated on deterministic CIFAR-100 **train**
examples excluded from the complete federated LT pool. The test split is never
opened by the Phase-2 data store.

The analysis reports two pools:

- `evidence_supporters` (primary): available clients with `N_kc > 0`, merged
  with normalized class counts;
- `all_clients` (secondary): all available clients, merged with sample-count
  FedAvg weights.

A1 makes all 30 clients available. A2 applies the actual common `frac=0.4`
schedule. Both potential (positive-only individual footprint) and actual
merged-state breadth are exported.

`BOTH` supports spatial and temporal topology→breadth arrows for this seed and
substrate. It does not establish breadth→accuracy; Phase 3 is still required.

Main outputs are `a1_spatial_summary.csv`, `a2_temporal_summary.csv`,
`paired_topology_contrasts.csv`, and `phase2_summary.json` under `analysis/`.

Stages can be run independently so the two topology-local runtimes can occupy
different GPUs: `manifests`, `clientlt`, `matched`, then `analyze`. Every saved
client state and the resumable actual-merge cache is bound to hashed inputs.
