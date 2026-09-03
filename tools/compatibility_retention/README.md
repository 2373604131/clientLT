# Compatibility-to-Retention Bridge

This is the single final bridge experiment following the completed
`tools.boundary_evidence` study. It does not alter or reinterpret the failed
pairwise-accuracy gate. Instead, it tests the newly observed margin-level
mechanism:

```text
under-constrained c+r specialization -> lower retention under identical rewriting
```

For each data seed, the runner reconstructs the exact controlled Client-LT
partition and trains every real client once from the same pre-federation
`theta0`. For each tail class `c`, it selects all clients with `n_k,c = 0` and
sample-count FedAvg-aggregates their updates. This produces one
`Delta_bg,c`. The exact same serialized background state is then applied to
every `c+h` and `c+r` pair for that seed and tail class:

```text
theta_post = theta0 + (theta_local - theta0) + (theta_bg - theta0)
```

Both update scales are fixed at one. There is no norm matching, scale sweep,
multi-round replay, alternate checkpoint, or alternate endpoint.

The sole primary endpoint is:

```text
G_local = M_c(theta_local) - M_c(theta0)
G_post  = M_c(theta_post)  - M_c(theta0)
R_c     = G_post / G_local
```

The five hard-negative pairs and both data seeds are averaged within each tail
class before forming the ratio. The 20 resulting tail-class ratios are the
inference units. The bridge is supported only when the mean contrast
`R(c+h)-R(c+r)` is positive and its 95% tail-class bootstrap interval excludes
zero.

Run after the source `output/boundary_evidence` experiment is complete:

```bash
bash scripts/run_compatibility_retention_bridge.sh
```

Or resume individual stages:

```bash
python -m tools.compatibility_retention.run --stage prepare
python -m tools.compatibility_retention.run --stage background
python -m tools.compatibility_retention.run --stage bridge
python -m tools.compatibility_retention.run --stage summarize
```

The result can directly link under-constrained specialization to reduced
retention under one identical real class-absent shared update. It cannot assign
all of the observed 13.85pp final accuracy gap to this mechanism.

## V2 background-adjusted correction

The completed V1 run showed that the class-absent background update directly
increased the target margin by far more than either local tail update. Its
original `G_post/G_local` ratio therefore became dominated by the unequal
denominators and is not retention evidence. V1 artifacts and provenance remain
unchanged, but its directional verdict is superseded.

V2 adds only the missing background-only evaluation and uses:

```text
R*_c = [M(theta0 + delta_tail + delta_bg) - M(theta0 + delta_bg)]
       / [M(theta0 + delta_tail) - M(theta0)]
```

No client is retrained. Run V2 on the same compute server while the V1
`background_aggregate_states` directory is still present:

```bash
bash scripts/run_compatibility_retention_bridge_v2.sh
```

The corrected result is written separately to
`output/compatibility_retention_bridge_v2`.
