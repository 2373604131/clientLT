# Method A simple controls — seed42

Fixed endpoint: mean committed-round Tail20 accuracy over rounds 81–100. All three gamma values are reported.
One development seed only: no seed standard deviation, significance, or generalization claim.

| Method | Status | Tail20 | Overall | Reason |
|---|---|---:|---:|---|
| s | complete | 68.6625 | 70.9100 |  |
| full-cp | complete | 70.9350 | 70.6330 |  |
| tailrw-g1 | missing |  |  |  |
| tailrw-g4 | missing |  |  |  |
| tailrw-g16 | missing |  |  |  |
| cover-cp | missing |  |  |  |

Paired differences (percentage points; only protocol-compatible pairs):

| Comparison | Tail20 | Overall |
|---|---:|---:|
| full-cp minus s | +2.2725 | -0.2770 |

Cover-CP retains source-based current targets and their cost. It isolates the priority rule only.
TailRW changes both normal factor aggregation stages and has no functional correction.
Higher tail accuracy with lower overall accuracy is a tradeoff, not an automatic win.
Missing predictions in legacy references stay unavailable. Costs are simulated messages and sequential runtime.
See pair_audit.csv for incompatible pairs. Reuse compares unchanged core training code; federated_main routing differs.
