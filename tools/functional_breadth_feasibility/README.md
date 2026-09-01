# Phase 0 + Phase 1: Functional Breadth feasibility

This pipeline intentionally performs **no training**.

## Phase 0

Phase 0 recursively audits legacy `run.log`/`log.txt` files and retains only
explicit `--partition client-longtail --frac 1.0` runs. It records the model,
parameter carrier, Client-LT controls, client count, local epochs, seed,
final/best tail accuracy, best-to-final drop, and trajectory availability.
Duplicate copied logs are collapsed by their launch command.

The output is descriptive. It must not be used as a causal comparison against
the current SCA experiment because the parameter carrier and protocol differ.

## Phase 1

Phase 1 is a CUDA forward-only analysis. It requires these completed artifacts:

- `output/carrier_access_audit/manifests`
- 80 `output/carrier_access_audit/experiment_b/candidate_states/*.pt` files
- 20 `output/post_write_rewrite_audit/d1/tail_writer_states/*.pt` files
- `output/e1_strength_breadth/protocol_v2/theta0_seed42.pt`

If any prerequisite is absent or stale, Phase 1 fails. It never calls the
Carrier-B or D1 training routines to recreate missing data.

All selection and evaluation evidence comes from frozen or deterministically
held-out **CIFAR-100-LT training samples**. A train-only data store rejects every
non-train sample ID and never opens the CIFAR test file.

This centralized access is acceptable for a simulator-side mechanism audit, but
it is **not** a privacy-preserving server-side FL method. The resulting matcher
must not be deployed or described as the final aggregation algorithm.

Phase 1 exports:

- validated candidate update tensors and state inventory;
- private gain for every candidate × tail × frozen hard boundary;
- held-out head/non-target harm;
- cosine to each direct-tail update;
- all 3,160 two-candidate pair screens per tail class;
- forward-evaluated actual merged boundary gains for the shortlist;
- one Broad/Narrow match decision per tail class.

`FEASIBLE` requires at least 12/20 tail classes to have an actual merged pair
that satisfies all preregistered strength, norm, safety, cosine, budget, and
breadth-separation gates. It only authorizes the next adaptation experiment; it
does not claim an accuracy or retention benefit.

## Run

From the repository root on one rented GPU:

```bash
python -u scripts/run_functional_breadth_p0_p1.py --stage all --gpu 0
```

The combined report is written to
`output/functional_breadth_p0_p1_seed42/p0_p1_report.md`.
