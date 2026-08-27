# V0b and V1 Offline Analysis

Both experiments consume the existing V0 dumps. They never repeat federated
training. A complete input inventory contains:

```text
output/v0_oracle_full/dumps/seed{1,42,2026}/v0_oracle/round_{020,050,080}/
  metadata.json
  round_state.pt
```

## V0b: multi-start solver audit

V0b fixes the original single-start derivative-free search. For every gamma,
it freezes four initialization families inside the same trust region:

- FedAvg;
- projected tail-support weighting;
- equal-client weighting;
- the frozen random-direction pool.

For each head/mid penalty pair, the runner selects the objective-best safe
initialization and refines it. All initializations remain in the final safe
selection pool, so the selected oracle is required to dominate a safe support
initialization on validation tail accuracy. Repeated model evaluations are
cached by candidate hash.

Run on an allocated three-GPU node:

```bash
mkdir -p output/v0_oracle_full
nohup env GPU_IDS="${CUDA_VISIBLE_DEVICES//,/ }" \
  V0_ROOT=output/v0_oracle_full \
  EVAL_BATCH_SIZE=256 \
  bash scripts/run_v0b_offline.sh \
  > output/v0_oracle_full/v0b_launcher.log 2>&1 &
```

The offline runner refuses to train if any dump is missing. It evaluates
gamma 0.2/0.4/0.8/1.0 with four solver iterations and writes:

```text
output/v0_oracle_full/oracle/train_v0b/
output/v0_oracle_full/summary/train_v0b/v0_aggregate.csv
output/v0_oracle_full/summary/train_v0b/v0_verdict.json
```

`support_regret = support tail gain - candidate tail gain`; positive oracle
regret means the oracle remained worse than the feasible support direction on
the reported test set. Gamma must not be chosen from these test results for a
paper method.

## V1: mode-representation stability

V1 uses only uploaded LoRA parameter deltas and no examples, labels, model
forward pass, validation set, or test set. It compares:

- whole-client updates;
- individual SVD atoms;
- near-degenerate singular subspaces via principal angles;
- per-transformer-block modes;
- cross-layer CountSketch modes lifted back to the full update space.

The perturbations cover client dropout, FedAvg-weight jitter, rank 4/8/16,
sketch seeds, cross-seed matching, and cross-round matching.

Run on a CPU allocation after all nine dumps exist:

```bash
OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 \
  V0_ROOT=output/v0_oracle_full/dumps \
  OUT_DIR=output/v1_mode_stability_full \
  bash scripts/run_v1_full.sh
```

Outputs:

```text
source_inventory.csv
singular_spectra.csv
mode_matches.csv
stability_units.csv
stability_summary.csv
v1_manifest.json
v1_verdict.json
```

`READY_FOR_V2` means only that a representation survived the frozen stability
thresholds. It does not show positive tail utility; V2 must test that causal
claim. Any raw LoRA-space winner must later be checked in basis-invariant
effective `BA` space with fixed-rank recompression.
