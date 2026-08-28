# V0c fast oracle screening

V0c is the fast replacement for the exhaustive V0b diagnostic. It never
trains a model and never regenerates a dump. It consumes the existing V0
`round_state.pt` files.

## Scientific question

V0c asks whether a safe direction with useful tail-class gain exists inside
the client-disagreement trust region. It is a mechanism screen, not the final
CMSA algorithm and not a paper-ready validation protocol.

The first screen uses round 80 for seeds 1, 42, and 2026. Earlier rounds are
expanded only if this three-seed screen finds a signal.

## Search protocol

For each gamma, V0c constructs one shared candidate bank containing:

- projected support and equal-client directions;
- random disagreement-span directions;
- positive and negative SVD-axis probes at two radii;
- random convex client mixtures.

The bank is evaluated once on a deterministic class-balanced optimization
subset. All head/mid penalty scalarizations reuse these metrics. Only the
union of the top tail candidates and scalarization winners is evaluated on the
full safe split. Candidates are frozen before official-test access.

Defaults:

- round: 80;
- optimization cap: 20 examples per class;
- gamma: 0.2, 0.4, 0.8, 1.0;
- rank: at most 8;
- random span candidates: 20 per gamma;
- convex candidates: 16 per gamma;
- safe top-k: 8 plus deduplicated scalarization winners.

## Run on a three-GPU allocation

Stop the old V0b workers before launching V0c so the two searches do not share
the same GPUs. Then run:

```bash
mkdir -p output/v0_oracle_full
nohup env GPU_IDS="${CUDA_VISIBLE_DEVICES//,/ }" V0_ROOT="output/v0_oracle_full" EVAL_BATCH_SIZE=256 bash scripts/run_v0c_fast.sh > output/v0_oracle_full/v0c_launcher.log 2>&1 & echo $!
```

Monitor the launcher and all three unit logs:

```bash
tail -f output/v0_oracle_full/v0c_launcher.log output/v0_oracle_full/logs/oracle_train_v0c_fast_seed*_round080.log
```

Successful completion ends with:

```text
V0c complete: output/v0_oracle_full/summary/train_v0c_fast/v0_verdict.json
```

Inspect:

```bash
cat output/v0_oracle_full/summary/train_v0c_fast/v0_report.md
python -m json.tool output/v0_oracle_full/summary/train_v0c_fast/v0_verdict.json
```

## Decision

- `SCREEN_PASS_EXPAND_ROUNDS`: rerun with `ROUNDS="20 50 80"`; completed
  round-80 units are skipped and the summary is rebuilt over all nine units.
- `SCREEN_NO_SIGNAL`: do not spend time on rounds 20/50. Reconsider the
  carrier, trust region, or the aggregation-side bottleneck.
- `PASS`: the expanded three-seed, three-round criterion passed.

Expansion command:

```bash
nohup env GPU_IDS="${CUDA_VISIBLE_DEVICES//,/ }" V0_ROOT="output/v0_oracle_full" ROUNDS="20 50 80" EVAL_BATCH_SIZE=256 bash scripts/run_v0c_fast.sh > output/v0_oracle_full/v0c_expand_launcher.log 2>&1 & echo $!
```

Train-based selection is deliberately optimistic and is suitable only for
this mechanism screen. A paper result still requires a frozen independent
validation protocol.
