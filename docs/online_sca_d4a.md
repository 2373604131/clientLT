# Online class-separable aggregation + D4-A

This experiment is the first non-oracle implementation after P0.

## Method boundary

- Shared vision-LoRA tensors: ordinary sample-weighted FedAvg.
- Tail class residual table: client rows are trained only for locally observed
  positive tail labels. The server aggregates row `c` only from selected
  clients containing class `c`, normalized by their class-`c` sample counts.
- Unsupported class row: retain the previous global row.
- No validation/test utility, class-wise oracle, adapter bank, or CLIP router
  is used by training.
- The primary run disables residual bias, so a gain cannot be explained by a
  disguised scalar logit adjustment.

This is a CAPT-like relaxed-information setting: the server consumes client
training class counts. It must not later be described as class-distribution
private.

## Why participation is 0.4

The launcher uses the same seed-42 schedule for FedAvg and SCA, with 12 of 30
clients selected per round. Partial participation is required for D4-A to
observe temporary supporter absences; under full participation the supporter
set is almost static.

## Outputs

- `online_sca_protocol.json`: frozen online protocol and information use.
- `round_metrics.csv`: ordinary global metrics.
- `d4a/d4a_per_class_round.csv`: supporter count, absence streak, retained-row
  status, class accuracy, true-class margin, and degradation from historical
  best. Test metrics are explicitly diagnostic and never control training.
  Margin is collected inside the existing global-test pass, so D4-A does not
  trigger a duplicate traversal of the test set.
- `online_sca_summary.json`: final matched comparison when both conditions are
  present.

## Foreground commands

Run SCA first:

```bash
STAGE=sca GPU=0 DATA_ROOT=DATA OUT_ROOT=output/online_sca_seed42 \
FREEZE=output/g0_d1_seed42/lora_freeze.json bash scripts/run_online_sca.sh
```

Then run the matched FedAvg carrier with the identical schedule:

```bash
STAGE=baseline GPU=0 DATA_ROOT=DATA OUT_ROOT=output/online_sca_seed42 \
FREEZE=output/g0_d1_seed42/lora_freeze.json bash scripts/run_online_sca.sh
```

Finally summarize without a GPU:

```bash
python -u scripts/run_online_sca.py --stage summary \
  --output-root output/online_sca_seed42
```

All launchers run in the foreground. A traceback is therefore visible
immediately and propagates as a non-zero shell exit code.
