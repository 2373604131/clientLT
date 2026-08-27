# V0 Label-Oracle Aggregation Headroom

V0 is an offline kill-test. It asks whether labels can select a head-safe tail
improvement from the current round's uploaded LoRA updates. It does not run
CMSA clustering or use `R_m`.

## 1. Produce a no-test ClipLoRA dump

Add these flags to an otherwise ordinary full-participation ClipLoRA/FedAvg
run. The dump round must be the final round of that run.

```powershell
python federated_main.py `
  --model fedavg `
  --trainer ClipLora `
  --frac 1.0 `
  --round 80 `
  --client_schedule_file <schedule.json> `
  --cliplora_aggregation fedavg `
  --experimentD_enable False `
  --v0_dump_enable True `
  --v0_dump_rounds 20,50,80 `
  <the remaining dataset and ClipLoRA arguments>
```

This path suppresses the round-0 zero-shot test and every global test. It
writes:

```text
<output-dir>/v0_oracle/round_{020,050,080}/{round_state.pt,metadata.json}
```

## 2. Run a cheap single-unit pilot

The formal runner requires a real `dataset.val` split. `train` selection is
available only as an explicitly optimistic engineering pilot.

```powershell
python scripts/run_v0_oracle.py `
  --dump-dir <output-dir>/v0_oracle/round_050 `
  --output-dir <v0-output-dir> `
  --selection-source val `
  --gammas 0 0.2 0.4 `
  --lambda-head 0 1 4 `
  --lambda-mid 0 1 `
  --solver-iterations 2 `
  --random-count 5 `
  --convex-random-count 8
```

Once the pilot has non-zero headroom, omit the reduced grids to run the frozen
formal defaults.

If a feasible support-weighted candidate outperforms the single-start span
search, do not interpret the latter as an oracle upper bound. Run the V0b
multi-start audit described in `docs/v0b_v1_offline_analysis.md`; it reuses the
same dumps, includes support/equal/random initializations, caches repeated
evaluations, and reports support regret.

## 3. Aggregate formal units

```powershell
python scripts/summarize_v0_oracle.py `
  --input-dirs <seed42-round20> <seed42-round50> <...> `
  --output-dir <v0-summary-dir>
```

The formal PASS criterion requires at least three seeds and three rounds. The
summarizer reports every gamma separately; it does not silently choose a gamma
from official-test performance.

## Interpretation

- `oracle_span` fails: the aggregation side has no demonstrated headroom.
- `oracle_convex_search` and `oracle_span` both work similarly: simple client
  reweighting may be enough.
- `oracle_span` works but `oracle_convex_search` does not: the strongest case
  for a mode-level CMSA aggregator.
- `support_only_tail_ceiling` is a per-class diagnostic reference, not one
  deployable global model.

The current implementation uses the executable uploaded LoRA parameter state.
The manifest records this choice. If V0 passes, confirm the result in
basis-invariant effective `BA` weight space with fixed-rank recompression before
using the number as the paper's final oracle ceiling.
