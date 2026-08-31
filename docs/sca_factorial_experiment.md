# SCA 2x2 factorial experiment

This experiment separates the class-residual architecture from its server
aggregation and changes topology independently:

| topology | residual FedAvg | class-separable aggregation |
| --- | --- | --- |
| Client-LT | `residual_fedavg_clientlt` | `online_sca` |
| fixed-marginal Dirichlet | `residual_fedavg_matched_dirichlet` | `online_sca_matched_dirichlet` |

All four cells use the same residual head, active tail rows, local gradient
mask, optimizer, LoRA configuration, client schedule, and evaluation schedule.
The residual aggregation mode is the only within-topology intervention.

`matched-dirichlet` reconstructs the Client-LT reference split with the same
split seed, copies every client total `n_k`, and resamples only the joint
client-class coupling using class-specific Dirichlet preferences. Every sample
is assigned exactly once, so every class total `n_c` is also identical.

## Recommended staged run

The existing Client-LT SCA result can stay in `online_sca`. First run only the
architecture-matched control:

```bash
STAGE=clientlt-control \
OUT_ROOT=output/online_sca_seed42_v2 \
FREEZE=output/g0_d1_seed42/lora_freeze.json \
bash scripts/run_online_sca.sh
```

Inspect `SCA - Residual-FedAvg` on Client-LT. If the aggregation gain is worth
continuing, run the two matched-topology cells. The launcher writes the full
checkpoint comparison to `clientlt_aggregation_screen.csv` and records final
deltas plus the fraction of positive checkpoints in `online_sca_summary.json`:

```bash
STAGE=matched \
OUT_ROOT=output/online_sca_seed42_v2 \
MATCHED_BETA=0.5 \
FREEZE=output/g0_d1_seed42/lora_freeze.json \
bash scripts/run_online_sca.sh
```

To explicitly run all three missing cells without the decision gate, use
`STAGE=factorial-new`. `STAGE=factorial-all` also reruns Client-LT SCA and will
refuse to overwrite a non-empty run directory.

## Analysis

When all four cells exist, the launcher automatically runs:

```bash
python scripts/analyze_sca_factorial.py \
  --output-root output/online_sca_seed42_v2
```

If the existing SCA run lives elsewhere, pass `--clientlt-sca-dir` explicitly.
The analyzer first fails closed unless:

- the two methods within each topology use exactly the same client-class map;
- Client-LT and matched Dirichlet have identical client margins `n_k`;
- Client-LT and matched Dirichlet have identical class margins `n_c`;
- their joint client-class matrices are actually different;
- all four cells contain the same evaluated rounds.

Outputs in `factorial_analysis/` are:

- `factorial_per_round.csv`: all cell values, both within-topology deltas, and DiD;
- `factorial_final.csv`: final common round;
- `factorial_best_common_round.csv`: DiD at one shared selected round;
- `factorial_best_per_cell_descriptive.csv`: independent maxima, explicitly not valid for causal DiD;
- `factorial_summary.json`: topology/protocol audits and the primary final decision fields.

The default primary metric is head-tail harmonic mean. Single-seed signs are
screening evidence only; inferential claims require fresh seeds.

## Three-GPU parallel run with automatic allocation exit

If `online_sca/` already contains the completed Client-LT SCA run, the three
missing cells can run concurrently on logical GPUs 0, 1, and 2. Parallel child
processes disable shared summary writes; the foreground shell waits for every
PID, writes the summaries once, and then exits the current interactive compute
shell even if a child failed.

Paste the following as one physical terminal line inside the rented compute
shell (do not wrap it in `bash -c`, because that would exit only the child
shell):

```bash
set -o pipefail; OUT_ROOT=output/online_sca_seed42_v2; FREEZE=output/g0_d1_seed42/lora_freeze.json; DATA_ROOT=DATA; LOG_DIR="$OUT_ROOT/parallel_logs"; mkdir -p "$LOG_DIR"; status=0; STAGE=clientlt-control GPU=0 OUT_ROOT="$OUT_ROOT" FREEZE="$FREEZE" DATA_ROOT="$DATA_ROOT" SKIP_SUMMARY=1 SKIP_FACTORIAL_ANALYSIS=1 bash scripts/run_online_sca.sh >"$LOG_DIR/clientlt_residual_fedavg.log" 2>&1 & p1=$!; STAGE=matched-control GPU=1 OUT_ROOT="$OUT_ROOT" FREEZE="$FREEZE" DATA_ROOT="$DATA_ROOT" SKIP_SUMMARY=1 SKIP_FACTORIAL_ANALYSIS=1 bash scripts/run_online_sca.sh >"$LOG_DIR/matched_residual_fedavg.log" 2>&1 & p2=$!; STAGE=matched-sca GPU=2 OUT_ROOT="$OUT_ROOT" FREEZE="$FREEZE" DATA_ROOT="$DATA_ROOT" SKIP_SUMMARY=1 SKIP_FACTORIAL_ANALYSIS=1 bash scripts/run_online_sca.sh >"$LOG_DIR/matched_sca.log" 2>&1 & p3=$!; wait "$p1" || status=$?; wait "$p2" || status=$?; wait "$p3" || status=$?; if [ "$status" -eq 0 ]; then python -u scripts/run_online_sca.py --stage summary --output-root "$OUT_ROOT" >"$LOG_DIR/summary.log" 2>&1 || status=$?; fi; if [ "$status" -eq 0 ]; then python -u scripts/analyze_sca_factorial.py --output-root "$OUT_ROOT" >"$LOG_DIR/factorial_analysis.log" 2>&1 || status=$?; fi; printf 'SCA factorial pipeline exit status: %s\nLogs: %s\n' "$status" "$LOG_DIR"; exit "$status"
```

`exit` releases only the shell in which the command is pasted. Verify that the
prompt is the allocated compute node rather than a persistent login node.
