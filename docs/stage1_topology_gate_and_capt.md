# Stage 1: topology-gate diagnosis and exact CAPT contrast

Stage 1 is diagnostic only. It does not change SCA or introduce a gate. It
answers two separate questions:

1. Do the existing four factorial trajectories support a topology-only gate?
2. Is CAPT's Client-LT advantage primarily a matched-topology base gap or a
   topology-robustness gap?

All results in this stage are seed-42 directional evidence. Rounds and classes
are not seed replicates.

## 1A: artifact-only topology diagnosis

Run on the existing four cells:

```bash
python -u scripts/analyze_stage1_topology_gate.py --output-root output/online_sca_seed42_v2
```

The analyzer fails closed unless the residual-FedAvg and SCA cells within each
topology have identical client-class matrices and executed-client schedules,
and unless Client-LT and matched Dirichlet have exact common `n_k`, exact
common `n_c`, and one common executed-client schedule.

For class `c`, the effective carrier count is

```text
N_eff(c) = n_c^2 / sum_k N_kc^2.
```

The excess concentration score is

```text
rho_c = max(0, (E_null[N_eff(c)] - N_eff(c)) / E_null[N_eff(c)]).
```

The null uniformly permutes class labels into the exact observed client
capacities. It therefore conditions on both `n_k` and `n_c`; it does not use
validation or test performance.

Primary artifacts in `stage1a_topology_gate/`:

- `stage1a_class_round.csv`: class x round x topology table, including
  accuracy, SCA-minus-control gain, supporter count and mass, effective
  carriers, top-1 mass, rho, absence, and absence streak;
- `stage1a_class_stage.csv`: early (1--20), middle (21--50), and late (51--80)
  class-level summaries;
- `stage1a_paired_class_stage.csv`: within-class topology contrasts, including
  delta-rho and the aggregation-gain difference-in-differences;
- `stage1a_spearman.csv`: all/head/tail class-level correlations and the paired
  topology contrast;
- `stage1a_rho_quartiles.csv` and `stage1a_gain_contribution.csv`: tail rho
  quartiles and their shares of positive/negative aggregation gain;
- `stage1a_absence_analysis.csv` and `stage1a_decay_association.csv`: tests of
  whether late decline is localized to long-absence classes;
- `stage1a_summary.json` and `stage1a_report.md`: compact directional verdict.

The four historical runs do not contain each supporter's pre-aggregation
residual update vector. `update_agreement` is therefore explicitly unavailable;
the analyzer does not use an aggregated row delta as a false proxy.

## 1B: communication-matched CAPT on both exact topologies

CAPT uses the same seed, split seed, 30 clients, 0.4 participation, 80 rounds,
3 local epochs, and the same schedule file as the four SCA cells. Before any
training starts, the launcher compares that file with the clients actually
logged by both existing SCA runs. After training, the analyzer checks CAPT's
actually executed clients, exact client-class matrices, both margins, and all
80 metric rows.

The repository's default CAPT branch uses official-test metrics to update an
MAB-controlled aggregation schedule and may skip aggregation at the final
round. That is unsuitable for a causal diagnostic. The explicit
`--capt_fixed_global_agg_freq 1` flag runs a communication-matched CAPT audit:
CAPT's clustering and class-aware aggregation remain active, global aggregation
occurs every round, and official-test metrics do not control future training.
The default CAPT behavior is unchanged when the flag is zero.

Run each cell separately with:

```bash
python -u scripts/run_stage1_capt_dual_topology.py --stage clientlt --gpu 0 --skip-analysis
python -u scripts/run_stage1_capt_dual_topology.py --stage matched --gpu 1 --skip-analysis
```

Then analyze:

```bash
python -u scripts/analyze_stage1_capt_gap.py --sca-output-root output/online_sca_seed42_v2 --capt-output-root output/online_sca_seed42_v2/stage1b_capt
```

For every one of overall, head, tail, H-mean, and macro-F1, and for both static
SCA and Residual-FedAvg comparators, the analyzer reports:

```text
BaseGap = CAPT_matched - ours_matched
TopologyRobustnessGap = (ours_matched - ours_ClientLT)
                      - (CAPT_matched - CAPT_ClientLT)
CAPT_ClientLT - ours_ClientLT = BaseGap + TopologyRobustnessGap
```

The decomposition is computed at the final round and at one best common round.
Independent per-cell maxima are descriptive only and are never substituted
into the decomposition.

## Two-GPU unattended run and automatic allocation exit

Paste the following as one physical line inside the rented compute shell. Do
not wrap it in `bash -c`: the final `exit` must execute in the allocation shell,
not in a disposable child shell.

```bash
set -o pipefail; OUT_ROOT=output/online_sca_seed42_v2; LOG_DIR="$OUT_ROOT/stage1_logs"; mkdir -p "$LOG_DIR"; status=0; python -u scripts/analyze_stage1_topology_gate.py --output-root "$OUT_ROOT" >"$LOG_DIR/stage1a.log" 2>&1 || status=$?; if [ "$status" -eq 0 ]; then python -u scripts/run_stage1_capt_dual_topology.py --stage clientlt --gpu 0 --skip-analysis --skip-finished --sca-output-root "$OUT_ROOT" --output-root "$OUT_ROOT/stage1b_capt" >"$LOG_DIR/capt_clientlt.log" 2>&1 & p1=$!; python -u scripts/run_stage1_capt_dual_topology.py --stage matched --gpu 1 --skip-analysis --skip-finished --sca-output-root "$OUT_ROOT" --output-root "$OUT_ROOT/stage1b_capt" >"$LOG_DIR/capt_matched.log" 2>&1 & p2=$!; wait "$p1" || status=$?; wait "$p2" || status=$?; fi; if [ "$status" -eq 0 ]; then python -u scripts/analyze_stage1_capt_gap.py --sca-output-root "$OUT_ROOT" --capt-output-root "$OUT_ROOT/stage1b_capt" >"$LOG_DIR/stage1b_analysis.log" 2>&1 || status=$?; fi; printf 'Stage-1 pipeline exit status: %s\nLogs: %s\n' "$status" "$LOG_DIR"; exit "$status"
```

The command needs two GPUs. If either child fails, analysis is skipped, the
nonzero status is printed, and the allocation shell still exits. `exit` should
only be used after confirming the prompt is the rented compute node, not the
persistent login node.

