# P0 head-damage-matched Pareto audit

P0 is an exploratory, offline gate for the CCAR direction. It reuses the
round-20/50/80 D2/D3 dumps and frozen D2b scalar/class endpoints. It does not
run local training or federated rounds.

## Protocol

- Seed: 42 only.
- Gamma grid: `0,0.1,...,1.0`.
- Logit-adjustment grid: `0,0.25,...,2.5`.
- Head-damage budgets relative to the frozen `FedAvg+LA` calibration choice:
  `0,0.5,1.0,2.0` percentage points.
- Direct matching tolerance: 0.25 head-accuracy points.
- Candidate selection: deterministic global-train calibration split inherited
  from D2b.
- Official test: accessed only after budget choices and direct matches have
  been written and hashed.

P0 evaluates exact model states. For class-conditional aggregation, every
nonzero gamma requires one state evaluation per tail class. Metric results are
cached after every gamma under `output/d23_seed42/p0/cache`; rerunning the same
command resumes completed work.

## Foreground server command

```bash
STAGE=p0 GPU=0 DATA_ROOT=DATA FREEZE=output/g0_d1_seed42/lora_freeze.json OUT_ROOT=output/d23_seed42 bash scripts/run_d23.sh
```

Do not use `nohup`: progress and errors are printed directly in the terminal.
The command requires the completed D2b artifacts under
`output/d23_seed42/d2b`.

## Primary outputs

- `p0/p0_verdict.json`: frozen decision and next action.
- `p0/p0_budget_report.csv`: scalar/class comparison for every head budget.
- `p0/p0_direct_match_report.csv`: calibration-frozen, head-matched pairs.
- `p0/p0_test_candidate_grid.csv`: full official-test gamma/tau grid.
- `p0/p0_test_pareto_frontier.csv`: nondominated scalar/class candidates.
- `p0/p0_test_pareto_auc.csv`: common-head-range Pareto envelope areas.
- `p0/p0_pareto_round_020.svg`, `050.svg`, `080.svg`: Pareto plots.

## Frozen decision

A budget passes when class-conditional aggregation, relative to scalar, has:

- H-mean gain at least 1 point;
- tail gain at least 2 points;
- balanced-accuracy gain at least -0.2 points;
- uncovered-tail drop no worse than 1 point;
- both frozen candidates still respect the head budget on test.

A round is positive when at least two budgets pass. P0 passes when at least two
of rounds 20/50/80 are positive.

Because seed-42 official-test results were inspected before P0 was designed, a
pass is exploratory. It permits V1a development but must be frozen and
confirmed on a fresh seed before becoming paper evidence.
