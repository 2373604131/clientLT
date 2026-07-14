# TCRM Prerequisite Experiments

This repo provides a single runner for the two prerequisite diagnostics:

```bash
python scripts/run_tcrm_prereq_experiments.py --experiment all --seeds 1 --rounds 20 --frac 0.4
```

The runner forwards TCRM stability defaults:

- `--clip_precision fp32`: mirrors the stable CAPT/PromptFL path where CLIP is
  converted to fp32 for ordinary training.
- `--prompt_grad_clip 1.0` and `--rho_grad_clip 1.0`: prevent one tiny local
  client update from dominating a round.
- `candidate_skip_count_k` is written to `tcrm_tail_diagnostics.csv` when a
  residual candidate is skipped because of non-finite logits/loss/gradients.

## 0c: Failure Mechanism Trajectories

Runs `decoupled_residual_fedavg` on both `noniid-labeldir` and
`client-longtail`, then plots tail accuracy, residual norm, and tail error
paths.

```bash
python scripts/run_tcrm_prereq_experiments.py --experiment 0c --seeds 1,42,3407 --rounds 20 --frac 0.4
```

Main outputs:

- `output/tcrm_prereq/0c/figures/exp0c_tail_rho_error_trajectories.png`
- `output/tcrm_prereq/0c/figures/exp0c_trajectory_points.csv`
- `output/tcrm_prereq/0c/figures/exp0c_summary.csv`

Use `--rerun true` to overwrite existing runs. By default, completed runs are
skipped and only aggregation/plotting is refreshed.

## 1a: Concentration Sweep and Federated Residual Gain

Runs the allocation concentration sweep over lambda values, then trains
`decoupled_residual_fedavg` on each controlled allocation. The plotted gain is
the final per-tail-class federated residual gain:

```text
G_k^fed = tail_accuracy_k - zero_shot_tail_accuracy_k
```

```bash
python scripts/run_tcrm_prereq_experiments.py --experiment 1a --concentration_levels 0.0,0.25,0.5,0.75,1.0
```

Main outputs:

- `output/tcrm_prereq/1a/figures/exp1a_concentration_vs_fed_gain.png`
- `output/tcrm_prereq/1a/figures/exp1a_concentration_fed_gain_points.csv`
- `output/tcrm_prereq/1a/figures/exp1a_summary.csv`

The controlled allocation is passed into `tcrm_main.py` with
`--controlled_allocation_csv`, so the allocation actually determines the client
training split. This is intentionally not the pooled oracle reporter, because a
pooled residual ignores client concentration when `M_k` is fixed.
