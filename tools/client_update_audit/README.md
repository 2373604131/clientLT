# E2 client-local functional update audit

E2 evaluates each client immediately after local vision-LoRA training and before
any server aggregation. It has two stages:

- E2A compares the natural Dirichlet and controlled Client-LT local update
  footprints from one shared `theta0`. It is descriptive, because natural
  client sizes and tail exposure differ.
- E2B keeps the Client-LT tail samples and every client size fixed and swaps
  only the non-tail companion samples. `narrow_related`, `broad_related`, and
  `broad_unrelated` form the causal semantic-access intervention.

The formal seed-42 compute-node command is one line:

```bash
bash scripts/run_e2_client_update_audit.sh
```

The runner is resumable by stage. If needed, execute individual stages:

```bash
python scripts/run_e2_client_update_audit.py --stage manifests
python scripts/run_e2_client_update_audit.py --stage e2a
python scripts/run_e2_client_update_audit.py --stage e2b
python scripts/run_e2_client_update_audit.py --stage summarize
```

The active Conda environment must contain `torch`, `torchvision`, `pandas`, and
`yacs`. CIFAR-100 must exist under `DATA/cifar-100/cifar-100-python`, and the
verified CLIP checkpoint must be available at
`$HOME/.cache/clip/ViT-B-16.pt`. DINOv2 is not used by E2.

Main outputs:

- `manifests/`: immutable sample, support, execution, and intervention tables;
- `e2a_local_footprint/local_all_class_footprints.csv`: the requested 100-class
  before/after local functional footprint;
- `e2a_local_footprint/local_tail_metrics.csv`: tail and semantic-neighbor
  metrics at local epochs 0--3;
- `e2b_access_intervention/`: the same client-local measurements for all three
  companion conditions;
- `e2b_access_intervention/companion_initial_difficulty.csv`: the pre-training
  `theta0` difficulty audit; imbalance beyond the frozen SMD threshold produces
  a confounded verdict instead of a causal claim;
- `analysis/e2_client_update_summary.json`: the formal gate and evidence
  boundary.

No aggregation function is called in either runtime stage. The runtime contract
and every fairness row record `server_aggregation_called=false`.
