$ErrorActionPreference = "Stop"

# PowerShell launcher for the A/B topology sweep aligned with the formal C/D setup.
# Run from the repository root after activating the clientLT conda environment:
#   .\scripts\run_ab_topology_formalCD.ps1

python experiments/analyze_clientlt_topology_sweep.py `
  --datadir DATA/cifar-100 `
  --output_dir output/topology_sweep/clientlt_control_users30_formalCD `
  --num_clients 30 `
  --tail_client_ratio 0.1 `
  --tail_class_ratio 0.2 `
  --imb_factor 0.01 `
  --imb_type exp `
  --seeds 1 42 2026 `
  --lambda_values 0 0.25 0.5 0.75 1.0 `
  --alpha_values 0.1 0.25 0.5 0.75 1.0 `
  --head_leakage_scale 3.0 `
  --fixed_alpha 0.5 `
  --fixed_lambda 0.75 `
  --num_rounds 100 `
  --participation_rate 1.0
