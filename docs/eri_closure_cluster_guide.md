# ERI 闭环实验：超算启动说明

这套实现把实验拆成互不混杂的三阶段：Phase 0 为已有 round dump 的离线闭环；Phase 1 是固定 `n_k`、固定 `n_c` 的 Client-LT / matched-Dirichlet FedAvg 比较；Phase 2 是相同固定边际下的 support-normalized 干预。所有梯度、路径积分与重放均只读取冻结的 CIFAR-100 训练集 probe；官方测试集只在训练结束后的普通 global test 中用于计算 BFD（best-to-final drop）。

## 先在登录节点准备一次

假设代码和数据都在共享文件系统，并且 CLIP ViT-B/16 权重已经缓存到每个计算节点可见的位置。把下面三个变量替换为集群实际路径：

```bash
export REPO_DIR=/shared/$USER/CAPT
export DATA_ROOT=/shared/datasets/DATA
export ERI_OUTPUT_ROOT=/shared/$USER/experiments/eri_closure_v1
cd "$REPO_DIR"
```

先用与你训练时一致的 Python/conda 环境验证导入：

```bash
python -m py_compile federated_main.py tools/eri_closure/*.py scripts/run_eri_closure.py
```

随后只运行一次 protocol；它生成全实验共同的 train-only probe manifest，以及每个 seed 的全参与 client schedule。**不要让训练数组自己第一次创建它们。**

```bash
python -u scripts/run_eri_closure.py \
  --stage protocol --output-root "$ERI_OUTPUT_ROOT" --data-root "$DATA_ROOT" \
  --seeds 1 2 3 42 2026
```

## Phase 0：已有 paired dump 的离线闭环

将现有 Client-LT 和 matched Dirichlet 的 `round_010` 目录映射到集群共享存储后运行：

```bash
python -u scripts/run_eri_phase0.py \
  --clientlt-dump /shared/.../client-longtail_seed42_round10/cusp_minimal/round_010 \
  --dirichlet-dump /shared/.../dirichlet_beta0.5_seed42_round10/cusp_minimal/round_010 \
  --output-root "$ERI_OUTPUT_ROOT/phase0" --data-root "$DATA_ROOT" --device cuda
```

结果包含 `analysis/round_signed_budgets.csv`、路径积分 completeness check，以及 `replay/frozen_replay_scores.csv`（FedAvg、support-normalized 和 100 个权重置换）。它不重训，也不把结果用于后续训练决策。

## Phase 1：固定边际的主效应（10 个训练任务）

先提交两个 FedAvg case × 五个 seed。`eri_closure_slurm_array.sbatch` 默认申请 1 GPU、8 CPU、48 GB、24 小时；请按实际卡型和本地队列策略修改 SBATCH 行以及 module/conda 初始化部分。

```bash
export ERI_STAGE=train
export ERI_CASES=clientlt_fedavg,matched_dirichlet_fedavg
export ERI_SEEDS=1,2,3,42,2026
sbatch --array=0-9 scripts/eri_closure_slurm_array.sbatch
```

全部完成后，先做固定边际验证。它会逐 seed 比较两个 topology 的每个 client 总量 `n_k` 和全局 class 总量 `n_c`；它**不会**错误地要求整个 client×class 矩阵相同，因为那正是 topology 操作量。

```bash
python -u scripts/run_eri_closure.py \
  --stage verify --output-root "$ERI_OUTPUT_ROOT" --data-root "$DATA_ROOT" \
  --cases clientlt_fedavg matched_dirichlet_fedavg --seeds 1 2 3 42 2026
```

再以相同数组设置运行路径积分归因；完成后对两个 FedAvg case 重放：

```bash
export ERI_STAGE=analyze
sbatch --array=0-9 scripts/eri_closure_slurm_array.sbatch

export ERI_STAGE=replay
sbatch --array=0-9 scripts/eri_closure_slurm_array.sbatch
```

`analyze` 与 `replay` 必须等待相应训练任务成功结束。建议在集群上使用 `--dependency=afterok:<jobid>`，例如 `sbatch --dependency=afterok:12345 --array=0-9 ...`。

## Phase 2：class-aware intervention（10 个训练任务）

Phase 1 的结果不改变 Phase 2 的预注册配置；仅当资源安排允许时启动 support-normalized 两个 case：

```bash
export ERI_STAGE=train
export ERI_CASES=clientlt_support_normalized,matched_dirichlet_support_normalized
sbatch --array=0-9 scripts/eri_closure_slurm_array.sbatch
```

完成后，对同一 pair 重复 `verify` 和 `analyze`。Phase 2 的主干预判据为在 Client-LT 内，support-normalized 相比 FedAvg 是否同时降低 CERI/ERI 并提高 tail retention；matched Dirichlet 是同样干预下的 topology control。

## 汇总与关键输出

所有 20 个任务的分析完成后运行：

```bash
python -u scripts/run_eri_closure.py \
  --stage summary --output-root "$ERI_OUTPUT_ROOT" --data-root "$DATA_ROOT"
```

主要文件位于 `$ERI_OUTPUT_ROOT/eri_closure_summary/`：

- `per_class_ceri_bfd.csv`：每个 tail class 的加权 `W,D,R,CERI` 与 best-to-final drop。
- `per_run_ceri_bfd_correlation.csv`：每个 run 的 Spearman(CERI, BFD)。
- `paired_topology_CERI.csv`：严格 seed、aggregation 配对的 Client-LT − matched-Dirichlet CERI 差。
- `paired_intervention_effects.csv`：同一 topology/seed 下 support-normalized − FedAvg 的 CERI 与 tail-retention 差。
- `eri_closure_summary.json`：可直接用于论文表格的总览。

每个 run 下的 `eri_closure/dumps/` 保留原始 local states 与服务器权重；`attribution_validity.csv` 记录路径积分与端点函数变化的 completeness 误差。若这个误差异常大，应增加 `--quadrature-points` 后重新做**离线归因**，无需重训。
