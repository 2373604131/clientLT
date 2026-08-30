# D2b：Scalar Aggregation Ceiling 与 CCAR 必要性闸门

D2b 不训练新的联邦模型。它复用 `output/d23_seed42/dump_seed42` 的 round 20/50/80 compact dumps，以及 D2 已产生的 `d2_client_class_utility.csv`，回答：一个全局客户端标量权重是否足以表达 client–class utility；如果不足，class-conditional aggregation 的优势在 logit adjustment 后是否仍然存在。

## 比较对象

每轮固定比较六组：

1. `fedavg`；
2. `fedavg_la`；
3. `scalar_oracle`：30 个客户端共享一组聚合权重；
4. `scalar_oracle_la`；
5. `class_conditional_oracle`：20 个 tail 类分别拥有一组客户端权重，并为每类精确构造一个 LoRA 聚合状态，只取该状态对应类别的 logit；
6. `class_conditional_oracle_la`。

这里的 oracle 是不可部署的上界，不是候选方法。权重在 global-train fit split 上优化，在不相交的 global-train calibration split 上选择 trust coefficient `gamma`。为避免低估 scalar ceiling，scalar 的整个 gamma grid 会用精确模型前向重新选择；class-conditional gamma 使用 functional approximation 选择后，再以20个真实类别状态审计近似误差。随后使用精确模型前向选择各自的 LA `tau`。所有权重和 `tau` 写入冻结文件并计算哈希后，程序才进行新的 official-test 前向。

D2 的 test-derived utility CSV 只用于分析 utility matrix 的 interaction energy，不参与任何候选权重、gamma 或 tau 的选择。

## 前台运行

同步代码后，在已有 `output/d23_seed42` 的单卡计算节点运行：

```bash
STAGE=d2b GPU=0 OUT_ROOT=output/d23_seed42 bash scripts/run_d23.sh
```

如果环境只有 `python3`：

```bash
PYTHON_BIN="$(command -v python3)" STAGE=d2b GPU=0 OUT_ROOT=output/d23_seed42 bash scripts/run_d23.sh
```

该命令不会进入 federated training，也不会重跑80轮。它需要多次模型前向：每轮先计算30个客户端的 train-only functional responses，再精确评估1个 scalar state、20个 class states，最后在 choices 冻结后评估 test。因此运行时间可能达到数小时；终端会持续打印 `train response i/30`、优化步数和 `exact ... class state i/20`。

若离线分析中断，重新执行同一条 `STAGE=d2b` 命令即可确定性覆盖未完成的 D2b 产物。

## 主要输出

- `d2b/d2b_utility_interaction.csv`：two-way additive decomposition、interaction energy、effective rank 和 client/class sign flips；
- `d2b/d2b_gamma_selection.csv`：train-calibration 上 scalar/class oracle 的 trust-region Pareto grid；
- `d2b/d2b_tau_selection.csv`：三个精确来源分别选择 LA tau 的全过程；
- `d2b/d2b_frozen_choices.csv`：测试前冻结的权重、gamma、tau、近似误差和哈希；
- `d2b/d2b_method_metrics.csv`：六组 test 指标，额外分开报告原 D1 覆盖的11个 tail 类和未覆盖的9个 tail 类；
- `d2b/d2b_round_contrasts.csv`：class oracle 相对 scalar oracle、以及二者各自 LA 后的直接差值；
- `d2b/d2b_verdict.json`：最终必要性判定。

## 判定

- `D2B_CCAR_NECESSITY_SUPPORTED`：utility interaction 显著；class oracle 在至少两轮超过 scalar oracle；各自 LA 后 class oracle 仍有至少1pp balanced/H 增益。
- `D2B_CLASS_GAP_REMOVED_BY_LOGIT_CALIBRATION`：classwise 上界存在，但 LA 后消失；不应实现完整 CCAR。
- `D2B_INTERACTION_EXISTS_BUT_SCALAR_CEILING_NOT_BROKEN`：utility 矩阵存在交互，但没有转化为精确预测收益。
- `D2B_SCALAR_AGGREGATION_NOT_REJECTED`：现有证据不能否定单一客户端标量权重。

所有结果仍是 seed-42 discovery gate，`method_ready` 固定为 false。
