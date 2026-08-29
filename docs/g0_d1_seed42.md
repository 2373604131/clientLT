# G0 → D1（CIFAR-100 ClientLT，seed 42）

这套脚本只用于确定现象和冻结实验载体，不是多 seed 的最终论文结果。

## 实验固定项

- 数据：仓库现有 `cifar100_LT` 与 `client-longtail` 划分。
- 客户端：30 个，其中 27 个 head client、3 个 tail specialist。
- 类别：按全局训练计数确定 80 个 head 类、20 个 tail 类；没有 mid 组。
- 发现阶段只运行 seed/split seed 42。
- 本地训练：3 epoch，vision-only ClipLoRA，FP32，客户端优化器隔离。

G0 只比较以下两套配置：

| 配置 | 视觉层 | rank | alpha | 目标矩阵 |
| --- | --- | ---: | ---: | --- |
| `old_r2` | top 3 blocks | 2 | 1 | q, v |
| `candidate_r4` | top 4 blocks | 4 | 2 | q, k, v |

每套配置都从同一个 incoming global 分别训练 3 个 tail specialist 和按样本量匹配的 3 个 head client。G0 不做服务器聚合。通过闸门后，启动器把唯一配置写入 `lora_freeze.json`；如果两套都通过，则选 tail specialist 的 held-out tail margin 中位增益更高者，同分选择较小的 `old_r2`。如果两套都失败，D1 会被强制阻止。

D1 只运行一次 seed 42、80 轮、全客户端参与的真实 FedAvg。第 20/50/80 轮从同一批本地更新构造以下离线反事实，不额外训练模型：

- `FedAvg`：所有 30 个客户端按样本量聚合。
- `support_actual`：只保留该 tail 类有效 supporter 的原始 FedAvg 权重质量。
- `non_support_actual`：只保留 non-supporter，用于检查其是否造成反向影响。
- `support_normalized`：只在有效 supporter 间重新归一化，是非部署式上界。
- matched random support：与真实 supporter 数量相同的随机客户端集合，20 次，报告 p95。

有效 supporter 完全复用 CAPT 的严格条件：该类样本占客户端本地训练集的比例 `> 0.1`。

## 前台运行

先进入已经激活 `clientlt` 环境的三卡计算节点。当前流程只使用一张 GPU，因为 G0 的两个配置必须依次完成并冻结，D1 随后只允许一条 80 轮轨迹：

```bash
GPU=0 DATA_ROOT=DATA OUT_ROOT=output/g0_d1_seed42 bash scripts/run_g0_d1.sh
```

命令不会使用 `nohup`，所有训练、报错和最终判定都会直接显示在当前终端。若环境只有 `python3`：

```bash
PYTHON_BIN="$(command -v python3)" GPU=0 DATA_ROOT=DATA OUT_ROOT=output/g0_d1_seed42 bash scripts/run_g0_d1.sh
```

也可以分阶段运行：

```bash
STAGE=g0 GPU=0 OUT_ROOT=output/g0_d1_seed42 bash scripts/run_g0_d1.sh
STAGE=d1 GPU=0 OUT_ROOT=output/g0_d1_seed42 bash scripts/run_g0_d1.sh
```

第二条命令没有通过的 `lora_freeze.json` 时会直接拒绝运行。中断后只跳过已经完整生成结果的阶段；如果 80 轮已经结束但最终 JSON 尚未生成，它也会直接从 CSV 恢复汇总，不会重新训练：

```bash
SKIP_COMPLETED=1 GPU=0 OUT_ROOT=output/g0_d1_seed42 bash scripts/run_g0_d1.sh
```

## 输出与判读

- `g0/*/g0_probe/g0_per_client.csv`：12 次本地能力检查中的逐客户端指标。
- `g0/*/g0_probe/g0_config_summary.json`：每套 LoRA 的能力汇总。
- `lora_freeze.json`：冻结选择及失败理由；这是 D1 的硬依赖。
- `d1_seed42/round_metrics.csv`：同一条 80 轮 FedAvg 轨迹。
- `d1_seed42/experiment_d/experiment_d_per_class.csv`：20/50/80 轮逐 tail 类反事实。
- `d1_seed42/experiment_d/experiment_d_round_summary.csv`：每个诊断轮的汇总。
- `d1_summary/d1_verdict.json`：三轮现象筛查判定。

D1 的筛查通过表示“tail supporter 的更新中存在 FedAvg 稀释掉的可恢复信号，并且不是任意同规模客户端集合都能实现”；它不表示新方法已经达到 SOTA。若失败，应先根据 `non_support_actual`、随机 p95 和 head damage 区分“没有信号”“support 定义无辨识度”和“有 tail 收益但代价过大”，不要直接添加 trick。
