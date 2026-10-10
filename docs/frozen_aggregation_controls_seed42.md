# 冻结 A 的两个聚合对照：GPU2 / GPU3，最多六客户端并行

本次只新增两条 seed42、100 轮正式训练，不重新训练已完成的新聚合。启动入口是 `scripts/run_cliplora_frozen_aggregation.py`。

| 设置 | GPU2：TailRW16 | GPU3：FedAvg |
|---|---|---|
| LoRA A | 初始值全程固定 | 相同 |
| LoRA B | 每轮每客户端3个epoch | 相同 |
| 客户端参与 | 每轮全部30个 | 相同 |
| 常规本地优化步数 | 105600 | 相同 |
| 额外 A/B 更新、保持修正、来源 C、直接校准 | 全部关闭 | 相同 |
| 本地损失、LA | 原始全局计数先验，tau=1 | 相同 |
| 客户端并行 | 每张GPU最多6个 | 相同 |
| 聚合权重 | `(n_j + 16*n_j_tail)/(N + 16*N_tail)` | `n_j/N` |

这里的“16倍”沿用历史 gamma=16 公式，不是将尾类客户端整份权重乘16，也不是给本地尾类损失加权。两组固定相同数据、初始化、客户端日程、优化器和学习率；只改变客户端聚合权重。每轮全部客户端完成后，按预定顺序只聚合一次。

## 同步与启动

先同步本次代码到服务器仓库根目录。若使用代码包，将 `frozen_aggregation_controls_parallel6.zip` 放到 `/data/yzh/clientLT`，执行：

```bash
cd /data/yzh/clientLT
unzip -o frozen_aggregation_controls_parallel6.zip
conda activate clientLT
python scripts/run_cliplora_frozen_aggregation.py --stage preflight
```

`preflight` 只检查文件、注册配置和权重，不运行GPU。检查通过后：

```bash
python scripts/run_cliplora_frozen_aggregation.py --stage server --gpus 2 3 --client-concurrency 6
```

**`server` 会自动完成每组独立的GPU smoke，然后继续该组100轮正式训练。它不是单纯检查。** 两张卡各运行一条实验；没有跨GPU拆分一个客户端，也没有启动A+B组。

默认输出目录：

```text
output/cifar100_LT/frozen_aggregation_controls_v1_parallel6/
  suite_plan.json
  tailrw16/runs/seed42/frozen/
  fedavg/runs/seed42/frozen/
  launcher_logs/tailrw16_server.log
  launcher_logs/fedavg_server.log
  analysis/
```

此目录与旧 `frozen_tailrw16_v1`、`client_aggregation_v2_parallel4` 分离。不会接着旧4并行目录跑，也不会覆盖旧结果。正式训练从相同初始化开始，不使用smoke的模型状态。

如只想试跑、不进入完整实验：

```bash
python scripts/run_cliplora_frozen_aggregation.py --stage smoke --gpus 2 3 --client-concurrency 6
```

每组smoke会实际执行一轮全部30个客户端，并在提交前另做同批次串行/六并行对照：8个客户端的队列以及独立的2客户端短队列。数值不一致或任何客户端失败，该组不会进入正式训练。短队列不会补假客户端，也不会等待凑齐6个。

六并行通过独立LoRA参数、梯度、优化器和CUDA stream执行；不可变骨干参数共享存储。batch顺序在主线程预先生成，避免并发随机数改变数据顺序。记录每组试跑的速度、显存和数值误差；**最多六并行不代表六倍加速**。

参考划分默认 `references/full10_clientlt`，数据默认 `DATA`；可使用 `--reference-run` 和 `--data-root` 改路径，但首次注册后重启必须保持设置相同。

## 进度、日志与恢复

```bash
python scripts/run_cliplora_frozen_aggregation.py --stage status
```

两条新实验均显示 `complete round=100` 才表示训练完成；`incomplete` 表示尚未完成，仅凭这一状态不能区分正在运行还是进程已中断。可同时查看日志：

```bash
tail -f output/cifar100_LT/frozen_aggregation_controls_v1_parallel6/tailrw16/launcher_logs/frozen_formal.log
tail -f output/cifar100_LT/frozen_aggregation_controls_v1_parallel6/fedavg/launcher_logs/frozen_formal.log
```

上面两条 `tail -f` 分别在两个终端运行。smoke阶段查看同目录的 `frozen_smoke.log`；启动失败查看套件根目录的 `launcher_logs/*_server.log`。

中断后重新执行相同的 `server` 命令：已完成的组会通过审计后跳过，其余组从最后提交的轮次恢复。代码、并行数和训练设置须保持注册时一致。仅补跑某一组可加 `--methods fedavg` 或 `--methods tailrw16`；它们仍分别映射GPU3、GPU2。

若六并行在服务器上显存不足，先保留失败日志；可将两组一起改为 `--client-concurrency 4 --output-root output/cifar100_LT/frozen_aggregation_controls_v1_parallel4`。不能在原注册目录内悄悄降并行数。自动静默回退会让两组执行条件不明确，因此没有这样处理。

## 汇总与打包

```bash
python scripts/run_cliplora_frozen_aggregation.py --stage pack
```

生成 `output/cifar100_LT/frozen_aggregation_controls_v1_parallel6_results.tar.gz`。代码会重新核验预测、实际聚合权重、本地步数、每轮A哈希及是否误执行额外更新。打包包含结果、配置、日志、诊断预测和试跑记录，不含大模型检查点；若实验未完成，报告会明确标记。

报告在 `analysis/report.md`。主指标在启动前固定为第81—100轮Overall均值，同时报告Head20、Middle60、Tail20，不挑最佳测试轮次。另保存：

- `curves.csv`：完整逐轮结果。
- `per_class.csv`：每类最后20轮均值。
- `sample_dynamics.csv`：初始能力保持、新答对、逐轮遗忘和始终未答对。
- `stage_effects.csv`：同状态同客户端更新的即时聚合比较。
- `costs.csv`、`parallel_pilot.csv`：训练预算、耗时和并行试跑。
- `pair_audit.json`：数据/初始化/预算/执行模式/源码/环境的配对核验。

默认只读加载 `output/cifar100_LT/client_aggregation_v2_parallel4` 中已完成的 `frozen` 组。若它在其他目录，汇总或打包时指定：

```bash
python scripts/run_cliplora_frozen_aggregation.py --stage pack --compare-root /实际目录/client_aggregation_v2_parallel4
```

新补两组都为6并行，是本次相同执行配置的比较；旧新聚合是4并行，报告明确标记 `execution_differs`。GPU试跑可以验证局部数值接近，但不能证明100轮完全相同，亦不能拿旧耗时直接计算4到6并行的全程加速比。只使用seed42，不作跨种子显著性结论。

旧E2多了额外B训练，旧16倍方案训练过A，均不放入此聚合对照。若新聚合参考目录不存在，两条新实验仍可运行和比较，历史比较显示 `missing/unavailable`。

## 本地验证范围

本地为CPU环境；自动化测试覆盖FedAvg和TailRW公式、六客户端独立更新与串行对齐、不足6个的队列、GPU任务映射、smoke失败阻止正式训练、预算/冻结审计及历史结果只读汇总。配置预检也使用实际参考划分和数据文件执行。服务器GPU能否承载六并行，由上述smoke实测，不以CPU测试代替。
