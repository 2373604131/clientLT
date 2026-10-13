# 论文主表汇总与普通 Dirichlet 补跑

日期：2026-10-13。

## 方法数量与分组

外部方法共 12 个：联邦长尾及基础/知识保持对照 8 个，LoRA 因子训练控制 4 个。加上冻结的 A 和 A+B，主比较共 14 个。已回收的 Client-LT 主比较有 13 个成绩，缺少 FFA-LoRA。S、A-flat、A-current、A-calibration 为内部控制，单列消融，不计入这 14 个。

| 分组 | 方法 | 实际比较范围 |
|---|---|---|
| `longtail` | FedAvg+LoRA、FedAvg+LoRA+LA | 共同基础参照 |
| `longtail` | CAPT、FedPuReL-Global、FedLF、FedYoYo、FedAvg-LoRA+FedReLa | 联邦长尾领域方法；保留当前仓库的适配范围 |
| `longtail` | FedNTD | 通用联邦知识保持对照，并非专门的联邦长尾方法 |
| `factors` | FFA-LoRA、RoLoRA、FedSVD、LoRA-A² | 固定因子、交替更新、服务器重参数化、自适应秩选择 |
| `ours` | A、A+B | 与历史主表相同的 Full-CP λ=10、μ=1；B 为 class-cyclic、w=.35 |

`ours` 使用历史大表对应的冻结方法；它不会自动切换到后来开发的 Method-A-margin、Av2 或其他 B 路线。

## 已有 Client-LT 结果

统一为 seed42、第 81–100 轮平均 ACC（%），Head/Middle/Tail 固定为 20/60/20 类。

| 分组 | 方法 | Overall | Head20 | Middle60 | Tail20 |
|---|---|---:|---:|---:|---:|
| longtail | FedAvg+LoRA | 64.5325 | 85.6300 | 66.9625 | 36.1450 |
| longtail | FedAvg+LoRA+LA | 71.1845 | 78.4525 | 72.3708 | 60.3575 |
| longtail | CAPT | 66.8210 | 79.2675 | 61.2208 | 71.1750 |
| longtail | FedPuReL-Global | 65.7625 | 70.5475 | 64.1892 | 65.6975 |
| longtail | FedNTD | 64.1465 | 86.1150 | 65.7792 | 37.2800 |
| longtail | FedLF | 67.7295 | 77.3550 | 70.5433 | 49.6625 |
| longtail | FedYoYo | 64.4635 | 67.8950 | 67.6892 | 51.3550 |
| longtail | FedAvg-LoRA+FedReLa | 65.6160 | 83.2900 | 67.9550 | 40.9250 |
| factors | FFA-LoRA | 未回收 | 未回收 | 未回收 | 未回收 |
| factors | RoLoRA | 63.2170 | 85.6725 | 64.6675 | 36.4100 |
| factors | FedSVD | 67.2670 | 82.9175 | 63.6842 | 62.3650 |
| factors | LoRA-A² | 65.0985 | 85.6950 | 67.2892 | 37.9300 |
| ours | A | 70.6330 | 76.1050 | 68.7083 | 70.9350 |
| ours | A+B | 70.6965 | 75.9550 | 68.7900 | 71.1575 |

来源：[原始合并表](../output/longtail_benchmarks_review_20261009/combined_performance.csv)。本次按分组整理为 [主表 CSV](../output/benchmark_tables_20261013/main_table_clientlt.csv)，内部四项控制在 [独立 CSV](../output/benchmark_tables_20261013/internal_controls_clientlt.csv)。精确数值与来源哈希保存在同目录 `provenance.json`。

CAPT 是固定聚合频率版本，FedPuReL 只有共享阶段；FedLF/FedYoYo/FedReLa 是 CLIP-LoRA 适配。上述结果不代表原论文原生配置的复现。A+B 相比 LA 的尾类提高 10.8000 个百分点，Overall 降低 .4880；当前证据支持尾类保持与头中类学习的取舍。

## 是否全部补跑 Dirichlet

正文出现一个方法，不会自动产生“必须再跑 Dirichlet”的要求。若只主张 Client-LT 下有效，可以限定结论；若希望说明主表结论不依赖特定客户端长尾构造，应在同一普通 Dirichlet 设置下比较完整主表。这次按后一目标补齐 14 个方法，内部消融不自动扩展。

CAPT 原论文以 Dirichlet α∈{.05,.1,.5} 评价不同异质性，见[实验设置与附录 B](https://arxiv.org/html/2503.06993v1#A2)。本项目补跑 α=.5 是我们的实验设计选择，不是会议的硬性规定，也不等于复现 CAPT 原协议。

当前脚本仅固定 seed42。完整主表补跑是第一步；跨种子的误差条、稳定性及更多 α 是另一项实验，不能由一个种子替代。

## 补跑协议

- `noniid-labeldir-fine`，对每个细类抽对称 Dirichlet，再以 multinomial 分配样本；默认 beta=.5（即论文常记的 α），split_seed=42。
- 原 Client-LT 的 10,847 张训练图像与全局各类样本数完全保留。分配前按 `(class_id, raw_sample_id)` 排序，顺序写入协议。不能直接复用旧普通 Dirichlet 结果，除非逐图像分配也一致。
- 每客户端至少 10 张；不固定 Client-LT 客户端容量。默认实际容量 229–500，合计每个本地 epoch 354 步。多数外部方法 100×3×354=106,200 步；LoRA-A² 另计筛选 epoch；A/A+B 另计原有额外阶段。
- 保留 CLIP ViT-B/16、rank4、视觉 top3 q/v、FP32、100 轮全参与、原方法损失/专用增强/超参数和固定测试集。
- 公共日程、probe、初始化与冻结骨干指纹来自 `references/full10_clientlt`。所有方法读取同一份新 manifest，实际模型仍严格校验指纹。
- A/A+B 从初始化开始，使用新的 manifest；不导入旧 Client-LT 已训练结果。

新入口：[scripts/run_dirichlet_benchmarks.py](../scripts/run_dirichlet_benchmarks.py)。默认输出：

```text
output/cifar100_LT/dirichlet_benchmarks_seed42_beta0.5_v1/
```

其中 `reference/` 为只生成一次、重复执行时严格核对的划分；`jobs/` 登记任务；`runs/` 保存外部方法；`ab_runs/` 保存 A/A+B；`smoke/` 保存独立短程检查；`analysis/` 为汇总。

## 启动

同步新增脚本、`tools/benchmarks/` 与修改的 `scripts/run_paper_benchmarks.py`。其余模型、数据与官方基线源码沿用此前成功环境；保留完整 `references/full10_clientlt`。在服务器仓库根目录执行。

生成划分、登记 14 个任务并打印命令，不训练：

```bash
python -u scripts/run_dirichlet_benchmarks.py --stage plan
```

依赖、数据、GPU、源码与协议预检：

```bash
python -u scripts/run_dirichlet_benchmarks.py --stage preflight
```

短程检查 12 个外部方法（每个两轮、两客户端、一个 batch，完整测试集评估）：

```bash
CUDA_VISIBLE_DEVICES=0 python -u scripts/run_dirichlet_benchmarks.py --stage smoke
```

A/A+B 沿用冻结的正式入口，未把它们强行改成短程版本。短程输出不进入主表。

一张卡依次训练全部 14 个方法：

```bash
CUDA_VISIBLE_DEVICES=0 python -u scripts/run_dirichlet_benchmarks.py --stage train
```

按用户要求的两类分别运行，A/A+B 单独启动：

```bash
CUDA_VISIBLE_DEVICES=0 python -u scripts/run_dirichlet_benchmarks.py --stage train --group longtail
CUDA_VISIBLE_DEVICES=1 python -u scripts/run_dirichlet_benchmarks.py --stage train --group factors
CUDA_VISIBLE_DEVICES=2 python -u scripts/run_dirichlet_benchmarks.py --stage train --group ours
```

三行分别在三个终端执行；每个组内前台串行。若已有调度器分配 GPU，省略 `CUDA_VISIBLE_DEVICES=...`，直接继承分配结果。

也可将完整 14 项分给四张卡，分别在四个终端执行（与上面的按组启动二选一，不重复启动同一方法）：

```bash
CUDA_VISIBLE_DEVICES=0 python -u scripts/run_dirichlet_benchmarks.py --stage train --num-shards 4 --shard-index 0
CUDA_VISIBLE_DEVICES=1 python -u scripts/run_dirichlet_benchmarks.py --stage train --num-shards 4 --shard-index 1
CUDA_VISIBLE_DEVICES=2 python -u scripts/run_dirichlet_benchmarks.py --stage train --num-shards 4 --shard-index 2
CUDA_VISIBLE_DEVICES=3 python -u scripts/run_dirichlet_benchmarks.py --stage train --num-shards 4 --shard-index 3
```

四份实际任务分别为：

| 分片 | 方法 |
|---|---|
| 0 | FedAvg+LoRA、FedNTD、FFA-LoRA、A |
| 1 | FedAvg+LoRA+LA、FedLF、RoLoRA、A+B |
| 2 | CAPT、FedYoYo、FedSVD |
| 3 | FedPuReL-Global、FedReLa、LoRA-A² |

中断后在对应原命令末尾追加 `--resume`。已完成且通过核验的任务自动跳过；源码、配置或划分改变时拒绝混用旧任务。可用 `--methods fedlf fedyoyo` 显式选择子集，覆盖 `--group`。

如改变 α，例如 `--dirichlet-beta 0.1`，默认输出自动变为 `...beta0.1_v1`。如果手动指定 `--output-root`，不同 α 必须使用不同目录。训练前不要覆盖正在运行的源码；新代码会改变任务指纹。

## 汇总和打包

```bash
python -u scripts/run_dirichlet_benchmarks.py --stage status
python -u scripts/run_dirichlet_benchmarks.py --stage summary
python -u scripts/run_dirichlet_benchmarks.py --stage pack
```

`analysis/performance.csv` 与 `analysis/report.md` 给出主表；`status.csv` 保留未开始、不完整和错误状态。只有完整核验的 100 轮结果进入主表；缺少结果不会填 0。分析包 `...beta0.5_v1_analysis.tar.gz` 包含逐轮、逐类和协议，不含模型权重。

短程状态需要显式指向其目录：

```bash
python -u scripts/run_dirichlet_benchmarks.py --stage summary --output-root output/cifar100_LT/dirichlet_benchmarks_seed42_beta0.5_v1/smoke
```

## 验证

新增 `tests/test_dirichlet_benchmarks.py` 覆盖真实样本池保持、实际重新分配、确定性重建、协议篡改拒绝、14 方法共同登记、四分片不重不漏、A/A+B 新划分参数/路径、旧 Client-LT 导入拒绝及短程与正式表隔离。

在本机 `D:/Anaconda/envs/clientLT/python.exe` 环境中，76 项组合回归检查全部通过、无跳过；汇总分组调整后，相关 31 项再次通过。模型启动检查包含随机小型 CLIP 的实际前向、反向与更新，不能替代完整预训练模型的服务器验收。

GPU 依赖预检通过不代表已完成训练。完整性能结论须等正式任务完成并汇总；脚本不会在默认 `plan` 阶段隐式启动训练。

本机 14 项依赖/数据/GPU 预检已通过。实际启动 FedAvg-LoRA 的 CUDA 短程检查后，数据与模型构建成功，但在初始化指纹核对处停止：`Initial LoRA factors differ from reference seed42`。这是此前 10 月 8 日本机检查也记录过的本地与服务器初始化差异。本次保留原参考与严格检查，没有替换参考哈希。尚未生成新的 Dirichlet 性能结果；正式补跑应在原大表成功运行的服务器环境先做 smoke，再启动完整训练。
