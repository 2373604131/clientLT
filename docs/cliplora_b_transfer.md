# 方法 B：周期性 Donor 引导的低秩更新迁移

## 1. 实现位置

入口：`scripts/run_cliplora_b_transfer.py`。

核心实现：`utils/cliplora_b_transfer.py`。

每轮的顺序是：

```text
共同 A、B → 全部客户端正常训练 B → 缓存原始本地 B 和 ΔB
→ 补充轮：逐接收端筛选 donor → 只优化小矩阵 C → 服务器重建补充
→ 原样本量权重聚合 B → 原方法 A 的更新与修正 → 评估并保存
```

在 `LAControlRuntime.train_phase` 的所有客户端训练结束、B 聚合开始之前接入。原入口未开启 B 迁移时，该接入点不做任何操作。

同一轮全部 donor 的更新均为 `ΔB_j = B_j_loc - B_start`。即使某个 donor 自己也是接收端，后续接收端使用的仍然是它未经迁移的原始更新。

默认搭配最近实现的 **A：full-cp，保持权重 λ=10、分类保持权重 μ=1**。μ=1 是这一轮固定的起始配置，不代表已经选出的最优值；不要在 B 学习率搜索的同时改变 A。也可显式传入 `--method full`，搭配原版 A。

## 2. 固定协议

| 项目 | 实现设置 |
|---|---|
| 数据与训练 | CIFAR100-LT；30 个客户端全部参与；100 轮；日常 B 训练 3 个本地 epoch |
| LoRA | vision，ViT-B/16，最后三个 block 的 q/v，共 6 个模块；rank=4，alpha=1，scaling=0.5 |
| A 时间表 | 第 1–90 轮执行原有 A 更新；第 91–100 轮不更新 A |
| B 迁移时间表 | 第 30、40、50、60、70、80、90、100 轮，共 8 次 |
| 接收端 | 持有任一全局 bottom20 尾类的客户端，包括因少量样本泄漏而持有尾类的客户端；不是仅选三个主要尾部客户端 |
| 候选来源 | 本轮其他客户端中，至少缺少接收端一个目标尾类的客户端 |
| 探测样本 | 每个本地尾类固定 min(8, 本地该类样本数) 张训练图片；与 A 的见证索引一致，但 B 仅用单一确定性视图 |
| 探测注入 | 原本地 B + 0.1 × 原始 donor ΔB；不同 donor 不累加 |
| 筛选 | 在 donor 缺失的尾类上，真实前向 LA 损失下降 > 0；保留全部合格 donor 的并集，无 Top-K |
| C | 每个接收端—donor—LoRA 模块一个 r×r 全矩阵，每次补充零初始化 |
| 优化器 | 独立 Adam；默认 lr=0.1，betas=(0.9,0.999)，eps=1e-8，weight_decay=0 |
| 优化次数 | 有 donor 的接收端执行恰好两个 mini-batch 更新步骤 |
| 平方正则 | 系数 0.001；各矩阵平方和除以 donor 数与模块数的乘积 |
| LA | 沿用全局训练先验、τ=1、CLIP logit scale 和全部 100 类输出，不重新计算小样本先验 |
| 聚合 | 原样本量加权 FedAvg，不修改客户端权重 |

C 的形状直接读取对应 B 张量的第二维。当前每个 donor 有六个 4×4 矩阵，共 96 个可训练标量。迁移模块本身没有把 4 写死；但本轮入口及原 A 实验协议仍固定 rank=4，未增加 rank 搜索。

### 两个校准 batch

存在非尾类时，每步最多抽取 16 张尾类图片和 16 张非尾类图片；只有尾类时，每步最多 32 张尾类图片。

- 尾类按打乱后的类别顺序循环取样，两个 batch 连续规划类别覆盖。样本充分时逐类均匀填充，不让某个大尾类占满 batch。
- 非尾类从本地非尾类图片中均匀抽样。
- 单个 batch 内不重复；样本少时不补重复图片，允许第二步复用第一步的图片。
- 两类都存在时，分类损失为 `0.5 × mean(LA_tail) + 0.5 × mean(LA_non_tail)`；只有尾类时使用尾类均值。
- 抽样只由 seed、轮次、接收端 ID 确定，不受 donor 数和 C 学习率影响。
- 使用确定性评估预处理，关闭 dropout，但 C 训练时启用梯度。

探测与校准均取自本地训练集，允许重叠；不是独立验证集，不能将这些损失变化当成测试集收益。

## 3. 小矩阵训练与并回 B

每层计算：

```text
R_k = mean_j(ΔB_j @ C_kj)
B_eff = B_k_loc + R_k
loss = LA_calibration + 0.001 × sum_j,l ||C_kj_l||² / (donor_count × module_count)
```

只把 C 交给优化器。A、本地 B、原始 donor ΔB、主干及文本特征保持固定。不同 donor 的补充先合成为 R，再执行一次模型前向，不建立多个教师网络。

实现采用临时可求导的层输出补充 `s × R × A × x`，等价于用 `B_eff` 前向，避免通过 `copy_` 替换 B 时切断 C 的梯度。第二步完成后移除临时补充分支。

接收端返回 donor ID 和 C；服务器用缓存的原始 ΔB 重建 R，得到 `B_k_new = B_k_loc + R_k`，再执行原 FedAvg。没有 donor 的接收端不执行 C 优化。本轮 C、donor 集合、优化器状态不进入下一轮训练；保存的 C 文件仅用于离线分析。推理 rank、B 形状及分支数均不变。

这是现有仓库中的串行联邦模拟，不是新增真实分布式网络服务。协议允许类别存在标记可见，额外通信量按候选 ΔB 下发、类别存在表及 C/ID 回传计算。训练侧诊断留在实验日志中，不作为聚合器的额外输入，也不计为方法协议上传；没有声称差分隐私或安全聚合。

## 4. 启动：三个节点各跑一组

在仓库根目录、已激活原训练环境下执行。每个节点分配一张 GPU，均以前台形式运行，无 `nohup`、无后台重定向。

用已有 Full-10 的完整结果目录复用划分、客户端顺序和计划；不是加载其训练后的权重。以下路径按原实验默认目录给出，若移动过目录，替换 `--reference-run` 后的路径即可。

### 节点一：C lr=0.03

```bash
python -u scripts/run_cliplora_b_transfer.py --transfer-lr 0.03 --reference-run output/cifar100_LT/sfra_v1/seed42/client-longtail/full/lambda10_protocol42
```

### 节点二：C lr=0.1

```bash
python -u scripts/run_cliplora_b_transfer.py --transfer-lr 0.1 --reference-run output/cifar100_LT/sfra_v1/seed42/client-longtail/full/lambda10_protocol42
```

### 节点三：C lr=0.3

```bash
python -u scripts/run_cliplora_b_transfer.py --transfer-lr 0.3 --reference-run output/cifar100_LT/sfra_v1/seed42/client-longtail/full/lambda10_protocol42
```

这三组都默认使用 `--method full-cp --retention-weight 10 --classification-weight 1`，只改变 C 学习率；分别执行完整的 100 轮，不是从第 100 轮接着训练。

对应 lr=0.1 的输出目录为：

```text
output/cifar100_LT/sfra_b_transfer/seed42/client-longtail/full-cp/lambda10_mu1_protocol42_b_lr0.1_probe0.1_reg0.001/
```

不会覆盖原 `sfra_v1` 或 `sfra_cp` 结果。

若选择搭配原版 A，在三组命令中一致追加 `--method full`；不要将不同 A 版本混在一组 B 学习率比较中。Dirichlet 使用 `--partition noniid-labeldir-fine`，并提供同划分的 `--reference-run`，不能复用 Client-LT 的划分文件。

### 断点续训

例如恢复默认 A 配置、C lr=0.1 的任务：

```bash
python -u scripts/run_cliplora_b_transfer.py --transfer-lr 0.1 --resume
```

从最近完成轮次的 `checkpoints/sfra_last.pt` 恢复，并重放原命令配置。若中断发生于迁移中间，则重做该轮，不接着使用半训练的 C。使用非默认 A 参数或输出根目录时，恢复命令需给出相同参数以定位原目录。

## 5. 结果记录和比较

保留既有 `round_metrics.csv`、逐类结果、A 的来源与修正日志。新增：

| 文件 | 内容 |
|---|---|
| `b_transfer_config.json` | B 协议和每个模块的实际 rank |
| `b_transfer_manifest.json` | 目标尾类、接收端、类别存在标记、固定探测及监测样本 |
| `b_transfer_rounds.csv` | 每次迁移的汇总、额外计算/通信量 |
| `b_transfer_rounds/rXXX/donor_scores.csv` | 缺类候选在各尾类上的探测前后 LA、gain 和是否为正 |
| `b_transfer_rounds/rXXX/calibration_manifest.json` | 最终 donor、两步校准图片位置、模块顺序 |
| `b_transfer_rounds/rXXX/optimization_steps.csv` | 两步损失、C 梯度范数、C 范数、有效权重补充范数 |
| `b_transfer_rounds/rXXX/matrices.npz` | 返回的 C，供离线核查，不跨轮复用 |
| `b_transfer_rounds/rXXX/receiver_summary.csv` | 接收端补充前后 Tail/Non-tail 训练侧功能及补充大小 |
| `b_transfer_rounds/rXXX/probe_metrics.csv` | 各客户端—类别在不同阶段的 LA、accuracy、margin |

阶段监测使用同一套固定本地训练图片：

1. `local_before → local_after`：接收端是否在本地吸收了帮助。
2. `ordinary_global_B → transferred_global_B`：在同一普通训练结果上，B 迁移对共享 B 的即时影响。
3. `transferred_global_B → committed_global`：之后的 A 更新是否保留这些收益；第 100 轮没有 A 更新。

非尾类监测是每接收端最多 16 张固定图片，不代表完整非尾类测试集。阶段汇总先在每个接收端内对存在的类别取均值，再对接收端取均值；不等于正式测试集的全局类别宏平均。最终模型优劣仍看正式测试结果，统一汇报第 81–100 轮平均的 Overall、Head20、Middle60、Tail20 及尾类遗忘。

B 的额外开销与 A 功能修正分开记录：`algorithm_forward_images` 包括筛选和 C 训练，`algorithm_backward_images` 只包含 C 训练，`diagnostic_forward_images` 是额外观察成本。`extra_downlink_bytes`/`extra_upload_bytes` 为 B 额外通信建模，普通训练的通信仍在原 `budget.csv` 中。`progress.json` 增加 B 优化步数及包含迁移和 A 修正的总步数，原普通训练预算字段的含义不变。

比较 B 是否有效时，使用**相同 A 版本、λ、μ、数据划分和训练协议的 A-only** 作对照。已有对应 A-only 可直接复用。原 Full-10 可用于复制实验协议，但它不是 `full-cp μ=1 + B` 的同 A 配置对照，不能把两者差值全部归因于 B。

### 汇总并打包

在各节点结果已经汇集到同一个 `sfra_b_transfer` 根目录后执行：

```bash
python -u scripts/run_cliplora_b_transfer.py --stage pack
```

输出：

```text
output/cifar100_LT/sfra_b_transfer/analysis/report.md
output/cifar100_LT/sfra_b_transfer_analysis.tar.gz
```

如需把已完成的同配置 A-only 也纳入汇总，使用 `--reference-run` 指向该 A-only 目录。压缩包包含本根目录的 CSV/JSON/NPZ 等分析文件，不含大型模型权重；外部参考目录不会被整体复制进包，参考性能会进入汇总表。

## 6. 本次验证边界

本次实现只进行源码检查和 Python 语法解析，没有启动训练、冒烟测试或 GPU 数值测试。因此尚未验证服务器端运行速度、显存占用和实验收益。
