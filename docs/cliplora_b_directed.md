# 方法 B：缺标签来源的尾类定向知识迁移

## 固定目的

**功能保持能够改善已有尾类能力的维护，但直接监督来源仍然有限。进一步分析表明，缺少目标尾类标签的客户端也能提供正向帮助，因此我们通过 B 的定向知识迁移，增强尾类能够获得的有效学习支持。**

B 在同一抗遗忘目标下处理“正向支持如何增强”。功能保持与支持增强构成相连的两个环节。本文档和每次运行保存的 `sfra_config.json`、`b_transfer_config.json` 均记录上述目的。

此处“正向帮助”的既有证据来自训练侧真实前向探测；新实现的全程测试增益尚未验证。代码正确性测试不能代替泛化结果。

## 当前已实现的边界

- 入口：`scripts/run_cliplora_b_directed.py`。
- 运行时：`utils/cliplora_b_directed.py`；筛选、重建与类别权重：`utils/b_directed_math.py`。
- 接收端从 Client-LT 协议指定的尾部客户端组取得。现有划分是 27、28、29，持有 119 张尾类训练图片，覆盖 20 个尾类。其他客户端上零散的尾类样本不再作为 B 的反馈数据。
- donor 必须来自目标客户端组之外，且**缺少正在帮助的那个目标类**。缺类按实际本地类别计数判定；donor 可以持有其他尾类。类别计数表与实际训练标签不一致时直接报错。
- B 的探测和 C 学习只读取目标端的尾类图片。没有非尾损失、Overall 优化项、尾/非尾权重 w 或非尾损害约束。
- 沿用完整 100 类 logits、全局训练先验 LA、同一共享模型。没有改变为仅区分 20 类的任务，也没有增加尾类推理分支。
- 维持原 A 设置、普通 B 训练与 FedAvg、迁移轮次 30/40/50/60/70/80/90/100、两次 C 优化、C 学习率 0.3 和矩阵正则 0.001。
- 新版本从头执行完整 100 轮，默认独立输出根目录 `output/cifar100_LT/sfra_b_directed`。旧共享 B、tradeoff、problem2 的入口、配置和断点继续保留。

## 为什么这次筛选会真正进入更新

旧版对每个接收端筛选，再把所有 donor 取并集，为并集中的每个 donor 配置一个自由矩阵 C。这样会丢掉“这个 donor 被哪个目标类认可”的关系。

新版保留 **目标类 → 缺标签 donor → 固定权重**，先构造各目标类的来源组合，再学习组合的变换。donor 并集只用于统计和存储，不是优化器的来源池。

### 1. 在普通共享 B 上逐类探测

普通本地 B 训练和 FedAvg 结束后，保存实际共享状态 B_bar，以及各客户端未经迁移的原始更新 ΔB_j。探测期间固定 A。

对目标尾类 c，仅在 `n[j,c]=0` 的外部客户端中考察 donor j。在 B_bar 上单独注入 `0.1 * ΔB_j`，所有 donor 都从同一 B_bar 出发，不累计注入。

`g[c,j] = L_c(B_bar) - L_c(B_bar + 0.1 * ΔB_j)`。

L_c 只使用目标尾部客户端组中 c 类的完整训练池：先汇总该类损失与样本数，再计算样本平均。它不按客户端数量重复加权。

每类取增益超过 1e-6 的 Top-3 donor，并以正增益归一化为 p[c,j]。并列时按客户端 ID 固定顺序。没有合格 donor 的类不补入负收益来源。

### 2. 把筛选关系固定在来源组合中

对每个有合格 donor 的类 c 和每个 LoRA 模块：

`U_c = sum_{j in D_c} p[c,j] * ΔB_j`。

这一轮中 D_c、p[c,j] 和 U_c 全部固定、停止梯度。只学习每个 **类别来源组合—模块** 的 r×r 矩阵 C_c。

这是一个明确的结构改变：C 从旧版的“每 donor 一个”改为“每个类别来源组合一个”。它使选中关系保留到实际更新，不能在校准时重新为整个 donor 并集分配自由矩阵。

### 3. 始终在最终共享模型上学习

设 T 是目标客户端组实际持有的尾类，T+ 是筛到来源的类。最终残差为：

`R(C) = (1 / |T|) * sum_{c in T+} U_c @ C_c`。

没有来源的类贡献零；分母仍是所有目标类数量，不能因来源缺失而自动放大其他类的更新。C 每次从零初始化，沿用两步联合 Adam：每一步收齐目标端所有尾类反馈后才更新一次。

目标函数为：

`L_tail(C) = (1 / |T|) * sum_c [sum_{k in target clients} sum_{x in D_kc} LA(x; B_bar + R(C)) / N_c]`

再加一次矩阵平方正则，归一化因子为有来源的类别数 × LoRA 模块数。各客户端可以分批计算，但每个样本都使用全局该类的 N_c 和同一个 C 版本，分批不重新归一化。

所有反馈都在同一个实际提交候选 `B_bar + R(C)` 上计算，避免分别优化不同局部模型后误称为共享模型收益。A、基础 B、原始 donor 更新及固定组合 U_c 不接受梯度。

最后只提交一次 `B_bar + R(C)`，随后继续执行 A。额外迁移项不再乘目标端样本权重，也不进行第二次 FedAvg。未选来源仍按原协议参与普通 FedAvg，但不进入相应类别的额外来源组合。

全部目标类都找不到合格来源时，本次额外残差精确为零，C 优化步数为零。

## 需要准确解释的范围

- 每类 Top-3 不代表整轮最多 3 个不同客户端。多个类可以选择不同 donor，并集仍可能较大；只要逐类关系和权重实际进入 U_c，就不等同于旧版不受约束的并集校准。
- 类 c 的来源组合只能包含缺 c 的 donor，但共享模型的任何更新都可能影响其他类。没有声称实现推理时的类别隔离或每类均无负迁移。
- 逐 donor 的正探测收益不保证合成后、变换后或测试集上仍然为正。B 使用固定第二步提交，不根据测试精度回退或挑检查点。
- 只用尾类监督不等于把其他类别当成不存在：完整 logits 仍包含其他类别作为分类竞争项。
- A/B 的职责按目标划分，不能预设 A 自动补偿所有副作用。正式结果仍保存 Overall/Head/Middle/Tail，B 的主要效果看固定 Tail20 的最后 20 轮测试均值。
- 没有使用独立留出数据来选择 donor。探测与优化都属于训练侧反馈；每次类损失使用完整目标尾类训练池，不能称为验证集增益。

## 与上一版审核稿的区别

用户这次明确固定了“缺少目标尾类标签”的来源定义。因此，上版稿中“放开缺类限制、按全部尾类目标选一个全局小集合、非零初始化 C”的提议不再采用。

本版实现为逐目标类缺失、每类正收益 Top-3、固定来源组合、每类组合 C、零初始化。此前的 E00/E10/E01/E11 重放不是本版的运行前置条件，不需要另开逐轮离线检测实验。

## 服务器启动

同步本次新增与修改的文件后，在服务器 `/data/yzh/clientLT`、已激活 `clientLT` 环境的终端执行。下方是一个完整训练，不是一组参数扫描。GPU 编号可按空闲卡修改。

```bash
CUDA_VISIBLE_DEVICES=1 python -u scripts/run_cliplora_b_directed.py --reference-run references/full10_clientlt --data-root DATA --fast-execution-v2 --feedback-batch-size 128 --feedback-cache-gib 4
```

这里复用划分、顺序和调度，不加载旧 B 的训练后权重。不要附加 `--transfer-tail-weight` 或 `--transfer-non-tail-sampling`；这两个旧权重/覆盖选项不控制新方法。

中断后，同一配置恢复：

```bash
CUDA_VISIBLE_DEVICES=1 python -u scripts/run_cliplora_b_directed.py --reference-run references/full10_clientlt --data-root DATA --fast-execution-v2 --feedback-batch-size 128 --feedback-cache-gib 4 --resume
```

恢复从最近已完成轮次继续；若某次迁移中断，该轮重做，不复用半训练 C。不能用旧方法的断点恢复为本方法，配置改变也不能复用同一断点。

完成后汇总与打包：

```bash
python scripts/run_cliplora_b_directed.py --stage pack --output-root output/cifar100_LT/sfra_b_directed
```

分析包：`output/cifar100_LT/sfra_b_directed_analysis.tar.gz`。打包不附加只有协议的 `references/full10_clientlt`。

## 文件与日志

| 文件 | 用途 |
|---|---|
| `sfra_config.json` / `b_transfer_config.json` | 固定目的、接收端、缺类规则、Top-K、阈值、C 结构与完整优化设置 |
| `b_transfer_manifest.json` | 目标组、实际反馈端、每类样本总数、完整训练位置、未观测尾类 |
| `b_transfer_rounds/rXXX/donor_scores.csv` | 所有合格候选的逐类增益、正负标记、是否选中、最终来源权重 |
| `source_weights.csv` | 实际进入重建的“类别—donor—权重”，以及 donor 的该类计数为零的证据 |
| `calibration_manifest.json` | 固定来源组合、C 的类别行序、目标样本位置；明确并集仅为统计 |
| `matrices.npz` | 类别索引、donor 索引、稀疏来源权重和每类 C |
| `commit.pt` | 原共享状态、实际类别组合 U、C、最终残差及提交状态 |
| `client_feedback_steps.csv` | 各目标端的加权尾类损失与图像数，非尾图像数始终为零 |
| `optimization_steps.csv` | 两步联合 C 优化的损失、正则、梯度范数与类别数量 |
| `probe_metrics.csv` | 普通 B、迁移 B、随后 A 提交的同一目标训练池监测 |
| `analysis/b_directed_sources.csv` / `b_directed_probes.csv` | 跨事件、跨运行的实际来源和候选汇总 |
| `analysis/performance.csv` | 正式第 81–100 轮测试均值，区分 directed 与旧 B |

CPU 已验证缺类约束、真实构造器的接收端过滤、同并集不同筛选关系会改变重建、类别平均权重、全模型联合梯度、rank-4/归一化 logits/LA 前向、冻结基础状态、单次提交、空来源零更新、断点与打包隔离。无本地 CUDA，未声称已完成真实 CIFAR/CLIP GPU 训练或验证性能提升。

代码文件：`federated_main.py`、`scripts/run_cliplora_sfra.py`、`scripts/run_cliplora_b_directed.py`、`utils/cliplora_sfra.py`、`utils/cliplora_b_directed.py`、`utils/b_directed_math.py`、`tools/sfra/summary.py`、`tools/sfra/b_problem2.py`。测试文件为 `tests/test_sfra_b_directed.py`。
