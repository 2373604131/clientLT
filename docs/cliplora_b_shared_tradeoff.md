# 方法 B：保留 donor + 矩阵 C 的覆盖与收益—代价优化

## 改动边界

继续使用现有 donor 迁移和共享完整矩阵 C。不改为直接训练 B，不改为标量 donor 加权，不新增蒸馏、额外模型、测试门控或回退。

本次只改变两件事：

1. 非尾校准样本从“随机抽图片”改为“轮流抽本地不同类别，再取图片”。
2. 校准数据项使用可调的尾类损失占比 `w`；预定比较 `0.5 / 0.35 / 0.2`。

旧入口 `run_cliplora_b_shared_transfer.py` 保留原随机非尾采样、50:50损失及旧目录。新入口为 `run_cliplora_b_shared_tradeoff.py`，默认 w=0.35、类别轮换采样。

## 固定不变的设置

| 项目 | 设置 |
| --- | --- |
| A | Full-CP，λ=10，μ=1，更新与修正全部不变 |
| 普通 B | 本地3个epoch、LA τ=1、样本量FedAvg不变 |
| LoRA | rank=4，视觉最后3个block的q/v，6个模块 |
| donor | 缺类且在普通共享B上有正向LA探测收益；探测步长0.1；共享并集 |
| C | 每donor每模块一个完整r×r矩阵，从该模块B形状读取r；当前4×4 |
| 注入 | `R = mean_j(ΔB_j @ C_j)`；普通FedAvg后提交 `B_bar + R` 一次 |
| 训练对象 | 只训练C；A、基础B、骨干与原始donor更新固定 |
| C优化器 | Adam，lr=0.3，betas=(0.9,0.999)，eps=1e-8，weight_decay=0 |
| C步数 | 每次迁移两步同步更新，每步所有接收端反馈后再step |
| 正则 | 0.001 × C平方和 / (donor数×模块数)，每步仅一次 |
| 时机 | 第30、40、50、60、70、80、90、100轮 |
| 接收端与尾类 | 原持有Tail20的客户端；不扩大为Few30，不增加反馈客户端 |
| 校准预算 | 两批原有尾类样本，非尾样本数与原两批逐批相同 |

## 非尾抽样的具体实现

先完整调用旧 `calibration_batches`，保留它产生的**两批**尾类样本以及各批非尾样本数量。随后只替换非尾样本，不改种子42的普通训练流，不改变探测样本。

新采样使用独立的NumPy RNG：`SeedSequence([seed, round, client, 271829])`。

- 将本地非尾类别随机排列。
- 依次轮流从类别中取一张图片，每批不重复使用同一张图片；该类别图片耗尽则继续其他类别。
- 类别游标跨两步延续，避免第二批总从同一批类别开始；每步重建类内图片袋，允许两步之间图片重叠，与原训练侧协议一致。
- 每批仍最多16张非尾图片，小客户端保持原数量；只有尾类的客户端不改变。
- 三个w设置使用完全相同的校准样本，w不会参与采样。

不根据测试准确率、当前哪一类失分或模型输出选择非尾类别，也不新增每类别权重参数。

## C 的新目标

对同时具有尾类和非尾类样本的接收客户端：

`local_loss = w * mean(LA_tail) + (1-w) * mean(LA_non_tail)`。

只有尾类时仍使用完整的 `mean(LA_tail)`，不无故将该客户端贡献乘w。各接收端依旧等权平均，再加原来的单次C正则。

`w`是**本地校准损失中尾组的系数**，不是客户端聚合权重、donor权重、迁移残差缩放、LA τ、方法A的λ/μ，也不是梯度贡献比例。类别轮换改善的是样本覆盖，并不保证精确的全局类别宏平均目标。

| 实验 | w | 含义 |
| --- | ---: | --- |
| B1 | 0.5 | 仅改变非尾覆盖，原组间权衡不变 |
| B2 | 0.35 | 优先尝试的收益—代价折中 |
| B3 | 0.2 | 更重视非尾学习的配置 |

三个配置全部报告；允许类别间取舍，不要求每组准确率同时提高。此阶段优化效果，暂不新增直接训练B、标量C等独立价值消融。

## 启动：每个单卡节点一条命令

同步本次代码至服务器，进入仓库根目录并激活原训练环境。以下均从头完整训练100轮、前台输出，启用与上一轮新B相同的fast-v2执行路径。

B1，w=0.5：

```bash
python -u scripts/run_cliplora_b_shared_tradeoff.py --transfer-tail-weight 0.5 --reference-run references/full10_clientlt --fast-execution-v2
```

B2，w=0.35：

```bash
python -u scripts/run_cliplora_b_shared_tradeoff.py --transfer-tail-weight 0.35 --reference-run references/full10_clientlt --fast-execution-v2
```

B3，w=0.2：

```bash
python -u scripts/run_cliplora_b_shared_tradeoff.py --transfer-tail-weight 0.2 --reference-run references/full10_clientlt --fast-execution-v2
```

同服务器多卡时，分别在命令前加 `CUDA_VISIBLE_DEVICES=实际GPU编号`。不要把三个命令接成一行而遗漏分隔，也不要放到同一张GPU同时训练。

`--reference-run`只复用已有划分和调度，不加载旧模型权重。必须使用服务器上实际存在的协议目录；已有A-only和旧B无需重跑。

输出在：

```text
output/cifar100_LT/sfra_b_shared_tradeoff/seed42/client-longtail/full-cp/
  lambda10_mu1_protocol42_bshared_b_lr0.3_probe0.1_reg0.001_ntclass-cyclic_tw0.5_fast_v2_f64_c4/
  lambda10_mu1_protocol42_bshared_b_lr0.3_probe0.1_reg0.001_ntclass-cyclic_tw0.35_fast_v2_f64_c4/
  lambda10_mu1_protocol42_bshared_b_lr0.3_probe0.1_reg0.001_ntclass-cyclic_tw0.2_fast_v2_f64_c4/
```

中断后原命令加 `--resume`，从最后完成的轮次恢复；不能拿旧B权重作为新配置的续训起点。不要更改w后声称续训同一个实验。

三组完成后：

```bash
python -u scripts/run_cliplora_b_shared_tradeoff.py --stage pack
```

输出 `output/cifar100_LT/sfra_b_shared_tradeoff_analysis.tar.gz`。打包时不必附带 `references/full10_clientlt`，因为那通常只是协议目录而不是完整实验结果。

## 记录与比较

- `sfra_config.json`中的B子配置、`b_transfer_config.json`保存新版本、采样规则和w；A子配置不变。
- `calibration_manifest.json`保存真实尾/非尾样本位置；结合原划分清单可核算类别覆盖与loss显式权重。
- `client_feedback_steps.csv`保存每客户端每步的实际尾/非尾损失权重，尾类独占客户端记为1/0。
- `performance.csv`及其余汇总带 `transfer_non_tail_sampling`、`transfer_tail_weight`，方法标记为 `full-cp+B-shared-tradeoff`。
- 沿用原来的矩阵C、梯度、有效残差、预算、官方逐类准确率等记录与checkpoint。

先比较B1与已完成的共享C，再比较B2/B3与B1。主要看Tail20/Few30收益、Overall和其他类代价，不把“B是否减少遗忘”当作它的主要任务。统一报告81–100轮平均，之后可离线重算Many/Medium/Few。

## 验证边界

本地进行CPU小张量单元检查、启动参数/恢复/汇总检查和代码审查；不加载CIFAR、CLIP，不进行GPU冒烟或正式训练。正式准确率与耗时要由服务器实验确定。

本次检查：`python -m unittest discover -s tests -p 'test_sfra*.py'` 共发现75项，74项通过，1项CUDA缓存放置检查因本地无GPU跳过。另与已归档的352批校准样本逐批核对，原尾类位置和非尾图片预算全部保留；旧共享C配置字典与归档一致。`git diff --check`通过。

## 学习率与2026-09-26新增的C损失诊断

“A设置不变；C学习率为0.3”是两个独立事实。并不是A/C都用0.3，也不是整个实验冻结A。

| 阶段 | 更新对象 | 其他对象 | 学习率/步长 |
| --- | --- | --- | --- |
| 普通B本地训练，100轮 | B | A固定 | SGD，0.001，每客户端3个epoch |
| 八次donor迁移 | 完整矩阵C | A、基础B、原始donor ΔB固定 | Adam，0.3，每次从零初始化，2步 |
| A本地提案，前90轮 | A | B固定 | SGD，0.001，每客户端1个epoch |
| A服务器功能修正，前90轮 | A的提案子空间坐标 | B固定 | 归一化坐标投影步长0.1，3步；不是普通A参数SGD学习率 |

C不是直接训练的LoRA B参数，而是通过 `mean_j(ΔB_j @ C_j)` 形成附加B更新。因此0.3与B的0.001并非同一参数化下的倍率比较，且优化器不同。不过不能由“C小、只训练两步”推断0.3一定合适；该数值沿用旧迁移扫描，需要在新目标下检查。

原 `optimization_steps.csv` 的两行 `la_before` 对应两个不同batch，不能直接以它们的差值认定loss下降。

新版默认增加只读诊断：在**相同的两个校准batch**上，分别测量 `C=0`、第一步后、第二步后，逐次写入：

```text
<run>/b_transfer_rounds/r030/c_loss_trace.csv
<run>/b_transfer_rounds/r040/c_loss_trace.csv
...
```

关键字段：

- `fixed_pool_la`：固定两批、客户端等权、尾/非尾按w加权的LA。
- `fixed_pool_objective`：上述LA加上实际系数乘过的C正则；可在三个时点直接比较。
- `tail_la / non_tail_la`及对应accuracy：分别显示两类组的变化，不把总loss下降误认为所有类别都受益。
- `batch1_la / batch2_la`：始终在同一批样本上比较每个C状态。
- `regularizer / regularization_penalty`：分别为未乘系数和已乘系数的正则项。
- `c_norm`、`effective_transfer_norm`、`transfer_to_ordinary_ratio`：C和真实LoRA改动规模；普通更新为零时比值留空。

`optimization_steps.csv`同步增加 `learning_rate`、`same_batch_la_after`、`objective_after_same_batch`、`c_update_norm`、固定集合前后目标等字段。控制台显示同batch前后LA和固定集合前后总目标。

这些数据是**训练侧优化诊断**，两批中可能重复出现同一图片，不是独立验证集，也不是官方测试集。accuracy为raw-logit组内样本均值再跨客户端/batch平均，与已有probe逐类平均口径不同。不同w对应不同的目标函数，不能仅按各配置的总loss绝对值比较好坏。

诊断增加3遍固定两批样本的无梯度前向，计入 `diagnostic_forward_images`，不增加C反向或optimizer steps；独立保存/恢复RNG，不改变学习率、采样、donor规则、提交位置，也不依据loss回退。现有协议每次两批共794个图像访问时，新增2382个诊断图像访问，而非新增训练图片。会有额外前向耗时。

原训练命令不变，更新代码后新启动或正常续训的进程自动记录。已在运行的旧Python进程不会热更新；已完成的旧轮次不能从现有日志补回这些新诊断，无需为记录loss重跑已完成实验。

原 `--stage pack` 自动汇总到 `analysis/b_transfer_loss_trace.csv` 并打包。旧归档没有该文件也仍可正常汇总。
