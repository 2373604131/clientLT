# PFRF 两阶段实验协议 v1

状态：第一阶段六组真实 GPU smoke 及 Max/Add 断点恢复已通过；第二阶段 100 轮 kill-test runner 已实现，真实运行与结果分析待完成。本文件不是已完成实验报告。对应配置清单为 `docs/pfrf_pilot_v1.json`；smoke 和 kill-test 训练参数分别由 `scripts/run_pfrf_smoke.py` 与 `scripts/run_pfrf_kill_test.py` 显式冻结。

目的：先验证实现正确，再检验客户端保存历史功能目标是否带来超过额外 memory 训练、类别加权和单轮目标的收益。seed 42 仅用于筛查，不能据此声明跨种子稳定或 CVPR 接收概率。

## 1. 研究问题与冻结范围

主要比较为 PFRF-Max 对 matched-memory CE、class-reweighted CE 和 instantaneous target。六组统一使用 Effective SVD-FedAvg；普通 CE 用于报告基础性能和额外计算成本。

本版明确选择“持久绝对目标记忆”语义。历史目标即使曾被全局模型超过也不会删除，因此全局后续退化时可再次触发。它与每次兑现就清零的旧 residual-Max 在超额兑现后不完全等价，论文和实现均不得再称无条件等价。

| 项目 | v1 默认值 |
| --- | --- |
| 数据 | CIFAR-100-LT，exp imbalance，imb_factor=0.01 |
| 划分 | client-longtail；seed=split_seed=42；30 clients |
| 拓扑参数 | specialization_lambda=0.75，intra_group_alpha=0.5，head_leakage_scale=3.0；head/tail client ratio=0.9/0.1；head_class_ratio=0.8 |
| 参与 | frac=1.0，各组共用 seed=42 生成的逐轮固定客户排列 |
| 模型 | CLIP ViT-B/16，vision-only，top3，q/v，rank=2，alpha=1，dropout=0，FP32 |
| 优化 | SGD，lr=0.001，constant LR；每客户端每轮重置 optimizer；本地阶段切换不重置 |
| Epoch / batch | 2 epoch CE proposal + 1 epoch correction；train batch=32，eval batch=64 |
| 时长 | smoke=5 轮；kill-test=100 轮，轮数在运行前固定 |
| 方法参数 | 温度 tau=1，auxiliary lambda=1，Max 无泄漏；memory class weight 均匀 |
| 额外组件 | SCA、Stage-3 proposal/restore、E1、Experiment D 均关闭；保持基线既有增强，不引入新增强 |

解析后的完整 optimizer 配置（含 momentum、weight decay、warmup）必须写入运行 manifest 并在六组间逐项相同，不能只记录上述摘要。调度器策略复用现有常量 LR / 单 epoch 单 scheduler-step 协议。

## 2. 数据访问与可重复性

保留原始联邦训练池 D_k，保持 n_k、n_c 和 client×class 矩阵在六组间完全一致。第一版不从稀缺尾类中移除 audit 样本。

- E_k：从 D_k 固定、按类均衡抽取最多 32 个 functional-memory 样本，可参与 CE 与辅助损失。
- A_k：从 D_k\E_k 固定抽取最多 28 个样本。仅用于 PFRF 控制之外的本地审计，但仍可能参加普通 CE；必须标注为训练集审计，不能当作未见数据泛化证据。
- U：复用 `tools/eri_closure/protocol.py` 的训练池外 probe 构造：每个 tail class 10 个 CIFAR-100 原始 train 样本，确保 raw ID 不属于整个联邦 LT 池。U 只用于离线诊断，不交给客户端损失或服务器聚合；它是额外评估数据访问，不能宣称方法训练使用了这些样本。
- 官方 test：只进入冻结实验的评估路径，不参与 H、lambda、memory 选择或训练停止规则。

只对 E_k 实际覆盖的类别 C_k^E 定义目标；未覆盖类别不填 0，不伪造功能值。记录每个客户端及 tail supporter 的 memory 类别覆盖率。A_k 为空时记录 NA，不能记为 0。

E_k/A_k/U 必须使用跨轮不变的视图，禁止每次读数重新随机裁剪。当前 CIFAR wrapper 实际使用确定性 ToTensor/Normalize/Resize(224)，PFRF 读取时还会保存并恢复 Python/NumPy/Torch RNG；若未来替换 dataset wrapper，必须重新验证视图指纹，不能假设 Stage-3 的 RNG 保留自动充分。普通 CE 保留仓库的原训练变换；六组按共同 seed 固定采样顺序，诊断不得消耗训练 RNG。

必须保存 split、样本 IDs、transform 配置、初始化、schedule、解析配置与代码版本的 hash。客户端原始类别数据与目标只能进入本地状态/离线仿真日志，不进入聚合 API；不据此声称差分隐私保证。

## 3. 功能量与六组精确定义

定义 p_y(x;M)=softmax(logits(x;M)/tau)_y，tau=1：

    f_{k,c}(M) = mean_{(x,c) in E_{k,c}} p_c(x;M)
    R(M;J) = mean_{c in C_k^E} relu(J_{k,c} - f_{k,c}(M))^2

冻结 CLIP logit_scale。概率有界只解决目标尺度和 Add 裁剪问题，不能保证置信度改善等于准确率改善；同时记录准确率、正确类概率和 true-class log-odds。不要声称有界概率天然解决过拟合。

每轮从同一个全局模型开始：先读 f0=F(M_t)，执行 2 epoch 普通 CE，读取 fhat=F(Mhat_CE)，detach 所有目标，再连续执行第 3 epoch。第 3 epoch 的每个普通 CE minibatch 配一个相同的完整 E_k forward/backward，合并损失后只执行一次 optimizer.step。完整 memory 上先算每类平均，再算 hinge；minibatch 内 hinge 的平均不等于该目标。

| ID | 名称 | 第三个 epoch 的目标/辅助损失 | 跨轮状态 |
| --- | --- | --- | --- |
| B0 | ordinary_ce | CE(B) | 无 |
| B1 | matched_memory_ce | CE(B) + lambda × mean_i CE(E_i) | 无 |
| B2 | class_reweighted_ce | CE(B) + lambda × mean_i w_{y_i} CE(E_i) | 仅固定本地频数 |
| B3 | instantaneous_target | CE(B) + lambda × R(M;fhat) | 无；每轮重建目标 |
| M1 | pfrf_max | CE(B) + lambda × R(M;H) | 持久 H |
| A1 | pfrf_add | CE(B) + lambda × R(M;T) | 持久 pending target T |

B2 使用本地原始 D_k 频数：raw_w_c=1/sqrt(n_{k,c})，再除以 memory 样本的 raw_w 均值，使 mean_i w_{y_i}=1。不能再额外乘 residual。该定义与 class-balanced memory 采样不同，避免 B1/B2 实际成为同一组。

M1 初始化 H=f0；之后只在 CE proposal 后执行 H=max(H,fhat)。上传后模型和后续全局模型均不得提高历史 H。校正前 deficit=max(H-fhat,0)，收到下一全局时的 gap=max(H-F(M_{t+1}),0)。H 在当前轮冻结、不可微。

A1 初始化 T_prev=f0；在第 t 轮：

    e = relu(T_prev - f0)
    a = relu(fhat - f0)
    raw_target = f0 + e + a
    T = min(raw_target, 1)

T 作为本轮绝对目标保存到下次参与，不由最终本地模型反向更新。记录 target_cap_rate 和 discarded_target=max(raw_target-1,0)。这是有裁剪的累积压力对照，不是“已证明可实现的历史目标”，不能与 M1 共用更新器。

B3 在 proposal 模型处初始 hinge 为零；M1 如果 CE 已经重现历史目标也会为零，这是预期行为。记录第 3 epoch 的 active-hinge 比例和辅助梯度范数；历史 gap 存在但 CE 已自动修复时，不应错误报告反馈实际触发。

B1–A1 精确匹配 memory 图像访问和 optimizer steps。B0 保持普通 CE，单独报告其更低的计算量；不声称 B0 与辅助组同 FLOPs。各 loss 的数值/梯度尺度不同，同 lambda 并不等于同梯度预算，需记录辅助/CE 梯度范数比。

## 4. Effective SVD 聚合的约定

逐 LoRA 矩阵聚合绝对有效适配器 D_k=s B_k A_k，q_k=n_k/sum_selected(n_k)，Dbar=sum_k q_k D_k。投影的是 Dbar，而不是只投影平均增量再加旧适配器（后者会突破目标 rank）。

运行时从模块读取 scaling。当前 `utils/loralib/layers.py` 使用 alpha/sqrt(r)，不能写死为 alpha/r。同时处理 fan_in_fan_out 与矩阵 reshape。

对 Dbar 做 rank-r SVD 投影，正奇异值方向采用平衡分解：

    B = U_r diag(sqrt(sigma_r / s))
    A = diag(sqrt(sigma_r / s)) V_r^T

零奇异值槽不能令 A/B 都为零，否则两个因子该槽的梯度都为零。零槽使用确定性非零 A 行、零 B 列；全零 Dbar 时保留旧非零 A（必要时用独立固定种子初始化），令 B=0。真实正奇异值不能随意舍弃；秩判定阈值需写入 manifest。

测试比较 sBA、logits 和尾奇异值误差，不要求不同 SVD 实现的 A/B 逐元素相等。六组使用完全相同分解、符号约定和零槽策略；重参数化可能改变下一轮优化动力学，不能将它称为与 factor-FedAvg 训练完全等价。

## 5. 第一步：确定性单测 + 5 轮 smoke

先跑 CPU 合成单测，再跑 5 轮真实 ClipLora。正确性事件不保证自然出现在 5 轮训练中，因此必须显式构造以下测试。

| 检查 | 构造 | 通过条件 |
| --- | --- | --- |
| 初始化/提高目标 | f0=.2，fhat=.5 | 初始 H=.2；proposal 后 H=.5；初始化 gap=0 |
| 不降低/不自举 | 后续 fhat=.45；最终上传 F=.9 | H 仍为 .5，不能取 .9 |
| 兑现和重现 | H=.5，全局 F依次 .3/.6/.35 | gap 依次 .2/0/.15，H 保留 |
| Max/Add 区分 | 旧目标 .5，f0=.2，fhat=.5 | Max target=.5；Add target=.8；Add=.8 不被描述为已达能力 |
| Add 上限 | Tprev=.9，f0=.1，fhat=.6 | raw_target=1.4，T=1，discarded=.4 |
| 缺席返回 | 客户端在轮 1 后缺席 2–4，轮 5 返回 | 目标不被缺席客户端循环意外覆盖，gap 按当前 F 重算 |
| 损失/梯度 | 人工 J>F；J=F；J<F | active 时梯度存在；后两者 hinge=0；target 不带梯度 |
| 聚合 | 单客户端、所有客户端相同、非对称小矩阵、rank>r | sBA 恢复正确；rank<=r；误差平方符合尾奇异值能量 |
| 因子不变性 | A'=RA，B'=BR^{-1} | Dbar 不变，投影结果等价 |
| 零适配器 | Dbar=0 后做一个 CE step | 前向不变且至少一个因子有可学习梯度，不进入双零死点 |
| 断点恢复 | 连续 5 轮 vs 2 轮存盘+恢复到 5 | 相同 manifest、状态及 RNG；最终 sBA/H/T/指标在容差内一致 |
| 状态隔离 | 客户端/condition 交错运行；修改数据 fingerprint | 不串状态；不兼容 resume 明确报错 |

数值验收：CPU FP64 合成矩阵 atol=1e-10、rtol=1e-8；FP32 sBA 重构与同环境 resume 默认 atol=1e-6、rtol=1e-5；frozen 权重必须完全不变。容差失败时记录误差并调查，不能在看见失败后悄悄放宽。重复/零奇异值的最优投影可能不唯一，以最优误差/前向一致性验证。

真实 smoke 先跑六组各 5 轮、完整 30 clients（缩小客户端数的运行只算调试，不代替此 gate）。M1/A1 额外从第 2 轮 checkpoint 分别恢复 3 轮。训练不强求准确率上升；必须有限值、步骤计数正确、目标/聚合/数据访问合约通过。

每客户端每轮 optimizer steps=3×len(train_loader_k)，scheduler steps=3；不能因为追加 memory 再多 step。smoke 为严格 resume 检查使用 num_workers=0；kill 若改 worker 数必须单独通过相同 resume 检查，否则沿用 0。

checkpoint 取全局轮边界，保存全局 LoRA 状态、全部客户端 H/T、round、全部 RNG、functional-memory 样本标识与配置约束。冻结 CLIP 权重由同一预训练模型重建，不在每轮重复存储；本实验无服务器 optimizer/scheduler 持久状态，客户端 optimizer/scheduler 每轮重建。恢复验证在轮边界进行，不承诺本地 minibatch 中途恢复。

输出 smoke_report.json，逐项 pass/fail/observed_error；只有全部通过才可进入第二步。已有仓库的聚合测试通过不等于本协议通过。

## 6. 第二步：seed 42 六组 kill-test

从共同 M0 全新启动六组，每组 100 rounds，不继承 smoke 的 H、模型或 optimizer。可以在六组各跑到 round 20 时做统一技术检查，所有有效运行仍续跑到 100，不按早期准确率淘汰。

统一入口为 `scripts/run_pfrf_kill_test.py`。它必须先读取已通过的 `output/pfrf_smoke_v1/smoke_report.json`，并将六组写入全新的 `output/pfrf_kill_v1/`。训练命令禁止 `--resume`，非空的未完成目录会直接拒绝续写。六组完成后生成 `kill_test_report.json`、`analysis/six_condition_metrics.csv` 和 `analysis/screening_summary.json`。

单节点六卡并行时使用 `--gpu-ids 0 1 2 3 4 5`，按六组冻结顺序一对一映射。父进程为每个子进程单独设置 `CUDA_VISIBLE_DEVICES`和确定性 cuBLAS 环境，每组标准输出写入自己的 `launcher.log`，不在六进程间共享模型、H/T 或可写日志。

若集群只允许分别申请六个单卡作业，每个作业使用 `--stage train --condition <name>`。此模式继承调度器提供的 `CUDA_VISIBLE_DEVICES`，并强制 PyTorch 只能看见一张卡；六个 condition 输出目录彼此隔离，共享的 schedule/protocol 使用跨节点唯一临时名和原子替换。六个作业都结束后，另执行一次 `--stage verify`。

第一个固定配置用 tau=1、lambda=1。若第 3 epoch 辅助梯度长期为零，优先判断目标是否已被 CE 兑现/概率是否饱和；不能直接认定历史反馈无价值。若辅助梯度非有限、相对 CE 严重失衡或 memory 覆盖失效，停止的是该实现/配置验收，需要修正版本并重跑六组。

首个配置阴性只能得出“v1 固定配置未通过”。若要继续调参，另建 v2 exploration：B1/B2/B3/M1/A1 获得同样的 lambda 候选 {0.1,1,10} 与轮数预算，选择仅用训练池外 U，官方 test 不可选超参。U 此后必须标为开发集；完整报告全部试验，已看过的 seed42 test 结果不作为未见确认。

数据池外 probe 本身较小，若效应接近其评估分辨率，应判为不确定，不因 probe 上几张图的变化直接终止方向。

主结果固定为 round 100 的 tail macro accuracy。辅报 round 81–100 均值、全程 tail AUC、non-tail macro、overall/macro、H-mean。按现有代码固定 bottom-20 classes 为 tail，其余 80 类为 non-tail；不混用 legacy cumulative head/medium/tail，当前兼容输出的 medium=0 不能当实验结果。

各轮保存 LoRA checkpoint并使用固定评估轮集合 1…100 计算 BFD_c=max_t accuracy_c(t)-accuracy_c(100)。可以离线统一评估，以免测试曲线干预训练。BFD 是辅助指标：低准确率模型也可能 BFD 很小，必须连同 final accuracy 解读。

pilot 的建议扩大实验门槛（工程效应阈值，不是统计显著性）：

- M1 的 final tail 比 max(B1,B2,B3) 至少高 1.0 个百分点，且末 20 轮平均差为正。
- 相对 B0 的 non-tail 与 overall 降幅各不超过 1.0 个百分点；同时列出对最强对照的变化，不隐藏 trade-off。
- U 上至少有相符的方向性证据；memory-only 概率上涨不足以通过。
- M1 存在实际历史约束触发，且改善不完全由额外数据访问或 B2 解释。

若收益为 0–1 pp 或评估方向不一致，判为不确定；如果 B1/B2/B3 已解释全部收益，判为当前历史反馈假设未获支持。只 Add 成功时单独研究累积压力，不能声称 Max 机制成立。单 seed 不计算“跨种子显著性”，不把 client/class/round 当独立训练重复。

通过后冻结方法，用 5 个新 seeds {1,2,3,2026,2027} 配对确认，报告 seed-level 差值和区间；这仍是第一数据集确认，之后才扩展 topology、partial participation 和第二数据集。

## 7. 机制日志与解释边界

每个 client/class/round 保存 f0、fhat、target_before、target_after、F(upload)、F(next_global)、gap_before、gap_after、correction_deficit、active_hinge_fraction、memory_count、target_raise_source、Add cap/discarded。

在同一冻结目标 J 下分开记录：

    local_shortfall = J - F(upload)
    local_global_difference = F(upload) - F(next_global)
    global_shortfall_signed = J - F(next_global)
                            = local_shortfall + local_global_difference

先保留有符号值再算正部。正部不满足上述可加性。若客户端自己都达不到 J，不能将全部 global gap 归因于聚合；非线性功能差也不是某个客户端的因果贡献。

gap 越大、后续 gain 越大也可能是回归均值；这种相关图仅属诊断，不能替代 M1 对 B3 的受控比较。W/P 如需保留，应另行定义分子分母与零分母规则，本轮不把未定义比值设为 gate。

目标不断提高不自动等于失败；Max 目标有界、单调，并可能因最大值选择产生乐观偏差。真正的失败证据是持续不可达、泛化不改善或压制其他类。H 不下降也不等于 residual 必然下降。

记录 CE/aux forward images、backward images、optimizer steps、额外诊断 backward 次数、CUDA 同步后的客户端时间、SVD 时间、训练/评估总时间、峰值显存和上传字节。梯度范数诊断固定在轮 {1,5,20,50,100}、每客户端 correction 的首个 batch，B1–A1 频率相同且不修改实际梯度。不能把模型推理与反向的图像数直接当精确 FLOPs。

## 8. 交付文件与执行成本

第一阶段默认输出根为 `output/pfrf_smoke_v1/`；每个 run 保存 `smoke_command.json`、`resolved_config.yaml`、`round_metrics.csv`、逐类指标、`client_split_fingerprint.json`、`effective_svd_contract.json`、`effective_svd_diagnostics.csv`，以及 `pfrf/` 下的 `runtime_contract.json`、`client_functional_state.csv`、`budget.csv` 和逐轮 checkpoint。私有离线日志不作为服务器输入。

分析交付六组对照表、tail/non-tail 曲线、训练 memory 与 U 泛化对照、活跃约束比例/缺口曲线、signed shortfall 分解和预算表。结论必须为 pass / inconclusive / fail / invalid 四者之一，并给出对应证据；invalid 表示实现或协议失效。

kill-test 基础工作量为 6×100×30=18,000 次客户端训练，54,000 client-epochs。smoke 六组完整轨迹共 900 次客户端训练；Max/Add 各额外独立跑 2 轮前缀和恢复后 3 轮，共再加 300 次，合计 1,200 次客户端训练、3,600 client-epochs。由真实 smoke 测得每组每轮中位耗时估算剩余 GPU-hours，另计 SVD、评估和保存开销；本设计不预报未经测量的小时数。

实现接入点：`trainers/cliplora.py` 加 epoch 内阶段切换和可微 memory loss；`federated_main.py` 加客户端状态及新聚合选择；`utils/stage3_private_state.py`/`utils/stage3_runtime.py` 仅复用样本 ID 与存储思想，使用独立 PFRF schema；`utils/lora_aggregation.py` 加 effective 聚合；`tools/eri_closure/protocol.py` 复用池外 probe 构造。

不要直接套用现有 ERI runner：它当前开启 ERI 专用审计并固定 factor-FedAvg 路径/约束，不能代替六组 PFRF runner。第一阶段统一入口为 `scripts/run_pfrf_smoke.py`。
