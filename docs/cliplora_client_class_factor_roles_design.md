# LoRA 的客户端—类别双层适配分工：实证验证协议

日期：2026-09-18。

**当前执行协议：`factor_roles_v1`。只比较 Client-LT 与普通逐细类别 Dirichlet `noniid-labeldir-fine`，beta=0.5；不使用 matched-Dirichlet。**

状态：第一阶段短分支代码已实现，未运行训练或冒烟测试，不增加哈希审计。已有 J/S 结果是动机，不是本协议的验证结果。入口为 `scripts/run_cliplora_factor_roles.py`。

## 1. 要验证的明确主张

在当前 CLIP-LoRA / Client-LT 设置中：

- A 是 **客户端级知识适配因子（client-level adaptation factor）**：主要调节不同客户端提供的更新如何进入共享适配器，对客户端证据组织及其聚合影响力的修正具有更强功能响应。
- B 是 **类别级判别适配因子（class-level discrimination factor）**：主要承担类别判别平衡的调整，对类别频率校正具有更强功能响应。

这是“哪种适配作用更占主导”的实证角色定义，不是互斥的信息存储划分。研究目标是检测交叉分工，而不是预先规定实验必须支持 A/B 分工。

若结果通过，可概括为：客户端—类别双层适配不对称性。仅观察 A 对拓扑敏感，而没有 B 对类别校正的优势，不足以宣布完整分工成立。

## 2. 对照原理：两类干预、两个因子

| 干预 | A-only 分支 | B-only 分支 | 目标证据 |
|---|---|---|---|
| 客户端影响力修正 | 只改变 A 的客户端聚合权重 | 只改变 B 的客户端聚合权重 | A 上的修正更能减轻 Client-LT 特有的尾类损失 |
| 类别频率修正 | 只训练 A 时，CE 改为 LA | 只训练 B 时，CE 改为 LA | B 上的修正产生更强的类别平衡收益 |

主要比较是“干预收益之差”，不是两个分支最终谁的分数更高。

等权聚合是诊断性干预，不是最终方法或新颖性主张。等权也会改变有效类别影响力，因此需要 B 对照、配对拓扑和类别校正对照共同解释；不能单凭 A 等权提高 Tail 宣称已证明因子角色。

## 3. 共同模型起点与预算

复用 S 的 Client-LT 与普通 `noniid-labeldir-fine` 两条训练轨迹，固定读取第 20、50、80 轮完成后的全局状态，共六个锚点。

来源是 `checkpoints/base_model.pt` 加 `events/r020_c000_main_refresh_A/state.pt` 等文件中的 `actual_after_lora_state`，即该轮 B 训练和 A 刷新均完成后的状态，不是 A 刷新之前。两个划分的训练分配直接读取各自 S 的 `partition_manifest.csv`，不重新划数据，也不约束 Dirichlet 匹配 Client-LT 的客户端容量。

如果普通 Dirichlet S 还没有完成之前已安排的补跑，先完成那一项；已有 S 不重跑。本入口不自动训练任何 S/E/J 基线，也不会拿旧 matched S 冒充普通 Dirichlet S。某来源只需已经保留所选轮次的完整状态即可启动对应诊断，无须为了这个诊断再完整训练一次。

每一个锚点都分别在两种原始数据划分上启动全部诊断分支。不能把 CLT 的 A 分支与 Dir 的 B 分支直接当作同起点实验；两种锚点来源均报告，避免有利来源选择。

- 原模型、rank4、top3 Q/V、缩放 0.5、冻结骨干保持不变。
- 每个分支从完全相同的锚点 A*/B* 出发；另一因子严格冻结。
- 必须使用训练后的非零 B*。不能用 B=0 的 LoRA 初始化测试 A-only，否则 A 梯度为零。
- A/B 活动参数量相同，均为 18,432。
- 两种因子都只训练一个 local epoch；不是原 S 的 B 三个 epoch 对 A 一个 epoch。
- 两边统一 SGD lr=0.001、momentum=0.9、weight_decay=0，无 scheduler；每个客户端新建优化器。
- 30 客户端全参与，客户端顺序固定为 0–29，`drop_last=False`，每客户端完整遍历一次。每种划分内 A/B/CE/LA 分支复用同一 batch 随机流；每个客户端都恢复共同模型并重新建立优化器。
- RNG 种子只依赖轮次、训练划分和客户端，不依赖因子、损失、聚合权重、起点来源或启动顺序。分支独立恢复随机流，支持分节点执行。
- 使用仓库现有 `DatasetCifar100` 包装器。特别注意：当前 CIFAR 包装器实际执行的是固定 `ToTensor → CIFAR Normalize → Resize(224)`，会覆盖传入的普通增强 transform；不能因为配置写了 random crop 就宣称实际用了它。这里保留现有行为，不顺便修改历史训练预处理。
- LA 固定 tau=1，先验使用原全局类别计数；CE 固定 tau=0。测试 logits 不添加 LA。

## 4. 完整短分支矩阵

每个锚点执行：

`2 划分 × 2 活动因子 × 2 损失 = 8 个单轮本地训练分支`。

每个分支缓存全部客户端 delta，并用两种权重分别重聚合：

- sample：q_k=n_k/N，原方法；
- uniform：q_k=1/30，客户端等权。

因此每个锚点得到 16 个聚合候选状态，但只需要 8 次本地训练分支；总共 48 个“全客户端单 epoch”分支、96 个原始聚合候选，以及96个幅度匹配候选。每个起点另评估一次自身，共198次官方测试集评估和198次小 probe 评估。

当前 seed42 清单下，Client-LT 每 epoch 352 步，普通 Dirichlet 每 epoch 354 步：合计 `6×4×(352+354)=16944` 个 optimizer steps；代码按实际清单计算，不硬编码成旧 matched 的16,896步。同一训练划分内 A/B/CE/LA 的步数完全相同；跨划分保持同一训练样本池、全客户端和完整一个 epoch，不为凑一样的 batch 数而丢数据。

这不是48次完整100轮实验，也不把相同步数说成相同 FLOPs。逐客户端记录 A/B 分支的时间、峰值显存、步数和样本呈现数。原 S 的 A 刷新起点在 B 聚合后而本实验起点在该轮结束后，原 S 的 B 训练又是3 epoch，因此旧更新不能当成本次同起点同预算的诊断分支。

## 5. 输出指标与主对比

所有候选用同一官方测试集，只作指标报告，不通过测试结果选择聚合权重、学习率或锚点。

- Overall、Head20、Middle60、Tail20，以及每一类别的准确率变化。
- 对已有固定训练池外 Tail probe（20类各10张），报告真类 log-odds `z_c − logsumexp(z_not_c)` 的变化；该 probe 不参与本地训练。不是 top1 competitor margin。
- 本诊断的 train/test/probe 统一走现有 CIFAR 包装器；复用旧 probe 样本清单，但不与旧 ERI 独立 transform 下的 log-odds 数值直接拼接比较。
- 若 probe 仅覆盖20尾类，不把其功能指标称为100类功能结果。
- 固定类别组，不按干预后结果重新划头尾类别。

### 5.1 客户端级角色的主对比

在 LA=1 条件下，对每个活动因子分别计算：

`U_factor(topology) = Tail_uniform − Tail_sample`。

然后计算：

`K_factor = U_factor(CLT) − U_factor(Dir)`。

核心比较 `K_A − K_B`。同时报告两种拓扑各自的 U、四组绝对分数与 Head/Middle/Overall，不能仅凭 Dir 性能被压低而声称缩小拓扑差距。

支持 A 客户端级主导作用的结果，应同时出现：

1. A 修正在 CLT 有实际正收益，且 CLT 上 A 的改善大于 B 的改善；
2. 其拓扑特异收益高于 B 修正，即 `K_A > K_B`；
3. 不是靠显著损坏 Dir 或头中部类别制造正差分；
4. 幅度控制后方向保持一致。

CE=0 分支下重复相同统计作为交互检查，不只挑选更有利的损失条件。

### 5.2 类别级角色的主对比

在 sample 聚合下，对每个活动因子分别计算：

`C_factor(topology) = Tail_LA − Tail_CE`。

主比较为同拓扑、同锚点内的 `C_B − C_A`，并同时显示全部类别组和 Overall。两种拓扑分别报告，不用一个平均数掩盖相反方向。

支持 B 类别级主导作用，需要 B 的类别校正收益大于 A，且不是只有降低头类分数而没有提升尾类能力。uniform 条件下重复该比较以检查聚合交互。

本实验直接支撑的是“类别频率校正的适配作用”，不是证明 B 独占所有语义类别知识。若最终论文主张更广泛的类别语义存储，还需独立语义迁移证据，不用本协议偷换概念。

## 6. 排除更新幅度造成的假分工

除了原始候选，增加幅度匹配的候选，并同时报告结果。

对每层有效矩阵改动计算：A-only 为 `s B* delta_A`，B-only 为 `s delta_B A*`；在全部 LoRA 层上合并 Frobenius 范数，不能直接比较 delta_A 和 delta_B 的参数范数。

每个锚点的16个候选，以它们最小的非零有效更新范数作为共同长度，使用一个全局缩放系数缩短每个非零更新，不改变方向，不逐层重新配比，不放大任何候选。此尺度只由更新张量确定，不看准确率。

零有效更新原样报告，不用于非零归一化，并说明该比较缺少可比较方向。若归一化尺度极小导致功能变化不可测，如实报告无辨识力，不在测试集上挑选更有利尺度。

代码记录 float32 参数实际落地后的有效范数，而不只记录理论缩放量；匹配误差超过0.1%的起点不标为幅度可比。匹配长度在任何测试评价之前确定。两种聚合严格复用同一份客户端 delta，幅度匹配也只重缩放缓存结果，不再训练。

该控制检验分工是否只是总体步幅差异；它不能自动排除所有逐层幅度和非线性差异。

## 7. 如何给角色定义下结论

主图是一张2×2收益矩阵：行=客户端影响力修正/类别频率修正，列=A/B。

支持完整分工的模式：

| 修正目标 | A | B |
|---|---|---|
| Client-LT 特有的客户端影响力问题 | 收益更大 | 收益更小 |
| 类别频率偏置 | 收益更小 | 收益更大 |

报告所有六个锚点、原始与幅度匹配结果，不事后挑选锚点、层或类别。

- 两行都由 A 占优：A 更有总体适配能力，不是双层分工。
- 两行都由 B 占优：B 更有总体适配能力，不是双层分工。
- 只有第一行符合：支持优先干预 A，不支持完整的 B 类别级主导结论。
- 明确交叉且跨锚点方向一致：支持当前配置下的角色假设，进入完整轨迹验证。
- 结果反向或不一致：不把它重新命名成同一假设的成功。

当前六个锚点来自两条 seed42 轨迹，不能当作六个独立训练种子，也不把100类或多个轮次当独立重复做显著性检验。

### 7.1 代码预先固定的判读，避免事后改变标准

`paired_role_scores.csv` 每个起点、每种幅度模式分别记录：

1. 客户端角色：`U_A(CLT)>0`、`U_A(CLT)>U_B(CLT)`、`K_A>K_B`。
2. 类别角色：两种训练划分分别满足 `C_B>0`、`C_B>C_A`。
3. 客户端 A 修正的代价：两种划分 Overall 不低于 −0.5 pp，Head20/Middle60 不低于 −1 pp；Dir Tail 不低于 −0.1 pp，避免依赖明显压低 Dir 的假改善。
4. 类别 B 修正的代价：两种划分 Overall 不低于 −0.5 pp，Head20/Middle60 不低于 −1 pp，且 Tail 必须实际增加。
5. 匹配有效，以上条件共同成立，才将该起点标为 `joint_pattern=True`。

这些代价界限是固定的 pilot 判读约定，不是统计显著性阈值，也不会控制训练或筛掉原始结果。报告会展示所有真实涨跌；即使跌幅未越界，也仍是代价，不称为“完全无损”。六个起点在 raw、norm_matched 都满足，才标记“完整预设模式出现”。未全部满足时保留实际方向一致点数，按具体失败项解释，不以均值为借口忽略反例。

`class_exposure.csv` 另记录样本量/等权下的名义类别权重 `sum_k q_k n_kc/n_k`，它是数据组成代理，不是测得的真实梯度/功能贡献。等权并不是与类别频率完全正交的干预，所以本实验定义的是“哪类修正在何因子上更有效”的功能角色，而不是互斥存储角色。

## 8. 从角色分工到尾类保持机制

短分支不能代替后期遗忘验证。以下只保留为第二阶段候选设计，本次不实现、不自动启动；是否进行由第一阶段结果决定。若交叉分工有信号，再考虑在同版本 S 上完成四个新增100轮运行：

| 变体 | A 聚合 | B 聚合 | 划分 |
|---|---|---|---|
| A-client-correction | 等权 | 样本量 | CLT、Dir 各一组 |
| B-client-correction | 样本量 | 等权 | CLT、Dir 各一组 |

现有同版本 S 为 sample/sample 基线；不修改 LA=1、B三epoch、A一epoch、A前90轮刷新或其他训练细节。新增优化步骤不增加，仅改变对应因子的聚合权重。

主指标：Tail末20轮、峰到最终回落、预定持续退化时点、固定窗口下降速度，以及 Overall/Head/Middle。沿用已有 W/H/D/R 离线归因，区分支持者维护改善和负向作用改变。

如果仅 A 修正显著减轻 CLT 后期遗忘，同时保持其余群体与 Dir 表现，才能把短分支角色结果连接到长期保持机制。等权结果只是机制诊断；最终功能感知 A 聚合需后续独立设计，不用等权本身宣称新方法。

## 9. 与已有代码、结果和文献的关系

- `utils/cliplora_la_control.py::LAControlRuntime.train_phase`：共同起点、冻结因子、逐客户端更新。
- `trainers/cliplora.py::cliplora_optimizer_step`：在 A/B 单因子分支中显式选择 CE/LA。
- `utils/cliplora_bridge_audit.py::BridgeAudit`：参考事件状态与客户端 delta 格式。
- `utils/cliplora_a_refresh.py`：因子选择、增量聚合、RNG隔离与有效更新诊断。
- `tools/eri_closure/`：原固定 probe 和功能归因。

服务器必须保留原模型、数据、完整 checkpoint/event state。轻量结果包不足以执行新分支。新入口独立于旧 `--stage analyze`，不改动历史训练代码。

## 10. 启动命令

在仓库根目录、原 `clientlt` 环境执行，全部前台运行，报错直接显示。默认数据在 `DATA`，默认读取：

```text
output/cifar100_LT/la_control/seed42/client-longtail/s/tau1_a1_protocol42/
output/cifar100_LT/la_control/seed42/noniid-labeldir-fine/s/tau1_a1_protocol42_beta0.5/
```

一个节点、一张 GPU，顺序完成全部48个分支：

```bash
python -u scripts/run_cliplora_factor_roles.py
```

如果两个节点各一张卡，可以分别跑（每条24个分支）：

```bash
python -u scripts/run_cliplora_factor_roles.py --anchor-origin client-longtail
```

```bash
python -u scripts/run_cliplora_factor_roles.py --anchor-origin noniid-labeldir-fine
```

**`--anchor-origin` 只选择模型从哪条 S 轨迹取出，不限制短分支的训练划分；每条命令仍会对每个模型同时测试 CLT/普通 Dir 两种分配。** 不要误以为两个命令是在拿不同起点直接比较 A/B。

还可以按起点轮次拆任务，例如：

```bash
python -u scripts/run_cliplora_factor_roles.py --anchor-origin client-longtail --rounds 20 50
python -u scripts/run_cliplora_factor_roles.py --anchor-origin client-longtail --rounds 80
python -u scripts/run_cliplora_factor_roles.py --anchor-origin noniid-labeldir-fine --rounds 20 50
python -u scripts/run_cliplora_factor_roles.py --anchor-origin noniid-labeldir-fine --rounds 80
```

四条可放在四个节点分别执行；同一输出目录不要并发执行重叠起点。单节点指定物理卡可加 `CUDA_VISIBLE_DEVICES=4`；申请到的节点只有一张可见卡则无需指定物理编号。

路径不同可附加 `--data-root /path/to/DATA --clt-run /path/to/clt_s --dirichlet-run /path/to/ordinary_dir_s`。默认 `--num-workers 0`，也可在所有任务统一设为4。每个起点的配置写入 `request.json`；重启沿用同样参数，完成的起点、单 epoch 分支和评价均跳过，只有未完成的工作继续。一个分支中途断掉时，该分支从头重做，不重复已完整保存的分支。

不同节点若没有共享文件系统，先合并各自的 `output/cifar100_LT/factor_roles/`（不同起点子目录），然后汇总：

```bash
python -u scripts/run_cliplora_factor_roles.py --stage summary
```

汇总是纯读结果的 CPU 工作，不需要加载 CLIP 或训练数据。少于六个起点时可先汇总，但报告会列出缺项。

主要结果在 `output/cifar100_LT/factor_roles/analysis/`：

- `report.md`：主对比、全部起点、真实收益与代价、预设判读。
- `role_matrix.png`：原始/等幅两张2×2收益矩阵；第一行是 K、第二行是 CLT 的 C。
- `paired_interactions.png`：三个主交互在所有起点上的变化，含 Dir 类别修正。
- `client_correction_costs.png`：两种划分的 Overall/Head/Middle/Tail 实际增减。
- `candidates.csv`、`intervention_effects.csv`、`paired_role_scores.csv`、`per_class.csv`、`budget.csv`、`class_exposure.csv`。
- `anchor_metadata.json`：类别组、类别先验、两种划分的客户端样本量、来源状态与匹配长度，随分析包一起带回。
- `evidence_status.json`：完成状态与预设模式出现情况，不是显著性检验。

打包轻量分析文件：

```bash
tar -czf factor_roles_analysis.tar.gz -C output/cifar100_LT/factor_roles analysis
```

`updates/*.pt` 保留在服务器供重新聚合使用，不必为常规表图分析传回。已有 S 和其他实验文件保持不变。
