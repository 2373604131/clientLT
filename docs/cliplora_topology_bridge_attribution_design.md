# 新框架下的 Client-LT 桥接与分阶段归因实验

状态：已实现新入口、分阶段采集、离线归因和四格汇总；仅做静态检查，未运行训练。日期：2026-09-14。启动见 [四节点运行说明](cliplora_topology_bridge_running.md)。

本设计对应用户提供的 2×2 桥接思路，以仓库现有 C1/C2 为唯一方法变量，不加入 C3、C4、access 调权、memory、投影、动态 gate 或新的训练损失。后文保留原设计规格；实际入口与存储细节以运行说明为准。实现采用各格独立的确定性 protocol 副本；normal_B 还保存原始客户端 B 状态以核对 float32 聚合，存储预算相应增加。

## 1. 实验问题与可以支持的结论

核心问题：在 rank4、固定 A 的日常 B-only 框架内，只改变 client–class coupling，尾类损失有多大？把同预算的额外 B 训练替换为普通 CE 的 A 刷新后，这个拓扑差异是否扩大？

分两层回答：

1. 结果层：topology × refresh strategy 的交互是否存在。
2. 机制层：这个交互伴随正常 B 阶段、额外刷新阶段中哪些类别功能贡献变化。

注意推断边界：

- C1/C2 对照是“额外更新哪个因子”的完整策略对照，不是有效步幅已匹配的纯方向实验。已有结果中两者有效更新范数不同，本桥接不同时调学习率。
- Matched Dirichlet 改变的是整个固定边际 joint coupling，包含支持集合、类别共现和样本归属等变化；不能仅凭四格结果把原因唯一指定为 access。
- 若 Dir-C2 稳定而 CLT-C2 退化，可以支持此配置下拓扑调节了 A-refresh 策略的效果，不能概括为“允许 A 学习本身永远没有问题”。
- 若两种拓扑同样退化，只能说没有检测到强 coupling 交互；还可能是普通优化不稳定或步幅过大，不能直接证明唯一原因是一般全局长尾。
- 已看过 CLT seed42 的结果，本轮是前瞻固定缺失对照的探索性桥接，不是完全未看结果的首次预注册验证。

## 2. 四格实验和运行数量

| 方法 | Client-LT | 固定边际 matched Dirichlet |
|---|---|---|
| C1：Fixed-A + extra-B CE | 已有准确率和阶段指标；缺向量归因 dump | 新跑 Dir-C1 |
| C2：Fixed-A + periodic A CE refresh | 已有准确率和阶段指标；缺向量归因 dump | 新跑 Dir-C2 |

两种方法日常都只训练 B；每 10 轮的额外阶段，C1 更新 B、C2 更新 A。这里的“方案 A”是框架名称，与 LoRA 因子 A 区分。

建议按两阶段安排，不先扩大方法网格：

- 阶段 I：新增 Dir-C1、Dir-C2 两次 100 轮训练；从这两次起就保存分阶段向量 dump。结合历史 CLT-C1/C2，计算完整性能四格、coupling gap 和刷新即时交互。
- 阶段 II：若要完成用户要求的四格 W/H/D/R 与 access 分解，补跑带相同 dump 的 CLT-C1、CLT-C2。不能用旧 gap 标量代替缺失的客户端更新向量。总计四次新训练，但只有两种方法、两个拓扑。
- 若现在就要一次得到完整归因，直接用同一版只增日志的实现跑四格；历史 CLT 结果作复现参考，新四格作归因主表，不按新旧结果谁好挑选。

历史 C0 仅作旁边的固定 A 参考；Dir-C0 是可选后续，不参与本次核心 DiD。没有必要重跑 C3 或 CAPT。

历史输入位置：

`output/a_refresh_c1_c2_c3_analysis/output/cifar100_LT/a_refresh_pilot/seed42/{c1,c2}`。

现有材料不含逐轮各客户端 LoRA 更新。即使服务器仍有 `a_refresh_last.pth.tar`，它也只覆盖最终全局 A/B，不能恢复第 10–90 轮的客户端更新。

## 3. 冻结训练协议

所有设置以历史 `command.json`、`resolved_config.yaml` 和实际客户端调度为基准，而非重新选择一套“更合理”的配置。

| 项目 | 固定值 |
|---|---|
| 数据与 backbone | CIFAR-100-LT；CLIP ViT-B/16 |
| 全局长尾 | exp，imb_factor=0.01；相同 10,847 张训练图像及原始 ID |
| 拓扑 | client-longtail / matched-dirichlet |
| Client-LT 参数 | specialization_lambda=0.75；intra_group_alpha=0.5；head_leakage_scale=3.0；client 比例 0.9/0.1；class 比例 0.8/0.2 |
| matched Dirichlet | beta=0.5，容量约束实现；不是普通 Dirichlet |
| 模型/聚合 | model=fedavg；trainer=ClipLora；按样本数聚合 |
| LoRA | vision top3，Q/V；rank=4；alpha=1；scaling=1/sqrt(4)=0.5；dropout=0 |
| 精度 | FP32 |
| 客户端 | 30；frac=1.0；每轮全部参与，顺序也相同 |
| 训练 | 100 rounds；正常阶段每客户端 3 epochs |
| batch / workers | train32；test64；workers8 |
| 正常优化器 | 每客户端重建 SGD；lr=0.001；momentum=0.9；weight decay=0.0005；constant LR |
| 刷新日程 | 10、20、30、40、50、60、70、80、90；第 100 轮不刷新 |
| 刷新优化器 | 每客户端重建 SGD；1 epoch；lr=0.001；momentum=0.9；weight decay=0 |
| 服务器权重 | q_k=n_k/10847；刷新时不重新分配权重 |
| 种子 | seed=42；split_seed=42；common_init_seed=42；schedule_seed=42 |
| 其他方法 | v2/static/progressive、SCA、PFRF、selective_sync、E1、Stage3、ExperimentD 全部关闭 |
| 常规评估 | 初始化和每一轮同一个完整官方测试集；按固定 80/20 类集合汇总 |

预算每格均为：

- 正常：`100 × 3 × Σ_k ceil(n_k/32) = 105600 steps`。
- 额外：`9 × 1 × 352 = 3168 steps`。
- 总计：108768 steps；正常图像呈现 3,254,100 次，额外 97,623 次。
- 沿用当前 C1/C2 相同的三次 private gap probe 流程，累计 292,869 次图像呈现；不因它们不用 gap 就只给新组删除这部分流程。

训练更新参数 A 或 B 各 18,432 个。相同步数和参数数不等于相同 FLOPs、墙钟时间或有效改动；均如实记录。

本桥接保持 refresh_lr=0.001，不与前一轮建议的小步幅排查混在一起。小步幅是之后独立的敏感性实验。

## 4. matched Dirichlet 的实现与必需的匹配审计

现有 `utils/datasplit.py::partition_fixed_marginal_dirichlet` 可以复用。

该分支先用相同 split seed 构造 Client-LT reference，取得每个具体 client ID 的容量 n_k；再用每类 Dirichlet 偏好及剩余容量约束重新分配所有样本。每个样本恰好分配一次，因此同时保持 n_k 和 n_c。

训练耦合 RNG 为 `RandomState(split_seed+100003)`；测试分区另用 `+100004`。两组模型都使用全局测试集作主评估，不用变化后的本地测试集拼成另一种指标。

必须保存并比较：

1. 所有类的总训练样本数向量 n_c：跨拓扑完全一致。
2. 每个 client ID 的 n_k：跨拓扑完全一致，不只是排序后的直方图一致。
3. 全体原始训练样本 ID 的集合：完全一致，无重复、无遗漏。
4. 同一拓扑 C1/C2 的 client→sample-ID 映射和有序数据列表：一致。
5. 跨拓扑的 client×class 矩阵应不同；客户端 membership hash 应允许不同，不能错误地要求所有指纹相等。
6. 每轮实际客户端有序列表、n_k 和 q_k：四格一致；不是只对排序后的客户端集合做比较。
7. 初始化全部 A/B hash、CLIP backbone 权重 hash、配置和测试标签顺序：一致。
8. 输出每类支持客户端数、所有权有效客户端数和 access，确认操作确实改变了关注的结构。不要在看到准确率后更换 beta 或挑一个更有利的划分。

`client_split_fingerprint.json` 主要是按客户端指纹；建议额外保存 `partition_manifest.csv`，至少含 raw_sample_id、class_id、client_id、local_position，以便跨拓扑验证全局样本集合。不能仅以相同类别计数替代样本身份匹配。

随机流约定：沿用现有训练路径、worker 数、显式调度和刷新 RNG。新增 dump 仅复制张量/写文件；新增评估只在离线执行。不同拓扑对应的输入图片不同，这是实验操作本身，不能要求同一 client/batch 含相同图像。若需要改成新的每 client/round 显式 normal RNG 方案或强制确定性内核，四格必须一起重跑，不能只改新 Dir 两格。

同拓扑 C1/C2 的第 1–9 轮应重合；跨拓扑从第一轮起可以不同。现有 common-init 保证初始化一致，不保证不同 GPU 训练后逐比特一致；保存环境版本和数值误差，不承诺后者。

## 5. 主分析：coupling gap 和交互

设 `Tbar(P,m)` 为固定 Tail 集合第 81–100 轮准确率均值，P∈{CLT,Dir}，m∈{C1,C2}。

`G_B = Tbar(Dir,C1) − Tbar(CLT,C1)`。

`G_A = Tbar(Dir,C2) − Tbar(CLT,C2)`。

`ΔG = G_A − G_B`

`   = [Tbar(Dir,C2)−Tbar(Dir,C1)] − [Tbar(CLT,C2)−Tbar(CLT,C1)]`。

解释：正 ΔG 表示从 extra-B 切换到 A-refresh 后，Dir 相对 CLT 的优势扩大。G_A 是“C2 整条轨迹的拓扑差”，不是只在 A 阶段发生的损失。

同样计算 Overall、Non-tail 的 G_B/G_A/ΔG，避免只保尾类却忽略整体学习。每个输出都记录方向约定 `Dir−CLT`，避免与旧分析 `CLT−Dir` 混淆。

主要表：四格 last20 Overall/Non-tail/Tail、G_B/G_A/ΔG，以及各拓扑内部 C2−C1。

补充表与曲线：

- 第 100 轮；末 10 轮均值；逐轮 gap/ΔG。
- 相对初始化的 final 和 last20 净收益。
- `Tail best-to-last20 drop=max_{r=1..100} Tail_r−mean_{r=81..100} Tail_r`。
- 测试峰值只作描述，不给每格选不同的“最佳轮次”代入 DiD。

历史 CLT last20 锚点：C1=(67.6975,67.68625,67.7425)，C2=(65.89,67.918125,57.7775)，顺序为 Overall/Non-tail/Tail。若采用全新四格归因运行，用新结果统一算主表；历史锚点只用于复现核对。

复用历史结果做阶段 I 时，`ΔG=[Tbar(Dir,C2)−Tbar(Dir,C1)]+9.965`。也就是说，两次新增运行首先回答：Dir 下从 extra-B 切换到 A-refresh，是否也会损失接近 10 个 Tail 百分点？若损失类似，不能把已有 CLT 退化认定为强 coupling 特异效应。

单 seed 为探索性证据。建议报告原始百分点和完整曲线，不做轮间独立样本 t 检验。若需要筛查标签，可前瞻固定 |gap|≤0.5 pp 为“数值较小”、gap≥2 pp 为“较大”，中间区间保留不确定；这不是统计等价/显著性界限，也不能为追求某个故事事后修改。

## 6. 分阶段定位：正常 B 与额外刷新

### 6.1 每个阶段的状态和更新

第 t 轮：

`M_pre=(A_t,B_t)`。

所有客户端从 M_pre 开始，正常训练 B，上传 ΔB^normal_k。

`M_bar=(A_t, B_t+Σ_k q_k ΔB^normal_k)`。

若 t∈{10,20,...,90}：

- C1：客户端从相同 M_bar 训练 B，得到 ΔB^extra_k；聚合后 `M_after=(A_t,B_bar+Σ q ΔB^extra_k)`。
- C2：客户端从相同 M_bar 训练 A，得到 ΔA^refresh_k；聚合后 `M_after=(A_t+Σ q ΔA^refresh_k,B_bar)`。

否则 `M_after=M_bar`。下一轮的 M_pre 即本轮 M_after。C2 刷新完 A 后，下一轮所有客户端加载最新 A，不恢复 A_initial。

### 6.2 不依赖新向量 dump 的阶段准确率分析

每次额外更新分别计算：

`δT_refresh(P,m,t)=T(M_after)−T(M_bar)`。

刷新即时交互：

`I_t=[δT_refresh(Dir,C2,t)−δT_refresh(CLT,C2,t)]`

`   −[δT_refresh(Dir,C1,t)−δT_refresh(CLT,C1,t)]`。

保留 C1 额外 B 阶段这个对照，不只比较 Dir-C2 与 CLT-C2。

后续适应：

- next-round：`T(M_after at t+1)−T(M_after at t)`；这里 t+1 没有额外刷新。
- following interval：`T(M_bar at t+10)−T(M_after at t)`；最后一次用第 100 轮最终结果作为端点。
- 正常阶段变化为 `T(M_bar)−T(M_pre)`，包含本地训练与随后聚合的合成结果，不宣称又把这二者的因果作用拆开。

由所有 9 次刷新阶段和普通逐轮指标，可以精确分解最终 DiD：最终 ΔG 减初始化 ΔG = 即时刷新交互之和 + 其余正常阶段交互之和。这是时间记账，不是独立因果分解；后续 B 已受先前 A 改动影响。

四格轨迹在后期不同，所以某轮 δ 的跨拓扑比较也包括历史路径差异。若将来要检验“同一共享状态下，拓扑对一次 A 更新的作用”，需要额外的共同 checkpoint 短分支重训，而不是把不同模型上的 δ 当成同一起点随机实验。本轮先不增加这个分支。

## 7. 真正的功能归因：W、H、D、R 与 access 分解

### 7.1 统一独立诊断 probe

复用 `tools/eri_closure/protocol.py::build_protocol` 的思路：从原 CIFAR-100 train split 中，不属于固定 LT 训练池的图像里，每个 Tail 类固定取 10 张，总计 200 张；固定 probe_seed=20260904，保存 manifest/hash，四格共用。

先验证 probe 与实际 LT 原始样本 ID 的交集为零。该 probe 是研究者离线诊断数据，不加入本地训练、不传给聚合算法、不用于选择学习率、划分或刷新时点，也不是服务器部署所需数据。

这和 C1/C2 原有的私有 training-set gap proxy 是两套东西，命名和输出目录必须分开。这里不会从只有几张尾类样本的客户端再扣训练数据，也不改变原来的 n_c/n_k。

功能指标沿用 ERI：

`F_c(M)=mean_{x∈probe_c}[z_c(x)−logsumexp_{j≠c} z_j(x)]`。

F 是可微的真类 log-odds，不是准确率，也不是 C3 的负 CE。新旧 W 对比需要相同功能定义和 probe；全局测试集只报告准确率结果，不能用于计算这些梯度。

默认只对 20 个 Tail 类做昂贵归因；Non-tail/Overall 的学习与保持由现有完整测试指标报告，不额外新增不匹配的数据抽样。

### 7.2 按实际阶段的更新做路径积分

对一个阶段 p，令 θ0 是该阶段起点的活动因子向量，u_k 是客户端实际训练得到的该因子增量，d=Σq_k u_k。其他全部模型参数固定为该阶段真实起点。

`e_{k,c,p}=q_k∫_0^1 <∇_θ F_c(θ0+s d),u_k> ds`。

这里 e 已包含服务器权重 q，后面求和不要再次乘 q。主分析使用 8 点 Gauss–Legendre；同时保留起点一阶近似作数值诊断，不用不闭合的一阶近似代替主归因。

理论闭合：`Σ_k e_{k,c,p}=F_c(M_after_phase)−F_c(M_before_phase)`。

normal_B、extra_B、refresh_A 必须是三种不同 phase，不把整个“先 B 后 A”的两段轨迹拉成一条联合 AB 直线。

### 7.3 支持集合与四种有符号贡献

统一使用 `S_c={k:n_kc>0}`。这是 ERI 代码定义；不是 CAPT 的本地占比 >0.1 的严格 supporter 阈值。

`W=Σ_{k∈S_c} max(e_k,0)`：supporter 正向贡献。

`H=Σ_{k∈S_c} max(−e_k,0)`：supporter 负向贡献。

`D=Σ_{k∉S_c} max(e_k,0)`：无该类样本客户端的正向 donor 贡献。

`R=Σ_{k∉S_c} max(−e_k,0)`：无该类样本客户端的负向 rewrite 贡献。

必须同时记录 H，使 `ΔF=W+D−H−R` 闭合；不能把 supporter 自己造成的负向更新也归入 R。

保留 `R/(W+D+epsilon)` 作为辅助 ERI，但主表优先列原始 W/H/D/R 和 ΔF；当正向预算接近零时明确标注，避免只展示很大的比率。不要在看到数据后添加新的干预目标。

### 7.4 W=access × success × strength 的精确定义

为避免与 LoRA A/B 混淆，本文用数学花体 access 和 Pplus 表示质量：

`access_c = Σ_{k∈S_c}q_k`。

`Pplus_{c,p}=Σ_{k∈S_c,e_{k,c,p}>0}q_k`。

`rho_{c,p}=Pplus/access`。

`mu_{c,p}=W/Pplus`。

因此 `W=access × rho × mu`（分母非零时）。rho 是按服务器质量加权的正向写入比例，不是正向客户端的简单人数比例；mu 是正向贡献按该质量归一化后的强度。

当 Pplus=0 且 access>0：W=0、rho=0、mu=NA；不能伪造 mu=0 来声称测量到了“强度很弱”。当 access=0：rho、mu 均为 NA。

尾类/阶段汇总必须先对 W、access、Pplus 使用同一权重求和或均值，再取比率。不能简单平均各类 rho、mu 后相乘，声称它等于平均 W。

极重要的结构事实：在本轮 frac=1、固定 q、固定划分下，access 不随时间变化，同一拓扑 C1/C2 的 access 也相同。A 刷新不改变支持集合。这里讨论的是“跨拓扑 access 更低”，以及“同一低 access 条件下，rho/mu 是否因训练状态改变而恶化”，不是 A 更新把 access 降低。

所有类的 supporter 每轮都参与，不能把全参与下的弱 access 描述成“supporter 经常缺席”。少数/小权重 supporter 的影响稀释，与抽样缺席不是同一机制。

### 7.5 聚合正确性与归因解释

每阶段只有一个因子活动：

- normal/extra B：`Σq(B_k A)=(Σq B_k)A`。
- A refresh：`Σq(B_bar A_k)=B_bar(Σq A_k)`。

因此这里可以归因到实际聚合路径，无需把因子乘积错配作为默认解释。路径积分给出的是沿选定路径的贡献分配，不等同于“移除客户端再训练”的反事实因果效应，也不是 Shapley 唯一责任。

## 8. 采集与离线计算预算

保存更新向量远比重新训练便宜，建议新运行采集全部事件：

- 100 个 normal_B dump。
- C1 9 个 extra_B / C2 9 个 refresh_A dump。
- 总计每格 109 个事件；不因为一开始只分析少量轮次就永久丢掉其余信息。

rank4 活动因子 18,432 个 FP32 参数，30 客户端的单阶段 delta 原始载荷约 2.11 MiB；保存两份完整全局 A/B 后，每事件约 2.39 MiB，不含序列化/元数据。109 个事件约 260 MiB/格，四格约 1 GiB。不要在每个客户端 dump 中复制整个 CLIP backbone；实际磁盘开销以文件为准。

第一批离线积分只计算：

- normal_B：10、20、40、60、80、90、100。
- 所有实际刷新：10、20、30、40、50、60、70、80、90。
- 共 16 个事件/格；200 张固定 Tail probe、8 个积分节点。每事件需 20×8 次类别梯度评价，不需要为每个客户端重新求一次梯度。

进一步定位后续适应时，使用已经保存的 normal_B：11、21、31、41、51、61、71、81、91，并补 30、50、70 的正常阶段；无需重训。

汇总规则必须与采样方式一致：

- 9 次刷新全部观测，刷新 W/H/D/R 可以直接逐事件求和，代表实际累计刷新贡献。
- 稀疏 normal_B 审计只报固定轮次均值和逐事件值，不冒充 100 轮真实累计量。
- 不直接复用旧 ERI summary 的梯形积分累计逻辑；A 刷新是离散事件，不能把第 10 与第 20 轮之间都插值成在持续刷新。
- 若需要完整正常阶段累计贡献，就对已保存的全部 100 个 normal_B 事件离线积分，随后直接求和。不要用少数有利轮次代替全程。

## 9. 逐类结构关系和更新幅度

每类记录并严格区分：

- n_c：全局该类样本数。
- N_c=|S_c|：支持客户端人数。
- `N_eff_ownership,c = n_c²/Σ_k n_kc²`：按该类样本所有权计算的有效客户端数，对应现有 class_topology 的 effective_client_number。
- access_c：supporter 的服务器权重总和。
- `N_eff_access,c=access_c²/Σ_{k∈S_c}q_k²`：支持权重有效客户端数，对应 ERI 的 support_effective_clients。不能与 ownership N_eff 混用。

预先固定两项探索性 Spearman：

1. 各拓扑 C2 内，`access_c` 与该类九次 A 刷新即时准确率变化的均值。若低 access 更容易下降，通常应为正相关。
2. 跨拓扑按同一 class ID 配对，`access_Dir−access_CLT` 与该类 C2−C1 效果的拓扑 DiD。

散点中显示 n_c，避免把不同类别难度或频率完全归因给 access。20 类、单 seed 的相关性只作诊断，不把 20 类×9 次刷新当 180 个独立样本；常量向量相关系数记 NA。类重采样区间也不能替代跨 seed 不确定性。

每个刷新继续保留现有 effective_delta_norm、delta_A/B_norm、new_direction_fraction、A 相对初始的子空间变化。正常 B 事件可从 dump 离线计算有效范数。

若 Dir 与 CLT 的 A 刷新幅度明显不同，这属于拓扑作用于本地优化后的中间结果，也提示机制解释需区分方向与幅度。保持本桥接训练原样，只把共同起点、有效范数匹配的短分支比较列为后续，不偷偷给其中一格裁剪/改 LR。

## 10. 仓库复用点与必须适配的代码

### 可直接复用

| 代码 | 用途 |
|---|---|
| `utils/datasplit.py::partition_fixed_marginal_dirichlet` | 固定 n_k/n_c 的 topology control |
| `scripts/run_cliplora_a_refresh.py::build_command` | 当前 C1/C2 全部训练参数基线 |
| `utils/cliplora_a_refresh.py::ARefreshRuntime` | 9 次额外阶段、客户端共同起点、优化器、冻结和预算 |
| 同文件 `update_diagnostics`、`isolated_rng` | 实际有效更新及 RNG 隔离工具 |
| `tools/eri_closure/attribution.py` | 路径积分、signed_budgets、W/access/rho/mu 基础数学函数 |
| `tools/eri_closure/protocol.py` | LT 池外 train-only probe manifest |
| `trainers/cliplora.py::build_cliplora_model` | 重建同架构模型，避免用旧实验默认 rank/精度 |
| 现有 round_metrics、a_refresh_stage_metrics/per_class | 结果与阶段准确率 |

### 不能直接叠加旧 ERI 开关的原因

1. `scripts/run_cliplora_a_refresh.py` 当前硬编码 `--partition client-longtail`，没有暴露 partition 参数。
2. `federated_main.py` 当前 ERI dump 在整轮刷新之后保存，却使用正常阶段的 local_weights 和轮初 pre_global_weights。C1 的终点多了一次 B 更新，已不是这些 normal local states 的聚合结果；C2 则可能因只检查 B 而遗漏 A 刷新。
3. 旧 `save_eri_round_dump` 只保存传入的 trainable_keys。B-only 下冻结的 A 不会被保存；但 C2 的当前 A 已不是初始化 A，离线重建不能使用默认 A_initial。
4. A-only 阶段同样必须保存真实固定 B_bar。只有 A 增量无法重建函数端点。
5. `TrainOnlyFunctionalEvaluator.gradient` 没有根据阶段主动打开 A 的 requires_grad。用 freeze-A 配置重建后直接对 A 求梯度不可用；allow_unused=True 不能解决参数 requires_grad=False。
6. 旧 `eri_audit_enable` 还会跳过初始化测试分支；直接给新 Dir 组打开会改变指标/执行路径，与历史 C1/C2 不再严格同协议。
7. 旧 summary 没有 phase 主键，采用不规则轮次梯形积分；不能直接混合 normal_B 和 refresh_A。

旧 `analyze_run` 还要求所有 dump 的活动参数 keys 一致；新分析需按 phase 分组或逐事件显式切换 spec，不能把 A-only/B-only 事件不加区分地交给旧入口。

因此，本设计采用独立的、只采集不改变训练的 bridge audit 接点，复用数学函数，不把旧 ERI flag 直接接到 C1/C2。

### 拟实施的最小文件边界

| 文件 | 拟改动 |
|---|---|
| `scripts/run_cliplora_a_refresh.py` | 增加向后兼容的 partition/beta 输入，默认仍为旧 Client-LT；不改其他默认值 |
| `scripts/run_cliplora_topology_bridge.py`（新） | 四格路径、共同 protocol/schedule、配置导出和运行组织；不是新算法 |
| `federated_main.py` | 在正常 B 聚合完成、任何额外刷新之前采集 normal_B 事件 |
| `utils/cliplora_a_refresh.py` | 每个刷新客户端产生 delta 时保留事件数据，在 aggregate_refresh_deltas 后采集 extra_B/refresh_A |
| `utils/cliplora_bridge_audit.py`（新） | 保存 phase-aware dump、完整 LoRA anchor、活动参数、端点和元数据；只读训练张量 |
| `scripts/analyze_cliplora_topology_bridge.py`（新） | 模型恢复、阶段活动参数梯度、ERI primitives 调用、四格汇总 |

建议每份事件包含：

```text
seed, topology, method, communication_round, phase
active_factor, active_keys, flatten_spec
anchor_lora_state: 完整当前 A/B，不只 requires_grad=True 的部分
actual_after_lora_state: 完整实际阶段终点 A/B
selected_client_ids: 有序
local_factor_deltas: 与起点相减的真实向量
server_weights, client_sample_counts, client_class_counts
optimizer_steps, parameter/sample budgets
backbone/config/initialization/schedule/probe hashes
aggregation arithmetic: normal-state-mean 或 refresh-base-plus-delta
```

离线重建顺序：已验证 backbone → 完整该阶段 LoRA anchor → 固定非活动参数 → 显式启用活动参数梯度 → 路径积分。每个阶段独立恢复，避免沿用上一事件错误的 A/B。

normal 用实际“有序参数平均”规则重构；refresh 用“middle+有序 delta 平均”规则重构。先核对实际矩阵端点，再核对功能闭合。归因代码不能只在一个错误重建的模型上自洽就算通过。

数值验收建议前瞻固定：

- 矩阵聚合重构 max_abs_error≤1e-5，并同时保存相对误差。
- 功能闭合每类 `|Σe−ΔF|≤1e-4+0.01Σ|e|`，保存全部残差和违反数量，而非只报总体均值。
- 同时比较路径端点与真实 trained endpoint 的 F，不能只比较路径积分自身两端。
- 若 8 点积分不够，可把 [0,1] 分成两段各用 8 点，再离线复算；现有基础工具只支持 1/2/4/8 点，不能假定已有 16 点参数。
- 不通过的事件标为 invalid，不能静默纳入机制结论或用增加 epsilon 掩盖。

以上是实验数据有效性记录，不是在训练流程里添加防御性回滚/裁剪/门控。实现后不启动用户未要求的冒烟训练。

## 11. 输出布局与交付表

拟新增输出根目录，绝不覆盖历史结果：

```text
output/cifar100_LT/a_refresh_topology_bridge/
  protocol/
    bridge_protocol.json
    probe_manifest.csv
    full_schedule_seed42.json
  seed42/
    client-longtail/c1/
    client-longtail/c2/
    matched-dirichlet/c1/
    matched-dirichlet/c2/
      command.json, resolved_config.yaml
      partition_manifest.csv, partition_summary.json, class_topology.csv
      client_class_counts.csv, client_split_fingerprint.json
      round_metrics.csv, per_class_accuracy_epoch_*.csv
      a_refresh_config/progress/summary/budget/stage_*.{json,csv}
      bridge_dumps/round_010/normal_B/...
      bridge_dumps/round_010/refresh_A/...
  analysis/
    protocol_audit.json
    factorial_performance.csv
    coupling_gap_by_round.csv
    refresh_event_interactions.csv
    phase_client_effects.csv
    phase_class_budgets.csv
    phase_factorization.csv
    attribution_validity.csv
    class_structure_relation.csv
    report.md
```

主图只需四张：四格 Tail/Non-tail 曲线；G_B/G_A/ΔG 曲线；9 次刷新即时 Tail 交互；按 normal_B/refresh 分面的 W/H/D/R 与 access/rho/mu 对比。逐类关系作为补图。

所有归因 CSV 的联合主键必须包含 seed、topology、method、round、phase、class_id；客户端表再加 client_id。不能只按 round 合并而把同一轮两个阶段覆盖。

## 12. 结果判定与后续路线

| 观察 | 本轮可支持的解释 | 下一步 |
|---|---|---|
| G_B 小，G_A 大，ΔG>0；Dir-C2 的 Tail/Non-tail 共同保持 | coupling 与 A-refresh 策略有强交互 | 查看 refresh_A 的 W/H/D/R、rho/mu，再决定是否干预 A 的贡献机制 |
| G_B 已大，G_A 更大 | B-only 路径已有拓扑敏感性，A-refresh 进一步放大 | 同时检查 normal_B 与 refresh_A，不只保护 A |
| G_B/G_A 都小，两种 C2 都明显退化 | 当前 A-refresh 失败缺乏强拓扑特异性 | 优先步幅/一般适配诊断，不包装为 coupling 新机制 |
| G_B/G_A 都大且相近 | 存在共同拓扑主效应，未见新增 A-refresh 放大 | 先定位两种策略共有的写入环节 |
| G_A<G_B 或某个 gap<0 | A-refresh 可能缩小差距，或 topology 效果与预期相反 | 如实报告，不调整划分追求预期方向 |
| CLT 的 access 低，rho/mu 与 Dir 接近 | 与结构性可进入权重不足相容 | 后续贡献/权重干预才可进一步验证原因 |
| CLT 的 access 低且 refresh_A 中 rho/mu 更低 | 除结构差异外，写入质量也较差 | 检查更新幅度、状态及方向，不只做服务器放大 |
| H 主导负向贡献 | supporter 自身负向更新不可忽略 | 不能把失败全部归因于 class-absent rewriting |

暂时不把归因结果直接写回训练。先完成桥接事实和机制定位，再设计下一版方法。

若扩展确认，建议新 seeds=1、2、3、2026，每个 seed 都跑完整四格，按 seed 计算 DiD；共 16 次确认训练。probe 定义和全部超参数保持不变。区分训练随机性与拓扑抽样随机性是后续更大设计，本次 seed42 不声称覆盖了它们。

结论模板应是：“在固定全局类别数与每个客户端样本量的条件下，观察到/未观察到 coupling 对 periodic A-refresh 策略效果的放大；该现象在阶段归因中表现为……”。不要提前写成“证明少数类无法影响 A”，把机制结论交给实验。
