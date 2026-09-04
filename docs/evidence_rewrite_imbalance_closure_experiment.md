# Evidence–Rewrite Imbalance 最终闭环实验

## 0. 实验要回答的唯一核心问题

本实验不再重复证明以下已经成立的局部事实：supporters 能够写入尾类功能、真实 FedAvg 会压缩这种写入、class-absent updates 同时包含 donors 与 rewriters、共享 LoRA 会长期侵蚀尾类 margin。它只回答三件尚未被直接联合测量的事：

1. 在严格匹配全局类别边际和客户端样本量边际后，Client-LT 是否具有更高的“破坏性 class-absent rewrite / 全部正向功能刷新”比率？
2. 该比率是否能预测类别随后发生的 best-to-final degradation，而不仅是与下降同时出现？
3. 当一个预注册、class-aware、与测试集无关的聚合干预增加正向 evidence access 时，该比率是否下降，且功能保留是否同步改善？

如果三层都通过，Evidence–Rewrite Imbalance（ERI）才从叙事概念变成被直接测量、具有结果关联并得到干预支持的机制。

本实验特别不检验、也不需要证明：

- Client-LT 的绝对 rewrite budget 一定大于 Dirichlet；
- 所有 class-absent clients 都有害；
- no-support event 是退化的必要条件；
- ERI 是尾类下降的唯一原因。

目标结论应当是：

> Client-LT 不一定制造更大的绝对破坏量；它使可进入全局模型的正向功能刷新更稀缺，从而让 class-absent destructive rewriting 更容易相对占优。这个相对不平衡能够预测后期退化，并可被 class-aware access control 定向缓解。

---

## 1. 为什么原始三项定义还需要收紧

用户提出的一阶定义方向正确：

\[
u_{k,c}^{t,(1)}=
\left\langle \nabla_\theta M_c(\theta^t),\Delta_k^t\right\rangle,
\qquad
\Delta_k^t=\theta_k^t-\theta^t.
\]

若 \(M_c\) 越大代表类别功能越好，且 \(\Delta_k^t\) 使用“local-after 减 global-before”的符号，则 \(u>0\) 是正向功能效应，\(u<0\) 是破坏效应。

但正式闭环前要修正四点。

### 1.1 只统计实际参与并实际进入聚合的客户端

令第 \(t\) 轮被选中的客户端集合为 \(S_t\)，实际服务器权重为 \(q_k^t\)，满足

\[
q_k^t\ge 0,
\qquad
\sum_{k\in S_t}q_k^t=1.
\]

该轮类别 \(c\) 的直接证据持有者为

\[
E_c^t=\{k\in S_t:N_{k,c}>0\},
\]

class-absent 集合为 \(A_c^t=S_t\setminus E_c^t\)。未参与客户端不进入任何一项。若某种聚合策略令一个已参与客户端 \(q_k^t=0\)，它的实际服务器影响也应为零。

### 1.2 必须加入 supporter-negative budget

原始 \(W,D,R\) 没有记录“持有类别 \(c\) 的客户端，却对 \(c\) 产生负向功能效应”的情形。为使 signed budget 完整，定义四个象限：

\[
\begin{aligned}
W_c^t &= \sum_{k\in E_c^t}[e_{k,c}^t]_+,
&&\text{direct-evidence positive writing},\\
H_c^t &= \sum_{k\in E_c^t}[-e_{k,c}^t]_+,
&&\text{supporter-side harmful update},\\
D_c^t &= \sum_{k\in A_c^t}[e_{k,c}^t]_+,
&&\text{class-absent donor refresh},\\
R_c^t &= \sum_{k\in A_c^t}[-e_{k,c}^t]_+,
&&\text{class-absent destructive rewrite}.
\end{aligned}
\]

其中一阶版本使用

\[
e_{k,c}^{t,(1)}=q_k^t u_{k,c}^{t,(1)}.
\]

这样可以同时验证：负向预算究竟主要来自广泛的 class-absent rewriting，还是来自 supporters 自身的冲突更新。如果 \(H\) 很大且主导负向预算，则论文必须把机制改写成更一般的 signed-update imbalance，不能把全部负向效应归因于 class-absent rewriting。

### 1.3 一阶内积必须通过功能近似校验

视觉 LoRA 的函数对 LoRA 因子并非严格线性，本地三轮更新也未必足够小。因此，不能默认 \(\langle\nabla M,\Delta\rangle\) 就等于真实功能效应。

正式版本采用沿该轮真实服务器更新路径的 path-integrated attribution：

\[
\delta^t=\sum_{j\in S_t}q_j^t\Delta_j^t,
\]

\[
e_{k,c}^{t,\mathrm{PI}}=
q_k^t\int_0^1
\left\langle
\nabla_\theta M_c(\theta^t+\alpha\delta^t),
\Delta_k^t
\right\rangle d\alpha.
\]

它有一个对闭环非常重要的 completeness 性质：

\[
\sum_{k\in S_t}e_{k,c}^{t,\mathrm{PI}}
=M_c(\theta^t+\delta^t)-M_c(\theta^t).
\]

因此在数值积分误差之外，四项满足

\[
\boxed{
\Delta M_c^t=(W_c^t+D_c^t)-(H_c^t+R_c^t)
}.
\]

实现时用预注册的 8 点 Gauss–Legendre quadrature；同时保存一阶版本。若一阶版本通过第 6 节的校验，可在全轮次使用一阶值、在审计轮次使用 PI 值确认；若不通过，主文只使用 PI attribution。

### 1.4 不能用逐轮 ratio 直接做时间平均

当某一轮正向刷新接近零时，逐轮 ERI 会爆炸。正式统计先累计预算，再做比值，而不是平均每轮比值。

令

\[
P_c^t=W_c^t+D_c^t
\]

表示全部正向功能刷新。对预注册轮次区间 \(I\)，定义

\[
P_c(I)=\sum_{t\in I}\omega_tP_c^t,
\qquad
R_c(I)=\sum_{t\in I}\omega_tR_c^t,
\]

其中每轮均审计时 \(\omega_t=1\)；稀疏审计时 \(\omega_t\) 是相邻审计点之间的预注册时间跨度权重。

核心指标为

\[
\boxed{
\operatorname{CERI}_c(I)=
\frac{R_c(I)}{P_c(I)+\epsilon}
}.
\]

同时报告更稳定、与 CERI 单调等价的 bounded share：

\[
\boxed{
\operatorname{ERIS}_c(I)=
\frac{R_c(I)}{R_c(I)+P_c(I)}\in[0,1].
}
\]

主文用 CERI 解释效应大小，用 ERIS 做主要推断。只要 \(R+P>0\)，ERIS 不需要任意 \(\epsilon\)。对于 \(P=0,R>0\) 的区间，原始 CERI 记为 \(+\infty\)，并单独报告 zero-refresh rate；不得仅靠增大 \(\epsilon\) 隐藏这种情形。

凡作图或回归确实需要 log-CERI 时，统一使用

\[
\operatorname{logCERI}_c=
\log[R_c+\epsilon_0]-\log[P_c+\epsilon_0],
\]

其中 \(\epsilon_0=10^{-6}\max(1,\operatorname{median}_c|M_c(\theta^0)|)\)，只由共同初始模型和 audit probes 决定，并在读取任何结果前冻结。主假设不依赖改变 \(\epsilon_0\) 才成立；应补充 \(10^{-5}\) 与 \(10^{-7}\) 倍率的敏感性结果。

两个必要的稳健性指标为

\[
\operatorname{CERI}^{\mathrm{all-neg}}_c=
\frac{R_c+H_c}{W_c+D_c+\epsilon},
\]

以及

\[
\operatorname{AbsentNegShare}_c=
\frac{R_c}{R_c+H_c+\epsilon}.
\]

前者检验结论是否依赖忽略 supporter harm，后者直接显示 class-absent rewrite 在全部负向预算中的占比。

---

## 2. 类别功能量 (M_c) 如何定义

### 2.1 主指标：平滑的 all-class log-odds margin

对类别 \(c\) 的固定、训练外 audit probe 集 \(\mathcal P_c\)，定义

\[
M_c(\theta)=
\frac{1}{|\mathcal P_c|}
\sum_{(x,c)\in\mathcal P_c}
\left[
z_c(x;\theta)-
\log\sum_{j\ne c}\exp z_j(x;\theta)
\right].
\]

该定义平滑、方向明确，并且不依赖每轮可能变化的 argmax competitor。主分析不使用 accuracy 的梯度，因为 accuracy 不可微且在尾类上过于离散。

### 2.2 Audit probe 与 official test 必须分工

- ERI 的梯度、donor/rewriter 符号、counterfactual 权重选择全部只使用固定 train-side audit probes。
- best-to-final drop、最终 tail accuracy 和最终 test margin 使用 official test，但必须在 ERI、干预规则与所有阈值冻结后才读取。
- 主实验可直接复用 `output/topology_breadth_phase2_seed42/manifests/heldout_tail_train_probes.csv` 的 20 类 × 10 张、明确排除在 federated LT pool 外的训练图像；新增 seed 必须按同一规则冻结并写入 manifest。
- 两种拓扑、两种聚合必须使用完全相同的 \(\mathcal P_c\)。不得为不同拓扑重新挑选“更合适”的 probes。

### 2.3 次指标

为与现有 E1、D1、D4 和 Stage 2-C 对齐，同时报告：

1. 固定 top-10 non-tail semantic-neighbor margin；neighbor 表使用已冻结的 `hard_boundaries.csv`，训练中不得更新；
2. per-class NLL；
3. per-class accuracy，仅用于最终可读性和与原图对齐，不用于梯度归因。

主结论必须先在 all-class log-odds margin 上成立。只在挑选出的 hard neighbors 上成立的结果，应被写成 boundary-specific 结果而非一般 ERI。

---

## 3. 正式实验设计：一个 2 × 2 配对因子实验

### 3.1 固定部分

- Dataset：CIFAR-100-LT，IF = 0.01，Bottom-20（80–99）为预定义尾类。
- Clients：30。
- Backbone：CLIP ViT-B/16。
- Trainable scope：vision-only LoRA，top-3 blocks，q/v，rank 2，alpha 1，FP32。
- Local training：3 local epochs；优化器、batch size、scheduler 与现有正式 ClipLoRA 运行完全一致。
- Rounds：100。
- Participation：主实验使用 full participation。
- Seeds：建议至少 5 个配对 seed，例如 1、2、3、42、2026；同一 seed 下模型初始化、LT sample pool、客户端样本量、调度和客户端本地随机种子全部跨条件配对。

主实验采用 full participation 是有意的：它移除 no-support sampling 这一放大器，使第一层结论直接回答“即使每轮都有支持者，固定边际下的 evidence topology 是否仍产生更高 ERI”。partial participation 已由 D4/Stage 2-C 解释，必要时只放入补充材料。

### 3.2 Topology 因子

两种拓扑必须同时满足：

\[
\sum_kN_{k,c}^{\mathrm{ClientLT}}
=\sum_kN_{k,c}^{\mathrm{Dir}}=n_c,
\]

\[
\sum_cN_{k,c}^{\mathrm{ClientLT}}
=\sum_cN_{k,c}^{\mathrm{Dir}}=n_k.
\]

即每个类别的总样本数和每个客户端的总样本数都逐项相等，只改变耦合矩阵 \(N_{k,c}\) 与 \(p(k\mid c)\)。

因此在 full-participation FedAvg 下，两种拓扑的 \(q_k=n_k/\sum_jn_j\) 也逐客户端完全相同；H1 的差异不能由客户端容量或服务器权重表不同解释。

实现上应复用 `output/topology_breadth_phase2_seed42/manifests/` 的 fixed-row/fixed-column 构造，或 `online_sca_seed42_v2` 中已经验证 row/column margins 相等的 split 生成路径。每个 seed 必须输出：

- row-margin exact equality；
- column-margin exact equality；
- LT sample-ID pool equality；
- matrix inequality；
- split fingerprint 和 SHA256。

现有 `ClipLora_SupportNormalized_2x2_seed42` 只保证相同全局类别边际，其 Client-LT 与 Dirichlet 客户端 row sums 并不逐项相等，因此可用于既有 intervention 佐证，但不应直接承担正式 H1 的 fixed-marginal topology claim。

### 3.3 Aggregation 因子

两个条件为：

1. Ordinary sample-weighted FedAvg；
2. 现有、单一全局模型可执行的 tail-support-normalized aggregation。

FedAvg 权重为

\[
q_k^{\mathrm{FA}}=\frac{n_k}{\sum_jn_j}.
\]

现有 LoRA support-normalized 策略不是为每个目标类训练一个模型，而是把所有尾类的 class-wise supporter distributions 取平均：

\[
q_k^{\mathrm{SN}}=
\frac{1}{|\mathcal C_{\mathrm{tail}}|}
\sum_{c'\in\mathcal C_{\mathrm{tail}}}
\mathbf 1[N_{k,c'}>0]
\frac{n_k}{\sum_j\mathbf 1[N_{j,c'}>0]n_j}.
\]

该权重只读取训练 split 的 support metadata，不读取 audit probe 或 official test。由于支持其他尾类但不支持目标类 \(c\) 的客户端仍可能获得非零权重，SN 下 \(R_c\) 不会被定义性地强制为零，这使 ERI 干预检验不是一个平凡恒等式。

形成四个正式条件：

| Topology | FedAvg | Support-normalized |
|---|---:|---:|
| fixed-marginal Client-LT | 主机制条件 | access intervention |
| fixed-marginal matched Dirichlet | topology control | specificity/safety control |

H1 与 H2 以两个 FedAvg 条件为主；H3 使用完整 2 × 2。

### 3.4 建议审计轮次

若计算和存储允许，保存每一轮的逐客户端 LoRA delta，并离线计算一阶 ERI；在以下预注册轮次计算 PI attribution：

\[
\mathcal T_{\mathrm{audit}}
=\{1,2,3,4,5,10,20,30,40,50,60,70,80,90,100\}.
\]

该集合在形成期更密、维护期每 10 轮一次。若只能保存稀疏轮次，则所有 CERI 均使用预先固定的梯形时间权重，且不得根据结果临时增加“有趣轮次”。

---

## 4. 三层核心假设与具体检验

## 4.1 H1：拓扑差异

主比较只使用普通 FedAvg，并在 fixed \(n_c\)、fixed \(n_k\)、相同 seed、相同初始化与 full participation 下进行。

维护区间预注册为 rounds 11–100；形成区间 rounds 1–10 单独报告，不与维护期混合解释。对每个 seed \(s\) 和尾类 \(c\)，计算

\[
\operatorname{ERIS}^{\mathrm{CLT}}_{s,c}(11{:}100),
\qquad
\operatorname{ERIS}^{\mathrm{Dir}}_{s,c}(11{:}100).
\]

主效应为 seed 内、class 内配对差：

\[
\Delta^{\mathrm{topo}}_{s,c}
=\operatorname{ERIS}^{\mathrm{CLT}}_{s,c}
-\operatorname{ERIS}^{\mathrm{Dir}}_{s,c}.
\]

必须同时画出并报告累计 \(W,H,D,R\)，尤其是：

\[
P^{\mathrm{CLT}} \text{ vs. }P^{\mathrm{Dir}},
\qquad
R^{\mathrm{CLT}} \text{ vs. }R^{\mathrm{Dir}}.
\]

这一步决定正确叙事：允许观察到 \(R^{\mathrm{CLT}}\le R^{\mathrm{Dir}}\)，只要 Client-LT 的 \(P\) 降得更多并使 ERI 上升。论文不得再把“相对失衡”偷换成“绝对 rewrite 更强”。

建议统计：

- seed 为最高层独立 block，class 为 seed 内配对 block；
- 报告 seed-cluster hierarchical bootstrap 95% CI；
- 报告每个 seed 的 20 类 macro mean 配对差；
- 方向性假设预注册后可使用 one-sided paired randomization test；至少 5 个 seed 才能避免最小可达 p 值过粗；
- 同时报告 20 类中 \(\Delta^{\mathrm{topo}}_{s,c}>0\) 的比例，但不得把 100 个 seed × class 点当作完全独立样本。

H1 通过条件：

1. maintenance ERIS 的 Client-LT minus Dirichlet 配对均值大于 0；
2. 预注册 95% cluster-bootstrap CI 下界大于 0；
3. 至少 4/5 seed 的 seed-level macro contrast 同向；
4. 使用 \((R+H)/(W+D)\) 后方向不反转；
5. 报告结果明确显示差异来自 \(P\)、\(R\) 或二者的何种组合。

## 4.2 H2：ERI 与后期下降的结果关联

### 类别级主结果

official test 上每轮计算与 audit 定义一致的 \(M_c^{\mathrm{test}}(t)\)。为降低单轮噪声，先做预注册的 5-round centered moving average；边界处使用可用轮次均值。定义：

\[
\operatorname{BestToFinalDrop}_{s,c}
=\max_{1\le t\le100}\bar M_{s,c}^{\mathrm{test}}(t)
-\frac{1}{5}\sum_{t=96}^{100}M_{s,c}^{\mathrm{test}}(t).
\]

同时报告 accuracy 版本，以连接现有 best-tail 与 final-tail 数字；但 margin 版本是主统计量。

第一个描述性检验是：

\[
\rho_{\mathrm{Spearman}}
\left(
\operatorname{CERI}_{s,c}(11{:}100),
\operatorname{BestToFinalDrop}_{s,c}
\right)>0.
\]

更强的配对检验使用 topology-induced differences：

\[
\Delta\log\operatorname{CERI}_{s,c}
=\log\operatorname{CERI}^{\mathrm{CLT}}_{s,c}
-\log\operatorname{CERI}^{\mathrm{Dir}}_{s,c},
\]

\[
\Delta\operatorname{Drop}_{s,c}
=\operatorname{Drop}^{\mathrm{CLT}}_{s,c}
-\operatorname{Drop}^{\mathrm{Dir}}_{s,c}.
\]

检验 \(\Delta\log\operatorname{CERI}\) 与 \(\Delta\operatorname{Drop}\) 的相关性，可以消去大量固定类别难度、类别样本量和 zero-shot 基线差异。

### 时间先行检验

仅用全程累计 ERI 关联全程下降仍可能被批评为 contemporaneous correlation。因此必须增加一个不重叠的 lagged test。令过去审计区间的 predictor 和下一审计区间的 outcome 分别为

\[
X_{s,c,i}=\operatorname{ERIS}_{s,c}([t_{i-1},t_i]),
\]

\[
Y_{s,c,i}=
-\left[
M_{s,c}^{\mathrm{test}}(t_{i+1})
-M_{s,c}^{\mathrm{test}}(t_i)
\right].
\]

两者时间窗口不重叠：过去的 imbalance 预测下一阶段的 margin loss，而不是用同一轮的功能变化解释自己。

拟合预注册 mixed-effects model：

\[
Y_{s,c,i}=
\beta_0+
\beta_1X_{s,c,i}+
\beta_2M_{s,c}^{\mathrm{test}}(t_i)+
\beta_3\mathrm{Topology}+
b_s+b_c+\varepsilon_{s,c,i}.
\]

其中 seed 和 class 至少作为随机截距或双向 cluster。\(\beta_1>0\) 表示更高的先行 imbalance 预测更大的下一阶段功能损失。

### 证明“ratio 比绝对破坏量更贴切”

并列拟合：

\[
\mathrm{Drop}\sim\log(R+\epsilon),
\]

以及

\[
\mathrm{Drop}\sim
\log(R+\epsilon)+\log(P+\epsilon).
\]

预期第二个模型中 rewrite 系数为正、positive-refresh 系数为负，并在 leave-one-seed-out prediction 上优于只用 \(R\) 的模型。这一步直接回应“Dirichlet 的绝对下降/改写甚至可能更大”的审稿质疑。

H2 通过条件：

1. CERI/ERIS 与 margin best-to-final drop 的相关系数为正，95% 双向 cluster-bootstrap CI 不跨 0；
2. topology-paired difference 相关性同向；
3. lagged model 的 \(\beta_1>0\)，95% CI 不跨 0；
4. 结论在控制起点 margin、class fixed/random effect 后仍成立；
5. accuracy drop 至少方向一致，即使因离散性未达到同等显著性。

## 4.3 H3：干预一致性

### 端到端干预

在每个 topology × seed 内，FedAvg 与 support-normalized 两条轨迹共享：

- 初始化；
- 数据 split；
- client schedule；
- local RNG contract；
- optimizer 和训练预算。

唯一改变的是预注册服务器聚合权重。对 Client-LT 计算

\[
\Delta^{\mathrm{int}}\operatorname{ERIS}_{s,c}
=\operatorname{ERIS}^{\mathrm{SN}}_{s,c}
-\operatorname{ERIS}^{\mathrm{FA}}_{s,c},
\]

\[
\Delta^{\mathrm{int}}\operatorname{Drop}_{s,c}
=\operatorname{Drop}^{\mathrm{SN}}_{s,c}
-\operatorname{Drop}^{\mathrm{FA}}_{s,c}.
\]

预期二者均小于 0，同时 final tail margin/accuracy 上升。

还应报告 difference-in-differences：

\[
\mathrm{DiD}_{\mathrm{ERI}}=
(\mathrm{SN}-\mathrm{FA})_{\mathrm{CLT}}
-(\mathrm{SN}-\mathrm{FA})_{\mathrm{Dir}},
\]

以及相同定义的 \(\mathrm{DiD}_{\mathrm{Drop}}\)。如果机制具有 topology specificity，Client-LT 的 ERI 降幅和 retention 改善应更大；Dirichlet 条件同时承担安全对照，检查是否产生 head 或整体性能损伤。

### 同一状态、同一批 local deltas 的 frozen replay

端到端两条轨迹会逐渐分叉，因此再加入一个更干净的局部因果检验。对每个 FedAvg 审计轮次冻结 \(\theta^t\) 和所有 \(\Delta_k^t\)，构造：

1. 原始 FedAvg 权重；
2. class-aware support-normalized 权重；
3. 20 个预注册的 permutation placebo：只在客户端之间置换 support-normalized 权重，因此权重分布和 \(N_{\mathrm{eff}}\) 完全相同，但 support identity 被破坏。

在相同 local deltas 上重算 ERI，并精确前向评价

\[
M_c\left(\theta^t+\sum_kq_k\Delta_k^t\right)-M_c(\theta^t).
\]

该 replay 不声称模拟重新训练后的长期轨迹；它只检验 class-aware placement 是否在同一状态、相同计算预算与相同权重集中度下，产生更低 ERI 和更好的即时 functional preservation。长期 retention 由上面的端到端对照负责。

H3 通过条件：

1. Client-LT 中 SN 相对 FedAvg 显著降低 maintenance ERIS/CERI；
2. Client-LT 中 SN 显著降低 best-to-final drop，并提高 final tail margin；
3. ERI 降幅与 drop 降幅在 class-paired 分析中同向相关；
4. frozen replay 中 class-aware 权重优于 permutation placebo 的均值，并超过其预注册 95th percentile；
5. matched Dirichlet 与 head 指标被完整报告，不允许只展示 Client-LT tail benefit。

---

## 5. 一阶近似与 PI attribution 的校验门槛

现有公式可以保留在论文正文，但必须注明它是局部一阶版本。审计轮次同时计算一阶值与 PI 值，至少报告：

1. client × tail-class effect 的 sign agreement；
2. Spearman correlation；
3. 四项累计预算的相对误差；
4. Client-LT vs Dirichlet 的 ERIS 排序是否一致；
5. PI completeness error：

\[
\operatorname{Err}_{c,t}=
\frac{left|
\sum_ke_{k,c}^{t,\mathrm{PI}}
-[M_c(\theta^{t+1})-M_c(\theta^t)]
\right|}
{|M_c(\theta^{t+1})-M_c(\theta^t)|+10^{-8}}.
\]

每个审计轮次的离线计算顺序固定为：

```text
theta_before, local_states, q, class_counts
  -> verify theta_after = theta_before + sum_k q_k * delta_k
  -> evaluate M_c(theta_before) and M_c(theta_after)
  -> for each of 8 path nodes, compute grad M_c(theta_before + alpha * delta_global)
  -> dot each class gradient with every client delta and integrate
  -> split effects by support bit and sign into W/H/D/R
  -> verify attribution completeness
  -> write only then join official-test retention outcomes
```

另计算服务器剂量下的 isolated finite difference

\[
\widetilde e_{k,c}^t=
M_c(\theta^t+q_k^t\Delta_k^t)-M_c(\theta^t),
\]

作为 sign sanity check。它不具备跨客户端可加性，因此不能替代 PI signed budget，但可检查 donor/rewriter 符号是否完全由所选归因路径的交互分摊造成。

建议门槛：

- median sign agreement ≥ 0.80；
- median Spearman \(\rho\ge0.70\)；
- PI completeness 的 median relative error ≤ 1%，95th percentile ≤ 5%；
- 一阶与 PI 的 topology contrast 同号。

如果一阶门槛失败，不代表 ERI 机制失败，只代表一阶估计器不够好。此时主结果全部使用 PI attribution。若 PI 数值积分仍无法满足 completeness，则增加 quadrature points；仍失败时检查模型状态加载、dropout/eval mode、参数 flatten 顺序与服务器重构，而不能继续报告 ERI。

作为更昂贵的最终备选，可在少量轮次使用 Monte-Carlo Shapley/permutation marginal attribution；只有 PI 路径因实现原因不可用时才需要它。

---

## 6. 数据完整性与防泄漏检查

每个被审计轮次必须保存以下内容：

- `global_before_trainable`；
- `local_trainable_states` 或无损 `local_deltas`；
- `global_after_trainable`；
- ordered selected-client IDs；
- 实际 server weights \(q_k^t\)；
- 每个客户端的样本数与 100 类训练 counts；
- trainable-key 顺序、shape、dtype 与 hash；
- 模型、数据、客户端和 augmentation RNG seeds；
- aggregation mode 与 protocol hash。

运行前后必须通过：

1. 按保存顺序用 \(q_k^t\) 重构服务器 state；
2. 重构 state 与真实 `global_after` tensor-exact，或最大误差在预注册 FP32 tolerance 内；
3. support bit 只能来自本轮训练 count，不得来自预测标签；
4. probe IDs 不属于任一 federated client；
5. 所有选择与阈值在 official test evaluation 前冻结；
6. FedAvg 与 SN 的本地训练在同一轮开始前都从各自真实 global state 出发，不能复用另一个条件的 local delta 充当端到端训练结果；frozen replay 例外，但必须明确标注为 replay。

---

## 7. 现有仓库产物能做什么，不能做什么

### 7.1 现在可以零训练完成的 pilot

以下两个成对 round-10 dump 都包含 global-before/global-after、30 个 local states、FedAvg weights 和 client-class counts：

- `output/cusp_minimal_refactor_20260801_163123/client-longtail_seed42_round10/cusp_minimal/round_010/round_state.pt`
- `output/functional_cusp_gate_seed42/dirichlet_beta0.5_seed42_round10/cusp_minimal/round_010/round_state.pt`

它们足以立即完成：

- round-10 的 \(W,H,D,R,\mathrm{ERI}\) snapshot；
- 一阶与 PI attribution 的 sign/rank/completeness 校验；
- Client-LT vs Dirichlet 的单 seed、单轮方向性 pilot；
- 同一 local deltas 上的 support-aware 与 permutation-placebo frozen replay。

该 pilot 的用途是验证实现和估算效应，不足以证明 H1–H3，因为它只有一个 seed、一个轮次，而且不是严格 fixed-\(n_k\) 主实验。

### 7.2 不能由现有 CSV 反推出的量

- PromptFL Experiment D 保存了 rounds 20/50/80 的 support-only、support-normalized 和 all-client 功能结果，以及 client update norms，但没有保存每个客户端的 update tensor；norm 不能恢复带方向的 \(u_{k,c}^t\)。
- 完整 PromptFL 轨迹保存了每轮 global prompt 和 per-class accuracy，但除了少数专门 dump 轮次，没有逐客户端 local state。
- 当前 ClipLoRA 2 × 2 和 Online SCA 轨迹保存了 per-round accuracy、aggregation weights 与拓扑 counts，但没有保存 global/local LoRA tensors。
- D1/D2 的 norm-equalized candidate updates 是 post-write 机制 replay，不是实际 on-trajectory FedAvg-weighted local budget，适合作为 signed-effect 外部验证，不应直接拼成主 ERI。
- `candidate_update_tensors.pt` 来自局部功能宽度审计，也不是 100 轮联邦轨迹。

因此，“完全不重新训练即可正式闭环”目前不成立。最省成本的可靠方案，是给现有 LoRA runner 加审计轮次 dump，然后重跑固定边际 2 × 2。若只加载已有 PromptFL global checkpoints 并重新做本地训练，应把所得量称为 state-conditioned probe-update ERI；除非可以 tensor-exact 重构原下一轮 global state，否则不能称为 realized on-trajectory ERI。

---

## 8. 推荐执行顺序

### Phase 0：实现与零训练 pilot

使用现有两个 round-10 dump：

1. 冻结 \(M_c\)、probe IDs、参数 scope 和符号约定；
2. 实现一阶与 PI client attribution；
3. 验证服务器 state reconstruction；
4. 验证四象限 closure；
5. 跑 support-normalized 与 20 个 permutation placebo；
6. 冻结 `eri_protocol.json`。

只有 Phase 0 的 reconstruction/completeness 通过后，才启动正式训练。

### Phase 1：正式 fixed-marginal FedAvg，先完成 H1 与 H2

先跑 2 topology × 5 seeds 的普通 FedAvg，并在审计轮次保存 local deltas。这样即使 intervention 尚未完成，也能先判断 ERI 是否确实是 topology-sensitive 且 outcome-relevant。

### Phase 2：加入 support-normalized，完成 H3

在完全相同的五组 split/seed 上跑 support-normalized，形成完整 2 × 2。随后完成端到端 retention、DiD 与 frozen replay placebo。

这个顺序避免在 H1 已失败时继续投入大量 intervention 计算。

---

## 9. 建议输出文件

```text
output/eri_closure/
├── protocol/
│   ├── eri_protocol.json
│   ├── probe_manifest.csv
│   ├── topology_pair_manifest.json
│   └── audit_rounds.json
├── dumps/<seed>/<topology>/<aggregation>/round_XXX/
│   ├── round_state.pt
│   ├── metadata.json
│   └── reconstruction.json
├── analysis/
│   ├── client_effects.parquet
│   ├── round_signed_budgets.csv
│   ├── cumulative_eri_per_class.csv
│   ├── attribution_validity.csv
│   ├── per_class_retention.csv
│   ├── topology_contrasts.csv
│   ├── lagged_models.csv
│   ├── intervention_contrasts.csv
│   ├── permutation_placebo.csv
│   └── eri_summary.json
└── figures/
    ├── eri_budget_and_ratio.pdf
    ├── eri_vs_best_to_final.pdf
    ├── intervention_arrows.pdf
    └── eri_temporal_heatmap.pdf
```

`client_effects.parquet` 至少包含：seed、topology、aggregation、round、class、client、support bit、\(q_k\)、一阶 effect、PI effect、sign、local update norm。

`round_signed_budgets.csv` 至少包含：\(W,H,D,R,P\)、raw ERI、ERIS、zero-refresh flag、actual \(\Delta M\)、attribution sum、completeness error。

---

## 10. 主文图表应该怎样画

### Figure A：不平衡究竟来自哪里

每个 topology 同时画累计 \(W,D,R,H\) 的 paired seed estimates，并在右侧画 ERIS。不能只画 ratio，否则读者无法判断是 \(R\) 上升还是 \(P\) 下降。

### Figure B：ERI 是否预测退化

x 轴为 maintenance log-CERI 或 ERIS，y 轴为 per-class margin best-to-final drop。颜色区分 topology，淡线连接同 seed、同 class 的 Client-LT/Dirichlet pair；图中报告 pooled 与 paired-difference Spearman 及 cluster CI。

### Figure C：干预闭环

每个 seed/class 画 FedAvg → support-normalized 的二维箭头：横轴 ERIS，纵轴 best-to-final drop。理想箭头指向左下。旁边显示 permutation placebo 的即时 margin-effect 分布。

### 主表

| Topology | Aggregation | \(W\) | \(D\) | \(R\) | \(H\) | ERIS | BFD margin | Final tail acc |
|---|---|---:|---:|---:|---:|---:|---:|---:|

所有列报告 seed-level mean 与 95% CI。\(W,D,R,H\) 不得省略。

---

## 11. 总通过规则与失败后的论文边界

采用固定顺序 gate：H1 → H2 → H3。只有前一层通过，后一层才作为 confirmatory evidence；这样无需对三个层次做事后挑选。

### 完整通过

- H1：fixed-marginal Client-LT 的 ERIS 显著高于 matched Dirichlet；
- H2：ERI 对 best-to-final 与 lagged future drop 均有正向预测；
- H3：class-aware support normalization 降低 ERI，并同步提高长期 retention，且优于 weight-distribution-matched placebo。

此时可以写：Evidence–Rewrite Imbalance 是被直接测量、与退化时间关联、且得到干预一致性支持的机制。

### H1 失败

不能再声称 Client-LT 具有更强 ERI。论文只能保留“positive access dilution”和“class-absent signed rewriting”两个并列现象。

### H1 通过、H2 失败

ERI 是 topology descriptor，但尚不能解释最终下降。

### H1/H2 通过、H3 失败

ERI 是有预测力的机制候选，但没有 causal-intervention closure。

### Support normalization 提升 retention，但 ERI 不降

说明现有 ERI 没有捕获干预路径；不能把 retention 改善归因于 ERI。

### ERI 降低，但 retention 不改善

说明 ERI 可能不是充分条件，或代价通过 head damage、supporter harm、优化不稳定等其他路径抵消。

### (H) 与 (R) 同量级或 (H) 主导

必须把故事从“broad class-absent rewriting 主导”收缩为“negative functional updates 相对 positive refresh 占优”，并把 class-absent attribution 作为子机制而非全部解释。

---

## 12. 通过后可直接进入论文的机制段落

> To directly quantify the proposed mechanism, we decompose each round's realized functional change for class \(c\) into positive updates from class-supporting clients, positive donor effects from class-absent clients, harmful effects from supporting clients, and destructive rewriting from class-absent clients. We define Evidence–Rewrite Imbalance as the cumulative destructive class-absent budget relative to all positive functional refresh. Under fixed global class counts and fixed client sizes, Client-LT exhibits a larger imbalance than matched Dirichlet. Importantly, this difference need not arise from a larger absolute rewrite budget: it is primarily the weaker positive refresh accessible under concentrated evidence ownership that allows rewriting to dominate. Class-wise ERI predicts subsequent best-to-final margin loss, and a test-independent support-aware aggregation intervention jointly reduces ERI and improves tail-function retention. These results close the link between sparse evidence ownership, broad shared-model rewrite authority, and long-horizon tail degradation.

中文对应表述：

> 我们将每轮类别功能变化直接分解为 supporter 的正向写入、class-absent donor 的正向刷新、supporter 的负向更新，以及 class-absent client 的破坏性改写，并将累计破坏性改写相对于全部正向功能刷新的比例定义为 Evidence–Rewrite Imbalance。在严格固定全局类别边际和客户端样本量边际后，Client-LT 的 ERI 高于 matched Dirichlet。该差异不要求 Client-LT 具有更大的绝对改写量；关键在于集中式证据所有权削弱了能够进入共享模型的正向刷新，使同等甚至更小的改写量也更容易占据主导。类别级 ERI 能够预测随后的 best-to-final margin loss，而与测试集无关的 support-aware aggregation 会同时降低 ERI 并改善尾类功能保留，从而闭合了“稀缺证据所有权—广泛共享改写权—长期尾类退化”这一机制链条。
