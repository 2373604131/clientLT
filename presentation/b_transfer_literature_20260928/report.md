# 头类知识如何支持尾类：迁移内容、类别冲突与方法 B 的实现选择

调研日期：2026-09-28。使用 `deep-research` skill，结合原始论文与本仓库结果。本文是研究决策文稿，候选改法尚未实现或验证。

后续设计更新：第8节保留为调研阶段的参数组合候选；当前建议优先研究的主方案见[目标尾类的正响应蒸馏](../../docs/method_b_positive_response_design_20260928.md)，其重点是将来源的正向响应转化为监督内容。两者均尚未实现或验证。

## 摘要

本次调研回答：头到尾迁移究竟传什么，如何处理帮助一个类别却损害另一个类别，以及如何适配当前联邦 CLIP-LoRA 方法 B。查阅的工作分别控制迁移内容、监督质量和联合优化方向。这些机制能够缓解特定条件下的负迁移，但未提供可直接用于我们设置的“每个类别测试精度均不退化”保证。现有 B 已把逐类 donor 关系保留到更新中；下一步值得优先研究的是，在保留这些来源约束的同时，用完整的目标类别反馈选择共享更新。非负系数只能限制方向变换，不能独立解决跨类冲突。本文提出一个在现有来源组合空间内优化的候选方案，并将训练损失保护与尾类测试泛化明确分开。

## 1. 研究问题与范围

研究问题在检索前固定为：

- **RQ1：** 普通长尾与相关少样本方法，实际从头类向尾类迁移什么？目标类身份如何保留？
- **RQ2：** 方法如何减少“帮助类别 N、损害类别 M”？证据是平均实验结果、逐类结果，还是有条件的优化保证？
- **RQ3：** 在共享 LoRA、目标尾类样本稀少、donor 缺少目标标签的约束下，哪些机制适合当前 B？

本文中的“类别 N/M”与我们论文的“方法 A/B”没有对应关系。B 的固定职责仍是增强目标尾类可获得的学习支持；评价以固定 Tail20 为主。讨论类别之间的取舍，不意味着给 B 增加 Overall 优化目标。

## 2. 检索与核验方法

三个并行视角分别检索头到尾的特征/统计迁移、语义污染与泛化反例、梯度冲突与多目标优化；主线另核对联邦及 CLIP 适配条件。关键词覆盖 head-to-tail transfer、class-generic features、distribution calibration、negative transfer、class-wise gradient conflict 和 long-tailed multi-objective optimization，并追踪核心论文引用及2025—2026年新工作。

本文纳入18篇代表性研究，时间跨度为2017—2026年。标题、作者、年份和方法陈述以 CVF/ECVA、OpenReview、PMLR、AAAI、NeurIPS 或作者 arXiv 原文核验。正文方法依据优先来自全文；TCL [7] 仅能核验 CVF 官方摘要，因而只讨论摘要明确列出的机制。DBG [16] 的 CVPR2026 接收信息来自作者 arXiv 声明，未独立核验正式会议录。没有把未读到原始机制的二手解读作为实现依据。

这是一份面向当前决策的定向综述，不是穷尽性文献普查。没有将不同数据集、骨干、任务或训练预算上的数值排成统一性能榜。

## 3. 三个干预位置

| 干预位置 | 核心问题 | 能提供的保护 | 仍存在的风险 |
|---|---|---|---|
| 迁移内容 | 哪部分知识可以跨类别复用？ | 减少把来源类别身份一起搬过去 | 类别共享假设可能错误，来源仍会不相关 |
| 监督与质量反馈 | 借来的内容是否产生有效监督？ | 过滤差样本、差教师或差增强策略 | 平均反馈会掩盖个别类退化；反馈可能过拟合 |
| 共享更新 | 多个目标使用同一模型时如何处理冲突？ | 控制所测目标上的更新方向与取舍 | 一阶、经验损失保证不等于有限步或测试精度保证 |

分类按主要干预位置组织，而不是将论文当成互不相关的方法列表；某些工作跨越多个位置。以下分别比较这些机制，再讨论适配 B 的边界。

## 4. 可迁移内容：借变化与结构，保留目标身份

FSA 显式拆分特征内容，Distribution Calibration 借用分布统计，GistNet 则共享类别几何；三者都比“某个来源整体有益，因此搬入其全部更新”更具体地定义了迁移对象。[1 FSA](https://arxiv.org/pdf/2008.03673)、[2 Distribution Calibration](https://arxiv.org/pdf/2101.06395)、[3 GistNet](https://arxiv.org/pdf/2105.00131)。

| 方法 | 迁移对象与接收方式 | 对“避免伤害”的实际含义 |
|---|---|---|
| FSA，ECCV2020 | 用 CAM 区分通用/类别特定局部特征，组合供体通用部分与尾类特定部分 | 减少身份混入；分解和选源是启发式 |
| Distribution Calibration，ICLR2021 | 从近邻 base 类借统计，结合 novel 类支持特征形成校准分布 | 源、目标标签可不交；新类 episodic 评估不保证旧类保持 |
| GistNet，ICCV2021 | 类别中心与共享形变分开学习，采样和梯度路径服务于不同参数 | 约束知识进入何处；不是逐类准确率保证 |

由此得到的 B 适配推断是：**donor 没有类别 N 的标签，只说明来源资格满足要求，并不说明其完整 LoRA 更新就是可共享的通用知识。** 当前正向 probe 证明的是这个更新在一批 N 类训练样本上的局部损失响应；它没有完成语义分解，也没有证明方向变换后的响应仍然相同。

如果未来转向特征迁移，可以在共同冻结表征坐标里，分别估计来源的类别中心和中心化变化，再用尾类自身样本或文本语义保留身份。但这需要新增统计估计、特征传递及分类器训练规则；每类4—9张目标图片不足以稳健估计高维协方差。因此它是另一条方法路线，不能作为替换 C 的几行代码来包装。

## 5. 质量反馈：更好的来源仍要成为可靠监督

MetaSAug 用平衡验证反馈选择增强方向，FASA 调节增强采样；DiVE 调整教师软监督的类别分布，TCL 则筛除错误专家知识。它们共同说明“生成或获得更多知识”与“知识值得学习”是两件事，同时也说明监督改良并不等同于逐类无损。[4 MetaSAug](https://arxiv.org/pdf/2103.12579)、[5 FASA](https://arxiv.org/pdf/2102.12867)、[6 DiVE](https://arxiv.org/pdf/2103.15042)、[7 TCL 官方摘要](https://openaccess.thecvf.com/content/CVPR2026/html/Zhou_Trust-calibrated_Collaborative_Learning_for_Long-Tailed_Visual_Recognition_CVPR_2026_paper.html)。

| 方法 | 反馈如何进入迁移 | 不能直接移植的条件 |
|---|---|---|
| MetaSAug，CVPR2021 | 用虚拟训练更新与平衡验证损失优化语义增强协方差 | 需要验证标签；平均验证目标仍允许类间取舍 |
| FASA，ICCV2021 | 根据验证损失变化调整特征增强采样 | 属于实例分割，且本类增强不等于跨类迁移；缺类验证采用簇反馈 |
| DiVE，ICCV2021 | 将教师预测视作虚拟样本，调整监督分布后训练学生 | 教师预测本身可能有偏，学生仍共享参数 |
| TCL，CVPR2026 | 官方摘要提出知识质量门控、尾类补偿和错误共识校准 | 本次未读到全文，不据此指定门控公式或声称保证 |

DiVE 给出了一个直接的类别组取舍例子：ImageNet-LT 上 CE 到 DiVE，Few 从8.07到31.46，而 Many 从65.02到64.06。它说明尾组受益可以伴随其他组退化；这仍是分组结果，不是我们关心的两个具体尾类之间的证明。[6，Table 2](https://arxiv.org/pdf/2103.15042)。

对 B 而言，若改成从 donor 的预测响应中蒸馏，就必须回答哪些响应可信、如何保留真实尾类标签、如何避免教师的其他类别偏置进入学生。完整 logits 蒸馏或多个教师的简单平均，都不能仅凭“蒸馏”这个名称获得安全性。该路线也不能自动消除共享学生中的梯度冲突。

## 6. 共享参数下，必须区分三种保证

直接面向长尾的 GBG 将类别按梯度相似性分组，再使用多目标优化；MGDA、GEM、PCGrad 和 CAGrad 则提供不同的方向选择或修正工具。GBG 的直接相关性更强，其他工作提供可借用的优化思想，但适用条件不能互相替换。[8 GBG](https://ojs.aaai.org/index.php/AAAI/article/download/28103/28211)、[9 MGDA](https://papers.neurips.cc/paper/7334-multi-task-learning-as-multi-objective-optimization.pdf)、[10 GEM](https://papers.nips.cc/paper/7225-gradient-episodic-memory-for-continual-learning.pdf)、[11 PCGrad](https://papers.neurips.cc/paper_files/paper/2020/file/3fe78a8acf5fda99de95303940a2420c-Paper.pdf)、[12 CAGrad](https://proceedings.neurips.cc/paper_files/paper/2021/file/9d27fdf2477ffbff837d73ef7ae23db9-Paper.pdf)。

| 方法 | 控制对象 | 保证边界 |
|---|---|---|
| GBG，AAAI2024 | 分组后的类别梯度 | 组层面的共同下降不等于每个类共同下降 |
| MGDA，NeurIPS2018 | 任务梯度凸包的最小范数向量 | 共同下降或 Pareto 驻点；任意 donor 更新不是任务梯度 |
| GEM，NeurIPS2017 | 当前更新与记忆任务梯度的内积 | 依赖局部线性与记忆代表性 |
| PCGrad，NeurIPS2020 | 逐对冲突梯度分量 | 不能推出任意多类的有限步逐类无损 |
| CAGrad，NeurIPS2021 | 平均方向附近的最差一阶收益 | 有条件的平均目标驻点结果；最差收益仍可能为负 |

不能把“采用多目标优化”直接当作改进结论。GBG 的 CIFAR10-LT IF100 消融中，BCL 为84.07，直接加 MOO 为74.58，再加 GBG 为85.05；作者讨论了 minibatch 中尾类缺席造成的目标变化。我们现有 B 已在每一步使用完整119张目标池，因而**不能把该退化原因直接套到我们头上**，也不必为了模仿论文先把20个目标类分成更粗的组。[8，Table 6](https://ojs.aaai.org/index.php/AAAI/article/download/28103/28211)。

以下三个层次需要分开：

1. 在当前参数附近，对所测训练损失的一阶方向有利。
2. 真正执行一次有限更新后，所测训练损失没有明显受损。
3. 在独立测试样本上，类别分类能力提高。

前一层并不自动推出后一层。对我们而言，“50/160个事件—类别的 LA 上升”属于第二层的现象；Tail20 是否提高属于第三层。

## 7. 与当前联邦 CLIP 设置的连接及反证

LIFT 在基础模型长尾学习中展示了训练适配与尾类泛化可能分离；CAPT 通过通用与类感知 prompt 及聚类组织共享知识；FedAFA 则在个性化联邦设置中生成少数类特征，并同时训练原始与生成内容。三者分别提示我们考虑泛化、共享粒度和边界保持，但没有一篇等同于当前缺标签 donor 的共享 LoRA 残差设置。[13 LIFT](https://proceedings.mlr.press/v235/shi24g.html)、[14 CAPT](https://arxiv.org/html/2503.06993v1)、[15 FedAFA](https://arxiv.org/pdf/2303.15168)。

FedAFA 尤其贴近“帮 N 害 M”的问题：它指出源特征和转化后的目标特征太接近，可能使源类被错判；它保留源特征和原始样本监督来学习区分边界。这是联合训练的经验机制，不是所有类测试无损定理。我们可以借鉴“使用外部支持时仍由真实目标数据确定边界”，不能照搬其个性化分类器目标到共享 CLIP。[15，§2.2.2](https://arxiv.org/pdf/2303.15168)。

结构分离也有边界：RIDE 采用互补专家，Decoupling 工作冻结表征再调整分类器，二者限制不同知识的共享方式，但分类输出仍在共同类别空间竞争。某个类别对应的模块没有改变，并不意味着其样本不会被另一个得分上升的类别抢走。[17 RIDE](https://arxiv.org/pdf/2010.01809)、[18 Decoupling](https://arxiv.org/pdf/1910.09217)。

近期 DBG 专门研究生成迁移对边界的影响；其 CIFAR100-LT IF100 表格中，CBDM 加 DBG 的 tail 从28.60到30.10，head 从69.10到67.30。这再次限定了“迁移有用”的含义：必须说明谁获益、代价是什么。[16 DBG，Table 1](https://arxiv.org/html/2605.01468v1)。

我们的本地结果还提供了直接反例，详见[逐类汇总](../b_directed_review_20260928/tail_class_summary.csv)：

| 类别 | 8次迁移中 LA 上升的次数 | 新版相对 A-only 的最后20轮测试精度变化 |
|---|---:|---:|
| 81 | 7 | +0.75 个百分点 |
| 86 | 0 | −0.55 个百分点 |
| 95 | 8 | +0.35 个百分点 |

两列不是同状态的长期因果对照，不能解释成“损害训练损失有利于泛化”。它们说明，当前数据不足以把训练 LA 冲突认定为 Tail20 没有提升的唯一原因。也不能把“受损类数量降到零”替代论文的效果目标。

## 8. 下一版 B 的优先候选：在固定来源组合内使用完整逐类反馈

以下为基于文献和代码的适配推导，**不是某篇论文已验证的联邦算法，也不是测试增益承诺**。它回答的是当前实现最直接暴露的问题：选源时认可的方向，如何在最终共享更新中使用，并显式考虑对其他目标尾类的影响。

### 8.1 固定迁移对象与坐标

仍对每个目标类 d 保留当前筛选规则：目标组之外、没有 d 类标签、在 d 类训练反馈上正收益、Top-3。用固定权重构造

\[
U_d=\sum_{j\in D_d}p_{dj}\,\Delta B_j.
\]

这里每个 U_d 是所有可迁移 LoRA-B 模块组成的完整方向集合。A 固定。建议用一个跨模块共享的非负标量 a_d 控制一个完整 U_d：

\[
R(a)=\sum_d a_d U_d,\qquad a_d\ge0.
\]

自由矩阵 C 暂时由这种受约束缩放替代。其作用是保留候选方向和固定来源比例，不是把原始 donor 更新“净化”为通用语义。固定组合本身的正收益也应测量，不能把各 donor 单独的正收益当作非线性组合的保证。

不要给每条“类—donor”边重新分配完全自由的系数，再全部加到同一个共享残差。对相同 donor 的系数可以在求和中合并，这会丢掉原本想保留的类别组合限制。固定 p、只调组合强度，保留的是候选更新集合的约束；它不是推理时的类别独立路由。

### 8.2 从单类认可扩展为完整的交叉影响

定义每个目标类 c 的样本平均训练损失 L_c。用当前普通 FedAvg 后的共享 B 作为共同参考，在相同 B 参数坐标中计算

\[
H_{cd}=-\left.\frac{\partial L_c(B_0+R(a))}{\partial a_d}\right|_{a=0}
=-\sum_m\langle\nabla_{B_m}L_c(B_0),U_{d,m}\rangle.
\]

H 是最多20×20的矩阵：列是固定类别来源组合，行是全部目标尾类。它直接回答“为 d 选择的组合，会如何影响 c”。由于 A 固定，B 坐标内的链式导数就是实际 LoRA 参数化的方向导数；不应把 B 张量与全权重矩阵的梯度混用。

**来源资格与损害测量是两件事。** donor 因为缺少 N 标签而有资格帮助 N，但判断是否损害 M 时，仍需要 M 的反馈，即使该 donor 自己拥有 M 标签。当前 donor 表只记录合格缺类配对，不足以直接拼出完整交叉矩阵。

H 可以通过各目标客户端对系数 a 求逐类梯度，再按该类的样本总量合并得到；本地数据无须集中。服务器最终需要的是每类对候选方向的响应。数据不出客户端本身不等于已经有形式化隐私保证。

### 8.3 联合选强度，允许明确的尾类取舍

如果只把对角线 H_dd 当成“这个组合有益”的证据，会漏掉全部交叉项。非负 a 也不能保证 Ha 的每一项非负。

一个可实现的软约束候选是：

\[
\begin{aligned}
\max_{a,s}\quad& \tfrac1{|T|}\mathbf1^\top Ha
-\lambda\|a\|_2^2-\rho\mathbf1^\top s\\
\text{s.t.}\quad&Ha\ge-\epsilon\mathbf1-s,\quad s\ge0,\quad a\ge0,\\
&\mathbf1^\top a\le b,\quad\|R(a)\|_{\mathrm{effective}}\le r.
\end{aligned}
\]

平均预测收益仍是主要目标；s 表示允许但需要付出代价的逐类损失上升。b 和 r 控制注入量，r 可沿用仓库计算有效 LoRA 权重更新范数的坐标定义。这里展示的是完整可审查的形式；实现时应合并冗余正则或预算，避免同时增加大量可调参数。当前结果没有给出这些数值的有效取值。

硬性要求所有目标类都不受损，可能只剩零更新。软约束保留收益与损害的权衡；若所有可用组合都不值得迁移，允许零更新是合理结果。不能用“强制系数和等于1”把一次没有有效方向的事件变成必然注入。

### 8.4 最后按真实模型检验有限更新

线性预测 H 只在当前参数附近成立。组合后需在同一完整目标池上做真实前向计算，核对实际平均收益与所允许的逐类损害；若偏离接受规则，就缩小注入或放弃本次残差。规则在实验前固定，不能按测试精度选择步长或回退。

这是 B 在迁移事件内部确定参数的步骤，使用目标端训练反馈；不需要另开第30/60/90轮离线诊断实验。它仍增加了逐类梯度、求解及可能回退的成本，不能把20个系数少于1920个参数直接写成已经更快。

当前所有反馈来自同一119张训练图片，每类4—9张。相同图片的多个增强视图不会成为新的独立样本。即使候选严格改善这批图片，也需要独立测试判断其尾类学习价值。

### 8.5 在仓库中的修改位置

| 现有位置 | 候选修改 |
|---|---|
| `utils/cliplora_b_directed.py::screen_sources` | 保留缺类来源筛选及记录，明确探测属于候选提名 |
| `utils/b_directed_math.py::build_class_bases` | 保留固定的类—donor 比例绑定 |
| `utils/cliplora_b_directed.py::calibrate_bases` | 改为逐类系数梯度、完整交叉响应和受约束的强度选择 |
| `utils/b_directed_math.py::reconstruct_class_residual` | 用完整方向的非负缩放重建共享残差 |
| `utils/cliplora_b_directed.py::apply_shared` | 真实候选前向核验及固定接受规则，随后按原流程执行 A |

上述只是一条优先候选路线。与直接改成特征生成、蒸馏或多专家相比，它保留了目前已核查的来源定义、共享模型和反馈范围，更容易解释这次改动到底解决了什么。

## 9. 尚未解决的问题与结论

**回答 RQ1：** 可迁移知识可以是类内变化、分布统计、共享几何或预测关系。头类知识有用，并不意味着其类别中心、全部 logits 或整段参数更新都适合目标尾类。缺目标标签并不是跨类迁移的原则性障碍，但目标端仍需可靠的类别锚和反馈。

**回答 RQ2：** 文献主要通过限制传递内容、反馈筛选、保留原始监督、结构分离或多目标方向控制缓解负迁移。已核验的机制没有提供能直接用于本任务的逐类测试无损保证；组均值上升、训练损失下降和测试分类提升必须分开陈述。

**回答 RQ3：** 下一版最适合优先讨论的是固定来源组合内的逐类反馈约束，而不是单独把 C 换成非负数。它能形成一个明确可检验的机制主张：让不同目标尾类的响应共同决定外部来源的使用。若其训练冲突明显减少、Tail20 仍无提升，下一步应转向迁移内容和泛化问题，不能继续只调损害惩罚来追求漂亮的训练指标。

当前尚无证据决定：这些 donor 更新是否包含足够的新判别信息，119张样本能否可靠选择可泛化的来源，以及减少当前 LA 冲突是否有助于长期尾类保持。方法目标可保持，但“已增强有效学习支持”仍要由实际效果证明。

## 参考文献

[1] Peng Chu, Xiao Bian, Shaopeng Liu, et al., “Feature Space Augmentation for Long-Tailed Data,” ECCV, 2020.

[2] Shuo Yang, Lu Liu, Min Xu, “Free Lunch for Few-shot Learning: Distribution Calibration,” ICLR, 2021.

[3] Bo Liu, Haoxiang Li, Hao Kang, et al., “GistNet: A Geometric Structure Transfer Network for Long-Tailed Recognition,” ICCV, 2021.

[4] Shuang Li, Kaixiong Gong, Chi Harold Liu, et al., “MetaSAug: Meta Semantic Augmentation for Long-Tailed Visual Recognition,” CVPR, 2021.

[5] Yuhang Zang, Chen Huang, Chen Change Loy, “FASA: Feature Augmentation and Sampling Adaptation for Long-Tailed Instance Segmentation,” ICCV, 2021.

[6] Yin-Yin He, Jianxin Wu, Xiu-Shen Wei, “Distilling Virtual Examples for Long-Tailed Recognition,” ICCV, 2021.

[7] Hao Zhou, Tingjin Luo, “Trust-calibrated Collaborative Learning for Long-Tailed Visual Recognition,” CVPR, 2026.

[8] Weiqi Li, Fan Lyu, Fanhua Shang, et al., “Long-Tailed Learning as Multi-Objective Optimization,” AAAI, 2024.

[9] Ozan Sener, Vladlen Koltun, “Multi-Task Learning as Multi-Objective Optimization,” NeurIPS, 2018.

[10] David Lopez-Paz, Marc'Aurelio Ranzato, “Gradient Episodic Memory for Continual Learning,” NeurIPS, 2017.

[11] Tianhe Yu, Saurabh Kumar, Abhishek Gupta, et al., “Gradient Surgery for Multi-Task Learning,” NeurIPS, 2020.

[12] Bo Liu, Xingchao Liu, Xiaojie Jin, et al., “Conflict-Averse Gradient Descent for Multi-task Learning,” NeurIPS, 2021.

[13] Jiang-Xin Shi, Tong Wei, Zhi Zhou, et al., “Long-Tail Learning with Foundation Model: Heavy Fine-Tuning Hurts,” ICML, 2024.

[14] Shihao Hou, Xinyi Shang, Shreyank N Gowda, et al., “CAPT: Class-Aware Prompt Tuning for Federated Long-Tailed Learning with Vision-Language Model,” arXiv:2503.06993, 2025.

[15] Yang Lu, Pinxin Qian, Gang Huang, et al., “Personalized Federated Learning on Long-Tailed Data via Adversarial Feature Augmentation,” ICASSP, 2023.

[16] Jiacheng Yang, Ruichi Zhang, Chikai Shang, et al., “Decision Boundary-aware Generation for Long-tailed Learning,” arXiv:2605.01468, 2026.

[17] Xudong Wang, Long Lian, Zhongqi Miao, et al., “Long-tailed Recognition by Routing Diverse Distribution-Aware Experts,” ICLR, 2021.

[18] Bingyi Kang, Saining Xie, Marcus Rohrbach, et al., “Decoupling Representation and Classifier for Long-Tailed Recognition,” ICLR, 2020.
