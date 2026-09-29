# 普通长尾识别的迁移对象与负迁移边界

检索日期：2026-09-28。范围：直接核验原始论文；五篇机制核心文献，加一篇 2026 年反证性检查。以下「B 适配」是研究建议，非原论文结果。

## 核验后的六篇核心文献

### 1. Feature Space Augmentation for Long-Tailed Data

Peng Chu, Xiao Bian, Shaopeng Liu, Haibin Ling. **ECCV 2020**，并非 CVPR 2020。[ECVA 原文](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123740681.pdf)，[作者 arXiv](https://arxiv.org/abs/2008.03673)。已读 §3–4、Algorithm 1。

CAM 将尾类的判别性局部特征与供体的通用局部特征分开；供体选尾类最容易混淆的类别，按随机比例抽取局部向量组成增强样本，标签仍属尾类。第二阶段同时保留真实 head 样本训练。限制：低 CAM 响应不等于严格的类别无关成分；近邻筛选与内容分离是启发式，不是逐类风险不增保证。集中式方法可访问源、目标有标签样本和特征。B 可借鉴「保留目标身份，仅移植变化」，不能据此证明完整 LoRA 可迁移。

### 2. Free Lunch for Few-shot Learning: Distribution Calibration

Shuo Yang, Lu Liu, Min Xu. **ICLR 2021**。[会议版本 PDF](https://arxiv.org/pdf/2101.06395)，[OpenReview](https://openreview.net/forum?id=JWOiYxMG92s)。已读 §3、Eq. 4–8。

对新类支持特征 x 选择最近的 k 个 base 类，迁移均值与协方差：μ′=(x+Σμ_h)/(k+1)，Σ′=ΣΣ_h/k+α（沿用原式的平滑记号），再采样训练新任务分类器。base/novel 标签集明确不交，因此供体不必拥有目标类别。限制：目标仍需有标签支持样本；依赖特征空间相似性和近似高斯性；评测新类 episodic 平均准确率，未验证保留旧类的共同分类性能，也无逐类 no-harm 定理。B 可迁移统计，但应避免未经验证地拖动目标均值。

### 3. GistNet: A Geometric Structure Transfer Network for Long-Tailed Recognition

Bo Liu, Haoxiang Li, Hao Kang, Gang Hua, Nuno Vasconcelos. **ICCV 2021**。[CVF 元数据](https://openaccess.thecvf.com/content/ICCV2021/html/Liu_GistNet_A_Geometric_Structure_Transfer_Network_for_Long-Tailed_Recognition_ICCV_2021_paper.html)，[作者 PDF](https://arxiv.org/pdf/2105.00131)。已读 §3、Fig. 4、Table 1。

每类分类器由中心 w_c 与共享形变 δ_j 构成 constellation：w_cj=g(w_c,δ_j)。class-balanced 损失训练中心；random-sampling 损失训练共享形变，后者自然主要来自 head。迁移的是共享几何，不是 head 中心。梯度分路限制 head 偏置进入哪里，但共享形状假设仍可能失配。原文的 guarantee 指参数更新路径，不是准确率。Places-LT 相对 Plain Model：many-shot 45.9→42.5，few-shot 0.36→32.1。B 可分离身份参数和可共享变化参数；需目标中心的本地监督。

注意：作者 arXiv v1 在 Table 1 后有一句把两种 sampling 写反；以上按 Fig. 4、§3.3 的公式及多处一致描述记录，勿引用该孤立句。

### 4. MetaSAug: Meta Semantic Augmentation for Long-Tailed Visual Recognition

Shuang Li, Kaixiong Gong, Chi Harold Liu, Yulin Wang, Feng Qiao, Xinjing Cheng. **CVPR 2021**。[CVF 元数据](https://openaccess.thecvf.com/content/CVPR2021/html/Li_MetaSAug_Meta_Semantic_Augmentation_for_Long-Tailed_Visual_Recognition_CVPR_2021_paper.html)，[作者 PDF](https://arxiv.org/pdf/2103.12579)。已读 §3、Eq. 1–6。

不是直接复制 head 协方差：将类条件协方差作为可优化超参数，经一次虚拟训练更新，对小型平衡验证集损失求超梯度，再更新模型。ISDA 的无限高斯扰动通过 CE 上界实现。它用接收方验证反馈筛增强方向，不以「head 必然正确」为前提。限制：优化平均验证损失仍允许个别类别退化；需要目标类验证标签，缺类客户端不能直接照搬。B 可使用此双层选择思路，但应额外报告逐类退化，而非只验平均收益。

### 5. FASA: Feature Augmentation and Sampling Adaptation for Long-Tailed Instance Segmentation

Yuhang Zang, Chen Huang, Chen Change Loy. **ICCV 2021**。[作者 PDF](https://arxiv.org/pdf/2102.12867)，[arXiv 元数据](https://arxiv.org/abs/2102.12867)。已读 §3、Table 1、补充 B.3。

这是无跨类迁移的必要对照：在线维护本类均值/对角标准差，采样 x̂_c=μ_c+σ_c⊙ε；验证损失好转则增加增强采样，恶化则减少。缺类或高噪声验证时，用特征相近簇的平均损失反馈。Table 1 的单 FA：rare AP 8.0→12.7，但 frequent 27.0→26.9；加入采样反馈为 17.8/27.2。这是实例分割 mask AP，非分类准确率；该小幅下降不构成统计显著性结论。反馈缓解并非逐类保证。B 应设本类增强对照，且不能把簇平均反馈当成缺失类的安全证明。

### 6. Decision Boundary-aware Generation for Long-tailed Learning

Jiacheng Yang, Ruichi Zhang, Chikai Shang, Mengke Li, Xinyi Shang, Junlong Gao, Yonggang Zhang, Yang Lu. **2026**；作者 arXiv 注明 Accepted by CVPR 2026，尚未另验正式 CVF proceedings。[元数据](https://arxiv.org/abs/2605.01468)，[全文](https://arxiv.org/html/2605.01468v1)。已读 §3–4、Table 1–3。

实证指出粗粒度 head-to-tail diffusion 迁移可增加类间重叠和 tail 离群率；提出近边界生成，结合原型距离、置信度过滤。这不支持所有语义移植都安全。其自身亦无统一 no-harm：Table 1，CIFAR100-LT IF100，CBDM→CBDM+DBG 的 tail 28.60→30.10，但 head 69.10→67.30。B 的验证应同时看类间混淆、目标增益及非目标退化，不能把类别分布更均匀当成边界更好。

## 对方法 B 的综合判断（本笔记推论）

1. **供体没有目标类并非原则性障碍。** 跨类迁移本来就在借其它类的变化、统计或形状。障碍是目标身份、供体相关性以及效果验证来自哪里。DC 最清楚地展示了源目标标签完全不交但目标仍需少量有标签支持样本。
2. **选择什么传与检验是否有害是两件事。** CAM 分解、相似类检索、共享几何约束回答前者；MetaSAug/FASA 的验证反馈回答后者。没有读到可直接迁移成「所有类准确率不降」的定理。
3. **不应把整段 LoRA 更新视作类内变化。** 可先在共同的冻结 CLIP 坐标中收集类中心与中心化残差/低秩协方差；按语义与视觉相似性选择供体；固定目标身份锚，仅控制接收方扰动方向和强度。这是需实验验证的候选适配。
4. **避免把不动 A 的参数写成不伤 A。** 即使 A 的特征和 logits 不动，B 的 logits 上升仍可能抢走 A 的样本。应评价联合类别空间下每个类的风险变化，而非仅核查被更新的模块。
5. **若 B 无目标类验证样本，必须降低主张。** 文本相似、协方差收缩、信任域、原型距离与教师一致性只能作保守代理；无法识别真实目标风险。可利用有该类的客户端回传验证统计，或将结论限定为可观测类别/已评测分布下的经验退化率。

书目补充：真正的 CVPR 2020 相关工作是 Jialun Liu, Yifan Sun, Chuchu Han, Zhaopeng Dou, Wenhui Li, *Deep Representation Learning on Long-Tailed Data: A Learnable Embedding Augmentation Perspective*（LEAP）。[CVF 元数据及摘要](https://openaccess.thecvf.com/content_CVPR_2020/html/Liu_Deep_Representation_Learning_on_Long-Tailed_Data_A_Learnable_Embedding_Augmentation_CVPR_2020_paper.html)。本笔记未成功读其完整 PDF，故未将它作为核心机制证据。
