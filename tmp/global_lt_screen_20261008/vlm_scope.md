# VLM 与 global / personalized 边界核验（2026-10-08）

本分支只读检索论文、官方代码和已有本地适配，不改训练代码。筛选问题：候选是否确实产出一个可独立评测的 global 模型；是否针对全局长尾；能否保留 CLIP ViT-B/16、视觉 LoRA rank 4 和固定文本分类器。下列“宜纳入”是实验设计判断，尚未进行复现运行。

## 建议结论

| 方法 | 正式题录 / 年份 | global 边界 | 本轮定位 | 代码与接入代价 |
|---|---|---|---|---|
| CAPT | *CAPT: Class-Aware Prompt Tuning for Federated Long-Tailed Learning with Vision-Language Model*, arXiv:2503.06993, 2025 | 聚类用于训练与聚合；官方代码随后聚合成一个模型、执行 global test，不能因出现 clustering 就当成 pFL | **保留主表的 VLM Fed-LT 参照**，显式标“fixed-schedule adaptation”；单列其 prompt 参数化差异 | [官方代码](https://github.com/shihaohou/CAPT)可用，本地已有 pinned snapshot 与适配；原方法双文本 prompt、视觉耦合、class-aware 聚合，和固定文本视觉 LoRA 不是相同结构 |
| FedVLS | *Exploring Vacant Classes in Label-Skewed Federated Learning*, AAAI 2025 | 明确优化并评测一个 global 模型；global 是本地蒸馏 teacher | **优先机制表**：缺失类别保留对照。若主表有 general-FL 组可加入，但不可标 FL-LT 专用 | [官方代码](https://github.com/krumpguo/FedVLS)可用；logit loss 可迁移到当前视觉 LoRA，固定文本不必替换；多一次 round-start teacher 前向 |
| CLIP2FL | *CLIP-Guided Federated Learning on Heterogeneity and Long-Tailed Data*, AAAI 2024 | 真正 global Fed-LT；CLIP 是 frozen teacher，学生是独立分类网络 | **结构扩展附表候选**，不宜塞进严格 fixed-text 主表 | [官方代码](https://github.com/shijiangming1/CLIP2FL)可用；本地 CLIP KD、服务器梯度匹配与文本原型约束合成特征、重训可训练 classifier；接入成本高 |
| FedLoGe | *FedLoGe: Joint Local and Generic Federated Learning under Long-tailed Data*, ICLR 2024 | 原框架同时产出 1 个 generic + K 个 personalized 模型；官方有独立 global 路径，不能一概称纯 pFL | **FedLoGe-Global 结构扩展附表**，本轮低于轻量 loss 方法优先级；不得使用 local/personalized 指标 | [官方代码](https://github.com/ZackZikaiXiao/FedLoGe)可用；需要 sparse ETF、辅助 trainable global head 和 global norm realignment，替换 fixed CLIP text head |
| FedGELA | *Federated Learning with Bilateral Curation for Partially Class-Disjoint Data*, NeurIPS 2023 | 同时服务 generic / personalized；代码明确 global 用原始 ETF，local 用本地频率缩放 ETF；可取 genuine global 评测 | **可选机制表**：missing-class 几何对照；本轮不优先。不是 FL-LT 专用，且比 FedVLS 更大结构变化 | [官方代码](https://github.com/MediaBrain-SJTU/FedGELA)可用；固定随机 ETF + 本地 class-frequency scaling + CE，保留 text head 就失去其核心几何条件 |
| FedCBL | **未核验到此准确缩写对应的原论文** | 未定 | **暂排除命名候选**，不能按缩写猜正式文献 | 多轮精确检索 FedCBL / Fed-CBL 未找到可用 scientific primary；若实际指 FedAvg + class-balanced loss，应按组合 baseline 命名 |

## CAPT：正确题录与已有 global 适配

作者：Shihao Hou, Xinyi Shang, Shreyank N Gowda, Yang Lu, Chao Wu, Yan Yan, Hanzi Wang。arXiv 官方仍显示仅 v1（2025-03-10），未给会议 journal reference。本次发现媒体称其 ACM MM 2026 录用，但官方 repo 与 arXiv 未确认，**当前可靠写法为 arXiv 2025，勿写 CVPR/AAAI 2025**。[arXiv 原记录](https://arxiv.org/abs/2503.06993)、[全文](https://arxiv.org/html/2503.06993v1)。

直接代码证据：本地 unmodified upstream `third_party/paper_baselines/capt/federated_main.py` 570–579 行将 local weights 汇成 `global_weights`、载入 global trainer，再执行 `global_test(is_global=True)`。因此类别 prompt 和 cluster 并不意味着按 client 选择个性化模型推理。

本地 `third_party/paper_baselines/README.md` 记录 CAPT commit 为 `840b7cab26c04fc9e65540a980b6ce020d62d618`；已适配为每轮 global 聚合、关闭 test-driven MAB、共享 global 起点和隔离本地 optimizer，并保留上游 step 后 mask 的实际更新行为。`trainers/baselines/capt.py` 对 loss、clustering、aggregation 作显式实现。应继续使用这一路径，别混入历史串行客户端继承的旧结果。它可在算法主表保留，但不能声称与视觉 LoRA rank4 使用完全相同的可训练结构。

## FedVLS：最适合补充现有 FedNTD 的机制基线

正式发表为 Guo, Ding, Liang, Wang, He, Tan，AAAI 39(16), 16960–16968，2025-04-11，DOI 10.1609/aaai.v39i16.33864；arXiv 最早 2024，不能据此写 AAAI 2024。repo About 的旧题名 *Empty Classes Matter...* 与正式题名不同。[出版记录](https://ojs.aaai.org/index.php/AAAI/article/view/33864)、[论文全文](https://arxiv.org/html/2401.02329v3)。

官方 `clientvls.py` 中 objective 是 `PriorCELoss + LADE_loss + lamda * VLS_loss`：本地 prior 校准 CE、logit suppression（代码固定系数 0.005）、只在当前 client 的 vacant classes 上蒸馏。`set_parameters` 将 global 权重载入学生并复制为 teacher。它与已有 FedNTD 的区别是“client 缺失类别集合”对“每个样本 not-true 类别集合”，正好检验支持集缺失，而非单纯全类知识保持。[官方 client 源码](https://raw.githubusercontent.com/krumpguo/FedVLS/master/system/flcore/clients/clientvls.py)、[server 源码](https://raw.githubusercontent.com/krumpguo/FedVLS/master/system/flcore/servers/servervls.py)。

实现判断：可复用现有 global teacher / LoRA 训练框架，保留官方三个损失，冻结文本分类器；zero / one vacant class 情况需明确零损失处理，避免空 softmax。适配后名称可写 FedVLS (CLIP-LoRA adaptation)，不能只搬 VLS KD 再当完整方法。原仓库视觉实验用 MobileNetV2，故新结果是统一骨干适配而非原数值复现。

## CLIP2FL：是 global VLM-guided FL-LT，但不是 CLIP 参数微调

作者 Jiangming Shi, Shanshan Zheng, Xiangbo Yin, Yang Lu, Yuan Xie, Yanyun Qu。出版页题名是 “Heterogeneity”，PDF/repo 题名为 “Heterogeneous”；引用以出版记录为准。[AAAI 出版记录](https://ojs.aaai.org/index.php/AAAI/article/view/29416)、[原 PDF](https://ojs.aaai.org/index.php/AAAI/article/download/29416/30672)。

原实验学生 CIFAR 使用 ResNet-8、ImageNet 使用 ResNet-50，另加 MLP 匹配维度，teacher 使用 CLIP ViT-B/32；并非当前 CLIP ViT-B/16 LoRA。只保留 teacher KD 会删除服务器 classifier retraining 的关键贡献。如果未来要公平对比，可另外定义共享 CLIP visual backbone + trainable head 的结构扩展组，保留合成特征与重训机制；只训练固定文本 head 会使方法不完整。官方仓库 README 和 options 给出客户端/特征优化/重训参数，但本次未运行验证环境。

## FedLoGe / FedGELA：应区分 global 可评测与 fixed-text 可接入

FedLoGe 的 [官方 README](https://github.com/ZackZikaiXiao/FedLoGe)明确 global 推理为 `realignment_global_head(backbone(input))`；local heads 用 `features.detach()` 更新，而 global realignment 只需 global head 的范数，因此可以省略 personalized inference，保留作者原有 global 路径。这是基于官方依赖关系的适配判断，不是“完整 FedLoGe 纯 global 原设定”。[论文算法](https://arxiv.org/html/2401.08977v2)给出 SSE-C、global head 与 local heads，global head 行归一化和 local norm transfer 是不同步骤。只给现有 fixed text 向量归一化不构成 FedLoGe：当前向量已常规归一化，而且没有其学习出的辅助 global head。

FedGELA 的 [NeurIPS 2023 正式记录](https://papers.nips.cc/paper_files/paper/2023/hash/65b721a1df04c1098567f70d483d6468-Abstract-Conference.html)核验到年份；2024 arXiv 上传日期并不更改会议年份。其 [官方 FedGELA.py](https://github.com/MediaBrain-SJTU/FedGELA/blob/main/FedGELA.py) 中 `compute_accuracy_g` 使用 `features @ ETF`，`compute_accuracy_l` 和 local training 使用 `features @ (EW_set[client] * ETF)`。不能把其 PA 与 GA 混排，也不能因为有 local prior 直接把它判为没有 global 模型。当前若不愿替换 text classifier，选择 FedVLS 更直接。

## 扩展检索中的 scope 排除

- [FedCART, CVPR 2026](https://openaccess.thecvf.com/content/CVPR2026/html/Qin_FedCART_Tackling_Long-Tailed_Distributions_in_Federated_Adversarial_Training_via_Classifier_CVPR_2026_paper.html)：题名为 *Tackling Long-Tailed Distributions in Federated Adversarial Training via Classifier Refinement*，目标带 adversarial training，本轮普通分类 global 主表不优先。
- [pFedMoAP, ICLR 2025](https://proceedings.iclr.cc/paper_files/paper/2025/hash/92369a01fbe8046a093746389b2c413e-Abstract-Conference.html)：依赖本地 gating 与多个非本地 prompt experts，明确 personalized，排除。
- Fed-Duet, ICLR 2026 的 [原文索引](https://openreview.net/pdf?id=Jk8g1OxyZY)针对 federated continual learning；搜索中的 long-tailed 只是相关工作对 CLIP2FL 的描述，不应据关键词当成新 Fed-LT 算法。

这一分支未确认到一个同时满足“比 CAPT 更新、global VLM Fed-LT、官方可复现、无新结构”的额外直接方法。这里是有范围的检索结论，不声称整个领域不存在这样的论文。近期 FedPuReL 已由主任务明确列为现有 global adaptation；不重复列为新候选。
