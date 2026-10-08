# 2025–2026 共享全局模型 Fed-LT 筛选核验

检索截止：2026-10-08。只读文献与官方代码，未运行论文代码、未改训练代码。问题：新方法是否输出共享全局模型；是否适用于 CIFAR-100 Client-LT；迁入 CLIP ViT-B/16 + LoRA r=4 是否保留核心机制。下列接入判断是依据论文/代码的工程推断，并非迁移性能结论。

## 建议排序

**优先 FedYoYo、FedReLa。MMC-FL 的 missing-class 动机非常贴近，但固定 CLIP 文本头与其可训练分类器机制不一致，且未找到已公开实现；FedNPC/FedSM 放在需要重训分类器的扩展对照。**

| 方法 | 精确题名、时间 | 全局模型证据 | 官方代码状态 | CLIP–LoRA 适配判断 |
|---|---|---|---|---|
| FedYoYo | *You Are Your Own Best Teacher: Achieving Centralized-level Performance in Federated Learning under Heterogeneous and Long-tailed Data*，Shanshan Yan 等，ICCV 2025，2750–2759；arXiv 2025-03-10 | 论文 Fig. 2/§3.3 明确 global model accuracy、global representations；客户端共享同一聚合模型 | [shanss132/FedYoYo](https://github.com/shanss132/FedYoYo) 公开，main_fedyoyo.py、models、data_loader、运行脚本可见 | 最适合优先接入。ASD：正确预测弱增强作为强增强蒸馏教师；DLA：融合局部与估计全局分布的 balanced softmax。原 ResNet-8/50；当前模型可保留固定文本头，在 logits/图像特征层实现损失。需双增强、warm-up、分布估计/通信，不能仅加 KL 就称完整复现 |
| FedReLa | *FedReLa: Imbalanced Federated Learning via Re-Labeling*，Guangzheng Hu 等，ICML 2026；arXiv 2026-06-24 | §4 明确服务器聚合本地 updates 得到 θ_global；Appendix B.1 明确 balanced test evaluates global model | [guangzhengh/FedReLa](https://github.com/guangzhengh/FedReLa) 公开且 README 标明官方 ICML 2026；有 FedETF/FedLoGe 训练入口及 util | 适合作为新数据级对照：在指定轮次用当前全局模型的本地 posterior 重标记**已标注训练样本**，再照常训练/聚合。无需训练分类器权重，因此固定头原则上可用；原文有 FedETF + FedReLa，支持固定头兼容性。尚未验证 CLIP-LoRA 性能 |
| MMC-FL | *Rethinking sparse supervision on federated long-tailed learning*，Yizhi Zhou 等，Knowledge-Based Systems 342:115819，2026-06-07；DOI 10.1016/j.knosys.2026.115819 | publisher 明确 global accuracy/global classifier optimization，非个性化目标 | publisher 仍写 “Code will be made publicly available upon acceptance.”；准确题名、MMC-FL、作者主页交叉检索未找到官方实现链接，不能说已有代码 | 动机近，但结构差异大：SE-RGM 对可训练 classifier 的 missing-class rows 屏蔽梯度，并用 simplex ETF 合成特征做 global-to-local classifier KD；SCC 按 Gaussian-kernel classifier similarity 协作聚合，backbone 用 FedAvg。固定文本头没有可训练 class rows，直接接入会退化；需增设可训练共享分类器/明确 adapted variant |
| FedNPC | *FedNPC: Stochastic Noise-driven Post-hoc Classifier Calibration Method for Federated Long-tailed Learning*，Jintong Gao 等，**CVPR Findings 2026**，7737–7746 | 官方代码对 global_model 的 classifier 做 post-hoc calibration，并 global_eval | [JintongGao/FedNPC](https://github.com/JintongGao/FedNPC) 公开，FedAvg-FedNPC.py 可读 | 随机高斯特征+均匀随机标签，冻结骨干，仅重训分类器以校正范数。CLIP 文本权重归一化时，范数机制无法原样作用；需额外可训练线性头。更适合扩展后处理对照 |
| FedSM | *FedSM: Robust Semantics-Guided Feature Mixup for Bias Reduction in Federated Learning with Long-Tail Data*，Jingrui Zhang 等，IEEE Internet of Things Journal 2026；arXiv 2025-10-31；DOI 10.1109/JIOT.2026.3652363 | 论文 Algorithm 1 输出 w_global；§3.3 Eq. 2 对本地模型加权聚合 | [DistriAI/FedSM](https://github.com/DistriAI/FedSM) 官方仓库公开；论文脚注给此链接 | VLM 语义引导选择 mixup 类别，本地特征和全局原型混合，末几轮重训 classifier。VLM 教师/原型通信/可训练头是核心额外结构；不能因使用 CLIP 语义就当成原生 CLIP-LoRA 方法 |

## FedReLa 的专项核验

这是**全局模型引导的训练标签自举**，并非另训个性化头。论文 §4 的 Algorithm 1 先由收到的全局模型推断本地训练样本 posterior，计算 class-wise z-score 和 tanh 重标记概率，再结合本地类别样本数差异，只允许较多样本类向较少样本类重分配。原始尾类标签被保护。一次重标记后，后续轮次继续使用修改后的本地标签。全局模型继续由参数更新聚合得到，推理无需客户端专属模型。

注意：§3 理论分析用了 posterior aggregation；Appendix A 解释这只是为推导 decision boundary，实际 §4 流程明确 parameter update aggregation，不能据理论写成“推理集成”。有固定 ETF head 的基线与 +FedReLa 实验，因此方法本身不要求训练头；移植可用 CLIP logits 得 posterior，继续只训练 LoRA。需要预先固定重标记轮次和阈值，并记录标签更改率；论文未证明这在 CLIP-LoRA 上同样有效。

支持短句（原文 Appendix B.1）："A balanced test dataset is used to evaluate the overall accuracy performance of the global model."

Primary：[论文全文](https://arxiv.org/pdf/2606.26037)，[arXiv 日期/作者](https://arxiv.org/abs/2606.26037)，[ICML 2026 官方列表](https://icml.cc/Downloads/2026)，[官方代码](https://github.com/guangzhengh/FedReLa)。ICML 列表与官方仓库相互印证；第三方网站出现 ICLR 2026 归属，不能采用。

## 可复现性细节

- FedNPC 官方 [FedAvg-FedNPC.py](https://raw.githubusercontent.com/JintongGao/FedNPC/main/FedAvg-FedNPC.py) L132–160 直接产生 Gaussian features 和 random class labels、冻结非 classifier 参数；L165–174 用 data_global_test 每轮评估并保存 test accuracy 最优的校准状态。若纳入大表，必须改成预定校准步数或 validation 选参，保持所有方法统一评估协议。
- FedWCM：*FedWCM: Unleashing the Potential of Momentum-based Federated Learning in Long-Tailed Scenarios*，Tianle Li 等，ICPP 2025，arXiv 2025-07-20。共享模型/全局 momentum 修正明确，针对全球及每轮分布动态调节动量，和 B 更新重组的对照价值高。但目前官方 [FedWCM-Supplement](https://github.com/Li-Tian-Le/FedWCM-Supplement) 仅 PDF、README，无可运行实现；“有官方 GitHub”不等于“开源实现”。[论文全文](https://arxiv.org/html/2507.14980v1)。
- SFD：*Tackling Federated Long-Tailed Learning via Synthetic Feature-Based Decoupled Training*，Huabin Zhu 等，KDD 2025，4168–4179，DOI 10.1145/3711896.3737143。作者 [CV](https://xenialll.github.io/assets/cv/CV_xinting_liao.pdf) 核实题名/发表。ACM PDF 返回 403，未找到官方代码；本报告不根据第三方摘要断言方法细节，暂缓实现推荐。

## 日期与范围纠偏

- FedLF *Adaptive Logit Adjustment and Feature Optimization in Federated Long-Tailed Learning* 是 **ACML 2024**，PMLR 260 论文集标 2025；arXiv 2024-09-18，官方 repo 也写 ACML '24。可作近年 baseline，但不应宣传为“2025 新方法”。[PMLR](https://proceedings.mlr.press/v260/lu25a.html)，[官方代码](https://github.com/18sym/FedLF)，[全文](https://arxiv.org/html/2409.12105v1)。
- FedCART 为 CVPR 2026 主会，但任务是 long-tailed **federated adversarial training**，不作为当前普通分类大表主推。[CVF](https://openaccess.thecvf.com/content/CVPR2026/html/Qin_FedCART_Tackling_Long-Tailed_Distributions_in_Federated_Adversarial_Training_via_Classifier_CVPR_2026_paper.html)。
- FEDTAIL 是 ICML 2025 **workshop**，重点 long-tailed domain generalization；不应当 ICML 主会同任务方法。[作者主页](https://sunnyinai.github.io/index.html)。
- FedPuReL / *Fine-Tuning Impairs the Balancedness of Foundation Models in Long-tailed Personalized Federated Learning*：CVPR 2026，明确 personalized，本次排除。[CVF 全文](https://openaccess.thecvf.com/content/CVPR2026/papers/Hou_Fine-Tuning_Impairs_the_Balancedness_of_Foundation_Models_in_Long-tailed_Personalized_CVPR_2026_paper.pdf)。

## 其余主要 primary links

- FedYoYo [ICCV 正式全文](https://openaccess.thecvf.com/content/ICCV2025/papers/Yan_You_Are_Your_Own_Best_Teacher_Achieving_Centralized-level_Performance_in_ICCV_2025_paper.pdf)，[可读 arXiv 全文](https://arxiv.org/html/2503.06916v1)。支持：Fig. 2 的 “Acc denotes global model accuracy”；§2.2–2.4 给 ASD、DLA。
- MMC-FL [publisher abstract/introduction](https://www.sciencedirect.com/science/article/abs/pii/S0950705126005459)，[作者主页](https://jxiao.wang/) 记载 2026-03 接收；出版社卷期日期 2026-06-07。publisher 页面可检索摘要/引言，全文下载未获；仅报告已核实结构，不声称读过完整算法。
- FedNPC [CVF Findings 正式页](https://openaccess.thecvf.com/content/CVPR2026F/html/Gao_FedNPC_Stochastic_Noise-driven_Post-hoc_Classifier_Calibration_Method_for_Federated_Long-tailed_CVPRF_2026_paper.html)。
- FedSM [arXiv 元数据](https://arxiv.org/abs/2510.27240)，[作者论文全文](https://arxiv.org/pdf/2510.27240)；metadata 明确 journal reference IEEE Internet of Things Journal, 2026。

本次未检索到足以新增推荐的 AAAI/IJCAI/NeurIPS 2025–2026 同任务新方法；这不是“不存在”的断言。检索覆盖精确 Fed-LT、imbalanced FL、class absence、会议站点和已找到论文引用/作者主页，优先保留可核实的全局模型方法。
