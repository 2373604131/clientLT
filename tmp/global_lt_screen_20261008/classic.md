2022–2024 全局共享联邦长尾分类 baseline 核验（2026-10-08）

结论：严格保留 CLIP ViT-B/16、视觉 LoRA rank=4、固定文本分类器时，本组最值得先做的是 **FedLF-adapted**。CReFF、Fed-GraB、CLIP2FL 确实针对全局长尾学习，但不能删去各自对分类头的关键操作后仍声称忠实复现。建议传统全局长尾对照保留 CReFF、Fed-GraB；若需要 VLM 相关经典方法，再加 CLIP2FL，并单列忠实保留方法组件的分类头设置。FedGELA 不列入纯全局长尾主比较。

以下“接入判断”是根据原文机制与当前固定文本头协议作出的技术推断，不是原论文实测结果。未运行任何基线或修改训练代码。

| 方法 | 正式题录与全局目标 | 核心机制 / 推理模型 | 当前协议接入判断 |
|---|---|---|---|
| CReFF | *Federated Learning on Heterogeneous and Long-Tailed Data via Classifier Re-Training with Federated Features*, IJCAI 2022, pp. 2218–2224 | 面向局部异质、总体长尾，训练共享全局模型；客户端上传按类平均的分类头梯度，服务器梯度匹配合成均衡特征并重训分类头。最终输出是共享特征提取器＋重训分类头。 | 视觉 LoRA 可替换特征学习参数，但固定文本头不能执行其核心 classifier re-training；需添加/替换为可训练线性分类头并保留梯度匹配、服务器合成特征流程。只保留固定文本头属于实质改变。 |
| Fed-GraB | *Fed-GraB: Federated Long-tailed Learning with Self-Adjusting Gradient Balancer*, NeurIPS 2023 | 全局长尾、客户端异质，目标为统一全局分类器。DPA 从聚合模型分类器各类别权重 L2 范数推断长尾先验，SGB 以闭环控制动态平衡各类正负 logit 梯度；推理为聚合模型。 | SGB 的梯度操作可反传到 LoRA，但归一化固定文本头各类范数相同，DPA 退化为均匀先验；未归一化固定文本范数也不反映训练数据频率。需可学习分类头保留 DPA；换成真实全局标签计数或其他估计器是方法改动且改变信息条件。 |
| CLIP2FL | AAAI 官网题录：*CLIP-Guided Federated Learning on Heterogeneity and Long-Tailed Data*, AAAI 2024, 38(13):14955–14963；论文PDF和作者repo把 Heterogeneity 写作 Heterogeneous，同一工作 | 全局 server model，非个性化。冻结 CLIP 教师向客户端学生做 KL 蒸馏；服务器合成特征由分类头梯度匹配＋CLIP 文本原型对比约束，再重训均衡分类头。最终用聚合学生特征提取器＋重训分类头。 | 原作学生为 ResNet-8/50＋维度匹配 MLP，教师 CLIP ViT-B/32。不是 LoRA CLIP 算法。更换为 ViT-B/16 LoRA 学生需保留独立冻结教师、可训练学生分类头、服务器合成/重训；若使用固定文本头而删除重训就不再是完整 CLIP2FL。 |
| FedLF | *FedLF: Adaptive Logit Adjustment and Feature Optimization in Federated Long-Tailed Learning*, ACML 2024；PMLR 260:303–318, **2025** | 明确学习一个全局模型；客户端三项损失：本地标签分布产生的乘性 logit 调整、类中心紧致/分离约束、特征去相关。模型按样本数 FedAvg；官方代码直接评估 fedavg_params，并不在推理中使用客户端特有头。 | 三个损失在固定文本 logits 与视觉特征上仍有定义，可只更新视觉 LoRA；无需服务器合成数据、可训练推理头或按客户端推理。属于最自然的适配，但应写 FedLF-adapted 并说明原作 ResNet-8 全参训练变为 CLIP+LoRA；局部类别中心不应被误写为个性化推理。 |
| FedGELA | *Federated Learning with Bilateral Curation for Partially Class-Disjoint Data*, NeurIPS **2023**（arXiv 2024 上传并非会议年份） | PCDD 任务，同时优化 generic/personal 两端；全局固定 simplex ETF，训练时按本地分布伸缩 ETF。可输出共享 backbone＋标准 ETF 用于全局，也输出个人 backbone＋调整后的 ETF。 | 它有全局分支，不能说“没有 global model”；但主设定是 PCDD，理论推导明确采用全局类别均衡，不是专为全局长尾设计。固定文本原型一般不具有 ETF 几何；替换成文本头会改变核心。排出本次纯全局长尾优先名单。 |

逐项 primary sources 与证据定位：

- CReFF：[IJCAI 官方页](https://www.ijcai.org/proceedings/2022/308)、[官方 PDF](https://www.ijcai.org/proceedings/2022/0308.pdf)、[论文指向的官方代码](https://github.com/shangxinyi/CReFF-FL)。PDF Sec.3.3、Eq.(3)–(7)、Algorithm 1：梯度针对重训分类器参数，算法输出 re-trained model，而非只输出原 FedAvg 模型。
- Fed-GraB：[NeurIPS 官方页](https://proceedings.neurips.cc/paper_files/paper/2023/hash/f4b8ddb9b1aa3cb11462d64a70b84db2-Abstract-Conference.html)、[官方 PDF](https://proceedings.neurips.cc/paper_files/paper/2023/file/f4b8ddb9b1aa3cb11462d64a70b84db2-Paper-Conference.pdf)、[官方代码](https://github.com/ZackZikaiXiao/FedGraB)。PDF Sec.3.2 DPA 明确使用 classifier weight L2 norm；Sec.3.3–3.4 为 SGB 与训练流程。代码入口 [fed_grab.py](https://github.com/ZackZikaiXiao/FedGraB/blob/main/fed_grab.py) 可见独立 nn.Linear 分类头；未对代码完成可运行性审计。
- CLIP2FL：[AAAI 官方题录](https://ojs.aaai.org/index.php/AAAI/article/view/29416)、[官方 PDF](https://ojs.aaai.org/index.php/AAAI/article/download/29416/30672)、[官方代码](https://github.com/shijiangming1/CLIP2FL)。DOI 10.1609/aaai.v38i13.29416；PDF Fig.2、Method、Algorithm 1 证实两模型及重训头；Implementation 段明确 ResNet-8/50、MLP、CLIP ViT-B/32。
- FedLF：[PMLR 正式题录](https://proceedings.mlr.press/v260/lu25a.html)、[卷260出版信息](https://proceedings.mlr.press/v260/)、[作者 arXiv 全文](https://arxiv.org/html/2409.12105v1)、[官方代码](https://github.com/18sym/FedLF)、[实际 local_train/global_eval 实现](https://github.com/18sym/FedLF/blob/main/algorithm/fedlf.py)。会议 2024-12-05 至 08，PMLR 卷于 2025-01-14 发表；应保存官方 BibTeX year=2025，同时可写 ACML 2024。方法细节依据作者全文及官方代码交叉核验；网页工具未能解析 PMLR 链接的最终 PDF，不声称逐页比对过终稿。
- FedGELA：[NeurIPS 官方 PDF](https://proceedings.neurips.cc/paper_files/paper/2023/file/65b721a1df04c1098567f70d483d6468-Paper-Conference.pdf)、[官方代码](https://github.com/MediaBrain-SJTU/FedGELA)、[作者 arXiv 页](https://arxiv.org/abs/2405.18972)。PDF Sec.3.2 明确 globally balanced 假设；Sec.3.3/Algorithm 1 明确双重输出及推理模型。

推荐顺序：

1. **FedLF-adapted**：当前严格同模型、同固定文本头协议的优先适配对象；保留三项损失，统一 LoRA 参数与聚合方式，报告改变。
2. **CReFF**：经典纯全局长尾机制对照；需允许线性头并完整保留服务器重训组件。
3. **Fed-GraB**：不同机制的纯全局长尾对照；同样需允许可学习头，不能悄悄替换 DPA。
4. **CLIP2FL**：当评审需要 CLIP 相关联邦长尾对照时增补；优先完整教师—学生—重训分类器流程，成本与模型/参数条件需单列。

如果固定文本分类器是完全不可放宽的硬约束，不应强凑本组四个为“同设置忠实 baseline”；保留 FedLF-adapted，再从其他已核验的原生 VLM 全局联邦方法补齐。
