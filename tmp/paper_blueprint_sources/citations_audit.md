# 中文论文蓝图引用独立核验

核验日期：2026-10-05。核验方式：paper-writer fresh-context verification ladder 第一级。核验者只接收拟引用表述和参考文献清单，未接收正文写作推理。每篇执行题名、作者加关键词年份、核心关键词至少三类检索，并以原始论文、作者提交的 arXiv 摘要或官方会议页面为证据。下列 VERIFIED 表示题录及所列限定表述均有依据；不代表相关论文的所有结论均被独立复现。

## 1. CAPT — VERIFIED

- 题录：Shihao Hou, Xinyi Shang, Shreyank N Gowda, Yang Lu, Chao Wu, Yan Yan, Hanzi Wang. *CAPT: Class-Aware Prompt Tuning for Federated Long-Tailed Learning with Vision-Language Model*. arXiv:2503.06993, 2025.
- 核验证据：原始摘要明确描述通用提示与类别感知提示，以及基于客户端数据分布的异质性感知聚类。
- 可用表述：CAPT 通过通用提示与类别感知提示构成的双提示机制，以及异质性感知的客户端聚类，处理联邦学习中的数据异质性与长尾分布。
- 边界：按已核实 arXiv 版本列为 2025 年预印本。没有以当前证据擅填会议。
- 来源：[arXiv 原始题录与摘要](https://arxiv.org/abs/2503.06993)。
- 已执行查询：`"CAPT: Class-Aware Prompt Tuning for Federated Long-Tailed Learning with Vision-Language Model"`；`Hou 2025 cluster aware federated long tailed prompt tuning CAPT`；`federated long tailed general category aware dual prompts client clustering CAPT`。

## 2. FedPuReL — VERIFIED

- 题录：Shihao Hou, Chikai Shang, Zhiheng Yang, Jiacheng Yang, Xinyi Shang, Junlong Gao, Yiqun Zhang, Yang Lu. *Fine-Tuning Impairs the Balancedness of Foundation Models in Long-tailed Personalized Federated Learning*. CVPR, 2026, pp. 17505–17514.
- 核验证据：CVPR 官方摘要确认零样本预测引导的局部梯度净化，以及冻结全局模型上的个性化残差；原文 §3.3.1、§3.3.2 进一步给出冲突时的梯度投影和个性化阶段冻结全局分支。用户 PDF 的文本提取 `fedpurel.txt` 也支持上述概括。
- 可用表述：FedPuReL 利用零样本预测引导局部梯度净化，以维护共享全局模型的类别平衡性；随后在冻结全局模型的基础上学习客户端特定的输出残差。
- 边界：不能把“维护平衡性”升格为对所有数据分布都保证严格类别平衡；也不能把零样本知识保持归给待写方法作为首次提出。
- 来源：[CVPR 官方题录与摘要](https://openaccess.thecvf.com/content/CVPR2026/html/Hou_Fine-Tuning_Impairs_the_Balancedness_of_Foundation_Models_in_Long-tailed_Personalized_CVPR_2026_paper.html)；[作者全文 §3.3](https://arxiv.org/html/2605.02247v1)。
- 已执行查询：`"Fine-Tuning Impairs the Balancedness of Foundation Models in Long-tailed Personalized Federated Learning"`；`Hou FedPuReL CVPR 2026 balancedness`；`FedPuReL zero shot gradient purification residual personalization`。

## 3. FedNTD — VERIFIED，建议明确问题范围

- 题录：Gihun Lee, Minchan Jeong, Yongjin Shin, Sangmin Bae, Se-Young Yun. *Preservation of the Global Knowledge by Not-True Distillation in Federated Learning*. NeurIPS, 2022.
- 核验证据：官方摘要同时描述全局模型对前轮知识的遗忘、局部训练对局部分布之外知识的遗忘，并说明 FedNTD 只针对非真实标签类别保留全局视角。
- 可用表述：FedNTD 从遗忘角度研究普通联邦学习中的数据异质性，通过针对非真实标签类别的蒸馏保留全局知识，缓解跨轮训练及局部适应导致的知识遗忘。
- 边界：这里的跨轮遗忘不是对基础模型预训练知识损失的直接研究；其动机类比持续学习，但不能据此改写成专门面向类别增量持续学习的方法。
- 来源：[NeurIPS 官方题录与摘要](https://proceedings.neurips.cc/paper_files/paper/2022/hash/fadec8f2e65f181d777507d1df69b92f-Abstract-Conference.html)。
- 已执行查询：`"Preservation of the Global Knowledge by Not-True Distillation in Federated Learning"`；`Lee 2022 federated not true distillation`；`FedNTD forgetting global knowledge local training non true classes`。

## 4. FFA-LoRA — VERIFIED

- 题录：Youbang Sun, Zitao Li, Yaliang Li, Bolin Ding. *Improving LoRA in Privacy-preserving Federated Learning*. ICLR, 2024.
- 核验证据：原始摘要说明固定随机初始化的非零矩阵，只微调零初始化矩阵。OpenReview 论文首页确认 ICLR 2024。
- 可用表述：FFA-LoRA 固定随机初始化的非零因子 A，仅训练零初始化的因子 B，以改进隐私保护联邦学习中的 LoRA 优化与聚合。
- 边界：“零初始化”指初始状态，不是 B 在整个训练中保持为零。不能笼统说所有单因子更新方法都保持同一随机子空间，因为 FedSVD 会重参数化 A。
- 来源：[arXiv 原始摘要](https://arxiv.org/abs/2403.12313)；[ICLR 官方 OpenReview 论文](https://openreview.net/pdf?id=NLPzL6HWNl)。
- 已执行查询：`"Improving LoRA in Privacy-preserving Federated Learning"`；`Sun 2024 FFA LoRA privacy preserving`；`FFA LoRA freeze randomly initialized non zero matrices`。

## 5. RoLoRA — VERIFIED，参考文献采用正式会议版作者

- 题录：Shuangyi Chen, Yuanxin Guo, Yue Ju, Hardik Dalal, Zhongwen Zhu, Ashish Khisti. *Robust Federated Finetuning of LLMs via Alternating Optimization of LoRA*. NeurIPS, 2025.
- 核验证据：会议全文 §3（PDF 第 4 页）明确规定跨奇偶通信轮交替更新；奇数轮固定 A、训练并聚合 B，后续轮固定 B、训练并聚合 A。
- 可用表述：RoLoRA 在相邻通信轮之间交替冻结和更新 LoRA 的两个因子，并由服务器聚合当轮更新的因子。
- 题录提醒：检索返回的早期 arXiv 作者列表少了 Zhongwen Zhu，且 Dalal 名字拼写与正式版不同；引用 NeurIPS 2025 时采用上述会议版作者列表。
- 边界：这里是通信轮之间的交替，不应改写成单个客户端内同一轮的顺序阶段。
- 来源：[NeurIPS 正式题录](https://proceedings.neurips.cc/paper_files/paper/2025/hash/adf17e7346b6be4c2f2bb40de572e5bc-Abstract-Conference.html)；[会议全文 §3](https://proceedings.neurips.cc/paper_files/paper/2025/file/adf17e7346b6be4c2f2bb40de572e5bc-Paper-Conference.pdf)。
- 已执行查询：`"Robust Federated Finetuning of LLMs via Alternating Optimization of LoRA"`；`Chen Guo 2025 "RoLoRA" alternating`；`robust federated lora alternating freezing communication rounds`。

## 6. FedSVD — VERIFIED

- 题录：Seanie Lee, Sangwoo Park, Dong Bok Lee, Dominik Wagner, Haebin Seong, Tobias Bocklet, Juho Lee, Sung Ju Hwang. *FedSVD: Adaptive Orthogonalization for Private Federated Learning with LoRA*. NeurIPS, 2025.
- 核验证据：官方 NeurIPS 摘要明确说明客户端只优化 B；服务器聚合 B 后，使用上一轮 A 形成乘积 BA，再用 SVD 重新分解为新的 A 和 B。
- 可用表述：FedSVD 在客户端仅更新 B，并在服务器聚合后对 BA 进行 SVD 重参数化，以得到自适应的正交因子 A 和相应因子 B。
- 边界：不能因为本地冻结 A 而把该方法概括为“全程固定随机 A”；A 在服务器阶段发生重参数化。
- 来源：[NeurIPS 正式题录与摘要](https://proceedings.neurips.cc/paper_files/paper/2025/hash/ad922aa85d4027ff3502e8e5f406e828-Abstract-Conference.html)；[arXiv 原始摘要](https://arxiv.org/abs/2505.12805)。
- 已执行查询：`"FedSVD: Adaptive Orthogonalization for Private Federated Learning with LoRA"`；`Lee 2025 FedSVD orthogonalization`；`FedSVD B matrix SVD decomposition A B frozen`。

## 7. LoRA-A² — VERIFIED

- 题录：Jabin Koo, Minwoo Jang, Jungseul Ok. *Towards Robust and Efficient Federated Low-Rank Adaptation with Heterogeneous Clients*. ACL, 2025, pp. 416–429.
- 核验证据：ACL 官方摘要直接说明方法结合交替冻结与自适应秩选择，针对低秩和高数据异质性场景。
- 可用表述：LoRA-A² 将交替冻结与自适应秩选择结合，用于提升异质客户端联邦低秩适应的鲁棒性和通信效率。
- 边界：本轮仅以摘要支持上述机制级概括，不据此断言其具体轮内训练顺序、秩分配公式或与待写方法完全相同。
- 来源：[ACL 官方题录与摘要](https://aclanthology.org/2025.acl-long.19/)。
- 已执行查询：`"Towards Robust and Efficient Federated Low-Rank Adaptation with Heterogeneous Clients"`；`Koo 2025 "LoRA" "heterogeneous clients"`；`"LoRA-A" "alternating" "rank"`。

## 8. FedFV — VERIFIED

- 题录：Zheng Wang, Xiaoliang Fan, Jianzhong Qi, Chenglu Wen, Cheng Wang, Rongshan Yu. *Federated Learning with Fair Averaging*. IJCAI, 2021, pp. 1615–1623.
- 核验证据：官方摘要将客户端不公平与冲突梯度及梯度幅度差异关联，提出在平均之前检测并缓解梯度冲突。
- 可用表述：FedFV 从客户端公平性出发，在聚合前检测并缓解客户端梯度冲突，调整梯度方向与幅度。
- 边界：客户端公平性不能直接等同类别平衡性，客户端梯度冲突也不能不加论证地等同 LoRA 因子语义冲突。
- 来源：[IJCAI 官方题录与摘要](https://www.ijcai.org/proceedings/2021/223)；[会议全文](https://www.ijcai.org/proceedings/2021/0223.pdf)。
- 已执行查询：`"Federated Learning with Fair Averaging"`；`Wang 2021 "FedFV"`；`"FedFV" "gradient" "conflicts"`。

## 9. ShapFed — VERIFIED，建议区分评估与聚合

- 题录：Nurbek Tastan, Samar Fares, Toluwani Aremu, Samuel Horváth, Karthik Nandakumar. *Redefining Contributions: Shapley-Driven Federated Learning*. IJCAI, 2024, pp. 5009–5017.
- 核验证据：官方摘要说明 ShapFed 利用 Shapley 值评估参与者的类别特定影响，并基于该贡献评估提出 ShapFed-WA 加权聚合。
- 可用表述：ShapFed 评估客户端的类别相关贡献，并基于贡献评估提出 ShapFed-WA 加权聚合策略。
- 建议修改：原“以类别相关贡献组织贡献评估/聚合”大体可保留，但以上完整句子更准确，避免让读者误读成按类别逐项实施参数聚合。
- 边界：不能根据摘要把它写成按 LoRA 因子贡献聚合，也不能说它完全没有类别相关处理。
- 来源：[IJCAI 官方题录与摘要](https://www.ijcai.org/proceedings/2024/554)。
- 已执行查询：`"Redefining Contributions: Shapley-Driven Federated Learning"`；`Tastan Fares 2024 ShapFed contribution`；`"ShapFed" "class" "contribution"`。最初另执行了错误作者关键词 Hashemi 查询，发现正确作者后已重新查询。

## 10. LIFT — VERIFIED，结论使用条件性措辞

- 题录：Jiang-Xin Shi, Tong Wei, Zhi Zhou, Jie-Jing Shao, Xin-Yan Han, Yu-Feng Li. *Long-Tail Learning with Foundation Model: Heavy Fine-Tuning Hurts*. ICML, PMLR 235, 2024, pp. 45014–45039.
- 核验证据：PMLR 官方摘要指出，重度微调可能降低尾类表现，并据此提出自适应轻量微调方法 LIFT。
- 可用表述：LIFT 分析了基础模型在长尾学习中的微调行为，发现重度微调可能损害尾类性能，并提出自适应轻量微调方法。
- 边界：不能改写成“任何微调都会降低性能”，也不能仅靠该论文支持联邦训练特有的因果机制。正式发表年份为 2024，早期 arXiv 发布于 2023。
- 来源：[PMLR 官方题录与摘要](https://proceedings.mlr.press/v235/shi24g.html)。
- 已执行查询：`"Long-Tail Learning with Foundation Model: Heavy Fine-Tuning Hurts"`；`Shi 2024 "LIFT" "long-tail"`；`"heavy fine-tuning" "lightweight" "long-tail"`。

## 总体结论与写作限制

十篇均为真实文献，采用上述题录及限定表述后没有未解决的 NOT_FOUND、METADATA_MISMATCH 或 OVERCLAIM 项。CAPT 保持 arXiv 2025 题录；其他会议标注均已在官方来源核实。无需删去引用。

本轮核验不支持任何“这些工作都没有某机制”“首次交替 LoRA”“首次利用零样本锚点”“首次类别相关贡献评估”等排他性新颖性断言。现有证据允许描述各方法做了什么，不能由摘要中未提到某内容推断该内容不存在。拟写方法与这些工作的差异应由具体机制、目标、训练粒度和实验验证来限定。
