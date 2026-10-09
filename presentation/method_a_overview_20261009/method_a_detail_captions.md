# 方法 A 右侧细节图

## 中文图注

**方法 A 在普通聚合后，利用客户端更新的类别影响和历史预测间隔构造保持目标，再对共享 LoRA A 进行三步修正。** ① 对每个非空“客户端—类别”组合 q，在共同模型上计算间隔梯度，与各客户端的更新做内积，保留原图与水平翻转两视图均为正的影响；正向影响的均值产生当前目标，其集中程度决定保持损失的权重。② 历史记录取已提交模型连续五轮、两个视图的最低间隔，满足改善条件后登记；保持目标取有效当前目标与历史目标中的较大者。③ 候选模型从普通聚合结果初始化，对未达到保持目标的间隔缺口施加带权惩罚，并对相对普通聚合结果的整体 LA 分类损失上升施加软惩罚；CLIP 与 B 固定，梯度仅更新 A，限制修正范围并提交第三步。正负矩阵、柱形和缺口都是机制示意；为清晰省略目标的视图下标、保持项的尺度归一化与视图求和，以及完整目标中的系数和邻近正则。

## English caption

**Method A constructs retention targets from class-wise client-update effects and historical margins, then corrects the shared LoRA A after ordinary aggregation.** (1) For every nonempty client–class pair q, margin gradients at the common model are paired with client updates; effects are retained only when positive in both the original and horizontally flipped views, with their mean determining the current target and their concentration determining the retention-loss weight. (2) Historical targets record the minimum margin over a nonoverlapping five-round block of committed models and both views, subject to the improvement condition; the larger valid current or historical target is used. (3) Starting from the ordinary proposal, the method penalizes target shortfalls and increases in the overall LA classification loss relative to that proposal, while keeping CLIP and B fixed and applying three bounded updates to A. Matrices and bars are schematic; view indices on targets, scale normalization, the view sum, objective coefficients, and the proximity regularizer are omitted for clarity.

## 实现口径核对

- 当前目标的上限 2 已在图中标明；正向响应先取两视图的较小值，再筛选。原图和翻转图始终来自同一固定训练图片。
- q 覆盖所有非空“客户端—类别”组合，并不预先只选择尾部类。图用一个 q 说明单个保持项，实际对所有有效 q 和两个视图汇总。
- F 为平均预测间隔；图中正确类与最强错误类两根柱只说明间隔的定义，实际是先逐样本求差、再平均，不是直接用两个平均类别分数作差。
- F 的基准与梯度在 B 已聚合、A 未更新的共同模型上计算。候选 A 的预测间隔在每次修正中重新计算。
- 保持权重由有效正向影响数的倒数经全局归一化得到；图中“越集中，权重越大”表达在其他单位不变时的相对关系，不是修改 FedAvg 的客户端样本量权重。
- 历史柱可理解为各已提交轮次的双视图最小间隔；满足相对原参照至少改善 0.001 后登记，新历史下一轮生效。历史区间互不重叠。
- 分类小图的 C(A) 是按原类频率、客户端样本量权重汇总的训练侧 LA 风险。先全局汇总，再对上升部分取正部和平方；不是对每个客户端的上升分别罚。
- 当前图没有承诺所有目标都达到、整体分类损失必不增加或各类准确率都改善。图展示惩罚项和操作，不展示方法性能。
- 三步修正中，目标、权重、尺度和普通提案固定；优化变量只有 A。完整目标另含邻近普通提案的正则，投影半径由各客户端 A 更新范数的未加权 RMS 给出。
- 图片和损失计算均留在客户端，服务器使用数值反馈共同更新候选；图中未把客户端图片画成上传服务器。

原始依据：docs/cliplora_method_a_full_cp.md。示例输入缩略图沿用 assets/manifest.json 所记录的 CIFAR-100 训练图片。
