# 中文图注

方法 A 根据客户端更新对本地类别预测的影响及历史预测水平，对普通聚合后的全局模型进行修正。各客户端正常训练 LoRA A，并使用每个非空客户端—类别组合的固定训练图片计算两视图预测间隔；普通模型聚合仍按样本量加权。正向更新影响用于计算当前改善目标和保持损失权重，历史目标记录不重叠五轮内正式模型及两个视图持续达到的最低预测间隔；两种目标均存在时取最大值，只有一项时使用该项。每一步修正在本地反馈样本上重新计算损失及梯度，固定执行三步投影梯度下降后提交一个全局模型；样本图标、得分条及响应符号均为示意，不代表实验结果或必然改善。

# English caption

Method A corrects the ordinary global aggregate using client update effects on local classes and historical prediction margins. Clients train LoRA A and measure two-view margins on fixed local training images for each nonempty client–class pair, while ordinary aggregation retains sample-size weights. Positive update effects determine the current target and retention weights; historical targets record sustained minimum margins over non-overlapping blocks of five committed rounds and both views. The maximum is used when both targets are available, and the existing target is used otherwise; local feedback losses and gradients are recomputed during three projected correction steps before committing one global model. Image icons, score bars, and response signs are schematic, not empirical results or guarantees of improvement.

# 符号与概览中省略的实现细节

- q=(k,c)：客户端 k 上的类别 c；v 为原图/水平翻转视图；j 遍历所有参与客户端。
- A 与 B 是 LoRA 因子，不是方法 A 与方法 B 两个算法模块。本图只展示方法 A 的活跃训练轮；B 的本轮普通训练与聚合已完成，随后 B 固定。
- F 是正确类余弦得分减最强错误类余弦得分的样本平均值，不是准确率，也不使用 LA 调整。
- r 是共同起点处的一阶近似。令 a_qj 为两个视图响应的较小值，仅在该值大于 1e-6 时保留，否则置零；u_q 为正 a_qj 的未加权均值。
- N_eff=(sum_j a_qj)^2 / sum_j a_qj^2；保持损失权重在所有激活 q 上归一化，无正向客户端时使用缓存权重。它不等于持有该类别的客户端数量。
- 初始模型仅作为历史首次登记的参照；每个五轮块的最小值超过有效历史（或初始参照）至少 0.001 后才登记，从下一轮开始使用。
- L_ret 是按固定尺度归一化的、对未达到目标的预测间隔施加的单侧平方惩罚；达到目标后不再惩罚。
- L_cls 是本地反馈样本上按原样本比例加权的 LA 分类损失相对普通提案上升时的单侧软惩罚，不保证全局准确率不下降。
- R 是客户端 A 更新范数的未加权 RMS。图中修正范围是相对于普通聚合结果的参数球，不是“客户端更新张成空间”的投影。
- 三步投影梯度下降的步长为 0.1；修正损失系数为 10，分类保持系数为 1。目标、权重和尺度在三步内固定，只对当前候选 A 求导。
- 本图是计算关系概览，不是逐包通信协议。原始图片留在客户端；右栏修正需要客户端反馈计算。

# 当前方法可以承载的叙述

普通聚合可能降低一部分类别在此前训练中取得的预测表现。方法 A 记录持续取得的预测改善，并在聚合后施加保持约束；再根据本轮正向更新影响设置当前改善目标和保持权重。将正向更新集中度作为保持优先级是设计假设，示意图没有把“集中度越高就必然遗忘”画成已证明事实。

历史保持是现有实验支持较明确的部分；当前目标和集中度权重是否值得额外复杂度，需要与简单客户端加权及简单保持对照，不能由示意图的复杂程度证明。
