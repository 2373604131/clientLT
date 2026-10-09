# 方法 A 模型示意图图注

## 中文

**方法 A 在普通联邦聚合后，根据客户端更新对类别预测的影响和历史预测间隔，对共享 LoRA A 进行修正。** 左：本地图片和类别文本分别经过 CLIP 的图像编码器和文本编码器，计算余弦相似度并使用 LA 分类损失训练 A。中：服务器沿用按样本量加权的普通聚合，得到 A 的初始提案。右：客户端在固定本地训练图片上计算候选模型的预测间隔与损失；客户端更新的一阶影响决定保持权重与当前目标，已登记的历史间隔提供历史目标。服务器根据客户端反馈，对 A 执行三步修正，再提交全局模型。雪花表示冻结，火焰表示训练，橙色虚线表示修正损失对 A 的梯度作用。

**范围与简化。** 图中只展示 A 阶段，B 已在此前的阶段完成训练和聚合；图像主干、文本编码器、文本提示和 logit scale 在 A 阶段均冻结。LoRA A/B 画在图像编码器旁边，表示内部投影层的低秩适配分支，并非一个直接接收原始图片的独立网络。服务器聚合仍使用原样本量权重；图中的保持权重作用于后续修正损失。图片和损失计算留在客户端。

**符号与实现。** q 表示一个非空的“客户端—类别”组合；固定图片来自该组合的训练样本，每组至多 8 张，使用原图和水平翻转两个视图。F 是正确类别与最强错误类别的余弦相似度之差的样本均值，不是准确率。g 在共同的聚合 B、更新前 A 上计算，与各客户端的 A 更新作内积，估计其对本地类别间隔的一阶影响。图中的单个向量示意省略两视图筛选和归一化。历史 H 按连续、互不重叠的五轮区间统计两视图最小间隔，满足改善条件才登记，下一轮生效；T 取有效当前目标和历史目标中的较大者。修正还包括邻近普通提案的正则和范围约束，图中为简洁省略；分类项惩罚相对普通提案的 LA 分类损失上升。

图中输入缩略图取自本地 CIFAR-100 训练数据，只用于说明输入类型，不表示实际抽中的固定样本。所有预测柱、历史柱与向量均为机制示意，不是实验结果或效果保证。图片来源索引见 assets/manifest.json。

## English

**Method A corrects the shared LoRA A after ordinary federated aggregation using class-wise update effects and historical prediction margins.** Left: local images and class text are encoded by CLIP, and A is trained with the LA classification loss. Middle: the server retains ordinary sample-weighted aggregation to form an initial A proposal. Right: clients evaluate candidate models on fixed local training images. First-order effects of client updates determine retention weights and current targets, while registered historical margins provide historical targets. The server uses client feedback for three correction steps before committing the global model. Snowflakes indicate frozen parameters, flames indicate trainable parameters, and dashed orange arrows indicate the correction-loss gradient acting on A.

The figure depicts only the A stage, after B has been trained and aggregated. B, the pretrained backbone, the text encoder, prompt parameters, and logit scale are fixed during this stage. The attached A/B blocks represent LoRA branches inside image-encoder projections. Images and loss evaluation stay on clients. Retention weights apply to the correction loss, while the original sample weights remain unchanged in ordinary aggregation.

For each nonempty client–class pair q, up to eight fixed local training images are evaluated using the original and horizontally flipped views. F is the mean true-class versus strongest-rival cosine margin. Its A-gradient g is measured at the common post-B, pre-A model; inner products with client updates estimate first-order class effects. The small vector illustration omits the two-view filter and normalization. Historical H uses the minimum over both views and a nonoverlapping five-round block, is registered only after sufficient improvement, and takes effect in the next round. The target combines valid current and historical targets by their maximum. The figure omits the proximity regularizer and correction-radius constraint; the classification term penalizes an increase in LA loss relative to the ordinary proposal.

Input thumbnails are illustrative CIFAR-100 training images, not an identified feedback subset. Prediction bars, historical bars, and update vectors are schematic and do not report experimental results.
