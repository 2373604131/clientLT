# 可用于论文的图注

文中的A、B指LoRA的两个因子；LA指logit adjustment。准确率用%，回落或差值用pp（百分点）。以下各图采用同一组CIFAR100-LT实验；英文图可以直接配合这些图注使用。

## Figure 1 候选：三联总览

**中文：** 相同的全局类别数量，在不同客户端分配下呈现不同的尾类保持轨迹；继续更新 A 也可以同时改善尾类与非尾类表现。(a) Client-LT与标准Dirichlet（β=0.5）使用同一批10,847张训练图片，每类数量相同；热图展示20个尾类在30个客户端的真实样本分配，并共用计数色标。(b) 使用相同LA校正和S更新策略时，Client-LT与Dirichlet的尾类峰后回落分别为3.45和0.90 pp；曲线未经平滑，S在第91–100轮仅训练B，灰色区域标示这一阶段。(c) Client-LT下E2、E3、S和J的第81–100轮平均准确率；E2固定A，E3有9次额外A更新，S有90次额外A更新，J为日常A/B联训加9次额外A更新，箭头表示E2到E3的变化。结果来自seed42的单次运行；S的预算更高，策略比较是描述性的。

**English:** Identical global class counts can accompany different tail-retention trajectories across client partitions, while updating A can improve both tail and non-tail accuracy. (a) Client-LT and standard Dirichlet (β=0.5) share the same 10,847 training images and per-class counts; heatmaps show the allocation of the 20 tail classes across 30 clients on a common count scale. (b) Under the same logit adjustment (LA) and S strategy, the peak-to-final tail drops are 3.45 pp and 0.90 pp, respectively; curves are unsmoothed, and the shaded interval marks rounds 91–100, when S trains only B. (c) Mean accuracy over rounds 81–100 on Client-LT: E2 fixes A, E3 and S use 9 and 90 extra A updates, respectively, and J combines daily joint A/B training with 9 extra A updates; the arrow connects E2 to E3. All results use seed 42; S has a larger training budget, so these strategy comparisons are descriptive.

## Insight 1：客户端分布

**中文：** 全局类别数量相同，并不意味着尾类最终表现相同。(a) 两种划分共享10,847张CIFAR100-LT图片及全部逐类数量；(b) 使用相同LA与S策略时，第81–100轮Tail20平均准确率分别为68.663%和72.298%；(c,d) 热图展示尾类编号80–99在30个客户端上的真实分配，共计153张尾类训练图片，色标一致，浅灰表示0。Dirichlet为β=0.5的标准逐类划分；客户端编号仅在各自划分内有意义。本图为seed42的描述性比较，分配变化同时涉及客户端容量、类别覆盖及本地组合，不能将差异全部归因于单一来源统计量。

**English:** Identical global class counts do not imply identical final tail performance. (a) The partitions share all 10,847 CIFAR100-LT training images and their per-class counts; (b) with the same LA and S strategy, mean Tail20 accuracy over rounds 81–100 is 68.663% for Client-LT and 72.298% for Dirichlet; (c,d) heatmaps show the actual allocation of tail class IDs 80–99 across 30 clients, using a common scale for the 153 tail training images, with zero counts in light gray. Dirichlet denotes the standard class-wise split with β=0.5, and client IDs are local to each partition. This single-seed comparison also changes client sizes, class coverage, and local class combinations, so it does not isolate the effect of any one source statistic.

## Insight 2：已有尾类能力的退化

**中文：** 在任务与全局训练数据不变的联邦训练中，模型已有的尾类能力仍会在后续轮次退化。图中给出S和J在Client-LT及标准Dirichlet（β=0.5）下第0–100轮的原始Tail20曲线，四组均采用LA；S为日常B训练加前90轮的额外A更新，J为日常A/B联训加9次额外A更新。回落定义为第1–100轮尾类组平均准确率峰值减第100轮：S为3.45/0.90 pp，J为12.15/1.80 pp（Client-LT/Dirichlet）。S中灰色区域为第91–100轮仅训练B的阶段；结果来自seed42单次运行，不包含平滑或跨种子置信区间。

**English:** Tail ability already present in the shared model can deteriorate during later federated rounds even when the task and global training data remain fixed. We show raw Tail20 accuracy from initialization (round 0) through round 100 for S and J under Client-LT and standard Dirichlet (β=0.5), with LA in all runs; S trains B daily and adds A updates in rounds 1–90, whereas J trains A/B jointly each round and adds 9 extra A updates. The group-level peak-to-final drop, defined as the maximum over rounds 1–100 minus round 100, is 3.45/0.90 pp for S and 12.15/1.80 pp for J (Client-LT/Dirichlet). Shading in S marks rounds 91–100 with B-only training; all curves are single-seed observations (seed 42), without smoothing or cross-seed confidence intervals.

## Insight 3：如何更新比是否永久冻结更关键

**中文：** 永久冻结 A 并非获得较好尾类表现的唯一途径，更新策略影响学习与保持的结果。(a) Client-LT下，E3相对固定A的E2在末20轮Non-tail80和Tail20准确率上分别提高1.53和1.61 pp；(b) 各策略达到的尾类峰值与最终值同时展示，以区分能力水平与峰后回落。E2为固定A及9次额外B更新，E3为9次额外A更新，S为90次额外A更新，J为日常A/B联训加9次额外A更新，均使用LA与seed42。E2/E3/J各有108,768次本地优化步，S为137,280次，因此本图比较完整训练策略，并不证明单独增加更新频率的因果效果。

**English:** Permanently freezing A is not the only way to obtain strong tail performance, and the update strategy affects both adaptation and retention. (a) On Client-LT, E3 improves mean Non-tail80 and Tail20 accuracy over rounds 81–100 by 1.53 pp and 1.61 pp relative to fixed-A E2; (b) peak and final tail accuracy are shown together to distinguish attained ability from subsequent decline. E2 fixes A and adds 9 extra B updates, E3 and S add 9 and 90 extra A updates, respectively, and J combines daily joint A/B training with 9 extra A updates; all runs use LA and seed 42. E2/E3/J each use 108,768 local optimizer steps, compared with 137,280 for S, so this is a descriptive comparison of complete strategies rather than an isolated causal test of update frequency.

## 补充图：逐类回落

**中文：** 尾类组平均曲线之外，各个尾类也呈现不同程度的峰后回落。每个点对应一个尾类，独立计算该类在第1–100轮的准确率峰值与第100轮之差；箱体为类别间四分位区间，横线为中位数，须延伸至1.5倍四分位距内的最远观测，所有类别以散点完整呈现。逐类回落的平均值不等于组平均曲线回落；各类峰值也可能受短期波动影响，因此该图是补充性描述，不能替代原始轨迹。结果来自seed42，20个类别不是20次独立实验。

**English:** Individual tail classes exhibit heterogeneous peak-to-final declines beyond what the group-average curve reveals. Each point represents one tail class and measures its own maximum accuracy over rounds 1–100 minus its round-100 accuracy; boxes show class quartiles, lines show medians, and whiskers reach the most extreme observations within 1.5 interquartile ranges, with every class displayed as a point. The mean of class-wise drops differs from the drop of the group-average curve, and individual peaks may reflect short-term fluctuations, so this plot complements the raw trajectories. Results come from seed 42; the 20 classes are not 20 independent experimental runs.
