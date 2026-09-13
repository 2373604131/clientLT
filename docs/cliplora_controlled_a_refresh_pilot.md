# C0–C3：受控共享 A 刷新 pilot 设计

状态：已按用户三项修正实现；仅做语法与静态审阅，未运行训练或冒烟测试。日期：2026-09-13。

运行入口：`python scripts/run_cliplora_a_refresh.py --method c1`，另两组将c1改为c2/c3；c0可用于重跑基线。默认rank4、seed42，每组独立输出到`output/cifar100_LT/a_refresh_pilot/seed42/<method>`，不指定GPU编号，沿用节点调度器分配。

## 1. 研究问题与边界

问题：在相同正常 B-only 训练、相同刷新时机、相同额外训练预算下，用客户端私有的局部功能缺口指导 A 刷新，是否优于普通 CE 刷新 A，并优于仅追加 B 训练？

现有结果支持把稳定性—可塑性作为工作假设，但尚未证明 fixed-A 的性能瓶颈就是方向不可变。rank4 初始 Tail=66.35、最终67.50，峰值68.05后仍有回落；历史 AB 可训练配置与当前存在聚合/精度等差异。不能直接把历史对比当成冻结 A 的因果实验。

本地 B-only 能改善某类，也不证明该类必须学习新 A 方向：该改善本身已经由旧 A 子空间内的 B 得到。本轮应称为“局部功能差异指导 A 更新”的 pilot；新方向必要性、Client-LT 特异性、动态门控与正式隐私保证都不是本轮已经能证明的命题。

## 2. 组别

| 组别 | 每轮正常阶段 | 每10轮的额外阶段 | 额外上传 | 作用 |
|---|---|---|---|---|
| C0 | 冻结当前 A，训练 B 3 epochs，FedAvg B | 无 | 无 | 已有 rank4 基线 |
| C1 | 同 C0 | 冻结 A，普通 CE 训练 B 1 epoch | delta B | 控制额外训练机会 |
| C2 | 同 C0 | 冻结共同全局 B，普通 CE 训练 A 1 epoch | delta A | 控制间歇开放 A |
| C3 | 同 C0 | 冻结共同全局 B，缺口加权 CE 训练 A 1 epoch | delta A | 检验功能缺口信号 |

主比较 C3–C2，必要比较 C3–C1，参考比较各组–C0。所有刷新都按固定日程，不加动态 gate、access 调权、PFRF、私有持久 B、投影约束、回放 memory、SVD 或自动回滚。

## 3. 固定配置

- CLIP ViT-B/16，vision，top3，第10/11/12个块，仅 q/v。
- rank=4，alpha=1，代码 scaling=alpha/sqrt(rank)=0.5，dropout=0，FP32。
- 原始骨干、文本编码器、提示参数和其他非 LoRA 参数均冻结。
- 30客户端，全参与，100正常轮，每客户端正常3 epochs，train batch32，test/probe batch64，drop_last=False。
- 正常阶段继续使用现有 SGD：lr=0.001、momentum=0.9、weight_decay=0.0005，无warmup，恒定学习率；每客户端重置正常优化器。
- 种子42首轮筛查；沿用当前数据划分与客户端日程。新种子确认必须成对运行，且先固定协议，不边看测试结果边改参数。
- 全部服务器权重始终 q_k=n_k/sum_j n_j，不使用类别频次分配服务器权重，不按gap重加权客户端。
- 刷新轮号是1-based的10/20/30/.../90，共9次；第100轮不刷新。每次刷新后都有正常B-only训练消化新A。发生在该轮正常B聚合之后，正式轮末评估在刷新聚合之后。
- 刷新阶段新建独立 SGD：lr=0.001、momentum=0.9、weight_decay=0、无scheduler、1 epoch。C1/C2/C3相同；C1更新B，C2/C3更新A。
- 刷新weight_decay=0是有意设计：避免没有缺口信号时仍因L2缩小A；三组刷新阶段统一。刷新动量不沿用正常B动量，也不跨客户端或刷新轮保留。
- server_refresh_lr=1：服务器对客户端delta直接按q平均，不加额外server optimizer、裁剪、范数归一化或插值门控。

这是一套保守的起始超参数，不保证A更新量与B更新量一样大，也不保证已经是最优学习率。是否真正产生有效 A 更新必须由日志验证。

## 4. 一轮的精确定义：先 B，后刷新，两个阶段分别聚合

令轮开始模型为 M_pre=(A^t,B^t)。A^t在C2/C3中可以是上一次刷新的结果，不能每轮恢复成初始A0。

### 正常阶段（全部四组相同）

1. 所有客户端从同一个M_pre出发；A^t冻结。
2. 用本地真实训练集D_k完成正常3 epochs，得到本地B_k^CE及M_k^CE=(A^t,B_k^CE)。
3. 正常上传/收集B_k^CE，服务器计算 B_bar=sum_k q_k B_k^CE。
4. 得到共同中间模型 M_bar=(A^t,B_bar)。

刷新轮在客户端临时保留其teacher的评估统计，不需要训练一个额外teacher。此teacher就是已有正常训练的结果，没有跨轮持久模型或回放数据。

### 刷新阶段

所有客户端必须重新加载同一份M_bar，而不是在自己的M_k^CE上直接继续修改A。

C1：固定A^t，B从B_bar开始，用CE训练1 epoch，上传delta B_k=B'_k-B_bar；服务器设 B^{t+1}=B_bar+sum q_k delta B_k，A^{t+1}=A^t。

C2/C3：固定B_bar，A从A^t开始，用对应loss训练1 epoch，上传delta A_k=A'_k-A^t；服务器设 A^{t+1}=A^t+sum q_k delta A_k，B^{t+1}=B_bar。

之后恢复B-only训练模式，丢弃刷新优化器，下轮从新的全局A/B开始。

### 为什么必须共同 B

正常阶段：sum q_k B_k^CE A^t = B_bar A^t。

A刷新阶段：sum q_k B_bar A'_k = B_bar(sum q_k A'_k)。

两阶段的矩阵加权平均都精确，不引入分别平均不同A_i/B_i的乘积偏差，不需要rank压缩。精确的是每个LoRA权重矩阵的平均，不是整个非线性网络预测或准确率的平均，也不保证尾类不下降。

错误实现是每客户端从自己的B_k^CE出发更新A_i，再分别平均A_i和B_k^CE；这会重新混入我们希望排除的A/B搭配问题。

## 5. 客户端功能缺口

### 5.1 数据和指标

第一版使用客户端现有训练集D_k，不新增memory、不取公共数据、不访问全局测试标签；用确定性eval transform、model.eval()、no_grad()评估。此次采用用户允许的小样本回退方案，不声称拥有独立held-out probe：核对seed42发现30个客户端都存在单样本类别，3个尾类客户端仅有46/39/45张样本、15/17/18个类别。按类完整留出会移走某些类别唯一训练样本，因此当前实现不改正常或刷新训练集，保留现有C0的可比性。

F_kc(M)=-mean_{(x,y) in D_kc} CE(M(x),y)，也就是类别c的平均真实标签log probability。使用完整100类softmax，不屏蔽本地缺失类。

优先负CE而不是准确率：某个客户端每类可能只有1–2张图，准确率差会过于离散。指标统一命名为training-set functional gap proxy，不称为“未兑现的真实知识”。它包含本地拟合优势，不是独立泛化证据。仅让probe避开额外刷新、但仍参加teacher的正常B训练，也不能完全消除这一问题；若采用真正独立probe，需重新明确划分及匹配C0，不能暗中从当前尾类少量训练样本中抽走数据。

### 5.2 两种不同的gap，必须明示定义

用户原始定义 h_kc=[F_kc(M_k^CE)-F_kc(M_pre)]_+：本地训练增益，回答本地能改善多少。

本协议建议C3实际采用 g_kc=[F_kc(M_k^CE)-F_kc(M_bar)]_+=[CE_kc(M_bar)-CE_kc(M_k^CE)]_+：在正常B聚合之后仍存在的本地teacher优势，更贴近“刷新当前共享模型尚未兑现的部分”。

这是对原始公式比较时点的明确调整，不是同一个量换名字。若共享B聚合已经补上本地优势，g会降为0。若本地teacher本身没有优于轮初模型，g仍可能为正，因此它不是“本轮新知识”的证明。h只作诊断，不再叠加h>0筛选或动态门控。

每个刷新轮的C1/C2/C3都用同样3次私有探测：M_pre、M_k^CE、M_bar。C1/C2计算同样的统计但不用于更新，匹配额外前向开销。使用同一批本地样本及确定性视图；M_pre和teacher统计可在正常阶段前后计算，M_bar统计在广播后计算。不需要保留三个完整模型。

g在刷新epoch开始前算一次，detach后保持固定，不在epoch内重算，不通过teacher或gap反向传播。未在本地出现的类不参与本地统计。

## 6. 避免类别均衡混淆：建议的C3损失

设 L_kc(theta) 是本地类c的平均CE，pi_kc=n_kc/n_k，只在本地计算。

普通CE：L_CE=sum_c pi_kc L_kc。

用户原始的 sum_c g_kc L_kc / sum_c g_kc 在g对所有类相等时退化成类均衡CE，而不是普通样本平均CE。直接比较这个C3和普通CE的C2，会同时改变类别均衡与gap信号。

为了四组pilot只检验gap，使用频率校准版本：

L_gap = sum_c pi_kc g_kc L_kc / (sum_c pi_kc g_kc + epsilon)，epsilon=1e-12。

设 Z_k=sum_c pi_kc g_kc，a_kc=g_kc/(Z_k+epsilon)。mini-batch损失为 mean_{j in batch} a_k,y_j CE_j。Z是整个客户端训练集的固定统计，不能每个batch用其中的权重和重新归一化，也不做类均衡采样。这样g为相同正数时近似退化为普通CE，差别只剩按gap相对重加权。

现有 fixed_denominator_cross_entropy 支持逐样本loss_weight与实际batch分母，可以直接复用。

该归一化保留客户端内各类gap的相对大小，但消除共同的绝对尺度。例如g=(0.001,0)与g=(0.5,0)在同一类频率下得到近乎相同的权重。因此C3检验的是gap-guided class direction，而不是gap越大A刷新就越强；g不改变客户端服务器权重q。不加入用总gap控制幅度的第二版机制。

这是已经采用的频率校准定义。若改回原始类均衡形式，应增加C2-balanced（类均衡CE刷新A）作为归因对照；否则C3优于C2不能独立归因于gap。

### 无正gap的客户端

所有g=0时，C3定义loss=0，delta A=0，不退回普通CE。刷新阶段独立优化器、初始动量为0且weight_decay=0，因此零梯度不会改变A。可仍执行同等数量的前向/反向/optimizer.step以保持计算预算；日志区分step调用与有非零有效更新。

服务器仍使用全部参与客户端的原始q，不对非零delta客户端重新归一化。不增加动态的刷新时机或客户端选择；零信号对应零更新是loss定义的一部分。

本版本没有gap幅度阈值；很小但非零的gap可能经归一化放大噪声，应记录gap总量、非零类数和最大样本权重。先不加入裁剪/阈值，不能把归一化后的大权重误解为强证据。较大的局部改善也可能来自易拟合或易记忆的样本，而非最缺少监督的类；gap没有天然的Tail安全保证。

## 7. 公平预算及成本

已知sum_k ceil(n_k/32)=352。

- 正常阶段：100×3×352=105600次optimizer.step。
- 额外阶段：9×1×352=3168次step。
- C1/C2/C3共108768次step，较C0多3%；另有三组相同的私有探测前向。
- rank4所有A与所有B各18432个参数，因此额外阶段更新参数数目相同，FP32单次每客户端delta理论载荷均73728 bytes（72KiB），不含协议头。
- 在原有100次正常B聚合之外，C1/C2/C3各增加9次刷新聚合。不是“没有新增通信”；C0只提供原预算基准。
- C3探测的gap0客户端可能实际更新为0；相同step调用不等于相同有效更新量。
- 相同参数量/step/学习率不等于相同FLOPs、墙钟时间、梯度范数或函数空间步长，A/B反向路径也可能不同。记录实测时间、样本数和有效更新量，不能声称所有计算成本完全一致。

正常与刷新/探测的随机数流隔离，探测和额外刷新不能推进下一轮正常数据增强的随机状态。每次刷新使用由(seed, round, client, phase)确定的相同数据顺序和增强；三组的normal与refresh角色分别对齐。新增观测不改变C0路径；第一轮刷新前1–9轮应与现有rank4结果一致。若重构破坏此性质，应先解释差异，而不是直接把旧C0当完全相同实现。

## 8. 最小机制记录，不增加控制逻辑

每轮继续保存round_metrics与per_class_accuracy。刷新轮额外固定评估三点：normal_pre、after_B_aggregation、after_refresh。各点评估只用于事后分析，不做刷新触发、择优或回滚。

刷新收益定义为after_refresh减after_B_aggregation，并分别报告Overall/Non-tail/Tail。另记录下轮正常训练后的变化，区分立刻收益与B重新适配后的收益。

服务器无类别信息的日志：轮号、阶段、q、delta范数、有效LoRA更新范数、聚合前后模型hash、正常/刷新step数、样本量、运行时间、A/B冻结状态。

客户端私有调试日志：类id、n_kc、F_pre/F_teacher/F_bar、h/g、样本权重、正gap类数；默认不作为服务器协议字段。实验模拟器可在授权离线分析阶段汇总，必须与算法输入隔离。

必须记录客户端频率加权gap均值g_bar=sum pi*g及对应deltaA范数，保存于`private_diagnostics/client_<id>/refresh.csv`；每轮`a_refresh_summary.csv`汇总非零gap客户端数、g_bar均值/最大值/按q加权均值、g_bar与deltaA范数及有效更新范数的Pearson相关系数。方差为0时相关系数留空，不把未定义关系写成0。日志只用于诊断，不反馈到聚合系数。

### A真的改变了方向吗

不能只看norm(A_new-A_old)，因为缩放和行空间内变换也会改变A。

对每个层用QR得到A_old行空间的正交列基Q，P=QQ^T，额外记录：

- A-refresh的实际矩阵变化 D=s B_bar delta A；B-refresh的变化 D=s delta B A_old。
- A-refresh中新输入方向分量 D_new=s B_bar delta A(I-P)，报告各层及拼接后的norm(D_new)、norm(D)，以及两者比值。
- A的初始/相邻刷新前后行空间角度或投影距离。

这能区分方向改变与单纯参数漂移；不把该指标用于梯度投影或门控。没有发现有效D或D_new时，C2/C3无收益不能证明新方向没用，可能只是这套刷新设置没有产生足够改变。

## 9. 判定规则

主指标：预先固定的第81–100轮平均Overall/Non-tail/Tail；补充最终、Tail峰值回落及所有刷新事件即时影响，不按测试峰值选checkpoint。

建议的pilot继续投入门槛（工程目标，不是统计显著性）：

- C3相对C2的末20轮Non-tail至少+0.5个百分点；
- Tail不低于C2及C0各自末20轮Tail减0.2个百分点；
- Overall高于C2，且与额外B预算C1相比没有明显劣势；
- 日志确认存在非零有效A更新，改善不是错误预算或不同数据划分产生。

若希望强主张“C3优于额外B训练”，还需C3在预定主指标上优于C1，而不能用“没有明显劣势”代替优越性证据。0.5/0.2是建议预注册容忍度，可在运行前修改，不能看结果后更换。

| 结果模式 | 合理解释 |
|---|---|
| C1≈C2≈C3且都优于C0 | 额外训练机会可能足够；无信号优势证据 |
| C2≈C3且优于C1 | 间歇A刷新有用；gap暂未体现额外价值 |
| C3优于C2和C1且Tail保持 | 支持gap引导A刷新值得继续验证 |
| C3优于C2但不如C1 | gap改善了A刷新，但没有证明应该选择A刷新 |
| C3仅Tail提高、Non-tail不动 | 更像另一种保留/重加权，不是可塑性恢复证据 |
| C2/C3的A有效更新接近零 | 刷新力度未验证，不能否定研究假设 |
| C2/C3每次刷新都明显伤害Tail | 稀疏更新A不自动带来安全性，需要重新检查loss和步长 |

C3若获胜，第二阶段优先验证gap匹配是否重要（如打乱gap对照）或同gap用于B是否也有效；这两者都不放进第一版。动态gate和与Static的组合再后置。新颖性需要专门相关工作检索，不能只凭C3赢C2宣布方法创新已成立。

## 10. 隐私边界与当前代码现实

可成立的说法：A刷新规则不需要服务器读取客户端类别id、逐类数量、gap向量或Tail标签；服务器只需要刷新delta和已有q。q需要客户端总样本量，不是各类样本量。普通阶段照常上传B，不能把“刷新只上传delta A”写成整个算法只上传delta A。

不能成立的说法：只上传delta A就保证服务器不知道类别或无法推断数据。梯度及模型更新可能泄漏信息，见[Deep Leakage from Gradients](https://proceedings.neurips.cc/paper_files/paper/2019/hash/60a6c4002cc7b29142def8871531281a-Abstract.html)及[Deep Leakage from Model in Federated Learning](https://proceedings.mlr.press/v234/zhao24b.html)。安全聚合与差分隐私需要单独威胁模型和实现，本轮不默认加入，也不声称形式化隐私保证。

当前federated_main.py会在模拟器中收集client_class_counts并写入诊断文件；cliplora_v2即使为fedavg也会求解基于类别统计的target weights。因此新pilot应设cliplora_v2=off，B/A聚合直接使用sample_weighted_client_weights，并把类别统计和Tail评估移到独立离线诊断接口。服务器更新函数签名不接收这些统计。若仍保留集中诊断，只能声称“算法不依赖这些字段”，不能声称当前模拟器进程从未见过类别信息。

## 11. 已实现代码与使用

新增模块`utils/cliplora_a_refresh.py`与启动脚本`scripts/run_cliplora_a_refresh.py`；在`federated_main.py`接入独立刷新阶段。不启用旧PFRF/selective_sync算法，仅复用已有RNG状态工具和普通ClipLora优化步骤。

模块职责：

1. 按名字显式枚举全部A键/B键，不能仅依赖当前requires_grad，因为普通阶段A是冻结的。
2. 构建本地训练样本的确定性probe loader，计算per-class CE。
3. 计算stop-gradient的gap和固定客户端归一化权重。
4. 切换A-only/B-only的requires_grad，并创建阶段独立优化器；切换前清空过期梯度，正常阶段恢复B-only。
5. 从共同中间模型运行1 epoch刷新并导出目标参数delta。
6. 计算上述不参与决策的诊断。

federated_main.py：在正常客户端循环中收集本地teacher探测统计，在正常B聚合后插入刷新客户端循环和单独delta聚合。调用已有aggregate_lora_state可实现指定A键/B键的加权平均，不走effective_svd。

`trainers/cliplora.py`无需修改：模块独立创建刷新SGD，不替换trainer注册的正常optimizer；复用cliplora_optimizer_step与fixed_denominator_cross_entropy传入loss_weight，不通过完整trainer.train()意外重复before_train或重置模型。下一正常客户端仍调用原有reset_optimizer_and_scheduler。

`a_refresh_last.pth.tar`保存全部LoRA A/B（即便当时A.requires_grad=False）、初始LoRA、已完成轮号、预算计数及RNG状态。服务器当前A跨轮持久。只在完整轮末保存，以免恢复后重复刷新。恢复时使用同一method/seed/rank/协议，并指定新输出根目录，避免中断轮的零散日志混入恢复结果，例如`python scripts/run_cliplora_a_refresh.py --method c3 --resume <checkpoint> --output-root output/cifar100_LT/a_refresh_resumed`；后续分析按轮号拼接两段结果。

新增主入口CLI：--a_refresh_variant c0/c1/c2/c3，--a_refresh_interval 10，--a_refresh_epochs 1，--a_refresh_lr 0.001。启动脚本对应--method、--refresh-interval、--refresh-epochs、--refresh-lr；支持--rank、--seed、--data-root、--output-root、--num-workers。首版gap定义和探测策略固定写入`a_refresh_config.json`，YACS配置另存`resolved_config.yaml`。本轮不需要调整这些默认超参数。

伪代码：

```text
for round in 1..100:
    pre = global_model
    for client in fixed_schedule[round]:
        load(pre); freeze_A_train_B()
        if refresh_round and variant != c0: private_probe(pre)
        normal_train(3_epochs)
        if refresh_round and variant != c0: private_probe(local_teacher)
        upload_B()
    middle = aggregate_B_with_sample_weights()
    if refresh_round and variant != c0:
        for client in fixed_schedule[round]:
            load(middle); private_probe(middle)
            calculate_private_gap()  # C1/C2 only observe, C3 uses it
            choose_B_for_c1_else_A()
            refresh_one_epoch_from_same_middle()
            upload_only_target_delta()
        global_model = aggregate_refresh_deltas(middle, original_q)
    else:
        global_model = middle
    restore_B_only_mode()
    evaluate_and_checkpoint_round_end()
```

### 输出清单

- `command.json`、`resolved_config.yaml`、`a_refresh_config.json`：实际运行命令与协议。
- 现有`round_metrics.csv`、逐类CSV、数据划分和聚合权重文件。
- `a_refresh_budget.csv`：正常/刷新阶段逐客户端预算。
- `a_refresh_stage_metrics.csv`、`a_refresh_stage_per_class.csv`：刷新轮三个时点评估。
- `a_refresh_summary.csv`：聚合更新、方向变化及gap关系。
- `private_diagnostics/client_*/class_gap.csv`和`refresh.csv`：本地私有诊断，非服务器算法输入。
- `a_refresh_progress.json`、`a_refresh_last.pth.tar`：进度及完整A/B恢复状态。

gap比较post-aggregation模型，C3保留样本频率先验；本轮只指导客户端内更新方向，不按总gap调节客户端权重或刷新强度。
