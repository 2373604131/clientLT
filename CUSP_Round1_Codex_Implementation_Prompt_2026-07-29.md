# CUSP 第一轮生死实验：Codex 实现 Prompt

## 编译说明

这份 Prompt 将当前第一轮实验编译成一个带硬性 Gate 的实现合同。由于当前对话工作区不包含实际 CAPT/PromptFL 代码仓库，Prompt 的第一阶段必须先只读检查仓库，确认真实入口、参数名、训练状态结构和已有输出约定；不得凭空创建与现有训练主线平行的新框架。

当前最新实验要求覆盖旧版 TCRM、CAPT 机制复用和多 PEFT 扩展。本轮只使用 PromptFL，先补齐 `support-raw`，再做固定第 10 轮的集中式 Oracle CUSP。

原始方案没有锁定以下实现细节。为使首轮实验可复现，本 Prompt 将它们预注册为首轮默认值：

- 若仓库没有现成且严格独立的验证集，则把 CIFAR-100 官方 test split 按类别、固定 seed 42 等分为 `oracle_val` 和 `oracle_test`，每类各 50 张。`oracle_val` 用于 Gate 0 机制诊断、hard-negative/utility/CUSP 求解，`oracle_test` 只用于所有候选方法冻结后的 Gate 1A 最终离线评估。该拆分仅用于第一轮机制生死验证，不作为最终论文标准测试协议。
- zero-shot hard-negative bank 固定 `K=5`。
- Gate 0 使用类别聚类 bootstrap：以类别为重采样单位、保留同一类别的三个诊断轮，10,000 次，seed 42。
- 随机更新重组固定 100 个随机凸组合，系数来自 `Dirichlet(1)`，seed 42；报告分布，不允许选择其中测试结果最好的组合。
- TIES 固定 trim density 为 0.2；只用于首轮基线，不做测试集调参。
- CUSP 在无量纲变量中固定 `lambda=0.1`、`mu=10.0`，head/overall 平均预测效用相对 FedAvg 的允许下降为 0.05。所有值必须进入输出元数据。
- “head/overall 基本保持”的首轮评估容忍度固定为相对 FedAvg 不下降超过 0.5 个百分点。该值只用于 Gate 判断，不是论文中的统计显著性结论。

如果仓库已经有与上述目的等价、且没有数据泄漏的固定验证协议，优先复用仓库协议，但必须在修改前报告差异并记录真实索引 fingerprint；不得同时维护两套默认协议。

---

## 可直接交给 Codex 的主 Prompt

```text
你需要在当前联邦长尾代码仓库中实现并验证“CUSP 第一轮生死实验”的最小闭环：

1. 在不改变 PromptFL/FedAvg 默认训练行为的前提下，保存固定诊断轮的客户端 Prompt 更新；
2. 实现离线 `support-normalized → support-raw → all-client` 因果诊断；
3. 实现固定第 10 轮的单轮集中式 Oracle CUSP，以及规定的五类等范数对照；
4. 完成静态检查、确定性单元测试和最小工程冒烟；
5. 只生成两次 20 轮完整训练及离线分析的准确命令，不要自动启动完整实验。

这不是完整 CUSP 在线训练框架，也不是最终论文实验。它只回答两个生死问题：

- Gate 0：在不重新归一化、不放大支持客户端更新时，加入非支持客户端更新后，tail 类已有收益是否仍明显下降？
- Gate 1A：在与 FedAvg 相同更新范数、相同客户端更新张成空间、保持 head/overall 的条件下，是否存在比 FedAvg 和简单合并基线更好的单轮全局方向？

如果 Gate 0 不通过，不得把 Oracle CUSP 的结果包装成支持聚合干扰假设；如果 Oracle CUSP 本身不通过，不得继续实现 secure aggregation、分布式 utility probing、Adapter、LoRA 或 CAPT+CUSP。


一、必须锁定的科学配置

第一轮完整运行只有两个配对条件：

| 项目 | Client-LT | Dirichlet |
|---|---|---|
| dataset | CIFAR100-LT | CIFAR100-LT |
| global imbalance factor | 0.01 | 0.01 |
| num_clients | 30 | 30 |
| frac | 1.0 | 1.0 |
| local_epochs | 3 | 3 |
| seed | 42 | 42 |
| rounds | 20 | 20 |
| local learner | PromptFL | PromptFL |
| partition | Client-LT | 标准 fine-class Dirichlet |
| partition parameters | lambda=0.75, alpha_1=0.5, head_leakage_scale=3.0 | beta=0.5 |
| diagnostic rounds | t={5,10,20} | t={5,10,20} |
| Oracle CUSP round | t=10 | t=10 |

轮次统一采用 1-based 语义：

- `theta_t` 是第 t 轮本地训练开始前的全局 Prompt 参数；
- `Delta_i = theta_i_local_after - theta_t` 是客户端 i 在第 t 轮产生的 Prompt 更新；
- `theta_all = theta_t + sum_i w_i Delta_i` 是该轮原始 FedAvg 结果；
- 因此诊断轮 t=20 使用第 20 轮开始模型和第 20 轮客户端更新，不需要第 21 轮训练。

两种划分必须共享以下不变量：

1. 完全相同的 CIFAR100-LT Global-LT 训练样本索引、每类总样本数和预处理；
2. 完全相同的 CLIP backbone、预训练权重、PromptFL 结构、可训练参数集合和 Prompt 初始化；
3. 完全相同的训练 seed、初始化 seed、优化器、学习率/调度、batch size、local_epochs、rounds；
4. 完全相同的客户端参与列表及其顺序。虽然 frac=1.0 使 30 个客户端全部参与，仍须复用并记录相同 schedule fingerprint；
5. 完全相同的评估代码、head/mid/tail 类别定义和验证/测试索引；
6. 原始 FedAvg 始终使用 `w_i = n_i / sum_j n_j`，不得改变归一化分母；
7. 除“训练样本如何分配给客户端”外，不允许有其他有意变化。

标准 Dirichlet 必须按 100 个 fine labels 划分。若仓库现有 `noniid-labeldir` 混用了 CIFAR-100 coarse labels，不得复用它冒充标准 Dirichlet；应复用仓库已有的 fine-class 分支，或在最小边界内新增明确命名的 fine-class 分支。

Client-LT 中 `alpha_1=0.5` 只表示该设置定义中的组内浓度，不得把它错误解释为对全部 30 个客户端直接执行的普通 Dirichlet。


二、待检验假设与统计单位

H0-Gate0：

对 tail 类 c，在同一 `theta_t` 和同一批客户端更新下，

    interference_c = G_support_raw,c - G_all,c

不存在稳定的正值；观察到的 support-only 优势主要来自重新归一化后的步长放大。

H1-Gate0：

`interference_c` 在 Client-LT 的三个固定诊断轮中方向稳定为正；Dirichlet 中也存在同方向现象，但允许更弱。这说明非支持客户端更新会进一步消耗已由支持客户端产生的类别收益。

H0-Gate1：

在客户端真实更新张成空间和 FedAvg 更新范数预算内，不存在稳定优于 FedAvg、随机重组和普通 class-wise weighting 的方向。

H1-Gate1：

Oracle CUSP 在 Client-LT 与 Dirichlet 上都能提高 tail 的实际 margin gain 和 attainable-gain survival，同时不依赖更大 update norm，且 head/overall 基本保持。

统计单位必须区分：

- 完整训练 run：`topology × seed`，本轮仅 2 × 1；
- Gate 0 配对单位：同一 topology、round、class 下的三个候选模型；
- Gate 0 bootstrap 单位：class，而不是把同一类三个 round 当作三个独立样本；
- Gate 1 比较单位：同一 topology、第 10 轮、同一组客户端更新、同一验证/测试数据上的 method；
- 每类指标必须保留，不能只输出平均值。


三、先只读检查仓库，再决定实际修改点

在编辑任何文件之前执行以下检查，并先在回复中给出一份简短的“仓库事实与实现地图”：

1. 阅读根目录及相关子目录中的 `AGENTS.md` 或其他仓库指令。
2. 找到实际联邦训练入口、PromptFL trainer、FedAvg 实现、CIFAR100-LT 构建、Client-LT 划分和 Dirichlet 划分的真实路径与调用链。
3. 确认：
   - 全局模型在每轮哪个位置被复制给客户端；
   - 本地 Prompt 参数训练前后状态在哪里可获得；
   - FedAvg 权重 `w_i` 在哪里计算；
   - 30 个客户端的 class count/support 信息在哪里可获得；
   - round/client schedule 在哪里生成或读取；
   - 现有评估函数如何返回 overall、macro、head、mid、tail 和 per-class accuracy；
   - 是否已有冻结 CLIP feature cache；
   - 是否已有客户端更新保存、update retention、实验 D、SVD、TIES 或 class-wise aggregation 代码可以安全复用；
   - 当前是否有用户未提交修改，避免覆盖无关改动。
4. 核对 PromptFL 的全部可训练参数。不要假设只有一个固定 key；保存和展平时必须依据实际 `requires_grad=True` 的 PromptFL 参数集合，并保存可逆的参数名称、shape、dtype、offset 规范。
5. 核对现有 head/mid/tail 定义并原样复用。若仓库没有唯一正式定义，停止编辑并报告，不能自行按类别编号猜测。
6. 核对训练集、验证集、测试集协议：
   - 若已有训练外、固定且不会泄漏 test 的分层验证集，复用；
   - 若没有，则为本轮 Oracle 实验把 CIFAR-100 官方 test split 按 class、seed 42 固定等分为 `oracle_val` 与 `oracle_test`，每类各 50 张；
   - 不得用 `oracle_test` 选择 hard negatives、求导、调 solver、选超参数或选择随机基线。
7. 核对以下预期路径是否真实存在，但不要把它们当成无条件事实：
   - 可能的主入口：`federated_main.py`
   - 可能的 PromptFL trainer：`trainers/promptfl.py`
   - 可能的 FedAvg：`utils/fed_utils.py`
   - 可能的数据划分：`utils/datasplit.py`
   - 可能的数据集封装：`datasets/cifar100_LT.py`
   若真实仓库不同，以实际路径为准并在实现地图中说明。

遇到以下任一情况，停止修改并报告，不要创建平行实现：

- 找不到唯一实际训练入口或 PromptFL 分支；
- 无法从同一轮获得 `theta_t`、30 个 `Delta_i`、`n_i` 和 class counts；
- Client-LT 与 fine-class Dirichlet 无法复用完全相同的 Global-LT 样本；
- head/mid/tail 定义不明确；
- 当前用户改动与拟修改位置直接冲突；
- 现有 PromptFL/FedAvg 行为无法通过 feature flag 保持原样。


四、修改边界

优先采用最小侵入式修改：

- 在现有联邦训练入口的 PromptFL/FedAvg 分支增加“只保存诊断”的显式 flag；
- 新增两个独立离线脚本，语义目标分别为：
  - `analyze_support_raw.py`
  - `oracle_cusp_single_round.py`
- 新增小型共享工具模块仅用于：
  - Prompt trainable state 的展平/还原；
  - dump schema 读写和校验；
  - margin、SVD、范数、cosine 等纯函数；
- 在现有 tests 目录加入确定性单元测试；
- 如仓库已有命名/目录规范，文件名可以按规范调整，但职责不能合并成难以审计的大脚本。

禁止事项：

1. 不修改 PromptFL 默认 loss、本地优化器、FedAvg 数学语义或默认输出。
2. 不改 CAPT、TCRM、FedITE、其他 trainer 或已有实验结果。
3. 不实现 Adapter、LoRA、EMA、SWA、Balanced Softmax、数据增强、memory、router、secure aggregation 或在线 distributed utility。
4. 不加入 PCGrad/CAGrad；它们不阻塞本轮 Gate。
5. 不用测试集选择 round。Oracle 固定第 10 轮，禁止事后改成“缺口最大轮”。
6. 不用测试集构造 hard-negative bank、计算 A、选择 CUSP 参数或挑随机组合。
7. 不把 support-only 权重重新归一化后的结果当作 support-raw。
8. 不保存或修改完整 CLIP backbone checkpoint；只保存重建诊断所需的 PromptFL trainable state、必要元数据和冻结特征。
9. 不为通过测试而降低断言、改变 baseline 定义或扩大 update norm。
10. 不自动启动两次 20 轮完整训练。

所有新增行为必须默认关闭。关闭 flag 时，相同 seed 的 PromptFL/FedAvg 运行必须与修改前保持一致；至少对模型参数、round metrics 和 client schedule 做严格或数值容差内的回归检查。


五、训练阶段：保存最小且完备的诊断包

为现有训练入口增加与仓库风格一致的 CLI/config。语义至少覆盖：

- 是否保存 CUSP/机制诊断；
- 诊断轮列表，完整实验固定为 `5,10,20`；
- 诊断输出根目录；
- oracle split seed，固定 42；
- hard-negative K，固定 5；
- 是否只保存 PromptFL trainable parameters。

不要盲目照抄这些建议 flag 名；应遵循仓库现有命名风格。但完成报告必须列出最终真实参数名。

每个 topology、seed、diagnostic round 保存一个自校验目录。建议逻辑结构：

    <output_root>/
      diagnostics/
        <topology>/
          seed_42/
            round_005/
            round_010/
            round_020/

每个 round dump 至少包含：

1. `global_before`：该轮开始时所有 PromptFL trainable parameters。
2. `client_updates`：30 个客户端的 `Delta_i`，按稳定 client_id 排列。
3. `fedavg_weights`：原始 `w_i=n_i/sum_j n_j`，float64 保存或计算。
4. `client_num_samples`：每个客户端实际用于本地训练的样本数 `n_i`。
5. `client_class_counts`：shape `[30,100]` 的实际训练样本计数。
6. `support_mask`：`support_mask[i,c] = client_class_counts[i,c] > 0`。
7. `global_after_fedavg`：实际训练主线得到的该轮 FedAvg Prompt state，供重建一致性校验。
8. `flatten_spec`：参数名、shape、dtype、offset、总维度和有序 key hash。
9. `metadata.json`，至少记录：
   - dataset/topology/partition parameters；
   - seed/split seed/schedule seed；
   - round（1-based）；
   - num_clients/frac/local_epochs；
   - optimizer/LR/scheduler/batch size；
   - Global-LT 样本 fingerprint 与每类总数；
   - client partition fingerprint；
   - client schedule fingerprint；
   - Prompt initialization fingerprint；
   - CLIP checkpoint/model identifier；
   - head/mid/tail class ids；
   - trainable parameter names；
   - Git commit（若可得）与工作树状态摘要；
   - dump schema version。

必须在保存时验证：

    global_before + sum_i w_i * Delta_i

逐参数重建出的 state 与 `global_after_fedavg` 一致。若误差超过适合 dtype 的容差，立即报错且将该 dump 标记为 invalid；不得继续做离线分析。

诊断保存不得改变客户端训练顺序、随机数状态、聚合值或全局评估结果。若保存操作本身会消费 RNG，必须改为不消费 RNG 的实现。


六、固定验证/测试特征与 hard-negative bank

优先复用仓库已有冻结 CLIP feature cache。否则以仓库当前 CLIP 图像/文本预处理准确生成一次缓存。

必须分别保存：

- `oracle_val` 图像特征与 labels；
- `oracle_test` 图像特征与 labels；
- 固定 zero-shot text prototypes；
- 每类固定 hard-negative ids；
- split indices 与 fingerprint；
- CLIP model/preprocess identifier。

如果采用本轮默认拆分：

- 源数据为 CIFAR-100 官方 test split；
- 对每个 fine class 使用 seed 42 的确定性打乱；
- 每类前 50 张进入 `oracle_val`，后 50 张进入 `oracle_test`；
- 两种 topology 复用完全相同 indices；
- `oracle_test` 的 labels/features 只能在候选更新已经完全确定后传给最终评估函数。

hard-negative bank 只根据冻结 zero-shot text prototype 之间的相似度生成：

- 对每个类别 c，排除自身；
- 取 cosine similarity 最高的 5 个 fine classes；
- 生成一次后冻结，两种 topology 和全部 method 共用；
- 不得根据训练后模型、val/test accuracy 或单轮结果改变。


七、Gate 0：support-normalized / support-raw / all-client

对 topology、round、class c：

    S_c = {i | client_class_counts[i,c] > 0}
    w_i = n_i / sum_j n_j

从同一个 `theta_t=global_before` 构造：

    theta_support_norm,c
      = theta_t
      + sum_{i in S_c} [w_i / sum_{j in S_c} w_j] * Delta_i

    theta_support_raw,c
      = theta_t
      + sum_{i in S_c} w_i * Delta_i

    theta_all
      = theta_t
      + sum_i w_i * Delta_i

注意：

- `support_raw` 必须保留原始全局 FedAvg 权重和原始分母；
- `support_norm` 仅用于量化 dilution，不是方法候选；
- `theta_all` 必须与 dump 中真实 `global_after_fedavg` 一致；
- 若 `S_c` 为空，记录 invalid/unsupported，不得除零或伪造零收益；
- 单客户端 class support 只能用于离线机制诊断，不能成为未来方法的服务器输入。

在 `oracle_val` 上分别计算每个模型的：

1. per-class accuracy；
2. per-class decision margin：

       m_c(theta)
         = mean_{x:y=c} [
             f_c(x;theta) - max_{h in H_c} f_h(x;theta)
           ]

3. 相对 `theta_t` 的 gain：

       G^acc_variant,c = acc_variant,c - acc_before,c
       G^margin_variant,c = margin_variant,c - margin_before,c

4. 分解：

       dilution_c = G_support_norm,c - G_support_raw,c
       interference_c = G_support_raw,c - G_all,c

accuracy 与 margin 都要计算，但 Gate 0 的主判据使用 margin interference，因为单轮 accuracy 离散且可能不变；accuracy 作为同方向辅助证据。

输出 `support_raw_per_class.csv`，每行一个
`topology × round × class`，至少包含：

- topology
- seed
- round
- class_id
- class_group (`head|mid|tail`)
- global_class_count
- support_client_count
- support_weight_mass = `sum_{i in S_c} w_i`
- before_acc
- support_norm_acc
- support_raw_acc
- all_acc
- gain_support_norm_acc
- gain_support_raw_acc
- gain_all_acc
- dilution_acc
- interference_acc
- before_margin
- support_norm_margin
- support_raw_margin
- all_margin
- gain_support_norm_margin
- gain_support_raw_margin
- gain_all_margin
- dilution_margin
- interference_margin
- valid
- invalid_reason

输出 `support_raw_summary.json` 和可读的 `support_raw_summary.md`，至少包含：

- 每个 topology、round、class_group 的 mean/median/standard error；
- interference>0 的 class fraction；
- Client-LT 和 Dirichlet 的类别聚类 bootstrap 95% CI；
- leave-one-round-out 的 pooled tail interference；
- 每个 round 单独的 tail interference；
- 按 support weight mass 分桶或相关性诊断；
- 有效/无效类别数量；
- Gate 0 自动判定与逐项理由。

bootstrap 规则：

- 只对 tail classes 重采样；
- 一次抽样抽取 class ids，并保留被抽中 class 的全部三个 round；
- 10,000 次，seed 42；
- 主统计量为三个 round 的 tail `interference_margin` 平均值；
- 同时输出 median 和 `interference_acc` 的辅助区间。

Gate 0 首轮通过条件必须编码为确定性报告规则：

1. Client-LT 在 t=5、10、20 三轮的 mean tail `interference_margin > 0`；
2. Client-LT pooled class-cluster bootstrap 95% CI 下界 `> 0`；
3. Client-LT 三个 leave-one-round-out pooled mean 都 `> 0`；
4. Dirichlet pooled mean tail `interference_margin > 0`，且至少 2/3 个诊断轮的 round mean `> 0`；Dirichlet 允许弱于 Client-LT；
5. 报告 top contributing classes 和 round，但不得因某一类别/轮次不满足就删除数据；
6. accuracy interference 至少方向不系统性反转。若 accuracy 大量为 0 变化，应明确标注“离散指标无分辨率”，不自动判失败。

若 Gate 0 不通过：

- 输出负结果与失败项；
- 不得执行 Oracle CUSP 的测试集评估；
- 不得把 CUSP 作为下一步在线方法继续扩展；
- 建议研究转向“局部收益生成不足”，但本任务不实现该替代路线。


八、Gate 1A：固定第 10 轮单轮集中式 Oracle CUSP

只有 Gate 0 报告为 pass 时才执行本阶段分析。实现代码可以提前完成和单元测试，但真实 round-10 分析必须读取 Gate 0 结果并硬性阻止越过失败 Gate，除非用户显式使用仅供调试的 override；override 结果必须醒目标记为非有效科学结果。

本阶段对 Client-LT 与 Dirichlet 分别使用各自第 10 轮 dump，但：

- method 之间共享完全相同的 `theta_10`、30 个 `Delta_i`、`w_i`、验证/测试特征；
- CUSP 求解和所有需要 utility 的基线只读取 `oracle_val`；
- 所有 method 的候选更新冻结后，统一在 `oracle_test` 评估一次；
- 禁止根据 `oracle_test` 结果改变方法、solver 参数或 baseline。


8.1 构建客户端更新子空间

按 `flatten_spec` 将每个客户端 Prompt 更新展平为 `Delta_i in R^p`，构造：

    D = [Delta_1, ..., Delta_30] in R^(p × 30)

使用稳定 SVD：

    D = U Sigma V^T

取最小 rank r，使：

    sum_{k=1}^r sigma_k^2 / sum_k sigma_k^2 >= 0.999

定义：

    Q = U[:, :r] in R^(p × r)
    Delta_FA = sum_i w_i Delta_i
    z_FA = Q^T Delta_FA
    B = ||Delta_FA||_2

要求：

- 第一版不调 SVD rank；
- 保存全部 singular values、累计能量和 r；
- 验证 `Q^T Q ≈ I`；
- 验证 `Q z_FA` 与 `Delta_FA` 的投影误差；
- 若 `B` 近零、SVD 全零或 rank=0，报告不可分析并停止；
- 不允许把范数预算改成客户端更新最大范数或 support-normalized 范数。


8.2 计算类别 utility matrix A

在 `oracle_val` 上定义：

    m_c(theta)
      = mean_{x:y=c} [
          f_c(x;theta) - max_{h in H_c} f_h(x;theta)
        ]

其中 `H_c` 是已冻结的 zero-shot top-5 hard-negative bank。

令 Prompt 参数为：

    theta(z) = theta_10 + unflatten(Q z)

使用 autograd 在 `z=0` 计算：

    A[c,k] = d m_c(theta(z)) / d z_k |_(z=0)

要求：

- A 的 shape 为 `[100,r]`；
- 类别 reduction 是每类样本均值，不按验证样本数再次加权；
- hard negative 的 max 必须保持可微；
- 用中心有限差分在小型 fixture 和至少若干真实 `(class,direction)` 对上检查 autograd；
- 记录每类 `||A_c||_2`、符号、有限值状态；
- 若 `||A_c||_2 <= eps`，将该类标为 invalid-for-normalized-utility，不参与除法，但仍保留原始 margin 评估；
- 不得用 `oracle_test` 计算 A。

定义：

    P_c = B * ||A_c||_2

它表示在线性近似和相同 L2 更新预算下，类别 c 在当前候选子空间中的理论可达 margin gain。

定义任意候选 `z` 的：

    predicted_gain_c(z) = A_c z
    predicted_survival_c(z) = A_c z / P_c

实际应用候选更新后，在 `oracle_test` 计算：

    actual_margin_gain_c
      = m_c(theta_10 + Qz) - m_c(theta_10)

    actual_survival_c
      = actual_margin_gain_c / P_c

raw ratio 必须原样保存，不得为了图好看在统计前 clip 到 [0,1]。


8.3 求解等范数 Oracle CUSP

为改善数值尺度，内部使用无量纲变量：

    u = z / B
    u_FA = z_FA / B
    a_c = A_c / ||A_c||_2

此时：

    predicted_survival_c = a_c u
    ||u||_2 <= 1

对所有 valid classes 求解：

    maximize_{u,tau,xi}
        tau
        - lambda * ||u-u_FA||_2^2
        - mu * mean_c(xi_c)

    subject to
        a_c u + xi_c >= tau,       for every valid class c
        xi_c >= 0
        ||u||_2 <= 1
        tau <= 1

并增加 head/overall 平均 utility 下界：

    mean_{c in valid head}(a_c u)
      >= mean_{c in valid head}(a_c u_FA) - 0.05

    mean_{c in all valid}(a_c u)
      >= mean_{c in all valid}(a_c u_FA) - 0.05

首轮固定：

    lambda = 0.1
    mu = 10.0

求解后：

    z_CUSP = B * u_star
    Delta_CUSP = Q z_CUSP

要求：

- 第一版可使用 CVXPY；
- 明确记录 solver、status、迭代/时间、目标值、tau、slack 总量/均值/最大值和 active constraints；
- solver 非 optimal/optimal_inaccurate 时不得静默使用结果；
- 对 `optimal_inaccurate` 必须额外验证全部约束残差与范数，只有在显式容差内才可评估；
- `||Delta_CUSP||_2 <= B + numerical_tolerance`；
- 不得在求解后为了提高测试准确率再次缩放；
- 若求解失败，保留 FedAvg，不得偷偷切换到另一个 solver/参数并只报告成功结果；可以报告诊断和建议，但本轮科学结果记为 solver failure。


九、Gate 1A 必须比较的六种方法

所有候选都从同一 `theta_10` 出发。除原始 FedAvg 外，候选更新最终都必须满足：

    ||Delta_method||_2 = B

若原始方法自然产生零向量，则不得强行归一化；应标记 invalid。CUSP 和 cone 使用 `<=B` 约束，报告真实 norm，不要求人为放大到恰好 B。

1. `fedavg`

       Delta_FA = sum_i w_i Delta_i

   这是原始基线，不重新缩放。

2. `random_recomposition`

   生成 100 组：

       alpha^(s) ~ Dirichlet(1,...,1), seed=42
       Delta_random^(s) = sum_i alpha_i^(s) Delta_i

   每个非零随机向量统一缩放到 B。所有 100 组在 `oracle_val` 和 `oracle_test` 的结果都要保存。

   主报告使用 100 组的 mean/std/95% empirical interval；不得选择 test 最优随机组作为主基线。可以额外报告 best-on-validation 一组，但选择只能使用 `oracle_val`，并明确标注。

3. `classwise_count_weighting`

   这是“普通 class-wise weighting”的首轮确定义，只使用训练 class counts，不访问 val/test utility：

       n_c = sum_i n_ic
       s_i = sum_{c:n_c>0} n_ic / n_c
       alpha_i = s_i / sum_j s_j
       Delta_CW = sum_i alpha_i Delta_i

   将非零 `Delta_CW` 缩放到 B。保存 `s_i`、`alpha_i` 和与原始 `w_i` 的差异。

4. `ties`

   对 weighted task vectors：

       v_i = w_i Delta_i

   实现或复用标准 TIES 的三步：

   - trim：每个 `v_i` 只保留绝对值最大的 20% 坐标；
   - elect sign：每个坐标根据 trimmed vectors 的加权/求和符号选举；
   - disjoint merge：只平均/合并与 elected sign 一致的非零坐标。

   固定 density=0.2，不用 test 调参。若仓库已有可信 TIES 实现，优先复用并记录其精确定义；若其数学语义与此处不同，修改前报告，不得把不同实现同名混用。最终非零更新缩放到 B。

5. `class_feasible_cone`

   使用与 CUSP 相同的 `A/Q/B/oracle_val`，在无量纲空间求离 FedAvg 最近的可行方向：

       minimize_u ||u-u_FA||_2^2

       subject to
           a_c u >= 0, for every valid tail class c
           mean_valid_head(a_c u)
             >= mean_valid_head(a_c u_FA) - 0.05
           mean_all_valid(a_c u)
             >= mean_all_valid(a_c u_FA) - 0.05
           ||u||_2 <= 1

   该基线没有 attainable-gain max-min 目标、没有 tau、没有 slack。保存可行性和约束残差。

6. `oracle_cusp`

   使用第 8.3 节的固定优化问题。

本轮不实现 PCGrad/CAGrad。只有 CUSP 出现明确正结果后再考虑补充。


十、Gate 1A 输出

建议目录：

    <output_root>/
      gate0_support_raw/
      gate1_oracle_cusp/
        <topology>/
          seed_42/
            round_010/

至少输出：

1. `method_metrics.csv`：每行一个 `topology × method`；random 主汇总另带分布字段。
2. `per_class_metrics.csv`：每行一个 `topology × method × class`。
3. `random_recomposition_runs.csv`：100 组随机结果。
4. `utility_matrix.npz`：A、P、Q/z 相关数组及 schema/version。
5. `svd_report.json`。
6. `solver_report.json`。
7. `oracle_cusp_summary.json`。
8. `oracle_cusp_summary.md`。

`method_metrics.csv` 至少包含：

- topology
- seed
- round
- method
- overall_acc
- macro_acc
- head_acc
- mid_acc
- tail_acc
- mean_head_margin_gain
- mean_mid_margin_gain
- mean_tail_margin_gain
- mean_tail_predicted_survival
- median_tail_predicted_survival
- mean_tail_actual_survival
- median_tail_actual_survival
- improved_class_count_margin
- harmed_class_count_margin
- improved_tail_class_count_margin
- harmed_tail_class_count_margin
- update_norm
- fedavg_norm
- norm_ratio
- cosine_with_fedavg
- pearson_predicted_vs_actual_margin_gain
- spearman_predicted_vs_actual_margin_gain
- solve_time_seconds
- svd_rank
- svd_energy_retained
- valid_utility_class_count
- solver_status

`per_class_metrics.csv` 至少包含：

- topology
- seed
- round
- method
- class_id
- class_group
- class_count_train_global
- support_client_count
- support_weight_mass
- attainable_gain_P
- predicted_margin_gain
- actual_margin_gain
- predicted_survival_ratio
- actual_survival_ratio
- before_margin
- after_margin
- before_acc
- after_acc
- acc_gain
- improved_margin
- valid_utility
- invalid_reason

相关性只在 valid、finite 类别上计算，并报告样本数。Pearson 和 Spearman 都必须输出。若常量数组导致相关性未定义，输出 null/NaN 和明确原因，不得替换成 0。

改进/受损类别按 actual margin gain 的正负统计；同时可附加 accuracy 版本，但不能替代 margin 版本。


十一、Gate 1A 的自动判定

对每个 topology，将 Oracle CUSP 与同一 round 的 FedAvg 配对比较。

首轮“继续开发”必须同时满足：

1. Client-LT 和 Dirichlet 的 CUSP mean tail actual margin gain 均高于 FedAvg；
2. 两种 topology 的 mean tail actual survival ratio 均高于 FedAvg；
3. 两种 topology 的 tail accuracy 相对 FedAvg 同方向改善或不变，不允许一边明显反向；
4. CUSP 的 overall accuracy 和 head accuracy 相对 FedAvg 各自下降不超过 0.5 个百分点；
5. CUSP update norm 不超过 FedAvg norm（仅允许数值误差）；
6. CUSP 在两种 topology 上都优于 random-recomposition 主分布均值；
7. CUSP 在两种 topology 上都优于 `classwise_count_weighting` 的 mean tail actual survival；
8. 相比 TIES 和 cone，CUSP 至少不能在两种 topology 上都更差；完整数值必须报告，不能删除强基线；
9. 线性 predicted gain 与 actual margin gain 至少总体方向一致；输出 Pearson/Spearman。若相关性很弱，Gate 可标为“conditional”而不是自动 pass，并明确指出线性化问题；
10. solver 状态、约束残差、SVD 和 dump 重建全部通过数值检查。

自动结论只能是：

- `PASS`：满足上述硬条件，可进入 5 轮在线冒烟；
- `CONDITIONAL`：margin/survival 明显改善但 accuracy 离散不变，或线性相关性不足；需要先修正 utility/线性化验证，不得直接宣称方法成立；
- `FAIL_GATE0`：support-raw 与 all 缺口不成立，停止聚合协调路线；
- `FAIL_ORACLE`：Oracle CUSP 不能在等预算下改善两种 topology，停止实现完整 CUSP；
- `FAIL_FAIRNESS_OR_IMPLEMENTATION`：范数、数据、schedule、重建、solver 或 test leakage 检查失败，结果无效，必须先修代码。

不得依据测试表现修改这些判定规则。


十二、输出 fingerprint 与公平性审计

为两个 topology 生成 `pairing_audit.json`，逐项比较：

- Global-LT sample fingerprint；
- global class counts；
- CLIP/preprocess identifier；
- Prompt initialization fingerprint；
- trainable parameter spec hash；
- optimizer/LR/scheduler/batch size/local_epochs/rounds；
- client schedule fingerprint；
- validation/test indices fingerprint；
- head/mid/tail ids；
- diagnostic rounds；
- hard-negative bank fingerprint。

允许不同且必须记录的只有：

- topology/partition name；
- client assignment fingerprint；
- 每客户端 class counts；
- 由实际 client sample counts 导致的 FedAvg weights。

若任何应共享项不一致，`pairing_audit` 必须 fail，并阻止离线科学比较。


十三、验证顺序

不要直接运行完整实验。严格按下列顺序验证。

Gate V0：静态与导入检查

- 对所有修改/新增 Python 文件运行仓库现有 formatter/linter（若有）；
- 运行 `python -m py_compile` 或等价导入检查；
- 验证 CLI help 能显示新增参数且默认关闭。

失败则停止。

Gate V1：确定性单元测试

至少覆盖：

1. 三客户端、两类别 synthetic fixture，手算验证：
   - support_norm；
   - support_raw 不重新归一化；
   - all；
   - dilution/interference。
2. flatten → unflatten 完全可逆，参数 key/shape 保持。
3. `global_before + sum_i w_i Delta_i` 能重建 FedAvg。
4. SVD 99.9% energy rank 选择正确，Q 正交。
5. A 的 autograd 与中心有限差分一致。
6. P、predicted survival 和 invalid zero-gradient class 处理正确。
7. FedAvg、random、classwise、TIES 的非零候选缩放后 norm 等于 B。
8. cone/CUSP 的 norm 与约束残差满足容差。
9. random baseline 在 seed 42 下完全确定。
10. solver 不可行/失败时不会静默评估。
11. `oracle_test` 不能被 utility/CUSP 求解函数读取；用接口隔离或 mock/spy 测试证明。
12. 诊断 flag 关闭时不创建 dump，不改变聚合返回值。

失败则停止，不得弱化断言。

Gate V2：最小工程冒烟

只运行一个不作为科学结果的最小条件：

- CIFAR100-LT；
- PromptFL；
- seed 42；
- 30 clients；
- frac=1.0；
- Client-LT 代表配置；
- 2 rounds；
- local_epochs=1；
- diagnostic round=1；
- 保存诊断开启；
- 不执行完整 Oracle 测试评估。

若 30 客户端的 2 轮冒烟仍明显过重，可以先增加一个纯 synthetic/in-memory integration fixture，但最终至少要完成一次真实数据的单轮 dump/reload 闭环；不要擅自把科学配置改成少客户端后宣称验证完成。

冒烟必须检查：

- 30 个客户端均从同一 global state 开始；
- dump 中恰有 30 个更新；
- `w_i` 和为 1；
- class counts 与实际 partition 相符；
- support mask 正确；
- FedAvg 重建误差通过；
- 保存前后 round metric 一致；
- dump 能被两个离线脚本加载；
- feature cache/split fingerprint 稳定；
- 没有读取 oracle_test 参与求解。

Gate V3：命令生成，不执行完整实验

V0–V2 全部通过后，根据仓库真实 CLI 输出以下准确命令：

1. Client-LT：20 rounds、local_epochs=3、seed=42、30 clients、frac=1.0、
   lambda=0.75、alpha_1=0.5、head_leakage_scale=3.0、
   diagnostic rounds 5/10/20。
2. fine-class Dirichlet：相同 Global-LT、20 rounds、local_epochs=3、
   seed=42、30 clients、frac=1.0、beta=0.5、
   相同 diagnostic rounds 和 schedule。
3. pairing audit 命令。
4. Gate 0 离线分析命令。
5. Gate 0 通过后的两个 topology 第 10 轮 Oracle CUSP 命令。
6. 汇总两个 topology 并生成最终 Gate 报告的命令。

不要使用占位符伪装成可运行命令。命令中的路径允许用清楚命名的用户需替换变量（如 DATA_ROOT/OUTPUT_ROOT），但实际 CLI flag 必须来自仓库。若某个命令仍无法确定，明确报告阻塞项。


十四、代码质量要求

- 保持人工可读、函数职责单一，不搭建新的大型框架。
- 优先复用现有 dataset/model/evaluation/serialization 逻辑。
- 数值计算明确 dtype/device，统计与权重建议用 float64，模型前向遵循原 dtype。
- 所有 CSV 字段顺序固定；JSON 使用 schema version。
- 错误信息必须指出 topology/round/class/client 和失败不变量。
- 对 100 类逐类构造候选模型时避免重复完整 CLIP 图像前向；优先复用冻结特征缓存，只重放 Prompt/logit 相关短路径。
- 不通过 pickle 加载不可信任外部对象；内部 `.pt` 文件明确仅限本实验可信产物。
- 不删除或覆盖用户已有输出；使用独立、确定性命名的实验目录。


十五、完成时必须汇报

最终回复只需要清楚汇报：

1. 实际读取到的训练入口和调用链；
2. 修改/新增的文件及各自用途；
3. 最终真实 CLI 参数名；
4. 实际执行的 V0/V1/V2 命令；
5. 单元测试和冒烟结果；
6. 生成的 artifact/schema 示例；
7. 默认关闭时的 PromptFL/FedAvg 回归结果；
8. 尚未解决的问题；
9. 是否达到“可以运行两次 20 轮训练”的工程 Gate；
10. 两次完整训练和全部离线分析的准确命令。

不要声称 Gate 0 或 Gate 1A 科学通过，除非对应两次 20 轮训练和离线结果已经真实存在并完成 pairing/fairness 审计。本次编码任务默认不授权自动启动完整训练。
```

---

## 实验要求覆盖审计

| 原实验要求 | Prompt 中的位置 | 验收方式 |
|---|---|---|
| CIFAR100-LT，IF=0.01 | 第一节 | `metadata.json` 与 `pairing_audit.json` |
| 30 clients，frac=1.0，local_epochs=3，seed=42，20轮 | 第一节 | 两条完整命令与 pairing audit |
| Client-LT λ=0.75、α₁=0.5、head leakage=3.0 | 第一节 | topology metadata |
| Dirichlet β=0.5 且为 fine-class | 第一、三节 | 划分调用链审计与 class-count fingerprint |
| PromptFL 为唯一首轮 learner | 第一、四节 | 禁止修改/加入 CAPT、Adapter、LoRA |
| 诊断轮 t={5,10,20} | 第一、五节 | round dump 目录和 metadata |
| 固定第10轮 Oracle，不挑轮 | 第一、四、八节 | round 字段与硬编码 Gate |
| 两拓扑完全相同 Global-LT 样本 | 第一、十二节 | sample fingerprint 和 global class counts |
| 相同参与调度 | 第一、十二节 | schedule fingerprint |
| 保存 θt、30个 Δi、FedAvg weights | 第五节 | round dump schema |
| support-normalized | 第七节 | 手算单测与 per-class CSV |
| support-raw 不重新归一化 | 第七、十三节 | synthetic exact test |
| all-client 原始 FedAvg | 第五、七节 | state 重建测试 |
| accuracy 和 margin gain | 第七节 | `support_raw_per_class.csv` |
| dilution / interference 分解 | 第七节 | exact formula unit test |
| Gate 0 三轮方向稳定 | 第七节 | per-round mean |
| bootstrap/配对统计 | 第七节 | class-cluster bootstrap 10,000次 |
| Dirichlet同方向、可更弱 | 第七节 | 单独的 Dirichlet Gate 条件 |
| 非单轮/少数类制造 | 第七节 | leave-one-round-out、top contributors |
| Gate 0失败停止CUSP路线 | 第七、八节 | 硬性 Gate 状态 |
| 单客户端支持信息仅离线诊断 | 第七节 | 禁止成为方法输入 |
| 固定训练外验证集与独立测试集 | 第三、六节 | split fingerprint 与接口隔离测试 |
| 测试集不得参与CUSP求解 | 第六、八、十三节 | mock/spy leakage test |
| Prompt参数展平 D=[Δ1,…,Δ30] | 第五、八节 | flatten spec 与可逆测试 |
| SVD保留99.9%能量，不调rank | 第八节 | `svd_report.json` |
| 固定zero-shot hard-negative bank | 第六、八节 | bank fingerprint，K=5 |
| A为类别margin方向导数 | 第八节 | autograd/有限差分测试 |
| P_c=B‖A_c‖，B=‖ΔFA‖ | 第八节 | unit test 与 per-class output |
| 与FedAvg相同范数预算 | 第八、九节 | norm/norm_ratio |
| CUSP max-min+slack+近FedAvg | 第八节 | CVXPY problem 与 solver report |
| head/overall效用下界 | 第八节 | 约束残差 |
| CVXPY集中式求解 | 第八节 | solver metadata |
| 原始FedAvg | 第九节 | method `fedavg` |
| 随机更新重组、同范数 | 第九节 | 100组 Dirichlet(1) 分布 |
| 普通class-wise weighting | 第九节 | 明确定义的 count-based baseline |
| TIES、同范数 | 第九节 | density=0.2 与 norm test |
| 类别可行锥投影 | 第九节 | 固定 convex projection |
| Oracle CUSP | 第八、九节 | method output |
| PCGrad/CAGrad不阻塞第一轮 | 第四、九节 | 明确禁止本轮实现 |
| overall/head/mid/tail accuracy | 第十节 | `method_metrics.csv` |
| macro accuracy | 第十节 | `macro_acc` |
| 每类margin gain | 第十节 | `per_class_metrics.csv` |
| tail attainable-gain survival ratio | 第八、十节 | predicted/actual raw ratios |
| 改善/受损类数量 | 第十节 | margin sign counts |
| update norm | 第十节 | norm 与 ratio |
| 与FedAvg cosine | 第十节 | `cosine_with_fedavg` |
| 线性预测与实际margin相关性 | 第十节 | Pearson/Spearman |
| 求解时间与SVD rank | 第十节 | metrics、solver/SVD reports |
| 不改PromptFL默认行为 | 第四、十三节 | flag-off 回归测试 |
| 不启动正式训练 | 开头、十三、十五节 | 只生成完整命令 |
| Oracle失败停止完整方法开发 | 第十一节 | `FAIL_ORACLE` |

### 审计结论

- 原始第一轮配置、Gate 0 公式、Gate 1A 算法、六种方法、全部输出指标和停止条件均已进入 Prompt。
- 补齐了原设计中会导致 Codex 自行猜测的实现缺口：轮次语义、验证/测试隔离、参数展平规范、dump 重建、baseline 数学定义、随机基线报告规则、数值异常、输出 schema、统计单位、测试泄漏检查和自动 Gate。
- 明确排除了旧版 TCRM、CAPT、Adapter、LoRA、EMA、数据增强、secure aggregation 与 PCGrad/CAGrad，防止第一轮任务膨胀。
- 当前唯一仍依赖目标仓库事实的内容是实际文件路径、CLI 名称和可复用函数；Prompt 已要求 Codex 在编辑前只读核对，并在不满足关键前提时停止。
