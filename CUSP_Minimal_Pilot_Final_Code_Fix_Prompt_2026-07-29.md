# CUSP 最小生死实验：最终代码收口 Prompt

## 编译说明

本 Prompt 以最新决定为准，覆盖此前 CUSP Round-1 的“两种划分、20 轮、100 个随机候选、完整 Gate 体系”方案。

本次只完成一个可直接启动的最小生死实验：

- `client-longtail`
- seed `42`
- PromptFL
- 训练 `10` 个通信轮
- 在第 `10` 轮保存 Oracle dump
- 离线比较 `fedavg`、`random_reweight`、`classwise_weighting`、`oracle_cusp`
- `random_reweight` 只生成 `10` 个固定随机候选
- 只报告 overall、macro、head/non-tail、tail accuracy 与更新范数

由于当前对话中可见代码与“本地已修改版本”的说明曾不一致，下面要求 Codex 必须先核对真实仓库，再修改实际活动路径，并用测试、dry-run 和产物证明完成状态。不得仅根据旧 Prompt 或修改说明判断代码已经完成。

---

```text
你需要在当前 CAPT 仓库中完成 CUSP 最小生死实验的最终代码收口。完成后，用户应能用一条命令依次完成：

1. 一个 Client-LT、seed 42、10 轮的 PromptFL/FedAvg 训练；
2. 第 10 轮 Oracle dump 的生成与完整性校验；
3. 基于同一组客户端更新的四候选离线构造；
4. 候选冻结后的 official test 评估；
5. 最小结果表和报告生成。

这不是重新设计论文最终实验，也不是实现在线 CUSP。优先修通真实活动路径，不要继续增加外围工程。

==================================================
一、实验目的与判定问题
==================================================

研究问题：

在同一个第 10 通信轮，固定全局模型、客户端本地训练结果和客户端更新，只改变更新的组合方式，能否在不增加总更新范数的前提下，比 FedAvg 更好地保留 tail 类收益，同时不明显损害 head/non-tail 和 overall accuracy？

待检验假设：

Oracle CUSP 在严格等范数条件下：

- tail accuracy 高于 FedAvg；
- tail accuracy 高于朴素 class-wise weighting；
- tail accuracy 高于 10 个 random reweight 候选的中位数；
- head/non-tail 与 overall accuracy 相对 FedAvg 的下降均不超过 0.5 个百分点。

这是单 topology、单 seed、单轮 Oracle 诊断，只用于决定 CUSP 路线是否值得扩展，不构成论文最终结论。

==================================================
二、本次锁定的实验配置
==================================================

本次只允许一个训练 run：

- dataset：`cifar100_LT`
- global imbalance type：`exp`
- imbalance factor：`0.01`
- partition：`client-longtail`
- backbone：`ViT-B/16`
- trainer：`PromptFL`
- aggregation：仓库原始 sample-weighted `fedavg`
- `CSC=True`
- `n_ctx=4`
- `n_general=1`
- clients：`30`
- participation fraction：`1.0`
- local epochs：`3`
- communication rounds：`10`
- train seed：`42`
- split seed：`42`
- client schedule seed：`42`
- Oracle round：`10`，CLI 和文件名统一使用 1-based round
- `specialization_lambda=0.75`
- `intra_group_alpha=0.5`
- `head_leakage_scale=3.0`
- learning rate：沿用现有 Experiment-D PromptFL 配置，预期为 `0.001`
- train batch size：沿用现有 Experiment-D PromptFL 配置，预期为 `32`
- `isolate_local_optimizer_state=True`
- `federated_single_scheduler_step=True`

如果仓库真实 CLI 名称与上述语义名称不同，复用现有 CLI，不要建立第二套参数体系。若真实 Experiment-D launcher 的学习率、batch size、PromptFL config 或其他关键训练参数与上述预期冲突，先在最终报告中明确指出，并以当前经过验证的正式 Experiment-D 配置为准；不得静默猜测。

本次不运行：

- fine-class Dirichlet；
- seeds 1 或 2026；
- 20 轮训练；
- Experiment-D 的 5/10/20 三轮 Gate；
- 100 个随机候选；
- 在线 CUSP；
- CAPT+CUSP 或其他训练器组合。

==================================================
三、先检查真实仓库，再编辑
==================================================

开始修改前：

1. 阅读仓库中的 `AGENTS.md`、README、环境文件和相关配置。
2. 检查 `git status` 与 `git diff`，保留用户已有改动，不覆盖无关文件。
3. 使用 `rg` 找到以下真实活动路径及调用链：
   - federated training 入口；
   - PromptFL 模型构建、forward、训练和评估；
   - Client-LT 划分；
   - FedAvg 聚合；
   - 第 10 轮 dump；
   - cached-feature replay；
   - Oracle 四候选构造；
   - shell launcher；
   - CUSP 测试。
4. 重点核对当前代码是否仍存在：
   - 真实 dump 加载后无条件 `RuntimeError`；
   - synthetic 或真实路径仍输出六候选；
   - `ties`、`tail_feasible_cone` 或假的 `classwise_support_normalized`；
   - `DRY_RUN=0` 固定退出；
   - `normalize_to_budget()` 只缩小、不执行本次要求的等范数；
   - finite difference 不稳定后仍继续求解；
   - CUSP solver 失败后仍读取 official test；
   - `candidate_frozen_at` 在 test 评估后才写；
   - dump/cache hash 只记录但加载时不验证；
   - 缺少新版 `trainers/promptfl.py` replay helper。

不要相信历史修改说明。最终报告必须基于实际文件、实际 diff 和实际执行结果。

如果真实仓库入口与预期文件名不同，修改真实活动路径，并在最终报告中列出实际路径。不要为了匹配本 Prompt 复制一套平行实现。

==================================================
四、修改边界
==================================================

优先复用并修改下列实际文件或同义文件：

- `federated_main.py`
- `trainers/promptfl.py`
- `utils/oracle_cusp.py`
- `scripts/oracle_cusp_single_round.py`
- `scripts/cusp_oracle_round1.sh`
- `tests/test_oracle_cusp.py`

允许新增少量、职责清晰的 helper 或 fixture，但不要引入新的框架。

禁止修改：

- PromptFL 原始 forward、loss、本地训练和优化器数学语义；
- `average_weights(...)` 或当前 FedAvg 的样本量加权语义；
- Client-LT 划分公式；
- CIFAR100-LT 全局样本生成；
- CLIP backbone 的冻结状态；
- CAPT、FedTEF、Adapter、LoRA 等无关训练器；
- 既有正式实验结果。

默认关闭 Oracle flag 时，原 PromptFL/FedAvg 训练行为和输出必须保持不变。

==================================================
五、训练侧第 10 轮 dump
==================================================

第 10 轮必须保存构造离线候选所需的最小完整状态。

5.1 必须保存的状态

保存：

- `global_before_trainable`：第 10 轮所有客户端本地训练开始前的全局可训练参数；
- 每个选中客户端本地训练后的全部 trainable state；
- 每个客户端的 `client_id`；
- 每个客户端样本数和原始 FedAvg weight；
- 每个客户端逐类训练样本数；
- `global_after_fedavg_trainable`；
- trainable keys、shape、dtype、numel；
- 动态 class counts、tail ids、head/non-tail ids；
- partition、seed、split seed、round、clients、frac、local epochs；
-完整 resolved args/config 路径；
-训练数据、client split 和 schedule fingerprint。

PromptFL `CSC=True` 时，Oracle 向量必须覆盖所有 `requires_grad=True` 的浮点参数，至少包括：

- `prompt_learner.general_ctx`
- `prompt_learner.class_aware_ctx`

不得只保存 `class_aware_ctx`。

5.2 稳定顺序与一致性

dump 中客户端按 `client_id` 升序排列，并对 local states、sample counts、FedAvg weights、class counts 使用完全相同的 permutation。

保存前验证：

- 恰好有 30 个客户端；
- client ids 无重复；
- FedAvg weights 均有限、非负、和为 1；
- 所有 local/global trainable states 的 key、shape、dtype 一致；
- 使用 dump 重建：

  `global_before + Σ_i w_i(local_i-global_before)`

  必须与 `global_after_fedavg_trainable` 在明确容差内一致。

5.3 train feature cache

utility 只能使用 train split，禁止使用 official test。

使用独立的 deterministic train-view：

- 对 Global-LT train split 使用确定性 evaluation-style transform；
- 不复用带 shuffle 或随机增强的 live training loader；
- 不修改 live training dataset/loader 的 transform；
- 样本身份使用稳定 `(client_id, dataset_index)`，不得使用 loader 遍历 offset；
- 按 `(client_id, dataset_index)` 排序；
- cache 至少包含 normalized image feature、label、client id、dataset index、source；
- `source="train"`；
- `test_used_for_utility=false`。

由于本次训练在第 10 轮结束，cache 只在训练完成后生成，不需要实现“第 11–20 轮轨迹一致性”工程。不要为本次任务继续开发 20 轮 RNG 轨迹对照。

5.4 原子写入与 hash 验证

一个有效 dump 目录至少包含：

- `round_state.pt`
- `train_feature_cache.pt`
- `metadata.json`

先写同级临时目录，完成自检和 SHA-256 后，再原子发布为 `round_010/`。失败时：

- 不得留下看似有效的 `round_010/`；
- 写出 `invalid_dump.json`，包含失败阶段、异常和已完成检查。

`metadata.json` 记录 `round_state.pt` 和 `train_feature_cache.pt` 的 hash。Oracle 加载时必须重新计算并核对，hash 缺失或不一致均停止，不能只记录不验证。

==================================================
六、真实 PromptFL cached-feature replay
==================================================

真实路径不得包含占位异常或 synthetic 代替逻辑。

优先在 `trainers/promptfl.py` 增加一个纯评估 helper，例如现有名称 `logits_from_cached_features(...)`，但以仓库真实接口为准。

helper 的语义：

1. 接收已经归一化的 cached image features；
2. 临时加载指定 candidate trainable state；
3. 使用真实 PromptFL：

   `prompts = prompt_learner()`

   `text_features = text_encoder(prompts, tokenized_prompts)`

   `text_features = normalize(text_features)`

   `logits = exp(logit_scale) * image_features @ text_features.T`

4. 不调用 image encoder；
5. 在 `try/finally` 中恢复原 trainable state 和原 train/eval mode；
6. 不修改参数、optimizer 或 scheduler；
7. CPU cache 按需移动到模型真实 device/dtype；
8. 输出与正常 PromptFL forward 在同一确定性小批样本上的 logits 数值一致。

必须增加一致性测试。测试失败时不得通过放宽到没有意义的容差来掩盖问题。

真实 replay 必须：

- 从 dump 恢复 `global_before_trainable`；
- 仅替换 dump 列出的 trainable keys；
- 重建同一 frozen CLIP/text encoder；
- 使用 `train_feature_cache.pt` 计算 utility；
- 对 `source!="train"`、`test_used_for_utility=true` 或 hash 不一致直接拒绝。

==================================================
七、四个候选的唯一定义
==================================================

活动路径中只允许以下方法名：

- `fedavg`
- `random_reweight`
- `classwise_weighting`
- `oracle_cusp`

不得输出或重新加入：

- `ties`
- `tail_feasible_cone`
- `classwise_support_normalized`
- PCGrad、CAGrad 或第五个候选。

定义：

`theta_0 = flatten(global_before_trainable)`

`delta_i = flatten(local_i - global_before)`

`delta_FA = Σ_i w_i delta_i`

`B = ||delta_FA||_2`

若 `B <= eps`，本轮无效并停止。

7.1 等范数公平性

本次比较明确为方向比较。除 FedAvg 外，每个有效 raw candidate direction 都必须缩放为：

`delta_final = B * delta_raw / ||delta_raw||_2`

因此：

`||delta_final||_2 = B`

要求：

- 零向量或非有限向量标记为 invalid，不能参与比较；
- 保存 `raw_norm`、`final_norm` 和 `scale_factor`；
- 使用统一绝对/相对 tolerance 校验等范数；
- train diagnostics、candidate hash 和 test evaluation 全部基于缩放后的 final candidate；
- 不允许一部分候选只满足 `<=B`、另一部分恰好等于 `B`。

7.2 `fedavg`

`delta_final = delta_FA`

必须与 dump 中 `global_after_fedavg_trainable` 一致。

7.3 `random_reweight`

固定：

- `num_random=10`
- `random_seed=42`
- `alpha^(s) ~ Dirichlet(1,...,1)`
- `delta_raw^(s)=Σ_i alpha_i^(s) delta_i`

每个随机方向都按 7.1 缩放到 B。

要求：

- 保留并评估全部 10 个候选；
- 不按 train 或 test 表现挑选“最好随机候选”；
- 主报告随机分布的 mean、std、min、p25、median、p75、max；
- CUSP 的科学比较使用 random median；
- 保存 coefficients/hash、raw norm、scale factor 和全部 test 指标。

7.4 `classwise_weighting`

对 `prompt_learner.class_aware_ctx[c]`：

`S_c = 本轮拥有类别 c 正样本的客户端集合`

`normalized_weight_i,c = w_i / Σ_{j∈S_c} w_j`

`classwise_state[c] = Σ_{i∈S_c} normalized_weight_i,c * local_state_i[c]`

对 `general_ctx` 和其他非 class-wise trainable key，raw state 使用原始 FedAvg state。

若某类没有支持客户端，该 class-aware row 保持 `global_before`，并记录 fallback class id。

构造完整 raw delta 后，按 7.1 对整个方向统一缩放到 B。不得任取前两个客户端平均，也不得把 Experiment-D 的 support counterfactual 冒充该 baseline。

7.5 `oracle_cusp`

客户端更新矩阵：

`D=[delta_1,...,delta_30]`

用 SVD 构造客户端更新子空间，保留 99.9% 平方奇异值能量。若截断后不能在容差内重建 FedAvg direction，追加 FedAvg residual 并重新正交化。

在 train cache 上定义每类平均 decision margin：

`margin(x,y)=logit_y-max_{h!=y}logit_h`

`m_c(theta)=mean_{x:y=c} margin(x,c)`

对正交基 `q_k` 做中心有限差分：

`A[c,k]=[m_c(theta_0+epsilon*q_k)-m_c(theta_0-epsilon*q_k)]/(2*epsilon)`

固定：

- `epsilon=1e-3`
- 用 `epsilon/2` 复核
- sign agreement `>=0.95`
- relative difference `<=0.10`

finite difference 不稳定时：

- 写明失败原因；
- 不调用 solver；
- 不构造 test loader/cache；
- 不读取 official test；
- 整个最小实验标记 `INCOMPLETE`。

对 valid class：

`a_c=A_c/||A_c||_2`

令 `u_FA` 为单位 FedAvg direction 在子空间中的坐标，求解：

maximize

`tau - 0.1*||u-u_FA||_2^2 - 10.0*mean(xi_c)`

subject to

`a_c u + xi_c >= tau`

`xi_c >= 0`

`||u||_2 <= 1`

`mean_valid_head(a_c u) >= mean_valid_head(a_c u_FA)-0.05`

`mean_valid_all(a_c u) >= mean_valid_all(a_c u_FA)-0.05`

solver 必须捕获并结构化记录：

- `cvxpy` dependency missing；
- solver missing；
- infeasible/unbounded；
- numerical error；
- unexpected exception。

solver 不成功时：

- 不得回退为 FedAvg 并伪装 CUSP 成功；
- 不得读取 official test；
- 写 `oracle_solver.json`；
- 退出非零。

solver 成功后，先得到 raw CUSP direction，再按 7.1 缩放到 B，并重新计算最终 train diagnostics。若等范数后的 final candidate 不满足有限性、子空间或预设 head/all train utility 约束，标记 invalid，不进入 test。

==================================================
八、候选冻结与 test 隔离
==================================================

执行顺序必须通过代码结构和测试固定：

1. 加载并验证 round dump；
2. 只用 train cache 计算 utility；
3. 构造全部候选：
   - 1 个 FedAvg；
   - 10 个 random reweight；
   - 1 个 classwise weighting；
   - 1 个 oracle CUSP；
4. 对所有候选完成等范数、有限性、状态重建和 hash 校验；
5. 原子写出：
   - `candidate_states.pt`
   - `candidate_manifest.json`
6. 在 manifest 中写入 `candidate_frozen_at` 和 `test_accessed=false`；
7. 关闭候选构造阶段；
8. 此后才允许创建或读取 official test loader/cache；
9. 用完全相同的 official test 数据和评估函数评估所有候选；
10. test 结果不得返回 solver、重新选择候选或修改超参数。

如果 CUSP solver 失败、finite difference 不稳定、任一确定性候选无效或 candidate freeze 失败，必须在读取 test 前停止。

增加可测试的阶段保护，例如：

- build phase 接口不接收 test loader/cache；
- evaluation phase 只接收已冻结 manifest；
- monkeypatch test loader 构造函数，证明失败路径不会访问 test。

test cache 若保存，必须单独标记：

- `source="official_test"`
- `purpose="test_evaluation_only"`

它不得传入 utility 或 solver API。

==================================================
九、类别定义与最小评估指标
==================================================

根据当前 run 的 Global-LT train class counts 动态定义：

- tail：样本数最少的 bottom 20% classes；
- head/non-tail：其余 80% classes。

排序 tie 使用 class id 作为确定性次级键。保存实际 class ids。

禁止：

- 硬编码 tail 为 80–99；
- 新建虚假的 mid/medium 组；
- 使用 test 分布定义 head/tail。

所有 accuracy 输出统一为 `[0,100]` 的百分点数值。若仓库 evaluator 返回 `[0,1]`，在输出层明确转换，并在 metadata 中写 `accuracy_scale="percent"`。

每个候选只报告：

- `overall_acc`
- `macro_acc`：100 个 per-class accuracy 的简单平均；
- `head_acc`：动态 non-tail classes 的 per-class accuracy 简单平均；
- `tail_acc`：动态 bottom-20% classes 的 per-class accuracy 简单平均；
- `update_norm`
- 相对 FedAvg 的四项 accuracy delta。

同时保存逐类：

- class id；
- group；
- support count；
- correct count；
- class accuracy。

本次不要继续实现 margin survival、Pearson/Spearman、跨 topology Gate 或复杂论文图表。CUSP 构造内部仍使用 train decision margin，但 test 主结果只保留上述准确率。

==================================================
十、输出文件
==================================================

最小实验结果目录至少生成：

1. `candidate_states.pt`
2. `candidate_manifest.json`
3. `oracle_method_summary.csv`
4. `oracle_per_class.csv`
5. `random_reweight_distribution.csv`
6. `oracle_solver.json`
7. `oracle_metadata.json`
8. `oracle_report.md`

`oracle_method_summary.csv` 恰好四个 method 名：

- `fedavg`
- `random_reweight`
- `classwise_weighting`
- `oracle_cusp`

其中 deterministic 方法填单值；`random_reweight` 行明确为 10 个候选的分布汇总，并至少包含 mean、std、min、p25、median、p75、max。不得填事后最佳随机结果。

`random_reweight_distribution.csv` 每个随机候选一行，共 10 行，至少包含：

- `candidate_id`
- coefficient hash
- `raw_norm`
- `final_norm`
- `scale_factor`
-四项 accuracy
-相对 FedAvg 的四项 delta

`oracle_per_class.csv` 保存三个 deterministic 方法的逐类结果；随机逐类结果可以保存 10 个候选的逐类行或确定性分布汇总，但必须在 schema 中明确。

`oracle_metadata.json` 至少记录：

-完整配置；
- dump/hash 校验；
- train/test 数据源；
-动态 head/tail ids；
- trainable keys；
- basis/SVD 信息；
- finite-difference 阈值和结果；
-等范数 tolerance；
-候选冻结时间；
- test 首次访问时间；
-无 test leakage 检查；
-软件依赖版本；
-运行时长。

`oracle_report.md` 用简短表格报告四个方法，并自动给出：

- `PASS_MINIMAL`：
  - CUSP tail acc 严格高于 FedAvg；
  - CUSP tail acc 严格高于 classwise weighting；
  - CUSP tail acc 严格高于 random median；
  - CUSP head acc 相对 FedAvg下降不超过 0.5 个百分点；
  - CUSP overall acc 相对 FedAvg下降不超过 0.5 个百分点；
  -全部候选等范数且无泄漏。
- `FAIL_MINIMAL`：实现完整，但上述科学条件不满足。
- `INCOMPLETE`：依赖、dump、finite difference、solver、候选冻结、test 或输出不完整。

报告必须注明：单 topology、单 seed、单轮 Oracle 结果不能直接作为论文结论。

==================================================
十一、一条命令完成真实最小实验
==================================================

修改现有 `scripts/cusp_oracle_round1.sh`，使其成为本次最小实验 launcher。若保留旧两 topology launcher 对历史复现更安全，则新增 `scripts/cusp_oracle_minimal.sh`，但最终只能指定一个“本次唯一正式 launcher”，避免两个脚本都自称 Round-1 正式入口。

正式 launcher 支持：

- `DRY_RUN=1`：只打印将执行的完整命令，不创建输出目录，不启动 Python，返回 0；
- `DRY_RUN=0`：真实执行；
- `STAGE=all|train|oracle`，默认 `all`；
- `PYTHON_BIN`、`DATA`、`OUTPUT_ROOT` 使用仓库已有风格，可由环境变量覆盖。

`STAGE=all DRY_RUN=0` 必须严格顺序执行：

1. 环境和数据 preflight；
2. 一个 Client-LT、seed 42、10 轮训练；
3. 验证 `round_010/round_state.pt`、`train_feature_cache.pt`、`metadata.json` 和 hash；
4. 运行真实 `oracle_cusp_single_round.py`；
5. 检查第十节全部输出和候选集合；
6. 返回真实状态码。

任一阶段失败立即停止。不得继续 test 或写科学 PASS。

覆盖规则：

- 默认拒绝非空训练输出目录；
- 不实现假的 `OVERWRITE=1` 或假的 `RESUME=1`；
- `STAGE=oracle` 只允许读取一个已经通过完整性校验的 dump，并写入新的/空的 Oracle 结果目录；
- 不删除用户旧结果。

`DRY_RUN=1` 应打印一条训练命令和一条 Oracle 命令，并明确显示：

- `client-longtail`
- seed 42
- 30 clients
- frac 1.0
- local epochs 3
- rounds 10
- Oracle round 10
- Client-LT 三个参数
-实际 dataset/trainer config

本次 Codex 修改任务不得执行 `DRY_RUN=0`，不得启动真实 10 轮训练。只验证 dry-run 和无需完整训练的 smoke。

==================================================
十二、环境 preflight
==================================================

先识别用户实际用于 CAPT 的 Python/conda 环境，不要擅自创建新环境，也不要静默安装或升级 PyTorch/CUDA。

真实实验至少检查：

- `torch`
- `yacs`
- `cvxpy`
-项目自身训练依赖
-可用 CVXPY solver
- CUDA/device
- dataset 根目录
- dataset/trainer config

测试阶段检查 `pytest`。

如果缺依赖：

- 代码仍需完整实现；
- launcher preflight 必须在训练前给出清楚错误；
- 最终报告给出与当前环境兼容的最小安装命令；
- 不得把代码占位问题归因于环境；
- 不得声称已经可以直接启动真实实验。

不要为了安装 `pytest`/`cvxpy` 升级现有 torch、torchvision、CUDA 或 numpy 主版本。

==================================================
十三、测试与验证
==================================================

13.1 静态检查

- 修改的 Python 文件全部通过 `python -m py_compile`；
- launcher 通过 `bash -n`；
-相关 CLI `--help` 可运行；
- `rg` 证明本次活动路径中没有：
  -真实路径占位 `RuntimeError`；
  -六候选集合；
  -`ties`；
  -`tail_feasible_cone`；
  -`classwise_support_normalized`；
  -非 dry-run 固定退出。

历史未调用 helper 若必须为兼容保留，可存在，但本次 launcher、Oracle 入口、输出和测试不得引用。

13.2 dump/cache 测试

至少覆盖：

- stable client sorting 后 state/weight/count 对齐；
- flatten/unflatten 覆盖 general/class-aware ctx；
- FedAvg state 精确重建；
- stable dataset index；
- train cache 拒绝 test source；
- hash 加载验证；
- cache 或 metadata 写入失败不发布有效 dump；
-默认关闭 Oracle 时不改变 PromptFL/FedAvg 活动训练路径。

13.3 replay 测试

至少覆盖：

- cached-feature logits 与真实 forward logits 一致；
-候选加载后在 `finally` 恢复模型参数和 mode；
-不调用 image encoder；
- CPU cache 正确移动到真实 device/dtype；
- utility API 不接受 test cache。

13.4 四候选测试

至少覆盖：

-方法集合恰好为四个固定名字；
- random 候选恰好 10 个、seed 42 可复现；
-不得选择 best random；
- classwise row 只聚合真实支持客户端；
-无支持类回退 global-before；
- general/non-classwise raw state 使用 FedAvg；
-每个有效候选最终 norm 与 B 一致；
-零/NaN direction 被拒绝；
- finite difference relative threshold 生效；
- finite difference 不稳定时不调用 solver/test；
- solver exception 写结构化失败；
- solver 失败时不构造 test loader；
-候选冻结前禁止 test；
- manifest 中 `candidate_frozen_at < test_first_accessed_at`；
- CUSP final candidate 仍在客户端更新子空间并通过最终约束复核。

13.5 synthetic/fixture smoke

构造不依赖数据下载的小型 fixture，完整走通：

`dump load -> train utility -> 4 methods/13 concrete candidates -> candidate freeze -> fake test evaluation -> outputs`

说明：

- 4 methods；
- 13 concrete candidates = 1 FedAvg + 10 random + 1 classwise + 1 CUSP。

不得再把“CSV 中出现四个名字”当作四候选真实成功的充分条件。所有 concrete candidates 必须有成功状态、状态向量、norm 和评估记录。

13.6 测试命令

使用项目实际 Python 环境运行：

- `python -m pytest tests/test_oracle_cusp.py -q`
-必要的邻近回归测试
- `bash -n <正式 launcher>`
- `DRY_RUN=1 bash <正式 launcher>`

若 Windows 中 WSL `bash` 不可用，允许使用 Git Bash，但最终报告写出实际可复制命令。

不得执行真实 10 轮训练。

==================================================
十四、启动实验前的硬验收门槛
==================================================

只有以下全部成立，最终报告才允许写：

`READY_FOR_MINIMAL_PILOT=true`

1. 真实 Oracle 路径不再包含占位异常；
2. 活动路径只包含四个固定 method；
3. 第 10 轮 dump 完整、原子、可重建、加载时验证 hash；
4. cached-feature replay 与真实 forward 一致；
5. finite difference 不稳定会在 test 前停止；
6. solver 失败会在 test 前停止；
7.全部候选先等范数、冻结、落盘，再读取 official test；
8. 13 个 concrete candidates 均可由 fixture 完整构造和评估；
9. 输出 schema 完整；
10. launcher 的 `DRY_RUN=0` 真实执行链存在；
11. launcher dry-run 参数与第二节完全一致；
12. `pytest`、静态检查、fixture smoke、launcher dry-run 全部通过；
13.真实项目环境具备 `yacs`、`cvxpy` 和可用 solver；
14.没有 test leakage；
15.本次没有自动启动真实训练。

任一项不满足：

`READY_FOR_MINIMAL_PILOT=false`

并逐项列出 blocker。不得用“代码应该可以”“依赖安装后大概可运行”代替验证。

==================================================
十五、明确暂缓，不要顺手实现
==================================================

本次不要修改或扩展：

- Gate 0 / `summarize_cusp_gate0.py`，除非它被本次 launcher 错误调用，需要解除调用；
- Gate 1 / `summarize_cusp_gate1.py`；
- Dirichlet 对照；
-多 seed；
- 100 random；
-第 11–20 轮训练轨迹一致性；
- margin survival；
- Pearson/Spearman；
-在线 CUSP；
- memory、EMA、residual、router、class-specific expert；
- secure aggregation；
- Adapter、LoRA、CAPT+CUSP；
-论文图表和大规模 sweep。

不要删除历史脚本或历史输出，只需保证本次唯一正式 launcher 不调用它们。

==================================================
十六、完成时汇报
==================================================

最终只汇报实际事实：

1. 实际训练入口、Oracle 入口、launcher；
2. 修改文件及每个文件解决的问题；
3. 旧版本问题中哪些确实存在、如何修复；
4. 四个 method 与 13 个 concrete candidates 的验证结果；
5. dump/cache/replay/冻结/test 隔离的验证结果；
6. 实际执行的全部命令和退出码；
7. pytest、fixture smoke、dry-run 结果；
8. 当前环境依赖与 solver 状态；
9. `READY_FOR_MINIMAL_PILOT=true/false`；
10. 若为 true，给出用户唯一需要执行的真实命令，例如：

   `DRY_RUN=0 STAGE=all bash <正式 launcher>`

11. 明确注明该真实命令未由 Codex 执行。

不要只给“已修改”的概述。若没有实际执行测试，不得报告测试通过。
```

## 最终口径

这份 Prompt 的完成目标是让用户得到一条可信的最小实验执行链，而不是提前完成论文最终实验。若最小实验得到正结果，再扩展 Dirichlet、三个 seed、100 个随机候选和完整机制分析；若最小实验失败，则先停止 CUSP 扩展并分析失败原因。
