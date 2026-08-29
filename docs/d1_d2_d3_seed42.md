# D1 汇总与 D2/D3（CIFAR-100 ClientLT，seed 42）

这组实验只负责确认现象并决定下一步方法方向，不是 SOTA 实验，也不允许直接写成论文结论。数据划分、客户端数量和训练载体保持现有仓库协议：`cifar100_LT`、正常 `client-longtail`、30 客户端、全参与、seed/split seed 42、vision-only ClipLoRA。G0 已冻结的配置必须是 `candidate_r4`：top 4 visual blocks、rank 4、alpha 2、q/k/v。

## 1. 重新汇总已有 D1（不训练）

旧判据把没有严格 supporter 的 tail 类也当作随机对照失败，因此混合了两个不同问题。新汇总分别报告：

- 现象：在存在严格 supporter 的类别上，support-normalized 是否优于 FedAvg 和 matched-random p95；
- 覆盖：`client class fraction > 0.1` 能覆盖多少 tail 类。

在服务器已激活 `clientlt` 环境后前台运行：

```bash
STAGE=d1-summary OUT_ROOT=output/g0_d1_seed42 bash scripts/run_g0_d1.sh
```

主要输出是 `output/g0_d1_seed42/d1_summary/d1_verdict.json`。判定含义：

- `D1_FULL_PASS`：稀释现象成立，严格 supporter 规则覆盖也足够；
- `D1_SUPPORTED_WITH_COVERAGE_GAP`：稀释现象成立，但硬 supporter 阈值漏掉太多 tail 类；
- `D1_PHENOMENON_NOT_SUPPORTED`：有效类别上也没有稳定稀释证据。

`method_ready` 只有现象和覆盖同时通过才会为 true；D1 通过本身仍不等于方法已经设计完成。

## 2. 一次训练，同时获取 D2/D3 载体

D1 当时没有保存每轮 30 个客户端的紧凑 LoRA 状态，因此 D2/D3 需要补跑一条 80 轮轨迹。它不是两次训练：同一条 seed-42 FedAvg 轨迹在 round 20/50/80 保存无测试访问的 compact dump，D2 和 D3 共同复用。

全流程前台运行：

```bash
GPU=0 DATA_ROOT=DATA FREEZE=output/g0_d1_seed42/lora_freeze.json OUT_ROOT=output/d23_seed42 bash scripts/run_d23.sh
```

命令不使用 `nohup`，训练、离线分析、报错和进度会全部打印在当前终端。若服务器只有 `python3`：

```bash
PYTHON_BIN="$(command -v python3)" GPU=0 DATA_ROOT=DATA FREEZE=output/g0_d1_seed42/lora_freeze.json OUT_ROOT=output/d23_seed42 bash scripts/run_d23.sh
```

也可以分阶段运行：

```bash
STAGE=dump GPU=0 OUT_ROOT=output/d23_seed42 bash scripts/run_d23.sh
STAGE=d2 GPU=0 OUT_ROOT=output/d23_seed42 bash scripts/run_d23.sh
STAGE=d3 GPU=0 OUT_ROOT=output/d23_seed42 bash scripts/run_d23.sh
```

已经完整结束的阶段可跳过：

```bash
SKIP_COMPLETED=1 GPU=0 OUT_ROOT=output/d23_seed42 bash scripts/run_d23.sh
```

如果 D2/D3 离线阶段中断，重新运行对应 `STAGE=d2` 或 `STAGE=d3` 即可；它会确定性覆盖未完成的离线产物，不会重新训练。训练 dump 若中断且三个轮次没有全部保存，则应换一个新的 `OUT_ROOT` 重跑，避免混用半条轨迹。

## 3. D2 在验证什么

D2 检查“TIES/PCGrad 式冲突量是否真的对应 tail damage”。它在上传的原始 LoRA 参数增量空间中计算：

- 客户端增量与其余 29 个客户端聚合方向的 cosine 和 sign disagreement（不使用类别信息，主判据）；
- 客户端增量与该 tail 类 supporter 共识方向的 cosine 和 sign disagreement（离线解释性次判据）；
- supporter 质量和参考覆盖。

类别参考方向对当前客户端做 leave-one-client-out。这样客户端自己的更新不会被用于证明自己与共识一致；若该类没有独立 peer supporter，则明确记为 reference unavailable。

几何量先写入 `d2_geometry_frozen.csv` 并记录哈希，之后才能读取 official test。真实 utility 使用“从 FedAvg 状态减去客户端实际 FedAvg 权重质量”的删除反事实：正值表示该客户端上传对该 tail 类有益，负值表示有害。

主要输出：

- `d2/d2_geometry_frozen.csv`：测试前冻结的几何量；
- `d2/d2_client_class_utility.csv`：逐 client × tail class 的真实贡献；
- `d2/d2_round_summary.csv`：相关性、harm AUC、false-protection rate 和有害损失；
- `d2/d2_verdict.json`：`D2_CONFLICT_PROXY_SUPPORTED` 或 `D2_CONFLICT_PROXY_NOT_SUPPORTED`。

D2 大约需要每轮 1 次 FedAvg 基线和 30 次客户端删除评估，共 93 次 CIFAR-100 test 前向。它没有反向传播，也不重新训练，但不会瞬间结束；终端会打印每个删除客户端的进度。

## 4. D3 在验证什么

D3 检查“表征里已有 tail 信息，但原始 CLIP 分类边界偏向 head”是否成立。每个诊断轮次先在全局训练集做固定、逐类分层的 fit/calibration 切分，并在 test 访问前冻结：

- train-calibration 选择的 logit-adjustment `tau`；
- nearest class centroid；
- class-balanced ridge linear probe。

之后只做一次 official-test 比较：native CLIP logits、logit adjustment、nearest centroid、balanced ridge probe。centroid 和 ridge 都是诊断探针，不是论文方法。

主要输出：

- `d3/d3_calibration_grid.csv`：只在 train calibration 上选 tau 的全过程；
- `d3/d3_frozen_choices.csv`：测试前冻结的选项与哈希；
- `d3/d3_method_metrics.csv`：四种分类头的 head/tail/balanced/H 指标；
- `d3/d3_round_contrasts.csv`：相对 native logits 的增益和 head damage；
- `d3/d3_verdict.json`：表征假设与校准假设是否分别成立。

## 5. 下一步方法决策

- D2 成立、D3 不成立：第一组件优先走 conflict-aware aggregation；
- D2 不成立、D3 成立：不要继续强行做几何冲突，优先做 federated classifier/logit calibration；
- D2、D3 都成立：先比较独立增益和 head 安全，再决定主机制与辅助校准，不能自动解释成 A+B；
- D2、D3 都不成立：当前四个候选解释中，这两条也被排除，需要回到数据/载体或寻找新的缺失现象。

所有 seed-42 判定只用于 discovery。现象与机制冻结后，才扩到其他 seeds 做复查。
