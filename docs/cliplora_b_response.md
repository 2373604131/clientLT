# 方法 B：独立 donor C 与尾类正响应监督

日期：2026-09-29。已实现 CPU 可验证版本，尚未进行服务器 CIFAR100／CLIP GPU 训练。

## 固定目的与实现

功能保持能够改善已有尾类能力的维护，但直接监督来源仍然有限。缺少目标尾类标签的客户端在训练侧探测中也可能提供正向帮助，B 尝试通过定向知识利用增强有效学习支持。是否产生独立的测试增益，由本次实验判断。

本版保留旧共享 B 的每 donor、每 LoRA 模块一个独立 C。它没有采用上一个 directed 版本的“先固定混合 donor、再给类别组合学习 C”，也没有改为直接优化 LoRA B。

| 项目 | 固定设置 |
|---|---|
| A | Full-CP，λ=10、μ=1，沿用既有训练规则 |
| 普通 B | 本地训练与按样本量 FedAvg 不变 |
| 反馈 | 协议尾部客户端组内的尾类完整训练池；当前27、28、29，共119张、20类 |
| 来源资格 | 在目标组之外，且实际本地训练数据缺少正在帮助的类别 |
| 来源探测 | 相同普通聚合模型 B₀ 上分别注入 0.1 ΔBⱼ；A 固定 |
| 筛选指标 | 原始 logits 正确类相对最强竞争类的间隔；先逐类求样本均值，再取两个视图增益的最小值 |
| 来源数量 | 每类正增益 Top-3，阈值 1e-6；按正增益归一化固定教师权重 |
| 两个视图 | 确定性评估图像及其水平翻转；不使用随机增强或独立验证集 |
| C | 每个选中 donor、每模块独立 r×r，当前 rank=4、6个模块，每次迁移从零开始 |
| 共享候选 | B₀ + meanⱼ(ΔBⱼ Cⱼ)，分母为选中来源并集大小 |
| 优化 | Adam，lr=0.3，2个全局同步步骤；收到全部目标反馈后才 step |
| 目标 | 类等权两视图 LA + 1.0 × 正响应损失 + 0.001 × C平方正则 |
| 迁移轮次 | 30、40、50、60、70、80、90、100；固定第二步提交，无测试门控 |
| 推理 | 单一共享100类模型；保持原 LoRA rank，无类别专属分支 |

响应损失系数1.0是首版固定开发配置，尚未证明最优；不额外扫描。只保留原矩阵正则，没有引入直接训练 B 设计稿中的更新范数约束。

## 教学目标如何构造

对目标样本 x、真实类别 c 和全部竞争类别 q，记 m_cq=z_c−z_q。各 donor 的增量都相对同一个 B₀ 计算。对该类选中的来源先求加权增量，再取跨视图较小值的正部：

```text
r_cq(x) = max(0, min_v sum_j p_cj [m_cq(x_v; B₀ + 0.1 ΔB_j) - m_cq(x_v; B₀)])
target_cq(x_v) = m_cq(x_v; B₀) + r_cq(x)
```

来源权重、教师输出、基准输出与目标全部停止梯度。不为每个竞争类别临时挑最大 donor。无正响应的关系使用基准作为软参考。

单样本响应损失为 `sum_(q != c) omega_cq * relu(target_cq - student_margin_cq)^2`。omega 是基准模型在非真实类别上的 softmax 概率，固定并停止梯度。达到或超过目标不受惩罚。LA 与响应损失均按“类内样本平均→类别等权→两视图平均”汇总；C 正则只加一次，除以 donor 数×模块数。

因此并集是 C 的参数基底，逐类关系还会单独改变该类的监督。参数化和教学内容分别承担职责。不同类仍共享一个模型，不能宣称来源在推理时完全隔离或每类都不会受损。

## 只新增两组完整实验

| 组别 | `--response-variant` | 用途 |
|---|---|---|
| 主方案 | `positive` | 独立 donor C 学习逐类正响应目标 |
| 匹配对照 | `zero` | 同样筛选来源、构造教师并训练同样的 C，仅把教学增量 r 置零 |

zero 保留基准响应软参考、真实尾类监督和 donor 更新基底，所以它不是“完全不使用 donor”或“直接训练 B”的对照。positive 与 zero 用于判断正响应监督的附加价值；不能据此单独证明 donor 筛选最优或 C 必需。

两组都依据**置零之前**测得的正响应决定事件是否可用：整次事件没有正响应则均跳过，不进行无支持的额外微调。它们从相同协议初始化开始完整训练，后续轨迹可能不同，不要求每个后续事件来源相同。

主指标：固定 Tail20 在提交轮81—100的测试准确率均值。末轮单列，不能替代主要指标。已有 A-only、旧共享 B、三个轮换版本与上一版 directed 作为历史参照，无需为本次补跑这些组；它们与新方法同时存在多个设计差别，不能当作单因素消融。

主方案和 zero 均使用两视图。当前119张目标图、2步对应476次反向图像访问，上一版 directed 的单视图为238次。新增探测与诊断前向单独记录，不能宣称和旧版训练计算完全相同。仍只有2次 C 梯度同步。

## 在服务器启动

先将本次修改同步到服务器，在 `/data/yzh/clientLT` 及原 `clientLT` 环境中执行。GPU编号按空闲卡调整；下面两个命令各占一张卡，均从头训练100轮。

主方案：

```bash
CUDA_VISIBLE_DEVICES=1 python -u scripts/run_cliplora_b_response.py --response-variant positive --reference-run references/full10_clientlt --data-root DATA --fast-execution-v2 --feedback-batch-size 128 --feedback-cache-gib 4
```

匹配对照：

```bash
CUDA_VISIBLE_DEVICES=2 python -u scripts/run_cliplora_b_response.py --response-variant zero --reference-run references/full10_clientlt --data-root DATA --fast-execution-v2 --feedback-batch-size 128 --feedback-cache-gib 4
```

`references/full10_clientlt` 应含已有划分和协议文件：`partition_manifest.csv`、`bridge_metadata.json`、`protocol/full_schedule.json`、`protocol/eri_protocol.json`、`protocol/probe_manifest.csv`。这里只复用划分/顺序/调度，不加载旧实验训练后的模型，也不要求这个目录存在 `sfra_config.json`。

不要附加旧版 `--transfer-tail-weight` 或 `--transfer-non-tail-sampling`。目标组默认从 Client-LT 协议推导，不在代码里硬编码27—29；其他划分需要明确目标端定义。

输出分别位于：

```text
output/cifar100_LT/sfra_b_response/seed42/client-longtail/full-cp/
  lambda10_mu1_protocol42_bshared_b_lr0.3_probe0.1_reg0.001_directed_k3_g1e-06_targetsprotocol_responseC_positive_rw1_fast_v2_f128_c4/
  lambda10_mu1_protocol42_bshared_b_lr0.3_probe0.1_reg0.001_directed_k3_g1e-06_targetsprotocol_responseC_zero_rw1_fast_v2_f128_c4/
```

中断后，在对应原命令末尾加 `--resume`。按已完成轮次恢复，半轮重做；positive/zero 不可互相续训，也不能使用旧 shared/directed 断点续训。若修改参数，应使用新输出目录。

## 汇总与打包

两组完成后执行一次：

```bash
python -u scripts/run_cliplora_b_response.py --stage pack
```

会先生成 `output/cifar100_LT/sfra_b_response/analysis`，再生成 `output/cifar100_LT/sfra_b_response_analysis.tar.gz`。该压缩包包含两组已存在的结果和证据，不包含大模型权重、原始图片或恢复断点。打包时不附加仅含协议的 reference 目录。

主要查看：

- `analysis/performance.csv`：含 positive/zero 标记、固定 Tail20 主要指标及其余组精度。
- `analysis/b_response_source_weights.csv`：逐类来源、权重与缺标签记录。
- `analysis/b_response_donor_scores.csv`：两个视图的探测增益和是否选中。
- `analysis/b_response_response_metrics.csv`：迁移前后的逐类响应损失、正响应覆盖和达到目标比例；这是训练侧量。
- `analysis/b_transfer_loss_trace.csv`：同一完整尾类池、两视图上 C=0、第一步后、第二步后的目标值。
- 各事件 `response_targets.npz`：本地位置、标签、基准/选中donor/提交学生 logits、教师增量和权重，可复算监督来源；不含原图。
- 各事件 `matrices.npz` 与 `calibration_manifest.json`：C 的 donor 索引、按类监督关系及模块顺序。
- 各事件 `commit.pt`：服务器本地留存，可精确重建更新；不进入分析包。

这些都是训练过程自动保存的记录，不需要另行执行逐轮离线重放。

## 同步清单与验证范围

新增：`utils/b_response_math.py`、`utils/cliplora_b_response.py`、`scripts/run_cliplora_b_response.py`、`tests/test_sfra_b_response.py`、本文档。

接线修改：`federated_main.py`、`utils/cliplora_sfra.py`、`scripts/run_cliplora_sfra.py`、`tools/sfra/summary.py`。旧版实现文件没有重写，原有配置字典在未启用 response profile 时保持原路径。

CPU 验证覆盖来源资格、双视图增量、同并集不同关系改变 C 梯度、独立 donor C、分批/联合优化一致性、真实归一化 logits＋LA、A/B 冻结、共享残差单次提交、空来源跳过、恢复配置隔离、汇总与打包。尚无本机 GPU，也没有在服务器启动正式实验；实际显存、耗时和泛化效果以服务器运行为准。
