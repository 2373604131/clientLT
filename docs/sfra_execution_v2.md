# SFRA 第二版执行加速与短程测速

目标是减少相同实验的耗时。没有修改训练 batch=32、FP32、学习率、客户端顺序、每类 witness 数量、两种视图、三步修正、CP 公式或 A/B 聚合规则。

## 这次改了什么

- **显卡缓存**：固定视觉前段特征可常驻 GPU，默认预算及硬上限均为 4 GiB，超过上限的配置会被拒绝；填入缓存时保留至少 4 GiB 空闲显存，不满足预算的条目留在 CPU。仅保存原精度特征，不缓存变化中的最终特征，不缓存随机训练增强。按当前 5,842 张 witness、两个视图、197×768 个 FP32 token 估算，完整前段缓存约 6.6 GiB，因此只缓存其中一部分到 GPU，其余留在 CPU。
- **批量反馈**：修正/测量的 forward batch 从旧版 32 调为默认 64，可试 128。仍在单个客户端内部合批，每个 token、每个视图分别按自己的样本数求均值，分类损失保留原客户端/类别权重。
- **减少小操作与等待**：用批量归约替代逐 token CUDA 操作；修正循环不再把非零系数列表搬回 CPU。全零系数批也做反向传播，换取取消同步；日志分别记录实际跳过数（v2 为 0）与全零系数批次数。这是速度上的取舍，不保证所有硬件都获益。
- **来源识别**：仍逐 token/视图计算独立完整 A 梯度、梯度范数与 QR 投影，复用 GPU 缓存。有限值检查移到整次测量返回之前，减少成千上万次主机等待；NaN/Inf 仍会报错，不能进入来源判断或参数提交。
- **独立记录**：新增 `execution_config.json` v2，目录后缀含批量/缓存设置，例如 `_fast_v2_f64_c4`。旧 `--fast-execution` 保留原执行路径和目录。

缓存预算不是峰值显存上限，也不是不会 OOM 的保证；模型、激活和 CUDA 工作区还要占显存。若显存不足，先减小反馈批量或缓存预算，在新配置目录重新试跑。设置 `--feedback-cache-gib 0` 可测试 CPU 缓存。未引入 AMP、TF32、torch.compile 或实验性的批量 Jacobian 路径。

## 在服务器先比较 3 轮

使用同一张 GPU、同一环境、同一数据划分，**依次**运行旧版和新版。不要同时跑这两个测速任务。下面用此前已经完成的均权 Full-CP 实验作为协议来源，重放它的划分与客户端顺序，不加载它训练后的参数。

```bash
conda activate clientLT
reference=output/cifar100_LT/sfra_b_aggregation/seed42/client-longtail/full-cp/lambda10_mu1_protocol42_bagg_uniform8_fast

CUDA_VISIBLE_DEVICES=4 python -u scripts/run_cliplora_sfra.py \
  --method full-cp --retention-weight 10 --classification-weight 1 \
  --b-aggregation uniform-transfer-rounds --reference-run "$reference" \
  --output-root output/sfra_speedcheck --fast-execution --stop-after-round 3

CUDA_VISIBLE_DEVICES=4 python -u scripts/run_cliplora_sfra.py \
  --method full-cp --retention-weight 10 --classification-weight 1 \
  --b-aggregation uniform-transfer-rounds --reference-run "$reference" \
  --output-root output/sfra_speedcheck --fast-execution-v2 \
  --feedback-batch-size 64 --feedback-cache-gib 4 --stop-after-round 3
```

`--stop-after-round 3` 在第 3 轮完成测试并保存 checkpoint 后退出。实验计划仍为 100 轮，不产生 `completion.json`。不需要手动中断。

对比脚本只读文件，不加载模型：

```bash
python scripts/compare_sfra_timing.py \
  output/sfra_speedcheck/seed42/client-longtail/full-cp/lambda10_mu1_protocol42_bagg_uniform8_fast \
  output/sfra_speedcheck/seed42/client-longtail/full-cp/lambda10_mu1_protocol42_bagg_uniform8_fast_v2_f64_c4
```

默认比较共同完成的第 2～90 轮，因此两次短跑实际比较第 2、3 轮，排除冷缓存轮次。输出每个阶段、反馈合计、已记录阶段总和；当完成轮数相同时，另列 `progress.json` 的 runtime 总耗时倍率（包含冷缓存与 checkpoint，但不包含它计时开始前的模型/数据构建）。程序检查算法配置、分区、客户端顺序与初始模型/测试集指纹，条件不匹配时不报告倍率。

3 轮只能初筛常规训练/反馈速度，不能代表后期历史状态、90 轮之后的 B-only 阶段或第 30 轮起的 B-transfer。最终还应比较完整运行时间与原有科学指标。

若 64 合批有收益且显存有余量，可用 `--feedback-batch-size 128` 再做一次相同短跑；目录自动分开。正式继续 64 的试跑时，在原新版命令后追加：

```bash
--resume --stop-after-round 0
```

恢复会继续原始参数和随机状态；仅暂停轮次允许显式改变。缓存重建，不能把 v1 checkpoint 当作 v2 checkpoint 接着跑。旧的 `command.json` 保留初始暂停设置，后续每次完整续跑都应显式指定 `--stop-after-round 0`。

## 如何判断能否快两倍

用户的第 89 轮记录：来源识别约 39 秒，三次修正约 87 秒，最终测量约 9 秒，反馈合计约 135 秒。这不是完整轮次时间。

若旧版每轮普通训练等耗时为 T，新版反馈耗时为 F，则只优化反馈时，总加速约为 `(T + 135) / (T + F)`。例如 T=135 秒、F=45 秒，反馈快了 3 倍，整轮只快了 1.5 倍。该数字仅为算例，不是性能预测。

如果反馈已经很快、普通训练占主导，下一步再测不同 DataLoader worker 数量以及设计可复用的加载器。不要给 30 个客户端盲目各开 8 个常驻 worker。扩大训练 batch、减少来源检查频率、减少三步修正或降低精度，都需要作为新的实验条件评估。

## 验证与边界

本地使用 CPU、小型真实仓库 VisionTransformer/LoRA 进行数值验证：不同 witness 微批与反馈合批大小、独立来源梯度范数/投影、不等 token 大小/类别频率、非活跃 token、全零系数、缓存复用与失效、异常恢复、完整三步修正与历史判断，以及暂停/续跑和测速协议校验。

2026-09-25 本地验证：相关测试共 65 项，64 项通过，1 项 CUDA 测试跳过；7 个相关 Python 文件语法检查通过。耗时对比脚本还在一份已有完整实验记录上做了自比较，输出为 1.00×，该检查不是新版性能测试。

```bash
python -m unittest discover -s tests -p "test_sfra*.py" -q
```

CUDA 缓存预算及混合 CPU/GPU 缓存的测试在无 CUDA 环境下跳过，可在服务器上执行：

```bash
CUDA_VISIBLE_DEVICES=4 python -m unittest discover -s tests -p "test_sfra_resident_feedback.py" -q
```

没有本地 GPU 性能结果，不承诺 2× 加速。合批与归约顺序会带来 FP32 舍入差异；固定单元测试的数值接近不代表 100 轮轨迹逐位相同，也不能保证所有临界阈值判断一致。

同步与数据传输优化依据：[PyTorch 性能指南](https://docs.pytorch.org/tutorials/recipes/recipes/tuning_guide)。
