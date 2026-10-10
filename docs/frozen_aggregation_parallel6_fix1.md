# 六客户端一致性检查失败：FIX1

**已被后续用户要求取代：** 当前v3代码取消串行/并行对比及阈值拦截，默认恢复原CUDA执行设置。请使用[当前启动说明](frozen_aggregation_controls_seed42.md)。下面仅保留FIX1的历史记录，不作为当前启动指令。

用户提供的 TailRW16 smoke 记录中，client21、client25 的串行/六并行最大差值分别为 `3.6931597e-6`、`7.8408048e-6`，超过原来的逐元素 `atol=1e-6, rtol=1e-5` 检查。client25 在两客户端短队列时只有 `1.8626451e-9`。这些记录证明并行执行没有通过一致性检查，但不能单凭差值确定是哪一个GPU算子造成的，也不能据此认定6并行普遍不可用。尚未开始100轮正式训练。

已发现的代码问题：训练使用多个CUDA stream，但没有固定cuBLAS工作区；`federated_main.py` 还会开启cuDNN自动算法搜索。[NVIDIA文档](https://docs.nvidia.com/cuda/cublas/index.html#results-reproducibility)说明多stream执行时内部工作区选择可以影响复现；[PyTorch文档](https://docs.pytorch.org/docs/stable/notes/randomness.html)说明cuDNN自动算法搜索及非确定性算法的影响。它们是本次修正针对的候选原因，尚无服务器GPU实测证明本次失败的唯一根因。

FIX1做了以下修改：

1. 在新训练子进程导入/初始化CUDA前设置 `CUBLAS_WORKSPACE_CONFIG=:4096:8`。
2. 两个冻结A对照默认使用 `--cuda-policy deterministic`：启用严格确定性算法、关闭cuDNN benchmark、启用cuDNN deterministic；TF32开关保持原设置，实际值另行记录。
3. 主程序初始化会重设benchmark，因此runtime在进入训练前再次应用策略，每轮检查实际设置。策略写入注册计划、客户端执行配置及 `cuda_numerics.json`。
4. 保留原来的误差阈值与六客户端并发上限。smoke增加同批次串行重复对照，所有串行重复/六并行/两客户端检查都必须通过；任何一次失败都会停止。
5. 报告每个客户端的参数误差、最差参数名、训练步数/样本数/调度步数、batch计划一致性及分配的执行槽；失败原因直接输出到日志。

损失、训练epoch、聚合公式及正式训练预算均未修改。确定性算法可能影响耗时，需以新的GPU试跑测量，不能沿用失败试跑的1.137倍当作修复后的加速结果。CPU测试不能确认GPU修复成功。

## 同步并重跑

将 `frozen_aggregation_parallel6_fix1.zip` 放入服务器仓库根目录，然后执行：

```bash
unzip -o frozen_aggregation_parallel6_fix1.zip
```

本地代码尚未自动提交或推送Git；仅 `git pull` 不会获取尚未推送的改动。

使用新目录 `frozen_aggregation_controls_v2_parallel6`，保留原v1失败记录。代码或数值策略改变后不能直接续跑旧注册计划，也不要手工改旧计划的指纹来绕过保护。

TailRW16，示例GPU0，一行命令：

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 python scripts/run_cliplora_joint_aggregation.py --stage run --aggregation-rule tailrw16 --client-concurrency 6 --cuda-policy deterministic --output-root output/cifar100_LT/frozen_aggregation_controls_v2_parallel6/tailrw16
```

FedAvg，示例GPU1，一行命令：

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 python scripts/run_cliplora_joint_aggregation.py --stage run --aggregation-rule fedavg --client-concurrency 6 --cuda-policy deterministic --output-root output/cifar100_LT/frozen_aggregation_controls_v2_parallel6/fedavg
```

两条命令分别放在两个终端。每条都先smoke，再自动正式训练。若使用两个独立计算节点，每条可分别使用各节点分配的GPU0；同一台服务器则选择不同GPU。两组必须使用同一修订版本和同一数值策略。

仍在单服务器GPU2/GPU3同时启动时可用：

```bash
python scripts/run_cliplora_frozen_aggregation.py --stage server --gpus 2 3 --client-concurrency 6 --cuda-policy deterministic
```

查看和收集v2结果：

```bash
python scripts/run_cliplora_frozen_aggregation.py --stage status --output-root output/cifar100_LT/frozen_aggregation_controls_v2_parallel6
python scripts/run_cliplora_frozen_aggregation.py --stage pack --output-root output/cifar100_LT/frozen_aggregation_controls_v2_parallel6
```

若smoke仍失败，日志会直接给出具体比较和client ID，并保留完整 `smoke/seed42/frozen/parallel_benchmark/factor_B.json`。此时没有证据允许跳过检查或放宽阈值，需根据新的串行重复和GPU策略记录进一步定位。

历史新聚合是4并行且未启用此确定性策略；报告会标记执行条件差异。若初始预测也出现变化，会明确标记不匹配，不会强行视为同条件对照。旧的完成结果依然可以读取核验。
