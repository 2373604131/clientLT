# CAPT：单卡四客户端并发试验

入口为 `scripts/run_capt_single_gpu_parallel.py`。使用大表中的 CAPT：seed42、固定的 Client-LT 原始样本分配、30 个客户端、100 轮、每客户端每轮 3 个本地 epoch、batch size 32、FP32、每轮固定聚合。调用现有 CAPT 模型、损失和聚类聚合函数。

这与 `run_capt_global_start.py` 默认保留 MAB 和旧优化器状态的原版入口不同。并发试验固定每个客户端从本轮全局参数出发，创建独立 SGD 优化器，不把一个客户端训练后的参数或动量交给下一个客户端。

## 单卡四并发测速

在服务器仓库根目录运行：

```bash
CUDA_VISIBLE_DEVICES=0 python -u scripts/run_capt_single_gpu_parallel.py --stage benchmark --seed 42 --parallel-clients 4 --reference-run references/full10_clientlt --data-root DATA --num-workers 0 --output-root output/cifar100_LT/capt_single_gpu_parallel_seed42_v1
```

默认使用第一轮日程中的 4 个客户端，完整训练各自的 3 个本地 epoch；串行与四并发均从相同初始参数开始。重复两次，第二次倒转先后顺序，比较中位耗时。保留预热；这里计时的是本地训练，不包含模型构建、聚合和测试集评估。四并发计时包含采样顺序规划和 CPU 图像处理，不能把这个加速比直接当作完整实验的端到端加速比。

读取 `parallel_benchmark/report.json`：

- `passed`：逐客户端参数及聚合参数是否在声明的误差范围内一致。
- `speedup`：相同工作量的串行耗时除以四并发耗时；大于 1 才表示加速。
- `timings`：各次耗时、优化步数、PyTorch 峰值显存、客户端与 stream 槽位记录。
- `evaluations`：添加 `--benchmark-evaluate` 时，记录两种执行方式得到的模型在完整测试集上的精度，仅作为执行核对。默认不评估测速模型，避免将训练吞吐测试与额外测试集前向混在一起。

默认测速只训练 4 个客户端，不能把其测试精度放进论文大表。需要用全体 30 个客户端测速时，加 `--benchmark-clients 30`。

快速检查可以加 `--benchmark-batches 1 --benchmark-repeats 1`，只检查一个 batch 的执行与一致性，不能用它代替完整本地训练的可靠速度测量。不同测速设置需使用新输出目录。

## 完整 Client-LT 实验

确认服务器上的一致性和速度后，在同一输出根目录启动正式训练：

```bash
CUDA_VISIBLE_DEVICES=0 python -u scripts/run_capt_single_gpu_parallel.py --stage train --seed 42 --parallel-clients 4 --reference-run references/full10_clientlt --data-root DATA --num-workers 0 --output-root output/cifar100_LT/capt_single_gpu_parallel_seed42_v1
```

同一张 GPU 上保持四个训练槽位，空闲槽位领取下一个客户端；收齐当轮全部 30 个客户端后才聚合。聚合保持原日程顺序，不能四个客户端一组提前更新全局模型。

恢复时在上述命令末尾添加 `--resume`。也可以用 `--stop-after 1` 检查完整第一轮，然后用同一命令移除 `--stop-after` 并添加 `--resume` 继续。恢复边界是完整已提交轮次，半轮不会混入下一次聚合。

```bash
python -u scripts/run_capt_single_gpu_parallel.py --stage status --output-root output/cifar100_LT/capt_single_gpu_parallel_seed42_v1
python -u scripts/run_capt_single_gpu_parallel.py --stage pack --output-root output/cifar100_LT/capt_single_gpu_parallel_seed42_v1
```

正式训练结果沿用大表汇总：第 81–100 轮平均 Overall / Head20 / Middle60 / Tail20，以及末轮结果。

## 执行边界

- 一个 Python 进程、四个线程和四条 CUDA stream，始终只使用可见设备 `cuda:0`，无需 MPS。
- 冻结的 CLIP 参数只读共享；可训练参数、梯度、优化器和模型对象分别独立。
- 先规划与旧串行入口一致的 DataLoader/RandomSampler 索引，再并行读取确定性的 CIFAR 图像变换。不会在线程中调用全局随机种子重置。
- 入口默认设置 `CUBLAS_WORKSPACE_CONFIG=:4096:8` 并启用 `torch.use_deterministic_algorithms(True)`；串行参照也使用相同设置。未经这些设置，真实 CAPT 的细小数值差异会在多步优化中积累。本测试核对相同确定性设置下的执行一致性，不承诺与此前未启用确定性设置的历史训练逐位相同。
- 本实现仅支持当前确定性 CIFAR100 变换、无随机层的 CAPT；不自动扩展到随机增强或其他算法。
- `--num-workers` 控制评估 DataLoader；客户端图像处理由四个线程执行，不能把该选项当作并发客户端数。
- 显存统计为同一进程的 PyTorch 统计；测速的串行与并发阶段都已经创建四个常驻副本，不能据此声称单客户端原入口的显存占用。
- 代码采用独立模块和输出目录，未修改现有 CAPT、A/B 或大表训练循环。

## 检查

```bash
python -m unittest tests.test_capt_single_gpu_parallel -v
```

检查覆盖原 DataLoader 的跨 epoch 样本顺序、随机状态隔离、相同批次重放的实际优化结果，以及四条真实 CUDA stream 的客户端独立性和冻结参数不变性。真实 CAPT 的本地 GPU 测试结果另见对应 `parallel_benchmark/report.json`，本机速度不能直接代表服务器。

cuBLAS 的多 stream 复现设置依据 [NVIDIA cuBLAS 文档](https://docs.nvidia.com/cuda/cublas/index.html#results-reproducibility)。

## 2026-10-09 本地实测

硬件为 Windows / RTX 4070 Ti SUPER 16GB，PyTorch 2.10.0+cu126。采用真实 CIFAR100 Client-LT 划分，第一轮客户端 `[4, 21, 9, 0]`，每客户端 3 个本地 epoch，保持 batch size 32。测试使用 `--benchmark-repeats 1`；两种执行方式各完成 141 次更新。

| 项目 | 串行 | 单卡四并发 |
|---|---:|---:|
| 本地训练耗时 | 50.818 秒 | 503.857 秒 |
| PyTorch 峰值分配 | 6.660 GiB | 23.782 GiB |
| 与串行的逐客户端参数差异 | — | 全部为 0 |
| 与串行的聚合参数差异 | — | 0 |

一致性通过，实测加速比为 0.101，即本机四并发约慢 9.9 倍。四并发的分配量超过本机物理显存，存在显存超额分配这一明确限制；没有通过 profiler 单独量化各项耗时，不能把全部耗时差异都归给某一个原因。服务器应在空闲 GPU 上重新测量，默认重复两次并倒转测量顺序。

报告位于 `output/capt_parallel_deterministic_local_check_20261009/parallel_benchmark/report.json`。新增的 4 项测试全部通过，覆盖采样顺序、重放更新、真实 CUDA 四并发及槽位复用、轮次恢复；原基准相关的 21 项测试也通过。本次没有运行完整 100 轮联邦训练，没有产生新的论文精度结果。

排查时未启用确定性计算的完整本地训练曾未通过严格参数一致性检查，失败报告保留于 `output/capt_parallel_full_local_check_20261009/parallel_benchmark/report.json`。当前入口已加入确定性计算设置；修改前的单 batch 检查不能替代本节的完整本地训练验证。
