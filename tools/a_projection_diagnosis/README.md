# A 投影学习诊断

按老师补充意见实施 E1、E2、E3 与行空间干预。完整协议见 [实验设计](../../docs/A投影学习机制诊断实验设计_20261012.md)。代码不修改训练损失或历史轨迹，不用测试数据选择候选、轮次或正则；目前尚未取得正式 GPU 测量结果。

## 服务器应该保存了什么

历史元数据记录仓库目录：

```text
/data/run01/scze847/yzh/code/clientLT
```

原始训练目录：

```text
output/cifar100_LT/plain_a_aggregation_v1_parallel8/
  fedavg/runs/seed42/plain_a/
  tailrw16/runs/seed42/plain_a/
```

现有代码每轮保存 B 事件，1–90 轮另保存 A 事件，共 190 个事件。`utils/cliplora_bridge_audit.py::BridgeAudit.save` 将完整 A/B 起点、终点、客户端 A 增量、ID、计数和权重写入 `events/.../state.pt`；`utils/cliplora_la_control.py` 保存 `checkpoints/base_model.pt`。例如：

```text
fedavg/runs/seed42/plain_a/checkpoints/base_model.pt
fedavg/runs/seed42/plain_a/events/r020_c000_main_refresh_A/state.pt
fedavg/runs/seed42/plain_a/events/r050_c000_main_refresh_A/state.pt
fedavg/runs/seed42/plain_a/events/r080_c000_main_refresh_A/state.pt
fedavg/runs/seed42/plain_a/events/r090_c000_main_refresh_A/state.pt
fedavg/runs/seed42/plain_a/events/r100_c000_main_normal_B/state.pt
```

0 轮由基座恢复；20/50/80/90 轮由 A 事件终点恢复；100 轮由 B 事件终点恢复。普通 A 只改 A，冻结主干及 B 的条件会在恢复时验证。历史 A 增量回加有 FP32 舍入限制，不能称逐位恢复客户端参数。

`tools/client_aggregation/analysis.py::pack_results` 的结果打包白名单不包含 `.pt`，并排除 `checkpoints`，所以本地分析包看不到权重。代码中这条保存路径不主动删除逐轮事件；服务器文件是否被人工清理，仍需 inventory 实查。

## 运行

在服务器仓库根目录、原 `clientlt` 环境执行。默认源目录已经指向上面的历史位置。

```bash
python scripts/run_a_projection_diagnosis.py --stage inventory
python scripts/run_a_projection_diagnosis.py --stage server --gpus 0 1 2 3 --output-root output/a_projection_diagnosis_v1_gpu4
```

`server` 完成文件检查，再将五项独立任务分配到最多四张 GPU；每项先做真实模型四图 smoke，再执行正式测量，全部结束后统一汇总。若缺少模型就停止并列出缺项，不自动重训 100 轮。真实模型依赖沿用训练环境及已有 CLIP ViT-B/16 缓存。

| 独立任务 | 内容 |
|---|---|
| `fedavg_round020` | 第20轮的 E2、E3、行空间干预及剩余样本复核 |
| `fedavg_round050` | 第50轮的相同诊断 |
| `fedavg_round080` | 第80轮的相同诊断 |
| `fedavg_e1` | FedAvg 的全部六个跨轮检查点 |
| `tailrw16_e1` | TailRW16 的全部六个跨轮检查点 |

队列先启动三个同轮任务及一个 E1 任务，先完成的 GPU 接着执行剩余 E1。同一事件的所有候选共享同一起点、固定输入、提案和评价样本；没有改变实验条件、训练种子或数值口径。四卡加速比仍需实测，不能把并行度直接当作4倍提速。每个进程独立加载一个模型，主机需要容纳最多四份模型与数据。

`--gpus` 接受1到4个**当前可见 GPU 编号**。例如环境为 `CUDA_VISIBLE_DEVICES=2,4,6,7` 时，`--gpus 0 1 2 3` 使用这四张已分配的卡；不会突破已有可见设备范围。每个子进程只看见自己的 GPU，并使用 `cuda:0`。指定 `--gpus` 时不要另选 `--device`。

只有两张卡时使用 `--gpus 0 1`；单卡可使用 `--device cuda:0`。默认每进程2个 CPU 数学线程、DataLoader workers=0，避免四个进程争用全部 CPU；可用 `--cpu-threads` 和 `--num-workers` 显式调整。多进程版本建议使用上面的新输出目录，保留旧版本已有输出。

可拆开检查与正式运行：

```bash
python scripts/run_a_projection_diagnosis.py --stage preflight --gpus 0 1 2 3 --output-root output/a_projection_diagnosis_v1_gpu4
python scripts/run_a_projection_diagnosis.py --stage smoke --gpus 0 1 2 3 --output-root output/a_projection_diagnosis_v1_gpu4
python scripts/run_a_projection_diagnosis.py --stage run --gpus 0 1 2 3 --output-root output/a_projection_diagnosis_v1_gpu4
python scripts/run_a_projection_diagnosis.py --stage pack --output-root output/a_projection_diagnosis_v1_gpu4
```

也可以手动拆任务；例如在不同终端同时执行同一轨迹的两个事件：

```bash
python scripts/run_a_projection_diagnosis.py --stage server --trajectories fedavg --part events --rounds 20 --device cuda:0 --output-root output/a_projection_diagnosis_v1_gpu4
python scripts/run_a_projection_diagnosis.py --stage server --trajectories fedavg --part events --rounds 50 --device cuda:1 --output-root output/a_projection_diagnosis_v1_gpu4
```

上述两条只完成所选事件。手动执行其他任务后统一 `pack`，汇总检查两轨迹及三个事件是否完整。自动队列更方便。

中断后重复同一命令：完整任务跳过，未完成任务复用前向缓存及已生成提案。各任务有独立缓存、costs、environment、smoke 和结果目录，各自持有任务锁；轨迹公共身份只在登记时短暂加锁。重复提交同一任务会报锁冲突。更改协议、代码、输入源、batch size 或 CPU 线程数时用新输出目录。

`parallel_plan.json` 记录任务与设备映射；`parallel_status.json` 记录状态、耗时、日志路径和失败退出码。日志在 `launcher_logs/<task>.log`。单任务失败不会丢弃其他任务；解决原因后重复命令即可。Ctrl+C 会停止本启动器创建的子进程。

## 缺少原提案时

只缺 `local_factor_deltas` 时自动从历史共同起点重新生成一次全部客户端 A 提案：每客户端新建 SGD，lr=.001、momentum=.9、wd=0、batch32、1 epoch、全局频数 LA。随机种子沿用原公式，但结果明确标为 regenerated paired proposals，不声称精确重放历史。

起点优先读取 A 事件 anchor，其次读取同轮 B 事件 endpoint。只有提交模型时，读取 19/20、49/50、79/80，拼成 `(A_previous, B_current)`。若这些模型以其他名字保存，在**原运行目录**添加明确映射，例如：

```json
{
  "committed": {
    "19": {"path": "checkpoints/r019.pt"},
    "20": {"path": "checkpoints/r020.pt"},
    "49": {"path": "checkpoints/r049.pt"},
    "50": {"path": "checkpoints/r050.pt"},
    "79": {"path": "checkpoints/r079.pt"},
    "80": {"path": "checkpoints/r080.pt"},
    "90": {"path": "checkpoints/r090.pt"},
    "100": {"path": "checkpoints/r100.pt"}
  }
}
```

文件名为 `projection_states.json`。支持完整 state_dict、`model_state_dict`/`state_dict`、仓库 `state_dict_overrides`，以及显式 `key` 指向事件中的状态字典。完整冻结基座仍必须保留。不要把最后一次 `sfra_last.pt` 当成任意历史轮次。

若服务器模型也已清理，沿用原普通 A 启动器补采 FedAvg，写新目录：

```bash
python scripts/run_cliplora_plain_a_aggregation.py --stage server --methods fedavg --gpus 0 --client-concurrency 8 --num-workers 8 --output-root output/cifar100_LT/a_projection_recollect
python scripts/run_a_projection_diagnosis.py --stage server --source-root output/cifar100_LT/a_projection_recollect --trajectories fedavg --output-root output/a_projection_recollect_diagnosis
```

单轨迹结果不能支持 TailRW16 的长期改善结论。没有新增静态聚合、方法 A 损失或 matched Dirichlet。

## 输出与判读

默认 `output/a_projection_diagnosis_v1/`：

| 位置 | 内容 |
|---|---|
| `artifact_inventory_*.json/csv` | 模型缺项、原服务器目录、恢复方案与跨轨迹元数据匹配 |
| `<trajectory>/probe_fit_manifest.csv` | 训练池每类至多32张，当前 seed42 共2472张 |
| `<trajectory>/<task>/probe_fit_manifest.csv`、`audit_eval_manifest.csv` | 每任务固定拟合样本和每类20/80评价清单；跨任务采用相同规则 |
| `<trajectory>/e1/representation_curves.csv` | 固定输入、固定 A、真实输入三组合；六模块8维与辅助48维 |
| `<trajectory>/e1/full_model_curves.csv` | 正常完整模型前向的准确率、CE、余弦间隔 |
| `fedavg/round*/local_class_response.csv` | 30×100 类别响应、类别持有情况与各权重 |
| `fedavg/round*/response_decomposition.csv` | 持有/未持有者正负加权作用及非线性差值 |
| `fedavg/round*/a_weight_counterfactuals.csv` | 同起点、同提案的三种 A 聚合，原幅度与缩小等幅 |
| `fedavg/round*/rowspace_interventions.csv` | 完整/空间内/空间外更新及 lost/gained，面板与剩余样本都报告 |
| `fedavg/round*/projection_probes.csv` | 上述全部候选使用共同起点输入的探针结果 |
| `fedavg/round*/norm_matching.csv` | 六模块统一系数、原有效范数、零候选和 low_power 标记 |
| `*/projection_geometry.csv` | 数值秩、奇异值、A 范数、行空间距离、有效映射范数 |
| `*/numerical_checks.json` | block9、90/100、不变性、分解或重复前向校验 |
| `fedavg/round*/provenance.json` | 历史提案或重新生成、起点哈希与重建误差 |
| `analysis/` | 主变化、配对区间、描述性分支、完整性及图表 |

客户端完整模型仅在2000张面板评估；全部预定非本地候选都在剩余8000张复核。后8000张是“剩余样本复核”，并非全新独立确认集。模型预测与逐模块探针预测保存到 npz，便于复核；测试标签不参与拟合、正则选择或权重求解。

探针：训练类等权使用采样后的 m_c；训练统计中心化 + 单一 RMS；FP64 SVD/ridge，固定 lambda=.01；没有逐通道 z-score、正则扫描或池化择优。中间投影沿模型 FP32 执行，基底/求解使用 FP64，运行时同秩空间内候选分数误差阈值2e-4，90/100相同固定输入阈值1e-10。probe 本身分辨能力不足时应报告尚未定位。

固定输入下降但真实输入未下降时，不能称实际信息退化。空间外干预也改变 Gram/尺度；等总有效范数仍未匹配激活或各层敏感度。空间内 probe 不变而模型受损，可以定位利用变化，不能直接判定 B 容量不足。所有干预都需同时看 lost 与 gained，保留有益新方向的可能。

汇总的1000次类别内配对 bootstrap 只覆盖给定模型与已拟合 probe 的测试样本不确定性，不能替代独立训练种子。统计不会自动确定下一版方法 A。

## 本地校验

```bash
python -m unittest tests.test_a_projection_diagnosis -v
python -m unittest tests.test_a_projection_parallel -v
```

测试包含可逆换基、空间内不变性、降秩、空间外 Gram 恒等式、零候选排除及只缩小范数匹配、采样、历史状态恢复和篡改检测，以及使用六个真实 pre-hook 的小模型完整 E1/E2/E3/干预流水线。小模型数据只验证程序正确性，不是 CIFAR-100 实验证据。

并行测试在 CPU 上验证4个并发任务、GPU可见设备映射、真实子进程启动与失败状态、独立任务目录和同轨迹三事件同时执行。尚未在真实4卡服务器测量吞吐或速度提升。
