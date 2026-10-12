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

## 不知道文件在哪时：跨目录搜索

将 `scripts/check_a_projection_artifacts.py` 复制到每台待检查服务器。它是独立脚本，只需要 Python 3 标准库，不依赖仓库、PyTorch 或 GPU。在仓库根目录执行：

```bash
python scripts/check_a_projection_artifacts.py --roots /data /home /mnt /media /scratch /work /workspace --archives
```

报告保存为 `output/a_projection_files_<主机名>.txt` 和同名 `.json`。提交这两个报告即可定位缺项，不必传输模型。存储挂载在其他位置时，将该路径加入 `--roots`。不存在的目录会跳过，权限错误写入报告；目录符号链接会跟随并去重，跳过 `.git`、`.cache`、Python/Node 环境等目录。未找到只说明本次可读搜索范围内没有匹配项，不能据此确认文件已经删除。

脚本检查 FedAvg/TailRW16 的元数据、冻结基座、20/50/80/90/100 轮模型、20/50/80 轮 pre-A 起点；兼容显式 `projection_states.json` 映射，并保留孤立模型和旧 `bridge_dumps` 目录作为线索。`--archives` 会扫描 ZIP/TAR 成员，不解压；压缩 TAR 必须流式读取，耗时取决于文件大小，可先省略此参数只查目录。

| 状态 | 含义 |
|---|---|
| `FILES_PRESENT` | 配置初筛符合且所需文件非空；仍需诊断入口核验分区、哈希、张量与提案 |
| `METADATA_ONLY` | 这个运行目录或归档中未找到可识别模型，不能直接进行投影诊断 |
| `ARCHIVE_FILES_PRESENT` | 所需文件位于压缩包；仍需恢复到目录并验证 |
| `PARTIAL_OR_OTHER_PROTOCOL` | 文件不全、配置不符或尚无法确认；详见缺项与配置字段 |

脚本只读目录、小 JSON 和可选归档头，不调用 `torch.load`。A 事件文件存在不等于客户端提案完整，提案内容留给正式入口验证。

## 单卡运行

找到完整输入后，在服务器仓库根目录、原 `clientLT` 环境执行：

```bash
python scripts/run_a_projection_diagnosis.py --stage server --device cuda:0 --output-root output/a_projection_diagnosis_v1_single
```

不传 `--gpus`，单个进程在一张 GPU 上逐项执行。`server` 先做文件检查和小样本真实模型 smoke，再依次执行 FedAvg E1、TailRW16 E1、FedAvg 第20/50/80轮的 E2/E3/行空间干预，最后汇总。每项结束后释放模型和 GPU 缓存；全部实验、采样、种子与数值口径保持原协议。`cuda:0` 指当前可见的第一张 GPU，兼容服务器已有的 `CUDA_VISIBLE_DEVICES` 分配。

未显式指定源路径时，入口搜索当前仓库的已知普通 A 目录、轻量归档中的原始路径及历史仓库位置。搜索经过协议检查，多个完整候选时要求明确路径，不按精度或时间自动挑选。搜索过程写入 `source_discovery.json`。如果查到其他位置，用 `--fedavg-run /实际路径/fedavg/runs/seed42/plain_a` 和 `--tailrw16-run /实际路径/tailrw16/runs/seed42/plain_a` 精确指定；`--source-root` 指定两者的共同实验根目录，`--search-root` 增加待查根目录。显式路径不会被自动替换。

若缺少模型，入口停止并列出缺项，不自动重训100轮。默认2个 CPU 数学线程、DataLoader workers=0，可用 `--cpu-threads`、`--num-workers` 调整。真实模型依赖沿用训练环境及已有 CLIP ViT-B/16 缓存。也可以拆开检查、运行和打包：

```bash
python scripts/run_a_projection_diagnosis.py --stage inventory --output-root output/a_projection_diagnosis_v1_single
python scripts/run_a_projection_diagnosis.py --stage preflight --output-root output/a_projection_diagnosis_v1_single
python scripts/run_a_projection_diagnosis.py --stage smoke --device cuda:0 --output-root output/a_projection_diagnosis_v1_single
python scripts/run_a_projection_diagnosis.py --stage run --device cuda:0 --output-root output/a_projection_diagnosis_v1_single
python scripts/run_a_projection_diagnosis.py --stage pack --output-root output/a_projection_diagnosis_v1_single
```

中断后重复同一命令：完整任务跳过，未完成任务复用前向缓存及已生成提案。各任务有独立缓存、costs、environment、smoke 和结果目录，各自持有任务锁；重复提交同一任务会报锁冲突。更改协议、代码、输入源、batch size 或 CPU 线程数时用新输出目录。运行期间控制台显示每项任务的 START/DONE。

可选的 `--gpus` 队列接口仍保留，但上述单卡命令不会启动它。

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

## 重新训练一条普通 A＋FedAvg（单卡、固定 Tail20）

先把新的代码包上传到服务器仓库根目录并解压。出现 `python: can't open file ...scripts/xxx.py` 表示脚本没有同步到该服务器，不能据此判断模型不存在。

```bash
python -m zipfile -e a_projection_recollect_single_20261012.zip .
python scripts/run_a_projection_recollect.py --stage server --gpu 0
```

新入口只补采一条 FedAvg。默认一张 GPU、客户端顺序执行、DataLoader workers=0，复用普通 A 训练实现：seed42、原10847张训练样本及30客户端划分、100轮、每轮 B3、前90轮额外 A1、最后10轮固定 A。继承原 LA、优化器、rank4 和数值设置。重新采集不声称逐位复现以前的并发执行轨迹。

类别标签不重新排序：Head=`0..19`，Middle=`20..79`，**Tail=`80..99`**。启动前核对原始样本/客户端/顺序/标签清单的语义哈希和尾类清单；运行时在本地 SGD 前再次检查分组；结果验证也核对同一分组。分组或划分改变会明确报错，不会自动选择另外20个类别。

`--stage server` 自动执行：预检 → 独立1轮 B+A smoke → 100轮正式训练 → 输入留存验证 → FedAvg 的 E1/E2/E3/行空间诊断 → 诊断结果打包。默认目录：

```text
output/cifar100_LT/a_projection_recollect_seed42_tail20/
  recollection_plan.json
  fedavg/runs/seed42/plain_a/checkpoints/base_model.pt
  fedavg/runs/seed42/plain_a/checkpoints/sfra_last.pt
  fedavg/runs/seed42/plain_a/events/.../state.pt
  fedavg/launcher_logs/plain_a_formal.log
  retained_inputs_verified.json
  diagnosis/
  diagnosis_results.zip
```

保留整个训练目录，尤其是 `checkpoints/` 和 `events/`。原训练代码会保存100个 B 事件及90个 A 事件；新入口在训练后检查190个事件文件，并实际验证冻结基座/初始化哈希、诊断模型、20/50/80轮各30份原始 A 提案与聚合重建、90/100轮 A 不变。只在这些检查通过后启动诊断。`diagnosis_results.zip` 是便于发回的分析包，不能替代服务器上的模型文件。

查看进度及日志：

```bash
python scripts/run_a_projection_recollect.py --stage status
tail -f output/cifar100_LT/a_projection_recollect_seed42_tail20/fedavg/launcher_logs/plain_a_formal.log
```

中断后重复原命令，正式训练从已提交轮次恢复，完整诊断任务跳过。若先只训练，添加 `--train-only`；之后运行 `--stage diagnose --gpu 0`。若数据根目录不同，使用 `--data-root`，训练与诊断保持同一参数。`--stage plan` 只验证并登记协议，`--stage preflight` 额外检查数据，`--stage audit` 验证已完成训练的输入留存；这些阶段不启动训练。

这条 FedAvg 轨迹能完成三个起点的聚合反事实，包含即时 TailRW16/joint 候选；TailRW16 的独立长期轨迹尚未补采，汇总会保留其缺失状态。没有新增方法 A 损失或 matched Dirichlet。

## 输出与判读

默认 `output/a_projection_diagnosis_v1/`：

| 位置 | 内容 |
|---|---|
| `artifact_inventory_*.json/csv` | 模型缺项、原服务器目录、恢复方案与跨轨迹元数据匹配 |
| `source_discovery*.json` | 搜索过的位置、候选目录、协议拒绝原因与最终输入来源 |
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
python -m unittest tests.test_a_projection_artifact_scan -v
python -m unittest tests.test_a_projection_discovery -v
python -m unittest tests.test_a_projection_recollect tests.test_plain_a_aggregation -v
```

测试包含可逆换基、空间内不变性、降秩、空间外 Gram 恒等式、零候选排除及只缩小范数匹配、采样、历史状态恢复和篡改检测，以及使用六个真实 pre-hook 的小模型完整 E1/E2/E3/干预流水线。小模型数据只验证程序正确性，不是 CIFAR-100 实验证据。

文件搜索测试覆盖完整/轻量 ZIP 和 TAR、显式历史模型映射、孤立模型、空文件和读取失败；单卡测试验证五项任务依次执行、上一模型释放后才创建下一模型，以及默认入口不会启动 GPU 队列。可选并行接口也保留 CPU 回归校验。尚未进行真实 GPU 测量。
