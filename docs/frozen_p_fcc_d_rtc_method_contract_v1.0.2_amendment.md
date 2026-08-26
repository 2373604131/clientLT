# P-FCC + D-RTC 方法运行契约 v1.0.2 修正案

本修正案只处理两轮真实集成检查暴露出的两个协议问题；其余
v1.0.1 条款保持不变。

## 1. 跨条件公平的 proposal clustering seed

服务器仍必须为每个实验条件维护独立、仅由该条件上一轮实际上传
构成的 proposal bank。聚类确定性种子修正为：

```text
stable_seed("proposal-cluster", global_seed, source_round)
```

禁止将 `condition` 加入聚类随机种子。因而相同轮次、相同上传集合
必须在 P-FCC、Combined 与 Random-Proposal 中产生相同的 cluster
membership 和 prototype；后续因各条件上传本身不同而自然产生的 bank
差异仍然允许。

## 2. Post-local P-FCC compatibility gate

Incoming-global proposal utility 仍用于选择最多两个正 utility proposal，
但不得直接假设该方向在本地 CE 更新后仍然兼容。完成普通本地训练并
获得 `delta_CE` 后，客户端构建固定 FCC multiplier：

```text
(0.0, 0.25, 0.5, 1.0)
```

每个候选均经过原有最终等范数组合，因此其上传范数仍严格匹配
`||delta_CE||_2`。客户端从同一个 post-local CE 状态出发，在私有
functional memory `E_k` 上独立评价四个实际上传候选，并选择 CE 最低的
multiplier；CE 相等时选择更小 multiplier。`0.0` 是强制安全回退，保证
P-FCC 相对无 FCC 候选不会在 `E_k` 上变差。

Combined 中四个候选保持同一个 D-RTC 分量，仅改变 FCC multiplier，
所以该 Gate 只裁决 FCC 的 post-local 兼容性，不替代 D-RTC 自身机制
验证。

Random-Proposal 必须执行完全相同的四个 post-local forward 以匹配计算
量，但不得使用其 CE 选择 multiplier，固定保留 `1.0`。独立 `A_k` 只做
离线审计，不参与 multiplier 选择。

## 3. 必须新增的审计

- 每个 multiplier 在 `E_k` 和独立 `A_k` 上的 CE；
- 实际选择的 multiplier；
- zero-FCC 到所选候选的 memory/audit gain；
- multiplier 为零的回退率，尤其是 tail-carrier clients；
- 原有最终上传范数 Gate 继续要求相对误差小于 `1e-6`。

本修正不引入服务器公共数据、客户端类别上行、额外同步轮次、客户端
统一重加权或更大的上传范数。
