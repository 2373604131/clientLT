# 方法 A 概览图：构图与事实约束

## 绘制前设计（figure-designer）

- 图型：solution-overview，三栏流程与跨轮历史反馈。
- 目的：展示当前 full-cp 如何保持普通聚合，并通过本地类别反馈设置权重和目标，随后修正全局模型。
- 主线：客户端训练与反馈 → 更新影响与目标计算 → 历史保持与全局修正。
- 不作为图中结论：集中度一定导致遗忘、性能必然提高、分类损失不会上升、无目标类别客户端已经产生有效迁移。
- 画布：18 cm × 9.3 cm；面向论文双栏通栏，正文标签约 8.6 pt，栏标题约 11 pt。
- 布局：左栏展示本地训练图片、冻结 CLIP / 固定 B / 训练 A、客户端更新和固定本地类别样本反馈；中栏展示样本量加权聚合、两视图一阶更新影响、当前目标与保持权重；右栏展示五轮正式模型历史、目标合并、三项修正目标及单一全局模型输出。
- 参考图只用于三栏大概览及颜色分区，不复制其梯度净化、个性化分支或模型模块。
- 配色：用户指定的亮蓝 #1479E0、橙 #F28E2B、绿 #159B75，黑色正文、白底、极浅同色面板。用户偏好覆盖 Vivid 默认低饱和配色。
- 工具：Vivid paper-technical-diagram 的 TikZ 路径，适合公式与精确箭头。不使用实验绘图模板或 AI 图片生成器编造架构。
- 中文讨论版与英文论文版为同一张图的两种语言。
- 图件交付即停止，不修改方法、论文正文或启动训练。

## 事实来源

- ../../docs/cliplora_method_a_full_cp.md（冻结版本 2026-09-21）
- ../../utils/cliplora_sfra.py
- ../../utils/sfra_math.py
- ../../utils/sfra_feedback.py

q=(k,c) 覆盖全部非空客户端—类别组合，而非预先选定尾类客户端。固定最多 8 张本地训练图片、两个确定性视图；r 为共同起点的一阶近似。正响应两视图均需超过阈值，u 为正响应未加权均值，N_eff 为有效正向客户端数。无当前正响应时不建立当前目标，权重使用缓存；没有当前或有效历史目标时不激活。

历史使用不重叠五轮正式模型及两视图最小值，超过上次参照 0.001 才登记，下一轮开始使用。初始模型只作为首次登记参照。目标两者存在时取最大值，仅有一项则使用该项。

修正作用在完整 LoRA A 参数空间，B 固定。L_ret 表示归一化预测间隔保持损失，L_cls 表示全局加权 LA 分类损失相对普通提案增加的单侧软惩罚。R 为客户端 A 更新范数的未加权 RMS。固定三步投影梯度下降，提交第三步。图示活跃 A 训练轮；第 91—100 轮当前配置仅更新 B。

<!-- BEGIN FIGURE_MANIFEST -->
## 图表清单（FIGURE_MANIFEST）

**数据图（matplotlib；paper-figure）：**
- none

**DrawIO 确定性技术图（paper-technical-diagram）：**
- none

**TikZ 精确图（paper-technical-diagram）：**
- tikz_method_a_overview_zh | claim=展示方法A的真实计算流程与历史反馈 | source=../../docs/cliplora_method_a_full_cp.md | section=方法概览 | language=zh | format=tex,pdf,png,svg | layout=single-landscape
- tikz_method_a_overview_en | claim=同一方法概览的英文版本 | source=../../docs/cliplora_method_a_full_cp.md | section=Method overview | language=en | format=tex,pdf,png,svg | layout=single-landscape

**AI 场景图（paper-illustration）：**
- none

**显式可选格式（HTML / Mermaid）：**
- none

**总数：** DATA=0, DRAWIO=0, TIKZ=2, ILLUSTRATION=0, OPTIONAL=0, ALL=2
<!-- END FIGURE_MANIFEST -->

## 成图检查（已完成）

已完整打开中文、英文实际成图并执行三轮局部修正，最后复查通过。两版各 78 个文本节点经 TeX 原生边界测量：文本重叠 0、超出画布 0、Overfull 0。矢量 PDF 字体已嵌入，不含栅格图片。示意图不包含实测性能曲线或改善数值。

原通用检查器对颜色明度及文本尺寸的误报保留在原始日志，具体判定与实际成图检查记录见 figure_review.json；没有改动检查器以消除提示。最终论文页面排版检查留待正文集成后进行。

交付对账：中文与英文均已生成 TEX、PDF、PNG、SVG，附重绘脚本、完整图注、符号解释和 LaTeX 引用片段。研究主张仍需简单加权与保持基线的实验支持。
