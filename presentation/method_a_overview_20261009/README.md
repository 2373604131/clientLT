# 方法 A 论文概览图

- 中文预览：figures/tikz_method_a_overview_zh.png
- 英文预览：figures/tikz_method_a_overview_en.png
- 论文插图：同名 PDF（矢量、嵌入字体）
- 矢量图形：同名 SVG（字体轮廓化；文字编辑请改 TikZ）
- 可编辑源码：同名 TEX；build_figure.py 控制两种语言的统一布局
- 图注与符号解释：captions.md
- 构图记录：PAPER_PLAN.md
- 成图审查：figure_review.json

## 重新生成

在本目录运行：

~~~powershell
python render_figure.py
python audit_geometry.py
~~~

运行环境由 .vivid/runtime.json 记录。需要 XeLaTeX、Microsoft YaHei、Arial、
pdftoppm 和 pdftocairo。未安装新的软件依赖。

## 使用范围

本图对应当前冻结 full-cp 实现，不改变算法或运行训练。图中不是新网络结构，
也不表示方法已证明优于简单加权。类别反馈使用本地训练样本；没有虚构实验曲线、
准确率或必然改善。

推荐通栏宽度 177.8 mm；源 PDF 宽约 181.45 mm，正文标签缩放后约 8.1 pt。
不要作为单栏图大幅缩小。最终论文页面还未排版，页内可读性需要在集成时确认。

## 绘图与审查

先使用 figure-designer 明确三栏构图与表达目的，再使用 Vivid 的
paper-technical-diagram / TikZ 路径渲染，最后打开中文和英文实际成图进行审查。
原通用 tikz_check.sh 的颜色计数将同色不同明度误当作不同颜色，并以通用字符宽度、
默认 0.75 cm 文本高度估计本图 8.3 pt 标签，因此产生大量误报。未改动该检查器；
保留其报告，并用 TeX 原生节点边界测量和实际成图审查核实具体问题。最终测量结果
见 geometry_review.json，配色用户覆盖说明及误报解释见 figure_review.json。
