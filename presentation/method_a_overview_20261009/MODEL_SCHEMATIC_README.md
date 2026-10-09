# 方法 A：模型示意图

本版本针对“上一版全是文字、像流程图”的反馈重新构图。主要元素是图片输入、CLIP 编码器、LoRA A/B、冻结与训练标志、图像/文本特征、预测、服务器及客户端。详细算法说明在独立图注中。

中文预览：figures/tikz_method_a_model_zh.png

英文预览：figures/tikz_method_a_model_en.png

同名 PDF 用于论文排版，SVG 用于矢量图形编辑，TEX 是完整可编辑源文件。SVG 中字体已经转为轮廓，修改文字请编辑 TEX 或生成脚本。PNG 为 240 dpi 预览；照片是 CIFAR-100 原始训练数组导出的输入示例，其余图形与文字为矢量内容。

运行 render_model_schematic.py 可重新生成全部中英文文件；运行 audit_model_schematic.py 检查实际 TeX 文字边界。运行环境沿用 .vivid/runtime.json。修改 build_model_schematic.py 中的布局和短标签即可重新生成。

MODEL_SCHEMATIC_PLAN.md 记录 figure-designer 绘制前构图，model_figure_review.json 记录实际成图审查，model_captions.md 提供中英文图注和简化说明。旧 overview 文件保留，当前主图是 model 文件。

这张图只展示当前方法 A 的 A 阶段，没有引入新的教师网络、可训练分类头、个性化模型或额外模块。B 在本图中冻结，是因为 B 阶段已经完成。
