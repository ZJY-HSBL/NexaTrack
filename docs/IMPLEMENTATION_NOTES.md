# Implementation notes / 实现说明

## Directly specified components / 直接按方法描述实现的部分

- Shared hierarchical shifted-window visual backbone. / 共享分层移位窗口视觉骨干网络。
- Cross-scale fusion between intermediate and deeper features. / 中层与深层特征的跨尺度融合。
- 256-dimensional Transformer feature width in the full profile. / 完整配置采用 256 维 Transformer 特征。
- 4 attention heads, 6 encoder layers, 1 decoder layer. / 4 个注意力头、6 层编码器、1 层解码器。
- 64-dimensional reduced representation inside context-aware attention. / 上下文感知注意力内部采用 64 维降维表示。
- Gaussian target mask with sigma = 0.05 on template tokens. / 模板 token 使用 sigma = 0.05 的高斯目标掩码。
- Two template inputs plus one search input. / 两个模板输入与一个搜索区域输入。
- Corner probability maps followed by expectation-based coordinate decoding. / 角点概率图与基于期望的坐标解码。
- IoU-family + L1 box loss. / IoU 类损失与 L1 边界框损失。
- AdamW-style optimization, weight decay 1e-4, backbone/main learning rates 1e-5 and 1e-4, ×0.1 decay after epoch 400. / AdamW 类优化，权重衰减 1e-4，骨干/主体学习率 1e-5 与 1e-4，第 400 个 epoch 后乘 0.1。

## Engineering decisions / 工程化补全

Some low-level tensor operations are not uniquely specified by the source description. To make the project executable and shape-safe, this repository uses the following explicit engineering choices:

原始方法描述并未唯一确定所有底层张量操作。为保证项目可运行、可训练且张量尺寸稳定，本仓库明确采用以下工程化方案：

1. Correlation vectors are resized into a fixed low-dimensional descriptor before the nested attention operation, then resized back to the original correlation-map axis. / 相关向量先缩放为固定低维描述子执行嵌套注意力，再恢复到原相关图维度。
2. The two decoder branches attend to encoded memory and pre-encoded fused memory respectively. / 解码器两分支分别关注编码增强后的 memory 与编码前的融合 memory。
3. Cross-scale features are adaptively pooled to configurable token grids before the global encoder, limiting quadratic attention memory. / 跨尺度特征进入全局编码器前通过自适应池化映射到可配置 token 网格，以控制二次复杂度显存开销。
4. Template input is cropped at 180×180 and internally resampled to 192×192 for stable hierarchical window partitioning. / 模板按 180×180 裁剪，在骨干内部转换到 192×192 工作尺寸以保证分层窗口划分稳定。
5. Generalized IoU is used as the IoU-family training term; its weighting relative to L1 remains configurable. / IoU 类训练项具体采用 generalized IoU，与 L1 的权重保持可配置。
6. The reliability head is trained against detached overlap quality as a soft target. / 可靠性预测头使用停止梯度后的重叠质量作为软目标进行训练。

These decisions are isolated in modular files so they can be replaced when stronger implementation details or checkpoints become available.

上述工程化方案均被隔离在独立模块中，后续如果获得更完整的实现细节或权重，可以直接替换对应模块。
