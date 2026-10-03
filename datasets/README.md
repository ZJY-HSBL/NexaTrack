# Dataset interface / 数据集接口

NexaTrack does not redistribute third-party datasets. Training consumes a JSONL manifest so different tracking datasets can be converted into one common interface.

NexaTrack 不分发第三方数据集。训练统一读取 JSONL manifest，因此可以将不同目标跟踪数据集转换到同一数据接口。

Each line must contain two template frames, one search frame, and their absolute `xyxy` target boxes:

每行必须包含两个模板帧、一个搜索帧，以及对应的绝对 `xyxy` 目标框：

```json
{"static_template":"a.jpg","dynamic_template":"b.jpg","search":"c.jpg","static_bbox":[x1,y1,x2,y2],"dynamic_bbox":[x1,y1,x2,y2],"search_bbox":[x1,y1,x2,y2]}
```

Recommended sampling strategy / 推荐采样策略：

1. Use the earliest valid frame as the static-template candidate. / 静态模板优先从序列前部有效帧采样。
2. Sample the dynamic template from a later frame near the search frame. / 动态模板从更靠近搜索帧的后续帧采样。
3. Keep all three samples from the same object trajectory. / 三个样本必须来自同一目标轨迹。
4. Exclude frames without a valid visible target for the default localization training setup. / 默认定位训练中排除目标完全不可见的帧。
5. Shuffle triplets across sequences during training. / 训练时在不同序列间随机打乱三元组。

The loader creates square crops from target area multipliers and transforms ground-truth boxes into normalized crop coordinates automatically.

数据加载器根据目标面积倍率构造正方形裁剪区域，并自动将真实框转换为裁剪区域内的归一化坐标。
