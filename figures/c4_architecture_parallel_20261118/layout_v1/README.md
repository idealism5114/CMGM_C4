# C4 主模型：并联结构与宏观／微观时间路径

2026-11-18 根据当前独立项目源码绘制。主模型为 `moe_phase_m/pred_global_dynamic`；训练默认值现为 lr=3e-4、weight_decay=1e-4，这两个优化器超参数不改变图中的网络结构。

| 用途 | 中文版 | 英文论文版 |
|---|---|---|
| 可编辑矢量图 | [SVG](c4_parallel_zh.svg) | [SVG](c4_parallel_en.svg) |
| 矢量排版 | [PDF](c4_parallel_zh.pdf) | [PDF](c4_parallel_en.pdf) |
| 高清插图，5940×2700 | [PNG](c4_parallel_zh.png) | [PNG](c4_parallel_en.png) |
| 屏幕预览 | [PNG](c4_parallel_zh_preview.png) | [PNG](c4_parallel_en_preview.png) |

采用原创矢量图标、浅色模块分组、平行编码路径、实线数据流和虚线路由／先验。布局参考正式发表论文中常见的模块分组与路径标识，未复制其模型结构或图像：

- Wu et al., *Connecting the Dots: Multivariate Time Series Forecasting with Graph Neural Networks*, KDD 2020, Figure 2. [作者论文PDF](https://www.xiaojun.ai/papers/kdd-2020-wu.pdf)
- Lim et al., *Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting*, International Journal of Forecasting, 2021, Figure 2. [期刊DOI](https://doi.org/10.1016/j.ijforecast.2021.03.012)、[作者预印本](https://arxiv.org/pdf/1912.09363)

## 图注

**中文：** C4的空间分支与时间分支并行编码同一历史市场窗口。空间分支通过历史加权、全局可学习图先验和两个EdgeAttnMixHop模块生成空间表示s。时间分支以因果Transformer生成宏观记忆H，并以H驱动三状态Regime过滤及潜在微观状态Z的递推；平衡读出和原时间投影生成t。四个专家分别读取Z、H、s以及s与t的低秩乘性交互，经过同一个共享预测头生成四期限候选收益率。样本条件化Router输出一组四专家权重，并在所有期限共享；最终在预测结果层进行凸加权。

**English:** Overview of C4. Parallel spatial and temporal encoders process the same historical market window. The temporal encoder combines causal macro memory with regime-dependent recurrent microstates. Four experts use recent latent change, long memory, spatial structure, and low-rank spatiotemporal interaction, respectively. A single shared forecast head produces horizon-specific candidate returns. A sample-conditioned global router supplies one expert-weight vector shared across all horizons; forecasts are combined at the prediction level.

## 关键语义

- 输入为 `[B,20,311,26]`，含263股票、25债券及23商品；输出为 `[B,4,23]`，期限行顺序为 `[1,5,10,20]`，表示冻结公共日历的交易索引步。
- 宏观／微观表示模型中的记忆与潜在动态层次，均来自同一个20期窗口，并非两种不同采样频率或完全独立的编码器。H参与Z的更新。
- 三个Regime专属生成器位于时间分支内部，区别于最终MoE的四个预测专家。近期专家的Z已包含递推历史，不能解释为只读取最近五期原始信息。
- 图先验A₀来自全局可学习节点参数，对不同样本共享；内容图注意力仍依赖输入。两个EdgeAttnMixHop模块各自保留K=2的内部传播。图中池化包括原节点归一化、三个市场的有效均值、拼接和原线性聚合。
- 平衡读出分别投影及归一化H_last/Z_last，拼接后经原state_readout与时间投影获得t。H和Z同时直接送入专家与Router，不通过detached日志缓存。
- Router读取 `concat(LN(s),LN(t),LN(ΔZ),LN(W_H mean(H)))`，MLP为256→32→4；唯一权重为 `[B,4]`，实现中广播到 `[B,4,4]`。没有期限输入或短／长任务掩码。
- 共享Head为Linear64→64、ReLU、Dropout0.3、Linear64→92。四专家共用参数，最终线性层按horizon-major输出行恢复 `[B,4horizon,4expert,23commodity]`；先预测，再加权。图中的P1—P4各自包含四期限预测。
- 没有原D0B的最终通道门控，没有加入后续动态图、GraphSkip或新的融合修正。原模型的有效性mask路径保留，图中省略了逐单元mask连线。
- 所有矩阵颜色、少量示意节点和Router柱高仅用于呈现结构，不是实际学习权重、attention或实验结果。箭头和专家名称不表示经济因果贡献或已经证明的预测分工。

## 复现与保护

在项目根目录执行 `/usr/bin/python3 tools/draw_c4_parallel.py`。系统librsvg/pycairo把原创SVG输出为PDF和PNG，不需要安装新包或调用图像生成服务。脚本仅生成本目录中的图稿，不读取训练样本、TEST、预测数组或checkpoint，不修改模型源码与旧版模型图。

源码与图稿SHA256见 [provenance.json](provenance.json)；旧图仍保留在 `../c4_architecture/`。
