# 固定模型与实验协议

## C4 计算路径

`FrozenDataset → C4Model → PhaseMLiveBackbone → 原空间/时间分支 → ExpertBank → 共享预测头 → C4Router 预测级加权`。

空间分支按原始顺序执行异构投影、逐节点历史时间注意力、AdaptiveGraphLearner、两层 EdgeAttnMixHop（每层内部 K=2）、原 LayerNorm、三个市场均值池化和 type aggregation。原可用性 mask 限制图 source/destination；全无效支持返回有限的中性表示。

时间分支保留 Market Encoder、两层因果 Transformer、Regime 递推、三个 regime-specific generators、Microstate 和 Balanced Readout。用于预测的 H/Z 是同一次前向中的 live tensor；diagnostics 仅作 detach 日志。

投影 `s,t∈R64`，`H∈R20×128`，`Z∈R20×64`。

四专家顺序固定：

1. Recent：`ReLU(W[Z_last, mean(Z_last5), Z_last−mean(Z_last5)])`，192→64。递推 Z 包含更早历史，不能称为仅依赖五个原始观测。
2. Long Memory：`ReLU(W[H_last, mean(H_all20)])`，256→64。
3. Spatial：`s + ReLU(Ws)`。
4. Interaction：`Wo[(Us s)⊙(Ut t)]`，rank16，**不含旧专家的额外 t 残差**。

Router：`concat(LN(s), LN(t), LN(Z_last−mean5Z), LN(W_H mean20H)) → Linear256,32 → GELU → Linear32,4 → softmax`。最终层初始化 Normal(0,1e−3)，bias0。四期限共享每个样本的一组权重；内部广播为 `[B,4,4]`。

共享 head 为 `Linear64,64 → ReLU → Dropout0.3 → Linear64,4C`，公共 body 一次作用于 `[B,4experts,64]`。最终层 horizon-major 行恢复候选预测 `[B,4horizon,4expert,23]`，再做加权求和。没有每专家独立 head，没有期限独立 Router；它是预测结果级融合。

## 初始化和恢复

模型构造顺序与原 C4 一致，包括原 gate 构造时消耗的随机数，随后按 C4 定义删除 gate。用于正式匹配训练的 `build(initial=True)` 严格加载原实验保存的未训练 state，而不是仅依赖 seed。`initial_states/c4.pt` 只抽取原初始归档的 C4/C0，保留逐张量值。

训练后的 best/latest 严格恢复原 state_dict，保留原参数 key；并从 checkpoint epoch 恢复 Switch warmup 状态。没有将 B-dynamic、动态图、GraphSkip 或其他 MoE 的参数混入主模型。

## 训练协议

| 项目 | 固定值 |
|---|---|
| seed | 42 |
| Optimizer | Adam；初始lr3e−4，weight_decay1e−4（2026-11-18起的通用默认值） |
| batch | 64，shuffle=False，drop_last=False，num_workers0 |
| 预算 | 最大200 epochs，patience10 |
| Scheduler | ReduceLROnPlateau；factor0.5，patience5 |
| selection/scheduler/early stopping | 完整 VAL5 商品 Macro MAE |
| 梯度裁剪 / AMP | 无 / 无 |
| epoch0 | 保持原 Phase M 协议，不是正式候选 |
| TRAIN 覆盖 | 2726 origins，每 epoch43 steps，尾 batch38 |
| VAL 覆盖 | 483 origins，8 batches，尾 batch35 |

2026-11-18经用户确认，`configs/project.json`的通用训练默认值同步为调优配置。今后主模型、消融及baseline使用同一训练入口时，未显式覆盖的lr/weight_decay均读取上述值；其他训练设置不变。历史运行目录中的配置、源码快照和checkpoint保持原样，不代表已有baseline也使用新参数重新训练过。复现或恢复旧参数运行时，应显式传入其原始`--lr`和`--weight-decay`；恢复仍须满足原有源码、数据及协议一致性检查。

每个 horizon 在其有效标签单元内计算 Huberδ0.02 均值，然后四项求和，不额外除以4。C4 保留原 Switch KL，仅加入一次，warmup20 到 beta_max0.0005；相关消融按原定义关闭相应逻辑；baseline 不加入 KL。任一 horizon 无有效标签则失败，禁止静默当作0损失。

选择指标先对每商品有效 origins 计算5d MAE，再对23商品平均，不能用 pooled MAE 代替。描述性四周期指标包含 Macro MAE、pooled MSE/RMSE/Hit；Hit 保持 `sign(pred)==sign(target)`，零收益规则未改变。

训练缓存仅包含 TRAIN/VAL，日期、商品、窗口、标签和 mask 不重新切分或拟合。TEST 是独立冻结 release，必须单独显式授权。没有把历史 TEST 排名用于本次整理或模型选择。

## 消融定义

| key | 实际干预 |
|---|---|
| full_c4 | 与主 C4 同一个 best checkpoint |
| no_spatial | 删除空间路径；仅 Recent/Long 候选，屏蔽 Spatial/Interaction 路由 |
| no_temporal | 空间专家 + 原共享 head；时间相关专家/Router/KL 不参与 |
| no_spatial_temporal_attention | 每节点可用历史均匀平均，图仍保留 |
| no_adaptive_graph | A=ones，保留两层内容图注意力 |
| no_edge_attn_mixhop | 保留学习图，替换为两层普通 MixHop，K2、beta0.05 |
| no_rpe | 仅将相对位置 bias 置零 |
| no_adaptive_regime_routing | regime/prior均匀，仍有三个 generator，KL0 |
| no_regime_specific_transition | 使用 generator0 共享，各候选不再独立 |
| no_microstate | 仅最终 balanced readout 的归一化微状态置零；递推 Z 与 Recent 专家仍存在 |
| no_balanced_readout | 仅绕过 long/micro readout LayerNorm，保留投影 |
| no_dynamic_routing | 四专家固定等权1/4；不等于可学习静态 Router |
| no_moe | 原 D0B channel gate + 原 head，无四专家/Router；复用 C0 epoch46 |

原消融 checkpoint 保留了被绕过模块的 state key；原审计确认不活跃的整组参数在工厂中冻结并排除 optimizer。未擅自删除以破坏 strict restore。参数总数、可训练数和无预测梯度参数的区别应分别报告。

## 版本与可解释性边界

8 个 baseline 为 RNN、GRU、LSTM、VanillaTransformer、GraphWaveNet、MTGNN、MSGNet、CrossGNN，使用2026-11-09统一 VAL5 选模版本。原始实现及第三方来源记录保留在 provenance，不宣称本地适配等同于每篇原论文的全部设置。

当前项目保留原 C4，不重新将四专家划分为短期/长期专属专家。不同来源有结构偏置，但不能仅凭名称或路由差异认定已学出任务分工。单 seed、反复使用 VAL/TEST 的限制仍然成立。
