# C4 Global Dynamic Prediction MoE 机制消融

共13行：Full C4，原9月17日10个骨干消融，四专家固定等权，取消MoE。
原w/o Adaptive Fusion删除D0B gate，但C4不存在gate，用户明确批准替换为w/o Dynamic Routing（四专家各1/4）。
取消MoE恢复原D0B gate，数学上就是Phase M C0，并非新增近似融合；Full复用C4，取消MoE复用C0。
这两组已用同seed、同数据、同初始化来源、同预算及VAL5 Macro MAE选模，因此不重复训练。
其余11组使用Phase M保存的**随机初始**state独立训练seed42，绝不从已训练C4热启动。

新代码仅新增 `models/phase_m_c4_ablation.py` 与 `scripts/phase_m_c4_ablation.py`，复用原MainInnovationAblation、
PhaseMLiveBackbone、专家/预测头和TRAIN/VAL loader；不修改任何旧模型或结果。

## 定义

- 去空间分支：不执行空间编码器，E3/E4禁用；近期E1/长期E2和剩余两候选动态路由保留。
- 去时间分支：不执行时间编码器，无Switch KL，仅E3空间专家和同一个共享head，路由退化为单一合法候选。
- 空间时间权重改均匀、图A改全1、EdgeAttnMixHop改普通MixHop×2、RPE置零：沿用封存Sep17定义。
- 去Adaptive Regime Routing：p/prior均1/3，三个独立generator保留，KL0。
- 去Regime-Specific Transitions：共享G0，Markov后验与native KL保留。
- 去Microstate：仅最终归一化后micro读出置零；递推和直接读取Z的近期专家保留。
  因C4专家绕过最终readout，这一消融不是移除所有潜状态信息，不把它解释为取消整个Z机制。
- 去Balanced Readout：仅绕过原long/micro LN。
- 固定等权：预测级四专家各1/4，不计算可训练router梯度；不是取消MoE，也不是可学习全局静态C1。
- 取消MoE：原D0B channel gate、原head，没有专家或router。

共有模块逐张量初值相同。结构上不使用的整参数张量保留在state_dict供审计，但requires_grad=False、排除optimizer，
分别报告注册总参数、有效可训练参数及冻结的非活动参数。部分路由坐标被mask，可能获得零梯度，明确记录。

## 协议与执行

新数据311节点、26特征、23商品；TRAIN2726/VAL483；batch64、shuffle=False、drop_last=False。
Adam1e-4/WD1e-5，max200、patience10、ReduceLROnPlateau factor.5/patience5。
四周期有效标签Huber(.02)均值之和，加一次native Switch KL；无时间分支或uniform regime时KL为0。
Checkpoint、scheduler、earlystop均使用**VAL5商品Macro MAE**，epoch0不是候选。
保留完整optimizer/scheduler/RNG/history/latest/best，中断只作完整状态恢复。

默认只预检：

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 ../.venv/bin/python -m cmgm.scripts.phase_m_c4_ablation --output experiments/phase_m_c4_ablation/<run>
```

预检通过后显式训练及冻结TEST：

```bash
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 ../.venv/bin/python -u -m cmgm.scripts.phase_m_c4_ablation --output experiments/phase_m_c4_ablation/<run> --run --device cuda:0
```

全部13行来源与best严格核验、models_frozen清单固定后，11组新模型只做一次TEST。
Full/noMoE的TEST预测通过原文件和checkpoint SHA256核验后复用。TEST使用已冻结release423 origins，
2025-01-02—2026-09-29，同23商品和mask，无scaler重拟合，不按TEST换epoch或结构。

主表 `main_TEST5_ablation_table.csv`，Δ=消融−Full C4；多周期见 `multi_horizon_TEST_ablation.csv`。
同时保存VAL表、完整TRAIN/VAL预测、TEST预测、history、选择审计、初值/源码/数据哈希和file_index。
独立checkpoint目录 `checkpoints/phase_m_c4_ablation/<run>/<key>/{best,latest}.pt`，不覆盖原产物。

C4是在历史TEST结果查看后选定的后续研究对象，本轮是探索性post-hoc消融。
单seed不支持显著性、纯机制因果或经济因果声明；不自动扩展seed、调参或替换默认主模型。
