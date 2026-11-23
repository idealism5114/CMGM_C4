# 当前 C4 科研模型图

图中模型为 **Phase M C4 — Global Dynamic Prediction MoE**，
ID：`moe_phase_m/pred_global_dynamic`。这是本轮继续保留的原 C4，使用原 Huber + Switch KL。

- [高清 PNG](c4_architecture.png)：3720 × 2355，适合插入 PPT 或预览。
- [矢量 PDF](c4_architecture.pdf)：适合论文排版与无损缩放。
- [可编辑 SVG](c4_architecture.svg)：适合 Inkscape / Illustrator 编辑。
- [生成源码](render_c4.py) 与 [来源审计](provenance.json)。

## 可直接用于论文的英文图注

**Architecture of the C4 global dynamic spatiotemporal prediction mixture of experts.**
(a) A 20-session history of 311 assets with 26 features per asset is processed once by each of the original spatial and temporal encoders. Four shared experts transform recent latent-state information, full-window memory, spatial structure, and low-rank spatiotemporal interactions into 64-dimensional representations. A single shared prediction head generates expert forecasts for 23 commodities at four horizons. A sample-conditioned router produces one four-expert weight vector shared across all horizons, and the final prediction is a convex combination of expert forecasts.
(b) The spatial encoder applies heterogeneous node projections, historical temporal attention, two EdgeAttnMixHop blocks with a globally learned graph prior, and masked market pooling.
(c) The temporal encoder combines causal Transformer memory, a three-state Markov regime filter, regime-specific recurrent latent generators, and balanced readout.
(d) Expert definitions and the global router are shown explicitly. The model is trained with the sum of four masked Huber losses and the original warmed Switch KL regularizer; checkpoints are selected by validation 5-session commodity Macro MAE.

## 读图与符号

输入为 `[B,20,311,26]`；311 节点由263股票、25债券、23商品组成。
输出为 `[B,4,23]`，周期顺序是公共参考日历上的未来 `[1,5,10,20]` 个索引步，
目标仍为累计收益率小数，不是自然日收益或分段增量。

`H` 是真实前向的 `[B,20,128]` Transformer 长记忆序列，`Z` 是 `[B,20,64]`
递推微状态序列，`s/t` 是原空间/时间分支投影后的64维表示。
图 a 的细竖线代表**带名字的特征接口 `(s,t,H,Z)`**，不是提前融合后的单一向量。

- E1 输入 `concat(Z_last, mean(Z_last5), Z_last−mean(Z_last5))`，经 Linear(192,64)+ReLU。
- E2 输入 `concat(H_last, mean(H_all20))`，经 Linear(256,64)+ReLU。
- E3 为 `s + ReLU(Linear(s))`。
- E4 为 `W_o((U_s s)⊙(U_t t))`，rank16；当前C4没有旧专家基类中的额外 `+t`。
- 四专家输出堆叠为 `[B,4,64]`。唯一共享Head为 Linear64→64 / ReLU / Dropout0.3 / Linear64→92；最后输出层按horizon-major行分组，得到 `[B,4,4,23]`，轴顺序为 batch/horizon/expert/commodity。
- Router 输入为四个64维块的拼接：`LN(s)`、`LN(t)`、`LN(Z_last−mean(Z_last5))`、`LN(W_R mean(H_all20))`。小MLP为256→32→4，中间GELU，softmax温度1。
- `π:[B,4]` 随样本改变，**同一样本四周期共享 π**。实现中可展开为 `[B,4,4]`，但没有四个独立Router或horizon embedding。
- `P_H/P_Z` 是balanced readout的两套投影，`W_R`是Router独立的长记忆投影；图中使用不同符号以避免误认为参数共享。

近期 `Z` 已递推整合更早状态，不能把E1解释为“只看最近5天原始数据”；20-session记忆也不表示无限长期记忆。
`A0` 为全局可学习图先验，在同一模型各样本间共享；EdgeAttnMixHop内容attention仍依赖当前节点输入。
每个EdgeAttnMixHop内部K=2，因此两块不等价于仅两次邻居平均。
Balanced readout将两个投影分别LayerNorm后拼接，再经Linear(128,64)，之后仍使用原`lstm_proj`。

当前C4没有D0B最终channel-wise gate；四专家全部执行，先预测再加权。
本图没有加入后续失败实验的动态图、GraphSkip、分歧融合、MAE替换、KL删除或四路由辅助项。
预测路径中的H/Z保持梯度连接；日志缓存可以detach。图中权重不表示金融经济因果贡献。

## 源码调用链

`NewDataPhaseMModel('C4').forward`
→ `PhaseMLiveBackbone.spatial / temporal`
→ 原 `gcn_proj / lstm_proj`
→ `ExpertBank` 与 `PhaseMRouter('C4')`
→ `expert_forecasts`（同一Head参数、horizon-major输出行）
→ `einsum('bhk,bhkc->bhc')`。

对应源码：

- [当前新数据 C4](../../../cmgm/models/phase_m_new_data.py)
- [四专家、Router及共享预测头调用](../../../cmgm/models/phase_m_prediction_moe.py)
- [原主干与mask适配](../../../cmgm/models/formal_d0b_main_ablation.py)
- [原专家基类](../../../cmgm/models/four_task_moe.py)

## 复现

仓库根目录运行：

```bash
/usr/bin/python3 figures/c4_architecture/20261112_v1/render_c4.py
```

使用系统 librsvg / PyGObject / pycairo 从确定性SVG导出PNG和PDF，无需安装新依赖。
本次仅读取源码和模型身份配置，没有执行训练、推理或读取TRAIN/VAL/TEST数值。
