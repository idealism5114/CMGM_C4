# Deep V3 baseline：与 C4 统一 VAL5 商品 Macro MAE 选模

本版本修复旧 Deep V3 新数据实验按四周期 VAL Huber 选模、C4 按 VAL5 商品 Macro MAE 选模的不一致。8 个模型固定为 RNN、GRU、LSTM、VanillaTransformer、GraphWaveNet、MTGNN、MSGNet、CrossGNN；seed42 各从头重新训练一次。旧结果与 checkpoint 保留。

Checkpoint、ReduceLROnPlateau、early stopping 全部监控 `VAL_5D_COMMODITY_MACRO_MAE`：每商品对完整 VAL 有效 origin 计算 MAE，再对全部23商品等权平均。没有有效标签的商品直接失败，不用 pooled MAE 或四周期 Huber替代。训练仍使用四周期 masked Huber(delta=.02)逐周期均值求和，不加 baseline 不具备的 Switch KL。

沿用 Adam lr1e-4、weight_decay1e-5、batch64、shuffle=False、drop_last=False、最多200epoch、early-stop patience10、scheduler factor.5/patience5；无新增梯度裁剪或 AMP。与原 C4 一致，epoch0不参与选模。更换 scheduler 监控会影响训练轨迹，必须重新训练，不能直接从旧 best 推断本次结果。

原数据和 mask 不变：311节点、26特征、23商品，TRAIN2726、VAL483，周期[1,5,10,20]。复用经过 SHA256 验证的原 baseline 中性输入缓存。V3 仍使用27个token（股票均值/标准差、债券均值/标准差、23商品）；C4保留311节点。固定槽的 baseline core 不具备完整动态图 availability mask，不声称两者内部信息或容量完全一致。GraphWaveNet原有3104个无梯度参数保留并审计，不借本次修改架构。

每个 baseline 先生成同一原有构造顺序的seed42随机初值并存档，训练时逐张量哈希核对，不加载训练过的权重。C4不重训，strict恢复原 Phase M epoch35 并复算 VAL。所有8个新 best checkpoint 的身份、selector、epoch和SHA256冻结后，才访问既有TEST release；不重新拟合 scaler，不读原始CSV，不用TEST调参。该项目既往多次查看TEST，本次属于探索性历史TEST评价。

默认仅预检：

```bash
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 ../.venv/bin/python -m cmgm.scripts.baselines_val5_macro --output experiments/baselines_deep_v3_c4_val5/seed42_20261109
```

用户已要求本次重跑。对通过预检的新目录训练并在全套冻结后评价TEST：

```bash
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 ../.venv/bin/python -u -m cmgm.scripts.baselines_val5_macro --output experiments/baselines_deep_v3_c4_val5/seed42_20261109 --resume --run --device cuda:0
```

不得重复启动正在运行的目录。逐epoch保存best/latest、optimizer、scheduler、RNG、history；只有确定性中断恢复允许继续既有运行。结果包含 `REPORT.md`、`VAL_5D_RANKING.csv`、`TEST_5D_RANKING.csv`、四周期/逐商品表、旧新selector对照、初始化/数据/source审计、`models_frozen.json`、`FINAL_VERIFICATION.json`及`file_index.json`。旧入口与默认协议完全不变。
