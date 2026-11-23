# CMGM_C4 迁移与核验报告

本目录整理原 Phase M C4、最新 VAL5 选模的8个 baseline，以及 Full +12 个消融。未启动正式训练或新 TEST 推理。

- 原仓库提交：`981de35aa9a0472e6974d5a6e4f98e1c5242e465`。HEAD、status、diff stat、binary diff hash 前后一致；用户已有修改保留。
- 528 个复制文件、810.52 MB，源/目标 SHA256 均一致，无指回旧仓库的软/硬链接。
- 22 个索引条目对应21个独立 best checkpoint；Full C4 为共享别名，不增加一次独立实验。
- 57项 CPU 测试通过，包含临时目录独立恢复、机制/梯度、原 train/eval 随机数与梯度对照、TEST 哨兵。
- 22个条目的完整 TRAIN/VAL 共44次冻结回放全部通过；GPU预测与历史缓存最大差0。
- C4 VAL5 Macro MAE：0.024297203261564；epoch35，567169参数。
- 独立 CPU CLI 复核 VAL5：0.024297203290524。
- 新默认训练入口使用原未训练初始状态，不使用已训练 checkpoint warm start；训练需显式 `--execute`。
- TEST 数组解析0、TEST推理0。复制和字节哈希不等同于重新评估；汇总引用既有指标JSON。

## 交付边界

预处理文件、标准化、mask、资产/商品顺序已独立复制；没有重建原始CSV处理流水线。新训练循环已通过静态/机制验证，但本轮没有正式重新训练它，不能承诺在不同硬件上逐bit复现训练轨迹。
主模型保留原始 C4 Huber+Switch KL。MAE/no-KL和路由辅助损失等后续研究没有悄悄改变默认定义。
baseline使用27个中性token，C4使用311完整节点；选模统一不等于结构和输入完全匹配。GraphWaveNet原3104个闲置参数保留。

## 本次修正的迁移工具问题

首次CPU回放的汇总文件名假设不适用于baseline，改为对原预测缓存直接重算相同指标；保留初始失败日志。测试对照中统一原/新CPU线程数，且分离两层TEST哨兵，未改变模型、数据或checkpoint来通过检查。

## 模型索引

| Model | Params | Trainable | Best epoch | VAL5 Macro MAE |
|---|---:|---:|---:|---:|
| c4/full | 567169 | 567169 | 35 | 0.024297203262 |
| baselines/RNN | 151388 | 151388 | 87 | 0.024635379497 |
| baselines/GRU | 430428 | 430428 | 48 | 0.032433120865 |
| baselines/LSTM | 569948 | 569948 | 13 | 0.025868768560 |
| baselines/VanillaTransformer | 366812 | 366812 | 12 | 0.038273901889 |
| baselines/GraphWaveNet | 260736 | 260736 | 23 | 0.024525201053 |
| baselines/MTGNN | 102057 | 102057 | 154 | 0.024667354105 |
| baselines/MSGNet | 82426 | 82426 | 128 | 0.024316032474 |
| baselines/CrossGNN | 678 | 678 | 59 | 0.024744105747 |
| ablations/full_c4 | 567169 | 567169 | 35 | 0.024297203262 |
| ablations/no_spatial | 567169 | 473387 | 21 | 0.024277618148 |
| ablations/no_temporal | 567169 | 100850 | 38 | 0.024339597368 |
| ablations/no_spatial_temporal_attention | 567169 | 567104 | 26 | 0.024301509388 |
| ablations/no_adaptive_graph | 567169 | 560748 | 35 | 0.024295921254 |
| ablations/no_edge_attn_mixhop | 592129 | 533889 | 25 | 0.024344199676 |
| ablations/no_rpe | 567169 | 567013 | 51 | 0.024292586140 |
| ablations/no_adaptive_regime_routing | 567169 | 566773 | 48 | 0.024301035388 |
| ablations/no_regime_specific_transition | 567169 | 501249 | 39 | 0.024395960383 |
| ablations/no_microstate | 567169 | 562881 | 25 | 0.024328447680 |
| ablations/no_balanced_readout | 567169 | 566913 | 25 | 0.024309467446 |
| ablations/no_dynamic_routing | 567169 | 550045 | 25 | 0.024320510882 |
| ablations/no_moe | 522269 | 522269 | 46 | 0.024297144480 |

## 文件

- 使用与命令：`README.md`。
- 结构、损失、消融定义：`docs/MODEL_AND_PROTOCOL.md`。
- 逐文件源/目标映射：`provenance/migration_manifest.json`。
- 全量回放：`verification/TRAIN_VAL_REPLAY.json`、`verification/gpu_replay.log`。
- 测试：`verification/tests.txt`。
- 最终审计：`verification/FINAL_AUDIT.json`。
- 保存结果汇总：`reports/existing_experiments/`，包含历史TEST指标，未重新推理。

**MIGRATION_PASS = true**。此PASS只表示本次独立整理及所列核验通过，不表示预测性能提升。
