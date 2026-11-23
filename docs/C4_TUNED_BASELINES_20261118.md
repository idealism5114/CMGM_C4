# 当前调优参数下的8个Deep V3 baseline

运行目录：`runs/baselines_deep_v3_tuned/seed42_20261118_01/`。

模型依次为RNN、GRU、LSTM、VanillaTransformer、GraphWaveNet、MTGNN、MSGNet、CrossGNN，保留本项目已迁移的最新Deep V3实现。各模型strict加载自己保存的未训练Seed42初值，从头训练，不加载历史已训练checkpoint。

统一使用当前通用配置：Adam，初始lr=3e-4，weight_decay=1e-4，batch64，shuffle=False，drop_last=False，每epoch完整2726个TRAIN起点、43次更新。最多200epochs，early stopping patience10，ReduceLROnPlateau factor0.5/patience5。保持原四周期masked Huberδ0.02，各周期有效标签均值求和；baseline不加入C4专有Switch KL。

Checkpoint、scheduler和early stopping统一使用完整VAL的5日商品Macro MAE。每组独立选择best epoch，不按TEST挑选模型或epoch。所有8组完成TRAIN/VAL并锁定checkpoint SHA256后，才按已有授权统一进行冻结TEST评估。

C4参照合法复用`runs/ablations_c4_tuned/seed42_20261118_01/models/full_c4/`：相同源码、数据与训练参数、同样VAL5选模，最佳epoch15。不会因baseline的TEST排序自动切换默认主模型。

## 命令

```bash
cd /home/yangxiaotong/projects/myresearch/CMGM_C4
# 默认只运行CPU前置测试，不训练，不读取TEST数值。
../.venv/bin/python tools/run_c4_baselines.py --output runs/baselines_deep_v3_tuned/seed42_20261118_01
# 正式运行全部8组；运行目录必须是新的。下面是本轮已启动的命令，不要重复启动。
../.venv/bin/python -u tools/run_c4_baselines.py --output runs/baselines_deep_v3_tuned/seed42_20261118_01 --device cuda:0 --gpu 0 --execute --allow-test
# 仅在确定性程序中断后，用同一代码、配置和数据继续同一组状态；不重置初值。
../.venv/bin/python -u tools/run_c4_baselines.py --output runs/baselines_deep_v3_tuned/seed42_20261118_01 --device cuda:0 --gpu 0 --execute --allow-test --resume
# 全套完成后，独立核对所有best/latest、初值、optimizer覆盖、日期、标签、mask及重算指标。
../.venv/bin/python tools/verify_c4_baselines.py runs/baselines_deep_v3_tuned/seed42_20261118_01
```

## 产物和解释

`PROGRESS.json`记录完成数量，`logs/`保存每模型训练输出。`models/`保存best/latest、optimizer/scheduler/RNG、配置、历史及TRAIN/VAL预测。所有8组VAL选择锁定于`MODELS_FROZEN.json`，之后的TEST输出在`test/`。

完成时生成`REPORT.md`、`VAL_RANKING.csv`、`TEST_COMPARISON.csv`、`ALL_HORIZONS.csv`、`PER_COMMODITY_5D.csv`、`TRAINING_HISTORY.csv`、`PARAMETER_AND_TIME_AUDIT.csv`、`FINAL_AUDIT.json`及`file_index.json`。独立复核另保存`INDEPENDENT_VERIFICATION.json`、40日期/2000次配对区块统计及四段时间结果。统计方案在部分VAL训练已运行、当前TEST尚未释放时固定，只用于描述，不是确认性预注册或选模规则。

V3 baseline读取27×26中性token（股票均值/标准差、债券均值/标准差、23商品），C4读取311完整节点；目标、日期、商品顺序、标签mask和指标分母相同，输入压缩方式与模型容量并不相同。GraphWaveNet原定义含3104个无预测梯度的参数，本轮保留并单独记录，未改动其结构。

固定C4调优后选出的lr/wd用于全部baseline，不代表每个baseline的独立调参最优值。既有baseline和调参/消融结果保持原样，不回写历史配置。单Seed42、反复使用过的VAL/TEST只能支持探索性比较，不宣称独立盲测或跨随机种子稳定性。

## 本轮完成状态

8/8组均已完成，74项前置测试和独立复核通过。全部best/latest、optimizer/scheduler/RNG、预测与指标已保存。MSGNet为本轮VAL5最佳baseline；C4的TEST5仅有极小点估计优势，详细结论见运行目录REPORT.md和PAIRED_BOOTSTRAP_5D.csv。没有自动替换主模型或追加实验。
