# CMGM C4 独立研究项目

本目录固定原 Phase M **C4 Global Dynamic Prediction MoE**，并整理它的 8 个最新 Deep V3 baseline 和 13 行消融结果（Full + 12）。运行代码、预处理数据和 checkpoint 均在本目录内；运行时不导入原 `Commedities` 仓库。原仓库与历史实验保留。

本次整理没有正式训练、没有新增 TEST 推理。已有 TEST 数据和预测文件只作字节复制与 SHA256 校验；结果汇总读取的是历史保存的指标 JSON。TEST 曾多次用于研究，历史指标属于探索性结果。

## 开始使用

```bash
cd /home/yangxiaotong/projects/myresearch/CMGM_C4
../.venv/bin/python run.py list
../.venv/bin/python run.py preflight
```

当前机器复用相邻的 Python 环境，仅复用软件包，模型和数据不依赖旧仓库。已验证的包版本见 [requirements.lock](requirements.lock)。整个文件夹可以一起迁移；另一台机器需要自行提供兼容的 Python/PyTorch 环境。无需运行导出工具。

也可以安装本地包：`python -m pip install -e '.[test]' --no-deps`，并在项目目录使用 `cmgm-c4`。默认 `run.py` 只列出模型；`train` 未指定 `--execute` 时只运行 preflight。

## 固定身份

| 项目 | 版本 |
|---|---|
| 默认主模型 | `c4/full`；原 ID `moe_phase_m/pred_global_dynamic` |
| Best epoch / 参数量 | 35 / 567169 |
| Best checkpoint SHA256 | `952dd88da6fb9be867c48bd089383ab12f4a55cc76fe38be7626d5bf4e6a612a` |
| 已复现 VAL5 商品 Macro MAE | 0.024297203261564152 |
| 输入 / 输出 | `[B,20,311,26]` / `[B,4,23]` |
| 市场节点 | 263 股票、25 债券、23 商品 |
| 周期 | `[1,5,10,20]`，冻结协议交易索引步，累计收益率小数 |
| TRAIN / VAL origins | 2726 / 483 |
| 统一选模指标 | VAL5 商品 Macro MAE；同时用于 scheduler 和 early stopping |
| Baseline 版本 | `baselines_deep_v3_c4_val5/seed42_20261109`，8 模型已重新按 VAL5 选模 |
| 消融版本 | `phase_m_c4_ablation/seed42_20261026_r1`，Full + 12 |

本项目默认保留原 C4 的四周期 masked Huber + 原 Switch KL。后续 MAE、取消 KL、辅助路由损失等研究版本未并入默认主模型。

## 目录

```text
src/cmgm_c4/
  models/          C4、四专家、Router、原 SLBR 原语、消融
  baselines/       8 个最新 Deep V3 定义及中性输入适配
  data/            冻结 TRAIN/VAL 读取；TEST 显式开关
  training/        独立训练入口、恢复、覆盖率检查
  evaluation/      原始指标定义和冻结推理
configs/           可移植模型索引、数据及训练协议
data/              完整预处理数据、mask、标准化和 manifest
initial_states/    原实验未训练的匹配初始状态
checkpoints/       复制的 best/latest 和训练状态
results/           原实验结果、历史与预测缓存
reports/           新生成的指标汇总
verification/      测试、完整 TRAIN/VAL 回放和迁移审计
provenance/        原配置、源码快照、哈希和旧→新路径关系
runs/              以后显式启动的新训练输出
```

`provenance` 中保留的旧绝对路径是来源记录，运行时路径由 `configs/registry.json` 的相对路径解析。`tools/export_from_cmgm.py` 和 `tools/capture_legacy_reference.py` 是一次性迁移工具，不是训练依赖；不要重新运行它们覆盖本目录。

## 复现与汇总

完整 TRAIN/VAL 冻结回放（CPU 或 CUDA）：

```bash
../.venv/bin/python run.py verify --device cuda:0
../.venv/bin/python run.py evaluate --model c4/full --split val --device cuda:0 --output runs/c4_val_check_01
../.venv/bin/python run.py summarize --output reports/archive_summary_02
```

`verify` 不读取 TEST 数组；它会先对复制文件进行字节哈希检查，再核验全部模型的 TRAIN/VAL。默认汇总只读取已保存的 JSON，不进行模型训练或 TEST 推理。输出目录必须是新的。

已有汇总：[ALL_RESULTS.csv](reports/existing_experiments/ALL_RESULTS.csv)、[VAL_RANKING.csv](reports/existing_experiments/VAL_RANKING.csv)、[历史 TEST_RANKING.csv](reports/existing_experiments/TEST_RANKING.csv)。Full C4 在主模型与消融表各出现一次，不是两次独立实验。

## 以后重新训练

以下命令需要用户主动执行，本次未执行。默认加载匹配的**未训练初始状态**，不会把 best checkpoint 当作初始化。

```bash
# 主模型
../.venv/bin/python run.py train --model c4/full --output runs/c4_seed42_01 --device cuda:0 --execute
# baseline 示例；其余名称通过 list 查看
../.venv/bin/python run.py train --model baselines/GRU --output runs/gru_seed42_01 --device cuda:0 --execute
# 消融示例
../.venv/bin/python run.py train --model ablations/no_adaptive_graph --output runs/no_adaptive_graph_seed42_01 --device cuda:0 --execute
# 中断恢复：相同代码/协议/数据，保留原运行目录
../.venv/bin/python run.py train --model c4/full --output runs/c4_seed42_01 --device cuda:0 --execute --resume
```

训练只访问 TRAIN/VAL。完成后保存 best/latest、optimizer/scheduler/RNG/history、最佳模型的 TRAIN/VAL 预测和指标；不自动进入 TEST。新训练只能写入本目录的 `runs/` 或目录外的新路径，不能覆盖迁移档案。

```bash
# 对完成的新训练结果汇总
../.venv/bin/python run.py summarize --runs-dir runs --output reports/new_runs_01
# 使用新训练正式冻结的 best checkpoint 评价 VAL
../.venv/bin/python run.py evaluate --run-dir runs/c4_seed42_01 --split val --device cuda:0 --output runs/new_c4_val_01
```

TEST 另需显式 `--split test --allow-test`。该开关仅提供以后用户授权后的手动入口，本次未调用。

## 实验解释边界

baseline 使用 27×26 中性 token（股票均值/标准差、债券均值/标准差、23 商品），C4 使用 311 个完整节点；它们共享目标、日期、标签 mask 和选模指标，输入读出能力并不完全匹配。

原 GraphWaveNet 保留了 3104 个没有预测梯度的参数；本次未删除，以保持 checkpoint 和历史实验定义一致。消融中的整支删除、均匀 Router、普通 MixHop 替代等含义见 [MODEL_AND_PROTOCOL.md](docs/MODEL_AND_PROTOCOL.md)。

本版本从已冻结预处理文件开始。原标准化参数、资产顺序和数据来源元数据已复制；没有重写从原始 CSV 重新生成该版本数据的生产流程，也不宣称解决了历史成员资格、真实发布时间或连续期货换月的所有数据问题。

检查结果见 [MIGRATION_REPORT.md](MIGRATION_REPORT.md)，模型图见 [figures/c4_architecture](figures/c4_architecture/README.md)。

## 2026-11-13 调参记录

9组Seed42已完成，VAL选择T08（lr3e-4、weight_decay1e-4），TEST5 Macro MAE为0.0242633933，较本轮默认对照约改善0.279%。原默认配置未覆盖。详见 [调参记录](docs/C4_TUNING_RESULT_20261113.md) 和 [完整报告](runs/tuning_lr_wd/seed42_20261113_01/REPORT.md)。

## 2026-11-18 调优 C4 消融

Full＋12消融均已从头完成Seed42训练及VAL锁定后的TEST评估。结果不支持所有组件都有稳定增益，完整呈现全部行：[消融报告](runs/ablations_c4_tuned/seed42_20261118_01/REPORT.md)。当前结果路径见 [RESULT_INDEX.json](RESULT_INDEX.json)，定义与命令见 [消融记录](docs/C4_TUNED_ABLATION_20261118.md)。原模型工厂默认checkpoint与配置未覆盖，恢复新结果请使用`--run-dir`。

## 2026-11-18 通用训练默认值更新

现已按用户要求将[通用配置](configs/project.json)更新为初始`lr=3e-4`、`weight_decay=1e-4`。今后`run.py train --execute`不传优化器覆盖参数时，主模型、消融和baseline均使用这套默认值；其他训练协议不变。命令行`--lr`、`--weight-decay`仍可显式覆盖。

此前“默认配置未覆盖”的记录描述当时状态。历史实验配置、源码快照、指标和checkpoint保留原样，已有baseline不会因此自动变成使用新参数训练的结果。复现或恢复旧参数运行需显式指定其原始lr/weight_decay，并满足原有源码、数据及协议校验。此次默认值更新不改变`evaluate --model c4/full`指向的历史checkpoint；评价当前调优版请继续使用对应的`--run-dir`。

## 2026-11-18 调优参数下的新baseline

8个Deep V3 baseline已从未训练Seed42初值重新完成训练，并在全部VAL5 checkpoint锁定后统一评价TEST。使用当前默认lr3e-4、weight_decay1e-4，其他预算及VAL5商品Macro MAE选模协议与调优C4一致。

MSGNet的VAL5最低；C4的TEST5点估计最低，但与MSGNet仅差约0.00000387，配对区块区间跨0，不能宣称稳定优于全部baseline。完整结果及限制见[报告](runs/baselines_deep_v3_tuned/seed42_20261118_01/REPORT.md)，运行/复核命令见[记录](docs/C4_TUNED_BASELINES_20261118.md)。当前路径记录于[RESULT_INDEX.json](RESULT_INDEX.json)，评价新baseline请使用相应`models/<模型名>`作为`--run-dir`。历史索引及结果保留。
