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

## 历史 C4 固定身份

| 项目 | 版本 |
|---|---|
| 历史主模型 | `c4/full`；原 ID `moe_phase_m/pred_global_dynamic` |
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
../.venv/bin/python run.py train --model c4v2/full --output runs/c4v2_full_seed42 --device cuda:0 --execute
# baseline 示例；其余名称通过 list 查看
../.venv/bin/python run.py train --model baselines/GRU --output runs/gru_seed42_01 --device cuda:0 --execute
# 消融示例
../.venv/bin/python run.py train --model ablations/no_adaptive_graph --output runs/no_adaptive_graph_seed42_01 --device cuda:0 --execute
# 中断恢复：相同代码/协议/数据，保留原运行目录
../.venv/bin/python run.py train --model c4v2/full --output runs/c4v2_full_seed42 --device cuda:0 --execute --resume
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

2026-11-18 将[通用配置](configs/project.json)更新为初始 `lr=3e-4`、`weight_decay=1e-4`。该版本的主模型、消融和 baseline 都使用这组默认值。当前 baseline 训练入口读取各自的配置，见下节。

复现或恢复旧实验时使用相应运行目录记录的源码快照、配置与数据。`evaluate --model c4/full` 指向历史 checkpoint。评价当前调优版使用对应的 `--run-dir`。

## 2026-11-18 调优参数下的新baseline

8个Deep V3 baseline已从未训练Seed42初值重新完成训练，并在全部VAL5 checkpoint锁定后统一评价TEST。使用当前默认lr3e-4、weight_decay1e-4，其他预算及VAL5商品Macro MAE选模协议与调优C4一致。

MSGNet的VAL5最低；C4的TEST5点估计最低，但与MSGNet仅差约0.00000387，配对区块区间跨0，不能宣称稳定优于全部baseline。完整结果及限制见[报告](runs/baselines_deep_v3_tuned/seed42_20261118_01/REPORT.md)，运行/复核命令见[记录](docs/C4_TUNED_BASELINES_20261118.md)。当前路径记录于[RESULT_INDEX.json](RESULT_INDEX.json)，评价新baseline请使用相应`models/<模型名>`作为`--run-dir`。历史索引及结果保留。

## Baseline 独立训练参数与 C4 百分比对比

`run.py train --model baselines/<模型名> --execute` 和八模型批量入口都读取 [configs/baselines](configs/baselines) 中对应的优化器、调度、梯度裁剪、批大小和训练预算。单模型入口支持 `--lr` 和 `--weight-decay` 覆盖。主模型和消融使用 [configs/project.json](configs/project.json)。

| 模型 | Adam 学习率 | Weight decay | 调度 | Batch / 最大 epoch / 早停 patience | 参数来源 |
|---|---:|---:|---|---|---|
| RNN、GRU、LSTM | 1e-3 | 0 | 恒定 | 64 / 200 / 10 | [PyTorch Adam 默认值](https://docs.pytorch.org/docs/stable/generated/torch.optim.Adam.html)，采用本项目预算 |
| VanillaTransformer | 1.3975e-3（峰值） | 0 | 4000 步 warmup，随后按步数平方根倒数衰减 | 64 / 200 / 10 | [原论文 §5.3](https://papers.neurips.cc/paper/7181-attention-isall-you-need.pdf)，按本地宽度 128 计算，采用本项目预算 |
| GraphWaveNet | 1e-3 | 1e-4 | 恒定，梯度裁剪 5 | 64 / 100 / 无 | [官方训练脚本](https://github.com/nnzhan/Graph-WaveNet/blob/6b162e80c59a1d494809252eca055cff93dc66b1/train.py) |
| MTGNN | 1e-3 | 1e-4 | 恒定，梯度裁剪 5 | 64 / 100 / 无 | [官方多步训练脚本](https://github.com/nnzhan/MTGNN/blob/master/train_multi_step.py) |
| MSGNet | 1e-4 | 0 | 官方 `type1` | 32 / 10 / 3 | [官方 ETTh1 脚本](https://github.com/YoZhibo/MSGNet/blob/main/scripts/ETTh1.sh)、[运行默认值](https://github.com/YoZhibo/MSGNet/blob/main/run_longExp.py) |
| CrossGNN | 1e-4 | 0 | 官方 `type1` | 32 / 10 / 3 | [官方运行默认值](https://github.com/hqh0728/CrossGNN/blob/0407abd085ee8342abe0bbe6de5b2ab17c44373c/run_longExp.py) |

RNN、GRU 和 LSTM 是本地 PyTorch 层的预测适配版，采用 Adam 默认值作为训练设定。VanillaTransformer 还采用论文中的 `betas=(0.9, 0.98)` 和 `eps=1e-9`，`--lr` 覆盖其峰值。MSGNet 和 CrossGNN 前两个 epoch 使用初始学习率，此后每轮减半。所有模型继续使用本项目的预测目标、架构适配、四周期 masked Huber 和 VAL5 商品 Macro MAE 选模。MTGNN 的四个预测周期共同训练。

```bash
python tools/run_c4_baselines.py --output runs/baselines_model_specific/seed42_01 --device cuda:0 --execute --allow-test
python tools/verify_c4_baselines.py runs/baselines_model_specific/seed42_01
```

批量入口使用已冻结的调优版 C4 Full 作参照。运行目录需为新路径，并需要冻结数据、初值和 C4 参照产物。TEST 需使用 `--allow-test`，在八组 VAL checkpoint 全部锁定后计算。

`REPORT.md`、`VAL_RANKING.csv` 和 `TEST_COMPARISON.csv` 给出 C4 相对各 baseline 的 5 日商品 Macro MAE 降幅，公式为 `(baseline - C4) / baseline × 100%`。正值表示 C4 误差更小，负值表示更大，分母为零时留空。`ALL_HORIZONS.csv` 提供 1、5、10、20 日各项指标的相对变化，Hit 使用越高越好的方向。`PER_COMMODITY_5D.csv` 提供各商品的 5 日 MAE 降幅。

## 当前主模型和分支消融

`c4v2/full` 是新训练的默认主模型。四个全局可学习专家分数叠加每个样本在 1、5、10、20 日上的调整量，再分别 softmax。调整层初始输出为零。每次只移除一个大模块，单独训练四组模型：

| 模型 | 预测路径 |
|---|---|
| Full | 空间与时间分支，四个专家，分周期动态路由 |
| 去时间分支 | 空间专家与共享预测头，关闭时间编码和时间 KL |
| 去空间分支 | 近期与长期专家，按周期在两个专家间分配权重 |
| 去 MoE | 保留两条编码分支，等权平均投影特征后使用共享预测头 |

在 Linux 服务器的项目根目录运行 `bash ablate.sh`。默认使用 GPU 1（遵循已设置的 `CUDA_VISIBLE_DEVICES`）、seed 42、Adam lr 3e-4 和 weight decay 1e-4。后台运行期间可以断开 SSH，再次执行同一命令可查看运行状态、续跑中断任务或显示已完成的报告。日志在 `runs/c4v2_branches/run.log`。四组都按验证集 5 日商品 Macro MAE 选模并锁定后评估测试集；`runs/c4v2_branches/seed42/REPORT.md` 按验证集和测试集分别报告 Full 相对各消融的误差降幅。历史 baseline 表中的 C4 参照为原版本。

本次服务器已完成的四组结果如下（seed 42，5 日商品 Macro MAE，按运行报告记录）。

| 模型 | VAL MAE5 | Full 相对此行降幅 | TEST MAE5 | Full 相对此行降幅 |
|---|---:|---:|---:|---:|
| Full C4（分周期路由） | 0.024234471 | +0.000% | 0.024236098 | +0.000% |
| 去时间分支 | 0.024228179 | −0.026% | 0.024240031 | +0.016% |
| 去空间分支 | 0.024258818 | +0.100% | 0.024262351 | +0.108% |
| 去 MoE（双分支等权融合） | 0.024228051 | −0.027% | 0.024242031 | +0.024% |

## C4v2 小组件消融

在同一服务器项目根目录运行 `bash ablate.sh small`。该命令在独立目录 `runs/c4v2_components/seed42` 训练九行，每行仅启用一个历史干预。空间侧包括时间注意力池化、自适应图和边注意力 MixHop。时间侧包括相对位置编码、状态路由、分状态转移、微状态读出和平衡读出。最后一行把四个专家固定为等权。微状态读出消融仅屏蔽最终读出中的微状态，近期专家仍接收状态序列。

报告从已完成的四组实验中读取 Full 的冻结预测，核对 checkpoint、数据协议、日期、标签和掩码。九个变体各自从同一 seed 42 的未训练共享初值出发，以 VAL5 选模。四专家固定等权这一行关闭动态配权，其余八行均使用分周期路由。完成后再次运行 `bash ablate.sh small` 可查看验证集和测试集百分比，日志为 `runs/c4v2_components/run.log`。`bash ablate.sh` 继续显示四组大分支实验。

## 历史 C4 递进消融与冻结路由诊断

主表固定为 A 仅空间专家、B 空间＋近期＋长期三专家等权、C 四专家等权、D 四专家全局可学习权重、E 原动态路由 Full C4。B→C 加入时空交互专家并重新均分权重。D 的四个 logits 从零开始，softmax 后跨样本和期限共享。D→E 衡量样本相关路由的增量。A 沿用原 `no_temporal`，关闭时间 KL，B–E 使用原时间 KL。A→B 衡量引入整套时间分支后的变化。

```bash
# A–E 合成预检
python tools/run_c4_ablation.py --progressive
# 直接使用已有 Full checkpoint 对应的 TRAIN、VAL 预测缓存诊断路由
python tools/run_c4_ablation.py --diagnose-full runs/ablations_c4_tuned/seed42_20261118_01/models/full_c4
# 使用默认种子 42 训练 A–E
python tools/run_c4_ablation.py --progressive --execute --output runs/c4_progressive_01
python tools/verify_c4_ablation.py runs/c4_progressive_01
```

默认仅使用种子 42，五行共用归档的 C4 未训练骨干和专家初值。主配置使用 Adam `lr=3e-4, weight_decay=1e-4`，共 5 次拟合。显式启用 `--sensitivity` 会用同一种子增加 D、E 两次拟合，使用 `lr=1e-4, weight_decay=1e-5`。各行都按 VAL5 商品 Macro MAE 选择 checkpoint。`--allow-test` 须在启动时指定，整套 checkpoint 锁定后评估历史 TEST。恢复同一运行使用相同参数和 `--resume`。

`REPORT.md` 和 `ABLATION_SUMMARY.csv` 默认报告种子 42 的 MAE 和相对改善，种子间标准差留空。`PAIRED_SEEDS.csv` 保留每个种子的结果。相对前一行、相对 A 的改善以参照行 MAE 为分母，Full 相对此行的改善以该行 MAE 为分母。所有百分比均在相同种子、配置和划分下配对计算。`PERIOD_STABILITY_5D.csv` 给出四段连续日期上的配对差值百分比。各组 `ROUTING_VAL.json` 使用同一 E 的专家预测，在 VAL 比较等权、TRAIN 平均权重和动态权重。

旧 Full＋12 消融及其附表见 [历史实验记录](docs/C4_TUNED_ABLATION_20261118.md)。旧入口显式指定 `--legacy`。
