# C4 Seed42 调参结果（2026-11-13）

实验：`runs/tuning_lr_wd/seed42_20261113_01/`。9组Adam学习率×weight_decay，原C4结构、Dropout、四周期Huber和Switch KL不变，共同未训练初值。

VAL选择：T08，lr=3e-4，weight_decay=1e-4，best epoch15，最后epoch25。默认T00最佳epoch35，完整复现原训练权重与VAL轨迹。

| 配置 | VAL5 Macro MAE | TEST1 MAE | TEST5 MAE | TEST10 MAE | TEST20 MAE |
|---|---:|---:|---:|---:|---:|
| 默认 T00 | 0.0242972033 | 0.0106228759 | 0.0243311584 | 0.0354877070 | 0.0504286740 |
| 调优 T08 | 0.0242420872 | 0.0104965014 | 0.0242633933 | 0.0355051102 | 0.0503300723 |

VAL5约改善0.227%，TEST5约改善0.279%，17/23商品TEST5改善；TEST10约退化0.049%。这是小幅探索性改善，不是跨seed稳定性证明。历史TEST已多次查看。

仅依据VAL选择并锁定配置后，评价T00和T08的TEST；其余7组不评估TEST。不继续Dropout调参或Seed43/44。原默认配置、历史checkpoint和结果保持不变。

完整报告：`runs/tuning_lr_wd/seed42_20261113_01/REPORT.md`。最佳checkpoint：`runs/tuning_lr_wd/seed42_20261113_01/trials/T08/best.pt`。

后续如需按该配置重新训练（本轮已完成，无需重复）：

```bash
../.venv/bin/python run.py train --model c4/full --lr 3e-4 --weight-decay 1e-4 --output runs/new_tuned_seed42 --device cuda:0 --execute
```

没有把调优参数写回原project.json；运行时应明确指定参数或恢复本轮冻结checkpoint。选模依据、源码快照、数据指纹、全部9组结果、23商品明细与独立复核均保存在本轮目录。
