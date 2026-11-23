# 固定 C4 的学习率与 weight decay 调参

2026-11-13，用户授权在独立 `CMGM_C4` 项目中运行9组Seed42，并在VAL选定配置后获取TEST结果。

网格：学习率 `[3e-5,1e-4,3e-4]` × weight decay `[1e-6,1e-5,1e-4]`。T00为原默认 `(1e-4,1e-5)`，首先重新训练以检查轨迹复现，其余点按固定顺序执行。全部使用相同未训练状态。

只改变两项Adam参数。原四周期masked Huberδ0.02、Switch KL、架构、Dropout、batch64、TRAIN顺序及所有mask不变。最大200轮，patience10，scheduler factor0.5/patience5。选checkpoint、scheduler、early stopping及最终配置全部使用完整VAL5商品Macro MAE。

正式运行：

```bash
CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
../.venv/bin/python -u tools/tune_c4.py \
  --output runs/tuning_lr_wd/seed42_20261113_01 \
  --device cuda:0 --execute --allow-test
```

`cuda:0` 指 `CUDA_VISIBLE_DEVICES=1` 映射后的物理GPU1。去掉 `--execute` 只执行CPU preflight。中断后保持相同目录、代码和协议，添加 `--resume`；不会重新初始化已完成trial。

查看进度，不加载模型或数据：

```bash
../.venv/bin/python tools/tuning_status.py runs/tuning_lr_wd/seed42_20261113_01
```

全部9组完成前，TEST访问哨兵持续生效。随后保存 `SELECTED_MODEL.json`、锁定其哈希，再对VAL胜出配置与本轮T00默认对照执行TEST。若胜出即T00只评价一次。其他候选不打开TEST，不依据TEST重新选参数。

输出包括预注册、source snapshot、每trial best/latest、optimizer/scheduler/RNG、逐epoch曲线、TRAIN/VAL预测、VAL排名、选模锁、获准模型的TEST指标和预测、23商品结果、最终审计与报告。

不自动继续Dropout搜索，不训练Seed43/44。历史TEST已多次查看，VAL亦反复用于开发，本轮结果为探索性调参，不能视作独立盲测。旧baseline未获得本轮搜索预算，不将调参后的差距全部解释成架构贡献。
