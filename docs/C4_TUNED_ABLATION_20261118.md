# 调优 C4 的 Full＋12 消融

本轮在 `runs/ablations_c4_tuned/seed42_20261118_01/` 新目录运行。13组均从未训练初值开始，包括Full重新复跑，不复用旧消融checkpoint。

统一使用VAL调参选出的Full超参：Seed42，Adam lr=3e-4、weight_decay=1e-4，batch64，最多200epochs，patience10，scheduler factor0.5/patience5。原四周期masked Huberδ0.02与原Switch KL保持不变；移除时间分支或固定regime等消融按原定义停用相应KL。每个模型都用自己的VAL5商品Macro MAE选择checkpoint、调度和早停。

固定13组：

1. Full C4。
2. 无空间分支（同时屏蔽空间/交互专家）。
3. 无时间分支（仅空间专家，关闭时间KL）。
4. 无空间分支中的历史时间注意力（改用有效历史均值）。
5. 无Adaptive Graph（均匀图先验，保留内容图注意力）。
6. 无EdgeAttnMixHop（两层普通MixHop替代）。
7. 无相对位置编码。
8. 无自适应regime routing（均匀概率，保留三个generator）。
9. 无regime-specific transitions（使用generator0共享）。
10. 无最终microstate readout（仍保留递推Z及Recent专家）。
11. 无Balanced Readout的两组LayerNorm（保留投影）。
12. 无动态MoE routing（四专家固定等权）。
13. 无MoE（恢复原D0B gate与共享预测头）。

共同参数逐张量匹配初始化；原整组inactive参数冻结并排除optimizer，兼容用的state key保留。Nominal参数数和trainable参数数分开报告，不把名义参数数视作有效统计容量。

```bash
CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
../.venv/bin/python -u tools/run_c4_ablation.py \
  --legacy \
  --output runs/ablations_c4_tuned/seed42_20261118_01 \
  --device cuda:0 --execute --allow-test
```

物理GPU1映射为cuda:0。默认去掉`--execute`只做preflight。发生可恢复中断时保持同一代码/协议/目录，添加`--resume`，从latest恢复optimizer/scheduler/RNG，不重新初始化已完成模型。

延续此前用户确认的C4消融＋TEST评估范围，只有全部13组TRAIN/VAL完成、best epoch与SHA锁定后才打开TEST。主模型预先指定为Full，不因TEST/消融排名改选主模型。单Seed42，历史TEST已反复查看，所有结论为探索性。

预检排除旧的历史结果汇总格式测试，因为它会主动读取历史TEST指标JSON；其余测试均在TEST访问哨兵内运行。本轮训练和VAL选择不读取TEST数组，不计算新的TEST指标。引用旧HPO完整性审计时只检查其PASS状态；该审计JSON含历史TEST摘要，这些值不进入模型、loss、排名或epoch选择。详见本轮READ_BOUNDARY_NOTE.json。

消融不分别调参；固定的是Full已选超参，因此结果不是每个变体各自最优性能。部分消融同时改变容量和随机执行路径，不能直接做严格单因素因果解释。全部结果均报告，包括优于Full的消融，不追加结构或新seed，不改历史数据及实验文件。

## 完成结果

13组均已完成，独立复核通过。Full的VAL5=0.0242420872，TEST5=0.0242633933。12个删减版中11个TEST5点估计略优于Full，只有去空间支路变差；绝大多数配对区间跨0，当前不支持所有模块都有稳定正向贡献。完整结果见本轮REPORT.md、TEST_ABLATION.csv和CONCLUSIONS.json。

恢复本轮Full应使用 `run.py evaluate --run-dir runs/ablations_c4_tuned/seed42_20261118_01/models/full_c4 --split val --device cuda:0 --output runs/new_ablation_full_check`；原`--model c4/full`仍对应归档旧checkpoint，未自动覆盖。
