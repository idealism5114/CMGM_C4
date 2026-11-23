# C4消融结果核验与结论

13/13结果完整：11次新seed42训练、Full C4和取消MoE两组合法复用。
Full C4 TEST MAE@5d = 0.0243311584。Δ=消融−Full，正值为移除后变差。

| Variant | TEST MAE@5d | ΔMAE | Relative Δ% | Best epoch |
|---|---:|---:|---:|---:|
| Full C4 Global Dynamic Prediction MoE | 0.0243311584 | +0.0000000000 | +0.0000 | 35 |
| w/o Spatial Branch | 0.0243423382 | +0.0000111799 | +0.0459 | 21 |
| w/o Temporal Branch | 0.0244742794 | +0.0001431210 | +0.5882 | 38 |
| w/o Spatial Temporal Attention | 0.0243512542 | +0.0000200958 | +0.0826 | 26 |
| w/o Adaptive Graph | 0.0243364056 | +0.0000052472 | +0.0216 | 35 |
| w/o EdgeAttnMixHop | 0.0244510014 | +0.0001198430 | +0.4925 | 25 |
| w/o Relative Position Encoding | 0.0243463542 | +0.0000151958 | +0.0625 | 51 |
| w/o Adaptive Regime Routing | 0.0243828727 | +0.0000517144 | +0.2125 | 48 |
| w/o Regime-Specific Transitions | 0.0244235325 | +0.0000923741 | +0.3797 | 39 |
| w/o Microstate | 0.0245100491 | +0.0001788908 | +0.7352 | 25 |
| w/o Balanced Readout | 0.0244551789 | +0.0001240205 | +0.5097 | 25 |
| w/o Dynamic Routing (uniform experts) | 0.0244909964 | +0.0001598381 | +0.6569 | 25 |
| w/o MoE (original D0B gate) | 0.0244150875 | +0.0000839291 | +0.3449 | 46 |

移除后TEST5变差：w/o Spatial Branch, w/o Temporal Branch, w/o Spatial Temporal Attention, w/o Adaptive Graph, w/o EdgeAttnMixHop, w/o Relative Position Encoding, w/o Adaptive Regime Routing, w/o Regime-Specific Transitions, w/o Microstate, w/o Balanced Readout, w/o Dynamic Routing (uniform experts), w/o MoE (original D0B gate)
移除后TEST5改善：无

以上仅描述单seed冻结TEST的边际差异，不代表统计显著。不能因为部分消融更低而临时换epoch、挑seed或追加结构搜索。
取消MoE就是独立从头训练的C0 D0B，Full就是本轮C4，两者不是用旧Huber选模结果替代。
原9月17日消融的定义被移植到C4，但模型选择统一改用Phase M的VAL5 Macro MAE；旧D0B消融表采用VAL Huber选模，二者不是严格相同协议的架构因果比较。
去Microstate仅移除最终micro读出，近期专家仍读live Z；不能将其解释为删除所有潜状态路径。
本任务在观察原Phase M TEST后开展，为post-hoc探索性研究。未自动修改默认模型，未启动额外seed/搜索。
