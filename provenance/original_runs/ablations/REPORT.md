# C4 Global Dynamic Prediction MoE — 正式机制消融（探索性）

完成训练/合法复用 13/13，完成 TEST 13/13。Full C4 与 no-MoE D0B 来自本轮 Phase M 匹配协议，未重复训练。
所有 Δ=消融−Full C4；正值表示移除后误差变大。主口径为23商品等权Macro MAE，收益率小数。
TEST423 origins：2025-01-02—2026-09-29。训练只用TRAIN/VAL；所有最佳checkpoint锁定后统一TEST。

| Variant | TEST MAE@5d | ΔMAE | Relative Δ% | RMSE@5d | Hit% | best epoch |
|---|---:|---:|---:|---:|---:|---:|
| Full C4 Global Dynamic Prediction MoE | 0.0243311584 | +0.0000000000 | +0.0000 | 0.0385833236 | 50.628 | 35 |
| w/o Spatial Branch | 0.0243423382 | +0.0000111799 | +0.0459 | 0.0386187436 | 50.493 | 21 |
| w/o Temporal Branch | 0.0244742794 | +0.0001431210 | +0.5882 | 0.0387353570 | 51.437 | 38 |
| w/o Spatial Temporal Attention | 0.0243512542 | +0.0000200958 | +0.0826 | 0.0386076218 | 50.389 | 26 |
| w/o Adaptive Graph | 0.0243364056 | +0.0000052472 | +0.0216 | 0.0385854601 | 50.130 | 35 |
| w/o EdgeAttnMixHop | 0.0244510014 | +0.0001198430 | +0.4925 | 0.0386687432 | 50.067 | 25 |
| w/o Relative Position Encoding | 0.0243463542 | +0.0000151958 | +0.0625 | 0.0385981760 | 49.933 | 51 |
| w/o Adaptive Regime Routing | 0.0243828727 | +0.0000517144 | +0.2125 | 0.0386236256 | 50.275 | 48 |
| w/o Regime-Specific Transitions | 0.0244235325 | +0.0000923741 | +0.3797 | 0.0386100258 | 50.441 | 39 |
| w/o Microstate | 0.0245100491 | +0.0001788908 | +0.7352 | 0.0386902219 | 49.445 | 25 |
| w/o Balanced Readout | 0.0244551789 | +0.0001240205 | +0.5097 | 0.0386767192 | 49.393 | 25 |
| w/o Dynamic Routing (uniform experts) | 0.0244909964 | +0.0001598381 | +0.6569 | 0.0386977936 | 48.936 | 25 |
| w/o MoE (original D0B gate) | 0.0244150875 | +0.0000839291 | +0.3449 | 0.0387062297 | 49.953 | 46 |

## Definitions and interpretation

原w/o Adaptive Fusion针对D0B gate，不适用C4，经用户明确确认换成四专家固定等权。no-MoE恢复原D0B gate，恰为已完成C0。
移除空间分支保留近期/长期两个专家，禁用空间/交互专家并在剩余候选归一化。移除时间分支仅剩空间专家，不调用时间编码器或KL。
w/o Microstate沿用原定义：仅归一化后最终micro读出为零，Z递推和近期Z专家仍保留；它不是取消所有latent信息。
不可用模块保留state_dict以核验共享初始化，但全量无梯度的参数冻结并排除optimizer；详见INITIALIZATION_AUDIT与PARAMETER_COUNTS。
全部新训练seed42、原四周期masked Huber(.02)+native KL一次；Adam1e-4、WD1e-5、batch64、max200、patience10、scheduler.5/patience5、shuffle=False、drop_last=False。
Checkpoint、scheduler、earlystop均按VAL5 Macro MAE，非旧9月17日Huber选模。epoch0不是候选。
C4是在先前TEST观察后选定用于本研究，故该消融为post-hoc探索性；不能视为盲测、统计显著或严格经济因果识别。
历史PIT/发布时间/连续期货换月仍未认证。未更改旧模型默认值、数据、报告或checkpoint；未增加其他seed/搜索。

逐项机制定义：
- Full C4 Global Dynamic Prediction MoE: Exact Phase M C4, reused seed42 best epoch35; prediction-level four-expert mixture
- w/o Spatial Branch: Remove graph spatial encoder/projection; disable spatial and ST experts; recent/long experts remain with masked two-candidate dynamic routing
- w/o Temporal Branch: Remove switching temporal encoder, KL and temporal-dependent experts/router; spatial expert through the same shared head
- w/o Spatial Temporal Attention: Uniform mean over available historical tokens per node; graph unchanged
- w/o Adaptive Graph: A=ones(N,N); native edge attention and both EdgeAttnMixHop blocks retained
- w/o EdgeAttnMixHop: Learned A retained; two ordinary MixHopPropagation(64,64,K=2,beta=.05), exactly the Sep17 definition
- w/o Relative Position Encoding: BaseRPE bias zero; complete causal Transformer retained
- w/o Adaptive Regime Routing: p/prior uniform; three distinct generators retained; KL zero
- w/o Regime-Specific Transitions: Adaptive Markov p/native KL retained; generator[0] shared across candidates
- w/o Microstate: Exact Sep17 post-normalization microstate null in final balanced readout; live Z recurrence AND recent-Z expert retained
- w/o Balanced Readout: Bypass only long/micro readout LayerNorm; both projections retained
- w/o Dynamic Routing (uniform experts): All four experts retained; fixed weights 1/4; dynamic context/router inactive (frozen and excluded from optimizer)
- w/o MoE (original D0B gate): Exact original D0B channel gate and shared head; no experts/router; reuse Phase M C0 seed42 best epoch46
