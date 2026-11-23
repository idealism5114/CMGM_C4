# Deep V3 baselines: C4-matched VAL5 Macro MAE selection

8/8 seed42 models retrained from scratch. Checkpoint, scheduler and early stopping all monitor VAL5 commodity Macro MAE.
Training remains four-horizon masked Huber(.02); all old architectures, adapter, optimizer, batch/order and 200-epoch cap retained. Epoch0 is not a candidate, matching original C4.
C4 is the unchanged Phase M global dynamic prediction MoE, reused after strict checkpoint and VAL replay audits. No trained checkpoint initialized any new baseline.

## VAL

| Rank | Model | 1d | 5d | 10d | 20d | Delta5 vs C4 | Best epoch |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | C4 | 0.0106115394 | 0.0242972033 | 0.0349443302 | 0.0489221023 | +0.0000000000 | 35 |
| 2 | MSGNet | 0.0107240288 | 0.0243160325 | 0.0354564121 | 0.0485367875 | +0.0000188292 | 128 |
| 3 | GraphWaveNet | 0.0105336983 | 0.0245252011 | 0.0359934853 | 0.0505321295 | +0.0002279978 | 23 |
| 4 | RNN | 0.0112468513 | 0.0246353795 | 0.0357756604 | 0.0496157224 | +0.0003381762 | 87 |
| 5 | MTGNN | 0.0109223909 | 0.0246673541 | 0.0353991245 | 0.0496013088 | +0.0003701508 | 154 |
| 6 | CrossGNN | 0.0106021795 | 0.0247441057 | 0.0351330085 | 0.0487386232 | +0.0004469025 | 59 |
| 7 | LSTM | 0.0132734709 | 0.0258687686 | 0.0360898573 | 0.0511373233 | +0.0015715653 | 13 |
| 8 | GRU | 0.0197236409 | 0.0324331209 | 0.0444557527 | 0.0630907562 | +0.0081359176 | 48 |
| 9 | VanillaTransformer | 0.0337363562 | 0.0382739019 | 0.0482762975 | 0.0627786104 | +0.0139766986 | 12 |

## TEST

| Rank | Model | 1d | 5d | 10d | 20d | Delta5 vs C4 | Best epoch |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | C4 | 0.0106228759 | 0.0243311584 | 0.0354877071 | 0.0504286741 | +0.0000000000 | 35 |
| 2 | MSGNet | 0.0107725130 | 0.0244490854 | 0.0357613968 | 0.0502770285 | +0.0001179270 | 128 |
| 3 | GraphWaveNet | 0.0106096941 | 0.0246364819 | 0.0362189865 | 0.0502032222 | +0.0003053235 | 23 |
| 4 | CrossGNN | 0.0106482709 | 0.0246859765 | 0.0354145372 | 0.0499414437 | +0.0003548181 | 59 |
| 5 | RNN | 0.0115827347 | 0.0251979569 | 0.0368214034 | 0.0518252419 | +0.0008667985 | 87 |
| 6 | MTGNN | 0.0111706689 | 0.0253955176 | 0.0363965672 | 0.0510006911 | +0.0010643592 | 154 |
| 7 | LSTM | 0.0146563698 | 0.0273903570 | 0.0392436999 | 0.0543523237 | +0.0030591986 | 13 |
| 8 | GRU | 0.0223773559 | 0.0373144568 | 0.0525453159 | 0.0755890066 | +0.0129832985 | 48 |
| 9 | VanillaTransformer | 0.0381014822 | 0.0442239310 | 0.0534941057 | 0.0664636263 | +0.0198927726 | 12 |

## Interpretation boundaries

The selection mismatch is repaired. The V3 27-token compressed input view, fixed-slot core masking limits, architecture capacities and absence of native Switch KL still differ from C4; this is not a fully identical information representation control.
Changing the scheduler/early-stop monitor can alter the training trajectory as well as the selected epoch. New results cannot be obtained by merely choosing a different old checkpoint.
No model/seed/structure was changed after results. TEST was evaluated only after all eight checkpoint identities were frozen. The project has repeatedly viewed TEST; these are exploratory historical TEST results, not a blind confirmation set.
Complete per-commodity metrics, all four horizons, prediction/target/mask/date caches, source/init hashes and best/latest optimizer/scheduler/RNG states are indexed in file_index.json.
GraphWaveNet retains its pre-existing unused final graph projection (3104 parameters); total/effective-gradient counts are in preflight.json. No baseline architecture was altered to fix this during the selector rerun.

## 本次结果解读

8/8模型完成，27项合成与回归测试通过；独立复算全部TRAIN/VAL/TEST、四周期MAE/MSE/RMSE/Hit/Huber与保存值的最大差为0。最佳epoch、scheduler重放、完整TRAIN覆盖、best/latest strict恢复及历史文件保护审计通过。

C4的VAL5 Macro MAE为0.0242972032616，最佳baseline MSGNet为0.0243160324738；TEST5分别为0.0243311583823和0.0244490854284，C4比MSGNet低0.000117927046（0.482%）。差异为单Seed、已多次使用过的VAL/TEST上的探索性结果，不构成统计显著性证明。

相较旧Huber监控协议，5个baseline的TEST5改善、3个退化；统一VAL监控不保证TEST单调改善。新表完整保留所有模型。C4并非全部期限最优：例如CrossGNN在TEST10/20日低于C4，MSGNet在VAL/TEST20日低于C4，不能将5日优势推广到全部周期。后续与C4对比请使用本目录新表，不与旧Huber选模表混用。

独立复核源码保存在audit_tools/verify_delivery.py；从仓库根目录设置PYTHONPATH=.运行。预检中的padding断言兼容及测试临时目录误报均在正式训练前修正，未改变数据或模型；正式8组没有重启、重抽seed或调参。
