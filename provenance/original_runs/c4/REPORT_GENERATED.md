# Phase M Horizon Conditioned Spatiotemporal Prediction MoE



Seven fresh seed42 TRAIN/VAL experiments COMPLETE. Lowest VAL5 commodity-macro MAE: C0 (0.0242971445).

All numerical differences below are candidate minus reference: negative means lower error. No TEST inference, dataset or labels.



## Table 1 Prediction performance



| Rank | Control | VAL_1d | VAL_5d | VAL_10d | VAL_20d | Delta5d_vs_D0B | Params | Best_Epoch |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | C0 | 0.01053748931 | 0.02429714448 | 0.03496883951 | 0.04922396974 | 0 | 522269 | 46 |
| 2 | C4 | 0.01061153939 | 0.02429720326 | 0.03494433025 | 0.04892210231 | 5.87811152e-08 | 567169 | 35 |
| 3 | C6 | 0.01054682516 | 0.02430282757 | 0.03492128344 | 0.04890757178 | 5.683087508e-06 | 567457 | 40 |
| 4 | C5 | 0.01058278161 | 0.02430911603 | 0.03497715001 | 0.04883340856 | 1.197155044e-05 | 567457 | 42 |
| 5 | C3 | 0.01068448442 | 0.0243128635 | 0.03506395891 | 0.04914352653 | 1.571902321e-05 | 550061 | 25 |
| 6 | C1 | 0.01061553423 | 0.02432370398 | 0.03511919134 | 0.04908474701 | 2.655950448e-05 | 550049 | 28 |
| 7 | C2 | 0.01062256857 | 0.02433747343 | 0.03511609518 | 0.04912533417 | 4.03289506e-05 | 550053 | 28 |


## Table 2 C5 expert use by horizon


| Horizon | recent_change | long_memory | spatial_structure | st_interaction | mean_sample_entropy_nats |
| --- | --- | --- | --- | --- | --- |
| 1 | 0.2942500971 | 0.1974651147 | 0.2205934146 | 0.2876913754 | 1.372110636 |
| 5 | 0.2942500971 | 0.1974651147 | 0.2205934146 | 0.2876913754 | 1.372110636 |
| 10 | 0.2942500971 | 0.1974651147 | 0.2205934147 | 0.2876913754 | 1.372110637 |
| 20 | 0.2942500971 | 0.1974651147 | 0.2205934147 | 0.2876913755 | 1.372110637 |


## Table 3 Key mechanisms


| Comparison | VAL5_difference | Delta_1d | Delta_10d | Delta_20d | Interpretation |
| --- | --- | --- | --- | --- | --- |
| C5 vs C0 | 1.197155044e-05 | 4.52923028e-05 | 8.310500169e-06 | -0.0003905611763 | candidate not lower |
| C5 vs C3 | -3.747472773e-06 | -0.0001017028102 | -8.680889683e-05 | -0.0003101179646 | candidate lower |
| C5 vs C4 | 1.191276932e-05 | -2.875777901e-05 | 3.281976462e-05 | -8.869375347e-05 | candidate not lower |
| C5 vs C6 | 6.288462929e-06 | 3.595645815e-05 | 5.586656819e-05 | -7.416322077e-05 | candidate not lower |


## Frozen C5 interventions


| Intervention | VAL5_macro_MAE | Delta5d_vs_native | prediction_mean_absolute_change |
| --- | --- | --- | --- |
| train_mean | 0.02431268513 | 3.56910015e-06 | 0.0002269309654 |
| paired_sample_shuffle | 0.02431006174 | 9.457110108e-07 | 4.301023865e-05 |
| uniform | 0.02437046914 | 6.135310474e-05 | 0.001359973884 |
| swap_1d_20d | 0.02430911603 | -2.788078796e-12 | 1.552824048e-10 |
| swap_5d_10d | 0.02430911603 | -2.858383669e-12 | 1.55062338e-10 |
| force_1d_recent_change | 0.02430911603 | -2.788078796e-12 | 0.001693370523 |
| force_1d_long_memory | 0.02430911603 | -2.788078796e-12 | 0.004671359243 |
| force_1d_spatial_structure | 0.02430911603 | -2.788078796e-12 | 0.001385935371 |
| force_1d_st_interaction | 0.02430911603 | -2.788078796e-12 | 0.001993689458 |
| force_5d_recent_change | 0.02564887635 | 0.001339760315 | 0.001921525019 |
| force_5d_long_memory | 0.0296852728 | 0.005376156766 | 0.003950997092 |
| force_5d_spatial_structure | 0.02593308165 | 0.001623965618 | 0.002293086551 |
| force_5d_st_interaction | 0.02519129447 | 0.0008821784409 | 0.001337095727 |
| force_10d_recent_change | 0.02430911603 | -2.788078796e-12 | 0.002154169767 |
| force_10d_long_memory | 0.02430911603 | -2.788078796e-12 | 0.004139123268 |
| force_10d_spatial_structure | 0.02430911603 | -2.788078796e-12 | 0.001954839324 |
| force_10d_st_interaction | 0.02430911603 | -2.788078796e-12 | 0.002159159423 |
| force_20d_recent_change | 0.02430911603 | -2.788078796e-12 | 0.001556166542 |
| force_20d_long_memory | 0.02430911603 | -2.788078796e-12 | 0.004122866043 |
| force_20d_spatial_structure | 0.02430911603 | -2.788078796e-12 | 0.003117199768 |
| force_20d_st_interaction | 0.02430911603 | -2.788078796e-12 | 0.001932775076 |


## Required answers


Q1. Yes: independently identified original gated D0B from sealed Sep17 source. Historical checkpoint is audit-only; every fit starts fresh.


Q2. Yes. Spatial graph/time weighting, market tokens, causal Transformer, regime/generators and balanced readout are unchanged. The temporal method is AST-identical until its live-tensor return.


Q3. Reused LongMemory and SpatialStructure directly. RecentChange adds the explicitly requested mean input; LowRankInteraction reuses rank16 matrices but removes the old temporal residual, per this protocol.


Q4. Yes. C5 predicts each expert before mixing; C6 mixes states before the same nonlinear head. Entire initial state is identical; dropout acts on different operands in training.


Q5. All seven seed42 fits and strict best checkpoint restores are complete.


Q6. C0 has lowest VAL5 macro MAE.


Q7. C5 vs C0: delta5d=+1.19715504366e-05; not lower. Single-run evidence only.


Q8. C5 vs C3: delta5d=-3.74747277343e-06; lower. Single-run evidence only.


Q9. C5 vs C4: delta5d=+1.19127693214e-05; not lower. Single-run evidence only.


Q10. C5 vs C6: delta5d=+6.28846292873e-06; not lower. Single-run evidence only.


Q11. 1d mean weights: {'recent_change': 0.2942500971489071, 'long_memory': 0.19746511468112346, 'spatial_structure': 0.22059341462996188, 'st_interaction': 0.28769137542193474}. These are routing weights, not economic causal influence.


Q12. 5d mean weights: {'recent_change': 0.2942500971489071, 'long_memory': 0.19746511468112346, 'spatial_structure': 0.22059341462996188, 'st_interaction': 0.28769137542193474}. These are routing weights, not economic causal influence.


Q13. 10d mean weights: {'recent_change': 0.2942500971489071, 'long_memory': 0.19746511471197473, 'spatial_structure': 0.22059341466081314, 'st_interaction': 0.28769137542193474}. These are routing weights, not economic causal influence.


Q14. 20d mean weights: {'recent_change': 0.2942500971489071, 'long_memory': 0.19746511471197473, 'spatial_structure': 0.2205934146916644, 'st_interaction': 0.28769137548363727}. These are routing weights, not economic causal influence.


Q15. C5 horizon-pair mean-route JS nats: [0.0, -1.0960373990191398e-17, -2.3202496160137903e-17, -1.0960373990191398e-17, -2.3202496160137903e-17, 2.193540812822847e-21]. Numerical horizon differences are reported without forcing a specialization interpretation.


Q16. See swap_1d_20d and swap_5d_10d rows; actual prediction/MAE changes are reported. No checkpoint was selected using these interventions.


Q17. Candidate forecast correlations/disagreements and per-expert MAE are in competence/correlation CSVs. Jointly trained single-expert inference is not an independently trained model. Differences alone do not establish useful complementarity.


Q18. All horizon argmax expert names: ['recent_change', 'recent_change', 'recent_change', 'recent_change']. Same argmax is not proof of collapse; inspect weights and entropy.


Q19. C5 maximum across-sample expert-weight std: 0.001730876543994966. Variation alone is not evidence of routing benefit.


Q20. Prediction evidence: {'C0': 1.1971550436595119e-05, 'C3': -3.7474727734264357e-06, 'C4': 1.1912769321390654e-05, 'C6': 6.2884629287301985e-06}. Mechanism evidence must additionally use mean/shuffled/swapped-route interventions. No general causal or significance claim.


Q21. All prediction and division-of-labor claims require matched seeds43/44 over all seven fixed variants, not only the winners; those runs were not started.


Q22. No TEST arrays/prices, TEST labels, loader or inference were accessed by Phase M. During initial version-document review, historical saved TEST summary numbers were visible; they were not used to define architecture, hyperparameters, ranking or training. This is not a first-blind-test claim.


## Scope and limitations


This explicitly authorized new-data protocol uses311 nodes,26 features,23 targets; old21/24 data and known Legacy bfill pollution are not used. TRAIN2726, VAL483; frozen dates/scaler/labels/masks retained. First19 TRAIN origins have padding; expert means include the original zero/carry padded live sequence positions, consistently across all controls.

Primary metric is valid-origin MAE per commodity then equal commodity mean. Four-horizon Huber training is retained, but all checkpoint/scheduler/earlystop decisions now use VAL5 macro MAE. Historic Huber-selected checkpoints are never ranked as new C0.

Single seed and previously used VAL are exploratory. Bootstrap block40 origins covers overlapping20-session targets; whole dates and commodity vectors stay paired. Intervals do not establish statistical significance. Historical PIT membership/release-time and continuous-futures roll certification remain unverified.

C1 versus C0 changes capacity and expert architecture. C4/C5 differ by288 router parameters. C5/C6 share expert/router/head initialization and structures but train dropout on different operands. No loss balance, routing warmup, oracle selection, parameter search, additional seeds or TEST follow-up. STOP.
