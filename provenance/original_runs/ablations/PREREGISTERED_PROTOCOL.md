# C4 ablations, scope frozen before fits

{
  "id": "CMGM_C4_MECHANISM_ABLATION_NEW23_SEED42_V1",
  "seed": 42,
  "input_nodes": 311,
  "feature_dim": 26,
  "targets": 23,
  "window": 20,
  "horizons": [
    1,
    5,
    10,
    20
  ],
  "target_unit": "decimal cumulative return",
  "optimizer": "Adam",
  "lr": 0.0001,
  "weight_decay": 1e-05,
  "batch_size": 64,
  "shuffle": false,
  "drop_last": false,
  "max_epochs": 200,
  "patience": 10,
  "scheduler_factor": 0.5,
  "scheduler_patience": 5,
  "train_loss": "SUM_h valid-label Huber(.02) means + native Switch KL ONCE",
  "selector": "VAL_5D_COMMODITY_MACRO_MAE",
  "selector_formula": "mean_c mean_valid_origin abs(pred-target); equal weight over23 commodities",
  "epoch0_candidate": false,
  "router_hidden": 32,
  "horizon_embedding_dim": 8,
  "rank": 16,
  "temperature": 1.0,
  "auxiliary_loss": false,
  "training_initialization": "Fresh original D0B construction then explicit common/expert/router state copying; no trained weight initialization",
  "dropout_note": "C5 head body on B,K,64; C6 on B,H,64. K=H=4: same random draw shape/count, distinct operands; eval has no dropout.",
  "TEST_policy": "User authorized full C4 ablations and TEST; all checkpoints freeze before TEST; no TEST selection",
  "bootstrap": {
    "run": false,
    "reason": "Not requested for this ablation task"
  },
  "user_data_override": "User explicitly chose new311/26/23 dataset over conflicting old21/24 requirement",
  "data_limitations": "Frozen new-data Research23 is exploratory; historical PIT membership, price release time and continuous futures roll method not certified",
  "configurations": [
    "full_c4",
    "no_spatial",
    "no_temporal",
    "no_spatial_temporal_attention",
    "no_adaptive_graph",
    "no_edge_attn_mixhop",
    "no_rpe",
    "no_adaptive_regime_routing",
    "no_regime_specific_transition",
    "no_microstate",
    "no_balanced_readout",
    "no_dynamic_routing",
    "no_moe"
  ],
  "reuse": {
    "full_c4": "C4",
    "no_moe": "C0"
  },
  "additional_training_runs": 11,
  "former_adaptive_fusion": "User approved replacement with uniform four-expert routing; gate ablation not applicable to C4",
  "status": "Exploratory post-hoc study; C4 selected for follow-up after observing prior TEST results",
  "inactive_parameters": "Structurally unused whole tensors frozen after synthetic/actual TRAIN preflight; optimizer excludes them. Partial masked router coordinates may have zero gradients."
}

Definitions:
- full_c4: Exact Phase M C4, reused seed42 best epoch35; prediction-level four-expert mixture
- no_spatial: Remove graph spatial encoder/projection; disable spatial and ST experts; recent/long experts remain with masked two-candidate dynamic routing
- no_temporal: Remove switching temporal encoder, KL and temporal-dependent experts/router; spatial expert through the same shared head
- no_spatial_temporal_attention: Uniform mean over available historical tokens per node; graph unchanged
- no_adaptive_graph: A=ones(N,N); native edge attention and both EdgeAttnMixHop blocks retained
- no_edge_attn_mixhop: Learned A retained; two ordinary MixHopPropagation(64,64,K=2,beta=.05), exactly the Sep17 definition
- no_rpe: BaseRPE bias zero; complete causal Transformer retained
- no_adaptive_regime_routing: p/prior uniform; three distinct generators retained; KL zero
- no_regime_specific_transition: Adaptive Markov p/native KL retained; generator[0] shared across candidates
- no_microstate: Exact Sep17 post-normalization microstate null in final balanced readout; live Z recurrence AND recent-Z expert retained
- no_balanced_readout: Bypass only long/micro readout LayerNorm; both projections retained
- no_dynamic_routing: All four experts retained; fixed weights 1/4; dynamic context/router inactive (frozen and excluded from optimizer)
- no_moe: Exact original D0B channel gate and shared head; no experts/router; reuse Phase M C0 seed42 best epoch46
