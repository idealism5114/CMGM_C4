# Deep V3 baseline selection repair

{
  "protocol_id": "CMGM_DEEP_V3_C4_VAL5_MACRO_SEED42_V1",
  "selector": "VAL_5D_COMMODITY_MACRO_MAE",
  "scheduler_metric": "VAL_5D_COMMODITY_MACRO_MAE",
  "early_stopping_metric": "VAL_5D_COMMODITY_MACRO_MAE",
  "seed": 42,
  "models": [
    "RNN",
    "GRU",
    "LSTM",
    "VanillaTransformer",
    "GraphWaveNet",
    "MTGNN",
    "MSGNet",
    "CrossGNN"
  ],
  "train": {
    "optimizer": "Adam",
    "lr": 0.0001,
    "weight_decay": 1e-05,
    "batch": 64,
    "shuffle": false,
    "drop_last": false,
    "max_epochs": 200,
    "patience": 10,
    "scheduler_factor": 0.5,
    "scheduler_patience": 5,
    "prediction_loss": "four-horizon masked Huber(.02) means SUM; no baseline auxiliary loss",
    "gradient_clip": null,
    "AMP": false,
    "epoch0_candidate": false
  },
  "changed_vs_old_baselines": [
    "checkpoint selector",
    "scheduler monitor",
    "early stopping monitor"
  ],
  "unchanged_vs_old_baselines": [
    "architecture and adapter",
    "data and masks",
    "loss",
    "seed/init order",
    "optimizer/budget"
  ],
  "comparison_limits": "Baselines use V3 27-token compressed view; C4 full311-node graph/native KL. This is a common selection protocol, not identical architecture/information capacity.",
  "test_policy": "All eight best checkpoints frozen before TEST access; reuse existing release, no new preprocessing/fitting. Historical TEST already viewed."
}
