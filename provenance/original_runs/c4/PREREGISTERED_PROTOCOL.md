# Phase M 新数据预注册协议

{
  "id": "CMGM_PHASE_M_NEW23_SEED42_MACRO5_V1",
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
  "TEST_policy": "No TEST numeric data, labels, arrays or inference. No automatic TEST option.",
  "bootstrap": {
    "block_origins": 40,
    "repeats": 2000,
    "seed": 42,
    "role": "paired date blocks; exploratory single seed",
    "run": true
  },
  "user_data_override": "User explicitly chose new311/26/23 dataset over conflicting old21/24 requirement",
  "data_limitations": "Frozen new-data Research23 is exploratory; historical PIT membership, price release time and continuous futures roll method not certified"
}

固定按C0至C6顺序各训练seed42一次。只按VAL5商品macro MAE选模，训练仍为四周期Huber加原生KL。C0必须重新训练，不能复用按Huber选出的历史最佳。不得根据结果更换专家、增加seed或读取TEST。
