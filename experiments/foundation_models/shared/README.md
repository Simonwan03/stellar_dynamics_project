# Shared Task B baselines

`train_spock_inspired_task_b_xgboost.py` is the reported 120-feature orbital-summary XGBoost baseline. `train_task_b_embedding_xgboost.py` and `train_task_b_logit_physics_fusion.py` are comparison experiments that can consume either Panda or Chronos-2 outputs. `recompute_task_b_base_logits.py` is a utility for saved probes. These scripts retain the Task B label ordering and split supplied by `--prefix-dir`.
