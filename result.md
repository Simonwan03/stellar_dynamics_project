## Linear SVM / 200 epochs
**T=100**
train_acc: 0.8500
test_acc : 0.8083

**T=200**
train_acc: 0.8955
test_acc : 0.8420

**T=500**

## 1D-CNN / 50 epochs
**T=100**
TEST | AUC 0.9544 | F1 0.8284 | Acc 0.8674
**T=200**
TEST | AUC 0.9630 | F1 0.8568 | Acc 0.8886
**T=500**
TEST | AUC: 0.9783 | F1: 0.9087 | Acc: 0.9308

Input
  └─ x: [B, in_ch, L]

Backbone
  ├─ Conv1d(in_ch → 128, k=7, pad=3)      -> [B, 128, L]
  ├─ BatchNorm1d(128)                     -> [B, 128, L]
  ├─ ReLU                                  -> [B, 128, L]
  ├─ Conv1d(128 → 256, k=5, pad=2)        -> [B, 256, L]
  ├─ BatchNorm1d(256)                     -> [B, 256, L]
  ├─ ReLU                                  -> [B, 256, L]
  ├─ Conv1d(256 → 256, k=3, pad=1)        -> [B, 256, L]
  ├─ BatchNorm1d(256)                     -> [B, 256, L]
  ├─ ReLU                                  -> [B, 256, L]
  └─ AdaptiveAvgPool1d(1)                  -> [B, 256, 1]

Head
  ├─ Flatten                               -> [B, 256]
  ├─ Dropout(p=0.3)                        -> [B, 256]
  └─ Linear(256 → 1)                       -> [B, 1]  

## Attention / 50 epochs
**T=100**
TEST | AUC 0.9515 | F1 0.8307 | Acc 0.8670
**T=200**
TEST | AUC 0.9637 | F1 0.8648 | Acc 0.8974
**T=500**
TEST | AUC 0.9783 | F1 0.9210 | Acc 0.9410

Positional Encoding
  ├─ Sin/Cos PE (max_len ≥ T(+1))          -> [1, T(+1), d_model]
  ├─ Add PE                                -> [B, T(+1), d_model]
  └─ Dropout(p=dropout)                    -> [B, T(+1), d_model]

Backbone: Transformer Encoder × num_layers
  └─ for each layer:
       ├─ (Norm-first) LayerNorm           -> [B, T(+1), d_model]
       ├─ Multi-Head Self-Attention
       │    ├─ QKV projections             -> [B, T(+1), d_model] ×3
       │    ├─ Attention (nhead heads)     -> [B, T(+1), d_model]
       │    └─ Output projection           -> [B, T(+1), d_model]
       ├─ Residual Add                     -> [B, T(+1), d_model]
       ├─ (Norm-first) LayerNorm           -> [B, T(+1), d_model]
       ├─ FeedForward (MLP)
       │    ├─ Linear(d_model → dim_ff)    -> [B, T(+1), dim_ff]
       │    ├─ GELU                        -> [B, T(+1), dim_ff]
       │    ├─ Dropout                     -> [B, T(+1), dim_ff]
       │    └─ Linear(dim_ff → d_model)    -> [B, T(+1), d_model]
       └─ Residual Add                     -> [B, T(+1), d_model]

Pooling (sequence → vector)
  ├─ if pooling="cls": take x[:,0,:]       -> [B, d_model]
  └─ if pooling="mean": mean over time     -> [B, d_model]

Head
  ├─ LayerNorm(d_model)                    -> [B, d_model]
  ├─ Dropout(p=0.3)                        -> [B, d_model]
  └─ Linear(d_model → 1)                   -> [B, 1]   (binary logit)

## Task C: Survival Curve Prediction
Setting
- Case-level split with all 8 perturbed trajectories from the same initial condition kept in the same set.
- Total cases: 5000
- Train / Val / Test: 3500 / 500 / 1000
- Input horizon: `T_in = 100`
- Input features per trajectory: 15 orbital features
- Target: survival curve at future times `[200, 400, 600, 800, 1000]`
- Physics constraint: predicted survival curve must be monotonically non-increasing

### Mean-Curve Baseline
Predict the training-set mean survival curve for every test case.

- TEST loss: `0.173331`
- TEST overall_mae: `0.387228`
- TEST mae_per_horizon: `[0.361827, 0.384286, 0.394061, 0.397485, 0.398479]`
- TEST monotone_fraction: `1.0000`

### Flat Case Model
Trajectory prefix is first reduced to a single case-level sequence `(100, 15)`, then modeled with a GRU encoder + monotone autoregressive decoder.

- Best fit: `best_epoch = 47`, `best_val = 0.020906`
- TEST loss: `0.023562`
- TEST overall_mae: `0.091460`
- TEST mae_per_horizon: `[0.083607, 0.086039, 0.089609, 0.097061, 0.100984]`
- TEST monotone_fraction: `1.0000`
- Improvement over mean baseline on TEST overall_mae: `0.295768`

Architecture
- Encoder: GRU over `(100, 15)` prefix
- Decoder: autoregressive GRUCell-style monotone survival decoder
- Constraint: `P_k = P_{k-1} * alpha_k`, `alpha_k in [0, 1]`, `P_0 = 1`

### Hierarchical Set-Seq Model
Keep the full perturbation set `(8, 100, 15)`. Each trajectory is encoded independently with a shared GRU, then aggregated with cross-trajectory attention before monotone decoding.

- Best fit: `best_epoch = 23`, `best_val = 0.014173`
- TEST loss: `0.015504`
- TEST overall_mae: `0.070258`
- TEST mae_per_horizon: `[0.075002, 0.061150, 0.066067, 0.071811, 0.077261]`
- TEST monotone_fraction: `1.0000`
- Improvement over mean baseline on TEST overall_mae: `0.316969`
- Improvement over flat case model on TEST overall_mae: `0.021202`

Architecture
- Trajectory encoder: shared GRU for each of the 8 trajectories
- Set aggregator: cross-trajectory attention
- Decoder: monotone autoregressive survival decoder

### Temporal + Set Dual Attention Model
Use temporal self-attention within each trajectory, then set attention across the 8 perturbations, followed by the same monotone survival decoder.

- Best fit: `best_epoch = 32`, `best_val = 0.011909`
- TEST loss: `0.016849`
- TEST overall_mae: `0.070382`
- TEST mae_per_horizon: `[0.075645, 0.060703, 0.064337, 0.071629, 0.079597]`
- TEST monotone_fraction: `1.0000`
- Improvement over mean baseline on TEST overall_mae: `0.316846`
- Improvement over flat case model on TEST overall_mae: `0.021078`
- Improvement over hierarchical set-seq model on TEST overall_mae: `-0.000124`

Architecture
- Temporal encoder: Transformer-style self-attention over each `(100, 15)` trajectory
- Set encoder: cross-trajectory attention over 8 trajectory tokens
- Decoder: monotone autoregressive survival decoder

### Summary
- The monotonic survival constraint is satisfied in all tested models: `monotone_fraction = 1.0000`.
- The mean-curve baseline is weak, showing that the prefix contains strong predictive signal.
- The flat GRU Seq2Seq model already gives a large improvement over the baseline.
- The hierarchical set-seq model is the strongest current model on test MAE.
- Explicitly modeling the 8 perturbed trajectories is more valuable than only increasing temporal model complexity.
- The dual-attention model is competitive, but does not materially outperform the simpler hierarchical set-seq model in the current setup.
