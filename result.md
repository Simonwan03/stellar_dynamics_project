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