# Latest Results for Tasks A--E

This document consolidates the latest saved experimental results currently available in the repository. No experiments were rerun while preparing this summary. Unless stated otherwise, the reported metrics are test-set metrics.

## Common data context

- The main dataset contains 4,998 complete valid systems, with 32 variations per system and 159,936 complete trajectories.
- The default initial-condition representation is `orbital15` (15 orbital channels per time step).
- Splits are made at the system level when the task is system-based; Task B uses a trajectory-level group split by `raw_case_id`.
- Metrics from different tasks use different targets and protocols and should not be compared numerically across tasks.

## Task A: Stable-fraction and binary stability prediction

### Protocol

The latest expanded experiment uses `seed=42`, a `70/10/20` system split, `orbital15` initial conditions, and a maximum horizon of 10,000. Two targets are evaluated:

1. `stable_fraction`: the fraction of the 32 variations that remain stable up to the selected horizon.
2. `binary_any`: a binary target whose positive class is a system for which all 32 variations remain stable within the horizon.

### Stable-fraction regression at horizon 10,000

| Model | MSE | MAE | R² | Spearman |
| --- | ---: | ---: | ---: | ---: |
| MLP-small | 0.08553 | 0.20680 | 0.42798 | 0.64393 |
| MLP-base | 0.09157 | 0.21485 | 0.38760 | 0.62934 |
| MLP-large | 0.09377 | **0.20326** | 0.37285 | 0.63198 |
| XGBoost-base | 0.08624 | 0.22388 | 0.42327 | 0.65034 |
| XGBoost-strong | **0.08336** | 0.21765 | **0.44251** | **0.66239** |

XGBoost-strong gives the best MSE, R², and rank correlation. MLP-large gives the lowest MAE.

### Binary-any classification at horizon 10,000

The positive-class rate in the test set is 15.8%.

| Model | F1 | AUROC | AUPRC | Balanced accuracy | MCC | Brier | ECE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| MLP-small | 0.43137 | 0.82748 | 0.47315 | 0.64911 | 0.36753 | 0.10756 | 0.04670 |
| MLP-base | **0.45890** | 0.82711 | 0.47549 | **0.67224** | **0.36884** | 0.10977 | 0.05861 |
| MLP-large | 0.42636 | 0.82360 | 0.45381 | 0.64733 | 0.35825 | 0.10928 | 0.05056 |
| XGBoost-small | 0.21053 | 0.82426 | 0.46076 | 0.55617 | 0.23279 | 0.10652 | **0.01183** |
| XGBoost-base | 0.29108 | 0.84806 | 0.48913 | 0.58385 | 0.26830 | 0.10256 | 0.01874 |
| XGBoost-strong | 0.35242 | **0.85299** | **0.50816** | 0.60936 | 0.31476 | **0.10120** | 0.01993 |

MLP-base has the best thresholded F1, balanced accuracy, and MCC. XGBoost-strong has the best AUROC, AUPRC, Brier score, and therefore the strongest ranking and probability quality overall.

### Horizon trend

For the horizon grid from 1,000 to 10,000, the mean stable fraction decreases from 0.4888 to 0.2974. The stable-fraction R² of MLP-base decreases from 0.50121 to 0.38760, while XGBoost-base decreases from 0.49256 to 0.42327. The binary positive rate decreases from 0.289 to 0.158, so F1 and AUPRC should be interpreted together with the changing class balance.

Sources: [expanded Task A summary](results/task_a_experiments/metrics_augmented/summary.md) and [Task A horizon grid](results/task_a_experiments/metrics_augmented/horizon_grid/summary.md).

## Task B: Early-trajectory binary instability prediction

### Protocol and encoder distinction

Task B predicts the binary stability state of an individual trajectory from an early prefix. The data contain 159,953 trajectories, and the split uses `seed=42`, `test_size=0.125`, `val_size=0.1`, and grouping by the base system. The canonical Task B target is the stable-positive label, `time_unbound >= 10000`; the Panda experiment also reports the inverse `unbound_positive` label, `time_unbound < 10000`.

The encoder-based results use frozen `GilpinLab/panda-72M` and Chronos-2 encoders. Panda uses `patch_mean_keep_channels` and produces a `15 x 768` representation; Chronos-2 uses its frozen `[REG]` representation with the same feature dimension. The raw `orbital15` SVM/CNN/TCN/GRU/Attention experiments below are complementary temporal baselines and prefix ablations.

### Latest Panda-encoder results: prefix = 512

These runs use the same prefix=512 split as the raw-feature ablation. The table reports the stable-positive target so that it is directly comparable with the canonical Task B metrics.

| Panda head | Accuracy | Precision | Recall | F1 | AUROC | AUPRC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Linear probe | **0.92890** | **0.86260** | 0.92198 | **0.89130** | 0.97872 | 0.95452 |
| Channel attention | 0.92154 | 0.82359 | **0.95680** | 0.88521 | 0.97934 | **0.95652** |
| Channel DeepSets | 0.92379 | 0.83408 | 0.94746 | 0.88716 | **0.97945** | 0.95611 |

The linear Panda head gives the best accuracy and F1. The attention head gives the best recall and AUPRC, while DeepSets gives the best AUROC. For the inverse unbound-positive label, the corresponding F1 scores are 0.94717, 0.94040, and 0.94247 for the linear, attention, and DeepSets heads, respectively; AUROC is unchanged by label inversion.

### Prefix = 96 frozen-encoder comparison

The aligned prefix=96 Panda linear probe gives stable-positive F1=0.80812, AUROC=0.94545, and AUPRC=0.88375. Its inverse unbound-positive F1 is 0.89840. This shorter-prefix result is useful for the encoder comparison but should not be mixed numerically with the prefix=512 results.

### Encoder and physics comparison at prefix = 96

The prefix=96 comparison uses the same system-grouped split for all methods. The physics representation contains 120 features: eight summary statistics (initial, final, mean, standard deviation, slope, minimum, maximum, and delta) for each of the 15 `orbital15` channels. This is a SPoCK-inspired physics-summary baseline, not a direct implementation of the original SPoCK/MEGNO calculation.

| Representation | Head | Accuracy | Stable F1 | AUROC | AUPRC |
| --- | --- | ---: | ---: | ---: | ---: |
| Panda frozen encoder | Linear probe | 0.86715 | 0.80812 | 0.94545 | 0.88375 |
| Chronos-2 frozen encoder | Linear probe | 0.86645 | 0.81051 | 0.95076 | 0.89956 |
| Panda frozen encoder | XGBoost | 0.86645 | 0.78390 | 0.93696 | 0.86645 |
| Chronos-2 frozen encoder | XGBoost | 0.87981 | 0.80566 | 0.95133 | 0.90352 |
| Physics summary | XGBoost | **0.89707** | **0.83479** | **0.96253** | **0.91790** |

At prefix=96, Physics+XGBoost is the strongest of these three main representation families on the stable-positive F1, AUROC, and AUPRC. The Chronos-2 linear probe is slightly better than the Panda linear probe on all three ranking/classification metrics shown here.

Additional prefix=96 fusion diagnostics are available: Chronos-2 plus physics fusion reaches stable-positive F1=0.82596, AUROC=0.95599, and AUPRC=0.90989; Panda plus physics fusion reaches 0.81629, 0.94924, and 0.89168, respectively. These are logistic fusion probes and should be distinguished from the Physics+XGBoost result.

### Latest prefix = 512 comparison beyond the Panda heads

The latest saved prefix=512 physics run uses the same 120-feature summary representation with XGBoost. Chronos-2 encoder metrics are currently available at prefix=96, but no corresponding prefix=512 Chronos-2 Task B classification result is saved. The separate Chronos-2 context=512 results in Task E are forecasting results and are not interchangeable with this Task B encoder experiment.

| Representation | Model | Accuracy | Stable F1 | AUROC | AUPRC |
| --- | --- | ---: | ---: | ---: | ---: |
| Physics summary | XGBoost | **0.93715** | **0.90242** | **0.98523** | **0.96517** |
| Raw `orbital15` | SVM | 0.78494 | 0.63846 | 0.84422 | 0.66840 |
| Raw `orbital15` | CNN | 0.89342 | 0.83567 | 0.95082 | 0.86541 |
| Raw `orbital15` | TCN | 0.90388 | 0.84920 | 0.95988 | 0.89190 |

At prefix=512, Physics+XGBoost is the strongest saved Task B result on the canonical stable-positive target. The Panda linear head still improves over the raw TCN baseline: F1 increases from 0.84920 to 0.89130 and AUROC from 0.95988 to 0.97872.

### Latest complete raw-prefix ablation

The complete raw-feature comparison covers prefixes 10, 20, 50, 100, 200, and 300 with SVM, CNN, TCN, GRU, and Attention.

| Prefix | Model | Accuracy | Precision | Recall | F1 | AUROC | AUPRC |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 200 | TCN | 0.8622 | 0.7575 | 0.8297 | 0.7920 | 0.9321 | 0.8293 |
| 300 | Attention | **0.8737** | 0.7546 | **0.8897** | **0.8166** | 0.9350 | 0.8158 |
| 300 | TCN | 0.8697 | **0.8123** | 0.7644 | 0.7876 | **0.9440** | **0.8654** |

The complete raw-feature ablation shows a clear improvement as the prefix grows from 10 to 200--300. At prefix=300, Attention gives the best accuracy, recall, and F1, whereas TCN gives the best precision, AUROC, and AUPRC.

Sources: [Panda linear-head metrics](experiments/panda_threebody/outputs/panda_unbound_dataset/panda_task_b_split_prefix_512/metrics.json), [Panda attention metrics](experiments/panda_threebody/outputs/panda_unbound_dataset/panda_task_b_split_prefix_512_attention/metrics.json), [Panda DeepSets metrics](experiments/panda_threebody/outputs/panda_unbound_dataset/panda_task_b_split_prefix_512_deepsets/metrics.json), [prefix=96 Panda metrics](experiments/panda_threebody/outputs/panda_unbound_dataset/panda_task_b_split_prefix_96_aligned/metrics.json), [Chronos-2 linear-probe metrics](experiments/panda_threebody/outputs/panda_unbound_dataset/chronos2_task_b_split_prefix96_reg/metrics.json), [Chronos-2 XGBoost metrics](experiments/panda_threebody/outputs/panda_unbound_dataset/chronos2_task_b_split_prefix96_reg_xgb/metrics.json), [Panda XGBoost metrics](experiments/panda_threebody/outputs/panda_unbound_dataset/panda_task_b_split_prefix96_xgb/metrics.json), [Panda plus physics fusion metrics](experiments/panda_threebody/outputs/panda_unbound_dataset/panda_task_b_split_prefix96_physics_fusion_logreg/metrics.json), [Chronos-2 plus physics fusion metrics](experiments/panda_threebody/outputs/panda_unbound_dataset/chronos2_task_b_split_prefix96_physics_fusion_logreg/metrics.json), [physics-summary XGBoost at prefix=96](experiments/panda_threebody/outputs/panda_unbound_dataset/spock_inspired_task_b_split_prefix96_xgb/metrics.json), [physics-summary XGBoost at prefix=512](experiments/panda_threebody/outputs/panda_unbound_dataset/spock_inspired_task_b_split_prefix512_xgb/metrics.json), [raw prefix=512 metrics](results/task_b_prefix_ablation/prefix_512/metrics.json), and [complete raw prefix-ablation report](results/task_b_prefix_ablation/readme.md).

## Task C: Full-system survival prediction

### Protocol

The latest Task C experiments use a 20-bin logarithmic survival representation over event times from 5 to 10,000 and predict the full 32-variation system target. Chronos-2 and PANDA use frozen embeddings extracted from a prefix of 96 steps; the raw-statistics and TCN baselines use a prefix of 100 steps. The split contains 4,998 systems with a 20% system-level test set.

`Overall MAE` is the mean error over the survival curve. `Final MAE` is the error at the final horizon. `Legacy-horizon MAE` is evaluated at the historical horizons 500, 800, 1,500, 2,500, 4,000, 6,500, and 10,000. IBS is the integrated Brier score; lower is better.

| Model | Input | Overall MAE | Final MAE | Legacy-horizon MAE | IBS |
| --- | --- | ---: | ---: | ---: | ---: |
| No-learned-predictor baseline | — | 0.30929 | 0.34780 | 0.37204 | 0.13094 |
| Chronos-2 embedding + attention | Frozen embedding | **0.05120** | 0.10682 | **0.07494** | **0.05007** |
| PANDA embedding + attention | Frozen embedding | 0.05265 | **0.10567** | 0.07604 | 0.05230 |
| Raw orbital15 statistics | Raw features | 0.06706 | 0.14988 | 0.10287 | 0.06357 |
| Raw orbital15 TCN | Raw sequence | 0.11602 | 0.20722 | 0.16250 | 0.09155 |

Chronos-2 attention achieves the best overall curve error, legacy-horizon MAE, and IBS. PANDA attention gives the lowest final-horizon MAE. Both frozen-embedding models substantially outperform the raw-feature baselines.

Sources: [Chronos-2 metrics](results/task_c_survival20_chronos_attention/metrics.json), [PANDA metrics](results/task_c_survival20_panda_attention/metrics.json), [raw-statistics metrics](results/task_c_survival20_stats_full_taskb8/metrics.json), and [TCN metrics](results/task_c_survival20_tcn/metrics.json).

## Task D: Lyapunov-like proxy-time regression

### Protocol

Task D predicts `log(tau_proxy)`, where `tau_proxy` is the first sampled time at which the semimajor-axis separation of a paired variation reaches a specified growth threshold. The system-level target uses the minimum crossing time over 16 pairs `(i, i+16)`. This is a discrete Lyapunov-like observational proxy, not a strict Lyapunov time obtained from variational equations.

The current recommended setting is threshold=10,000 with minimum pair aggregation. It contains 4,998 systems, 4,968 uncensored systems (0.600% censored), a median proxy time of 150,000, and a 90th-percentile proxy time of 1,565,000.

### Main result at threshold = 10,000, aggregation = minimum

All error metrics below are on `log(tau_proxy)`. Factor-2 and factor-3 accuracy measure the fraction of predictions within the corresponding multiplicative factor of the target.

| Model | MAE | RMSE | Spearman | Factor-2 accuracy | Factor-3 accuracy |
| --- | ---: | ---: | ---: | ---: | ---: |
| Constant-median baseline | 1.1449 | 1.5105 | — | 0.3239 | 0.6992 |
| MLP | **0.7297** | **0.9992** | **0.7891** | **0.6137** | **0.7907** |
| Random forest | 0.8304 | 1.1433 | 0.6734 | 0.5151 | 0.7334 |

The MLP is the best model under the recommended threshold=10,000 setting.

### Threshold sensitivity for the MLP with minimum aggregation

| Threshold | MAE | RMSE | Spearman | Factor-2 | Factor-3 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1,000 | **0.6794** | **0.9105** | 0.7158 | **0.6230** | 0.8050 |
| 3,000 | 0.6798 | 0.9164 | 0.7354 | 0.6182 | **0.8116** |
| 10,000 | 0.7297 | 0.9992 | **0.7891** | 0.6137 | 0.7907 |
| 30,000 | 0.7644 | 1.0704 | 0.7383 | 0.5735 | 0.7771 |

The lower thresholds yield lower log-time errors, while threshold=10,000 gives the highest Spearman correlation in this sensitivity grid. Results from the historical threshold=1,000 initial run should not be mixed with the current threshold=10,000 main setting.

Sources: [Task D sensitivity summary](results/task_d_proxy_sensitivity/summary.csv), [threshold=10,000 metrics](results/task_d_proxy_sensitivity/threshold_10000/aggregation_min/metrics.json), and [Task D protocol documentation](readme.md#task-d-lyapunov-proxy-regression).

## Task E: Long-horizon orbital forecasting

### Protocol

The latest unified benchmark uses `orbital15`, a context of 512 steps, and a one-shot prediction length of 128 steps. E1 is one-shot forecasting; E2 recursively rolls out four 128-step blocks for a total horizon of 512. Results use system-balanced aggregation through the hierarchy `system -> trajectory -> forecasting window`.

The test set contains 578 systems and 9,648 trajectories. It contains 38,245 E1 test windows and 17,485 E2 rolling windows.

### E1: one-shot 512 -> 128 forecasting

MSE, MAE, and RMSE are reported on the standardized z-scale. Unit-circle error evaluates the orbital phase representation.

| Model | MSE (z) | MAE (z) | RMSE (z) | Pearson | Unit-circle error |
| --- | ---: | ---: | ---: | ---: | ---: |
| Chronos2 | 0.2331 | **0.2282** | 0.4828 | 0.8580 | **0.2265** |
| PANDA zero-shot | **0.2217** | 0.2371 | **0.4709** | **0.8625** | 0.2862 |
| TOTO2 | 0.2262 | 0.2303 | 0.4757 | 0.8601 | 0.2310 |
| TiRex2 | 0.2422 | 0.2408 | 0.4921 | 0.8508 | 0.2277 |

PANDA zero-shot obtains the best MSE, RMSE, and Pearson correlation. Chronos2 obtains the best MAE and unit-circle error.

### E2: recursive 512-step rollout

| Model | Block 1 MAE | Block 2 MAE | Block 3 MAE | Block 4 MAE | Error amplification |
| --- | ---: | ---: | ---: | ---: | ---: |
| Chronos2 | **0.2201** | **0.2895** | **0.3179** | **0.3397** | 1.5436 |
| PANDA zero-shot | 0.2267 | 0.3033 | 0.3260 | 0.3416 | 1.5065 |
| TOTO2 | 0.2234 | 0.2957 | 0.3250 | 0.3483 | 1.5592 |
| TiRex2 | 0.2352 | 0.3039 | 0.3310 | 0.3507 | **1.4909** |

All models accumulate error during recursive rollout. Chronos2 has the lowest absolute MAE in every rollout block, while TiRex2 has the lowest relative amplification because its first-block error is already higher.

Sources: [Chronos2](runs/task_e_chronos2_orbital15_ctx512_pred128/summary.md), [PANDA zero-shot](runs/task_e_panda_zero_shot_orbital15_ctx512_pred128/summary.md), [TOTO2](runs/task_e_toto2_orbital15_ctx512_pred128/summary.md), and [TiRex2](runs/task_e_tirex2_orbital15_ctx512_pred128/summary.md).

## Overall takeaways

- **Task A:** XGBoost-strong is the strongest regression and ranking model; MLP-base is best when the binary decision is evaluated by F1.
- **Task B:** The latest prefix=512 Physics+XGBoost result reaches stable-positive F1=0.9024, AUROC=0.9852, and AUPRC=0.9652. Frozen Panda and Chronos-2 encoders are also competitive, with the latest Chronos-2 probe currently available at prefix=96.
- **Task C:** Frozen foundation-model embeddings followed by attention provide the best full-system survival prediction, with Chronos-2 leading on overall curve metrics and PANDA leading at the final horizon.
- **Task D:** The MLP predicts the Lyapunov-like proxy substantially better than the constant and random-forest baselines. The target remains a discrete observational proxy rather than a strict Lyapunov time.
- **Task E:** Foundation time-series models are competitive for one-shot forecasting, but recursive rollout still shows clear error accumulation over the 512-step horizon.
