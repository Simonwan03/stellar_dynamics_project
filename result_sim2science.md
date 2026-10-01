# Result for Sim2Science workshop

This document consolidates the latest saved experimental results currently available in
the repository. The Equation (2)/(4) and official pretrained-MLP Task A baselines were
run on the shared split for this update; other sections summarize existing saved runs.
Unless stated otherwise, the reported metrics are test-set metrics.

## Common data context

- The main dataset contains 4,998 complete valid systems, with 32 variations per system and 159,936 complete trajectories.
- The default initial-condition representation is `orbital15` (15 orbital channels per time step).
- The current code uses one shared `raw_case_id` manifest for Tasks A--C: `3498/750/750`
  complete systems under the 70/15/15 protocol. The saved metrics below predate that
  protocol unless explicitly stated and must be rerun before cross-task comparison.
- Metrics from different tasks use different targets and protocols and should not be compared numerically across tasks.

## Task A: Stable-fraction and binary stability prediction

### Protocol

The current formal Task A results use `seed=42`, the shared `70/15/15` system split
(`3498/750/750` systems), `orbital15` initial conditions, and a horizon of 10,000.
All six methods below use the same 4,998-system universe and the same 750-system test
set. Two targets are evaluated:

1. `stable_fraction`: the fraction of the 32 variations that remain stable up to the selected horizon.
2. `binary_any`: a binary target whose positive class is a system for which all 32 variations remain stable within the horizon.

Equation (2) is the Mardling-Aarseth criterion. Equation (4) is the updated
Vynatheya et al. (2022) criterion with inner-eccentricity and nonlinear mutual-inclination
corrections. For `binary_any`, a physical criterion predicts stable only if all 32
variation margins are positive. For `stable_fraction`, its prediction is the fraction of
the 32 hard per-variation decisions that are stable. Neither physical criterion is fitted
to the train or validation labels. MLP and XGBoost use the 15-dimensional `orbital15`
representation from the first variation; the physical criteria directly use every
variation's `aratios`, `eccs`, `masses`, and `h_uvecs` fields.
The standard official pretrained MLP is the upstream 2022
`mlp_model_trip_v1.2.2.pkl`, which
takes `[q_in,q_out,a_in/a_out,e_in,e_out,i_mut/pi]` for every variation and has six
hidden layers of 50 logistic units. Its class 1 probability means unstable. The
`binary_any` score is the minimum stable probability
over 32 variations. Results report both the fixed upstream threshold 0.5 and a secondary
validation-F1 operating point with frozen weights. The `stable_fraction` score is the
mean stable probability.
The additional `mlp_model_trip_ghost_v1.2.2.pkl` baseline uses the identical inputs
and aggregation, but was trained with the 2023 ghost-orbit label: a bound system is
also unstable when a minimally perturbed companion orbit diverges beyond the upstream
threshold.

### Current shared-split binary-any results at horizon 10,000

The test positive rate is `119/750 = 15.87%`. The locally trained MLP and XGBoost
thresholds are selected on the validation split by maximum F1; Equations (2) and (4)
retain their fixed physical boundary (`margin > 0`), and both official pretrained MLPs
retain their upstream threshold of 0.5 while also reporting a secondary validation-F1
threshold for a matched operating-point comparison.

#### Thresholded classification

| Model | Accuracy | Balanced acc. | Precision | Recall | F1 | Specificity | MCC | TN / FP / FN / TP |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Equation (2) | 0.6187 | 0.7529 | 0.2875 | 0.9496 | 0.4414 | 0.5563 | 0.3701 | 351 / 280 / 6 / 113 |
| Equation (4) | 0.6600 | 0.7536 | 0.3046 | 0.8908 | 0.4540 | 0.6165 | 0.3716 | 389 / 242 / 13 / 106 |
| Official pretrained MLP (fixed 0.5) | 0.6453 | 0.7858 | 0.3081 | **0.9916** | 0.4701 | 0.5800 | 0.4178 | 366 / 265 / 1 / 118 |
| Official pretrained MLP (val threshold 0.8232) | 0.7933 | 0.8158 | 0.4244 | 0.8487 | 0.5658 | 0.7829 | 0.4958 | 494 / 137 / 18 / 101 |
| Official Ghost MLP (fixed 0.5) | 0.7267 | 0.7898 | 0.3547 | 0.8824 | 0.5060 | 0.6973 | 0.4333 | 440 / 191 / 14 / 105 |
| Official Ghost MLP (val threshold 0.6872) | **0.8120** | **0.8201** | **0.4500** | 0.8319 | **0.5841** | 0.8082 | **0.5137** | 510 / 121 / 20 / 99 |
| MLP-base | 0.7507 | 0.7530 | 0.3629 | 0.7563 | 0.4905 | 0.7496 | 0.3929 | 473 / 158 / 29 / 90 |
| XGBoost-base | 0.8053 | 0.7650 | 0.4308 | 0.7059 | 0.5350 | **0.8241** | 0.4414 | 520 / 111 / 35 / 84 |

#### Ranking, prevalence, and calibration diagnostics

| Model | Threshold | Predicted stable rate | AUROC | AUPRC | Brier | Log loss | ECE | MCE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Equation (2) | 0.5000 | 0.5240 | 0.7529 | 0.2810 | 0.3813 | 6.1464 | 0.3813 | 0.7125 |
| Equation (4) | 0.5000 | 0.4640 | 0.7536 | 0.2887 | 0.3400 | 5.4802 | 0.3400 | 0.6954 |
| Official pretrained MLP (fixed 0.5) | 0.5000 | 0.5107 | **0.9124** | 0.6934 | 0.2626 | 0.9168 | 0.3418 | 0.6737 |
| Official pretrained MLP (val threshold) | 0.8232 | 0.3173 | **0.9124** | 0.6934 | 0.2626 | 0.9168 | 0.3418 | 0.6737 |
| Official Ghost MLP (fixed 0.5) | 0.5000 | 0.3947 | 0.9086 | **0.7184** | 0.1859 | 0.6400 | 0.2489 | 0.6147 |
| Official Ghost MLP (val threshold) | 0.6872 | 0.2933 | 0.9086 | **0.7184** | 0.1859 | 0.6400 | 0.2489 | 0.6147 |
| MLP-base | 0.1613 | 0.3307 | 0.8416 | 0.5139 | 0.1050 | 0.3280 | 0.0315 | **0.1768** |
| XGBoost-base | 0.2406 | **0.2600** | 0.8525 | 0.5089 | **0.1018** | **0.3200** | **0.0248** | 0.2627 |

The physical criteria output hard decisions rather than calibrated probabilities, so
their Brier, log-loss, ECE, MCE, AUROC, and AUPRC entries describe a two-level score and
should not be interpreted as probability calibration. Equation (4) improves on Equation
(2) in accuracy, precision, F1, specificity, AUPRC, and false-positive count, while losing
some stable recall. XGBoost-base gives the best accuracy, precision, and specificity.
The standard official pretrained MLP has the best AUROC and misses
only one truly stable test system, but its upstream threshold predicts 51.07% of systems
stable versus the true 15.87%, producing many false positives. This indicates strong
cross-dataset ranking with substantial threshold/calibration shift. Selecting threshold
0.8232 on validation raises test accuracy to 0.7933 and F1 to 0.5658, the best F1 in the
original comparison, without updating pretrained weights. The Ghost model slightly
reduces AUROC but raises AUPRC to 0.7184 and is less shifted at its fixed threshold.
Its validation-selected threshold 0.6872 gives the overall best accuracy (0.8120),
balanced accuracy (0.8201), F1 (0.5841), and MCC (0.5137).

### Current shared-split stable-fraction results at horizon 10,000

#### Regression and correlation metrics

| Model | MSE | RMSE | MAE | Median AE | R² | Explained var. | Pearson | Spearman | Mean bias |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Equation (2) | 0.2488 | 0.4988 | 0.3073 | 0.0469 | -0.5653 | -0.3023 | 0.5057 | 0.5256 | 0.2045 |
| Equation (4) | 0.2268 | 0.4763 | 0.2853 | **0.0312** | -0.4269 | -0.2956 | 0.5073 | 0.5400 | 0.1445 |
| Official pretrained MLP | 0.1294 | 0.3597 | 0.2397 | 0.1261 | 0.1859 | 0.3919 | **0.6870** | **0.7315** | 0.1810 |
| Official Ghost MLP | 0.1056 | 0.3249 | **0.2176** | 0.1239 | 0.3359 | 0.3846 | 0.6703 | 0.6847 | 0.0880 |
| MLP-base | 0.0901 | 0.3001 | 0.2247 | 0.1644 | 0.4335 | 0.4353 | 0.6599 | 0.6812 | 0.0169 |
| XGBoost-base | **0.0884** | **0.2972** | 0.2330 | 0.1948 | **0.4442** | **0.4443** | 0.6738 | 0.6928 | **0.0051** |

#### Prediction scale, tolerance, and sampling-noise diagnostics

The common test target mean is 0.3195 and the estimated 32-trial binomial sampling MSE
is 0.001827 for every row.

| Model | Prediction mean | Within 0.05 | Within 0.10 | Within 0.20 | MSE / binomial-noise MSE |
| --- | ---: | ---: | ---: | ---: | ---: |
| Equation (2) | 0.5240 | 0.5000 | 0.5653 | 0.6107 | 136.16 |
| Equation (4) | 0.4640 | **0.5253** | **0.5893** | **0.6360** | 124.12 |
| Official pretrained MLP | 0.5005 | 0.3800 | 0.4653 | 0.5827 | 70.82 |
| Official Ghost MLP | 0.4076 | 0.3667 | 0.4627 | 0.5973 | 57.77 |
| MLP-base | 0.3364 | 0.2267 | 0.3427 | 0.5493 | 49.28 |
| XGBoost-base | **0.3247** | 0.1627 | 0.2933 | 0.5147 | **48.35** |

The physical criteria predict exact multiples of `1/32`, which explains their very low
median error and high narrow-tolerance rates on systems near exactly 0 or 1. Their much
higher MSE and negative R² show that they perform poorly on the full stable-fraction
distribution. Both overpredict the mean stable fraction; Equation (4) reduces the bias
and improves every main error and correlation metric relative to Equation (2). XGBoost
has the best MSE, RMSE, R², explained variance, and bias, while MLP has the best MAE.
The standard official pretrained MLP substantially outperforms both equations and has the best
Pearson/Spearman ranking, but its mean prediction of 0.5005 remains far above the target
mean 0.3195, limiting its MSE and R² relative to locally trained models.
The Ghost model reduces that prediction mean to 0.4076 and consequently improves MSE,
RMSE, MAE, and R² over the standard pretrained model. It obtains the lowest MAE overall,
although XGBoost still has the best MSE/RMSE/R² and the standard pretrained model still
has the strongest correlation ranking.

Current machine-readable sources: [binary MLP/XGBoost/Equation (2)](results/task_a/sampling_2/task_a_common/binary/metrics.json),
[binary Equation (4)](results/task_a/sampling_2/task_a_common/equation4_binary/metrics.json),
[binary official pretrained MLP](results/task_a/sampling_2/task_a_common/pretrained_mlp_binary/metrics.json),
[binary official Ghost MLP](results/task_a/sampling_2/task_a_common/pretrained_ghost_mlp_binary/metrics.json),
[stable-fraction MLP/XGBoost](results/task_a/sampling_2/task_a_common/stable_fraction/metrics.json),
[stable-fraction Equation (2)](results/task_a/sampling_2/task_a_common/equation2_stable_fraction/metrics.json),
[stable-fraction Equation (4)](results/task_a/sampling_2/task_a_common/equation4_stable_fraction/metrics.json),
and [stable-fraction official pretrained MLP](results/task_a/sampling_2/task_a_common/pretrained_mlp_stable_fraction/metrics.json),
plus [stable-fraction official Ghost MLP](results/task_a/sampling_2/task_a_common/pretrained_ghost_mlp_stable_fraction/metrics.json).

The official artifact was trained with scikit-learn 1.2.2 and evaluated here with 1.8.0
after verifying SHA-256
`f8bc2f192813145f42683669adaef6448a287523875895343453055b3acf1436`.
The Ghost artifact SHA-256 is
`a260845fc9c5b80ede3ab0028ac582bffa8450dad0c23f3ce78d73987932b866`.
Of 160,000 IC rows, 512 rows (16 complete systems; 2 in the test split) fall below the
published `q_out >= 0.01` training range. This small extrapolation subset is retained and
flagged in the saved prediction files.

The source paper defines stability as remaining bound for 100 outer periods while both
semimajor axes change by no more than 10%. Task A instead uses the repository's
`time_unbound >= 10000` labels. These results therefore evaluate transfer of the
published physical criteria to Task A; they do not reproduce the paper's reported 93%
Equation (4) accuracy.

## Task B: Early-trajectory stability prediction

### Protocol and encoder distinction

Task B predicts the binary stability state of an individual trajectory from an early prefix.
The historical runs later in this section use the older `test_size=0.125`, `val_size=0.1`
configuration. The paper results immediately below use the shared `70/15/15` manifest,
with the canonical 4,998-system universe and 159,936 complete trajectories. As in
manuscript Table 10, the positive class for F1 and AUPRC is stable at `H=10000`
(`time_unbound >= 10000`).

The encoder-based results use frozen `GilpinLab/panda-72M` and Chronos-2 encoders. Panda uses `patch_mean_keep_channels` and produces a `15 x 768` representation; Chronos-2 uses its frozen `[REG]` representation with the same feature dimension. The raw `orbital15` SVM/CNN/TCN/GRU/Attention experiments below are complementary temporal baselines and prefix ablations.

### Latest paper results: shared split, matched linear probes

**These are the latest Task B results intended for the paper.** Panda and Chronos-2 use
the same 4,998-system universe, case-level `3498/750/750` train/validation/test split,
input rows, target labels, 11,521-parameter linear probe, optimizer settings, and seed.
The test set contains 24,000 trajectories from 750 systems. Train, validation, and test
have no overlapping `raw_case_id` values.

The following classification metrics use stability as the positive class, matching
manuscript Table 10. The complete table below also reports system-level stable-fraction
errors and the other model families.

| Prefix | Encoder | Test accuracy | Stable F1 | AUROC | Stable AUPRC |
| ---: | --- | ---: | ---: | ---: | ---: |
| 96 | PANDA | **0.8640** | **0.8070** | 0.9467 | 0.8911 |
| 96 | Chronos-2 | 0.8623 | 0.8067 | **0.9531** | **0.9083** |
| 512 | PANDA | **0.9217** | 0.8845 | 0.9786 | 0.9550 |
| 512 | Chronos-2 | 0.9209 | **0.8848** | **0.9791** | **0.9552** |

At both prefix lengths, Chronos-2 has slightly higher AUROC and stable AUPRC;
PANDA has slightly higher accuracy. Their stable F1 values are nearly identical.
The overall metrics include trajectories that have already become unbound within the
observed prefix: 9,776 test trajectories at prefix=96 and 13,807 at prefix=512.
The at-risk ablation below evaluates forward prediction after excluding these cases.

Sources: [Panda prefix=96 metrics](experiments/foundation_models/outputs/panda_unbound_dataset/task_b_common/panda_prefix96_linear/metrics.json), [Chronos-2 prefix=96 metrics](experiments/foundation_models/outputs/panda_unbound_dataset/task_b_common/chronos2_prefix96_linear/metrics.json), [Panda prefix=512 metrics](experiments/foundation_models/outputs/panda_unbound_dataset/task_b_common/panda_prefix512_linear/metrics.json), and [Chronos-2 prefix=512 metrics](experiments/foundation_models/outputs/panda_unbound_dataset/task_b_common/chronos2_prefix512_linear/metrics.json).

### Complete common-split comparison: encoder, physics, and raw-sequence baselines

The following table is the complete matched comparison on the new common split at two
evaluation levels. All rows use 24,000 test trajectories from the same 750 test systems.
The trajectory-binary columns use the stable-positive target
(`time_unbound >= 10000`). For system stability, the 32 predicted stable probabilities
within each system are averaged and compared with the empirical stable fraction at
`H=10000`. Thus MAE, MSE, and R² are computed over 750 systems, not over 24,000
trajectories. The raw-sequence models use the same `orbital15` channels, normalization,
seed, 50-epoch maximum, and validation-AUROC early stopping as the historical raw
baseline. Dynamical summaries use 120 prefix-summary features and XGBoost.

#### Trajectory-binary and aggregated system-stability metrics

| Prefix | Representation | Model | Trajectory Accuracy | Trajectory F1 | Trajectory AUROC | Trajectory AUPRC | System MAE ↓ | System MSE ↓ | System R² ↑ |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 96 | Panda encoder | Linear | 0.8640 | 0.8070 | 0.9467 | 0.8911 | 0.1031 | 0.0256 | 0.8390 |
| 96 | Chronos-2 encoder | Linear | 0.8623 | 0.8067 | 0.9531 | 0.9083 | 0.1112 | 0.0281 | 0.8230 |
| 96 | Dynamical summaries | XGBoost | **0.9000** | **0.8404** | **0.9649** | **0.9319** | **0.0703** | **0.0153** | **0.9040** |
| 96 | Raw sequence | SVM | 0.7493 | 0.5800 | 0.8135 | 0.6166 | 0.2373 | 0.0927 | 0.4165 |
| 96 | Raw sequence | CNN | 0.7937 | 0.6092 | 0.8888 | 0.7526 | 0.1759 | 0.0786 | 0.5056 |
| 96 | Raw sequence | TCN | 0.8167 | 0.7402 | 0.8971 | 0.7523 | 0.1570 | 0.0581 | 0.6347 |
| 512 | Panda encoder | Linear | 0.9217 | 0.8845 | 0.9786 | 0.9550 | 0.0534 | 0.0071 | 0.9554 |
| 512 | Chronos-2 encoder | Linear | 0.9209 | 0.8848 | 0.9791 | 0.9552 | 0.0614 | 0.0087 | 0.9452 |
| 512 | Dynamical summaries | XGBoost | **0.9329** | **0.8960** | **0.9851** | **0.9673** | **0.0398** | **0.0051** | **0.9676** |
| 512 | Raw sequence | SVM | 0.7590 | 0.5918 | 0.8282 | 0.6232 | 0.2184 | 0.0818 | 0.4855 |
| 512 | Raw sequence | CNN | 0.8839 | 0.8205 | 0.9470 | 0.8543 | 0.0985 | 0.0278 | 0.8249 |
| 512 | Raw sequence | TCN | 0.8973 | 0.8411 | 0.9593 | 0.8900 | 0.0765 | 0.0177 | 0.8886 |

Dynamical summaries plus XGBoost are strongest at both evaluation levels and both
prefix lengths. At prefix 96 the aggregated system-level result is MAE=0.0703,
MSE=0.0153, and R²=0.9040; at prefix 512 it improves to MAE=0.0398, MSE=0.0051,
and R²=0.9676.

#### At-risk system ablation: excluding systems already unbound inside the prefix

To test whether the gains from trajectory prefixes are driven only by systems whose
instability is already visible inside the observed window, this evaluation removes an
entire system if any of its 32 variations becomes unbound before the relevant cutoff.
With the dataset cadence of 5 time units per step, prefix 96 ends at `t=480` and prefix
512 ends at `t=2560`. The prediction target remains the system stable fraction at
`H=10000`. This is an evaluation-only ablation: the saved Task A initial-condition
XGBoost and Task B dynamical-summary XGBoost models are not retrained.

For the initial-condition versus prefix-96 comparison, 257 of the 750 test systems have
all 32 variations survive through `t=480`:

| Available information | Systems | MAE ↓ | MSE ↓ | R² ↑ |
| --- | ---: | ---: | ---: | ---: |
| Initial conditions | 257 | 0.3432 | 0.1551 | -0.8771 |
| Prefix 96 | 257 | **0.1103** | **0.0269** | **0.6746** |

Prefix 96 reduces MAE by 67.8% and MSE by 82.7%. The paired system-bootstrap
differences (prefix 96 minus initial conditions) are -0.2329 for MAE, with 95% CI
`[-0.2597, -0.2057]`, and -0.1282 for MSE, with 95% CI
`[-0.1488, -0.1086]`.

For the prefix-96 versus prefix-512 comparison, 174 test systems have all 32 variations
survive through `t=2560`:

| Available information | Systems | MAE ↓ | MSE ↓ | R² ↑ |
| --- | ---: | ---: | ---: | ---: |
| Initial conditions | 174 | 0.3985 | 0.1940 | -35.8425 |
| Prefix 96 | 174 | 0.0897 | 0.0187 | -2.5556 |
| Prefix 512 | 174 | **0.0520** | **0.0079** | **-0.4984** |

Prefix 512 reduces MAE by 42.0% and MSE by 57.9% relative to prefix 96. The paired
differences (prefix 512 minus prefix 96) are -0.0377 for MAE, with 95% CI
`[-0.0469, -0.0293]`, and -0.0108 for MSE, with 95% CI
`[-0.0154, -0.0068]`. Thus both incremental gains remain after removing systems whose
instability is already visible in the relevant prefix.

The restricted prefix-512 cohort has mean stable fraction 0.9668 and very little target
variance, so its negative R² values are unstable and should not be used as the primary
comparison; MAE, MSE, and paired differences are more informative. Machine-readable
results are available in
[the at-risk ablation directory](results/comparisons/dataset_unknown/task_ab_future_only_ablation/), and the analysis
is reproduced by
[future_only_ablation.py](tasks/comparisons/future_only_ablation.py).

#### At-risk trajectory-level ablation

This ablation asks whether high prefix-based classification performance is explained
only by trajectories whose instability is already visible in the input. For an at-risk
time `t_P`, it retains only trajectories with `time_unbound > t_P` and predicts whether
they remain stable through `H=10000`. The strict inequality excludes trajectories that
become unbound exactly at the observed endpoint. Metrics use the stable-positive target
and a fixed probability threshold of 0.5. `Stable %` reports the positive prevalence.

| At-risk cohort | Prefix | Representation | Model | Trajectories | Stable % | Accuracy | Stable F1 | AUROC | AUPRC |
| --- | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `T>480` | 96 | Panda frozen encoder | Linear probe | 14,192 | 54.04 | 0.77086 | 0.80748 | 0.87160 | 0.89231 |
| `T>480` | 96 | Chronos-2 frozen encoder | Linear probe | 14,192 | 54.04 | 0.76719 | 0.80672 | 0.88484 | 0.90879 |
| `T>480` | 96 | Dynamical summaries | XGBoost | 14,192 | 54.04 | **0.83089** | **0.84043** | **0.91228** | **0.93203** |
| `T>2560` | 96 | Panda frozen encoder | Linear probe | 10,189 | 75.27 | 0.78771 | 0.86313 | 0.81720 | 0.93007 |
| `T>2560` | 96 | Chronos-2 frozen encoder | Linear probe | 10,189 | 75.27 | 0.78850 | **0.86485** | 0.83414 | 0.93978 |
| `T>2560` | 96 | Dynamical summaries | XGBoost | 10,189 | 75.27 | **0.79988** | 0.86109 | **0.86554** | **0.95371** |
| `T>2560` | 512 | Panda frozen encoder | Linear probe | 10,189 | 75.27 | 0.81804 | 0.88592 | 0.87872 | 0.95782 |
| `T>2560` | 512 | Chronos-2 frozen encoder | Linear probe | 10,189 | 75.27 | 0.81470 | 0.88541 | 0.87807 | 0.95728 |
| `T>2560` | 512 | Dynamical summaries | XGBoost | 10,189 | 75.27 | **0.84208** | **0.89611** | **0.90400** | **0.96739** |

The `T>480` rows measure future-stability prediction from the native 96-step input.
The `T>2560` rows form the stronger matched-cohort comparison: every prefix-96 and
prefix-512 row contains exactly the same 10,189 trajectories and targets. Increasing
the observed prefix from 96 to 512 steps improves AUROC from 0.81720 to 0.87872 for
Panda, from 0.83414 to 0.87807 for Chronos-2, and from 0.86554 to 0.90400 for dynamical
summaries. Thus the longer-prefix gains cannot be attributed solely to removing
trajectories that have already become unstable. The full 12-model benchmark above is
kept separate because this focused ablation is intended to test that methodological
concern rather than provide another model ranking.

The checked-in tables are available as
[Markdown](results/comparisons/dataset_unknown/task_ab_future_only_ablation/trajectory_at_risk_table.md) and
[LaTeX](results/comparisons/dataset_unknown/task_ab_future_only_ablation/trajectory_at_risk_table.tex). The analysis
is reproduced by
[future_only_ablation.py](tasks/comparisons/future_only_ablation.py).

The checked-in two-level aggregation is available as [LaTeX](results/task_b/sampling_2/task_b_common/two_level_evaluation/task_b_two_level_table.tex)
and [Markdown](results/task_b/sampling_2/task_b_common/two_level_evaluation/task_b_two_level_table.md) tables.
Each model's raw metrics and test probabilities are stored under
[the common Task B output directory](experiments/foundation_models/outputs/panda_unbound_dataset/task_b_common/).

## Task C: Full-system survival prediction

### Protocol

Task C uses a 20-bin logarithmic survival representation over event times from 5 to
10,000 and predicts the full 32-variation system target. The current paper comparison
uses prefix=96 for every model, the shared system-level manifest
`dataset_v3/common_case_split_sampling_2_g32_h10000_r70-15-15_seed42.csv`, and 20
training epochs on the GPU. This is the latest common-split result; the historical
20%-test-set runs are retained below for provenance only.

`Overall MAE` is the mean error over the survival curve. `Final MAE` is the error at the final horizon. `Legacy-horizon MAE` is evaluated at the historical horizons 500, 800, 1,500, 2,500, 4,000, 6,500, and 10,000. IBS is the integrated Brier score; lower is better.

### Latest common split: prefix=96, 20-bin event-time/censoring target

The split contains 3,498 train systems, 750 validation systems, and 750 test systems;
all 32 variations of a system remain in the same split. The right-censoring fraction is
0.315126 at the variation level.

Architecture naming: the raw rows differ in their trajectory encoder, while all use the
same MLP hazard predictor. The PANDA/Chronos rows use a frozen foundation encoder, one
of three representation pooling choices, and the same MLP hazard predictor. In the table,
`shared MLP hazard head` means `LayerNorm → Linear → GELU → Dropout → Linear(20)`; the
input dimension changes with the pooled representation, but the predictor structure is
the same. The CLI/artifact names (`cnn`, `embedding_direct`, etc.) are retained in the
source paths for reproducibility.

| Encoder / representation | Predictor | Overall MAE | Final MAE | IBS |
| --- | --- | ---: | ---: | ---: |
| No learned encoder (train-mean curve) | None (baseline) | 0.309931 | 0.351022 | 0.131530 |
| Raw orbital15 CNN encoder | Shared MLP hazard head | 0.090960 | 0.172839 | 0.075234 |
| Raw orbital15 Stats encoder | Shared MLP hazard head | 0.066928 | 0.147462 | 0.063156 |
| Raw orbital15 TCN encoder | Shared MLP hazard head | 0.066243 | 0.156397 | 0.060562 |
| Frozen PANDA encoder + mean/std pooling | Shared MLP hazard head | 0.060236 | 0.103509 | 0.055019 |
| Frozen PANDA encoder + channel-attention pooling | Shared MLP hazard head | 0.045852 | 0.089240 | 0.048635 |
| Frozen PANDA encoder + direct pooling | Shared MLP hazard head | 0.042693 | 0.081913 | 0.045943 |
| Frozen Chronos-2 encoder + mean/std pooling | Shared MLP hazard head | 0.047562 | 0.093931 | 0.048029 |
| Frozen Chronos-2 encoder + channel-attention pooling | Shared MLP hazard head | 0.041431 | 0.088733 | 0.042572 |
| Frozen Chronos-2 encoder + direct pooling | Shared MLP hazard head | **0.037562** | 0.082615 | **0.041600** |

Chronos-2 direct-pooling + shared MLP has the best overall curve MAE and IBS; PANDA
direct-pooling + shared MLP has the best final-bin MAE (0.081913). The raw TCN encoder
with the shared MLP is the strongest raw-feature
baseline by overall MAE and IBS. Sources: [latest raw common-split metrics](results/task_c/sampling_2/task_c_survival20_common_prefix96_raw/metrics.json), [latest PANDA common-split metrics](results/task_c/sampling_2/task_c_survival20_common_prefix96_panda/metrics.json), [latest Chronos-2 common-split metrics](results/task_c/sampling_2/task_c_survival20_common_prefix96_chronos/metrics.json), and [shared split manifest](dataset_v3/common_case_split_sampling_2_g32_h10000_r70-15-15_seed42.csv).

### Per-bin test errors on the latest common split

The following tables use the 750-system test split. `System MAE` is `horizon_mae`, the
absolute error of the system-level survival fraction at each bin upper edge. `Brier` is
`brier_by_bin`, averaged over the 32 variations and test systems. Lower is better in both
tables. The 20 rows correspond to the 20 logarithmic bins, not the legacy seven horizons.

#### System-level MAE by bin

| Bin | Upper edge | Mean baseline | CNN+shared MLP | Stats+shared MLP | TCN+shared MLP | PANDA mean/std+shared MLP | PANDA attention+shared MLP | PANDA direct+shared MLP | Chronos mean/std+shared MLP | Chronos attention+shared MLP | Chronos direct+shared MLP |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 7.31 | 0.093252 | 0.051311 | 0.040894 | 0.035501 | 0.047504 | 0.039274 | 0.036154 | 0.040640 | 0.033700 | 0.029684 |
| 2 | 10.69 | 0.150522 | 0.066139 | 0.048043 | 0.037915 | 0.057444 | 0.044852 | 0.041913 | 0.045799 | 0.035408 | 0.032138 |
| 3 | 15.64 | 0.186214 | 0.074177 | 0.050469 | 0.038740 | 0.061544 | 0.046701 | 0.043187 | 0.047472 | 0.035308 | 0.032530 |
| 4 | 22.87 | 0.210642 | 0.076915 | 0.049459 | 0.035682 | 0.060031 | 0.042634 | 0.039887 | 0.044655 | 0.030929 | 0.029249 |
| 5 | 33.44 | 0.241966 | 0.077692 | 0.047262 | 0.034895 | 0.057366 | 0.037603 | 0.035535 | 0.039797 | 0.026266 | 0.024242 |
| 6 | 48.90 | 0.269184 | 0.075638 | 0.043274 | 0.036469 | 0.053853 | 0.031524 | 0.029064 | 0.034498 | 0.022299 | 0.021034 |
| 7 | 71.50 | 0.294162 | 0.066073 | 0.036976 | 0.035232 | 0.044931 | 0.023148 | 0.022656 | 0.026890 | 0.016498 | 0.015537 |
| 8 | 104.56 | 0.315761 | 0.057306 | 0.035597 | 0.035891 | 0.039291 | 0.022129 | 0.021138 | 0.023034 | 0.016983 | 0.016088 |
| 9 | 152.91 | 0.336607 | 0.049818 | 0.035377 | 0.033374 | 0.035888 | 0.018562 | 0.018686 | 0.019311 | 0.013868 | 0.013191 |
| 10 | 223.61 | 0.354754 | 0.046471 | 0.033902 | 0.031470 | 0.031980 | 0.016821 | 0.016754 | 0.015687 | 0.013554 | 0.011301 |
| 11 | 326.99 | 0.368285 | 0.043089 | 0.029887 | 0.029468 | 0.028950 | 0.015667 | 0.016102 | 0.015755 | 0.013704 | 0.011451 |
| 12 | 478.18 | 0.377458 | 0.054618 | 0.026956 | 0.032076 | 0.031155 | 0.017150 | 0.016675 | 0.016726 | 0.014883 | 0.011979 |
| 13 | 699.26 | 0.382394 | 0.077647 | 0.048333 | 0.052738 | 0.045779 | 0.035516 | 0.033349 | 0.035337 | 0.032721 | 0.028512 |
| 14 | 1022.57 | 0.385376 | 0.099720 | 0.071056 | 0.075419 | 0.059403 | 0.049700 | 0.046980 | 0.050192 | 0.048967 | 0.042704 |
| 15 | 1495.35 | 0.385947 | 0.119785 | 0.091578 | 0.097044 | 0.073485 | 0.062370 | 0.058763 | 0.064547 | 0.062220 | 0.055128 |
| 16 | 2186.72 | 0.381903 | 0.134718 | 0.107421 | 0.112933 | 0.082929 | 0.072329 | 0.065801 | 0.074623 | 0.071801 | 0.064410 |
| 17 | 3197.76 | 0.378769 | 0.148777 | 0.121147 | 0.127226 | 0.091236 | 0.080026 | 0.072522 | 0.082243 | 0.079195 | 0.072093 |
| 18 | 4676.24 | 0.371866 | 0.159678 | 0.132574 | 0.138689 | 0.097493 | 0.084702 | 0.077121 | 0.088270 | 0.084748 | 0.077739 |
| 19 | 6838.31 | 0.362541 | 0.166785 | 0.140898 | 0.147693 | 0.100944 | 0.087082 | 0.079661 | 0.091834 | 0.086839 | 0.079611 |
| 20 | 10000.00 | 0.351022 | 0.172839 | 0.147462 | 0.156397 | 0.103509 | 0.089240 | 0.081913 | 0.093931 | 0.088733 | 0.082615 |

#### Variation-level Brier error by bin

| Bin | Upper edge | Mean baseline | CNN+shared MLP | Stats+shared MLP | TCN+shared MLP | PANDA mean/std+shared MLP | PANDA attention+shared MLP | PANDA direct+shared MLP | Chronos mean/std+shared MLP | Chronos attention+shared MLP | Chronos direct+shared MLP |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 7.31 | 0.022960 | 0.043130 | 0.040178 | 0.036609 | 0.040767 | 0.037435 | 0.036088 | 0.038178 | 0.032200 | 0.029948 |
| 2 | 10.69 | 0.046872 | 0.058432 | 0.050910 | 0.041392 | 0.053661 | 0.047052 | 0.045290 | 0.047187 | 0.037287 | 0.035076 |
| 3 | 15.64 | 0.064345 | 0.062292 | 0.052211 | 0.039036 | 0.055283 | 0.047056 | 0.043571 | 0.046578 | 0.035376 | 0.033782 |
| 4 | 22.87 | 0.077373 | 0.063023 | 0.050396 | 0.036245 | 0.053270 | 0.042807 | 0.039244 | 0.042732 | 0.031602 | 0.030181 |
| 5 | 33.44 | 0.094511 | 0.060661 | 0.046143 | 0.032194 | 0.048374 | 0.035840 | 0.032702 | 0.036414 | 0.025697 | 0.024368 |
| 6 | 48.90 | 0.110087 | 0.056672 | 0.040900 | 0.030577 | 0.041412 | 0.027934 | 0.024954 | 0.029445 | 0.019976 | 0.019085 |
| 7 | 71.50 | 0.122165 | 0.050846 | 0.035798 | 0.031289 | 0.033502 | 0.019958 | 0.018084 | 0.024330 | 0.015486 | 0.015127 |
| 8 | 104.56 | 0.132993 | 0.045594 | 0.034382 | 0.032622 | 0.030522 | 0.020329 | 0.018645 | 0.020858 | 0.016116 | 0.015036 |
| 9 | 152.91 | 0.143111 | 0.041457 | 0.035595 | 0.032429 | 0.027425 | 0.018140 | 0.016573 | 0.017950 | 0.013299 | 0.012914 |
| 10 | 223.61 | 0.151981 | 0.040008 | 0.037733 | 0.032766 | 0.025934 | 0.018464 | 0.016721 | 0.017088 | 0.013960 | 0.013680 |
| 11 | 326.99 | 0.159602 | 0.040839 | 0.035066 | 0.034249 | 0.026062 | 0.019501 | 0.017688 | 0.018801 | 0.015563 | 0.015196 |
| 12 | 478.18 | 0.164550 | 0.052656 | 0.033062 | 0.036682 | 0.030457 | 0.024992 | 0.021548 | 0.023752 | 0.019671 | 0.018004 |
| 13 | 699.26 | 0.167911 | 0.071803 | 0.053423 | 0.055769 | 0.048508 | 0.044626 | 0.041949 | 0.043252 | 0.040574 | 0.039204 |
| 14 | 1022.57 | 0.170011 | 0.091068 | 0.074367 | 0.076145 | 0.065228 | 0.062109 | 0.059707 | 0.060806 | 0.058572 | 0.057841 |
| 15 | 1495.35 | 0.171634 | 0.104879 | 0.089612 | 0.091961 | 0.076389 | 0.073903 | 0.070962 | 0.072404 | 0.070324 | 0.069061 |
| 16 | 2186.72 | 0.170626 | 0.115391 | 0.101268 | 0.103881 | 0.084764 | 0.082728 | 0.078826 | 0.080767 | 0.078570 | 0.077132 |
| 17 | 3197.76 | 0.170275 | 0.121923 | 0.108405 | 0.110876 | 0.088135 | 0.085864 | 0.082271 | 0.084139 | 0.081539 | 0.080897 |
| 18 | 4676.24 | 0.167241 | 0.126550 | 0.112877 | 0.116575 | 0.091081 | 0.088894 | 0.085458 | 0.086579 | 0.083497 | 0.082668 |
| 19 | 6838.31 | 0.163356 | 0.128483 | 0.115311 | 0.119294 | 0.090649 | 0.088668 | 0.085255 | 0.085621 | 0.082182 | 0.082001 |
| 20 | 10000.00 | 0.158993 | 0.128962 | 0.115480 | 0.120645 | 0.088953 | 0.086405 | 0.083317 | 0.083704 | 0.079959 | 0.080797 |

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

Sources: [threshold=10,000 metrics](results/task_d/sampling_2/task_d_proxy_sensitivity/threshold_10000/aggregation_min/metrics.json) and [Task D protocol documentation](readme.md#task-d-lyapunov-proxy-regression).

## Task E: Long-horizon orbital forecasting

### Protocol

The latest unified benchmark uses `orbital15`, a context of 512 steps, and a one-shot prediction length of 128 steps. E1 is one-shot forecasting; E2 recursively rolls out four 128-step blocks for a total horizon of 512. Results use system-balanced aggregation through the hierarchy `system -> trajectory -> forecasting window`.

The test set contains 578 systems and 9,648 trajectories. It contains 38,245 E1 test windows and 17,485 E2 rolling windows.

### E1: one-shot 512 -> 128 forecasting

MSE, MAE, and RMSE are reported on the standardized z-scale. Unit-circle error evaluates the orbital phase representation.

| Model | MSE (z) | MAE (z) | RMSE (z) | Pearson | Unit-circle error |
| --- | ---: | ---: | ---: | ---: | ---: |
| Chronos2 | 0.2331 | **0.2282** | 0.4828 | 0.8580 | **0.2265** |
| PANDA zero-shot | 0.2217 | 0.2371 | 0.4709 | 0.8625 | 0.2862 |
| iTransformer | **0.2123** | 0.2429 | **0.4608** | **0.8673** | 0.3152 |
| TOTO2 | 0.2262 | 0.2303 | 0.4757 | 0.8601 | 0.2310 |
| TiRex2 | 0.2422 | 0.2408 | 0.4921 | 0.8508 | 0.2277 |

iTransformer obtains the best MSE, RMSE, and Pearson correlation. Chronos2 obtains the best MAE and unit-circle error.

### E2: recursive 512-step rollout

| Model | Block 1 MAE | Block 2 MAE | Block 3 MAE | Block 4 MAE | Error amplification |
| --- | ---: | ---: | ---: | ---: | ---: |
| Chronos2 | **0.2201** | **0.2895** | **0.3179** | **0.3397** | 1.5436 |
| PANDA zero-shot | 0.2267 | 0.3033 | 0.3260 | 0.3416 | 1.5065 |
| iTransformer | 0.2380 | 0.2993 | 0.3294 | 0.3515 | **1.4772** |
| TOTO2 | 0.2234 | 0.2957 | 0.3250 | 0.3483 | 1.5592 |
| TiRex2 | 0.2352 | 0.3039 | 0.3310 | 0.3507 | 1.4909 |

All models accumulate error during recursive rollout. Chronos2 has the lowest absolute MAE in every rollout block, while iTransformer has the lowest relative amplification.

Sources: [Chronos2](results/task_e/sampling_2/task_e_chronos2_orbital15_ctx512_pred128/summary.md), [PANDA zero-shot](results/task_e/sampling_2/task_e_panda_zero_shot_orbital15_ctx512_pred128/summary.md), [iTransformer](results/task_e/sampling_2/task_e_itransformer_orbital15_ctx512_pred128/summary.md), [TOTO2](results/task_e/sampling_2/task_e_toto2_orbital15_ctx512_pred128/summary.md), and [TiRex2](results/task_e/sampling_2/task_e_tirex2_orbital15_ctx512_pred128/summary.md).

## Overall takeaways

- **Task A:** On the current shared split, the Ghost pretrained MLP has the best binary
  AUPRC and, after validation-only threshold selection, the best accuracy, balanced
  accuracy, F1, and MCC. The standard pretrained MLP retains the best binary AUROC and
  stable-fraction correlations. Ghost reduces its probability bias and stable-fraction
  error, but XGBoost-base still has the best stable-fraction MSE/R².
  Equation (4) consistently improves on Equation (2), while both analytic criteria
  remain weaker than the pretrained and locally trained models.
- **Task B:** On the shared case-level `70/15/15` split, dynamical summaries plus
  XGBoost are strongest at both trajectory and aggregated-system levels. At prefix=96
  they achieve trajectory AUROC=0.9649 and system R²=0.9040; at prefix=512 these improve
  to AUROC=0.9851 and system R²=0.9676. Among frozen linear probes, Chronos-2 has
  slightly higher AUROC than Panda, while Panda has lower system-level MAE and MSE.
- **Task C:** On the latest prefix=96 common split, Chronos-2 direct frozen embeddings lead overall curve MAE, legacy-horizon MAE, and IBS; PANDA direct has the best final-bin MAE, while raw TCN is the strongest raw-feature baseline.
- **Task D:** The MLP predicts the Lyapunov-like proxy substantially better than the constant and random-forest baselines. The target remains a discrete observational proxy rather than a strict Lyapunov time.
- **Task E:** Foundation time-series models are competitive for one-shot forecasting, but recursive rollout still shows clear error accumulation over the 512-step horizon.
