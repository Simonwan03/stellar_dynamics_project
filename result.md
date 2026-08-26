# Latest Results for Tasks A--E

This document consolidates the latest saved experimental results currently available in the repository. No experiments were rerun while preparing this summary. Unless stated otherwise, the reported metrics are test-set metrics.

## Common data context

- The main dataset contains 4,998 complete valid systems, with 32 variations per system and 159,936 complete trajectories.
- The default initial-condition representation is `orbital15` (15 orbital channels per time step).
- The current code uses one shared `raw_case_id` manifest for Tasks A--C: `3498/750/750`
  complete systems under the 70/15/15 protocol. The saved metrics below predate that
  protocol unless explicitly stated and must be rerun before cross-task comparison.
- Metrics from different tasks use different targets and protocols and should not be compared numerically across tasks.

## Task A: Stable-fraction and binary stability prediction

### Protocol

The latest saved expanded experiment uses `seed=42`, an older `70/10/20` system split,
`orbital15` initial conditions, and a maximum horizon of 10,000. Two targets are
evaluated. It is retained as a historical result; new runs should use the shared
`70/15/15` manifest described in `readme.md`.

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

Task B predicts the binary stability state of an individual trajectory from an early prefix.
The historical runs later in this section use the older `test_size=0.125`, `val_size=0.1`
configuration. The paper results immediately below use the shared `70/15/15` manifest,
with the canonical 4,998-system universe and 159,936 complete trajectories. The canonical
Task B target is the stable-positive label, `time_unbound >= 10000`; the new comparison
reports the inverse `unbound_positive` label, `time_unbound < 10000`, because it directly
measures detection of instability.

The encoder-based results use frozen `GilpinLab/panda-72M` and Chronos-2 encoders. Panda uses `patch_mean_keep_channels` and produces a `15 x 768` representation; Chronos-2 uses its frozen `[REG]` representation with the same feature dimension. The raw `orbital15` SVM/CNN/TCN/GRU/Attention experiments below are complementary temporal baselines and prefix ablations.

### Latest paper results: shared split, matched linear probes

**These are the latest Task B results intended for the paper.** Panda and Chronos-2 use
the same 4,998-system universe, case-level `3498/750/750` train/validation/test split,
input rows, target labels, 11,521-parameter linear probe, optimizer settings, and seed.
The test set contains 24,000 trajectories from 750 systems. Train, validation, and test
have no overlapping `raw_case_id` values.

| Prefix | Encoder | Test accuracy | AUROC | AUPRC | Late-unbound recall |
| ---: | --- | ---: | ---: | ---: | ---: |
| 96 | Panda | **0.8640** | 0.9467 | 0.9751 | **0.6334** |
| 96 | Chronos-2 | 0.8623 | **0.9531** | **0.9781** | 0.6139 |
| 512 | Panda | **0.9217** | 0.9786 | 0.9902 | **0.4517** |
| 512 | Chronos-2 | 0.9209 | **0.9791** | **0.9905** | 0.4006 |

At prefix=96, Chronos-2 has the stronger ranking metrics, while Panda has 1.95 percentage
points higher recall for trajectories that become unbound only after the prefix. At
prefix=512 the overall ranking metrics are effectively tied; Panda has 5.11 percentage
points higher late-unbound recall, whereas Chronos-2 correctly retains 1.24 percentage
points more trajectories that remain stable through the final horizon. Thus neither
encoder is uniformly superior at the default probability threshold of 0.5.

The overall metrics include trajectories that have already become unbound within the
observed prefix: 9,776 test trajectories at prefix=96 and 13,807 at prefix=512. Both
encoders detect virtually all of these cases. Genuine forward prediction is evaluated
separately in the trajectory-level at-risk ablation below, including a matched
`T>2560` cohort that compares 96- and 512-step inputs on exactly the same trajectories.

Sources: [Panda prefix=96 metrics](experiments/panda_threebody/outputs/panda_unbound_dataset/task_b_common/panda_prefix96_linear/metrics.json), [Chronos-2 prefix=96 metrics](experiments/panda_threebody/outputs/panda_unbound_dataset/task_b_common/chronos2_prefix96_linear/metrics.json), [Panda prefix=512 metrics](experiments/panda_threebody/outputs/panda_unbound_dataset/task_b_common/panda_prefix512_linear/metrics.json), and [Chronos-2 prefix=512 metrics](experiments/panda_threebody/outputs/panda_unbound_dataset/task_b_common/chronos2_prefix512_linear/metrics.json).

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
[the at-risk ablation directory](results/task_ab_future_only_ablation/), and the analysis
is reproduced by
[evaluate_task_ab_future_only_ablation.py](scripts/evaluate_task_ab_future_only_ablation.py).

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

Machine-readable results and ready-to-use tables are available as
[CSV](results/task_ab_future_only_ablation/trajectory_at_risk_metrics.csv),
[Markdown](results/task_ab_future_only_ablation/trajectory_at_risk_table.md), and
[LaTeX](results/task_ab_future_only_ablation/trajectory_at_risk_table.tex). The analysis
is reproduced by
[evaluate_task_ab_future_only_ablation.py](scripts/evaluate_task_ab_future_only_ablation.py).

The two-level machine-readable aggregation is
[task_b_two_level_metrics.csv](results/task_b_common/two_level_evaluation/task_b_two_level_metrics.csv),
with matching [LaTeX](results/task_b_common/two_level_evaluation/task_b_two_level_table.tex)
and [Markdown](results/task_b_common/two_level_evaluation/task_b_two_level_table.md) tables.
Each model's raw metrics and test probabilities are stored under
[the common Task B output directory](experiments/panda_threebody/outputs/panda_unbound_dataset/task_b_common/).

### Historical Panda-encoder results: prefix = 512

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

### Historical prefix = 512 comparison beyond the Panda heads

The earlier prefix=512 physics run uses the same 120-feature summary representation with
XGBoost, but predates the shared paper split. A matched prefix=512 Chronos-2 result is now
reported in the latest-paper-results subsection above. The separate Chronos-2 context=512
results in Task E are forecasting results and are not interchangeable with Task B.

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
baseline by overall MAE and IBS. Sources: [latest raw common-split metrics](results/task_c_survival20_common_prefix96_raw/metrics.json), [latest PANDA common-split metrics](results/task_c_survival20_common_prefix96_panda/metrics.json), [latest Chronos-2 common-split metrics](results/task_c_survival20_common_prefix96_chronos/metrics.json), and [shared split manifest](dataset_v3/common_case_split_sampling_2_g32_h10000_r70-15-15_seed42.csv).

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

### Historical 20% test split (not used for the latest comparison)

The following saved runs use the previous 3,498/500/1,000 system split and should not be
mixed with the common-split table above.

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

Sources: [Chronos2](runs/task_e_chronos2_orbital15_ctx512_pred128/summary.md), [PANDA zero-shot](runs/task_e_panda_zero_shot_orbital15_ctx512_pred128/summary.md), [iTransformer](runs/task_e_itransformer_orbital15_ctx512_pred128/summary.md), [TOTO2](runs/task_e_toto2_orbital15_ctx512_pred128/summary.md), and [TiRex2](runs/task_e_tirex2_orbital15_ctx512_pred128/summary.md).

## Overall takeaways

- **Task A:** XGBoost-strong is the strongest regression and ranking model; MLP-base is best when the binary decision is evaluated by F1.
- **Task B:** On the shared case-level `70/15/15` split, dynamical summaries plus
  XGBoost are strongest at both trajectory and aggregated-system levels. At prefix=96
  they achieve trajectory AUROC=0.9649 and system R²=0.9040; at prefix=512 these improve
  to AUROC=0.9851 and system R²=0.9676. Among frozen linear probes, Chronos-2 has
  slightly higher AUROC than Panda, while Panda has lower system-level MAE and MSE.
- **Task C:** On the latest prefix=96 common split, Chronos-2 direct frozen embeddings lead overall curve MAE, legacy-horizon MAE, and IBS; PANDA direct has the best final-bin MAE, while raw TCN is the strongest raw-feature baseline.
- **Task D:** The MLP predicts the Lyapunov-like proxy substantially better than the constant and random-forest baselines. The target remains a discrete observational proxy rather than a strict Lyapunov time.
- **Task E:** Foundation time-series models are competitive for one-shot forecasting, but recursive rollout still shows clear error accumulation over the 512-step horizon.
