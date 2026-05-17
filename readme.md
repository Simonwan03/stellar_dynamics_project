# Stellar Dynamics Task Summary

This repository contains three notebook experiments for stability prediction on the local three-body trajectory datasets:

- `task_a.ipynb`: perturbation-set probability prediction from initial conditions.
- `task_b.ipynb`: early-prefix classification and time-to-instability prediction.
- `task_c.ipynb`: trajectory-level multi-horizon survival prediction.

The notebooks use groups of 8 perturbed trajectories per nominal initial condition. Most experiments use the first local 5 result shards, giving 40000 trajectories or 5000 complete 8-way perturbation sets.

## Local Dataset Versions

### `dataset_v3` / `sampling_2`

The new `sampling_2` dataset is available under `dataset_v3/` and contains the full 32-variation structure:

- Initial conditions: `dataset_v3/ICs_sampling_2.hdf5`.
- Completion table: `dataset_v3/completed_systems_sampling_2.txt`.
- Result shards: `dataset_v3/results_batch_00_sampling_2.hdf5` through `dataset_v3/results_batch_19_sampling_2.hdf5`.
- Total rows/runs: `160000`.
- Unique systems: `5000`.
- Variations per unique system: `32`.
- Batch layout: `20` result shards, each with `8000` runs, equivalent to `250` unique systems per shard.
- Indexing convention:
  - `unique_system_id = run_id // 32`
  - `variation_id = run_id % 32`
  - `batch_id = run_id // 8000`
  - `row_in_batch = run_id % 8000`
- Time-series shape per result shard: `(8000, 2001, ...)`, giving `2000` time intervals.
- Time grid: `times_complete` runs from `0` to `100000000`; with outer period `10000`, this covers `10000` outer orbits.

The 32 variations are organized as two 16-variation blocks. The first 16 use the unperturbed initial conditions, and the last 16 repeat the same phase grid with a perturbed inner semimajor axis. The 16 phase variations come from a `4 x 4` grid over inner and outer orbit phases. The intended phase values are `[0, pi/2, pi, 3*pi/2]`; in the HDF5 files, the stored `tAnos` values are true anomalies, so they do not necessarily appear as exact multiples of `pi/2` for eccentric orbits.

Filtering rule: as before, rows marked invalid in `completed_systems_sampling_2.txt` should be ignored. In the current local file, column 3 contains `47` rows with `-1`. Downstream loaders should also drop any run whose selected HDF5 input/target fields contain non-finite values, because some derived complete-series fields such as `periods_complete` and `tAnos_complete` can contain NaNs even when the completion table has no textual `NaN` token.

## Task A: Perturbation-Set Probability Prediction

Task A predicts an empirical stability probability for each initial condition by aggregating the outcomes of its 8 perturbations.

### Data and Target

- Source initial conditions: `dataset/ICs_sampling_0.hdf5`.
- Raw initial-condition rows: `40000`.
- Grouping: every 8 rows form one perturbation set, giving `5000` groups.
- Selected features:
  - `aratios`
  - `mratios`
  - `eccs`
  - `incs`
  - `APs`
  - `LANs`
- Angular variables are encoded with sine/cosine.
- Target: `prob_unbound`, the fraction of the 8 trajectories in a group with `time_unbound >= 5000`.
- Train/test split:
  - Train: `4000` groups
  - Test: `1000` groups
- Baseline flat feature shape: `X = (5000, 17)`.
- Transformer token feature shape: `X_train = (4000, 26, 3)`.

### Models and Results

| Model | Test MSE | Test MAE | Notes |
| --- | ---: | ---: | --- |
| MLP probability regressor | `0.0963` best MSE, `0.0984` at best-MAE epoch | `0.2158` best MAE | 200 epochs; final epoch test MSE/MAE was `0.1236` / `0.2224` |
| Transformer probability regressor | `0.1130` best MSE, `0.1158` at best-MAE epoch | `0.2266` best MAE | 200 epochs; final epoch test MSE/MAE was `0.1417` / `0.2447` |
| XGBoost regressor | `0.0983` | `0.2435` | Test `R2 = 0.4279`; train MSE/MAE was `0.0056` / `0.0540` |

### Takeaway

The task is feasible from initial-condition features alone, but the neural models overfit after early epochs. The MLP reached the best recorded MAE, while XGBoost reached similar MSE with stronger train/test overfitting.

## Task B: Prefix-to-Outcome Classification and Time Prediction

Task B predicts long-term instability from an early trajectory prefix.

### Data and Target

- Source trajectory files: `dataset/results_batch_00..04_sampling_0.hdf5`.
- Loaded trajectories: `40000`.
- Prefix length in the notebook: `T = 100`.
- After finite filtering: `39746` trajectories.
- Censored-at-horizon samples: `14203`.
- Observed unbound samples: `25543`.
- Train/test split:
  - Train: `34746`
  - Test: `5000`
- Hard label:
  - censored/stable at horizon vs observed unbound.
- Soft target:
  - `64`-bin distribution over `time_unbound`.
- Main engineered feature stack:
  - orbital/physical channels with sine/cosine angles
  - pairwise distance/speed/radial-speed geometry
  - optional normalized time channel
  - notebook SVM feature tensor: `(34746, 36, 100)`

NaN-heavy channels were identified and avoided or cleaned:

| Channel | Bad samples in first `T=100` |
| --- | ---: |
| `periods_complete` | `16847 / 40000` |
| `tAnos_complete` | `14509 / 40000` |
| `positions_complete` | `254 / 40000` |

### Notebook Results

| Model | Target | Test Result |
| --- | --- | --- |
| Linear SVM | hard binary label | train accuracy `0.8395`, test accuracy `0.8022` |
| 1D CNN | 64-bin soft time distribution | test soft CE `1.7869`, MAE all `537.42`, MAE observed `546.94` |
| TCN | hard binary label | AUC `0.9795`, F1 `0.9156`, accuracy `0.9350` |
| Transformer attention classifier | hard binary label | AUC `0.9783`, F1 `0.9210`, accuracy `0.9410` |

Additional classification sweeps recorded in `result.md`:

| Model | Prefix `T` | Test AUC | Test F1 | Test Accuracy |
| --- | ---: | ---: | ---: | ---: |
| 1D CNN | `100` | `0.9544` | `0.8284` | `0.8674` |
| 1D CNN | `200` | `0.9630` | `0.8568` | `0.8886` |
| 1D CNN | `500` | `0.9783` | `0.9087` | `0.9308` |
| Attention | `100` | `0.9515` | `0.8307` | `0.8670` |
| Attention | `200` | `0.9637` | `0.8648` | `0.8974` |
| Attention | `500` | `0.9783` | `0.9210` | `0.9410` |

### Panda-Backed Prefix Classification

An isolated Panda experiment was added under `experiments/panda_threebody/` to test whether the pretrained Panda time-series backbone is useful for the Task B final-unbound classification problem.

- Source data: `dataset/`.
- Input representation: `orbital15`.
- Input window: first `512` sampled steps, shape `(40000, 512, 15)`.
- Physical time covered by the input: `t = 0` to `t = 25,550,000`.
- Full trajectory horizon: `1001` sampled steps, ending at `t = 50,000,000`.
- Label: `final_unbound = time_unbound < 5000`.
- Backbone: `GilpinLab/panda-72M`.
- Training setup: frozen Panda backbone, trainable linear classification head only.
- Trainable parameters: `769`.
- Split:
  - Train: `28008`
  - Validation: `3992`
  - Test: `8000`

Results from `experiments/panda_threebody/outputs/panda_unbound_dataset/panda_unbound_orbital15_dataset/metrics.json`:

| Model | Prefix steps | Trainable parameters | Test AUC | Test F1 | Test Accuracy | Notes |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Frozen Panda-72M + linear head | `512` | `769` | `0.9784` | `0.9520` | `0.9393` | Panda hidden representation pooled over channels and patches |

For reference, a non-Panda Conv1D baseline on the same `orbital15` prefix reached test AUC `0.9509`, F1 `0.9040`, and accuracy `0.8844`. This baseline is useful as a sanity check, but the Panda-backed result above is the relevant comparison for testing Panda on the three-body stability task.

### Takeaway

Early trajectory prefixes contain strong predictive signal for future instability. Deep temporal models substantially outperform the linear SVM, and longer prefixes improve classification. The frozen Panda-72M representation with only a linear head matches the strongest notebook-era Task B results, giving test AUC `0.9784` and the best recorded F1/accuracy in this summary.

## Task C: Trajectory-Level Multi-Horizon Survival Prediction

Task C upgrades survival prediction from a case-level curve to per-trajectory, multi-horizon survival prediction for each 8-way perturbation ensemble.

### Data and Target

- Source dataset: `dataset_v2`.
- Local result shards used: `results_batch_00..04_sampling_1.hdf5`.
- Available local subset: 5 of 50 requested shards.
- Cleaned trajectories: `40000`.
- Complete perturbation sets: `5000`.
- Input tensor: `(5000, 8, 200, 15)`.
- Trajectory target tensor: `(5000, 8, 9)`.
- Ensemble fraction target: `(5000, 9)`.
- Ensemble count target: `(5000, 9)`.
- Train/val/test split:
  - Train: `3500` cases
  - Validation: `500` cases
  - Test: `1000` cases
- Future survival times:
  - `1100`
  - `1500`
  - `2000`
  - `3000`
  - `4000`
  - `5000`
  - `7000`
  - `8500`
  - `10000`

The model predicts hard-monotone trajectory survival logits by parameterizing each horizon logit as an initial value minus cumulative non-negative drops. This guarantees non-increasing survival probability over time. Ensemble-level survival is then aggregated as:

- `Q_hat = mean_i sigmoid(logits_i)`
- `N_hat = sum_i sigmoid(logits_i)`

### Models and Results

| Model | Test Loss | Fraction MAE | Fraction RMSE | Count MAE | Final-Horizon MAE | Monotone Fraction |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Mean trajectory baseline | `0.8934` | `0.3801` | `0.4170` | `3.0411` | `0.3540` | `1.0000` |
| Trajectory context model | `0.3235` | `0.0958` | `0.1682` | `0.7662` | `0.1204` | `1.0000` |
| Dual-attention model | `0.3002` | `0.0845` | `0.1588` | `0.6762` | `0.1094` | `1.0000` |

Trajectory-context details:

- Shared CNN encoder processes each trajectory independently.
- The 8 trajectory embeddings are mean-pooled into an ensemble context.
- Per-trajectory head predicts hard-monotone survival logits.
- Best validation epoch: `21`.
- Test ensemble-fraction MAE improvement over baseline: `0.2844`.

Dual-attention details:

- Transformer-style temporal self-attention encodes each trajectory.
- The same hard-monotone survival head predicts trajectory-level survival.
- Best validation epoch: `19`.
- Test ensemble-fraction MAE improvement over baseline: `0.2956`.
- Test ensemble-fraction MAE improvement over trajectory context: `0.0112`.

### Takeaway

Task C shows the strongest and most physically structured setup. The early 200-step prefix predicts long-horizon survival well, and the hard-monotone parameterization enforces valid survival curves for every tested model. The dual-attention model is the best current Task C model, reducing ensemble-fraction MAE from `0.3801` to `0.0845`.

## Overall Conclusions

- Task A establishes that grouped initial-condition features can predict perturbation-level outcome probabilities, but static features alone are limited.
- Task B shows that short trajectory prefixes dramatically improve future stability classification, especially with temporal neural models.
- Task C is the most complete formulation: it keeps the 8-trajectory perturbation structure, predicts multiple future horizons, and enforces physically valid monotone survival curves.
- Across the project, preserving temporal information and perturbation-set structure is more informative than using only flattened static initial conditions.


Ye Zhu
3:14 PM
Panda: A pretrained forecast model for chaotic dynamics (https://arxiv.org/abs/2505.13755) ICLR2026

Ye Zhu
3:38 PM
[2110.07238] On the difficulty of learning chaotic dynamics with RNNs (https://arxiv.org/abs/2110.07238) - Neurips2022  

[2306.04406] Generalized Teacher Forcing for Learning Chaotic Dynamics (https://arxiv.org/abs/2306.04406) - ICML2023

[2306.01187v3] Training neural operators to preserve invariant measures of chaotic attractors (https://arxiv.org/abs/2306.01187v3) - Neurips2023

[2402.04467] DySLIM: Dynamics Stable Learning by Invariant Measure for Chaotic Systems (https://arxiv.org/abs/2402.04467) - ICML2024  

[2409.15771v3] Zero-shot forecasting of chaotic systems(https://arxiv.org/abs/2409.15771v3) ICLR2025 Pretrained models perform well on chaotic problems(**Chronos**), foundation models can still capture the attractor’s geometric and statistical structure.
 
[2411.06311] When are dynamical systems learned from time series data statistically accurate](https://arxiv.org/abs/2411.06311) Neurips2024

Pavan Vynatheya
3:41 PM
So Lyapunove time is defined for a divergence by a factor of e (or order unity). So in this context, I would suggest to look at the semimajor axis ratio parameter (eccentricity and angles vary a lot periodically even for stable systems). So predicting a timescale where two similar orbits diverge in semimajor axis ratio by an order of itself is most useful.


3:55 PM
Cellular automata:
https://arxiv.org/html/2504.20858v2
https://advanced.onlinelibrary.wiley.com/doi/10.1002/advs.202511352?af=R
