# Task Packages

Each task package contains its reusable implementation and command-line entry point.
Run them from the repository root. On the current machine the examples use the `stellar`
conda environment; on AWS, use whichever environment contains `h5py`, `numpy`, `pandas`,
`scikit-learn`, `matplotlib`, and `torch`.

## Layout

```text
tasks/
├── common/   # code shared by multiple tasks
├── task_a/   # stability classification
├── task_b/   # prefix classification
├── task_c/   # survival prediction
├── task_d/   # divergence-time regression
├── task_e/   # trajectory forecasting
└── task_g/   # boundary sensitivity
```

Every task can be launched as a package, for example
`python -m tasks.task_e --help`. Task-specific implementation stays inside its
package; imports between tasks are explicit. General visualization utilities
live in `tools/`.

Task E defaults to group-aware splitting (`32` variations per base system),
completion-table filtering, and scaler statistics computed only from valid
pre-event training windows. Adapter checkpoints exclude frozen pretrained
backbones and record the backbone model ID in `config.json`.

## Task A

Group-level stable-at-horizon modeling from tokenized initial conditions.
By default this predicts the empirical stable fraction across the 32 perturbations
for each system, matching the original Task A distribution-label objective.

Set `--target-mode binary_any` to make this a system-level binary classifier:
the label is `1` only when all 32 variations survive to `--time-horizon`; if any
variation is unbound before that time, the label is `0`.

```bash
conda run -n stellar python -m tasks.task_a \
  --output-dir results/task_a \
  --models equation2,equation4,mlp,xgboost \
  --epochs 80
```

Binary all-stable classification:

```bash
conda run -n stellar python -m tasks.task_a \
  --output-dir results/task_a_binary_any \
  --target-mode binary_any \
  --time-horizon 10000 \
  --models equation2,equation4,mlp,xgboost \
  --epochs 50 \
  --device cuda
```

For `binary_any`, MLP defaults to `BCEWithLogitsLoss` and XGBoost uses a
`binary:logistic` objective. The output metrics include accuracy, balanced
accuracy, precision, recall, F1, specificity, FPR/FNR, MCC, AUROC, average
precision, Brier score, log loss, confusion-matrix counts, and 10-bin ECE/MCE.
For `stable_fraction`, regression metrics additionally include median absolute
error, explained variance, Pearson/Spearman correlation, bias, tolerance
accuracies, target-bin breakdowns, and the MSE relative to the 32-trial
binomial sampling-noise estimate.

The non-trainable `equation2` baseline is the Mardling-Aarseth criterion;
`equation4` implements Vynatheya et al. (2022), Equations (3)-(4), including
the inner-eccentricity and nonlinear mutual-inclination corrections. For both
criteria, `binary_any` requires all variation margins to be positive, while
`stable_fraction` returns the fraction of variations classified stable.

The optional `pretrained_mlp` baseline uses the official Vynatheya et al. 2022
`mlp_model_trip_v1.2.2.pkl`. It takes
`[q_in,q_out,a_in/a_out,e_in,e_out,i_mut/pi]`; class 1 is unstable. For
`binary_any`, Task A uses the minimum stable probability across variations at
the upstream threshold 0.5 and also reports a validation-F1 operating point
without changing model weights. For `stable_fraction`, it averages stable
probabilities. Download the external artifact before selecting this model:

```bash
mkdir -p external_checkpoints/triple_stability
curl -L \
  https://raw.githubusercontent.com/pavanvyn/triple-stability/main/mlp_model_trip_v1.2.2.pkl \
  -o external_checkpoints/triple_stability/mlp_model_trip_v1.2.2.pkl
```

The adapter verifies SHA-256
`f8bc2f192813145f42683669adaef6448a287523875895343453055b3acf1436`
before loading the pickle. The artifact is intentionally not tracked by Git.

The optional `pretrained_ghost_mlp` baseline has the same input and aggregation
interface, but uses the official ghost-orbit divergence labels. Download it with:

```bash
curl -L \
  https://raw.githubusercontent.com/pavanvyn/triple-stability/main/mlp_model_trip_ghost_v1.2.2.pkl \
  -o external_checkpoints/triple_stability/mlp_model_trip_ghost_v1.2.2.pkl
```

Its verified SHA-256 is
`a260845fc9c5b80ede3ab0028ac582bffa8450dad0c23f3ce78d73987932b866`.

MLP presets:
- `--mlp-size small`: hidden `(64,64,32)`, dropout `0.1`
- `--mlp-size base`: hidden `(128,128,64)`, dropout `0.2`
- `--mlp-size large`: hidden `(256,256,128)`, dropout `0.2`

MLP loss modes:
- `--mlp-loss mse`: `Sigmoid + MSELoss`
- `--mlp-loss bce`: `Linear + BCEWithLogitsLoss`; useful for `binary_any`, and also supports soft `stable_fraction` labels

MLP early stopping:
- default: `--early-stopping-patience 15 --early-stopping-min-delta 1e-4`
- set `--early-stopping-patience 0` to disable and always run the full `--epochs`

XGBoost presets:
- `--xgb-size small`: `300` trees, depth `2`
- `--xgb-size base`: `1000` trees, depth `3`
- `--xgb-size strong`: `2000` trees, depth `4`

Use `--regroup --overwrite-tokenized` if you need to rebuild
`dataset_v3/ICs_sampling_2_groupof32.hdf5`.

## Task B

Hard-label classification from early trajectory prefixes.

```bash
conda run -n stellar python -m tasks.task_b \
  --output-dir results/task_b \
  --representation orbital15 \
  --prefix-len 100 \
  --models svm,cnn,gru,tcn,attention \
  --epochs 50
```

To predict the stable fraction across each 32-trajectory perturbation group
from early prefixes:

```bash
conda run -n stellar python -m tasks.task_b \
  --output-dir results/task_b_group_fraction \
  --representation orbital15 \
  --target-mode group_fraction \
  --group-aggregation set_attention \
  --prefix-len 100 \
  --models group_attention \
  --epochs 50
```

Use `--group-aggregation concat --models svm,cnn,gru,tcn,attention` for the simpler
baseline that flattens the 32 variations into the feature channel dimension.

`Task B` now defaults to the `orbital15` sequence representation:
`aratios_complete(1) + mratios_complete(2) + eccs_complete(2) + incs_complete(2) + sin/cos(APs_complete)(4) + sin/cos(LANs_complete)(4) = 15` channels per time step.
The default label horizon is `--time-horizon 10000`.
Use `--representation full` to restore the older wider stack with extra orbital channels plus optional geometry/time features.

The temporal models now auto-scale with the effective number of input steps `T`:
- `SVM`: params follow `T * 15`
- `GRU`: two-layer unidirectional GRU; its hidden width is auto-scaled with `T`
- `T=100`: `cnn ~92K`, `gru ~155K`, `tcn ~355K`, `attention ~399K`
- `T=200`: `cnn ~141K`, `gru ~155K`, `tcn ~454K`, `attention ~597K`
- `T=500`: `cnn ~200K`, `gru ~240K`, `tcn ~706K`, `attention ~931K`

For the default 15-channel input, the automatic GRU scale is approximately
`155K` parameters for `T <= 200` (`hidden=128, layers=2`) and `240K` for
longer prefixes (`hidden=160, layers=2`). Use `--gru-layers` or
`--hidden-dim` to override its size.

If you set `--hidden-dim`, that hidden width is used directly instead of the automatic per-`T` scale. You can also override `--tcn-levels`, `--attention-layers`, and `--attention-heads`.
Early stopping defaults to `--early-stopping-patience 8`; set it to `0` to disable.

For a quick AWS smoke test:

```bash
conda run -n stellar python -m tasks.task_b \
  --output-dir /tmp/task_b_smoke \
  --max-shards 1 \
  --rows-per-shard 512 \
  --prefix-len 32 \
  --time-stride 4 \
  --models svm,cnn \
  --epochs 1
```

## Task C

Trajectory-level multi-horizon survival prediction for 32-variation ensembles.

```bash
conda run -n stellar python -m tasks.task_c \
  --output-dir results/task_c \
  --prefix-len 200 \
  --time-stride 1 \
  --models baseline,direct,direct_mean_std,traj,dual \
  --epochs 40
```

`direct` is an aggregation-before-encoding weak baseline that averages the 32
variation trajectories before predicting the system survival curve.
`direct_mean_std` is a stronger system-level baseline that concatenates the
per-time-step mean, standard deviation, minimum, and maximum over the 32
variations before applying the same direct curve model and loss.

All Task C neural models use a discrete-time hazard head. The network predicts
conditional failure hazards for the future intervals, and the survival curve is
the cumulative product of one minus those hazards. `traj` and `dual` now have
two heads: a variation-level head and a direct system-level head. Their joint
objective combines the variation-level masked hazard negative log-likelihood,
the system-level binomial hazard negative log-likelihood, and a consistency
penalty between the system head and the mean variation-head survival curve.
`direct` and `direct_mean_std` continue to optimize the interval-count binomial
hazard negative log-likelihood from the aggregated 32-variation survival
fractions.

### Event-time/censoring upgrade (20 bins)

For the event-time formulation use the separate entry point:

```bash
conda run -n stellar python -m tasks.task_c.survival \
  --prefix-cache results/task_c_ablation/_cache \
  --prefix-len 100 \
  --prefix-representation orbital15 \
  --models baseline,stats,cnn \
  --hazard-bins 20 \
  --hazard-t-max 10000 \
  --variation-target full32 \
  --epochs 40 \
  --device cuda
```

This path constructs logarithmic time edges and uses each raw
`time_unbound` as an event time. `time_unbound == 10000` (and values beyond
the selected upper edge) are right-censored: the target is zero and the
at-risk mask remains one in every bin. For an observed event, the masked BCE
target is zero before the event bin, one in the event bin, and masked after
it. The model predicts 20 conditional hazards per variation, obtains
survival by `cumprod(1 - sigmoid(logits))`, and trains with
`L_variation_NLL + 0.2 * MAE(mean_variation_survival, system_survival_target)`.
The default `--hazard-t-min` is inferred from the earliest positive event in
the valid data (5 for sampling_2); pass it explicitly when the prefix defines
a stricter observation boundary. Outputs contain 20-bin MAE/RMSE,
variation NLL, Brier/IBS, monotonicity checks, and a readout at the historical
seven horizons under `old_*` metrics. The original `python -m tasks.task_c`
entry point remains unchanged for direct comparison with the old seven-
fraction training objective.

The raw-trajectory `stats` encoder reuses the Task B SPoCK-inspired summary
definition for every `orbital15` channel: `initial`, `final`, `mean`, `std`,
`slope`, `min`, `max`, and `delta`. This gives `15 x 8 = 120` features per
variation in Task C, with the same channel-major ordering as the Task B
XGBoost baseline. Task C still applies train-only trajectory standardisation
before the summaries are computed.

Precomputed frozen Panda/Chronos embeddings can use the same target, loss, and
hazard head without running the backbone again. In addition to `embedding_direct`
and `embedding_meanstd`, `embedding_attention` learns a softmax weight over the
15 channel embeddings (`[15,768]`) and feeds their weighted sum to the hazard
head:

```bash
conda run -n stellar python -m tasks.task_c.survival \
  --embedding-cache experiments/panda_threebody/outputs/panda_unbound_dataset/panda_task_b_prefix_96_features_aligned.float16.npy \
  --embedding-labels results/task_b_prefix_ablation/prefix_100/labels_filtered.csv \
  --prefix-len 96 \
  --models baseline,embedding_direct,embedding_meanstd,embedding_attention \
  --hazard-bins 20 --epochs 10 --device cuda \
  --output-dir results/task_c_survival20_panda
```

Use `--models baseline,embedding_attention` to run this pooling variant. It
does not unfreeze or rerun either frozen backbone.

`embedding_direct` feeds the flattened `15 x 768 = 11520` representation
directly into the first linear layer of `HazardHead`; `embedding_meanstd`
reshapes it to `[15,768]`, pools mean and standard deviation, and uses 1536
input features. The cache loader verifies the label fingerprint before grouping
rows into complete `[system,32,embedding]` tensors.

For the initial-condition control experiment, use `--input-mode initial
--models baseline,ic_mlp`. It reads one `orbital15` vector at `t=0` per system
and predicts the same seven-horizon survival curve with a `15 -> 128 -> 128 ->
64 -> 7` MLP. No later trajectory steps are loaded in this mode.

The run saves five result figures:

- `figure_1_mean_survival_curve.png`: true and predicted mean survival curves
- `figure_2_representative_cases.png`: stable, middle, and unstable examples
- `figure_3_per_horizon_mae.png`: model comparison at each horizon
- `figure_4_q_pred_vs_true.png`: predicted versus true `q` at `t=10000`
- `figure_5_survival_heatmaps.png`: true, predicted, and absolute-error heatmaps

For the prefix/variation-count ablation, run the reproducible grid driver:

```bash
conda run -n stellar python -m tasks.task_c.ablation \
  --output-dir results/task_c_ablation \
  --prefix-lens 10,20,50,100 \
  --variation-counts 2,4,8,16,32 \
  --models baseline,direct,traj,dual \
  --prefix-representation orbital15 \
  --variation-target full32 \
  --device cuda
```

The ablation uses balanced nested variation subsets and predicts the full
32-variation system fraction (`q_32`) from each subset. The default
`orbital15` representation keeps the grid comparable to the earlier Task C
notebook and avoids loading the large geometry/full-stack arrays. Each cell is saved
under `prefix_<T>/variations_<N>/`, with the grid
summary in `summary.csv`. Existing cells with `metrics.json` are skipped so the
grid can be resumed; pass `--force` to rerun them. The runner first builds a
shared prefix cache under `results/task_c_ablation/_cache`; use `--cache-dir`
to put this cache on a faster local disk and `--rebuild-cache` to recreate it.
The horizons stay on the original `time_unbound` absolute time axis; the
trajectory sampling clock uses different units and is not added to the horizons.

## Task D

Task D uses initial conditions only to predict the conditional 25th, 50th, and
75th percentiles of a system-level semimajor-axis-ratio Lyapunov-time proxy.
For pair `(i, i+16)`, define

`G_ij(t) = |Q_i(t)-Q_j(t)| / |Q_i(0)-Q_j(0)|`, with `Q=a_in/a_out`.

The default pair label is `T_ij = t_cross(G=e^5)/5`, measured in initial outer
periods. A system label is the reciprocal of the median valid pair growth rate,
`T_system = 1 / median_ij(1/T_ij)`, requiring at least eight valid pairs. This
is an observable finite-time proxy, not a variational-equation maximum
Lyapunov time.

Inputs are the `orbital15` initial conditions, with angles encoded as sine/cosine.
The 16 nominal variations have the same physical values for these 15 features;
their observed differences are only floating-point noise. Tabular models can
therefore use `--tabular-features first` to take the first variation directly,
instead of the default redundant mean/std/min/max summary. All models predict
`log(T_system/P_out,0)` with q25/q50/q75 pinball losses.

```bash
conda run -n stellar python -m tasks.task_d \
  --dataset-dir dataset_sampling_3_prefix1024 \
  --threshold-power 5 \
  --models constant,gbr,mlp,deepsets \
  --output-dir results/task_d_initial_quantiles \
  --device cuda
```

Run the recommended label sensitivity comparison with:

```bash
conda run -n stellar python -m tasks.task_d.sensitivity \
  --threshold-powers 4 5 6 \
  --models constant,gbr \
  --output-dir results/task_d_initial_quantile_sensitivity
```

Metrics include pinball loss and empirical coverage for every quantile, central
50% interval coverage and width, median MAE/RMSE/Spearman, factor-two accuracy,
and quantile-crossing rate. Crossing interpolation is linear in log growth and
can be disabled with `--no-interpolation`.

### Label-threshold sensitivity ablation (2026-09-08)

Using the same seed (`42`), feature representation, split protocol, and models,
we compared crossing thresholds `e^4`, `e^5`, and `e^6`. The table reports the
main test metrics for GBR; the complete constant baseline, per-quantile
coverage, predictions, and plots are summarized in
`results/task_d_initial_quantile_sensitivity/summary.md`.

| Threshold | Valid systems | Target P25/P50/P75 | GBR mean pinball ↓ | GBR P50 log-MAE ↓ | Central 50% coverage | Spearman ↑ | Factor-two accuracy ↑ |
|---:|---:|---|---:|---:|---:|---:|---:|
| `e^4` | 4,997 | 0.9544 / 3.4319 / 9.1262 | **0.3227** | **0.7559** | 0.4827 | 0.6589 | **54.13%** |
| `e^5` | 4,996 | 0.9113 / 3.4091 / 11.2786 | 0.3611 | 0.8475 | 0.4780 | 0.6686 | 51.27% |
| `e^6` | 4,988 | 0.8811 / 3.3414 / 12.0871 | 0.3930 | 0.9139 | 0.4920 | **0.6810** | 48.13% |

GBR outperforms the constant quantile baseline at all three thresholds.
Increasing the threshold widens the upper tail of the target distribution and
steadily increases GBR pinball loss and absolute error, while central 50%
coverage remains close to nominal. The valid system membership changes slightly
because of the minimum-eight-valid-pairs filter, so these are label-sensitivity
trends rather than a strictly paired comparison.

### First-variation ablation (2026-09-06)

The first-variation experiment used the same 4,996 systems, seed 42, and
3,497/750/749 train/validation/test split as the summary-feature experiment.
It requires XGBoost >= 2.0 for the native quantile objective.

```bash
conda run --no-capture-output -n stellar python -m tasks.task_d \
  --dataset-dir dataset_sampling_3_prefix1024 \
  --threshold-power 5 \
  --tabular-features first \
  --models mlp,xgboost \
  --output-dir results/task_d_first_variation_mlp_xgboost \
  --device cuda
```

| Metric | Summary MLP (60D) | First-variation MLP (15D) | First-variation XGBoost (15D) |
|---|---:|---:|---:|
| Mean pinball loss (lower is better) | **0.2760** | 0.2762 | 0.3442 |
| Median log-MAE (lower is better) | 0.6375 | **0.6349** | 0.7993 |
| Median log-RMSE (lower is better) | 0.8456 | **0.8295** | 1.0268 |
| Median Spearman (higher is better) | 0.8186 | **0.8261** | 0.7039 |
| Median factor-two accuracy (higher is better) | 62.88% | **64.22%** | 53.67% |
| Central 50% interval coverage | 45.66% | 41.12% | **46.19%** |

Using only the first variation leaves MLP pinball loss effectively unchanged
and slightly improves its point-prediction and ranking metrics. This supports
the conclusion that the predictive signal comes from the base system's 15
physical initial-condition features, rather than across-variation summaries or
floating-point differences. MLP is more accurate than XGBoost, although its
central interval is too narrow (41.12% empirical coverage versus the nominal
50%). Full outputs are in `results/task_d_first_variation_mlp_xgboost/`.

## Task E

Unified past-window -> future-window benchmark:

```bash
conda run -n stellar python -m tasks.task_e \
  --data_glob "dataset_v3/results_batch_*_sampling_2.hdf5" \
  --representation orbital15 \
  --model ridge \
  --context-len 512 \
  --pred-len 128 \
  --stride 20 \
  --output_dir runs/task_e_ridge_orbital15_ctx512_pred128
```

For a large run, prepare the HDF5 window index and train-only scaler once:

```bash
conda run -n stellar python -m tasks.task_e.prepare \
  --cache-dir runs/task_e_prepared_cache_orbital15_ctx512_pred128 \
  --data-glob "dataset_v3/results_batch_*_sampling_2.hdf5" \
  --completion-table dataset_v3/completed_systems_sampling_2.txt \
  --context-len 512 --pred-len 128 --stride 20 \
  --drop-nan-windows
```

Then reuse it for each model:

```bash
conda run -n stellar python -m tasks.task_e \
  --prepared-cache runs/task_e_prepared_cache_orbital15_ctx512_pred128 \
  --model chronos2 \
  --foundation_batch_size 32 \
  --output_dir runs/task_e_chronos2
```

The cache stores window records and metadata, not a copy of the HDF5 files. It
is rejected automatically when source files or preprocessing settings differ.
The unified entry supports `orbital15` and `raw_state18`/`trajectory`, with
base-system group train/validation/test splits, completion-aware windows, and
train-window-only standardization. Available models are:

- `linear_extrapolation`
- `ridge`
- `tcn`, `patchtst`, `transformer`, `itransformer`
- `chronos2`, `panda_zero_shot`, `tirex2`, `toto2`

`chronos2` 使用 `amazon/chronos-2` 的原生多变量预测，不再把各通道拆成
独立的单变量序列。
`tirex2` 使用 `NX-AI/TiRex-2` 的原生多目标预测，同样保留 15 个通道的联合建模。
`toto2` 使用 `Datadog/Toto-2.0-22m` 的原生多变量预测，并选取 0.5 分位数作为点预测。

`linear_extrapolation` 是不训练的局部趋势基线：对 prefix 最后
`--linear-fit-window`（默认 32）步逐通道拟合最小二乘直线并外推未来；对
`orbital15` 的 sin/cos 角度通道默认投影回单位圆（可用
`--no-linear-angle-projection` 关闭）。它与其他模型共享完全相同的数据窗口、scaler、
system-level split、E1/E2 评估和 bootstrap 协议。

Task E 的主协议包括 E1 one-shot `512 -> 128` 和 E2 四个递归 `128` 步 block
（最长 horizon `512`）。E2 通过同一次运行的 `--run-rolling --rolling-blocks 4`
启用，不重新训练模型，也不会使用未来 ground truth 更新 context。

默认 sampling protocol 保留全部 candidate starts，但每个 epoch 每条 train trajectory
随机采样 8 个窗口；validation/test 固定采样每条 trajectory 的 4 个 early/middle/late
窗口；E2 每条 trajectory 固定采样 2 个分散 rolling starts。正文 metrics 按
`system -> trajectory -> window` 层级聚合并对 system 等权，`--bootstrap-samples 1000`
使用 system-level bootstrap 计算 95% CI。对应 CLI 为
`--train-windows-per-trajectory`、`--val-windows-per-trajectory`、
`--test-windows-per-trajectory` 和 `--rolling-windows-per-trajectory`；设置为 0
可恢复全量 candidate windows。

多个模型运行完成后，可用以下命令生成两张 horizon 曲线和汇总文件：

```bash
conda run -n stellar python -m tasks.task_e.report \
  --run-dirs runs/task_e_e1e2_ridge runs/task_e_e1e2_tcn runs/task_e_e1e2_panda \
  --output-dir results/task_e_e1e2_summary
```

Every run writes `config.json`, `feature_metadata.json`, `scaler.npz`,
`val_metrics.json`, `test_metrics.json`, `horizon_metrics.csv`,
`channel_metrics.csv`, and `predictions_sample.npz`. Trained models also write
`train_log.csv` and `best_model.pt`; ridge writes `best_model.pkl`.

Small end-to-end smoke test:

```bash
conda run -n stellar python -m tasks.task_e \
  --data_glob "dataset_v3/results_batch_00_sampling_2.hdf5" \
  --representation orbital15 \
  --model tcn \
  --context_len 32 \
  --pred_len 8 \
  --max_trajectories 100 \
  --max_scaler_trajectories 70 \
  --max_windows_per_trajectory 5 \
  --epochs 2 \
  --output_dir runs/task_e_tcn_smoke
```

Foundation dependencies are optional. Missing Chronos-2 or Panda
packages/checkpoints produce a model-specific installation/checkpoint error;
they do not affect ridge or supervised models. `panda_zero_shot`
supports the native multivariate Panda path. Historical `panda_adapter` code is
not an active model option. The `itransformer` option
is the documented simple variable-token implementation.

Dataset trajectory visualization:

```bash
conda run -n stellar python tools/visualize_dataset_v3_trajectories.py \
  --row 0 \
  --time-stride 4 \
  --output-dir results/dataset_v3_trajectory_viz
```

Add `--animation --time-stride 20` to save a GIF animation without making the
file too large.

Foundation zero-shot and Panda adapter experiments are available through the
unified `tasks/task_e/main.py` entry point. Their reusable loaders and
adapter modules live under `tasks/task_e/`.
