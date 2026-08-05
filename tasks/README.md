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
└── task_f/   # boundary sensitivity
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
  --models mlp,xgboost \
  --epochs 80
```

Binary all-stable classification:

```bash
conda run -n stellar python -m tasks.task_a \
  --output-dir results/task_a_binary_any \
  --target-mode binary_any \
  --time-horizon 10000 \
  --models mlp,xgboost \
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
the cumulative product of one minus those hazards. `traj` and `dual` optimize a
variation-level masked hazard negative log-likelihood. `direct` and
`direct_mean_std` optimize the corresponding interval-count binomial hazard
negative log-likelihood from the aggregated 32-variation survival fractions.

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

## Task D

Task D now predicts a finite-time Lyapunov-time proxy rather than threshold-crossing time.
For each of the 16 `(i, i+16)` perturbation pairs, it fits the slope of `log R(t)`
in the configured early-growth interval, rejects low-quality fits, aggregates the
accepted positive slopes, and uses
`log_lyapunov_time_proxy = log(1 / lambda_phase_space)` as the training target.
The old `R(t) > 1000` crossing time remains an auxiliary diagnostic.

```bash
conda run -n stellar python -m tasks.task_d \
  --task-setting initial \
  --output-dir results/task_d_initial \
  --growth-lower 1 \
  --growth-upper 100 \
  --growth-sensitivity-uppers 10 30 100 \
  --lambda-r2-min 0.8 \
  --min-valid-lambda-pairs 8 \
  --max-pair-log-mad 0.75 \
  --mlp-epochs 80
```

The prefix setting predicts the same Lyapunov-time proxy using `cnn_orbital`,
`cnn_phase_space`, and `set_cnn`. Its current label fit uses the stored full
trajectory, including the prefix interval, so a future-only non-overlapping fit
is required before interpreting it as strict forecasting.

The saved dataset is generally too coarsely sampled to obtain five points in
the `R=1–10`, `1–30`, or `1–100` growth ranges. The quality filter intentionally
rejects such systems. Production Lyapunov labels therefore require higher-cadence
early trajectory output or variational/tangent-equation integrations with periodic
renormalization. The result should be described as a finite-time Lyapunov-time
proxy, not a strict asymptotic Lyapunov time.

## Task E

Unified past-window -> future-window benchmark:

```bash
conda run -n stellar python -m tasks.task_e \
  --data_glob "dataset_v3/results_batch_*_sampling_2.hdf5" \
  --representation orbital15 \
  --model persistence \
  --context_len 200 \
  --pred_len 50 \
  --stride 20 \
  --output_dir runs/task_e_persistence_orbital15_ctx200_pred50
```

The unified entry supports `orbital15` and `raw_state18`/`trajectory`, with
base-system group train/validation/test splits, completion-aware windows, and
train-window-only standardization. Available models are:

- `persistence`, `ridge`
- `tcn`, `patchtst`, `transformer`, `itransformer`
- `timesfm`, `chronos`, `panda_zero_shot`, `panda_adapter`

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

Foundation dependencies are optional. Missing TimesFM, Chronos, or Panda
packages/checkpoints produce a model-specific installation/checkpoint error;
they do not affect persistence, ridge, or supervised models. `panda_zero_shot`
and the current `panda_adapter` require `orbital15`. The `itransformer` option
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
