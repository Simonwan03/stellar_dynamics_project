# Task Experiment Scripts

These entrypoints convert the original task notebooks into reusable command-line jobs.
Run them from the repository root. On the current machine the examples use the `stellar`
conda environment; on AWS, use whichever environment contains `h5py`, `numpy`, `pandas`,
`scikit-learn`, `matplotlib`, and `torch`.

## Task A

Group-level stable-at-horizon modeling from tokenized initial conditions.
By default this predicts the empirical stable fraction across the 32 perturbations
for each system, matching the original Task A distribution-label objective.

```bash
conda run -n stellar python scripts/task_a_stability_classifier.py \
  --output-dir results/task_a \
  --models mlp,xgboost \
  --epochs 80
```

MLP presets:
- `--mlp-size small`: hidden `(64,64,32)`, dropout `0.1`
- `--mlp-size base`: hidden `(128,128,64)`, dropout `0.2`
- `--mlp-size large`: hidden `(256,256,128)`, dropout `0.2`

MLP loss modes:
- `--mlp-loss mse`: `Sigmoid + MSELoss`
- `--mlp-loss bce`: `Linear + BCEWithLogitsLoss` with the same `stable_fraction` labels

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
conda run -n stellar python scripts/task_b_prefix_classifiers.py \
  --output-dir results/task_b \
  --representation orbital15 \
  --prefix-len 100 \
  --models svm,cnn,tcn,attention \
  --epochs 50
```

To predict the stable fraction across each 32-trajectory perturbation group
from early prefixes:

```bash
conda run -n stellar python scripts/task_b_prefix_classifiers.py \
  --output-dir results/task_b_group_fraction \
  --representation orbital15 \
  --target-mode group_fraction \
  --group-aggregation set_attention \
  --prefix-len 100 \
  --models group_attention \
  --epochs 50
```

Use `--group-aggregation concat --models svm,cnn,tcn,attention` for the simpler
baseline that flattens the 32 variations into the feature channel dimension.

`Task B` now defaults to the `orbital15` sequence representation:
`aratios_complete(1) + mratios_complete(2) + eccs_complete(2) + incs_complete(2) + sin/cos(APs_complete)(4) + sin/cos(LANs_complete)(4) = 15` channels per time step.
The default label horizon is `--time-horizon 10000`.
Use `--representation full` to restore the older wider stack with extra orbital channels plus optional geometry/time features.

The temporal models now auto-scale with the effective number of input steps `T`:
- `SVM`: params follow `T * 15`
- `T=100`: `cnn ~92K`, `tcn ~355K`, `attention ~399K`
- `T=200`: `cnn ~141K`, `tcn ~454K`, `attention ~597K`
- `T=500`: `cnn ~200K`, `tcn ~706K`, `attention ~931K`

If you set `--hidden-dim`, that hidden width is used directly instead of the automatic per-`T` scale. You can also override `--tcn-levels`, `--attention-layers`, and `--attention-heads`.
Early stopping defaults to `--early-stopping-patience 8`; set it to `0` to disable.

For a quick AWS smoke test:

```bash
conda run -n stellar python scripts/task_b_prefix_classifiers.py \
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
conda run -n stellar python scripts/task_c_survival.py \
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

The run saves five result figures:

- `figure_1_mean_survival_curve.png`: true and predicted mean survival curves
- `figure_2_representative_cases.png`: stable, middle, and unstable examples
- `figure_3_per_horizon_mae.png`: model comparison at each horizon
- `figure_4_q_pred_vs_true.png`: predicted versus true `q` at `t=10000`
- `figure_5_survival_heatmaps.png`: true, predicted, and absolute-error heatmaps

## Task D

System-level phase-space divergence regression has two leakage-aware settings.
The default initial-condition experiment predicts `log1p(tau_system)` from
`t=0` features with a constant baseline, random forest, and MLP:

```bash
conda run -n stellar python scripts/task_d_lyapunov_regression.py \
  --task-setting initial \
  --output-dir results/task_d_initial \
  --growth-lower 1 \
  --growth-upper 100 \
  --mlp-epochs 80
```

The prefix setting computes the actual observation end time from
`times_complete`, removes systems with `tau_system <= t_obs`, and predicts
`log1p(tau_system - t_obs)`:

```bash
conda run -n stellar python scripts/task_d_lyapunov_regression.py \
  --task-setting prefix \
  --prefix-len 200 \
  --prefix-time-stride 1 \
  --output-dir results/task_d_prefix_tobs_200 \
  --cnn-epochs 80
```

Prefix models are `cnn_orbital`, `cnn_phase_space`, and `set_cnn`. Neural
regressors use Huber loss by default; select `--loss mse` for the previous MSE
objective. The target is an empirical Lyapunov-like divergence timescale, not a
strict infinitesimal Lyapunov time.

## Task E

Dataset trajectory visualization:

```bash
conda run -n stellar python scripts/visualize_dataset_v3_trajectories.py \
  --row 0 \
  --time-stride 4 \
  --output-dir results/dataset_v3_trajectory_viz
```

Add `--animation --time-stride 20` to save a GIF animation without making the
file too large.

Transformer orbital forecasting baseline. Debug first, then full.

```bash
conda run -n stellar python scripts/task_e_transformer_orbital.py \
  --mode debug \
  --conda-env stellar \
  --debug-epochs 3

conda run -n stellar python scripts/task_e_transformer_orbital.py \
  --mode full \
  --conda-env stellar \
  --full-epochs 50 \
  --amp
```

Direct trajectory forecasting baseline for comparison. By default this predicts
`positions_complete,velocities_complete` from the same past channels.

```bash
conda run -n stellar python scripts/task_e_transformer_trajectory.py \
  --mode debug \
  --conda-env stellar \
  --debug-epochs 3

conda run -n stellar python scripts/task_e_transformer_trajectory.py \
  --mode full \
  --conda-env stellar \
  --full-epochs 10 \
  --amp
```

The trajectory wrapper defaults to a faster short-horizon full run:
`context_len=64`, `pred_len=8`, `stride=128`, a 2-layer `d_model=128` Transformer, and capped
train/validation/test windows. It also estimates normalization from a capped
training-trajectory subset. Increase `--max-train-windows`, lower
`--full-stride`, raise `--max-normalization-trajectories`, or raise the model
size when you want a slower exhaustive run.

Pretrained foundation-model zero-shot trajectory baselines. Univariate models
(`timesfm`, `chronos`, `patchtst`) treat each position/velocity channel as an
independent series; `panda` consumes the full multivariate window directly.

```bash
conda run -n stellar python scripts/task_e_foundation_forecasters.py \
  --models naive,timesfm,chronos,patchtst,panda \
  --representation trajectory \
  --num-windows 256 \
  --context-len 512 \
  --pred-len 128 \
  --batch-size 128 \
  --device cuda
```

`timesfm`, `chronos`, and `panda` are optional dependencies. If they are not
installed or their checkpoints are unavailable, the script records a per-model
error JSON and continues with the remaining models. PatchTST defaults to the
public `ibm/patchtst-etth1-forecasting` checkpoint and can try comma-separated
fallbacks via `--patchtst-fallback-model-ids`. Foundation checkpoints are
loaded from the local Hugging Face cache by default to avoid long offline retry
delays; pass `--no-hf-local-files-only` when you intentionally want to download
missing checkpoints.

For Panda-compatible orbital baselines, use the `orbital15` representation:

```bash
conda run -n stellar python scripts/task_e_foundation_forecasters.py \
  --models naive,chronos,patchtst,panda \
  --representation orbital15 \
  --output-dir runs/foundation_orbital_baselines \
  --num-windows 256 \
  --context-len 512 \
  --pred-len 128 \
  --batch-size 128 \
  --device cuda
```

The default `GilpinLab/panda-72M` checkpoint expects `orbital15` style input
with 15 channels. The script will raise a clear error if you try to use it with
the 17-channel `orbital` representation or the 18-channel raw trajectory stack.

Panda temporal-window adapter fine-tuning. This freezes Panda by default and
trains two temporal MLPs around it:
`input_len -> panda_context_len` before Panda and `panda_pred_len -> output_len`
after Panda. The default setup adapts `256 -> 512` context steps and
`128 -> 64` forecast steps.

```bash
conda run -n stellar python scripts/train_panda_temporal_adapter.py \
  --representation orbital15 \
  --input-len 256 \
  --output-len 64 \
  --num-windows 4096 \
  --epochs 10 \
  --batch-size 32 \
  --device cuda \
  --amp
```

Use `--no-freeze-panda` only when you intentionally want to fine-tune the Panda
backbone as well as the temporal adapters.
The adapter trainer drops windows with non-finite values or any absolute channel
value above `--max-abs-value` (`1e4` by default), because extreme orbital
element outliers can make the Panda backbone return NaNs.

For position/velocity features instead of `orbital15`, use the trajectory
adapter. It defaults to the center-of-mass reference frame, standardizes the 18
relative state channels, maps `18 -> Panda channels` before Panda, then maps
back to 18 channels after Panda:

```bash
conda run -n stellar python scripts/train_panda_trajectory_adapter.py \
  --input-len 256 \
  --output-len 64 \
  --num-windows 4096 \
  --epochs 10 \
  --batch-size 32 \
  --device cuda \
  --output-dir runs/panda_trajectory_adapter_full
```

Use `--reference-frame absolute` to train on raw absolute coordinates, or
`--reference-frame body0` to use body 0 as the origin.
