# Stellar Dynamics Research Repo

This repository studies long-horizon prediction on hierarchical three-body stability datasets. The codebase started as a set of exploratory notebooks and now includes reusable training and evaluation scripts for five tasks:

- `Task A`: group-level outcome prediction from initial conditions
- `Task B`: early-prefix instability classification
- `Task C`: multi-horizon survival prediction for full perturbation sets
- `Task D`: Lyapunov-like semimajor-axis divergence time regression
- `Task E`: direct forecasting baselines, including Transformers and pretrained time-series models

The current script entrypoints live under [scripts](</home/wan/stellar_dynamics/scripts>), shared model/data utilities live under [src](</home/wan/stellar_dynamics/src>), and the Panda subproject is isolated under [experiments/panda_threebody](</home/wan/stellar_dynamics/experiments/panda_threebody>).

**Repository Layout**

- [dataset](</home/wan/stellar_dynamics/dataset>): original `sampling_0` data used by the early notebooks
- [dataset_v2](</home/wan/stellar_dynamics/dataset_v2>): longer-horizon 8-variation dataset used heavily by Task C and Panda
- [dataset_v3](</home/wan/stellar_dynamics/dataset_v3>): full 32-variation dataset used by Tasks A, B, D, and E scripts
- [scripts](</home/wan/stellar_dynamics/scripts>): command-line task runners
- [src](</home/wan/stellar_dynamics/src>): reusable data/model utilities
- [experiments/panda_threebody](</home/wan/stellar_dynamics/experiments/panda_threebody>): vendored Panda code plus project-specific adapters
- [result.md](/home/wan/stellar_dynamics/result.md): working experiment notes and intermediate results
- [results/readme.md](/home/wan/stellar_dynamics/results/readme.md): earlier project summary

## Datasets

Three local dataset versions appear in this repo. They share the same physical problem family but differ in grouping structure, available horizon length, and script usage.

### `dataset/` (`sampling_0`)

This is the original 8-variation setup used by the notebook-era Tasks A and B.

- Initial conditions: `dataset/ICs_sampling_0.hdf5`
- Completion table: `dataset/completed_systems_sampling_0.txt`
- Result shards: `dataset/results_batch_00_sampling_0.hdf5` through `dataset/results_batch_04_sampling_0.hdf5`
- Total trajectories: `40000`
- Perturbation sets: `5000`
- Variations per set: `8`
- Time length per trajectory: `1001` sampled steps
- Typical instability horizon label maximum: `time_unbound = 5000`

This dataset is still relevant for:

- Task A notebook logic
- Task B prefix classification
- Panda-backed final-unbound classification in `experiments/panda_threebody`

### `dataset_v2/` (`sampling_1`)

This is an 8-variation dataset with a longer trajectory horizon and is the source of the clean `orbital15` representation used in Task C and Panda forecasting.

- Initial conditions: `dataset_v2/ICs_sampling_1.hdf5`
- Completion table: `dataset_v2/completed_systems_sampling_1.txt`
- Local result shards currently present: `dataset_v2/results_batch_00_sampling_1.hdf5` through `results_batch_04_sampling_1.hdf5`
- Local trajectories represented by available shards: `40000`
- Complete perturbation sets from local shards: `5000`
- Variations per set: `8`
- Time length per trajectory: `2001` sampled steps
- Typical instability horizon label maximum: `time_unbound = 10000`

This dataset is mainly used for:

- Task C survival prediction
- Panda orbital15 export, zero-shot evaluation, and finetuning

### `dataset_v3/` (`sampling_2`)

This is the current full 32-variation dataset used by the reusable scripts for Tasks A, B, D, and E.

- Initial conditions: `dataset_v3/ICs_sampling_2.hdf5`
- Completion table: `dataset_v3/completed_systems_sampling_2.txt`
- Result shards: `dataset_v3/results_batch_00_sampling_2.hdf5` through `results_batch_19_sampling_2.hdf5`
- Total trajectories: `160000`
- Unique systems: `5000`
- Variations per system: `32`
- Time length per trajectory: `2001` sampled steps
- Local result layout: `20` shards, `8000` runs per shard, `250` unique systems per shard

Useful indexing conventions:

- `unique_system_id = run_id // 32`
- `variation_id = run_id % 32`
- `batch_id = run_id // 8000`
- `row_in_batch = run_id % 8000`

The 32 variations form two 16-variation blocks. The second block repeats the same phase grid with a perturbed inner semimajor axis.

### Common Feature Families

Across the datasets you will encounter these recurring arrays:

- Initial-condition features:
  - `aratios`, `mratios`, `eccs`, `incs`, `APs`, `LANs`
  - plus raw Cartesian state arrays such as `positions`, `velocities`
- Time-series features:
  - `*_complete` versions of the same orbital features
  - `positions_complete`, `velocities_complete`
  - optional derived quantities such as `energy_complete`, `angMomentum_complete`, `periods_complete`, `tAnos_complete`

Two canonical feature stacks show up repeatedly:

- `orbital15`
  - `aratios_complete`: `1`
  - `mratios_complete`: `2`
  - `eccs_complete`: `2`
  - `incs_complete`: `2`
  - `sin/cos(APs_complete)`: `4`
  - `sin/cos(LANs_complete)`: `4`
  - total channels: `15`
- `trajectory/raw_state18`
  - flattened `positions_complete` (`3 bodies x 3 coords = 9`)
  - flattened `velocities_complete` (`3 bodies x 3 coords = 9`)
  - total channels: `18`

### Data Quality Notes

- Rows marked invalid in the completion tables should be dropped.
- Some `*_complete` channels can contain NaNs even when the completion table is valid.
- `periods_complete` and `tAnos_complete` are especially NaN-heavy in earlier datasets.
- Many scripts therefore use filtered orbital subsets (`orbital15`) or explicit finite-value screening before training.

## Shared Task Conventions

Most task scripts follow the same structural ideas:

- Split at the case/system level, not per trajectory, when perturbation grouping matters.
- Keep all perturbations from the same base system in the same train/val/test split whenever possible.
- Use early prefixes as inputs and late-time outcomes or future rollouts as targets.
- Record results under `runs/`, `outputs/`, or `results/` rather than mutating the source notebooks.

For notebook-era work, the perturbation-set size was often `8`. For the newer `dataset_v3` scripts, the natural grouping size is `32`.

## Task A

**Goal**

Predict a system-level outcome summary from initial conditions alone. Historically this was framed as perturbation-set stability probability regression on the 8-variation dataset; the reusable script version is group-level stability classification on the 32-variation dataset.

### Inputs and Targets

Notebook-era Task A:

- Input: grouped initial-condition features from `dataset/ICs_sampling_0.hdf5`
- Group size: `8`
- Features: orbital descriptors with sine/cosine expansion for angular variables
- Target: empirical stability probability or grouped stable/unstable outcome summary

Scripted Task A (`scripts/task_a_stability_classifier.py`):

- Dataset: `dataset_v3`
- Group size: `32`
- Input: grouped initial-condition tokens derived from `ICs_sampling_2.hdf5`
- Target: stable-at-horizon classification at the group level

### Pipeline

1. Load initial-condition arrays.
2. Regroup trajectories into full perturbation sets.
3. Build feature tokens from orbital parameters.
4. Train group-level classifiers or regressors.
5. Save grouped metrics and artifacts.

### Current Script Entry

```bash
conda run -n stellar python scripts/task_a_stability_classifier.py \
  --output-dir results/task_a \
  --models mlp,xgboost \
  --epochs 20
```

If regrouped tokens need to be rebuilt:

```bash
conda run -n stellar python scripts/task_a_stability_classifier.py \
  --output-dir results/task_a \
  --models mlp,xgboost \
  --epochs 20 \
  --regroup \
  --overwrite-tokenized
```

### Existing Findings

From notebook and working-note results:

- MLP and XGBoost both extract signal from static orbital features.
- Performance is meaningful but clearly below prefix-based temporal models.
- Static features alone are insufficient for the strongest long-horizon predictions.

## Task B

**Goal**

Predict whether a trajectory will later become unstable using only an early prefix of the time series.

### Inputs and Targets

- Main trajectory source: `dataset/` in the notebooks, `dataset_v3` in the reusable script
- Typical prefix lengths explored: `100`, `200`, `500`
- Typical target:
  - hard binary label: stable vs unstable by horizon
  - sometimes softened as a distribution over unbinding time bins

The common input feature families are:

- orbital features with angle encodings
- pairwise geometric features
- optional raw prefix features

### Pipeline

1. Load trajectories from HDF5 shards.
2. Truncate to an early prefix.
3. Filter invalid or non-finite samples.
4. Build binary labels from `time_unbound` or related horizon logic.
5. Train prefix-based classifiers such as:
   - SVM
   - CNN
   - TCN
   - Transformer attention
   - IC + prefix hybrids

### Current Script Entry

```bash
conda run -n stellar python scripts/task_b_prefix_classifiers.py \
  --output-dir results/task_b \
  --prefix-len 100 \
  --models svm,cnn,tcn,attention,ic_prefix \
  --epochs 20
```

Quick smoke test:

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

### Existing Findings

Recorded working results show:

- TCN and attention models reach AUC around `0.97+`
- Longer prefixes help materially
- Prefix-based temporal models are much stronger than static IC-only baselines

The repo also includes a Panda-backed classification experiment under [experiments/panda_threebody](</home/wan/stellar_dynamics/experiments/panda_threebody>) using:

- representation: `orbital15`
- prefix length: `512`
- backbone: `GilpinLab/panda-72M`

That subproject is isolated from the main task scripts.

## Task C

**Goal**

Predict multi-horizon survival for all perturbations in a full system, then aggregate to system-level survival statistics.

### Inputs and Targets

- Dataset: `dataset_v2`
- Local complete perturbation sets: `5000`
- Input tensor used in the main setup: `(5000, 8, 200, 15)`
- Representation: `orbital15`
- Target:
  - trajectory-level survival at multiple future horizons
  - ensemble-level survival fraction and count

Typical future horizons in the current cleaned setup:

- `1100`, `1500`, `2000`, `3000`, `4000`, `5000`, `7000`, `8500`, `10000`

### Pipeline

1. Load locally available `dataset_v2` shards.
2. Build the clean `orbital15` feature tensor.
3. Keep only complete 8-way perturbation sets.
4. Split by case ID.
5. Train models that predict monotone survival curves.
6. Aggregate per-trajectory survival into ensemble statistics.

### Model Families

- Mean trajectory baseline
- Trajectory-context encoder
- Dual-attention temporal + set model
- Earlier notebook variants including flat case-level GRU and hierarchical set-seq

### Current Script Entry

```bash
conda run -n stellar python scripts/task_c_survival.py \
  --output-dir results/task_c \
  --prefix-len 200 \
  --time-stride 2 \
  --models baseline,direct,traj,dual \
  --epochs 40
```

### Existing Findings

The strongest current Task C models:

- preserve perturbation-set structure
- enforce monotone survival over future horizons
- significantly outperform mean-curve baselines

This is the most physically structured prediction task in the repo.

## Task D

**Goal**

Estimate a Lyapunov-like system divergence time from early information, using semimajor-axis ratio divergence as the instability proxy.

### Definition

The current Task D pipeline defines pairwise divergence times from semimajor-axis ratio separation:

- `delta_a(t) = |a_i(t) - a_j(t)|`
- `R_a(t) = delta_a(t) / delta_a(0)`
- declare divergence when `R_a(t) > threshold`

Then system-level targets are aggregated from valid pairwise times, typically with:

- threshold: `1e3`
- aggregation: `median`
- target transform: `log_tau_system = log1p(tau_system)`

### Inputs and Targets

- Dataset: `dataset_v3`
- Natural group size: `32`
- Features:
  - static orbital summaries
  - optional prefix-derived sequence summaries
  - optional raw position/velocity features

### Pipeline

1. Load and regroup 32-variation systems.
2. Compute valid pairwise semimajor-axis divergence times.
3. Aggregate pairwise times into one system-level target.
4. Build static or prefix-augmented system features.
5. Train regressors:
   - Random forest
   - MLP
   - CNN

### Current Script Entry

```bash
conda run -n stellar python scripts/task_d_lyapunov_regression.py \
  --output-dir outputs/task_d_lyapunov \
  --mlp-epochs 80 \
  --cnn-epochs 80
```

Useful optional flags include:

- `--include-prefix-features`
- `--prefix-len 200`
- `--include-positions-velocities`
- `--sequence-aggregation mean_std|flatten`

### Existing Findings

Current working notes show:

- Random forest is a strong baseline
- CNN can achieve the best Spearman rank correlation among the tested neural models
- Predicting `log1p(tau_system)` is more stable than direct raw-time regression

## Task E

**Goal**

Forecast future orbital or Cartesian trajectory segments directly from an early prefix. This task contains both learned Transformers and pretrained foundation-model baselines.

### Subtasks

Task E currently has three main branches:

1. Transformer orbital forecasting
2. Transformer direct trajectory forecasting
3. Pretrained foundation-model forecasting

### Task E.1: Transformer Orbital Forecasting

- Script: [scripts/task_e_transformer_orbital.py](/home/wan/stellar_dynamics/scripts/task_e_transformer_orbital.py)
- Dataset: `dataset_v3`
- Input features: orbital channels from result shards
- Debug setup: smaller synthetic subset and shorter run
- Full setup defaults:
  - `context_len = 256`
  - `pred_len = 64`
  - `stride = 32`

Command:

```bash
conda run -n stellar python scripts/task_e_transformer_orbital.py \
  --mode debug \
  --conda-env stellar \
  --debug-epochs 3
```

Full run:

```bash
conda run -n stellar python scripts/task_e_transformer_orbital.py \
  --mode full \
  --conda-env stellar \
  --full-epochs 50 \
  --amp
```

### Task E.2: Transformer Direct Trajectory Forecasting

- Script: [scripts/task_e_transformer_trajectory.py](/home/wan/stellar_dynamics/scripts/task_e_transformer_trajectory.py)
- Default channels: `positions_complete,velocities_complete`
- Full-run defaults:
  - `context_len = 64`
  - `pred_len = 8`
  - `stride = 128`
  - compact 2-layer Transformer for faster sweeps

Command:

```bash
conda run -n stellar python scripts/task_e_transformer_trajectory.py \
  --mode debug \
  --conda-env stellar \
  --debug-epochs 3
```

Full run:

```bash
conda run -n stellar python scripts/task_e_transformer_trajectory.py \
  --mode full \
  --conda-env stellar \
  --full-epochs 10 \
  --amp
```

### Task E.3: Pretrained Foundation Forecasting

- Script: [scripts/task_e_foundation_forecasters.py](/home/wan/stellar_dynamics/scripts/task_e_foundation_forecasters.py)
- Supported model keys:
  - `naive`
  - `timesfm`
  - `chronos`
  - `patchtst`
  - `panda`
- Supported representations:
  - `trajectory`
  - `orbital`
  - `orbital15`

#### Representation Behavior

- `trajectory`
  - uses flattened `positions_complete + velocities_complete`
  - `18` channels
  - univariate models are evaluated channel-by-channel
- `orbital`
  - uses a 17-channel orbital stack with log transforms for ratios and `sin/cos` expansion for angles
- `orbital15`
  - matches the Panda-oriented orbital representation
  - required for the default `GilpinLab/panda-72M` checkpoint path to make semantic sense

#### Current Default Window

The current default forecasting window is:

- `context_len = 512`
- `pred_len = 128`

#### Univariate vs Multivariate Baselines

- `timesfm`, `chronos`, `patchtst`
  - treated as univariate baselines in this script
  - one multivariate sample with `C` channels becomes `C` separate series
- `panda`
  - consumes the full multivariate window directly

This is why `trajectory` mode with `256` windows becomes:

- `256 x 18 = 4608` univariate series for `timesfm`, `chronos`, and `patchtst`
- but still only `256` multivariate windows for `panda`

#### Default Commands

Trajectory foundation baselines:

```bash
conda run -n stellar python scripts/task_e_foundation_forecasters.py \
  --models naive,timesfm,chronos,patchtst \
  --representation trajectory \
  --output-dir runs/foundation_trajectory_baselines \
  --num-windows 256 \
  --context-len 512 \
  --pred-len 128 \
  --batch-size 128 \
  --device cuda
```

Panda-compatible orbital baselines:

```bash
conda run -n stellar python scripts/task_e_foundation_forecasters.py \
  --models naive,chronos,patchtst,panda \
  --representation orbital15 \
  --output-dir runs/foundation_orbital_baselines \
  --num-windows 256 \
  --context-len 512 \
  --pred-len 128 \
  --batch-size 128 \
  --device cuda \
  --no-hf-local-files-only
```

Important operational notes:

- `timesfm`, `chronos`, and `panda` are optional dependencies.
- The script records per-model failures and continues unless `--fail-fast` is set.
- Hugging Face checkpoints default to `local_files_only=True`; use `--no-hf-local-files-only` to allow download.
- `timesfm` is often the slowest model because it combines the largest parameter count with a list-based forecasting API and many univariate series.
- `panda` uses the vendored Panda implementation from `experiments/panda_threebody/vendor/panda`, not a separate custom reimplementation.

## Panda Subproject

The Panda integration under [experiments/panda_threebody](</home/wan/stellar_dynamics/experiments/panda_threebody>) is intentionally isolated from the rest of the repo.

It contains:

- a vendored Panda source snapshot
- adapters that export three-body trajectories into Panda Arrow format
- zero-shot and finetune scripts for `orbital15`
- a Panda-backed classifier for final-unbound prediction

The vendored Panda code itself is not where the main task logic was written. The project-specific glue is in:

- [experiments/panda_threebody/scripts](</home/wan/stellar_dynamics/experiments/panda_threebody/scripts>)
- [experiments/panda_threebody/adapters](</home/wan/stellar_dynamics/experiments/panda_threebody/adapters>)
- [experiments/panda_threebody/configs](</home/wan/stellar_dynamics/experiments/panda_threebody/configs>)

For Panda-specific usage details, read [experiments/panda_threebody/README.md](/home/wan/stellar_dynamics/experiments/panda_threebody/README.md).

## Environment and Dependencies

Most examples in this repo assume the `stellar` conda environment.

Common dependencies:

- `h5py`
- `numpy`
- `pandas`
- `matplotlib`
- `scikit-learn`
- `torch`
- `transformers`

Optional Task E foundation dependencies:

- `timesfm`
- `chronos`
- Panda vendor dependencies under `experiments/panda_threebody`

## Outputs and Artifacts

You will typically find outputs under:

- `results/`: earlier notebook-linked summaries
- `runs/`: Task E and fast experiment artifacts
- `outputs/`: Task D and Panda subproject artifacts

Typical saved files include:

- `config.json`
- `metrics.json`
- `summary_metrics.csv`
- horizon-wise metric CSVs
- example forecast CSVs
- model checkpoints
- plots such as loss curves and example predictions

## Recommended Entry Points

If you want the fastest orientation path:

1. Read [scripts/README.md](/home/wan/stellar_dynamics/scripts/README.md) for executable commands.
2. Read [experiments/panda_threebody/README.md](/home/wan/stellar_dynamics/experiments/panda_threebody/README.md) for the Panda branch.
3. Check [result.md](/home/wan/stellar_dynamics/result.md) for the latest working notes and metric snapshots.

If you want to reproduce the main code-driven tasks:

1. Task A: `scripts/task_a_stability_classifier.py`
2. Task B: `scripts/task_b_prefix_classifiers.py`
3. Task C: `scripts/task_c_survival.py`
4. Task D: `scripts/task_d_lyapunov_regression.py`
5. Task E: `scripts/task_e_transformer_*` and `scripts/task_e_foundation_forecasters.py`
