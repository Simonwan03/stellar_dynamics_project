# Data Audit

## Scope

This audit is limited to read-only inspection of the existing repository plus the vendored Panda repo. No files outside `experiments/foundation_models/` were modified.

## Source Repo Inventory

### Primary data roots

1. `dataset/`
   - `ICs_sampling_0.hdf5`
   - `results_batch_00_sampling_0.hdf5` .. `results_batch_04_sampling_0.hdf5`
   - `completed_systems_sampling_0.txt`
2. `dataset_v2/`
   - `ICs_sampling_1.hdf5`
   - `results_batch_00_sampling_1.hdf5` .. `results_batch_04_sampling_1.hdf5`
   - `completed_systems_sampling_1.txt`

### Existing data / modeling logic

- `read_data.ipynb`
  - repo-level dataset summary
  - multi-shard HDF5 loader for trajectory tensors
- `task_a.ipynb`
  - IC-level grouping into 8-way perturbation sets
  - static initial-condition feature handling
- `task_b.ipynb`
  - multishard trajectory loading for `dataset/`
  - NaN reporting
  - label sorting / mask alignment
  - richer forecasting/classification feature stacks
- `task_c.ipynb`
  - forecasting-oriented 15D orbital stack on `dataset_v2/`
  - complete-case filtering at perturbation-set granularity
  - deterministic stratified case split

## Trajectory Sources Usable for Panda Forecasting

### Candidate 1: `dataset/` results shards

- Paths:
  - `dataset/results_batch_00_sampling_0.hdf5` .. `dataset/results_batch_04_sampling_0.hdf5`
- Saved notebook output:
  - each shard exposes:
    - `positions_complete`: `(8000, 1001, 3, 3)`
    - `velocities_complete`: `(8000, 1001, 3, 3)`
    - `times_complete`: `(8000, 1001)`
    - plus orbital channels such as `aratios_complete`, `eccs_complete`, `incs_complete`, `APs_complete`, `LANs_complete`, `masses_complete`, `energy_complete`
  - concatenated local total:
    - `40000` trajectories
    - effective prefix example from notebook: `(40000, 100, 3, 3)` for `positions_complete`
- Labels:
  - `dataset/completed_systems_sampling_0.txt`
  - rows: `40000`
  - `num_batch` range: `0..4`
  - `time_unbound` max: `5000`
  - no negative `time_unbound`
  - no nonzero diagnosis codes in the saved file scan

### Candidate 2: `dataset_v2/` results shards

- Paths:
  - `dataset_v2/results_batch_00_sampling_1.hdf5` .. `dataset_v2/results_batch_04_sampling_1.hdf5`
- Saved notebook output:
  - each shard exposes:
    - `positions_complete`: `(8000, 2001, 3, 3)`
    - `velocities_complete`: `(8000, 2001, 3, 3)`
    - `times_complete`: `(8000, 2001)`
    - plus orbital channels matching the v1 schema
  - `task_c.ipynb` loads the locally present five shards and builds:
    - trajectory tensor: `(40000, 200, 15)`
    - complete perturbation-set tensor: `(5000, 8, 200, 15)`
    - future target tensor: `(5000, 8, 9)`
- Labels:
  - `dataset_v2/completed_systems_sampling_1.txt`
  - rows: `400000`
  - `num_batch` range: `0..49`
  - local repo only contains result shards `0..4`
  - negative `time_unbound`: `139`
  - nonzero diagnosis: `139`
  - `task_c.ipynb` already filters invalid labels and keeps only complete 8-trajectory cases

## Shapes and Alignment

### Raw IC-level tensors

From saved output in `task_a.ipynb` for `dataset/ICs_sampling_0.hdf5`:

- `aratios`: `(40000, 1)`
- `mratios`: `(40000, 2)`
- `eccs`: `(40000, 2)`
- `incs`: `(40000, 2)`
- `APs`: `(40000, 2)`
- `LANs`: `(40000, 2)`
- `positions`: `(40000, 3, 3)`
- `velocities`: `(40000, 3, 3)`
- grouped 8-way tokenized IC example:
  - `APs: (40000, 2) -> (5000, 8, 2)`
  - same grouping used for other static channels

### Forecasting-ready sequence tensors

1. `orbital15` from `task_c.ipynb`
   - input shape before case grouping: `(40000, 200, 15)`
   - after grouping by complete perturbation sets: `(5000, 8, 200, 15)`
   - clean and already used in the local forecasting notebook
2. `raw_state18`
   - derivable from `positions_complete` + `velocities_complete`
   - expected per-step dim: `3 bodies * (3 xyz + 3 vxyz) = 18`
3. `geom9`
   - derivable from notebook helper `pairwise_geom`
   - expected per-step dim: `9`
4. `full38`
   - task-b notebook stack:
   - orbital/physical channels + geometry + normalized time
   - total approximately `38` channels

## NaN / Padding / Mask Findings

From saved output in `task_b.ipynb` on the 5 local `dataset/` shards:

- `periods_complete`: `16847 / 40000` samples have NaN within the first `T=100`
- `tAnos_complete`: `14509 / 40000`
- `positions_complete`: `254 / 40000`
- `velocities_complete`: `0 / 40000`
- many other orbital channels report `0 / 40000`

Current masking logic already present in the notebook:

- sort labels by `["num_batch", "index"]`
- compute:
  - `bad_pos = ~isfinite(positions_complete[:, :T]).all(...)`
  - `bad_vel = ~isfinite(velocities_complete[:, :T]).all(...)`
- use shared boolean mask for both data and labels

This audit therefore treats `periods_complete` and `tAnos_complete` as low-priority forecasting inputs, and keeps masking explicit in the adapter before export.

## Label Alignment

### Per-trajectory labels

The local repo stores label rows as:

- `index`
- `num_batch`
- `time_unbound`
- an auxiliary float column
- `diagnosis` / `dignosis`

Existing alignment conventions discovered in notebooks:

- sort labels by `["num_batch", "index"]` to match concatenated shard order
- define `raw_case_id = np.arange(len(labels)) // 8`
- later refactor to `case_id` after cleaning complete sets

### Existing forecasting-oriented targets

The repo’s own historical tasks are stability-oriented, but `task_c.ipynb` already reframes this as future survival forecasting on perturbation sets:

- future checkpoints:
  - `[1100, 1500, 2000, 3000, 4000, 5000, 7000, 8500, 10000]`
- target tensors:
  - per-trajectory survival curve: `(B, 8, 9)`
  - per-case fraction curve: `(B, 9)`

For Panda integration, the primary target is simpler:

- export full trajectories as multivariate time series
- let Panda train on `prefix -> future trajectory`
- keep stability labels only for grouped evaluation and optional probe

## Representation Priority

### 1. `orbital15` (chosen primary)

Why first:

- already exercised on `dataset_v2` in forecasting-style notebook code
- smallest clean representation with direct physical meaning
- no reliance on NaN-heavy `periods_complete` / `tAnos_complete`
- preserves the exact feature definition already validated locally

### 2. `raw_state18`

Why second:

- closest to a direct trajectory forecast target
- low ambiguity: body positions and velocities only
- useful as a plain baseline against feature-engineered orbital channels

### 3. `geom9`

Why third:

- translation/rotation-invariant
- may be easier for cross-system generalization
- useful for later comparison with zero-shot behavior

### 4. `full38`

Why fourth:

- richer but includes more potential instability / redundancy
- heavier channels with more preprocessing assumptions
- better suited as a later extension than the first MVP

## Panda Compatibility Notes

Vendored Panda repo confirms:

- train entrypoint: `vendor/panda/scripts/panda/train.py`
- eval entrypoint: `vendor/panda/scripts/panda/evaluate.py`
- Hydra config root: `vendor/panda/config/`
- multivariate loading path:
  - `FileDataset(path=..., freq="h", one_dim_target=False)`
  - `MultivariateTimeSeriesDataset(...)`
- expected Arrow layout:
  - `{data_root}/{system_name}/*.arrow`

The adapter therefore exports each trajectory as one Arrow file containing a multivariate target matrix with shape `(channels, time)`, then validates that Panda can read it back with `one_dim_target=False`.

## Recommended MVP Path

1. Use `dataset_v2/` local shards `00..04` only.
2. Export `orbital15` to Panda Arrow.
3. Validate round-trip shapes with Panda utilities.
4. Run:
   - zero-shot with HF pretrained Panda weights
   - small finetune from pretrained init
   - small scratch sanity check
5. Evaluate on held-out test systems and summarize metrics per horizon and per subgroup.

