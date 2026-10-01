# Foundation models for three-body dynamics

This directory holds the pretrained-model adapters and Task B probe pipelines used by the paper. Run commands from the repository root. The raw sampling 3 dataset lives outside Git; point `--dataset-dir` at the matching HDF5 shards and completion table. See the [paper reproduction map](../../docs/reproduction.md) for the reported configurations.

| Directory | Role |
| --- | --- |
| `panda/` | Panda-72M Task B frozen embeddings and linear/other probes; [historical Panda pilot notes](panda/README.md) |
| `chronos2/` | Chronos-2 Task B REG embeddings and linear probe |
| `shared/` | Model-independent orbital-summary XGBoost and optional embedding/fusion comparisons |
| `adapters/` | Orbital15 and other input representations shared by the extractors |
| `vendor/panda/` | Panda source git submodule, initialized with `git submodule update --init --recursive` |
| `scripts/` | Compatibility entry points and older Panda pilot launchers; see [scripts/README.md](scripts/README.md) |
| `configs/`, `reports/` | Earlier Panda pilot settings and reports, not the sampling 3 paper protocol |
| `outputs/` | Ignored local caches, checkpoints, and predictions; paths moved with this directory on the current machine |

## From a fresh clone

The submodule provides Panda **source code only** at the repository-pinned commit. It does not contain `GilpinLab/panda-72M` model weights, simulation HDF5 shards, completion tables, or generated embedding caches. After cloning the committed repository:

```bash
git submodule update --init --recursive
conda env create -f environment.yml
conda activate stellar
uv pip sync requirements.lock --python "$CONDA_PREFIX/bin/python" --torch-backend cu130 --require-hashes
python -m pip check
python scripts/check_environment.py
```

The single locked environment targets Linux x86_64 with CUDA 13.0 PyTorch wheels. A clean installation on another machine has not yet been tested. The Task E loader defaults to `GilpinLab/panda-72M` and **local-files-only** checkpoint loading. Populate the Hugging Face cache or pass a local model path with `--panda_model_id` before running. To permit an explicit download in the Task E wrapper, set `STELLAR_ALLOW_MODEL_DOWNLOAD=1` (network access and access to the model are still required):

```bash
STELLAR_ALLOW_MODEL_DOWNLOAD=1 bash scripts/task_e.sh panda_zero_shot /path/to/sampling3
```

The sampling 3 directory must contain `results_batch_*_sampling_3.hdf5` and `completed_systems_sampling_3.txt`. The wrapper prepares a train-only scaler and system-disjoint split before E1/E2 inference. For Task B/C frozen-embedding results, first reproduce the aligned Task B labels/split, then extract model features; the submodule alone cannot recreate those caches.

The Task C survival head reads aligned Panda/Chronos-2 feature caches but is implemented in `tasks.task_c.survival`. Task E forecasting uses the main `tasks.task_e` entry point (`--model panda_zero_shot` or `chronos2`), not the older pilot scripts here.

Examples for a prepared sampling 3 Task B prefix split:

```bash
python experiments/foundation_models/shared/train_spock_inspired_task_b_xgboost.py \
  --dataset-dir dataset_sampling_3_prefix1024 --sampling-id 3 \
  --prefix-dir results/task_b/sampling_3/task_b_sampling3_prefix512 --prefix-steps 128 \
  --output-dir results/reproduce_task_b_stats_xgboost
```

Panda and Chronos-2 probes require their pretrained weights or a matching cached feature file. Their exact label order and system split must match the Task B prefix run. Use `--help` on the scripts under `panda/` and `chronos2/` before supplying cache paths.
