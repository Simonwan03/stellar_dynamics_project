# Stellar dynamics learning experiments

This repository contains research code for predicting stability, survival, divergence time, and future motion in perturbed three-body systems. Start with the [manuscript reproduction map](docs/reproduction.md) and [release audit](docs/reproducibility_audit.md). The longer [research log](readme.md) preserves earlier experiments and result tables.

## Install

The supported environment targets Linux x86_64, Python 3.12.12, and the PyTorch CUDA 13.0 wheels. `requirement.txt` is the single dependency input for all tasks, including the foundation models; `requirements.lock` fixes the complete pip dependency graph and wheel hashes. The Panda source is pinned by the Git submodule and its model weights are external.

```bash
conda env create -f environment.yml
conda activate stellar
git submodule update --init --recursive
uv pip sync requirements.lock --python "$CONDA_PREFIX/bin/python" --torch-backend cu130 --require-hashes
python -m pip check
python scripts/check_environment.py
python -m pytest -q
```

Run these commands from the repository root. To change dependencies, edit `requirement.txt`, then regenerate and review the lock with:

```bash
uv pip compile requirement.txt --python "$CONDA_PREFIX/bin/python" --python-platform x86_64-unknown-linux-gnu --torch-backend cu130 --exclude-newer 2026-10-01T00:00:00Z --generate-hashes --no-annotate --custom-compile-command 'See README.md#install for the exact regeneration command' -o requirements.lock
```

Advance `--exclude-newer` deliberately when updating packages. A clean GPU installation and all model checkpoint runs still need validation on the target machine. Use `python scripts/task_<letter>.py --resolution low|high --help` to inspect each supported task profile; see the [entry point map](scripts/readme.md) for availability and defaults.

## Data and code

The simulation data is **not tracked**, and this repository provides no verified public download location. Supply sampling 3 shards as `dataset_sampling_3_prefix1024/results_batch_*_sampling_3.hdf5` plus `completed_systems_sampling_3.txt`, or pass the path to your own copy. On this development machine, `dataset_sampling_3_prefix1024` is an ignored symlink to `/data/dataset_sampling_3_prefix1024_float64`; a fresh clone needs its own data path or symlink. Earlier sampling 2 experiments use `dataset_v3/`; in this checkout that path is a machine-specific symlink. The completion table and shards must have the same sampling ID. For example, run `python scripts/task_b.py --resolution high --dataset-dir /path/to/sampling3 --prefix-len 128 --models mlp,cnn,tcn --split-manifest configs/splits/common_case_split_sampling_3_g32_h10000_r70-15-15_seed42.csv --output-dir results/task_b/sampling_3/reproduce_task_b`, or pass that directory as the second argument to `scripts/task_e.sh`.

`tasks/` owns preprocessing, models, evaluation, and auxiliary experiments; `scripts/` contains Task A–F entry points; `experiments/foundation_models/` separates Panda, Chronos-2, shared baselines, and the Panda source submodule. `results/` is the single output tree. It groups classification, survival, divergence, forecasting, robustness, cross-task comparisons, and shared analysis by task and dataset. The [artifact inventory](docs/artifact_inventory.md) lists current paths, the [CSV index](docs/artifact_inventory.csv) records classification sources, the [path map](docs/artifact_path_map.csv) translates historical paths, and the [merge map](docs/artifact_merge_map.csv) records the final consolidation. New outputs are ignored by Git. `archive/` holds archived research notes; `tests/` contains basic checks.

The reported Task B/C short-prefix inputs use the same 15-channel `orbital15` representation; the [reproduction map](docs/reproduction.md#task-bc-prefix-input) lists the exact fields and distinguishes frozen embeddings from raw sequences.

## Reproduce results

Run `python -m tasks.common.preflight --task B --dataset-dir /path/to/sampling3` before an expensive sampling-3 run. For Task A, use `--task A --dataset-dir /path/to/sampling2`. The checker reports missing data and validates the preserved B/C/F system split.

[The reproduction map](docs/reproduction.md) connects Tasks A–F, each final model family, source code, and saved configuration. Training entry points evaluate validation and test in the same process. For the sampling 3 Task E E1/E2 protocol:

```bash
bash scripts/task_e.sh chronos2 dataset_sampling_3_prefix1024
```

Choose `chronos2`, `panda_zero_shot`, `toto2`, `itransformer`, or `tcn`. This creates a prepared cache and writes metrics under `results/task_e/sampling_3/reproduce_task_e_<model>/`. The supervised models run their saved base-and-resume sequence. Full reproduction requires externally supplied raw data, pretrained weights for foundation models, and significant compute.

## Citation

Paper citation pending; cite the final paper once its bibliographic details are available.
