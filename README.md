# Stellar dynamics learning experiments

This repository contains research code for predicting stability, survival, divergence time, and future motion in perturbed three-body systems. Start with the [manuscript reproduction map](docs/reproduction.md). The longer [research log](readme.md) preserves earlier experiments and result tables.

## Install

Use Python 3.12. The committed `environment.yml` and `requirement.txt` are historical environment snapshots; `requirements-foundation.txt` adds optional pretrained model dependencies. A starting setup is:

```bash
conda env create -f environment.yml
conda activate stellar
git submodule update --init --recursive
python -m pytest -q
```

PyTorch GPU builds must match the local CUDA runtime. Run entry points from the repository root. Each task exposes `python -m tasks.task_a --help` through `python -m tasks.task_f --help`.

## Data and code

The simulation data is **not included**, and this repository provides no verified public download location. Supply sampling 3 shards as `dataset_sampling_3_prefix1024/results_batch_*_sampling_3.hdf5` plus `completed_systems_sampling_3.txt`, or pass the path to your own copy. Earlier sampling 2 experiments use `dataset_v3/`; in this checkout that path is a machine-specific symlink. The completion table and shards must have the same sampling ID.

`tasks/` owns preprocessing, models and evaluation; `scripts/` holds high-resolution and comparison drivers; `experiments/panda_threebody/` holds frozen-encoder adapters and the Panda submodule. `results/` and `runs/` contain generated artifacts and saved provenance for reported experiments. New outputs are ignored by Git. `legacy/` labels historical code; `tests/` contains basic checks.

## Reproduce results

[The reproduction map](docs/reproduction.md) connects Tasks A–F, each final model family, source code, and saved configuration. Training entry points evaluate validation and test in the same process. For the sampling 3 Task E E1/E2 protocol:

```bash
bash scripts/reproduce/task_e.sh chronos2 dataset_sampling_3_prefix1024
```

Choose `chronos2`, `panda_zero_shot`, `toto2`, `itransformer`, or `tcn`. This creates a prepared cache and writes metrics under `runs/reproduce_task_e_<model>/`. The supervised models run their saved base-and-resume sequence. Full reproduction requires the absent raw data, pretrained weights for foundation models, and significant compute.

## Citation

Paper citation pending; cite the final paper once its bibliographic details are available.
