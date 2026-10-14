# Learning What Is Predictable in Chaotic Three-Body Dynamics

Official codebase for our study of **learnability in chaotic three-body dynamics**.

We use hierarchical three-body systems near the stability boundary as a controlled testbed to study how predictability depends on the **prediction target, available observations, physical representation, and model class**.

## Overview

We study a hierarchy of prediction problems on hierarchical three-body systems with controlled perturbations:

- **Stability prediction** from initial conditions and short trajectory prefixes (Tasks A and B)
- **Survival prediction** over multiple time horizons (Task C)
- **Finite-time divergence** prediction (Task D)
- **Trajectory forecasting** at short and long horizons (Task E)
- **Partial-observation robustness** under missing trajectory data (Task F)

Our experiments compare standard supervised models, neural sequence models, and pretrained time-series foundation models. Coarse-grained stability can often be predicted from limited observations, while detailed trajectory forecasting is more sensitive to chaotic error growth. The [research record](record_en.md) contains task definitions, experiment settings, and result tables.

## Repository Structure

```text
.
├── tasks/                         # Data processing, models, training, and evaluation
├── scripts/                       # Task A–F command-line entry points
├── configs/                       # Shared split manifests and configurations
├── experiments/foundation_models/ # Foundation-model integrations
├── results/                       # Saved experimental results
├── figures/                       # Research figures
├── docs/                          # Reproduction and artifact documentation
└── record.md                      # Detailed research record
```

See the [task code map](tasks/README.md) and [reproduction map](docs/reproduction.md) for the implementation and saved configurations of each experiment.

## Dataset

The dataset comprises approximately **160,000 hierarchical three-body trajectories** from 5,000 base systems, with 32 controlled variations per system. Trajectories include orbital and state representations, completion information, and the targets used by the prediction tasks. The [research record](record.md) describes the sampling 2 files and fields.

The simulation data is **not tracked in this repository**, and there is no verified public download location here. Supply the data separately: sampling 2 experiments use `dataset_v3/`, while current sampling 3 experiments use `dataset_sampling_3_prefix1024/` or a path passed with `--dataset-dir`. The HDF5 shards and completion table must have matching sampling IDs. See the [reproduction map](docs/reproduction.md) for the required inputs and split manifests.

## Experiments

The main entry points are `scripts/task_a.py` through `scripts/task_f.py`. For example, to run the sampling 3 short-prefix stability experiment after supplying its data:

```bash
python -m tasks.common.preflight --task B --dataset-dir /path/to/sampling3
python scripts/task_b.py --resolution high \
  --dataset-dir /path/to/sampling3 \
  --prefix-len 128 --models mlp,cnn,tcn \
  --split-manifest configs/splits/common_case_split_sampling_3_g32_h10000_r70-15-15_seed42.csv \
  --output-dir results/task_b/sampling_3/reproduce_task_b
```

The [reproduction map](docs/reproduction.md) links every task to its commands, reported settings, and saved results. Full reproduction requires the external simulation data; foundation-model experiments also require their pretrained weights.

## Installation

The supported environment targets Linux x86_64, Python 3.12.12, and PyTorch CUDA 13.0 wheels. From the repository root:

```bash
conda env create -f environment.yml
conda activate stellar
git submodule update --init --recursive
uv pip sync requirements.lock --python "$CONDA_PREFIX/bin/python" --torch-backend cu130 --require-hashes
python -m pip check
python scripts/check_environment.py
```

`requirement.txt` is the dependency input; `requirements.lock` pins the resolved packages and wheel hashes. Installation and model checkpoints still need validation on the target GPU machine.

## Citation

Paper citation details will be added when the manuscript is available.

## License

This repository does not currently include a license file.
