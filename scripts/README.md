# Task Experiment Scripts

These entrypoints convert the original task notebooks into reusable command-line jobs.
Run them from the repository root. On the current machine the examples use the `stellar`
conda environment; on AWS, use whichever environment contains `h5py`, `numpy`, `pandas`,
`scikit-learn`, `matplotlib`, and `torch`.

## Task A

Group-level stable-at-horizon classification from tokenized initial conditions.

```bash
conda run -n stellar python scripts/task_a_stability_classifier.py \
  --output-dir results/task_a \
  --models mlp,xgboost \
  --epochs 20
```

Use `--regroup --overwrite-tokenized` if you need to rebuild
`dataset_v3/ICs_sampling_2_groupof32.hdf5`.

## Task B

Hard-label classification from early trajectory prefixes.

```bash
conda run -n stellar python scripts/task_b_prefix_classifiers.py \
  --output-dir results/task_b \
  --prefix-len 100 \
  --models svm,cnn,tcn,attention,ic_prefix \
  --epochs 20
```

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
  --time-stride 2 \
  --models baseline,direct,traj,dual \
  --epochs 40
```

## Task D

System-level semimajor-axis divergence regression. This wraps the existing
`src/task_d_lyapunov.py` pipeline.

```bash
conda run -n stellar python scripts/task_d_lyapunov_regression.py \
  --output-dir outputs/task_d_lyapunov \
  --mlp-epochs 80 \
  --cnn-epochs 80
```

## Task E

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

Pretrained foundation-model zero-shot trajectory baselines. This treats each
position/velocity channel as an independent univariate series and evaluates
TimesFM, Chronos, PatchTST, plus a persistence baseline on sampled windows.

```bash
conda run -n stellar python scripts/task_e_foundation_forecasters.py \
  --models naive,timesfm,chronos,patchtst \
  --num-windows 256 \
  --context-len 64 \
  --pred-len 8 \
  --batch-size 128 \
  --device cuda
```

`timesfm` and `chronos` are optional dependencies. If they are not installed,
the script records a per-model error JSON and continues with the remaining
models.
