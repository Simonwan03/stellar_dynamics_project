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
