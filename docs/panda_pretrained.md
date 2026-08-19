# Panda Pretrained Integration

This project uses Panda primarily as a pretrained foundation forecaster.  The
Panda source code stays external as a Git submodule, while this repository owns
the three-body data preparation, adapters, evaluation scripts, and results.

## Dependency Layout

- Panda source: `experiments/panda_threebody/vendor/panda`
- Main forecasting entry: `tasks/task_e/main.py`
- Legacy isolated Panda experiments: `experiments/panda_threebody/scripts`
- Shared Panda loader: `tasks/task_e/panda_dependency.py`

Initialize the Panda source after cloning:

```bash
git submodule update --init --recursive
```

If you keep Panda somewhere else, point the loader to it:

```bash
export STELLAR_PANDA_VENDOR_ROOT=/path/to/panda
```

## Checkpoints

The default pretrained checkpoint is:

```text
GilpinLab/panda-72M
```

Keep pretrained weights out of Git.  Use either the local Hugging Face cache or
a local ignored checkpoint directory such as:

```text
checkpoints/panda/GilpinLab--panda-72M/
external_checkpoints/panda/
```

The `.gitignore` excludes these directories and common model artifact formats.

## Recommended Commands

Zero-shot Panda evaluation on the project benchmark:

```bash
conda run -n stellar python -m tasks.task_e \
  --data_glob "dataset_v3/results_batch_*_sampling_2.hdf5" \
  --representation orbital15 \
  --model panda_zero_shot \
  --context_len 200 \
  --pred_len 50 \
  --num_eval_windows 256 \
  --panda_model_id GilpinLab/panda-72M \
  --hf-local-files-only \
  --output_dir runs/task_e_panda_zero_shot
```

The same entrypoint also accepts the 18-channel Cartesian state representation:

```bash
conda run -n stellar python -m tasks.task_e \
  --data_glob "dataset_v3/results_batch_*_sampling_2.hdf5" \
  --representation raw_state18 \
  --model panda_zero_shot \
  --context_len 200 \
  --pred_len 50 \
  --panda_model_id GilpinLab/panda-72M \
  --hf-local-files-only \
  --output_dir runs/task_e_panda_zero_shot_raw18
```

Panda shares its channel embedding and forecast head, so the checkpoint's
nominal `num_input_channels=1` does not restrict inference to one channel.
Both the 15-channel `orbital15` and 18-channel `raw_state18` representations
are passed as multivariate windows.

Allow a first-time Hugging Face download:

```bash
conda run -n stellar python -m tasks.task_e \
  --representation orbital15 \
  --model panda_zero_shot \
  --panda_model_id GilpinLab/panda-72M \
  --no-hf-local-files-only \
  --output_dir runs/task_e_panda_zero_shot
```

Use a local checkpoint directory:

```bash
conda run -n stellar python -m tasks.task_e \
  --representation orbital15 \
  --model panda_zero_shot \
  --panda_model_id checkpoints/panda/GilpinLab--panda-72M \
  --hf-local-files-only \
  --output_dir runs/task_e_panda_zero_shot_local
```

Historical adapter example (not part of the current Task E protocol; the CLI now rejects `panda_adapter`):

```bash
conda run -n stellar python -m tasks.task_e \
  --representation orbital15 \
  --model panda_adapter \
  --context_len 256 \
  --pred_len 64 \
  --num_train_windows_for_adapter 4096 \
  --freeze_panda \
  --panda_model_id GilpinLab/panda-72M \
  --hf-local-files-only \
  --output_dir runs/task_e_panda_adapter
```

When Panda is frozen, `best_model.pt` stores only the temporal adapter state;
the 72M frozen backbone is not duplicated. Restoring the run first loads
`panda_model_id` from the Hugging Face cache (or a local checkpoint path) and
then applies the adapter-only state. If `--no-freeze_panda` is used, the full
fine-tuned model state is saved instead.

## Role of `experiments/panda_threebody`

Treat `experiments/panda_threebody` as a historical/isolated Panda experiment
area.  New comparisons for the paper should prefer the main Task E entrypoints
above, so forecasting results share the same data split, scaler, metrics, and
output layout as the other baselines.
