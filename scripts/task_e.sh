#!/usr/bin/env bash
# Sampling 3 first-window Task E E1/E2 protocol.
set -euo pipefail
if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "Usage: bash scripts/task_e.sh {chronos2|panda_zero_shot|toto2|itransformer|tcn} [dataset_dir]" >&2
  exit 2
fi
model=$1
data_dir=${2:-dataset_sampling_3_prefix1024}
case "$model" in
  chronos2|panda_zero_shot|toto2|itransformer|tcn) ;;
  *) echo "Unsupported reported model: $model" >&2; exit 2 ;;
esac
python -m tasks.common.preflight --task E --dataset-dir "$data_dir"
cache_dir=results/task_e/sampling_3/reproduce_task_e_prepared
if [[ ! -f "$cache_dir/manifest.json" ]]; then
  python -m tasks.task_e.prepare \
    --cache-dir "$cache_dir" \
    --data-glob "$data_dir/results_batch_*_sampling_3.hdf5" \
    --completion-table "$data_dir/completed_systems_sampling_3.txt" \
    --representation orbital15 --context-len 512 --pred-len 128 \
    --stride 20 --split-seed 42 --group-size 32 \
    --max-windows-per-trajectory 1 --window-sampling sequential \
    --drop-nan-windows
fi
run_model() {
  local run_dir=$1
  shift
  local model_download_args=()
  if [[ "${STELLAR_ALLOW_MODEL_DOWNLOAD:-0}" == 1 ]]; then
    model_download_args=(--no-hf-local-files-only)
  fi
  python -m tasks.task_e \
    --prepared-cache "$cache_dir" --model "$model" \
    --data_glob "$data_dir/results_batch_*_sampling_3.hdf5" \
    --completion_table "$data_dir/completed_systems_sampling_3.txt" \
    --representation orbital15 --context-len 512 --pred-len 128 \
    --stride 20 --split_seed 42 --group_size 32 --seed 42 \
    --max_windows_per_trajectory 1 \
    --train-windows-per-trajectory 1 --val-windows-per-trajectory 1 \
    --test-windows-per-trajectory 1 --rolling-windows-per-trajectory 1 \
    --run-rolling --rolling-blocks 4 --bootstrap-samples 1000 \
    --output_dir "$run_dir" "${model_download_args[@]}" "$@"
}
case "$model" in
  chronos2)
    run_model results/task_e/sampling_3/reproduce_task_e_chronos2 --foundation_batch_size 32
    ;;
  panda_zero_shot|toto2)
    run_model "results/task_e/sampling_3/reproduce_task_e_${model}"
    ;;
  itransformer)
    base_dir=results/task_e/sampling_3/reproduce_task_e_itransformer_base
    run_model "$base_dir" --batch_size 512 --epochs 30
    run_model results/task_e/sampling_3/reproduce_task_e_itransformer --batch_size 512 --epochs 3 \
      --resume_from "$base_dir/best_model.pt"
    ;;
  tcn)
    base_dir=results/task_e/sampling_3/reproduce_task_e_tcn_base
    run_model "$base_dir" --batch_size 512 --epochs 25
    run_model results/task_e/sampling_3/reproduce_task_e_tcn --batch_size 512 --epochs 3 \
      --resume_from "$base_dir/best_model.pt"
    ;;
esac
