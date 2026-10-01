#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_ENV="${PANDA_CONDA_ENV:-stellar}"
export PYTHONNOUSERSITE=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-}"
export ACCELERATE_USE_CPU="${ACCELERATE_USE_CPU:-true}"
export HF_HOME="${HF_HOME:-$ROOT/outputs/cache/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$ROOT/outputs/cache/huggingface/transformers}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$ROOT/outputs/cache/huggingface/hub}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$ROOT/outputs/logs/matplotlib}"

FINETUNE_JSON="$ROOT/outputs/checkpoints/finetune_orbital15_small_latest.json"
SCRATCH_JSON="$ROOT/outputs/checkpoints/scratch_orbital15_small_latest.json"

if [[ -f "$FINETUNE_JSON" ]]; then
  FINETUNE_CKPT="$(conda run -n "$CONDA_ENV" python -c 'import json,sys; print(json.load(open(sys.argv[1]))["checkpoint_final"])' "$FINETUNE_JSON")"
  PYTHONPATH="$ROOT:$ROOT/vendor/panda:${PYTHONPATH:-}" conda run -n "$CONDA_ENV" python "$ROOT/scripts/run_panda_eval.py" --config "$ROOT/configs/eval.yaml" --checkpoint "$FINETUNE_CKPT" --run-name finetune_orbital15_small_test --split test
fi

if [[ -f "$SCRATCH_JSON" ]]; then
  SCRATCH_CKPT="$(conda run -n "$CONDA_ENV" python -c 'import json,sys; print(json.load(open(sys.argv[1]))["checkpoint_final"])' "$SCRATCH_JSON")"
  PYTHONPATH="$ROOT:$ROOT/vendor/panda:${PYTHONPATH:-}" conda run -n "$CONDA_ENV" python "$ROOT/scripts/run_panda_eval.py" --config "$ROOT/configs/eval.yaml" --checkpoint "$SCRATCH_CKPT" --run-name scratch_orbital15_small_test --split test
fi

PYTHONPATH="$ROOT:$ROOT/vendor/panda:${PYTHONPATH:-}" conda run -n "$CONDA_ENV" python "$ROOT/scripts/summarize_results.py"
