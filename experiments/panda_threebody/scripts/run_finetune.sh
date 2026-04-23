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
PYTHONPATH="$ROOT:$ROOT/vendor/panda:${PYTHONPATH:-}" conda run -n "$CONDA_ENV" python "$ROOT/scripts/run_train.py" --config configs/finetune.yaml
