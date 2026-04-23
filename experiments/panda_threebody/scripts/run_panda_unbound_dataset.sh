#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_ENV="${PANDA_CONDA_ENV:-stellar}"

export PYTHONNOUSERSITE=1
export PYTHONPATH="$ROOT:$ROOT/vendor/panda:${PYTHONPATH:-}"
export HF_HOME="${HF_HOME:-$ROOT/outputs/cache/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$ROOT/outputs/cache/huggingface/transformers}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$ROOT/outputs/cache/huggingface/hub}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$ROOT/outputs/logs/matplotlib}"

conda run --no-capture-output -n "$CONDA_ENV" python "$ROOT/scripts/train_panda_unbound_classifier.py" \
  --config "$ROOT/configs/panda_unbound_dataset.yaml" \
  "$@"
