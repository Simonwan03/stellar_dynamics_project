#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_ENV="${PANDA_CONDA_ENV:-stellar}"

export PYTHONNOUSERSITE=1
export PYTHONPATH="$ROOT:$ROOT/vendor/panda:${PYTHONPATH:-}"
export HF_HOME="${HF_HOME:-$ROOT/outputs/cache/huggingface}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$ROOT/outputs/logs/matplotlib}"

conda run --no-capture-output -n "$CONDA_ENV" python "$ROOT/scripts/train_unbound_classifier.py" \
  --config "$ROOT/configs/unbound_dataset.yaml" \
  "$@"
