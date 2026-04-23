#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_ENV="${PANDA_CONDA_ENV:-stellar}"

PY="conda run -n $CONDA_ENV python"
export PYTHONPATH="$ROOT:$ROOT/vendor/panda:${PYTHONPATH:-}"
export PYTHONNOUSERSITE=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-}"
export ACCELERATE_USE_CPU="${ACCELERATE_USE_CPU:-true}"
export HF_HOME="${HF_HOME:-$ROOT/outputs/cache/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$ROOT/outputs/cache/huggingface/transformers}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$ROOT/outputs/cache/huggingface/hub}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$ROOT/outputs/logs/matplotlib}"

$PY "$ROOT/scripts/audit_repo.py" --write
$PY "$ROOT/scripts/export_to_panda_arrow.py" --config "$ROOT/configs/export.yaml" --representation orbital15
$PY "$ROOT/scripts/validate_panda_dataset.py" --config "$ROOT/configs/export.yaml" --representation orbital15
$PY "$ROOT/scripts/run_zero_shot.py" --config "$ROOT/configs/zero_shot.yaml"
bash "$ROOT/scripts/run_finetune.sh"
bash "$ROOT/scripts/run_scratch.sh"
bash "$ROOT/scripts/run_eval.sh"
$PY "$ROOT/scripts/summarize_results.py"
