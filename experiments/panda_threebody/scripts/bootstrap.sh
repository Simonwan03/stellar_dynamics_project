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

conda run -n "$CONDA_ENV" python -m pip install \
  pyyaml pyarrow "gluonts[arrow]" hydra-core accelerate wandb click tensorboard orjson statsmodels \
  "transformers==4.40.2" "huggingface_hub>=0.19,<1" "tokenizers==0.19.1" "safetensors>=0.4" \
  "dysts @ git+https://github.com/williamgilpin/dysts@master"

echo "Bootstrap complete"
echo "Runtime: conda run -n $CONDA_ENV python"
echo "Panda source: $ROOT/vendor/panda"
