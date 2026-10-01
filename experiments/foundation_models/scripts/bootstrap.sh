#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROJECT_ROOT="$(cd "$ROOT/../.." && pwd)"
CONDA_ENV="${PANDA_CONDA_ENV:-stellar}"
PYTHON_BIN="$(conda run -n "$CONDA_ENV" python -c 'import sys; print(sys.executable)')"

conda run -n "$CONDA_ENV" uv pip sync "$PROJECT_ROOT/requirements.lock" \
  --python "$PYTHON_BIN" --torch-backend cu130 --require-hashes

echo "Environment synced: $CONDA_ENV"
echo "Panda source: $ROOT/vendor/panda"
