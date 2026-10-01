"""Compatibility entry point; implementation lives in chronos2/train_chronos2_task_b_linear.py."""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.foundation_models.chronos2.train_chronos2_task_b_linear import *  # noqa: F401,F403,E402

if __name__ == "__main__":
    main()
