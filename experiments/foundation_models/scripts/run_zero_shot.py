from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import EXPERIMENT_ROOT
from adapters.project_data import load_yaml


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/zero_shot.yaml")
    args = parser.parse_args()

    cfg = load_yaml(EXPERIMENT_ROOT / args.config)
    command = [
        sys.executable,
        str(EXPERIMENT_ROOT / "scripts" / "run_panda_eval.py"),
        "--config",
        str(EXPERIMENT_ROOT / args.config),
        "--checkpoint",
        str(cfg["checkpoint"]),
        "--run-name",
        str(cfg["run_name"]),
        "--split",
        "test",
    ]
    print("Running zero-shot:", " ".join(command))
    env = os.environ.copy()
    env["PYTHONNOUSERSITE"] = "1"
    env.setdefault("CUDA_VISIBLE_DEVICES", "")
    env.setdefault("ACCELERATE_USE_CPU", "true")
    env.setdefault("PYTHONPATH", f"{EXPERIMENT_ROOT}:{EXPERIMENT_ROOT / 'vendor' / 'panda'}")
    env.setdefault("HF_HOME", str(EXPERIMENT_ROOT / "outputs" / "cache" / "huggingface"))
    env.setdefault("TRANSFORMERS_CACHE", str(EXPERIMENT_ROOT / "outputs" / "cache" / "huggingface" / "transformers"))
    env.setdefault("HF_HUB_CACHE", str(EXPERIMENT_ROOT / "outputs" / "cache" / "huggingface" / "hub"))
    subprocess.run(command, check=True, cwd=EXPERIMENT_ROOT, env=env)


if __name__ == "__main__":
    main()
