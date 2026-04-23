from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = EXPERIMENT_ROOT.parents[1]
VENDOR_PANDA = EXPERIMENT_ROOT / "vendor" / "panda"
VENV_PYTHON = EXPERIMENT_ROOT / ".venv" / "bin" / "python"
DEFAULT_CONDA_ENV = os.environ.get("PANDA_CONDA_ENV", "stellar")


def load_yaml(path: Path) -> dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f)


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def nowstamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def choose_device(preference: str = "auto") -> str:
    if preference != "auto":
        return preference
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def python_cmd() -> list[str]:
    override = os.environ.get("PANDA_PYTHON_CMD")
    if override:
        return shlex.split(override)
    if VENV_PYTHON.exists():
        return [str(VENV_PYTHON)]
    return ["conda", "run", "-n", DEFAULT_CONDA_ENV, "python"]


def hydra_list(values: list[Any]) -> str:
    return "[" + ",".join(str(v) for v in values) + "]"


def write_json(path: Path, payload: Any) -> None:
    ensure_dir(path.parent)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def find_latest_checkpoint(run_name: str, checkpoint_root: Path) -> Path:
    candidates = sorted(checkpoint_root.glob(f"{run_name}-*"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        raise FileNotFoundError(f"No checkpoint run directory found for pattern {run_name}-* under {checkpoint_root}")
    checkpoint = candidates[0] / "checkpoint-final"
    if not checkpoint.exists():
        raise FileNotFoundError(f"Expected checkpoint-final under {candidates[0]}")
    return checkpoint


def stream_subprocess(cmd: list[str], log_path: Path, env: dict[str, str] | None = None, cwd: Path | None = None) -> None:
    ensure_dir(log_path.parent)
    merged_env = os.environ.copy()
    merged_env["PYTHONPATH"] = f"{EXPERIMENT_ROOT}:{VENDOR_PANDA}:{merged_env.get('PYTHONPATH', '')}"
    merged_env["PYTHONNOUSERSITE"] = "1"
    merged_env.setdefault("CUDA_VISIBLE_DEVICES", "")
    merged_env.setdefault("ACCELERATE_USE_CPU", "true")
    if not merged_env.get("PANDA_KEEP_DISTRIBUTED_ENV"):
        for key in ["LOCAL_RANK", "RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT"]:
            merged_env.pop(key, None)
    merged_env.setdefault("MPLCONFIGDIR", str(EXPERIMENT_ROOT / "outputs" / "logs" / "matplotlib"))
    merged_env.setdefault("HF_HOME", str(EXPERIMENT_ROOT / "outputs" / "cache" / "huggingface"))
    merged_env.setdefault("TRANSFORMERS_CACHE", str(EXPERIMENT_ROOT / "outputs" / "cache" / "huggingface" / "transformers"))
    merged_env.setdefault("HF_HUB_CACHE", str(EXPERIMENT_ROOT / "outputs" / "cache" / "huggingface" / "hub"))
    if env:
        merged_env.update(env)
    with open(log_path, "w") as log_file:
        process = subprocess.Popen(
            cmd,
            cwd=str(cwd) if cwd else None,
            env=merged_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            sys.stdout.write(line)
            log_file.write(line)
        ret = process.wait()
        if ret != 0:
            raise subprocess.CalledProcessError(ret, cmd)
