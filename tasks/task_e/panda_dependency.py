from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PANDA_VENDOR_ROOT = ROOT / "experiments" / "panda_threebody" / "vendor" / "panda"


def panda_vendor_root() -> Path:
    """Return the Panda source checkout used for pretrained model loading."""
    override = os.environ.get("STELLAR_PANDA_VENDOR_ROOT")
    if override:
        return Path(override).expanduser().resolve()
    return DEFAULT_PANDA_VENDOR_ROOT


def ensure_panda_importable() -> Path:
    vendor_root = panda_vendor_root()
    if str(vendor_root) not in sys.path:
        sys.path.insert(0, str(vendor_root))
    return vendor_root


def load_panda_pretrained(
    model_id_or_path: str,
    device: str | torch.device,
    *,
    local_files_only: bool = True,
) -> torch.nn.Module:
    """Load the Panda paper checkpoint using the vendored Panda model code.

    `model_id_or_path` can be a Hugging Face repo id such as
    `GilpinLab/panda-72M` or a local directory containing the same pretrained
    checkpoint files. Local checkpoint directories should live outside Git.
    """
    vendor_root = ensure_panda_importable()
    try:
        from panda.patchtst.patchtst import PatchTSTForPrediction  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "Panda is not importable. Initialize the submodule with "
            "`git submodule update --init --recursive`, or set "
            "STELLAR_PANDA_VENDOR_ROOT to a Panda source checkout. "
            f"Expected default path: {vendor_root}"
        ) from exc

    try:
        model = PatchTSTForPrediction.from_pretrained(
            model_id_or_path,
            local_files_only=local_files_only,
        ).to(device)
    except Exception as exc:
        raise RuntimeError(
            format_panda_checkpoint_error(model_id_or_path, local_files_only, exc)
        ) from exc

    model.eval()
    return model


def format_panda_checkpoint_error(model_id_or_path: str, local_files_only: bool, exc: Exception) -> str:
    source = Path(model_id_or_path).expanduser()
    if source.exists():
        return f"Failed to load local Panda checkpoint at {source}: {exc}"
    if local_files_only:
        return (
            f"Panda checkpoint {model_id_or_path!r} was not available in the local Hugging Face cache. "
            "Populate the cache first, pass --no-hf-local-files-only to allow a download, "
            "or pass --panda-model-id /path/to/local/checkpoint."
        )
    return f"Failed to load Panda checkpoint {model_id_or_path!r}: {exc}"


def panda_dependency_metadata(model_id_or_path: str, local_files_only: bool) -> dict[str, Any]:
    vendor_root = panda_vendor_root()
    return {
        "panda_vendor_root": str(vendor_root),
        "panda_model_id_or_path": model_id_or_path,
        "hf_local_files_only": bool(local_files_only),
        "panda_vendor_present": vendor_root.exists(),
    }
