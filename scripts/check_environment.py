"""Smoke-check the project's single locked Linux CUDA environment."""

from __future__ import annotations

import argparse
import sys


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--panda-checkpoint",
        help="Optional local Panda checkpoint or cached Hugging Face model ID to load on CPU",
    )
    args = parser.parse_args()

    if sys.version_info[:2] != (3, 12):
        raise RuntimeError(f"Expected Python 3.12, got {sys.version.split()[0]}")

    import torch

    if torch.__version__ != "2.10.0+cu130":
        raise RuntimeError(f"Expected torch 2.10.0+cu130, got {torch.__version__}")

    from chronos import Chronos2Pipeline  # noqa: F401
    from tirex2 import TimeseriesType, load_model  # noqa: F401
    from toto2 import Toto2Model  # noqa: F401
    from gluonts.dataset.common import ListDataset  # noqa: F401
    from tasks.task_e.panda_dependency import ensure_panda_importable, load_panda_pretrained

    vendor_root = ensure_panda_importable()
    from panda.patchtst.patchtst import PatchTSTForPrediction  # noqa: F401

    if args.panda_checkpoint:
        model = load_panda_pretrained(args.panda_checkpoint, "cpu", local_files_only=True)
        with torch.no_grad():
            context = torch.zeros(
                1, model.config.context_length, model.config.num_input_channels
            )
            prediction = model(past_values=context).prediction_outputs
        print(f"Panda prediction shape: {tuple(prediction.shape)}")

    print(f"Environment imports OK; Panda source: {vendor_root}")


if __name__ == "__main__":
    main()
