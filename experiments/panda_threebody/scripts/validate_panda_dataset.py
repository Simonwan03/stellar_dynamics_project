from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from gluonts.dataset.common import FileDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "vendor" / "panda"))

from adapters.project_data import get_export_root, load_yaml
from panda.dataset import MultivariateTimeSeriesDataset


def inspect_split(split_dir: Path, context_length: int, prediction_length: int) -> None:
    system_dirs = sorted([p for p in split_dir.iterdir() if p.is_dir()])
    print(f"## split={split_dir.name}")
    print("system_names:", [p.name for p in system_dirs])
    all_files = sorted(split_dir.glob("*/*.arrow"))
    print("num_arrow_files:", len(all_files))
    if not all_files:
        return

    bad_nan = 0
    bad_zero = 0
    bad_short = 0
    for file_path in all_files:
        entry = next(iter(FileDataset(path=file_path, freq="h", one_dim_target=False)))
        target = np.asarray(entry["target"])
        if np.isnan(target).any():
            bad_nan += 1
        if np.allclose(target, 0.0):
            bad_zero += 1
        if target.shape[-1] < context_length + prediction_length:
            bad_short += 1

    first_entry = next(iter(FileDataset(path=all_files[0], freq="h", one_dim_target=False)))
    target = np.asarray(first_entry["target"])
    print("single_target_shape:", tuple(target.shape))
    dataset = MultivariateTimeSeriesDataset(
        datasets=[FileDataset(path=file_path, freq="h", one_dim_target=False) for file_path in all_files[: min(8, len(all_files))]],
        probabilities=[1.0 / min(8, len(all_files))] * min(8, len(all_files)),
        context_length=context_length,
        prediction_length=prediction_length,
        num_test_instances=1,
        window_style="single",
        window_start=context_length,
        model_type="predict",
        mode="test",
    )
    sample = next(iter(dataset))
    print("windowed_past_shape:", tuple(sample["past_values"].shape))
    print("windowed_future_shape:", tuple(sample["future_values"].shape))
    print("nan_files:", bad_nan)
    print("all_zero_files:", bad_zero)
    print("too_short_files:", bad_short)
    print()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/export.yaml")
    parser.add_argument("--representation", default=None)
    args = parser.parse_args()

    exp_root = Path(__file__).resolve().parents[1]
    config = load_yaml(exp_root / args.config)
    representation = args.representation or config.get("representation", "orbital15")
    export_root = get_export_root(config, representation=representation)
    manifest_dir = export_root / "manifests"

    print("Export root:", export_root)
    for split in ["train", "val", "test"]:
        manifest_path = manifest_dir / f"{split}_manifest.csv"
        if manifest_path.exists():
            manifest = pd.read_csv(manifest_path)
            print(f"{split}_manifest_rows:", len(manifest))
            print(f"{split}_systems:", sorted(manifest['system_name'].unique().tolist()))
        split_dir = export_root / split
        if split_dir.exists():
            inspect_split(split_dir, int(config["context_length"]), int(config["prediction_length"]))


if __name__ == "__main__":
    main()
