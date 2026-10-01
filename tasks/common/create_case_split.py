from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tasks.common.case_split import (  # noqa: E402
    default_case_split_manifest_path,
    ensure_case_split_manifest,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create the shared Task A/B/C system split manifest.")
    parser.add_argument("--dataset-dir", type=Path, default=Path("dataset_v3"))
    parser.add_argument("--sampling-id", type=int, default=2)
    parser.add_argument("--group-size", type=int, default=32)
    parser.add_argument("--time-horizon", type=float, default=10000.0)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    labels_path = args.dataset_dir / f"completed_systems_sampling_{args.sampling_id}.txt"
    manifest_path = args.output or default_case_split_manifest_path(
        args.dataset_dir,
        args.sampling_id,
        args.group_size,
        args.time_horizon,
        args.seed,
        args.train_ratio,
        args.val_ratio,
        args.test_ratio,
    )
    manifest = ensure_case_split_manifest(
        labels_path,
        manifest_path,
        group_size=args.group_size,
        time_horizon=args.time_horizon,
        seed=args.seed,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        overwrite=args.overwrite,
    )
    counts = manifest["split"].value_counts().to_dict()
    print(f"Wrote common case split: {manifest_path}")
    print(f"Systems: {len(manifest)}; counts: {counts}")


if __name__ == "__main__":
    main()
