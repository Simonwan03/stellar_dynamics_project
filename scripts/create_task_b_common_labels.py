from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Align Task B labels to the common complete-system manifest."
    )
    parser.add_argument(
        "--source-labels",
        type=Path,
        default=Path("results/task_b_prefix_ablation/prefix_512/labels_filtered.csv"),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("dataset_v3/common_case_split_sampling_2_g32_h10000_r70-15-15_seed42.csv"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/task_b_common/prefix_512/labels_filtered.csv"),
    )
    parser.add_argument("--group-size", type=int, default=32)
    args = parser.parse_args()

    if args.group_size <= 0:
        raise ValueError("--group-size must be positive.")
    labels = pd.read_csv(args.source_labels)
    manifest = pd.read_csv(args.manifest)
    required_label_columns = {"index", "num_batch", "time_unbound", "diagnosis"}
    missing_labels = sorted(required_label_columns - set(labels.columns))
    if missing_labels:
        raise ValueError(f"Source labels are missing columns: {missing_labels}")
    if "raw_case_id" not in manifest or "split" not in manifest:
        raise ValueError("The common manifest must contain raw_case_id and split columns.")

    labels = labels.copy()
    labels["raw_case_id"] = labels["index"].to_numpy(dtype=np.int64) // args.group_size
    common_ids = manifest["raw_case_id"].to_numpy(dtype=np.int64)
    labels = labels[labels["raw_case_id"].isin(common_ids)].copy().reset_index(drop=True)
    split_by_case = dict(
        zip(manifest["raw_case_id"].astype(np.int64), manifest["split"].astype(str))
    )
    labels["split"] = labels["raw_case_id"].map(split_by_case)

    if labels["split"].isna().any():
        raise RuntimeError("Some aligned labels have no manifest split.")
    if labels["raw_case_id"].nunique() != len(manifest):
        raise RuntimeError(
            f"Expected {len(manifest)} complete systems, got {labels['raw_case_id'].nunique()}."
        )
    counts = labels.groupby("raw_case_id").size()
    if not np.all(counts.to_numpy(dtype=np.int64) == args.group_size):
        raise RuntimeError("Aligned labels do not contain exactly group_size rows per system.")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    labels.to_csv(args.output, index=False)
    split_indices = {
        name: np.flatnonzero(labels["split"].to_numpy() == name).astype(np.int64)
        for name in ("train", "val", "test")
    }
    np.savez(
        args.output.parent / "split_indices.npz",
        train_idx=split_indices["train"],
        val_idx=split_indices["val"],
        test_idx=split_indices["test"],
        case_ids=labels["raw_case_id"].to_numpy(dtype=np.int64),
    )
    metadata = {
        "source_labels": str(args.source_labels),
        "manifest": str(args.manifest),
        "group_size": int(args.group_size),
        "rows": int(len(labels)),
        "cases": int(labels["raw_case_id"].nunique()),
        "split_counts": {
            name: int((labels["split"] == name).sum()) for name in ("train", "val", "test")
        },
        "split_indices": str(args.output.parent / "split_indices.npz"),
    }
    args.output.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
