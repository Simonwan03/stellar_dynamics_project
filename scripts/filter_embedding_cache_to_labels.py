from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def labels_fingerprint(labels: pd.DataFrame) -> str:
    columns = ["index", "num_batch"]
    if "row_in_batch" in labels.columns:
        columns.append("row_in_batch")
    identity = labels[columns].to_numpy(dtype=np.int64, copy=True)
    return hashlib.sha256(identity.tobytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Filter an embedding cache into the common Task B label order."
    )
    parser.add_argument("--source-features", type=Path, required=True)
    parser.add_argument("--source-labels", type=Path, required=True)
    parser.add_argument("--target-labels", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prefix-steps", type=int, required=True)
    parser.add_argument("--pooling", required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--chunk-size", type=int, default=4096)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.prefix_steps <= 0 or args.chunk_size <= 0:
        raise ValueError("--prefix-steps and --chunk-size must be positive.")
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"Output already exists: {args.output}; use --overwrite to replace it.")

    source_labels = pd.read_csv(args.source_labels)
    target_labels = pd.read_csv(args.target_labels)
    for name, labels in (("source", source_labels), ("target", target_labels)):
        missing = sorted({"index", "num_batch"} - set(labels.columns))
        if missing:
            raise ValueError(f"{name} labels are missing columns: {missing}")

    source_features = np.load(args.source_features, mmap_mode="r")
    if source_features.ndim != 2 or source_features.shape[0] != len(source_labels):
        raise ValueError(
            f"Source feature shape {source_features.shape} does not match source labels {len(source_labels)}."
        )
    source_index = source_labels["index"].to_numpy(dtype=np.int64)
    if np.unique(source_index).size != source_index.size:
        raise ValueError("Source labels contain duplicate trajectory indices.")
    target_index = target_labels["index"].to_numpy(dtype=np.int64)
    source_rows = {int(index): row for row, index in enumerate(source_index)}
    try:
        selected_rows = np.asarray([source_rows[int(index)] for index in target_index], dtype=np.int64)
    except KeyError as exc:
        raise ValueError(f"Target trajectory index is absent from source labels: {exc.args[0]}") from exc
    if np.unique(selected_rows).size != selected_rows.size:
        raise ValueError("Target labels select duplicate source feature rows.")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    output = np.lib.format.open_memmap(
        args.output,
        mode="w+",
        dtype=source_features.dtype,
        shape=(len(target_labels), source_features.shape[1]),
    )
    for start in range(0, len(target_labels), args.chunk_size):
        stop = min(start + args.chunk_size, len(target_labels))
        output[start:stop] = source_features[selected_rows[start:stop]]
    output.flush()
    del output

    metadata = {
        "rows": int(len(target_labels)),
        "feature_dim": int(source_features.shape[1]),
        "prefix_steps": int(args.prefix_steps),
        "pooling": str(args.pooling),
        "label_fingerprint": labels_fingerprint(target_labels),
        "dtype": str(np.dtype(source_features.dtype)),
        "source_features": str(args.source_features),
        "source_labels": str(args.source_labels),
        "target_labels": str(args.target_labels),
    }
    if args.checkpoint is not None:
        metadata["checkpoint"] = str(args.checkpoint)
    args.output.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
