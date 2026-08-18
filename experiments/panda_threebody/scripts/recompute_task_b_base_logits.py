#!/usr/bin/env python3
"""Recompute aligned probabilities from a saved frozen linear Task B probe."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_PREFIX_DIR = REPO_ROOT / "results/task_b_prefix_ablation/prefix_512"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-cache", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--prefix-dir", type=Path, default=DEFAULT_PREFIX_DIR)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=4096)
    args = parser.parse_args()

    labels = pd.read_csv(args.prefix_dir / "labels_filtered.csv")
    split_file = np.load(args.prefix_dir / "split_indices.npz")
    split_membership = np.full(len(labels), "", dtype=object)
    for split in ("train", "val", "test"):
        split_membership[split_file[f"{split}_idx"]] = split
    if np.any(split_membership == ""):
        raise ValueError("Task B split does not cover every row.")

    features = np.load(args.feature_cache, mmap_mode="r")
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    feature_dim = int(checkpoint["feature_dim"])
    if features.shape != (len(labels), feature_dim):
        raise ValueError(f"Feature/checkpoint shape mismatch: {features.shape} vs {feature_dim}")
    state = checkpoint["model_state_dict"]
    weight = state["classifier.weight"].detach().to(dtype=torch.float32)
    bias = state["classifier.bias"].detach().to(dtype=torch.float32)
    if tuple(weight.shape) != (1, feature_dim):
        raise ValueError(f"Expected a linear classifier weight, got {tuple(weight.shape)}")

    logits = np.empty(len(labels), dtype=np.float32)
    with torch.no_grad():
        for start in range(0, len(labels), args.batch_size):
            stop = min(start + args.batch_size, len(labels))
            batch = torch.from_numpy(np.asarray(features[start:stop], dtype=np.float32))
            logits[start:stop] = (batch @ weight.T + bias).squeeze(1).numpy()
            if start == 0 or stop == len(labels) or (start // args.batch_size) % 10 == 0:
                print(f"logits {stop}/{len(labels)}", flush=True)

    probability_unbound = 1.0 / (1.0 + np.exp(-np.clip(logits, -40.0, 40.0)))
    output = labels[["index", "num_batch", "time_unbound", "diagnosis"]].copy()
    output["task_b_row"] = np.arange(len(labels), dtype=np.int64)
    output["row_in_batch"] = labels["row_in_batch"].to_numpy(dtype=np.int64)
    output["split"] = split_membership
    output["target_unbound"] = (labels["time_unbound"].to_numpy(dtype=np.float32) < 10000.0).astype(np.int8)
    output["logit_unbound"] = logits
    output["prob_unbound"] = probability_unbound
    output["prob_stable"] = 1.0 - probability_unbound
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False)
    print(f"Wrote {len(output)} aligned rows to {args.output}", flush=True)


if __name__ == "__main__":
    main()
