#!/usr/bin/env python3
"""Generate a resumable Chronos-2 feature cache for the Task B split."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd
import torch

EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = EXPERIMENT_ROOT.parents[1]
sys.path.insert(0, str(EXPERIMENT_ROOT))

from adapters.project_data import build_representation, required_keys_for_representation  # noqa: E402


DEFAULT_MODEL = EXPERIMENT_ROOT / "outputs/cache/huggingface/chronos-2"
DEFAULT_LABELS = REPO_ROOT / "results/task_b_prefix_ablation/prefix_512/labels_filtered.csv"
DEFAULT_OUTPUT = (
    EXPERIMENT_ROOT / "outputs/panda_unbound_dataset/chronos2_task_b_prefix96_reg_features.float16.npy"
)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def labels_fingerprint(labels: pd.DataFrame) -> str:
    columns = ["index", "num_batch"]
    if "row_in_batch" in labels.columns:
        columns.append("row_in_batch")
    identity = labels[columns].to_numpy(dtype=np.int64, copy=True)
    return hashlib.sha256(identity.tobytes()).hexdigest()


def feature_dim(pooling: str) -> int:
    if pooling in {"reg", "mean"}:
        return 15 * 768
    if pooling == "full":
        return 15 * 8 * 768
    raise ValueError(f"Unknown pooling mode: {pooling}")


def resolve_local_rows(labels: pd.DataFrame, full_labels: pd.DataFrame, positions: np.ndarray) -> np.ndarray:
    group = labels.iloc[positions]
    if "row_in_batch" in group.columns:
        return group["row_in_batch"].to_numpy(dtype=np.int64)
    batch_id = int(group.iloc[0]["num_batch"])
    batch_labels = full_labels.loc[full_labels["num_batch"] == batch_id].sort_values("index")
    index_to_local = {
        int(index): row for row, index in enumerate(batch_labels["index"].to_numpy(dtype=np.int64))
    }
    return np.asarray([index_to_local[int(index)] for index in group["index"]], dtype=np.int64)


def load_batch(
    labels: pd.DataFrame,
    full_labels: pd.DataFrame,
    dataset_dir: Path,
    batch_id: int,
    positions: np.ndarray,
    prefix_steps: int,
    sampling_id: int,
) -> np.ndarray:
    local_rows = resolve_local_rows(labels, full_labels, positions)
    read_order = np.argsort(local_rows)
    restore_order = np.argsort(read_order)
    shard = dataset_dir / f"results_batch_{int(batch_id):02d}_sampling_{sampling_id}.hdf5"
    data: dict[str, np.ndarray] = {}
    with h5py.File(shard, "r") as handle:
        for key in required_keys_for_representation("orbital15"):
            sorted_values = np.asarray(
                handle[key][local_rows[read_order], :prefix_steps, ...], dtype=np.float32
            )
            data[key] = sorted_values[restore_order]
    x = build_representation(data, "orbital15")
    expected = (len(positions), prefix_steps, 15)
    if x.shape != expected:
        raise ValueError(f"Expected batch shape {expected}, got {x.shape}")
    if not np.isfinite(x).all():
        raise ValueError(f"Found non-finite values in dataset batch {batch_id}.")
    return x


def pool_embeddings(embeddings: list[torch.Tensor], pooling: str) -> np.ndarray:
    if not embeddings:
        raise ValueError("Chronos-2 returned no embeddings.")
    stacked = torch.stack(embeddings, dim=0)
    # Chronos-2 returns (n_variates, num_context_patches + 2, d_model).
    # Its encoder input is ordered as context patches, [REG], then the masked
    # output patch.  Therefore [REG] is the penultimate token, not token 0.
    if pooling == "reg":
        pooled = stacked[:, :, -2, :]
    elif pooling == "mean":
        pooled = stacked[:, :, :-2, :].mean(dim=2)
    elif pooling == "full":
        pooled = stacked
    else:
        raise ValueError(f"Unknown pooling mode: {pooling}")
    return pooled.float().cpu().numpy().reshape(len(embeddings), -1)


def remove_if_exists(path: Path) -> None:
    if path.exists():
        path.unlink()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--dataset-dir", type=Path, default=REPO_ROOT / "dataset_v3")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--prefix-steps", type=int, default=96)
    parser.add_argument("--sampling-id", type=int, default=2)
    parser.add_argument("--pooling", choices=["reg", "mean", "full"], default="reg")
    parser.add_argument("--sample-batch-size", type=int, default=32)
    parser.add_argument(
        "--embed-batch-size",
        type=int,
        default=256,
        help="Chronos-2 batch size measured in individual time series, not multivariate samples.",
    )
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--dtype", choices=["auto", "bfloat16", "float32"], default="bfloat16")
    parser.add_argument("--limit", type=int, default=None, help="Process only the first N labels (for a smoke test).")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.prefix_steps <= 0 or args.sample_batch_size <= 0 or args.embed_batch_size <= 0:
        raise ValueError("prefix and batch sizes must be positive.")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto":
        device = "cpu"

    labels_all = pd.read_csv(args.labels)
    if args.limit is not None:
        if args.limit <= 0:
            raise ValueError("--limit must be positive when provided.")
        labels = labels_all.iloc[: args.limit].copy().reset_index(drop=True)
    else:
        labels = labels_all.copy().reset_index(drop=True)
    if labels.empty:
        raise ValueError("No Task B labels were found.")

    expected_rows = len(labels)
    expected_dim = feature_dim(args.pooling)
    fingerprint = labels_fingerprint(labels)
    output_path = args.output
    metadata_path = output_path.with_suffix(".json")
    partial_path = output_path.with_name(f"{output_path.stem}.partial.npy")
    progress_path = output_path.with_name(f"{output_path.stem}.progress.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if args.overwrite:
        for path in (output_path, metadata_path, partial_path, progress_path):
            remove_if_exists(path)

    expected_metadata = {
        "rows": expected_rows,
        "feature_dim": expected_dim,
        "prefix_steps": args.prefix_steps,
        "sampling_id": args.sampling_id,
        "pooling": args.pooling,
        "label_fingerprint": fingerprint,
        "dtype": "float16",
    }
    if output_path.exists() and metadata_path.exists():
        metadata = read_json(metadata_path)
        if all(metadata.get(key) == value for key, value in expected_metadata.items()):
            print(f"Reusing completed Chronos-2 cache: {output_path}", flush=True)
            return
        raise ValueError(f"Existing cache has incompatible metadata: {output_path}")

    completed_batches: set[int] = set()
    if progress_path.exists():
        progress = read_json(progress_path)
        for key, value in expected_metadata.items():
            if progress.get(key) != value:
                raise ValueError(f"Partial cache metadata mismatch for {key}: {progress.get(key)!r} != {value!r}")
        completed_batches = {int(batch_id) for batch_id in progress.get("completed_batches", [])}
        if not partial_path.exists():
            raise FileNotFoundError(f"Progress file exists but partial cache is missing: {partial_path}")
        print(f"Resuming {len(completed_batches)} completed batch(es) from {partial_path}", flush=True)
    elif partial_path.exists():
        raise FileExistsError(
            f"Found an untracked partial cache {partial_path}; use --overwrite to discard it or restore its progress file."
        )
    else:
        progress = {**expected_metadata, "completed_batches": []}

    if partial_path.exists():
        features = np.lib.format.open_memmap(partial_path, mode="r+", dtype=np.float16, shape=(expected_rows, expected_dim))
    else:
        features = np.lib.format.open_memmap(partial_path, mode="w+", dtype=np.float16, shape=(expected_rows, expected_dim))
        features.flush()
        write_json(progress_path, progress)

    from chronos import Chronos2Pipeline

    pipeline_kwargs: dict[str, object] = {"device_map": device}
    if args.dtype == "bfloat16":
        pipeline_kwargs["dtype"] = torch.bfloat16
    elif args.dtype == "float32":
        pipeline_kwargs["dtype"] = torch.float32
    print(f"Loading Chronos-2 from {args.model_path}", flush=True)
    pipeline = Chronos2Pipeline.from_pretrained(str(args.model_path), **pipeline_kwargs)

    batch_groups = labels.groupby("num_batch", sort=True).groups
    total_start = time.perf_counter()
    processed = 0
    try:
        for batch_value, group_index in batch_groups.items():
            batch_id = int(batch_value)
            positions = np.asarray(group_index, dtype=np.int64)
            if batch_id in completed_batches:
                processed += len(positions)
                continue
            batch_start = time.perf_counter()
            x = load_batch(
                labels, labels_all, args.dataset_dir, batch_id, positions,
                args.prefix_steps, args.sampling_id,
            )
            for start in range(0, len(x), args.sample_batch_size):
                stop = min(start + args.sample_batch_size, len(x))
                context = torch.from_numpy(np.transpose(x[start:stop], (0, 2, 1)))
                if device == "cuda":
                    torch.cuda.synchronize()
                with torch.inference_mode():
                    embeddings, _ = pipeline.embed(context, batch_size=args.embed_batch_size)
                if device == "cuda":
                    torch.cuda.synchronize()
                if len(embeddings) != stop - start:
                    raise RuntimeError(
                        f"Chronos-2 returned {len(embeddings)} embeddings for {stop - start} samples."
                    )
                features[positions[start:stop]] = pool_embeddings(embeddings, args.pooling).astype(np.float16)
                if stop == len(x) or stop % (args.sample_batch_size * 16) == 0:
                    print(
                        f"chronos2_cache batch={batch_id:02d} batch_rows={stop}/{len(x)}",
                        flush=True,
                    )
            features.flush()
            completed_batches.add(batch_id)
            progress["completed_batches"] = sorted(completed_batches)
            write_json(progress_path, progress)
            processed += len(positions)
            elapsed = time.perf_counter() - total_start
            rate = processed / max(elapsed, 1e-9)
            eta = (expected_rows - processed) / max(rate, 1e-9)
            print(
                f"chronos2_cache batch={batch_id:02d} rows={processed}/{expected_rows} "
                f"batch_seconds={time.perf_counter() - batch_start:.1f} "
                f"rate={rate:.2f}/s eta_hours={eta / 3600:.2f}",
                flush=True,
            )
    except BaseException:
        features.flush()
        raise

    features.flush()
    del features
    os.replace(partial_path, output_path)
    metadata = {
        **expected_metadata,
        "model_path": str(args.model_path),
        "num_batches": len(batch_groups),
        "sample_batch_size": args.sample_batch_size,
        "embed_batch_size": args.embed_batch_size,
        "device": device,
        "model_dtype": args.dtype,
        "feature_layout": "flattened (sample, variate, hidden)" if args.pooling != "full" else "flattened (sample, variate, token, hidden)",
        "completed_batches": sorted(completed_batches),
        "total_seconds": time.perf_counter() - total_start,
    }
    write_json(metadata_path, metadata)
    print(f"Completed Chronos-2 cache: {output_path}", flush=True)
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
