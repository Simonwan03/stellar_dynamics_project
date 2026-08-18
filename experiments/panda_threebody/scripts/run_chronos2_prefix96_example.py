#!/usr/bin/env python3
"""Run Chronos-2 embeddings for one orbital15 Task B sample."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

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
DEFAULT_OUTPUT = EXPERIMENT_ROOT / "outputs/chronos2_prefix96_example"


def _find_local_row(labels: pd.DataFrame, task_row: int) -> tuple[int, int, pd.Series]:
    """Return (batch id, row within HDF5 shard, selected label row)."""
    selected = labels.iloc[int(task_row)]
    batch_id = int(selected["num_batch"])
    if "row_in_batch" in labels.columns:
        return batch_id, int(selected["row_in_batch"]), selected

    # Older label files do not contain row_in_batch.  Reconstruct it using the
    # same index ordering used by the Task B Panda feature extractor.
    batch_labels = labels.loc[labels["num_batch"] == batch_id].sort_values("index")
    positions = {int(index): row for row, index in enumerate(batch_labels["index"].to_numpy())}
    return batch_id, positions[int(selected["index"])], selected


def load_orbital15_sample(
    labels_path: Path,
    dataset_dir: Path,
    task_row: int,
    prefix_steps: int,
) -> tuple[np.ndarray, pd.Series, int, int]:
    labels = pd.read_csv(labels_path)
    if task_row < 0 or task_row >= len(labels):
        raise IndexError(f"task_row={task_row} is outside [0, {len(labels)})")
    batch_id, local_row, selected = _find_local_row(labels, task_row)
    shard = dataset_dir / f"results_batch_{batch_id:02d}_sampling_2.hdf5"
    keys = required_keys_for_representation("orbital15")
    with h5py.File(shard, "r") as handle:
        data = {
            key: np.asarray(handle[key][local_row : local_row + 1, :prefix_steps, ...], dtype=np.float32)
            for key in keys
        }
    x = build_representation(data, "orbital15")  # (1, time, 15)
    if x.shape != (1, prefix_steps, 15):
        raise ValueError(f"Expected (1, {prefix_steps}, 15), got {x.shape}")
    if not np.isfinite(x).all():
        raise ValueError("The selected orbital15 sample contains non-finite values.")
    return x, selected, batch_id, local_row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--dataset-dir", type=Path, default=REPO_ROOT / "dataset_v3")
    parser.add_argument("--task-row", type=int, default=0)
    parser.add_argument("--prefix-steps", type=int, default=96)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--dtype", choices=["auto", "bfloat16", "float32"], default="bfloat16")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    if args.prefix_steps <= 0:
        raise ValueError("--prefix-steps must be positive.")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto":
        device = "cpu"

    x, label, batch_id, local_row = load_orbital15_sample(
        args.labels, args.dataset_dir, args.task_row, args.prefix_steps
    )
    # Chronos-2 uses (batch, variates, history), while the project
    # representation is stored as (batch, history, channels).
    context = torch.from_numpy(np.transpose(x, (0, 2, 1)))

    from chronos import Chronos2Pipeline

    pipeline_kwargs: dict[str, object] = {"device_map": device}
    if args.dtype == "bfloat16":
        pipeline_kwargs["dtype"] = torch.bfloat16
    elif args.dtype == "float32":
        pipeline_kwargs["dtype"] = torch.float32

    print(f"Loading Chronos-2 from {args.model_path}", flush=True)
    pipeline = Chronos2Pipeline.from_pretrained(str(args.model_path), **pipeline_kwargs)
    if device == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.inference_mode():
        embeddings, loc_scales = pipeline.embed(context)
    if device == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    # NumPy does not support direct conversion from CUDA/bfloat16 tensors;
    # persist a portable float32 copy of the frozen embedding.
    embedding = embeddings[0].float().numpy()
    loc, scale = loc_scales[0]
    loc_np = loc.float().numpy()
    scale_np = scale.float().numpy()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.save(args.output_dir / "input_orbital15_prefix96.npy", x[0])
    np.save(args.output_dir / "chronos2_embedding.npy", embedding)
    np.save(args.output_dir / "loc.npy", loc_np)
    np.save(args.output_dir / "scale.npy", scale_np)
    metadata = {
        "model_path": str(args.model_path),
        "task_row": int(args.task_row),
        "num_batch": batch_id,
        "row_in_batch": local_row,
        "index": int(label["index"]),
        "time_unbound": float(label["time_unbound"]),
        "diagnosis": int(label["diagnosis"]),
        "prefix_steps": int(args.prefix_steps),
        "input_shape_batch_variates_time": list(context.shape),
        "embedding_shape_variates_tokens_hidden": list(embedding.shape),
        "loc_shape": list(loc.shape),
        "scale_shape": list(scale.shape),
        "device": device,
        "dtype": args.dtype,
        "embedding_seconds": elapsed,
    }
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")

    print(json.dumps(metadata, indent=2), flush=True)
    print(f"Saved Chronos-2 embedding to {args.output_dir / 'chronos2_embedding.npy'}", flush=True)


if __name__ == "__main__":
    main()
